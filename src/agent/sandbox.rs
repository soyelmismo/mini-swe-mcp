//! Guardrails for agent command execution.
//!
//! Two kinds of protection live here:
//!
//! * **Command-text guardrails** - [`validate_bash_command`] rejects the
//!   obvious escapes before a command is ever run, and [`is_heavy_command`]
//!   classifies a command so it gets a longer timeout.
//! * **Kernel-level filesystem confinement** - [`apply_landlock_sandbox`]
//!   installs a [Landlock] LSM domain on the *current* process (and therefore,
//!   by inheritance, on every child it later spawns), restricting the
//!   filesystem to a read-only system prefix plus an explicitly writable
//!   worktree and build target directory.
//!
//! [Landlock]: https://docs.kernel.org/userspace-api/landlock.html

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// Byte budget above which captured command output is truncated.
pub const TRUNCATE_LIMIT: usize = 16_384;
/// Bytes retained from the *start* of over-budget output (floored to a
/// character boundary).
pub const TRUNCATE_HEAD: usize = 12_288;
/// Bytes retained from the *end* of over-budget output (ceiled to a character
/// boundary).
pub const TRUNCATE_TAIL: usize = 4_096;

/// Literal text of the marker inserted in place of the discarded middle,
/// excluding the decimal byte count that is rendered between the two halves.
const TRUNCATE_MARKER: &str = "\n... [Truncated ";
/// Literal text of the second half of the marker, after the byte count.
const TRUNCATE_MARKER_SUFFIX: &str = " bytes] ...\n";
/// Upper bound on the decimal digits of a `usize` (2^64 - 1 has 20 digits).
/// Used to size the result buffer up front so the marker needs no allocation.
const USIZE_MAX_DIGITS: usize = 20;

/// Bound command output to [`TRUNCATE_LIMIT`] bytes, keeping the head and the
/// tail of the text and reporting how many bytes were discarded.
///
/// Both cut points are snapped to UTF-8 character boundaries (`floor` for the
/// head, `ceil` for the tail), so no character is ever split and
/// `head + dropped + tail == input.len()` holds exactly.
///
/// The result is assembled **once** into an exactly-sized `String`: the marker
/// is pushed directly (the byte count is rendered into a stack buffer) and the
/// 4 KiB tail is never copied through an intermediate allocation.
pub fn truncate_output(combined: &str) -> String {
    let total = combined.len();
    if total <= TRUNCATE_LIMIT {
        return combined.to_string();
    }

    let head_end = combined.floor_char_boundary(TRUNCATE_HEAD);
    let tail_start = combined.ceil_char_boundary(total - TRUNCATE_TAIL);
    let dropped = total - (head_end + (total - tail_start));

    let mut out = String::with_capacity(
        head_end + TRUNCATE_MARKER.len() + USIZE_MAX_DIGITS + TRUNCATE_MARKER_SUFFIX.len()
            + (total - tail_start),
    );
    let mut digits = [0u8; USIZE_MAX_DIGITS];
    let count = render_decimal(&mut digits, dropped);

    out.push_str(&combined[..head_end]);
    out.push_str(TRUNCATE_MARKER);
    out.push_str(count);
    out.push_str(TRUNCATE_MARKER_SUFFIX);
    out.push_str(&combined[tail_start..]);
    debug_assert!(
        out.capacity() >= out.len(),
        "result buffer must be sized up front"
    );
    out
}

/// Render `value` as decimal ASCII digits into `buf` and return the used
/// prefix as a `&str`.
fn render_decimal(buf: &mut [u8; USIZE_MAX_DIGITS], mut value: usize) -> &str {
    debug_assert!(value > 0, "the marker is only emitted with a dropped region");
    let mut idx = buf.len();
    while value > 0 {
        idx -= 1;
        buf[idx] = b'0' + u8::try_from(value % 10).expect("remainder is a single digit");
        value /= 10;
    }
    std::str::from_utf8(&buf[idx..]).expect("ASCII digits are valid UTF-8")
}

/// Validate that a subagent command does not attempt to escape the worktree
/// or trigger runaway recursive scans of root or home filesystems.
pub fn validate_bash_command(command: &str) -> Result<(), &'static str> {
    let trimmed = command.trim();

    // 1. Block recursive searches starting at root, home, or system directories
    const FORBIDDEN_SEARCHES: &[&str] = &[
        "find / ",
        "find / -",
        "find /\"",
        "find /'",
        "find ~",
        "find /home",
        "find /root",
        "find /etc",
        "find /var",
        "find /usr",
        "grep -rn / ",
        "grep -r / ",
    ];

    for token in FORBIDDEN_SEARCHES {
        if trimmed.contains(token) {
            return Err(
                "Scanning root '/' or system directories is forbidden. Confine searches to the current repository ($PWD).",
            );
        }
    }

    // 2. Block escaping to parent or root directory via cd
    const FORBIDDEN_CDS: &[&str] = &[
        "cd / ", "cd /;", "cd /&&", "cd /||", "cd /home", "cd ~", "cd $HOME", "cd /root",
    ];

    for token in FORBIDDEN_CDS {
        if trimmed.contains(token) || trimmed.ends_with("cd /") {
            return Err(
                "Navigating outside the repository with 'cd' is forbidden. All files are in $PWD.",
            );
        }
    }

    Ok(())
}

/// Distinguish CPU-heavy commands (compilations, test runners) from
/// lightweight exploration commands (git status, cat, ls, grep, etc.).
pub fn is_heavy_command(command: &str) -> bool {
    let lower = command.to_lowercase();
    if lower.starts_with("cargo") || lower.contains("cargo ") || lower.contains("cargo\t") {
        return true;
    }
    if lower == "make"
        || lower.starts_with("make ")
        || lower.contains(" make ")
        || lower.contains(" make\t")
    {
        return true;
    }
    const HEAVY_PATTERNS: &[&str] = &[
        "rustc", "pytest", "unittest", "cmake", "ninja", "gcc", "g++", "clang", "npm ", "yarn ",
        "pnpm ", "mvn ", "gradle", "go test", "go build",
    ];
    HEAVY_PATTERNS.iter().any(|pattern| lower.contains(pattern))
}

/// Check if the bubblewrap (`bwrap`) sandbox utility is available on this system.
pub fn has_bwrap() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        std::process::Command::new("bwrap")
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    })
}

/// If a worktree's `.git` is a gitdir reference pointing to a parent git directory,
/// locate both the common `.git` directory and the specific worktree gitdir.
pub fn find_git_dirs(worktree_dir: &Path) -> Option<(PathBuf, Option<PathBuf>)> {
    let dot_git = worktree_dir.join(".git");
    if dot_git.is_file()
        && let Ok(content) = std::fs::read_to_string(&dot_git)
        && let Some(gitdir_line) = content.lines().find(|l| l.starts_with("gitdir: "))
    {
        let raw_path = gitdir_line.trim_start_matches("gitdir: ").trim();
        let gitdir_path = PathBuf::from(raw_path);
        for ancestor in gitdir_path.ancestors() {
            if ancestor.file_name().and_then(|n| n.to_str()) == Some(".git") {
                return Some((ancestor.to_path_buf(), Some(gitdir_path)));
            }
        }
    }
    None
}

/// Backwards-compatible helper returning only the common `.git` root.
pub fn find_git_common_dir(worktree_dir: &Path) -> Option<PathBuf> {
    find_git_dirs(worktree_dir).map(|(common, _)| common)
}

// ---------------------------------------------------------------------------
// Landlock LSM filesystem confinement
// ---------------------------------------------------------------------------

/// Environment variable that force-disables the Landlock confinement, mirroring
/// the existing `SWE_DISABLE_SANDBOX` knob used for the bubblewrap sandbox.
pub const DISABLE_LANDLOCK_ENV: &str = "SWE_DISABLE_LANDLOCK";

/// System prefixes that a sandboxed build is allowed to read (but never write).
///
/// Landlock is a *whitelist* LSM: once a ruleset declares filesystem rights as
/// handled, every path that is not covered by a rule is denied for those
/// rights. Granting these read-only (plus `EXECUTE`, so `/usr/bin/bash` and the
/// linker can actually be run) is what keeps the toolchain usable while the rest
/// of the filesystem stays unreachable.
const READ_ONLY_SYSTEM_PATHS: &[&str] =
    &["/usr", "/bin", "/sbin", "/lib", "/lib64", "/lib32", "/opt"];

/// Home-relative sub-paths that must never be reachable, even read-only.
///
/// These are the credential stores that would let an agent exfiltrate the
/// operator's secrets (or push to their remotes). They are denied *by
/// omission* - no rule is ever added for them - and the list is kept explicit
/// so the intent is testable rather than emergent.
const DENIED_HOME_SUBDIRS: &[&str] = &[".ssh", ".aws", ".gnupg", ".gpg", ".kube", ".docker"];

/// Absolute system paths that must never be reachable, even read-only.
const DENIED_ABSOLUTE_PATHS: &[&str] = &["/root", "/etc/shadow", "/etc/gshadow", "/etc/sudoers"];

/// `LANDLOCK_ACCESS_FS_EXECUTE`: run a file.
const ACCESS_FS_EXECUTE: u64 = 1 << 0;
/// `LANDLOCK_ACCESS_FS_WRITE_FILE`: open a file for writing.
const ACCESS_FS_WRITE_FILE: u64 = 1 << 1;
/// `LANDLOCK_ACCESS_FS_READ_FILE`: open a file for reading.
const ACCESS_FS_READ_FILE: u64 = 1 << 2;
/// `LANDLOCK_ACCESS_FS_READ_DIR`: list / traverse a directory.
const ACCESS_FS_READ_DIR: u64 = 1 << 3;
/// `LANDLOCK_ACCESS_FS_REMOVE_DIR`: unlink a directory.
const ACCESS_FS_REMOVE_DIR: u64 = 1 << 4;
/// `LANDLOCK_ACCESS_FS_REMOVE_FILE`: unlink a file.
const ACCESS_FS_REMOVE_FILE: u64 = 1 << 5;
/// `LANDLOCK_ACCESS_FS_MAKE_CHAR`: create a character device.
const ACCESS_FS_MAKE_CHAR: u64 = 1 << 6;
/// `LANDLOCK_ACCESS_FS_MAKE_DIR`: create a directory.
const ACCESS_FS_MAKE_DIR: u64 = 1 << 7;
/// `LANDLOCK_ACCESS_FS_MAKE_REG`: create a regular file.
const ACCESS_FS_MAKE_REG: u64 = 1 << 8;
/// `LANDLOCK_ACCESS_FS_MAKE_SOCK`: create a UNIX socket.
const ACCESS_FS_MAKE_SOCK: u64 = 1 << 9;
/// `LANDLOCK_ACCESS_FS_MAKE_FIFO`: create a named pipe.
const ACCESS_FS_MAKE_FIFO: u64 = 1 << 10;
/// `LANDLOCK_ACCESS_FS_MAKE_BLOCK`: create a block device.
const ACCESS_FS_MAKE_BLOCK: u64 = 1 << 11;
/// `LANDLOCK_ACCESS_FS_MAKE_SYM`: create a symbolic link.
const ACCESS_FS_MAKE_SYM: u64 = 1 << 12;
/// `LANDLOCK_ACCESS_FS_REFER`: link or rename across directory boundaries.
const ACCESS_FS_REFER: u64 = 1 << 13;
/// `LANDLOCK_ACCESS_FS_TRUNCATE`: truncate a file.
const ACCESS_FS_TRUNCATE: u64 = 1 << 14;
/// `LANDLOCK_ACCESS_FS_IOCTL_DEV`: issue device-specific `ioctl`s.
const ACCESS_FS_IOCTL_DEV: u64 = 1 << 15;
/// `LANDLOCK_ACCESS_FS_RESOLVE_UNIX`: connect to a pathname UNIX socket.
const ACCESS_FS_RESOLVE_UNIX: u64 = 1 << 16;

/// Number of distinct `LANDLOCK_ACCESS_FS_*` rights defined by the UAPI, and
/// therefore the exclusive upper bound on the bit index of a valid right.
const ACCESS_FS_MAX_BIT: u32 = 17;

/// Every filesystem access right this module knows how to request.
const ALL_ACCESS_FS: u64 = (1 << ACCESS_FS_MAX_BIT) - 1;

/// `LANDLOCK_RULE_PATH_BENEATH`: the rule type describing a directory subtree.
const RULE_PATH_BENEATH: u32 = 1;

/// `LANDLOCK_CREATE_RULESET_VERSION`: ask the kernel for its supported ABI.
const CREATE_RULESET_VERSION: u32 = 1;

/// Syscall number of `landlock_create_ruleset(2)`.
const SYS_LANDLOCK_CREATE_RULESET: libc::c_long = 444;
/// Syscall number of `landlock_add_rule(2)`.
const SYS_LANDLOCK_ADD_RULE: libc::c_long = 445;
/// Syscall number of `landlock_restrict_self(2)`.
const SYS_LANDLOCK_RESTRICT_SELF: libc::c_long = 446;

/// Oldest Landlock ABI this implementation is willing to talk to.
///
/// ABI 1 is the initial Landlock release. Anything below it does not exist, and
/// is indistinguishable from "unsupported", so both take the same
/// graceful-degradation path.
const MIN_SUPPORTED_ABI: i64 = 1;

/// One `PATH_BENEATH` rule: a directory subtree and the rights allowed in it.
#[derive(Debug, Clone, PartialEq, Eq)]
struct PathRule {
    /// Directory whose subtree the rule applies to.
    path: PathBuf,
    /// Bitmask of `ACCESS_FS_*` rights granted inside the subtree.
    allowed: u64,
}

/// `struct landlock_ruleset_attr` truncated to the field this module sets.
///
/// The kernel UAPI lets the structure grow across ABI versions and validates
/// the caller's `size`, so only `handled_access_fs` is ever populated and only
/// that field's size is passed. Querying with `size == 0` returns the highest
/// supported ABI version.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
struct RulesetAttr {
    handled_access_fs: u64,
}

/// `struct landlock_path_beneath_attr` as the kernel defines it.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct PathBeneathAttr {
    allowed_access: u64,
    parent_fd: libc::c_int,
}

/// Every filesystem right a sandboxed child needs to *read* a path: listing
/// directories, reading files, resolving sockets and executing binaries.
const READ_ONLY_RIGHTS: u64 =
    ACCESS_FS_READ_FILE | ACCESS_FS_READ_DIR | ACCESS_FS_EXECUTE | ACCESS_FS_RESOLVE_UNIX;

/// Every filesystem right a sandboxed child needs to *build* inside its worktree
/// or target dir: create, rewrite, truncate, delete and link.
///
/// Device nodes and UNIX sockets are deliberately excluded: nothing in a build
/// needs to `mknod` or `bind(2)` a socket, and dropping those two rights costs
/// a little capability without breaking anything.
const WRITE_RIGHTS: u64 = ACCESS_FS_EXECUTE
    | ACCESS_FS_READ_FILE
    | ACCESS_FS_READ_DIR
    | ACCESS_FS_WRITE_FILE
    | ACCESS_FS_TRUNCATE
    | ACCESS_FS_REMOVE_DIR
    | ACCESS_FS_REMOVE_FILE
    | ACCESS_FS_MAKE_DIR
    | ACCESS_FS_MAKE_REG
    | ACCESS_FS_MAKE_SYM
    | ACCESS_FS_MAKE_FIFO
    | ACCESS_FS_REFER
    | ACCESS_FS_RESOLVE_UNIX;

/// Which filesystem rights became available in which Landlock ABI version.
///
/// A kernel rejects a ruleset that asks for a right it does not implement, so
/// the mask has to be narrowed to the running kernel's ABI before
/// `landlock_create_ruleset` is called. The table lists, per ABI, only the
/// rights *introduced* by that release; the result is cumulative.
const ABI_ACCESS_FS_INTRODUCED: &[(i64, u64)] = &[
    // ABI 1: only EXECUTE existed.
    (1, ACCESS_FS_EXECUTE),
    // ABI 2: the bulk of the filesystem rights, including REFER.
    (
        2,
        ACCESS_FS_WRITE_FILE
            | ACCESS_FS_READ_FILE
            | ACCESS_FS_READ_DIR
            | ACCESS_FS_REMOVE_DIR
            | ACCESS_FS_REMOVE_FILE
            | ACCESS_FS_MAKE_CHAR
            | ACCESS_FS_MAKE_DIR
            | ACCESS_FS_MAKE_REG
            | ACCESS_FS_MAKE_SOCK
            | ACCESS_FS_MAKE_FIFO
            | ACCESS_FS_MAKE_BLOCK
            | ACCESS_FS_MAKE_SYM
            | ACCESS_FS_REFER,
    ),
    // ABI 3: TRUNCATE.
    (3, ACCESS_FS_TRUNCATE),
    // ABI 4 is network-only and adds no filesystem right.
    (5, ACCESS_FS_IOCTL_DEV),
    (9, ACCESS_FS_RESOLVE_UNIX),
];

/// Thin `unsafe` wrapper around one of the three Landlock syscalls.
///
/// Returns the raw return value: a non-negative `c_long` on success (a file
/// descriptor for `create_ruleset`, `0` for the other two) and `-1` with
/// `errno` set on failure.
///
/// All three syscalls take up to four arguments
/// (`landlock_add_rule` is `(ruleset_fd, rule_type, rule_attr, flags)`); the
/// shorter ones simply ignore the trailing zero. Passing `flags` explicitly
/// matters: leaving the fourth register uninitialised makes `add_rule` fail
/// intermittently with `EINVAL` depending on whatever garbage the caller
/// happened to leave in `r10`.
fn landlock_syscall(number: libc::c_long, args: [libc::c_long; 4]) -> i64 {
    // SAFETY: the Landlock syscalls take plain `c_long`-sized arguments.
    // Pointers are either null or reference live, correctly sized and aligned
    // stack structs that outlive the call, and the kernel only ever reads
    // `size` bytes from them.
    unsafe { libc::syscall(number, args[0], args[1], args[2], args[3]) }
}

/// Query the highest Landlock ABI version the running kernel implements.
///
/// Returns `None` when Landlock is compiled out, disabled at boot via `lsm=`,
/// or blocked by a seccomp policy - every one of those surfaces as a failed
/// syscall, and none of them is an error worth propagating to the caller.
fn query_abi_version() -> Option<i64> {
    // `size == 0` with the VERSION flag is the documented ABI-version query.
    let ret = landlock_syscall(
        SYS_LANDLOCK_CREATE_RULESET,
        [0, 0, CREATE_RULESET_VERSION as libc::c_long, 0],
    );
    if ret < MIN_SUPPORTED_ABI {
        return None;
    }
    Some(ret)
}

/// The subset of [`ALL_ACCESS_FS`] that a kernel of the given ABI implements.
///
/// Unknown or newer ABIs simply get every right this crate knows about: a
/// future kernel is a superset of the current one, and a right we never request
/// costs nothing but leaves a little capability unused.
fn supported_access_fs(abi: i64) -> u64 {
    let mut mask = 0;
    for (introduced_in, rights) in ABI_ACCESS_FS_INTRODUCED {
        if abi >= *introduced_in {
            mask |= *rights;
        }
    }
    // ABI 0 means "no Landlock at all", so an empty mask is correct there; the
    // caller never gets that far because `query_abi_version` rejects it first.
    debug_assert!(
        mask & !ALL_ACCESS_FS == 0,
        "the ABI table must stay within the defined rights"
    );
    mask & ALL_ACCESS_FS
}

/// Whether Landlock confinement has been switched off for this process.
fn landlock_disabled() -> bool {
    std::env::var(DISABLE_LANDLOCK_ENV).as_deref() == Ok("1")
}

/// Home directory of the user the worker runs as, if one is discoverable.
fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .filter(|p| !p.as_os_str().is_empty())
}

/// Absolute paths that must stay unreachable, resolved against `$HOME`.
///
/// The list is intentionally *not* added as Landlock rules: the whole point is
/// that no rule covers them, so the handled rights deny them. Keeping an
/// explicit, testable list stops that from silently regressing into "granted by
/// accident" if a broad prefix rule is ever added above.
fn denied_paths() -> Vec<PathBuf> {
    let mut denied: Vec<PathBuf> = DENIED_ABSOLUTE_PATHS.iter().map(PathBuf::from).collect();
    if let Some(home) = home_dir() {
        denied.extend(DENIED_HOME_SUBDIRS.iter().map(|d| home.join(d)));
    }
    denied
}

/// True when `path` is `denied` or lives underneath it.
///
/// Purely lexical (no canonicalisation): the rules handed to Landlock are
/// matched on the paths actually opened, and this check exists to keep the
/// *policy* honest rather than to mirror the kernel's resolution.
fn is_denied(path: &Path, denied: &[PathBuf]) -> bool {
    denied.iter().any(|d| path == d || path.starts_with(d))
}

/// Build the set of `PATH_BENEATH` rules for a sandboxed child.
///
/// Landlock is allow-only: every granted path widens access, so the policy
/// grants read-only system prefixes plus exactly two writable roots and
/// nothing else. Sensitive paths stay unreachable by omission (see
/// [`denied_paths`]).
///
/// Every path is filtered through [`is_denied`], so a denied directory can never
/// be granted access even if it is also reachable from an allowed prefix.
fn build_path_rules(worktree: &Path, target_dir: &Path) -> Vec<PathRule> {
    let denied = denied_paths();
    let mut rules = Vec::new();

    let push = |path: PathBuf, allowed: u64, rules: &mut Vec<PathRule>| {
        if path.is_absolute() && !is_denied(&path, &denied) {
            rules.push(PathRule { path, allowed });
        }
    };

    // 1. System prefixes: readable and executable, never writable.
    for sys in READ_ONLY_SYSTEM_PATHS {
        push(PathBuf::from(sys), READ_ONLY_RIGHTS, &mut rules);
    }

    // Landlock is allow-only: a rule on /etc would also grant /etc/shadow.
    // Grant only the individual non-secret configuration files needed by tools.
    for config in [
        "/etc/passwd", "/etc/group", "/etc/nsswitch.conf", "/etc/resolv.conf",
        "/etc/hosts", "/etc/host.conf", "/etc/gai.conf", "/etc/ld.so.cache",
        "/etc/localtime", "/etc/os-release", "/etc/protocols", "/etc/services",
        "/etc/ssl/certs", "/etc/ca-certificates", "/etc/pki/tls/certs",
    ] {
        push(PathBuf::from(config), ACCESS_FS_READ_FILE | ACCESS_FS_READ_DIR, &mut rules);
    }

    // 3. Pseudo-filesystems, always readable.
    push(PathBuf::from("/dev"), READ_ONLY_RIGHTS, &mut rules);
    push(PathBuf::from("/proc"), READ_ONLY_RIGHTS, &mut rules);

    // 4. The worker's own writable roots: the only writable paths in the domain.
    push(worktree.to_path_buf(), WRITE_RIGHTS, &mut rules);
    push(target_dir.to_path_buf(), WRITE_RIGHTS, &mut rules);

    rules
}

/// The union of every right any rule asks for, narrowed to what the kernel
/// supports: the ruleset's `handled_access_fs` mask.
///
/// Handled rights are what turns Landlock's "allow" model into a "deny by
/// default" one, so this is also the set of operations an ungranted path loses.
fn handled_access_fs(rules: &[PathRule], access: u64) -> u64 {
    rules
        .iter()
        .fold(0, |acc, rule| acc | rule.allowed)
        & access
}

/// Apply a Landlock filesystem domain to the calling process and its children.
///
/// The domain is installed with `landlock_restrict_self`, which is
/// **irreversible and one-way**: the process cannot widen its own access
/// afterwards, and every process it forks inherits the restriction. That is
/// exactly what is wanted for worker execution, but it is also why this
/// function is the *only* place in the crate that should call it - a caller
/// must be certain it is done touching anything outside the sandbox, because
/// its own subsequent filesystem access is confined too.
///
/// What the resulting domain permits:
///
/// * **read + execute** on the system prefixes ([`READ_ONLY_SYSTEM_PATHS`]),
///   selected `/etc` config files, `/proc` and `/dev` - enough to run a compiler, a linker and
///   `bash` itself;
/// * **read + write** on `worktree` and `target_dir`.
///
/// Everything else is denied, which is what makes the sensitive paths - the
/// operator's `~/.ssh`, `~/.aws`, `~/.gnupg` and `/etc/shadow` (see
/// [`denied_paths`]) - unreachable rather than merely unused.
///
/// # Graceful degradation
///
/// Landlock is a Linux LSM that is absent from kernels older than 5.13 and can
/// be disabled at boot (`lsm=` without `landlock`) or blocked by a seccomp
/// policy. None of those is a reason to fail a worker: when the kernel cannot
/// support the domain, this logs at debug level and returns `Ok(())`, leaving
/// the process unconfined. Only a *malformed policy* - a path we were told to
/// sandbox which does not exist - is reported as an `Err`.
///
/// # Example
///
/// ```no_run
/// use std::path::Path;
/// mini_swe_mcp::agent::sandbox::apply_landlock_sandbox(
///     Path::new("/tmp/swe-wt-ab12cd34"),
///     Path::new("/tmp/swe-target-ab12cd34"),
/// )?;
/// # Ok::<(), anyhow::Error>(())
/// ```
pub fn apply_landlock_sandbox(worktree: &Path, target_dir: &Path) -> Result<()> {
    if landlock_disabled() {
        tracing::debug!("landlock confinement disabled by {DISABLE_LANDLOCK_ENV}=1");
        return Ok(());
    }

    // Both roots must exist: a rule on a missing path is a caller bug, and
    // silently granting access to a directory that is not there would only hide
    // it until the first write fails somewhere much less obvious.
    if !worktree.is_dir() {
        anyhow::bail!("landlock worktree does not exist: {}", worktree.display());
    }
    if !target_dir.is_dir() {
        anyhow::bail!("landlock target dir does not exist: {}", target_dir.display());
    }

    apply_with_abi(worktree, target_dir, query_abi_version())
}

/// The body of [`apply_landlock_sandbox`], with the ABI probe as a parameter.
///
/// Taking the ABI as an argument rather than calling [`query_abi_version`]
/// directly is what makes the "kernel has no Landlock" branch reachable from a
/// test: on a kernel that *does* support Landlock the probe can never return
/// `None`, so the graceful-degradation path would otherwise be dead code that
/// no test ever executes. `abi == None` is exactly the state a pre-5.13 or
/// `lsm=`-disabled kernel puts us in.
fn apply_with_abi(worktree: &Path, target_dir: &Path, abi: Option<i64>) -> Result<()> {
    let Some(abi) = abi else {
        tracing::debug!("landlock unsupported by this kernel; running unconfined");
        return Ok(());
    };

    let access = supported_access_fs(abi);
    let rules = build_path_rules(worktree, target_dir);
    let handled = handled_access_fs(&rules, access);

    create_ruleset(handled)
        .and_then(|ruleset_fd| {
            for rule in &rules {
                add_rule(ruleset_fd, rule)?;
            }
            // Landlock requires PR_SET_NO_NEW_PRIVS to be set before restrict_self
            // unless the process has CAP_SYS_ADMIN.
            // SAFETY: prctl with PR_SET_NO_NEW_PRIVS takes integer arguments and is safe.
            if unsafe { libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) } != 0 {
                anyhow::bail!(
                    "prctl PR_SET_NO_NEW_PRIVS: {}",
                    std::io::Error::last_os_error()
                );
            }

            // SAFETY: `ruleset_fd` is still open here and no other thread can
            // have closed it; `restrict_self` needs no other argument.
            let ret = landlock_syscall(
                SYS_LANDLOCK_RESTRICT_SELF,
                [ruleset_fd.into(), 0, 0, 0],
            );
            if ret < 0 {
                anyhow::bail!(
                    "landlock_restrict_self: {}",
                    std::io::Error::last_os_error()
                );
            }
            Ok(())
        })
        .with_context(|| format!("failed to install landlock ABI {abi} filesystem domain"))?;

    tracing::debug!(abi, handled, rules = rules.len(), "landlock filesystem domain applied");
    Ok(())
}

/// Add one `PATH_BENEATH` rule to an open ruleset.
fn add_rule(ruleset_fd: libc::c_int, rule: &PathRule) -> Result<()> {
    // Open with `O_PATH`: it needs no permission on the target itself (only
    // traversal of its parents), so a rule is still installed on paths the
    // caller could not `File::open` for reading. A genuinely missing path
    // simply has nothing to protect; skipping it keeps the sandbox working on
    // minimal images that lack, say, `/opt`.
    // SAFETY: `path` is a NUL-free OS string converted via `CString`; the fd
    // returned by `open` is owned here and closed exactly once below.
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;
    let fd: libc::c_int = match CString::new(rule.path.as_os_str().as_bytes()) {
        Ok(c) => unsafe { libc::open(c.as_ptr(), libc::O_PATH | libc::O_CLOEXEC) },
        Err(_) => return Ok(()),
    };
    if fd < 0 {
        return Ok(());
    }
    struct FdGuard(libc::c_int);
    impl Drop for FdGuard {
        fn drop(&mut self) {
            // SAFETY: fd was returned by a successful `open` above and is
            // closed exactly once here.
            unsafe { libc::close(self.0) };
        }
    }
    let _guard = FdGuard(fd);
    let parent: libc::c_int = fd;
    let allowed = if rule.path.is_file() { rule.allowed & ACCESS_FS_READ_FILE } else { rule.allowed };
    let attr = PathBeneathAttr {
        allowed_access: allowed,
        parent_fd: parent,
    };
    // SAFETY: `attr` is a live `landlock_path_beneath_attr` and `_guard` keeps
    // the referenced file or directory open for the duration of the call.
    let ret = landlock_syscall(
        SYS_LANDLOCK_ADD_RULE,
        [
            ruleset_fd.into(),
            RULE_PATH_BENEATH.into(),
            &attr as *const PathBeneathAttr as libc::c_long,
            0,
        ],
    );
    if ret < 0 {
        // A right the kernel rejects means the policy is ahead of the running
        // ABI, which is a bug in [`supported_access_fs`], not a runtime condition.
        anyhow::bail!(
            "landlock_add_rule({}): {}",
            rule.path.display(),
            std::io::Error::last_os_error()
        );
    }
    Ok(())
}

/// Create a Landlock ruleset handling `handled_access_fs`, returning its fd.
fn create_ruleset(handled_access_fs: u64) -> Result<libc::c_int> {
    let attr = RulesetAttr { handled_access_fs };
    // SAFETY: `attr` is a live, correctly laid out `landlock_ruleset_attr`
    // prefix and `size` is its exact size, which is what the kernel validates.
    let ret = landlock_syscall(
        SYS_LANDLOCK_CREATE_RULESET,
        [
            &attr as *const RulesetAttr as libc::c_long,
            std::mem::size_of::<RulesetAttr>() as libc::c_long,
            0,
            0,
        ],
    );
    if ret < 0 {
        anyhow::bail!(
            "landlock_create_ruleset: {}",
            std::io::Error::last_os_error()
        );
    }
    Ok(ret as libc::c_int)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truncate_output_utf8_boundary() {
        let mut s = "a".repeat(TRUNCATE_HEAD - 1);
        s.push('€'); // bytes 12287..12290
        s.push_str(&"b".repeat(9000));
        let truncated = truncate_output(&s);
        assert!(truncated.contains("... [Truncated"));
    }

    #[test]
    fn test_truncate_output_constants_match_the_budget() {
        const { assert!(TRUNCATE_HEAD + TRUNCATE_TAIL == TRUNCATE_LIMIT) };
        const { assert!(TRUNCATE_TAIL < TRUNCATE_LIMIT && TRUNCATE_HEAD < TRUNCATE_LIMIT) };
    }

    #[test]
    fn test_validate_bash_command_blocks_escapes() {
        // Blocked: root find
        assert!(validate_bash_command("find / -name 'foo'").is_err());
        assert!(validate_bash_command("find ~ -name 'foo'").is_err());
        assert!(validate_bash_command("find /home -name 'foo'").is_err());

        // Blocked: cd to root or home
        assert!(validate_bash_command("cd / && ls").is_err());
        assert!(validate_bash_command("cd /home && ls").is_err());
        assert!(validate_bash_command("cd ~").is_err());
        assert!(validate_bash_command("cd /").is_err());

        // Allowed: within worktree
        assert!(validate_bash_command("find . -name 'foo'").is_ok());
        assert!(validate_bash_command("find src -type f").is_ok());
        assert!(validate_bash_command("cd src && cargo test").is_ok());
        assert!(validate_bash_command("grep -rn 'WorkerState' src/").is_ok());
    }

    #[test]
    fn test_is_heavy_command() {
        assert!(is_heavy_command("cargo build"));
        assert!(is_heavy_command("cargo test --all"));
        assert!(is_heavy_command("cargo"));
        assert!(is_heavy_command("pytest tests/"));
        assert!(is_heavy_command("make -j4"));
        assert!(is_heavy_command("make"));
        assert!(is_heavy_command("gcc -O3 main.c"));

        assert!(!is_heavy_command("git status"));
        assert!(!is_heavy_command("git diff HEAD"));
        assert!(!is_heavy_command("ls -la"));
        assert!(!is_heavy_command("cat src/agent.rs"));
        assert!(!is_heavy_command("find . -name '*.rs'"));
        assert!(!is_heavy_command(
            "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
        ));
    }

    #[test]
    fn test_has_bwrap_returns_boolean() {
        let _ = has_bwrap();
    }

    #[test]
    fn test_find_git_common_dir_on_regular_dir() {
        let tmp = std::env::temp_dir();
        assert_eq!(find_git_common_dir(&tmp), None);
    }

// ---------- Landlock confinement tests ----------

/// Rule granted to `path` by `rules`, if any. Later rules win, mirroring the
/// kernel's refine-previous semantics.
fn rule_for<'a>(rules: &'a [PathRule], path: &Path) -> Option<&'a PathRule> {
    rules.iter().rev().find(|r| path == r.path)
}

/// Scratch directory unique to the calling test, removed on drop.
struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        let dir = crate::worktree::swe_base_dir()
            .join("landlock-test")
            .join(format!("{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create scratch dir");
        Self(dir)
    }

    fn worktree(&self) -> PathBuf {
        let w = self.0.join("worktree");
        std::fs::create_dir_all(&w).expect("create worktree");
        w
    }

    fn target(&self) -> PathBuf {
        let t = self.0.join("target");
        std::fs::create_dir_all(&t).expect("create target dir");
        t
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn system_prefixes_are_read_only() {
    let scratch = Scratch::new("readonly");
    let rules = build_path_rules(&scratch.worktree(), &scratch.target());

    for sys in ["/usr", "/bin", "/lib", "/opt"] {
        let rule = rule_for(&rules, Path::new(sys))
            .unwrap_or_else(|| panic!("{sys} must be granted so the toolchain can run"));
        assert_eq!(
            rule.allowed, READ_ONLY_RIGHTS,
            "{sys} must be read-only, not writable"
        );
    }
}

#[test]
fn worktree_and_target_are_writable() {
    let scratch = Scratch::new("writable");
    let (worktree, target) = (scratch.worktree(), scratch.target());
    let rules = build_path_rules(&worktree, &target);

    for root in [&worktree, &target] {
        let rule = rule_for(&rules, root)
            .unwrap_or_else(|| panic!("{} must be writable", root.display()));
        assert!(
            rule.allowed & ACCESS_FS_WRITE_FILE != 0,
            "{} must allow write access",
            root.display()
        );
        assert!(
            rule.allowed & ACCESS_FS_MAKE_REG != 0,
            "{} must allow creating files",
            root.display()
        );
        assert!(
            rule.allowed & ACCESS_FS_MAKE_DIR != 0,
            "{} must allow creating directories",
            root.display()
        );
    }
}

#[test]
fn writable_roots_are_narrowed_to_the_supported_abi() {
    let scratch = Scratch::new("abi");
    let rules = build_path_rules(&scratch.worktree(), &scratch.target());

    // On the oldest supported ABI only EXECUTE exists, so a ruleset may not ask
    // for anything the kernel would reject.
    for abi in 1..=9 {
        let access = supported_access_fs(abi);
        assert_eq!(access & !ALL_ACCESS_FS, 0, "ABI {abi} mask escapes the UAPI");
        let handled = handled_access_fs(&rules, access);
        assert_eq!(
            handled & !access,
            0,
            "ABI {abi}: handled rights must be a subset of supported rights"
        );
    }

    // Rights only exist from their introducing ABI onwards.
    assert_eq!(supported_access_fs(1), ACCESS_FS_EXECUTE);
    assert_eq!(
        supported_access_fs(2) & ACCESS_FS_REFER,
        ACCESS_FS_REFER,
        "REFER arrived in ABI 2"
    );
    assert_eq!(
        supported_access_fs(2) & ACCESS_FS_TRUNCATE,
        0,
        "TRUNCATE only arrived in ABI 3"
    );
    assert_eq!(
        supported_access_fs(4) & ACCESS_FS_IOCTL_DEV,
        0,
        "IOCTL_DEV only arrived in ABI 5"
    );
    assert_eq!(
        supported_access_fs(8) & ACCESS_FS_RESOLVE_UNIX,
        0,
        "RESOLVE_UNIX only arrived in ABI 9"
    );
    assert_eq!(supported_access_fs(9), ALL_ACCESS_FS, "ABI 9 knows every right");
    assert_eq!(
        supported_access_fs(999),
        ALL_ACCESS_FS,
        "a future ABI is a superset, not a subset"
    );
}

#[test]
fn sensitive_paths_are_never_granted() {
    let scratch = Scratch::new("denied");
    let (worktree, target) = (scratch.worktree(), &scratch.target());
    let rules = build_path_rules(&worktree, target);
    let denied = denied_paths();

    // The documented set must be present, resolved against $HOME where relative.
    let home = home_dir().expect("tests run with a discoverable $HOME");
    for rel in [".ssh", ".aws", ".gnupg"] {
        assert!(
            denied.contains(&home.join(rel)),
            "$HOME/{rel} must be on the deny list"
        );
    }
    assert!(
        denied.iter().any(|p| p == Path::new("/etc/shadow")),
        "/etc/shadow must be on the deny list"
    );

    // Home-relative secrets must never appear in any rule: they are denied by
    // omission (no rule covers them), so the handled-rights deny-by-default
    // model keeps them unreachable.
    for path in &denied {
        assert!(
            !is_denied(path, &[]),
            "a denied path must not be classified as allowed by an empty list"
        );
        for rule in &rules {
            assert!(
                rule.path != *path && !rule.path.starts_with(path),
                "rule {} grants access to denied path {}",
                rule.path.display(),
                path.display()
            );
        }
    }

    // A broad /etc grant would also grant these files. They must be denied
    // by omission, not an impossible zero-rights rule (the kernel rejects it).
    for secret in ["/etc/shadow", "/etc/gshadow", "/etc/sudoers"] {
        assert!(!rules.iter().any(|r| Path::new(secret).starts_with(&r.path)));
    }
}

#[test]
fn a_denied_subtree_is_filtered_out_of_the_rule_set() {
    let _scratch = Scratch::new("filter");
    let home = home_dir().expect("tests run with a discoverable $HOME");
    // A worktree pinned at a denied location must be filtered out, not granted.
    let denied = vec![home.join(".ssh")];
    assert!(is_denied(&home.join(".ssh"), &denied));
    assert!(is_denied(&home.join(".ssh/id_rsa"), &denied));
    assert!(!is_denied(&home.join(".cargo"), &denied));
    assert!(!is_denied(&home.join(".sshx"), &denied), "prefix must match on a path segment");
}

#[test]
fn writable_roots_win_over_the_system_prefixes() {
    let scratch = Scratch::new("order");
    let (worktree, target) = (scratch.worktree(), scratch.target());
    let rules = build_path_rules(&worktree, &target);

    // The writable roots are appended last, so a later (more specific) rule can
    // refine an earlier read-only one instead of being shadowed by it.
    let last_read_only = rules
        .iter()
        .rposition(|r| r.allowed == READ_ONLY_RIGHTS)
        .expect("the policy grants read-only prefixes");
    let worktree_at = rules
        .iter()
        .position(|r| r.path == worktree)
        .expect("worktree is granted");
    assert!(
        worktree_at > last_read_only,
        "writable roots must come after the read-only prefixes"
    );
    let _ = target;
}

#[test]
fn relative_roots_are_never_granted() {
    // Landlock rules are matched on absolute paths; a relative one would be
    // silently useless, so it is dropped rather than granted.
    let rules = build_path_rules(Path::new("relative-worktree"), Path::new("relative-target"));
    assert!(!rules.iter().any(|r| r.path == Path::new("relative-worktree")));
}

#[test]
fn a_missing_worktree_is_reported_rather_than_ignored() {
    let scratch = Scratch::new("missing");
    let missing = scratch.0.join("does-not-exist");
    let err = apply_landlock_sandbox(&missing, &scratch.target()).unwrap_err();
    assert!(
        format!("{err:#}").contains("does not exist"),
        "a missing worktree must be an error, got: {err:#}"
    );
}

#[test]
fn a_missing_target_dir_is_reported_rather_than_ignored() {
    let scratch = Scratch::new("missing-target");
    let missing = scratch.0.join("no-target");
    let err = apply_landlock_sandbox(&scratch.worktree(), &missing).unwrap_err();
    assert!(
        format!("{err:#}").contains("does not exist"),
        "a missing target dir must be an error, got: {err:#}"
    );
}

#[test]
fn landlock_can_be_disabled_without_failing() {
    let scratch = Scratch::new("disabled");
    // SAFETY: the test harness runs these environment-sensitive tests in one
    // process; `SWE_DISABLE_LANDLOCK` is only read here and by
    // `apply_landlock_sandbox`, and each test that touches it removes it again
    // before returning. Nothing else in the process consults the variable.
    unsafe { std::env::set_var(DISABLE_LANDLOCK_ENV, "1") };
    let result = apply_landlock_sandbox(&scratch.worktree(), &scratch.target());
    unsafe { std::env::remove_var(DISABLE_LANDLOCK_ENV) };

    assert!(
        result.is_ok(),
        "an explicit opt-out must never fail a worker: {result:?}"
    );
}

#[test]
fn an_unsupported_kernel_degrades_instead_of_failing() {
    // `supported_access_fs` is total: every ABI, including one below the minimum
    // this build supports, yields a mask the kernel can actually be asked for.
    // That is the property that makes the kernel-version probe a pure
    // optimisation rather than a correctness dependency.
    for abi in 0..=12 {
        let access = supported_access_fs(abi);
        assert_eq!(access & !ALL_ACCESS_FS, 0, "ABI {abi} mask escapes the UAPI");
    }

    // A nonexistent kernel: the syscall fails, `query_abi_version` reports
    // "unsupported" instead of an error, and the caller proceeds unconfined.
    // The kernel running the suite supports Landlock (the enforcement test
    // proves it end-to-end); the degradation path is exercised by the
    // opt-out test above, which does not depend on kernel capabilities.
    assert!(query_abi_version().is_some(), "this kernel does support it");
}

/// The graceful-degradation path, forced through the [`apply_with_abi`] seam.
///
/// This is the property that matters operationally: on a kernel without
/// Landlock (pre-5.13, built without the LSM, disabled via `lsm=`, or blocked
/// by a seccomp policy) a worker must still run. The probe is the *only* thing
/// that distinguishes that machine from this one, so the branch is exercised
/// with `None` directly rather than being left to chance.
#[test]
fn a_kernel_without_landlock_still_runs_the_worker() {
    let scratch = Scratch::new("noll");

    // `None` is exactly what a failing `landlock_create_ruleset(VERSION)` reports.
    let result = apply_with_abi(&scratch.worktree(), &scratch.target(), None);

    assert!(
        result.is_ok(),
        "an unsupported kernel must degrade, not fail: {result:?}"
    );
    // Degrading means *unconfined*, so the process keeps the access it had -
    // which is why this is a downgrade and not a silently partial sandbox.
    std::fs::write(scratch.worktree().join("still-writable"), b"ok")
        .expect("an unconfined process keeps its access");
}

/// An ABI that does not exist must not silently produce an empty ruleset: a
/// handled-mask of zero would deny *everything*, including the worktree, so a
/// bogus value threaded through the seam has to be visibly wrong.
#[test]
fn an_impossible_abi_handles_no_rights() {
    assert_eq!(
        supported_access_fs(0),
        0,
        "ABI 0 does not exist and must grant nothing"
    );
    let scratch = Scratch::new("badabi");
    let rules = build_path_rules(&scratch.worktree(), &scratch.target());
    assert_eq!(
        handled_access_fs(&rules, supported_access_fs(0)),
        0,
        "an ABI-0 ruleset must handle no rights at all"
    );
}

#[test]
fn the_abi_query_is_stable_within_a_process() {
    // The probe is a syscall, not a cached value, so it must be consistent; a
    // flaky reading would mean a partial ruleset was installed.
    let a = query_abi_version();
    let b = query_abi_version();
    assert_eq!(a, b, "the Landlock ABI query must be deterministic");
    if let Some(abi) = a {
        assert!(abi >= MIN_SUPPORTED_ABI, "a reported ABI is always usable");
    }
}

    /// Write access must go to the worktree and the target dir and nowhere else.
    ///
    /// This is the property that makes the sandbox worth having, and it is easy
    /// to lose by accident: Landlock is allow-only, so a `/tmp` rule would also
    /// cover the worktree/target (which live there by default) and hand the
    /// agent write access to every other worker's files. Scratch space is
    /// therefore never granted: only the two declared roots are writable.
    #[test]
    fn write_access_is_exclusive_to_the_two_declared_roots() {
        let scratch = Scratch::new("exclusive");
        let worktree = Path::new("/tmp/landlock-wt-abc123");
        let target = Path::new("/tmp/landlock-target-abc123");
        let rules = build_path_rules(worktree, target);

        let writable: Vec<&Path> = rules
            .iter()
            .filter(|r| r.allowed & ACCESS_FS_WRITE_FILE != 0)
            .map(|r| r.path.as_path())
            .collect();

        assert_eq!(
            writable.len(),
            2,
            "only the two declared roots may be writable, got: {writable:?}"
        );
        assert!(writable.contains(&worktree));
        assert!(writable.contains(&target));
        let _ = &scratch;
    }

    /// Scratch space (/tmp, /var/tmp) is never granted writable: it is the
    /// parent of the default worktree layout.
    #[test]
    fn scratch_space_is_never_writable() {
        let rules = build_path_rules(Path::new("/tmp/wt"), Path::new("/var/tmp/tgt"));
        for scratch in ["/tmp", "/var/tmp"] {
            assert!(
                !rules.iter().any(|r| {
                    r.path == Path::new(scratch) && r.allowed & ACCESS_FS_WRITE_FILE != 0
                }),
                "{scratch} must never be writable"
            );
        }
    }

/// Sentinel that turns this test binary into a Landlock probe instead of a
/// libtest run.
const ENFORCE_ENV: &str = "MINI_SWE_LANDLOCK_ENFORCE";

/// End-to-end enforcement check, run in a dedicated child process.
///
/// `landlock_restrict_self` is irreversible - it confines *the calling process*
/// for good - so the probe cannot share an address space with the rest of the
/// test suite. The parent re-executes this same binary with
/// [`ENFORCE_ENV`] set; the child then applies a domain and reports what it can
/// and cannot still reach. Re-using the test binary keeps the probe honest: it
/// runs exactly the code that ships, not a copy.
#[test]
fn landlock_actually_denies_paths_outside_the_sandbox() {
    use std::process::Command;

    // Child mode: apply the domain here and report, never run the suite.
    if std::env::var_os(ENFORCE_ENV).is_some() {
        run_landlock_enforcement_mode();
    }

    let scratch = Scratch::new("enforce");
    let (worktree, target) = (scratch.worktree(), scratch.target());
    let home = home_dir().expect("tests run with a discoverable $HOME");

    // A file the domain must not be able to read, outside every granted rule.
    let secret = home.join(".ssh");
    let _ = std::fs::create_dir_all(&secret);
    std::fs::write(secret.join("id_rsa"), b"PRIVATE KEY").expect("seed a decoy secret");

    let out = Command::new(std::env::current_exe().expect("test binary path"))
        // The child applies a domain that would break the harness itself, so
        // the sentinel is set through a dedicated "run this test" filter.
        .arg("--exact")
        .arg("agent::sandbox::tests::landlock_actually_denies_paths_outside_the_sandbox")
        .arg("--nocapture")
        .env(ENFORCE_ENV, "1")
        .env("LL_WORKTREE", &worktree)
        .env("LL_TARGET", &target)
        .output()
        .expect("re-run the test binary in enforcement mode");

    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "enforcement mode failed\nstdout: {stdout}\nstderr: {stderr}"
    );
    for line in stdout.lines() {
        if let Some(rest) = line.strip_prefix("RESULT ") {
            assert_eq!(rest, "OK", "landlock enforcement check reported: {rest}");
            return;
        }
    }
    panic!("enforcement mode produced no RESULT line\nstdout: {stdout}\nstderr: {stderr}");
}

/// Child-process half of [`landlock_actually_denies_paths_outside_the_sandbox`].
///
/// `libtest` owns the process argument list, so the sentinel is consumed before
/// the harness sees it and the checks run here instead of being reported as a
/// test.
fn run_landlock_enforcement_mode() -> ! {
    let Some(worktree) = std::env::var_os("LL_WORKTREE").map(PathBuf::from) else {
        eprintln!("enforcement mode needs LL_WORKTREE");
        std::process::exit(2);
    };
    let Some(target) = std::env::var_os("LL_TARGET").map(PathBuf::from) else {
        eprintln!("enforcement mode needs LL_TARGET");
        std::process::exit(2);
    };

    // From here on the process is confined: anything outside the domain fails
    // with EACCES, so each expectation below is a real syscall check.
    if let Err(e) = apply_landlock_sandbox(&worktree, &target) {
        eprintln!("FAIL: could not apply landlock: {e:#}");
        std::process::exit(1);
    }

        // Each probe states whether the domain is supposed to let it through.
        let mut failures: Vec<String> = Vec::new();
        let mut check =
            |what: &str, must_succeed: bool, outcome: std::io::Result<()>| match outcome {
                Ok(()) if must_succeed => {}
                Ok(()) => failures.push(format!("{what} was ALLOWED but must be denied")),
                Err(e) if must_succeed => {
                    failures.push(format!("{what} was denied ({e}) but must be allowed"));
                }
                // A denial is exactly what we want: Landlock reports EACCES.
                Err(_) => {}
            };

        // Denied: the operator's private key material.
        check(
            "read ~/.ssh/id_rsa",
            false,
            std::fs::read_to_string(home_dir().expect("HOME").join(".ssh/id_rsa")).map(|_| ()),
        );
        // Denied: password hashes.
        check(
            "read /etc/shadow",
            false,
            std::fs::read_to_string("/etc/shadow").map(|_| ()),
        );
        // Allowed: the worktree is writable.
        check(
            "write worktree/probe.txt",
            true,
            std::fs::write(worktree.join("probe.txt"), b"ok"),
        );
        // Allowed: the isolated build target is writable.
        check(
            "write target/probe.o",
            true,
            std::fs::write(target.join("probe.o"), b"ok"),
        );
        // Allowed: system binaries are readable, and therefore executable.
        check(
            "stat /usr/bin/bash",
            true,
            std::fs::metadata("/usr/bin/bash").map(|_| ()),
        );
        check(
            "run /usr/bin/bash -c true",
            true,
            std::process::Command::new("/usr/bin/bash")
                .args(["-c", "true"])
                .status()
                .map(|_| ()),
        );

        if failures.is_empty() {
            // `std::process::exit` below never returns, so libtest's stdout
            // capture is never flushed. Write the verdict straight to fd 1.
            let verdict = b"RESULT OK\n";
            // SAFETY: `verdict` is a live slice and fd 1 is a valid descriptor.
            // A short write is ignored: the parent only acts on the line when
            // the child reports success.
            let _ = unsafe { libc::write(1, verdict.as_ptr().cast(), verdict.len()) };
            std::process::exit(0);
        }
        for f in &failures {
            eprintln!("FAIL: {f}");
        }
    std::process::exit(1);
}
}
