//! Guardrails for agent command execution.
//!
//! Two kinds of protection live here:
//!
//! * **Command-text guardrails** - [`validate_bash_command`] rejects the
//!   obvious escapes before a command is ever run, and [`is_heavy_command`]
//!   classifies a command so it gets a longer timeout.
//! * **Kernel-level filesystem confinement** - the [Landlock] LSM restricts
//!   the filesystem to a read-only system prefix plus an explicitly writable
//!   worktree and build target directory. Two entry points share one policy:
//!   [`apply_landlock_sandbox`] confines the *calling* process (and everything
//!   it later spawns), while [`build_landlock_plan`] prepares the same domain
//!   for installation in a forked child through a `pre_exec` hook - which is
//!   how `exec.rs` confines a worker when bubblewrap is unavailable, without
//!   ever confining the daemon itself.
//!
//! [Landlock]: https://docs.kernel.org/userspace-api/landlock.html

use anyhow::{Context, Result};
#[cfg(unix)]
use std::ffi::CString;
#[cfg(unix)]
use std::os::unix::ffi::OsStrExt;
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
pub const TRUNCATE_MARKER: &str = "\n... [Truncated ";
/// Literal text of the second half of the marker, after the byte count.
pub const TRUNCATE_MARKER_SUFFIX: &str = " bytes] ...\n";
/// Upper bound on the decimal digits of a `usize` (2^64 - 1 has 20 digits).
/// Used to size the result buffer up front so the marker needs no allocation.
pub const USIZE_MAX_DIGITS: usize = 20;

/// Bound output to [`TRUNCATE_LIMIT`] bytes, keeping head and tail and
/// reporting bytes discarded. Cut points snap to UTF-8 boundaries so no
/// character splits and `head + dropped + tail == input.len()` holds.
/// Assembles once into an exactly-sized `String` (marker pushed directly,
/// byte count rendered into a stack buffer, tail never copied twice).
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

/// Reject commands that escape the worktree or recursively scan root/home.
pub fn validate_bash_command(command: &str) -> Result<(), &'static str> {
    let trimmed = command.trim();

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

/// True for CPU-heavy commands (builds, test runners) vs lightweight ones.
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

/// Whether the `bwrap` sandbox utility is available.
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

/// Resolve a worktree's gitdir reference to the common `.git` and worktree gitdir.
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

/// Return only the common `.git` root.
pub fn find_git_common_dir(worktree_dir: &Path) -> Option<PathBuf> {
    find_git_dirs(worktree_dir).map(|(common, _)| common)
}

// ---------------------------------------------------------------------------
// Landlock LSM filesystem confinement
// ---------------------------------------------------------------------------

/// Env var that force-disables Landlock confinement (mirrors `SWE_DISABLE_SANDBOX`).
pub const DISABLE_LANDLOCK_ENV: &str = "SWE_DISABLE_LANDLOCK";

/// System prefixes a sandboxed build may read (never write).
///
/// Landlock is a whitelist LSM: once rights are handled, every uncovered path
/// is denied. Read-only plus `EXECUTE` keeps the toolchain usable while the
/// rest of the filesystem stays unreachable.
const READ_ONLY_SYSTEM_PATHS: &[&str] =
    &["/usr", "/bin", "/sbin", "/lib", "/lib64", "/lib32", "/opt"];

/// Home-relative credential stores that must never be reachable, even read-only.
///
/// Denied by omission (no rule ever covers them); kept explicit so the intent
/// is testable rather than emergent.
const DENIED_HOME_SUBDIRS: &[&str] = &[".ssh", ".aws", ".gnupg", ".gpg", ".kube", ".docker"];

/// Absolute system paths that must never be reachable, even read-only.
const DENIED_ABSOLUTE_PATHS: &[&str] = &["/root", "/etc/shadow", "/etc/gshadow", "/etc/sudoers"];

/// Individual `/etc` files a sandboxed build may read.
///
/// Landlock is allow-only, so a rule on `/etc` would also grant `/etc/shadow`.
/// Granting files individually keeps the toolchain working while secrets stay
/// unreachable by omission. All entries are non-secret config files.
const CONFIG_PATHS: &[&str] = &[
    "/etc/passwd",
    "/etc/group",
    "/etc/nsswitch.conf",
    "/etc/resolv.conf",
    "/etc/hosts",
    "/etc/host.conf",
    "/etc/gai.conf",
    "/etc/ld.so.cache",
    "/etc/localtime",
    "/etc/os-release",
    "/etc/protocols",
    "/etc/services",
    "/etc/ssl/certs",
    "/etc/ca-certificates",
    "/etc/pki/tls/certs",
];

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

/// Number of `LANDLOCK_ACCESS_FS_*` rights in the UAPI; exclusive upper bound
/// on a valid right's bit index.
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

/// Oldest Landlock ABI this implementation accepts.
///
/// ABI 1 is the initial release; anything below it does not exist and is
/// indistinguishable from "unsupported", so both degrade the same way.
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
/// The UAPI lets the struct grow across ABI versions and validates `size`, so
/// only `handled_access_fs` is populated and only its size is passed. A
/// `size == 0` query returns the highest supported ABI.
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

/// An owned file descriptor that never escapes into a child process.
///
/// Two complementary mechanisms: `open(2)` uses `O_CLOEXEC` where the crate
/// chooses the descriptor itself, and descriptors the *kernel* hands back
/// (notably `landlock_create_ruleset`, which takes no flags) get `FD_CLOEXEC`
/// set explicitly, since `O_CLOEXEC` is not retroactive. Without this, a
/// ruleset fd opened before `Command::spawn` would be inherited by every
/// worker, leaking a handle to a live policy. Dropping closes it exactly once.
struct Fd(libc::c_int);

impl Fd {
    /// Adopt a raw descriptor, marking it close-on-exec.
    ///
    /// Returns `None` if `FD_CLOEXEC` cannot be set, so a descriptor that could
    /// leak into a child is never held.
    fn new(raw: libc::c_int) -> Option<Self> {
        if raw < 0 {
            return None;
        }
        // SAFETY: `raw` is a live descriptor owned by this function until the
        // `Fd` below takes over. `F_GETFD`/`F_SETFD` only read and write the
        // descriptor's own flag word, and the previous value is preserved.
        unsafe {
            let flags = libc::fcntl(raw, libc::F_GETFD);
            if flags < 0 || libc::fcntl(raw, libc::F_SETFD, flags | libc::FD_CLOEXEC) < 0 {
                libc::close(raw);
                return None;
            }
        }
        Some(Self(raw))
    }

    /// The raw descriptor, for passing into a syscall.
    fn raw(&self) -> libc::c_int {
        self.0
    }
}

impl Drop for Fd {
    fn drop(&mut self) {
        // SAFETY: the descriptor came from a successful `open`/syscall, is owned
        // by this value, and is closed exactly once here.
        unsafe { libc::close(self.0) };
    }
}

/// Rights a sandboxed child needs to read a path: list dirs, read files,
/// resolve sockets, execute binaries.
const READ_ONLY_RIGHTS: u64 =
    ACCESS_FS_READ_FILE | ACCESS_FS_READ_DIR | ACCESS_FS_EXECUTE | ACCESS_FS_RESOLVE_UNIX;

/// Rights a sandboxed child needs to build in its worktree/target dir.
///
/// Device nodes and UNIX sockets are deliberately excluded: nothing in a build
/// needs `mknod` or `bind(2)`, and dropping them costs little capability.
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

/// Filesystem rights introduced per Landlock ABI version.
///
/// A kernel rejects a ruleset asking for a right it does not implement, so the
/// mask is narrowed to the running ABI before `landlock_create_ruleset`. The
/// table lists only rights *introduced* per release; the result is cumulative.
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
/// Returns the raw `c_long` (a ruleset fd for `create_ruleset`, `0` otherwise)
/// or `-1` with `errno` set. All three take up to four arguments; the shorter
/// ones ignore the trailing zero. Passing `flags` explicitly matters: an
/// uninitialised fourth register makes `add_rule` fail intermittently with
/// `EINVAL` depending on whatever garbage the caller left in `r10`.
fn landlock_syscall(number: libc::c_long, args: [libc::c_long; 4]) -> i64 {
    // SAFETY: the Landlock syscalls take plain `c_long`-sized arguments.
    // Pointers are either null or reference live, correctly sized and aligned
    // stack structs that outlive the call, and the kernel only ever reads
    // `size` bytes from them.
    unsafe { libc::syscall(number, args[0], args[1], args[2], args[3]) }
}

/// Query the highest Landlock ABI the running kernel implements.
///
/// Returns `None` when Landlock is compiled out, disabled via `lsm=`, or
/// blocked by seccomp - all surface as a failed syscall, none worth
/// propagating to the caller.
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

/// The subset of [`ALL_ACCESS_FS`] a kernel of the given ABI implements.
///
/// Unknown/newer ABIs get every right this crate knows: a future kernel is a
/// superset, and an unrequested right costs nothing.
fn supported_access_fs(abi: i64) -> u64 {
    let mut mask = 0;
    for (introduced_in, rights) in ABI_ACCESS_FS_INTRODUCED {
        if abi >= *introduced_in {
            mask |= *rights;
        }
    }
    // ABI 0 means "no Landlock", so an empty mask is correct; the caller never
    // gets that far because `query_abi_version` rejects it first.
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
/// Intentionally *not* added as Landlock rules: no rule covers them, so the
/// handled rights deny them. An explicit, testable list stops a broad prefix
/// rule from silently granting them by accident.
fn denied_paths() -> Vec<PathBuf> {
    let mut denied = Vec::with_capacity(DENIED_ABSOLUTE_PATHS.len() + DENIED_HOME_SUBDIRS.len());
    denied.extend(DENIED_ABSOLUTE_PATHS.iter().map(PathBuf::from));
    if let Some(home) = home_dir() {
        denied.extend(DENIED_HOME_SUBDIRS.iter().map(|d| home.join(d)));
    }
    denied
}

/// True when `path` is `denied` or lives underneath it.
///
/// Purely lexical and sound because every caller-supplied path was already
/// resolved by [`canonical_root`]. The fixed system prefixes are literal
/// absolute paths with no symlink components, so string comparison matches
/// what the kernel opens.
fn is_denied(path: &Path, denied: &[PathBuf]) -> bool {
    denied.iter().any(|d| path == d || path.starts_with(d))
}

/// Resolve a caller-supplied writable root to the real directory it names.
///
/// The two writable roots are the only policy paths from outside the crate,
/// hence the only ones a symlink can influence. The kernel resolves a
/// `PATH_BENEATH` rule onto the real inode but [`is_denied`] compares path
/// strings, so a worktree that is really a symlink to `~/.ssh` would pass the
/// lexical check and hand the sandbox a write grant over the operator's keys.
/// Canonicalising makes the string check and the rule agree.
///
/// An un-canonicalisable root (missing or unreadable component) is returned
/// unchanged: the caller already validated existence, and the `O_PATH` open in
/// [`add_rule`] is the real check. Falling back is the safe direction - it
/// keeps current behaviour instead of silently dropping a writable root.
fn canonical_root(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

/// Rights meaningful only on a *non-directory* object.
///
/// The complement of the rights `PATH_BENEATH` rules may carry on a directory.
/// The kernel rejects a rule holding any of these with `EINVAL` when the
/// referenced object is not a directory; it does not merely ignore them.
const NON_DIRECTORY_RIGHTS: u64 = ACCESS_FS_READ_DIR
    | ACCESS_FS_REMOVE_DIR
    | ACCESS_FS_REMOVE_FILE
    | ACCESS_FS_MAKE_CHAR
    | ACCESS_FS_MAKE_DIR
    | ACCESS_FS_MAKE_REG
    | ACCESS_FS_MAKE_SOCK
    | ACCESS_FS_MAKE_FIFO
    | ACCESS_FS_MAKE_BLOCK
    | ACCESS_FS_MAKE_SYM
    | ACCESS_FS_REFER
    | ACCESS_FS_TRUNCATE;

/// Narrow a rule's rights to what the kernel accepts for `path`.
///
/// A `PATH_BENEATH` rule may carry directory-only rights only when naming a
/// *directory*; against a regular file or character device the kernel fails
/// the whole `landlock_add_rule` with `EINVAL`.
///
/// "Not a directory" is not "is a regular file": `Path::is_file` answers
/// `false` for `/dev/null` and the other sinks, so asking it would let a
/// `READ_DIR` bit reach the kernel on exactly the nodes that most need it
/// filtered - and an `is_file` branch that keeps directory rights would hand
/// `/dev/null` a `MAKE_*` grant. Masking by [`NON_DIRECTORY_RIGHTS`] asks the
/// kernel's actual question.
///
/// The survivors - `READ_FILE`, `WRITE_FILE`, `EXECUTE`, `IOCTL_DEV`,
/// `RESOLVE_UNIX` - are all meaningful on a character device, which keeps
/// `> /dev/null` working. An un-stat-able path is treated as a non-directory:
/// the narrower mask is the safe direction to be wrong in.
fn rights_for(rule_path: &Path, allowed: u64) -> u64 {
    match std::fs::metadata(rule_path) {
        Ok(meta) if meta.is_dir() => allowed,
        Ok(_) | Err(_) => allowed & !NON_DIRECTORY_RIGHTS,
    }
}

/// Build the `PATH_BENEATH` rules for a sandboxed child.
///
/// Landlock is allow-only: every granted path widens access, so the policy
/// grants read-only system prefixes plus exactly two writable roots and
/// nothing else. Sensitive paths stay unreachable by omission (see
/// [`denied_paths`]).
///
/// Every path is filtered through [`is_denied`], so a denied directory can
/// never be granted even if reachable from an allowed prefix. The two
/// caller-supplied roots are canonicalised first ([`canonical_root`]) so a
/// symlink cannot point one at a denied directory.
fn build_path_rules(worktree: &Path, target_dir: &Path) -> Vec<PathRule> {
    let denied = denied_paths();
    let capacity = READ_ONLY_SYSTEM_PATHS.len() + CONFIG_PATHS.len() + 3 + 2 + 2;
    let mut rules = Vec::with_capacity(capacity);

    let mut push = |path: PathBuf, allowed: u64| {
        if path.is_absolute() && !is_denied(&path, &denied) {
            rules.push(PathRule { path, allowed });
        }
    };

    // System prefixes: readable and executable, never writable.
    for sys in READ_ONLY_SYSTEM_PATHS {
        push(PathBuf::from(sys), READ_ONLY_RIGHTS);
    }

    // Landlock is allow-only: a rule on /etc would also grant /etc/shadow.
    // Grant only the individual non-secret configuration files needed by tools.
    for config in CONFIG_PATHS {
        push(PathBuf::from(config), ACCESS_FS_READ_FILE | ACCESS_FS_READ_DIR);
    }

    // Pseudo-filesystems, always readable.
    push(PathBuf::from("/dev"), READ_ONLY_RIGHTS);
    push(PathBuf::from("/proc"), READ_ONLY_RIGHTS);

    // Redirection sinks must stay writable or `> /dev/null` breaks in almost
    // every shell command. The grant is limited to the null/zero/full sinks:
    // writing there discards data anyway, and `MAKE_CHAR` is not granted, so
    // this is not a path back to arbitrary device I/O. The broader `/dev`
    // rule above stays read-only.
    for sink in ["/dev/null", "/dev/zero", "/dev/full"] {
        push(PathBuf::from(sink), READ_ONLY_RIGHTS | ACCESS_FS_WRITE_FILE);
    }

    // The only caller-supplied paths, hence the only ones that can be
    // symlinks: canonicalised first (see [`canonical_root`]) so the
    // deny-check and the rule name the same real directory.
    push(canonical_root(worktree), WRITE_RIGHTS);
    push(canonical_root(target_dir), WRITE_RIGHTS);

    rules
}

/// Union of every right any rule asks for, narrowed to what the kernel
/// supports: the ruleset's `handled_access_fs` mask.
///
/// Handled rights turn Landlock's "allow" model into "deny by default", so
/// this is also the set of operations an ungranted path loses.
fn handled_access_fs(rules: &[PathRule], access: u64) -> u64 {
    rules
        .iter()
        .fold(0, |acc, rule| acc | rule.allowed)
        & access
}

/// Apply a Landlock filesystem domain to the calling process and its children.
///
/// `landlock_restrict_self` is **irreversible and one-way**: the process cannot
/// widen its own access afterwards, and every fork inherits the restriction.
/// That is what worker execution wants, but it is also why this is the *only*
/// place in the crate that should call it - a caller must be done touching
/// anything outside the sandbox, because its own subsequent access is confined
/// too.
///
/// The domain permits:
///
/// * **read + execute** on the system prefixes ([`READ_ONLY_SYSTEM_PATHS`]),
///   selected `/etc` config files, `/proc` and `/dev` - enough to run a
///   compiler, linker and `bash`;
/// * **read + write** on `worktree` and `target_dir`.
///
/// Everything else is denied, which is what makes the sensitive paths (the
/// operator's `~/.ssh`, `~/.aws`, `~/.gnupg`, `/etc/shadow`; see
/// [`denied_paths`]) unreachable rather than merely unused.
///
/// # Graceful degradation
///
/// Landlock is absent from kernels older than 5.13 and can be disabled at boot
/// (`lsm=` without `landlock`) or blocked by seccomp. None of those is a reason
/// to fail a worker: when the kernel cannot support the domain, this logs at
/// debug level and returns `Ok(())`, leaving the process unconfined. Only a
/// *malformed policy* - a path we were told to sandbox that does not exist - is
/// reported as an `Err`.
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
        anyhow::bail!(
            "landlock target dir does not exist: {}",
            target_dir.display()
        );
    }

    apply_with_abi(worktree, target_dir, query_abi_version())
}

/// Body of [`apply_landlock_sandbox`], with the ABI probe as a parameter.
///
/// Taking the ABI as an argument makes the "kernel has no Landlock" branch
/// reachable from a test: on a kernel that *does* support Landlock the probe
/// can never return `None`, so the degradation path would otherwise be dead
/// code no test executes. `abi == None` is exactly the state a pre-5.13 or
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
                add_rule(ruleset_fd.raw(), rule)?;
            }
            // Landlock requires PR_SET_NO_NEW_PRIVS before restrict_self unless
            // the process has CAP_SYS_ADMIN.
            // SAFETY: prctl with PR_SET_NO_NEW_PRIVS takes integer arguments.
            if unsafe { libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) } != 0 {
                anyhow::bail!(
                    "prctl PR_SET_NO_NEW_PRIVS: {}",
                    std::io::Error::last_os_error()
                );
            }

            // SAFETY: `ruleset_fd` is still open (closed only when this
            // closure's `Fd` drops) and no other thread closed it.
            let ret = landlock_syscall(
                SYS_LANDLOCK_RESTRICT_SELF,
                [ruleset_fd.raw().into(), 0, 0, 0],
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

// ----------
// Signal-safe pre-exec plan
// ----------

/// A fully prepared, allocation-free description of a Landlock domain.
///
/// # Why this exists
///
/// [`apply_landlock_sandbox`] is not safe to call from a `Command::pre_exec`
/// closure: that closure runs in the child between `fork(2)` and `exec(2)`,
/// where only async-signal-safe operations are permitted. Everything the
/// policy needs - canonicalising paths, `CString` construction, `Vec` growth,
/// `tracing`, `anyhow` formatting - allocates, and allocating in a forked child
/// of a multi-threaded server can deadlock on the allocator lock some other
/// thread held at the instant of the fork.
///
/// [`LandlockPlan`] moves all of that into the *parent* and leaves the child
/// nothing but raw syscalls (`open`, `landlock_create_ruleset`,
/// `landlock_add_rule`, `prctl`, `landlock_restrict_self`, `close`), none of
/// which allocates, locks or logs, so the closure is genuinely
/// async-signal-safe.
///
/// [`build_landlock_plan`] returns `Ok(None)` when the kernel has no Landlock
/// or confinement is disabled - the same graceful degradation
/// [`apply_landlock_sandbox`] performs - so the caller registers no `pre_exec`
/// hook at all.
pub struct LandlockPlan {
    /// Bitmask of `ACCESS_FS_*` rights the ruleset handles.
    handled: u64,
    /// One pre-resolved, NUL-terminated rule path per rule.
    ///
    /// Allocated in the parent and inherited across the fork, so the child
    /// never builds a string of its own.
    paths: Vec<CString>,
    /// Rights granted for `paths[i]`, one entry per rule.
    allowed: Vec<u64>,
}

impl std::fmt::Debug for LandlockPlan {
    /// Summarise the plan without dumping every rule path.
    ///
    /// A plan is a security policy: printing it wholesale would turn a debug
    /// line into a host directory listing, so only the shape is exposed.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LandlockPlan")
            .field("handled", &self.handled)
            .field("rules", &self.paths.len())
            .finish()
    }
}

impl LandlockPlan {
    /// Absolute path of rule `index`, or `None` when absent.
    ///
    /// Exposed so a caller/test can verify a named root is granted without
    /// reaching into the plan's representation.
    pub fn rule_path(&self, index: usize) -> Option<&std::path::Path> {
        use std::os::unix::ffi::OsStrExt as _;
        let bytes = self.paths.get(index)?.as_bytes();
        // NUL-terminated on construction and a `CString` cannot hold an
        // interior NUL, so the slice up to the terminator is the original path.
        let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
        Some(std::path::Path::new(std::ffi::OsStr::from_bytes(
            &bytes[..end],
        )))
    }

    /// Number of `PATH_BENEATH` rules this plan will install.
    pub fn rule_count(&self) -> usize {
        self.paths.len()
    }

    /// Bitmask of `ACCESS_FS_*` rights the ruleset will handle.
    pub fn handled_access(&self) -> u64 {
        self.handled
    }

    /// Apply the domain to the calling process.
    ///
    /// # Safety
    ///
    /// Only sound between `fork(2)` and `exec(2)`, where the calling process is
    /// single-threaded by construction. Kept to raw syscalls so it is
    /// async-signal-safe, but **irreversible**: `landlock_restrict_self`
    /// confines the calling process for good, along with everything it forks.
    pub(crate) unsafe fn apply(&self) -> std::io::Result<()> {
        // Handled mask was already narrowed to the running ABI by
        // `build_landlock_plan`, so the kernel cannot reject it.
        let attr = RulesetAttr {
            handled_access_fs: self.handled,
        };
        // SAFETY: `attr` is a live, correctly laid out `landlock_ruleset_attr`
        // prefix and `size` is its exact size, which is what the kernel
        // validates. Both are stack locals that outlive the call.
        let ruleset_fd = {
            landlock_syscall(
                SYS_LANDLOCK_CREATE_RULESET,
                [
                    &attr as *const RulesetAttr as libc::c_long,
                    std::mem::size_of::<RulesetAttr>() as libc::c_long,
                    0,
                    0,
                ],
            )
        };
        if ruleset_fd < 0 {
            return Err(std::io::Error::last_os_error());
        }
        // The descriptor belongs to the child alone (this ran after the fork),
        // so it is closed exactly once on the way out rather than through `Fd`,
        // whose `Drop` would add a branch and flag dance to signal-unsafe code.
        // It must stay open until `landlock_restrict_self` consumes it.

        // `O_PATH` needs no permission on the target itself, only traversal of
        // its parents, so a rule installs even on paths this process could not
        // open for reading. `O_CLOEXEC` closes the leak window into the exec'd
        // image.
        for (c_path, &allowed_access) in self.paths.iter().zip(&self.allowed) {
            // SAFETY: `c_path` is a NUL-terminated OS string that outlives the
            // call, as `open(2)` requires.
            let parent_fd = unsafe { libc::open(c_path.as_ptr(), libc::O_PATH | libc::O_CLOEXEC) };
            if parent_fd < 0 {
                // A path that cannot be opened grants nothing; skipping keeps
                // the sandbox working on minimal images lacking, say, `/opt`.
                continue;
            }

            let rule_attr = PathBeneathAttr {
                allowed_access,
                parent_fd,
            };
            // SAFETY: `rule_attr` is a live, correctly laid out
            // `landlock_path_beneath_attr` and `parent_fd` stays open for the
            // duration of the call. The fourth argument is the explicit
            // `flags` register: leaving it uninitialised makes this call fail
            // intermittently with `EINVAL` depending on the garbage the caller
            // happened to leave in `r10`.
            let ret = {
                landlock_syscall(
                    SYS_LANDLOCK_ADD_RULE,
                    [
                        ruleset_fd,
                        RULE_PATH_BENEATH.into(),
                        &rule_attr as *const PathBeneathAttr as libc::c_long,
                        0,
                    ],
                )
            };
            // SAFETY: `parent_fd` is owned by this loop iteration and is not
            // referenced again after the call above.
            unsafe { libc::close(parent_fd) };
            if ret < 0 {
                // Close the ruleset before bailing: leaving it open would leak
                // a handle to a live policy if the hook were ever reused.
                //
                // SAFETY: `ruleset_fd` is owned by this function and is not
                // referenced again.
                unsafe { libc::close(ruleset_fd as libc::c_int) };
                return Err(std::io::Error::last_os_error());
            }
        }

        // Landlock refuses to install a domain on a process that could still
        // regain privilege through `execve`, so `no_new_privs` is a hard
        // prerequisite, not a hardening nicety.
        // SAFETY: `prctl(PR_SET_NO_NEW_PRIVS)` takes plain integers and is
        // async-signal-safe.
        if unsafe { libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) } != 0 {
            // SAFETY: `ruleset_fd` is owned by this function and unused after
            // this point.
            unsafe { libc::close(ruleset_fd as libc::c_int) };
            return Err(std::io::Error::last_os_error());
        }

        // One-way door: confine this process and everything it forks.
        // SAFETY: the ruleset descriptor is live and fully populated here, and
        // `restrict_self` takes no other argument.
        let ret = { landlock_syscall(SYS_LANDLOCK_RESTRICT_SELF, [ruleset_fd, 0, 0, 0]) };
        // `landlock_restrict_self` consumed the ruleset, so the descriptor is
        // dead weight. Closed on both outcomes: on success the child must not
        // carry a policy handle into the exec'd image; on failure the hook is
        // about to abort the spawn.
        //
        // SAFETY: `ruleset_fd` is owned by this function and referenced no
        // further.
        unsafe { libc::close(ruleset_fd as libc::c_int) };
        if ret < 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(())
    }
}

/// Build a [`LandlockPlan`] for `worktree` and `target_dir`, or `Ok(None)` when
/// this process cannot be confined.
///
/// The two conditions that make a plan impossible are the two that make the
/// sandbox optional, and they are deliberately the *same* ones
/// [`apply_landlock_sandbox`] treats as graceful:
///
/// * `SWE_DISABLE_LANDLOCK=1` - the operator asked for no confinement;
/// * the Landlock syscalls are unavailable (a pre-5.13 kernel, a build
///   without `CONFIG_SECURITY_LANDLOCK`, `lsm=` without `landlock`, or a
///   seccomp policy that kills them). `query_abi_version` reports all of them
///   as `None` and the worker runs unconfined rather than failing.
///
/// A malformed policy is *not* in that list: a worktree or target dir that
/// does not exist is a caller bug and is reported as an `Err`, exactly as
/// [`apply_landlock_sandbox`] reports it. A caller that only ever gets here
/// with directories the parent just created never reaches that branch.
///
/// Everything this does - the ABI probe, canonicalisation, `CString`
/// construction, the syscall-backed existence checks - runs in the parent
/// process, where allocating is safe. The returned plan holds no file
/// descriptor, so nothing leaks into the child through an inherited one.
pub fn build_landlock_plan(worktree: &Path, target_dir: &Path) -> Result<Option<LandlockPlan>> {
    build_plan_with_abi(worktree, target_dir, query_abi_version())
}

/// The body of [`build_landlock_plan`], with the ABI probe as a parameter.
///
/// This mirrors the [`apply_with_abi`] seam and exists for the same reason.
/// `query_abi_version` is a syscall against the machine running the test suite,
/// which supports Landlock, so the "this kernel cannot confine" branch can
/// never be taken there - leaving the single most important robustness
/// property of the whole feature (a worker still runs on a host with no
/// Landlock) as code no test ever executes. `abi == None` is exactly the state
/// a pre-5.13, `lsm=`-disabled or seccomp-blocked kernel puts us in, so the
/// branch is driven directly instead of being left to chance.
fn build_plan_with_abi(
    worktree: &Path,
    target_dir: &Path,
    abi: Option<i64>,
) -> Result<Option<LandlockPlan>> {
    if landlock_disabled() {
        tracing::debug!("landlock confinement disabled by {DISABLE_LANDLOCK_ENV}=1");
        return Ok(None);
    }

    // A rule on a missing path is a caller bug; silently granting access to a
    // directory that is not there would only hide it until the first write
    // fails somewhere far less obvious.
    if !worktree.is_dir() {
        anyhow::bail!("landlock worktree does not exist: {}", worktree.display());
    }
    if !target_dir.is_dir() {
        anyhow::bail!(
            "landlock target dir does not exist: {}",
            target_dir.display()
        );
    }

    // A kernel without Landlock is not a reason to fail a worker: the caller
    // gets `None`, registers no hook, and the command runs unconfined.
    let Some(abi) = abi else {
        tracing::debug!("landlock unsupported by this kernel; running unconfined");
        return Ok(None);
    };

    let rules = build_path_rules(worktree, target_dir);
    let handled = handled_access_fs(&rules, supported_access_fs(abi));

    // Counts are fixed by the policy, so size both vectors up front.
    let mut paths = Vec::with_capacity(rules.len());
    let mut allowed = Vec::with_capacity(rules.len());
    for rule in &rules {
        // A path with an interior NUL cannot be a C string; a rule we cannot
        // name must not be pretended installed.
        let Ok(c_path) = CString::new(rule.path.as_os_str().as_bytes()) else {
            tracing::debug!(
                path = %rule.path.display(),
                "skipping unusable landlock rule path"
            );
            continue;
        };
        // Directory-only rights are meaningless on a non-directory; the kernel
        // rejects a rule carrying them rather than masking them out.
        let allowed_access = rights_for(&rule.path, rule.allowed);
        paths.push(c_path);
        allowed.push(allowed_access);
    }

    tracing::debug!(
        abi,
        handled,
        rules = paths.len(),
        "prepared landlock plan for the exec pre_exec hook"
    );
    Ok(Some(LandlockPlan {
        handled,
        paths,
        allowed,
    }))
}

/// Add one `PATH_BENEATH` rule to an open ruleset.
fn add_rule(ruleset_fd: libc::c_int, rule: &PathRule) -> Result<()> {
    // `O_PATH` needs no permission on the target itself (only traversal of
    // its parents), so a rule installs even on paths the caller could not
    // `File::open` for reading. A genuinely missing path has nothing to
    // protect; skipping keeps the sandbox working on minimal images.
    let Some(c_path) = CString::new(rule.path.as_os_str().as_bytes()).ok() else {
        return Ok(());
    };
    // SAFETY: `c_path` is a NUL-terminated OS string that outlives the call.
    // `O_CLOEXEC` keeps the descriptor out of any child spawned later; `Fd`
    // also sets `FD_CLOEXEC` as belt-and-braces and owns the descriptor.
    let fd = Fd::new(unsafe { libc::open(c_path.as_ptr(), libc::O_PATH | libc::O_CLOEXEC) });
    let Some(parent) = fd else {
        return Ok(());
    };
    let allowed = rights_for(&rule.path, rule.allowed);
    let attr = PathBeneathAttr {
        allowed_access: allowed,
        parent_fd: parent.raw(),
    };
    // SAFETY: `attr` is a live `landlock_path_beneath_attr` and `parent` keeps
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
///
/// The returned [`Fd`] carries `FD_CLOEXEC`: the kernel hands the ruleset
/// descriptor back without it, and a leaked ruleset fd would be inherited by
/// every command the daemon spawns from here on. [`Fd`] also closes it once the
/// ruleset has been applied.
fn create_ruleset(handled_access_fs: u64) -> Result<Fd> {
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
    // A descriptor we could not mark close-on-exec must not be used, or it
    // would leak into children.
    Fd::new(ret as libc::c_int).ok_or_else(|| {
        anyhow::anyhow!(
            "landlock_create_ruleset: could not mark the ruleset fd close-on-exec: {}",
            std::io::Error::last_os_error()
        )
    })
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

/// A host with no Landlock must still run workers.
///
/// The probe is the only thing distinguishing such a machine from this one, so
/// the "unsupported" branch is driven through the [`build_plan_with_abi`] seam
/// rather than left to whatever kernel the suite runs on. A `bail!` here would
/// take every worker on every kernel without Landlock offline, buying no
/// security.
#[test]
fn a_kernel_without_landlock_yields_no_plan_instead_of_an_error() {
    let scratch = Scratch::new("noplan");

    let built = build_plan_with_abi(&scratch.worktree(), &scratch.target(), None);

    assert!(
        built.is_ok(),
        "a kernel without Landlock must degrade, not fail: {built:?}"
    );
    assert!(
        built.unwrap().is_none(),
        "no kernel support must yield *no plan*, so the caller registers no \
         pre_exec hook and the worker runs unconfined"
    );

    // Degrading means unconfined: the process keeps the access it had, which
    // is what makes this a downgrade rather than a silent partial sandbox.
    std::fs::write(scratch.worktree().join("still-writable"), b"ok")
        .expect("an unconfined process keeps its access");
}

/// The malformed-policy check must survive the unsupported-kernel path.
///
/// Ordering matters: a missing root is a caller bug reported *before* the ABI
/// probe, so a host with no Landlock still gets told its arguments are wrong.
#[test]
fn a_missing_root_is_reported_even_where_landlock_is_unavailable() {
    let scratch = Scratch::new("noplan-missing");
    let missing = scratch.0.join("does-not-exist");

    let err = build_plan_with_abi(&missing, &scratch.target(), None).unwrap_err();
    assert!(
        format!("{err:#}").contains("does not exist"),
        "a malformed policy must be reported regardless of kernel support: {err:#}"
    );
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

/// Directory-only rights must never reach the kernel on a non-directory.
///
/// A `PATH_BENEATH` rule naming a regular file or character device is rejected
/// outright with `EINVAL` when it carries them, aborting the whole spawn: a
/// policy correct on paper but rejected at install confines *nothing* and fails
/// every worker.
///
/// `/dev/null` is the case that matters - it is a character device, so the
/// obvious `Path::is_file` test answers `false`, and both failure modes are
/// live: keeping `READ_DIR` gets the rule rejected, and narrowing to
/// `READ_FILE` silently takes away the write every `> /dev/null` needs.
#[test]
fn non_directory_rules_are_narrowed_without_losing_their_write() {
    for path in ["/dev/null", "/dev/zero", "/dev/full"] {
        let p = Path::new(path);
        if !p.exists() {
            continue; // Minimal images may not ship every sink.
        }
        let narrowed = rights_for(p, READ_ONLY_RIGHTS | ACCESS_FS_WRITE_FILE);

        assert_eq!(
            narrowed & NON_DIRECTORY_RIGHTS,
            0,
            "{path} is not a directory, so no directory-only right may survive: {narrowed:#x}"
        );
        assert_ne!(
            narrowed & ACCESS_FS_WRITE_FILE,
            0,
            "{path} must stay writable, or `> {path}` breaks in every command"
        );
        assert_ne!(
            narrowed & ACCESS_FS_READ_FILE,
            0,
            "{path} must stay readable"
        );
    }
}

/// A directory keeps every right it was granted.
#[test]
fn directory_rules_keep_every_right_they_were_granted() {
    let scratch = Scratch::new("dirrights");
    let dir = scratch.target();
    assert!(dir.is_dir());
    assert_eq!(
        rights_for(&dir, WRITE_RIGHTS),
        WRITE_RIGHTS,
        "a directory must not lose any right - that would silently make the \
         worktree read-only"
    );
}

/// Redirection sinks are writable without making the rest of `/dev` so.
#[test]
fn null_sinks_are_writable_but_the_dev_pseudo_filesystem_is_not() {
    let scratch = Scratch::new("devsink");
    let rules = build_path_rules(&scratch.worktree(), &scratch.target());

    let dev = rule_for(&rules, Path::new("/dev")).expect("/dev is granted");
    assert_eq!(
        dev.allowed & ACCESS_FS_WRITE_FILE,
        0,
        "/dev itself must stay read-only: a writable pseudo-filesystem is a \
         much larger grant than a redirection sink"
    );

    for sink in ["/dev/null", "/dev/zero", "/dev/full"] {
        if !Path::new(sink).exists() {
            continue;
        }
        let rule = rule_for(&rules, Path::new(sink))
            .unwrap_or_else(|| panic!("{sink} must be granted as a sink"));
        assert_ne!(
            rule.allowed & ACCESS_FS_WRITE_FILE,
            0,
            "{sink} must be writable or every `> {sink}` fails"
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

/// Descriptors that outlive the call must never reach a child process.
///
/// `O_CLOEXEC` in `open(2)` only covers descriptors *this* code creates;
/// `landlock_create_ruleset` returns one the kernel made, with no flags it
/// could honour, so `FD_CLOEXEC` has to be set explicitly. A ruleset fd
/// inherited by a spawned worker would be a live handle to a policy the child
/// was never meant to be able to reach.
#[test]
fn landlock_descriptors_are_close_on_exec() {
    let ruleset = create_ruleset(ALL_ACCESS_FS).expect("create a ruleset");
    // SAFETY: `ruleset` owns a live descriptor for the whole block.
    let flags = unsafe { libc::fcntl(ruleset.raw(), libc::F_GETFD) };
    assert!(flags >= 0, "F_GETFD must succeed on a live descriptor");
    assert_ne!(
        flags & libc::FD_CLOEXEC,
        0,
        "the ruleset descriptor must be close-on-exec or it leaks into children"
    );
}

/// Dropping the ruleset must close the descriptor.
///
/// Before the [`Fd`] wrapper, `landlock_create_ruleset`'s descriptor was never
/// closed at all: it stayed open for the remaining life of the (long-running,
/// multi-threaded) daemon, and was inherited by every process subsequently
/// spawned. Closing on drop is what actually bounds its lifetime to the one
/// call that needs it.
#[test]
fn the_ruleset_descriptor_is_closed_when_it_goes_out_of_scope() {
    let fd = {
        let ruleset = create_ruleset(ALL_ACCESS_FS).expect("create a ruleset");
        let raw = ruleset.raw();
        // SAFETY: `raw` is a live descriptor owned by the `Fd` here.
        assert!(unsafe { libc::fcntl(raw, libc::F_GETFD) } >= 0);
        raw
    };
    // `fcntl` on a stale descriptor must now fail; a leaked one would still
    // answer, and the number was just freed so it cannot name another live
    // descriptor.
    // SAFETY: the descriptor is expected to be closed, so this probes, not uses.
    let reopened = unsafe { libc::fcntl(fd, libc::F_GETFD) };
    assert_eq!(
        reopened, -1,
        "the ruleset descriptor must be closed on drop, not leaked for the process lifetime"
    );
}

/// A symlinked writable root must not smuggle a grant onto a denied directory.
///
/// This is the whole point of [`canonical_root`]: Landlock resolves a
/// `PATH_BENEATH` rule onto the real inode, but [`is_denied`] compares path
/// strings. A root supplied as a symlink pointing at `~/.ssh` therefore looks
/// innocent to the lexical check while the rule would really cover the
/// operator's private keys. Canonicalising makes the two agree.
#[test]
fn a_symlinked_root_cannot_smuggle_a_grant_onto_a_denied_directory() {
    let scratch = Scratch::new("symlink");
    let home = home_dir().expect("tests run with a discoverable $HOME");
    let secret_dir = home.join(".ssh");
    std::fs::create_dir_all(&secret_dir).expect("create a denied directory");

    // Ordinary-looking scratch space that is really a symlink onto the
    // operator's key material.
    let link = scratch.0.join("looks-innocent");
    std::os::unix::fs::symlink(&secret_dir, &link).expect("create the symlink");

    // Without canonicalisation the lexical deny-check would pass and the rule
    // would cover ~/.ssh.
    let resolved = canonical_root(&link);
    assert_eq!(
        resolved, secret_dir,
        "canonicalisation must resolve the link to its real target"
    );

    let rules = build_path_rules(&link, &scratch.target());
    assert!(
        !rules.iter().any(|r| r.path.starts_with(&secret_dir) || r.path == secret_dir),
        "a symlinked root must never be granted once it resolves into a denied path"
    );
    assert!(
        !rules.iter().any(|r| r.path == link),
        "the un-resolved symlink path must not be granted either"
    );
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

        // The null/zero/full sinks are writable as *sinks* - see
        // `build_path_rules` - but nothing else is, and in particular neither
        // `/dev` nor `/tmp` gains a grant that would cover another worker's
        // files. The property being pinned is that the writable set is
        // exhausted by a short, explicit list: a new prefix rule would have to
        // be added here deliberately.
        const WRITABLE_SINKS: [&str; 3] = ["/dev/null", "/dev/zero", "/dev/full"];

        for sink in WRITABLE_SINKS {
            assert!(
                writable.iter().any(|w| *w == Path::new(sink)),
                "{sink} must stay usable as a redirection sink, got: {writable:?}"
            );
        }
        for path in &writable {
            let is_root = *path == worktree || *path == target;
            let is_sink = WRITABLE_SINKS.contains(&path.to_str().unwrap_or_default());
            assert!(
                is_root || is_sink,
                "{} may not be writable: only the two declared roots and the \
                 null sinks are, got: {writable:?}",
                path.display()
            );
        }
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

    // From here on the process is confined: anything outside the domain fails with EACCES.
    if let Err(e) = apply_landlock_sandbox(&worktree, &target) {
        eprintln!("FAIL: could not apply landlock: {e:#}");
        std::process::exit(1);
    }

        let mut failures: Vec<String> = Vec::new();
        let mut check =
            |what: &str, must_succeed: bool, outcome: std::io::Result<()>| match outcome {
                Ok(()) if must_succeed => {}
                Ok(()) => failures.push(format!("{what} was ALLOWED but must be denied")),
                Err(e) if must_succeed => {
                    failures.push(format!("{what} was denied ({e}) but must be allowed"));
                }
                // Denial is the desired outcome; Landlock reports EACCES.
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
            // `std::process::exit` never returns, so libtest's stdout capture
            // is never flushed; write the verdict straight to fd 1.
            let verdict = b"RESULT OK\n";
            // SAFETY: `verdict` is a live slice and fd 1 is valid. A short
            // write is ignored: the parent only acts on the line when the
            // child reports success.
            let _ = unsafe { libc::write(1, verdict.as_ptr().cast(), verdict.len()) };
            std::process::exit(0);
        }
        for f in &failures {
            eprintln!("FAIL: {f}");
        }
    std::process::exit(1);
}
}
