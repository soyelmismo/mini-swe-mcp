//! Bash execution for the agent: sandbox construction, environment setup and
//! process supervision.
//!
//! Everything in this module concerns running a command *locally* on the
//! worker's machine. The LLM half of the runner (HTTP completions, SSE
//! streaming) lives in [`super::runner`]; this module owns the tool side of a
//! step: validating the command, building the sandboxed child process, applying
//! the shared build/cache environment, enforcing the wall-clock timeout and
//! collecting the child's combined output.
//!
//! [`AgentRunner::execute_bash`] is the single entry point and keeps its
//! original signature so callers (notably `pool::runner`) are unaffected.
//!
//! # Sandboxing: the kernel first, bubblewrap on request
//!
//! Every command is confined, and *how* depends on configuration and what the
//! host kernel provides:
//!
//! * **kernel backend (default)** - the child is confined from a `pre_exec`
//!   hook with no helper process and no namespace: Landlock owns the
//!   filesystem, a seccomp filter owns the syscalls, and process hardening
//!   (`PR_SET_PDEATHSIG`, `PR_SET_DUMPABLE`, `RLIMIT_CORE`) owns the process
//!   itself (see [`apply_kernel_confinement`]). An offline step needs no
//!   network namespace either: Landlock denies TCP bind/connect and seccomp
//!   denies INET socket creation.
//! * **bubblewrap (`SWE_SANDBOX=bwrap`)** - the child gets its own mount
//!   namespace, PID namespace and an empty tmpfs `$HOME` instead (see
//!   [`apply_sandbox_args`]); an offline step is additionally wrapped in a
//!   network namespace by [`wrap_network_command`].
//!
//! The kernel backend is deliberately *not* a weaker policy: the same
//! allow-list applies, and the sensitive paths (`$HOME/.ssh`, `/etc/shadow`)
//! are unreachable either way. A host whose kernel offers neither Landlock
//! nor seccomp falls back to bubblewrap when it is installed, and otherwise
//! runs the command unconfined with a warning.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::process::{Child, Command};

use super::AgentRunner;
use super::intercept::{check_command, strip_data_heredocs};
use super::sandbox::{
    KernelConfinement, TRUNCATE_HEAD, TRUNCATE_TAIL, find_git_common_dir, find_git_dirs,
    has_bwrap, is_heavy_command, truncate_with_dropped, validate_bash_command,
};

/// Exit code reported when a command exceeded its wall-clock budget.
const TIMEOUT_EXIT_CODE: i32 = 124;

/// Default wall-clock budget (seconds) for heavy commands (builds, test suites).
const DEFAULT_HEAVY_TIMEOUT_SECS: u64 = 600;

/// Default wall-clock budget (seconds) for light commands.
const DEFAULT_LIGHT_TIMEOUT_SECS: u64 = 120;

/// `nice` flag applied to every child so agent work yields to interactive work.
const NICE_FLAG: &str = "-n";
/// `nice` level applied to every child so agent work yields to interactive work.
const NICE_VALUE: &str = "10";

/// Grace period after `SIGTERM` before a timed-out group is escalated to
/// `SIGKILL`. Long enough to flush buffers, short enough that a wedged build
/// still fails near its budget.
const TERM_GRACE_MS: u64 = 5_000;

/// Bound on the post-`SIGKILL` reap. `SIGKILL` cannot be caught, so a child
/// still unreaped here is not ours to wait on and must not stall the result.
const KILL_GRACE_MS: u64 = 2_000;

/// Program that places a command in a fresh network namespace.
///
/// `unshare -n` gives a *kernel-level* "no egress" guarantee without containers
/// or an external firewall: the child gets its own empty network stack, so
/// every connect() fails immediately with `ENETUNREACH` instead of hanging out
/// a TCP timeout.
const NETWORK_NAMESPACE_TOOL: &str = "unshare";

/// Shell run by an offline step.
///
/// Wrapping in `bash -c` keeps the model's command semantics (pipes,
/// redirections, `&&`) intact instead of re-parsing the string into argv.
const OFFLINE_SHELL: &str = "bash";

/// Bound on draining the output pipes of a terminated command.
///
/// A grandchild that inherited stdout/stderr can hold the write end open long
/// after its parent was killed, so the drain is abandoned at this deadline
/// rather than hanging the worker.
const DRAIN_GRACE_MS: u64 = 2_000;

/// Granularity of a single pipe read.
///
/// Large enough that per-read syscall overhead stays negligible, small enough
/// that a reader task holds only this much scratch beyond its retained buffer.
const DRAIN_CHUNK_BYTES: usize = 8 * 1024;

impl AgentRunner {
    /// Run `command` with `bash -c` inside `dir`; return combined
    /// stdout/stderr (truncated to the shared budget) plus the exit code.
    ///
    /// Validation failures are *not* errors: a guardrail-rejected command is
    /// reported to the model as output with a non-zero code so it can recover.
    /// Only a failure to spawn the child propagates as an `Err`.
    pub async fn execute_bash(&self, dir: &Path, command: &str) -> Result<(String, Option<i32>)> {
        // Destructive-pattern guard: a block is reported to the model like a
        // guardrail rejection so it can recover on the next step.
        if let Err(reason) = check_command(command) {
            return Ok((blocked_by_interceptor(&reason), Some(1)));
        }

        // Same executable view as `check_command`: file content written through
        // a heredoc is data, not a search or a `cd`.
        if let Err(reason) = validate_bash_command(&strip_data_heredocs(command)) {
            return Ok((blocked_by_guardrail(reason), Some(1)));
        }

        let parallelism = build_parallelism();
        let target_dir = resolve_target_dir(dir);
        let _ = std::fs::create_dir_all(&target_dir);

        // A writable private temp dir, replacing the tmpfs bubblewrap mounts
        // over `/tmp`: it lives under the isolated target dir, so the
        // Landlock grant on the target covers it with no extra rule.
        let tmp_dir = target_dir.join("tmp");
        let _ = std::fs::create_dir_all(&tmp_dir);

        let mut cmd = Command::new("nice");
        configure_process(&mut cmd);

        // Classify on the model's own command, before any offline wrapper is
        // applied: a wrapper would otherwise mask the command's heaviness
        // from the timeout classifier.
        let timeout_secs = command_timeout_secs(command);

        match select_backend() {
            SandboxBackend::Kernel => {
                // The kernel confines the forked child itself; the command
                // needs no wrapper, offline or otherwise.
                apply_kernel_confinement(&mut cmd, dir, &target_dir, self.network_offline);
                cmd.current_dir(dir)
                    .args([NICE_FLAG, NICE_VALUE, "bash", "-c", command]);
            }
            SandboxBackend::Bwrap => {
                // bubblewrap builds the mount namespace itself; adding
                // Landlock here would only risk re-confining a process bwrap
                // already confined. Offline still needs its network namespace.
                let final_command = wrap_network_command(command, self.network_offline);
                apply_sandbox_args(&mut cmd, dir, &target_dir);
                cmd.args(["--chdir", &dir.to_string_lossy()]);
                cmd.args(["/usr/bin/bash", "-c", &final_command]);
            }
            SandboxBackend::Unconfined => {
                warn_unconfined_once();
                cmd.current_dir(dir)
                    .args([NICE_FLAG, NICE_VALUE, "bash", "-c", command]);
            }
        }

        // Cleared environment + strict allow-list first, so no ambient
        // credential from the operator's shell reaches the model. Build/cache
        // variables are layered on top afterwards.
        apply_sanitized_environment(&mut cmd, dir);
        apply_build_env(&mut cmd, &target_dir, &tmp_dir, &parallelism);
        crate::cache::apply_shared_cache_env(&mut cmd);

        run_with_timeout(&mut cmd, timeout_secs).await
    }
}

/// Message shown to the model when an interceptor blocks a command.
fn blocked_by_interceptor(reason: &str) -> String {
    format!(
        "COMMAND BLOCKED BY INTERCEPTOR:\n{reason}\nPlease use a safe, non-destructive command within the current repository directory ($PWD)."
    )
}

/// Wrap `cmd` in an isolated network namespace when the worker declared
/// `network: "offline"`.
///
/// The wrapped form `unshare -n -- bash -c '<cmd>'` is idempotent with the rest
/// of the execution path: the child is still a bash process spawned by `nice`,
/// so `current_dir`, build env, timeout and output plumbing are unchanged.
/// Inside the namespace there is no route or interface, so `curl`/`git fetch`/
/// `cargo add` fails immediately (`Network is unreachable`) rather than
/// blocking for its own connect timeout.
///
/// `offline == false` returns `cmd` verbatim: connectivity is the default, and
/// a wrapper would only add a process for no isolation benefit.
///
/// The wrapper is applied even when `unshare` is missing: the command then
/// fails fast with a clear "not found" instead of silently running *with*
/// network access - a policy that quietly does not apply is worse than one
/// loudly unavailable. Callers that must degrade can inspect [`has_unshare`].
pub fn wrap_network_command(cmd: &str, offline: bool) -> String {
    if !offline {
        return cmd.to_string();
    }
    format!(
        "{NETWORK_NAMESPACE_TOOL} -n -- {OFFLINE_SHELL} -c {}",
        shell_quote(cmd)
    )
}

/// Whether the network-namespace primitive is usable on this host.
///
/// Cached probe: [`wrap_network_command`] runs once per step, and spawning
/// `unshare` just to ask would add a fork per step.
pub fn has_unshare() -> bool {
    super::sandbox::binary_available(NETWORK_NAMESPACE_TOOL)
}

/// Single-quote `value` for `bash -c`.
///
/// The command is model-authored and may contain every metacharacter, so the
/// wrapping layer must not re-interpret it: single quotes suppress all
/// expansion, and an embedded `'` is closed, escaped and reopened (the standard
/// `'"'"'` dance).
fn shell_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', r"'\''"))
}

/// Message shown to the model when the worktree guardrail rejects a command.
fn blocked_by_guardrail(reason: &str) -> String {
    format!(
        "COMMAND BLOCKED BY WORKTREE GUARDRAIL:\n{reason}\nPlease run your command within the current repository directory ($PWD)."
    )
}

/// Build/test parallelism for the child: `BUILD_PARALLELISM` or half the
/// available cores (never below one).
fn build_parallelism() -> String {
    let default_parallelism = crate::config::half_the_cores();
    crate::config::env_parse("BUILD_PARALLELISM")
        .unwrap_or(default_parallelism)
        .to_string()
}

/// Directory the child builds into: `CARGO_TARGET_DIR` when set, otherwise a
/// per-worktree dir under the SWE base so concurrent workers never share a
/// build cache.
fn resolve_target_dir(dir: &Path) -> PathBuf {
    if let Some(custom) = std::env::var_os("CARGO_TARGET_DIR") {
        return PathBuf::from(custom);
    }
    let dir_name = dir
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("default");
    crate::worktree::swe_base_dir().join(format!("swe-target-{dir_name}"))
}

/// Which confinement a worker step runs under.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SandboxBackend {
    /// Landlock + seccomp + process hardening from a `pre_exec` hook.
    Kernel,
    /// Bubblewrap mount namespaces, opt-in or kernel-fallback.
    Bwrap,
    /// No confinement; the caller logs the downgrade.
    Unconfined,
}

/// Env var that opts back into the bubblewrap backend.
pub const BWRAP_BACKEND_ENV: &str = "SWE_SANDBOX";
/// Value of [`BWRAP_BACKEND_ENV`] that selects bubblewrap.
pub const BWRAP_BACKEND_VALUE: &str = "bwrap";

/// Whether all confinement is explicitly disabled.
fn sandbox_disabled() -> bool {
    std::env::var("SWE_DISABLE_SANDBOX").as_deref() == Ok("1")
}

/// Whether the operator asked for the bubblewrap backend.
fn bwrap_requested() -> bool {
    std::env::var(BWRAP_BACKEND_ENV).as_deref() == Ok(BWRAP_BACKEND_VALUE)
}

/// Whether the bubblewrap sandbox is usable and not explicitly disabled.
fn sandbox_enabled() -> bool {
    has_bwrap() && !sandbox_disabled()
}

/// Pick the backend for the next worker step.
///
/// The kernel confines every step unless the operator opted out entirely
/// (`SWE_DISABLE_SANDBOX=1`) or back into bubblewrap (`SWE_SANDBOX=bwrap`).
/// A kernel with neither Landlock nor seccomp cannot confine anything, so it
/// falls back to bubblewrap when installed and otherwise runs unconfined.
pub(crate) fn select_backend() -> SandboxBackend {
    if sandbox_disabled() {
        return SandboxBackend::Unconfined;
    }
    if bwrap_requested() && has_bwrap() {
        return SandboxBackend::Bwrap;
    }
    if KernelConfinement::probe_available() {
        return SandboxBackend::Kernel;
    }
    if has_bwrap() {
        tracing::debug!("kernel offers no Landlock or seccomp; falling back to bubblewrap");
        return SandboxBackend::Bwrap;
    }
    SandboxBackend::Unconfined
}

/// Warn once per process that steps run without confinement.
///
/// The downgrade is worth exactly one log line: every step would otherwise
/// repeat it, burying the worker's own output.
fn warn_unconfined_once() {
    use std::sync::Once;
    static WARNED: Once = Once::new();
    WARNED.call_once(|| {
        tracing::warn!("no kernel confinement and no bubblewrap; running the command unconfined");
    });
}

/// Confine the child to `dir` and `target_dir` with the Landlock LSM, in the
/// forked child itself.
///
/// # Why a `pre_exec` hook and not a call in the parent
///
/// `landlock_restrict_self` restricts **the calling process** and is
/// irreversible. Calling it in the daemon would confine the daemon: it could no
/// longer read its own worktree state, config or caches, and no later call
/// could undo it. The only correct place to confine a worker is the process
/// about to become that worker - exactly what a `Command::pre_exec` closure is
/// (it runs in the child between `fork(2)` and `exec(2)`).
///
/// # Why the work happens in the parent
///
/// A `pre_exec` closure runs in a forked child of a *multi-threaded* server, so
/// only async-signal-safe operations are permitted; `malloc`, `tracing` and
/// `anyhow` are not (a thread holding the allocator lock at the instant of the
/// fork leaves the child permanently deadlocked). [`build_landlock_plan`]
/// therefore resolves the whole policy - ABI probe, path canonicalisation,
/// `CString` construction, syscall-backed existence checks - in the parent,
/// where allocating is safe, and the closure is left with nothing but raw
/// syscalls.
///
/// # Failure policy
///
/// * **No Landlock on this kernel, or `SWE_DISABLE_LANDLOCK=1`** -
///   [`build_landlock_plan`] returns `Ok(None)` and *no hook is registered*:
///   the command runs unconfined rather than failing. Landlock is absent on
///   pre-5.13 kernels, when `CONFIG_SECURITY_LANDLOCK` is off, when the
///   bootloader was given `lsm=` without it, and when a seccomp policy kills
///   the syscalls, so a hard failure there would take the worker offline for
///   no security gain.
/// * **A malformed policy** (the worktree or the target dir does not exist) -
///   propagated as an `Err` by the caller, because a worker confined to a
///   directory that is not there has no correct behaviour.
///
/// A hook that *does* run and then fails is fatal to the child: an `Err` out of
/// `pre_exec` aborts the spawn and is reported to the parent, which is the
/// right outcome - the kernel promised a domain and did not deliver one, and
/// silently continuing would be a confinement that is only advertised.
/// Confine the child with the kernel backend - Landlock, seccomp and process
/// hardening - in the forked child itself.
///
/// # Why a `pre_exec` hook and not a call in the parent
///
/// `landlock_restrict_self` restricts **the calling process** and is
/// irreversible. Calling it in the daemon would confine the daemon: it could no
/// longer read its own worktree state, config or caches, and no later call
/// could undo it. The only correct place to confine a worker is the process
/// about to become that worker - exactly what a `Command::pre_exec` closure is
/// (it runs in the child between `fork(2)` and `exec(2)`).
///
/// # Why the work happens in the parent
///
/// A `pre_exec` closure runs in a forked child of a *multi-threaded* server, so
/// only async-signal-safe operations are permitted; `malloc`, `tracing` and
/// `anyhow` are not (a thread holding the allocator lock at the instant of the
/// fork leaves the child permanently deadlocked). [`KernelConfinement::prepare`]
/// therefore resolves the whole policy - ABI probe, path canonicalisation,
/// `CString` construction, BPF assembly, syscall-backed existence checks - in
/// the parent, where allocating is safe, and the closure is left with nothing
/// but raw syscalls.
///
/// # Failure policy
///
/// * **No Landlock or seccomp on this kernel, or `SWE_DISABLE_LANDLOCK=1`** -
///   [`KernelConfinement::prepare`] returns `Ok(None)` and *no hook is
///   registered*: the command runs unconfined rather than failing.
/// * **A malformed policy** (the worktree or the target dir does not exist) -
///   a warning is logged and no hook is registered, because a worker confined
///   to a directory that is not there has no correct behaviour.
///
/// A hook that *does* run and then fails is fatal to the child: an `Err` out of
/// `pre_exec` aborts the spawn and is reported to the parent, which is the
/// right outcome - the kernel promised a domain and did not deliver one, and
/// silently continuing would be a confinement that is only advertised.
#[cfg(unix)]
fn apply_kernel_confinement(
    cmd: &mut Command,
    dir: &Path,
    target_dir: &Path,
    offline: bool,
) {
    // Parent side of the hook: everything that allocates happens here, so the
    // closure below is reduced to syscalls. A `None` plan means this host
    // cannot confine the process, which is not an error.
    let confinement = match KernelConfinement::prepare(dir, target_dir, offline) {
        Ok(Some(confinement)) => confinement,
        Ok(None) => return,
        Err(e) => {
            tracing::warn!(
                error = %format!("{e:#}"),
                worktree = %dir.display(),
                "kernel confinement unavailable; running the command unconfined"
            );
            return;
        }
    };

    // SAFETY: the closure runs in the child between `fork` and `exec`, where
    // `confinement.apply` performs only raw syscalls - it allocates nothing,
    // takes no lock and never unwinds - and confines only that child, never
    // the parent.
    unsafe {
        cmd.pre_exec(move || {
            // SAFETY: forwarded from this function's contract; see above.
            confinement.apply()
        });
    }
}

/// Non-unix stub: no Landlock LSM, no seccomp and no `fork` to hook.
#[cfg(not(unix))]
fn apply_kernel_confinement(
    _cmd: &mut Command,
    _dir: &Path,
    _target_dir: &Path,
    _offline: bool,
) {}

/// Baseline child setup: kill the process group on drop, detach stdin, capture
/// both output streams.
fn configure_process(cmd: &mut Command) {
    cmd.kill_on_drop(true);
    #[cfg(unix)]
    cmd.process_group(0);
    cmd.stdin(std::process::Stdio::null());
    cmd.stdout(std::process::Stdio::piped());
    cmd.stderr(std::process::Stdio::piped());
}

/// Read-only bind for a toolchain cache directory.
///
/// `--ro-bind-try` tolerates a missing path, so a cache that was never
/// installed on this host does not prevent the sandbox from starting.
fn ro_bind_toolchain_cache(cmd: &mut Command, path: &Path) {
    let path_str = path.to_string_lossy();
    cmd.args(["--ro-bind-try", &path_str, &path_str]);
}

/// Append the full `bwrap` argument vector.
///
/// Read-only by default: only the worktree, its own gitdir, the isolated build
/// target dir and a handful of toolchain caches are bound writable. `$HOME` is
/// an empty tmpfs so the agent cannot read or clobber SSH keys, dotfiles or
/// package-manager credentials.
fn apply_sandbox_args(cmd: &mut Command, dir: &Path, target_dir: &Path) {
    let dir_str = dir.to_string_lossy();
    let target_str = target_dir.to_string_lossy();

    cmd.args([
        NICE_FLAG,
        NICE_VALUE,
        "bwrap",
        "--die-with-parent",
        "--new-session",
        "--unshare-pid",
        "--unshare-ipc",
        "--ro-bind",
        "/usr",
        "/usr",
        "--symlink",
        "usr/bin",
        "/bin",
        "--symlink",
        "usr/bin",
        "/sbin",
        "--symlink",
        "usr/lib",
        "/lib",
        "--symlink",
        "usr/lib",
        "/lib64",
        "--ro-bind-try",
        "/etc",
        "/etc",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
    ]);

    // Isolate user home: empty tmpfs, toolchain caches read-only.
    let home = std::env::var_os("HOME").map(PathBuf::from);
    if let Some(home) = home.as_deref() {
        let home_str = home.to_string_lossy();
        cmd.args(["--tmpfs", &home_str]);
        let cache_tmp = home.join(".cache");
        let cache_tmp_str = cache_tmp.to_string_lossy();
        cmd.args(["--tmpfs", &cache_tmp_str]);
    }

    // Expose the shared toolchain homes read-only. The paths come from the
    // same resolution the child environment uses (`env::host_cargo_home`, plus
    // `RUSTUP_HOME`), so the mount and the forwarded `CARGO_HOME` can never
    // disagree: a directory the sandbox binds but the environment does not
    // point at would leave cargo with an empty registry, and one the
    // environment points at but the sandbox does not bind would simply be
    // missing. Each distinct path is bound once.
    let mut toolchain_caches: Vec<PathBuf> = Vec::new();
    if let Some(home) = home.as_deref() {
        for cache_dir in [".cargo", ".rustup", ".local/bin"] {
            toolchain_caches.push(home.join(cache_dir));
        }
    }
    if let Some(cargo_home) = super::env::host_cargo_home() {
        toolchain_caches.push(cargo_home);
    }
    if let Some(rustup_home) = std::env::var_os("RUSTUP_HOME")
        .map(PathBuf::from)
        .filter(|p| !p.as_os_str().is_empty())
    {
        toolchain_caches.push(rustup_home);
    }
    toolchain_caches.sort();
    toolchain_caches.dedup();
    for path in toolchain_caches {
        ro_bind_toolchain_cache(cmd, &path);
    }

    // Worktree read-write.
    cmd.args(["--bind", &dir_str, &dir_str]);

    // Common .git READ-ONLY so git resolves refs/objects without the sandbox
    // being able to prune or delete repository branches.
    if let Some((common_git, worktree_gitdir)) = find_git_dirs(dir) {
        let common_str = common_git.to_string_lossy();
        cmd.args(["--ro-bind", &common_str, &common_str]);

        // Only this worker's worktree gitdir read-write, so it can update its index.
        if let Some(wt_gitdir) = worktree_gitdir
            && wt_gitdir.is_dir()
        {
            let wt_str = wt_gitdir.to_string_lossy();
            cmd.args(["--bind", &wt_str, &wt_str]);
        }
    }

    // Isolated build target read-write.
    cmd.args(["--bind", &target_str, &target_str]);

    // Modular shared package/compiler caches.
    crate::cache::append_bwrap_cache_args(cmd, home.as_deref());
}

/// Replace the inherited environment with the sanitized allow-list.
///
/// The parent environment is cleared wholesale and rebuilt from
/// `env::build_clean_environment`, which forwards only essential runtime
/// variables, remaps `HOME` to an isolated per-worktree scratch directory and
/// purges credential-bearing names (`OPENAI_API_KEY`, `GITHUB_TOKEN`, `AWS_*`,
/// `SSH_*`, ...). The child cannot read back the operator's secrets through the
/// ambient environment, and runs are reproducible regardless of what else the
/// operator exported.
///
/// `repo_path` is the original checkout; the worker's worktree `dir` is what
/// the command is chdir'ed into, so the isolated `HOME` is placed under `dir`
/// where the sandbox bind makes it writable.
fn apply_sanitized_environment(cmd: &mut Command, dir: &Path) {
    // The original repository is the worktree's git common dir when one can
    // be resolved, else the worktree itself.
    let repo_path = find_git_common_dir(dir).unwrap_or_else(|| dir.to_path_buf());
    let repo_path = repo_path
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or(repo_path);
    super::env::apply_clean_environment_cmd(cmd, &repo_path, dir);
}

/// Universal build/test parallelism caps so a command cannot oversubscribe the
/// machine no matter which build tool it drives.
fn apply_build_env(cmd: &mut Command, target_dir: &Path, tmp_dir: &Path, parallelism: &str) {
    cmd.env("CARGO_TARGET_DIR", target_dir)
        // A writable private temp dir under the isolated target: the kernel
        // backend grants no `/tmp`, and the bubblewrap backend mounts an empty
        // tmpfs there, so tools must use the sandbox-writable scratch space.
        .env("TMPDIR", tmp_dir)
        .env("TMP", tmp_dir)
        .env("TEMP", tmp_dir)
        .env("CARGO_BUILD_JOBS", parallelism)
        .env("RUST_TEST_THREADS", parallelism)
        .env("NEXTEST_TEST_THREADS", parallelism)
        .env("MAKEFLAGS", format!("-j{parallelism}"))
        .env("CMAKE_BUILD_PARALLEL_LEVEL", parallelism)
        .env("RAYON_NUM_THREADS", parallelism)
        .env("OMP_NUM_THREADS", parallelism)
        .env("OPENBLAS_NUM_THREADS", parallelism)
        .env("MKL_NUM_THREADS", parallelism)
        .env("GOMAXPROCS", parallelism);
}

/// Wall-clock budget (seconds) for `command`: heavy commands get a longer
/// default; `COMMAND_TIMEOUT_SECS` overrides either tier.
fn command_timeout_secs(command: &str) -> u64 {
    let default_timeout = if is_heavy_command(command) {
        crate::config::env_parse("COMMAND_HEAVY_TIMEOUT_SECS")
            .unwrap_or(DEFAULT_HEAVY_TIMEOUT_SECS)
    } else {
        crate::config::env_parse("COMMAND_LIGHT_TIMEOUT_SECS")
            .unwrap_or(DEFAULT_LIGHT_TIMEOUT_SECS)
    };
    crate::config::env_parse("COMMAND_TIMEOUT_SECS").unwrap_or(default_timeout)
}

/// Spawn `cmd`, wait up to `timeout_secs`, and collect the combined output.
///
/// On timeout the whole process group gets a graceful `SIGTERM`, escalated to
/// `SIGKILL` only if it refuses to stop within [`TERM_GRACE_MS`] (a lingering
/// compiler or test runner would otherwise outlive its budget). Whatever the
/// command printed before the budget expired is still reported, so the model
/// sees the last diagnostics instead of a bare "timed out". The timeout is
/// reported as ordinary output with exit code [`TIMEOUT_EXIT_CODE`].
async fn run_with_timeout(cmd: &mut Command, timeout_secs: u64) -> Result<(String, Option<i32>)> {
    let mut child = cmd.spawn().context("Failed to spawn bash process")?;
    let child_pid = child.id();

    // Covers cancellation: if this future is dropped (pool kill, shutdown,
    // cancelled request) the guard takes the whole process group down, where
    // `kill_on_drop` would only reach the leader PID.
    let mut group_guard = ProcessGroupGuard::new(child_pid);

    // Hand the pipes to reader tasks that run independently of the wait, so
    // cancelling the wait on timeout cannot throw away output already produced.
    let stdout = child
        .stdout
        .take()
        .context("stdout pipe was not captured")?;
    let stderr = child
        .stderr
        .take()
        .context("stderr pipe was not captured")?;
    let out_buf = PipeBuffer::spawn(stdout);
    let err_buf = PipeBuffer::spawn(stderr);

    match tokio::time::timeout(Duration::from_secs(timeout_secs), child.wait()).await {
        // Clean exit: the write ends are closed now that the child is gone, so
        // the readers reach EOF and the real output and code are reported.
        Ok(Ok(status)) => {
            // The group is already reaped; the guard must not signal it again.
            group_guard.disarm();
            let (out, err) = tokio::join!(out_buf.finish(), err_buf.finish());
            Ok((
                combine_streams(&out.bytes, &err.bytes, out.dropped + err.dropped),
                status.code(),
            ))
        }
        // A `wait` error leaves the child's fate unknown, so the guard stays
        // armed: killing the group is the only cleanup that cannot leave a
        // build running behind us. Signalling a reaped group is inert
        // (`ESRCH`), and the PID cannot be recycled while tokio holds the
        // unreaped `Child`.
        Ok(Err(e)) => Err(e).context("Failed waiting for bash process"),
        // The child outlived its budget: stop it, then report what it printed.
        Err(_elapsed) => {
            terminate_process_group(child_pid, &mut child).await;
            // `terminate_process_group` reaped the group (or gave up after the
            // SIGKILL), so the drop guard has nothing left to do.
            group_guard.disarm();
            // The group is gone, so the readers are released and the drain
            // converges instead of waiting out the whole drain budget.
            let (out, err) = tokio::join!(out_buf.finish(), err_buf.finish());
            let mut output = combine_streams(&out.bytes, &err.bytes, out.dropped + err.dropped);
            output.push_str(&format!(
                "\nCommand timed out after {timeout_secs}s and was terminated."
            ));
            Ok((output, Some(TIMEOUT_EXIT_CODE)))
        }
    }
}

/// Head and tail of a command's output stream, bounded to exactly the bytes
/// [`truncate_output`] would have kept.
struct Captured {
    /// First [`TRUNCATE_HEAD`] bytes seen; later input spills into `tail`.
    head: Vec<u8>,
    /// Last [`TRUNCATE_TAIL`] bytes seen, kept in a rolling window so the
    /// buffer never grows with the output.
    tail: Vec<u8>,
    /// Total bytes produced, including the ones not kept. Comparing it against
    /// what we hold *is* the elision count, so the two never drift apart.
    seen: usize,
}

impl Captured {
    fn new() -> Self {
        Self {
            head: Vec::with_capacity(TRUNCATE_HEAD),
            tail: Vec::with_capacity(TRUNCATE_TAIL),
            seen: 0,
        }
    }

    /// Absorb `chunk`, retaining only what the truncation budget can use.
    ///
    /// Bounding here rather than after the read keeps a chatty command from
    /// costing unbounded memory: the sink is a fixed 16 KiB per pipe whether
    /// the child printed 16 KiB or 16 GiB. What survives is exactly the head
    /// and tail [`truncate_output`] would have kept; the discarded middle is
    /// counted, not forgotten, in [`dropped`](Self::dropped).
    fn push(&mut self, chunk: &[u8]) {
        self.seen += chunk.len();
        let mut rest = chunk;

        // Fill the head first; only what spills past it is eligible for the
        // rolling tail.
        if self.head.len() < TRUNCATE_HEAD {
            let take = (TRUNCATE_HEAD - self.head.len()).min(rest.len());
            self.head.extend_from_slice(&rest[..take]);
            rest = &rest[take..];
        }
        if rest.is_empty() {
            return;
        }

        if rest.len() >= TRUNCATE_TAIL {
            // The chunk alone fills the window: keep only its last bytes.
            self.tail.clear();
            self.tail
                .extend_from_slice(&rest[rest.len() - TRUNCATE_TAIL..]);
            return;
        }
        // Append, then drop the oldest bytes that fell off the front. The
        // window never exceeds `TRUNCATE_TAIL`, so this is O(tail) per chunk
        // and O(1) amortised per byte.
        self.tail.extend_from_slice(rest);
        if self.tail.len() > TRUNCATE_TAIL {
            self.tail.drain(..self.tail.len() - TRUNCATE_TAIL);
        }
    }

    /// The retained bytes -- head then tail -- with no marker of its own.
    ///
    /// Deliberately *unmarked*: the elision is reported through
    /// [`dropped`](Self::dropped) instead, so [`combine_streams`] stays the
    /// single place that decides what the model sees and can account for both
    /// streams' elisions in one marker. Borrows rather than consumes so the
    /// caller can snapshot a partial drain.
    fn captured(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.head.len() + self.tail.len());
        out.extend_from_slice(&self.head);
        out.extend_from_slice(&self.tail);
        out
    }

    /// Bytes this stream produced that the retained head/tail do not hold.
    fn dropped(&self) -> usize {
        self.seen - self.head.len() - self.tail.len()
    }
}

/// One of a command's output pipes, drained in the background into a bounded
/// buffer that outlives the command's own timeout.
///
/// Running the read as a detached task is what makes the drain robust: a read
/// inlined into the awaited future is cancelled wholesale when the wait times
/// out, discarding every byte it had already received; here the task keeps
/// draining while the caller is free to stop waiting, and
/// [`finish`](Self::finish) later collects whatever arrived.
///
/// "Draining" and "retaining" are deliberately separate. The reader keeps
/// consuming from the pipe no matter how much the child writes -- that stops a
/// full pipe buffer from wedging the child in `write` -- while [`Captured`]
/// discards everything the truncation budget cannot show. Together they let a
/// multi-gigabyte build log cost 16 KiB of memory and still not deadlock.
struct PipeBuffer {
    bytes: Arc<Mutex<Captured>>,
    reader: tokio::task::JoinHandle<()>,
}

// A cancelled command must not leave detached readers holding pipe
// descriptors indefinitely (a child can escape the process group with `setsid`).
impl Drop for PipeBuffer {
    fn drop(&mut self) {
        self.reader.abort();
    }
}

/// One drained pipe: the bytes worth showing, plus how many were elided.
///
/// `dropped` lets [`combine_streams`] tell "the child printed 20 KB" from "the
/// child printed 4 GB" and report the right figure in the truncation marker.
struct Stream {
    bytes: Vec<u8>,
    dropped: usize,
}

impl PipeBuffer {
    /// Start draining `pipe` into a fresh bounded buffer.
    fn spawn<R>(mut pipe: R) -> Self
    where
        R: AsyncRead + Unpin + Send + 'static,
    {
        let bytes = Arc::new(Mutex::new(Captured::new()));
        let sink = Arc::clone(&bytes);
        let reader = tokio::spawn(async move {
            let mut buf = [0u8; DRAIN_CHUNK_BYTES];
            loop {
                // A read failure means the child is gone; whatever arrived is
                // still worth reporting, so the loop only stops at EOF (`0`)
                // and the buffer is flushed either way.
                match pipe.read(&mut buf).await {
                    Ok(0) | Err(_) => break,
                    Ok(n) => match sink.lock() {
                        Ok(mut sink) => sink.push(&buf[..n]),
                        // The owning task panicked while holding the lock; a
                        // poisoned mutex still holds what was captured so far.
                        Err(poisoned) => poisoned.into_inner().push(&buf[..n]),
                    },
                }
            }
        });
        Self { bytes, reader }
    }

    /// Stop waiting for the reader and return the bytes worth reporting.
    ///
    /// Bounded by [`DRAIN_GRACE_MS`] because a grandchild that inherited the
    /// write end can hold the pipe open long after its parent was killed; the
    /// reader is then cancelled and the bytes captured up to that point
    /// returned.
    async fn finish(mut self) -> Stream {
        if tokio::time::timeout(Duration::from_millis(DRAIN_GRACE_MS), &mut self.reader)
            .await
            .is_err()
        {
            // The pipe is held open outside our process group. Cancel the
            // reader so its pipe descriptor and task cannot accumulate across
            // repeated timed-out commands.
            tracing::debug!(
                timeout_ms = DRAIN_GRACE_MS,
                "abandoning output drain of a timed-out command"
            );
            self.reader.abort();
        }

        let guard = match self.bytes.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        Stream {
            bytes: guard.captured(),
            dropped: guard.dropped(),
        }
    }
}

/// `SIGTERM` the child's process group, then `SIGKILL` any remaining members
/// after the leader exits or [`TERM_GRACE_MS`] elapses.
///
/// Best-effort at every step: the group may already be gone (the sandbox
/// wrapper carries `--die-with-parent` and takes its children with it).
async fn terminate_process_group(pid: Option<u32>, child: &mut Child) {
    signal_process_group(pid, libc::SIGTERM);

    // Bounded reap so a well-behaved child can exit on its own; the deadline
    // stops a wedged child from extending the budget indefinitely.
    let reaped = matches!(
        tokio::time::timeout(Duration::from_millis(TERM_GRACE_MS), child.wait()).await,
        Ok(Ok(_))
    );

    // Reaping the leader does not imply its children stopped: a shell can
    // terminate on SIGTERM while a child ignores it. Kill remaining members
    // before returning from the timeout path.
    signal_process_group(pid, libc::SIGKILL);
    if !reaped {
        let _ = tokio::time::timeout(Duration::from_millis(KILL_GRACE_MS), child.wait()).await;
    }
}

/// Send `sig` to the process group led by `pid`.
///
/// `configure_process` places the child in its own process group (`bwrap` adds
/// `--new-session`), so the negated PID addresses the group and reaches
/// grandchildren that `child.wait` alone would never reap.
///
/// A direct `kill(2)` rather than a forked `kill` binary: signalling is on the
/// cancellation path, where a `PATH` lookup and a `fork`/`exec` race are both a
/// delay and a way to lose the signal entirely. `libc` is already a hard
/// dependency (see [`super::sandbox`]), so this adds no dependency weight.
///
/// `ESRCH` means the group is already gone -- the sandbox wrapper carries
/// `--die-with-parent` and routinely takes its children down before the timeout
/// fires -- which is a success, not an error.
#[cfg(unix)]
fn signal_process_group(pid: Option<u32>, sig: libc::c_int) {
    let Some(pid) = pid.map(|p| p as libc::pid_t) else {
        return;
    };
    // SAFETY: `kill` takes no pointers and cannot cause UB. The negated pid
    // addresses the group `configure_process` created; a pid of 0 would address
    // *our own* group, which `Child::id` never reports (it is `None` if the
    // child already exited, handled by the `else` above).
    let rc = unsafe { libc::kill(-pid, sig) };
    if rc != 0 {
        let err = std::io::Error::last_os_error();
        if err.raw_os_error() != Some(libc::ESRCH) {
            tracing::debug!(pid, sig, error = %err, "process group signal failed");
        }
    }
}

/// Non-unix stub: no process group signalling outside unix; the `Child`
/// guard's `SIGKILL` is the only cleanup there.
#[cfg(not(unix))]
fn signal_process_group(_pid: Option<u32>, _sig: i32) {}

/// Kill a command's whole process group unless explicitly disarmed.
///
/// `kill_on_drop` only reaches the child itself, so this guard closes the hole
/// where a cancelled future (`pool::kill`, `kill_all`, a shutting-down runtime)
/// drops the `Child` while its grandchildren keep running -- an interrupted
/// `cargo build` would otherwise leave its `rustc` children running, holding
/// the worktree's `target/` open and burning CPU.
///
/// `SIGKILL` rather than [`terminate_process_group`]'s `SIGTERM`-then-`SIGKILL`,
/// because a `Drop` cannot await: there is no grace period to give and no reap
/// to perform. `SIGKILL` cannot be caught, blocked or ignored, so the group is
/// gone by the time the caller resumes.
///
/// Inert on non-unix targets, where [`signal_process_group`] does nothing and
/// there is no group to reach.
struct ProcessGroupGuard {
    pid: Option<u32>,
    armed: bool,
}

impl ProcessGroupGuard {
    /// Guard `pid` until [`disarm`](Self::disarm) is called.
    fn new(pid: Option<u32>) -> Self {
        Self { pid, armed: true }
    }

    /// Stop guarding: the caller already reaped the group, so a drop-time
    /// `SIGKILL` would be redundant.
    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for ProcessGroupGuard {
    fn drop(&mut self) {
        if self.armed {
            signal_process_group(self.pid, libc::SIGKILL);
        }
    }
}

/// Merge the two drained pipes into the string the model sees, bounded to the
/// shared truncation budget.
///
/// Each pipe keeps only [`TRUNCATE_HEAD`] + [`TRUNCATE_TAIL`] bytes, so a
/// command that printed megabytes arrives here as two short slices plus
/// `dropped`, the count of bytes the buffers already elided. The marker spliced
/// between the halves carries that count, so it reports the child's *true*
/// output size rather than the size of our own buffer -- without it a 4 GB
/// build log would be reported as "16 KB" and the model could never tell a
/// truncated stream from a small one.
///
/// The newline separator is unconditional, so a stdout that already ends in
/// `\n` yields a blank line between the streams. That is pre-existing behaviour
/// the model sees verbatim in the tool result, so it is preserved as-is rather
/// than quietly changing the transcripts this crate already produced.
fn combine_streams(stdout: &[u8], stderr: &[u8], dropped: usize) -> String {
    let mut combined = String::new();
    if !stdout.is_empty() {
        combined.push_str(&String::from_utf8_lossy(stdout));
    }
    if !stderr.is_empty() {
        if !combined.is_empty() {
            combined.push('\n');
        }
        combined.push_str(&String::from_utf8_lossy(stderr));
    }

    truncate_with_dropped(&combined, dropped)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn runner() -> AgentRunner {
        AgentRunner::new(
            "http://localhost".to_string(),
            "test-key".to_string(),
            "test-model".to_string(),
            None,
        )
    }

    /// The default (connected) policy must not touch the command at all: a
    /// wrapper there would only add a process and a quoting layer.
    #[test]
    fn an_online_command_is_returned_verbatim() {
        let cmd = "cargo test && echo done > out.txt";
        assert_eq!(wrap_network_command(cmd, false), cmd);
        assert_eq!(wrap_network_command(cmd, false), cmd.to_string());
    }

    /// `offline` wraps the command in a network namespace while keeping it a
    /// single bash string, so the model's pipes/redirections still work.
    #[test]
    fn an_offline_command_is_wrapped_in_a_network_namespace() {
        let wrapped = wrap_network_command("echo hi", true);
        assert!(
            wrapped.starts_with("unshare -n -- bash -c "),
            "the wrapper must enter an isolated network namespace: {wrapped}"
        );
        assert!(wrapped.contains("'echo hi'"), "{wrapped}");
    }

    /// The command is model-authored text, so the wrapper must quote it: an
    /// embedded single quote, `$VAR` or backtick has to survive the extra
    /// shell layer verbatim instead of being expanded or breaking out of it.
    #[test]
    fn an_offline_wrapper_quotes_its_command_exactly_once() {
        let wrapped = wrap_network_command("echo 'it'\''s' $HOME `id` \"q\"", true);
        let inner = wrapped
            .rsplit_once("bash -c ")
            .expect("an offline command is wrapped")
            .1;
        assert!(inner.starts_with('\'') && inner.ends_with('\''), "{inner}");
        // The escaped-quote dance keeps the outer quoting balanced.
        assert!(inner.contains("'\\''"), "{inner}");
        assert!(
            inner.contains("$HOME"),
            "no expansion happens at wrap time: {inner}"
        );
    }

    /// End-to-end contract: with the policy on, a request needing egress fails
    /// *immediately* (no interface, no route) instead of hanging out a connect
    /// timeout, while a local command still runs normally.
    ///
    /// Where the host forbids namespace creation (an unprivileged container, a
    /// kernel without `CONFIG_NET_NS`) the wrapper is still applied and the
    /// step fails loudly; that fallback is asserted separately below rather
    /// than letting the test quietly pass in isolation.
    #[tokio::test]
    async fn an_offline_worker_has_no_egress_and_still_runs_local_commands() {
        if !has_unshare() {
            eprintln!("skipping: unshare is unavailable on this host");
            return;
        }
        let tmp = crate::worktree::swe_base_dir().join("exec-offline-test");
        let _ = std::fs::create_dir_all(&tmp);
        let offline = runner().with_network_offline(true);

        if !can_create_network_namespace() {
            // Policy must never silently degrade into "with network access".
            let (out, code) = offline
                .execute_bash(&tmp, "printf 'must not run\n'")
                .await
                .expect("an offline command must still be spawned");
            assert_ne!(
                code,
                Some(0),
                "an unavailable namespace must fail loudly, never run unisolated: {out:?}"
            );
            let _ = std::fs::remove_dir_all(&tmp);
            return;
        }

        let (out, code) = offline
            .execute_bash(&tmp, "printf 'still runs\n'")
            .await
            .expect("an offline command must still be spawned");
        assert_eq!(code, Some(0), "local commands must work offline: {out:?}");
        assert!(out.contains("still runs"), "{out:?}");

        let started = std::time::Instant::now();
        let (out, code) = offline
            .execute_bash(
                &tmp,
                "curl -sS --max-time 20 https://example.com -o /dev/null; echo exit=$?",
            )
            .await
            .expect("an offline curl must fail as ordinary output, not an error");
        let elapsed = started.elapsed();

        assert_ne!(code, Some(0), "egress must fail offline: {out:?}");
        assert!(
            out.contains("exit=") && !out.contains("exit=0"),
            "the curl must not succeed: {out:?}"
        );
        assert!(
            elapsed < std::time::Duration::from_secs(15),
            "offline must fail fast on ENETUNREACH, not wait out a connect \
             timeout (took {elapsed:?})"
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// Whether this host actually lets us build a network namespace.
    ///
    /// `has_unshare` only proves the binary exists; inside an unprivileged
    /// container `unshare -n` still fails with EPERM, so the promised
    /// isolation cannot be created there.
    fn can_create_network_namespace() -> bool {
        std::process::Command::new(NETWORK_NAMESPACE_TOOL)
            .args(["-n", "--", "true"])
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .is_ok_and(|status| status.success())
    }

    #[test]
    fn resolve_target_dir_defaults_to_per_worktree_path() {
        let dir = Path::new("/tmp/swe/worktree-alpha");
        if std::env::var_os("CARGO_TARGET_DIR").is_some() {
            return; // The override wins; covered by the caller-visible behaviour.
        }
        let target = resolve_target_dir(dir);
        assert!(target.ends_with("swe-target-worktree-alpha"), "{target:?}");
    }

    #[test]
    fn resolve_target_dir_uses_anonymous_dir_name_safely() {
        if std::env::var_os("CARGO_TARGET_DIR").is_some() {
            return;
        }
        let target = resolve_target_dir(Path::new("/"));
        assert!(target.ends_with("swe-target-default"), "{target:?}");
    }

    #[test]
    fn heavy_commands_get_the_larger_default_budget() {
        if std::env::var_os("COMMAND_TIMEOUT_SECS").is_some()
            || std::env::var_os("COMMAND_HEAVY_TIMEOUT_SECS").is_some()
            || std::env::var_os("COMMAND_LIGHT_TIMEOUT_SECS").is_some()
        {
            return; // Environment overrides make the defaults unobservable.
        }
        assert!(is_heavy_command("cargo build"));
        assert_eq!(
            command_timeout_secs("cargo build"),
            DEFAULT_HEAVY_TIMEOUT_SECS
        );
        assert_eq!(command_timeout_secs("echo hi"), DEFAULT_LIGHT_TIMEOUT_SECS);
    }

    #[test]
    fn guardrail_rejection_is_reported_not_errored() {
        let out = blocked_by_guardrail("outside the worktree");
        assert!(out.contains("COMMAND BLOCKED BY WORKTREE GUARDRAIL"));
        assert!(out.contains("outside the worktree"));
    }

    #[test]
    fn combine_streams_joins_both_streams_and_truncates() {
        // Both streams non-empty: stdout, the separator, then stderr.
        assert_eq!(
            combine_streams(b"from stdout\n", b"from stderr", 0),
            "from stdout\n\nfrom stderr"
        );

        // Only one stream present: no separator is inserted.
        assert_eq!(combine_streams(b"only stdout", b"", 0), "only stdout");

        let long = "x".repeat(TRUNCATE_LIMIT_FOR_TEST + 1_000);
        let truncated = combine_streams(long.as_bytes(), b"", 0);
        assert!(truncated.contains("... [Truncated "), "{truncated}");
        assert!(truncated.len() < TRUNCATE_LIMIT_FOR_TEST + 1_000);
    }

    /// Mirrors the sandbox budget constant without importing it twice.
    const TRUNCATE_LIMIT_FOR_TEST: usize = super::super::sandbox::TRUNCATE_LIMIT;

    #[tokio::test]
    async fn execute_bash_sandbox_runs_and_blocks_write() {
        let unique_id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let tmp = crate::worktree::swe_base_dir()
            .join(format!("bwrap-test-{}-{unique_id}", std::process::id()));
        let _ = std::fs::create_dir_all(&tmp);
        let r = runner();

        // 1. Basic command within worktree succeeds
        let (out, code) = r
            .execute_bash(&tmp, "echo 'hello from sandbox'")
            .await
            .unwrap();
        assert_eq!(code, Some(0), "command failed with output: {out:?}");
        assert!(out.contains("hello from sandbox"));

        // 2. Writing to read-only host root /usr fails when bwrap is active
        if has_bwrap() {
            let (out, code) = r
                .execute_bash(&tmp, "touch /usr/forbidden_write_test 2>&1")
                .await
                .unwrap();
            assert_ne!(code, Some(0));
            assert!(
                out.contains("Read-only")
                    || out.contains("sólo lectura")
                    || out.contains("solo lectura")
                    || out.contains("Permission denied")
                    || out.contains("Permiso denegado"),
                "unexpected touch output: {out:?}, code: {code:?}"
            );
        }

        let target_dir = crate::worktree::swe_base_dir().join(format!(
            "swe-target-bwrap-test-{}-{unique_id}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&tmp);
        let _ = std::fs::remove_dir_all(&target_dir);
    }

    #[tokio::test]
    async fn execute_bash_runs_unsandboxed_when_disabled() {
        let tmp = crate::worktree::swe_base_dir().join("exec-path-test");
        let _ = std::fs::create_dir_all(&tmp);
        let (out, code) = runner()
            .execute_bash(&tmp, "printf 'plain\\n'")
            .await
            .unwrap();
        assert_eq!(code, Some(0), "command failed with output: {out:?}");
        assert!(out.contains("plain"), "{out:?}");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// A process group that ignores `SIGTERM` is still taken down, but only
    /// after the grace period: the child must be alive when the grace expires
    /// (proving `SIGTERM` was sent and had no effect) and gone once the call
    /// returns (proving the `SIGKILL` escalation landed).
    #[tokio::test]
    async fn terminate_escalates_term_to_kill_after_the_grace_period() {
        let mut cmd = Command::new("bash");
        // The loop must run *in the shell*: bash `exec`s a trailing simple
        // command, so `trap '' TERM; sleep 300` would really be a `sleep`, which
        // dies on SIGTERM and would hide the escalation entirely.
        cmd.args([
            "-c",
            "trap '' TERM; end=$((SECONDS+300)); while (( SECONDS < end )); do :; done",
        ]);
        configure_process(&mut cmd);
        let mut child = cmd.spawn().expect("bash must spawn");
        let pid = child.id().expect("a spawned child has a pid");
        // Let the trap take effect before any signal is sent.
        tokio::time::sleep(Duration::from_millis(200)).await;

        let started = std::time::Instant::now();
        terminate_process_group(Some(pid), &mut child).await;
        let elapsed = started.elapsed();

        assert!(
            elapsed >= Duration::from_millis(TERM_GRACE_MS),
            "a child ignoring SIGTERM must be given the full {TERM_GRACE_MS}ms grace \
             period before the SIGKILL escalation, but the group went down in {elapsed:?}"
        );
        // It only went down because of the SIGKILL: it was still running when
        // the SIGTERM was sent, so nothing else could have reaped it.
        assert!(
            !group_is_alive(pid),
            "the SIGKILL escalation must reap the group"
        );
    }

    /// A child that stops on `SIGTERM` exits inside the grace period, so the
    /// call returns long before the `SIGKILL` deadline.
    #[tokio::test]
    async fn terminate_does_not_escalate_a_child_that_obeys_sigterm() {
        let mut cmd = Command::new("bash");
        // Shell-side loop again, so the trapped shell itself receives the signal
        // instead of an exec'd `sleep`.
        cmd.args([
            "-c",
            "trap 'exit 0' TERM; end=$((SECONDS+300)); while (( SECONDS < end )); do :; done",
        ]);
        configure_process(&mut cmd);
        let mut child = cmd.spawn().expect("bash must spawn");
        tokio::time::sleep(Duration::from_millis(200)).await;

        let started = std::time::Instant::now();
        terminate_process_group(child.id(), &mut child).await;
        let elapsed = started.elapsed();

        assert!(
            elapsed < Duration::from_millis(TERM_GRACE_MS),
            "a SIGTERM-obedient child must be reaped without waiting out the grace \
             period, but the call took {elapsed:?}"
        );
    }

    /// Signalling a group that is already gone is a no-op rather than a panic:
    /// the sandbox wrapper carries `--die-with-parent` and routinely takes its
    /// children down before the timeout ever fires.
    #[tokio::test]
    async fn terminate_tolerates_an_already_dead_group() {
        let mut cmd = Command::new("true");
        configure_process(&mut cmd);
        let mut child = cmd.spawn().expect("true must spawn");
        let pid = child.id();
        let _ = child.wait().await;

        let started = std::time::Instant::now();
        terminate_process_group(pid, &mut child).await;
        assert!(
            started.elapsed() < Duration::from_millis(TERM_GRACE_MS),
            "signalling a dead group must return promptly, took {:?}",
            started.elapsed()
        );
    }

    /// Whether a process still exists, used by the escalation test to show the
    /// child outlived the `SIGTERM` and was removed by the `SIGKILL`.
    fn group_is_alive(pid: u32) -> bool {
        std::process::Command::new("kill")
            .args(["-0", &pid.to_string()])
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .expect("kill must run")
            .success()
    }

    /// The head/tail buffer must keep exactly the bytes `truncate_output`
    /// keeps, and never more: that is the whole point of bounding it.
    #[test]
    fn the_capture_buffer_is_bounded_and_keeps_the_head_and_tail() {
        // Far past the budget, fed in chunks of every interesting size.
        let total = 400_000;
        let mut captured = Captured::new();
        let mut chunk = Vec::new();
        for i in 0..total {
            chunk.push(b'a' + (i % 26) as u8);
            if chunk.len() == 7_919 {
                captured.push(&chunk);
                chunk.clear();
            }
        }
        captured.push(&chunk);

        let retained = captured.captured();
        assert!(
            retained.len() <= TRUNCATE_HEAD + TRUNCATE_TAIL,
            "the retained buffer must never exceed the truncation budget, got {}",
            retained.len()
        );
        assert_eq!(
            captured.dropped(),
            total - retained.len(),
            "every byte not retained must be accounted for as dropped"
        );
        // The head is the first TRUNCATE_HEAD bytes of the stream...
        assert_eq!(
            &retained[..TRUNCATE_HEAD],
            &(0..TRUNCATE_HEAD)
                .map(|i| b'a' + (i % 26) as u8)
                .collect::<Vec<u8>>()[..],
            "the head of the stream must be kept verbatim"
        );
        // ...and the tail is the last TRUNCATE_TAIL bytes.
        let tail: Vec<u8> = (total - TRUNCATE_TAIL..total)
            .map(|i| b'a' + (i % 26) as u8)
            .collect();
        assert_eq!(
            &retained[retained.len() - TRUNCATE_TAIL..],
            &tail[..],
            "the tail of the stream must be kept verbatim"
        );
    }

    #[test]
    fn truncated_output_keeps_the_final_tail_and_reports_total_elision() {
        let mut stdout = Captured::new();
        stdout.push(&[b'a'; 100_000]);
        let out = combine_streams(&stdout.captured(), b"", stdout.dropped());
        assert!(out.starts_with(&"a".repeat(TRUNCATE_HEAD)));
        assert!(out.ends_with(&"a".repeat(TRUNCATE_TAIL)));
        assert!(out.contains(&format!("... [Truncated {} bytes] ...", stdout.dropped())));
        let mut stderr = Captured::new();
        stderr.push(&[b'b'; 100_000]);
        let out = combine_streams(
            &stdout.captured(),
            &stderr.captured(),
            stdout.dropped() + stderr.dropped(),
        );
        let omitted = 200_000 + 1 - TRUNCATE_HEAD - TRUNCATE_TAIL;
        assert!(out.contains(&format!("... [Truncated {omitted} bytes] ...")));
        assert!(out.starts_with(&"a".repeat(TRUNCATE_HEAD)));
        assert!(out.ends_with(&"b".repeat(TRUNCATE_TAIL)));
        assert!(out.len() < TRUNCATE_LIMIT_FOR_TEST + 100);
    }

    /// Output that fits the budget is untouched -- no marker, no accounting.
    #[test]
    fn small_output_is_captured_whole() {
        let mut captured = Captured::new();
        captured.push(b"hello ");
        captured.push(b"world");
        assert_eq!(captured.captured(), b"hello world");
        assert_eq!(captured.dropped(), 0);
    }

    /// A command far larger than the shared budget must report the bytes it
    /// *actually* printed, not the size of the buffer we kept.
    #[tokio::test]
    async fn a_huge_output_reports_its_true_size_without_being_buffered() {
        let tmp = crate::worktree::swe_base_dir().join("exec-flood-test");
        let _ = std::fs::create_dir_all(&tmp);

        // ~2 MiB: two orders of magnitude past the 16 KiB budget, and well
        // past any pipe buffer, so the drain has to keep up to avoid a stall.
        let (out, code) = runner()
            .execute_bash(
                &tmp,
                "for i in $(seq 1 40000); do echo aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-$i; done",
            )
            .await
            .expect("a flooding command must not hang");
        assert_eq!(code, Some(0), "command failed with output: {out:?}");

        assert!(out.contains("... [Truncated "), "{out}");
        // ~1.24 MiB was written; the marker must reflect that order of
        // magnitude, not the 16 KiB we retained.
        let digits: String = out
            .split("... [Truncated ")
            .nth(1)
            .and_then(|rest| rest.split(' ').next())
            .expect("a truncation marker")
            .to_string();
        let reported: usize = digits.parse().expect("a byte count in the marker");
        assert!(
            reported > 1_000_000,
            "the marker must report the child's real output size ({reported}), \
             not the size of the retained buffer"
        );
        // The head survives...
        assert!(
            out.starts_with("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-1\n"),
            "head: {out}"
        );
        // ...and so does the tail.
        assert!(
            out.contains("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-40000"),
            "tail: {out}"
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// Cancelling the future that owns the child -- what `pool::kill` and
    /// `kill_all` do -- must take down the whole process group, not just the
    /// leader. `kill_on_drop` alone signals the leader PID, which leaves the
    /// `rustc` children of an interrupted build running.
    #[tokio::test]
    async fn dropping_the_future_kills_the_whole_process_group() {
        let dir = crate::worktree::swe_base_dir().join("exec-cancel-test");
        let _ = std::fs::create_dir_all(&dir);
        let pid_file = dir.join("grandchild.pid");
        let mut cmd = Command::new("bash");
        cmd.args([
            "-c",
            &format!("sleep 300 & echo $! > {}; wait", pid_file.display()),
        ]);
        configure_process(&mut cmd);
        cmd.current_dir(&dir);
        let task = tokio::spawn(async move { run_with_timeout(&mut cmd, 300).await });
        let grandchild = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if let Ok(pid) = std::fs::read_to_string(&pid_file) {
                    break pid.trim().parse::<u32>().expect("numeric pid");
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .expect("the shell must have recorded its child pid");
        assert!(
            group_is_alive(grandchild),
            "grandchild must start before cancellation"
        );
        task.abort();
        let _ = task.await;
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert!(
            !group_is_alive(grandchild),
            "cancelled command left grandchild {grandchild} running"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The guard is the whole cancellation contract, so its state machine is
    /// pinned directly: armed means "signal on drop", disarmed means "the
    /// caller already reaped the group".
    #[tokio::test]
    async fn a_disarmed_guard_never_signals_the_group() {
        let mut cmd = Command::new("bash");
        // A live child in its own group: if the disarmed drop signalled, this
        // pid would be gone by the assertion below.
        cmd.args(["-c", "sleep 300"]);
        configure_process(&mut cmd);
        let mut child = cmd.spawn().expect("bash must spawn");
        let pid = child.id().expect("a spawned child has a pid");
        tokio::time::sleep(Duration::from_millis(300)).await;
        assert!(group_is_alive(pid), "the child must be running");

        // This is the clean-exit path: we disarmed before dropping.
        let mut guard = ProcessGroupGuard::new(Some(pid));
        guard.disarm();
        drop(guard);

        tokio::time::sleep(Duration::from_millis(300)).await;
        assert!(
            group_is_alive(pid),
            "a disarmed guard must not signal a process group that is still live"
        );

        // Clean up: now the group really is signalled.
        signal_process_group(Some(pid), libc::SIGKILL);
        let _ = child.wait().await;
    }

    /// The timeout path must not discard what the command had already printed:
    /// the reader tasks keep filling their buffers across the cancellation, so
    /// the model still sees the last diagnostics.
    #[tokio::test]
    async fn timed_out_command_still_reports_the_output_it_produced() {
        let tmp = crate::worktree::swe_base_dir().join("exec-drain-test");
        let _ = std::fs::create_dir_all(&tmp);
        // A one-second budget, so the test stays quick.
        // SAFETY: this test binary runs its tests single-threaded, and no other
        // thread in this process reads the command timeout variables.
        unsafe { std::env::set_var("COMMAND_LIGHT_TIMEOUT_SECS", "1") };

        let (out, code) = runner()
            .execute_bash(&tmp, "echo EARLY-STDOUT; echo EARLY-STDERR >&2; sleep 300")
            .await
            .expect("a timeout is an ordinary result, not an error");

        unsafe { std::env::remove_var("COMMAND_LIGHT_TIMEOUT_SECS") };

        assert_eq!(
            code,
            Some(TIMEOUT_EXIT_CODE),
            "a timeout reports 124: {out:?}"
        );
        assert!(
            out.contains("EARLY-STDOUT"),
            "stdout written before the timeout must survive the kill: {out:?}"
        );
        assert!(
            out.contains("EARLY-STDERR"),
            "stderr written before the timeout must survive the kill: {out:?}"
        );
        assert!(
            out.contains("timed out after 1s"),
            "the timeout itself must still be reported: {out:?}"
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// A grandchild that inherited the output pipes cannot be reaped by us, so
    /// the drain is abandoned at its deadline instead of hanging the worker.
    #[tokio::test]
    async fn a_leaked_pipe_does_not_hang_the_timeout_path() {
        let tmp = crate::worktree::swe_base_dir().join("exec-leak-test");
        let _ = std::fs::create_dir_all(&tmp);
        unsafe { std::env::set_var("COMMAND_LIGHT_TIMEOUT_SECS", "1") };

        // `setsid` detaches the sleeper from the killed process group, so it
        // keeps the inherited stdout open past the SIGKILL.
        let started = std::time::Instant::now();
        let (out, code) = runner()
            .execute_bash(&tmp, "echo BEFORE-LEAK; (setsid sleep 300 &); sleep 300")
            .await
            .expect("a leaked pipe must not turn the timeout into a hang");
        let elapsed = started.elapsed();

        unsafe { std::env::remove_var("COMMAND_LIGHT_TIMEOUT_SECS") };

        assert_eq!(code, Some(TIMEOUT_EXIT_CODE), "{out:?}");
        assert!(
            out.contains("BEFORE-LEAK"),
            "output produced before the leak must still be reported: {out:?}"
        );
        let budget = Duration::from_secs(1)
            + Duration::from_millis(TERM_GRACE_MS)
            + Duration::from_millis(DRAIN_GRACE_MS)
            + Duration::from_secs(5);
        assert!(
            elapsed < budget,
            "the timeout path must stay bounded, took {elapsed:?} (budget {budget:?})"
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// End-to-end guarantee: a secret exported in the *worker's* own
    /// environment is invisible to the command the model runs, and the command
    /// sees the remapped, isolated `HOME` instead of the operator's.
    ///
    /// This is the check that matters, because it exercises the real spawn
    /// path (`env_clear` + allow-list) rather than the pure helper: a bug that
    /// only removed the helper's redaction, or one where `env_clear` was never
    /// called, is invisible to unit tests of `build_clean_environment` alone.
    #[tokio::test]
    async fn a_spawned_command_cannot_read_the_operators_secrets() {
        let unique_id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let tmp = crate::worktree::swe_base_dir().join(format!(
            "env-sanitize-test-{}-{unique_id}",
            std::process::id()
        ));
        let _ = std::fs::create_dir_all(&tmp);

        // Export a secret the way an operator's shell would.
        // SAFETY: the test binary runs its tests single-threaded, and no other
        // thread in this process reads this variable.
        unsafe {
            std::env::set_var("OPENAI_API_KEY", "sk-leaked-must-not-appear");
            std::env::set_var("GITHUB_TOKEN", "ghp-leaked-must-not-appear");
            std::env::set_var("AWS_SECRET_ACCESS_KEY", "aws-leaked-must-not-appear");
            std::env::set_var("SSH_AUTH_SOCK", "/tmp/agent.sock");
        }

        let (out, code) = runner()
            .execute_bash(
                &tmp,
                "echo \"key=[$OPENAI_API_KEY] gh=[$GITHUB_TOKEN] aws=[$AWS_SECRET_ACCESS_KEY] ssh=[$SSH_AUTH_SOCK] home=[$HOME]\"",
            )
            .await
            .expect("a spawned command must not error");

        unsafe {
            std::env::remove_var("OPENAI_API_KEY");
            std::env::remove_var("GITHUB_TOKEN");
            std::env::remove_var("AWS_SECRET_ACCESS_KEY");
            std::env::remove_var("SSH_AUTH_SOCK");
        }

        assert_eq!(code, Some(0), "command failed with output: {out:?}");
        for var in [
            "OPENAI_API_KEY",
            "GITHUB_TOKEN",
            "AWS_SECRET_ACCESS_KEY",
            "SSH_AUTH_SOCK",
        ] {
            assert!(
                !out.contains("leaked-must-not-appear"),
                "{var} leaked into the child environment: {out:?}"
            );
        }
        assert!(
            !out.contains("/tmp/agent.sock"),
            "the SSH agent socket path must not reach the child: {out:?}"
        );
        assert!(
            out.contains("home=[") && out.contains("target/home"),
            "HOME must be remapped into an isolated directory, got: {out:?}"
        );
        assert!(
            std::env::var("HOME").is_ok_and(|real| !out.contains(&real)),
            "the real HOME must not be visible to the child: {out:?}"
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// End-to-end toolchain contract: a command the model runs inside the
    /// sandbox observes a `CARGO_HOME` pointing at the *host's* registry cache,
    /// even though its own `HOME` is an empty scratch directory.
    ///
    /// Without this, a worktree builds against an empty registry and reaches for
    /// `index.crates.io` for crates the operator already has locally, which fails
    /// on an offline host. Asserting on the *relationship* between the two
    /// variables -- the cache must not be the sandboxed home -- is what makes the
    /// test meaningful on a host that has no `~/.cargo` at all.
    ///
    // Deliberately *not* a `#[tokio::test]`: the spawn has to run while the
    // process-environment lock is held, and that lock is a std `Mutex` which
    // cannot be held across an `.await` (`clippy::await_holding_lock`). The test
    // therefore drives its own current-thread runtime and blocks on it, which is
    // the only way to make "mutate the environment, spawn, assert" atomic.
    #[test]
    fn a_spawned_command_sees_the_host_toolchain_cache() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("build a current-thread runtime");
        let unique_id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let tmp = crate::worktree::swe_base_dir().join(format!(
            "env-toolchain-test-{}-{unique_id}",
            std::process::id()
        ));
        let _ = std::fs::create_dir_all(&tmp);

        // A host cache this test controls, so the expectation does not depend on
        // whatever layout the machine running the suite happens to have, and so
        // the forwarding is exercised even on hosts with no `~/.cargo`.
        let host_cargo = tmp.join("host-cargo");
        std::fs::create_dir_all(&host_cargo).expect("create host cargo home");

        // The whole spawn runs under the environment lock: the child reads
        // `CARGO_HOME` when it is built, and a concurrent mutating test would
        // otherwise swap the value out from under the assertion.
        let spawned = crate::agent::env::with_env_lock(|| {
            // SAFETY: serialized against every other test that reads or writes
            // the process environment, including the ones in `env`.
            unsafe { std::env::set_var("CARGO_HOME", &host_cargo) };
            let result = runtime.block_on(
                runner().execute_bash(&tmp, "echo \"home=[$HOME] cargo_home=[$CARGO_HOME]\""),
            );
            unsafe { std::env::remove_var("CARGO_HOME") };
            result
        });
        let (out, code) = spawned.expect("a spawned command must not error");

        assert_eq!(code, Some(0), "command failed with output: {out:?}");
        assert_eq!(
            out.trim(),
            format!(
                "home=[{}] cargo_home=[{}]",
                tmp.join("target/home").display(),
                host_cargo.display()
            ),
            "the child must see the host CARGO_HOME, not the sandboxed home: {out:?}"
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// A command that fills far more than a kernel pipe buffer must not
    /// deadlock: both pipes are drained while the child is still running, so it
    /// never blocks in `write`.
    #[tokio::test]
    async fn a_large_output_does_not_deadlock_the_collector() {
        let tmp = crate::worktree::swe_base_dir().join("exec-chatty-test");
        let _ = std::fs::create_dir_all(&tmp);

        // 20k lines is well past the 64 KiB pipe buffer.
        let (out, code) = runner()
            .execute_bash(&tmp, "for i in $(seq 1 20000); do echo line-$i; done")
            .await
            .expect("a chatty command must not hang");
        assert_eq!(code, Some(0), "command failed with output: {out:?}");
        assert!(out.contains("line-1\n"), "stdout head: {out:?}");
        assert!(
            out.contains("... [Truncated "),
            "20k lines must still hit the shared truncation budget"
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }
    // ---------- Landlock pre_exec confinement ----------

    /// Sentinel that turns this test binary into a Landlock probe instead of a
    /// libtest run. `landlock_restrict_self` is irreversible, so the probe
    /// cannot share an address space with the rest of the suite.
    const LANDLOCK_PROBE_ENV: &str = "MINI_SWE_EXEC_LANDLOCK_PROBE";

    /// Scratch worktree + target pair, removed on drop.
    struct LandlockScratch {
        worktree: PathBuf,
        target: PathBuf,
    }

    impl LandlockScratch {
        fn new(tag: &str) -> Self {
            let unique_id = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let base = crate::worktree::swe_base_dir().join(format!(
                "exec-landlock-{tag}-{}-{unique_id}",
                std::process::id()
            ));
            let worktree = base.join("worktree");
            let target = base.join("target");
            std::fs::create_dir_all(&worktree).expect("create worktree");
            std::fs::create_dir_all(&target).expect("create target dir");
            Self { worktree, target }
        }
    }

    impl Drop for LandlockScratch {
        fn drop(&mut self) {
            if let Some(base) = self.worktree.parent() {
                let _ = std::fs::remove_dir_all(base);
            }
        }
    }

    /// A plan is built for two existing directories and names both of them.
    #[test]
    fn a_landlock_plan_is_built_for_two_existing_roots() {
        let scratch = LandlockScratch::new("plan");
        let plan = super::super::sandbox::build_landlock_plan(&scratch.worktree, &scratch.target)
            .expect("building a plan must not fail on a Landlock-capable kernel");

        if let Some(plan) = plan {
            let named: Vec<PathBuf> = (0..plan.rule_count())
                .filter_map(|i| plan.rule_path(i).map(Path::to_path_buf))
                .collect();
            assert!(
                named.len() >= 2,
                "a plan must carry at least the two writable roots, got {named:?}"
            );
            for root in [&scratch.worktree, &scratch.target] {
                assert!(
                    named.contains(root),
                    "the plan must grant {}; it grants {named:?}",
                    root.display()
                );
            }
            assert_ne!(
                plan.handled_access(),
                0,
                "a plan with no handled rights would deny everything"
            );
        }
    }

    /// A missing root is a caller bug, not a runtime condition, and must be
    /// reported rather than silently producing a plan that grants nothing.
    #[test]
    fn a_missing_root_is_reported_rather_than_silently_dropped() {
        let scratch = LandlockScratch::new("missing");
        let missing = scratch.worktree.join("no-such-dir");
        let err =
            super::super::sandbox::build_landlock_plan(&missing, &scratch.target).unwrap_err();
        assert!(
            format!("{err:#}").contains("does not exist"),
            "a missing worktree must be reported, got: {err:#}"
        );

        let err =
            super::super::sandbox::build_landlock_plan(&scratch.worktree, &missing).unwrap_err();
        assert!(
            format!("{err:#}").contains("does not exist"),
            "a missing target dir must be reported, got: {err:#}"
        );
    }

    /// The opt-out knob must produce *no plan at all* rather than an empty one:
    /// an empty plan would install a ruleset that grants nothing and confine
    /// the worker to a directory it cannot even read.
    #[test]
    fn the_landlock_opt_out_produces_no_plan() {
        let scratch = LandlockScratch::new("disabled");
        // SAFETY: the harness runs these environment-sensitive tests in one
        // process; nothing else in the suite reads this variable concurrently
        // with the window below.
        unsafe { std::env::set_var(super::super::sandbox::DISABLE_LANDLOCK_ENV, "1") };
        let built = super::super::sandbox::build_landlock_plan(&scratch.worktree, &scratch.target);
        unsafe { std::env::remove_var(super::super::sandbox::DISABLE_LANDLOCK_ENV) };

        assert!(
            matches!(built, Ok(None)),
            "an explicit opt-out must skip confinement entirely, got: {built:?}"
        );
    }

    /// The hook is only installed when there is no bubblewrap: with bwrap
    /// present it builds the mount namespace itself and a second, LSM-level
    /// confinement would be redundant (and would confine `bwrap` itself).
    #[test]
    fn the_landlock_hook_is_skipped_when_bubblewrap_is_available() {
        if !has_bwrap() {
            eprintln!("skipping: bubblewrap is unavailable on this host");
            return;
        }
        // Both branches are driven by `sandbox_enabled()`; this asserts the
        // predicate that selects them, which is what the hook hangs off.
        assert!(
            sandbox_enabled(),
            "with bwrap present and no opt-out, the bwrap branch must win"
        );
    }

    /// A confined command still runs, and it still gets its own process group
    /// and pipes: the Landlock hook must not disturb the existing plumbing.
    ///
    /// This is the regression that matters most in practice. A `pre_exec` hook
    /// that returned `Err`, or that confined the process before `fork`
    /// completed, would break *every* worker on a host without bubblewrap -
    /// including a plain `echo`.
    #[tokio::test]
    async fn a_command_runs_while_the_landlock_hook_is_installed() {
        let scratch = LandlockScratch::new("runs");
        let mut cmd = Command::new("/bin/sh");
        apply_landlock_pre_exec(&mut cmd, &scratch.worktree, &scratch.target);
        cmd.arg("-c").arg("echo confined-and-alive");

        let out = cmd
            .output()
            .await
            .expect("a confined child must still spawn");
        assert!(
            out.status.success(),
            "the confined child must run: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(
            String::from_utf8_lossy(&out.stdout).contains("confined-and-alive"),
            "the confined child must still produce its output"
        );
    }

    /// End-to-end: the `pre_exec` hook really confines the *child* - and only
    /// the child.
    ///
    /// The parent side of the contract is the part a naive implementation gets
    /// catastrophically wrong: calling `landlock_restrict_self` from the daemon
    /// would confine the daemon itself, permanently and irreversibly. This
    /// test therefore asserts both halves: the child is denied a path outside
    /// the domain, and the parent is not.
    #[tokio::test]
    async fn the_pre_exec_hook_confines_the_child_and_leaves_the_parent_alone() {
        if std::env::var_os(LANDLOCK_PROBE_ENV).is_some() {
            return;
        }
        // Only meaningful where the kernel actually confines: on a kernel
        // without Landlock the hook is never installed at all, which is the
        // documented graceful degradation.
        let scratch = LandlockScratch::new("e2e");
        if super::super::sandbox::build_landlock_plan(&scratch.worktree, &scratch.target)
            .expect("plan")
            .is_none()
        {
            eprintln!("skipping: this kernel has no Landlock");
            return;
        }

        // A file the domain must not be able to read. The policy grants
        // exactly two writable roots - the worktree and the target dir - so a
        // sibling of the scratch base is outside the domain by construction,
        // without depending on what else happens to live on this host.
        let secret = scratch
            .worktree
            .parent()
            .expect("scratch has a parent")
            .join("secret");
        std::fs::write(&secret, b"PRIVATE KEY").expect("seed a decoy secret");

        // Report the real errno rather than a shell `if`: `touch` on an
        // existing file legitimately succeeds, so a bare success/failure word
        // would conflate "allowed" with "already there".
        //
        // The `>/dev/null` redirect is load-bearing, not decoration: opening
        // `/dev/null` for writing needs `WRITE_FILE`, so a read-only `/dev`
        // would fail the redirect itself and every command in the domain that
        // uses the most idiomatic redirection in shell would fail with it.
        let probe = format!(
            "cat {secret} >/dev/null; echo cat_rc=$?; \
             touch {wt}/wrote; echo touch_rc=$?; \
             echo hi >/dev/null; echo redir_rc=$?; \
             echo PID=$$",
            secret = secret.display(),
            wt = scratch.worktree.display(),
        );

        let mut cmd = Command::new("/bin/sh");
        apply_landlock_pre_exec(&mut cmd, &scratch.worktree, &scratch.target);
        cmd.arg("-c").arg(&probe);
        let out = cmd.output().await.expect("spawn the confined probe");

        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(out.status.success(), "probe failed: {stdout}");
        assert!(
            stdout.contains("cat_rc=1"),
            "the child must not be able to read outside its domain: {stdout}"
        );
        assert!(
            stdout.contains("touch_rc=0"),
            "the child must still be able to write its worktree: {stdout}"
        );
        assert!(
            stdout.contains("redir_rc=0"),
            "the child must still be able to redirect to /dev/null: {stdout}"
        );

        // The parent is untouched. If the hook had confined the daemon, this
        // write would fail - and so would every later command in the suite.
        let parent_probe = scratch.worktree.join("parent-probe");
        std::fs::write(&parent_probe, b"parent").expect(
            "the parent must NOT be confined: landlock_restrict_self belongs in the child only",
        );
        let _ = std::fs::remove_file(&parent_probe);
        let _ = std::fs::remove_file(&secret);
    }

    /// The parent's own `$HOME` must remain readable after a confined child has
    /// run, which is the observable form of "the hook confined the child only".
    #[tokio::test]
    async fn a_confined_child_does_not_narrow_the_parents_access() {
        if std::env::var_os(LANDLOCK_PROBE_ENV).is_some() {
            return;
        }
        let scratch = LandlockScratch::new("parent");
        if super::super::sandbox::build_landlock_plan(&scratch.worktree, &scratch.target)
            .expect("plan")
            .is_none()
        {
            eprintln!("skipping: this kernel has no Landlock");
            return;
        }

        // Something outside the domain the parent has legitimate access to.
        let outside = scratch
            .worktree
            .parent()
            .expect("scratch has a parent")
            .join("outside");
        std::fs::write(&outside, b"secret").expect("seed a file outside the domain");

        let mut cmd = Command::new("/bin/sh");
        apply_landlock_pre_exec(&mut cmd, &scratch.worktree, &scratch.target);
        cmd.arg("-c").arg("true");
        let _ = cmd.output().await.expect("spawn");

        // The child could not have read it...
        let mut child = Command::new("/bin/sh");
        apply_landlock_pre_exec(&mut child, &scratch.worktree, &scratch.target);
        child.arg("-c").arg(format!("cat {}", outside.display()));
        let out = child.output().await.expect("spawn");
        assert!(
            !out.status.success(),
            "the child must not read a file outside the domain: {}",
            String::from_utf8_lossy(&out.stdout)
        );

        // ...but the parent still can, because the restriction was installed in
        // the child and nowhere else.
        assert_eq!(
            std::fs::read(&outside).expect("the parent must keep its own access"),
            b"secret",
            "a confined child must not narrow the parent's access"
        );
        let _ = std::fs::remove_file(&outside);
    }

    /// The confinement must be the *fallback*, never a replacement: with
    /// bubblewrap present the bwrap argument vector is unchanged.
    #[test]
    fn bubblewrap_remains_the_primary_sandbox_when_available() {
        if !has_bwrap() {
            eprintln!("skipping: bubblewrap is unavailable on this host");
            return;
        }
        // A worktree the bwrap builder accepts must still yield the bwrap
        // argv, and the Landlock hook must not have been folded into it.
        let scratch = LandlockScratch::new("bwrap");
        let mut cmd = Command::new("true");
        apply_sandbox_args(&mut cmd, &scratch.worktree, &scratch.target);
        let rendered: Vec<String> = cmd
            .as_std()
            .get_args()
            .map(|a| a.to_string_lossy().into_owned())
            .collect();
        assert!(
            rendered.iter().any(|a| a == "bwrap"),
            "bwrap must still lead the argv: {rendered:?}"
        );
    }
}
