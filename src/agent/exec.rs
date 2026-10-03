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
use std::hash::{DefaultHasher, Hash, Hasher};
use std::io::Read;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWriteExt};
use tokio::process::{Child, Command};
use tracing::{info, warn};

use super::AgentRunner;
use super::intercept::{check_command, strip_data_heredocs};
use super::jobs::JobState;
use super::sandbox::{
    KernelConfinement, TRUNCATE_HEAD, TRUNCATE_TAIL, find_git_common_dir, find_git_dirs, has_bwrap,
    is_heavy_command, truncate_with_dropped, validate_bash_command,
};

/// Exit code reported when a command exceeded its wall-clock budget.
pub(super) const TIMEOUT_EXIT_CODE: i32 = 124;

/// Default wall-clock budget (seconds) for heavy commands (builds, test suites).
const DEFAULT_HEAVY_TIMEOUT_SECS: u64 = 600;

/// Default wall-clock budget (seconds) for light commands.
const DEFAULT_LIGHT_TIMEOUT_SECS: u64 = 120;

/// `nice` flag applied to every child so agent work yields to interactive work.
const NICE_FLAG: &str = "-n";
/// `nice` level applied to every child so agent work yields to interactive work.
const NICE_VALUE: &str = "10";

/// Env var choosing a heavy command's I/O scheduling class.
///
/// `0` leaves the inherited class alone, `idle` (or its class number, `3`)
/// opts into the idle class, and anything else — including an unset variable —
/// keeps the best-effort default described on [`IoClass`].
pub const HEAVY_IONICE_ENV: &str = "HUB_HEAVY_IONICE";

/// `IOPRIO_CLASS_SHIFT` from `<linux/ioprio.h>`: a priority value is the class
/// in its high bits and the level in the low ones.
const IOPRIO_CLASS_SHIFT: i32 = 13;

/// `IOPRIO_CLASS_BE`: best-effort, the class every ordinary command runs in.
const IOPRIO_CLASS_BE: i32 = 2;

/// `IOPRIO_CLASS_IDLE`: the disk is only used when nothing else wants it.
const IOPRIO_CLASS_IDLE: i32 = 3;

/// `IOPRIO_WHO_PROCESS`: the priority value addresses one pid.
const IOPRIO_WHO_PROCESS: i32 = 1;

/// Best-effort level the kernel gives a task nobody has re-prioritised
/// (`IOPRIO_BE_NR_LEVELS / 2`), so a light command is pinned to *normal*
/// rather than to the top of its class.
const IOPRIO_BE_NORMAL: i32 = 4;

/// Lowest best-effort level (`IOPRIO_BE_NR_LEVELS - 1`): the bottom of the
/// class, but still a class the scheduler serves whenever the disk is free.
const IOPRIO_BE_LOWEST: i32 = 7;

/// `ionice` arguments for the default heavy class: best-effort, lowest level.
const IONICE_HEAVY_ARGS: &str = "-c2 -n7";

/// `ionice` arguments for the idle class, the opt-in heavy class.
const IONICE_IDLE_ARGS: &str = "-c3";

/// Grace period after `SIGTERM` before a timed-out group is escalated to
/// `SIGKILL`. Long enough to flush buffers, short enough that a wedged build
/// still fails near its budget.
const TERM_GRACE_MS: u64 = 5_000;

/// Grace period after `SIGTERM` before a *finished* step's group is escalated
/// to `SIGKILL`. The step is over, so a job the shell backgrounded with `&`
/// only needs a moment to exit on its own before the escalation.
const STEP_TERM_GRACE_MS: u64 = 500;

/// Poll interval while waiting for a signalled process group to empty.
const GROUP_POLL_MS: u64 = 25;

/// [`TERM_GRACE_MS`] as a [`Duration`], for the callers that pass a grace.
pub(super) const TERM_GRACE: Duration = Duration::from_millis(TERM_GRACE_MS);

/// [`STEP_TERM_GRACE_MS`] as a [`Duration`], for the callers that pass a grace.
const STEP_TERM_GRACE: Duration = Duration::from_millis(STEP_TERM_GRACE_MS);

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

/// Cap on the log file a background job's output streams to.
///
/// A build log is unbounded by nature, so the file is capped: what survives is
/// the most recent window rather than the oldest bytes. The head and tail the
/// model sees come from the bounded in-memory buffer instead, so nothing it
/// could read is lost here.
const JOB_LOG_MAX_BYTES: u64 = 4 * 1024 * 1024;

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

        let parallelism = build_parallelism(self.build_jobs);
        // The worker already holds its build directory for its whole lifetime
        // (see `WorktreeGuard::build_dir`), so a step only reads the
        // grant: no second lock is taken per command.
        let target_dir = self.build_target_dir.clone();
        let sandbox_target = target_dir.as_deref().unwrap_or(dir);
        let tmp_dir = crate::worktree::scratch_dir(dir);
        std::fs::create_dir_all(&tmp_dir).context("Failed to create worker scratch directory")?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&tmp_dir, std::fs::Permissions::from_mode(0o700))
                .context("Failed to make worker scratch private")?;
        }

        let mut cmd = Command::new("nice");
        configure_process(&mut cmd);

        // Classify on the model's own command, before any offline wrapper is
        // applied: a wrapper would otherwise mask the command's heaviness
        // from the timeout classifier.
        let heavy = is_heavy_command(command);
        let timeout_secs = self
            .command_timeout_override
            .unwrap_or_else(|| command_timeout_secs(heavy));
        // Disk priority rides the same classification: a saturated disk must
        // slow the build down, not the light commands around it.
        let io_plan = io_plan(
            heavy,
            ioprio_syscall_supported(),
            heavy_ionice_setting(std::env::var(HEAVY_IONICE_ENV).ok().as_deref()),
        );

        match select_backend() {
            SandboxBackend::Kernel => {
                // The kernel confines the forked child itself. Offline needs
                // no wrapper when the seccomp filter denies INET sockets; if
                // this kernel could not install one, the network namespace
                // still enforces the policy rather than silently dropping it.
                let network_denied = match apply_kernel_confinement(
                    &mut cmd,
                    dir,
                    sandbox_target,
                    self.network_offline,
                    io_plan,
                ) {
                    Ok(denied) => denied,
                    Err(e) => {
                        tracing::error!(
                            error = %format!("{e:#}"),
                            worktree = %dir.display(),
                            "sandbox could not be prepared; refusing to run the command"
                        );
                        return Ok((
                            format!(
                                "BLOCKED: the sandbox could not be prepared ({e:#}); the command was not run."
                            ),
                            Some(1),
                        ));
                    }
                };
                let final_command = if network_denied {
                    command.to_string()
                } else {
                    wrap_network_command(command, self.network_offline)
                };
                let final_command = apply_io_wrapper(final_command, io_plan);
                cmd.current_dir(dir)
                    .args([NICE_FLAG, NICE_VALUE, "bash", "-c", &final_command]);
            }
            SandboxBackend::Bwrap => {
                // bubblewrap builds the mount namespace itself; adding
                // Landlock here would only risk re-confining a process bwrap
                // already confined. Offline still needs its network namespace.
                let final_command = wrap_network_command(command, self.network_offline);
                let final_command = apply_io_wrapper(final_command, io_plan);
                apply_sandbox_args(&mut cmd, dir, sandbox_target);
                cmd.args(["--chdir", &dir.to_string_lossy()]);
                cmd.args(["/usr/bin/bash", "-c", &final_command]);
                if let IoPlan::Set(class) = io_plan {
                    apply_io_priority(&mut cmd, class);
                }
            }
            SandboxBackend::Unconfined => {
                warn_unconfined_once();
                // No confinement is never "with network access" when the
                // policy says offline.
                let final_command = wrap_network_command(command, self.network_offline);
                let final_command = apply_io_wrapper(final_command, io_plan);
                cmd.current_dir(dir)
                    .args([NICE_FLAG, NICE_VALUE, "bash", "-c", &final_command]);
                if let IoPlan::Set(class) = io_plan {
                    apply_io_priority(&mut cmd, class);
                }
            }
        }

        // Cleared environment + strict allow-list first, so no ambient
        // credential from the operator's shell reaches the model. Build/cache
        // variables are layered on top afterwards, then the per-command
        // overlay (the differential verify gate's divergent environment), so
        // the divergent values win over every default.
        apply_sanitized_environment(&mut cmd, dir);
        apply_build_env(&mut cmd, target_dir.as_deref(), &tmp_dir, &parallelism);
        crate::cache::apply_shared_cache_env(&mut cmd);
        for (name, value) in &self.extra_env {
            // The overlay is the variant-B environment: credential-bearing
            // names are refused here as well, so a tampered snapshot can never
            // ride the extra env into a child.
            if crate::agent::env::is_secret_name(name) {
                continue;
            }
            cmd.env(name, value);
        }

        // A command that outlives its budget keeps running as a background
        // job, so its output needs a log to stream to. The worker's private
        // scratch is where the step's own scratch already lives, and it is
        // deleted with the worktree.
        let job_log_dir = self.job_handle().map(|_| tmp_dir.clone());
        match run_with_timeout(&mut cmd, timeout_secs, job_log_dir.as_deref()).await? {
            RunOutcome::Finished { output, code } => Ok((output, code)),
            RunOutcome::Backgrounded(backgrounded) => {
                Ok(self.continue_as_job(*backgrounded, command).await)
            }
        }
    }

    /// Report a command that outlived its budget.
    ///
    /// With a job table the command keeps running as job `<n>`, which the
    /// worker can wait on or stop; without one there is nobody to wait on it,
    /// so it is stopped here and reported as the timeout it is.
    async fn continue_as_job(
        &self,
        backgrounded: Backgrounded,
        command: &str,
    ) -> (String, Option<i32>) {
        let Backgrounded {
            pid,
            mut child,
            out,
            err,
            timeout_secs,
            log,
            mut guard,
        } = backgrounded;
        let Some(handle) = self.job_handle() else {
            // Nobody owns this runner's jobs, so there is nobody to wait on the
            // command: stop it and report the timeout it is.
            terminate_process_group(pid, &mut child, TERM_GRACE).await;
            let (out, err) = tokio::join!(out.finish(), err.finish());
            let mut output = combine_streams(&out.bytes, &err.bytes, out.dropped + err.dropped);
            output.push_str(&format!(
                "\nCommand timed out after {timeout_secs}s and was terminated."
            ));
            return (output, Some(TIMEOUT_EXIT_CODE));
        };
        let job = JobState::new(pid, child, out, err, job_label(command), log.clone());
        let id = handle.spawn(job);
        // The job owns the group now, so the guard must not signal it on drop.
        guard.disarm();
        self.set_last_job_id(id);
        info!(
            job = id,
            timeout_secs,
            log = %log.display(),
            "Command outlived its budget; continuing as a background job"
        );
        // Reported with the timeout's exit code, not 0: the command has not
        // finished, and a completion gate that outlived its budget has not
        // passed.
        (
            backgrounded_message(id, timeout_secs, &log),
            Some(TIMEOUT_EXIT_CODE),
        )
    }
}

/// Fingerprint the worktree content from git's view, so a step can report
/// the exact tree it ran on.
///
/// The fingerprint is the `HEAD` commit, the binary diff of every tracked
/// change, and the untracked non-ignored files each hashed with its bytes.
/// It changes whenever any file a suite could observe changes -- a tracked
/// edit, a new commit, or a created or edited untracked file -- and stays
/// identical otherwise, which is what lets the completion gate reuse a
/// verify run on an unchanged tree. `None` when git could not answer, so a
/// fingerprint that could not be taken is never read as "unchanged".
pub(crate) fn tree_fingerprint(dir: &Path) -> Option<String> {
    let head = crate::worktree::git(dir, "rev-parse HEAD", &["rev-parse", "HEAD"]).ok()?;
    let diff = crate::worktree::git(
        dir,
        "diff HEAD --binary",
        &["diff", "--no-ext-diff", "--no-textconv", "HEAD", "--binary"],
    )
    .ok()?;
    let others = crate::worktree::git(
        dir,
        "ls-files --others --exclude-standard",
        &["ls-files", "--others", "--exclude-standard", "-z"],
    )
    .ok()?;
    if !head.status.success() || !diff.status.success() || !others.status.success() {
        return None;
    }
    let mut hasher = DefaultHasher::new();
    head.stdout.hash(&mut hasher);
    diff.stdout.hash(&mut hasher);
    let mut names: Vec<_> = others
        .stdout
        .split(|b| *b == 0)
        .filter(|p| !p.is_empty())
        .collect();
    names.sort_unstable();
    let mut buffer = [0; 8192];
    for name in names {
        name.hash(&mut hasher);
        let path = dir.join(std::ffi::OsStr::from_bytes(name));
        let metadata = std::fs::symlink_metadata(&path).ok()?;
        metadata.permissions().mode().hash(&mut hasher);
        if metadata.file_type().is_symlink() {
            std::fs::read_link(&path)
                .ok()?
                .as_os_str()
                .as_bytes()
                .hash(&mut hasher);
        } else if metadata.is_file() {
            // Stream large untracked files rather than allocating their contents.
            let mut file = std::fs::File::open(path).ok()?;
            let mut size = 0u64;
            loop {
                let count = file.read(&mut buffer).ok()?;
                if count == 0 {
                    break;
                }
                hasher.write(&buffer[..count]);
                size += count as u64;
            }
            size.hash(&mut hasher);
        } else {
            return None;
        }
    }
    Some(format!("{:016x}", hasher.finish()))
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

/// Build/test parallelism for the child: `BUILD_PARALLELISM`, else the
/// admission controller's granted job count, else half the available cores
/// (never below one).
fn build_parallelism(granted: Option<usize>) -> String {
    if let Some(parallelism) = crate::config::env_parse::<usize>("BUILD_PARALLELISM") {
        return parallelism.to_string();
    }
    granted
        .map(|jobs| jobs.max(1).to_string())
        .unwrap_or_else(|| crate::config::half_the_cores().to_string())
}

/// A light step without slot affinity leaves Cargo's normal target selection intact.
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

/// Pick the backend for the next worker step.
///
/// The kernel confines every step unless the operator opted out entirely
/// (`SWE_DISABLE_SANDBOX=1`) or back into bubblewrap (`SWE_SANDBOX=bwrap`).
/// A kernel with neither Landlock nor seccomp cannot confine anything, so it
/// falls back to bubblewrap when installed and otherwise runs unconfined.
pub(crate) fn select_backend() -> SandboxBackend {
    choose_backend(
        sandbox_disabled(),
        bwrap_requested(),
        KernelConfinement::probe_available,
        has_bwrap,
    )
}

/// The selection rule itself, free of environment and host probes so it can
/// be tested without mutating process-global state under concurrent tests.
fn choose_backend(
    disabled: bool,
    bwrap_opt_in: bool,
    kernel_available: impl FnOnce() -> bool,
    bwrap_installed: impl Fn() -> bool,
) -> SandboxBackend {
    if disabled {
        return SandboxBackend::Unconfined;
    }
    if bwrap_opt_in && bwrap_installed() {
        return SandboxBackend::Bwrap;
    }
    if kernel_available() {
        return SandboxBackend::Kernel;
    }
    if bwrap_installed() {
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
///
/// Returns whether the installed hook itself denies the network for an
/// `offline` step (the seccomp filter refuses every INET socket), so the
/// caller knows when a network namespace is still required.
#[cfg(unix)]
fn apply_kernel_confinement(
    cmd: &mut Command,
    dir: &Path,
    target_dir: &Path,
    offline: bool,
    io_plan: IoPlan,
) -> Result<bool> {
    // Parent side of the hook: everything that allocates happens here, so the
    // closure below is reduced to syscalls. A `None` plan means this host
    // cannot confine the process, which is not an error. A plan that cannot be
    // built (the worktree or the target dir is gone) fails CLOSED: the command
    // is not run rather than run without the confinement it was promised.
    let confinement = match KernelConfinement::prepare(dir, target_dir, offline)? {
        Some(confinement) => confinement,
        None => return Ok(false),
    };
    let network_denied = offline && confinement.has_seccomp();

    // SAFETY: the closure runs in the child between `fork` and `exec`, where
    // `confinement.apply` performs only raw syscalls - it allocates nothing,
    // takes no lock and never unwinds - and confines only that child, never
    // the parent.
    // The I/O class rides this same closure rather than a second `pre_exec`
    // registration, so the confinement can never be the half that is dropped.
    let io_hook = match io_plan {
        IoPlan::Set(class) => Some(IoPriorityHook(class)),
        IoPlan::Wrap(_) | IoPlan::Inherit => None,
    };

    unsafe {
        cmd.pre_exec(move || {
            // SAFETY: forwarded from this function's contract; see above.
            confinement.apply()?;
            if let Some(hook) = &io_hook {
                let _ = hook.apply();
            }
            Ok(())
        });
    }
    Ok(network_denied)
}

/// Non-unix stub: no Landlock LSM, no seccomp and no `fork` to hook.
#[cfg(not(unix))]
fn apply_kernel_confinement(
    _cmd: &mut Command,
    _dir: &Path,
    _target_dir: &Path,
    _offline: bool,
    _io_plan: IoPlan,
) -> Result<bool> {
    Ok(false)
}

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

/// I/O scheduling class a command's process group runs in.
///
/// The CPU side of "agent work yields to everything else" is already `nice`;
/// this is the disk side of the same idea. A build is I/O-bound and mostly
/// indifferent to latency, while the light commands that make up most steps
/// (grep, sed, git, cat) are latency-bound and do almost no I/O. Sending the
/// first to the bottom of the best-effort class is what keeps the second
/// responsive when several builds saturate the disk.
///
/// The idle class is deliberately *not* the default: it is only served when no
/// best-effort or realtime queue anywhere on the host has pending I/O, so on a
/// machine with continuous background traffic (sync clients, network mounts,
/// browsers) an idle build can starve for minutes. Best-effort at its lowest
/// level reduces contention without requiring every other queue to be empty.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum IoClass {
    /// Best-effort at the normal level: the class an ordinary command runs in.
    BestEffort,
    /// Best-effort at the lowest level: the default for a heavy command.
    Heavy,
    /// Idle: the disk is only used once nothing else wants it. Opt-in.
    Idle,
}

/// How [`HEAVY_IONICE_ENV`] configures a heavy command's I/O class.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum HeavyIonice {
    /// Best-effort at the lowest level: the default.
    Lowest,
    /// The idle class, for hosts that want it.
    Idle,
    /// Leave the inherited class alone.
    Inherit,
}

/// Read [`HEAVY_IONICE_ENV`].
///
/// `idle` (or its class number, `3`) opts a host into the idle class, `0`
/// disables the demotion outright, and anything else — including an unset or
/// unparsable variable — keeps the best-effort default.
fn heavy_ionice_setting(raw: Option<&str>) -> HeavyIonice {
    match raw.map(str::trim) {
        Some("0") => HeavyIonice::Inherit,
        Some("idle") | Some("3") => HeavyIonice::Idle,
        _ => HeavyIonice::Lowest,
    }
}

/// How a command's child is placed in its I/O scheduling class.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum IoPlan {
    /// Install the class with `ioprio_set` in the child's `pre_exec` hook.
    Set(IoClass),
    /// Wrap the command in `ionice`: this host has no working `ioprio_set`.
    Wrap(IoClass),
    /// Leave the class the child inherits alone.
    Inherit,
}

/// Decide how `command`'s child gets its I/O scheduling class.
///
/// A heavy command is sent to the bottom of the best-effort class so a build
/// cannot starve the light commands that make up most steps, and
/// [`HEAVY_IONICE_ENV`] picks between that, the idle class and no demotion at
/// all. A light command is pinned to best-effort *normal* so it can never
/// inherit a demotion from whatever spawned it. `syscall_ok` picks the
/// mechanism: `ioprio_set` in the child, or the `ionice` wrapper on a host
/// where the syscall does not answer.
fn io_plan(heavy: bool, syscall_ok: bool, ionice: HeavyIonice) -> IoPlan {
    let class = match (heavy, ionice) {
        (false, _) => IoClass::BestEffort,
        // `HUB_HEAVY_IONICE=0` disables the feature outright: the child keeps
        // whatever class it inherited, demotion and pinning alike.
        (true, HeavyIonice::Inherit) => return IoPlan::Inherit,
        (true, HeavyIonice::Idle) => IoClass::Idle,
        (true, HeavyIonice::Lowest) => IoClass::Heavy,
    };
    if syscall_ok {
        return IoPlan::Set(class);
    }
    match class {
        // No `ioprio_set` on this host, so `ionice` is the only way to demote.
        IoClass::Heavy | IoClass::Idle => IoPlan::Wrap(class),
        // Nothing to demote: the child keeps the class it inherited.
        IoClass::BestEffort => IoPlan::Inherit,
    }
}

/// The `ioprio_set` argument for `class`: the class in the high bits, the
/// level below it.
fn ioprio_value(class: IoClass) -> i32 {
    let (class, level) = match class {
        IoClass::BestEffort => (IOPRIO_CLASS_BE, IOPRIO_BE_NORMAL),
        IoClass::Heavy => (IOPRIO_CLASS_BE, IOPRIO_BE_LOWEST),
        // The idle class has no levels; the low bits are ignored.
        IoClass::Idle => (IOPRIO_CLASS_IDLE, 0),
    };
    (class << IOPRIO_CLASS_SHIFT) | level
}

/// Whether `ioprio_set` answers on this host, probed once.
///
/// The probe asks for a pid that cannot exist, so it changes nothing: `ESRCH`
/// means the kernel implements the syscall, anything else means it cannot be
/// used here and the `ionice` wrapper has to carry the demotion instead.
fn ioprio_syscall_supported() -> bool {
    static SUPPORTED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *SUPPORTED.get_or_init(|| {
        // SAFETY: a raw syscall over three integers that reads and writes no
        // memory we own; the pid is deliberately nonexistent.
        let rc = unsafe {
            libc::syscall(
                libc::SYS_ioprio_set,
                IOPRIO_WHO_PROCESS as libc::c_long,
                -1,
                ioprio_value(IoClass::Idle) as libc::c_long,
            )
        };
        rc != 0 && std::io::Error::last_os_error().raw_os_error() == Some(libc::ESRCH)
    })
}

/// Child-side hook placing the calling process in an I/O scheduling class.
///
/// Tolerant by design: a kernel or a locked-down container that refuses the
/// call leaves the inherited class rather than failing the step, because a
/// command that runs at the wrong I/O priority still beats one that never
/// runs.
struct IoPriorityHook(IoClass);

impl IoPriorityHook {
    fn apply(&self) -> std::io::Result<()> {
        // SAFETY: `ioprio_set` takes three integers and touches no memory we
        // own, and it is async-signal-safe -- which is exactly what a
        // `pre_exec` closure in the forked child of a multi-threaded server
        // is allowed to do.
        let rc = unsafe {
            libc::syscall(
                libc::SYS_ioprio_set,
                IOPRIO_WHO_PROCESS as libc::c_long,
                0,
                ioprio_value(self.0) as libc::c_long,
            )
        };
        if rc != 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(())
    }
}

/// Install the I/O class on `cmd`'s child through its `pre_exec` hook.
///
/// Only for the backends that register no other hook. The kernel backend
/// carries the class inside its confinement closure instead: a second
/// registration that replaced the confinement would run the command
/// unconfined, which is not a price worth any priority.
#[cfg(unix)]
fn apply_io_priority(cmd: &mut Command, class: IoClass) {
    let hook = IoPriorityHook(class);
    // SAFETY: the closure runs in the child between `fork` and `exec` and does
    // nothing but the raw syscall above, which allocates nothing and takes no
    // lock.
    unsafe {
        cmd.pre_exec(move || {
            let _ = hook.apply();
            Ok(())
        });
    }
}

/// Non-unix stub: no I/O scheduling class to install.
#[cfg(not(unix))]
fn apply_io_priority(_cmd: &mut Command, _class: IoClass) {}

/// Apply the `ionice` fallback to an already-wrapped command string.
///
/// `ionice` sets its own class and then execs the shell, so the whole process
/// group inherits that class just as the syscall path would leave it.
fn apply_io_wrapper(command: String, plan: IoPlan) -> String {
    let args = match plan {
        IoPlan::Wrap(IoClass::Heavy) => IONICE_HEAVY_ARGS,
        IoPlan::Wrap(IoClass::Idle) => IONICE_IDLE_ARGS,
        IoPlan::Wrap(IoClass::BestEffort) | IoPlan::Set(_) | IoPlan::Inherit => return command,
    };
    format!("ionice {args} {OFFLINE_SHELL} -c {}", shell_quote(&command))
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

    // Shared build target and worker-private scratch read-write.
    cmd.args(["--bind", &target_str, &target_str]);
    let scratch = crate::worktree::scratch_dir(dir);
    let scratch_str = scratch.to_string_lossy();
    cmd.args(["--bind", &scratch_str, &scratch_str]);

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

/// `WORKER_BUILD_DEBUG=1` keeps Cargo's default (full) debug info in worker
/// builds.
const WORKER_BUILD_DEBUG_VAR: &str = "WORKER_BUILD_DEBUG";

/// Cargo settings that keep a worker's private target directory small.
///
/// Debug info is the bulk of what a Rust build writes, and a worker's target
/// directory is rebuilt from scratch far more often than it is reused, so the
/// DWARF can dwarf the object code that is actually verifiable. Incremental
/// state is the same trade: a large, worker-private artefact set whose reuse
/// does not survive the next worker. Neither is needed to *read* a failure -
/// an assertion message is printed verbatim, and a panic still prints a
/// backtrace, because the symbol names come from the symbol table that
/// `debug = 0` leaves in place.
///
/// The defaults apply only when the operator has not spoken: a value already
/// exported is forwarded verbatim (`lookup` reports it), and
/// `WORKER_BUILD_DEBUG=1` disables the defaults outright. `lookup` is the
/// process environment in production and a synthetic map in tests, so the
/// policy is testable without mutating process-global state.
fn cargo_artifact_diet(
    lookup: &dyn Fn(&str) -> Option<std::ffi::OsString>,
) -> Vec<(String, String)> {
    let keep_debug = lookup(WORKER_BUILD_DEBUG_VAR).as_deref() == Some(std::ffi::OsStr::new("1"));
    [
        "CARGO_PROFILE_DEV_DEBUG",
        "CARGO_PROFILE_TEST_DEBUG",
        "CARGO_INCREMENTAL",
    ]
    .into_iter()
    .filter_map(|name| match lookup(name) {
        Some(value) => Some((name.to_string(), value.to_string_lossy().into_owned())),
        // The operator asked for full debug info: let Cargo apply its default.
        None if keep_debug => None,
        None => Some((name.to_string(), "0".to_string())),
    })
    .collect()
}

/// The operator's environment, as [`cargo_artifact_diet`] reads it.
fn operator_build_env(name: &str) -> Option<std::ffi::OsString> {
    std::env::var_os(name)
}

/// Universal build/test environment: parallelism caps so a command cannot
/// oversubscribe the machine no matter which build tool it drives, a private
/// scratch directory, and the Cargo artifact diet ([`cargo_artifact_diet`])
/// that keeps a Rust worker's target directory small.
fn apply_build_env(
    cmd: &mut Command,
    target_dir: Option<&Path>,
    tmp_dir: &Path,
    parallelism: &str,
) {
    if let Some(target) = target_dir {
        cmd.env("CARGO_TARGET_DIR", target);
    }
    // Trim what a per-worker target directory writes; see [`cargo_artifact_diet`].
    for (name, value) in cargo_artifact_diet(&operator_build_env) {
        cmd.env(name, value);
    }
    // Private scratch is never shared with another slot user.
    cmd.env("TMPDIR", tmp_dir)
        .env("TMP", tmp_dir)
        .env("TEMP", tmp_dir)
        // A mini-swe run nested inside the step (this crate's own test suite,
        // or a worker driving the CLI) keeps its scratch data there too:
        // the default `/var/tmp` is outside every sandbox's writable set.
        .env("SWE_TEMP_DIR", tmp_dir)
        .env("CARGO_BUILD_JOBS", parallelism)
        .env("RUST_TEST_THREADS", parallelism)
        .env("NEXTEST_TEST_THREADS", parallelism)
        .env("MAKEFLAGS", format!("-j{parallelism}"))
        .env("CMAKE_BUILD_PARALLEL_LEVEL", parallelism)
        .env("RAYON_NUM_THREADS", parallelism)
        .env("OMP_NUM_THREADS", parallelism)
        .env("OPENBLAS_NUM_THREADS", parallelism)
        .env("MKL_NUM_THREADS", parallelism)
        .env("GOMAXPROCS", parallelism)
        // JVM: Maven's `-T` thread count and Gradle's worker cap. Both are
        // read as JVM/system properties, so the same granted job count as
        // Cargo's `CARGO_BUILD_JOBS` applies without touching the command.
        .env("MAVEN_OPTS", format!("-T{parallelism}"))
        .env(
            "GRADLE_OPTS",
            format!("-Dorg.gradle.workers.max={parallelism}"),
        )
        // Python: pytest-xdist sizes its worker pool from this variable.
        .env("PYTEST_XDIST_AUTO_NUM_WORKERS", parallelism);
}

/// Wall-clock budget (seconds) for a command of weight `heavy`: heavy commands
/// get a longer default; `COMMAND_TIMEOUT_SECS` overrides either tier.
///
/// `heavy` is the caller's classification of the model's own command, so the
/// budget and the I/O priority can never disagree about what a command is.
fn command_timeout_secs(heavy: bool) -> u64 {
    let default_timeout = if heavy {
        crate::config::env_parse("COMMAND_HEAVY_TIMEOUT_SECS").unwrap_or(DEFAULT_HEAVY_TIMEOUT_SECS)
    } else {
        crate::config::env_parse("COMMAND_LIGHT_TIMEOUT_SECS").unwrap_or(DEFAULT_LIGHT_TIMEOUT_SECS)
    };
    crate::config::env_parse("COMMAND_TIMEOUT_SECS").unwrap_or(default_timeout)
}

/// What became of a command that ran under [`run_with_timeout`].
enum RunOutcome {
    /// The command ended within its budget.
    Finished { output: String, code: Option<i32> },
    /// The command outlived its budget and is still running in its own group.
    ///
    /// Boxed: the pipes and the guard dwarf the finished output, and this value
    /// is moved once per command.
    Backgrounded(Box<Backgrounded>),
}

/// A command that outlived its budget, still running in its own process group.
struct Backgrounded {
    /// Group leader, for whoever takes the group over.
    pid: Option<u32>,
    child: Child,
    out: PipeBuffer,
    err: PipeBuffer,
    /// The budget the command just exhausted.
    timeout_secs: u64,
    /// Where its output streams to, inside the worker's private scratch.
    log: PathBuf,
    /// Still armed: whoever takes the group over disarms it, so a cancellation
    /// in between cannot leave the process running.
    guard: ProcessGroupGuard,
}

/// Spawn `cmd`, wait up to `timeout_secs`, and collect the combined output.
///
/// A command that ends within its budget has its whole process group taken
/// down with it -- a job the shell backgrounded with `&` is still a member, and
/// leaving it running would outlive the worker that started it -- and its
/// output is reported with the real exit code.
///
/// A command that *outlives* its budget is not killed: it is handed back as
/// [`RunOutcome::Backgrounded`], still running in its own group with its pipes
/// still draining, for the caller to turn into a background job. `log_dir` is
/// where that job's output streams to; `None` keeps the in-memory drain only,
/// which is all a command that is not going to outlive its budget needs.
async fn run_with_timeout(
    cmd: &mut Command,
    timeout_secs: u64,
    log_dir: Option<&Path>,
) -> Result<RunOutcome> {
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
    // Named after the process, which is unique per job and known before the
    // job number is.
    let job_log = log_dir.map(|dir| dir.join(format!("job-{}.log", child_pid.unwrap_or(0))));
    let out_buf = PipeBuffer::spawn(stdout, job_log.clone());
    let err_buf = PipeBuffer::spawn(stderr, job_log.clone());

    match tokio::time::timeout(Duration::from_secs(timeout_secs), child.wait()).await {
        // Clean exit: the write ends are closed now that the child is gone, so
        // the readers reach EOF and the real output and code are reported.
        Ok(Ok(status)) => {
            // The step is over, so the group goes down with it: a job the shell
            // backgrounded with `&` is still a member, and leaving it running
            // would outlive the worker that started it.
            terminate_process_group(child_pid, &mut child, STEP_TERM_GRACE).await;
            // The group is already reaped; the guard must not signal it again.
            group_guard.disarm();
            let (out, err) = tokio::join!(out_buf.finish(), err_buf.finish());
            Ok(RunOutcome::Finished {
                output: combine_streams(&out.bytes, &err.bytes, out.dropped + err.dropped),
                code: status.code(),
            })
        }
        // A `wait` error leaves the child's fate unknown, so the guard stays
        // armed: killing the group is the only cleanup that cannot leave a
        // build running behind us. Signalling a reaped group is inert
        // (`ESRCH`), and the PID cannot be recycled while tokio holds the
        // unreaped `Child`.
        Ok(Err(e)) => {
            // The child's fate is unknown, so the group is taken down here too:
            // an error is an end of the step like any other.
            terminate_process_group(child_pid, &mut child, STEP_TERM_GRACE).await;
            Err(e).context("Failed waiting for bash process")
        }
        // The child outlived its budget. It is *not* stopped here: the caller
        // turns it into a background job, so the group guard stands down only
        // once someone has taken the group over, and the pipes keep draining
        // into the job's log.
        Err(_elapsed) => Ok(RunOutcome::Backgrounded(Box::new(Backgrounded {
            pid: child_pid,
            child,
            out: out_buf,
            err: err_buf,
            timeout_secs,
            log: job_log.unwrap_or_else(|| PathBuf::from("job.log")),
            guard: group_guard,
        }))),
    }
}

/// The tool result a command that outlived its budget produces.
///
/// The command is still running, so nothing about its outcome is reported: the
/// worker is told the job number, how to wait on it and how to stop it.
fn backgrounded_message(id: u64, timeout_secs: u64, log: &Path) -> String {
    format!(
        "Command is still running after {timeout_secs}s as job {id}; it was not killed.\n\
         `echo WAIT_JOB {id}` waits for it (up to {}s per call) and then reports its exit code and the tail of its output.\n\
         `echo KILL_JOB {id}` stops it. It is also stopped after {}s or when this worker ends.\n\
         Its output is streaming to {}.",
        super::jobs::wait_job_secs(),
        super::jobs::job_max_secs(),
        log.display()
    )
}

/// Bounded one-line label for a background job, from the command that started
/// it. Local to the agent layer, which does not depend on the pool that drives
/// it, and bounded so one long command can never blow up a status view.
fn job_label(command: &str) -> String {
    let first = command.lines().next().unwrap_or("").trim();
    let cut = first.len().min(60);
    let mut label = first[..first.floor_char_boundary(cut)].to_string();
    if label.len() < first.len() {
        label.push_str("...");
    }
    label
}

/// Head and tail of a command's output stream, bounded to exactly the bytes
/// [`truncate_output`] would have kept.
pub(super) struct Captured {
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
    pub(super) fn captured(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.head.len() + self.tail.len());
        out.extend_from_slice(&self.head);
        out.extend_from_slice(&self.tail);
        out
    }

    /// Bytes this stream produced that the retained head/tail do not hold.
    pub(super) fn dropped(&self) -> usize {
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
pub(super) struct PipeBuffer {
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

/// Open (creating if needed) the log a background job streams to.
async fn open_job_log(path: &Path) -> Option<tokio::fs::File> {
    match tokio::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .await
    {
        Ok(file) => Some(file),
        Err(e) => {
            tracing::debug!(error = %e, path = %path.display(), "could not open a job log");
            None
        }
    }
}

/// Append `chunk` to a job's log, keeping the file within [`JOB_LOG_MAX_BYTES`].
///
/// Returns the new byte count. Past the cap the log restarts, so what survives
/// is the most recent window of a stream that has no natural end.
async fn append_job_log(file: &mut tokio::fs::File, chunk: &[u8], written: u64) -> u64 {
    let mut written = written;
    if written + chunk.len() as u64 > JOB_LOG_MAX_BYTES {
        let _ = file.set_len(0).await;
        written = 0;
    }
    if file.write_all(chunk).await.is_err() {
        return written;
    }
    written + chunk.len() as u64
}

/// One drained pipe: the bytes worth showing, plus how many were elided.
///
/// `dropped` lets [`combine_streams`] tell "the child printed 20 KB" from "the
/// child printed 4 GB" and report the right figure in the truncation marker.
pub(super) struct Stream {
    bytes: Vec<u8>,
    dropped: usize,
}

impl PipeBuffer {
    /// Start draining `pipe` into a fresh bounded buffer, appending the raw
    /// stream to `log` when one is given.
    ///
    /// The log is where a background job's output streams while the worker
    /// waits on it: the bytes the command printed, capped so a chatty build
    /// cannot fill the worker's scratch. A log that cannot be written is
    /// dropped rather than failing the drain -- the pipe must keep being read
    /// either way, or the child wedges in `write`.
    fn spawn<R>(mut pipe: R, log: Option<PathBuf>) -> Self
    where
        R: AsyncRead + Unpin + Send + 'static,
    {
        let bytes = Arc::new(Mutex::new(Captured::new()));
        let sink = Arc::clone(&bytes);
        let reader = tokio::spawn(async move {
            let mut log = match log {
                Some(path) => open_job_log(&path).await,
                None => None,
            };
            let mut written = 0u64;
            let mut buf = [0u8; DRAIN_CHUNK_BYTES];
            loop {
                // A read failure means the child is gone; whatever arrived is
                // still worth reporting, so the loop only stops at EOF (`0`)
                // and the buffer is flushed either way.
                match pipe.read(&mut buf).await {
                    Ok(0) | Err(_) => break,
                    Ok(n) => {
                        match sink.lock() {
                            Ok(mut sink) => sink.push(&buf[..n]),
                            // The owning task panicked while holding the lock; a
                            // poisoned mutex still holds what was captured so far.
                            Err(poisoned) => poisoned.into_inner().push(&buf[..n]),
                        }
                        if let Some(file) = log.as_mut() {
                            written = append_job_log(file, &buf[..n], written).await;
                        }
                    }
                }
            }
        });
        Self { bytes, reader }
    }

    /// The buffer this pipe drains into, shared with its reader task.
    ///
    /// A background job reads its tail through this while the job is still
    /// running, long before [`finish`](Self::finish) collects the drain.
    pub(super) fn shared(&self) -> Arc<Mutex<Captured>> {
        Arc::clone(&self.bytes)
    }

    /// Stop waiting for the reader and return the bytes worth reporting.
    ///
    /// Bounded by [`DRAIN_GRACE_MS`] because a grandchild that inherited the
    /// write end can hold the pipe open long after its parent was killed; the
    /// reader is then cancelled and the bytes captured up to that point
    /// returned.
    pub(super) async fn finish(mut self) -> Stream {
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
/// after the group goes quiet or `term_grace` elapses.
///
/// Called on *every* way a step can end -- clean exit, error and timeout -- so a
/// job the shell backgrounded with `&` never outlives the step that started it.
///
/// Best-effort at every step: the group may already be gone (the sandbox
/// wrapper carries `--die-with-parent` and takes its children with it).
pub(super) async fn terminate_process_group(
    pid: Option<u32>,
    child: &mut Child,
    term_grace: Duration,
) {
    signal_process_group(pid, libc::SIGTERM);

    // The members are snapshotted once and the snapshot is what the grace
    // period waits on. On the clean-exit path the leader is already reaped, so
    // `child.wait()` alone cannot measure whether the rest of the group is
    // gone, and re-reading `/proc` on every poll would rescan every process on
    // the host.
    let members = crate::agent::reap::process_group_members(pid.unwrap_or(0));
    if !members.is_empty() {
        warn!(
            count = members.len(),
            pids = ?members,
            "Step left processes running in its process group; terminating them"
        );
    }
    await_group_gone(&members, child, term_grace).await;

    // Reaping the leader does not imply its children stopped: a shell can
    // terminate on SIGTERM while a child ignores it. Kill remaining members
    // before returning.
    signal_process_group(pid, libc::SIGKILL);
    let _ = tokio::time::timeout(Duration::from_millis(KILL_GRACE_MS), child.wait()).await;
}

/// Wait until `child` is reaped *and* every pid in `members` is gone, or until
/// `grace` elapses.
///
/// Both conditions are required: a reaped leader says nothing about the rest of
/// its group, and a live member says nothing about whether the leader was ever
/// waited on.
async fn await_group_gone(members: &[u32], child: &mut Child, grace: Duration) -> bool {
    let deadline = tokio::time::Instant::now() + grace;
    while tokio::time::Instant::now() < deadline {
        let reaped = matches!(child.try_wait(), Ok(Some(_)));
        if reaped
            && members
                .iter()
                .all(|pid| !crate::agent::reap::pid_is_alive(*pid))
        {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(GROUP_POLL_MS)).await;
    }
    false
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
pub(super) fn signal_process_group(pid: Option<u32>, sig: libc::c_int) {
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
pub(super) fn signal_process_group(_pid: Option<u32>, _sig: i32) {}

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
pub(super) fn combine_streams(stdout: &[u8], stderr: &[u8], dropped: usize) -> String {
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
    /// instead of reaching the network, while a local command still runs
    /// normally. The kernel backend denies INET sockets with seccomp and TCP
    /// bind/connect with Landlock, so no network namespace is needed; the
    /// bubblewrap backend keeps the `unshare -n` wrapper instead.
    #[tokio::test]
    async fn an_offline_worker_has_no_egress_and_still_runs_local_commands() {
        let scratch = crate::test_support::TestScratch::new("exec-offline");
        let tmp = scratch.path().to_path_buf();
        let offline = runner().with_network_offline(true);

        let (out, code) = offline
            .execute_bash(&tmp, "printf 'still runs\n'")
            .await
            .expect("an offline command must still be spawned");
        assert_eq!(code, Some(0), "local commands must work offline: {out:?}");
        assert!(out.contains("still runs"), "{out:?}");

        // Creating an IPv4 socket must fail outright under the offline policy.
        let (out, code) = offline
            .execute_bash(
                &tmp,
                "python3 -c 'import socket; socket.socket(socket.AF_INET)'; echo exit=$?",
            )
            .await
            .expect("an offline socket probe must fail as ordinary output, not an error");
        assert_eq!(code, Some(0), "the probe step itself must succeed: {out:?}");
        assert!(
            out.contains("exit=1"),
            "creating an INET socket must fail offline: {out:?}"
        );

        let started = std::time::Instant::now();
        let (out, code) = offline
            .execute_bash(
                &tmp,
                "curl -sS --max-time 20 https://example.com -o /dev/null; echo exit=$?",
            )
            .await
            .expect("an offline curl must fail as ordinary output, not an error");
        let elapsed = started.elapsed();

        assert_eq!(code, Some(0), "the probe step itself must succeed: {out:?}");
        assert!(
            out.contains("exit=") && !out.contains("exit=0"),
            "the curl must not succeed: {out:?}"
        );
        assert!(
            elapsed < std::time::Duration::from_secs(15),
            "offline must fail fast, not wait out a connect timeout (took {elapsed:?})"
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn the_leased_build_dir_is_exported_to_cargo() {
        let target = Path::new("/tmp/swe-target-0123456789abcdef-0");
        let mut cmd = Command::new("true");
        apply_build_env(&mut cmd, Some(target), Path::new("/tmp/private"), "1");
        assert!(
            cmd.as_std()
                .get_envs()
                .any(|(key, value)| key == "CARGO_TARGET_DIR" && value == Some(target.as_os_str()))
        );
        // A worker without a lease builds in its own worktree, never in a
        // directory another live worker could be building in.
        let mut plain = Command::new("true");
        apply_build_env(&mut plain, None, Path::new("/tmp/private"), "1");
        assert!(
            !plain
                .as_std()
                .get_envs()
                .any(|(key, _)| key == "CARGO_TARGET_DIR")
        );
    }

    /// The per-ecosystem parallelism caps all carry the granted job count, so
    /// a Go, JVM or pytest-xdist build is dosed exactly like a Cargo one.
    #[test]
    fn build_env_carries_the_per_ecosystem_parallelism_caps() {
        let mut cmd = Command::new("true");
        apply_build_env(&mut cmd, None, Path::new("/tmp/private"), "3");
        let value = |name: &str| {
            cmd.as_std()
                .get_envs()
                .find(|(key, _)| *key == std::ffi::OsStr::new(name))
                .and_then(|(_, value)| value.map(|v| v.to_string_lossy().into_owned()))
                .unwrap_or_else(|| panic!("{name} must be set"))
        };
        assert_eq!(value("GOMAXPROCS"), "3");
        assert_eq!(value("MAVEN_OPTS"), "-T3");
        assert_eq!(value("GRADLE_OPTS"), "-Dorg.gradle.workers.max=3");
        assert_eq!(value("PYTEST_XDIST_AUTO_NUM_WORKERS"), "3");
        // Cargo's own caps are unchanged.
        assert_eq!(value("CARGO_BUILD_JOBS"), "3");
        assert_eq!(value("MAKEFLAGS"), "-j3");
    }

    /// Worker builds discard Cargo's debug info and incremental state by
    /// default: both are large writes into a target directory that is leased
    /// per worker and rebuilt often.
    #[test]
    fn build_env_discards_cargo_debug_info_and_incremental_state() {
        if std::env::var_os(WORKER_BUILD_DEBUG_VAR).is_some()
            || std::env::var_os("CARGO_PROFILE_DEV_DEBUG").is_some()
            || std::env::var_os("CARGO_PROFILE_TEST_DEBUG").is_some()
            || std::env::var_os("CARGO_INCREMENTAL").is_some()
        {
            return; // Operator overrides make the defaults unobservable.
        }
        let mut cmd = Command::new("true");
        apply_build_env(&mut cmd, None, Path::new("/tmp/private"), "1");
        let value = |name: &str| {
            cmd.as_std()
                .get_envs()
                .find(|(key, _)| *key == std::ffi::OsStr::new(name))
                .and_then(|(_, value)| value.map(|v| v.to_string_lossy().into_owned()))
                .unwrap_or_else(|| panic!("{name} must be set"))
        };
        assert_eq!(value("CARGO_PROFILE_DEV_DEBUG"), "0");
        assert_eq!(value("CARGO_PROFILE_TEST_DEBUG"), "0");
        assert_eq!(value("CARGO_INCREMENTAL"), "0");
    }

    /// An operator-exported value is forwarded verbatim; only the names the
    /// operator left alone take the diet's default.
    #[test]
    fn cargo_artifact_diet_keeps_operator_values() {
        let lookup = |name: &str| match name {
            "CARGO_PROFILE_DEV_DEBUG" => Some(std::ffi::OsString::from("2")),
            _ => None,
        };
        let diet: std::collections::HashMap<String, String> =
            cargo_artifact_diet(&lookup).into_iter().collect();
        assert_eq!(
            diet.get("CARGO_PROFILE_DEV_DEBUG").map(String::as_str),
            Some("2")
        );
        assert_eq!(
            diet.get("CARGO_PROFILE_TEST_DEBUG").map(String::as_str),
            Some("0")
        );
        assert_eq!(diet.get("CARGO_INCREMENTAL").map(String::as_str), Some("0"));
    }

    /// `WORKER_BUILD_DEBUG=1` restores Cargo's own defaults for the names the
    /// operator did not set, so debug info and backtraces stay complete.
    #[test]
    fn worker_build_debug_suppresses_the_diet_defaults() {
        let keep_debug = |name: &str| match name {
            WORKER_BUILD_DEBUG_VAR => Some(std::ffi::OsString::from("1")),
            _ => None,
        };
        assert!(cargo_artifact_diet(&keep_debug).is_empty());

        // Only `1` is the opt-out; any other value keeps the diet.
        let zero = |name: &str| match name {
            WORKER_BUILD_DEBUG_VAR => Some(std::ffi::OsString::from("0")),
            _ => None,
        };
        assert_eq!(cargo_artifact_diet(&zero).len(), 3);

        // An explicitly exported Cargo value still rides along.
        let explicit = |name: &str| match name {
            WORKER_BUILD_DEBUG_VAR => Some(std::ffi::OsString::from("1")),
            "CARGO_INCREMENTAL" => Some(std::ffi::OsString::from("0")),
            _ => None,
        };
        assert_eq!(
            cargo_artifact_diet(&explicit),
            vec![("CARGO_INCREMENTAL".to_string(), "0".to_string())]
        );
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
        assert_eq!(command_timeout_secs(true), DEFAULT_HEAVY_TIMEOUT_SECS);
        assert_eq!(command_timeout_secs(false), DEFAULT_LIGHT_TIMEOUT_SECS);
    }

    // ---------- I/O priority ----------

    /// Scheduling decisions are independent of ambient environment variables.
    #[test]
    fn io_plan_demotes_heavy_commands_and_pins_light_ones() {
        use HeavyIonice::{Idle, Inherit, Lowest};
        let cases = [
            (true, true, Lowest, IoPlan::Set(IoClass::Heavy)),
            (true, false, Lowest, IoPlan::Wrap(IoClass::Heavy)),
            (true, true, Idle, IoPlan::Set(IoClass::Idle)),
            (true, false, Idle, IoPlan::Wrap(IoClass::Idle)),
            (true, true, Inherit, IoPlan::Inherit),
            (true, false, Inherit, IoPlan::Inherit),
            (false, true, Lowest, IoPlan::Set(IoClass::BestEffort)),
            (false, true, Idle, IoPlan::Set(IoClass::BestEffort)),
            (false, true, Inherit, IoPlan::Set(IoClass::BestEffort)),
            (false, false, Lowest, IoPlan::Inherit),
        ];
        for (heavy, supported, setting, expected) in cases {
            assert_eq!(io_plan(heavy, supported, setting), expected);
        }
    }

    #[test]
    fn heavy_ionice_setting_accepts_opt_out_and_idle_opt_in() {
        for (raw, expected) in [
            (None, HeavyIonice::Lowest),
            (Some(""), HeavyIonice::Lowest),
            (Some("1"), HeavyIonice::Lowest),
            (Some("7"), HeavyIonice::Lowest),
            (Some("unknown"), HeavyIonice::Lowest),
            (Some("0"), HeavyIonice::Inherit),
            (Some(" 0 "), HeavyIonice::Inherit),
            (Some("idle"), HeavyIonice::Idle),
            (Some(" idle "), HeavyIonice::Idle),
            (Some("3"), HeavyIonice::Idle),
        ] {
            assert_eq!(heavy_ionice_setting(raw), expected, "{raw:?}");
        }
    }

    /// A priority value is the class in its high bits and the level below it,
    /// and best-effort normal is the level the kernel gives an unpinned task.
    #[test]
    fn ioprio_values_carry_the_class_and_the_level() {
        assert_eq!(
            ioprio_value(IoClass::Heavy),
            (IOPRIO_CLASS_BE << IOPRIO_CLASS_SHIFT) | 7
        );
        assert_eq!(
            ioprio_value(IoClass::Idle),
            IOPRIO_CLASS_IDLE << IOPRIO_CLASS_SHIFT
        );
        assert_eq!(
            ioprio_value(IoClass::BestEffort),
            (IOPRIO_CLASS_BE << IOPRIO_CLASS_SHIFT) | IOPRIO_BE_NORMAL
        );
    }

    /// The `ionice` fallback re-quotes the command exactly once, so a command
    /// carrying its own quotes and metacharacters survives the wrapper.
    #[tokio::test]
    async fn the_ionice_wrapper_runs_the_command_unchanged() {
        if !crate::agent::sandbox::binary_available("ionice") {
            return; // No `ionice` on this host: nothing to observe.
        }
        let wrapped = apply_io_wrapper(
            "echo 'a b' | tr ' ' '_'".to_string(),
            IoPlan::Wrap(IoClass::Heavy),
        );
        assert!(
            wrapped.starts_with("ionice -c2 -n7 bash -c "),
            "the wrapper must prefix the command: {wrapped}"
        );
        let mut cmd = Command::new("bash");
        cmd.args(["-c", &wrapped]);
        let out = cmd.output().await.expect("the wrapper must run");
        assert!(out.status.success(), "wrapper failed: {out:?}");
        assert_eq!(
            String::from_utf8_lossy(&out.stdout).trim(),
            "a_b",
            "the wrapped command must behave exactly as the bare one"
        );
        let idle = apply_io_wrapper("true".into(), IoPlan::Wrap(IoClass::Idle));
        assert!(idle.starts_with("ionice -c3 bash -c "));
        // Every other plan leaves the command string untouched.
        let plain = "echo hi".to_string();
        assert_eq!(apply_io_wrapper(plain.clone(), IoPlan::Inherit), plain);
        assert_eq!(
            apply_io_wrapper(plain.clone(), IoPlan::Set(IoClass::Idle)),
            plain
        );
    }

    /// A process's I/O priority value, or `None` when the host refuses to say.
    ///
    /// `ioprio_get` is the only way to observe the class: `/proc/<pid>/io`
    /// carries byte counters, not scheduling.
    #[cfg(unix)]
    fn ioprio_of(pid: u32) -> Option<i32> {
        // SAFETY: a raw syscall over two integers that reads no memory we own.
        let rc = unsafe {
            libc::syscall(
                libc::SYS_ioprio_get,
                IOPRIO_WHO_PROCESS as libc::c_long,
                pid as libc::c_long,
            )
        };
        (rc >= 0).then_some(rc as i32)
    }

    /// Both best-effort levels and the idle opt-in are observable in the
    /// child, not merely decided in the parent.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_child_inherits_the_io_class_of_its_command() {
        if !ioprio_syscall_supported() {
            return; // No `ioprio_set` here, so there is nothing to observe.
        }
        for (heavy, setting, expected) in [
            (true, HeavyIonice::Lowest, IoClass::Heavy),
            (false, HeavyIonice::Lowest, IoClass::BestEffort),
            (true, HeavyIonice::Idle, IoClass::Idle),
        ] {
            let mut cmd = Command::new("sleep");
            configure_process(&mut cmd);
            cmd.arg("30");
            let IoPlan::Set(class) = io_plan(heavy, true, setting) else {
                panic!("a syscall-capable host must plan to set the class");
            };
            apply_io_priority(&mut cmd, class);
            // `spawn` only returns once the child has exec'd, and the
            // `pre_exec` hook runs before that, so the class is already set.
            let mut child = cmd.spawn().expect("sleep must spawn");
            let pid = child.id().expect("the child must still be running");
            let value = ioprio_of(pid).expect("the child's ioprio must be readable");
            assert_eq!(
                value,
                ioprio_value(expected),
                "a {} command's child must run at the expected I/O priority",
                if heavy { "heavy" } else { "light" }
            );
            child.kill().await.expect("kill the probe child");
            child.wait().await.expect("reap the probe child");
        }
    }

    #[test]
    fn build_parallelism_uses_the_granted_job_count() {
        if std::env::var_os("BUILD_PARALLELISM").is_some() {
            return; // An env override makes the granted count unobservable.
        }
        // The admission controller's grant wins over the default, and zero or
        // any granted count still yields at least one job.
        assert_eq!(build_parallelism(Some(3)), "3");
        assert_eq!(build_parallelism(Some(1)), "1");
        // Without a grant the default is half the cores, never below one.
        assert_eq!(
            build_parallelism(None),
            crate::config::half_the_cores().to_string()
        );
    }

    #[test]
    fn guardrail_rejection_is_reported_not_errored() {
        let out = blocked_by_guardrail("outside the worktree");
        assert!(out.contains("COMMAND BLOCKED BY WORKTREE GUARDRAIL"));
        assert!(out.contains("outside the worktree"));
    }

    /// The per-command overlay must reach the child: the differential verify
    /// gate replays the same command in the dispatcher's ambient environment,
    /// so an overlay that is dropped would make variant B indistinguishable
    /// from variant A.
    #[test]
    fn extra_env_reaches_the_child() {
        crate::agent::env::with_env_lock(|| {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("runtime");
            rt.block_on(extra_env_probe());
        });
    }

    async fn extra_env_probe() {
        let scratch = crate::test_support::TestScratch::new("extra-env");
        let tmp = scratch.path().to_path_buf();
        let runner = runner().with_extra_env(vec![
            ("SWE_EXTRA_ENV_PROBE".to_string(), "present".to_string()),
            ("HOME".to_string(), "/tmp".to_string()),
        ]);
        let (out, code) = runner
            .execute_bash(&tmp, "printf '%s-%s' \"$SWE_EXTRA_ENV_PROBE\" \"$HOME\"")
            .await
            .unwrap();
        assert_eq!(code, Some(0), "command failed: {out:?}");
        assert_eq!(
            out.trim(),
            "present-/tmp",
            "the overlay must reach the child: {out:?}"
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// The retained head/tail must be decodable: `Captured::push` cuts at the
    /// raw byte budget, so a stream whose head or tail boundary falls inside a
    /// multi-byte code point used to hand `combine_streams` bytes that decoded
    /// to U+FFFD.
    #[test]
    fn captured_keeps_head_and_tail_on_char_boundaries() {
        // Every char is 3 bytes and the budget is a multiple of 3, but the
        // chunked push below lands the cut mid-char regardless of alignment
        // once the stream is padded by a leading ASCII byte.
        let mut captured = Captured::new();
        let mut stream = String::from("x");
        stream.push_str(&"\u{65e5}".repeat(20_000));
        captured.push(stream.as_bytes());

        let retained = captured.captured();
        let text = std::str::from_utf8(&retained)
            .unwrap_or_else(|err| panic!("retained bytes are not valid UTF-8: {err}"));
        assert!(
            !text.contains('\u{fffd}'),
            "a truncated code point must not become a replacement character"
        );
        assert!(
            text.starts_with('x') && text.ends_with('\u{65e5}'),
            "the kept head and tail must be whole code points"
        );
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
        let scratch = crate::test_support::TestScratch::new("bwrap-test");
        let tmp = scratch.path().to_path_buf();
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

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[tokio::test]
    async fn execute_bash_runs_unsandboxed_when_disabled() {
        let scratch = crate::test_support::TestScratch::new("exec-path");
        let tmp = scratch.path().to_path_buf();
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
        terminate_process_group(Some(pid), &mut child, TERM_GRACE).await;
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
        terminate_process_group(child.id(), &mut child, TERM_GRACE).await;
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
        terminate_process_group(pid, &mut child, TERM_GRACE).await;
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
        let scratch = crate::test_support::TestScratch::new("exec-flood");
        let tmp = scratch.path().to_path_buf();

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
        let scratch = crate::test_support::TestScratch::new("exec-cancel");
        let dir = scratch.path().to_path_buf();
        let pid_file = dir.join("grandchild.pid");
        let mut cmd = Command::new("bash");
        cmd.args([
            "-c",
            &format!("sleep 300 & echo $! > {}; wait", pid_file.display()),
        ]);
        configure_process(&mut cmd);
        cmd.current_dir(&dir);
        // The command outlives its budget here, so it comes back as a
        // `Backgrounded` rather than as output: dropping that future must still
        // take the group down, which is what the guard it carries is for.
        let task = tokio::spawn(async move { run_with_timeout(&mut cmd, 300, None).await });
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
        let scratch = crate::test_support::TestScratch::new("exec-drain");
        let tmp = scratch.path().to_path_buf();

        let (out, code) = runner()
            .with_command_timeout(1)
            .execute_bash(&tmp, "echo EARLY-STDOUT; echo EARLY-STDERR >&2; sleep 300")
            .await
            .expect("a timeout is an ordinary result, not an error");

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
        let scratch = crate::test_support::TestScratch::new("exec-leak");
        let tmp = scratch.path().to_path_buf();

        // `setsid` detaches the sleeper from the killed process group, so it
        // keeps the inherited stdout open past the SIGKILL.
        let started = std::time::Instant::now();
        let (out, code) = runner()
            .with_command_timeout(1)
            .execute_bash(
                &tmp,
                "echo BEFORE-LEAK; (setsid sleep 300 & echo $! > leaked.pid); sleep 300",
            )
            .await
            .expect("a leaked pipe must not turn the timeout into a hang");
        let elapsed = started.elapsed();
        // The detached sleeper is the point of the test, but it must not
        // outlive it: the harness audit rightly reports a suite that leaves
        // processes behind.
        if let Some(pid) = std::fs::read_to_string(tmp.join("leaked.pid"))
            .ok()
            .and_then(|pid| pid.trim().parse::<i32>().ok())
        {
            // SAFETY: `kill` takes plain integers; a stale pid only yields ESRCH.
            unsafe { libc::kill(pid, libc::SIGKILL) };
        }

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
        let scratch = crate::test_support::TestScratch::new("env-sanitize-test");
        let tmp = scratch.path().to_path_buf();

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
        let scratch = crate::test_support::TestScratch::new("env-toolchain-test");
        let tmp = scratch.path().to_path_buf();

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
        let scratch = crate::test_support::TestScratch::new("exec-chatty");
        let tmp = scratch.path().to_path_buf();

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
        let plan =
            super::super::sandbox::build_landlock_plan(&scratch.worktree, &scratch.target, false)
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
        let err = super::super::sandbox::build_landlock_plan(&missing, &scratch.target, false)
            .unwrap_err();
        assert!(
            format!("{err:#}").contains("does not exist"),
            "a missing worktree must be reported, got: {err:#}"
        );

        let err = super::super::sandbox::build_landlock_plan(&scratch.worktree, &missing, false)
            .unwrap_err();
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
        let built =
            super::super::sandbox::build_landlock_plan(&scratch.worktree, &scratch.target, false);
        unsafe { std::env::remove_var(super::super::sandbox::DISABLE_LANDLOCK_ENV) };

        assert!(
            matches!(built, Ok(None)),
            "an explicit opt-out must skip confinement entirely, got: {built:?}"
        );
    }

    /// Backend selection: the kernel confines by default, bubblewrap is an
    /// explicit opt-in, and a kernel that cannot confine falls back to
    /// bubblewrap when it is installed.
    #[test]
    fn backend_selection_prefers_the_kernel_and_falls_back() {
        use SandboxBackend::*;
        let yes = || true;
        let no = || false;
        // An explicit opt-out confines nothing, whatever the host offers.
        assert_eq!(choose_backend(true, true, yes, yes), Unconfined);
        // The kernel confines by default, even with bubblewrap installed.
        assert_eq!(choose_backend(false, false, yes, yes), Kernel);
        // The opt-in selects bubblewrap where it is installed...
        assert_eq!(choose_backend(false, true, yes, yes), Bwrap);
        // ...and is ignored where it is not.
        assert_eq!(choose_backend(false, true, yes, no), Kernel);
        // A kernel that cannot confine degrades to bubblewrap, then to none.
        assert_eq!(choose_backend(false, false, no, yes), Bwrap);
        assert_eq!(choose_backend(false, false, no, no), Unconfined);
    }

    /// A step's temp and nested-scratch variables point at one private dir the
    /// sandbox lets it write, whichever backend confines it.
    #[tokio::test]
    async fn a_step_gets_a_writable_private_scratch_dir() {
        let scratch = crate::test_support::TestScratch::new("exec-scratch");
        let tmp = scratch.path().to_path_buf();
        let (out, code) = runner()
            .execute_bash(
                &tmp,
                r#"[ "$TMPDIR" = "$SWE_TEMP_DIR" ] && mkdir -p "$SWE_TEMP_DIR/nested" && touch "$TMPDIR/probe" && echo scratch-ok"#,
            )
            .await
            .expect("the probe must spawn");
        let _ = std::fs::remove_dir_all(&tmp);
        assert_eq!(code, Some(0), "{out:?}");
        assert!(out.contains("scratch-ok"), "{out:?}");
    }

    /// A sandbox that cannot be prepared (here: the worktree no longer exists)
    /// fails closed: the command is refused, never run unconfined.
    #[tokio::test]
    async fn a_missing_worktree_refuses_the_command_instead_of_running_it_unconfined() {
        if select_backend() != SandboxBackend::Kernel {
            eprintln!("skipping: the kernel backend is not in use on this host");
            return;
        }
        let scratch = crate::test_support::TestScratch::missing("exec-gone");
        let gone = scratch.path().to_path_buf();
        let marker = std::env::temp_dir().join(format!("exec-gone-marker-{}", std::process::id()));
        let _ = std::fs::remove_file(&marker);
        let (out, code) = runner()
            .execute_bash(&gone, &format!("touch {}", marker.display()))
            .await
            .expect("a refused command is ordinary output");
        assert_eq!(code, Some(1), "{out:?}");
        assert!(
            out.contains("BLOCKED: the sandbox could not be prepared"),
            "{out:?}"
        );
        assert!(!marker.exists(), "the command must not have run");
    }

    /// Run a python snippet under the kernel backend and return its stdout.
    ///
    /// Python's `ctypes` issues raw syscalls, which is the only way to check
    /// that the seccomp filter answers each number as intended.
    #[cfg(target_arch = "x86_64")]
    async fn run_confined_python(offline: bool, tag: &str, script: &str) -> String {
        let scratch = LandlockScratch::new(tag);
        let mut cmd = Command::new("python3");
        apply_kernel_confinement(
            &mut cmd,
            &scratch.worktree,
            &scratch.target,
            offline,
            IoPlan::Inherit,
        )
        .expect("prepare the kernel confinement");
        cmd.arg("-c").arg(script);
        let out = cmd
            .output()
            .await
            .expect("python3 must spawn under the sandbox");
        String::from_utf8_lossy(&out.stdout).into_owned()
    }

    /// Privileged and cross-process syscalls fail with EPERM, and an x32-ABI
    /// syscall (which would match none of the native rules) fails with ENOSYS.
    #[cfg(target_arch = "x86_64")]
    #[tokio::test]
    async fn the_seccomp_filter_denies_escapes_and_foreign_abis() {
        if !super::super::sandbox::KernelConfinement::probe_available()
            || std::process::Command::new("python3")
                .arg("-V")
                .output()
                .is_err()
        {
            eprintln!("skipping: no kernel confinement or python3 on this host");
            return;
        }
        let script = r#"
import ctypes
libc = ctypes.CDLL(None, use_errno=True)
def call(nr, *args):
    r = libc.syscall(nr, *args)
    return ctypes.get_errno() if r == -1 else 0
print("ptrace", call(101, 0, 0, 0, 0))
print("mount", call(165, 0, 0, 0, 0, 0))
print("unshare_user", call(272, 0x10000000))
print("io_uring_setup", call(425, 1, 0))
print("x32_getpid", call(0x40000000 | 39))
print("getpid", call(39))
"#;
        let out = run_confined_python(false, "seccomp-deny", script).await;
        for (name, errno) in [
            ("ptrace", libc::EPERM),
            ("mount", libc::EPERM),
            ("unshare_user", libc::EPERM),
            ("io_uring_setup", libc::EPERM),
            ("x32_getpid", libc::ENOSYS),
            ("getpid", 0),
        ] {
            assert!(
                out.lines().any(|l| l == format!("{name} {errno}")),
                "{name} must answer {errno}; child printed:\n{out}"
            );
        }
    }

    /// Offline: no IPv4/IPv6 socket can be created (UDP included), while a
    /// unix-domain socket still works.
    #[cfg(target_arch = "x86_64")]
    #[tokio::test]
    async fn an_offline_worker_cannot_create_inet_sockets() {
        if !super::super::sandbox::KernelConfinement::probe_available()
            || std::process::Command::new("python3")
                .arg("-V")
                .output()
                .is_err()
        {
            eprintln!("skipping: no kernel confinement or python3 on this host");
            return;
        }
        let script = r#"
import socket
for name, fam, kind in [("tcp4", socket.AF_INET, socket.SOCK_STREAM),
                        ("udp4", socket.AF_INET, socket.SOCK_DGRAM),
                        ("tcp6", socket.AF_INET6, socket.SOCK_STREAM),
                        ("unix", socket.AF_UNIX, socket.SOCK_STREAM)]:
    try:
        socket.socket(fam, kind).close(); print(name, "ok")
    except OSError as e:
        print(name, "denied", e.errno)
"#;
        let offline = run_confined_python(true, "seccomp-offline", script).await;
        for name in ["tcp4", "udp4", "tcp6"] {
            assert!(
                offline
                    .lines()
                    .any(|l| l.starts_with(&format!("{name} denied"))),
                "{name} must be denied offline:\n{offline}"
            );
        }
        assert!(
            offline.lines().any(|l| l == "unix ok"),
            "unix sockets stay usable:\n{offline}"
        );

        // The online half only means something where this process may open
        // INET sockets itself (not, say, inside an offline worker step).
        if std::net::UdpSocket::bind("127.0.0.1:0").is_ok() {
            let online = run_confined_python(false, "seccomp-online", script).await;
            assert!(
                online.lines().any(|l| l == "tcp4 ok"),
                "online workers keep inet sockets:\n{online}"
            );
        }
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
        apply_kernel_confinement(
            &mut cmd,
            &scratch.worktree,
            &scratch.target,
            false,
            IoPlan::Inherit,
        )
        .expect("prepare the kernel confinement");
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
        if super::super::sandbox::build_landlock_plan(&scratch.worktree, &scratch.target, false)
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
        apply_kernel_confinement(
            &mut cmd,
            &scratch.worktree,
            &scratch.target,
            false,
            IoPlan::Inherit,
        )
        .expect("prepare the kernel confinement");
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
        if super::super::sandbox::build_landlock_plan(&scratch.worktree, &scratch.target, false)
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
        apply_kernel_confinement(
            &mut cmd,
            &scratch.worktree,
            &scratch.target,
            false,
            IoPlan::Inherit,
        )
        .expect("prepare the kernel confinement");
        cmd.arg("-c").arg("true");
        let _ = cmd.output().await.expect("spawn");

        // The child could not have read it...
        let mut child = Command::new("/bin/sh");
        apply_kernel_confinement(
            &mut child,
            &scratch.worktree,
            &scratch.target,
            false,
            IoPlan::Inherit,
        )
        .expect("prepare the kernel confinement");
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

    /// The bubblewrap backend stays intact as the opt-in: with bubblewrap
    /// present the bwrap argument vector is unchanged.
    #[test]
    fn bubblewrap_remains_available_as_the_opt_in_backend() {
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

    /// The tree fingerprint the completion gate reuses a verify run on.
    mod fingerprint {
        use crate::agent::exec::tree_fingerprint;
        use crate::test_support::TestScratch;
        use std::path::Path;

        fn git(dir: &Path, args: &[&str]) {
            let out = std::process::Command::new("git")
                .current_dir(dir)
                .args(args)
                .output()
                .unwrap_or_else(|e| panic!("git {args:?}: {e}"));
            assert!(
                out.status.success(),
                "git {args:?} failed: {}",
                String::from_utf8_lossy(&out.stderr)
            );
        }

        /// A fresh git repository in `scratch`, with one baseline commit.
        fn init_repo(scratch: &TestScratch) {
            let dir = scratch.path();
            git(dir, &["init", "-b", "master"]);
            git(dir, &["config", "user.name", "t"]);
            git(dir, &["config", "user.email", "t@localhost"]);
            std::fs::write(dir.join("seed.txt"), "seed\n").unwrap();
            git(dir, &["add", "seed.txt"]);
            git(dir, &["commit", "-m", "baseline"]);
        }

        #[test]
        fn an_unchanged_tree_fingerprints_identically() {
            let scratch = TestScratch::new("exec-fp-stable");
            init_repo(&scratch);
            let a = tree_fingerprint(scratch.path()).expect("fingerprint");
            let b = tree_fingerprint(scratch.path()).expect("fingerprint");
            assert_eq!(a, b, "an unchanged tree must fingerprint identically");
        }

        /// An untracked file is part of the tree's content even though git
        /// tracks no prior version of it to diff, so both creating one and
        /// editing it must move the fingerprint.
        #[test]
        fn an_untracked_file_moves_the_fingerprint() {
            let scratch = TestScratch::new("exec-fp-untracked");
            init_repo(&scratch);
            let dir = scratch.path();
            let before = tree_fingerprint(dir).expect("fingerprint before");

            std::fs::write(dir.join("new.rs"), "fn main() {}\n").unwrap();
            let after_new = tree_fingerprint(dir).expect("fingerprint after new file");
            assert_ne!(
                before, after_new,
                "a new untracked file must move the fingerprint"
            );

            std::fs::write(dir.join("new.rs"), "fn main() { println!(\"hi\"); }\n").unwrap();
            let after_edit = tree_fingerprint(dir).expect("fingerprint after edit");
            assert_ne!(
                after_new, after_edit,
                "editing an untracked file must move the fingerprint"
            );
        }

        /// A directory git cannot read as a repository yields no
        /// fingerprint, so the gate re-runs rather than reusing an unknown.
        #[test]
        fn a_directory_that_is_not_a_repository_has_no_fingerprint() {
            let scratch = TestScratch::new("exec-fp-nonrepo");
            assert_eq!(
                tree_fingerprint(scratch.path()),
                None,
                "a non-repository must yield no fingerprint, never a stable one"
            );
        }
    }
}
