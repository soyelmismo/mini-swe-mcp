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

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::process::{Child, Command};

use super::AgentRunner;
use super::sandbox::{
    find_git_common_dir, find_git_dirs, has_bwrap, is_heavy_command, truncate_output,
    validate_bash_command,
};

/// Exit code reported to the model when a command exceeded its wall-clock
/// budget and its process group was killed.
const TIMEOUT_EXIT_CODE: i32 = 124;

/// Default wall-clock budget (seconds) for commands `is_heavy_command`
/// classifies as heavy (builds, test suites, package installs).
const DEFAULT_HEAVY_TIMEOUT_SECS: u64 = 600;

/// Default wall-clock budget (seconds) for every other command.
const DEFAULT_LIGHT_TIMEOUT_SECS: u64 = 120;

/// `nice` level applied to every child so agent work yields to interactive
/// work on a shared machine.
const NICE_LEVEL: &str = "-n";

/// Grace period a timed-out command gets to stop itself after `SIGTERM` before
/// its process group is escalated to `SIGKILL`.
///
/// Long enough for a tool to flush its buffers and tidy up after itself, short
/// enough that a wedged build still fails close to its wall-clock budget.
const TERM_GRACE_MS: u64 = 5_000;

/// Bound on the second, post-`SIGKILL` reap. `SIGKILL` cannot be caught or
/// ignored, so a child still unreaped here is not ours to wait on any longer
/// and must not stall the tool result.
const KILL_GRACE_MS: u64 = 2_000;

/// Bound on draining the output pipes of a command that has been terminated.
///
/// A grandchild that inherited stdout/stderr can hold the write end open long
/// after its parent was killed, so the drain is abandoned at this deadline
/// rather than being allowed to hang the worker.
const DRAIN_GRACE_MS: u64 = 2_000;

impl AgentRunner {
    /// Run `command` with `bash -c` inside `dir` and return its combined
    /// stdout/stderr (truncated to the shared byte budget) plus the exit code.
    ///
    /// Validation failures are *not* errors: a command rejected by the
    /// worktree guardrail is reported back to the model as output with a
    /// non-zero code so it can recover on the next step. Only a failure to
    /// spawn the child propagates as an `Err`.
    pub async fn execute_bash(&self, dir: &Path, command: &str) -> Result<(String, Option<i32>)> {
        if let Err(reason) = validate_bash_command(command) {
            return Ok((blocked_by_guardrail(reason), Some(1)));
        }

        let parallelism = build_parallelism();
        let target_dir = resolve_target_dir(dir);
        let _ = std::fs::create_dir_all(&target_dir);

        let mut cmd = Command::new("nice");
        configure_process(&mut cmd);

        if sandbox_enabled() {
            apply_sandbox_args(&mut cmd, dir, &target_dir);
            cmd.args(["--chdir", &dir.to_string_lossy()]);
            cmd.args(["/usr/bin/bash", "-c", command]);
        } else {
            cmd.current_dir(dir)
                .args([NICE_LEVEL, "10", "bash", "-c", command]);
        }

        // Environment hygiene first: the child is spawned with a cleared
        // environment and a strict allow-list, so no ambient credential from
        // the operator's shell can reach the model. Build/cache variables are
        // layered on top of the sanitized base afterwards.
        apply_sanitized_environment(&mut cmd, dir);
        apply_build_env(&mut cmd, &target_dir, &parallelism);
        crate::cache::apply_shared_cache_env(&mut cmd);

        let timeout_secs = command_timeout_secs(command);
        run_with_timeout(&mut cmd, timeout_secs).await
    }
}

/// Message shown to the model when the worktree guardrail rejects a command.
fn blocked_by_guardrail(reason: &str) -> String {
    format!(
        "COMMAND BLOCKED BY WORKTREE GUARDRAIL:\n{reason}\nPlease run your command within the current repository directory ($PWD)."
    )
}

/// Build/test parallelism shared by the child, honouring `BUILD_PARALLELISM`
/// and defaulting to half the available cores (never below one).
fn build_parallelism() -> String {
    let default_parallelism = std::thread::available_parallelism()
        .map(|n| (n.get() / 2).max(1))
        .unwrap_or(2);
    std::env::var("BUILD_PARALLELISM")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default_parallelism)
        .to_string()
}

/// Directory the child builds into: `CARGO_TARGET_DIR` when set, otherwise a
/// per-worktree directory under the SWE base dir so concurrent workers never
/// share a build cache.
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

/// Whether the bubblewrap sandbox is usable and not explicitly disabled.
fn sandbox_enabled() -> bool {
    has_bwrap() && std::env::var("SWE_DISABLE_SANDBOX").as_deref() != Ok("1")
}

/// Baseline child process setup: kill the whole process group on drop, detach
/// stdin, and capture both output streams.
fn configure_process(cmd: &mut Command) {
    cmd.kill_on_drop(true);
    #[cfg(unix)]
    cmd.process_group(0);
    cmd.stdin(std::process::Stdio::null());
    cmd.stdout(std::process::Stdio::piped());
    cmd.stderr(std::process::Stdio::piped());
}

/// Append the full `bwrap` argument vector.
///
/// The sandbox is read-only by default: only the worktree, its own gitdir, the
/// isolated build target dir and a handful of toolchain caches are bound
/// writable. `$HOME` is replaced by an empty tmpfs so the agent cannot read or
/// clobber SSH keys, dotfiles or package-manager credentials.
fn apply_sandbox_args(cmd: &mut Command, dir: &Path, target_dir: &Path) {
    let dir_str = dir.to_string_lossy();
    let target_str = target_dir.to_string_lossy();

    cmd.args([
        NICE_LEVEL,
        "10",
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

    // Isolate user home: mount empty tmpfs, expose only toolchain caches read-only
    let home = std::env::var_os("HOME").map(PathBuf::from);
    if let Some(home) = home.as_deref() {
        let home_str = home.to_string_lossy();
        cmd.args(["--tmpfs", &home_str]);
        for cache_dir in [".cargo", ".rustup", ".local/bin"] {
            let full = home.join(cache_dir);
            if full.exists() {
                let full_str = full.to_string_lossy();
                cmd.args(["--ro-bind-try", &full_str, &full_str]);
            }
        }
        let cache_tmp = home.join(".cache");
        let cache_tmp_str = cache_tmp.to_string_lossy();
        cmd.args(["--tmpfs", &cache_tmp_str]);
    }

    // Expose the shared toolchain homes read-only when they exist.
    for var in ["CARGO_HOME", "RUSTUP_HOME"] {
        if let Some(p) = std::env::var_os(var).map(PathBuf::from)
            && p.exists()
        {
            let p = p.to_string_lossy();
            cmd.args(["--ro-bind-try", &p, &p]);
        }
    }

    // Expose the worktree directory read-write
    cmd.args(["--bind", &dir_str, &dir_str]);

    // Expose the common .git directory READ-ONLY so git can resolve refs/objects
    // without permitting the sandbox to prune or delete repository branches!
    if let Some((common_git, worktree_gitdir)) = find_git_dirs(dir) {
        let common_str = common_git.to_string_lossy();
        cmd.args(["--ro-bind", &common_str, &common_str]);

        // Expose ONLY this worker's worktree gitdir read-write so it can update its local index
        if let Some(wt_gitdir) = worktree_gitdir
            && wt_gitdir.is_dir()
        {
            let wt_str = wt_gitdir.to_string_lossy();
            cmd.args(["--bind", &wt_str, &wt_str]);
        }
    }

    // Bind isolated build target directory read-write
    cmd.args(["--bind", &target_str, &target_str]);

    // Modular shared package/compiler caches
    crate::cache::append_bwrap_cache_args(cmd, home.as_deref());
}

/// Replace the inherited environment with the sanitized allow-list.
///
/// The parent process environment is cleared wholesale and rebuilt from
/// `env::build_clean_environment`, which forwards only the essential runtime
/// variables, remaps `HOME` to an isolated per-worktree scratch directory and
/// purges credential-bearing names (`OPENAI_API_KEY`, `GITHUB_TOKEN`, `AWS_*`,
/// `SSH_*`, ...). The child therefore cannot read back the operator's secrets
/// through the ambient environment, and two runs are reproducible regardless of
/// what else the operator happens to have exported.
///
/// The `repo_path` argument is the original checkout; the worker's worktree
/// `dir` is what the command is chdir'ed into, so the isolated `HOME` is placed
/// under `dir` where the sandbox bind makes it writable.
fn apply_sanitized_environment(cmd: &mut Command, dir: &Path) {
    // The worker's worktree `dir`; the original repository is its git common
    // dir when one can be resolved, else the worktree itself.
    let repo_path = find_git_common_dir(dir).unwrap_or_else(|| dir.to_path_buf());
    let repo_path = repo_path
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or(repo_path);
    super::env::apply_clean_environment_cmd(cmd, &repo_path, dir);
}

/// Universal build and test parallelism caps, so a command cannot oversubscribe
/// the machine no matter which build tool it drives.
fn apply_build_env(cmd: &mut Command, target_dir: &Path, parallelism: &str) {
    cmd.env("CARGO_TARGET_DIR", target_dir)
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

/// Wall-clock budget for `command` in seconds: heavy commands get a longer
/// default, and `COMMAND_TIMEOUT_SECS` overrides either tier.
fn command_timeout_secs(command: &str) -> u64 {
    let default_timeout = if is_heavy_command(command) {
        std::env::var("COMMAND_HEAVY_TIMEOUT_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_HEAVY_TIMEOUT_SECS)
    } else {
        std::env::var("COMMAND_LIGHT_TIMEOUT_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_LIGHT_TIMEOUT_SECS)
    };
    std::env::var("COMMAND_TIMEOUT_SECS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default_timeout)
}

/// Spawn `cmd`, wait up to `timeout_secs`, and collect the combined output.
///
/// On timeout the child's whole process group is asked to stop with a graceful
/// `SIGTERM` and escalated to `SIGKILL` only if it refuses to stop within
/// [`TERM_GRACE_MS`] (a lingering compiler or test runner would otherwise
/// outlive its budget). Whatever the command printed before the budget expired
/// is still reported, so the model sees the last diagnostics instead of a bare
/// "timed out". The timeout itself is reported as ordinary output with exit
/// code [`TIMEOUT_EXIT_CODE`].
async fn run_with_timeout(cmd: &mut Command, timeout_secs: u64) -> Result<(String, Option<i32>)> {
    let mut child = cmd.spawn().context("Failed to spawn bash process")?;
    let child_pid = child.id();

    // Take the pipes out of the child's hands and hand them to reader tasks.
    // The readers keep running independently of the wait, so cancelling the
    // wait on timeout cannot throw away output the command already produced.
    let stdout = child.stdout.take().context("stdout pipe was not captured")?;
    let stderr = child.stderr.take().context("stderr pipe was not captured")?;
    let out_buf = PipeBuffer::spawn(stdout);
    let err_buf = PipeBuffer::spawn(stderr);

    match tokio::time::timeout(Duration::from_secs(timeout_secs), child.wait()).await {
        // Clean exit: the write ends are closed now that the child is gone, so
        // the readers reach EOF and the real output and code are reported.
        Ok(Ok(status)) => {
            let (out, err) = tokio::join!(out_buf.finish(), err_buf.finish());
            Ok((combine_output(&out, &err), status.code()))
        }
        Ok(Err(e)) => Err(e).context("Failed waiting for bash process"),
        // The child outlived its budget: stop it, then report what it printed.
        Err(_elapsed) => {
            terminate_process_group(child_pid, &mut child).await;
            // The group is gone by now, so the readers are released and the
            // drain converges instead of waiting out the whole drain budget.
            let (out, err) = tokio::join!(out_buf.finish(), err_buf.finish());
            let mut output = combine_output(&out, &err);
            output.push_str(&format!(
                "\nCommand timed out after {timeout_secs}s and was terminated."
            ));
            Ok((output, Some(TIMEOUT_EXIT_CODE)))
        }
    }
}

/// One of a command's output pipes, drained in the background into a buffer
/// that outlives the command's own timeout.
///
/// Running the read as a detached task is what makes the drain robust. A read
/// inlined into the awaited future is cancelled wholesale when the wait times
/// out, discarding every byte it had already received; here the task keeps
/// filling `bytes` while the caller is free to stop waiting, and
/// [`finish`](Self::finish) later collects whatever arrived.
struct PipeBuffer {
    bytes: Arc<Mutex<Vec<u8>>>,
    reader: tokio::task::JoinHandle<()>,
}

impl PipeBuffer {
    /// Start draining `pipe` into a fresh buffer.
    fn spawn<R>(mut pipe: R) -> Self
    where
        R: AsyncRead + Unpin + Send + 'static,
    {
        let bytes = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&bytes);
        let reader = tokio::spawn(async move {
            let mut buf = Vec::new();
            // A read failure means the child is gone, but whatever arrived is
            // still worth reporting, so the buffer is flushed either way.
            let _ = pipe.read_to_end(&mut buf).await;
            match sink.lock() {
                Ok(mut sink) => sink.extend_from_slice(&buf),
                // The owning task panicked while holding the lock; a poisoned
                // mutex still holds the bytes collected so far.
                Err(poisoned) => poisoned.into_inner().extend_from_slice(&buf),
            }
        });
        Self { bytes, reader }
    }

    /// Stop waiting for the reader and return every byte it collected.
    ///
    /// Bounded by [`DRAIN_GRACE_MS`] because a grandchild that inherited the
    /// write end can hold the pipe open long after its parent was killed; the
    /// reader is then abandoned and the bytes read up to that point returned.
    async fn finish(self) -> Vec<u8> {
        if tokio::time::timeout(Duration::from_millis(DRAIN_GRACE_MS), self.reader)
            .await
            .is_err()
        {
            // Still blocked on a pipe somebody else holds open. Dropping the
            // handle detaches the reader, and the `Arc` keeps the buffer alive
            // until the runtime reaps the task when that pipe finally closes.
            tracing::debug!(
                timeout_ms = DRAIN_GRACE_MS,
                "abandoning output drain of a timed-out command"
            );
        }

        match self.bytes.lock() {
            Ok(guard) => guard.clone(),
            Err(poisoned) => poisoned.into_inner().clone(),
        }
    }
}

/// `SIGTERM` the child's process group, then `SIGKILL` it if it is still alive
/// after [`TERM_GRACE_MS`].
///
/// Best-effort at every step: the group may already be gone (the sandbox
/// wrapper carries `--die-with-parent` and takes its children with it), and a
/// group that stops promptly on the `SIGTERM` is never escalated.
async fn terminate_process_group(pid: Option<u32>, child: &mut Child) {
    signal_process_group(pid, "TERM");

    // Reap with a bounded wait so a well-behaved child can exit on its own and
    // close its pipes; the deadline stops a wedged child from extending the
    // budget by an unbounded amount.
    if tokio::time::timeout(Duration::from_millis(TERM_GRACE_MS), child.wait())
        .await
        .is_ok()
    {
        return;
    }

    signal_process_group(pid, "KILL");
    // Reap again after the kill, still bounded: `SIGKILL` is uncatchable, so a
    // child surviving this long is not ours to wait on any further.
    let _ = tokio::time::timeout(Duration::from_millis(KILL_GRACE_MS), child.wait()).await;
}

/// Send `signal` (`"TERM"` or `"KILL"`) to the process group led by `pid`.
///
/// `configure_process` places the child in its own process group (`bwrap` adds
/// `--new-session`), so the negated PID addresses the group and also reaches
/// grandchildren that `child.wait` alone would never reap.
fn signal_process_group(pid: Option<u32>, signal: &str) {
    #[cfg(unix)]
    if let Some(pid) = pid {
        let _ = std::process::Command::new("kill")
            .args([&format!("-{signal}"), &format!("-{pid}")])
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status();
    }
    #[cfg(not(unix))]
    let _ = (pid, signal);
}

/// Merge stdout and stderr into a single string, separated by a newline when
/// both are non-empty, then bound it to the shared truncation budget.
///
/// The separator is unconditional, so a stdout that already ends in `\n`
/// yields a blank line between the streams. That is pre-existing behaviour the
/// model sees verbatim in the tool result, so it is preserved as-is rather
/// than quietly changing the transcripts this crate already produced.
fn combine_output(stdout: &[u8], stderr: &[u8]) -> String {
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

    truncate_output(&combined)
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
    fn combine_output_joins_both_streams_and_truncates() {
        // Both streams non-empty: stdout, the separator, then stderr.
        assert_eq!(
            combine_output(b"from stdout\n", b"from stderr"),
            "from stdout\n\nfrom stderr"
        );

        // Only one stream present: no separator is inserted.
        assert_eq!(combine_output(b"only stdout", b""), "only stdout");

        let long = "x".repeat(TRUNCATE_LIMIT_FOR_TEST + 1_000);
        let truncated = combine_output(long.as_bytes(), b"");
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
        assert!(!group_is_alive(pid), "the SIGKILL escalation must reap the group");
    }

    /// A child that stops on `SIGTERM` is never escalated: it exits inside the
    /// grace period, so the call returns long before the `SIGKILL` deadline.
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

        assert_eq!(code, Some(TIMEOUT_EXIT_CODE), "a timeout reports 124: {out:?}");
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

    /// The end-to-end guarantee: a secret exported in the *worker's* own
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
}
