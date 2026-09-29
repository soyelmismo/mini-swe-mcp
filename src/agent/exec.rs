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
use std::time::Duration;
use tokio::process::Command;

use super::AgentRunner;
use super::sandbox::{
    find_git_dirs, has_bwrap, is_heavy_command, truncate_output, validate_bash_command,
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
/// On timeout the child's whole process group is `SIGKILL`ed (a lingering
/// compiler or test runner would otherwise outlive the budget) and the timeout
/// is reported to the model as ordinary output with exit code 124.
async fn run_with_timeout(cmd: &mut Command, timeout_secs: u64) -> Result<(String, Option<i32>)> {
    let child = cmd.spawn().context("Failed to spawn bash process")?;
    let child_pid = child.id();

    let output =
        match tokio::time::timeout(Duration::from_secs(timeout_secs), child.wait_with_output())
            .await
        {
            Ok(Ok(out)) => out,
            Ok(Err(e)) => return Err(e).context("Failed waiting for bash process"),
            Err(_) => {
                kill_process_group(child_pid);
                return Ok((
                    format!("Command timed out after {timeout_secs}s and was terminated."),
                    Some(TIMEOUT_EXIT_CODE),
                ));
            }
        };

    Ok((combine_output(&output), output.status.code()))
}

/// `SIGKILL` the child's process group. Best-effort: the child may already be
/// gone, and the sandbox wrapper may have taken its group down with it.
fn kill_process_group(pid: Option<u32>) {
    #[cfg(unix)]
    if let Some(pid) = pid {
        let _ = std::process::Command::new("kill")
            .args(["-KILL", &format!("-{pid}")])
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status();
    }
    #[cfg(not(unix))]
    let _ = pid;
}

/// Merge stdout and stderr into a single string, separated by a newline when
/// both are non-empty, then bound it to the shared truncation budget.
///
/// The separator is unconditional, so a stdout that already ends in `\n`
/// yields a blank line between the streams. That is pre-existing behaviour the
/// model sees verbatim in the tool result, so it is preserved as-is rather
/// than quietly changing the transcripts this crate already produced.
fn combine_output(output: &std::process::Output) -> String {
    let mut combined = String::new();
    if !output.stdout.is_empty() {
        combined.push_str(&String::from_utf8_lossy(&output.stdout));
    }
    if !output.stderr.is_empty() {
        if !combined.is_empty() {
            combined.push('\n');
        }
        combined.push_str(&String::from_utf8_lossy(&output.stderr));
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
        let out = std::process::Output {
            status: exit_status(0),
            stdout: b"from stdout\n".to_vec(),
            stderr: b"from stderr".to_vec(),
        };
        assert_eq!(combine_output(&out), "from stdout\n\nfrom stderr");

        // Only one stream present: no separator is inserted.
        let stdout_only = std::process::Output {
            status: exit_status(0),
            stdout: b"only stdout".to_vec(),
            stderr: Vec::new(),
        };
        assert_eq!(combine_output(&stdout_only), "only stdout");

        let long = "x".repeat(TRUNCATE_LIMIT_FOR_TEST + 1_000);
        let truncated = combine_output(&std::process::Output {
            status: exit_status(0),
            stdout: long.into_bytes(),
            stderr: Vec::new(),
        });
        assert!(truncated.contains("... [Truncated "), "{truncated}");
        assert!(truncated.len() < TRUNCATE_LIMIT_FOR_TEST + 1_000);
    }

    /// Mirrors the sandbox budget constant without importing it twice.
    const TRUNCATE_LIMIT_FOR_TEST: usize = super::super::sandbox::TRUNCATE_LIMIT;

    fn exit_status(code: i32) -> std::process::ExitStatus {
        use std::os::unix::process::ExitStatusExt;
        std::process::ExitStatus::from_raw(code << 8)
    }

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
}
