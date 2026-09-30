//! Shared harness for the integration test suite.
//!
//! Every integration test that needs to touch the outside world (spawn the
//! crate's binary, run `git`, or scratch on disk) goes through this module so
//! the boilerplate lives in exactly one place:
//!
//! * [`binary_path`] locates the freshly built `mini-swe-mcp` executable.
//!   Cargo already exports `CARGO_BIN_EXE_<name>` for integration tests, so no
//!   `target/<profile>/` guessing is needed (and no copy can drift out of sync).
//! * [`TempDir`] hands out collision-free scratch directories that clean
//!   themselves up on drop, even if the test panics.
//! * [`run_exe`] / [`run_exe_in_dir`] / [`run_exe_on_manifest`] run the binary
//!   and turn a spawn failure into a readable panic.
//! * [`git`] wraps the `git` CLI; [`git_ref_exists`] and
//!   [`worktree_is_registered`] are the two probes the worktree tests need.
//! * [`stdout_of`] / [`stderr_of`] decode captured output.
//!
//! The crate deliberately keeps its dependency graph small, so this harness is
//! plain `std` — no `tempfile`, no `assert_cmd`, no `predicates`.

#![allow(dead_code)] // not every integration test opts into every helper

pub mod fake_llm;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

// ----------
// Binary location
// ----------

/// Absolute path of the `mini-swe-mcp` executable under test.
///
/// Cargo sets `CARGO_BIN_EXE_<name>` for every integration test target, so
/// this is exact and needs no `target/<profile>/` fallback.
pub fn binary_path() -> PathBuf {
    PathBuf::from(
        std::env::var("CARGO_BIN_EXE_mini-swe-mcp")
            .expect("cargo did not set CARGO_BIN_EXE_mini-swe-mcp for this integration test"),
    )
}

// ----------
// Agent identity
// ----------

/// The identity a child of this test process resolves to: this process, named
/// with its pid and its start time (see [`mini_swe_mcp::hub::identity`]).
///
/// A CLI or MCP child a test spawns is an agent session whose host is the test
/// process itself, so this is the owner identity its workers carry.
pub fn host_of_this_process() -> String {
    let me =
        mini_swe_mcp::hub::identity::process(std::process::id()).expect("read /proc/self/stat");
    format!("host:{}:{}:{}", me.comm, me.pid, me.starttime)
}

// ----------
// Unique names
// ----------

/// Monotonic counter, so two calls in the same nanosecond still differ.
static COUNTER: AtomicU64 = AtomicU64::new(0);

/// A suffix that is unique per process, per call and per instant.
///
/// PID alone is not enough: several tests in one binary, or one test re-run
/// concurrently, would otherwise share a scratch directory and then delete
/// each other's state.
pub fn unique_suffix(tag: &str) -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock is before the unix epoch")
        .as_nanos();
    format!(
        "{tag}-{}-{nanos}-{}",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    )
}

// ----------
// Scratch directories
// ----------

/// A uniquely named scratch directory that removes itself on drop.
pub struct TempDir {
    path: PathBuf,
}

impl TempDir {
    /// Create (clearing any stale entry first) a scratch directory under `base`.
    pub fn new(base: &Path, tag: &str) -> Self {
        let path = base.join(format!("swe-test-{tag}-{}", unique_suffix("dir")));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path)
            .unwrap_or_else(|e| panic!("failed to create temp dir {}: {e}", path.display()));
        Self { path }
    }

    /// Create a scratch directory under the system temp dir.
    pub fn new_in_tmp(tag: &str) -> Self {
        Self::new(&std::env::temp_dir(), tag)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Create and return a fresh subdirectory (cleaned up with the parent).
    pub fn subdir(&self, name: &str) -> PathBuf {
        let path = self.path.join(name);
        std::fs::create_dir_all(&path)
            .unwrap_or_else(|e| panic!("failed to create {}: {e}", path.display()));
        path
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        // Best effort: a leftover directory must never fail an otherwise good test.
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

// ----------
// Running the crate's binary
// ----------
//
// Every helper pins `MINI_SWE_NO_DAEMON=1`: these tests exercise the CLI
// itself, and must never auto-start or reach the developer's real hub daemon.
// The hub transport is tested end to end in tests/hub_test.rs.

/// Run the binary in the current working directory.
pub fn run_exe(args: &[&str]) -> Output {
    let exe = binary_path();
    Command::new(&exe)
        .args(args)
        .env("MINI_SWE_NO_DAEMON", "1")
        .output()
        .unwrap_or_else(|e| panic!("failed to run {} {args:?}: {e}", exe.display()))
}

/// Run the binary with `dir` as its working directory and a hermetic
/// environment: `HOME` and `XDG_CONFIG_HOME` are pinned inside `dir` and
/// `OPENAI_API_KEY` is removed, so a child can never read or mutate the
/// developer's real environment, config or repository.
pub fn run_exe_in_dir(exe: &Path, dir: &Path, args: &[&str]) -> Output {
    Command::new(exe)
        .args(args)
        .current_dir(dir)
        .env("MINI_SWE_NO_DAEMON", "1")
        .env("HOME", dir)
        .env("XDG_CONFIG_HOME", dir.join(".config"))
        .env("ENV_FILE", dir.join(".env.does-not-exist"))
        .env_remove("OPENAI_API_KEY")
        .output()
        .unwrap_or_else(|e| {
            panic!(
                "failed to run {} {args:?} in {}: {e}",
                exe.display(),
                dir.display()
            )
        })
}

/// Run the binary against the crate's own `models.yaml`, with a dummy API key
/// that the manifest/list code paths never actually use.
pub fn run_exe_on_manifest(args: &[&str]) -> Output {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    Command::new(binary_path())
        .args(args)
        .current_dir(manifest_dir)
        .env("MINI_SWE_NO_DAEMON", "1")
        .env("OPENAI_API_KEY", "test-key-not-used-by-manifest-or-list")
        .env("ENV_FILE", manifest_dir.join(".env.does-not-exist"))
        .env("MODELS_FILE", manifest_dir.join("models.yaml"))
        .output()
        .unwrap_or_else(|e| panic!("failed to run mini-swe-mcp {args:?}: {e}"))
}

// ----------
// git
// ----------

/// Run a `git` subcommand in `dir`, panicking with git's own stderr on failure.
pub fn git(dir: &Path, args: &[&str]) -> String {
    let output = Command::new("git")
        .current_dir(dir)
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("failed to run git {args:?} in {}: {e}", dir.display()));
    assert!(
        output.status.success(),
        "git {args:?} in {} failed: {}",
        dir.display(),
        stderr_of(&output)
    );
    stdout_of(&output)
}

/// True when `branch` resolves in `repo`.
pub fn git_ref_exists(repo: &Path, branch: &str) -> bool {
    Command::new("git")
        .current_dir(repo)
        .args(["rev-parse", "--verify", "--quiet", branch])
        .output()
        .is_ok_and(|o| o.status.success())
}

/// True when `git worktree list` reports `path` as a registered worktree.
pub fn worktree_is_registered(repo: &Path, path: &Path) -> bool {
    let list = git(repo, &["worktree", "list", "--porcelain"]);
    list.lines()
        .any(|line| line.strip_prefix("worktree ") == path.to_str())
}

// ----------
// Output decoding
// ----------

pub fn stdout_of(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

pub fn stderr_of(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}
