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
use std::sync::Mutex;
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

/// A short, unique scratch directory name.
///
/// The leaf is deliberately short: a scratch directory can hold a Unix socket,
/// whose path is bounded by `sockaddr_un::sun_path` (about 108 bytes) and is
/// prefixed by the ambient `TMPDIR`. The divergent-verify gate re-runs the
/// suite with `TMPDIR` inside the worktree, so a verbose name stops binding;
/// the tag is truncated and uniquified by pid and the same counter instead.
pub fn scratch_name(tag: &str) -> String {
    let tag: String = tag.chars().take(8).collect();
    format!(
        "swe-{tag}-{}-{}",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    )
}

// ----------
// Isolated scratch roots
// ----------

/// A pool over its own temporary scratch root, plus the root itself.
///
/// In-process tests must never touch the real registry under `swe_base_dir()`
/// (`SWE_TEMP_DIR` or `/var/tmp`): tests in one binary run in parallel threads
/// and share ids such as `h3-mine`/`agent-a`, so one test sees or removes
/// another's rows. Every pool a test builds goes through this helper, and the
/// returned [`TempDir`] must be kept alive for the pool's whole lifetime.
pub struct IsolatedPool {
    /// The pool under test, filing every row, mailbox and history file under
    /// the scratch root.
    pub pool: mini_swe_mcp::pool::WorkerPool,
    /// Owns the scratch root; dropping it removes the directory.
    pub scratch: TempDir,
}

impl IsolatedPool {
    /// A pool with `max_concurrent` slots over a fresh temporary root.
    pub fn new(max_concurrent: usize, tag: &str) -> Self {
        let scratch = TempDir::new_in_tmp(tag);
        let root = mini_swe_mcp::worktree::ScratchRoot::new(scratch.path());
        let pool = mini_swe_mcp::pool::WorkerPool::with_scratch(
            max_concurrent,
            "http://localhost:1".to_string(),
            "test-key".to_string(),
            root,
        );
        Self { pool, scratch }
    }

    /// A pool over a fresh temporary root, without the wrapper.
    ///
    /// The returned [`TempDir`] must stay alive for the pool's lifetime.
    pub fn pool(max_concurrent: usize, tag: &str) -> (mini_swe_mcp::pool::WorkerPool, TempDir) {
        let owned = Self::new(max_concurrent, tag);
        (owned.pool, owned.scratch)
    }

    /// The pool's scratch root, for the `*_in` registry/history/steer helpers.
    pub fn root(&self) -> mini_swe_mcp::worktree::ScratchRoot {
        mini_swe_mcp::worktree::ScratchRoot::new(self.scratch.path())
    }
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
        let path = base.join(scratch_name(tag));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path)
            .unwrap_or_else(|e| panic!("failed to create temp dir {}: {e}", path.display()));
        Self { path }
    }

    /// Create a scratch directory under the daemon's scratch root.
    ///
    /// Deliberately not the raw `TMPDIR`: the hub tests bind a Unix socket
    /// inside these directories, and a long `TMPDIR` would push the socket path
    /// past `SUN_LEN`. [`mini_swe_mcp::worktree::swe_base_dir`] makes the same
    /// short-base choice the daemon itself does.
    pub fn new_in_tmp(tag: &str) -> Self {
        Self::new(&mini_swe_mcp::worktree::swe_base_dir(), tag)
    }

    /// Own a scratch directory the caller has already created, removing it on
    /// drop. The caller keeps responsibility for creating it.
    pub fn own(path: PathBuf) -> Self {
        Self { path }
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
        reclaim_scratch_root(&self.path);
        // A pool a test killed leaves its worker's checkout behind, and the
        // runner rebuilds that checkout in a `spawn_blocking` task the abort
        // cannot cancel -- so it reappears *after* the drop above, once the
        // pool's own clone is gone. Registering the root for the exit sweep
        // reclaims it then, which is the only moment nothing is running again.
        remember_for_exit_sweep(&self.path);
    }
}

/// Reclaim a scratch root and everything the code under test derived from it.
///
/// Shared by [`TempDir::drop`] and the process-exit sweep, so a root is
/// reclaimed the same way whenever the last chance to do it arrives.
fn reclaim_scratch_root(path: &Path) {
    // A worktree's private scratch and its leased build directories are filed
    // next to the scratch base, keyed by the worktree leaf and by the repository
    // hash, so removing the tree alone leaves them behind.
    mini_swe_mcp::worktree::remove_target_dirs(path);
    mini_swe_mcp::cache::remove_build_dir_leases(path);
    // A pool is handed this directory as its scratch root, so the checkouts and
    // sidecars *it* created live inside it and go with it -- except for the
    // worktrees a killed worker leaves behind, which its aborted task no longer
    // owns. Those are filed under the pool's own root by leaf name.
    mini_swe_mcp::worktree::remove_scratch_root_worktrees(path);
    // The companions of the checkouts just dropped, then the root itself.
    mini_swe_mcp::worktree::remove_target_dirs(path);
    // Best effort: a leftover directory must never fail an otherwise good test.
    let _ = std::fs::remove_dir_all(path);
}

/// Own the short fallback directory a too-deep hub directory's socket moves to,
/// when [`mini_swe_mcp::hub::HubPaths::endpoint`] chose one.
///
/// A daemon removes this directory on a graceful exit and when its `run` future
/// is dropped, but a daemon a test SIGKILLs (or leaves running) does not, so the
/// test that caused it owns it. `None` when the socket fits in the hub
/// directory or the endpoint fell back to a Linux abstract socket.
pub fn fallback_socket_dir(hub_dir: &Path) -> Option<TempDir> {
    match mini_swe_mcp::hub::HubPaths::new(hub_dir.to_path_buf()).endpoint() {
        mini_swe_mcp::hub::HubEndpoint::Path(path) => path
            .parent()
            .filter(|parent| *parent != hub_dir)
            .map(|parent| TempDir::own(parent.to_path_buf())),
        mini_swe_mcp::hub::HubEndpoint::Abstract(_) => None,
    }
}

impl std::ops::Deref for TempDir {
    type Target = Path;

    fn deref(&self) -> &Path {
        &self.path
    }
}

impl AsRef<Path> for TempDir {
    fn as_ref(&self) -> &Path {
        &self.path
    }
}

impl AsRef<std::ffi::OsStr> for TempDir {
    fn as_ref(&self) -> &std::ffi::OsStr {
        self.path.as_os_str()
    }
}

// ----------
// Process-lifetime scratch
// ----------

/// Scratch directories removed when the test binary exits.
static PROCESS_SCRATCH: Mutex<Vec<PathBuf>> = Mutex::new(Vec::new());

/// Scratch roots re-swept when the test binary exits, because the code under
/// test can still write into one after its owner dropped it.
static EXIT_SWEPT_SCRATCH: Mutex<Vec<PathBuf>> = Mutex::new(Vec::new());

/// Registers the exit sweep with libc the first time a root asks for one.
static EXIT_SWEEP_ARMED: std::sync::Once = std::sync::Once::new();

unsafe extern "C" {
    /// Registered with libc so a directory that must outlive every test in the
    /// binary still leaves nothing behind when the binary exits.
    fn atexit(handler: extern "C" fn()) -> i32;
}

extern "C" fn remove_process_scratch() {
    let paths = PROCESS_SCRATCH
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    for path in paths.iter() {
        let _ = std::fs::remove_dir_all(path);
    }
    let roots = EXIT_SWEPT_SCRATCH
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    for path in roots.iter() {
        reclaim_scratch_root(path);
    }
}

/// Reclaim `path` once more when the test binary exits.
///
/// A pool that outlives its [`TempDir`] -- because the test kept a clone, or
/// because a killed worker's runner task is still winding down -- writes its
/// checkouts into a root that has already been dropped, and a `spawn_blocking`
/// task cannot be cancelled by aborting the worker that awaited it. The exit
/// sweep is the only point at which no test is running, so it is the only place
/// those late writes can be undone.
fn remember_for_exit_sweep(path: &Path) {
    {
        let mut roots = EXIT_SWEPT_SCRATCH
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if !roots.iter().any(|root| root == path) {
            roots.push(path.to_path_buf());
        }
    }
    EXIT_SWEEP_ARMED.call_once(|| {
        // SAFETY: the handler only locks statics and removes directories, so it
        // is safe to run from `atexit`.
        unsafe { atexit(remove_process_scratch) };
    });
}

/// A scratch directory that lives for the whole test binary.
///
/// Some tests point process-global state (`SWE_TEMP_DIR`, the registry) at a
/// directory every sibling test in the binary needs, so one test's [`TempDir`]
/// must not own it. Registering the directory with `atexit` gives it the
/// binary's lifetime while still leaving nothing behind.
pub fn process_temp_dir(tag: &str) -> PathBuf {
    let path = mini_swe_mcp::worktree::swe_base_dir().join(scratch_name(tag));
    let _ = std::fs::remove_dir_all(&path);
    std::fs::create_dir_all(&path)
        .unwrap_or_else(|e| panic!("create process scratch {}: {e}", path.display()));
    PROCESS_SCRATCH
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .push(path.clone());
    // SAFETY: the handler only locks a static and removes directories, so it is
    // safe to run from `atexit`.
    unsafe { atexit(remove_process_scratch) };
    path
}

// ----------
// Running the crate's binary
// ----------
//
// Every helper pins `MINI_SWE_NO_DAEMON=1`: these tests exercise the CLI
// itself, and must never auto-start or reach the developer's real hub daemon.
// The hub transport is tested end to end in tests/hub_test.rs.

/// Remove every environment variable that names an agent identity from `cmd`.
///
/// A child a test spawns would otherwise inherit the session the suite was
/// launched from — a `CLAUDE_CODE_SESSION_ID` in an agent's shell qualifies the
/// owner the child reports — so every spawn of the crate's binary scrubs the
/// session variables, the operator override and the watch token first. A test
/// that deliberately exercises one of them sets it *after* scrubbing.
pub fn scrub_identity_env(cmd: &mut Command) {
    for var in mini_swe_mcp::hub::identity::SESSION_ENV_VARS {
        cmd.env_remove(var);
    }
    cmd.env_remove("MINI_SWE_AGENT_ID");
    cmd.env_remove(mini_swe_mcp::hub::identity::WATCH_TOKEN_ENV);
}

/// A `Command` for `exe` with every inherited agent identity scrubbed; see
/// [`scrub_identity_env`].
pub fn binary_command(exe: &Path) -> Command {
    let mut cmd = Command::new(exe);
    scrub_identity_env(&mut cmd);
    cmd
}

/// Run the binary in the current working directory.
pub fn run_exe(args: &[&str]) -> Output {
    let exe = binary_path();
    binary_command(&exe)
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
    binary_command(exe)
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
    binary_command(&binary_path())
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
