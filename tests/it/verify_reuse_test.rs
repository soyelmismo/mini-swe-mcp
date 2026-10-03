//! The completion gate must not re-run a verification the worker already
//! passed on the same tree.
//!
//! Before requesting completion a worker runs the project's gates itself. The
//! completion gate then re-runs the very same command on the very same tree,
//! recompiling and re-testing everything. When the worker's last run of the
//! verify command passed on an unchanged tree, the gate reuses it: variant A
//! (the canonical run) is skipped, while variant B and the side-effect audit
//! still run, because B is what A cannot prove.
//!
//! The reuse is disclosed to the model in the completion turn, so the durable
//! history carries a "verify reused from step N" note exactly when the gate
//! skipped a fresh variant A -- and carries nothing of the sort when it ran.

use crate::common;
use mini_swe_mcp::pool::{WorkerPool, WorkerState};
use std::path::{Path, PathBuf};

const TEST_OWNER: &str = "verify-reuse";

/// A heavy command that reads the tree but never writes it, so its pass
/// certifies a stable tree and is eligible for reuse. `rustc --version` is the
/// heavy segment; it exits zero and touches no file.
const VERIFY: &str = "rustc --version";

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

/// A throwaway git repository the worker is dispatched against.
struct TestRepo {
    dir: PathBuf,
}

impl TestRepo {
    fn new(tag: &str) -> Self {
        let dir = std::env::temp_dir().join(common::unique_suffix(&format!("vreuse-{tag}")));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create scratch repo");
        let dir = dir.canonicalize().expect("canonicalize scratch repo");
        git(&dir, &["init", "-b", "master"]);
        git(&dir, &["config", "user.name", "mini-swe-test"]);
        git(&dir, &["config", "user.email", "test@localhost"]);
        std::fs::write(dir.join("README.md"), "# scratch\n").expect("seed file");
        git(&dir, &["add", "README.md"]);
        git(&dir, &["commit", "-m", "baseline"]);
        Self { dir }
    }

    fn path(&self) -> &Path {
        &self.dir
    }
}

impl Drop for TestRepo {
    fn drop(&mut self) {
        // The worker's heavy verify leases a build directory keyed by this
        // repo's hash; it is filed next to the scratch base, not inside the
        // repo, so removing the repo has to take the lease with it.
        mini_swe_mcp::cache::remove_build_dir_leases(&self.dir);
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

async fn wait_for_terminal(pool: &WorkerPool, worker_id: &str) -> WorkerState {
    for _ in 0..600 {
        if let Some(state) = pool.get_worker_state(worker_id).await {
            match state {
                WorkerState::Completed { .. }
                | WorkerState::Failed { .. }
                | WorkerState::Exhausted { .. } => return state,
                WorkerState::Running { .. } | WorkerState::Paused { .. } => {}
            }
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    panic!("worker {worker_id} did not reach a terminal state");
}

/// The durable history log for a worker under `root`, as one string.
fn history_of(root: &mini_swe_mcp::worktree::ScratchRoot, worker_id: &str) -> String {
    let path = mini_swe_mcp::pool::revision::history_log_path_in(root, worker_id);
    std::fs::read_to_string(&path).unwrap_or_default()
}

/// Run a scripted worker over an isolated repository, registry and history.
///
/// `first` and `second` are the worker's own turns (the gate command, then a
/// no-op or a tree change); `verify` is the gate the completion reuses; `env`
/// is the dispatcher's ambient snapshot for variant B.
async fn exercise(
    first: &str,
    second: &str,
    verify: &str,
    env: Vec<(String, String)>,
) -> (WorkerState, String) {
    let repo = TestRepo::new("gate");
    let llm = common::fake_llm::FakeLlm::spawn(first, second).await;
    let scratch = common::TempDir::new_in_tmp("vreuse-pool");
    let root = mini_swe_mcp::worktree::ScratchRoot::new(scratch.path());
    let pool = WorkerPool::with_scratch(
        1,
        llm.base_url().to_string(),
        "test-key".to_string(),
        root.clone(),
    );
    let worker_id = pool
        .dispatch(
            TEST_OWNER.to_string(),
            "exercise the reuse gate".to_string(),
            "test-model".to_string(),
            None,
            repo.path().to_path_buf(),
            6,
            Some("verify-reuse".to_string()),
            None,
            false,
            Some(verify.to_string()),
            env,
        )
        .await
        .expect("dispatch worker");
    let state = wait_for_terminal(&pool, &worker_id).await;
    let history = history_of(&root, &worker_id);
    (state, history)
}

fn assert_verified(state: &WorkerState) {
    assert!(
        matches!(
            state,
            WorkerState::Completed {
                verified: Some(true),
                ..
            }
        ),
        "expected verified completion: {state:?}"
    );
}

/// A worker that runs the verify command itself and then completes must have
/// the gate reuse that run: the history carries the "verify reused" note,
/// proving the canonical run was skipped.
#[tokio::test]
async fn a_passed_gate_on_an_unchanged_tree_is_reused() {
    let (state, history) = exercise(VERIFY, "echo done", VERIFY, Vec::new()).await;
    assert_verified(&state);
    assert!(
        history.contains("verify reused from step 1"),
        "the completion turn must disclose the reuse, got history:\n{history}"
    );
}

/// A file changed between the worker's run and completion moves the tree
/// fingerprint, so the gate runs variant A afresh instead of reusing.
#[tokio::test]
async fn a_changed_file_runs_canonical_verification_again() {
    let (state, history) = exercise(VERIFY, "echo changed >> README.md", VERIFY, Vec::new()).await;
    assert_verified(&state);
    assert!(
        !history.contains("verify reused"),
        "a changed tree must not be reused, got history:\n{history}"
    );
}

/// A completion whose verify command differs from the one the worker ran has
/// no recorded success to reuse, so variant A runs.
#[tokio::test]
async fn a_different_command_runs_canonical_verification_again() {
    let (state, history) =
        exercise(VERIFY, "echo done", "rustc --version && true", Vec::new()).await;
    assert_verified(&state);
    assert!(
        !history.contains("verify reused"),
        "a different command must not be reused, got history:\n{history}"
    );
}

/// A command whose earlier run failed is never recorded, so the gate runs it
/// afresh rather than reusing a pass that never happened.
#[tokio::test]
async fn a_failed_prior_run_is_not_reused() {
    let verify = "rustc --version && test -f ready";
    let (state, history) = exercise(verify, "touch ready", verify, Vec::new()).await;
    assert_verified(&state);
    assert!(
        history.contains("exit code: 1"),
        "the worker's own run must have failed first, got history:\n{history}"
    );
    assert!(
        !history.contains("verify reused"),
        "a failed run must not be reused, got history:\n{history}"
    );
}

/// Reusing variant A must not skip variant B: a command that only passes in the
/// canonical environment is still caught, so the completion is refused even
/// though A was reused.
#[tokio::test]
async fn reuse_does_not_skip_divergent_verification() {
    let verify = "rustc --version && test -z \"$SWE_REUSE_PROBE\"";
    let (state, history) = exercise(
        verify,
        "echo done",
        verify,
        vec![("SWE_REUSE_PROBE".to_string(), "set".to_string())],
    )
    .await;
    assert!(
        history.contains("verify reused from step 1"),
        "the canonical run must be reused, got history:\n{history}"
    );
    assert!(
        history.contains("orchestrator's environment"),
        "variant B must still run and refuse, got history:\n{history}"
    );
    assert!(
        !matches!(
            state,
            WorkerState::Completed {
                verified: Some(true),
                ..
            }
        ),
        "variant B's failure must not verify the completion: {state:?}"
    );
}
