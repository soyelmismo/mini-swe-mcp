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
//! The verify command here counts its own executions in a worktree file, so
//! the committed counter proves how many times it ran: two with reuse (the
//! worker's own turn plus variant B), three without (a fresh variant A too).

mod common;

use std::path::{Path, PathBuf};

use mini_swe_mcp::pool::{WorkerPool, WorkerState};

const TEST_OWNER: &str = "verify-reuse";

/// A heavy command that appends to a worktree counter and then exits zero.
///
/// `rustc --version` makes the whole command heavy (so the gate records it),
/// and the counter is the observable side effect: it lives in the worktree, so
/// it is committed to the worker's branch and readable after the run, and it
/// does not touch the shared repository the side-effect audit watches.
fn counting_verify() -> String {
    "n=$(cat vcount 2>/dev/null || echo 0); echo $((n+1)) > vcount; rustc --version".to_string()
}

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

fn git_capture(dir: &Path, args: &[&str]) -> String {
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
    String::from_utf8_lossy(&out.stdout).trim().to_string()
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
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

async fn wait_for_terminal(pool: &WorkerPool, worker_id: &str) -> WorkerState {
    for _ in 0..600 {
        if let Some(state) = pool.get_worker_state(worker_id).await {
            match state {
                WorkerState::Completed { .. } | WorkerState::Failed { .. } => return state,
                WorkerState::Running { .. } | WorkerState::Paused { .. } => {}
            }
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    panic!("worker {worker_id} did not reach a terminal state");
}

/// A worker that runs the verify command itself and then completes must have
/// the gate reuse that run: the committed counter reads two (its own turn plus
/// variant B), not three (which a fresh variant A would make it).
#[tokio::test]
async fn a_passed_gate_on_an_unchanged_tree_is_reused() {
    let repo = TestRepo::new("reuse");
    let verify = counting_verify();
    // Turn 1 runs the verify command; turn 2 is a no-op that leaves the tree
    // unchanged; turn 3 completes.
    let llm = common::fake_llm::FakeLlm::spawn(&verify, "echo done").await;

    let scratch = common::TempDir::new_in_tmp("vreuse-pool");
    let root = mini_swe_mcp::worktree::ScratchRoot::new(scratch.path());
    let pool =
        WorkerPool::with_scratch(1, llm.base_url().to_string(), "test-key".to_string(), root);

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
            Some(verify),
            Vec::new(),
        )
        .await
        .expect("dispatch the worker");

    let state = wait_for_terminal(&pool, &worker_id).await;
    match state {
        WorkerState::Completed { verified, .. } => assert_eq!(
            verified,
            Some(true),
            "the reused gate must still complete the worker verified"
        ),
        other => panic!("worker must complete, got {other:?}"),
    }

    let branch = format!("worker-{worker_id}");
    let count = git_capture(repo.path(), &["show", &format!("{branch}:vcount")]);
    assert_eq!(
        count, "2",
        "the gate must reuse the worker's own passing run: variant A is skipped, \
         so the command ran once for the worker and once for variant B (got {count})"
    );

    // The scratch root outlives the assertions so the pool's files stay put.
    drop(scratch);
}
