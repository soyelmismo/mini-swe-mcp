//! A worker that is making progress at its budget gets one automatic
//! extension instead of stopping.
//!
//! The decision is a pure function over recent history (see
//! `pool::runner::turn::grant_extension`): progress (a changed diff or a
//! test/gate command in the window) and no recent guard fire means the
//! worker is close, so it gets one bounded extension; anything else stops
//! as today. These tests drive a scripted fake LLM (no real model) to
//! exercise the extension end to end.

mod common;

use std::path::{Path, PathBuf};
use std::time::Duration;

use mini_swe_mcp::pool::{WorkerPool, WorkerState};
use mini_swe_mcp::worktree::ScratchRoot;

use common::fake_llm::FakeLlm;

const OWNER: &str = "budget-ext-owner";
const GROUP: &str = "budget-ext-group";

/// A throwaway git repository the worker is dispatched against.
struct TestRepo {
    _scratch: common::TempDir,
    dir: PathBuf,
}

impl TestRepo {
    fn new(tag: &str) -> Self {
        let scratch = common::TempDir::new_in_tmp(&format!("budget-ext-repo-{tag}"));
        let dir = scratch
            .path()
            .canonicalize()
            .expect("canonicalize the scratch repo");
        common::git(&dir, &["init", "-b", "master"]);
        common::git(&dir, &["config", "user.name", "mini-swe-test"]);
        common::git(&dir, &["config", "user.email", "test@localhost"]);
        std::fs::write(dir.join("README.md"), "# scratch\n").expect("seed file");
        common::git(&dir, &["add", "README.md"]);
        common::git(&dir, &["commit", "-m", "baseline"]);
        Self {
            _scratch: scratch,
            dir,
        }
    }

    fn path(&self) -> &Path {
        &self.dir
    }
}

impl Drop for TestRepo {
    fn drop(&mut self) {
        mini_swe_mcp::cache::remove_build_dir_leases(&self.dir);
    }
}

/// Poll until the worker reaches any terminal state.
async fn wait_terminal(pool: &WorkerPool, id: &str) -> WorkerState {
    for _ in 0..1200 {
        if let Some(state) = pool.get_worker_state(id).await {
            match state {
                WorkerState::Running { .. } | WorkerState::Paused { .. } => {}
                other => return other,
            }
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("worker {id} never reached a terminal state");
}

#[tokio::test]
async fn a_worker_making_progress_completes_inside_the_extension() {
    // A heavy command on turn 1, benign commands on turns 2 and 3, then the
    // completion sentinel from turn 4 on. With a budget of 3 the worker runs
    // out of turns at step 3, but the heavy command in its recent history
    // earns one automatic extension, so it completes on turn 4.
    let server = FakeLlm::spawn_gate_then_complete("cargo --version").await;
    let repo = TestRepo::new("progress");
    let scratch = common::TempDir::new_in_tmp("budget-ext-pool");
    let pool = WorkerPool::with_scratch(
        1,
        server.base_url().to_string(),
        "test-key".to_string(),
        ScratchRoot::new(scratch.path()),
    );
    let worker_id = pool
        .dispatch(
            OWNER.to_string(),
            "make progress and finish inside the extension".to_string(),
            "test-model".to_string(),
            None,
            repo.path().to_path_buf(),
            3,
            Some(GROUP.to_string()),
            None,
            false,
            None,
            Vec::new(),
        )
        .await
        .expect("dispatch the worker");

    let state = wait_terminal(&pool, &worker_id).await;
    match &state {
        WorkerState::Completed { turns, metrics, .. } => {
            assert_eq!(*turns, 4, "the extension carried the worker to turn 4");
            assert_eq!(
                metrics.auto_extensions_granted, 1,
                "exactly one automatic extension was granted"
            );
        }
        other => panic!("expected Completed, got {other:?}"),
    }
}

#[tokio::test]
async fn a_stagnating_worker_is_exhausted_not_extended() {
    // A looping script that never completes and never runs a heavy command:
    // no progress signal, so the budget stops the worker as today.
    let server = FakeLlm::spawn_looping().await;
    let repo = TestRepo::new("stagnation");
    let scratch = common::TempDir::new_in_tmp("budget-ext-pool-stag");
    let pool = WorkerPool::with_scratch(
        1,
        server.base_url().to_string(),
        "test-key".to_string(),
        ScratchRoot::new(scratch.path()),
    );
    let worker_id = pool
        .dispatch(
            OWNER.to_string(),
            "loop without progress".to_string(),
            "test-model".to_string(),
            None,
            repo.path().to_path_buf(),
            3,
            Some(GROUP.to_string()),
            None,
            false,
            None,
            Vec::new(),
        )
        .await
        .expect("dispatch the worker");

    let state = wait_terminal(&pool, &worker_id).await;
    match &state {
        WorkerState::Exhausted { turns, metrics, .. } => {
            assert_eq!(*turns, 3, "the whole budget was spent");
            assert_eq!(
                metrics.auto_extensions_granted, 0,
                "no automatic extension was granted"
            );
        }
        other => panic!("expected Exhausted, got {other:?}"),
    }
}
