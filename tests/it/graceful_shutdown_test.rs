//! A graceful hub shutdown must leave live workers interrupted, not failed.
//!
//! `WorkerPool::kill_all` is the SIGINT/SIGTERM path. A planned restart has to
//! be at least as good as a crash: each live worker's uncommitted work is
//! committed onto its branch and its registry row is left `Interrupted`, so the
//! next daemon's recovery auto-continues it (see `interrupted_workers` and
//! `auto_continue_budget`). An explicit user `kill` still ends `Failed`.

use crate::common;
use mini_swe_mcp::pool::{
    LogBuffer, MAX_AUTO_CONTINUES, RegistryStatus, WorkerMeta, WorkerMetrics, WorkerPool,
    WorkerRecord, WorkerState, load_registry_entry_in,
};
use mini_swe_mcp::worktree::ScratchRoot;
use std::path::{Path, PathBuf};

const OWNER: &str = "graceful-shutdown-test";

/// A repo with one commit and a worktree on `worker-<id>` carrying an
/// uncommitted file, so the shutdown has something to salvage.
fn repo_with_dirty_worktree(scratch: &common::TempDir, id: &str) -> (PathBuf, PathBuf) {
    let repo = scratch.subdir("repo");
    common::git(&repo, &["init", "-b", "master"]);
    common::git(&repo, &["config", "user.name", "t"]);
    common::git(&repo, &["config", "user.email", "t@t"]);
    std::fs::write(repo.join("base.txt"), "base\n").expect("seed the repo");
    common::git(&repo, &["add", "base.txt"]);
    common::git(&repo, &["commit", "-m", "seed"]);

    let checkout = scratch.path().join(format!("swe-wt-{id}"));
    let branch = format!("worker-{id}");
    common::git(
        &repo,
        &[
            "worktree",
            "add",
            "-b",
            &branch,
            &checkout.to_string_lossy(),
            "HEAD",
        ],
    );
    std::fs::write(checkout.join("dirty.txt"), "unsaved\n").expect("dirty the checkout");
    (repo, checkout)
}

/// A live (running) synthetic record, without an LLM behind it.
fn running_worker(id: &str) -> WorkerRecord {
    WorkerRecord {
        id: id.to_string(),
        task: "keep the work going".to_string(),
        model: "test-model".to_string(),
        owner: OWNER.to_string(),
        state: WorkerState::Running {
            step: 1,
            last_command: "cargo test".to_string(),
            started_at: 0,
        },
        metrics: WorkerMetrics::default(),
        logs: LogBuffer::new(),
        pending_steer: Vec::new(),
        resume_tx: None,
        handle: None,
        revision: 0,
    }
}

fn meta(id: &str, repo: &Path) -> WorkerMeta {
    WorkerMeta {
        task: "keep the work going".to_string(),
        repo_path: Some(repo.to_string_lossy().into_owned()),
        ..WorkerMeta::test_meta(id, OWNER)
    }
}

/// A live worker goes through the shutdown path: its row ends `Interrupted`
/// with the salvage branch named, recovery picks it up for continuation, and
/// its uncommitted work is committed onto its branch. A worker the user killed
/// explicitly stays `Failed` and is never offered for continuation.
#[tokio::test]
async fn graceful_shutdown_interrupts_live_workers_only() {
    let scratch = common::TempDir::new_in_tmp("graceful-shutdown");
    let root = ScratchRoot::new(scratch.path());
    let pool = WorkerPool::with_scratch(
        4,
        "http://localhost:1".to_string(),
        "k".to_string(),
        root.clone(),
    );

    let (repo, checkout) = repo_with_dirty_worktree(&scratch, "live1");
    let live = "live1";
    let killed = "killed1";

    // Both workers are live: one the daemon will shut down, one the user kills.
    pool.__test_save_status(
        &meta(live, &repo),
        "test-model",
        RegistryStatus::Running,
        1,
        10,
        "cargo test",
        None,
    );
    pool.__test_insert_worker(running_worker(live)).await;
    pool.__test_register_worktree(live, checkout).await;

    pool.__test_save_status(
        &meta(killed, &repo),
        "test-model",
        RegistryStatus::Running,
        1,
        10,
        "cargo test",
        None,
    );
    pool.__test_insert_worker(running_worker(killed)).await;

    // The explicit kill happens first, so it is already terminal when the
    // shutdown walks the pool and only `live1` is left to interrupt.
    assert!(pool.kill(killed).await, "kill must find the worker");
    let interrupted = pool.kill_all().await;
    assert_eq!(interrupted, 1, "only the live worker is shut down");

    // The live worker's row is Interrupted on disk, naming where its work went.
    let row = load_registry_entry_in(&root, live).expect("the interrupted row survives");
    assert_eq!(
        row.status,
        RegistryStatus::Interrupted,
        "a graceful shutdown must interrupt, not fail"
    );
    assert_eq!(
        row.last_command,
        format!("hub stopped; work saved on branch worker-{live}"),
        "the row must name the salvage branch"
    );
    assert!(
        row.question.is_none(),
        "an interrupted worker has no question"
    );

    // Recovery's own probes pick it up for continuation...
    let candidates = pool.interrupted_workers().await;
    assert!(
        candidates.contains(&live.to_string()),
        "recovery must offer the interrupted worker, got {candidates:?}"
    );
    assert_eq!(
        pool.auto_continue_budget(live).await,
        MAX_AUTO_CONTINUES,
        "a fresh interrupted worker still has its whole continuation budget"
    );

    // ...and its uncommitted work is committed onto its branch.
    let branch = format!("worker-{live}");
    let subject = common::git(&repo, &["log", "-1", "--format=%s", &branch]);
    assert!(
        subject.contains("checkpoint before kill"),
        "the shutdown must have committed the worktree, branch head was {subject:?}"
    );
    assert_eq!(
        common::git(&repo, &["show", &format!("{branch}:dirty.txt")]).trim(),
        "unsaved",
        "the live worker's uncommitted file must survive on its branch"
    );

    // An explicit kill is unchanged: Failed on disk, never offered as a
    // continuation candidate.
    let killed_row = load_registry_entry_in(&root, killed).expect("the failed row survives");
    assert_eq!(
        killed_row.status,
        RegistryStatus::Failed,
        "an explicit kill stays failed"
    );
    assert!(
        !pool
            .interrupted_workers()
            .await
            .contains(&killed.to_string()),
        "an explicitly killed worker must not be offered for continuation"
    );
}
