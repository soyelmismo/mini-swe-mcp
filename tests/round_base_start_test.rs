//! A continued worker integrates its consolidator's pinned base before turn one.
mod common;

use mini_swe_mcp::pool::{
    WorkerHistory, WorkerMeta, WorkerMetrics, WorkerPool, WorkerState, save_worker_history_in,
};
use mini_swe_mcp::worktree::{ScratchRoot, WorktreeGuard};
use std::time::Duration;

#[tokio::test]
async fn revision_start_does_not_integrate_the_moving_master_tip() {
    let repo = common::TempDir::new_in_tmp("round-start-repo");
    let scratch = common::TempDir::new_in_tmp("round-start-root");
    let root = ScratchRoot::new(scratch.path());
    common::git(repo.path(), &["init", "-b", "master"]);
    common::git(repo.path(), &["config", "user.name", "test"]);
    common::git(repo.path(), &["config", "user.email", "test@localhost"]);
    std::fs::write(repo.path().join("seed"), "seed\n").unwrap();
    common::git(repo.path(), &["add", "."]);
    common::git(repo.path(), &["commit", "-m", "seed"]);
    let mut guard = WorktreeGuard::new_in(&root, repo.path(), "pinned").unwrap();
    std::fs::write(guard.path.join("worker-file"), "work\n").unwrap();
    guard.commit_changes("worker work").unwrap();
    let history = WorkerHistory {
        task: "integrate round".into(),
        group: Some("round".into()),
        role: mini_swe_mcp::pool::WorkerRole::Worker,
        model: "test-model".into(),
        temperature: None,
        repo_path: repo.path().to_string_lossy().into(),
        base_commit: guard.base_commit.clone(),
        base_branch: guard.base_branch.clone(),
        branch: guard.branch.clone(),
        network_offline: false,
        verify: None,
        client_env: Vec::new(),
        max_turns: 2,
        review_after: None,
        revision: 0,
        auto_continues: 0,
        owner: Some("owner".into()),
        messages: Vec::new(),
    };
    drop(guard);
    save_worker_history_in(&root, "pinned", &history).unwrap();
    std::fs::write(repo.path().join("round-file"), "round\n").unwrap();
    common::git(repo.path(), &["add", "."]);
    common::git(repo.path(), &["commit", "-m", "round base"]);
    let base = common::git(repo.path(), &["rev-parse", "HEAD"]);
    std::fs::write(repo.path().join("outside-round"), "later\n").unwrap();
    common::git(repo.path(), &["add", "."]);
    common::git(repo.path(), &["commit", "-m", "later master"]);
    let llm = common::fake_llm::FakeLlm::spawn("echo ASK_ORCHESTRATOR: inspect", "echo no").await;
    let pool = WorkerPool::with_scratch(1, llm.base_url().into(), "key".into(), root.clone());
    let actor = WorkerMeta {
        id: "actor".into(),
        task: "consolidate".into(),
        group: Some("round".into()),
        role: mini_swe_mcp::pool::WorkerRole::Consolidate,
        repo_path: Some(history.repo_path.clone()),
        owner: "owner".into(),
        started_at: 0,
        pid: std::process::id(),
        revision: 0,
        auto_continues: 0,
        metrics: WorkerMetrics::default(),
        report: None,
        verified: None,
    };
    let mut target = actor.clone();
    target.id = "pinned".into();
    target.role = mini_swe_mcp::pool::WorkerRole::Worker;
    mini_swe_mcp::pool::save_registry_entry_in(
        &root,
        &target.entry(
            "test-model",
            mini_swe_mcp::pool::RegistryStatus::Completed,
            2,
            2,
            "done",
            None,
        ),
    );
    std::fs::write(root.join("swe-wt-actor.round-base"), base.trim()).unwrap();
    let observation = pool
        .consolidate_steer(&actor, "pinned", "continue".into())
        .await;
    assert!(observation.contains("revising"), "{observation}");
    tokio::time::timeout(Duration::from_secs(20), async {
        loop {
            if matches!(
                pool.get_worker_state("pinned").await,
                Some(WorkerState::Paused { .. })
            ) {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("worker pauses on its first turn");
    let files = common::git(
        &root.join("swe-wt-pinned"),
        &["ls-tree", "--name-only", "HEAD"],
    );
    assert!(files.contains("round-file"), "{files}");
    assert!(!files.contains("outside-round"), "{files}");
    pool.kill("pinned").await;
}
