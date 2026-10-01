//! A worker that runs out of turns must be reported as stopped, not done.
//!
//! The turn budget is a resource, not a result: an implementer that spends it
//! all without emitting the completion sentinel has checkpointed work but has
//! not finished. Real runs were delivered as `completed` with a diff and no
//! verification, which read as success. These tests drive a scripted fake LLM
//! (no real model) until the budget is gone and assert the distinct
//! `Exhausted` outcome, its stopped-not-done rendering, and its exclusion from
//! the consolidator's round manifest.

mod common;

use std::path::{Path, PathBuf};
use std::time::Duration;

use serde_json::json;

use mini_swe_mcp::mcp::{ConnectionContext, EventKind, McpServer, WorkerView, render_for_test};
use mini_swe_mcp::pool::{WorkerPhase, WorkerPool, WorkerState};
use mini_swe_mcp::worktree::ScratchRoot;

use common::fake_llm::FakeLlm;

const OWNER: &str = "exhausted-owner";
const GROUP: &str = "exhausted-group";

/// A throwaway git repository the worker is dispatched against.
struct TestRepo {
    _scratch: common::TempDir,
    dir: PathBuf,
}

impl TestRepo {
    fn new(tag: &str) -> Self {
        let scratch = common::TempDir::new_in_tmp(&format!("exhaust-repo-{tag}"));
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
        // Build directories are leased by repository hash, filed next to the
        // scratch base rather than inside the repository.
        mini_swe_mcp::cache::remove_build_dir_leases(&self.dir);
    }
}

/// Dispatch `max_turns` of budget against a looping LLM that never completes.
async fn dispatch_exhausted(
    server: &FakeLlm,
    repo: &Path,
    max_turns: usize,
) -> (WorkerPool, String, common::TempDir) {
    let scratch = common::TempDir::new_in_tmp("exhaust-pool");
    let pool = WorkerPool::with_scratch(
        1,
        server.base_url().to_string(),
        "test-key".to_string(),
        ScratchRoot::new(scratch.path()),
    );
    let worker_id = pool
        .dispatch(
            OWNER.to_string(),
            "spend the whole budget without finishing".to_string(),
            "test-model".to_string(),
            None,
            repo.to_path_buf(),
            max_turns,
            Some(GROUP.to_string()),
            None,
            false,
            None,
            Vec::new(),
        )
        .await
        .expect("dispatch the worker");
    (pool, worker_id, scratch)
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
async fn a_worker_that_runs_out_of_turns_is_exhausted_not_completed() {
    let server = FakeLlm::spawn_looping().await;
    let repo = TestRepo::new("distinct");
    let (pool, id, _scratch) = dispatch_exhausted(&server, repo.path(), 3).await;

    let state = wait_terminal(&pool, &id).await;
    let summary = state.to_summary();
    match &state {
        WorkerState::Exhausted {
            turns,
            summary,
            branch,
            ..
        } => {
            assert_eq!(*turns, 3, "the whole budget was spent");
            assert!(
                summary.contains("turn_budget_exhausted"),
                "the summary must not read as a success: {summary}"
            );
            assert_eq!(
                branch.as_deref(),
                Some(format!("worker-{id}").as_str()),
                "the work is checkpointed on the branch"
            );
        }
        other => panic!("expected Exhausted, got {other:?}"),
    }
    assert!(
        !matches!(state, WorkerState::Completed { .. }),
        "running out of turns is not a completion"
    );
    assert_eq!(summary["status"], "Exhausted");
    assert_eq!(summary["reason"], "turn_budget_exhausted");
    assert!(
        common::git_ref_exists(repo.path(), &format!("worker-{id}")),
        "the branch must survive the exhaustion"
    );
    assert_eq!(
        pool.worker_progress(&id).await.expect("progress").phase,
        WorkerPhase::Exhausted
    );
}

#[tokio::test]
async fn exhausted_status_list_and_review_read_as_stopped_not_done() {
    let server = FakeLlm::spawn_looping().await;
    let repo = TestRepo::new("render");
    let (pool, id, _scratch) = dispatch_exhausted(&server, repo.path(), 2).await;
    let state = wait_terminal(&pool, &id).await;
    assert!(matches!(state, WorkerState::Exhausted { .. }));

    let mcp = McpServer::new(pool.clone(), "ninja".to_string());
    let ctx = ConnectionContext {
        agent_id: Some(OWNER.to_string()),
        ..ConnectionContext::hub_connection(3)
    };

    let status = mcp
        .execute_tool_for(
            "worker",
            json!({ "action": "status", "worker_id": id }),
            &ctx,
        )
        .await
        .expect("status answers");
    assert_eq!(status["state"]["state"], "Exhausted");
    let next_step = status["next_step"].as_str().unwrap_or_default();
    assert!(
        next_step.contains("Stopped, not done"),
        "status must read as stopped: {next_step}"
    );
    assert!(
        next_step.contains(&format!("steer {id} \"continue\" --max-turns 2")),
        "status must name the continuation: {next_step}"
    );

    let list = mcp
        .execute_tool_for("worker", json!({ "action": "list" }), &ctx)
        .await
        .expect("list answers");
    let row = list["workers"]
        .as_array()
        .expect("workers array")
        .iter()
        .find(|row| row["id"] == json!(id))
        .expect("the worker is listed");
    assert_eq!(row["state"]["status"], "Exhausted");

    let review = mcp
        .execute_tool_for(
            "worker",
            json!({ "action": "review", "worker_id": id }),
            &ctx,
        )
        .await
        .expect("review answers");
    assert_eq!(review["state"], "Exhausted");
    assert!(
        review["verified"].is_null(),
        "an exhausted worker never verified: {review}"
    );
    let next_command = review["next_command"].as_str().unwrap_or_default();
    assert!(
        next_command.contains("steer") && next_command.contains("--max-turns"),
        "review must point at the continuation: {next_command}"
    );
}

#[tokio::test]
async fn exhausted_event_rendering_says_stopped_not_done() {
    let server = FakeLlm::spawn_looping().await;
    let repo = TestRepo::new("event");
    let (pool, id, _scratch) = dispatch_exhausted(&server, repo.path(), 2).await;
    let state = wait_terminal(&pool, &id).await;
    assert!(matches!(state, WorkerState::Exhausted { .. }));

    let view = WorkerView {
        worker_id: id.clone(),
        event: Some(EventKind::Exhausted),
        status: "exhausted".to_string(),
        turns: 2,
        branch: Some(format!("worker-{id}")),
        ..WorkerView::default()
    };
    let text = render_for_test(&view, EventKind::Exhausted);
    assert!(
        text.contains("Stopped, not done"),
        "the notification must read as stopped: {text}"
    );
    assert!(
        text.contains(&format!("steer {id} \"continue\" --max-turns 2")),
        "the notification must name the continuation: {text}"
    );
}

#[tokio::test]
async fn exhausted_worker_is_not_ready_for_the_round_manifest() {
    let server = FakeLlm::spawn_looping().await;
    let repo = TestRepo::new("manifest");
    let (pool, id, _scratch) = dispatch_exhausted(&server, repo.path(), 2).await;
    let state = wait_terminal(&pool, &id).await;
    assert!(matches!(state, WorkerState::Exhausted { .. }));

    let manifest = pool.round_manifest(OWNER, GROUP, repo.path()).await;
    assert!(
        manifest.ready.is_empty(),
        "an exhausted worker must not be ready to integrate: {}",
        manifest.render()
    );
    let not_ready = manifest
        .not_ready
        .iter()
        .find(|worker| worker.id == id)
        .unwrap_or_else(|| panic!("it must be listed not ready: {}", manifest.render()));
    assert_eq!(not_ready.state, "Exhausted");
    assert!(
        manifest.render().contains("ready (completed, branch not yet merged):\n  (none)"),
        "the ready section must be empty: {}",
        manifest.render()
    );
}
