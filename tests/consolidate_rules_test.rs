//! Consolidation attribution and paused-question regression tests.
mod common;

use mini_swe_mcp::agent::CONSOLIDATOR_INSTRUCTIONS;
use mini_swe_mcp::pool::{RegistryStatus, WorkerMeta, WorkerRole, save_registry_entry_in};

fn actor() -> WorkerMeta {
    WorkerMeta {
        task: "integrate".into(),
        group: Some("round".into()),
        role: WorkerRole::Consolidate,
        ..WorkerMeta::test_meta("consolidator", "owner")
    }
}

#[test]
fn instructions_explain_interactions_and_paused_answers() {
    assert!(CONSOLIDATOR_INSTRUCTIONS.contains("type/function/field"));
    assert!(CONSOLIDATOR_INSTRUCTIONS.contains("never CONSOLIDATE_WAIT a paused worker"));
}

#[tokio::test]
async fn wait_returns_full_paused_question_immediately() {
    let harness = common::IsolatedPool::new(2, "consolidate-question");
    let question = "Which field should I use?\nThe new WorkerMeta field is absent here.\nPlease supply its default.";
    let mut meta = actor();
    meta.id = "worker".into();
    meta.role = WorkerRole::Worker;
    save_registry_entry_in(
        &harness.root(),
        &meta.entry(
            "test",
            RegistryStatus::Paused,
            1,
            10,
            "question",
            Some(question.into()),
        ),
    );
    let result = tokio::time::timeout(
        std::time::Duration::from_secs(1),
        harness
            .pool
            .consolidate_wait(&actor(), &["worker".into()], Some(60)),
    )
    .await
    .expect("paused targets must not wait");
    assert!(result.contains(question), "{result}");
    assert!(
        result.contains("answer it with CONSOLIDATE_STEER"),
        "{result}"
    );
}

use mini_swe_mcp::pool::{WorkerPool, WorkerState};
use mini_swe_mcp::worktree::ScratchRoot;
use std::path::Path;

async fn wait_state(pool: &WorkerPool, id: &str, paused: bool) -> WorkerState {
    wait_state_where(pool, id, |state| {
        (matches!(state, WorkerState::Paused { .. }) == paused)
            && !matches!(state, WorkerState::Running { .. })
    })
    .await
}

async fn wait_state_where(
    pool: &WorkerPool,
    id: &str,
    predicate: impl Fn(&WorkerState) -> bool,
) -> WorkerState {
    tokio::time::timeout(std::time::Duration::from_secs(20), async {
        let mut changes = pool.subscribe_changes();
        loop {
            if let Some(state) = pool.get_worker_state(id).await
                && predicate(&state)
            {
                return state;
            }
            changes.changed().await.expect("pool notification");
        }
    })
    .await
    .expect("worker reaches requested state")
}

async fn dispatch(pool: &WorkerPool, repo: &Path, role: WorkerRole) -> String {
    pool.dispatch_with_role(
        "owner".into(),
        "exercise consolidation rules".into(),
        "test-model".into(),
        None,
        repo.to_path_buf(),
        10,
        Some("round".into()),
        None,
        false,
        None,
        Vec::new(),
        role,
    )
    .await
    .expect("dispatch")
}

fn repo(tag: &str) -> common::TempDir {
    let repo = common::TempDir::new_in_tmp(tag);
    common::git(repo.path(), &["init", "-b", "master"]);
    common::git(repo.path(), &["config", "user.name", "test"]);
    common::git(repo.path(), &["config", "user.email", "test@localhost"]);
    std::fs::write(repo.path().join("seed"), "baseline\n").unwrap();
    common::git(repo.path(), &["add", "."]);
    common::git(repo.path(), &["commit", "-m", "baseline"]);
    repo
}

#[tokio::test]
async fn steered_question_goes_to_consolidator_not_orchestrator_watch() {
    let repo = repo("consolidate-routing-repo");
    let scratch = common::TempDir::new_in_tmp("consolidate-routing-pool");
    let llm = common::fake_llm::FakeLlm::spawn(
        "echo ASK_ORCHESTRATOR: initial question",
        "echo ASK_ORCHESTRATOR: steered question",
    )
    .await;
    let pool = WorkerPool::with_scratch(
        2,
        llm.base_url().into(),
        "test".into(),
        ScratchRoot::new(scratch.path()),
    );
    let worker = dispatch(&pool, repo.path(), WorkerRole::Worker).await;
    wait_state(&pool, &worker, true).await;
    let consolidator = dispatch(&pool, repo.path(), WorkerRole::Consolidate).await;
    wait_state(&pool, &consolidator, true).await;
    let mut actor = actor();
    actor.id = consolidator.clone();
    let server = mini_swe_mcp::mcp::McpServer::new(pool.clone(), "test-model".into());
    let mut ctx = mini_swe_mcp::mcp::ConnectionContext::stdio();
    ctx.agent_id = Some("owner".into());
    let watch = serde_json::json!({"action":"watch", "worker_ids":[worker], "timeout_secs":0});
    let initial = server
        .execute_tool_for("worker", watch.clone(), &ctx)
        .await
        .unwrap();
    assert!(initial.to_string().contains("needs_input"), "{initial}");
    let result = pool
        .consolidate_steer(&actor, &worker, "answer".into())
        .await;
    assert!(result.contains("resumed"), "{result}");
    let state = wait_state_where(&pool, &worker, |state| {
        matches!(state, WorkerState::Paused { question, .. } if question.contains("steered question"))
    }).await;
    assert!(
        matches!(state, WorkerState::Paused { ref question, .. } if question.contains("steered question")),
        "{state:?}"
    );
    let waited = pool
        .consolidate_wait(&actor, std::slice::from_ref(&worker), Some(60))
        .await;
    assert!(waited.contains("steered question"), "{waited}");
    let event = server
        .execute_tool_for("worker", watch.clone(), &ctx)
        .await
        .unwrap();
    assert!(!event.to_string().contains("needs_input"), "{event}");
    // An orchestrator steer takes back responsibility for the next question.
    pool.steer(&worker, "orchestrator answer".into())
        .await
        .unwrap();
    assert!(!pool.question_for_consolidator(&worker).await);
    pool.kill(&worker).await;
    pool.kill(&consolidator).await;
}

#[tokio::test]
async fn completion_integrates_round_base_not_new_master_tip() {
    let repo = repo("consolidate-base-repo");
    let scratch = common::TempDir::new_in_tmp("consolidate-base-pool");
    let llm = common::fake_llm::FakeLlm::spawn(
        "echo ASK_ORCHESTRATOR: ready",
        "echo worker > worker-file",
    )
    .await;
    let pool = WorkerPool::with_scratch(
        2,
        llm.base_url().into(),
        "test".into(),
        ScratchRoot::new(scratch.path()),
    );
    let worker = dispatch(&pool, repo.path(), WorkerRole::Worker).await;
    wait_state(&pool, &worker, true).await;
    pool.steer(&worker, "finish initial work".into())
        .await
        .unwrap();
    let initial = wait_state(&pool, &worker, false).await;
    assert!(
        matches!(initial, WorkerState::Completed { .. }),
        "{initial:?}"
    );
    std::fs::write(repo.path().join("round-file"), "round base\n").unwrap();
    common::git(repo.path(), &["add", "."]);
    common::git(repo.path(), &["commit", "-m", "round base"]);
    let base = common::git(repo.path(), &["rev-parse", "HEAD"]);
    let consolidator = dispatch(&pool, repo.path(), WorkerRole::Consolidate).await;
    wait_state(&pool, &consolidator, true).await;
    std::fs::write(repo.path().join("new-tip"), "outside round\n").unwrap();
    common::git(repo.path(), &["add", "."]);
    common::git(repo.path(), &["commit", "-m", "new moving tip"]);
    let mut actor = actor();
    actor.id = consolidator.clone();
    let result = pool
        .consolidate_steer(&actor, &worker, "complete".into())
        .await;
    assert!(result.contains("revising"), "{result}");
    let completed = wait_state(&pool, &worker, false).await;
    assert!(
        matches!(completed, WorkerState::Completed { .. }),
        "{completed:?}"
    );
    let branch = format!("worker-{worker}");
    common::git(
        repo.path(),
        &["merge-base", "--is-ancestor", base.trim(), &branch],
    );
    let files = common::git(repo.path(), &["ls-tree", "--name-only", &branch]);
    assert!(files.contains("round-file"), "{files}");
    assert!(!files.contains("new-tip"), "{files}");
    // The consolidator itself still integrates the moving base at completion.
    pool.steer(&consolidator, "complete".into()).await.unwrap();
    let completed = wait_state(&pool, &consolidator, false).await;
    assert!(
        matches!(completed, WorkerState::Completed { .. }),
        "{completed:?}"
    );
    let files = common::git(
        repo.path(),
        &["ls-tree", "--name-only", &format!("worker-{consolidator}")],
    );
    assert!(files.contains("new-tip"), "{files}");
}
