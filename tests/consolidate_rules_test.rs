//! Consolidation attribution and paused-question regression tests.
mod common;

use mini_swe_mcp::agent::CONSOLIDATOR_INSTRUCTIONS;
use mini_swe_mcp::pool::{
    RegistryStatus, WorkerMeta, WorkerMetrics, WorkerRole, save_registry_entry_in,
};

fn actor() -> WorkerMeta {
    WorkerMeta {
        id: "consolidator".into(), task: "integrate".into(), group: Some("round".into()),
        role: WorkerRole::Consolidate, repo_path: None, owner: "owner".into(),
        started_at: 0, pid: std::process::id(), revision: 0, auto_continues: 0,
        metrics: WorkerMetrics::default(), report: None,
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
    save_registry_entry_in(&harness.root(), &meta.entry("test", RegistryStatus::Paused, 1, 10, "question", Some(question.into())));
    let result = tokio::time::timeout(std::time::Duration::from_secs(1), harness.pool.consolidate_wait(&actor(), &["worker".into()], Some(60))).await.expect("paused targets must not wait");
    assert!(result.contains(question), "{result}");
    assert!(result.contains("answer it with CONSOLIDATE_STEER"), "{result}");
}
