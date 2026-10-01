//! A consolidator that steered a worker owns its lifecycle events until the
//! round is over.
//!
//! E14 routes a steered worker's *question* to the consolidator through the
//! steer source; the same rule applies to its completion. While that source
//! names a live consolidator the owner's watch stays quiet, and once the
//! consolidator has finished the stopped worker is visible to its owner again.

mod common;

use common::IsolatedPool;
use mini_swe_mcp::mcp::{ConnectionContext, McpServer};
use mini_swe_mcp::pool::{
    RegistryStatus, WorkerMeta, WorkerMetrics, WorkerPool, WorkerRole, save_registry_entry_in,
};
use mini_swe_mcp::worktree::ScratchRoot;

/// The owner of every worker in this test.
const OWNER: &str = "owner";

/// The registry row a worker of `role` in the round leaves behind.
fn write_row(root: &ScratchRoot, id: &str, role: WorkerRole, status: RegistryStatus) {
    let meta = WorkerMeta {
        report: None,
        id: id.to_string(),
        task: "exercise consolidate routing".to_string(),
        group: Some("round-1".to_string()),
        role,
        repo_path: None,
        owner: OWNER.to_string(),
        started_at: 0,
        pid: std::process::id(),
        revision: 0,
        auto_continues: 0,
        metrics: WorkerMetrics::default(),
        verified: None,
    };
    save_registry_entry_in(
        root,
        &meta.entry("test-model", status, 3, 10, "cargo test", None),
    );
}

/// The steer source that `consolidate_steer` writes for a steered worker.
///
/// The file is what production reads back; writing it directly keeps this test
/// to the routing rule and out of a real consolidator dispatch.
fn write_steer_source(root: &ScratchRoot, worker: &str, consolidator: &str) {
    let source = serde_json::json!({"consolidator": consolidator, "round_base": null});
    std::fs::write(
        root.join(format!("swe-wt-{worker}.steer-source")),
        serde_json::to_vec(&source).unwrap(),
    )
    .unwrap();
}

fn watch_args(worker: &str) -> serde_json::Value {
    serde_json::json!({"action": "watch", "worker_ids": [worker], "timeout_secs": 0})
}

#[tokio::test]
async fn a_steered_completion_waits_for_the_consolidator_to_finish() {
    let harness = IsolatedPool::new(2, "consolidate-events");
    let root = harness.root();
    // A terminal registry row survives the loader only while its worktree
    // still exists; the event router is the reader that keeps it.
    std::fs::create_dir_all(root.join("swe-wt-w1")).unwrap();
    write_row(&root, "w1", WorkerRole::Worker, RegistryStatus::Completed);
    write_row(
        &root,
        "c1",
        WorkerRole::Consolidate,
        RegistryStatus::Running,
    );
    write_steer_source(&root, "w1", "c1");

    let server = McpServer::new(WorkerPool::clone(&harness.pool), "test-model".into());
    let mut ctx = ConnectionContext::stdio();
    ctx.agent_id = Some(OWNER.to_string());

    let hidden = server
        .execute_tool_for("worker", watch_args("w1"), &ctx)
        .await
        .unwrap();
    assert!(
        !hidden.to_string().contains("completed"),
        "a completion steered to a live consolidator must not wake the owner: {hidden}"
    );

    // The consolidator finished, so its steer source names no live owner and
    // normal delivery resumes: the stopped worker is visible again.
    write_row(
        &root,
        "c1",
        WorkerRole::Consolidate,
        RegistryStatus::Completed,
    );
    let shown = server
        .execute_tool_for("worker", watch_args("w1"), &ctx)
        .await
        .unwrap();
    assert!(
        shown.to_string().contains("completed"),
        "the stopped worker must be visible once its consolidator is gone: {shown}"
    );
}
