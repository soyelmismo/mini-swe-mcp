//! Round delivery must persist the same acknowledgments as individual delivery.
use crate::common;
use mini_swe_mcp::mcp::{ConnectionContext, McpServer};
use mini_swe_mcp::pool::{LogBuffer, WorkerMetrics, WorkerRecord, WorkerState};
use std::collections::BTreeSet;
#[tokio::test]
async fn a_round_acknowledgment_survives_router_restart() {
    let isolated = common::IsolatedPool::new(2, "round-ack");
    let hub = common::TempDir::new_in_tmp("round-ack-hub");
    isolated
        .pool
        .__test_insert_worker(WorkerRecord {
            id: "round-worker".into(),
            task: "contribution".into(),
            model: "test".into(),
            owner: "round-owner".into(),
            state: WorkerState::Completed {
                summary: "done".into(),
                turns: 1,
                diff: String::new(),
                completed_at: 0,
                artifacts: vec![],
                metrics: WorkerMetrics::default(),
                revision: 0,
                branch: Some("worker-round-worker".into()),
                verified: Some(true),
                report: None,
                verdicts: None,
            },
            metrics: WorkerMetrics::default(),
            logs: LogBuffer::new(),
            pending_steer: vec![],
            resume_tx: None,
            handle: None,
            revision: 0,
        })
        .await;
    let mut ctx = ConnectionContext::hub_connection(1);
    ctx.agent_id = Some("round-owner".into());
    let one = ["round-worker".to_string()].into_iter().collect();
    let server = McpServer::new(isolated.pool.clone(), "test".into());
    let events = server.start_hub_events(Some(hub.path())).await;
    let result = server
        .watch_poll(&ctx, &one, &BTreeSet::new(), true, true)
        .await
        .unwrap();
    assert_eq!(result["events"][0]["event"], "round", "{result}");
    events.abort();
    let _ = events.await;
    let restarted = McpServer::new(isolated.pool.clone(), "test".into());
    let events = restarted.start_hub_events(Some(hub.path())).await;
    let result = restarted
        .watch_poll(&ctx, &one, &BTreeSet::new(), true, false)
        .await
        .unwrap();
    assert!(
        result["events"].as_array().is_some_and(Vec::is_empty),
        "the acknowledged round must not replay: {result}"
    );
    events.abort();
    let _ = events.await;
}
