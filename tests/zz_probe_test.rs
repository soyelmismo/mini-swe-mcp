// Probe 2: can a NON-OWNER watch drop another agent's stall episode from its backlog?
mod common;
use common::{IsolatedPool, unique_suffix};
use mini_swe_mcp::pool::{
    LogBuffer, RegistryStatus, WorkerMeta, WorkerMetrics, WorkerPool, WorkerRecord,
    WorkerRegistryEntry, WorkerRole, WorkerState, save_registry_entry_in,
};
use mini_swe_mcp::worktree::ScratchRoot;
use serde_json::json;

const OWNER: &str = "agent-a";

fn cmeta(id: &str, owner: &str) -> WorkerMeta {
    WorkerMeta { task: "t".into(), group: Some("g".into()), role: WorkerRole::Consolidate, ..WorkerMeta::test_meta(id, owner) }
}
fn wrow(id: &str, owner: &str) -> WorkerRegistryEntry {
    WorkerRegistryEntry { task: "t".into(), status: RegistryStatus::Running, step: 3,
        last_command: "cargo test".into(), group: Some("g".into()), ..WorkerRegistryEntry::test_row(id, owner) }
}
async fn ins(pool: &WorkerPool, root: &ScratchRoot, id: &str, owner: &str) {
    save_registry_entry_in(root, &wrow(id, owner));
    pool.__test_insert_worker(WorkerRecord { id: id.into(), task: "t".into(), model: "test".into(),
        owner: owner.into(), state: WorkerState::Running { step: 1, last_command: "cargo test".into(), started_at: 0 },
        metrics: WorkerMetrics::default(), logs: LogBuffer::new(), pending_steer: vec![], resume_tx: None, handle: None, revision: 0 }).await;
}

#[test]
fn foreign_agent_watch_and_suppression() {
    let h = IsolatedPool::new(4, "probe-foreign");
    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build().unwrap();
    rt.block_on(async move {
        let pool = h.pool.clone();
        let w = format!("w1-{}", unique_suffix("w"));
        ins(&pool, &h.root(), &w, "agent-b").await;
        // agent-b consolidator starts a wait (holds the mark)
        let c = format!("consol-{}", unique_suffix("c"));
        ins(&pool, &h.root(), &c, "agent-b").await;
        let p = pool.clone(); let m = cmeta(&c, "agent-b"); let ww = w.clone();
        let t = tokio::spawn(async move { p.consolidate_wait(&m, &[ww], Some(2)).await });
        while !pool.harness_wait_in_flight(&c) { tokio::time::sleep(std::time::Duration::from_millis(5)).await; }
        // agent-b's view: in flight (command_started_at present)
        let mut v = json!({"worker_id":c,"owner":"agent-b","group":"g","model":"t","status":"running",
            "step":0,"revision":0,"branch":format!("worker-{c}"),"last_step_at":0,
            "command_started_at": mini_swe_mcp::pool::unix_timestamp(),
            "metrics": mini_swe_mcp::pool::WorkerMetrics::default()});
        v["last_step_at"] = json!(mini_swe_mcp::pool::unix_timestamp().saturating_sub(1800));
        let snap: std::collections::BTreeMap<String, serde_json::Value> =
            std::collections::BTreeMap::from([(c.clone(), v)]);
        println!("snapshot built: {}", snap.len());
        t.await.unwrap();
    });
}
