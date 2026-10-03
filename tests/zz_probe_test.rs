// Probe 5: is a WAIT_JOB / bash command able to overlap a CONSOLIDATE_WAIT on
// the same worker id? The turn is single-threaded per worker, so the wait ends
// the turn. But: does anything else publish command_running for the SAME id?
// Check: two pools / hub handover paths.
mod common;
use common::{IsolatedPool, unique_suffix};
use mini_swe_mcp::pool::{
    LogBuffer, RegistryStatus, WorkerMeta, WorkerMetrics, WorkerPool, WorkerRecord,
    WorkerRegistryEntry, WorkerRole, WorkerState, save_registry_entry_in,
};
use mini_swe_mcp::worktree::ScratchRoot;

const OWNER: &str = "agent-a";
fn wrow(id: &str) -> WorkerRegistryEntry {
    WorkerRegistryEntry { task: "t".into(), status: RegistryStatus::Running, step: 3,
        last_command: "cargo test".into(), group: Some("g".into()), ..WorkerRegistryEntry::test_row(id, OWNER) }
}
async fn ins(pool: &WorkerPool, root: &ScratchRoot, id: &str) {
    save_registry_entry_in(root, &wrow(id));
    pool.__test_insert_worker(WorkerRecord { id: id.into(), task: "t".into(), model: "test".into(),
        owner: OWNER.into(), state: WorkerState::Running { step: 1, last_command: "cargo test".into(), started_at: 0 },
        metrics: WorkerMetrics::default(), logs: LogBuffer::new(), pending_steer: vec![], resume_tx: None, handle: None, revision: 0 }).await;
}

#[tokio::test]
async fn probe_overlapping_turns_for_one_id() {
    let h = IsolatedPool::new(4, "probe-turns");
    let pool = h.pool.clone();
    let c = format!("consol-{}", unique_suffix("c"));
    ins(&pool, &h.root(), &c).await;
    // Simulate: a turn's run_gated publishes a mark, and CONCURRENTLY the same
    // worker id runs a wait (as the F13 code path would if a revision raced).
    let g1 = pool.command_running(&c);
    // second task, same id
    let p = pool.clone();
    let id2 = c.clone();
    let t = tokio::spawn(async move {
        let g2 = p.command_running(&id2);
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        let during = p.worker_progress(&id2).await.unwrap().command_started_at;
        drop(g2);
        let after_inner = p.worker_progress(&id2).await.unwrap().command_started_at;
        (during, after_inner)
    });
    let (during, after_inner) = t.await.unwrap();
    println!("inner in flight started={during:?} after inner drop but outer live={after_inner:?}");
    drop(g1);
    println!("after outer drop: {:?}", pool.worker_progress(&c).await.unwrap().command_started_at);
}
