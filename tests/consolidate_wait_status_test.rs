//! A consolidator blocked in `CONSOLIDATE_WAIT` is running a command as far as
//! the status view and the stall detector are concerned: the wait holds the
//! pool's `command_running` mark and names itself as the command in flight.
//! These tests drive [`WorkerPool::consolidate_wait`] directly -- no LLM, no
//! sandbox -- and never touch the host repository.

mod common;

use common::{IsolatedPool, unique_suffix};
use mini_swe_mcp::cli::watch::select_event;
use mini_swe_mcp::pool::{
    LogBuffer, RegistryStatus, WorkerMeta, WorkerMetrics, WorkerPool, WorkerRecord,
    WorkerRegistryEntry, WorkerRole, WorkerState, save_registry_entry_in,
};
use mini_swe_mcp::worktree::ScratchRoot;
use serde_json::json;

const OWNER: &str = "agent-a";
const GROUP: &str = "round-1";

struct Harness {
    pool: IsolatedPool,
}

impl Harness {
    fn new(tag: &str) -> Self {
        Self {
            pool: IsolatedPool::new(4, tag),
        }
    }

    fn root(&self) -> ScratchRoot {
        self.pool.root()
    }

    fn consolidator(&self, id: &str) -> WorkerMeta {
        WorkerMeta {
            task: "integrate the round".to_string(),
            group: Some(GROUP.to_string()),
            role: WorkerRole::Consolidate,
            ..WorkerMeta::test_meta(id, OWNER)
        }
    }
}

fn worker_row(id: &str, owner: &str, status: RegistryStatus) -> WorkerRegistryEntry {
    WorkerRegistryEntry {
        task: "do the work".to_string(),
        status,
        step: 3,
        last_command: "cargo test".to_string(),
        group: Some(GROUP.to_string()),
        ..WorkerRegistryEntry::test_row(id, owner)
    }
}

async fn insert_live_worker(pool: &WorkerPool, root: &ScratchRoot, id: &str, owner: &str) {
    save_registry_entry_in(root, &worker_row(id, owner, RegistryStatus::Running));
    pool.__test_insert_worker(WorkerRecord {
        id: id.to_string(),
        task: "do the work".to_string(),
        model: "test".to_string(),
        owner: owner.to_string(),
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
    })
    .await;
}

/// The watch view of a live worker, as `watch_snapshot` builds it.
fn watch_view(progress: &mini_swe_mcp::pool::WorkerProgress, now: u64) -> serde_json::Value {
    json!({
        "worker_id": "c",
        "status": "running",
        "step": progress.step,
        "last_step_at": now.saturating_sub(1800),
        "command_started_at": progress.command_started_at,
    })
}

/// While the wait blocks, the consolidator's progress names the wait as the
/// command in flight and carries a start time, so the stall detector reads the
/// step as work even past the idle threshold.
#[test]
fn a_waiting_consolidator_is_running_a_command() {
    let h = Harness::new("wait-status");
    let worker = format!("w1-{}", unique_suffix("w"));
    let consolidator = format!("consol-{}", unique_suffix("c"));
    let _meta = h.consolidator(&consolidator);
    let root = h.root();
    let pool = h.pool.pool.clone();
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("runtime");
    rt.block_on(async move {
        insert_live_worker(&pool, &root, &worker, OWNER).await;
        insert_live_worker(&pool, &root, &consolidator, OWNER).await;
        let wait_pool = pool.clone();
        let wait_meta = h.consolidator(&consolidator);
        let wait_worker = worker.clone();
        let wait = tokio::spawn(async move {
            wait_pool
                .consolidate_wait(&wait_meta, &[wait_worker], Some(2))
                .await
        });
        // Poll until the wait has taken its mark.
        let progress = loop {
            let progress = pool.worker_progress(&consolidator).await.unwrap();
            if progress.command_started_at.is_some() {
                break progress;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        };
        assert!(
            progress
                .last_command
                .as_deref()
                .is_some_and(|c| c.starts_with("CONSOLIDATE_WAIT")),
            "the wait should name itself as the command in flight: {:?}",
            progress.last_command
        );
        let now = mini_swe_mcp::pool::unix_timestamp();
        assert!(
            select_event(&watch_view(&progress, now), None, now + 1800).is_none(),
            "a waiting consolidator must not stall"
        );
        wait.await.expect("wait task");
        // After the wait returns, ordinary stall detection applies again: the
        // mark is gone, so an idle step past the threshold stalls.
        let progress = pool.worker_progress(&consolidator).await.unwrap();
        assert!(progress.command_started_at.is_none());
        assert!(
            select_event(&watch_view(&progress, now), None, now + 1800).is_some(),
            "an idle step after the wait must stall again"
        );
    });
}
