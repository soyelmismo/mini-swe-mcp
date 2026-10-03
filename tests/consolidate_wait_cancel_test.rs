//! A `CONSOLIDATE_WAIT` that is cancelled mid-wait leaves nothing behind.
//!
//! A kill aborts the worker task, a hub shutdown interrupts it, and a panic
//! unwinds it: the wait's future is dropped with the group still running. The
//! pool must release the claim `harness_wait_in_flight` reads on the way out,
//! exactly as it releases the `command_running` mark. A claim that outlived
//! the wait would own the worker's reported command for good -- every later
//! step of a revision continued on the same worker id would find the claim
//! held and skip its own write, so `status` would name a wait the pool was
//! never running, for the rest of that worker's life.
//!
//! These tests drive [`WorkerPool::consolidate_wait`] directly -- no LLM, no
//! sandbox -- and never touch the host repository.

mod common;

use common::{IsolatedPool, unique_suffix};
use mini_swe_mcp::pool::{
    LogBuffer, RegistryStatus, WorkerMeta, WorkerMetrics, WorkerPool, WorkerRecord,
    WorkerRegistryEntry, WorkerRole, WorkerState, save_registry_entry_in,
};
use mini_swe_mcp::worktree::ScratchRoot;

const OWNER: &str = "agent-a";
const GROUP: &str = "round-1";

fn consolidator_meta(id: &str) -> WorkerMeta {
    WorkerMeta {
        task: "integrate the round".to_string(),
        group: Some(GROUP.to_string()),
        role: WorkerRole::Consolidate,
        ..WorkerMeta::test_meta(id, OWNER)
    }
}

fn worker_row(id: &str) -> WorkerRegistryEntry {
    WorkerRegistryEntry {
        task: "do the work".to_string(),
        status: RegistryStatus::Running,
        step: 3,
        last_command: "cargo test".to_string(),
        group: Some(GROUP.to_string()),
        ..WorkerRegistryEntry::test_row(id, OWNER)
    }
}

async fn insert_live_worker(pool: &WorkerPool, root: &ScratchRoot, id: &str) {
    save_registry_entry_in(root, &worker_row(id));
    pool.__test_insert_worker(WorkerRecord {
        id: id.to_string(),
        task: "do the work".to_string(),
        model: "test".to_string(),
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
    })
    .await;
}

/// A wait aborted with its group still running gives up its claim on the
/// command label, so the next step owns it again.
#[test]
fn an_aborted_wait_releases_its_claim_on_the_command_label() {
    let harness = IsolatedPool::new(4, "wait-cancel");
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("runtime");
    rt.block_on(async move {
        let pool = harness.pool.clone();
        let worker = format!("w1-{}", unique_suffix("w"));
        let consolidator = format!("consol-{}", unique_suffix("c"));
        insert_live_worker(&pool, &harness.root(), &worker).await;
        insert_live_worker(&pool, &harness.root(), &consolidator).await;
        let wait_pool = pool.clone();
        let wait_worker = worker.clone();
        let wait_meta = consolidator_meta(&consolidator);
        let wait = tokio::spawn(async move {
            wait_pool
                .consolidate_wait(
                    &wait_meta,
                    &[wait_worker],
                    // Long enough that the wait is still in flight when the
                    // abort lands: the wait only ends on its own when the
                    // group stops, and this one never does.
                    Some(3600),
                )
                .await
        });
        while !pool.harness_wait_in_flight(&consolidator) {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        // What a kill does to a worker task: abort it where it waits.
        wait.abort();
        let _ = wait.await;
        assert!(
            !pool.harness_wait_in_flight(&consolidator),
            "an aborted wait must release the claim on the command label, or every \
             later step of a revision on this worker id skips its own write"
        );
        // The command-in-flight mark is released the same way: the worker is
        // idle again, so ordinary stall detection applies.
        let progress = pool
            .worker_progress(&consolidator)
            .await
            .expect("the consolidator is still a live record");
        assert_eq!(
            progress.command_started_at, None,
            "an aborted wait must not leave the worker marked as running a command"
        );
    });
}
