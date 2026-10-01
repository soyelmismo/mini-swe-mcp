//! Integration tests for the consolidator's harness-mediated steer and wait.
//!
//! A consolidator routes a failure back to the worker that owns it by echoing
//! `CONSOLIDATE_STEER <id> <message>`, and blocks until its group stops by
//! echoing `CONSOLIDATE_WAIT <id> ... [timeout=<secs>]`; both are executed on
//! the harness, never in the sandbox. These tests drive
//! [`mini_swe_mcp::pool::WorkerPool::consolidate_steer`] and
//! [`mini_swe_mcp::pool::WorkerPool::consolidate_wait`] directly -- no LLM, no
//! sandbox -- and never touch the host repository.

mod common;

use common::{IsolatedPool, unique_suffix};
use mini_swe_mcp::pool::{
    LogBuffer, RegistryStatus, WorkerMeta, WorkerMetrics, WorkerPool, WorkerRecord,
    WorkerRegistryEntry, WorkerRole, WorkerState, save_registry_entry_in,
};
use mini_swe_mcp::worktree::ScratchRoot;

const OWNER: &str = "agent-a";
const FOREIGN_OWNER: &str = "agent-b";
const GROUP: &str = "round-1";

/// A pool over its own temporary scratch root.
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

    /// The consolidator's own metadata, as its dispatch wrote it.
    fn consolidator(&self, id: &str) -> WorkerMeta {
        WorkerMeta {
            id: id.to_string(),
            task: "integrate the round".to_string(),
            group: Some(GROUP.to_string()),
            role: WorkerRole::Consolidate,
            repo_path: None,
            owner: OWNER.to_string(),
            started_at: 0,
            pid: std::process::id(),
            revision: 0,
            auto_continues: 0,
            metrics: WorkerMetrics::default(),
        }
    }
}

/// The registry row a target worker leaves behind.
fn worker_row(id: &str, owner: &str, status: RegistryStatus) -> WorkerRegistryEntry {
    WorkerRegistryEntry {
        id: id.to_string(),
        pid: std::process::id(),
        task: "do the work".to_string(),
        model: "test".to_string(),
        status,
        step: 3,
        max_turns: 10,
        last_command: "cargo test".to_string(),
        question: None,
        started_at: 0,
        updated_at: 0,
        group: Some(GROUP.to_string()),
        role: WorkerRole::Worker,
        repo_path: None,
        owner: Some(owner.to_string()),
        metrics: WorkerMetrics::default(),
        base_branch: None,
        base_commit: None,
        revision: 0,
        auto_continues: 0,
    }
}

/// A live worker of `owner` in the consolidator's group: the registry row the
/// delegation check reads, plus the in-process record a steer queues on.
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

/// The state a worker reaches when it finishes; `verified` is what the wait
/// line reports.
fn completed(id: &str, verified: bool) -> WorkerState {
    WorkerState::Completed {
        turns: 3,
        diff: String::new(),
        summary: "done".to_string(),
        completed_at: 0,
        artifacts: Vec::new(),
        branch: Some(format!("worker-{id}")),
        verified: Some(verified),
        metrics: WorkerMetrics::default(),
        revision: 0,
    }
}

#[test]
fn steer_reaches_a_live_worker_of_the_same_owner_and_group() {
    let h = Harness::new("steer-live");
    let worker = format!("w1-{}", unique_suffix("w"));
    let consolidator = format!("consol-{}", unique_suffix("c"));
    let meta = h.consolidator(&consolidator);
    let root = h.root();
    let pool = &h.pool.pool;
    let observation = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime")
        .block_on(async {
            insert_live_worker(pool, &root, &worker, OWNER).await;
            pool.consolidate_steer(&meta, &worker, "fix the parser".to_string())
                .await
        });
    assert_eq!(observation, format!("{worker} steered (revision 0)"));
}

#[test]
fn steer_of_another_owners_worker_is_refused() {
    let h = Harness::new("steer-own");
    let worker = format!("w1-{}", unique_suffix("w"));
    let consolidator = format!("consol-{}", unique_suffix("c"));
    let meta = h.consolidator(&consolidator);
    let root = h.root();
    let pool = &h.pool.pool;
    let observation = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime")
        .block_on(async {
            insert_live_worker(pool, &root, &worker, FOREIGN_OWNER).await;
            pool.consolidate_steer(&meta, &worker, "fix the parser".to_string())
                .await
        });
    assert_eq!(
        observation,
        format!("{worker} refused: the target belongs to another owner")
    );
}

#[test]
fn wait_returns_once_the_target_completes() {
    let h = Harness::new("wait-done");
    let worker = format!("w1-{}", unique_suffix("w"));
    let consolidator = format!("consol-{}", unique_suffix("c"));
    let meta = h.consolidator(&consolidator);
    let root = h.root();
    let pool = h.pool.pool.clone();
    let observation = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime")
        .block_on(async {
            insert_live_worker(&pool, &root, &worker, OWNER).await;
            let waiting = tokio::spawn({
                let ids = vec![worker.clone()];
                let pool = pool.clone();
                async move { pool.consolidate_wait(&meta, &ids, Some(30)).await }
            });
            // The target is still running, so the wait must block rather than
            // answer with a state the worker never reached.
            for _ in 0..200 {
                assert!(
                    !waiting.is_finished(),
                    "the wait answered while its target was still running"
                );
                tokio::task::yield_now().await;
            }
            // It stops; the wait answers with the state it reached.
            pool.__test_set_worker_state(&worker, completed(&worker, true))
                .await;
            waiting.await.expect("the wait task panicked")
        });
    assert_eq!(observation, format!("{worker} completed verified"));
}

#[test]
fn wait_reports_a_timeout_for_a_worker_that_never_stops() {
    let h = Harness::new("wait-time");
    let worker = format!("w1-{}", unique_suffix("w"));
    let consolidator = format!("consol-{}", unique_suffix("c"));
    let meta = h.consolidator(&consolidator);
    let root = h.root();
    let pool = &h.pool.pool;
    let observation = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime")
        .block_on(async {
            insert_live_worker(pool, &root, &worker, OWNER).await;
            pool.consolidate_wait(&meta, std::slice::from_ref(&worker), Some(1))
                .await
        });
    assert_eq!(
        observation,
        format!("{worker} running (still running after 1s)")
    );
}
