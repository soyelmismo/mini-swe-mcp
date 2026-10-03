//! A consolidator blocked in `CONSOLIDATE_WAIT` is running a command as far as
//! the status view and the stall detector are concerned: the wait holds the
//! pool's `command_running` mark and names itself as the command in flight.
//! These tests drive [`WorkerPool::consolidate_wait`] directly -- no LLM, no
//! sandbox -- and never touch the host repository.

mod common;

use std::collections::{BTreeMap, BTreeSet};

use common::{IsolatedPool, unique_suffix};
use mini_swe_mcp::cli::watch::{ROUND_STALL_SECS, Snapshot, round_event, select_event};
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

/// A one-worker `--all` round snapshot for the consolidator, as the round
/// watch builds it from the live progress.
fn round_snapshot(
    consolidator: &str,
    progress: &mini_swe_mcp::pool::WorkerProgress,
    now: u64,
) -> Snapshot {
    let mut view = watch_view(progress, now);
    view["worker_id"] = json!(consolidator);
    view["group"] = json!(GROUP);
    BTreeMap::from([(consolidator.to_string(), view)])
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
        // The 20-minute `--all` round threshold is the same rule: a command in
        // flight keeps the round from reading the wait as a stalled step.
        assert!(
            round_event(
                &round_snapshot(&consolidator, &progress, now),
                &BTreeSet::from([consolidator.clone()]),
                &BTreeSet::from([GROUP.to_string()]),
                now + ROUND_STALL_SECS + 600,
                |_| true,
                |_| true,
            )
            .is_none(),
            "a waiting consolidator must not stall its --all round"
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
        // Past the `--all` threshold and with the mark gone, the round reports
        // the same worker as stalled again.
        let event = round_event(
            &round_snapshot(&consolidator, &progress, now),
            &BTreeSet::from([consolidator.clone()]),
            &BTreeSet::from([GROUP.to_string()]),
            now + ROUND_STALL_SECS + 600,
            |_| true,
            |_| true,
        )
        .expect("an idle consolidator stalls its --all round");
        assert_eq!(event["workers"][0]["outcome"], "stalled");
    });
}

/// The wait stamps its command start exactly once, so the status view's
/// `running for` clock only ever grows. A wait that re-published the mark on
/// every poll reset the start time each time, so a wait that had been going for
/// half an hour still read as `running for 0s`.
#[test]
fn a_long_wait_reports_a_growing_elapsed_time() {
    let h = Harness::new("wait-clock");
    let worker = format!("w1-{}", unique_suffix("w"));
    let consolidator = format!("consol-{}", unique_suffix("c"));
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
                .consolidate_wait(&wait_meta, &[wait_worker], Some(3))
                .await
        });
        // Poll until the wait has taken its mark.
        let first = loop {
            let progress = pool.worker_progress(&consolidator).await.unwrap();
            if progress.command_started_at.is_some() {
                break progress;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        };
        let started = first.command_started_at.expect("the wait takes the mark");
        // Let the wait run past the point where a re-publish would have reset
        // the clock, then confirm the start time never moved.
        tokio::time::sleep(std::time::Duration::from_millis(1_200)).await;
        let later = pool.worker_progress(&consolidator).await.unwrap();
        assert_eq!(
            later.command_started_at,
            Some(started),
            "a wait must stamp its command start once, not on every poll"
        );
        // The status view turns that one stamp into a growing elapsed clock.
        let now = mini_swe_mcp::pool::unix_timestamp();
        let elapsed = now.saturating_sub(started);
        assert!(
            elapsed >= 1,
            "a wait in flight must report a non-zero elapsed: {elapsed}s"
        );
        wait.await.expect("wait task");
    });
}

/// The step recorder decides whether to keep the wait's label from the pool's
/// own record of it, never from the recorded command text.
///
/// A step's label is model-written: `CONSOLIDATE_WAIT` is a first word any
/// command can begin with, and the harness's wait is spelled exactly that way.
/// A recorder that re-derived ownership from the text let a worker issuing
/// `CONSOLIDATE_WAIT ...` freeze its own reported command for good -- every
/// later step read the same unchanged label and skipped its own write -- so
/// `status` named a harness wait the pool was never running. The pool must
/// report a wait in flight only while it is really publishing one.
#[test]
fn only_a_wait_the_pool_publishes_counts_as_a_wait_in_flight() {
    let h = Harness::new("wait-ownership");
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("runtime");
    rt.block_on(async {
        let pool = h.pool.pool.clone();
        let worker = format!("w1-{}", unique_suffix("w"));
        let consol = format!("consol-{}", unique_suffix("c"));
        insert_live_worker(&pool, &h.root(), &worker, OWNER).await;
        insert_live_worker(&pool, &h.root(), &consol, OWNER).await;
        // Nothing has been published yet: a step owns the label outright, so
        // even a step whose own command is named like the wait takes it.
        assert!(
            !pool.harness_wait_in_flight(&consol),
            "no wait is in flight before one is published"
        );
        // A real wait publishes one; the pool says so while it runs.
        let wait_pool = pool.clone();
        let wait_meta = h.consolidator(&consol);
        let wait_worker = worker.clone();
        let wait = tokio::spawn(async move {
            wait_pool
                .consolidate_wait(&wait_meta, &[wait_worker], Some(2))
                .await
        });
        while !pool.harness_wait_in_flight(&consol) {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        wait.await.expect("wait task");
        // Once it returns, the claim is gone: a later step writes its own
        // command again, whatever that command happens to be named.
        assert!(
            !pool.harness_wait_in_flight(&consol),
            "a finished wait must leave no claim on the label"
        );
    });
}
