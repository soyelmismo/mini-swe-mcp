//! F13 regression: a consolidator waiting in `CONSOLIDATE_WAIT` must show its
//! real wait time and must never read as stalled, in plain watch mode or in
//! the `--all` round.
//!
//! Both halves are driven off the pool, with no LLM, no sandbox and no
//! repository: the wait is [`WorkerPool::consolidate_wait`] running against
//! workers inserted by the test, and the stall rules are the ones the CLI and
//! the daemon share ([`select_event`], [`round_event`]).

mod common;

use std::collections::{BTreeMap, BTreeSet};

use common::{IsolatedPool, unique_suffix};
use mini_swe_mcp::cli::watch::{ROUND_STALL_SECS, Snapshot, round_event, select_event};
use mini_swe_mcp::pool::{
    LogBuffer, RegistryStatus, WorkerMeta, WorkerMetrics, WorkerPool, WorkerRecord, WorkerRole,
    WorkerState, save_registry_entry_in,
};
use mini_swe_mcp::worktree::ScratchRoot;
use serde_json::json;

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

fn row(id: &str, status: RegistryStatus) -> mini_swe_mcp::pool::WorkerRegistryEntry {
    mini_swe_mcp::pool::WorkerRegistryEntry {
        task: "do the work".to_string(),
        status,
        step: 3,
        last_command: "cargo test".to_string(),
        group: Some(GROUP.to_string()),
        ..mini_swe_mcp::pool::WorkerRegistryEntry::test_row(id, OWNER)
    }
}

async fn insert_live_worker(pool: &WorkerPool, root: &ScratchRoot, id: &str) {
    save_registry_entry_in(root, &row(id, RegistryStatus::Running));
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

/// The watch view of a live worker: the step's own last-step time, plus the
/// command-in-flight mark exactly as `watch_snapshot` publishes it.
fn watch_view(progress: &mini_swe_mcp::pool::WorkerProgress, now: u64) -> serde_json::Value {
    json!({
        "worker_id": "c",
        "status": "running",
        "step": progress.step,
        "last_step_at": now.saturating_sub(1800),
        "command_started_at": progress.command_started_at,
    })
}

fn round_snapshot(id: &str, progress: &mini_swe_mcp::pool::WorkerProgress, now: u64) -> Snapshot {
    let mut view = watch_view(progress, now);
    view["worker_id"] = json!(id);
    view["group"] = json!(GROUP);
    BTreeMap::from([(id.to_string(), view)])
}

fn selection(id: &str) -> (BTreeSet<String>, BTreeSet<String>) {
    (
        BTreeSet::from([id.to_string()]),
        BTreeSet::from([GROUP.to_string()]),
    )
}

/// The wait must stamp its command start once: a wait that keeps republishing
/// the mark resets the start time on every poll, so `status` reports
/// "running for 0s" for a wait that has been going for half an hour.
#[test]
fn a_long_wait_reports_its_real_wait_time() {
    let harness = IsolatedPool::new(4, "wait-clock");
    let worker = format!("w1-{}", unique_suffix("w"));
    let consolidator = format!("consol-{}", unique_suffix("c"));
    let root = harness.root();
    let pool = harness.pool.clone();
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("runtime");
    rt.block_on(async move {
        insert_live_worker(&pool, &root, &worker).await;
        insert_live_worker(&pool, &root, &consolidator).await;
        let wait_pool = pool.clone();
        let wait_meta = consolidator_meta(&consolidator);
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
        assert!(
            first
                .last_command
                .as_deref()
                .is_some_and(|c| c.starts_with("CONSOLIDATE_WAIT")),
            "the wait must name itself as the command in flight: {:?}",
            first.last_command
        );
        // The wait stays in flight for a while; the start time must not move,
        // so the elapsed time the status view reports only ever grows.
        tokio::time::sleep(std::time::Duration::from_millis(1_100)).await;
        let later = pool.worker_progress(&consolidator).await.unwrap();
        assert_eq!(
            later.command_started_at, first.command_started_at,
            "a wait must stamp its command start once, not on every poll"
        );
        // The `status` view turns that one stamp into a growing clock.
        let now = mini_swe_mcp::pool::unix_timestamp();
        let elapsed = now.saturating_sub(later.command_started_at.unwrap());
        assert!(elapsed >= 1, "the wait must report a growing elapsed: {elapsed}");
        wait.await.expect("wait task");
        // The mark is gone once the wait returns: ordinary detection resumes.
        let done = pool.worker_progress(&consolidator).await.unwrap();
        assert!(
            done.command_started_at.is_none(),
            "the mark must clear when the wait returns"
        );
    });
}

/// A wait in flight is work, not inactivity: a consolidator that has been
/// waiting for half an hour produces no stall in either mode, and ordinary
/// stall detection resumes the moment the wait returns.
#[test]
fn a_waiting_consolidator_never_reads_as_stalled() {
    let harness = IsolatedPool::new(4, "wait-nostall");
    let worker = format!("w1-{}", unique_suffix("w"));
    let consolidator = format!("consol-{}", unique_suffix("c"));
    let root = harness.root();
    let pool = harness.pool.clone();
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("runtime");
    rt.block_on(async move {
        insert_live_worker(&pool, &root, &worker).await;
        insert_live_worker(&pool, &root, &consolidator).await;
        // A progress snapshot taken while the wait is in flight, held so the
        // two stall rules see the same view after the wait has returned.
        let wait_pool = pool.clone();
        let wait_meta = consolidator_meta(&consolidator);
        let wait_worker = worker.clone();
        let wait = tokio::spawn(async move {
            wait_pool
                .consolidate_wait(&wait_meta, &[wait_worker], Some(1))
                .await
        });
        let waiting = loop {
            let progress = pool.worker_progress(&consolidator).await.unwrap();
            if progress.command_started_at.is_some() {
                break progress;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        };
        let now = mini_swe_mcp::pool::unix_timestamp();
        // Past both thresholds: 1800 s of no step is a plain stall and past
        // the 1200 s `--all` round threshold as well.
        assert!(
            select_event(&watch_view(&waiting, now), None, now + 1800).is_none(),
            "a worker with a command in flight must not stall in plain mode"
        );
        let (ids, groups) = selection(&consolidator);
        assert!(
            round_event(
                &round_snapshot(&consolidator, &waiting, now),
                &ids,
                &groups,
                now + ROUND_STALL_SECS + 600,
                |_| true,
                |_| true,
            )
            .is_none(),
            "a worker with a command in flight must not stall its --all round"
        );
        wait.await.expect("wait task");
        // After the wait, the same step with the mark gone stalls again.
        let idle = pool.worker_progress(&consolidator).await.unwrap();
        assert!(
            select_event(&watch_view(&idle, now), None, now + 1800).is_some(),
            "ordinary stall detection must resume after the wait"
        );
        let event = round_event(
            &round_snapshot(&consolidator, &idle, now),
            &ids,
            &groups,
            now + ROUND_STALL_SECS + 600,
            |_| true,
            |_| true,
        )
        .expect("an idle step after the wait must stall its --all round");
        assert_eq!(event["workers"][0]["outcome"], "stalled");
    });
}
