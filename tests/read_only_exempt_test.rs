//! The read-only escalation, scoped to the workers it is a guard against.
//!
//! The detector exists for an *implementer* that reads instead of writing, so
//! a round consolidator and the review phases are exempt: reviewing, merging
//! (`CONSOLIDATE_MERGE`), steering, waiting (`CONSOLIDATE_WAIT`) and running
//! the gate are their job, and a long read-only streak there is the work
//! itself. Independently of the role, a turn the harness answered itself -- a
//! consolidator verb, a background-job wait -- is progress the worktree sample
//! cannot see, so it never counts towards the streak.
//!
//! The tests run against the thresholds the pool ships (15, 30 and 45 read-only
//! turns), which the code under test reads from the environment at dispatch
//! time. They therefore *set* nothing and *restore* nothing: a test that
//! lowered the thresholds would mutate process-global state its parallel
//! siblings could observe. The price is a turn budget sized to reach past the
//! pause threshold.

mod common;

use std::path::Path;
use std::time::Duration;

use mini_swe_mcp::pool::{WorkerPool, WorkerRole, WorkerState};
use mini_swe_mcp::worktree::ScratchRoot;

/// Owner recorded for the workers this test dispatches: the escalation is what
/// is under test, not the per-agent ownership check.
const TEST_OWNER: &str = "test-agent";

/// The read-only turn the escalation's last step fires on: the default pause
/// threshold is three times the default nudge threshold of 15.
const PAUSE_TURN: usize = 45;

/// A consolidator dispatch that names a file, so the exemption is what is
/// being measured: without it the detector is armed and would fire.
const TASK: &str = "Review src/lib.rs and integrate the group's branches.";

/// One read-only turn for a consolidator: it reads the diff, which is its job
/// and changes nothing in the worktree.
fn review_turn(n: usize) -> &'static str {
    REVIEW_TURNS[n % REVIEW_TURNS.len()]
}

/// Six distinct reads, so the repetition detector never answers a turn in
/// place of the escalation.
const REVIEW_TURNS: [&str; 6] = [
    "sed -n '1,40p' src/lib.rs",
    "git -C . log --oneline -n 5",
    "git -C . diff --stat HEAD~1",
    "grep -rn 'pub fn' src/lib.rs | head -20",
    "cat AGENTS.md | head -30",
    "ls -la src",
];

/// Wait until `id` is parked on the orchestrator, or `None` when it reached a
/// terminal state or kept running without ever parking. Polls the pool rather
/// than sleeping a fixed time, so a slow machine does not make the test flaky.
async fn wait_for_paused(pool: &WorkerPool, id: &str) -> Option<String> {
    for _ in 0..900 {
        match pool.get_worker_state(id).await {
            Some(WorkerState::Paused { question, .. }) => return Some(question),
            Some(WorkerState::Completed { .. } | WorkerState::Failed { .. }) => return None,
            _ => {}
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    None
}

/// The metrics of a finished worker, whatever terminal state it reached.
async fn wait_for_terminal_metrics(pool: &WorkerPool, id: &str) -> mini_swe_mcp::pool::WorkerMetrics {
    let state = tokio::time::timeout(Duration::from_secs(120), async {
        let mut changes = pool.subscribe_changes();
        loop {
            if let Some(state) = pool.get_worker_state(id).await
                && matches!(
                    state,
                    WorkerState::Completed { .. }
                        | WorkerState::Failed { .. }
                        | WorkerState::Exhausted { .. }
                )
            {
                return state;
            }
            changes.changed().await.expect("pool notification");
        }
    })
    .await
    .unwrap_or_else(|_| panic!("worker {id} did not reach a terminal state"));
    match state {
        WorkerState::Completed { metrics, .. }
        | WorkerState::Failed { metrics, .. }
        | WorkerState::Exhausted { metrics, .. } => metrics,
        other => panic!("expected a terminal worker state, got {other:?}"),
    }
}

/// A throwaway git repository with a seeded commit, in a temporary location.
fn repo(tag: &str) -> common::TempDir {
    let repo = common::TempDir::new_in_tmp(tag);
    common::git(repo.path(), &["init", "-b", "master"]);
    common::git(repo.path(), &["config", "user.name", "test"]);
    common::git(repo.path(), &["config", "user.email", "test@localhost"]);
    std::fs::create_dir_all(repo.path().join("src")).expect("seed src directory");
    std::fs::write(repo.path().join("src/lib.rs"), "pub fn seed() {}\n").expect("seed file");
    std::fs::write(repo.path().join("README.md"), "# scratch\n").expect("seed file");
    common::git(repo.path(), &["add", "."]);
    common::git(repo.path(), &["commit", "-m", "baseline"]);
    repo
}

async fn dispatch(pool: &WorkerPool, repo: &Path, role: WorkerRole, max_turns: usize) -> String {
    pool.dispatch_with_role(
        TEST_OWNER.to_string(),
        TASK.to_string(),
        "test-model".to_string(),
        None,
        repo.to_path_buf(),
        max_turns,
        Some("round".to_string()),
        None,
        false,
        None,
        Vec::new(),
        role,
    )
    .await
    .expect("dispatch the worker")
}

async fn pool_for(tag: &str, base_url: &str) -> (WorkerPool, common::TempDir) {
    let scratch = common::TempDir::new_in_tmp(tag);
    let pool = WorkerPool::with_scratch(
        1,
        base_url.to_string(),
        "test-key".to_string(),
        ScratchRoot::new(scratch.path()),
    );
    (pool, scratch)
}

/// A consolidator that reviews the group's work for longer than the pause
/// threshold, and then blocks on the group, is never parked: its reads and its
/// waits are the job, and both are progress the detector must not read as a
/// worker stuck exploring.
#[tokio::test]
async fn a_consolidator_that_only_reviews_and_waits_is_never_paused() {
    let repo = repo("read-only-exempt-consolidator");
    let mut commands: Vec<&str> = (0..PAUSE_TURN + 10)
        .map(|n| review_turn(n))
        .collect();
    // The closing wait: a `CONSOLIDATE_WAIT` names workers that never existed
    // in this test, so it is refused immediately and answers the turn without
    // blocking on anything.
    commands.push("echo \"CONSOLIDATE_WAIT nobody timeout=1\"");
    let llm = common::fake_llm::FakeLlm::spawn_scripted(&commands).await;
    let (pool, _scratch) = pool_for("read-only-exempt-pool", llm.base_url()).await;
    let worker_id = dispatch(&pool, repo.path(), WorkerRole::Consolidate, PAUSE_TURN + 20).await;

    let metrics = wait_for_terminal_metrics(&pool, &worker_id).await;
    assert_eq!(
        metrics.loop_pauses, 0,
        "a consolidator must never be parked for reviewing or waiting, got {metrics:?}"
    );
    assert_eq!(
        metrics.stagnation_nudges, 0,
        "an exempt worker must not be nudged to edit either, got {metrics:?}"
    );

    // The wait was answered by the harness and recorded as a turn, so the
    // script really did reach the point where the streak would have fired.
    let bodies = llm.request_bodies().await;
    assert!(
        bodies.len() as usize >= PAUSE_TURN,
        "the script must have run past the pause threshold, got {} turns",
        bodies.len()
    );
}

/// The same run, as an ordinary worker, is still escalated: the exemption is
/// the role, not a general loosening of the guard.
#[tokio::test]
async fn an_ordinary_worker_is_still_paused_by_the_same_run() {
    let repo = repo("read-only-exempt-worker");
    let commands: Vec<&str> = (0..PAUSE_TURN).map(review_turn).collect();
    let llm = common::fake_llm::FakeLlm::spawn_scripted(&commands).await;
    let (pool, _scratch) = pool_for("read-only-exempt-worker-pool", llm.base_url()).await;
    let worker_id = dispatch(&pool, repo.path(), WorkerRole::Worker, PAUSE_TURN + 10).await;

    let question = wait_for_paused(&pool, &worker_id)
        .await
        .expect("a worker that only reads must still park on the orchestrator");
    assert!(
        question.contains("read-only turns"),
        "the pause must still name the streak, got {question:?}"
    );
}
