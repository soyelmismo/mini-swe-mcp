//! The read-only escalation, scoped to the workers it is a guard against.
//!
//! The detector exists for an *implementer* that reads instead of writing, so
//! a round consolidator and the review phases are exempt: reviewing, merging
//! (`CONSOLIDATE_MERGE`), steering, waiting (`CONSOLIDATE_WAIT`) and running
//! the gate are their job, and a long read-only streak there is the work
//! itself. Independently of the role, a turn the harness answered itself -- a
//! consolidator verb, a background-job wait -- is progress the worktree sample
//! cannot see, but only when its answer changed: a wait that reports the same
//! states again taught the worker nothing, so it does not restart the streak
//! (and the loop detector in `loop_escalation_test` parks it instead).
//!
//! The tests run against the thresholds the pool ships (15, 30 and 45 read-only
//! turns), which the code under test reads from the environment at dispatch
//! time. They therefore *set* nothing and *restore* nothing: a test that
//! lowered the thresholds would mutate process-global state its parallel
//! siblings could observe. The price is a turn budget sized to reach past the
//! pause threshold.

use crate::common;
use mini_swe_mcp::pool::{WorkerPool, WorkerRole, WorkerState};
use mini_swe_mcp::worktree::ScratchRoot;
use std::path::Path;
use std::time::Duration;

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
async fn wait_for_terminal_metrics(
    pool: &WorkerPool,
    id: &str,
) -> mini_swe_mcp::pool::WorkerMetrics {
    let state = tokio::time::timeout(Duration::from_secs(90), async {
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
    let mut commands: Vec<&str> = (0..PAUSE_TURN + 10).map(review_turn).collect();
    // The closing wait: a `CONSOLIDATE_WAIT` names workers that never existed
    // in this test, so it is refused immediately and answers the turn without
    // blocking on anything. The budget ends on that same turn, so the run
    // stops instead of pausing on a socket that has run out of script.
    commands.push("echo \"CONSOLIDATE_WAIT nobody timeout=1\"");
    let llm = common::fake_llm::FakeLlm::spawn_scripted(&commands).await;
    let (pool, _scratch) = pool_for("read-only-exempt-pool", llm.base_url()).await;
    let worker_id = dispatch(&pool, repo.path(), WorkerRole::Consolidate, commands.len()).await;

    let metrics = wait_for_terminal_metrics(&pool, &worker_id).await;
    assert_eq!(
        metrics.loop_pauses, 0,
        "a consolidator must never be parked for reviewing or waiting, got {metrics:?}"
    );

    // The script really did run past the point where the streak would have
    // fired, and the closing wait was answered by the harness as a turn.
    let bodies = llm.request_bodies().await;
    assert!(
        bodies.len() as usize >= PAUSE_TURN,
        "the script must have run past the pause threshold, got {} turns",
        bodies.len()
    );
    // None of the three read-only steps reached the model: neither the demand,
    // nor the plan, nor the pause. The stagnation detector is a different
    // guard and keeps its own timing, so its nudges are not what is asserted
    // here.
    for (turn, body) in bodies.iter().enumerate() {
        let told = body.to_string();
        assert!(
            !told.contains("read-only turns") && !told.contains("write the first edit now"),
            "turn {turn} must not carry the read-only escalation, got {told}"
        );
    }
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

/// A turn spent waiting on a background job is progress, not another read-only
/// turn, whatever role the worker has: an ordinary implementer that alternates
/// its reads with `WAIT_JOB` -- the sanctioned alternative to sleep-polling --
/// never reaches the escalation, though every one of those turns leaves the
/// worktree unchanged.
///
/// The waits here report a *different* job each turn, which is what makes each
/// answer new information. A wait that comes back with the states it already
/// reported is not progress, and `loop_escalation_test` covers that side.
#[tokio::test]
async fn a_wait_job_turn_whose_answer_changed_is_progress() {
    let repo = repo("read-only-exempt-wait-job");
    // Reads interleaved with waits, one more turn than the pause threshold:
    // every turn here leaves the worktree exactly as it found it.
    let commands: Vec<String> = (0..PAUSE_TURN + 5)
        .flat_map(|n| {
            [
                review_turn(n).to_string(),
                format!("echo WAIT_JOB {}", n + 1),
            ]
        })
        .collect();
    let scripted: Vec<&str> = commands.iter().map(String::as_str).collect();
    let llm = common::fake_llm::FakeLlm::spawn_scripted(&scripted).await;
    let (pool, _scratch) = pool_for("read-only-exempt-wait-pool", llm.base_url()).await;
    let worker_id = dispatch(&pool, repo.path(), WorkerRole::Worker, commands.len()).await;

    let paused = wait_for_paused(&pool, &worker_id).await;
    assert!(
        paused.is_none(),
        "a worker waiting on its jobs must never be parked, got {paused:?}"
    );
    let metrics = wait_for_terminal_metrics(&pool, &worker_id).await;
    assert_eq!(
        metrics.loop_pauses, 0,
        "the waits must have kept the streak from escalating, got {metrics:?}"
    );
}
