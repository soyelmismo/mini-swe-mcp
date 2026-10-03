//! The equivalent-command loop detector, for every role.
//!
//! A worker that re-runs one command spelled several ways -- `cargo test 2>&1
//! | grep ...`, `cargo test 2>&1 | tail`, `cargo fmt --check && ... && cargo
//! test`, `cargo test 2>&1 > /tmp/out` -- with an unchanged worktree and an
//! unchanged answer is stuck, and the byte-identical repetition detector
//! cannot see it. The detector in `pool::runner::turn` normalizes each command
//! to its base, counts the runs inside a sliding window, nudges the worker,
//! and parks it on the orchestrator when the loop survives the nudge.
//!
//! The escalation is deliberately not scoped to the implementer: a round
//! consolidator and the review phases reach it too, because the read-only
//! exemption is about *reading*, not about re-running one command with nothing
//! to show for it. The tests below drive a consolidator and an ordinary worker
//! through the same loop and assert the orchestrator question each one gets.

use crate::common;
use std::path::Path;
use std::time::Duration;
use mini_swe_mcp::pool::{WorkerPool, WorkerRole, WorkerState};
use mini_swe_mcp::worktree::ScratchRoot;

/// Owner recorded for the workers this test dispatches: the loop is what is
/// under test, not the per-agent ownership check.
const TEST_OWNER: &str = "test-agent";

/// A dispatch that names a file, so the read-only detector is armed for the
/// worker role: the loop detector has to fire first, on its own evidence.
const TASK: &str = "Review src/lib.rs and integrate the group's branches.";

/// Six spellings of one command that all exit 1 with an empty output, so the
/// only thing that changes between turns is how the command is written.
///
/// `sh -c 'exit 1'` is used rather than a test runner: the detector must fire
/// on any command, and a scratch repository has no toolchain to run. The
/// redirections are chosen to keep the exit code and the output identical --
/// a pipe would report the filter's status instead.
const LOOP_TURNS: [&str; 6] = [
    "sh -c 'exit 1'",
    "sh -c 'exit 1' 2>&1",
    "sh -c 'exit 1' 2>&1 > /dev/null",
    "cd . && sh -c 'exit 1' 2>&1",
    "RUST_BACKTRACE=1 sh -c 'exit 1'",
    "sh -c 'exit 1' 2>&1 > /dev/null",
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

/// A throwaway git repository with a seeded commit, in a temporary location.
fn repo(tag: &str) -> common::TempDir {
    let repo = common::TempDir::new_in_tmp(tag);
    common::git(repo.path(), &["init", "-b", "master"]);
    common::git(repo.path(), &["config", "user.name", "test"]);
    common::git(repo.path(), &["config", "user.email", "test@localhost"]);
    std::fs::create_dir_all(repo.path().join("src")).expect("seed src directory");
    std::fs::write(repo.path().join("src/lib.rs"), "pub fn seed() {}\n").expect("seed file");
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

/// A consolidator that re-runs one command in six spellings is parked with the
/// loop summary: the read-only exemption does not cover this, and the older
/// guards -- a byte-identical repetition check and nudges that never escalate
/// for a consolidator -- let it run.
#[tokio::test]
async fn a_consolidator_looping_on_one_command_is_paused_with_the_summary() {
    let repo = repo("loop-escalation-consolidator");
    let llm = common::fake_llm::FakeLlm::spawn_scripted(&LOOP_TURNS).await;
    let (pool, _scratch) = pool_for("loop-escalation-pool", llm.base_url()).await;
    let worker_id = dispatch(
        &pool,
        repo.path(),
        WorkerRole::Consolidate,
        LOOP_TURNS.len() + 5,
    )
    .await;

    let question = wait_for_paused(&pool, &worker_id)
        .await
        .expect("a consolidator looping on one command must be parked on the orchestrator");
    assert!(
        question.contains("is stuck in a loop"),
        "the pause must be the loop question, got {question:?}"
    );
    assert!(
        question.contains("`sh -c 'exit 1'`"),
        "the question must name the base command, got {question:?}"
    );
    assert!(
        question.contains("6 times"),
        "the question must say how often it ran, got {question:?}"
    );
    assert!(
        question.contains("exit code 1"),
        "the question must carry the last exit code, got {question:?}"
    );

    // The nudge reached the model before the pause: the request that followed
    // the fourth run carries it.
    let bodies = llm.request_bodies().await;
    assert!(
        bodies
            .iter()
            .any(|body| body.to_string().contains("Loop detected")),
        "the worker must be nudged before it is parked"
    );
}

/// The same loop, run by an ordinary worker: the escalation is not a
/// consolidator-only guard either.
#[tokio::test]
async fn an_ordinary_worker_looping_on_one_command_is_paused_too() {
    let repo = repo("loop-escalation-worker");
    let llm = common::fake_llm::FakeLlm::spawn_scripted(&LOOP_TURNS).await;
    let (pool, _scratch) = pool_for("loop-escalation-worker-pool", llm.base_url()).await;
    let worker_id = dispatch(&pool, repo.path(), WorkerRole::Worker, LOOP_TURNS.len() + 5).await;

    let question = wait_for_paused(&pool, &worker_id)
        .await
        .expect("a worker looping on one command must be parked on the orchestrator");
    assert!(
        question.contains("is stuck in a loop"),
        "the pause must be the loop question, got {question:?}"
    );
}

/// A wait that comes back with the states it already reported is not progress:
/// the loop detector counts it, and the worker is parked instead of being
/// allowed to poll a job that never changes.
#[tokio::test]
async fn a_wait_that_returns_the_same_states_again_is_not_progress() {
    let repo = repo("loop-escalation-wait");
    // The same `WAIT_JOB` for a job that never existed: the harness answers
    // every one of them with the same refusal, and the worktree never moves.
    let commands: Vec<&str> = vec!["echo WAIT_JOB 1"; 8];
    let llm = common::fake_llm::FakeLlm::spawn_scripted(&commands).await;
    let (pool, _scratch) = pool_for("loop-escalation-wait-pool", llm.base_url()).await;
    let worker_id = dispatch(&pool, repo.path(), WorkerRole::Worker, commands.len() + 5).await;

    let question = wait_for_paused(&pool, &worker_id)
        .await
        .expect("a wait that reports the same states again must not count as progress");
    assert!(
        question.contains("is stuck in a loop"),
        "the pause must be the loop question, got {question:?}"
    );
    assert!(
        question.contains("`echo WAIT_JOB 1`"),
        "the question must name the base command, got {question:?}"
    );
}
