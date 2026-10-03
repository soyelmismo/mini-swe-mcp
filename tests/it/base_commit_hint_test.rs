//! Every worker is told its base commit, because the harness checkpoints.
//!
//! The harness commits the worker's uncommitted work every 20 steps, so a
//! bare `git diff` after the first checkpoint is empty. A model that reads it
//! as "my edits are gone" spends its remaining turns re-checking and
//! re-applying them (worker affab410). Three surfaces have to name the base:
//!
//! * the opening message, with the exact `git diff <base>` to run,
//! * the system prompt's rule 13, for the same reason,
//! * the checkpoint notice itself, which is the turn the model sees its
//!   working tree emptied on.

use crate::common;
use mini_swe_mcp::agent::SYSTEM_PROMPT;
use mini_swe_mcp::pool::{WorkerPool, WorkerState, opening_task_message};
use mini_swe_mcp::worktree::ScratchRoot;
use std::path::Path;
use std::time::Duration;

/// Owner recorded for the worker dispatched here: what is under test is the
/// wording it is told, not the per-agent ownership check.
const TEST_OWNER: &str = "test-agent";

/// The checkpoint interval the engine commits on: turn 20 of a run.
const CHECKPOINT_TURN: usize = 20;

/// One past the checkpoint, so the notice the commit pushed is in the request
/// body the scripted turn for that step receives.
const AFTER_CHECKPOINT: usize = 22;

const BASE: &str = "0123456789abcdef0123456789abcdef01234567";

/// The opening message names the base commit and the one command that shows
/// the worker's whole change set.
#[test]
fn the_opening_message_names_the_base_commit_and_how_to_diff_against_it() {
    let message = opening_task_message("do the thing", None, BASE, Path::new("/nonexistent"));
    assert!(
        message.contains(&format!("git diff {BASE}")),
        "the opening message must name the exact diff against the base, got:\n{message}"
    );
    assert!(
        message.contains(BASE),
        "the opening message must carry the base commit sha, got:\n{message}"
    );
    assert!(
        message.contains("checkpoint"),
        "the opening message must say the harness checkpoints the work, got:\n{message}"
    );
}

/// A dispatch whose base commit could not be read names no base: a hint built
/// from an empty sha would send the worker to `git diff ` .
#[test]
fn the_opening_message_omits_the_base_hint_without_a_base_commit() {
    let message = opening_task_message("do the thing", None, "  ", Path::new("/nonexistent"));
    assert!(
        !message.contains("git diff"),
        "no base commit means no diff hint, got:\n{message}"
    );
}

/// Rule 13 is where a worker reads that the harness commits for it, so it has
/// to say a checkpoint lands mid-run and that `git diff <base>` is the whole
/// change set.
#[test]
fn rule_13_explains_the_checkpoints_and_the_full_diff() {
    let rule = SYSTEM_PROMPT
        .lines()
        .find(|line| line.starts_with("13."))
        .expect("the system prompt must keep rule 13");
    assert!(
        rule.contains("checkpoint"),
        "rule 13 must name the checkpoints, got:\n{rule}"
    );
    assert!(
        rule.contains("git diff <base commit>"),
        "rule 13 must point at the full-change-set diff, got:\n{rule}"
    );
}

/// The base commit the pool recorded for this worker: the first line of its
/// conversation log, which is the [`WorkerHistory`] metadata line.
fn recorded_base_commit(root: &ScratchRoot, worker_id: &str) -> String {
    let path = mini_swe_mcp::pool::history_log_path_in(root, worker_id);
    let log =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let meta: mini_swe_mcp::pool::WorkerHistory =
        serde_json::from_str(log.lines().next().expect("a metadata line"))
            .expect("parse the metadata line");
    assert!(
        !meta.base_commit.is_empty(),
        "a dispatched worker always records a base commit"
    );
    meta.base_commit
}

/// The checkpoint turn itself tells the model what happened and how to see
/// everything it changed, because that is the turn where `git diff` goes empty.
#[tokio::test]
async fn the_checkpoint_turn_tells_the_model_its_full_change_set() {
    let repo = common::TempDir::new_in_tmp("checkpoint-notice-repo");
    common::git(repo.path(), &["init", "-b", "master"]);
    common::git(repo.path(), &["config", "user.name", "test"]);
    common::git(repo.path(), &["config", "user.email", "test@localhost"]);
    std::fs::write(repo.path().join("README.md"), "# scratch\n").expect("seed file");
    common::git(repo.path(), &["add", "."]);
    common::git(repo.path(), &["commit", "-m", "baseline"]);

    // Every turn writes a distinct file, so neither the repetition detector nor
    // the stagnation guard fires and the checkpoint at turn 20 has something to
    // commit. The budget runs out instead of a completion sentinel, so the
    // scripted turns (which carry no REPORT block) end the worker in
    // `Exhausted` -- a terminal state this test can still read.
    let commands: Vec<String> = (1..=AFTER_CHECKPOINT)
        .map(|turn| format!("echo turn {turn} > note-{turn}.txt"))
        .collect();
    // The budget is the run's end, but the engine asks for one more turn after
    // it, so the script runs past it rather than running dry mid-poll.
    let commands: Vec<String> = commands
        .into_iter()
        .cycle()
        .take(AFTER_CHECKPOINT + 8)
        .collect();
    let scripted: Vec<&str> = commands.iter().map(String::as_str).collect();
    let llm = common::fake_llm::FakeLlm::spawn_scripted(&scripted).await;

    let scratch = common::TempDir::new_in_tmp("checkpoint-notice-pool");
    let pool = WorkerPool::with_scratch(
        1,
        llm.base_url().to_string(),
        "test-key".to_string(),
        ScratchRoot::new(scratch.path()),
    );
    let worker_id = pool
        .dispatch(
            TEST_OWNER.to_string(),
            "exercise the checkpoint notice".to_string(),
            "test-model".to_string(),
            None,
            repo.path().to_path_buf(),
            AFTER_CHECKPOINT,
            Some("checkpoint-notice".to_string()),
            None,
            false,
            None,
            Vec::new(),
        )
        .await
        .expect("dispatch the worker");

    let mut last_state = None;
    let state = tokio::time::timeout(Duration::from_secs(120), async {
        let mut changes = pool.subscribe_changes();
        loop {
            if let Some(state) = pool.get_worker_state(&worker_id).await {
                if matches!(
                    state,
                    WorkerState::Completed { .. }
                        | WorkerState::Failed { .. }
                        | WorkerState::Exhausted { .. }
                ) {
                    return state;
                }
                last_state = Some(format!("{state:?}"));
            }
            changes.changed().await.expect("pool notification");
        }
    })
    .await
    .unwrap_or_else(|_| {
        panic!(
            "worker {worker_id} did not reach a terminal state after {} requests; last state {:?}",
            llm.requests(),
            last_state
        )
    });
    assert!(
        matches!(state, WorkerState::Exhausted { .. }),
        "the scripted run must end on its own budget, got {state:?} after {} requests",
        llm.requests()
    );

    // The base the run was actually dispatched against, read from the log the
    // pool wrote rather than assumed from the seed repository.
    let base = recorded_base_commit(&ScratchRoot::new(scratch.path()), &worker_id);
    // The conversation the model saw: the opening message plus every notice.
    let bodies = llm.request_bodies().await;
    assert!(
        bodies.len() as usize > CHECKPOINT_TURN,
        "the run must reach past the checkpoint turn, got {} requests",
        bodies.len()
    );
    let last = bodies.last().expect("a captured request").to_string();
    let notice = last
        .split("Checkpoint committed (")
        .nth(1)
        .map(|rest| rest.split('`').next().unwrap_or_default().to_string())
        .unwrap_or_else(|| panic!("the checkpoint notice must reach the model, got:\n{last}"));
    assert!(
        notice.starts_with("19 files); your full change set: "),
        "the notice must name the file count and the full-change-set diff, got:\n{last}"
    );
    assert!(
        last.contains(&format!("git diff {base}")),
        "the notice must diff against the worker's base commit {base}, got:\n{last}"
    );
}
