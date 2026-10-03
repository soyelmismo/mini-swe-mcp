//! An interrupted worker stays continuable even when it never committed.
//!
//! A hub shutdown interrupts every live worker. When a worker had only read
//! code -- five turns, no file touched -- there is nothing for
//! `WorkerPool::kill_all`'s checkpoint to commit, so its `worker-<id>` branch
//! still points at the base commit and looks exactly like a branch with no
//! work. Teardown deleted that branch, and the next daemon then refused to
//! auto-continue the worker (`Worker branch ... no longer exists`), leaving a
//! row and a conversation with nothing to run on.
//!
//! Two properties are pinned here:
//!
//! * the branch of an interrupted worker survives teardown whatever it points
//!   at, so the next daemon can auto-continue it, and
//! * the orphan sweep does not treat an Interrupted worker's history as
//!   orphaned, because that conversation is what the continuation replays.

use crate::common;
use mini_swe_mcp::pool::{
    LogBuffer, RegistryStatus, WorkerHistory, WorkerMeta, WorkerMetrics, WorkerPool, WorkerRecord,
    WorkerRole, WorkerState, append_history_message_in, history_log_path_in,
    load_registry_entry_in, prune_orphan_histories_with_retention_and_grace_in,
    save_registry_entry_in,
};
use mini_swe_mcp::worktree::{ScratchRoot, WorktreeGuard};
use std::path::{Path, PathBuf};

const OWNER: &str = "interrupted-no-commit-test";

/// A repository with one commit on `master`, and its head sha.
fn seed_repo(scratch: &common::TempDir) -> (PathBuf, String) {
    let repo = scratch.subdir("repo");
    common::git(&repo, &["init", "-b", "master"]);
    common::git(&repo, &["config", "user.name", "t"]);
    common::git(&repo, &["config", "user.email", "t@t"]);
    std::fs::write(repo.join("base.txt"), "base\n").expect("seed the repo");
    common::git(&repo, &["add", "base.txt"]);
    common::git(&repo, &["commit", "-m", "seed"]);
    let head = common::git(&repo, &["rev-parse", "HEAD"])
        .trim()
        .to_string();
    (repo, head)
}

/// A live (running) synthetic record, without an LLM behind it.
fn running_worker(id: &str) -> WorkerRecord {
    WorkerRecord {
        id: id.to_string(),
        task: "read the code and report".to_string(),
        model: "test-model".to_string(),
        owner: OWNER.to_string(),
        state: WorkerState::Running {
            step: 6,
            last_command: "grep -rn".to_string(),
            started_at: 0,
        },
        metrics: WorkerMetrics::default(),
        logs: LogBuffer::new(),
        pending_steer: Vec::new(),
        resume_tx: None,
        handle: None,
        revision: 0,
    }
}

fn meta(id: &str, repo: &Path) -> WorkerMeta {
    WorkerMeta {
        task: "read the code and report".to_string(),
        repo_path: Some(repo.to_string_lossy().into_owned()),
        ..WorkerMeta::test_meta(id, OWNER)
    }
}

/// The conversation a worker that only read code leaves behind: the opening
/// pair plus the turns it did before the hub stopped it.
fn write_history(root: &ScratchRoot, id: &str, repo: &Path, base_commit: &str) {
    use mini_swe_mcp::agent::{ChatMessage, Role};
    let history = WorkerHistory {
        task: "read the code and report".to_string(),
        group: None,
        role: WorkerRole::Worker,
        model: "test-model".to_string(),
        temperature: None,
        repo_path: repo.to_string_lossy().to_string(),
        base_commit: base_commit.to_string(),
        base_branch: Some("master".to_string()),
        branch: format!("worker-{id}"),
        network_offline: false,
        verify: None,
        client_env: Vec::new(),
        max_turns: 10,
        review_after: None,
        revision: 0,
        auto_continues: 0,
        owner: Some(OWNER.to_string()),
        messages: vec![
            ChatMessage::text(Role::System, "system prompt"),
            ChatMessage::text(Role::User, "TASK:\nread the code and report"),
            ChatMessage::text(Role::Assistant, "I read the parser; here is what it does."),
        ],
    };
    for msg in &history.messages {
        append_history_message_in(root, id, &history, msg).expect("append the conversation");
    }
}

/// A worker whose branch has no commit beyond the base keeps that branch.
///
/// This is the F4 regression itself: `kill_all` checkpoints what the worker
/// left uncommitted, and a worker that changed nothing has nothing to
/// checkpoint, so the branch only ever pointed at the base commit. Teardown
/// used to delete it as a branch with no work, which left the Interrupted row
/// and its conversation un-continuable across the restart.
#[test]
fn an_interrupted_worker_with_no_commit_keeps_its_branch() {
    let scratch = common::TempDir::new_in_tmp("interrupted-no-commit");
    let root = ScratchRoot::new(scratch.path());
    let (repo, head) = seed_repo(&scratch);
    let id = "reader1";

    // A worker that read code and changed nothing: its branch is cut at the
    // base commit and never moved.
    let guard = WorktreeGuard::new_in(&root, &repo, id).expect("worktree");
    assert_eq!(guard.base_commit, head, "the branch is cut at HEAD");
    let branch = guard.branch.clone();
    let checkout = guard.path.clone();
    assert!(
        common::git_ref_exists(&repo, &branch),
        "the branch exists while the worker runs"
    );

    // The hub interrupts it: nothing is committed, so the branch still sits on
    // the base commit and its teardown must not delete it.
    WorktreeGuard::mark_interrupted(&checkout);
    drop(guard);

    assert!(!checkout.exists(), "the worktree itself is still reclaimed");
    assert!(
        common::git_ref_exists(&repo, &branch),
        "an interrupted worker's branch survives teardown even with no commits"
    );
    assert_eq!(
        common::git(&repo, &["rev-parse", &branch]).trim(),
        head,
        "the branch still points at the base commit"
    );

    // The next daemon auto-continues on that branch, which is only possible
    // because the ref is still there.
    let reattached = WorktreeGuard::reopen_in(&root, &repo, id, &head).expect("reopen");
    assert_eq!(reattached.branch, branch);
    drop(reattached);
    let _ = std::fs::remove_dir_all(&repo);
}

/// A worker that ended some other way still gets a commitless branch deleted.
///
/// The interruption is a property of the *shutdown*, not of the worker id: the
/// next run of the same id must clean up after itself as before.
#[test]
fn a_branch_with_no_commits_is_still_deleted_without_an_interruption() {
    let scratch = common::TempDir::new_in_tmp("interrupted-scope");
    let root = ScratchRoot::new(scratch.path());
    let (repo, _head) = seed_repo(&scratch);
    let id = "reader2";

    let guard = WorktreeGuard::new_in(&root, &repo, id).expect("worktree");
    let branch = guard.branch.clone();
    drop(guard);

    assert!(
        !common::git_ref_exists(&repo, &branch),
        "a branch with no commits past the base is still pruned"
    );
    let _ = std::fs::remove_dir_all(&repo);
}

/// The interrupted marker is consumed by the teardown that honours it.
#[test]
fn the_interruption_marker_does_not_outlive_the_teardown() {
    let scratch = common::TempDir::new_in_tmp("interrupted-marker");
    let root = ScratchRoot::new(scratch.path());
    let (repo, _head) = seed_repo(&scratch);
    let id = "reader3";

    let guard = WorktreeGuard::new_in(&root, &repo, id).expect("worktree");
    let checkout = guard.path.clone();
    WorktreeGuard::mark_interrupted(&checkout);
    drop(guard);

    let mut marker = checkout.as_os_str().to_os_string();
    marker.push(".interrupted");
    let marker = PathBuf::from(marker);
    assert!(
        !marker.exists(),
        "the marker must not survive the drop that honoured it"
    );

    // The same id runs again and ends without committing: now its branch is
    // pruned, because nothing marked this run as interrupted.
    let guard = WorktreeGuard::new_in(&root, &repo, id).expect("worktree");
    let branch = guard.branch.clone();
    drop(guard);
    assert!(
        !common::git_ref_exists(&repo, &branch),
        "a later run is not covered by an earlier run's interruption"
    );
    let _ = std::fs::remove_dir_all(&repo);
}

/// The full restart: the shutdown path, then the next daemon's continuation.
#[tokio::test]
async fn a_worker_interrupted_before_any_commit_is_auto_continued_by_the_next_daemon() {
    let scratch = common::TempDir::new_in_tmp("interrupted-restart");
    let root = ScratchRoot::new(scratch.path());
    let pool = WorkerPool::with_scratch(
        4,
        "http://localhost:1".to_string(),
        "k".to_string(),
        root.clone(),
    );
    let (repo, head) = seed_repo(&scratch);
    let id = "reader4";

    // Dispatch the worker: a real worktree, a real registry row, and the
    // conversation its turns left behind. The guard stands in for the one the
    // worker's own task owns, so it is still alive when the shutdown runs.
    let guard = WorktreeGuard::new_in(&root, &repo, id).expect("worktree");
    let branch = guard.branch.clone();
    let checkout = guard.path.clone();
    write_history(&root, id, &repo, &head);
    pool.__test_save_status(
        &meta(id, &repo),
        "test-model",
        RegistryStatus::Running,
        6,
        10,
        "grep -rn",
        None,
    );
    pool.__test_insert_worker(running_worker(id)).await;
    pool.__test_register_worktree(id, checkout.clone()).await;

    // The graceful stop. The checkpoint finds nothing to commit, and the abort
    // that follows drops the guard, which is what reclaims the checkout.
    assert_eq!(pool.kill_all().await, 1, "the live worker is interrupted");
    drop(guard);

    let row = load_registry_entry_in(&root, id).expect("the interrupted row survives");
    assert_eq!(row.status, RegistryStatus::Interrupted);
    assert!(
        common::git_ref_exists(&repo, &branch),
        "the branch of an interrupted worker with no commits must survive"
    );
    assert!(
        pool.interrupted_workers().await.contains(&id.to_string()),
        "the next daemon offers it for auto-continuation"
    );

    // The continuation the next daemon performs: it re-attaches to the branch
    // the shutdown preserved, which is what used to fail with
    // "Worker branch worker-reader4 no longer exists".
    let reattached = WorktreeGuard::reopen_in(
        &root,
        &repo,
        id,
        &row.base_commit.clone().unwrap_or(head.clone()),
    )
    .expect("the interrupted worker must be continuable");
    assert_eq!(reattached.branch, branch);
    drop(reattached);
    let _ = std::fs::remove_dir_all(&repo);
}

/// A pruned branch is recreated from the commit the row recorded.
///
/// The other half of continuability: even if the ref is gone by the time the
/// continuation runs, the row names the commit the worker branched from, so
/// the worker is restarted rather than refused.
#[tokio::test]
async fn a_pruned_branch_is_recreated_from_the_recorded_base_commit() {
    let scratch = common::TempDir::new_in_tmp("interrupted-recreate");
    let root = ScratchRoot::new(scratch.path());
    let pool = WorkerPool::with_scratch(
        4,
        "http://localhost:1".to_string(),
        "k".to_string(),
        root.clone(),
    );
    let (repo, head) = seed_repo(&scratch);
    let id = "reader5";

    let guard = WorktreeGuard::new_in(&root, &repo, id).expect("worktree");
    let branch = guard.branch.clone();
    // The worker's own teardown already removed the commitless branch; make
    // sure of the starting point before pruning one that exists.
    drop(guard);
    common::git(&repo, &["branch", &branch, &head]);
    assert!(common::git_ref_exists(&repo, &branch));
    common::git(&repo, &["branch", "-D", &branch]);
    assert!(!common::git_ref_exists(&repo, &branch));

    // An Interrupted row that recorded the commit it branched from.
    let row = mini_swe_mcp::pool::WorkerRegistryEntry {
        task: "read the code and report".to_string(),
        model: "test-model".to_string(),
        status: RegistryStatus::Interrupted,
        step: 6,
        last_command: "hub stopped; work saved on branch worker-reader5".into(),
        repo_path: Some(repo.to_string_lossy().to_string()),
        owner: Some(OWNER.to_string()),
        base_branch: Some("master".into()),
        base_commit: Some(head.clone()),
        updated_at: mini_swe_mcp::pool::unix_timestamp(),
        ..mini_swe_mcp::pool::WorkerRegistryEntry::test_row(id, OWNER)
    };
    save_registry_entry_in(&root, &row);

    // A continuation on the same id must find a branch to run on.
    let outcome = pool
        .steer(id, "the hub restarted".into())
        .await
        .expect("an interrupted worker with a recorded base commit is continuable");
    assert!(matches!(
        outcome,
        mini_swe_mcp::pool::SteerOutcome::Continuing { .. }
    ));
    assert!(
        common::git_ref_exists(&repo, &branch),
        "the branch must be back so the continuation has a base to work from"
    );
    let _ = std::fs::remove_dir_all(&repo);
}

/// The sweep must not treat an Interrupted worker's history as orphaned.
///
/// Its conversation is exactly what the auto-continuation replays, and its
/// branch points nowhere past the base, so the "branch is gone" rule would
/// retire a worker that is still meant to run.
#[test]
fn the_orphan_sweep_keeps_an_interrupted_workers_history() {
    let scratch = common::TempDir::new_in_tmp("interrupted-sweep");
    let root = ScratchRoot::new(scratch.path());
    let (repo, head) = seed_repo(&scratch);
    let id = "reader6";

    write_history(&root, id, &repo, &head);
    let row = mini_swe_mcp::pool::WorkerRegistryEntry {
        task: "read the code and report".to_string(),
        model: "test-model".to_string(),
        status: RegistryStatus::Interrupted,
        step: 6,
        last_command: "hub stopped; work saved on branch worker-reader6".into(),
        repo_path: Some(repo.to_string_lossy().to_string()),
        owner: Some(OWNER.to_string()),
        base_branch: Some("master".into()),
        base_commit: Some(head.clone()),
        // A real timestamp: `test_row`'s zero would read as "never written",
        // which both age rules treat as still-in-grace, and the sweep would
        // then prove nothing.
        updated_at: mini_swe_mcp::pool::unix_timestamp(),
        ..mini_swe_mcp::pool::WorkerRegistryEntry::test_row(id, OWNER)
    };
    save_registry_entry_in(&root, &row);

    // Retention is zero and the grace is zero: every age-based rule says this
    // worker's history is spent.
    let pruned = prune_orphan_histories_with_retention_and_grace_in(&root, &repo, 0, 0);
    assert_eq!(
        pruned, 0,
        "an interrupted worker's history is not an orphan"
    );
    assert!(
        history_log_path_in(&root, id).exists(),
        "the conversation the continuation replays must survive the sweep"
    );
    assert!(
        load_registry_entry_in(&root, id).is_some(),
        "the row the recovery reads must survive the sweep"
    );

    // A *finished* worker under the same conditions is still swept: the
    // exemption is the interrupted status, not this row's age. Its branch is
    // gone, which is the case the sweep exists for.
    save_registry_entry_in(
        &root,
        &mini_swe_mcp::pool::WorkerRegistryEntry {
            status: RegistryStatus::Completed,
            ..row
        },
    );
    assert!(
        !common::git_ref_exists(&repo, &format!("worker-{id}")),
        "the finished worker has no branch left to protect it"
    );
    assert_eq!(
        prune_orphan_histories_with_retention_and_grace_in(&root, &repo, 0, 0),
        1,
        "a finished worker's expired history is still swept"
    );
    assert!(
        !history_log_path_in(&root, id).exists(),
        "the sweep really did remove it"
    );
    let _ = std::fs::remove_dir_all(&repo);
}
