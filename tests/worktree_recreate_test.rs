//! A worktree that outlives its checkout directory must be recoverable.
//!
//! `git worktree add` refuses a path that is still *registered* even when the
//! directory is gone ("is a missing but already registered worktree"), so a
//! checkout directory deleted without unregistering it -- a cleanup whose
//! `worktree remove` did not run, a filesystem-level delete -- left the next
//! dispatch or revision of the same worker unable to create its worktree at
//! all, and the whole revision failed until an operator ran `git worktree
//! prune` by hand.
//!
//! Two properties are pinned here:
//!
//! * a worktree directory deleted without unregistering is recreated
//!   successfully for the same worker, and
//! * cleanup leaves no registration behind (`git worktree list` no longer
//!   reports the path), so the next revision does not hit the stale row.
//!
//! Every test runs against its own throwaway repository and its own scratch
//! root, so no sweep, registry or another test's checkout is involved.

mod common;

use mini_swe_mcp::worktree::{ScratchRoot, WorktreeGuard};
use std::path::PathBuf;

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

/// Deleting the checkout directory alone, the way a cleanup that never
/// unregistered it does, must not make the next `worktree add` fail.
#[test]
fn a_deleted_worktree_directory_is_recreated_for_the_same_worker() {
    let scratch = common::TempDir::new_in_tmp("wt-recreate");
    let (repo, _head) = seed_repo(&scratch);
    let root = ScratchRoot::new(scratch.subdir("workers"));

    let first = WorktreeGuard::new_in(&root, &repo, "recreate").expect("first dispatch");
    let path = first.path.clone();
    // `forget` skips `Drop`, which is the point: the crash scenario is a
    // checkout whose registration outlived the process that made it.
    std::mem::forget(first);

    // The crash scenario: the directory is gone, but git still has the row.
    std::fs::remove_dir_all(&path).expect("simulate a cleanup that lost the dir");
    assert!(
        common::worktree_is_registered(&repo, &path),
        "fixture must reproduce the stale registration this test recovers from"
    );

    let second = WorktreeGuard::new_in(&root, &repo, "recreate")
        .expect("a worktree whose directory was deleted must be recreated");
    assert!(second.path.is_dir(), "the checkout must be on disk again");
    assert_eq!(second.path, path, "the same worker owns the same path");
}

/// The recovery must not be specific to `new`: a revision re-attaches to the
/// existing branch with `worktree add <path> <branch>`, which is the form git
/// rejects with "is a missing but already registered worktree".
#[test]
fn a_revision_reopens_a_worktree_whose_directory_was_deleted() {
    let scratch = common::TempDir::new_in_tmp("wt-reopen");
    let (repo, head) = seed_repo(&scratch);
    let root = ScratchRoot::new(scratch.subdir("workers"));

    let mut guard = WorktreeGuard::new_in(&root, &repo, "reopen").expect("first dispatch");
    let path = guard.path.clone();
    // A finished run leaves its branch behind with commits on it.
    std::fs::write(path.join("worker.txt"), "done\n").expect("worker writes a file");
    guard
        .commit_changes("worker: finished work")
        .expect("commit the worker change");
    guard.preserve_branch = true;
    std::mem::forget(guard);

    std::fs::remove_dir_all(&path).expect("simulate a cleanup that lost the dir");

    let reopened = WorktreeGuard::reopen_in(&root, &repo, "reopen", &head)
        .expect("a revision must recover from a stale registration");
    assert!(
        std::fs::read_to_string(reopened.path.join("worker.txt")).is_ok(),
        "the revision must re-attach to the preserved work"
    );
}

/// Cleanup must unregister the worktree, not just delete its directory: a
/// surviving registration is what makes the *next* revision fail.
#[test]
fn cleanup_leaves_no_worktree_registration_behind() {
    let scratch = common::TempDir::new_in_tmp("wt-unregister");
    let (repo, _head) = seed_repo(&scratch);
    let root = ScratchRoot::new(scratch.subdir("workers"));

    let guard = WorktreeGuard::new_in(&root, &repo, "unregister").expect("dispatch");
    let path = guard.path.clone();
    assert!(
        common::worktree_is_registered(&repo, &path),
        "a live worktree must be registered"
    );

    drop(guard);

    assert!(
        !common::worktree_is_registered(&repo, &path),
        "cleanup left {} registered; the next revision would fail with \
         'missing but already registered worktree'",
        path.display()
    );
    assert!(!path.exists(), "cleanup must delete the directory too");
}

/// The recovery is scoped to the worker's own path: another worker's live
/// registration in the same repository must survive it untouched.
#[test]
fn recovery_never_unregisters_another_workers_worktree() {
    let scratch = common::TempDir::new_in_tmp("wt-foreign");
    let (repo, head) = seed_repo(&scratch);
    let root = ScratchRoot::new(scratch.subdir("workers"));

    let live = WorktreeGuard::new_in(&root, &repo, "sibling").expect("sibling dispatch");
    let sibling_path = live.path.clone();

    let stale = WorktreeGuard::new_in(&root, &repo, "stale").expect("stale dispatch");
    let stale_path = stale.path.clone();
    std::mem::forget(stale);
    std::fs::remove_dir_all(&stale_path).expect("lose the stale checkout directory");

    // A *revision* of the stale worker is the flow that shares the sibling's
    // repository and hits git's "already registered" refusal, so use it here.
    let recovered = WorktreeGuard::reopen_in(&root, &repo, "stale", &head)
        .expect("recover the stale registration without touching the sibling");

    assert_eq!(recovered.path, stale_path);
    assert!(
        common::worktree_is_registered(&repo, &sibling_path),
        "recovering one worker's worktree unregistered a live sibling"
    );
    drop(recovered);
    drop(live);
}
