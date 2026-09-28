//! Integration tests for [`mini_swe_mcp::worktree::WorktreeGuard`].
//!
//! These tests drive a real git repository: they create worktrees, inspect
//! diffs, and assert that dropping a guard removes both the branch and the
//! directory. Because they mutate the repository they run in, each test uses
//! its own throwaway worker id (and therefore its own worktree/branch).

use mini_swe_mcp::worktree::WorktreeGuard;
use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static COUNTER: AtomicUsize = AtomicUsize::new(0);

fn run(dir: &Path, args: &[&str]) -> String {
    let output = Command::new("git")
        .current_dir(dir)
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("failed to run git {args:?} in {}: {e}", dir.display()));
    assert!(
        output.status.success(),
        "git {args:?} in {} failed: {}",
        dir.display(),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout).to_string()
}

/// Absolute path of the repository the test binary is running in.
fn repo_root() -> PathBuf {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let top = run(&root, &["rev-parse", "--show-toplevel"]);
    PathBuf::from(top.trim())
}

fn unique_worker_id(prefix: &str) -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!(
        "test-{prefix}-{}-{nanos}-{}",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::SeqCst)
    )
}

fn branch_exists(repo: &Path, branch: &str) -> bool {
    Command::new("git")
        .current_dir(repo)
        .args(["rev-parse", "--verify", "--quiet", branch])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn worktree_is_registered(repo: &Path, path: &Path) -> bool {
    let list = run(repo, &["worktree", "list", "--porcelain"]);
    list.lines()
        .any(|line| line.strip_prefix("worktree ") == Some(path.to_str().unwrap_or("")))
}

#[test]
fn concurrent_guards_use_distinct_branches_and_paths() {
    let repo = repo_root();

    let id_a = unique_worker_id("a");
    let id_b = unique_worker_id("b");

    let (branch_a, path_a, branch_b, path_b) = {
        let guard_a = WorktreeGuard::new(&repo, &id_a).expect("first worktree failed");
        let guard_b = WorktreeGuard::new(&repo, &id_b).expect("second worktree failed");

        // Distinct branches and distinct on-disk locations.
        assert_ne!(guard_a.branch, guard_b.branch, "branches must differ");
        assert_ne!(guard_a.path, guard_b.path, "worktree paths must differ");
        assert_eq!(guard_a.branch, format!("worker-{id_a}"));
        assert_eq!(guard_b.branch, format!("worker-{id_b}"));
        assert!(guard_a.path.is_dir(), "worktree A directory missing");
        assert!(guard_b.path.is_dir(), "worktree B directory missing");

        // Both branches are registered and both worktrees are visible to git.
        assert!(branch_exists(&repo, &guard_a.branch));
        assert!(branch_exists(&repo, &guard_b.branch));
        assert!(worktree_is_registered(&repo, &guard_a.path));
        assert!(worktree_is_registered(&repo, &guard_b.path));

        // Each worktree checks out its own branch and sees the same HEAD commit.
        assert_eq!(
            run(&guard_a.path, &["rev-parse", "--abbrev-ref", "HEAD"]).trim(),
            guard_a.branch
        );
        assert_eq!(
            run(&guard_b.path, &["rev-parse", "--abbrev-ref", "HEAD"]).trim(),
            guard_b.branch
        );
        assert_eq!(
            run(&guard_a.path, &["rev-parse", "HEAD"]),
            run(&guard_b.path, &["rev-parse", "HEAD"])
        );

        (
            guard_a.branch.clone(),
            guard_a.path.clone(),
            guard_b.branch.clone(),
            guard_b.path.clone(),
        )
    };

    // Guards were dropped at the end of the scope: everything is cleaned up.
    for (branch, path) in [(&branch_a, &path_a), (&branch_b, &path_b)] {
        assert!(!branch_exists(&repo, branch), "branch {branch} still exists");
        assert!(!worktree_is_registered(&repo, path), "worktree {path:?} still registered");
        assert!(!path.exists(), "directory {path:?} still exists");
    }
}

#[test]
fn get_diff_reports_untracked_and_modified_files() {
    let repo = repo_root();
    let id = unique_worker_id("diff");
    let guard = WorktreeGuard::new(&repo, &id).expect("worktree creation failed");

    // Pristine worktree => empty diff.
    assert_eq!(guard.get_diff().expect("get_diff failed on clean tree"), "");

    // 1. An untracked (new) file.
    let new_file = guard.path.join("brand_new_file.txt");
    std::fs::write(&new_file, "hello from the worker\n").expect("failed to create untracked file");

    // 2. A modification of a file tracked in the worktree.
    let tracked = guard.path.join("tracked_file.txt");
    std::fs::write(&tracked, "original contents\n").expect("failed to create tracked file");
    run(&guard.path, &["add", "tracked_file.txt"]);
    run(&guard.path, &["commit", "-m", "add baseline file"]);

    std::fs::write(&tracked, "modified contents\n").expect("failed to modify tracked file");

    let diff = guard.get_diff().expect("get_diff failed");

    // Untracked file is surfaced (intent-to-add makes it appear in the diff).
    assert!(
        diff.contains("brand_new_file.txt"),
        "diff is missing the untracked file:\n{diff}"
    );
    assert!(
        diff.contains("hello from the worker"),
        "diff is missing the untracked file contents:\n{diff}"
    );

    // Modified tracked file is surfaced with both removed and added lines.
    assert!(
        diff.contains("tracked_file.txt"),
        "diff is missing the modified file:\n{diff}"
    );
    assert!(
        diff.contains("-original contents") && diff.contains("+modified contents"),
        "diff is missing the modification hunks:\n{diff}"
    );

    // Both files live inside the worktree, not in the parent repository.
    assert!(new_file.starts_with(&guard.path));
    assert!(tracked.starts_with(&guard.path));
}
