//! Integration tests for [`mini_swe_mcp::worktree::WorktreeGuard`].
//!
//! These tests run against an isolated temporary Git repository per test.
//! They never touch or mutate the host repository.

use mini_swe_mcp::worktree::WorktreeGuard;
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

struct TestRepo {
    dir: PathBuf,
}

impl TestRepo {
    fn new(prefix: &str) -> Self {
        let dir = std::env::temp_dir().join(format!(
            "swe-test-repo-{prefix}-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::SeqCst)
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        run(&dir, &["init", "-b", "master"]);
        run(&dir, &["config", "user.name", "mini-swe-test"]);
        run(&dir, &["config", "user.email", "test@localhost"]);

        // Create an initial commit with a README.md
        let readme = dir.join("README.md");
        std::fs::write(&readme, "# Test Repository\nInitial baseline content\n").unwrap();
        run(&dir, &["add", "README.md"]);
        run(&dir, &["commit", "-m", "Initial baseline commit"]);
        Self { dir }
    }

    fn path(&self) -> &Path {
        &self.dir
    }
}

impl Drop for TestRepo {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
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
    let test_repo = TestRepo::new("concurrent");
    let repo = test_repo.path();

    let id_a = unique_worker_id("a");
    let id_b = unique_worker_id("b");

    let (branch_a, path_a, branch_b, path_b) = {
        let guard_a = WorktreeGuard::new(repo, &id_a).expect("first worktree failed");
        let guard_b = WorktreeGuard::new(repo, &id_b).expect("second worktree failed");

        // Distinct branches and distinct on-disk locations.
        assert_ne!(guard_a.branch, guard_b.branch, "branches must differ");
        assert_ne!(guard_a.path, guard_b.path, "worktree paths must differ");
        assert_eq!(guard_a.branch, format!("worker-{id_a}"));
        assert_eq!(guard_b.branch, format!("worker-{id_b}"));
        assert!(guard_a.path.is_dir(), "worktree A directory missing");
        assert!(guard_b.path.is_dir(), "worktree B directory missing");

        // Both branches are registered and both worktrees are visible to git.
        assert!(branch_exists(repo, &guard_a.branch));
        assert!(branch_exists(repo, &guard_b.branch));
        assert!(worktree_is_registered(repo, &guard_a.path));
        assert!(worktree_is_registered(repo, &guard_b.path));

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
        assert!(!branch_exists(repo, branch), "branch {branch} still exists");
        assert!(!worktree_is_registered(repo, path), "worktree {path:?} still registered");
        assert!(!path.exists(), "directory {path:?} still exists");
    }
}

#[test]
fn get_diff_reports_untracked_and_modified_files() {
    let test_repo = TestRepo::new("diff");
    let repo = test_repo.path();
    let id = unique_worker_id("diff");
    let guard = WorktreeGuard::new(repo, &id).expect("worktree creation failed");

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

/// True when `git worktree list` mentions `path` as a registered worktree.
fn worktree_list_mentions(repo: &std::path::Path, path: &std::path::Path) -> bool {
    let list = run(repo, &["worktree", "list"]);
    let target = path.to_str().unwrap_or_default();
    list.lines()
        .filter_map(|line| line.split_whitespace().next())
        .any(|entry| entry == target)
}

/// True when `git branch --list <branch>` reports the branch as existing.
fn branch_is_listed(repo: &std::path::Path, branch: &str) -> bool {
    let list = run(repo, &["branch", "--list", branch]);
    list.lines().any(|line| {
        let name = line.trim().trim_start_matches('*').trim();
        !name.is_empty()
    })
}

#[test]
fn test_worktree_cleanup_on_drop_with_uncommitted_files() {
    let test_repo = TestRepo::new("dirty");
    let repo = test_repo.path();
    let id = unique_worker_id("dirty");
    let (branch, path) = {
        let guard = WorktreeGuard::new(repo, &id).expect("worktree creation failed");
        let path = guard.path.clone();
        let branch = guard.branch.clone();

        // 1. An untracked (never added) file inside the worktree.
        let untracked = path.join("untracked_dirty_file.txt");
        std::fs::write(&untracked, "untracked payload\n").expect("failed to write untracked file");
        assert!(untracked.exists(), "untracked file was not created");

        // 2. A modification of a file that is tracked in the worktree.
        let tracked = path.join("README.md");
        assert!(tracked.exists(), "expected a tracked README.md in the worktree");
        let original = std::fs::read_to_string(&tracked).expect("failed to read tracked file");
        std::fs::write(&tracked, format!("{original}\nmodified by the worker\n"))
            .expect("failed to modify tracked file");

        // 3. Sanity check: git really sees both kinds of uncommitted change,
        //    so cleanup is not trivially satisfied by a pristine tree.
        let status = run(&path, &["status", "--porcelain"]);
        assert!(
            status.contains("untracked_dirty_file.txt"),
            "untracked file missing from git status:\n{status}"
        );
        assert!(
            status.contains("README.md"),
            "modified file missing from git status:\n{status}"
        );

        // Dropping the guard must clean up regardless of the dirty state.
        drop(guard);

        (branch, path)
    };

    assert!(
        !path.exists(),
        "worktree directory {path:?} still exists after dropping the guard"
    );
    assert!(
        !worktree_list_mentions(repo, &path),
        "git worktree list still mentions {path:?}:\n{}",
        run(repo, &["worktree", "list"])
    );
    assert!(
        !worktree_is_registered(repo, &path),
        "worktree {path:?} is still registered"
    );
    assert!(
        !branch_is_listed(repo, &branch),
        "git branch --list still reports branch {branch}:\n{}",
        run(repo, &["branch", "--list", &branch])
    );
    assert!(
        !branch_exists(repo, &branch),
        "branch {branch} still exists after dropping the guard"
    );
}

#[test]
fn test_sync_artifacts_preserves_reports_to_repo_root() {
    let test_repo = TestRepo::new("artifacts");
    let repo = test_repo.path();
    let id = unique_worker_id("artifacts");
    let guard = WorktreeGuard::new(repo, &id).expect("worktree creation failed");

    let audit_dir = guard.path.join("audits");
    std::fs::create_dir_all(&audit_dir).expect("failed to create audits dir in worktree");
    let audit_file = audit_dir.join(format!("audit_{id}.md"));
    std::fs::write(&audit_file, "# Subagent Audit Report\nAll clear.").expect("failed to write audit file");

    let synced = guard.sync_artifacts().expect("sync_artifacts failed");
    let expected_rel = format!("audits/audit_{id}.md");
    assert!(
        synced.contains(&expected_rel),
        "expected synced to contain {expected_rel}, got: {synced:?}"
    );

    let destination = repo.join(&expected_rel);
    assert!(destination.exists(), "artifact was not copied to repo root: {destination:?}");
    let content = std::fs::read_to_string(&destination).expect("failed to read copied artifact");
    assert!(content.contains("# Subagent Audit Report"));

    let _ = std::fs::remove_file(&destination);

    drop(guard);
}

#[test]
fn test_commit_changes_preserves_branch_on_drop() {
    let test_repo = TestRepo::new("commit");
    let repo = test_repo.path();
    let id = unique_worker_id("commit");
    let branch = format!("worker-{id}");

    {
        let mut guard = WorktreeGuard::new(repo, &id).expect("worktree creation failed");
        let new_file = guard.path.join("preserved_feature.txt");
        std::fs::write(&new_file, "Preserved code from subagent\n").expect("write file");

        let committed_branch = guard
            .commit_changes("worker(test): preserve this work")
            .expect("commit failed");
        assert_eq!(committed_branch, Some(branch.clone()));
        assert!(guard.preserve_branch);
        // Guard drops here
    }

    // Worktree directory and registration are gone
    let path = mini_swe_mcp::worktree::swe_base_dir().join(format!("swe-wt-{id}"));
    assert!(!path.exists(), "worktree dir should be cleaned up");
    assert!(!worktree_is_registered(repo, &path));

    // But the git branch is PRESERVED in the test repository
    assert!(branch_exists(repo, &branch), "worker branch should be preserved");

    let log = run(repo, &["log", "-1", "--pretty=%s", &branch]);
    assert!(
        log.contains("worker(test): preserve this work"),
        "commit message not found in preserved branch: {log}"
    );
}

#[test]
fn test_sync_artifacts_skips_unchanged_files_but_copies_changed_ones() {
    let test_repo = TestRepo::new("unchanged");
    let repo = test_repo.path();
    let id = unique_worker_id("unchanged");
    let guard = WorktreeGuard::new(repo, &id).expect("worktree creation failed");

    let audit_dir = guard.path.join("audits");
    std::fs::create_dir_all(&audit_dir).expect("failed to create audits dir in worktree");
    let worktree_file = audit_dir.join("stable_audit.md");
    std::fs::write(&worktree_file, "content v1\n").expect("failed to write audit file");

    // First sync copies the file and reports it.
    let first = guard.sync_artifacts().expect("first sync_artifacts failed");
    assert!(
        first.contains(&"audits/stable_audit.md".to_string()),
        "expected the first sync to report audits/stable_audit.md, got: {first:?}"
    );

    let destination = repo.join("audits/stable_audit.md");
    assert!(destination.exists(), "artifact was not copied to repo root");

    // Stamp the destination with a distinctive mtime. An unchanged file must not
    // be rewritten, so the stamp has to survive a second sync untouched.
    let sentinel = std::time::UNIX_EPOCH + std::time::Duration::from_secs(1_000_000);
    std::fs::File::options()
        .write(true)
        .open(&destination)
        .expect("failed to open destination for stamping")
        .set_times(std::fs::FileTimes::new().set_modified(sentinel))
        .expect("failed to stamp destination mtime");

    // The unchanged file is still reported as in sync...
    let second = guard
        .sync_artifacts()
        .expect("second sync_artifacts failed");
    assert!(
        second.contains(&"audits/stable_audit.md".to_string()),
        "an unchanged but present artifact must still be reported, got: {second:?}"
    );
    // ...but it was not rewritten.
    let mtime = std::fs::metadata(&destination)
        .expect("failed to stat destination")
        .modified()
        .expect("failed to read destination mtime");
    assert_eq!(
        mtime, sentinel,
        "unchanged file was rewritten during re-sync (mtime advanced)"
    );

    // Once the worktree copy actually changes, the sync must pick it up.
    std::fs::write(&worktree_file, "content v2\n").expect("failed to modify audit file");
    guard.sync_artifacts().expect("third sync_artifacts failed");
    let body = std::fs::read_to_string(&destination).expect("failed to read copied artifact");
    assert_eq!(body, "content v2\n", "changed artifact was not re-copied");

    let _ = std::fs::remove_file(&destination);
    drop(guard);
}

#[test]
fn test_sync_artifacts_never_mirrors_git_build_or_node_modules() {
    let test_repo = TestRepo::new("guards");
    let repo = test_repo.path();
    let id = unique_worker_id("guards");
    let guard = WorktreeGuard::new(repo, &id).expect("worktree creation failed");

    // A worker that ran a build or an install inside an artifact directory
    // leaves dependency caches and build output behind. None of it may cross
    // into the repository root.
    let skipped = [
        ".git",
        "node_modules",
        "target",
        "build",
        "dist",
        "__pycache__",
        ".venv",
    ];
    let audit_dir = guard.path.join("audits");
    for name in skipped {
        let nested = audit_dir.join(name).join("nested");
        std::fs::create_dir_all(&nested).expect("failed to create skipped dir in worktree");
        std::fs::write(nested.join("payload.bin"), "should never be mirrored")
            .expect("failed to write payload");
    }
    std::fs::write(audit_dir.join("keep.md"), "keep me\n").expect("failed to write artifact");

    let synced = guard.sync_artifacts().expect("sync_artifacts failed");

    // The genuine artifact is synced and lands in the repo root.
    assert!(
        synced.contains(&"audits/keep.md".to_string()),
        "expected audits/keep.md to be synced, got: {synced:?}"
    );
    assert_eq!(
        std::fs::read_to_string(repo.join("audits/keep.md")).expect("failed to read artifact"),
        "keep me\n"
    );

    for name in skipped {
        assert!(
            !synced.iter().any(|p| p.contains(name)),
            "skipped directory {name} leaked into the synced list: {synced:?}"
        );
        assert!(
            !repo.join("audits").join(name).exists(),
            "skipped directory {name} was materialized in the repo root"
        );
    }

    drop(guard);
}

#[test]
fn test_sync_artifacts_publishes_files_atomically_without_staging_debris() {
    let test_repo = TestRepo::new("atomic");
    let repo = test_repo.path();
    let id = unique_worker_id("atomic");
    let guard = WorktreeGuard::new(repo, &id).expect("worktree creation failed");

    let report_dir = guard.path.join("reports");
    std::fs::create_dir_all(&report_dir).expect("failed to create reports dir in worktree");
    const FILES: usize = 50;
    for i in 0..FILES {
        std::fs::write(
            report_dir.join(format!("report_{i}.md")),
            format!("payload {i}\n"),
        )
        .expect("failed to write report");
    }

    guard.sync_artifacts().expect("sync_artifacts failed");

    let destination = repo.join("reports");
    // Every file landed with its exact contents...
    for i in 0..FILES {
        let name = format!("report_{i}.md");
        assert_eq!(
            std::fs::read_to_string(destination.join(&name)).expect("failed to read report"),
            format!("payload {i}\n"),
            "artifact {name} did not survive the sync"
        );
    }
    // ...and no staging file from the atomic write was left behind.
    let staging: Vec<String> = std::fs::read_dir(&destination)
        .expect("failed to read destination reports dir")
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".tmp"))
        .collect();
    assert!(
        staging.is_empty(),
        "atomic sync left staging files behind in the repo root: {staging:?}"
    );

    drop(guard);
}
