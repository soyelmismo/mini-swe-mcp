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

/// A throwaway repository plus the scratch root its worktrees live under.
///
/// Every checkout this fixture drives is built with
/// [`WorktreeGuard::new_in`] over a root the fixture owns and removes, so the
/// suite never files a `swe-wt-*` checkout or its `swe-tmp-*` companion in the
/// real scratch base that [`mini_swe_mcp::worktree::swe_base_dir`] resolves.
struct TestRepo {
    dir: PathBuf,
    root: PathBuf,
}

impl TestRepo {
    fn new(prefix: &str) -> Self {
        let unique = format!(
            "swe-test-repo-{prefix}-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::SeqCst)
        );
        // One parent per fixture holds both the repository and the worktree
        // scratch root, so `Drop` takes the pair with a single removal.
        let base = std::env::temp_dir().join(unique);
        let _ = std::fs::remove_dir_all(&base);
        let dir = base.join("repo");
        std::fs::create_dir_all(&dir).unwrap();
        run(&dir, &["init", "-b", "master"]);
        run(&dir, &["config", "user.name", "mini-swe-test"]);
        run(&dir, &["config", "user.email", "test@localhost"]);

        // Create an initial commit with a README.md
        let readme = dir.join("README.md");
        std::fs::write(&readme, "# Test Repository\nInitial baseline content\n").unwrap();
        run(&dir, &["add", "README.md"]);
        run(&dir, &["commit", "-m", "Initial baseline commit"]);
        Self { dir, root: base }
    }

    fn path(&self) -> &Path {
        &self.dir
    }

    /// The scratch root every checkout of this repository is created under.
    fn scratch(&self) -> mini_swe_mcp::worktree::ScratchRoot {
        mini_swe_mcp::worktree::ScratchRoot::new(&self.root)
    }

    /// A worker checkout of this repository, isolated from the real scratch base.
    fn guard(&self, worker_id: &str) -> WorktreeGuard {
        WorktreeGuard::new_in(&self.scratch(), &self.dir, worker_id)
            .expect("worktree creation failed")
    }
}

impl Drop for TestRepo {
    fn drop(&mut self) {
        // A worker leases build directories keyed by this repo's hash, and a
        // checkout it ran derives a private `swe-tmp-<leaf>`; both are filed
        // next to the scratch base, so removing the fixture has to take them.
        mini_swe_mcp::cache::remove_build_dir_leases(&self.dir);
        mini_swe_mcp::worktree::remove_target_dirs(&self.dir);
        mini_swe_mcp::worktree::remove_scratch_root_worktrees(&self.root);
        let _ = std::fs::remove_dir_all(&self.root);
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
        let guard_a = test_repo.guard(&id_a);
        let guard_b = test_repo.guard(&id_b);

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
        assert!(
            !worktree_is_registered(repo, path),
            "worktree {path:?} still registered"
        );
        assert!(!path.exists(), "directory {path:?} still exists");
    }
}

/// Checkpoint commits made while the worker runs must not hide earlier work:
/// the reported diff spans everything since the worker's base commit.
#[test]
fn get_diff_spans_checkpoint_commits_and_uncommitted_work() {
    let test_repo = TestRepo::new("diff-checkpoint");
    let repo = test_repo.path();
    let id = unique_worker_id("diff-checkpoint");
    let mut guard = test_repo.guard(&id);

    std::fs::write(guard.path.join("before_checkpoint.txt"), "first\n").unwrap();
    guard
        .commit_changes("checkpoint")
        .expect("checkpoint commit failed");
    std::fs::write(guard.path.join("after_checkpoint.txt"), "second\n").unwrap();

    let diff = guard.get_diff().expect("get_diff failed");
    assert!(
        diff.contains("before_checkpoint.txt"),
        "checkpointed work missing:\n{diff}"
    );
    assert!(
        diff.contains("after_checkpoint.txt"),
        "uncommitted work missing:\n{diff}"
    );
}

#[test]
fn get_diff_reports_untracked_and_modified_files() {
    let test_repo = TestRepo::new("diff");
    let repo = test_repo.path();
    let id = unique_worker_id("diff");
    let guard = test_repo.guard(&id);

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

    // The file committed inside the worktree and then modified is reported as
    // the worker's net change since its base commit: its current contents.
    assert!(
        diff.contains("tracked_file.txt"),
        "diff is missing the modified file:\n{diff}"
    );
    assert!(
        diff.contains("+modified contents") && !diff.contains("original contents"),
        "diff must show the net change since the base commit:\n{diff}"
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
        let guard = test_repo.guard(&id);
        let path = guard.path.clone();
        let branch = guard.branch.clone();

        // 1. An untracked (never added) file inside the worktree.
        let untracked = path.join("untracked_dirty_file.txt");
        std::fs::write(&untracked, "untracked payload\n").expect("failed to write untracked file");
        assert!(untracked.exists(), "untracked file was not created");

        // 2. A modification of a file that is tracked in the worktree.
        let tracked = path.join("README.md");
        assert!(
            tracked.exists(),
            "expected a tracked README.md in the worktree"
        );
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
    let guard = test_repo.guard(&id);

    let audit_dir = guard.path.join("audits");
    std::fs::create_dir_all(&audit_dir).expect("failed to create audits dir in worktree");
    let audit_file = audit_dir.join(format!("audit_{id}.md"));
    std::fs::write(&audit_file, "# Subagent Audit Report\nAll clear.")
        .expect("failed to write audit file");

    let synced = guard.sync_artifacts();
    let expected_rel = format!("audits/audit_{id}.md");
    assert!(
        synced.contains(&expected_rel),
        "expected synced to contain {expected_rel}, got: {synced:?}"
    );

    let destination = repo.join(&expected_rel);
    assert!(
        destination.exists(),
        "artifact was not copied to repo root: {destination:?}"
    );
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
        let mut guard = test_repo.guard(&id);
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
    assert!(
        branch_exists(repo, &branch),
        "worker branch should be preserved"
    );

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
    let guard = test_repo.guard(&id);

    let audit_dir = guard.path.join("audits");
    std::fs::create_dir_all(&audit_dir).expect("failed to create audits dir in worktree");
    let worktree_file = audit_dir.join("stable_audit.md");
    std::fs::write(&worktree_file, "content v1\n").expect("failed to write audit file");

    // First sync copies the file and reports it.
    let first = guard.sync_artifacts();
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
    let second = guard.sync_artifacts();
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
    guard.sync_artifacts();
    let body = std::fs::read_to_string(&destination).expect("failed to read copied artifact");
    assert_eq!(body, "content v2\n", "changed artifact was not re-copied");

    let _ = std::fs::remove_file(&destination);
    drop(guard);
}

/// A seeded artifact is the worker's starting point, not its output. When the
/// repo root's copy moves on while the worker runs (another worker's result
/// merged, the user edited it), re-publishing the worker's untouched copy would
/// silently revert that newer content.
#[test]
fn test_sync_never_reverts_a_newer_repo_file_the_worker_never_touched() {
    let test_repo = TestRepo::new("no-revert");
    let repo = test_repo.path();
    std::fs::create_dir_all(repo.join("audits")).expect("failed to create audits dir in repo");
    let repo_file = repo.join("audits/memory.md");
    std::fs::write(&repo_file, "seeded v1\n").expect("failed to write seeded artifact");

    let id = unique_worker_id("no-revert");
    let guard = test_repo.guard(&id);
    assert!(
        guard.path.join("audits/memory.md").exists(),
        "the artifact was not seeded into the worktree"
    );

    // Another writer advances the repo root while the worker runs. The worker
    // never reads or writes its seeded copy.
    std::fs::write(&repo_file, "newer content from another worker\n")
        .expect("failed to edit the repo artifact after seeding");

    let synced = guard.sync_artifacts();
    assert_eq!(
        std::fs::read_to_string(&repo_file).expect("failed to read repo artifact"),
        "newer content from another worker\n",
        "the sync reverted a repo file that moved on after seeding"
    );
    // A file the worker never touched is not its artifact: the sync neither
    // copies it nor reports it, so the completion view stays the worker's own.
    assert!(
        !synced.contains(&"audits/memory.md".to_string()),
        "an inherited artifact must not be reported, got: {synced:?}"
    );

    drop(guard);
    assert_eq!(
        std::fs::read_to_string(&repo_file).expect("failed to read repo artifact"),
        "newer content from another worker\n",
        "the teardown sync reverted a repo file that moved on after seeding"
    );
}

/// The other half of the rule: a seeded file the worker *did* edit is its
/// output, and must reach the repo root even though the path was seeded.
#[test]
fn test_sync_publishes_a_seeded_file_the_worker_modified() {
    let test_repo = TestRepo::new("edited-seed");
    let repo = test_repo.path();
    std::fs::create_dir_all(repo.join("reports")).expect("failed to create reports dir in repo");
    let repo_file = repo.join("reports/status.md");
    std::fs::write(&repo_file, "seeded v1\n").expect("failed to write seeded report");

    let id = unique_worker_id("edited-seed");
    let guard = test_repo.guard(&id);
    std::fs::write(guard.path.join("reports/status.md"), "worker revision\n")
        .expect("failed to edit the seeded report in the worktree");

    let synced = guard.sync_artifacts();
    assert!(
        synced.contains(&"reports/status.md".to_string()),
        "expected the edited artifact to be reported, got: {synced:?}"
    );
    assert_eq!(
        std::fs::read_to_string(&repo_file).expect("failed to read synced report"),
        "worker revision\n",
        "the worker's edit to a seeded file was not synced"
    );

    drop(guard);
}

/// A path the worker invented was never seeded, so it has no fingerprint to
/// match and always travels back to the repo root.
#[test]
fn test_sync_publishes_a_new_file_the_worker_created() {
    let test_repo = TestRepo::new("new-artifact");
    let repo = test_repo.path();
    let id = unique_worker_id("new-artifact");
    let guard = test_repo.guard(&id);

    let audit_dir = guard.path.join("audits");
    std::fs::create_dir_all(&audit_dir).expect("failed to create audits dir in worktree");
    std::fs::write(audit_dir.join("audit_new.md"), "# New audit\n")
        .expect("failed to write the new artifact");

    let synced = guard.sync_artifacts();
    assert!(
        synced.contains(&"audits/audit_new.md".to_string()),
        "expected the new artifact to be reported, got: {synced:?}"
    );
    assert_eq!(
        std::fs::read_to_string(repo.join("audits/audit_new.md"))
            .expect("failed to read the new artifact"),
        "# New audit\n",
        "a file the worker created was not synced"
    );

    drop(guard);
}

#[test]
fn test_sync_artifacts_never_mirrors_git_build_or_node_modules() {
    let test_repo = TestRepo::new("guards");
    let repo = test_repo.path();
    let id = unique_worker_id("guards");
    let guard = test_repo.guard(&id);

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

    let synced = guard.sync_artifacts();

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
    let guard = test_repo.guard(&id);

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

    guard.sync_artifacts();

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

/// `WorktreeGuard::new` is public, so its `worker_id` is untrusted input even
/// though the pool only ever mints hex ids. A hostile id must not be able to
/// steer the worktree directory out of the scratch base, nor reach git as an
/// option.
#[test]
fn hostile_worker_id_cannot_escape_the_scratch_base_or_become_a_git_flag() {
    let test_repo = TestRepo::new("hostile");
    let repo = test_repo.path();

    // A traversal payload and a leading-dash flag payload, in one dispatch each.
    let traversal = unique_worker_id("hostile/../../escape");
    let flag = format!("--upload-pack=/bin/sh-{traversal}");

    for (id, label) in [(&traversal, "traversal"), (&flag, "option-injection")] {
        let guard = test_repo.guard(id);

        let name = guard
            .path
            .file_name()
            .and_then(|n| n.to_str())
            .expect("worktree path has a UTF-8 name");
        assert!(
            name.starts_with("swe-wt-"),
            "{label}: worktree dir {name:?} escaped the swe-wt- namespace"
        );
        assert!(
            !name.contains('/') && !name.contains(".."),
            "{label}: worktree dir {name:?} still contains a separator or traversal"
        );

        // The worktree must still be a real, registered git worktree: hardening
        // the id may not break the checkout itself.
        assert!(guard.path.is_dir(), "{label}: worktree directory missing");
        assert!(
            worktree_is_registered(repo, &guard.path),
            "{label}: git does not know this worktree"
        );
        assert!(
            branch_exists(repo, &guard.branch),
            "{label}: branch was not created"
        );

        drop(guard);
    }
}

/// The worktree holds a full checkout of the repository plus whatever secrets
/// and artifacts the subagent writes. It must not be readable by any other
/// local account, regardless of the host umask.
#[cfg(unix)]
#[test]
fn worktree_directory_is_private_to_its_owner() {
    use std::os::unix::fs::PermissionsExt;

    let test_repo = TestRepo::new("private");
    let repo = test_repo.path();
    let id = unique_worker_id("private");
    let guard = test_repo.guard(&id);

    let mode = std::fs::metadata(&guard.path)
        .expect("failed to stat the worktree directory")
        .permissions()
        .mode()
        & 0o777;
    assert_eq!(
        mode, 0o700,
        "worktree directory is {mode:o}; it must be 0700 so no other local \
         account can read the checkout, its secrets or its artifacts"
    );

    drop(guard);
}

/// `reopen` re-attaches to the preserved branch: same branch name, previous
/// commits intact, original diff base, and a missing branch is a clear error.
#[test]
fn reopen_reattaches_to_the_preserved_branch() {
    let test_repo = TestRepo::new("reopen");
    let repo = test_repo.path();
    let id = unique_worker_id("reopen");

    let base = run(repo, &["rev-parse", "HEAD"]).trim().to_string();
    let branch = format!("worker-{id}");
    let first_path = {
        let mut guard = test_repo.guard(&id);
        std::fs::write(guard.path.join("fix.txt"), "fix\n").expect("worker change");
        guard
            .commit_changes("worker: fix")
            .expect("checkpoint commit");
        guard.path.clone()
    };
    assert!(
        branch_exists(repo, &branch),
        "the finished run must preserve its branch"
    );
    assert!(
        !first_path.exists(),
        "the finished run must remove its checkout"
    );

    let guard = WorktreeGuard::reopen(repo, &id, &base).expect("reopen must re-attach");
    assert_eq!(guard.branch, branch, "the revision keeps the same branch");
    assert_eq!(
        guard.base_commit, base,
        "the diff base stays the original commit"
    );
    assert!(
        guard.path.join("fix.txt").is_file(),
        "checkpoints survive the re-attach"
    );
    assert!(
        worktree_is_registered(repo, &guard.path),
        "the re-attached checkout is a registered worktree"
    );
}

/// Reopening a branch that no longer exists names the branch in the error.
#[test]
fn reopen_on_a_missing_branch_is_a_clear_error() {
    let test_repo = TestRepo::new("reopen-missing");
    let repo = test_repo.path();
    let id = unique_worker_id("reopen-missing");

    let err = match WorktreeGuard::reopen(repo, &id, "abc123") {
        Ok(_) => panic!("no such branch exists"),
        Err(e) => e,
    };
    assert!(
        err.to_string().contains(&format!("worker-{id}")),
        "the error must name the missing branch, got: {err}"
    );
}

fn sync_base(guard: &WorktreeGuard) -> mini_swe_mcp::worktree::BaseSync {
    WorktreeGuard::sync_base_at(
        &guard.path,
        &guard.repo_root,
        &guard.branch,
        &guard.base_commit,
        guard.base_branch.as_deref(),
    )
    .expect("base integration failed")
}

#[test]
fn moving_base_is_merged_and_excluded_from_worker_diff() {
    use mini_swe_mcp::worktree::BaseSync;
    let repo = TestRepo::new("sync-clean");
    let guard = repo.guard(&unique_worker_id("sync-clean"));
    assert_eq!(guard.base_branch.as_deref(), Some("master"));
    assert_eq!(sync_base(&guard), BaseSync::Unchanged);
    std::fs::write(guard.path.join("worker.txt"), "worker change\n").unwrap();
    std::fs::write(repo.path().join("base.txt"), "base change\n").unwrap();
    run(repo.path(), &["add", "."]);
    run(repo.path(), &["commit", "-m", "move base"]);
    assert_eq!(
        sync_base(&guard),
        BaseSync::Merged {
            branch: "master".into()
        }
    );
    assert_eq!(
        std::fs::read_to_string(guard.path.join("base.txt")).unwrap(),
        "base change\n"
    );
    assert_eq!(run(&guard.path, &["status", "--porcelain"]), "");
    let diff = guard.get_diff().unwrap();
    assert!(diff.contains("worker.txt"));
    assert!(
        !diff.contains("base.txt"),
        "base-only work leaked into diff: {diff}"
    );
    assert_eq!(
        run(&guard.path, &["rev-list", "--parents", "-n", "1", "HEAD"])
            .split_whitespace()
            .count(),
        3
    );
    assert_eq!(sync_base(&guard), BaseSync::Unchanged);

    // A second base advance is integrated without forgetting the worker's first commit.
    std::fs::write(repo.path().join("later.txt"), "later base change\n").unwrap();
    run(repo.path(), &["add", "."]);
    run(repo.path(), &["commit", "-m", "move base again"]);
    assert!(matches!(sync_base(&guard), BaseSync::Merged { .. }));
    let diff = guard.get_diff().unwrap();
    assert!(diff.contains("worker.txt"));
    assert!(!diff.contains("base.txt") && !diff.contains("later.txt"));
}

#[test]
fn conflicting_base_refuses_completion_until_markers_are_resolved() {
    use mini_swe_mcp::worktree::BaseSync;
    let repo = TestRepo::new("sync-conflict");
    let mut guard = repo.guard(&unique_worker_id("sync-conflict"));
    std::fs::write(guard.path.join("README.md"), "worker intent\n").unwrap();
    std::fs::write(repo.path().join("README.md"), "base intent\n").unwrap();
    run(repo.path(), &["add", "."]);
    run(repo.path(), &["commit", "-m", "conflicting base"]);
    let conflicts = BaseSync::Conflicts {
        branch: "master".into(),
        files: vec!["README.md".into()],
    };
    assert_eq!(sync_base(&guard), conflicts);
    assert!(WorktreeGuard::merge_in_progress_at(&guard.path).unwrap());
    let markers = std::fs::read_to_string(guard.path.join("README.md")).unwrap();
    assert!(markers.contains("<<<<<<<"));
    assert!(markers.contains("base intent") && markers.contains("worker intent"));
    let head = run(&guard.path, &["rev-parse", "HEAD"]);
    assert!(guard.commit_changes("checkpoint during conflict").is_err());
    assert_eq!(
        sync_base(&guard),
        conflicts,
        "markers must refuse a second completion too"
    );
    assert_eq!(run(&guard.path, &["rev-parse", "HEAD"]), head);

    // Even if the model stages the file, the harness checks working-tree markers.
    run(&guard.path, &["add", "README.md"]);
    assert_eq!(sync_base(&guard), conflicts);
    std::fs::write(guard.path.join("README.md"), "base intent\nworker intent\n").unwrap();
    std::fs::write(guard.path.join("renamed.txt"), "<<<<<<< unresolved\n").unwrap();
    assert_eq!(
        sync_base(&guard),
        BaseSync::Conflicts {
            branch: "master".into(),
            files: vec!["renamed.txt".into()],
        }
    );
    std::fs::remove_file(guard.path.join("renamed.txt")).unwrap();
    assert_eq!(
        sync_base(&guard),
        BaseSync::Merged {
            branch: "master".into()
        }
    );
    assert!(!WorktreeGuard::merge_in_progress_at(&guard.path).unwrap());
    assert_eq!(run(&guard.path, &["status", "--porcelain"]), "");
    assert_eq!(
        run(&guard.path, &["rev-list", "--parents", "-n", "1", "HEAD"])
            .split_whitespace()
            .count(),
        3
    );
    let diff = guard.get_diff().unwrap();
    assert!(diff.contains("+worker intent"));
    assert!(!diff.contains("+base intent"));
}

#[test]
fn detached_dispatch_has_no_base_to_sync() {
    let repo = TestRepo::new("sync-detached");
    run(repo.path(), &["switch", "--detach"]);
    let guard = repo.guard(&unique_worker_id("sync-detached"));
    assert_eq!(guard.base_branch, None);
    std::fs::write(guard.path.join("worker.txt"), "pending\n").unwrap();
    assert_eq!(
        sync_base(&guard),
        mini_swe_mcp::worktree::BaseSync::Unchanged
    );
    assert_eq!(
        run(&guard.path, &["rev-parse", "HEAD"]).trim(),
        guard.base_commit
    );
    assert!(guard.get_diff().unwrap().contains("worker.txt"));
}

#[test]
fn sync_base_env_zero_leaves_worker_untouched() {
    // Isolate the env override in a child so concurrent tests cannot observe it.
    const CHILD: &str = "SWE_SYNC_BASE_TEST_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "worktree_test::sync_base_env_zero_leaves_worker_untouched",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .env("WORKER_SYNC_BASE", "0")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }
    let repo = TestRepo::new("sync-disabled");
    let guard = repo.guard(&unique_worker_id("sync-disabled"));
    std::fs::write(guard.path.join("worker.txt"), "pending\n").unwrap();
    std::fs::write(repo.path().join("base.txt"), "base\n").unwrap();
    run(repo.path(), &["add", "."]);
    run(repo.path(), &["commit", "-m", "move base"]);
    assert_eq!(
        sync_base(&guard),
        mini_swe_mcp::worktree::BaseSync::Unchanged
    );
    assert_eq!(
        run(&guard.path, &["rev-parse", "HEAD"]).trim(),
        guard.base_commit
    );
    assert_eq!(
        run(&guard.path, &["status", "--porcelain"]),
        "?? worker.txt\n"
    );
    assert!(!guard.path.join("base.txt").exists());
    assert!(!WorktreeGuard::merge_in_progress_at(&guard.path).unwrap());
}
