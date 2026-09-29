//! Tests for the abandoned-worktree sweep (`prune_stale_worktrees`) and the
//! PID-lease scheme it relies on.
//!
//! These cover the code paths that decide whether a subagent worktree may be
//! destroyed, which previously had no coverage at all. Every test runs against
//! its own throwaway Git repository and its own scratch base directory, so the
//! sweep can be driven explicitly without mutating the process-global
//! `SWE_TEMP_DIR` (see `prune_stale_worktrees_in`).
//!
//! Safety contract asserted throughout: the sweep may only ever delete a
//! worktree whose owning process is provably gone. Everything ambiguous
//! (missing, unreadable, malformed or foreign lease) must be preserved.

use mini_swe_mcp::worktree::{
    WorktreeGuard, is_process_alive, prune_stale_worktrees_in, worktree_is_stale_for_test,
};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static COUNTER: AtomicUsize = AtomicUsize::new(0);

fn unique(prefix: &str) -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!(
        "prune-{prefix}-{}-{nanos}-{}",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::SeqCst)
    )
}

fn try_run(dir: &Path, args: &[&str]) -> bool {
    Command::new("git")
        .current_dir(dir)
        .args(args)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

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

struct Fixture {
    repo: PathBuf,
    base: PathBuf,
    /// Worktrees created by the test, so `Drop` can unregister them even when
    /// the assertions already removed them.
    pending: Vec<(String, PathBuf, PathBuf)>,
}

impl Fixture {
    /// A repo plus an isolated scratch base dir, both removed on drop.
    fn new(prefix: &str) -> Self {
        let id = unique(prefix);
        let root = std::env::temp_dir().join(id);
        let repo = root.join("repo");
        let base = root.join("base");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&repo).unwrap();
        std::fs::create_dir_all(&base).unwrap();

        run(&repo, &["init", "-b", "master"]);
        run(&repo, &["config", "user.name", "mini-swe-test"]);
        run(&repo, &["config", "user.email", "test@localhost"]);
        std::fs::write(repo.join("README.md"), "# Baseline\n").unwrap();
        run(&repo, &["add", "README.md"]);
        run(&repo, &["commit", "-m", "Initial baseline commit"]);
        Self {
            repo,
            base,
            pending: Vec::new(),
        }
    }

    /// Register a `worker-*` branch checked out at `<base>/swe-wt-<id>` and
    /// return `(branch, dir, pid_file)`. The worktree is detached from its
    /// `WorktreeGuard` so that `Drop` does not clean it up.
    fn add_worktree(&self, id: &str) -> (String, PathBuf, PathBuf) {
        let branch = format!("worker-{id}");
        let dir = self.base.join(format!("swe-wt-{id}"));
        let pid_file = {
            let mut s = dir.clone().into_os_string();
            s.push(".pid");
            PathBuf::from(s)
        };
        run(
            &self.repo,
            &[
                "worktree",
                "add",
                "-b",
                &branch,
                dir.to_str().unwrap(),
                "HEAD",
            ],
        );
        (branch, dir, pid_file)
    }

    /// Write a lease naming this (live) test process, matching what
    /// `WorktreeGuard::new` writes.
    fn write_live_lease(&self, pid_file: &Path) {
        std::fs::write(pid_file, format!("{}", std::process::id())).unwrap();
    }

    /// Write a lease naming a PID that is guaranteed not to exist.
    ///
    /// `/proc/sys/kernel/pid_max` bounds the range; a pid of `u32::MAX - 1`
    /// is never reused within any plausible pid_max, so it is always dead.
    const DEAD_PID: u32 = 4_294_967_294;

    fn write_dead_lease(&self, pid_file: &Path) {
        std::fs::write(pid_file, Self::DEAD_PID.to_string()).unwrap();
    }

    fn sweep(&self) {
        prune_stale_worktrees_in(&self.repo, std::slice::from_ref(&self.base));
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        // Remove any worktree registrations first so git does not leave
        // dangling metadata behind in the throwaway repo.
        for (branch, dir, _) in self.pending.drain(..) {
            let _ = try_run(&self.repo, &["worktree", "remove", "--force", dir.to_str().unwrap_or("")]);
            let _ = try_run(&self.repo, &["branch", "-D", &branch]);
        }
        if let Some(root) = self.base.parent() {
            let _ = std::fs::remove_dir_all(root);
        }
    }
}

// `pending` lives in a separate impl block so the Drop impl above stays
// readable; the field itself is declared by the struct patch below.
impl Fixture {
    fn track(&mut self, entry: (String, PathBuf, PathBuf)) {
        self.pending.push(entry);
    }
}

/// A PID that `is_process_alive` must report as dead.
#[test]
fn dead_pid_is_reported_dead_and_self_alive() {
    assert!(
        is_process_alive(std::process::id()),
        "this test process must be reported as alive"
    );
    assert!(
        !is_process_alive(Fixture::DEAD_PID),
        "PID {} must be reported as dead",
        Fixture::DEAD_PID
    );
}

/// Test 1 — a worktree leased by a live process survives the sweep.
#[test]
fn live_worktree_is_preserved() {
    let mut f = Fixture::new("live");
    let (branch, dir, pid_file) = f.add_worktree(&unique("live"));
    f.track((branch.clone(), dir.clone(), pid_file.clone()));
    f.write_live_lease(&pid_file);

    f.sweep();

    assert!(dir.is_dir(), "live worktree directory was pruned");
    assert!(pid_file.exists(), "live worktree lease was removed");
    assert!(
        branch_exists(&f.repo, &branch),
        "live worktree branch {branch} was pruned"
    );
    assert!(
        worktree_registered(&f.repo, &dir),
        "live worktree was unregistered"
    );
}

/// Test 2 — a worktree leased by a dead process loses directory, branch and
/// lease in one sweep.
#[test]
fn dead_worktree_is_pruned_completely() {
    let mut f = Fixture::new("dead");
    let (branch, dir, pid_file) = f.add_worktree(&unique("dead"));
    f.track((branch.clone(), dir.clone(), pid_file.clone()));
    f.write_dead_lease(&pid_file);

    f.sweep();

    assert!(!dir.exists(), "zombie worktree directory survived the sweep");
    assert!(!pid_file.exists(), "zombie lease file survived the sweep");
    assert!(
        !branch_exists(&f.repo, &branch),
        "zombie branch {branch} survived the sweep"
    );
    assert!(
        !worktree_registered(&f.repo, &dir),
        "zombie worktree is still registered"
    );
}

/// Test 3 — no lease at all: fail open and keep the worktree.
#[test]
fn unleased_worktree_is_preserved() {
    let mut f = Fixture::new("nolease");
    let (branch, dir, pid_file) = f.add_worktree(&unique("nolease"));
    f.track((branch.clone(), dir.clone(), pid_file.clone()));
    assert!(!pid_file.exists(), "precondition: no lease written");

    f.sweep();

    assert!(dir.is_dir(), "worktree without a lease was pruned");
    assert!(branch_exists(&f.repo, &branch), "branch of unleased worktree was pruned");
}

/// Test 4 — a corrupt lease is handled identically by the registered-worktree
/// path and the orphan-directory path, and never destroys live work.
#[test]
fn corrupt_lease_is_handled_consistently() {
    let mut f = Fixture::new("corrupt-reg");
    let (branch, dir, pid_file) = f.add_worktree(&unique("corrupt-reg"));
    f.track((branch.clone(), dir.clone(), pid_file.clone()));
    std::fs::write(&pid_file, "not-a-pid").unwrap();

    f.sweep();

    assert!(
        dir.is_dir(),
        "registered worktree with a corrupt lease was pruned (fail-open expected)"
    );
    assert!(branch_exists(&f.repo, &branch), "corrupt-lease branch was pruned");

    // Same corrupt data seen by the orphan-directory path: also fail open.
    let orphan = f.base.join(format!("swe-wt-{}", unique("corrupt-orphan")));
    std::fs::create_dir_all(&orphan).unwrap();
    {
        let mut s = orphan.clone().into_os_string();
        s.push(".pid");
        std::fs::write(PathBuf::from(s), "not-a-pid").unwrap();
    }

    f.sweep();

    assert!(
        orphan.is_dir(),
        "orphan directory with a corrupt lease was pruned (fail-open expected)"
    );
}

/// Test 5 — a dead lease whose worktree was never registered loses both the
/// directory and the lease, regardless of `read_dir` ordering.
///
/// This is the regression test for the original leak: the `.pid` entry used to
/// be deleted by a separate branch of the loop, after which the directory
/// branch could no longer read it, failed open, and left the directory behind
/// forever. The sweep is run repeatedly to shake out any ordering dependence.
#[test]
fn orphan_directory_and_lease_are_removed_together_in_any_order() {
    let f = Fixture::new("orphan");
    for _ in 0..25 {
        let orphan = f.base.join(format!("swe-wt-{}", unique("orphan")));
        std::fs::create_dir_all(&orphan).unwrap();
        let mut pid_name = orphan.clone().into_os_string();
        pid_name.push(".pid");
        let pid_file = PathBuf::from(pid_name);
        f.write_dead_lease(&pid_file);

        f.sweep();

        assert!(
            !orphan.exists(),
            "orphan directory {} leaked: lease was removed but directory kept",
            orphan.display()
        );
        assert!(
            !pid_file.exists(),
            "lease {} leaked alongside its orphaned directory",
            pid_file.display()
        );
    }
}

/// A lease whose worktree directory is already gone protects nothing, so it
/// is reclaimed instead of accumulating forever.
#[test]
fn dangling_lease_without_directory_is_reclaimed() {
    let f = Fixture::new("dangling");
    let mut pid_name = f.base.join(format!("swe-wt-{}", unique("dangling")));
    pid_name.set_file_name({
        let mut s = pid_name.file_name().unwrap().to_os_string();
        s.push(".pid");
        s
    });
    f.write_dead_lease(&pid_name);
    assert!(!f.base.join(
        pid_name.file_stem().map(PathBuf::from).unwrap()
    ).exists());

    f.sweep();

    assert!(
        !pid_name.exists(),
        "lease {} outlived its missing worktree",
        pid_name.display()
    );
}

/// Test 6 — an unmerged `worker-*` branch without a worktree is preserved,
/// while a merged one is reclaimed.
#[test]
fn orphan_branch_pruning_respects_unmerged_commits() {
    let f = Fixture::new("branch");

    // Unmerged: a commit that exists only on the worker branch.
    let unmerged = format!("worker-{}", unique("unmerged"));
    run(&f.repo, &["checkout", "-b", &unmerged]);
    std::fs::write(f.repo.join("feature.txt"), "unmerged work\n").unwrap();
    run(&f.repo, &["add", "feature.txt"]);
    run(&f.repo, &["commit", "-m", "unmerged worker commit"]);
    run(&f.repo, &["checkout", "master"]);

    // Merged: points at HEAD, so deleting it cannot lose work.
    let merged = format!("worker-{}", unique("merged"));
    run(&f.repo, &["branch", &merged]);

    f.sweep();

    assert!(
        branch_exists(&f.repo, &unmerged),
        "unmerged worker branch {unmerged} was destroyed"
    );
    let log = run(&f.repo, &["log", "-1", "--pretty=%s", &unmerged]);
    assert!(
        log.contains("unmerged worker commit"),
        "unmerged branch lost its commit: {log}"
    );
    assert!(
        !branch_exists(&f.repo, &merged),
        "merged worker branch {merged} was not reclaimed"
    );
}

/// Test 7 — `keep = true` preserves the worktree, and the preserved worktree
/// also survives a later sweep (its lease is not left pointing at a PID that
/// will eventually die).
#[test]
fn kept_worktree_survives_its_own_drop_and_a_later_sweep() {
    let repo = std::env::temp_dir().join(unique("keep"));
    let _ = std::fs::remove_dir_all(&repo);
    std::fs::create_dir_all(&repo).unwrap();
    run(&repo, &["init", "-b", "master"]);
    run(&repo, &["config", "user.name", "mini-swe-test"]);
    run(&repo, &["config", "user.email", "test@localhost"]);
    std::fs::write(repo.join("README.md"), "# Baseline\n").unwrap();
    run(&repo, &["add", "README.md"]);
    run(&repo, &["commit", "-m", "Initial baseline commit"]);

    let (path, branch) = {
        let id = unique("keep");
        let mut guard = WorktreeGuard::new(&repo, &id).expect("worktree creation failed");
        guard.keep = true;
        std::fs::write(guard.path.join("work.txt"), "kept\n").unwrap();
        (guard.path.clone(), guard.branch.clone())
    };

    assert!(path.is_dir(), "keep = true still removed the worktree directory");
    assert!(branch_exists(&repo, &branch), "keep = true removed the branch");

    // The lease must be gone: leaving it would let a later sweep see a dead PID
    // and destroy the worktree the user asked to keep.
    let mut pid_name = path.clone().into_os_string();
    pid_name.push(".pid");
    let pid_file = PathBuf::from(pid_name);
    assert!(
        !pid_file.exists(),
        "kept worktree left a lease behind that would later be treated as a zombie"
    );

    // A later sweep, run with this repo's worktree listed but the keeper process
    // (the test binary) still alive, must still keep it.
    let base = path.parent().unwrap().to_path_buf();
    prune_stale_worktrees_in(&repo, std::slice::from_ref(&base));
    assert!(
        path.is_dir(),
        "kept worktree was pruned by a later sweep after its owner exited"
    );

    let _ = try_run(&repo, &["worktree", "remove", "--force", path.to_str().unwrap()]);
    let _ = try_run(&repo, &["branch", "-D", &branch]);
    let _ = std::fs::remove_dir_all(&repo);
}

fn branch_exists(repo: &Path, branch: &str) -> bool {
    Command::new("git")
        .current_dir(repo)
        .args(["rev-parse", "--verify", "--quiet", branch])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn worktree_registered(repo: &Path, path: &Path) -> bool {
    let list = run(repo, &["worktree", "list", "--porcelain"]);
    list.lines()
        .filter_map(|l| l.strip_prefix("worktree "))
        .any(|p| p == path.to_str().unwrap_or(""))
}

/// Regression for audit §02: a registered worktree whose directory was removed
/// out from under us (manual `rm -rf`, `tmpwatch`, CI cleanup, ...) must not be
/// treated as stale while its owning process is still alive, and its branch
/// must never be deleted with `branch -D` when it carries unmerged commits.
///
/// The old ordering evaluated `!wt_path.exists()` first and pruned the branch
/// unconditionally, destroying unmerged commits.
#[test]
fn live_worktree_whose_directory_vanished_keeps_its_unmerged_branch() {
    let f = Fixture::new("vanished");
    let (branch, dir, pid_file) = f.add_worktree(&unique("vanished"));
    f.write_live_lease(&pid_file);

    // Give the worker branch a commit that is not in master, then delete the
    // working directory from underneath git, exactly as an external cleaner
    // would.
    run(&dir, &["checkout", &branch]);
    std::fs::write(dir.join("precious.txt"), "unmerged worker output\n").unwrap();
    run(&dir, &["add", "precious.txt"]);
    run(&dir, &["commit", "-m", "unmerged work that must not be destroyed"]);
    std::fs::remove_dir_all(&dir).unwrap();
    assert!(!dir.exists(), "precondition: worktree directory is gone");

    f.sweep();

    assert!(
        branch_exists(&f.repo, &branch),
        "branch {branch} was destroyed even though its commits are unmerged"
    );
    let log = run(&f.repo, &["log", "-1", "--pretty=%s", &branch]);
    assert!(
        log.contains("unmerged work that must not be destroyed"),
        "unmerged commit was lost: {log}"
    );
}


/// Direct coverage of the §1 staleness decision, which the end-to-end sweep
/// cannot reach: `git worktree prune` at the top of the sweep unregisters a
/// worktree whose directory is already gone, so §2 ends up owning the branch
/// instead. Here the registration is irrelevant — only the ordering inside the
/// decision matters.
///
/// The old code evaluated `!wt_path.exists()` first and answered "stale"
/// without ever consulting the lease, so a live owner lost its worktree.
#[test]
fn staleness_decision_prefers_the_lease_over_the_missing_directory() {
    let f = Fixture::new("decision");
    let (_branch, dir, pid_file) = f.add_worktree(&unique("decision"));
    f.write_live_lease(&pid_file);

    // Directory present, owner alive -> in use.
    assert!(
        !worktree_is_stale_for_test(dir.to_str().unwrap()),
        "a worktree leased by a live process must not be considered stale"
    );

    // Directory removed out from under us, owner still alive -> still in use.
    std::fs::remove_dir_all(&dir).unwrap();
    assert!(
        !worktree_is_stale_for_test(dir.to_str().unwrap()),
        "a live lease must keep its worktree even when the directory is gone          (the old code returned true here and destroyed the branch)"
    );

    // Same, but the owner is gone -> genuinely abandoned.
    f.write_dead_lease(&pid_file);
    assert!(
        worktree_is_stale_for_test(dir.to_str().unwrap()),
        "a dead lease with no directory must be considered stale"
    );

    // No lease at all and no directory -> nothing to protect, reclaim it.
    std::fs::remove_file(&pid_file).unwrap();
    assert!(
        worktree_is_stale_for_test(dir.to_str().unwrap()),
        "an unleased worktree with no directory must be considered stale"
    );
}
