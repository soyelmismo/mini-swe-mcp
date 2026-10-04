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
    claim_lease_for_test, is_process_alive, prune_stale_worktrees_in, worktree_is_stale_for_test,
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
            let _ = try_run(
                &self.repo,
                &["worktree", "remove", "--force", dir.to_str().unwrap_or("")],
            );
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

    assert!(
        !dir.exists(),
        "zombie worktree directory survived the sweep"
    );
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
    assert!(
        branch_exists(&f.repo, &branch),
        "branch of unleased worktree was pruned"
    );
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
    assert!(
        branch_exists(&f.repo, &branch),
        "corrupt-lease branch was pruned"
    );

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
    assert!(
        !f.base
            .join(pid_name.file_stem().map(PathBuf::from).unwrap())
            .exists()
    );

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

/// Test 7 — a registered worktree whose owner died with uncommitted work is
/// salvaged before the prune: the change is committed onto its own
/// `worker-<id>` branch, which is then preserved, while the directory goes.
#[test]
fn dirty_worktree_is_salvaged_before_prune() {
    let f = Fixture::new("salvage");
    let id = unique("salvage");
    let dir = f.base.join(format!("swe-wt-{id}"));
    let branch = format!("worker-{id}");
    let pid_file = {
        let mut s = dir.clone().into_os_string();
        s.push(".pid");
        PathBuf::from(s)
    };
    run(
        &f.repo,
        &[
            "worktree",
            "add",
            "-b",
            &branch,
            dir.to_str().unwrap(),
            "HEAD",
        ],
    );
    f.write_dead_lease(&pid_file);
    std::fs::write(dir.join("precious.txt"), "uncommitted worker output\n").unwrap();

    f.sweep();

    assert!(!dir.exists(), "dirty worktree directory survived the sweep");
    assert!(
        branch_exists(&f.repo, &branch),
        "salvaged branch {branch} was destroyed"
    );
    let log = run(&f.repo, &["log", "-1", "--pretty=%s", &branch]);
    assert!(
        log.contains("salvaged uncommitted work before prune"),
        "salvage commit missing: {log}"
    );
    let body = run(&f.repo, &["show", &format!("{branch}:precious.txt")]);
    assert!(
        body.contains("uncommitted worker output"),
        "salvaged change was not committed: {body}"
    );
}

/// A dead worker's worktree is salvaged through the same gate as every other
/// harness commit: the cache and the oversized file its tooling left behind die
/// with the directory, while its real edit is committed onto the branch the
/// sweep preserves.
#[test]
fn salvage_leaves_cache_and_oversized_paths_out_of_the_commit() {
    let f = Fixture::new("salvage-gate");
    let id = unique("salvage-gate");
    let dir = f.base.join(format!("swe-wt-{id}"));
    let branch = format!("worker-{id}");
    let pid_file = {
        let mut s = dir.clone().into_os_string();
        s.push(".pid");
        PathBuf::from(s)
    };
    run(
        &f.repo,
        &[
            "worktree",
            "add",
            "-b",
            &branch,
            dir.to_str().unwrap(),
            "HEAD",
        ],
    );
    f.write_dead_lease(&pid_file);
    std::fs::write(dir.join("edit.rs"), "fn kept() {}\n").unwrap();
    std::fs::write(dir.join("big.bin"), vec![0u8; 20 * 1024 * 1024]).unwrap();
    let home = dir.join(".envcheck/home");
    std::fs::create_dir_all(home.join(".cache/kache/store")).unwrap();
    std::fs::write(home.join(".cache/kache/store/blob"), "cache payload\n").unwrap();

    f.sweep();

    assert!(!dir.exists(), "the worktree directory survived the sweep");
    assert!(
        branch_exists(&f.repo, &branch),
        "salvaged branch {branch} was destroyed"
    );
    let committed = run(&f.repo, &["ls-tree", "-r", "--name-only", &branch]);
    assert!(
        committed.contains("edit.rs"),
        "the worker's edit was not salvaged: {committed:?}"
    );
    assert!(
        !committed.contains("big.bin"),
        "an oversized file was salvaged: {committed:?}"
    );
    assert!(
        !committed.contains(".envcheck"),
        "a cache path was salvaged: {committed:?}"
    );
    // The worktree shared the repository's object database, so a refused file
    // must not have left a large blob behind for a later push to trip over.
    let objects = run(&f.repo, &["cat-file", "--batch-all-objects", "--batch-check"]);
    assert!(
        !objects.lines().any(|line| {
            let size = line.split_whitespace().nth(2).unwrap_or("0");
            size.parse::<u64>().unwrap_or(0) > 1024 * 1024
        }),
        "a large blob reached the object database: {objects}"
    );
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
    run(
        &dir,
        &["commit", "-m", "unmerged work that must not be destroyed"],
    );
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

/// A process that has exited but not yet been reaped is a **zombie**: the
/// kernel keeps every per-process resource (`/proc/<pid>`, the pid slot, the
/// exit status) alive, so both the `/proc/<pid>` presence check and `kill -0`
/// happily report a process that can never read or write its worktree again.
/// The lease of such an owner must be treated as dead, or the abandoned
/// worktree, its branch and its target directory are pinned until the pid slot
/// is finally recycled.
#[cfg(target_os = "linux")]
fn spawn_zombie() -> Option<i32> {
    use std::process::Stdio;

    // `--list` makes the test binary enumerate its tests and exit immediately:
    // a short-lived child, with no shell and no other dependency.
    let exe = std::env::current_exe().ok()?;
    let child = Command::new(exe)
        .arg("--list")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    let pid = child.id() as i32;
    // `std::process::Child` does not reap on drop, so the child stays in state
    // `Z` (unreaped) and keeps both its `/proc` entry and its pid slot.
    std::mem::forget(child);

    // Wait for the child to actually exit; only then is it a zombie.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    loop {
        if let Ok(status) = std::fs::read_to_string(format!("/proc/{pid}/status")) {
            if status
                .lines()
                .any(|l| l.trim_start().starts_with("State:") && l.contains("(zombie)"))
            {
                return Some(pid);
            }
        } else {
            return None; // reaped or vanished: no zombie to test
        }
        if std::time::Instant::now() >= deadline {
            return None;
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
}

/// Placeholder kept so the binary re-executed by [`spawn_zombie`] has a body to
/// enumerate; `--list` runs no test at all.
#[test]
#[ignore = "not a real test; only enumerated by --list"]
fn __zombie_placeholder__() {}

#[test]
#[cfg(target_os = "linux")]
fn zombie_owner_is_treated_as_dead() {
    let Some(zpid) = spawn_zombie() else {
        panic!("failed to create a real zombie process on this host");
    };
    let zpid_u32 = u32::try_from(zpid).expect("positive pid");

    // `/proc/<pid>` exists, so the old presence-only check reported "alive".
    assert!(
        Path::new(&format!("/proc/{zpid}")).exists(),
        "precondition: the zombie still has a /proc entry"
    );
    assert!(
        !is_process_alive(zpid_u32),
        "a zombie owner must not count as alive (it can never touch the worktree)"
    );
    assert!(
        is_process_alive(std::process::id()),
        "a real, running process must still count as alive"
    );
}

/// End-to-end: an orphan worktree whose lease names a **zombie** owner is
/// reclaimed, together with its lease and its scratch target directory.
#[test]
#[cfg(target_os = "linux")]
fn zombie_leased_orphan_is_reclaimed() {
    let Some(zpid) = spawn_zombie() else {
        panic!("failed to create a real zombie process on this host");
    };
    let f = Fixture::new("zombie");

    let orphan = f.base.join(format!("swe-wt-{}", unique("zombie")));
    let pid_file = {
        let mut s = orphan.clone().into_os_string();
        s.push(".pid");
        PathBuf::from(s)
    };
    std::fs::create_dir_all(&orphan).unwrap();
    std::fs::write(&pid_file, zpid.to_string()).unwrap();
    let target_dir = f.base.join(format!(
        "swe-target-{}",
        orphan.file_name().unwrap().to_str().unwrap()
    ));
    std::fs::create_dir_all(&target_dir).unwrap();

    f.sweep();

    assert!(!orphan.exists(), "zombie-leased orphan directory survived");
    assert!(!pid_file.exists(), "zombie lease survived its directory");
    assert!(
        !target_dir.exists(),
        "scratch target directory of a zombie-leased worktree survived"
    );

    // Idempotent: a second sweep has nothing left to do and changes nothing.
    f.sweep();
    assert!(!orphan.exists(), "second sweep resurrected nothing");
}

/// The orphan sweep must be idempotent: repeated sweeps converge to the same
/// state and never leave a half-removed directory behind, however many times it
/// is run.
#[test]
fn orphan_sweep_is_idempotent() {
    let f = Fixture::new("idem");

    let mut orphans = Vec::new();
    for i in 0..5 {
        let orphan = f.base.join(format!("swe-wt-{}", unique("idem")));
        let pid_file = {
            let mut s = orphan.clone().into_os_string();
            s.push(".pid");
            PathBuf::from(s)
        };
        std::fs::create_dir_all(&orphan).unwrap();
        f.write_dead_lease(&pid_file);
        orphans.push((orphan.clone(), pid_file.clone(), i));
    }

    // Run the sweep several times: the first pass reclaims, later passes are
    // no-ops that must neither error nor resurrect anything.
    for pass in 0..4 {
        f.sweep();
        for (dir, pid_file, _) in &orphans {
            assert!(
                !dir.exists(),
                "pass {pass}: orphan {} came back",
                dir.display()
            );
            assert!(
                !pid_file.exists(),
                "pass {pass}: lease {} came back",
                pid_file.display()
            );
        }
    }
}

/// A lease carrying a *malformed* owner line (a truncated write, or tampering)
/// must fail open. It must never be silently downgraded to the weaker "bare pid"
/// form and obeyed: an unreadable uid is exactly the case where we cannot tell a
/// stale lease from a foreign one, so the sweep must keep the worktree.
#[test]
fn lease_with_malformed_owner_line_is_refused() {
    let f = Fixture::new("malformed");
    let (_branch, dir, pid_file) = f.add_worktree(&unique("malformed"));

    // A DEAD pid, but with a garbage owner line: the stale decision must be
    // withheld because the lease is not trustworthy, and the worktree kept.
    std::fs::write(&pid_file, format!("{}\nnot-a-uid", Fixture::DEAD_PID)).unwrap();

    assert!(
        !worktree_is_stale_for_test(dir.to_str().unwrap()),
        "a lease with a malformed owner line must not authorise removal"
    );

    f.sweep();
    assert!(
        dir.is_dir(),
        "malformed-owner lease destroyed the worktree it does not validly describe"
    );
    assert!(pid_file.exists(), "malformed lease was removed anyway");
}

/// The atomic claim succeeds **exactly once**, so of two concurrent sweeps
/// racing over the same abandoned worktree precisely one may proceed with the
/// destructive steps. This is the property that makes a reclaim all-or-nothing
/// instead of two sweeps interleaving their directory/lease/target removals into
/// a half-reclaimed worktree.
///
/// The race itself cannot be provoked from a single-threaded test, so the
/// primitive is exercised directly: the first claim wins, the second is refused,
/// and the loser neither duplicates nor destroys the winner's claimed lease.
#[test]
fn lease_claim_succeeds_exactly_once() {
    let f = Fixture::new("claim");
    let lease = f.base.join(format!("swe-wt-{}.pid", unique("claim")));
    f.write_dead_lease(&lease);

    let claimed = claim_lease_for_test(&lease).expect("first claim must win");
    assert!(claimed.exists(), "claim did not move the lease aside");
    assert!(
        !lease.exists(),
        "claiming the lease must take it out of the sweep's namespace"
    );
    // The claimed name is inert: it is not a `swe-wt-` directory, so no later
    // sweep's name-based classifier can mistake it for a worktree or a lease.
    let name = claimed.file_name().unwrap().to_string_lossy().into_owned();
    assert!(
        name.starts_with(".swe-wt-lease-claimed."),
        "claimed lease must be hidden from the name-based classifier, got {name}"
    );

    // A second concurrent sweep loses the election and must not touch anything.
    assert!(
        claim_lease_for_test(&lease).is_err(),
        "a second sweep must not be able to claim the same lease"
    );
    assert!(
        claimed.exists(),
        "the losing claim destroyed the winner's claimed lease"
    );

    let _ = std::fs::remove_file(&claimed);
}

/// The claim leaves no debris in the shared scratch directory: the claimed copy
/// is dropped once the reclaim finishes, so repeated sweeps do not litter a base
/// dir that every other subagent on the host also writes into.
#[test]
fn reclaim_leaves_no_claim_debris() {
    let f = Fixture::new("debris");
    let orphan = f.base.join(format!("swe-wt-{}", unique("debris")));
    let mut pid_name = orphan.clone().into_os_string();
    pid_name.push(".pid");
    let pid_file = PathBuf::from(pid_name);
    std::fs::create_dir_all(&orphan).unwrap();
    f.write_dead_lease(&pid_file);

    f.sweep();
    f.sweep();

    assert!(!orphan.exists(), "orphan survived two sweeps");
    assert!(!pid_file.exists(), "lease survived two sweeps");
    let leftovers: Vec<String> = std::fs::read_dir(&f.base)
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.contains("claimed"))
        .collect();
    assert!(
        leftovers.is_empty(),
        "claim debris left in the base dir: {leftovers:?}"
    );
}

/// The target's name may sort before its worktree's name in the directory
/// stream. Reclamation must remove both regardless of encounter order.
#[test]
fn orphan_removes_target_in_custom_base() {
    let f = Fixture::new("target-order");
    let orphan = f.base.join(format!("swe-wt-{}", unique("target-order")));
    let target = f.base.join(format!(
        "swe-target-{}",
        orphan.file_name().unwrap().to_str().unwrap()
    ));
    let mut name = orphan.clone().into_os_string();
    name.push(".pid");
    std::fs::create_dir_all(&orphan).unwrap();
    std::fs::create_dir_all(&target).unwrap();
    f.write_dead_lease(&PathBuf::from(name));
    f.sweep();
    assert!(!orphan.exists());
    assert!(!target.exists());
}
