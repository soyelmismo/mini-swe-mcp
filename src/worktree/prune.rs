//! Garbage collection for abandoned subagent worktrees.
//!
//! A worktree is leased by a sibling `<worktree>.pid` file holding the owner's
//! pid and uid. The sweep consumes those leases to reclaim worktrees,
//! `worker-*` branches and `swe-target-*` scratch directories left behind by
//! crashed processes — *failing open* whenever a lease is missing or
//! unreadable, so an ambiguous marker can never destroy live work.
//!
//! Two properties underpin the whole sweep:
//!
//! * **Liveness means "able to do work".** A crashed owner stays visible to the
//!   kernel until reaped, surviving for minutes as a *zombie*: `/proc/<pid>`
//!   and `kill -0` both answer "alive" while nothing will ever touch the
//!   worktree again. [`is_process_alive`] therefore inspects
//!   `/proc/<pid>/status` and reports zombies as dead (audit §05).
//! * **Every reclaim is all-or-nothing and order-free.** An orphan worktree and
//!   its lease are removed as one atomic unit keyed by the directory itself, so
//!   the sweep converges to the same state no matter how `read_dir` orders the
//!   entries, and re-running it changes nothing (audit §01/§09).

use super::{force_remove_dir, git, pid_file_for, remove_target_dirs, swe_base_dirs};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use tracing::{error, info};

/// True when a process with `pid` is alive **and able to do work** on this host.
///
/// A crashed owner is not reaped instantly: until its parent runs `wait()` the
/// kernel keeps it as a *zombie*, and neither `/proc/<pid>` nor `kill -0` can
/// tell a zombie from a real process. Reporting such a lease as live pins its
/// worktree, branch and scratch directory for as long as the owner lingers. On
/// Linux the `State:` field of `/proc/<pid>/status` is the authority and `Z` is
/// answered as dead.
///
/// `/proc` is consulted first (cheapest, most reliable); `kill -0` is the
/// portable fallback and the authority where `/proc/<pid>/status` cannot be
/// read (restricted containers, `hidepid=2`, non-Linux kernels).
pub fn is_process_alive(pid: u32) -> bool {
    // pid 0 addresses the process group and is never a valid lease owner.
    if pid == 0 {
        return false;
    }

    #[cfg(target_os = "linux")]
    {
        match proc_state(pid) {
            // Readable `State:` field: trust it, so zombies are never mistaken
            // for live owners.
            Some(alive) => return alive,
            // Entry exists but `status` is unreadable (hidepid, racing exit).
            // The pid is allocated, so keep the worktree; `kill -0` decides.
            None => {
                if Path::new(&format!("/proc/{pid}")).exists() {
                    return kill_zero_says_alive(pid);
                }
            }
        }
    }

    kill_zero_says_alive(pid)
}

/// `kill -0` verdict, failing open towards keeping work.
///
/// When the check cannot be performed (no `kill` binary, no permission to
/// signal) liveness is unprovable, so `true` is returned: an unprovable death
/// is not a death. Only a successful `kill -0` reporting "no such process"
/// counts as dead.
#[cfg(unix)]
fn kill_zero_says_alive(pid: u32) -> bool {
    // `kill(pid, 0)` probes for the process without signalling it. `EPERM`
    // means the process exists but belongs to another user (still alive);
    // `ESRCH` means no such process. Any other error is unprovable, so fail
    // open towards keeping work.
    // SAFETY: `kill` takes no pointers; signal 0 only checks for existence.
    match unsafe { libc::kill(pid as libc::pid_t, 0) } {
        0 => true,
        -1 => {
            let err = std::io::Error::last_os_error();
            err.raw_os_error() != Some(libc::ESRCH)
        }
        _ => true,
    }
}

/// `kill -0` is unavailable off-unix; treat liveness as unprovable (keep).
#[cfg(not(unix))]
fn kill_zero_says_alive(_pid: u32) -> bool {
    true
}

/// `Some(bool)` when `/proc/<pid>/status` exposes the process state, `None`
/// when the process is not visible or its `status` cannot be read.
#[cfg(target_os = "linux")]
fn proc_state(pid: u32) -> Option<bool> {
    let status = std::fs::read_to_string(format!("/proc/{pid}/status")).ok()?;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("State:") {
            // `State:\tZ (zombie)`; the letter is always the first field.
            let code = rest.split_whitespace().next()?;
            // `Z` (zombie) and `X` (dead) can never do work again; all else keeps its lease.
            return Some(code != "Z" && code != "X");
        }
    }
    None
}

/// Owner identity recorded inside a `.pid` file so pruning refuses to touch
/// worktrees belonging to a different user (mitigates PID reuse across
/// accounts on shared hosts, audit §12).
fn current_uid() -> Option<u32> {
    #[cfg(unix)]
    {
        *CURRENT_UID.get_or_init(read_proc_uid)
    }
    #[cfg(not(unix))]
    {
        None
    }
}

#[cfg(unix)]
fn read_proc_uid() -> Option<u32> {
    // SAFETY: `getuid` takes no arguments and cannot fail.
    Some(unsafe { libc::getuid() })
}

static CURRENT_UID: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();

/// Contents written to a `<worktree>.pid` file: the owning process id followed
/// by the owning uid, one per line.
pub(super) fn pid_file_contents() -> String {
    match current_uid() {
        Some(uid) => format!("{}\n{uid}", std::process::id()),
        None => std::process::id().to_string(),
    }
}

/// Liveness decision for a `<worktree>.pid` file:
///
/// * `Some(true)`  -> owner provably gone, worktree abandoned;
/// * `Some(false)` -> owner alive, worktree in use;
/// * `None`        -> **no usable lease**; caller must fail open and keep.
///
/// `None` covers a missing, unreadable, malformed or foreign-owner marker. The
/// policy is deliberately "keep" in all four cases, applied identically by the
/// registered-worktree and orphan-directory paths: a bare-pid lease that the
/// kernel recycles freely must never destroy a live worktree, nor may a corrupt
/// file. A bare-pid lease stays readable (older builds wrote one) and is
/// answered by liveness alone; pid recycling cannot make it dangerous, because
/// a recycled pid is either alive (work kept) or dead (no owner left to lose
/// work). Dangling leases with no worktree are cleaned up separately by
/// `worktree_dir_of_pid_file`, a decision that cannot lose work
/// (audit §01/§03/§08/§12).
fn pid_file_is_stale(pid_file: &Path) -> Option<bool> {
    let content = std::fs::read_to_string(pid_file).ok()?;
    let mut lines = content.lines().map(str::trim);
    let pid: u32 = lines.next()?.parse().ok()?;
    if pid == 0 {
        return None; // process-group sentinel, never a valid owner
    }

    // A lease recorded for another uid is not ours to act on — and neither is
    // one whose owner line we cannot read. Accepting an unparseable line would
    // silently downgrade a truncated or tampered lease to the weaker "bare pid"
    // form and hand it authority it was never granted.
    if let Some(ours) = current_uid()
        && let Some(uid) = lines.next()
        && !matches!(uid.parse::<u32>(), Ok(uid) if uid == ours)
    {
        return None;
    }

    Some(!is_process_alive(pid))
}

/// The worktree directory a `swe-wt-*.pid` lease belongs to, if it exists.
fn worktree_dir_of_pid_file(pid_file: &Path) -> Option<PathBuf> {
    let name = pid_file.file_name()?.to_str()?;
    let wt_name = name.strip_suffix(".pid")?;
    let base = pid_file.parent()?;
    let dir = base.join(wt_name);
    dir.is_dir().then_some(dir)
}

/// True when `name` is a scratch worktree directory created by this crate.
fn is_worktree_dir_name(name: &str) -> bool {
    name.starts_with("swe-wt-")
}

/// True when `name` is the lease belonging to a scratch worktree directory
/// (`swe-wt-*.pid`), whether or not that directory currently exists.
fn is_worktree_lease_name(name: &str) -> bool {
    name.starts_with("swe-wt-")
        && name.ends_with(".pid")
        && name.len() > "swe-wt-".len() + ".pid".len()
}

/// Reclaim one abandoned scratch worktree as an **atomic, idempotent unit**:
/// the worktree directory, its lease and its scratch target directory all go,
/// or none of them do.
///
/// Atomicity matters because the sweep walks `read_dir` output while other
/// subagents keep creating and dropping entries in the same base directory. The
/// decision is taken from a single source — the worktree directory's own lease
/// — *before* the first destructive step, so a worktree is never observed
/// half-removed and `read_dir` order cannot change the outcome
/// (audit §01/§09). Idempotence follows: every step tolerates an already-missing
/// path, so re-running the sweep is a no-op.
///
/// Fails open: only `Some(true)` — a readable lease naming a process that can
/// no longer do work — authorizes the removal.
fn reclaim_abandoned_worktree(dir: &Path) -> bool {
    let pid_file = pid_file_for(dir);
    if pid_file_is_stale(&pid_file) != Some(true) {
        return false;
    }
    info!(path = %dir.display(), "Pruning orphaned worktree directory");
    // Claim the lease by renaming it to a private name before destroying
    // anything. `rename` within a directory is atomic, so of N concurrent
    // sweeps exactly one observes a successful claim and only that one
    // proceeds; the losers get `NotFound` and leave the entry alone. Without
    // this, two sweeps could both read a live-looking lease, both decide
    // "stale", and interleave their removals into a half-reclaimed worktree.
    //
    // Claiming before the directory is removed keeps the unit all-or-nothing
    // for *readers* too: the unleased window a concurrent sweep would otherwise
    // see (and fail open on, stranding the directory forever — the leak audit
    // §01 was written about) is bounded to a single `rename`, and the reclaiming
    // sweep is already committed to finishing.
    let Ok(claimed) = claim_lease(&pid_file) else {
        // Another sweep claimed this lease first, or it vanished under us;
        // either way the worktree is that sweep's to finish, not ours.
        return false;
    };
    // The lease may have been replaced between the initial read and the claim.
    // Only the claimed contents authorize deletion; never overwrite a lease
    // created concurrently while restoring a changed one.
    if pid_file_is_stale(&claimed) != Some(true) || pid_file.exists() {
        if !pid_file.exists() {
            let _ = std::fs::hard_link(&claimed, &pid_file);
        }
        let _ = std::fs::remove_file(&claimed);
        return false;
    }
    // The reclaim is now ours alone. The claimed copy is dropped afterwards.
    salvage_dirty_worktree(dir);
    force_remove_dir(dir);
    remove_target_dirs(dir);
    let _ = std::fs::remove_file(&claimed);
    true
}

/// Commit any uncommitted changes in a dead worker's worktree onto the branch
/// it has checked out, so an interrupted worker's work survives the prune.
///
/// Both prune paths call this before deleting anything. A registered worktree
/// is on its own `worker-<id>` branch; the salvage commit makes that branch
/// unmerged, so the branch sweep preserves it. An orphan directory whose git
/// metadata is already gone cannot be committed and is skipped (the status
/// probe fails). Fallback credentials match `guard::commit_changes`.
pub(crate) fn salvage_dirty_worktree(dir: &Path) -> bool {
    let Ok(status) = git(dir, "status --porcelain", &["status", "--porcelain"]) else {
        return false;
    };
    if !status.status.success() {
        return false;
    }
    if status.stdout.is_empty() {
        return true;
    }
    let id = dir
        .file_name()
        .and_then(|n| n.to_str())
        .map(|n| n.strip_prefix("swe-wt-").unwrap_or(n))
        .unwrap_or("unknown");
    let msg = format!("worker({id}): salvaged uncommitted work before prune");
    if !git(dir, "add -A", &["add", "-A"]).is_ok_and(|out| out.status.success()) {
        return false;
    }
    let committed = git(
        dir,
        "commit",
        &[
            "-c",
            "user.name=mini-swe",
            "-c",
            "user.email=mini-swe@localhost",
            "commit",
            "-m",
            &msg,
        ],
    );
    match committed {
        Ok(out) if out.status.success() => {
            info!(path = %dir.display(), "Salvaged uncommitted worker changes before prune");
            true
        }
        _ => {
            error!(path = %dir.display(), "Could not salvage uncommitted worker changes");
            false
        }
    }
}

/// Atomically take exclusive ownership of a lease by renaming it aside.
///
/// Returns the claimed path on success. Any error — most importantly
/// `NotFound`, meaning another sweep claimed it first — leaves the original
/// lease untouched and tells the caller to abandon the reclaim.
fn claim_lease(pid_file: &Path) -> std::io::Result<PathBuf> {
    let Some(parent) = pid_file.parent() else {
        return Err(std::io::ErrorKind::NotFound.into());
    };
    let claimed = parent.join(format!(
        ".swe-wt-lease-claimed.{}.{}",
        std::process::id(),
        CLAIM_COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::rename(pid_file, &claimed)?;
    Ok(claimed)
}

/// Process-unique counter backing [`claim_lease`], so two sweeps in the same
/// process never pick the same claimed-lease name.
static CLAIM_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Test seam: the atomic lease claim itself. A race between two concurrent
/// sweeps cannot be provoked from a single-threaded test, but the property that
/// makes it safe — the claim succeeds exactly once — is directly observable.
#[doc(hidden)]
pub fn claim_lease_for_test(pid_file: &Path) -> std::io::Result<PathBuf> {
    claim_lease(pid_file)
}

/// Reclaim leases that describe nothing: a `swe-wt-*.pid` whose worktree
/// directory no longer exists. They protect no work, cannot lose data, and are
/// handled independently of the directory walk so ordering between the two
/// entry kinds is irrelevant.
fn reclaim_dangling_lease(lease: &Path) {
    if worktree_dir_of_pid_file(lease).is_none() {
        let _ = std::fs::remove_file(lease);
    }
}

/// Remove one `worker-*` worktree: its git registration, its directory, its
/// sibling `.pid` file and its scratch target directory.
///
/// The branch is deleted only when it is merged into `HEAD` (or carries no
/// commits beyond `HEAD`), so unmerged work is never destroyed (audit §02).
fn remove_worker_worktree(repo_root: &Path, wt: &str, br: &str) {
    let wt_path = Path::new(wt);
    let pid_file = pid_file_for(wt_path);

    salvage_dirty_worktree(wt_path);
    let _ = git(
        repo_root,
        "worktree remove",
        &["worktree", "remove", "--force", wt],
    );
    if is_branch_merged(repo_root, br) {
        let _ = git(repo_root, "branch -D", &["branch", "-D", br]);
    } else {
        info!(branch = %br, "Preserving unmerged worker branch with commits");
    }
    force_remove_dir(wt_path);
    let _ = std::fs::remove_file(&pid_file);
    remove_target_dirs(wt_path);
    // The worker is gone past review: retire its saved conversation (and its
    // steering mailbox, which the finished run's guard may never have dropped
    // on a crash) from every scratch base.
    if let Some(name) = wt_path.file_name().and_then(|n| n.to_str())
        && let Some(id) = name.strip_prefix("swe-wt-")
    {
        crate::pool::remove_worker_history(id);
        crate::pool::remove_steer_file(id);
    }
}

/// True when `branch` has no commits missing from `HEAD`, i.e. deleting it
/// cannot lose work. Unreadable/absent branches count as safe to delete.
fn is_branch_merged(repo_root: &Path, branch: &str) -> bool {
    git(
        repo_root,
        "merge-base",
        &["merge-base", "--is-ancestor", branch, "HEAD"],
    )
    .map(|o| o.status.success())
    .unwrap_or(false)
}

/// Decide whether the registered worktree `wt` / branch `br` is abandoned.
///
/// Process liveness is checked **before** the on-disk directory: a worktree
/// whose owner is still running must never be destroyed just because something
/// else removed its directory (audit §02). A missing or unreadable `.pid`
/// fails open, a malformed or foreign-owned marker fails closed.
fn registered_worktree_is_stale(wt: &str) -> bool {
    let wt_path = Path::new(wt);
    match pid_file_is_stale(&pid_file_for(wt_path)) {
        // Owner gone (or marker unusable) -> abandoned regardless of directory.
        Some(stale) => stale,
        // No marker at all: fall back to the directory. Without a lease there
        // is nothing to protect, and an unregistered-but-present directory is
        // almost always a leftover from a crashed run.
        None => !wt_path.exists(),
    }
}

/// Handle one entry of `git worktree list --porcelain`.
///
/// Stale entries are removed together with their branch; live ones are recorded
/// in `active_branches` so the orphan-branch sweep leaves them alone.
fn prune_worktree_if_stale(
    repo_root: &Path,
    wt: &str,
    br: &str,
    active_branches: &mut Vec<String>,
) {
    if !br.starts_with("worker-") {
        return;
    }

    if registered_worktree_is_stale(wt) {
        info!(path = %wt, branch = %br, "Pruning zombie subagent worktree");
        remove_worker_worktree(repo_root, wt, br);
    } else {
        active_branches.push(br.to_string());
    }
}

/// Exposed for tests: the liveness verdict for a registered worktree path.
///
/// The end-to-end sweep cannot reach this decision for a worktree whose
/// directory was deleted, because `git worktree prune` unregisters it first;
/// this seam lets the ordering contract be asserted directly.
#[doc(hidden)]
pub fn worktree_is_stale_for_test(wt: &str) -> bool {
    registered_worktree_is_stale(wt)
}

/// Prune abandoned subagent worktrees registered with `repo_root`, using the
/// default scratch base directories (`swe_base_dirs`).
pub fn prune_stale_worktrees(repo_root: &Path) {
    prune_stale_worktrees_in(repo_root, &swe_base_dirs());
}

/// Test seam: same sweep as [`prune_stale_worktrees`] but with an explicit set
/// of scratch base directories, so tests never have to mutate `SWE_TEMP_DIR`
/// (which is process-global and therefore racy across `cargo test` threads).
///
/// The sweep is a single pass over each source, with exactly one
/// `git worktree prune` at the start (audit §06/§07):
///   1. registered `worker-*` worktrees whose owner process is dead,
///   2. `worker-*` branches with no worktree at all (merged ones only),
///   3. unregistered `swe-wt-*` directories in the scratch base dirs,
///   4. `swe-target-*` scratch dirs whose worktree is gone.
pub fn prune_stale_worktrees_in(repo_root: &Path, base_dirs: &[PathBuf]) {
    let _ = git(repo_root, "worktree prune", &["worktree", "prune"]);

    // 1. Prune registered worktrees whose owner process is dead or directory is missing.
    let mut active_branches = Vec::new();
    if let Ok(output) = git(
        repo_root,
        "worktree list",
        &["worktree", "list", "--porcelain"],
    ) {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut current_wt: Option<String> = None;
        let mut current_branch: Option<String> = None;
        for line in stdout.lines() {
            if let Some(path) = line.strip_prefix("worktree ") {
                current_wt = Some(path.to_string());
            } else if let Some(branch) = line.strip_prefix("branch refs/heads/") {
                current_branch = Some(branch.to_string());
            } else if line.is_empty()
                && let (Some(wt), Some(br)) = (current_wt.take(), current_branch.take())
            {
                prune_worktree_if_stale(repo_root, &wt, &br, &mut active_branches);
            }
        }
        if let (Some(wt), Some(br)) = (current_wt, current_branch) {
            prune_worktree_if_stale(repo_root, &wt, &br, &mut active_branches);
        }
    }

    // 2. Prune orphaned worker-* branches with no registered worktrees AND merged into HEAD.
    if let Ok(output) = git(
        repo_root,
        "branch --list",
        &["branch", "--list", "worker-*"],
    ) {
        let stdout = String::from_utf8_lossy(&output.stdout);
        for line in stdout.lines() {
            let branch = line
                .trim()
                .trim_start_matches('*')
                .trim_start_matches('+')
                .trim();
            if branch.starts_with("worker-") && !active_branches.iter().any(|b| b == branch) {
                if is_branch_merged(repo_root, branch) {
                    info!(branch = %branch, "Pruning merged or empty worker branch");
                    let _ = git(repo_root, "branch -D", &["branch", "-D", branch]);
                } else {
                    info!(branch = %branch, "Preserving unmerged worker branch with commits");
                }
            }
        }
    }

    for base in base_dirs {
        // Stream one directory walk instead of retaining the entire (possibly
        // huge) system temp directory in memory. A target encountered before
        // its worktree is handled when that worktree is reclaimed below.
        if let Ok(entries) = std::fs::read_dir(base) {
            for entry in entries.flatten() {
                let p = entry.path();
                let Some(name) = p.file_name().and_then(|n| n.to_str()) else {
                    error!(path = %p.display(), "Skipping non-UTF-8 scratch entry");
                    continue;
                };
                let is_dir = entry
                    .file_type()
                    .map(|t| t.is_dir())
                    .unwrap_or_else(|_| p.is_dir());
                if is_dir && is_worktree_dir_name(name) {
                    if reclaim_abandoned_worktree(&p) {
                        // `base_dirs` can be supplied independently of the default
                        // scratch bases, so clean the matching target here too.
                        force_remove_dir(&base.join(format!("swe-target-{name}")));
                    }
                } else if !is_dir && is_worktree_lease_name(name) {
                    reclaim_dangling_lease(&p);
                } else if is_dir
                    && let Some(wt_name) = name.strip_prefix("swe-target-")
                    && wt_name.starts_with("swe-wt-")
                    && !base.join(wt_name).exists()
                {
                    force_remove_dir(&p);
                }
            }
        }
    }

    // Post-pass prune: phase 1 may have removed worktrees, leaving new stale
    // administrative entries; this second pass clears them. Together the pre-
    // and post-passes form a legitimate double pass (pre collects old garbage,
    // post collects what we just removed) — both are needed.
    let _ = git(repo_root, "worktree prune", &["worktree", "prune"]);
}
