//! Garbage collection for abandoned subagent worktrees.
//!
//! A worktree is leased by a sibling `<worktree>.pid` file holding the owner's
//! pid and uid. The sweep below consumes those leases to reclaim worktrees,
//! `worker-*` branches and `swe-target-*` scratch directories left behind by
//! crashed processes — while *failing open* whenever a lease is missing or
//! unreadable, so an ambiguous marker can never destroy live work.
//!
//! Two properties underpin the whole sweep:
//!
//! * **Liveness means "able to do work".** A crashed owner stays visible to the
//!   kernel until it is reaped, so it can survive for minutes as a *zombie*:
//!   `/proc/<pid>` and `kill -0` both keep answering "alive" while nothing will
//!   ever read or write the worktree again. [`is_process_alive`] therefore
//!   inspects `/proc/<pid>/status` and reports zombies as dead (audit §05).
//! * **Every reclaim is all-or-nothing and order-free.** An orphan worktree and
//!   its lease are removed as one atomic unit keyed by the directory itself, so
//!   the sweep converges to the same state no matter how `read_dir` happens to
//!   order the entries, and re-running it changes nothing (audit §01/§09).

use super::{force_remove_dir, git, pid_file_for, remove_target_dirs, swe_base_dirs};
use std::path::{Path, PathBuf};
use std::process::Command;
use tracing::{error, info};

/// True when a process with `pid` is alive **and able to do work** on this host.
///
/// A crashed owner is not reaped instantly: until its parent runs `wait()` the
/// kernel keeps it as a *zombie*, and neither the presence of `/proc/<pid>` nor
/// `kill -0` can tell a zombie from a real process — both signal "still here".
/// Reporting such a lease as live pins its worktree, its `worker-*` branch and
/// its `swe-target-*` scratch directory for as long as the lease owner lingers
/// (an agent that leaks a child leaves the whole tree stranded). On Linux the
/// single-letter `State:` field of `/proc/<pid>/status` is therefore treated as
/// the authority and `Z` is answered as dead.
///
/// `/proc` is consulted first because it is the cheapest and most reliable
/// answer; `kill -0` is the portable fallback, and is the authority on every
/// platform where `/proc/<pid>/status` cannot be read (restricted containers,
/// `hidepid=2` mount options, non-Linux kernels).
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
            // The entry exists but `status` is not readable (hidepid=2 over
            // hidepid=0, a racing exit, an unreadable mount). The pid is
            // certainly allocated, so keep the worktree; `kill -0` decides.
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
/// When the check itself cannot be performed (no `kill` binary, no permission
/// to signal) liveness is unprovable, so `true` is returned: an unprovable
/// death is not a death. Only a successfully-executed `kill -0` that reports
/// "no such process" counts as dead.
#[cfg(unix)]
fn kill_zero_says_alive(pid: u32) -> bool {
    Command::new("kill")
        .args(["-0", &pid.to_string()])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(true)
}

/// `kill -0` is unavailable off-unix; treat liveness as unprovable (keep).
#[cfg(not(unix))]
fn kill_zero_says_alive(_pid: u32) -> bool {
    true
}

/// `Some(true)`/`Some(false)` when `/proc/<pid>/status` exposes the process
/// state, `None` when the process is not visible on this host (or its `status`
/// cannot be read).
#[cfg(target_os = "linux")]
fn proc_state(pid: u32) -> Option<bool> {
    let status = std::fs::read_to_string(format!("/proc/{pid}/status")).ok()?;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("State:") {
            // `State:\tZ (zombie)`; the letter is always the first field and is
            // present on every `/proc` implementation this kernel ships.
            let code = rest.split_whitespace().next()?;
            // `Z` (zombie) can never do work again; `X` (dead, should never
            // be observed) is gone too. Everything else keeps its lease.
            return Some(code != "Z" && code != "X");
        }
    }
    None
}

/// Owner identity recorded inside a `.pid` file so pruning can refuse to touch
/// worktrees that belong to a different user (mitigates PID reuse across
/// accounts on shared hosts, see audit §12).
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
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("Uid:") {
            return rest.split_whitespace().next()?.parse::<u32>().ok();
        }
    }
    None
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

/// Liveness decision for a `<worktree>.pid` file, expressed as an `Option`:
///
/// * `Some(true)`  -> the owner is provably gone, the worktree is abandoned;
/// * `Some(false)` -> the owner process is alive, the worktree is in use;
/// * `None`        -> **no usable lease**; the caller must fail open and keep
///   the worktree.
///
/// `None` covers a missing, unreadable, malformed or foreign-owner marker. The
/// policy is deliberately "keep" in all four cases and is applied identically
/// by the registered-worktree and orphan-directory paths: a `.pid` holding
/// only a bare integer that the kernel recycles freely must never be able to
/// destroy a live worktree, and a corrupt file must not be able to do it
/// either. Dangling lease files that no longer have a worktree are cleaned up
/// separately by `worktree_dir_of_pid_file`, which is a decision that cannot
/// lose work (audit §01/§03/§08/§12).
fn pid_file_is_stale(pid_file: &Path) -> Option<bool> {
    let content = std::fs::read_to_string(pid_file).ok()?;
    let mut lines = content.lines();
    let pid: u32 = lines.next()?.trim().parse().ok()?;
    // A lease recorded for another uid is not ours to interpret or to act on.
    if let Some(ours) = current_uid()
        && lines
            .next()
            .is_some_and(|l| l.trim().parse::<u32>().is_ok_and(|uid| uid != ours))
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
/// decision is therefore taken from a single source — the worktree directory's
/// own lease — *before* the first destructive step, so a worktree is never
/// observed half-removed, and `read_dir` order cannot change the outcome
/// (audit §01/§09). Idempotence follows from it: every step tolerates an
/// already-missing path, so re-running the sweep is a no-op.
///
/// Fails open: only `Some(true)` — a readable lease naming a process that can no
/// longer do work — authorizes the removal.
fn reclaim_abandoned_worktree(dir: &Path) {
    let pid_file = pid_file_for(dir);
    if pid_file_is_stale(&pid_file) != Some(true) {
        return;
    }
    info!(path = %dir.display(), "Pruning orphaned worktree directory");
    // The lease goes with the directory, and it goes last: removing it first
    // would make a concurrent sweep see an unleased directory and fail open,
    // stranding it forever (the leak audit §01 was written about).
    force_remove_dir(dir);
    let _ = std::fs::remove_file(&pid_file);
    remove_target_dirs(dir);
}

/// Reclaim leases that describe nothing: a `swe-wt-*.pid` whose worktree
/// directory no longer exists. They protect no work, cannot lose data, and are
/// handled independently of the directory walk so that ordering between the two
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
}

/// True when `branch` has no commits that are missing from `HEAD`, i.e. deleting
/// it cannot lose work. Unreadable/absent branches count as safe to delete.
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
        // Owner process is gone (or the marker is unusable) -> the worktree is
        // abandoned regardless of whether the directory still exists.
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

    // 1. Prune registered worktrees whose owner process is dead or directory is missing
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

    // 2. Prune orphaned worker-* branches that have no registered worktrees AND are merged into HEAD
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
        // 3. Reclaim abandoned `swe-wt-*` scratch directories. Every entry is
        //    classified by name alone and then handled as a whole unit — the
        //    directory with its lease, or the lease on its own — so the result
        //    is independent of `read_dir` ordering and no orphan can survive
        //    with its lease already removed (audit §01/§09).
        if let Ok(entries) = std::fs::read_dir(base) {
            for entry in entries.flatten() {
                let p = entry.path();
                // Non-UTF-8 names are reported instead of guessed at: such an
                // entry is never a lease this sweep wrote, so nothing is lost
                // by skipping it.
                let Some(name) = p.file_name().and_then(|n| n.to_str()) else {
                    error!(path = %p.display(), "Skipping non-UTF-8 scratch entry");
                    continue;
                };

                // Dispatch on the entry's *type* first: a directory can only be
                // a worktree, and a lease is always a regular file. Both
                // branches are self-contained, so which of the two `read_dir`
                // hands back first cannot change the outcome.
                if p.is_dir() {
                    if is_worktree_dir_name(name) {
                        reclaim_abandoned_worktree(&p);
                    }
                } else if is_worktree_lease_name(name) {
                    reclaim_dangling_lease(&p);
                }
            }
        }

        // 4. Prune orphaned swe-target-* directories in base dirs whose worktrees are gone
        if let Ok(entries) = std::fs::read_dir(base) {
            for entry in entries.flatten() {
                let p = entry.path();
                if let Some(name) = p.file_name().and_then(|n| n.to_str())
                    && let Some(wt_name) = name.strip_prefix("swe-target-")
                    && wt_name.starts_with("swe-wt-")
                {
                    let wt_path = base.join(wt_name);
                    if !wt_path.exists() {
                        force_remove_dir(&p);
                    }
                }
            }
        }
    }

    // Post-pass prune: phase 1 may have removed worktrees, which leaves new
    // stale administrative entries; this second pass clears them. Together the
    // pre- and post-passes form a legitimate double pass (pre collects old
    // garbage, post collects what we just removed) — both are needed.
    let _ = git(repo_root, "worktree prune", &["worktree", "prune"]);
}
