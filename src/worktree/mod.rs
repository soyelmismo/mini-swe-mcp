//! Isolated `git` worktrees for subagents.
//!
//! Every dispatched subagent gets its own worktree on a dedicated `worker-<id>`
//! branch under [`swe_base_dir`], so concurrent agents never contend for the
//! repository working tree. This package owns both halves of that lifecycle:
//!
//! * `guard` — the RAII [`WorktreeGuard`] that creates a worktree, syncs
//!   artifacts, produces diffs, commits work and cleans up on `Drop`;
//! * `prune` — the garbage-collection sweep that reclaims worktrees, branches
//!   and scratch directories abandoned by crashed or finished processes.
//!
//! The public surface stays flat: everything callers need is re-exported here.

mod guard;
pub(crate) mod prune;

pub use guard::{FileFingerprint, WorktreeGuard};
pub use prune::{
    claim_lease_for_test, is_process_alive, prune_stale_worktrees, prune_stale_worktrees_in,
    worktree_is_stale_for_test,
};

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::process::Command;
use tracing::error;

/// Root directory hosting all subagent scratch data (worktrees, target dirs, caches).
pub fn swe_base_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("SWE_TEMP_DIR") {
        PathBuf::from(dir)
    } else {
        let var_tmp = PathBuf::from("/var/tmp");
        if var_tmp.is_dir() {
            var_tmp
        } else {
            std::env::temp_dir()
        }
    }
}

/// All directories that can host `swe-wt-*` / `swe-target-*` scratch data.
///
/// Yielded at most once each: `swe_base_dir()` frequently *is* the system temp
/// dir, and sweeping it twice used to re-scan the same tree (audit §07).
pub(crate) fn swe_base_dirs() -> Vec<PathBuf> {
    let mut dirs = vec![swe_base_dir()];
    let tmp = std::env::temp_dir();
    if !dirs.contains(&tmp) {
        dirs.push(tmp);
    }
    dirs
}

/// Run `git` in `dir`, attaching the operation to the error when spawning fails.
pub(crate) fn git(dir: &Path, operation: &str, args: &[&str]) -> Result<std::process::Output> {
    Command::new("git")
        .current_dir(dir)
        .args(args)
        .output()
        .with_context(|| format!("Failed to execute git {operation}"))
}

/// The `.pid` sibling path for a worktree directory, built without `format!`
/// allocations in the common case (audit §10).
pub(crate) fn pid_file_for(path: &Path) -> PathBuf {
    let mut sibling = path.as_os_str().to_os_string();
    sibling.push(".pid");
    PathBuf::from(sibling)
}

/// Remove a directory tree, tolerating an already-missing path.
///
/// Centralises the `exists() && remove_dir_all` pattern repeated in three
/// places (audit §04/§04b).
pub(crate) fn force_remove_dir(path: &Path) {
    if let Err(e) = std::fs::remove_dir_all(path)
        && e.kind() != std::io::ErrorKind::NotFound
    {
        error!(error = %e, path = %path.display(), "Failed to delete leftover worktree directory");
    }
}

/// Delete a worktree's scratch/target directories from every known base dir.
pub(crate) fn remove_target_dirs(wt_path: &Path) {
    if let Some(wt_name) = wt_path.file_name().and_then(|n| n.to_str()) {
        for base in swe_base_dirs() {
            force_remove_dir(&base.join(format!("swe-target-{wt_name}")));
        }
    }
}
