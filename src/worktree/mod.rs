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

pub use guard::{BaseSync, BranchMerge, FileFingerprint, WorktreeGuard};
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

/// The scratch root every per-worker path is resolved under.
///
/// Resolved once, when a pool is built: [`WorkerPool`](crate::pool::WorkerPool)
/// holds one for its lifetime instead of re-reading `SWE_TEMP_DIR` on every
/// call, so two pools in one process can never resolve different roots
/// mid-run. The free functions in `pool` stay thin wrappers over
/// [`ScratchRoot::from_env`], which is what the CLI, the monitor and the hub
/// daemon use.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScratchRoot {
    root: PathBuf,
    /// Directories swept alongside `root`. Only the default resolution sweeps
    /// the system temp dir, so an injected root sees exactly its own rows and
    /// never another pool's.
    extra: Vec<PathBuf>,
}

impl ScratchRoot {
    /// Resolve the default root exactly as [`swe_base_dir`] does.
    pub fn from_env() -> Self {
        let root = swe_base_dir();
        let mut extra = Vec::new();
        let tmp = std::env::temp_dir();
        if tmp != root {
            extra.push(tmp);
        }
        Self { root, extra }
    }

    /// An explicit root, sweeping nothing but itself.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            extra: Vec::new(),
        }
    }

    /// A root over an already-resolved set of scratch directories: the first
    /// is the root, the rest are swept alongside it.
    pub fn from_dirs(dirs: &[PathBuf]) -> Self {
        let root = dirs.first().cloned().unwrap_or_else(swe_base_dir);
        let extra = dirs
            .iter()
            .skip(1)
            .filter(|dir| **dir != root)
            .cloned()
            .collect();
        Self { root, extra }
    }

    /// The root itself, which every per-worker path is built under.
    pub fn path(&self) -> &Path {
        &self.root
    }

    /// `rel` resolved under this root.
    pub fn join(&self, rel: impl AsRef<Path>) -> PathBuf {
        self.root.join(rel)
    }

    /// Every directory that can host this root's scratch data.
    ///
    /// Yielded at most once each: the root frequently *is* the system temp
    /// dir, and sweeping it twice used to re-scan the same tree (audit §07).
    pub fn base_dirs(&self) -> Vec<PathBuf> {
        let mut dirs = vec![self.root.clone()];
        for dir in &self.extra {
            if !dirs.contains(dir) {
                dirs.push(dir.clone());
            }
        }
        dirs
    }
}

impl Default for ScratchRoot {
    fn default() -> Self {
        Self::from_env()
    }
}

/// All directories that can host `swe-wt-*` / `swe-target-*` scratch data.
pub(crate) fn swe_base_dirs() -> Vec<PathBuf> {
    ScratchRoot::from_env().base_dirs()
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

/// Private scratch root for one worker, independent of its shared build slot.
pub(crate) fn scratch_dir(worktree: &Path) -> PathBuf {
    let name = worktree
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("default");
    swe_base_dir().join(format!("swe-tmp-{name}"))
}

/// Delete private scratch and legacy targets, never shared build dirs.
pub(crate) fn remove_target_dirs(wt_path: &Path) {
    remove_target_dirs_in(&ScratchRoot::from_env(), wt_path)
}

/// [`remove_target_dirs`] under an explicit scratch root.
pub(crate) fn remove_target_dirs_in(root: &ScratchRoot, wt_path: &Path) {
    if let Some(wt_name) = wt_path.file_name().and_then(|n| n.to_str()) {
        for base in root.base_dirs() {
            force_remove_dir(&base.join(format!("swe-target-{wt_name}")));
            force_remove_dir(&base.join(format!("swe-tmp-{wt_name}")));
        }
    }
}

#[cfg(test)]
mod target_tests {
    use super::*;

    #[test]
    fn teardown_only_removes_private_and_legacy_directories() {
        let base = swe_base_dir();
        let name = format!("swe-wt-cleanup-{}", uuid::Uuid::new_v4());
        let worktree = base.join(&name);
        let legacy = base.join(format!("swe-target-{name}"));
        let scratch = scratch_dir(&worktree);
        let lease = crate::cache::BuildDirLease::acquire(&base).unwrap();
        let shared = lease.dir().to_path_buf();
        std::fs::create_dir_all(&legacy).unwrap();
        std::fs::create_dir_all(&scratch).unwrap();
        remove_target_dirs(&worktree);
        assert!(!legacy.exists());
        assert!(!scratch.exists());
        assert!(shared.exists());
        drop(lease);
        force_remove_dir(&shared);
    }
}
