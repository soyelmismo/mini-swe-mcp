//! The [`WorktreeGuard`] lifecycle: create an isolated worktree for one
//! subagent, hand it its artifact directories, report and preserve its work,
//! and reclaim it on `Drop`.
//!
//! Lease bookkeeping (writing the `.pid` marker) and the sweep that consumes
//! those markers live in the sibling [`prune`](super::prune) module; this file
//! only owns one worktree at a time.

use super::{
    force_remove_dir, git, pid_file_for, prune::pid_file_contents, remove_target_dirs, swe_base_dir,
};
use anyhow::{Context, Result};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use tracing::info;

/// Unversioned directories that are copied into a fresh worktree so subagents
/// can read them without git tracking, and synced back on the way out.
const ARTIFACT_DIRS: &[&str] = &["audits", "reports", ".agents", "artifacts"];

/// RAII guard around one subagent's `git` worktree.
pub struct WorktreeGuard {
    pub path: PathBuf,
    pub branch: String,
    pub repo_root: PathBuf,
    pub base_commit: String,
    /// When set, [`Drop`] leaves the worktree on disk instead of removing it.
    ///
    /// The `.pid` lease is deleted on preservation so the abandoned-worktree
    /// sweep keeps failing open on it instead of eventually treating it as a
    /// zombie (audit §05/§08).
    pub keep: bool,
    pub preserve_branch: bool,
}

impl WorktreeGuard {
    pub fn new(repo_root: &Path, worker_id: &str) -> Result<Self> {
        let branch = format!("worker-{}", worker_id);
        let path = swe_base_dir().join(format!("swe-wt-{}", worker_id));

        // Ensure target directory and branch don't exist. `worktree prune` is
        // deliberately not called here: `prune_stale_worktrees` owns a single
        // prune per sweep, and the `worktree add` below fails loudly if a stale
        // registration is still in place (audit §06).
        force_remove_dir(&path);
        let _ = git(repo_root, "branch -D", &["branch", "-D", &branch]);

        info!(repo = %repo_root.display(), branch = %branch, path = %path.display(), "Creating git worktree");

        let path_str = path
            .to_str()
            .context("Worktree path contains invalid UTF-8")?;

        let output = git(
            repo_root,
            "worktree add",
            &["worktree", "add", "-b", &branch, path_str, "HEAD"],
        )?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("git worktree add failed: {}", stderr);
        }

        // Write host PID (+ owner uid) to a sibling file so running worktrees are
        // never pruned by concurrent instances, and so a stale marker left by a
        // different user is never mistaken for ours (audit §12).
        let _ = std::fs::write(pid_file_for(&path), pid_file_contents());

        // Seed unversioned directories into the worktree so subagents can read them without git tracking
        for dir in ARTIFACT_DIRS {
            let src = repo_root.join(dir);
            if src.is_dir() {
                let dst = path.join(dir);
                let mut dummy = BTreeSet::new();
                let _ = copy_dir_all(&src, &dst, &mut dummy, repo_root);
            }
        }

        let base_commit_out = git(repo_root, "rev-parse HEAD", &["rev-parse", "HEAD"])?;
        if !base_commit_out.status.success() {
            let stderr = String::from_utf8_lossy(&base_commit_out.stderr);
            anyhow::bail!("git rev-parse HEAD failed: {}", stderr.trim());
        }
        let base_commit = String::from_utf8_lossy(&base_commit_out.stdout)
            .trim()
            .to_string();

        Ok(Self {
            path,
            branch,
            repo_root: repo_root.to_path_buf(),
            base_commit,
            keep: false,
            preserve_branch: false,
        })
    }

    pub fn get_diff(&self) -> Result<String> {
        // Stage untracked files intent-to-add so git diff captures new files as well
        let _ = git(&self.path, "add", &["add", "-N", "."]);

        let output = git(&self.path, "diff HEAD", &["diff", "HEAD"])?;
        if output.status.success() {
            let diff = String::from_utf8_lossy(&output.stdout).to_string();
            if !diff.trim().is_empty() {
                return Ok(diff);
            }
        }

        // If working tree diff is empty, check if subagent committed changes to this branch
        if !self.base_commit.is_empty() {
            let output = git(
                &self.path,
                "diff base_commit..HEAD",
                &["diff", &format!("{}..HEAD", self.base_commit)],
            )?;
            if output.status.success() {
                let diff = String::from_utf8_lossy(&output.stdout).to_string();
                if !diff.trim().is_empty() {
                    return Ok(diff);
                }
            }
        }

        Ok(String::new())
    }

    /// Sync report and audit directories (audits, reports, .agents, artifacts)
    /// from the worktree back into the repository root.
    pub fn sync_artifacts(&self) -> Result<Vec<String>> {
        let mut synced = BTreeSet::new();

        for dir in ARTIFACT_DIRS {
            let src_dir = self.path.join(dir);
            if src_dir.is_dir() {
                let dest_dir = self.repo_root.join(dir);
                let _ = copy_dir_all(&src_dir, &dest_dir, &mut synced, &self.path);
            }
        }

        Ok(synced.into_iter().collect())
    }

    /// Commit all dirty changes in the worktree to preserve work in git history,
    /// marking the branch to be retained upon worktree cleanup.
    pub fn commit_changes(&mut self, message: &str) -> Result<Option<String>> {
        // Stage all changes (both tracked and untracked)
        let _ = git(&self.path, "add", &["add", "-A"]);

        // Check if there are changes to commit
        let status = git(&self.path, "status", &["status", "--porcelain"])?;
        if !status.status.success() {
            let stderr = String::from_utf8_lossy(&status.stderr);
            anyhow::bail!("git status failed: {}", stderr.trim());
        }
        if status.stdout.is_empty() {
            // Even if working tree is clean, check if branch already has commits beyond base_commit
            if self.branch_has_commits() {
                self.preserve_branch = true;
                return Ok(Some(self.branch.clone()));
            }
            return Ok(None);
        }

        // Commit with fallback credentials so lack of git config never errors
        let commit_out = git(
            &self.path,
            "commit",
            &[
                "-c",
                "user.name=mini-swe",
                "-c",
                "user.email=mini-swe@localhost",
                "commit",
                "-m",
                message,
            ],
        )?;
        if !commit_out.status.success() {
            let stderr = String::from_utf8_lossy(&commit_out.stderr);
            anyhow::bail!("git commit failed: {}", stderr.trim());
        }

        self.preserve_branch = true;
        Ok(Some(self.branch.clone()))
    }

    /// True when this worker's branch carries commits beyond `base_commit`.
    ///
    /// Shared by [`WorktreeGuard::commit_changes`] and [`Drop`]: a branch with
    /// committed work is never deleted, whatever else cleanup decides.
    fn branch_has_commits(&self) -> bool {
        if self.base_commit.is_empty() {
            return false;
        }
        git(
            &self.repo_root,
            "rev-list",
            &[
                "rev-list",
                "--count",
                &format!("{}..{}", self.base_commit, self.branch),
            ],
        )
        .map(|o| {
            String::from_utf8_lossy(&o.stdout)
                .trim()
                .parse::<u64>()
                .unwrap_or(0)
                > 0
        })
        .unwrap_or(false)
    }
}

impl Drop for WorktreeGuard {
    fn drop(&mut self) {
        let pid_file = pid_file_for(&self.path);

        if self.keep {
            // The worktree outlives this process on purpose. Its lease must not
            // outlive the leaseholder, or a later sweep would see a dead PID and
            // delete a worktree the user asked to keep. Dropping the `.pid` makes
            // the pruner fail open and treat the directory as active forever
            // (audit §05/§08).
            let _ = std::fs::remove_file(&pid_file);
            info!(path = %self.path.display(), "Preserving worktree");
            return;
        }

        // Sync report/audit artifacts to repo root before cleanup
        let _ = self.sync_artifacts();
        info!(path = %self.path.display(), branch = %self.branch, "Cleaning up git worktree");

        let path_str = self.path.to_string_lossy();
        let _ = git(
            &self.repo_root,
            "worktree remove",
            &["worktree", "remove", "--force", &path_str],
        );
        // If the branch has commits beyond base_commit, ALWAYS preserve it
        let has_commits = self.branch_has_commits();

        if self.preserve_branch || has_commits {
            info!(branch = %self.branch, "Preserving worker branch with committed changes");
        } else {
            let _ = git(
                &self.repo_root,
                "branch -D",
                &["branch", "-D", &self.branch],
            );
        }

        // `worktree remove --force` normally deleted the directory already; this
        // is the fallback for when it could not (audit §04).
        force_remove_dir(&self.path);
        let _ = std::fs::remove_file(&pid_file);
        remove_target_dirs(&self.path);
    }
}

/// Recursively copy `src` into `dst`, recording every copied file (relative to
/// `worktree_root`) in `collected`.
///
/// Deduplication uses a `HashSet` so overlapping artifact directories scale
/// linearly instead of quadratically (audit §11).
fn copy_dir_all(
    src: &Path,
    dst: &Path,
    collected: &mut BTreeSet<String>,
    worktree_root: &Path,
) -> std::io::Result<()> {
    if !src.exists() {
        return Ok(());
    }
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let ft = entry.file_type()?;
        if ft.is_symlink() {
            continue;
        }
        let src_path = entry.path();
        let dst_path = dst.join(entry.file_name());
        if ft.is_dir() {
            copy_dir_all(&src_path, &dst_path, collected, worktree_root)?;
        } else if ft.is_file() {
            std::fs::copy(&src_path, &dst_path)?;
            if let Ok(rel) = src_path.strip_prefix(worktree_root) {
                collected.insert(rel.to_string_lossy().into_owned());
            }
        }
    }
    Ok(())
}
