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
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::UNIX_EPOCH;
use tracing::{debug, info};

/// Unversioned directories that are copied into a fresh worktree so subagents
/// can read them without git tracking, and synced back on the way out.
const ARTIFACT_DIRS: &[&str] = &["audits", "reports", ".agents", "artifacts"];

/// Directory names that are never mirrored between a worktree and the repo root.
///
/// A subagent that installs dependencies or runs a build inside an artifact
/// directory would otherwise drag an entire `.git`, `node_modules` or compiled
/// output tree across the boundary on every sync — megabytes of files the repo
/// root neither wants nor can meaningfully merge. Skipping by name at every
/// recursion level keeps the guard cheap and, more importantly, keeps foreign
/// build output and dependency caches out of the repository working tree.
const SKIP_DIR_NAMES: &[&str] = &[
    ".git",
    "node_modules",
    "target",
    "build",
    "dist",
    ".venv",
    "venv",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".cargo",
    ".next",
    ".turbo",
];

/// True when a directory entry must never be copied out of a worktree.
///
/// Matching the basename at every level is enough to stop `.git`,
/// `node_modules` and the various build/output directories *before* their
/// contents are walked, which is where the real cost would be.
fn is_skipped_dir_name(name: &str) -> bool {
    SKIP_DIR_NAMES.contains(&name)
}

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

        // Seed unversioned directories into the worktree so subagents can read
        // them without git tracking. Uses the same copy path (and therefore the
        // same skipped-directory guards) as the sync back, so a `.git` or
        // `node_modules` tree planted inside an artifact directory is never
        // mirrored in either direction.
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
    ///
    /// Returns the sorted, duplicate-free list of repository-relative files that
    /// are in sync after the call. The copy itself is conservative: unchanged
    /// files are left untouched, dependency caches and build output
    /// ([`SKIP_DIR_NAMES`]) are never mirrored, and each file is published
    /// atomically so a concurrent reader in the repo root never observes a
    /// partially written artifact.
    ///
    /// Per-directory I/O errors are logged rather than propagated: both call
    /// sites discard the `Result` (`pool::runner` wants the artifact count,
    /// `Drop` is a best-effort safety net), so failing here would abort a
    /// worker's teardown over a single unreadable report.
    pub fn sync_artifacts(&self) -> Result<Vec<String>> {
        let mut synced = BTreeSet::new();

        for dir in ARTIFACT_DIRS {
            let src_dir = self.path.join(dir);
            if src_dir.is_dir() {
                let dest_dir = self.repo_root.join(dir);
                if let Err(e) = copy_dir_all(&src_dir, &dest_dir, &mut synced, &self.path) {
                    debug!(
                        dir = %dir,
                        path = %src_dir.display(),
                        error = %e,
                        "Artifact sync failed for directory"
                    );
                }
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

        // Sync report/audit artifacts to repo root before cleanup. This is the
        // safety net for teardown paths that never reached the explicit call in
        // `pool::runner`; a second run is cheap because unchanged files are
        // skipped rather than recopied.
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

/// Process-unique counter backing [`tmp_sibling_name`], so two concurrent
/// workers writing into the same repo root never collide on a staging path.
static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// A staging filename in `dir` that no other writer can pick: it combines the
/// process id, a monotonic counter and the clock, so neither a concurrent
/// worker nor a stale file from a crashed one can share it.
fn tmp_sibling_name(dir: &Path, file_name: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let seq = TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    dir.join(format!(
        ".{file_name}.{}.{nanos}.{seq}.tmp",
        std::process::id()
    ))
}

/// True when `dst` already holds `src`'s exact bytes.
///
/// Length is compared first (one `stat`, and it rules out the overwhelming
/// majority of files), then content: a partial or truncated write from an
/// interrupted earlier sync must never be mistaken for a completed one.
/// Timestamps alone are not enough — coarse `mtime` granularity would hide two
/// distinct writes made inside the same tick.
fn is_up_to_date(src: &Path, dst: &Path) -> bool {
    let Ok(src_meta) = std::fs::metadata(src) else {
        return false;
    };
    let Ok(dst_meta) = std::fs::metadata(dst) else {
        return false;
    };
    if !src_meta.is_file() || !dst_meta.is_file() || src_meta.len() != dst_meta.len() {
        return false;
    }
    match (std::fs::read(src), std::fs::read(dst)) {
        (Ok(a), Ok(b)) => a == b,
        _ => false,
    }
}

/// Copy `src` onto `dst` atomically.
///
/// The bytes land in a uniquely named staging file *in the destination
/// directory* first, then a single `rename` publishes them. Because the staging
/// path shares the destination's filesystem, the `rename` is atomic: a reader in
/// the repo root observes either the previous file or the complete new one,
/// never a half-written artifact — which the previous in-place `std::fs::copy`
/// could produce, and which the orchestrator reading artifacts while a second
/// sync overwrote them would then report as corrupt.
fn copy_file_atomic(src: &Path, dst: &Path) -> std::io::Result<()> {
    let file_name = dst.file_name().unwrap_or_default().to_string_lossy();
    let staging = tmp_sibling_name(dst.parent().unwrap_or(dst), &file_name);

    // Any failure must not leave staging debris in the repository root.
    let result = (|| {
        std::fs::copy(src, &staging)?;
        std::fs::rename(&staging, dst)
    })();

    if result.is_err() {
        let _ = std::fs::remove_file(&staging);
    }
    result
}

/// Recursively copy `src` into `dst`, recording every copied file (relative to
/// `worktree_root`) in `collected`.
///
/// Three properties make the sync cheap enough to run on every worker teardown:
///
/// * **Skipped directories** ([`SKIP_DIR_NAMES`]) are pruned before their
///   contents are walked, so `.git`, `node_modules` and build output never
///   cross the worktree boundary.
/// * **Symlinks are not followed**, so the walk cannot escape the worktree or
///   publish a link pointing at arbitrary host paths.
/// * **Unchanged files are left alone.** A destination that already holds the
///   exact same bytes is not rewritten, which turns a repeated sync into a
///   metadata scan and keeps destination mtimes stable for downstream watchers.
///
/// A file that is already in sync is still recorded in `collected`, so callers
/// see a complete and stable artifact list across repeated syncs. The set also
/// deduplicates by relative path instead of re-checking membership per file,
/// so overlapping artifact directories scale linearly (audit §11).
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
            // Never mirror dependency caches or build output into the repo root.
            if entry.file_name().to_str().is_some_and(is_skipped_dir_name) {
                debug!(
                    path = %src_path.display(),
                    "Skipping build/cache directory during artifact sync"
                );
                continue;
            }
            copy_dir_all(&src_path, &dst_path, collected, worktree_root)?;
        } else if ft.is_file() {
            if !is_up_to_date(&src_path, &dst_path) {
                copy_file_atomic(&src_path, &dst_path)?;
            }
            if let Ok(rel) = src_path.strip_prefix(worktree_root) {
                collected.insert(rel.to_string_lossy().into_owned());
            }
        }
    }
    Ok(())
}
