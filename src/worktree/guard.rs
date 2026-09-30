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
use std::collections::BTreeMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::UNIX_EPOCH;
use tracing::{debug, info};

/// Unversioned dirs copied into a fresh worktree (readable without git
/// tracking) and synced back on the way out.
const ARTIFACT_DIRS: &[&str] = &["audits", "reports", ".agents", "artifacts"];

/// Directory names never mirrored between a worktree and the repo root.
///
/// Skipping by name at every recursion level keeps dependency caches and build
/// output out of the repository working tree.
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
fn is_skipped_dir_name(name: &str) -> bool {
    SKIP_DIR_NAMES.contains(&name)
}

/// Ceiling on a sanitized worker id, so a caller cannot force a path or branch
/// name long enough to hit a filesystem or git limit.
const MAX_WORKER_ID_LEN: usize = 64;

/// Reduce a caller-supplied worker id to a token safe in *both* a path
/// component and a git branch name.
///
/// `WorktreeGuard::new` is public, so `worker_id` is untrusted input. Only
/// `[A-Za-z0-9_-]` survives and every rejected byte collapses to a single `-`,
/// which blocks two classes of abuse before the value reaches a path or argv:
///
/// * **Traversal.** A `/` can no longer appear, so `../../home/user/.ssh`
///   degrades to `home-user-ssh` and stays a single component inside
///   `swe_base_dir()`. The same value names the directory `Drop` removes.
/// * **Option injection.** Leading `-` is stripped even when the id is later
///   used directly as a git argument.
///
/// An id that sanitizes away to nothing has no safe rendering, so it is
/// replaced with a fresh random one rather than collapsing distinct callers
/// onto the same worktree.
fn sanitize_worker_id(worker_id: &str) -> String {
    let mut out = String::with_capacity(worker_id.len().min(MAX_WORKER_ID_LEN));
    let mut last_was_dash = false;
    for ch in worker_id.chars() {
        if out.len() >= MAX_WORKER_ID_LEN {
            break;
        }
        if ch.is_ascii_alphanumeric() || ch == '_' {
            out.push(ch);
            last_was_dash = false;
        } else if !last_was_dash {
            // Collapse a run of rejected bytes into one separator so `"a///b"`
            // and `"a/b"` cannot produce two different worktrees.
            out.push('-');
            last_was_dash = true;
        }
    }

    let trimmed = out.trim_matches('-');
    if trimmed.is_empty() {
        return format!("w{}", uuid::Uuid::new_v4().simple());
    }
    trimmed.to_string()
}

/// Create the checkout directory private *before* git writes into it.
/// Git accepts an existing empty directory for `worktree add` and preserves
/// its mode; chmodding only after checkout exposes contents under umask 022.
fn create_private_worktree_dir(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Could not create worktree base {}", parent.display()))?;
    }
    let mut builder = std::fs::DirBuilder::new();
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder
        .create(path)
        .with_context(|| format!("Could not create worktree directory {}", path.display()))
}

/// Verify that git did not relax the directory mode after checkout.
fn harden_worktree_dir(path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700)).with_context(
            || {
                format!(
                    "Could not restrict worktree directory {} to 0700",
                    path.display()
                )
            },
        )?;
    }
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
}

/// RAII guard around one subagent's `git` worktree.
pub struct WorktreeGuard {
    pub path: PathBuf,
    pub branch: String,
    pub repo_root: PathBuf,
    pub base_commit: String,
    pub preserve_branch: bool,
    /// Fingerprint of every artifact file seeded into the worktree, keyed by
    /// repository-relative path. A file still matching its entry was never
    /// touched by the worker, so [`WorktreeGuard::sync_artifacts`] leaves the
    /// repository root's copy alone.
    seeded: BTreeMap<String, FileFingerprint>,
}

impl WorktreeGuard {
    pub fn new(repo_root: &Path, worker_id: &str) -> Result<Self> {
        // Branch and directory names derive from the *sanitized* id, so they
        // can never describe different worktrees nor escape `swe_base_dir()`.
        let worker_id = sanitize_worker_id(worker_id);
        let branch = format!("worker-{worker_id}");
        let path = swe_base_dir().join(format!("swe-wt-{worker_id}"));

        // Ensure target directory and branch don't exist. `worktree prune` is
        // deliberately not called here: `prune_stale_worktrees` owns a single
        // prune per sweep, and `worktree add` fails loudly if a stale
        // registration is still in place (audit §06).
        force_remove_dir(&path);
        let _ = git(repo_root, "branch -D", &["branch", "-D", &branch]);

        info!(repo = %repo_root.display(), branch = %branch, path = %path.display(), "Creating git worktree");

        let path_str = path
            .to_str()
            .context("Worktree path contains invalid UTF-8")?;
        // Create it private before checkout so no reader observes files under
        // a permissive umask.
        create_private_worktree_dir(&path)?;

        let output = git(
            repo_root,
            "worktree add",
            &["worktree", "add", "-b", &branch, path_str, "HEAD"],
        )?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("git worktree add failed: {}", stderr);
        }

        // Verify git left the checkout private before seeding artifacts.
        harden_worktree_dir(&path)?;

        // Write host PID (+ owner uid) to a sibling file so running worktrees
        // are never pruned by concurrent instances, and a stale marker left by
        // a different user is never mistaken for ours (audit §12).
        let _ = std::fs::write(pid_file_for(&path), pid_file_contents());

        // Seed unversioned directories so subagents can read them without git
        // tracking. Uses the same copy path (and skipped-directory guards) as
        // the sync back, so a `.git` or `node_modules` tree planted inside an
        // artifact directory is never mirrored in either direction. The walk
        // records what each seeded file contained, so the sync back can tell a
        // file the worker edited from one it merely inherited.
        let mut seeded = BTreeMap::new();
        for dir in ARTIFACT_DIRS {
            let src = repo_root.join(dir);
            if src.is_dir() {
                let dst = path.join(dir);
                let _ = copy_dir_all(&src, &dst, &mut seeded, repo_root, None);
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
            preserve_branch: false,
            seeded,
        })
    }

    pub fn get_diff(&self) -> Result<String> {
        // Stage untracked files intent-to-add so git diff captures new files.
        let _ = git(&self.path, "add", &["add", "-N", "."]);

        let output = git(&self.path, "diff HEAD", &["diff", "HEAD"])?;
        if output.status.success() {
            let diff = String::from_utf8_lossy(&output.stdout).to_string();
            if !diff.trim().is_empty() {
                return Ok(diff);
            }
        }

        // If the working tree diff is empty, check for commits on this branch.
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
    /// Only what the worker actually produced travels: a file the worker
    /// created, or one whose content still differs from the copy seeded into
    /// its worktree. A seeded file left untouched is *not* copied, so a repo
    /// root that moved on while the worker ran (another worker's result merged,
    /// the user edited the file) keeps its newer content instead of being
    /// reverted to the stale copy the worker never looked at.
    ///
    /// Returns the sorted, duplicate-free list of repository-relative files in
    /// sync after the call, including the seeded-unchanged ones, so callers
    /// keep counting every artifact the worktree holds. The copy is otherwise
    /// conservative: unchanged files are left untouched, dependency caches and
    /// build output ([`SKIP_DIR_NAMES`]) are never mirrored, and each file is
    /// published atomically so a concurrent reader never observes a partially
    /// written artifact.
    ///
    /// Per-directory I/O errors are logged rather than propagated: both call
    /// sites discard the `Result` (`pool::runner` wants the artifact count,
    /// `Drop` is a best-effort safety net), so failing here would abort a
    /// worker's teardown over a single unreadable report.
    pub fn sync_artifacts(&self) -> Vec<String> {
        let mut synced = BTreeMap::new();

        for dir in ARTIFACT_DIRS {
            let src_dir = self.path.join(dir);
            if src_dir.is_dir() {
                let dest_dir = self.repo_root.join(dir);
                if let Err(e) = copy_dir_all(
                    &src_dir,
                    &dest_dir,
                    &mut synced,
                    &self.path,
                    Some(&self.seeded),
                ) {
                    debug!(
                        dir = %dir,
                        path = %src_dir.display(),
                        error = %e,
                        "Artifact sync failed for directory"
                    );
                }
            }
        }

        synced.into_keys().collect()
    }

    /// Commit all dirty changes in the worktree to preserve work in git history,
    /// marking the branch to be retained upon worktree cleanup.
    pub fn commit_changes(&mut self, message: &str) -> Result<Option<String>> {
        // Stage all changes (both tracked and untracked).
        let _ = git(&self.path, "add", &["add", "-A"]);

        // Check if there are changes to commit.
        let status = git(&self.path, "status", &["status", "--porcelain"])?;
        if !status.status.success() {
            let stderr = String::from_utf8_lossy(&status.stderr);
            anyhow::bail!("git status failed: {}", stderr.trim());
        }
        if status.stdout.is_empty() {
            // Even if the working tree is clean, check if the branch already
            // has commits beyond base_commit.
            if self.branch_has_commits() {
                self.preserve_branch = true;
                return Ok(Some(self.branch.clone()));
            }
            return Ok(None);
        }

        // Commit with fallback credentials so lack of git config never errors.
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

        // Sync report/audit artifacts to repo root before cleanup. This is the
        // safety net for teardown paths that never reached the explicit call in
        // `pool::runner`; a second run is cheap because unchanged files are
        // skipped rather than recopied.
        self.sync_artifacts();
        info!(path = %self.path.display(), branch = %self.branch, "Cleaning up git worktree");

        let _ = git(
            &self.repo_root,
            "worktree remove",
            &[
                "worktree",
                "remove",
                "--force",
                &self.path.to_string_lossy(),
            ],
        );
        // If the branch has commits beyond base_commit, ALWAYS preserve it.
        let has_commits = !self.preserve_branch && self.branch_has_commits();

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

/// Fingerprint of one artifact file's content: its byte length plus a hash of
/// the bytes.
///
/// Length is the cheap discriminator and the hash only has to separate files of
/// the same size, so a collision costs a redundant re-copy, never a lost edit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FileFingerprint {
    len: u64,
    hash: u64,
}

impl FileFingerprint {
    fn of(bytes: &[u8]) -> Self {
        let mut hasher = DefaultHasher::new();
        bytes.hash(&mut hasher);
        Self {
            len: bytes.len() as u64,
            hash: hasher.finish(),
        }
    }

    /// Fingerprint of the file at `path`, or `None` when it cannot be read as
    /// a regular file.
    fn of_path(path: &Path) -> Option<Self> {
        let meta = std::fs::metadata(path).ok()?;
        if !meta.is_file() {
            return None;
        }
        Some(Self::of(&std::fs::read(path).ok()?))
    }
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

/// Recursively copy `src` into `dst`, recording every file it walks
/// (relative to `worktree_root`) in `collected` together with the fingerprint
/// of the bytes that were copied.
///
/// Four properties make the sync cheap enough to run on every worker teardown:
///
/// * **Skipped directories** ([`SKIP_DIR_NAMES`]) are pruned before their
///   contents are walked, so `.git`, `node_modules` and build output never
///   cross the worktree boundary.
/// * **Symlinks are not followed**, so the walk cannot escape the worktree or
///   publish a link pointing at arbitrary host paths.
/// * **Unchanged files are left alone.** A destination that already holds the
///   exact same bytes is not rewritten, which turns a repeated sync into a
///   metadata scan and keeps destination mtimes stable for downstream watchers.
/// * **Seeded, untouched files are left alone too**, when `seeded` is given:
///   a file whose bytes still match the fingerprint recorded when it was
///   copied into the worktree was never changed by the worker, so re-publishing
///   it could only revert whatever the repository root grew in the meantime.
///   The seeding pass itself passes `None` and fills the map instead.
///
/// A file that is already in sync is still recorded in `collected`, so callers
/// see a complete and stable artifact list across repeated syncs. The map also
/// deduplicates by relative path instead of re-checking membership per file,
/// so overlapping artifact directories scale linearly (audit §11).
fn copy_dir_all(
    src: &Path,
    dst: &Path,
    collected: &mut BTreeMap<String, FileFingerprint>,
    worktree_root: &Path,
    seeded: Option<&BTreeMap<String, FileFingerprint>>,
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
            copy_dir_all(&src_path, &dst_path, collected, worktree_root, seeded)?;
        } else if ft.is_file() {
            let rel = src_path
                .strip_prefix(worktree_root)
                .ok()
                .map(|rel| rel.to_string_lossy().into_owned());
            // An unreadable file is still walked, just not fingerprinted: it
            // counts as changed and travels back rather than going missing.
            let fingerprint = FileFingerprint::of_path(&src_path);
            let inherited = match (seeded, &rel, fingerprint) {
                (Some(seeded), Some(rel), Some(fingerprint)) => {
                    seeded.get(rel) == Some(&fingerprint)
                }
                _ => false,
            };
            if !inherited && !is_up_to_date(&src_path, &dst_path) {
                copy_file_atomic(&src_path, &dst_path)?;
            }
            if let (Some(rel), Some(fingerprint)) = (rel, fingerprint) {
                collected.insert(rel, fingerprint);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The sanitized id is spliced into a branch name and into a path that
    /// `Drop` recursively deletes, so these are the two properties it must hold.
    #[test]
    fn sanitized_id_is_a_single_safe_path_and_branch_component() {
        for raw in [
            "deadbeef",
            "../../home/user/.ssh",
            "a/b\\c",
            "  spaced  ",
            "--upload-pack=/bin/sh",
            "trailing///",
            "///leading",
        ] {
            let id = sanitize_worker_id(raw);
            assert!(!id.is_empty(), "id {raw:?} must not sanitize to nothing");
            assert!(
                !id.starts_with('-'),
                "id {raw:?} would be read as a git flag"
            );
            assert!(
                !id.contains('/') && !id.contains('\\'),
                "id {raw:?} kept a separator"
            );
            assert!(!id.contains('.'), "id {raw:?} kept a dot");
            assert!(
                id.bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-'),
                "id {raw:?} kept a byte outside [A-Za-z0-9_-]: {id:?}"
            );
            assert!(id.len() <= MAX_WORKER_ID_LEN, "id {raw:?} exceeded the cap");
        }
    }

    /// Two spellings of the same logical id must not produce two worktrees, and
    /// a genuinely hostile id must still get a usable worktree of its own.
    #[test]
    fn sanitize_collapses_equivalent_ids_and_replaces_unusable_ones() {
        assert_eq!(sanitize_worker_id("a/b"), sanitize_worker_id("a//b"));
        assert_eq!(sanitize_worker_id("deadbeef"), "deadbeef");

        // Nothing usable survives: replaced, not collapsed to an empty id that
        // every such caller would then share.
        for raw in ["", "///", "   "] {
            let id = sanitize_worker_id(raw);
            assert!(!id.is_empty(), "id {raw:?} must be replaced, not emptied");
            assert!(id
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-'));
        }
        assert_ne!(sanitize_worker_id(""), sanitize_worker_id(""));

        // The cap is enforced before the trim, so a long id cannot outrun it.
        assert!(sanitize_worker_id(&"x".repeat(500)).len() <= MAX_WORKER_ID_LEN);
    }

    /// Git must receive a private directory, not create a public one and wait
    /// for a post-checkout chmod to close the exposure window.
    #[cfg(unix)]
    #[test]
    fn worktree_dir_is_private_before_git_checkout() {
        use std::os::unix::fs::PermissionsExt;

        let root = std::env::temp_dir().join(format!(
            "swe-precheckout-test-{}",
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir(&root).unwrap();
        let dir = root.join("missing-base").join("worktree");
        create_private_worktree_dir(&dir).unwrap();
        let mode = std::fs::metadata(&dir).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o700);
        std::fs::remove_dir_all(root).unwrap();
    }

    /// The worktree directory must never be group- or world-readable: it holds
    /// the full checkout plus any secrets and artifacts seeded into it.
    #[cfg(unix)]
    #[test]
    fn harden_worktree_dir_leaves_only_owner_bits() {
        use std::os::unix::fs::PermissionsExt;

        let dir = std::env::temp_dir().join(format!(
            "swe-harden-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        // Start world-readable, the way `git worktree add` under umask 022 does.
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o755)).unwrap();

        harden_worktree_dir(&dir).unwrap();
        let mode = std::fs::metadata(&dir).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o700, "worktree dir ended up {mode:o}");

        let _ = std::fs::remove_dir_all(&dir);
    }
}
