//! The [`WorktreeGuard`] lifecycle: create an isolated worktree for one
//! subagent, hand it its artifact directories, report and preserve its work,
//! and reclaim it on `Drop`.
//!
//! Both harness-mediated integrations go through
//! [`WorktreeGuard::merge_reference_at`]: the pre-completion base sync and a
//! consolidator's `CONSOLIDATE_MERGE` of a finished worker's branch.
//!
//! Lease bookkeeping (writing the `.pid` marker) and the sweep that consumes
//! those markers live in the sibling [`prune`](super::prune) module; this file
//! only owns one worktree at a time.
//!
//! Load-bearing: teardown deletes a `worker-<id>` branch that carries no
//! commit beyond the base, *unless* the worker was interrupted (see
//! [`WorktreeGuard::mark_interrupted`]). An interrupted worker must stay
//! continuable across a hub restart, and the branch is what its continuation
//! re-attaches to -- a worker that only read code has no commit to preserve,
//! so the ref itself is the only thing left of it.
//!
//! Load-bearing: a checkout directory and git's *registration* for it are one
//! unit, in both directions. A directory deleted while its registration
//! survives makes the next `worktree add` for the same worker fail outright
//! ("is a missing but already registered worktree"), which failed whole
//! revisions until an operator ran `git worktree prune` by hand. So teardown
//! unregisters what it deletes ([`unregister_worktree`]) and creation recovers
//! from a registration left behind by an earlier run. A live worker's row is
//! never collateral damage: the recovery is scoped to one worker's own path.

use super::{
    ScratchRoot, force_remove_dir, git, pid_file_for, prune::pid_file_contents,
    remove_target_dirs_in,
};
use anyhow::{Context, Result};
use std::collections::{BTreeMap, HashSet};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::UNIX_EPOCH;
use tracing::{debug, info, warn};

use crate::config::env_parse;

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
    ".rustup",
    ".cache",
    ".next",
    ".turbo",
];

/// True when a directory entry must never be copied out of a worktree, and a
/// path component must never be staged into a harness commit.
fn is_skipped_dir_name(name: &str) -> bool {
    SKIP_DIR_NAMES.contains(&name)
}

/// Ceiling, in megabytes, on one file a harness commit may stage.
///
/// A worker that points `HOME` (or a tool's own state directory) inside its
/// worktree fills the tree with a compile cache; committing it pushed 164 MB
/// blobs into history that a remote then refused. The cap is read at each
/// commit, so a test can lower it without touching process state.
const MAX_COMMIT_FILE_MB_ENV: &str = "MINI_SWE_MAX_COMMIT_FILE_MB";

/// The commit cap used when [`MAX_COMMIT_FILE_MB_ENV`] is unset or unparsable.
const DEFAULT_MAX_COMMIT_FILE_MB: u64 = 10;

/// The commit cap in bytes, `0` when the operator disabled it.
fn commit_file_cap_bytes() -> u64 {
    env_parse::<u64>(MAX_COMMIT_FILE_MB_ENV).unwrap_or(DEFAULT_MAX_COMMIT_FILE_MB) * 1024 * 1024
}

/// Why a harness commit left a path out of the index.
#[derive(Debug, Clone, PartialEq, Eq)]
enum SkipReason {
    /// The file is larger than the commit cap.
    TooLarge,
    /// The path sits in a cache directory or a tool home the worker created.
    CacheDir,
}

/// One path a harness commit refused to stage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkippedPath {
    /// Repository-relative path, as `git status` spelled it.
    pub path: String,
    /// The file's size in bytes, when it could be measured.
    pub bytes: Option<u64>,
    reason: SkipReason,
}

impl SkippedPath {
    /// The one-line explanation a checkpoint notice shows the worker.
    fn notice_line(&self) -> String {
        let why = match self.reason {
            SkipReason::TooLarge => "larger than the commit cap",
            SkipReason::CacheDir => "cache directory",
        };
        let size = self
            .bytes
            .map(format_size)
            .unwrap_or_else(|| "unknown size".to_string());
        format!(
            "not committed: {} ({size}), {why} -- move it to $TMPDIR",
            self.path
        )
    }
}

/// Render a byte count the way a worker reads it: megabytes once the number is
/// large enough to matter, kilobytes below that, bytes below that.
fn format_size(bytes: u64) -> String {
    const MIB: f64 = 1024.0 * 1024.0;
    const KIB: f64 = 1024.0;
    let bytes = bytes as f64;
    if bytes >= MIB {
        format!("{:.1} MB", bytes / MIB)
    } else if bytes >= KIB {
        format!("{:.1} KB", bytes / KIB)
    } else {
        format!("{bytes} B")
    }
}

/// What one harness commit did: whether it committed, and what it left out.
#[derive(Debug, Default, Clone)]
pub struct CommitReport {
    /// The branch that carries the commit, when one was made.
    pub branch: Option<String>,
    /// Paths the commit refused to stage, deduplicated, in path order.
    pub skipped: Vec<SkippedPath>,
}

impl CommitReport {
    /// True when a commit was created.
    pub fn committed(&self) -> bool {
        self.branch.is_some()
    }

    /// The lines a checkpoint notice shows the worker, longest-first so the
    /// paths a worker can act on are the ones it reads. `None` when nothing
    /// was refused.
    pub fn notice_lines(&self) -> Option<String> {
        if self.skipped.is_empty() {
            return None;
        }
        const MAX_REPORTED: usize = 20;
        let shown = self.skipped.len().min(MAX_REPORTED);
        let mut lines: Vec<String> = self.skipped[..shown]
            .iter()
            .map(SkippedPath::notice_line)
            .collect();
        let rest = self.skipped.len() - shown;
        if rest > 0 {
            lines.push(format!("... and {rest} more paths left out"));
        }
        Some(lines.join("\n"))
    }
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

/// The commit a branch points at, or `None` when the branch is absent.
///
/// `show-ref --verify` answers without listing every ref, so a revision probe
/// stays O(1) however many worker branches the repository holds.
fn branch_ref(repo_root: &Path, branch: &str) -> Option<String> {
    let output = git(
        repo_root,
        "show-ref",
        &[
            "show-ref",
            "--verify",
            "--quiet",
            &format!("refs/heads/{branch}"),
        ],
    )
    .ok()?;
    output.status.success().then(|| branch.to_string())
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

/// Whether git still holds a registration row for the worktree at `path`.
///
/// Read from `worktree list --porcelain` rather than inferred from the
/// directory: a row outliving its directory is exactly the stale state this
/// module now has to recognise, and the two are independent.
fn is_registered_worktree(repo_root: &Path, path: &Path) -> bool {
    let Ok(output) = git(
        repo_root,
        "worktree list",
        &["worktree", "list", "--porcelain"],
    ) else {
        // Unprovable registration: assume the worst, so a caller that only
        // recovers from a stale row can never mistake a live one for it.
        return true;
    };
    if !output.status.success() {
        return true;
    }
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| line.strip_prefix("worktree "))
        .any(|registered| Path::new(registered.trim()) == path)
}

/// Drop git's registration row for the worktree at `path`, and only that one.
///
/// A registration whose directory is already gone is what makes the *next*
/// `worktree add` for the same worker fail with "is a missing but already
/// registered worktree" (audit §13: stale worktree registrations), so every exit
/// path that deletes a checkout directory has to unregister it. Escalates,
/// because no single git command covers every state this is called in:
///
/// * `worktree remove --force` is the precise unregister -- it names one
///   worktree -- but it refuses while the directory survives, and it *validates*
///   the directory's `.git` pointer, so it also refuses a half-deleted checkout.
/// * Deleting the directory first clears that second case, after which the
///   narrow command works again.
/// * `worktree prune` is the last resort. It is the *global* sweep, so it only
///   ever runs once the two narrow steps failed, and it can only drop rows git
///   already calls stale -- which by definition excludes a live checkout.
///
/// Never touches another worker's registration: the first two steps name one
/// path, and the last can only clear rows with no live directory. Returns
/// whether `path` ends up unregistered.
fn unregister_worktree(repo_root: &Path, path: &Path) -> bool {
    if !is_registered_worktree(repo_root, path) {
        return true;
    }
    let path_str = path.to_string_lossy().to_string();
    let remove = |repo_root: &Path| -> bool {
        git(
            repo_root,
            "worktree remove",
            &["worktree", "remove", "--force", &path_str],
        )
        .is_ok_and(|out| out.status.success())
    };
    if !path.exists() && remove(repo_root) {
        debug!(path = %path.display(), "Unregistered stale git worktree");
        return true;
    }
    // Either the directory is still there or the narrow step failed. Both git
    // escape hatches refuse a path whose directory exists, so the row can only
    // be cleared once the checkout is gone: delete the directory, then unregister
    // (audit §13: stale worktree registrations).
    //
    // The caller decides this path is abandoned, so deleting it here is the same
    // step `Drop` and the sweep already take. It is *not* taken for a healthy
    // checkout -- that one already unregistered on the first branch above.
    force_remove_dir(path);
    if remove(repo_root) {
        debug!(path = %path.display(), "Unregistered worktree after removing its directory");
        return true;
    }
    // Last resort: the global sweep, which drops every row git calls stale --
    // this one included, and never a live checkout's.
    let pruned = git(repo_root, "worktree prune", &["worktree", "prune"])
        .is_ok_and(|out| out.status.success());
    if pruned && !is_registered_worktree(repo_root, path) {
        debug!(path = %path.display(), "Pruned stale git worktree registration");
        return true;
    }
    warn!(
        path = %path.display(),
        "Could not unregister git worktree; a later add of this path will fail"
    );
    false
}

/// Result of the harness's pre-completion integration step.
#[derive(Debug, PartialEq, Eq)]
pub enum BaseSync {
    Unchanged,
    Merged { branch: String },
    Conflicts { branch: String, files: Vec<String> },
}

/// What an unfinished merge in a checkout is waiting for.
enum PendingMerge {
    None,
    Conflicts(Vec<String>),
    Concluded,
}

/// Result of merging one worker branch into a consolidator's worktree.
#[derive(Debug, PartialEq, Eq)]
pub enum BranchMerge {
    Merged { files: usize },
    Conflicts { files: Vec<String> },
}

/// RAII guard around one subagent's `git` worktree.
pub struct WorktreeGuard {
    pub path: PathBuf,
    pub branch: String,
    pub repo_root: PathBuf,
    /// The root this guard's checkout was filed under.
    ///
    /// The checkout path alone does not say where its private `swe-tmp-<leaf>`
    /// scratch lives: the runner derives that from the scratch *base*, so a
    /// guard built under an injected root must reclaim the companion under that
    /// root. Holding the root also keeps [`Drop`] from reaching into the real
    /// base, where a sibling agent's identically named scratch lives.
    scratch_root: ScratchRoot,
    /// This worker's exclusive build directory, leased on its first heavy
    /// command and held until the guard drops.
    build_dir: Option<crate::cache::BuildDirLease>,
    pub base_commit: String,
    /// Branch checked out at dispatch; detached checkouts have no sync target.
    pub base_branch: Option<String>,
    pub preserve_branch: bool,
    /// Fingerprint of every artifact file seeded into the worktree, keyed by
    /// repository-relative path. A file still matching its entry was never
    /// touched by the worker, so [`WorktreeGuard::sync_artifacts`] leaves the
    /// repository root's copy alone.
    seeded: BTreeMap<String, FileFingerprint>,
}

impl WorktreeGuard {
    /// Re-attach to the branch a finished worker left behind (`worker-<id>`).
    ///
    /// Used by a revision: the branch carries the previous run's checkpoints
    /// and final commit, so the worktree is checked out from the branch tip
    /// instead of `HEAD` (reusing [`WorktreeGuard::new`]'s checkout path, not
    /// a second copy of it). The diff base stays the original `base_commit`,
    /// so uncommitted work plus every checkpoint still reports as one diff.
    /// Errors when the branch is gone -- the orchestrator must know the branch
    /// it reviewed no longer exists instead of silently restarting elsewhere.
    pub fn reopen(repo_root: &Path, worker_id: &str, base_commit: &str) -> Result<Self> {
        Self::reopen_in(&ScratchRoot::from_env(), repo_root, worker_id, base_commit)
    }

    /// [`WorktreeGuard::reopen`] under an explicit scratch root.
    pub fn reopen_in(
        root: &ScratchRoot,
        repo_root: &Path,
        worker_id: &str,
        base_commit: &str,
    ) -> Result<Self> {
        let worker_id = sanitize_worker_id(worker_id);
        let branch = format!("worker-{worker_id}");
        if branch_ref(repo_root, &branch).is_none() {
            anyhow::bail!(
                "Worker branch {branch} no longer exists; the finished worker cannot be revised"
            );
        }
        Self::checkout(
            root,
            repo_root,
            &worker_id,
            &branch,
            branch.as_str(),
            base_commit,
            false,
        )
    }

    pub fn new(repo_root: &Path, worker_id: &str) -> Result<Self> {
        Self::new_in(&ScratchRoot::from_env(), repo_root, worker_id)
    }

    /// [`WorktreeGuard::new`] under an explicit scratch root.
    pub fn new_in(root: &ScratchRoot, repo_root: &Path, worker_id: &str) -> Result<Self> {
        // Branch and directory names derive from the *sanitized* id, so they
        // can never describe different worktrees nor escape `swe_base_dir()`.
        let worker_id = sanitize_worker_id(worker_id);
        let branch = format!("worker-{worker_id}");
        let path = root.join(format!("swe-wt-{worker_id}"));

        // Ensure target directory and branch don't exist.
        force_remove_dir(&path);
        // A stale registration pins its branch: `git branch -D` refuses a branch
        // "used by worktree", so deleting the branch *before* dropping the row
        // silently does nothing, and the `-b` in `checkout` then collides with
        // the branch the previous run left behind. Unregistering first makes the
        // branch deletable and the path reusable, both scoped to this worker
        // (audit §13: stale worktree registrations).
        unregister_worktree(repo_root, &path);
        let _ = git(repo_root, "branch -D", &["branch", "-D", &branch]);

        let branch_out = git(
            repo_root,
            "symbolic-ref",
            &["symbolic-ref", "--quiet", "--short", "HEAD"],
        )?;
        let base_branch = branch_out.status.success().then(|| {
            String::from_utf8_lossy(&branch_out.stdout)
                .trim()
                .to_string()
        });
        let base_commit_out = git(repo_root, "rev-parse HEAD", &["rev-parse", "HEAD"])?;
        if !base_commit_out.status.success() {
            let stderr = String::from_utf8_lossy(&base_commit_out.stderr);
            anyhow::bail!("git rev-parse HEAD failed: {}", stderr.trim());
        }
        let base_commit = String::from_utf8_lossy(&base_commit_out.stdout)
            .trim()
            .to_string();

        let mut guard = Self::checkout(
            root,
            repo_root,
            &worker_id,
            &branch,
            &base_commit,
            &base_commit,
            true,
        )?;
        guard.base_branch = base_branch;
        guard.base_commit = base_commit;
        Ok(guard)
    }

    /// Check out `start_point` at `path` on `branch`: the one checkout path
    /// [`WorktreeGuard::new`] and [`WorktreeGuard::reopen`] share, so neither
    /// duplicates the private-dir creation, the `worktree add`, the mode
    /// hardening, the lease marker or the artifact seeding.
    ///
    /// `fresh_branch` selects `-b` (create, for a dispatch) versus `--detach`-
    /// free re-attach (for a revision, where the branch already exists).
    /// `base_commit` is recorded as the diff base without re-reading `HEAD` of
    /// the repo root, which may have moved on since the original run.
    fn checkout(
        root: &ScratchRoot,
        repo_root: &Path,
        worker_id: &str,
        branch: &str,
        start_point: &str,
        base_commit: &str,
        fresh_branch: bool,
    ) -> Result<Self> {
        let path = root.join(format!("swe-wt-{worker_id}"));

        // A revision never leaves a stale checkout behind: the finished run's
        // `Drop` removed it, but a crashed run may not have (audit §06).
        force_remove_dir(&path);

        // Deleting the checkout is not enough to make the path reusable: git
        // keeps its registration row until it is told to drop it, and
        // `worktree add` then refuses the path outright ("is a missing but
        // already registered worktree"). A crashed run leaves exactly that
        // state, and it used to fail the whole revision until an operator ran
        // `git worktree prune` by hand (audit §13: stale worktree registrations).
        //
        // Scoped to this worker's own path, so recovering one worker never
        // unregisters a live sibling's worktree.
        unregister_worktree(repo_root, &path);

        info!(repo = %repo_root.display(), branch = %branch, path = %path.display(), "Creating git worktree");

        let path_str = path
            .to_str()
            .context("Worktree path contains invalid UTF-8")?;
        let add = |repo_root: &Path| -> Result<std::process::Output> {
            if fresh_branch {
                git(
                    repo_root,
                    "worktree add",
                    &["worktree", "add", "-b", branch, path_str, start_point],
                )
            } else {
                git(
                    repo_root,
                    "worktree add",
                    &["worktree", "add", path_str, start_point],
                )
            }
        };

        // Create it private before checkout so no reader observes files under
        // a permissive umask.
        create_private_worktree_dir(&path)?;
        let mut output = add(repo_root)?;

        // A row can appear between the unregister above and this `add` (a
        // concurrent sweep, a second daemon on the same repository). One retry,
        // and only when the failure is this path's own stale registration --
        // every other failure is reported unchanged, so a real error is never
        // masked by a speculative prune.
        if !output.status.success()
            && is_registered_worktree(repo_root, &path)
            && unregister_worktree(repo_root, &path)
        {
            warn!(
                path = %path.display(),
                "Retrying worktree creation after clearing a stale registration"
            );
            // The failed `add` left its own placeholder behind.
            force_remove_dir(&path);
            create_private_worktree_dir(&path)?;
            output = add(repo_root)?;
        }

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

        Ok(Self {
            path,
            branch: branch.to_string(),
            repo_root: repo_root.to_path_buf(),
            scratch_root: root.clone(),
            build_dir: None,
            base_commit: base_commit.to_string(),
            base_branch: None,
            preserve_branch: false,
            seeded,
        })
    }

    pub fn get_diff(&self) -> Result<String> {
        Self::diff_with_base_at(&self.path, &self.base_commit, self.base_branch.as_deref())
    }

    /// Resolve the shared ancestor with the current base tip, excluding base-only work.
    pub fn diff_base_at(
        path: &Path,
        base_commit: &str,
        base_branch: Option<&str>,
    ) -> Result<String> {
        if let Some(branch) = base_branch {
            let reference = format!("refs/heads/{branch}");
            let output = git(path, "merge-base", &["merge-base", "HEAD", &reference])?;
            if !output.status.success() {
                anyhow::bail!(
                    "git merge-base failed: {}",
                    String::from_utf8_lossy(&output.stderr).trim()
                );
            }
            return Ok(String::from_utf8_lossy(&output.stdout).trim().to_string());
        }
        Ok(if base_commit.is_empty() {
            "HEAD"
        } else {
            base_commit
        }
        .to_string())
    }

    /// Take the worker diff against its shared ancestor with the moving base.
    pub fn diff_with_base_at(
        path: &Path,
        base_commit: &str,
        base_branch: Option<&str>,
    ) -> Result<String> {
        let base = Self::diff_base_at(path, base_commit, base_branch)?;
        Self::diff_at(path, &base)
    }

    /// The final diff of the checkout at `path` against `base_commit`.
    ///
    /// Shared core of [`WorktreeGuard::get_diff`] so the worker's final tail
    /// can run off the runtime thread on owned copies of the paths.
    pub fn diff_at(path: &Path, base_commit: &str) -> Result<String> {
        // Stage untracked files intent-to-add so git diff captures new files.
        let _ = git(path, "add", &["add", "-N", "."]);

        // Diff the working tree against the commit the worker started from, so
        // checkpoint commits made along the way (auto-checkpoints, pause on an
        // LLM error) and the still-uncommitted tail are reported together.
        // Diffing against HEAD would hide everything before the last checkpoint.
        let base = if base_commit.is_empty() {
            "HEAD"
        } else {
            base_commit
        };
        let output = git(path, "diff base", &["diff", base])?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("git diff {base} failed: {}", stderr.trim());
        }
        Ok(String::from_utf8_lossy(&output.stdout).into_owned())
    }

    /// Whether this checkout has a live merge or a committed pending integration.
    pub fn merge_in_progress_at(path: &Path) -> Result<bool> {
        if Self::git_merge_in_progress_at(path)? {
            return Ok(true);
        }
        // The WIP merge already contains the base parent. Its durable trailer
        // keeps completion gated even after worktree metadata is reclaimed.
        let message = checked_git(
            path,
            "read pending integration",
            &["log", "-1", "--format=%B"],
        )?;
        Ok(String::from_utf8_lossy(&message.stdout)
            .lines()
            .any(|line| line == "Worker-Pending-Base-Integration: true"))
    }

    fn git_merge_in_progress_at(path: &Path) -> Result<bool> {
        Ok(git(
            path,
            "rev-parse MERGE_HEAD",
            &["rev-parse", "--verify", "--quiet", "MERGE_HEAD"],
        )?
        .status
        .success())
    }

    /// Integrate the dispatch's base branch before verification, on the harness.
    /// Conflicts remain in place for the model; only a later completion may commit them.
    pub fn sync_base_at(
        path: &Path,
        repo_root: &Path,
        branch: &str,
        base_commit: &str,
        base_branch: Option<&str>,
    ) -> Result<BaseSync> {
        if std::env::var_os("WORKER_SYNC_BASE").as_deref() == Some(std::ffi::OsStr::new("0")) {
            return Ok(BaseSync::Unchanged);
        }
        let Some(base_branch) = base_branch else {
            return Ok(BaseSync::Unchanged);
        };
        Self::sync_reference_at(
            path,
            repo_root,
            branch,
            base_commit,
            &format!("refs/heads/{base_branch}"),
            base_branch,
        )
    }

    /// Integrate a pinned round commit using the same conflict lifecycle as base sync.
    pub fn sync_round_base_at(
        path: &Path,
        repo_root: &Path,
        branch: &str,
        base_commit: &str,
        round_base: &str,
    ) -> Result<BaseSync> {
        if std::env::var_os("WORKER_SYNC_BASE").as_deref() == Some(std::ffi::OsStr::new("0")) {
            return Ok(BaseSync::Unchanged);
        }
        Self::sync_reference_at(path, repo_root, branch, base_commit, round_base, round_base)
    }

    fn sync_reference_at(
        path: &Path,
        repo_root: &Path,
        branch: &str,
        base_commit: &str,
        reference: &str,
        base_branch: &str,
    ) -> Result<BaseSync> {
        let mut merged = false;
        match Self::pending_merge_at(path)? {
            PendingMerge::Conflicts(files) => {
                return Ok(BaseSync::Conflicts {
                    branch: base_branch.to_string(),
                    files,
                });
            }
            // A merge the harness started and the model has since resolved is
            // concluded here, and counts as this call's integration.
            PendingMerge::Concluded => merged = true,
            PendingMerge::None => {}
        }

        let ancestor = checked_git(
            path,
            "resolve integration ancestor",
            &["merge-base", "HEAD", reference],
        )?;
        let base = String::from_utf8_lossy(&ancestor.stdout).trim().to_string();
        let tip = checked_git(
            path,
            "resolve base tip",
            &["rev-parse", "--verify", reference],
        )?;
        if base == String::from_utf8_lossy(&tip.stdout).trim() {
            return Ok(if merged {
                BaseSync::Merged {
                    branch: base_branch.to_string(),
                }
            } else {
                BaseSync::Unchanged
            });
        }

        // The base sync may fast-forward: a worker with nothing of its own yet
        // simply moves to the base tip.
        match Self::merge_reference_at(path, repo_root, branch, base_commit, reference, false)? {
            BranchMerge::Merged { .. } => Ok(BaseSync::Merged {
                branch: base_branch.to_string(),
            }),
            BranchMerge::Conflicts { files } => Ok(BaseSync::Conflicts {
                branch: base_branch.to_string(),
                files,
            }),
        }
    }

    /// What an unfinished merge in the checkout is waiting for.
    fn pending_merge_at(path: &Path) -> Result<PendingMerge> {
        if !Self::merge_in_progress_at(path)? {
            return Ok(PendingMerge::None);
        }
        let files = Self::conflict_markers_at(path)?;
        if !files.is_empty() {
            return Ok(PendingMerge::Conflicts(files));
        }
        checked_git(path, "add resolved merge", &["add", "-A"])?;
        if Self::git_merge_in_progress_at(path)? {
            checked_git(
                path,
                "commit resolved merge",
                &[
                    "-c",
                    "user.name=mini-swe",
                    "-c",
                    "user.email=mini-swe@localhost",
                    "commit",
                    "--no-edit",
                ],
            )?;
        } else {
            Self::commit_staged_at(path, "worker: resolve pending base integration")?;
        }
        Ok(PendingMerge::Concluded)
    }

    fn conflict_markers_at(path: &Path) -> Result<Vec<String>> {
        // Git searches working-tree content even when the index is unmerged.
        let markers = git(
            path,
            "grep conflict markers",
            &[
                "grep",
                "--no-textconv",
                "--untracked",
                "--exclude-standard",
                "-a",
                "-l",
                "-z",
                "-e",
                "^<<<<<<<",
                "--",
            ],
        )?;
        match markers.status.code() {
            Some(0) => Ok(nul_paths(&markers.stdout)),
            Some(1) => Ok(Vec::new()),
            _ => anyhow::bail!(
                "Could not check conflict markers: {}",
                String::from_utf8_lossy(&markers.stderr).trim()
            ),
        }
    }

    /// Preserve a pending integration without treating its markers as resolved.
    fn preserve_pending_integration_at(path: &Path) -> Result<()> {
        if !Self::merge_in_progress_at(path)? {
            return Ok(());
        }
        let files = Self::conflict_markers_at(path)?;
        let dirty = checked_git(
            path,
            "status pending integration",
            &["status", "--porcelain"],
        )?;
        if !Self::git_merge_in_progress_at(path)? && dirty.stdout.is_empty() {
            return Ok(());
        }
        checked_git(path, "stage pending integration", &["add", "-A"])?;
        Self::commit_staged_at(
            path,
            &format!(
                "worker: WIP base integration (unresolved: {})\n\nWorker-Pending-Base-Integration: true",
                files.join(", ")
            ),
        )?;
        Ok(())
    }

    /// Merge `reference` into the checkout at `path`, on the harness.
    ///
    /// The core both harness integrations share: uncommitted work is
    /// checkpointed first, no editor is opened, and a conflict is left in the
    /// worktree for the model to resolve. `no_ff` forces a merge commit, which
    /// a consolidator's integration needs so every merged worker stays visible
    /// on its branch.
    fn merge_reference_at(
        path: &Path,
        repo_root: &Path,
        branch: &str,
        base_commit: &str,
        reference: &str,
        no_ff: bool,
    ) -> Result<BranchMerge> {
        Self::commit_changes_at(
            path,
            repo_root,
            branch,
            base_commit,
            "worker: checkpoint before merge",
        )?;
        let before = checked_git(path, "record merge base", &["rev-parse", "HEAD"])?;
        let before = String::from_utf8_lossy(&before.stdout).trim().to_string();
        let mut args = vec![
            "-c",
            "user.name=mini-swe",
            "-c",
            "user.email=mini-swe@localhost",
            "merge",
        ];
        if no_ff {
            args.push("--no-ff");
        }
        args.extend(["--no-edit", reference]);
        let output = git(path, "merge branch", &args)?;
        if !output.status.success() {
            let conflicts = checked_git(
                path,
                "list merge conflicts",
                &["diff", "--name-only", "--diff-filter=U", "-z"],
            )?;
            let files = nul_paths(&conflicts.stdout);
            if Self::merge_in_progress_at(path)? && !files.is_empty() {
                return Ok(BranchMerge::Conflicts { files });
            }
            anyhow::bail!(
                "git merge {reference} failed: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            );
        }
        let after = checked_git(path, "record merge head", &["rev-parse", "HEAD"])?;
        let after = String::from_utf8_lossy(&after.stdout).trim().to_string();
        let files = if after == before {
            0
        } else {
            let changed = checked_git(
                path,
                "count merged files",
                &["diff", "--name-only", &format!("{before}..{after}")],
            )?;
            String::from_utf8_lossy(&changed.stdout)
                .lines()
                .filter(|line| !line.trim().is_empty())
                .count()
        };
        Ok(BranchMerge::Merged { files })
    }

    /// Merge `worker-<id>` into the checkout at `path`, on the harness.
    ///
    /// The consolidator's `CONSOLIDATE_MERGE` verb: the same machinery as the
    /// base sync, always as a merge commit, plus the file count its observation
    /// reports so the model sees what each merge brought in without running git
    /// itself. A branch the repository no longer carries is an error the caller
    /// reports as a refusal, not a silent skip.
    pub fn merge_branch_at(
        path: &Path,
        repo_root: &Path,
        branch: &str,
        base_commit: &str,
        worker_id: &str,
    ) -> Result<BranchMerge> {
        let worker_branch = format!("worker-{worker_id}");
        if branch_ref(repo_root, &worker_branch).is_none() {
            anyhow::bail!("branch {worker_branch} does not exist");
        }
        match Self::pending_merge_at(path)? {
            PendingMerge::Conflicts(files) => return Ok(BranchMerge::Conflicts { files }),
            PendingMerge::Concluded | PendingMerge::None => {}
        }
        Self::merge_reference_at(
            path,
            repo_root,
            branch,
            base_commit,
            &format!("refs/heads/{worker_branch}"),
            true,
        )
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
    /// Returns the sorted, duplicate-free list of repository-relative files
    /// the worker created or changed, so a status view never has to carry the
    /// seeded files the repo already held. The copy itself is conservative:
    /// unchanged files are left untouched, dependency caches and build output
    /// (`SKIP_DIR_NAMES`) are never mirrored, and each file is published
    /// atomically so a concurrent reader never observes a partially written
    /// artifact.
    ///
    /// Per-directory I/O errors are logged rather than propagated: both call
    /// sites discard the `Result` (`pool::runner` wants the artifact count,
    /// `Drop` is a best-effort safety net), so failing here would abort a
    /// worker's teardown over a single unreadable report.
    pub fn sync_artifacts(&self) -> Vec<String> {
        Self::sync_artifacts_at(&self.path, &self.repo_root, &self.seeded)
    }

    /// Sync artifacts from `path` (a worker's checkout) back into `repo_root`,
    /// skipping files still matching `seeded`.
    ///
    /// Shared core of [`WorktreeGuard::sync_artifacts`] so the worker's final
    /// tail can run off the runtime thread on owned copies of the paths.
    pub fn sync_artifacts_at(
        path: &Path,
        repo_root: &Path,
        seeded: &BTreeMap<String, FileFingerprint>,
    ) -> Vec<String> {
        let mut synced = BTreeMap::new();

        for dir in ARTIFACT_DIRS {
            let src_dir = path.join(dir);
            if src_dir.is_dir() {
                let dest_dir = repo_root.join(dir);
                if let Err(e) = copy_dir_all(&src_dir, &dest_dir, &mut synced, path, Some(seeded)) {
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

    /// The seeded artifact fingerprints, cloned so the final tail can cross
    /// into a `spawn_blocking` thread.
    pub fn seeded(&self) -> BTreeMap<String, FileFingerprint> {
        self.seeded.clone()
    }

    /// The build directory this worker's commands build in, leasing one on the
    /// first call.
    ///
    /// The lease is exclusive for the guard's lifetime and is dropped with it,
    /// so a completed, failed or killed worker releases its directory for the
    /// next one while a live worker never shares one with another.
    pub(crate) async fn build_dir(&mut self) -> Option<PathBuf> {
        if self.build_dir.is_none()
            && let Some(lease) = self.lease_build_dir().await
        {
            self.build_dir = Some(lease);
        }
        self.build_dir
            .as_ref()
            .map(|lease| lease.dir().to_path_buf())
    }

    /// The build directory this worker already leased, without leasing one.
    ///
    /// A light command reuses the worker's directory when it has one and
    /// otherwise builds in its worktree, so a worker that never runs a heavy
    /// command never takes a directory from the pool.
    pub(crate) fn leased_build_dir(&self) -> Option<&Path> {
        self.build_dir.as_ref().map(|lease| lease.dir())
    }

    /// Acquire the lease off the runtime: it waits on the sweep lock.
    async fn lease_build_dir(&mut self) -> Option<crate::cache::BuildDirLease> {
        let repo = self.repo_root.clone();
        let acquired =
            tokio::task::spawn_blocking(move || crate::cache::BuildDirLease::acquire(&repo)).await;
        match acquired {
            Ok(Ok(lease)) => Some(lease),
            Ok(Err(error)) => {
                debug!(%error, repo = %self.repo_root.display(), "Build directory lease unavailable");
                None
            }
            Err(error) => {
                debug!(%error, repo = %self.repo_root.display(), "Build directory lease task failed");
                None
            }
        }
    }

    /// Stage and commit the worktree at `path`, reporting what was left out.
    ///
    /// Split out of [`WorktreeGuard::commit_changes`] so the pool's `kill`
    /// path can commit the uncommitted work of a worker whose guard it does
    /// not own: a `kill` must not lose work, and the aborted task tears its
    /// worktree down on the way out. A clean tree is not an error and simply
    /// reports no commit, so the caller can stay quiet about it.
    ///
    /// Every harness commit goes through here, so the staging rules of
    /// [`stage_commitable_changes_at`] hold for auto-checkpoints, the final
    /// commit and the checkpoint before a merge alike.
    pub(crate) fn commit_all(
        path: &Path,
        base_commit: &str,
        message: &str,
    ) -> Result<CommitReport> {
        // Completion owns merge resolution; checkpoints must not commit markers.
        if Self::merge_in_progress_at(path)? {
            anyhow::bail!("Base merge is still in progress; resolve it and request completion");
        }
        let skipped = Self::stage_commitable_changes_at(path, base_commit)?;

        // Check if there are changes to commit.
        let status = git(path, "status", &["status", "--porcelain"])?;
        if !status.status.success() {
            let stderr = String::from_utf8_lossy(&status.stderr);
            anyhow::bail!("git status failed: {}", stderr.trim());
        }
        if status.stdout.is_empty() {
            return Ok(CommitReport {
                branch: None,
                skipped,
            });
        }

        Self::commit_staged_at(path, message)?;
        log_skipped(path, &skipped);
        Ok(CommitReport {
            branch: None,
            skipped,
        })
    }

    fn commit_staged_at(path: &Path, message: &str) -> Result<()> {
        // Commit with fallback credentials so lack of git config never errors.
        let commit_out = git(
            path,
            "commit",
            &[
                "-c",
                "user.name=mini-swe",
                "-c",
                "user.email=mini-swe@localhost",
                "commit",
                "--allow-empty",
                "-m",
                message,
            ],
        )?;
        if !commit_out.status.success() {
            let stderr = String::from_utf8_lossy(&commit_out.stderr);
            anyhow::bail!("git commit failed: {}", stderr.trim());
        }
        Ok(())
    }

    /// Commit all dirty changes in the worktree to preserve work in git history,
    /// marking the branch to be retained upon worktree cleanup.
    pub fn commit_changes(&mut self, message: &str) -> Result<CommitReport> {
        let report = Self::commit_changes_at(
            &self.path,
            &self.repo_root,
            &self.branch,
            &self.base_commit,
            message,
        )?;
        if report.committed() {
            self.preserve_branch = true;
        }
        Ok(report)
    }

    /// Shared core of [`WorktreeGuard::commit_changes`] on owned paths, so it
    /// can run off the runtime thread: commit the checkout at `path` when it is
    /// dirty, and name `branch` when it carries work (new or already
    /// committed beyond `base_commit`). The caller marks the branch preserved.
    pub fn commit_changes_at(
        path: &Path,
        repo_root: &Path,
        branch: &str,
        base_commit: &str,
        message: &str,
    ) -> Result<CommitReport> {
        let mut report = Self::commit_all(path, base_commit, message)?;
        if report.committed() || Self::branch_has_commits_at(repo_root, base_commit, branch) {
            report.branch = Some(branch.to_string());
        }
        Ok(report)
    }

    /// Stage every change the harness may commit, and report what it refused.
    ///
    /// This is the gate every harness commit stages its files through, so a
    /// compile cache a worker grew inside its own worktree (a `HOME` pointed at
    /// `.envcheck/home`, say) can never reach the index and the history a
    /// remote pulls. The rules, in order:
    ///
    /// * a deletion is always staged, because it only shrinks the tree;
    /// * a path the base commit already tracks is always staged, so a
    ///   repository that legitimately carries a large fixture or a `target/`
    ///   directory keeps committing it;
    /// * anything else is refused when a path component names a cache or tool
    ///   directory ([`SKIP_DIR_NAMES`]), when it sits under a directory the
    ///   worker created that holds a `.cache` entry, or when the file is larger
    ///   than the commit cap.
    ///
    /// A refused path is never staged, so its content never reaches the object
    /// database, and anything an earlier pass left staged for it is unstaged.
    /// The caller reports the refusals to the worker.
    fn stage_commitable_changes_at(path: &Path, base_commit: &str) -> Result<Vec<SkippedPath>> {
        let status = git(
            path,
            "status",
            &["status", "--porcelain", "-z", "--untracked-files=all"],
        )?;
        if !status.status.success() {
            anyhow::bail!(
                "git status failed: {}",
                String::from_utf8_lossy(&status.stderr).trim()
            );
        }
        let changed = status_entries(&status.stdout);
        if changed.is_empty() {
            return Ok(Vec::new());
        }
        let cap = commit_file_cap_bytes();
        let base = base_tree_at(path, base_commit)?;
        let mut keep: Vec<String> = Vec::new();
        let mut skipped: Vec<SkippedPath> = Vec::new();
        for (index_status, worktree_status, rel) in changed {
            let deleted = index_status == 'D' || worktree_status == 'D';
            if deleted || base.tracks(&rel) {
                keep.push(rel);
                continue;
            }
            match Self::commit_refusal(path, &rel, cap, &base) {
                Some((reason, bytes)) => skipped.push(SkippedPath { path: rel, bytes, reason }),
                None => keep.push(rel),
            }
        }

        // A path an earlier pass staged (`add -A` during a merge, `add -N`
        // while diffing) must not survive in the index once it is refused:
        // unstaging it keeps the blob out of the commit this pass is about to
        // make.
        if !skipped.is_empty() {
            let refused: Vec<String> = skipped.iter().map(|s| s.path.clone()).collect();
            run_pathspecs(path, "reset", &["reset", "--quiet"], &refused);
        }
        log_skipped(path, &skipped);
        if keep.is_empty() {
            return Ok(skipped);
        }
        run_pathspecs(path, "add", &["add"], &keep);
        Ok(skipped)
    }

    /// Why a new path must not be staged, or `None` when it may be.
    ///
    /// `None` also covers a file that vanished between the status read and this
    /// pass: staging it would fail the whole `git add`, and the next
    /// checkpoint's status reports the deletion and stages it then.
    fn commit_refusal(
        worktree: &Path,
        rel: &str,
        cap: u64,
        base: &BaseTree,
    ) -> Option<(SkipReason, Option<u64>)> {
        let Ok(meta) = std::fs::metadata(worktree.join(rel)) else {
            return None;
        };
        let bytes = meta.len();
        if rel.split('/').any(is_skipped_dir_name) || throwaway_home(worktree, rel, base) {
            return Some((SkipReason::CacheDir, Some(bytes)));
        }
        if cap > 0 && bytes > cap {
            return Some((SkipReason::TooLarge, Some(bytes)));
        }
        None
    }

    /// True when this worker's branch carries commits beyond `base_commit`.
    ///
    /// Shared by [`WorktreeGuard::commit_changes`] and [`Drop`]: a branch with
    /// committed work is never deleted, whatever else cleanup decides.
    fn branch_has_commits(&self) -> bool {
        Self::branch_has_commits_at(&self.repo_root, &self.base_commit, &self.branch)
    }

    fn branch_has_commits_at(repo_root: &Path, base_commit: &str, branch: &str) -> bool {
        if base_commit.is_empty() {
            return false;
        }
        git(
            repo_root,
            "rev-list",
            &["rev-list", "--count", &format!("{base_commit}..{branch}")],
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

    /// Mark the worktree at `path` as belonging to a worker the hub
    /// interrupted.
    ///
    /// Called by the hub's shutdown path before the worker's task is
    /// aborted: a planned shutdown ends the worker `Interrupted`, so
    /// its branch (and therefore its continuability) must outlive the
    /// worktree teardown, even when the worker never committed
    /// anything.
    ///
    /// The pool only tracks the checkout *path* -- the guard itself
    /// lives inside the worker's task, which the abort drops after
    /// this method has returned -- so the mark is a file beside the
    /// worktree, named from the worktree's own directory: the guard's
    /// `Drop` reads it and `clear_interrupted_marker` removes it
    /// once honoured.
    pub fn mark_interrupted(path: &Path) {
        let _ = std::fs::write(interrupted_marker_path(path), b"interrupted");
    }

    /// Mark the worktree at `path` as still being torn down by this process.
    ///
    /// The hub's shutdown writes it before it aborts a worker and awaits the
    /// guard's drop; the drop removes it (via `TeardownMarker`) once the
    /// checkpoint, the worktree removal/unregistration and the target cleanup
    /// have all finished. A replacement hub refuses to recover the worker while
    /// the marker names a live owner, so a teardown that outlived the bounded
    /// shutdown wait can never race the recovery that recreates the worktree
    /// (H18). The marker carries this process's pid so a marker left by a
    /// crashed teardown is recognised as stale instead of blocking forever.
    pub fn mark_teardown(path: &Path) {
        let _ = std::fs::write(
            crate::worktree::teardown_marker_for(path),
            std::process::id().to_string(),
        );
    }

    /// Whether `path`'s worktree teardown is still owned by a live process.
    ///
    /// A marker left by a process that is gone (a crash mid-teardown) names no
    /// owner, so it is removed here and reported as not pending: failing open
    /// keeps a crashed teardown from stranding the worker forever.
    pub fn teardown_pending(path: &Path) -> bool {
        let marker = crate::worktree::teardown_marker_for(path);
        let Ok(contents) = std::fs::read_to_string(&marker) else {
            return false;
        };
        let owner = contents
            .lines()
            .next()
            .and_then(|line| line.trim().parse::<u32>().ok());
        if owner.is_some_and(crate::worktree::is_process_alive) {
            return true;
        }
        let _ = std::fs::remove_file(&marker);
        false
    }

    /// Whether this guard belongs to an interrupted worker.
    ///
    /// The marker file is what both the shutdown path and the
    /// abort-drop agree on: the former writes it, the latter reads
    /// and consumes it.
    fn interrupted(&self) -> bool {
        self.interrupted_marker_path().is_file()
    }

    /// Remove the interruption marker.
    ///
    /// [`Drop`] calls this after the branch decision, honoured or
    /// not, so a *later* run of the same worker id starts unmarked:
    /// its teardown prunes a branch that points nowhere past the base
    /// again.
    fn clear_interrupted_marker(&self) {
        let _ = std::fs::remove_file(self.interrupted_marker_path());
    }

    /// The interruption marker path: beside the worktree, named from
    /// the worktree's own directory, so the shutdown path and the
    /// dropping task agree on it.
    fn interrupted_marker_path(&self) -> PathBuf {
        interrupted_marker_path(&self.path)
    }

    /// Whether `refs/heads/{branch}` still resolves.
    ///
    /// A branch that never resolved was never a worker's branch at
    /// all (a worktree whose branch someone already deleted), so
    /// nothing needs preserving; this keeps the interrupted path
    /// from "preserving" a branch that has nothing to preserve.
    fn branch_ref_exists(&self) -> bool {
        branch_ref(&self.repo_root, &self.branch).is_some()
    }

    /// The worker id this guard belongs to, recovered from the worktree
    /// directory name (`swe-wt-<id>`): the sweep logs it, and the id is not
    /// stored on the guard.
    fn worker_id(&self) -> String {
        self.path
            .file_name()
            .and_then(|name| name.to_str())
            .and_then(|name| name.strip_prefix("swe-wt-"))
            .unwrap_or("unknown")
            .to_string()
    }

    /// Every directory this worker's commands may run in: the worktree, its
    /// private scratch dir, its target dirs and the build dir it leased.
    ///
    /// This list is the whole definition of "belongs to this worker" for the
    /// sweep in [`crate::agent::reap`], and it deliberately names no other
    /// worker's directories, so a sweep can never take down a live sibling.
    fn worker_process_dirs(&self) -> Vec<PathBuf> {
        let mut dirs = crate::agent::reap::worker_dirs(&self.path);
        if let Some(build_dir) = self.leased_build_dir() {
            dirs.push(build_dir.to_path_buf());
        }
        dirs
    }

    /// Test hook: slow this worker's teardown by `delay`, once.
    ///
    /// A handover test needs the old daemon's worktree cleanup to take long
    /// enough to observe the ordering guarantee; the delay is keyed by worker
    /// id so parallel tests are unaffected, and is consumed on use.
    pub fn __test_set_teardown_delay(worker_id: &str, delay: std::time::Duration) {
        teardown_delays()
            .lock()
            .expect("teardown delay lock poisoned")
            .insert(worker_id.to_string(), delay);
    }

    /// Clear a teardown delay set by [`Self::__test_set_teardown_delay`].
    pub fn __test_clear_teardown_delay(worker_id: &str) {
        teardown_delays()
            .lock()
            .expect("teardown delay lock poisoned")
            .remove(worker_id);
    }
}

/// Per-worker teardown delays for tests; empty in production.
fn teardown_delays() -> &'static std::sync::Mutex<BTreeMap<String, std::time::Duration>> {
    static DELAYS: std::sync::Mutex<BTreeMap<String, std::time::Duration>> =
        std::sync::Mutex::new(BTreeMap::new());
    &DELAYS
}

/// Take the teardown delay for `worker_id`, if a test set one.
fn take_teardown_delay(worker_id: &str) -> Option<std::time::Duration> {
    teardown_delays()
        .lock()
        .expect("teardown delay lock poisoned")
        .remove(worker_id)
}

impl Drop for WorktreeGuard {
    fn drop(&mut self) {
        // Hold the teardown marker for the whole drop, on every exit path
        // including the early `return` below: the hub writes it before the
        // abort, and this removes it only once the teardown is really done, so
        // a replacement hub waiting on it can never recreate the worktree while
        // it is still being removed.
        let _teardown = TeardownMarker(self.path.clone());
        // A test can hold teardown open to prove the replacement daemon waits
        // for it; production never sets a delay, so this is a no-op there.
        if let Some(delay) = take_teardown_delay(&self.worker_id()) {
            std::thread::sleep(delay);
        }
        // A job the worker detached from every process group (`setsid cmd &`, a
        // double fork) is reparented to init and outlives its step, so the only
        // thing that still ties it to this worker is its working directory.
        // This runs before anything is deleted: the directories are what the
        // sweep matches on, and a live process may still be writing into the
        // tree that is about to be salvaged.
        let dirs = self.worker_process_dirs();
        crate::agent::reap::sweep_worker_processes(&self.worker_id(), &dirs);

        // Teardown must retain partially resolved hunks on every exit path.
        if let Err(e) = Self::preserve_pending_integration_at(&self.path) {
            warn!(path = %self.path.display(), error = %e, "Keeping worktree after integration salvage failed");
            // The checkout is kept because it may still hold unresolved work --
            // but only if it is still one. A directory whose git metadata is
            // already gone (a half-deleted checkout) cannot be salvaged, holds
            // nothing, and would otherwise leave a registration that fails every
            // later attempt to create this worker's worktree (audit §13: stale worktree registrations).
            unregister_worktree(&self.repo_root, &self.path);
            return;
        }

        let pid_file = pid_file_for(&self.path);

        // Sync report/audit artifacts to repo root before cleanup. This is the
        // safety net for teardown paths that never reached the explicit call in
        // `pool::runner`; a second run is cheap because unchanged files are
        // skipped rather than recopied.
        self.sync_artifacts();
        info!(path = %self.path.display(), branch = %self.branch, "Cleaning up git worktree");

        // `worktree remove --force` is the precise unregister, but it fails
        // when the directory has already gone or holds no `.git` pointer, and
        // a surviving row is what makes the *next* revision of this worker
        // fail (audit §13: stale worktree registrations). Its result is deliberately not checked here: the
        // authoritative cleanup is the unregister below, which also covers the
        // states `remove` refuses.
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
        // An interrupted worker may have made no commit at all -- its
        // branch then points exactly at the base commit -- but the
        // branch must survive regardless: the next daemon's recovery
        // auto-continues an Interrupted worker by re-attaching to it,
        // so deleting the ref would strand the worker's row and
        // history with nothing to continue on. The marker is consumed
        // here, honoured or not, so a *later* run of the same worker
        // id that ends some other way prunes its branch again. A
        // worker that ended some other way keeps deleting a branch
        // that points nowhere past the base.
        let interrupted =
            !self.preserve_branch && !has_commits && self.interrupted() && self.branch_ref_exists();
        self.clear_interrupted_marker();

        if self.preserve_branch || has_commits || interrupted {
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
        // Then drop the registration itself. Ordering matters: git refuses to
        // unregister a path whose directory still exists without a valid
        // `.git` pointer, so the directory has to be gone first, and the
        // registration has to be gone before this worker can ever be recreated
        // (audit §13: stale worktree registrations).
        unregister_worktree(&self.repo_root, &self.path);
        let _ = std::fs::remove_file(&pid_file);
        remove_target_dirs_in(&self.scratch_root, &self.path);
    }
}

/// The interruption marker path of the worktree at `path`: a sibling
/// file named from the worktree's own directory.
///
/// [`WorktreeGuard::mark_interrupted`] writes it from the shutdown
/// path; the guard's [`Drop`] reads and removes it. Keeping the
/// naming in one free function lets both sides derive it from a bare
/// path, which is all the pool tracks.
fn interrupted_marker_path(path: &Path) -> PathBuf {
    let mut marker = path.as_os_str().to_os_string();
    marker.push(".interrupted");
    PathBuf::from(marker)
}

/// Removes a worktree's teardown marker when the teardown that owns it ends.
///
/// Constructed at the start of [`WorktreeGuard`]'s drop and dropped at its end,
/// so the marker outlives the checkpoint, worktree removal, unregistration and
/// target cleanup on *every* exit path, including an early return and a panic.
struct TeardownMarker(PathBuf);

impl Drop for TeardownMarker {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(crate::worktree::teardown_marker_for(&self.0));
    }
}

/// How many paths one `git add` / `git reset` invocation takes.
///
/// Small enough that a change set of tens of thousands of files cannot push a
/// single argv past the kernel's argument limit, large enough that the spawn
/// cost stays negligible next to the commit itself.
const PATHSPEC_CHUNK: usize = 256;

/// Run `git args…` once per chunk of `paths`, each path wrapped in literal
/// pathspec magic so a filename that starts with `:` or holds a glob character
/// is never read as a pattern.
///
/// A chunk git refuses is logged and skipped rather than propagated: the
/// `git add -A` this replaces was ignored on failure too, and one unreadable
/// path must not cost a worker its whole checkpoint.
fn run_pathspecs(path: &Path, operation: &str, args: &[&str], paths: &[String]) {
    for chunk in paths.chunks(PATHSPEC_CHUNK) {
        let specs: Vec<String> = chunk.iter().map(|p| format!(":(literal){p}")).collect();
        let mut argv: Vec<&str> = args.to_vec();
        argv.push("--");
        argv.extend(specs.iter().map(String::as_str));
        match git(path, operation, &argv) {
            Ok(output) if output.status.success() => {}
            Ok(output) => warn!(
                worktree = %path.display(),
                paths = chunk.len(),
                error = %String::from_utf8_lossy(&output.stderr).trim(),
                "git {} could not stage every path it was given", operation
            ),
            Err(error) => warn!(
                worktree = %path.display(),
                error = %error,
                "git {} could not run", operation
            ),
        }
    }
}

/// The paths and directories a commit's base commit tracks.
///
/// A path the base tracks is pre-existing work, so neither the size cap nor the
/// cache rules apply to it: the repository chose to carry it before this worker
/// started, and it is already in history.
#[derive(Default)]
struct BaseTree {
    files: HashSet<String>,
    dirs: HashSet<String>,
}

impl BaseTree {
    /// True when `rel` is a file the base commit tracks.
    fn tracks(&self, rel: &str) -> bool {
        self.files.contains(rel)
    }

    /// True when `dir`, a repository-relative directory, existed in the base.
    fn tracks_dir(&self, dir: &str) -> bool {
        self.dirs.contains(dir)
    }
}

/// Read the tree `base_commit` tracks.
///
/// A caller without a base (the pool's kill path) reads the checkout's own
/// `HEAD` instead: whatever that commit carries is in history either way, so it
/// keeps being committed normally. A base the checkout cannot read yields an
/// empty tree, which refuses nothing by the tracked rule and leaves the cache
/// and size rules to apply to every path.
fn base_tree_at(path: &Path, base_commit: &str) -> Result<BaseTree> {
    let reference = if base_commit.is_empty() { "HEAD" } else { base_commit };
    let out = git(
        path,
        "ls-tree",
        &["ls-tree", "-r", "--name-only", "-z", reference],
    )?;
    let mut tree = BaseTree::default();
    if !out.status.success() {
        debug!(
            base = %reference,
            "Could not read the base tree; the commit rules apply to every path"
        );
        return Ok(tree);
    }
    for entry in out.stdout.split(|b| *b == 0).filter(|e| !e.is_empty()) {
        let rel = String::from_utf8_lossy(entry).into_owned();
        let mut dir = rel.as_str();
        while let Some((parent, _)) = dir.rsplit_once('/') {
            dir = parent;
            tree.dirs.insert(dir.to_string());
        }
        tree.files.insert(rel);
    }
    Ok(tree)
}

/// Whether `rel` sits under a throwaway tool home the worker created.
///
/// A worker that points `HOME` (or one tool's state directory) inside its
/// worktree fills that directory with caches and tool state. The marker is a
/// `.cache` entry: a directory the base commit does not track that holds one is
/// a home a tool made for itself, and everything under it is tool state rather
/// than work product. The checkout root is never one, so a repository that
/// keeps a `.cache` of its own is unaffected -- its paths are refused by name
/// instead.
fn throwaway_home(worktree: &Path, rel: &str, base: &BaseTree) -> bool {
    let mut dir = rel;
    while let Some((parent, _)) = dir.rsplit_once('/') {
        dir = parent;
        if base.tracks_dir(dir) {
            continue;
        }
        if worktree.join(dir).join(".cache").is_dir() {
            return true;
        }
    }
    false
}

/// The changed paths `git status --porcelain -z` reported, as
/// `(index status, worktree status, path)` triples.
///
/// `-z` keeps the records NUL-terminated and the paths unquoted, so a filename
/// holding a quote or a newline survives the round trip. A rename or a copy
/// carries its origin as a second NUL-terminated field, consumed here and
/// dropped: the new name is the one that has to be staged.
fn status_entries(bytes: &[u8]) -> Vec<(char, char, String)> {
    let mut out = Vec::new();
    let mut fields = bytes.split(|b| *b == 0).filter(|f| !f.is_empty());
    while let Some(field) = fields.next() {
        let text = String::from_utf8_lossy(field).into_owned();
        let mut chars = text.chars();
        let (Some(x), Some(y), Some(' ')) = (chars.next(), chars.next(), chars.next()) else {
            continue;
        };
        if x == 'R' || x == 'C' || y == 'R' || y == 'C' {
            fields.next();
        }
        out.push((x, y, text[3..].to_string()));
    }
    out
}

/// Log the paths a harness commit refused, once per commit, at INFO.
fn log_skipped(worktree: &Path, skipped: &[SkippedPath]) {
    if skipped.is_empty() {
        return;
    }
    let named = skipped
        .iter()
        .take(5)
        .map(|s| s.path.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    info!(
        worktree = %worktree.display(),
        count = skipped.len(),
        paths = %named,
        "Left cache or oversized paths out of the harness commit"
    );
}

fn checked_git(path: &Path, operation: &str, args: &[&str]) -> Result<std::process::Output> {
    let output = git(path, operation, args)?;
    if !output.status.success() {
        anyhow::bail!(
            "git {operation} failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(output)
}

fn nul_paths(bytes: &[u8]) -> Vec<String> {
    let mut paths: Vec<_> = bytes
        .split(|b| *b == 0)
        .filter(|p| !p.is_empty())
        .map(|p| String::from_utf8_lossy(p).into_owned())
        .collect();
    paths.sort();
    paths.dedup();
    paths
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
pub struct FileFingerprint {
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
            // Only the worker's own output is collected. A seeded file whose
            // content still matches is not the worker's artifact, so reporting
            // it would inflate every completion view with pre-existing files.
            if !inherited && let (Some(rel), Some(fingerprint)) = (rel, fingerprint) {
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
            assert!(
                id.bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
            );
        }
        assert_ne!(sanitize_worker_id(""), sanitize_worker_id(""));

        // The cap is enforced before the trim, so a long id cannot outrun it.
        assert!(sanitize_worker_id(&"x".repeat(500)).len() <= MAX_WORKER_ID_LEN);
    }

    /// A guard over `repo` with no worktree on disk: the build-dir lease is
    /// the only part under test here.
    fn guard_for(repo: &Path) -> WorktreeGuard {
        WorktreeGuard {
            path: repo.join("worktree"),
            branch: "worker-test".to_string(),
            repo_root: repo.to_path_buf(),
            scratch_root: ScratchRoot::from_env(),
            build_dir: None,
            base_commit: String::new(),
            base_branch: None,
            preserve_branch: false,
            seeded: BTreeMap::new(),
        }
    }

    /// Whether another worker could still take `dir`: the lock file a lease
    /// holds is what the sweep probes to decide a dir is busy.
    fn dir_is_free(dir: &Path) -> bool {
        let file = match std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(dir.join(".swe-target.lease"))
        {
            Ok(file) => file,
            Err(_) => return false,
        };
        // SAFETY: `file` owns a live descriptor for the duration of the call.
        let locked = unsafe {
            libc::flock(
                std::os::fd::AsRawFd::as_raw_fd(&file),
                libc::LOCK_EX | libc::LOCK_NB,
            )
        };
        if locked == 0 {
            // SAFETY: releasing a lock this call just took.
            unsafe { libc::flock(std::os::fd::AsRawFd::as_raw_fd(&file), libc::LOCK_UN) };
            true
        } else {
            false
        }
    }

    fn repo_dir(tag: &str) -> PathBuf {
        let path = crate::worktree::swe_base_dir()
            .join(format!("swe-lease-{tag}-{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&path).expect("repository root must be creatable");
        path
    }

    /// The worker's dir is leased once and held for its whole lifetime, so a
    /// light step between two heavy ones cannot lose it to another worker.
    #[tokio::test]
    async fn a_workers_build_dir_survives_light_and_heavy_alternation() {
        let repo = repo_dir("alternating");
        let mut worker = guard_for(&repo);

        assert_eq!(
            worker.leased_build_dir(),
            None,
            "a worker that ran no heavy command must hold no dir"
        );
        let first = worker
            .build_dir()
            .await
            .expect("a heavy step must lease a dir");
        assert_eq!(
            worker.leased_build_dir(),
            Some(first.as_path()),
            "a light step must reuse the worker's dir"
        );
        assert_eq!(
            worker.build_dir().await.as_deref(),
            Some(first.as_path()),
            "a later heavy step must reuse the worker's dir"
        );
        assert!(!dir_is_free(&first), "the worker must hold its dir");

        // A second live worker of the same repository gets a dir of its own.
        let mut other = guard_for(&repo);
        let second = other
            .build_dir()
            .await
            .expect("a second worker must lease a dir");
        assert_ne!(second, first, "two live workers shared {}", first.display());

        drop(worker);
        assert!(
            dir_is_free(&first),
            "ending the worker must release its dir"
        );
        assert!(!dir_is_free(&second), "the other worker must keep its dir");

        drop(other);
        let _ = std::fs::remove_dir_all(&first);
        let _ = std::fs::remove_dir_all(&second);
        let _ = std::fs::remove_dir_all(&repo);
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
