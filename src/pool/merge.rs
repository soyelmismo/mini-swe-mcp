//! One-command merge of a finished worker's branch into its base branch.
//!
//! Merging a worker by hand is eight commands: trial-merge against the base,
//! run the gate, `git merge --no-ff`, delete the branch, drop the history file,
//! reclaim the worktree. Each one is a chance to merge into the wrong branch,
//! to merge over uncommitted work, or to leave a branch and a worktree behind.
//! This module performs the whole sequence as one auditable operation.
//!
//! The invariants it keeps, in the order they are enforced:
//!
//! * **Nothing is merged into a branch the operator did not name.** The target
//!   is the base branch recorded for the worker, and the repository must
//!   already have it checked out; a merge never moves `HEAD` itself.
//! * **The operator's working tree is never touched.** Untracked and unrelated
//!   local changes are left exactly as they are -- no stash, no checkout, no
//!   reset. Only a file the merge would actually write is a reason to refuse.
//! * **The gate runs on the merge result, not on the branch tip.** The merge is
//!   materialised in a throwaway worktree under the scratch root, so a branch
//!   that only passes alone cannot land.
//! * **A refusal leaves no trace.** Every check that can fail runs before the
//!   real merge, so a refused merge changes neither the repository nor the
//!   worker's files.
//!
//! The gate itself is deliberately the same command the worker ran -- the one
//! its dispatch named, or the auto-detected one -- replayed verbatim; nothing
//! here assumes a language or a test runner.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result};

use super::registry::load_registry_entry_in;
use super::revision::{WorkerHistory, load_worker_history_log_in, remove_worker_history_in};
use crate::worktree::{ScratchRoot, force_remove_dir, git, remove_target_dirs_in};

/// How many trailing lines of a failed gate a refusal carries.
///
/// A failing suite prints its summary last, so the tail is the part that says
/// what broke; the head is almost always build noise.
const GATE_TAIL_LINES: usize = 40;

/// Longest merge subject accepted from a task's first line.
///
/// A task is prose, and `git log --oneline` is read at a glance; a subject that
/// runs to a paragraph is truncated rather than dropped, so the message still
/// names the work.
const MAX_SUBJECT_BYTES: usize = 200;

/// One merge request, resolved against one scratch root.
pub struct MergeRequest<'a> {
    /// Worker whose branch is merged (`worker-<id>`).
    pub worker_id: &'a str,
    /// The worker's last verify verdict, when the caller has it: the live pool
    /// owns it, and a cross-process caller only has the on-disk row. `None`
    /// means "unknown", which never skips the gate.
    pub verified: Option<bool>,
    /// Keep the worker branch after merging (the CLI's `--no-delete`).
    pub keep_branch: bool,
}

/// What a successful merge did, in the words the CLI prints.
#[derive(Debug, Clone)]
pub struct MergeReport {
    pub worker_id: String,
    /// The branch that was merged (`worker-<id>`).
    pub branch: String,
    /// The branch it landed on.
    pub base_branch: String,
    /// Repository the merge happened in.
    pub repo_path: PathBuf,
    /// Abbreviated commit the merge produced.
    pub commit: String,
    /// Whether the gate ran; `false` when it was skipped as already verified.
    pub gate_ran: bool,
    /// The command the gate ran, when it ran one.
    pub gate_command: Option<String>,
    /// True when the worker branch was deleted.
    pub branch_deleted: bool,
    /// Human-readable list of what the cleanup reclaimed.
    pub cleaned: Vec<String>,
}

/// [`merge_worker_in`] over the default scratch root.
pub fn merge_worker(worker_id: &str) -> Result<MergeReport> {
    merge_worker_in(
        &ScratchRoot::from_env(),
        &MergeRequest {
            worker_id,
            verified: None,
            keep_branch: false,
        },
    )
}

/// Merge `worker_id`'s branch into its recorded base branch, or refuse.
///
/// Every refusal is an `Err` naming the reason and, where the operator can act
/// on it, the command that acts: a conflict points at `steer`, a dirty file at
/// the file itself. Nothing is written before the last check passes.
pub fn merge_worker_in(root: &ScratchRoot, req: &MergeRequest) -> Result<MergeReport> {
    let worker_id = req.worker_id;
    let resolved = resolve(root, worker_id)?;
    let repo = resolved.repo.as_path();

    if let Some(entry) = load_registry_entry_in(root, worker_id)
        && entry.status.is_live()
    {
        anyhow::bail!(
            "worker {worker_id} is still {}; wait for it to finish (or kill it) before merging",
            entry.status.display_name()
        );
    }
    if !branch_exists(repo, &resolved.branch)? {
        anyhow::bail!(
            "worker branch {} does not exist in {}; it may already have been merged or pruned",
            resolved.branch,
            repo.display()
        );
    }
    let checked_out = checked_out_branch(repo);
    if checked_out.as_deref() != Some(resolved.base_branch.as_str()) {
        anyhow::bail!(
            "{} has {} checked out, not the worker's base branch {}; check out {} first \
             (the merge never moves HEAD for you)",
            repo.display(),
            checked_out.unwrap_or_else(|| "a detached HEAD".to_string()),
            resolved.base_branch,
            resolved.base_branch
        );
    }

    // Trial merge: `merge-tree` computes the merge without touching a single
    // file, so a conflict is a refusal rather than a half-merged tree.
    let tree = match merge_tree(repo, &resolved.base_branch, &resolved.branch)? {
        MergeTree::Clean(tree) => tree,
        MergeTree::Conflicts(files) => {
            anyhow::bail!(
                "{} does not merge cleanly into {}; conflicting file(s): {}. \
                 Send the conflicts back to the worker: steer {} \"merge conflicts in {}\"",
                resolved.branch,
                resolved.base_branch,
                files.join(", "),
                worker_id,
                files.join(", ")
            );
        }
    };

    // Only the files the merge would write matter: an unrelated dirty or
    // untracked file elsewhere in the tree is the operator's business.
    let blocked = blocked_by_dirty_tree(repo, &resolved.base_branch, &resolved.branch)?;
    if !blocked.is_empty() {
        anyhow::bail!(
            "{} has uncommitted change(s) in file(s) this merge would touch: {}. \
             Commit or stash them first; untouched files are left alone",
            repo.display(),
            blocked.join(", ")
        );
    }

    let gate_command = resolved
        .verify
        .clone()
        .or_else(|| super::detect_verify_command(repo));
    let up_to_date = is_ancestor(repo, &resolved.base_branch, &resolved.branch)?;
    let already_verified = req.verified == Some(true);
    let gate_ran = !(up_to_date && (already_verified || gate_command.is_none()));
    if gate_ran {
        let command = gate_command.as_deref().ok_or_else(|| {
            anyhow::anyhow!(
                "worker {worker_id} needs a verify gate but none is recorded and none could be \
                 detected for {}; pass one with steer, or merge by hand",
                repo.display()
            )
        })?;
        run_gate(root, worker_id, repo, &tree, command)?;
    }

    let subject = merge_subject(&resolved.task, worker_id);
    git(
        repo,
        "merge --no-ff",
        &[
            "merge",
            "--no-ff",
            "--no-edit",
            "-m",
            &subject,
            &resolved.branch,
        ],
    )
    .with_context(|| {
        format!(
            "git merge --no-ff {} into {} failed",
            resolved.branch, resolved.base_branch
        )
    })?;
    let commit = git(repo, "rev-parse", &["rev-parse", "--short", "HEAD"])
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string());

    let (branch_deleted, cleaned) =
        cleanup(root, worker_id, repo, &resolved.branch, req.keep_branch);

    Ok(MergeReport {
        worker_id: worker_id.to_string(),
        branch: resolved.branch,
        base_branch: resolved.base_branch,
        repo_path: resolved.repo,
        commit,
        gate_ran,
        gate_command: if gate_ran { gate_command } else { None },
        branch_deleted,
        cleaned,
    })
}

/// Everything the merge needs about one worker, read from disk.
struct Resolved {
    repo: PathBuf,
    base_branch: String,
    branch: String,
    task: String,
    verify: Option<String>,
}

/// Resolve the worker's repository, base branch, branch, task and gate.
///
/// The saved conversation is the authority (it is what a revision relaunches
/// from); the registry row fills in a worker whose history was already reaped,
/// and the base branch falls back to what the repository itself reports.
fn resolve(root: &ScratchRoot, worker_id: &str) -> Result<Resolved> {
    let entry = load_registry_entry_in(root, worker_id);
    let history: Option<WorkerHistory> = load_worker_history_log_in(root, worker_id).ok();
    let repo = entry
        .as_ref()
        .and_then(|e| e.repo_path.clone())
        .or_else(|| history.as_ref().map(|h| h.repo_path.clone()))
        .map(PathBuf::from)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "Unknown worker {worker_id}: no registry row and no saved conversation name its \
                 repository"
            )
        })?;
    let base_branch = history
        .as_ref()
        .and_then(|h| h.base_branch.clone())
        .or_else(|| entry.as_ref().and_then(|e| e.base_branch.clone()))
        .or_else(|| super::revision::detect_base_branch(&repo))
        .ok_or_else(|| {
            anyhow::anyhow!(
                "worker {worker_id} records no base branch and none could be detected in {}; \
                 merge it by hand",
                repo.display()
            )
        })?;
    Ok(Resolved {
        repo,
        base_branch,
        branch: history
            .as_ref()
            .map(|h| h.branch.clone())
            .unwrap_or_else(|| format!("worker-{worker_id}")),
        task: history
            .as_ref()
            .map(|h| h.task.clone())
            .or_else(|| entry.as_ref().map(|e| e.task.clone()))
            .unwrap_or_default(),
        verify: history
            .as_ref()
            .and_then(|h| h.verify.clone())
            .filter(|v| !v.trim().is_empty()),
    })
}

/// The branch `repo` currently has checked out, or `None` when detached.
fn checked_out_branch(repo: &Path) -> Option<String> {
    let out = git(
        repo,
        "symbolic-ref",
        &["symbolic-ref", "--quiet", "--short", "HEAD"],
    )
    .ok()?;
    if !out.status.success() {
        return None;
    }
    let name = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if name.is_empty() { None } else { Some(name) }
}

/// Whether `branch` resolves in `repo`.
fn branch_exists(repo: &Path, branch: &str) -> Result<bool> {
    Ok(git(
        repo,
        "rev-parse",
        &[
            "rev-parse",
            "--verify",
            "--quiet",
            &format!("refs/heads/{branch}"),
        ],
    )?
    .status
    .success())
}

/// Whether `ancestor` is reachable from `descendant`.
fn is_ancestor(repo: &Path, ancestor: &str, descendant: &str) -> Result<bool> {
    Ok(git(
        repo,
        "merge-base --is-ancestor",
        &["merge-base", "--is-ancestor", ancestor, descendant],
    )?
    .status
    .success())
}

/// The outcome of a trial merge, computed without touching the working tree.
enum MergeTree {
    /// The tree the merge would produce.
    Clean(String),
    /// The files that conflict, in the order git reported them.
    Conflicts(Vec<String>),
}

/// Trial-merge `branch` into `base` with `git merge-tree --write-tree`.
///
/// Exit status 0 is a clean merge, 1 is a conflict list; anything else is a git
/// failure (unknown ref, unsupported option) and is reported verbatim rather
/// than guessed at.
fn merge_tree(repo: &Path, base: &str, branch: &str) -> Result<MergeTree> {
    let out = git(
        repo,
        "merge-tree",
        &["merge-tree", "--write-tree", "--name-only", base, branch],
    )?;
    let stdout = String::from_utf8_lossy(&out.stdout);
    let mut lines = stdout.lines();
    // Line one is always the tree oid; the conflict list follows it, then a
    // blank line and git's informational messages.
    let tree = lines
        .next()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "git merge-tree produced no merge tree for {branch} into {base}: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            )
        })?
        .to_string();
    if out.status.success() {
        return Ok(MergeTree::Clean(tree));
    }
    if out.status.code() == Some(1) {
        let files: Vec<String> = lines
            .take_while(|line| !line.trim().is_empty())
            .map(|line| line.trim().to_string())
            .filter(|line| !line.is_empty())
            .collect();
        return Ok(MergeTree::Conflicts(files));
    }
    anyhow::bail!(
        "git merge-tree {base} {branch} failed ({}): {}",
        out.status
            .code()
            .map(|c| c.to_string())
            .unwrap_or_else(|| "signal".to_string()),
        String::from_utf8_lossy(&out.stderr).trim()
    );
}

/// The files a merge of `branch` into `base` would write: everything either
/// side changed since their merge base.
fn touched_files(repo: &Path, base: &str, branch: &str) -> Result<Vec<String>> {
    let Some(merge_base) = git(repo, "merge-base", &["merge-base", base, branch])
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
    else {
        // No common history: every file either side has is in play.
        return Ok(Vec::new());
    };
    let mut files: Vec<String> = Vec::new();
    for tip in [base, branch] {
        let out = git(
            repo,
            "diff --name-only",
            &["diff", "--name-only", "-z", &merge_base, tip],
        )?;
        for path in String::from_utf8_lossy(&out.stdout).split('\0') {
            if !path.is_empty() && !files.iter().any(|p| p == path) {
                files.push(path.to_string());
            }
        }
    }
    Ok(files)
}

/// Every path the repository's working tree has changed: modified, staged or
/// untracked. `-z` keeps the paths unquoted, so a name with a space or a quote
/// still compares equal to the merge's own list.
fn dirty_paths(repo: &Path) -> Result<Vec<String>> {
    let mut paths: Vec<String> = Vec::new();
    for args in [
        vec!["diff", "--name-only", "-z"],
        vec!["diff", "--cached", "--name-only", "-z"],
        vec!["ls-files", "--others", "--exclude-standard", "-z"],
    ] {
        let out = git(repo, "status", &args)?;
        for path in String::from_utf8_lossy(&out.stdout).split('\0') {
            if !path.is_empty() && !paths.iter().any(|p| p == path) {
                paths.push(path.to_string());
            }
        }
    }
    Ok(paths)
}

/// The touched files that are also dirty, i.e. the ones that refuse the merge.
fn blocked_by_dirty_tree(repo: &Path, base: &str, branch: &str) -> Result<Vec<String>> {
    let touched = touched_files(repo, base, branch)?;
    if touched.is_empty() {
        return Ok(Vec::new());
    }
    let dirty = dirty_paths(repo)?;
    Ok(touched
        .into_iter()
        .filter(|path| dirty.iter().any(|d| d == path))
        .collect())
}

/// Replay `command` on the merge result in a throwaway worktree.
///
/// `git worktree add` refuses a tree object, so the worktree is created
/// detached at the base tip and the merge tree is read into it; the result is
/// byte-for-byte what the merge would produce. The worktree lives under the
/// scratch root, never inside the operator's checkout, and is removed whatever
/// the command does.
fn run_gate(
    root: &ScratchRoot,
    worker_id: &str,
    repo: &Path,
    tree: &str,
    command: &str,
) -> Result<()> {
    let gate_dir = root.join(format!("swe-merge-{worker_id}"));
    force_remove_dir(&gate_dir);
    let created = git(
        repo,
        "worktree add",
        &[
            "worktree",
            "add",
            "--detach",
            "--no-checkout",
            &gate_dir.to_string_lossy(),
            &base_tip_ref(repo)?,
        ],
    );
    if let Err(e) = created {
        force_remove_dir(&gate_dir);
        return Err(e).with_context(|| {
            format!(
                "could not create a gate worktree for worker {worker_id} under the scratch root"
            )
        });
    }
    let materialised = git(&gate_dir, "read-tree", &["read-tree", tree])
        .and_then(|_| git(&gate_dir, "checkout-index", &["checkout-index", "-a", "-f"]));
    let gate = match materialised {
        Ok(_) => run_gate_command(&gate_dir, command),
        Err(e) => Err(e).with_context(|| {
            format!("could not materialise the merge result for worker {worker_id}")
        }),
    };
    reclaim_gate_worktree(repo, &gate_dir);
    let (code, text) = gate?;
    if code == Some(0) {
        return Ok(());
    }
    anyhow::bail!(
        "verify gate failed on the merge result of worker {worker_id} ({command}, exit {}):\n{}",
        code.map(|c| c.to_string())
            .unwrap_or_else(|| "signal".to_string()),
        tail(&text, GATE_TAIL_LINES)
    );
}

/// Run `command` in `dir`, returning its exit status and its combined output.
fn run_gate_command(dir: &Path, command: &str) -> Result<(Option<i32>, String)> {
    let output = Command::new("sh")
        .arg("-c")
        .arg(command)
        .current_dir(dir)
        .output()
        .with_context(|| format!("could not run the verify gate: {command}"))?;
    let mut text = String::from_utf8_lossy(&output.stdout).into_owned();
    if !output.stderr.is_empty() {
        if !text.is_empty() {
            text.push('\n');
        }
        text.push_str(&String::from_utf8_lossy(&output.stderr));
    }
    Ok((output.status.code(), text))
}

/// The ref the gate worktree is created at: the base branch tip.
///
/// The worktree only needs *a* commit to exist; the merge tree is read over it
/// immediately afterwards.
fn base_tip_ref(repo: &Path) -> Result<String> {
    let out = git(repo, "symbolic-ref", &["symbolic-ref", "--short", "HEAD"])?;
    let name = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if name.is_empty() {
        anyhow::bail!(
            "cannot create a gate worktree: {} has a detached HEAD",
            repo.display()
        );
    }
    Ok(name)
}

/// Remove a gate worktree and its registration, whatever the gate did.
fn reclaim_gate_worktree(repo: &Path, gate_dir: &Path) {
    let _ = git(
        repo,
        "worktree remove",
        &["worktree", "remove", "--force", &gate_dir.to_string_lossy()],
    );
    force_remove_dir(gate_dir);
    let _ = git(repo, "worktree prune", &["worktree", "prune"]);
}

/// The last `lines` lines of `text`, with a note when anything was dropped.
fn tail(text: &str, lines: usize) -> String {
    let all: Vec<&str> = text.lines().collect();
    if all.len() <= lines {
        return text.trim_end().to_string();
    }
    let kept = all[all.len() - lines..].join("\n");
    format!(
        "... ({} earlier line(s) omitted)\n{kept}",
        all.len() - lines
    )
}

/// The merge subject: the task's first line, credited to the worker.
fn merge_subject(task: &str, worker_id: &str) -> String {
    let first = task.lines().next().unwrap_or("").trim();
    let subject = if first.is_empty() {
        format!("worker {worker_id}")
    } else if first.len() > MAX_SUBJECT_BYTES {
        format!(
            "{}...",
            &first[..first.floor_char_boundary(MAX_SUBJECT_BYTES)]
        )
    } else {
        first.to_string()
    };
    format!("{subject} (worker {worker_id})")
}

/// Post-merge cleanup: the worktree leftovers, the branch and the history file.
///
/// Reuses the same helpers the prune sweep uses, so a merged worker leaves
/// exactly what a pruned one does. The registry row survives: `status` and
/// `collect` still answer about the run, and `reap` expires it on its own TTL.
fn cleanup(
    root: &ScratchRoot,
    worker_id: &str,
    repo: &Path,
    branch: &str,
    keep_branch: bool,
) -> (bool, Vec<String>) {
    // The worktree goes first: a leftover that is still registered would make
    // the branch undeletable, and `git worktree prune` clears the registration
    // once its directory is gone.
    let worktree = root.join(format!("swe-wt-{worker_id}"));
    let reclaimed = worktree.exists();
    if reclaimed {
        force_remove_dir(&worktree);
        remove_target_dirs_in(root, &worktree);
        let _ = git(repo, "worktree prune", &["worktree", "prune"]);
    }
    let branch_deleted = !keep_branch
        && git(repo, "branch -D", &["branch", "-D", branch]).is_ok_and(|o| o.status.success());
    remove_worker_history_in(root, worker_id);

    let mut cleaned = Vec::new();
    if branch_deleted {
        cleaned.push(format!("branch {branch} deleted"));
    }
    cleaned.push("history file removed".to_string());
    if reclaimed {
        cleaned.push("worktree leftovers removed".to_string());
    }
    (branch_deleted, cleaned)
}
