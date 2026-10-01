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

use anyhow::{Context, Result};

use super::admission::{AdmissionClass, AdmissionController};
use super::registry::{RegistryStatus, load_all_registry_entries_in, load_registry_entry_in};
use super::revision::{
    RetireContext, WorkerHistory, load_worker_history_log_in, retire_worker_with,
};
use crate::agent::AgentRunner;
use crate::worktree::{ScratchRoot, force_remove_dir, git, remove_target_dirs_in};

/// How many trailing lines of a failed gate a refusal carries.
///
/// A failing suite prints its summary last, so the tail is the part that says
/// what broke; the head is almost always build noise.
const GATE_TAIL_LINES: usize = 40;

/// Longest task excerpt a merge subject keeps.
///
/// A task is prose, and `git log --oneline` is read at a glance; the subject
/// keeps at most this many characters of the task's first line, so a
/// consolidator's paragraph-length task cannot become a paragraph-length
/// subject.
const MAX_SUBJECT_TASK_CHARS: usize = 72;

/// Marks the task excerpt of a clamped merge subject as shortened.
const SUBJECT_ELLIPSIS: &str = "...";

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
    /// The pool's admission controller, so the gate's build competes for the
    /// host's heavy-command budget like any worker's. `None` for a caller with
    /// no pool (tests, one-shot tools): the gate still runs confined, only the
    /// host budget is not reserved.
    pub admission: Option<AdmissionController>,
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
            admission: None,
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
        run_gate(
            root,
            worker_id,
            repo,
            &tree,
            command,
            req,
            &resolved.client_env,
        )?;
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
    let commit = head_commit(repo);

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

/// Abbreviated commit `repo` currently has checked out.
fn head_commit(repo: &Path) -> String {
    git(repo, "rev-parse", &["rev-parse", "--short", "HEAD"])
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string())
}

/// Everything the merge needs about one worker, read from disk.
struct Resolved {
    repo: PathBuf,
    base_branch: String,
    branch: String,
    task: String,
    verify: Option<String>,
    /// The dispatcher's filtered ambient environment, replayed into the gate so
    /// it sees what the worker's own verify run saw.
    client_env: Vec<(String, String)>,
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
        client_env: history
            .as_ref()
            .map(|h| h.client_env.clone())
            .unwrap_or_default(),
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
fn merge_tree(repo: &Path, base: &str, branch: &str) -> Result<MergeTree> {
    merge_tree_with(repo, &[], base, branch)
}

/// Trial-merge `theirs` into `ours` with `git merge-tree --write-tree`.
///
/// `options` are the extra flags the caller needs -- a batch composes a branch
/// on top of an already merged tree, so it names the merge base explicitly.
///
/// Exit status 0 is a clean merge, 1 is a conflict list; anything else is a git
/// failure (unknown ref, unsupported option) and is reported verbatim rather
/// than guessed at.
fn merge_tree_with(repo: &Path, options: &[&str], ours: &str, theirs: &str) -> Result<MergeTree> {
    let mut args: Vec<&str> = vec!["merge-tree", "--write-tree", "--name-only"];
    args.extend_from_slice(options);
    args.push(ours);
    args.push(theirs);
    let out = git(repo, "merge-tree", &args)?;
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
                "git merge-tree produced no merge tree for {theirs} into {ours}: {}",
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
        "git merge-tree {ours} {theirs} failed ({}): {}",
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
///
/// The command itself is model-written code, so it goes through the very
/// executor a worker's bash step uses: kernel confinement (Landlock + seccomp,
/// or bubblewrap), a cleared and allow-listed environment, the heavy-command
/// timeout and a build directory leased from the repository's pool rather than
/// a fresh `target/` inside the throwaway worktree. A merge is a rare,
/// one-command operation, so the gate drives its own current-thread runtime
/// instead of forcing every caller of this module to be async.
fn run_gate(
    root: &ScratchRoot,
    worker_id: &str,
    repo: &Path,
    tree: &str,
    command: &str,
    req: &MergeRequest<'_>,
    client_env: &[(String, String)],
) -> Result<()> {
    let (code, text) = run_gate_result(root, worker_id, repo, tree, command, req, client_env)?;
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

/// [`run_gate`] without the verdict: the exit code and the bounded output, so a
/// batch can gate its combined tree once and still report the failing tail.
fn run_gate_result(
    root: &ScratchRoot,
    label: &str,
    repo: &Path,
    tree: &str,
    command: &str,
    req: &MergeRequest<'_>,
    client_env: &[(String, String)],
) -> Result<(Option<i32>, String)> {
    let gate_dir = root.join(format!("swe-merge-{label}"));
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
            format!("could not create a gate worktree for {label} under the scratch root")
        });
    }
    let materialised = git(&gate_dir, "read-tree", &["read-tree", tree])
        .and_then(|_| git(&gate_dir, "checkout-index", &["checkout-index", "-a", "-f"]));
    let gate = match materialised {
        Ok(_) => {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .context("could not start the verify gate's runtime");
            runtime.and_then(|runtime| {
                runtime.block_on(run_gate_confined(repo, &gate_dir, command, req, client_env))
            })
        }
        Err(e) => {
            Err(e).with_context(|| format!("could not materialise the merge result for {label}"))
        }
    };
    reclaim_gate_worktree(root, repo, &gate_dir);
    gate
}

/// Run the gate command confined, admitted and with a leased build directory.
///
/// The executor's own truncation keeps the head *and* the tail of the output,
/// which is what a failing suite needs: the panic and the summary are at the
/// end, so nothing here re-buffers the streams.
async fn run_gate_confined(
    repo: &Path,
    gate_dir: &Path,
    command: &str,
    req: &MergeRequest<'_>,
    client_env: &[(String, String)],
) -> Result<(Option<i32>, String)> {
    // Fail closed: the gate replays model-written code, so a host that can
    // confine nothing at all refuses the merge instead of running it as the
    // user. A worker's own step may degrade with a warning; a merge is the one
    // place that must not.
    if crate::agent::exec::select_backend() == crate::agent::exec::SandboxBackend::Unconfined {
        anyhow::bail!(
            "no sandbox is available on this host (neither Landlock/seccomp nor bubblewrap), \
             so the verify gate cannot run confined; refusing to merge worker's branch"
        );
    }

    // A build directory leased from the repository's pool: warm for the next
    // build, and never a fresh multi-gigabyte `target/` inside a worktree that
    // is about to be deleted.
    let repo_owned = repo.to_path_buf();
    let lease =
        tokio::task::spawn_blocking(move || crate::cache::BuildDirLease::acquire(&repo_owned))
            .await
            .context("the build-directory lease task for the verify gate failed")??;
    let build_dir = lease.dir().to_path_buf();

    // The heavy slot is held for the whole gate run and released on every exit
    // path, including a refusal, because the permit is a guard.
    let _permit = match &req.admission {
        // The gate is a completion verification, so it queues in that class
        // and is preferred over a worker still exploring.
        Some(controller) => Some(controller.acquire(AdmissionClass::Completion).await),
        None => None,
    };

    // The bash path never dials the API, so the transport fields are unused;
    // only the confinement, environment and build directory matter here.
    let mut runner = AgentRunner::new(String::new(), String::new(), String::new(), None)
        .with_extra_env(client_env.to_vec())
        // The gate is run by the harness, not the model, so it must never
        // become a background job; its budget is the absolute job ceiling, so a
        // slow gate takes longer instead of failing.
        .without_job_conversion()
        .with_command_timeout(crate::agent::jobs::job_max_secs());
    runner.build_target_dir = Some(build_dir);
    let (text, code) = runner.execute_bash(gate_dir, command).await?;
    Ok((code, text))
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
///
/// The executor also opens a private scratch directory beside the worktree
/// (`swe-tmp-<name>`), so the same helper the prune sweep uses reclaims both.
/// The repository's leased build directory is named after the repository, not
/// after this worktree, so it survives for the next build to reuse.
fn reclaim_gate_worktree(root: &ScratchRoot, repo: &Path, gate_dir: &Path) {
    let _ = git(
        repo,
        "worktree remove",
        &["worktree", "remove", "--force", &gate_dir.to_string_lossy()],
    );
    force_remove_dir(gate_dir);
    remove_target_dirs_in(root, gate_dir);
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

/// The merge subject: `Merge worker-<id>: <task first line clamped>`.
///
/// The worker is named first so `git log --oneline` reads as a stack of merges,
/// and the task's first line is clamped to [`MAX_SUBJECT_TASK_CHARS`] so a
/// paragraph-length task cannot produce a paragraph-length subject.
fn merge_subject(task: &str, worker_id: &str) -> String {
    let first = task.lines().next().unwrap_or("").trim();
    if first.is_empty() {
        return format!("Merge worker-{worker_id}");
    }
    let clamped = if first.chars().count() > MAX_SUBJECT_TASK_CHARS {
        let keep = MAX_SUBJECT_TASK_CHARS - SUBJECT_ELLIPSIS.len();
        let kept: String = first.chars().take(keep).collect();
        format!("{}{SUBJECT_ELLIPSIS}", kept.trim_end())
    } else {
        first.to_string()
    };
    format!("Merge worker-{worker_id}: {clamped}")
}

/// Post-merge retirement: the merged worker leaves nothing behind at all.
///
/// A merged worker is fully integrated -- its commits are in the base branch --
/// so nothing of it is needed any more. The merge therefore calls the one
/// shared [`retire_worker_with`] rather than picking which leftovers to drop:
/// branch, registry row, history, steering mailbox and steer-source, watch
/// acknowledgements, and the worktree, scratch and build directories it held go
/// together, so `list` shows only live and awaiting-integration workers without
/// any hiding logic.
///
/// `--no-delete` is the one exception: the operator asked to keep the branch, so
/// the branch stays and only the scratch traces are reclaimed. A sweep does not
/// see a branch that outlived its merge and therefore leaves that worker alone
/// until the branch itself goes.
///
/// Returns whether the branch was deleted and what was reclaimed, in the words
/// the CLI prints.
fn cleanup(
    root: &ScratchRoot,
    worker_id: &str,
    repo: &Path,
    branch: &str,
    keep_branch: bool,
) -> (bool, Vec<String>) {
    let worktree = root.join(format!("swe-wt-{worker_id}"));
    let reclaimed = worktree.exists();
    let branch_deleted = !keep_branch
        && git(repo, "branch -D", &["branch", "-D", branch]).is_ok_and(|o| o.status.success());
    retire_worker_with(
        root,
        worker_id,
        &RetireContext {
            repo: Some(repo),
            ack_dir: None,
        },
    );

    let mut cleaned = Vec::new();
    if branch_deleted {
        cleaned.push(format!("branch {branch} deleted"));
    } else if keep_branch {
        cleaned.push(format!("branch {branch} kept (--no-delete)"));
    }
    cleaned.push("worker retired (row, history, mailbox, scratch)".to_string());
    if reclaimed {
        cleaned.push("worktree leftovers removed".to_string());
    }
    (branch_deleted, cleaned)
}

// ----------
// Batch merge: one gate for a whole round of approved workers
// ----------

/// One batch merge request, resolved against one scratch root.
pub struct MergeApprovedRequest<'a> {
    /// The agent whose approved workers may be merged (H-3). `None` is the
    /// admin override: every owner's approved workers.
    pub owner: Option<&'a str>,
    /// Only the approved workers of this group.
    pub group: Option<&'a str>,
    /// The pool's admission controller, so the batch's single gate competes for
    /// the host's heavy-command budget like any worker's. `None` for a caller
    /// with no pool: the gate still runs confined, only the budget is not
    /// reserved.
    pub admission: Option<AdmissionController>,
}

/// One approved worker whose branch landed.
#[derive(Debug, Clone)]
pub struct MergedWorker {
    pub worker_id: String,
    /// Abbreviated commit that worker's merge produced.
    pub commit: String,
}

/// One approved worker skipped because its branch does not compose.
#[derive(Debug, Clone)]
pub struct SkippedWorker {
    pub worker_id: String,
    /// The files that conflict, in the order git reported them.
    pub files: Vec<String>,
    /// The steer that sends those conflicts back to the worker that owns them.
    pub steer: String,
}

/// What a batch merge did, in the words the CLI prints.
#[derive(Debug, Clone)]
pub struct MergeApprovedReport {
    /// The workers whose branch landed, in merge order.
    pub merged: Vec<MergedWorker>,
    /// The workers skipped because their branch conflicts.
    pub skipped: Vec<SkippedWorker>,
    /// The one command the batch gated on.
    pub gate_command: Option<String>,
    /// How long that gate took, in milliseconds.
    pub gate_duration_ms: u128,
    /// The branch every worker landed on.
    pub base_branch: String,
    /// Repository the merges happened in.
    pub repo_path: PathBuf,
    /// Human-readable list of what the per-worker cleanup reclaimed.
    pub cleaned: Vec<String>,
}

/// [`merge_approved_in`] over the default scratch root.
pub fn merge_approved(req: &MergeApprovedRequest<'_>) -> Result<MergeApprovedReport> {
    merge_approved_in(&ScratchRoot::from_env(), req)
}

/// Land every approved worker of a round with one gate, or refuse.
///
/// The orchestrator gates a round, not a worker: this selects the caller's
/// completed workers that carry an approval (optionally one group), composes
/// their branches into a single tree, runs the shared verify gate on that tree
/// once and -- only if it passes -- merges each branch with `--no-ff` and
/// cleans up after it. A branch that conflicts is skipped and reported rather
/// than failing the round; a failing gate merges nothing at all.
///
/// Every refusal is taken before the first merge, so a refused batch changes
/// neither the repository nor any worker's files.
pub fn merge_approved_in(
    root: &ScratchRoot,
    req: &MergeApprovedRequest<'_>,
) -> Result<MergeApprovedReport> {
    let ids = approved_workers(root, req)?;
    if ids.is_empty() {
        anyhow::bail!(
            "no completed worker of yours carries an approval{}; approve the ones to land first",
            group_clause(req.group)
        );
    }
    let mut resolved: Vec<(String, Resolved)> = Vec::new();
    for id in &ids {
        resolved.push((id.clone(), resolve(root, id)?));
    }
    let repo = resolved[0].1.repo.clone();
    let base_branch = resolved[0].1.base_branch.clone();
    for (id, other) in &resolved[1..] {
        if other.repo != repo || other.base_branch != base_branch {
            anyhow::bail!(
                "worker {id} records base branch {} in {}, not {} in {}; merge each repository's \
                 approved workers in its own batch",
                other.base_branch,
                other.repo.display(),
                base_branch,
                repo.display()
            );
        }
    }
    let repo = repo.as_path();

    // The checkout refusals `merge <id>` takes, before anything is written.
    let checked_out = checked_out_branch(repo);
    if checked_out.as_deref() != Some(base_branch.as_str()) {
        anyhow::bail!(
            "{} has {} checked out, not the workers' base branch {}; check out {} first \
             (the merge never moves HEAD for you)",
            repo.display(),
            checked_out.unwrap_or_else(|| "a detached HEAD".to_string()),
            base_branch,
            base_branch
        );
    }
    let mut touched: Vec<String> = Vec::new();
    for (_, worker) in &resolved {
        for path in touched_files(repo, &base_branch, &worker.branch)? {
            if !touched.contains(&path) {
                touched.push(path);
            }
        }
    }
    let dirty = dirty_paths(repo)?;
    let blocked: Vec<String> = touched
        .iter()
        .filter(|path| dirty.contains(path))
        .cloned()
        .collect();
    if !blocked.is_empty() {
        anyhow::bail!(
            "{} has uncommitted change(s) in file(s) this merge would touch: {}. \
             Commit or stash them first; untouched files are left alone",
            repo.display(),
            blocked.join(", ")
        );
    }

    // Compose the round: each branch on top of the previous result, so the gate
    // sees the batch as the one tree it would produce.
    let base_tree = tree_of(repo, &base_branch);
    let mut composed: Option<String> = None;
    let mut included: Vec<&(String, Resolved)> = Vec::new();
    let mut skipped: Vec<SkippedWorker> = Vec::new();
    for worker in &resolved {
        let branch = worker.1.branch.as_str();
        let trial = match &composed {
            // The first branch is merged exactly as `merge <id>` merges it.
            None => merge_tree(repo, &base_branch, branch)?,
            // The rest are merged onto the composed tree, with the point the
            // branch forked from the base as the merge base: the same three-way
            // merge a sequential `git merge` would compute.
            Some(tree) => {
                // The point the branch forked from the base is the merge base;
                // with no common history at all git decides on its own.
                match merge_base_of(repo, &base_branch, branch).or_else(|| base_tree.clone()) {
                    Some(forked) => {
                        merge_tree_with(repo, &["--merge-base", &forked], tree, branch)?
                    }
                    None => merge_tree_with(repo, &[], tree, branch)?,
                }
            }
        };
        match trial {
            MergeTree::Clean(tree) => {
                composed = Some(tree);
                included.push(worker);
            }
            MergeTree::Conflicts(files) => skipped.push(SkippedWorker {
                worker_id: worker.0.clone(),
                steer: steer_hint(&worker.0, &files),
                files,
            }),
        }
    }
    if included.is_empty() {
        let hints: Vec<String> = skipped.iter().map(|s| s.steer.clone()).collect();
        anyhow::bail!(
            "no approved worker's branch composes into {}: {}",
            base_branch,
            hints.join("; ")
        );
    }

    // One gate for the whole batch, on the combined tree.
    let gate_command = shared_gate_command(repo, &included)?;
    let gate_req = MergeRequest {
        worker_id: "approved",
        verified: None,
        keep_branch: false,
        admission: req.admission.clone(),
    };
    // The gate replays the dispatcher's filtered environment; a round's workers
    // share it, so the first included worker's copy is the batch's.
    let client_env = included[0].1.client_env.clone();
    // `included` is not empty here, so the composition produced a tree.
    let composed = composed.as_deref().unwrap_or_default();
    let started = std::time::Instant::now();
    let (code, text) = run_gate_result(
        root,
        "approved",
        repo,
        composed,
        &gate_command,
        &gate_req,
        &client_env,
    )?;
    let gate_duration_ms = started.elapsed().as_millis();
    if code != Some(0) {
        let landed: Vec<&str> = included.iter().map(|(id, _)| id.as_str()).collect();
        anyhow::bail!(
            "verify gate failed on the combined merge result of {} ({gate_command}, exit {}):\n{}\n{}",
            landed.join(", "),
            code.map(|c| c.to_string())
                .unwrap_or_else(|| "signal".to_string()),
            tail(&text, GATE_TAIL_LINES),
            attribute_failures(&text, repo, &base_branch, &included),
        );
    }

    let mut merged: Vec<MergedWorker> = Vec::new();
    let mut cleaned: Vec<String> = Vec::new();
    for (id, worker) in &included {
        let subject = merge_subject(&worker.task, id);
        git(
            repo,
            "merge --no-ff",
            &[
                "merge",
                "--no-ff",
                "--no-edit",
                "-m",
                &subject,
                &worker.branch,
            ],
        )
        .with_context(|| {
            format!(
                "git merge --no-ff {} into {} failed",
                worker.branch, base_branch
            )
        })?;
        let commit = head_commit(repo);
        let (_, worker_cleaned) = cleanup(root, id, repo, &worker.branch, false);
        cleaned.extend(worker_cleaned);
        merged.push(MergedWorker {
            worker_id: id.clone(),
            commit,
        });
    }

    Ok(MergeApprovedReport {
        merged,
        skipped,
        gate_command: Some(gate_command),
        gate_duration_ms,
        base_branch,
        repo_path: repo.to_path_buf(),
        cleaned,
    })
}

/// The caller's completed, approved workers, in approval order.
///
/// The registry is the only cross-process view of the pool, so the selection
/// reads it: a worker this process still owns and one another process merged
/// into its row are both seen here. Ties on the approval stamp fall back to the
/// id, so the order is deterministic.
fn approved_workers(root: &ScratchRoot, req: &MergeApprovedRequest<'_>) -> Result<Vec<String>> {
    let mut rows: Vec<(u64, String)> = load_all_registry_entries_in(root)
        .into_iter()
        .filter(|entry| entry.status == RegistryStatus::Completed)
        .filter(|entry| entry.approved.is_some())
        .filter(|entry| {
            req.owner
                .is_none_or(|owner| entry.owner.as_deref() == Some(owner))
        })
        .filter(|entry| {
            req.group
                .is_none_or(|group| entry.group.as_deref() == Some(group))
        })
        .map(|entry| {
            (
                entry.approved.as_ref().map(|a| a.at).unwrap_or_default(),
                entry.id,
            )
        })
        .collect();
    rows.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    Ok(rows.into_iter().map(|(_, id)| id).collect())
}

/// `group` as a clause of a refusal, so the message names what was looked for.
fn group_clause(group: Option<&str>) -> String {
    match group {
        Some(group) => format!(" in group {group}"),
        None => String::new(),
    }
}

/// The tree `branch` points at, when it resolves.
fn tree_of(repo: &Path, branch: &str) -> Option<String> {
    git(
        repo,
        "rev-parse",
        &["rev-parse", &format!("{branch}^{{tree}}")],
    )
    .ok()
    .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
    .filter(|s| !s.is_empty())
}

/// The commit `branch` forked from `base` at, when they share history.
fn merge_base_of(repo: &Path, base: &str, branch: &str) -> Option<String> {
    git(repo, "merge-base", &["merge-base", base, branch])
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
}

/// The one command the batch gates on.
///
/// The verify command the workers recorded, when they recorded the same one;
/// otherwise the auto-detected project gate. A round shares one acceptance
/// criterion, so a batch with neither is refused rather than landed ungated.
fn shared_gate_command(repo: &Path, included: &[&(String, Resolved)]) -> Result<String> {
    let mut recorded: Vec<&str> = Vec::new();
    for (_, worker) in included {
        if let Some(command) = worker.verify.as_deref()
            && !recorded.contains(&command)
        {
            recorded.push(command);
        }
    }
    recorded
        .first()
        .filter(|_| {
            recorded.len() == 1 && included.iter().all(|(_, worker)| worker.verify.is_some())
        })
        .map(|command| (*command).to_string())
        .or_else(|| super::detect_verify_command(repo))
        .ok_or_else(|| {
            anyhow::anyhow!(
                "the approved workers record no verify command they agree on and none could be \
                 detected for {}; pass one with steer, or merge by hand",
                repo.display()
            )
        })
}

/// The steer that sends a skipped worker's conflicts back to it.
fn steer_hint(worker_id: &str, files: &[String]) -> String {
    format!(
        "steer {worker_id} \"merge conflicts in {}\"",
        files.join(", ")
    )
}

/// Attribute the files a failing gate named to the workers whose branch touched
/// them.
///
/// A gate names a file as `path:line`; the worker whose branch touched it is the
/// one to steer. A file several workers touched is an interaction point: no
/// single branch is wrong, so the orchestrator has to decide between them.
fn attribute_failures(
    text: &str,
    repo: &Path,
    base_branch: &str,
    included: &[&(String, Resolved)],
) -> String {
    let mut owners: Vec<(String, Vec<String>)> = Vec::new();
    for (id, worker) in included {
        let Some(forked) = merge_base_of(repo, base_branch, &worker.branch) else {
            continue;
        };
        let Ok(diff) = git(
            repo,
            "diff --name-only",
            &["diff", "--name-only", "-z", &forked, &worker.branch],
        ) else {
            continue;
        };
        for path in String::from_utf8_lossy(&diff.stdout)
            .split(' ')
            .filter(|p| !p.is_empty())
            .map(str::to_string)
        {
            match owners.iter_mut().find(|(known, _)| *known == path) {
                Some((_, ids)) => {
                    if !ids.contains(id) {
                        ids.push(id.clone());
                    }
                }
                None => owners.push((path, vec![id.clone()])),
            }
        }
    }
    let known: Vec<String> = owners.iter().map(|(path, _)| path.clone()).collect();
    let named = files_named_in(text, &known);
    if named.is_empty() {
        return "no file the failure names was touched by a worker in this batch".to_string();
    }
    let mut lines =
        vec!["the failing files and the workers whose branch touched them:".to_string()];
    for path in named {
        let ids = &owners
            .iter()
            .find(|(known, _)| *known == path)
            .expect("a named file is one the owners map holds")
            .1;
        if ids.len() > 1 {
            lines.push(format!(
                "  {path}: {} (interaction point: several workers touched it)",
                ids.join(", ")
            ));
        } else {
            lines.push(format!("  {path}: {}", ids[0]));
        }
    }
    lines.join("\n")
}

/// The paths `text` names, restricted to `known`.
///
/// A gate names a file as `path:line`; a bare token that is exactly a file the
/// batch touches counts too, so a message that prints only the path still
/// attributes. Anything else in the output is prose and is ignored, which is
/// what keeps the attribution to files a worker really changed.
fn files_named_in(text: &str, known: &[String]) -> Vec<String> {
    let mut named: Vec<String> = Vec::new();
    for token in text.split_whitespace() {
        let token = token.trim_matches(|c: char| {
            matches!(
                c,
                '(' | ')'
                    | '['
                    | ']'
                    | '{'
                    | '}'
                    | '<'
                    | '>'
                    | '\''
                    | '"'
                    | '`'
                    | ','
                    | ';'
                    | '*'
                    | '='
                    | '|'
            )
        });
        let token = token.trim_start_matches(['-', '>']);
        let candidate = path_of(token);
        if known.iter().any(|path| path == candidate) && !named.iter().any(|path| path == candidate)
        {
            named.push(candidate.to_string());
        }
    }
    named
}

/// The path part of a `path:line` token, or the token itself when it names no
/// line.
fn path_of(token: &str) -> &str {
    let Some((head, rest)) = token.split_once(':') else {
        return token;
    };
    let digits = rest.chars().take_while(|c| c.is_ascii_digit()).count();
    // `path:12` and `path:12:5` are a location; `note:text` is not.
    if digits > 0 && (rest.len() == digits || rest.as_bytes()[digits] == b':') {
        head
    } else {
        token
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The subject names the worker first and keeps the whole first task line
    /// when it fits, so an ordinary merge is still readable at a glance.
    #[test]
    fn test_merge_subject_names_the_worker_and_keeps_a_short_task() {
        assert_eq!(
            merge_subject("do the w1 work\nignored detail", "w1"),
            "Merge worker-w1: do the w1 work"
        );
        assert_eq!(
            merge_subject("", "w9"),
            "Merge worker-w9",
            "an empty task still names the worker"
        );
    }

    /// A paragraph-length task is clamped to `MAX_SUBJECT_TASK_CHARS` so the
    /// merge subject stays one readable line.
    #[test]
    fn test_merge_subject_clamps_a_long_task_to_the_limit() {
        let at_limit = "y".repeat(MAX_SUBJECT_TASK_CHARS);
        assert_eq!(
            merge_subject(&at_limit, "w2"),
            format!("Merge worker-w2: {at_limit}"),
            "a task exactly at the limit is kept whole"
        );

        let over = "x".repeat(MAX_SUBJECT_TASK_CHARS + 20);
        let subject = merge_subject(&over, "w3");
        let task = subject
            .strip_prefix("Merge worker-w3: ")
            .expect("the subject must name the worker");
        assert_eq!(task.chars().count(), MAX_SUBJECT_TASK_CHARS, "{subject}");
        assert!(task.ends_with(SUBJECT_ELLIPSIS), "{subject}");
    }

    /// Clamping never splits a multi-byte character.
    #[test]
    fn test_merge_subject_clamps_on_a_character_boundary() {
        let over = "\u{e9}".repeat(MAX_SUBJECT_TASK_CHARS + 5);
        let subject = merge_subject(&over, "w4");
        let task = subject
            .strip_prefix("Merge worker-w4: ")
            .expect("the subject must name the worker");
        assert_eq!(task.chars().count(), MAX_SUBJECT_TASK_CHARS, "{subject}");
        assert!(task.ends_with(SUBJECT_ELLIPSIS), "{subject}");
    }
}
