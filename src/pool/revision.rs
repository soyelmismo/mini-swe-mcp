//! Revision persistence: the finished worker's conversation and relaunch facts.
//!
//! The orchestrator reviews a finished worker's branch and then steers it with
//! corrections. The model must continue on its own branch with its full
//! context rather than restart from the task, so [`WorkerPool::run_worker`]
//! and its auto-checkpoints serialize the live conversation (system prompt,
//! task, every assistant turn
//! with its reasoning, every tool result -- exactly what the next request
//! replays) plus the metadata the launch took apart again, into
//! `swe_base_dir()/swe-wt-<id>.history.json`. [`WorkerPool::steer`] reloads the
//! file, appends the revision request, and re-launches the same loop on the
//! same branch. History without its metadata is a dead letter, so one atomic
//! file carries both.
//!
//! The canonical store is the append-only log `swe-wt-<id>.history.jsonl`:
//! line one is the metadata [`WorkerHistory`] carries, and every following
//! line is one message as it is pushed. Appending one line per message keeps
//! the conversation durable across a crash -- only the in-flight turn is
//! lost -- and a torn final line (a crash mid-append) is dropped on load
//! instead of failing the read. The legacy whole-file `.history.json` is still
//! read for workers written by an older build.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::{Path, PathBuf};

use crate::agent::{ChatMessage, Role};

/// Prefix of the user message a revision appends after the reloaded history.
///
/// Kept in one place so the pool, the MCP schema text and the tests agree on
/// the shape of the turn the model sees after its terminal summary.
pub const REVISION_PREFIX: &str =
    "REVISION REQUEST from the orchestrator after reviewing your branch:";

/// Prefix of the user message a continuation appends when the previous run
/// stopped for a reason other than a completed review.
///
/// A `failed`, `killed` or `interrupted` worker is not a dead end: the branch
/// still holds its work, so the steer continues it with the same id and the
/// same context instead of dispatching a replacement.
pub const CONTINUE_PREFIX: &str = "CONTINUE: your previous run stopped";

/// Automatic continuations the hub spends on one interrupted worker before it
/// leaves it to the orchestrator.
///
/// A worker that the hub keeps restarting and that keeps being interrupted is
/// not making progress, so after this many automatic "the hub restarted"
/// continuations the row stays `interrupted` and only an explicit
/// `steer <id> "..."` moves it.
pub const MAX_AUTO_CONTINUES: usize = 3;

/// The branch `repo_path` has checked out, the way dispatch detects it.
///
/// A conversation saved before base-branch tracking existed carries no
/// `base_branch`, so its continuation would never sync the base before
/// completing. Detecting it here -- once, at continuation time -- and storing
/// it on the history restores the pre-completion base sync for that worker.
pub fn detect_base_branch(repo_path: &std::path::Path) -> Option<String> {
    let out = crate::worktree::git(
        repo_path,
        "symbolic-ref",
        &["symbolic-ref", "--quiet", "--short", "HEAD"],
    )
    .ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// Fill in `base_branch` on a history that predates base-branch tracking.
///
/// Returns the branch that was detected, if any, so the caller can record it
/// on the registry row too. A history that already names its base branch is
/// left untouched.
pub async fn ensure_base_branch(
    history: &mut WorkerHistory,
    repo_path: &std::path::Path,
) -> Option<String> {
    if history.base_branch.is_some() {
        return None;
    }
    let repo = repo_path.to_path_buf();
    let detected = tokio::task::spawn_blocking(move || detect_base_branch(&repo))
        .await
        .ok()
        .flatten()?;
    history.base_branch = Some(detected.clone());
    Some(detected)
}

/// Fresh turn budget a steered revision starts with.
///
/// The previous run may have burned its whole dispatch budget, so a revision
/// restarts the loop from this (or from an explicit `max_turns` on the steer)
/// instead of inheriting whatever the finished worker had left.
pub const DEFAULT_REVISION_TURNS: usize = 60;

/// The metadata a finished run leaves behind so a later steer can relaunch it.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WorkerHistory {
    pub task: String,
    /// Owning group of the original dispatch, preserved across revisions.
    #[serde(default)]
    pub group: Option<String>,
    pub model: String,
    pub temperature: Option<f32>,
    pub repo_path: String,
    /// Commit the previous worktree measured its diff against. The revision
    /// re-attaches to the branch with this as the diff base, so history and
    /// checkpoints survive.
    pub base_commit: String,
    /// Branch checked out at the original dispatch, absent in older histories.
    #[serde(default)]
    pub base_branch: Option<String>,
    /// The branch the previous run committed to (`worker-<id>`).
    pub branch: String,
    pub network_offline: bool,
    /// `None` disables the verify gate, exactly as on dispatch.
    pub verify: Option<String>,
    /// The dispatcher's filtered ambient environment for the differential
    /// verify gate. Absent in files written before the gate existed.
    #[serde(default)]
    pub client_env: Vec<(String, String)>,
    pub max_turns: usize,
    pub review_after: Option<String>,
    /// Revision counter as of the run that wrote the file.
    pub revision: usize,
    /// Automatic "the hub restarted" continuations already spent on this
    /// worker. The daemon continues an interrupted worker at most
    /// [`MAX_AUTO_CONTINUES`] times; past that it stays `interrupted` for the
    /// orchestrator to steer by hand.
    #[serde(default)]
    pub auto_continues: usize,
    /// Agent that owns the worker; a revision keeps it. `None` in a file
    /// written before ownership was tracked.
    #[serde(default)]
    pub owner: Option<String>,
    /// The live conversation: system prompt, task, every assistant turn with
    /// its reasoning, every tool result.
    pub messages: Vec<ChatMessage>,
}

/// Path of the conversation file for `worker_id` (0600, beside the mailbox).
pub fn history_path(worker_id: &str) -> PathBuf {
    crate::worktree::swe_base_dir().join(format!("swe-wt-{worker_id}.history.json"))
}

/// Path of the append-only conversation log for `worker_id`.
///
/// Line one carries the metadata [`WorkerHistory`] has today; every following
/// line is one message as it is pushed, so the durable conversation grows one
/// line at a time instead of being rewritten whole.
pub fn history_log_path(worker_id: &str) -> PathBuf {
    crate::worktree::swe_base_dir().join(format!("swe-wt-{worker_id}.history.jsonl"))
}

/// Append one message to `worker_id`'s conversation log, creating it with the
/// metadata line when it does not exist yet.
///
/// The append is a single `write_all` of one line on a file opened in append
/// mode, so a crash can only ever tear the *last* line: the loader drops it
/// and keeps everything before it. The file is 0600 (it may carry tool output
/// with secrets) and the directory is created on first use.
pub fn append_history_message(
    worker_id: &str,
    meta: &WorkerHistory,
    message: &ChatMessage,
) -> Result<()> {
    let path = history_log_path(worker_id);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Could not create history dir {}", parent.display()))?;
    }
    let fresh = !path.exists();
    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .with_context(|| format!("Could not open history log {}", path.display()))?;
    restrict_to_owner(&path)?;
    let mut payload = String::new();
    if fresh {
        // The metadata line carries the relaunch facts only: the conversation
        // is the lines that follow it, one per message.
        let mut meta = meta.clone();
        meta.messages.clear();
        payload.push_str(
            &serde_json::to_string(&meta).context("Could not serialize history metadata")?,
        );
        payload.push('\n');
    }
    payload
        .push_str(&serde_json::to_string(message).context("Could not serialize history message")?);
    payload.push('\n');
    let mut file = file;
    std::io::Write::write_all(&mut file, payload.as_bytes())
        .with_context(|| format!("Could not append to history log {}", path.display()))?;
    Ok(())
}

/// Rewrite the metadata line (line one) of `worker_id`'s conversation log.
///
/// A continuation changes `revision` and the turn budget but appends only
/// message lines, so without this the log keeps naming the revision of the
/// dispatch that created it and every later continuation reads the same
/// counter again. The message lines are copied through untouched; only line
/// one is replaced, atomically, so a reader never sees a half-written log.
fn save_history_metadata(worker_id: &str, meta: &WorkerHistory) -> Result<()> {
    let path = history_log_path(worker_id);
    let mut meta = meta.clone();
    meta.messages.clear();
    let line = serde_json::to_string(&meta).context("Could not serialize history metadata")?;
    let existing = match std::fs::read_to_string(&path) {
        Ok(raw) => raw,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(e) => {
            return Err(e)
                .with_context(|| format!("Could not read history log {}", path.display()));
        }
    };
    let mut payload = line;
    payload.push('\n');
    // Skip the stale metadata line; every line after it is a message.
    for message_line in existing.lines().skip(1) {
        payload.push_str(message_line);
        payload.push('\n');
    }
    write_private_atomic(&path, payload.as_bytes())
}

/// Load the conversation log, tolerating a torn last line.
///
/// A crash mid-append leaves a partial JSON object behind; that line is
/// dropped and the messages before it are kept, so a reload never fails on
/// the one line a crash was in the middle of writing.
pub fn load_worker_history_log(worker_id: &str) -> Result<WorkerHistory> {
    let path = history_log_path(worker_id);
    let raw = std::fs::read_to_string(&path).with_context(|| {
        format!(
            "No saved conversation for worker {worker_id} at {}; it cannot be revised",
            path.display()
        )
    })?;
    let mut lines = raw.lines().filter(|l| !l.trim().is_empty());
    let meta_line = lines.next().with_context(|| {
        format!(
            "Conversation log for worker {worker_id} at {} is empty; it cannot be revised",
            path.display()
        )
    })?;
    let mut history: WorkerHistory = serde_json::from_str(meta_line).with_context(|| {
        format!(
            "Conversation log for worker {worker_id} at {} is corrupt; it cannot be revised",
            path.display()
        )
    })?;
    // The metadata line is the relaunch facts; the conversation is the lines
    // after it, so anything it carried is discarded here.
    history.messages.clear();
    for line in lines {
        // A torn final line is the one failure a crash can leave: skip it and
        // every later line (there is none in practice) rather than losing the
        // whole conversation.
        let Ok(message) = serde_json::from_str::<ChatMessage>(line) else {
            tracing::warn!(
                worker = %worker_id,
                path = %path.display(),
                "Dropping torn last line of the conversation log"
            );
            break;
        };
        history.messages.push(message);
    }
    if !is_replayable(&history.messages) {
        anyhow::bail!(
            "Saved conversation for worker {worker_id} at {} is not replayable; it cannot be revised",
            path.display()
        );
    }
    Ok(history)
}

/// Persist `history` atomically with owner-only permissions.
///
/// A rename-into-place write must not expose the file early under umask 022
/// (the conversation may carry secrets from tool output), so the staging file
/// is chmodded *before* the first byte lands; the rename is then what commits
/// the record. `swe_base_dir()` is not world-writable by construction.
pub fn save_worker_history(worker_id: &str, history: &WorkerHistory) -> Result<()> {
    let path = history_path(worker_id);
    let json = serde_json::to_string(history).context("Could not serialize worker history")?;
    write_private_atomic(&path, json.as_bytes())
}

/// Replace `path` with `bytes` through an owner-only staging file and a
/// rename, so a reader never observes a half-written record.
fn write_private_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    static STAGING_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Could not create history dir {}", parent.display()))?;
    }
    let mut staging = path.as_os_str().to_os_string();
    staging.push(format!(
        ".{}.{}.tmp",
        std::process::id(),
        STAGING_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    let staging = PathBuf::from(staging);

    let file = std::fs::File::create(&staging)
        .with_context(|| format!("Could not create history file {}", staging.display()))?;
    restrict_to_owner(&staging)?;
    std::io::Write::write_all(&mut { file }, bytes)
        .with_context(|| format!("Could not write history file {}", staging.display()))?;
    std::fs::rename(&staging, path)
        .with_context(|| format!("Could not commit history file {}", path.display()))?;
    Ok(())
}

#[cfg(unix)]
fn restrict_to_owner(path: &std::path::Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
        .with_context(|| format!("Could not restrict history file {}", path.display()))
}

#[cfg(not(unix))]
fn restrict_to_owner(_path: &std::path::Path) -> Result<()> {
    Ok(())
}

/// Is `messages` a conversation the provider accepts?
///
/// The pool records only valid turns, but a file survives crashes, edits and
/// older builds: a `tool` turn without a call id (or an assistant turn with an
/// empty call list) would fail the *whole* request, so the loader rejects the
/// file instead of starting a worker that can never call the model.
pub fn is_replayable(messages: &[ChatMessage]) -> bool {
    if messages.is_empty() {
        return false;
    }
    if !matches!(messages[0].role(), Role::System) {
        return false;
    }
    messages.iter().all(|m| m.is_wire_valid())
}

/// Load the conversation the finished worker left behind.
///
/// The append-only log is the canonical store; the legacy whole-file
/// `.history.json` is still read for a worker an older build wrote and no
/// newer run has appended to.
pub fn load_worker_history(worker_id: &str) -> Result<WorkerHistory> {
    if history_log_path(worker_id).is_file() {
        return load_worker_history_log(worker_id);
    }
    let path = history_path(worker_id);
    let raw = std::fs::read_to_string(&path).with_context(|| {
        format!(
            "No saved conversation for worker {worker_id} at {}; it cannot be revised",
            path.display()
        )
    })?;
    let history: WorkerHistory = serde_json::from_str(&raw).with_context(|| {
        format!(
            "Saved conversation for worker {worker_id} at {} is corrupt; it cannot be revised",
            path.display()
        )
    })?;
    if !is_replayable(&history.messages) {
        anyhow::bail!(
            "Saved conversation for worker {worker_id} at {} is not replayable; it cannot be revised",
            path.display()
        );
    }
    Ok(history)
}

/// Delete the conversation file of `worker_id` (on prune).
///
/// Every known base dir is swept: a `SWE_TEMP_DIR` that moved is not a reason
/// to leak an owner-only file carrying tool output.
pub fn remove_worker_history(worker_id: &str) {
    for base in crate::worktree::swe_base_dirs() {
        let path = base.join(format!("swe-wt-{worker_id}.history.json"));
        remove_quietly(worker_id, &path);
        let path = base.join(format!("swe-wt-{worker_id}.history.jsonl"));
        remove_quietly(worker_id, &path);
    }
}

fn remove_quietly(worker_id: &str, path: &Path) {
    match std::fs::remove_file(path) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => tracing::warn!(
            worker = %worker_id,
            path = %path.display(),
            error = %e,
            "Failed to remove worker history file"
        ),
    }
}

/// Delete the saved conversations of `repo_root`'s workers whose branch is
/// gone (merged and deleted, or pruned): nothing can revise them any more.
///
/// A finished worker has no worktree left, so the worktree sweep of `prune`
/// never reaches its history file; this is the sweep that does.
pub fn prune_orphan_histories(repo_root: &Path) -> usize {
    /// The two fields the sweep needs, without parsing the conversation.
    #[derive(Deserialize)]
    struct Owner {
        repo_path: String,
        branch: String,
    }
    let Ok(root) = repo_root.canonicalize() else {
        return 0;
    };
    let mut removed = 0;
    for base in crate::worktree::swe_base_dirs() {
        let Ok(entries) = std::fs::read_dir(&base) else {
            continue;
        };
        for entry in entries.flatten() {
            let name = entry.file_name();
            // Both stores carry the same metadata on their first line, so one
            // sweep covers the append-only log and the legacy whole-file.
            let Some(id) = name
                .to_str()
                .and_then(|n| n.strip_prefix("swe-wt-"))
                .and_then(|n| {
                    n.strip_suffix(".history.jsonl")
                        .or_else(|| n.strip_suffix(".history.json"))
                })
            else {
                continue;
            };
            let Some(owner) = std::fs::read_to_string(entry.path()).ok().and_then(|raw| {
                // The log's first line is the metadata; the whole-file
                // form is the same object, so both parse the same way.
                let first = raw.lines().find(|l| !l.trim().is_empty())?;
                serde_json::from_str::<Owner>(first).ok()
            }) else {
                continue;
            };
            if Path::new(&owner.repo_path).canonicalize().ok().as_deref() != Some(root.as_path()) {
                continue;
            }
            let reference = format!("refs/heads/{}", owner.branch);
            let branch_exists = crate::worktree::git(
                &root,
                "rev-parse",
                &["rev-parse", "--verify", "--quiet", &reference],
            )
            .is_ok_and(|out| out.status.success());
            if !branch_exists {
                remove_worker_history(id);
                removed += 1;
            }
        }
    }
    removed
}

use super::runner::WorkerLaunchConfig;

/// What [`WorkerPool::steer_with_budget`] actually did, so the reply can only
/// claim what happened.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SteerOutcome {
    /// A running worker took the message: it is applied on its next step.
    Queued,
    /// A paused worker was resumed with the message.
    Resumed,
    /// A stopped worker was continued on its own id and branch. `revision`
    /// counts the run; `cold` is true when the conversation had to be rebuilt
    /// from the registry row because no history survived.
    Continuing { revision: usize, cold: bool },
}

impl SteerOutcome {
    /// The verb the reply uses: `steered`, `resumed`, `revising` or
    /// `continuing`. A cold continuation is a continuation, not a revision --
    /// nothing was replayed.
    pub fn verb(self) -> &'static str {
        match self {
            Self::Queued => "steered",
            Self::Resumed => "resumed",
            Self::Continuing { cold: true, .. } => "continuing",
            Self::Continuing { .. } => "revising",
        }
    }
}

/// Revision counter of a [`SteerOutcome::Continuing`], for the reply.
fn outcome_revision(outcome: &SteerOutcome) -> usize {
    match outcome {
        SteerOutcome::Continuing { revision, .. } => *revision,
        _ => 0,
    }
}

impl super::WorkerPool {
    /// Continue a stopped worker `id` with `message`, whatever stopped it.
    ///
    /// With a surviving conversation the worker is revised on its own branch
    /// (the [`REVISION_PREFIX`] path after a completed review, the
    /// [`CONTINUE_PREFIX`] path after anything else). Without one -- a legacy
    /// worker, or one whose log was lost -- it is continued cold: the same id
    /// and branch, a fresh conversation that names the work the branch already
    /// holds. Only a missing branch is an error.
    pub async fn continue_worker(
        &self,
        id: &str,
        message: String,
        revision_turns: Option<usize>,
    ) -> anyhow::Result<SteerOutcome> {
        match super::load_worker_history(id) {
            Ok(mut history) => {
                let repo_path = std::path::PathBuf::from(&history.repo_path);
                // A history from before base-branch tracking would never sync
                // the base before completing; detect and store it now.
                let detected = ensure_base_branch(&mut history, &repo_path).await;
                let reason = super::load_registry_entry(id)
                    .map(|e| e.status)
                    .unwrap_or(super::RegistryStatus::Stopped);
                let prefix = match reason {
                    super::RegistryStatus::Completed => REVISION_PREFIX.to_string(),
                    other => format!(
                        "{CONTINUE_PREFIX} ({reason}). Orchestrator:",
                        reason = other.display_name()
                    ),
                };
                let outcome = self
                    .revise_with_prefix(id, history, prefix, message, revision_turns)
                    .await?;
                if let Some(branch) = detected {
                    // Persist the detected base branch on the row so the next
                    // continuation and every reader see it.
                    self.record_base_branch(id, &branch).await;
                }
                Ok(outcome)
            }
            Err(e) => {
                tracing::debug!(
                    worker = %id,
                    error = %e,
                    "No surviving conversation; continuing the worker cold"
                );
                self.cold_continue(id, message, revision_turns).await
            }
        }
    }

    /// Store the base branch detected during a continuation on the registry
    /// row, so the pre-completion base sync keeps running for that worker.
    async fn record_base_branch(&self, id: &str, base_branch: &str) {
        let base_branch = base_branch.to_string();
        let id = id.to_string();
        let _ = tokio::task::spawn_blocking(move || {
            let Some(mut entry) = super::load_registry_entry(&id) else {
                return;
            };
            if entry.base_branch.as_deref() == Some(base_branch.as_str()) {
                return;
            }
            entry.base_branch = Some(base_branch);
            super::save_registry_entry(&entry);
        })
        .await;
    }

    /// Continue a worker with no surviving conversation: same id, same branch,
    /// a fresh conversation built from the registry row.
    ///
    /// The registry row still names the task, the model and the branch, so the
    /// fresh conversation is the system prompt, the original task, a note that
    /// the branch already holds the previous attempt's work, and the steer
    /// message. Only a missing branch is an error.
    async fn cold_continue(
        &self,
        id: &str,
        message: String,
        revision_turns: Option<usize>,
    ) -> anyhow::Result<SteerOutcome> {
        let entry = super::load_registry_entry(id).with_context(|| {
            format!(
                "Worker {id} has no saved conversation and no registry row, so it cannot be continued"
            )
        })?;
        let repo_path = entry
            .repo_path
            .clone()
            .map(std::path::PathBuf::from)
            .filter(|p| p.is_dir())
            .with_context(|| {
                format!("Repository of worker {id} no longer exists; it cannot be continued")
            })?;
        let branch = format!("worker-{id}");
        // Fail fast when the branch is gone, before anything is relaunched:
        // this is the one condition a continuation cannot work around.
        {
            let repo = repo_path.clone();
            let reference = format!("refs/heads/{branch}");
            let exists = tokio::task::spawn_blocking(move || {
                crate::worktree::git(
                    &repo,
                    "show-ref",
                    &["show-ref", "--verify", "--quiet", &reference],
                )
                .map(|o| o.status.success())
                .unwrap_or(false)
            })
            .await
            .unwrap_or(false);
            if !exists {
                anyhow::bail!("branch {branch} no longer exists");
            }
        }

        let max_turns = revision_turns.unwrap_or(super::DEFAULT_REVISION_TURNS);
        if max_turns == 0 {
            anyhow::bail!("Revision budget for worker {id} must be at least 1 turn");
        }

        let system = crate::manifest::build_system_prompt(&repo_path, &entry.model);
        let base_branch = detect_base_branch(&repo_path);
        // The base the diff is measured from: the recorded one, else the
        // merge-base of the branch with the base branch.
        let base_commit = self
            .resolve_continuation_base(&repo_path, &branch, base_branch.as_deref())
            .await
            .or_else(|| entry.base_commit.clone())
            .unwrap_or_default();
        let task = entry.task.clone();
        let conversation = format!(
            "The task you were dispatched with:\n{task}\n\n\
             CONTINUATION: your previous run stopped ({reason}) without finishing, and its \
             conversation was not saved. The branch `{branch}` still holds the work that run \
             left behind -- start by reading it:\n  \
             git log --oneline <base>..HEAD\n  \
             git diff <base>...HEAD --stat\n\
             then continue from there. The orchestrator's message:\n{message}",
            reason = entry.status.display_name(),
        );
        let history = WorkerHistory {
            task,
            group: entry.group.clone(),
            model: entry.model.clone(),
            temperature: None,
            repo_path: repo_path.to_string_lossy().to_string(),
            base_commit,
            base_branch,
            branch: branch.clone(),
            network_offline: false,
            verify: None,
            client_env: Vec::new(),
            max_turns,
            review_after: None,
            revision: entry.revision,
            auto_continues: entry.auto_continues,
            owner: entry.owner.clone(),
            messages: vec![
                ChatMessage::text(crate::agent::Role::System, system),
                ChatMessage::text(crate::agent::Role::User, conversation),
            ],
        };
        let outcome = self
            .revise_with_prefix(id, history, String::new(), String::new(), Some(max_turns))
            .await?;
        Ok(SteerOutcome::Continuing {
            revision: outcome_revision(&outcome),
            cold: true,
        })
    }

    /// Relaunch a finished worker as a revision: same id, same branch, full
    /// context plus the orchestrator's corrections.
    ///
    /// The history file supplies the saved conversation and the relaunch facts
    /// (model, repo, base commit, network policy, verify gate). The steer text
    /// is appended as a user message with the [`REVISION_PREFIX`] marker, the
    /// turn budget restarts at `revision_turns` (or [`DEFAULT_REVISION_TURNS`]),
    /// the state returns to `Running`, and the same loop runs again -- same
    /// verify gate, same checkpoints, same completion payload. The revision
    /// counter in the record (and in every payload) is bumped once.
    ///
    /// Works for a worker held in this process *and* for a registry-only one
    /// (e.g. after a hub restart): the latter is re-registered here first, so
    /// both cases converge on one relaunch path. Refuses with a clear error
    /// when the history file is missing or the branch is gone (which of the
    /// two is named in the message).
    pub async fn revise(
        &self,
        id: &str,
        message: String,
        revision_turns: Option<usize>,
    ) -> anyhow::Result<()> {
        let mut history = super::load_worker_history(id)?;
        let repo_path = std::path::PathBuf::from(&history.repo_path);
        // A history from before base-branch tracking would never sync the base
        // before completing; detect and store it here too.
        let detected = ensure_base_branch(&mut history, &repo_path).await;
        let outcome = self
            .revise_with_prefix(
                id,
                history,
                REVISION_PREFIX.to_string(),
                message,
                revision_turns,
            )
            .await?;
        if let Some(branch) = detected {
            self.record_base_branch(id, &branch).await;
        }
        let _ = outcome;
        Ok(())
    }

    /// [`WorkerPool::revise`] with an explicit conversation and message prefix.
    ///
    /// A completed worker is revised with [`REVISION_PREFIX`]; one that stopped
    /// for any other reason is continued with [`CONTINUE_PREFIX`], which names
    /// the reason so the model knows what it is picking up from.
    pub(crate) async fn revise_with_prefix(
        &self,
        id: &str,
        mut history: WorkerHistory,
        prefix: String,
        message: String,
        revision_turns: Option<usize>,
    ) -> anyhow::Result<SteerOutcome> {
        let max_turns = revision_turns.unwrap_or(super::DEFAULT_REVISION_TURNS);
        if max_turns == 0 {
            anyhow::bail!("Revision budget for worker {id} must be at least 1 turn");
        }

        let repo_path = std::path::PathBuf::from(&history.repo_path);
        if !repo_path.is_dir() {
            anyhow::bail!(
                "Repository {} of worker {id} no longer exists; the finished worker cannot be revised",
                history.repo_path
            );
        }
        // Fail fast when the reviewed branch is gone, before the record is
        // touched: the error names the branch, not just the worker.
        {
            let branch = history.branch.clone();
            let repo = repo_path.clone();
            let exists = tokio::task::spawn_blocking(move || {
                crate::worktree::git(
                    &repo,
                    "show-ref",
                    &[
                        "show-ref",
                        "--verify",
                        "--quiet",
                        &format!("refs/heads/{branch}"),
                    ],
                )
                .map(|o| o.status.success())
                .unwrap_or(false)
            })
            .await
            .unwrap_or(false);
            if !exists {
                anyhow::bail!(
                    "Worker branch {} no longer exists; the finished worker {id} cannot be revised",
                    history.branch
                );
            }
        }

        if !prefix.is_empty() {
            history.messages.push(ChatMessage::text(
                crate::agent::Role::User,
                format!("{prefix}\n{message}"),
            ));
        }
        // Three stores carry the counter and any of them can lag: the log's
        // metadata line is written once at dispatch, the registry row is the
        // only copy a reaped worker leaves, and the in-process record is what
        // a running worker updates. Bump the highest, so a stale copy can
        // never make a continuation repeat the previous number.
        let persisted = {
            let in_memory = self
                .workers
                .read()
                .await
                .get(id)
                .map(|w| w.revision)
                .unwrap_or(0);
            let registry = super::load_registry_entry(id)
                .map(|e| e.revision)
                .unwrap_or(0);
            history.revision.max(in_memory).max(registry)
        };
        history.max_turns = max_turns;
        history.revision = persisted + 1;
        let revision = history.revision;
        // Append the messages the log does not have yet, so a worker relaunched
        // from it replays the same conversation. A warm continuation adds one
        // line; a cold one writes its whole fresh conversation. This happens
        // before the relaunch is visible anywhere, so the conversation is never
        // behind the record that points at it.
        let already = super::load_worker_history_log(id)
            .map(|logged| logged.messages.len())
            .unwrap_or(0);
        for message in history.messages.iter().skip(already) {
            if let Err(e) = super::append_history_message(id, &history, message) {
                tracing::warn!(
                    worker = %id,
                    error = %e,
                    "Could not append the continuation message to the history log"
                );
            }
        }

        // The log's metadata line still names the dispatch's revision; rewrite
        // it with the counter bumped above so the *next* continuation reads N,
        // not the stale N-1.
        if let Err(e) = save_history_metadata(id, &history) {
            tracing::warn!(
                worker = %id,
                error = %e,
                "Could not persist the revision counter to the history log"
            );
        }

        // A registry-only worker has no record here yet: register it so the
        // relaunch below -- and every poll on it -- sees one process's worker,
        // with the same id and branch the orchestrator reviewed.
        let now = super::unix_timestamp();
        let owner = history
            .owner
            .clone()
            .unwrap_or_else(|| super::UNATTRIBUTED_OWNER.to_string());
        let running = super::WorkerState::Running {
            step: 0,
            last_command: format!("revision {revision} starting"),
            started_at: now,
        };
        {
            let mut lock = self.workers.write().await;
            match lock.get_mut(id) {
                // Two steers racing on one finished worker must not launch
                // two revisions on the same branch.
                Some(w)
                    if matches!(
                        w.state,
                        super::WorkerState::Running { .. } | super::WorkerState::Paused { .. }
                    ) =>
                {
                    anyhow::bail!(
                        "Worker {id} is already running (revision {} in progress)",
                        w.revision
                    );
                }
                Some(w) => {
                    w.state = running;
                    w.model = history.model.clone();
                    w.task = history.task.clone();
                    w.revision = revision;
                }
                None => {
                    lock.insert(
                        id.to_string(),
                        super::WorkerRecord {
                            id: id.to_string(),
                            task: history.task.clone(),
                            model: history.model.clone(),
                            owner: owner.clone(),
                            state: running,
                            metrics: super::WorkerMetrics::default(),
                            logs: super::LogBuffer::with_policy(self.log_policy),
                            pending_steer: Vec::new(),
                            resume_tx: None,
                            handle: None,
                            revision,
                        },
                    );
                }
            }
            self.notify_change();
        }
        // The row that makes the revision visible to registry readers before
        // its first turn writes one; built before the conversation moves out.
        let row = super::WorkerRegistryEntry {
            id: id.to_string(),
            pid: std::process::id(),
            task: history.task.clone(),
            model: history.model.clone(),
            status: super::RegistryStatus::Running,
            step: 0,
            max_turns,
            last_command: format!("revision {revision} starting"),
            question: None,
            started_at: now,
            updated_at: now,
            group: history.group.clone(),
            repo_path: Some(history.repo_path.clone()),
            metrics: super::WorkerMetrics::default(),
            base_branch: history.base_branch.clone(),
            base_commit: Some(history.base_commit.clone()),
            revision,
            auto_continues: history.auto_continues,
            owner: Some(owner.clone()),
        };

        let pool = self.clone();
        let wid = id.to_string();
        let base_commit = history.base_commit.clone();
        let model_for_fail = history.model.clone();
        let meta = super::WorkerMeta {
            id: wid.clone(),
            task: history.task.clone(),
            group: history.group.clone(),
            repo_path: Some(history.repo_path.clone()),
            started_at: now,
            pid: std::process::id(),
            metrics: super::WorkerMetrics::default(),
            revision,
            auto_continues: history.auto_continues,
            owner,
        };
        let mut meta_for_fail = meta;
        let config = WorkerLaunchConfig {
            task: history.task.clone(),
            model: history.model.clone(),
            temperature: history.temperature,
            repo_path,
            max_turns,
            review_after: history.review_after.clone(),
            network_offline: history.network_offline,
            verify: history.verify.clone(),
            // A revision re-runs the same differential gate: the snapshot is
            // the one the original dispatch carried, so the check stays
            // deterministic even when the revising connection differs.
            client_env: history.client_env.clone(),
            resume_messages: Some(std::mem::take(&mut history.messages)),
            resume_base_commit: Some(base_commit.clone()),
            resume_base_branch: history.base_branch.clone(),
        };
        let handle = tokio::spawn(async move {
            if let Err(e) = pool
                .run_worker(wid.clone(), config, &mut meta_for_fail)
                .await
            {
                tracing::error!(worker = %wid, error = %e, "Revision failed with error");
                pool.update_worker(&wid, |w| w.fail(e.to_string())).await;
                pool.save_status(
                    &meta_for_fail,
                    &model_for_fail,
                    super::RegistryStatus::Failed,
                    0,
                    max_turns,
                    &format!("error: {e}"),
                    None,
                );
            }
        });
        {
            let mut lock = self.workers.write().await;
            if let Some(w) = lock.get_mut(id) {
                w.handle = Some(handle);
            }
        }
        super::save_registry_entry(&row);
        tracing::info!(worker = %id, revision, max_turns, "Worker revision started");
        Ok(SteerOutcome::Continuing {
            revision,
            cold: false,
        })
    }
}
