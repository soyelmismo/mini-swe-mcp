//! Revision persistence: the finished worker's conversation and relaunch facts.
//!
//! The orchestrator reviews a finished worker's branch and then steers it with
//! corrections. The model must continue on its own branch with its full
//! context rather than restart from the task, so [`WorkerPool::run_worker`]
//! serializes the live conversation (system prompt, task, every assistant turn
//! with its reasoning, every tool result -- exactly what the next request
//! replays) plus the metadata the launch took apart again, into
//! `swe_base_dir()/swe-wt-<id>.history.json`. [`WorkerPool::steer`] reloads the
//! file, appends the revision request, and re-launches the same loop on the
//! same branch. History without its metadata is a dead letter, so one atomic
//! file carries both.

use anyhow::{Context, Result};
use std::path::PathBuf;

use crate::agent::{ChatMessage, Role};

/// Prefix of the user message a revision appends after the reloaded history.
///
/// Kept in one place so the pool, the MCP schema text and the tests agree on
/// the shape of the turn the model sees after its terminal summary.
pub const REVISION_PREFIX: &str =
    "REVISION REQUEST from the orchestrator after reviewing your branch:";

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
    /// The branch the previous run committed to (`worker-<id>`).
    pub branch: String,
    pub network_offline: bool,
    /// `None` disables the verify gate, exactly as on dispatch.
    pub verify: Option<String>,
    pub max_turns: usize,
    pub review_after: Option<String>,
    /// Revision counter as of the run that wrote the file.
    pub revision: usize,
    /// The live conversation: system prompt, task, every assistant turn with
    /// its reasoning, every tool result.
    pub messages: Vec<ChatMessage>,
}

/// Path of the conversation file for `worker_id` (0600, beside the mailbox).
pub fn history_path(worker_id: &str) -> PathBuf {
    crate::worktree::swe_base_dir().join(format!("swe-wt-{worker_id}.history.json"))
}

/// Persist `history` atomically with owner-only permissions.
///
/// A rename-into-place write must not expose the file early under umask 022
/// (the conversation may carry secrets from tool output), so the staging file
/// is chmodded *before* the first byte lands; the rename is then what commits
/// the record. `swe_base_dir()` is not world-writable by construction.
pub fn save_worker_history(worker_id: &str, history: &WorkerHistory) -> Result<()> {
    let path = history_path(worker_id);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Could not create history dir {}", parent.display()))?;
    }
    let mut staging = path.as_os_str().to_os_string();
    staging.push(format!(".{}.{}.tmp", std::process::id(), history.revision));
    let staging = PathBuf::from(staging);

    let file = std::fs::File::create(&staging)
        .with_context(|| format!("Could not create history file {}", staging.display()))?;
    restrict_to_owner(&staging)?;
    let json = serde_json::to_string(history).context("Could not serialize worker history")?;
    std::io::Write::write_all(&mut { file }, json.as_bytes())
        .with_context(|| format!("Could not write history file {}", staging.display()))?;
    std::fs::rename(&staging, &path)
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

/// Load the conversation file the finished worker left behind.
pub fn load_worker_history(worker_id: &str) -> Result<WorkerHistory> {
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

/// Delete the conversation file of `worker_id` (prune, reap, collect).
///
/// Every known base dir is swept: a `SWE_TEMP_DIR` that moved is not a reason
/// to leak an owner-only file carrying tool output.
pub fn remove_worker_history(worker_id: &str) {
    for base in crate::worktree::swe_base_dirs_for_cleanup() {
        let path = base.join(format!("swe-wt-{worker_id}.history.json"));
        match std::fs::remove_file(&path) {
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
}

use super::runner::WorkerLaunchConfig;

impl super::WorkerPool {
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
                    &["show-ref", "--verify", "--quiet", &format!("refs/heads/{branch}")],
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

        history.messages.push(ChatMessage::text(
            crate::agent::Role::User,
            format!("{REVISION_PREFIX}\n{message}"),
        ));
        history.max_turns = max_turns;
        history.revision += 1;
        let revision = history.revision;

        // A registry-only worker has no record here yet: register it so the
        // relaunch below -- and every poll on it -- sees one process's worker,
        // with the same id and branch the orchestrator reviewed.
        let now = super::unix_timestamp();
        {
            let mut lock = self.workers.write().await;
            match lock.get_mut(id) {
                Some(w) => {
                    w.state = super::WorkerState::Running {
                        step: 0,
                        last_command: format!("revision {revision} starting"),
                        started_at: now,
                    };
                    w.model = history.model.clone();
                    w.task = history.task.clone();
                    w.revision = revision;
                }
                None => {
                    let record = super::WorkerRecord {
                        id: id.to_string(),
                        task: history.task.clone(),
                        model: history.model.clone(),
                        state: super::WorkerState::Running {
                            step: 0,
                            last_command: format!("revision {revision} starting"),
                            started_at: now,
                        },
                        metrics: super::WorkerMetrics::default(),
                        logs: super::LogBuffer::with_policy(self.log_policy),
                        pending_steer: Vec::new(),
                        resume_tx: None,
                        handle: None,
                        revision,
                    };
                    lock.insert(id.to_string(), record);
                }
            }
        }

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
            resume_messages: Some(history.messages),
            resume_base_commit: Some(base_commit.clone()),
        };
        let handle = tokio::spawn(async move {
            if let Err(e) = pool.run_worker(wid.clone(), config, &mut meta_for_fail).await {
                tracing::error!(worker = %wid, error = %e, "Revision failed with error");
                let mut lock = pool.workers.write().await;
                if let Some(w) = lock.get_mut(&wid) {
                    w.fail(e.to_string());
                }
                meta_for_fail.save_status(
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
        // The revision's launch is what a re-attached `status` reports until
        // the first turn writes its own row (`history.messages` moved into the
        // launch above, so the row is rebuilt from the surviving fields).
        WorkerHistory {
            task: history.task.clone(),
            group: history.group.clone(),
            model: history.model.clone(),
            temperature: history.temperature,
            repo_path: history.repo_path.clone(),
            base_commit: base_commit.clone(),
            branch: history.branch.clone(),
            network_offline: history.network_offline,
            verify: history.verify.clone(),
            max_turns,
            review_after: history.review_after.clone(),
            revision,
            messages: Vec::new(),
        }
        .save_revision_status(id, revision, max_turns);
        tracing::info!(worker = %id, revision, max_turns, "Worker revision started");
        Ok(())
    }
}

impl WorkerHistory {
    /// The registry row that makes a just-started revision visible to
    /// cross-process readers before its first turn writes its own.
    fn save_revision_status(&self, worker_id: &str, revision: usize, max_turns: usize) {
        // Reuse the canonical row writer with the history's identity: the row
        // for the worker id carries the fresh budget and the revision marker.
        super::save_registry_entry(&super::WorkerRegistryEntry {
            id: worker_id.to_string(),
            pid: std::process::id(),
            task: self.task.clone(),
            model: self.model.clone(),
            status: super::RegistryStatus::Running,
            step: 0,
            max_turns,
            last_command: format!("revision {revision} starting"),
            question: None,
            started_at: super::unix_timestamp(),
            updated_at: super::unix_timestamp(),
            group: self.group.clone(),
            repo_path: Some(self.repo_path.clone()),
            metrics: super::WorkerMetrics::default(),
        });
    }
}
