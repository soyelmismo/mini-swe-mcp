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
