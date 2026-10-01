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
use tracing::{debug, warn};

use crate::agent::{ChatMessage, Role};

use super::registry::WorkerRegistryEntry;
use super::state::{retention_expired, within_retired_grace};
use super::steer::remove_steer_file_in;

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
    /// Dispatch authority, preserved across warm and cold continuations.
    #[serde(default)]
    pub role: super::WorkerRole,
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
    history_path_in(&ScratchRoot::from_env(), worker_id)
}

/// [`history_path`] under an explicit scratch root.
pub fn history_path_in(root: &ScratchRoot, worker_id: &str) -> PathBuf {
    root.join(format!("swe-wt-{worker_id}.history.json"))
}

/// Path of the append-only conversation log for `worker_id`.
///
/// Line one carries the metadata [`WorkerHistory`] has today; every following
/// line is one message as it is pushed, so the durable conversation grows one
/// line at a time instead of being rewritten whole.
pub fn history_log_path(worker_id: &str) -> PathBuf {
    history_log_path_in(&ScratchRoot::from_env(), worker_id)
}

/// [`history_log_path`] under an explicit scratch root.
pub fn history_log_path_in(root: &ScratchRoot, worker_id: &str) -> PathBuf {
    root.join(format!("swe-wt-{worker_id}.history.jsonl"))
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
    append_history_message_in(&ScratchRoot::from_env(), worker_id, meta, message)
}

/// [`append_history_message`] under an explicit scratch root.
pub fn append_history_message_in(
    root: &ScratchRoot,
    worker_id: &str,
    meta: &WorkerHistory,
    message: &ChatMessage,
) -> Result<()> {
    let path = history_log_path_in(root, worker_id);
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
fn save_history_metadata_in(
    root: &ScratchRoot,
    worker_id: &str,
    meta: &WorkerHistory,
) -> Result<()> {
    let path = history_log_path_in(root, worker_id);
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
    load_worker_history_log_in(&ScratchRoot::from_env(), worker_id)
}

/// [`load_worker_history_log`] under an explicit scratch root.
pub fn load_worker_history_log_in(root: &ScratchRoot, worker_id: &str) -> Result<WorkerHistory> {
    let path = history_log_path_in(root, worker_id);
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
    save_worker_history_in(&ScratchRoot::from_env(), worker_id, history)
}

/// [`save_worker_history`] under an explicit scratch root.
pub fn save_worker_history_in(
    root: &ScratchRoot,
    worker_id: &str,
    history: &WorkerHistory,
) -> Result<()> {
    let path = history_path_in(root, worker_id);
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
    load_worker_history_in(&ScratchRoot::from_env(), worker_id)
}

/// [`load_worker_history`] under an explicit scratch root.
pub fn load_worker_history_in(root: &ScratchRoot, worker_id: &str) -> Result<WorkerHistory> {
    if history_log_path_in(root, worker_id).is_file() {
        return load_worker_history_log_in(root, worker_id);
    }
    let path = history_path_in(root, worker_id);
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
    remove_worker_history_in(&ScratchRoot::from_env(), worker_id);
}

/// [`remove_worker_history`] under an explicit scratch root.
pub fn remove_worker_history_in(root: &ScratchRoot, worker_id: &str) {
    for base in root.base_dirs() {
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

/// [`retire_worker_in`] under the default scratch root.
pub fn retire_worker(worker_id: &str) {
    retire_worker_in(&ScratchRoot::from_env(), worker_id);
}

/// Everything a retirement needs to reach outside the scratch root: the
/// repository the worker's branch lives in, and the hub directory holding the
/// persisted watch acknowledgements.
///
/// Both are optional because the callers are not alike. A prune that only
/// knows the scratch root still removes every scratch trace; a merge knows the
/// repository and takes the branch with it.
#[derive(Debug, Clone, Default)]
pub struct RetireContext<'a> {
    /// Repository the worker's `worker-<id>` branch lives in. `None` leaves the
    /// branch alone rather than guessing a repository that may hold other
    /// work.
    pub repo: Option<&'a Path>,
    /// Hub directory holding `watch_acks.json`. `None` skips the ack store.
    pub ack_dir: Option<&'a Path>,
    /// Whether the `worker-<id>` branch itself survives.
    ///
    /// `merge --no-delete` sets this: the operator asked to keep the branch, so
    /// the scratch traces still go but the ref stays. Everything else retires
    /// the branch too, because a branch whose commits are already in the base is
    /// exactly what the sweep would delete on its next pass.
    pub keep_branch: bool,
}

/// Retire every trace of `worker_id`: its branch, its registry row, its saved
/// conversation, its steering mailbox and steer-source, its watch
/// acknowledgements, and the worktree, scratch and build directories it held.
///
/// The one deletion path, so a row can never outlive the conversation it names
/// (or the other way round) and leave a half-known worker behind. A retired
/// worker is fully integrated: its commits are in the base branch, so nothing
/// here can lose work that is not already in the repository.
///
/// Called for an integrated worker *immediately* -- by `merge`, by
/// `merge --approved`, and when a consolidator lands -- and by
/// [`sweep_retired_workers_in`] for whatever a merge could not reach. Every
/// step is best effort: retirement is idempotent, and a file that is already
/// gone is the desired end state, not an error.
pub fn retire_worker_in(root: &ScratchRoot, worker_id: &str) {
    retire_worker_with(root, worker_id, &RetireContext::default());
}

/// What a retirement actually did, as the callers must report it.
///
/// The retirement is the only code that deletes the branch, so its *result*
/// is the truth a caller reports: `cleanup` used to test `git branch -D`
/// separately and could report a deletion the retirement never performed (or
/// miss one it did), which then silently skipped a consolidator's round.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RetireOutcome {
    /// The `worker-<id>` ref is gone. False when the branch was kept, when it
    /// was already absent, or when the repository could not be probed -- the
    /// last case is deliberately not "deleted", because nothing proved it.
    pub branch_deleted: bool,
    /// A leftover worktree directory was reclaimed.
    pub worktree_reclaimed: bool,
    /// The registry row was deleted. A kept branch keeps its row.
    pub row_removed: bool,
}

/// [`retire_worker_in`] with the repository and hub directory the retirement
/// can also clean.
pub fn retire_worker_with(root: &ScratchRoot, worker_id: &str, ctx: &RetireContext<'_>) {
    retire_worker_reporting(root, worker_id, ctx);
}

/// [`retire_worker_with`], returning what it actually did.
///
/// The reporting twin of the retirement, so no caller has to re-derive the
/// result of a deletion it did not perform itself.
pub fn retire_worker_reporting(
    root: &ScratchRoot,
    worker_id: &str,
    ctx: &RetireContext<'_>,
) -> RetireOutcome {
    let branch = format!("worker-{worker_id}");
    // The worktree goes first: a leftover that is still registered would make
    // the branch undeletable, and `git worktree prune` clears the registration
    // once its directory is gone.
    let worktree = root.join(format!("swe-wt-{worker_id}"));
    let reclaimed = worktree.exists();
    if reclaimed {
        crate::worktree::force_remove_dir(&worktree);
    }
    let mut branch_deleted = false;
    crate::worktree::remove_target_dirs_in(root, &worktree);
    // The worker's build-directory *lease* is deliberately not touched here: the
    // directories are filed per repository, not per worker, and are shared by
    // every live worker of that repository. The lease itself is a guard that
    // releases when the worker's guard drops, which is what frees the directory
    // for the next worker; the warm directory stays, to be reclaimed by the
    // build-dir sweep once it is idle.
    if let Some(repo) = ctx.repo.filter(|repo| repo.is_dir() && !ctx.keep_branch) {
        let _ = crate::worktree::git(repo, "worktree prune", &["worktree", "prune"]);
        // `git branch -D` refuses a branch a worktree still has checked out;
        // the prune above just released it. An already-absent branch is not an
        // error: the end state is the same, so report "gone" either way.
        if crate::worktree::git(repo, "branch -D", &["branch", "-D", &branch])
            .is_ok_and(|out| out.status.success())
        {
            branch_deleted = true;
        }
    }
    for suffix in ["steer-source", "round-base"] {
        let _ = std::fs::remove_file(root.join(format!("swe-wt-{worker_id}.{suffix}")));
    }
    remove_worker_history_in(root, worker_id);
    remove_steer_file_in(root, worker_id);
    // A branch that was deliberately kept leaves a worker that is still known:
    // `status`, `list` and a later `merge` all read its row. Its branch is what
    // keeps it meaningful, and the row is retired with the branch the moment
    // that goes -- by the sweep, or by the next merge.
    let mut row_removed = false;
    if !ctx.keep_branch {
        row_removed = super::remove_registry_entry_in(root, worker_id);
        if let Some(dir) = ctx.ack_dir {
            crate::mcp::events::forget_watch_acks(dir, worker_id);
        }
    } else if let Some(mut row) = super::load_registry_entry_in(root, worker_id)
        && !row.keep_branch
    {
        // Durable `--no-delete`: mark the row so every later sweep skips this
        // worker while its branch lives, instead of only the one pass that
        // happens to follow the merge.
        row.keep_branch = true;
        row.updated_at = super::unix_timestamp();
        super::save_registry_entry_in(root, &row);
    }
    if reclaimed {
        debug!(worker = %worker_id, "Reclaimed retired worker leftovers");
    }
    RetireOutcome {
        branch_deleted,
        worktree_reclaimed: reclaimed,
        row_removed,
    }
}

/// [`sweep_retired_workers_in`] under the default scratch root.
pub fn sweep_retired_workers() -> RetireSweep {
    sweep_retired_workers_in(&ScratchRoot::from_env(), None)
}

/// What one retirement sweep reclaimed.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RetireSweep {
    /// Workers whose branch is merged into its base, retired with it.
    pub workers: Vec<String>,
    /// History, steer or steer-source files with neither a registry row nor a
    /// branch to go with them.
    pub orphans: usize,
}

/// Retire every worker that is already integrated, and delete the leftovers of
/// workers nobody can ever continue again.
///
/// Two independent jobs, both cheap and bounded:
///
/// * a **merged** worker -- its `worker-<id>` branch is an ancestor of the base
///   branch its row records -- is retired outright: its commits are in the base
///   branch, so branch, row, history, mailbox and scratch all go now. This is
///   what keeps `list` showing only live and awaiting-integration workers, with
///   no hiding logic anywhere.
/// * an **orphan** file -- a history, steer or steer-source file with neither a
///   registry row nor a branch -- is deleted, because nothing can consume it
///   again. This is what reclaims the hundreds of history files old workers
///   left behind.
///
/// The [`retired grace`](within_retired_grace) period is deliberately *not*
/// applied here: it protects a branch that vanished **without** being merged
/// (see [`prune_orphan_histories_with_retention_and_grace_in`]), which may
/// still be recreated from its recorded head. A branch proven merged needs no
/// such protection, so retiring it immediately is safe.
///
/// Every git probe is read-only and every step is best effort, so an
/// unprobeable repository retires nothing rather than retiring the wrong
/// worker.
pub fn sweep_retired_workers_in(root: &ScratchRoot, ack_dir: Option<&Path>) -> RetireSweep {
    let mut sweep = RetireSweep::default();
    // Grouped by (repository, base): one `for-each-ref` per group answers the
    // merged-branch question for every row of that group, instead of one git
    // process per row.
    let mut groups: std::collections::BTreeMap<(PathBuf, String), Vec<String>> =
        std::collections::BTreeMap::new();
    for entry in super::load_registry_entries_read_only_in(root) {
        if entry.status.is_live() {
            continue;
        }
        let Some(repo) = entry.repo_path.as_deref().map(Path::new) else {
            continue;
        };
        let Some(base) = entry.base_branch.clone().filter(|base| !base.is_empty()) else {
            continue;
        };
        // A kept branch is a durable operator instruction, not a one-off pass:
        // leave the whole worker alone while its branch still lives.
        if entry.keep_branch
            && local_branches(repo).is_some_and(|set| set.contains(&format!("worker-{}", entry.id)))
        {
            continue;
        }
        groups
            .entry((repo.to_path_buf(), base))
            .or_default()
            .push(entry.id);
    }
    for ((repo, base), ids) in &groups {
        // The probe failed: retire nothing, and never guess a repository.
        let Some(merged) = merged_branches(repo, base) else {
            continue;
        };
        let ctx = RetireContext {
            repo: Some(repo.as_path()),
            ack_dir,
            keep_branch: false,
        };
        for id in ids {
            if !merged.contains(&format!("worker-{id}")) {
                continue;
            }
            retire_worker_with(root, id, &ctx);
            sweep.workers.push(id.clone());
        }
    }
    sweep.orphans = remove_orphan_worker_files(root);
    sweep
}

/// Every local branch of `repo`.
///
/// `None` means the probe itself failed, which is different from an empty set:
/// an empty set proves no branch exists, while `None` proves nothing and must
/// never authorise a retirement or a deletion.
fn local_branches(repo: &Path) -> Option<std::collections::HashSet<String>> {
    refs_of(
        repo,
        &["for-each-ref", "--format=%(refname:short)", "refs/heads/"],
    )
}

/// The local branches of `repo` already contained in `base`.
fn merged_branches(repo: &Path, base: &str) -> Option<std::collections::HashSet<String>> {
    refs_of(
        repo,
        &[
            "for-each-ref",
            "--format=%(refname:short)",
            &format!("--merged={base}"),
            "refs/heads/",
        ],
    )
}

/// Run one `git for-each-ref` and parse its branch lines, or `None` on failure.
fn refs_of(repo: &Path, args: &[&str]) -> Option<std::collections::HashSet<String>> {
    if !repo.is_dir() {
        return Some(std::collections::HashSet::new());
    }
    let out = crate::worktree::git(repo, "for-each-ref", args).ok()?;
    if !out.status.success() {
        return None;
    }
    Some(
        String::from_utf8_lossy(&out.stdout)
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .map(str::to_string)
            .collect(),
    )
}

/// The metadata line of a history file: the repository and branch its worker
/// worked on.
#[derive(Deserialize)]
struct OrphanOwner {
    repo_path: String,
    branch: String,
}

/// Cap on the metadata line read, so the scan never pulls a whole conversation
/// into memory to decide whether one file is reachable. The line is one small
/// JSON object; the cap only guards against a corrupt file.
const ORPHAN_METADATA_MAX_BYTES: u64 = 64 * 1024;

/// The worker id encoded in a scratch companion's name, if this name is one.
fn worker_id_from_name(name: &str) -> Option<&str> {
    let rest = name.strip_prefix("swe-wt-")?;
    [
        ".history.jsonl",
        ".history.json",
        ".steer",
        ".steer-source",
        ".round-base",
    ]
    .iter()
    .find_map(|suffix| rest.strip_suffix(suffix))
}

/// The ownership a history file states, read from its first line only.
///
/// A `.steer`, `.steer-source` or `.round-base` names no repository of its own
/// and yields `None`; the caller shares the ownership of the same worker's
/// history instead of assuming the file is unreachable.
fn read_orphan_owner(path: &Path) -> Option<OrphanOwner> {
    use std::io::{BufRead, BufReader, Read};
    let name = path.to_string_lossy();
    if !(name.ends_with(".history.jsonl") || name.ends_with(".history.json")) {
        return None;
    }
    let file = std::fs::File::open(path).ok()?;
    let mut reader = BufReader::new(file.take(ORPHAN_METADATA_MAX_BYTES));
    let mut line = String::new();
    reader.read_line(&mut line).ok()?;
    serde_json::from_str(line.trim()).ok()
}

/// Delete the per-worker files no row and no branch can ever claim again.
///
/// Ownership is shared across a worker's files: a history states the repository
/// and branch, and the worker's `.steer`, `.steer-source` and `.round-base`
/// companions inherit that ownership rather than being assumed branch-less. So a
/// worker's files are removed only when its ownership was actually probed *and*
/// none of its candidate branches exists. When nothing about a worker could be
/// probed -- no history to name an owner, or an unprobeable repository -- its
/// files are kept, because deleting a reachable conversation is unrecoverable
/// while keeping an unreachable one only costs space.
///
/// Bounded: the branch set is read once per repository, not once per file, and
/// only the metadata line of a history is read.
fn remove_orphan_worker_files(root: &ScratchRoot) -> usize {
    let live: std::collections::HashSet<String> = super::load_registry_entries_read_only_in(root)
        .into_iter()
        .map(|entry| entry.id)
        .collect();
    let mut files: Vec<(String, PathBuf)> = Vec::new();
    let mut owners: std::collections::HashMap<String, Vec<OrphanOwner>> =
        std::collections::HashMap::new();
    for base in root.base_dirs() {
        let Ok(entries) = std::fs::read_dir(&base) else {
            continue;
        };
        for entry in entries.flatten() {
            let name = entry.file_name();
            let Some(name) = name.to_str() else {
                continue;
            };
            let Some(id) = worker_id_from_name(name) else {
                continue;
            };
            let path = entry.path();
            if let Some(owner) = read_orphan_owner(&path) {
                owners.entry(id.to_string()).or_default().push(owner);
            }
            files.push((id.to_string(), path));
        }
    }
    let mut branches: std::collections::HashMap<PathBuf, Option<std::collections::HashSet<String>>> =
        std::collections::HashMap::new();
    let mut removed = 0;
    for (id, path) in files {
        if live.contains(&id) {
            continue;
        }
        let Some(candidates) = owners.get(&id) else {
            // Nothing names this worker's repository: its reachability cannot be
            // disproven, so the file stays.
            continue;
        };
        let mut probed = false;
        let mut live_branch = false;
        for owner in candidates {
            let repo = PathBuf::from(&owner.repo_path);
            let set = branches
                .entry(repo.clone())
                .or_insert_with(|| local_branches(&repo));
            match set {
                Some(set) => {
                    probed = true;
                    live_branch |= set.contains(&owner.branch);
                }
                // An unprobeable repository proves nothing about this file.
                None => live_branch = true,
            }
        }
        if !probed || live_branch {
            continue;
        }
        match std::fs::remove_file(&path) {
            Ok(()) => removed += 1,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => warn!(
                path = %path.display(),
                error = %e,
                "Failed to remove orphan worker file"
            ),
        }
    }
    removed
}

/// Retire the durable state of every terminal worker whose retention expired.
///
/// Age-based and repository-agnostic: a worker whose branch still exists keeps
/// its row and its conversation until this retention runs out, which is what
/// lets an orchestrator continue a worker it finished days ago. Returns how
/// many workers were retired.
pub fn retire_expired_terminal_workers_in(root: &ScratchRoot, retention_secs: u64) -> usize {
    let now = super::unix_timestamp();
    let mut retired = 0;
    for entry in super::load_registry_entries_read_only_in(root) {
        if !entry.status.is_terminal() || !retention_expired(entry.updated_at, retention_secs, now)
        {
            continue;
        }
        retire_worker_in(root, &entry.id);
        retired += 1;
    }
    retired
}

/// Delete the saved conversations of `repo_root`'s workers whose branch is
/// gone (merged and deleted, or pruned): nothing can revise them any more.
///
/// A finished worker has no worktree left, so the worktree sweep of `prune`
/// never reaches its history file; this is the sweep that does. A worker whose
/// branch survives is kept unless its retention expired, so a finished worker
/// stays continuable for as long as its branch does. One whose branch is gone
/// is kept through its retired grace period (see
/// [`super::state::DEFAULT_WORKER_RETIRED_GRACE_SECS`]) so an orchestrator that
/// reverts the merge can still continue it.
pub fn prune_orphan_histories(repo_root: &Path) -> usize {
    prune_orphan_histories_in(&ScratchRoot::from_env(), repo_root)
}

/// [`prune_orphan_histories`] under an explicit scratch root.
pub fn prune_orphan_histories_in(root: &ScratchRoot, repo_root: &Path) -> usize {
    prune_orphan_histories_with_retention_in(
        root,
        repo_root,
        super::state::terminal_retention_secs(),
    )
}

/// [`prune_orphan_histories_in`] with an explicit retention, so a caller (or a
/// test) can name the age instead of reading the environment.
///
/// The retired grace is still read from the environment; use
/// [`prune_orphan_histories_with_retention_and_grace_in`] to name both.
pub fn prune_orphan_histories_with_retention_in(
    root: &ScratchRoot,
    repo_root: &Path,
    retention_secs: u64,
) -> usize {
    prune_orphan_histories_with_retention_and_grace_in(
        root,
        repo_root,
        retention_secs,
        super::state::worker_retired_grace_secs(),
    )
}

/// [`prune_orphan_histories_with_retention_in`] with an explicit retired grace
/// too, so a caller (or a test) can name both ages instead of reading the
/// environment.
///
/// A worker whose branch is gone keeps its registry row and conversation for
/// `grace_secs` after its row was last written, because the merge that pruned
/// the branch may still be reverted and the worker continued. Without a row
/// there is nothing to recreate the branch from, so the conversation is
/// retired at once.
pub fn prune_orphan_histories_with_retention_and_grace_in(
    root: &ScratchRoot,
    repo_root: &Path,
    retention_secs: u64,
    grace_secs: u64,
) -> usize {
    /// The two fields the sweep needs, without parsing the conversation.
    #[derive(Deserialize)]
    struct Owner {
        repo_path: String,
        branch: String,
    }
    let Ok(repo) = repo_root.canonicalize() else {
        return 0;
    };
    let now = super::unix_timestamp();
    let mut removed = 0;
    for base in root.base_dirs() {
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
            if Path::new(&owner.repo_path).canonicalize().ok().as_deref() != Some(repo.as_path()) {
                continue;
            }
            let reference = format!("refs/heads/{}", owner.branch);
            let branch_exists = crate::worktree::git(
                &repo,
                "rev-parse",
                &["rev-parse", "--verify", "--quiet", &reference],
            )
            .is_ok_and(|out| out.status.success());
            let entry = super::load_registry_entry_in(root, id);
            // The branch outlived its retention, or it is gone with nothing
            // left to recreate it from, or its grace has run out: either way
            // the worker cannot be continued any more and the whole trace goes.
            let expired = entry
                .as_ref()
                .is_some_and(|e| retention_expired(e.updated_at, retention_secs, now));
            let kept_through_grace = entry
                .as_ref()
                .is_some_and(|e| within_retired_grace(e.updated_at, grace_secs, now));
            if expired || (!branch_exists && !kept_through_grace) {
                retire_worker_in(root, id);
                removed += 1;
            }
        }
    }
    removed
}

use super::runner::WorkerLaunchConfig;
use crate::worktree::ScratchRoot;

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
pub(crate) fn outcome_revision(outcome: &SteerOutcome) -> usize {
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
        match super::load_worker_history_in(&self.scratch, id) {
            Ok(mut history) => {
                let repo_path = std::path::PathBuf::from(&history.repo_path);
                // A history from before base-branch tracking would never sync
                // the base before completing; detect and store it now.
                let detected = ensure_base_branch(&mut history, &repo_path).await;
                let reason = super::load_registry_entry_in(&self.scratch, id)
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
        let root = self.scratch.clone();
        let _ = tokio::task::spawn_blocking(move || {
            let Some(mut entry) = super::load_registry_entry_in(&root, &id) else {
                return;
            };
            if entry.base_branch.as_deref() == Some(base_branch.as_str()) {
                return;
            }
            entry.base_branch = Some(base_branch);
            super::save_registry_entry_in(&root, &entry);
        })
        .await;
    }

    /// Record the branch head commit of a finished run on the registry row.
    ///
    /// A continuation recreates `worker-<id>` from this commit after a merge
    /// pruned the branch (within the retired grace period), so the row must
    /// name where the branch ended. Written at completion; rewritten when a
    /// later run finishes on a new commit.
    pub(crate) async fn record_head_commit(&self, id: &str, head_commit: Option<String>) {
        let Some(head_commit) = head_commit else {
            return;
        };
        let id = id.to_string();
        let root = self.scratch.clone();
        let _ = tokio::task::spawn_blocking(move || {
            let Some(mut entry) = super::load_registry_entry_in(&root, &id) else {
                return;
            };
            if entry.head_commit.as_deref() == Some(head_commit.as_str()) {
                return;
            }
            entry.head_commit = Some(head_commit);
            super::save_registry_entry_in(&root, &entry);
        })
        .await;
    }

    /// Make sure `branch` exists before a continuation relaunches on it.
    ///
    /// A merged worker's branch is pruned, but its row and conversation are
    /// kept through the retired grace period. When the branch is gone, recreate
    /// it at the head commit the row recorded at completion, so the
    /// continuation keeps the worker's work instead of failing. The error names
    /// the branch when there is nothing to recreate it from.
    async fn ensure_worker_branch(
        &self,
        id: &str,
        repo_path: &std::path::Path,
        branch: &str,
    ) -> anyhow::Result<()> {
        if Self::worker_branch_exists(repo_path, branch).await {
            return Ok(());
        }
        let Some(head_commit) =
            super::load_registry_entry_in(&self.scratch, id).and_then(|entry| entry.head_commit)
        else {
            anyhow::bail!(
                "Worker branch {branch} no longer exists; the finished worker {id} cannot be revised"
            );
        };
        let repo = repo_path.to_path_buf();
        let branch_for_git = branch.to_string();
        let head = head_commit.clone();
        let recreated = tokio::task::spawn_blocking(move || {
            crate::worktree::git(&repo, "branch", &["branch", &branch_for_git, &head])
                .map(|out| out.status.success())
                .unwrap_or(false)
        })
        .await
        .unwrap_or(false);
        if !recreated {
            anyhow::bail!(
                "Worker branch {branch} no longer exists and could not be recreated from commit {head_commit}; the finished worker {id} cannot be revised"
            );
        }
        tracing::info!(
            worker = %id,
            branch = %branch,
            commit = %head_commit,
            "Recreated a pruned worker branch from its recorded head"
        );
        Ok(())
    }

    /// Whether `refs/heads/{branch}` exists (the blocking git runs off the
    /// runtime).
    async fn worker_branch_exists(repo_path: &std::path::Path, branch: &str) -> bool {
        let repo = repo_path.to_path_buf();
        let reference = format!("refs/heads/{branch}");
        tokio::task::spawn_blocking(move || {
            crate::worktree::git(
                &repo,
                "show-ref",
                &["show-ref", "--verify", "--quiet", &reference],
            )
            .map(|o| o.status.success())
            .unwrap_or(false)
        })
        .await
        .unwrap_or(false)
    }

    /// Continue a worker with no surviving conversation: same id, same branch,
    /// a fresh conversation built from the registry row.
    ///
    /// The registry row still names the task, the model and the branch, so the
    /// fresh conversation is the system prompt, the original task, a note that
    /// the branch already holds the previous attempt's work, and the steer
    /// message. A branch pruned after a merge is recreated from the head
    /// commit the row recorded.
    async fn cold_continue(
        &self,
        id: &str,
        message: String,
        revision_turns: Option<usize>,
    ) -> anyhow::Result<SteerOutcome> {
        let entry = super::load_registry_entry_in(&self.scratch, id).with_context(|| {
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
        // The branch may have been pruned after a merge; recreate it from the
        // row's recorded head before anything is relaunched.
        self.ensure_worker_branch(id, &repo_path, &branch).await?;

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
            role: entry.role,
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
        let mut history = super::load_worker_history_in(&self.scratch, id)?;
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
        // The branch may have been pruned after a merge; recreate it from the
        // row's recorded head before the record is touched.
        self.ensure_worker_branch(id, &repo_path, &history.branch)
            .await?;

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
            let registry = super::load_registry_entry_in(&self.scratch, id)
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
        let already = super::load_worker_history_log_in(&self.scratch, id)
            .map(|logged| logged.messages.len())
            .unwrap_or(0);
        for message in history.messages.iter().skip(already) {
            if let Err(e) = super::append_history_message_in(&self.scratch, id, &history, message) {
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
        if let Err(e) = save_history_metadata_in(&self.scratch, id, &history) {
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
            role: history.role,
            repo_path: Some(history.repo_path.clone()),
            metrics: super::WorkerMetrics::default(),
            base_branch: history.base_branch.clone(),
            base_commit: Some(history.base_commit.clone()),
            head_commit: None,
            revision,
            auto_continues: history.auto_continues,
            owner: Some(owner.clone()),
            report: None,
            // A revision changes the branch, so the previous review no longer
            // applies: drop any approval this row carried.
            approved: None,
            verified: None,
            integrated: Vec::new(),
            keep_branch: false,
        };

        let pool = self.clone();
        let wid = id.to_string();
        let base_commit = history.base_commit.clone();
        let model_for_fail = history.model.clone();
        let meta = super::WorkerMeta {
            id: wid.clone(),
            task: history.task.clone(),
            group: history.group.clone(),
            role: history.role,
            repo_path: Some(history.repo_path.clone()),
            started_at: now,
            pid: std::process::id(),
            metrics: super::WorkerMetrics::default(),
            revision,
            auto_continues: history.auto_continues,
            owner,
            report: None,
            verified: None,
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
        super::save_registry_entry_in(&self.scratch, &row);
        tracing::info!(worker = %id, revision, max_turns, "Worker revision started");
        Ok(SteerOutcome::Continuing {
            revision,
            cold: false,
        })
    }
}
