//! On-disk worker registry: one JSON file per worker, shared across processes.
//!
//! Registry rows are the only cross-process view of the pool: `list_workers`,
//! the monitor and crash recovery all read them, so a row must be written on
//! every state transition and removed exactly once. The dispatch role persists
//! across steering and cold continuations.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use super::state::{WorkerMetrics, WorkerReport};
use crate::worktree::ScratchRoot;

/// Lifecycle status of a worker, as recorded in the on-disk registry.
///
/// Serialized to lowercase so the on-disk JSON stays byte-identical to the
/// historical stringly-typed rows. An unknown value deserializes to
/// [`RegistryStatus::Stopped`] for forward compatibility: a newer server that
/// writes a status this build does not know must not crash the reader.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RegistryStatus {
    Running,
    Paused,
    Reviewing,
    Completed,
    Failed,
    /// The worker ran out of turns before it completed. Terminal and
    /// continuable: its branch is checkpointed, but it must not be integrated
    /// as a finished contribution.
    Exhausted,
    Stopped,
    /// A hub crash interrupted a live worker. Terminal for listing -- its
    /// uptime is frozen -- but continuable: its branch and its conversation
    /// are intact, so `steer <id> "..."` resumes it on the same id.
    Interrupted,
}

impl RegistryStatus {
    /// Whether the worker has finished and its uptime is frozen.
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Completed | Self::Failed | Self::Exhausted | Self::Stopped | Self::Interrupted
        )
    }

    /// Whether the worker is still live (its uptime keeps counting).
    pub fn is_live(self) -> bool {
        matches!(self, Self::Running | Self::Paused | Self::Reviewing)
    }

    /// The user-visible, title-cased name of the status.
    pub fn display_name(self) -> &'static str {
        match self {
            Self::Running => "Running",
            Self::Paused => "Paused",
            Self::Reviewing => "Reviewing",
            Self::Completed => "Completed",
            Self::Failed => "Failed",
            Self::Exhausted => "Exhausted",
            Self::Stopped => "Stopped",
            Self::Interrupted => "Interrupted",
        }
    }
}

/// Owner label of a row that carries none, i.e. one written before ownership
/// was tracked. Rendered in `list` payloads so the gap is visible instead of
/// showing up as a missing field.
pub const UNATTRIBUTED_OWNER: &str = "unattributed";

/// The owning agent named by `entry`, or [`UNATTRIBUTED_OWNER`].
pub fn registry_owner_label(entry: &WorkerRegistryEntry) -> &str {
    entry.owner.as_deref().unwrap_or(UNATTRIBUTED_OWNER)
}

/// Dispatch authority: consolidators integrate only their owner's group.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WorkerRole {
    #[default]
    Worker,
    Consolidate,
}

/// The orchestrator's verdict on a completed worker.
///
/// Recorded in the worker's registry row so it outlives the in-memory record
/// `collect` evicts; a new revision drops it, because a changed branch needs a
/// fresh review. The batch merge reads it too, and lands the workers in the
/// order the stamps were recorded in.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkerApproval {
    /// Unix time the worker was approved.
    pub at: u64,
    /// Optional note the orchestrator left with the approval.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerRegistryEntry {
    pub id: String,
    pub pid: u32,
    pub task: String,
    pub model: String,
    pub status: RegistryStatus,
    pub step: usize,
    pub max_turns: usize,
    pub last_command: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub question: Option<String>,
    pub started_at: u64,
    pub updated_at: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub group: Option<String>,
    #[serde(default)]
    pub role: WorkerRole,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repo_path: Option<String>,
    /// Agent identity that dispatched the worker. `None` on a row written
    /// before ownership was tracked, which is why it deserializes with a
    /// default instead of failing the read.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub owner: Option<String>,
    /// Per-worker health counters as of this write. `#[serde(default)]` so a
    /// row written by an older build still parses.
    #[serde(default)]
    pub metrics: WorkerMetrics,
    /// Base branch the worker's diff is measured against, detected at dispatch
    /// or at the first continuation. `None` on a row written before base-branch
    /// tracking existed, which is why it deserializes with a default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_branch: Option<String>,
    /// Base commit the worker branched from.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_commit: Option<String>,
    /// Commit `worker-<id>` pointed at when the run finished, recorded so a
    /// continuation can recreate the branch after a merge pruned it (see the
    /// retired grace period). `None` on a row written before head tracking.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub head_commit: Option<String>,
    /// How many revisions this worker has run. `0` on a row written before
    /// revisions were counted.
    #[serde(default)]
    pub revision: usize,
    /// Automatic "the hub restarted" continuations already spent on this
    /// worker, capped at [`super::MAX_AUTO_CONTINUES`].
    #[serde(default)]
    pub auto_continues: usize,
    /// The structured report of the completion turn. Kept on the row, not only
    /// in the in-memory record, because a terminal record is evicted after its
    /// TTL while its row outlives it: `status`, `collect` and `watch` must
    /// still be able to say what the run did.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub report: Option<WorkerReport>,
    /// The orchestrator's approval of the completed worker, or `None` while it
    /// is unreviewed. Persisted so it survives the in-memory eviction.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub approved: Option<WorkerApproval>,
    /// Whether the completion passed its verify gate. Written with the terminal
    /// row and cleared when a revision restarts the worker, so a view built
    /// from the row alone still reports it. `#[serde(default)]` keeps a row
    /// written before the flag was recorded readable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verified: Option<bool>,
    /// Workers whose branches a consolidator merged into its own branch, in
    /// merge order. Recorded on the consolidator's row because that row is the
    /// only durable record of the round it integrated: when the consolidator
    /// itself is merged, every worker it absorbed is fully integrated too and
    /// is retired with it.
    ///
    /// `#[serde(default)]` keeps a row written before consolidators recorded
    /// their round readable; such a consolidator falls back to the sweep, which
    /// proves each worker's branch is merged by itself.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub integrated: Vec<String>,
}

impl WorkerRegistryEntry {
    /// A fully defaulted registry row for tests.
    ///
    /// `id` and `owner` are the two fields nearly every fixture varies; a
    /// caller adjusts the rest with struct update syntax or a setter. The
    /// helper is hidden but public so the integration tests under `tests/` can
    /// build rows too: a new field on the struct then touches this builder
    /// alone instead of every fixture literal.
    #[doc(hidden)]
    pub fn test_row(id: impl Into<String>, owner: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            pid: std::process::id(),
            task: "task".into(),
            model: "test".into(),
            status: RegistryStatus::Running,
            step: 0,
            max_turns: 10,
            last_command: String::new(),
            question: None,
            started_at: 0,
            updated_at: 0,
            group: None,
            role: WorkerRole::Worker,
            repo_path: None,
            owner: Some(owner.into()),
            metrics: WorkerMetrics::default(),
            base_branch: None,
            base_commit: None,
            head_commit: None,
            revision: 0,
            auto_continues: 0,
            report: None,
            approved: None,
            verified: None,
            integrated: Vec::new(),
        }
    }
}

/// The immutable per-worker fields shared by every registry write for a worker.
///
/// Only the status/step/max_turns/last_command/question/updated_at/model vary
/// between writes, so a worker builds this once and reuses it via
/// [`WorkerMeta::save_status`].
pub struct WorkerMeta {
    pub id: String,
    pub task: String,
    pub group: Option<String>,
    pub role: WorkerRole,
    pub repo_path: Option<String>,
    /// Agent identity owning this worker, copied into every row this meta
    /// writes — including the review phase's, which keeps one worker.
    pub owner: String,
    pub started_at: u64,
    pub pid: u32,
    /// How many revisions this worker has run; the history log's metadata line
    /// carries the same counter.
    pub revision: usize,
    /// Automatic "the hub restarted" continuations already spent, so the
    /// daemon's cap survives a restart.
    pub auto_continues: usize,
    /// The phase loop's running counters, written with every status update.
    ///
    /// The loop owns the counters and lends them to the turn engine, which
    /// moves them at the exact point each guard fires; the meta carries them to
    /// disk so a cross-process reader (the monitor, a `status` answered from a
    /// registry row) sees the same numbers as the live record.
    pub metrics: WorkerMetrics,
    /// The completion report, set by the phase loop once the worker has
    /// finished and written with the terminal row.
    pub report: Option<WorkerReport>,
    /// Whether the completion passed its verify gate, written with the
    /// terminal row so the row carries the same verdict as the report.
    pub verified: Option<bool>,
}

impl WorkerMeta {
    /// A fully defaulted meta for tests, the [`WorkerRegistryEntry::test_row`]
    /// counterpart for [`WorkerMeta`].
    #[doc(hidden)]
    pub fn test_meta(id: impl Into<String>, owner: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            task: "task".into(),
            group: None,
            role: WorkerRole::Worker,
            repo_path: None,
            owner: owner.into(),
            started_at: 0,
            pid: std::process::id(),
            revision: 0,
            auto_continues: 0,
            metrics: WorkerMetrics::default(),
            report: None,
            verified: None,
        }
    }

    /// The row this worker's next status update describes.
    ///
    /// Built as a value so the pool's [`RegistryWriter`] can decide whether it
    /// is worth a write at all; `save_status` stays the unconditional path.
    #[allow(clippy::too_many_arguments)]
    pub fn entry(
        &self,
        model: &str,
        status: RegistryStatus,
        step: usize,
        max_turns: usize,
        last_command: &str,
        question: Option<String>,
    ) -> WorkerRegistryEntry {
        WorkerRegistryEntry {
            id: self.id.clone(),
            pid: self.pid,
            task: self.task.clone(),
            model: model.to_string(),
            status,
            step,
            max_turns,
            last_command: last_command.into(),
            question,
            started_at: self.started_at,
            updated_at: super::unix_timestamp(),
            group: self.group.clone(),
            role: self.role,
            repo_path: self.repo_path.clone(),
            owner: Some(self.owner.clone()),
            metrics: self.metrics,
            base_branch: None,
            base_commit: None,
            head_commit: None,
            revision: self.revision,
            auto_continues: self.auto_continues,
            report: self.report.clone(),
            approved: None,
            verified: self.verified,
            integrated: Vec::new(),
        }
    }

    /// Persist one status update for this worker, unconditionally.
    ///
    /// Files the row under the default scratch root; a pool routes its own
    /// writes through [`RegistryWriter`], which carries the pool's root.
    pub fn save_status(
        &self,
        model: &str,
        status: RegistryStatus,
        step: usize,
        max_turns: usize,
        last_command: &str,
        question: Option<String>,
    ) {
        save_registry_entry(&self.entry(model, status, step, max_turns, last_command, question));
    }
}

/// How often one worker's *step-only* registry row may be rewritten.
///
/// A step update carries no lifecycle information — the monitor reads the
/// status, and a step counter that lags by a few seconds changes nothing an
/// operator acts on — while a busy worker would otherwise rewrite its row on
/// every turn. Status transitions are never throttled (see [`RegistryWriter`]).
const STEP_WRITE_INTERVAL: Duration = Duration::from_secs(3);

/// Coalescing front-end to the on-disk registry.
///
/// The registry is the cross-process view of the pool, so it has to stay
/// accurate about *lifecycle*: `running -> paused -> running` and every
/// terminal state are written the moment they happen, because the monitor and
/// crash recovery act on them. Everything else — a step counter moving, a
/// `last_command` label changing — is coalesced to at most one write per
/// worker per [`STEP_WRITE_INTERVAL`].
///
/// The last row written per worker is kept in memory so a kill (which has no
/// `WorkerMeta` at hand, the loop owning it is being aborted) can still end
/// that worker's row on a terminal status. The map is bounded by the number of
/// live workers: entries leave with `collect` and `reap`.
pub struct RegistryWriter {
    rows: HashMap<String, WorkerRegistryEntry>,
    last_write: HashMap<String, Instant>,
    /// The scratch root this writer's rows are filed under.
    root: ScratchRoot,
}

impl Default for RegistryWriter {
    fn default() -> Self {
        Self::new(ScratchRoot::from_env())
    }
}

impl RegistryWriter {
    /// A writer that files every row under `root`.
    pub fn new(root: ScratchRoot) -> Self {
        Self {
            rows: HashMap::new(),
            last_write: HashMap::new(),
            root,
        }
    }

    /// Write `entry`, unless it is a step-only update inside the throttle
    /// window of a row that already says the same thing.
    pub fn save(&mut self, entry: WorkerRegistryEntry) {
        // A consolidator records the round it integrated on its own row, and
        // that list is written by a *different* code path than the status
        // updates. Every status write rebuilds the row from `WorkerMeta`, which
        // knows nothing about the round, so merging the two halves here is what
        // makes the list survive the consolidator's own completion: without it,
        // the very next step would erase the round and the workers it names
        // could never be retired. The writer is the single choke point every
        // registry write passes through, so this is the one place that has to
        // know.
        let mut entry = entry;
        if entry.integrated.is_empty() {
            // The writer's own cache first (no I/O on the common path), then the
            // row on disk, so a round recorded by an earlier process or before
            // this writer started also survives.
            let known = self.rows.get(&entry.id).cloned().or_else(|| {
                super::load_registry_entry_in(&self.root, &entry.id)
            });
            if let Some(known) = known.filter(|known| !known.integrated.is_empty()) {
                entry.integrated = known.integrated;
            }
        }
        let now = Instant::now();
        let transition = self
            .rows
            .get(&entry.id)
            .is_none_or(|last| last.status != entry.status);
        if !transition
            && self
                .last_write
                .get(&entry.id)
                .is_some_and(|at| now.duration_since(*at) < STEP_WRITE_INTERVAL)
        {
            return;
        }
        self.last_write.insert(entry.id.clone(), now);
        self.rows.insert(entry.id.clone(), entry.clone());
        save_registry_entry_in(&self.root, &entry);
    }

    /// The row last written for `worker_id`, if this process wrote one.
    pub fn entry(&self, worker_id: &str) -> Option<&WorkerRegistryEntry> {
        self.rows.get(worker_id)
    }

    /// Forget a worker whose record left the pool, so the map stays bounded.
    pub fn remove(&mut self, worker_id: &str) {
        self.rows.remove(worker_id);
        self.last_write.remove(worker_id);
    }

    /// Forget the last-write timestamp of one row (test support).
    #[doc(hidden)]
    pub fn reset_throttle(&mut self, worker_id: &str) {
        self.last_write.remove(worker_id);
    }
}

/// Whether a consolidator may act on `target`, or the reason it may not.
///
/// The consolidator's whole authority is this check: it may integrate only
/// workers its own owner dispatched into its own group, never itself and never
/// another consolidator (whose branch is an integration, not work). A row with
/// no recorded owner or group is never authority, so a legacy row is refused
/// rather than trusted.
pub fn check_consolidate_delegation(
    actor: &WorkerMeta,
    target: &WorkerRegistryEntry,
) -> Result<(), String> {
    if actor.role != WorkerRole::Consolidate {
        return Err("the caller is not a consolidator".to_string());
    }
    if actor.owner.is_empty() || target.owner.as_deref() != Some(actor.owner.as_str()) {
        return Err("the target belongs to another owner".to_string());
    }
    let group = actor.group.as_deref().unwrap_or("").trim();
    if group.is_empty() || target.group.as_deref() != Some(group) {
        return Err("the target is in another group".to_string());
    }
    if actor.id == target.id {
        return Err("a consolidator cannot integrate itself".to_string());
    }
    if target.role == WorkerRole::Consolidate {
        return Err("the target is itself a consolidator".to_string());
    }
    Ok(())
}

pub fn extract_group(task: &str) -> Option<String> {
    let trimmed = task.trim();
    if trimmed.starts_with('[')
        && let Some(end) = trimmed.find(']')
    {
        let tag = trimmed[1..end].trim();
        if !tag.is_empty() {
            return Some(tag.to_string());
        }
    }
    None
}

pub fn registry_dir() -> PathBuf {
    registry_dir_in(&ScratchRoot::from_env())
}

/// [`registry_dir`] under an explicit scratch root.
pub fn registry_dir_in(root: &ScratchRoot) -> PathBuf {
    root.join("swe-registry")
}

pub fn save_registry_entry(entry: &WorkerRegistryEntry) {
    save_registry_entry_in(&ScratchRoot::from_env(), entry);
}

/// [`save_registry_entry`] under an explicit scratch root.
pub fn save_registry_entry_in(root: &ScratchRoot, entry: &WorkerRegistryEntry) {
    let dir = registry_dir_in(root);
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join(format!("{}.json", entry.id));
    if let Ok(json) = serde_json::to_string(entry) {
        let _ = std::fs::write(path, json);
    }
}

pub fn remove_registry_entry(worker_id: &str) {
    remove_registry_entry_in(&ScratchRoot::from_env(), worker_id);
}

/// [`remove_registry_entry`] under an explicit scratch root.
///
/// Returns whether a row was actually there to remove, so a caller reporting
/// what it reclaimed never claims a deletion that did not happen.
pub fn remove_registry_entry_in(root: &ScratchRoot, worker_id: &str) -> bool {
    let mut removed = false;
    for dir in root.base_dirs() {
        let path = dir.join("swe-registry").join(format!("{worker_id}.json"));
        // A base dir this process never wrote to fails with `NotFound`, which
        // is not a removal; any other failure is likewise not a removal.
        if std::fs::remove_file(path).is_ok() {
            removed = true;
        }
    }
    removed
}

fn worktree_exists_in(root: &ScratchRoot, worker_id: &str) -> bool {
    root.join(format!("swe-wt-{worker_id}")).is_dir()
}

fn branch_exists(
    item: &WorkerRegistryEntry,
    cache: &mut std::collections::HashMap<PathBuf, std::collections::HashSet<String>>,
) -> bool {
    let repo_dir = item
        .repo_path
        .as_ref()
        .map(PathBuf::from)
        .filter(|p| p.is_dir())
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

    let branches = cache.entry(repo_dir.clone()).or_insert_with(|| {
        let mut set = std::collections::HashSet::new();
        if let Ok(output) = crate::worktree::git(
            &repo_dir,
            "for-each-ref",
            &[
                "for-each-ref",
                "--format=%(refname:short)",
                "refs/heads/worker-*",
            ],
        ) && output.status.success()
        {
            for line in String::from_utf8_lossy(&output.stdout).lines() {
                let branch = line.trim();
                if !branch.is_empty() {
                    set.insert(branch.to_string());
                }
            }
        }
        set
    });

    branches.contains(&format!("worker-{}", item.id))
}

/// Read registry rows without Git probes or pruning: the statusLine path.
///
/// One scan via [`raw_registry_entries`] with dead-pid normalisation on top,
/// so the fast view agrees with the dashboard about liveness without ever
/// deleting a row or shelling out to git.
pub fn load_registry_entries_read_only() -> Vec<WorkerRegistryEntry> {
    load_registry_entries_read_only_in(&ScratchRoot::from_env())
}

/// [`load_registry_entries_read_only`] under an explicit scratch root.
pub fn load_registry_entries_read_only_in(root: &ScratchRoot) -> Vec<WorkerRegistryEntry> {
    raw_registry_entries_in(root)
        .into_iter()
        .map(|(_, mut entry)| {
            if entry.status.is_live() && !crate::worktree::is_process_alive(entry.pid) {
                entry.status = RegistryStatus::Stopped;
            }
            entry
        })
        .collect()
}

fn raw_registry_entries_in(root: &ScratchRoot) -> Vec<(PathBuf, WorkerRegistryEntry)> {
    let mut seen = std::collections::HashSet::new();
    root.base_dirs()
        .into_iter()
        .flat_map(|base| {
            std::fs::read_dir(base.join("swe-registry"))
                .into_iter()
                .flatten()
                .flatten()
        })
        .filter_map(move |file| {
            let path = file.path();
            if path.extension()?.to_str()? != "json" {
                return None;
            }
            let entry: WorkerRegistryEntry =
                serde_json::from_slice(&std::fs::read(&path).ok()?).ok()?;
            // IDs are path components, never paths supplied by registry contents.
            if entry.id.is_empty()
                || !entry
                    .id
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
                || path.file_stem()?.to_str()? != entry.id
                || !seen.insert(entry.id.clone())
            {
                return None;
            }
            Some((path, entry))
        })
        .collect()
}

/// Rewrite the dead rows of a crashed hub into interrupted ones before serving.
///
/// Recovery treats any `running`/`paused`/`reviewing` row with a dead pid as
/// an orphan of the previous hub: salvage the checkout onto `worker-<id>` and
/// mark it interrupted with the branch name, keeping branch and history for revision.
/// The checkout is released only after a successful salvage. Rows of this daemon are live work, so they are never orphans.
/// Ids of every registry row the last hub left `interrupted`.
///
/// Read at daemon startup to decide which workers to continue: the row is
/// terminal for listing but its branch and conversation are intact.
/// [`interrupted_registry_entries`] under an explicit scratch root.
pub(crate) fn interrupted_registry_entries_in(root: &ScratchRoot) -> Vec<WorkerRegistryEntry> {
    raw_registry_entries_in(root)
        .into_iter()
        .map(|(_, entry)| entry)
        .filter(|e| e.status == RegistryStatus::Interrupted)
        .collect()
}

pub(crate) fn recover_orphaned_workers() -> usize {
    recover_orphaned_workers_in(&ScratchRoot::from_env())
}

/// [`recover_orphaned_workers`] under an explicit scratch root.
pub(crate) fn recover_orphaned_workers_in(root: &ScratchRoot) -> usize {
    recover_entries_in(root, raw_registry_entries_in(root))
}

fn recover_entries_in(
    root: &ScratchRoot,
    entries: impl IntoIterator<Item = (PathBuf, WorkerRegistryEntry)>,
) -> usize {
    let mut recovered = 0;
    for (path, mut entry) in entries {
        if !entry.status.is_live()
            || entry.pid == std::process::id()
            || crate::worktree::is_process_alive(entry.pid)
        {
            continue;
        }
        // The base dir the worker was created under, not the registry row's own
        // grandparent: the registry may live anywhere, and the target/scratch
        // cleanup below resolves through `swe_base_dir()` the same way teardown
        // does, so the two must agree on where the worktree was.
        let checkout = root.join(format!("swe-wt-{}", entry.id));
        // The orphan's commands may have detached themselves from every process
        // group (`setsid cmd &`, a double fork), so nothing but their working
        // directory still ties them to the worker that is gone. The sweep runs
        // before the salvage: the worktree must still exist to be matched on,
        // and a live process may still be writing into the tree that is about
        // to be committed.
        crate::agent::reap::sweep_worker_processes(
            &entry.id,
            &crate::agent::reap::worker_dirs(&checkout),
        );
        let salvaged =
            !checkout.is_dir() || crate::worktree::prune::salvage_dirty_worktree(&checkout);
        if salvaged && checkout.is_dir() {
            // Release the registration so steer can reattach to the branch.
            // Unlike prune, recovery never retires the branch or history.
            let removed = crate::worktree::git(
                &checkout,
                "worktree remove",
                &["worktree", "remove", "--force", &checkout.to_string_lossy()],
            );
            if !removed.is_ok_and(|out| out.status.success()) {
                tracing::warn!(worker = %entry.id, "Could not release recovered worktree");
            }
        }
        // The orphan's private build dirs and its lease outlive the worktree
        // directory itself, so they need their own cleanup: without it every
        // hub restart leaks one `swe-target-<id>` tree and one stale `.pid`.
        if !checkout.is_dir() {
            crate::worktree::remove_target_dirs_in(root, &checkout);
            let _ = std::fs::remove_file(crate::worktree::pid_file_for(&checkout));
        }
        // Interrupted, not failed: the worker stopped because the hub did, not
        // because it cannot continue. Its branch and conversation survive, so
        // the daemon continues it and the orchestrator can steer it.
        entry.status = RegistryStatus::Interrupted;
        entry.question = None;
        entry.last_command = if salvaged {
            format!("hub restarted; work salvaged on branch worker-{}", entry.id)
        } else {
            format!(
                "hub restarted; salvage failed; work retained in {}",
                checkout.display()
            )
        };
        entry.updated_at = super::unix_timestamp();
        match serde_json::to_vec(&entry)
            .map_err(std::io::Error::other)
            .and_then(|json| std::fs::write(&path, json))
        {
            Ok(()) => recovered += 1,
            Err(error) => {
                tracing::warn!(%error, worker = %entry.id, "Could not record hub recovery")
            }
        }
    }
    recovered
}

pub fn load_all_registry_entries() -> Vec<WorkerRegistryEntry> {
    load_all_registry_entries_in(&ScratchRoot::from_env())
}

/// [`load_all_registry_entries`] under an explicit scratch root.
pub fn load_all_registry_entries_in(root: &ScratchRoot) -> Vec<WorkerRegistryEntry> {
    let mut entries = Vec::new();
    let mut branches_by_repo: std::collections::HashMap<
        PathBuf,
        std::collections::HashSet<String>,
    > = std::collections::HashMap::new();

    // The directory scan, parsing and id dedupe live in [`raw_registry_entries`];
    // this loader only adds liveness normalisation and terminal-row pruning.
    for (path, mut item) in raw_registry_entries_in(root) {
        if item.status.is_live() && !crate::worktree::is_process_alive(item.pid) {
            item.status = RegistryStatus::Stopped;
        }

        if item.status.is_terminal()
            && !worktree_exists_in(root, &item.id)
            && !branch_exists(&item, &mut branches_by_repo)
        {
            let _ = std::fs::remove_file(&path);
            continue;
        }

        entries.push(item);
    }
    entries.sort_by_key(|a| std::cmp::Reverse(a.updated_at));
    entries
}

/// Load a single registry entry by id, normalizing its liveness exactly as
/// [`load_all_registry_entries`] does: a `running`/`paused`/`reviewing` row
/// whose pid is dead is reported as `stopped`.
pub fn load_registry_entry(worker_id: &str) -> Option<WorkerRegistryEntry> {
    load_registry_entry_in(&ScratchRoot::from_env(), worker_id)
}

/// [`load_registry_entry`] under an explicit scratch root.
pub fn load_registry_entry_in(root: &ScratchRoot, worker_id: &str) -> Option<WorkerRegistryEntry> {
    for dir in root.base_dirs() {
        let path = dir.join("swe-registry").join(format!("{worker_id}.json"));
        if let Ok(content) = std::fs::read_to_string(&path)
            && let Ok(mut item) = serde_json::from_str::<WorkerRegistryEntry>(&content)
        {
            if item.status.is_live() && !crate::worktree::is_process_alive(item.pid) {
                item.status = RegistryStatus::Stopped;
            }
            return Some(item);
        }
    }
    None
}

#[cfg(test)]
mod recovery_cleanup_tests {
    use super::*;
    use crate::worktree::pid_file_for;

    /// A pid that is certainly dead: a child that has already exited, so
    /// `is_process_alive` reports it dead and the sweep treats the row as an
    /// orphan rather than a live worker it must not touch.
    fn dead_pid() -> u32 {
        let mut child = std::process::Command::new("true")
            .spawn()
            .expect("spawn a short-lived child");
        let pid = child.id();
        let _ = child.wait();
        pid
    }

    /// A registry row whose `pid` is dead: the shape of a worker orphaned by a
    /// hub crash.
    fn orphan_row(id: &str) -> WorkerRegistryEntry {
        WorkerRegistryEntry {
            pid: dead_pid(),
            task: "orphan".to_string(),
            step: 1,
            last_command: "orphaned".to_string(),
            ..WorkerRegistryEntry::test_row(id, "agent-a")
        }
    }

    /// The sweep releases the worktree registration so `steer` can reattach to
    /// the branch. That alone leaves the worker's private build dirs and its
    /// lease behind, and every hub restart would leak another pair.
    #[test]
    fn recovery_removes_the_orphans_target_dir_and_pid_file() {
        // A private root, so the sweep cannot touch the real registry.
        let base = std::env::temp_dir().join(format!(
            "swe-recovery-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4().simple()
        ));
        let root = ScratchRoot::new(&base);
        let id = format!("recovery-{}", uuid::Uuid::new_v4().simple());
        let worktree = base.join(format!("swe-wt-{id}"));
        let target = base.join(format!("swe-target-swe-wt-{id}"));
        let scratch = base.join(format!("swe-tmp-swe-wt-{id}"));
        let pid = pid_file_for(&worktree);
        std::fs::create_dir_all(&target).unwrap();
        std::fs::create_dir_all(&scratch).unwrap();
        std::fs::write(&pid, serde_json::json!({"pid": 1}).to_string()).unwrap();
        // A private registry file: the sweep is driven directly, so no other
        // process's rows can be recovered by accident.
        let registry = base.join("swe-registry");
        std::fs::create_dir_all(&registry).unwrap();
        let path = registry.join(format!("{id}.json"));
        let row = orphan_row(&id);
        std::fs::write(&path, serde_json::to_vec(&row).unwrap()).unwrap();

        let recovered = recover_entries_in(&root, [(path.clone(), row)]);
        assert_eq!(recovered, 1, "the orphan row must be recovered");
        let row: WorkerRegistryEntry =
            serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        assert_eq!(row.status, RegistryStatus::Interrupted);

        assert!(!target.exists(), "the orphan's target dir must be removed");
        assert!(
            !scratch.exists(),
            "the orphan's scratch dir must be removed"
        );
        assert!(!pid.exists(), "the orphan's lease must be removed");
        let _ = std::fs::remove_dir_all(&base);
    }
}
