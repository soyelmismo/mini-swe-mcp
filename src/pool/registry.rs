//! On-disk worker registry: one JSON file per worker, shared across processes.
//!
//! Registry rows are the only cross-process view of the pool: `list_workers`,
//! the monitor and crash recovery all read them, so a row must be written on
//! every state transition and removed exactly once.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use super::state::WorkerMetrics;

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
            Self::Completed | Self::Failed | Self::Stopped | Self::Interrupted
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
    /// How many revisions this worker has run. `0` on a row written before
    /// revisions were counted.
    #[serde(default)]
    pub revision: usize,
    /// Automatic "the hub restarted" continuations already spent on this
    /// worker, capped at [`super::MAX_AUTO_CONTINUES`].
    #[serde(default)]
    pub auto_continues: usize,
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
}

impl WorkerMeta {
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
            repo_path: self.repo_path.clone(),
            owner: Some(self.owner.clone()),
            metrics: self.metrics,
            base_branch: None,
            base_commit: None,
            revision: self.revision,
            auto_continues: self.auto_continues,
        }
    }

    /// Persist one status update for this worker, unconditionally.
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
#[derive(Default)]
pub struct RegistryWriter {
    rows: HashMap<String, WorkerRegistryEntry>,
    last_write: HashMap<String, Instant>,
}

impl RegistryWriter {
    /// Write `entry`, unless it is a step-only update inside the throttle
    /// window of a row that already says the same thing.
    pub fn save(&mut self, entry: WorkerRegistryEntry) {
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
        save_registry_entry(&entry);
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
    crate::worktree::swe_base_dir().join("swe-registry")
}

pub fn save_registry_entry(entry: &WorkerRegistryEntry) {
    let dir = registry_dir();
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join(format!("{}.json", entry.id));
    if let Ok(json) = serde_json::to_string(entry) {
        let _ = std::fs::write(path, json);
    }
}

pub fn remove_registry_entry(worker_id: &str) {
    for dir in crate::worktree::swe_base_dirs() {
        let path = dir.join("swe-registry").join(format!("{worker_id}.json"));
        let _ = std::fs::remove_file(path);
    }
}

fn worktree_exists(worker_id: &str) -> bool {
    for base in crate::worktree::swe_base_dirs() {
        if base.join(format!("swe-wt-{worker_id}")).is_dir() {
            return true;
        }
    }
    false
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
            &["for-each-ref", "--format=%(refname:short)", "refs/heads/worker-*"],
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
    raw_registry_entries()
        .map(|(_, mut entry)| {
            if entry.status.is_live() && !crate::worktree::is_process_alive(entry.pid) {
                entry.status = RegistryStatus::Stopped;
            }
            entry
        })
        .collect()
}

fn raw_registry_entries() -> impl Iterator<Item = (PathBuf, WorkerRegistryEntry)> {
    let mut seen = std::collections::HashSet::new();
    crate::worktree::swe_base_dirs()
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
}

/// Rewrite the dead rows of a crashed hub into interrupted ones before serving.
///
/// Recovery treats any `running`/`paused`/`reviewing` row with a dead pid as
/// an orphan of the previous hub: salvage the checkout onto `worker-<id>` and
/// mark it interrupted with the branch name, keeping branch and history for revision.
/// The checkout is released only after a successful salvage. Rows of this daemon are live work, so they are never orphans.
pub(crate) fn recover_orphaned_workers() -> usize {
    let mut recovered = 0;
    for (path, mut entry) in raw_registry_entries() {
        if !entry.status.is_live()
            || entry.pid == std::process::id()
            || crate::worktree::is_process_alive(entry.pid)
        {
            continue;
        }
        let base = path
            .parent()
            .and_then(|dir| dir.parent())
            .expect("registry base");
        let checkout = base.join(format!("swe-wt-{}", entry.id));
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
    let mut entries = Vec::new();
    let mut branches_by_repo: std::collections::HashMap<PathBuf, std::collections::HashSet<String>> =
        std::collections::HashMap::new();

    // The directory scan, parsing and id dedupe live in [`raw_registry_entries`];
    // this loader only adds liveness normalisation and terminal-row pruning.
    for (path, mut item) in raw_registry_entries() {
        if item.status.is_live() && !crate::worktree::is_process_alive(item.pid) {
            item.status = RegistryStatus::Stopped;
        }

        if item.status.is_terminal()
            && !worktree_exists(&item.id)
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
    for dir in crate::worktree::swe_base_dirs() {
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
