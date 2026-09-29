//! On-disk worker registry: one JSON file per worker, shared across processes.
//!
//! Registry rows are the only cross-process view of the pool: `list_workers`,
//! the monitor and crash recovery all read them, so a row must be written on
//! every state transition and removed exactly once.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

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
}

impl RegistryStatus {
    /// Whether the worker has finished and its uptime is frozen.
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Stopped)
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
        }
    }
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
    pub started_at: u64,
    pub pid: u32,
}

impl WorkerMeta {
    /// Persist one status update for this worker.
    pub fn save_status(
        &self,
        model: &str,
        status: RegistryStatus,
        step: usize,
        max_turns: usize,
        last_command: &str,
        question: Option<String>,
    ) {
        save_registry_entry(&WorkerRegistryEntry {
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
        });
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

pub fn load_all_registry_entries() -> Vec<WorkerRegistryEntry> {
    let mut entries = Vec::new();
    let mut seen_ids = std::collections::HashSet::new();
    let mut branches_by_repo: std::collections::HashMap<PathBuf, std::collections::HashSet<String>> =
        std::collections::HashMap::new();

    for dir in crate::worktree::swe_base_dirs() {
        let dir = dir.join("swe-registry");
        if let Ok(read_dir) = std::fs::read_dir(dir) {
            for entry in read_dir.flatten() {
                let p = entry.path();
                if p.extension().and_then(|e| e.to_str()) == Some("json")
                    && let Ok(content) = std::fs::read_to_string(&p)
                    && let Ok(mut item) = serde_json::from_str::<WorkerRegistryEntry>(&content)
                    && seen_ids.insert(item.id.clone())
                {
                    if item.status.is_live() && !crate::worktree::is_process_alive(item.pid) {
                        item.status = RegistryStatus::Stopped;
                    }

                    if item.status.is_terminal()
                        && !worktree_exists(&item.id)
                        && !branch_exists(&item, &mut branches_by_repo)
                    {
                        let _ = std::fs::remove_file(&p);
                        continue;
                    }

                    entries.push(item);
                }
            }
        }
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
