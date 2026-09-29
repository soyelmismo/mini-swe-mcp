//! On-disk worker registry: one JSON file per worker, shared across processes.
//!
//! Registry rows are the only cross-process view of the pool: `list_workers`,
//! the monitor and crash recovery all read them, so a row must be written on
//! every state transition and removed exactly once.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerRegistryEntry {
    pub id: String,
    pub pid: u32,
    pub task: String,
    pub model: String,
    pub status: String,
    pub step: usize,
    pub max_turns: usize,
    pub last_command: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub question: Option<String>,
    pub started_at: u64,
    pub updated_at: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub group: Option<String>,
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
    for dir in [registry_dir(), std::env::temp_dir().join("swe-registry")] {
        let path = dir.join(format!("{worker_id}.json"));
        let _ = std::fs::remove_file(path);
    }
}

pub fn load_all_registry_entries() -> Vec<WorkerRegistryEntry> {
    let mut entries = Vec::new();
    let mut seen_ids = std::collections::HashSet::new();

    for dir in [registry_dir(), std::env::temp_dir().join("swe-registry")] {
        if let Ok(read_dir) = std::fs::read_dir(dir) {
            for entry in read_dir.flatten() {
                let p = entry.path();
                if p.extension().and_then(|e| e.to_str()) == Some("json")
                    && let Ok(content) = std::fs::read_to_string(&p)
                    && let Ok(mut item) = serde_json::from_str::<WorkerRegistryEntry>(&content)
                    && seen_ids.insert(item.id.clone())
                {
                    if (item.status == "running" || item.status == "paused")
                        && !crate::worktree::is_process_alive(item.pid)
                    {
                        item.status = "stopped".to_string();
                    }
                    entries.push(item);
                }
            }
        }
    }
    entries.sort_by_key(|a| std::cmp::Reverse(a.updated_at));
    entries
}
