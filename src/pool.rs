use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::{RwLock, Semaphore};
use tokio::task::JoinHandle;
use tracing::{error, info};

use crate::agent::{AgentRunner, AgentStepLog, ChatMessage, Role, SYSTEM_PROMPT};
use crate::worktree::WorktreeGuard;

#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "state", content = "details")]
pub enum WorkerState {
    Running {
        step: usize,
        last_command: String,
        started_at: u64,
    },
    Paused {
        question: String,
        step: usize,
        paused_at: u64,
    },
    Completed {
        turns: usize,
        diff: String,
        summary: String,
        completed_at: u64,
        #[serde(default)]
        artifacts: Vec<String>,
        #[serde(default)]
        branch: Option<String>,
    },
    Failed {
        error: String,
        step: usize,
        failed_at: u64,
    },
}

impl WorkerState {
    /// The worker loop turn this state corresponds to, for every variant.
    pub fn step(&self) -> usize {
        match self {
            WorkerState::Running { step, .. } | WorkerState::Paused { step, .. } => *step,
            WorkerState::Completed { turns, .. } => *turns,
            WorkerState::Failed { step, .. } => *step,
        }
    }

    pub fn to_summary(&self) -> serde_json::Value {
        match self {
            WorkerState::Running { step, last_command, started_at } => serde_json::json!({
                "status": "Running",
                "step": step,
                "last_command": last_command,
                "started_at": started_at,
            }),
            WorkerState::Paused { question, step, paused_at } => serde_json::json!({
                "status": "Paused",
                "step": step,
                "question": question,
                "paused_at": paused_at,
            }),
            WorkerState::Completed { turns, summary, completed_at, artifacts, branch, .. } => serde_json::json!({
                "status": "Completed",
                "turns": turns,
                "summary": summary,
                "completed_at": completed_at,
                "artifacts": artifacts,
                "branch": branch,
            }),
            WorkerState::Failed { error, step, failed_at } => serde_json::json!({
                "status": "Failed",
                "step": step,
                "error": error,
                "failed_at": failed_at,
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// Step-log retention policy (audit 07 — R1/R2/F1/F2/F6)
// ---------------------------------------------------------------------------

/// Default number of step-log entries retained **per worker**.
///
/// The buffer is a sliding window, so memory per worker is O(1) in the number
/// of turns instead of O(n).
pub const DEFAULT_MAX_RETAINED_LOGS: usize = 200;
/// Hard ceiling for [`DEFAULT_MAX_RETAINED_LOGS`]; larger `WORKER_MAX_RETAINED_LOGS`
/// values are clamped to this so a misconfiguration cannot reintroduce the
/// unbounded growth the audit flagged.
pub const MAX_RETAINED_LOGS_CEILING: usize = 1000;

/// Default number of step-log entries inlined into a single MCP response.
pub const DEFAULT_MAX_EMITTED_LOGS: usize = 40;
/// Hard ceiling for [`DEFAULT_MAX_EMITTED_LOGS`].
pub const MAX_EMITTED_LOGS_CEILING: usize = 500;

/// Default age (seconds) after which a `Completed`/`Failed` worker record is
/// evicted from the pool.
pub const DEFAULT_TERMINAL_TTL_SECS: u64 = 300;

/// Hard cap on the `output` field of a retained [`AgentStepLog`].
///
/// Unlike the previous implementation the truncation marker is *charged
/// against* this budget, so the stored value is `<= MAX_LOG_OUTPUT_BYTES`
/// rather than "budget + marker" (audit 07, F6).
pub const MAX_LOG_OUTPUT_BYTES: usize = 2048;

/// Hard cap on the `command` field of a retained [`AgentStepLog`].
pub const MAX_LOG_COMMAND_BYTES: usize = 64;

/// Worst-case charged cost of one retained entry: the two text fields plus the
/// inline `AgentStepLog` struct. Used to derive the per-worker byte budget from
/// the entry-count window.
pub fn worst_case_entry_bytes() -> usize {
    std::mem::size_of::<AgentStepLog>() + MAX_LOG_COMMAND_BYTES + MAX_LOG_OUTPUT_BYTES
}

/// How a worker's step-log history is retained and how much of it is emitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LogRetentionPolicy {
    /// Maximum number of entries kept in the window.
    pub max_retained: usize,
    /// Total byte budget for the retained payloads (both fields combined).
    pub max_bytes: usize,
    /// Maximum number of entries inlined into one MCP response.
    pub max_emitted: usize,
}

impl Default for LogRetentionPolicy {
    fn default() -> Self {
        Self {
            max_retained: DEFAULT_MAX_RETAINED_LOGS,
            max_bytes: DEFAULT_MAX_RETAINED_LOGS * worst_case_entry_bytes(),
            max_emitted: DEFAULT_MAX_EMITTED_LOGS,
        }
    }
}

impl LogRetentionPolicy {
    /// Build a policy from the environment, clamping every value to its ceiling
    /// and falling back to the defaults for zero / non-numeric input.
    pub fn from_env() -> Self {
        let default = Self::default();
        let max_retained = std::env::var("WORKER_MAX_RETAINED_LOGS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&v| v > 0)
            .map(|v| v.min(MAX_RETAINED_LOGS_CEILING))
            .unwrap_or(default.max_retained);
        let max_emitted = std::env::var("WORKER_MAX_EMITTED_LOGS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&v| v > 0)
            .map(|v| v.min(MAX_EMITTED_LOGS_CEILING))
            .unwrap_or(default.max_emitted);
        Self {
            max_retained,
            max_bytes: max_retained * worst_case_entry_bytes(),
            max_emitted,
        }
    }
}

/// A bounded, append-only-with-eviction step-log history.
///
/// # Invariants
///
/// * `entries.len() <= policy.max_retained` and `bytes <= policy.max_bytes`
///   hold after every [`LogBuffer::push`].
/// * Entries are evicted strictly oldest-first, so the retained window is the
///   *tail* of the worker's history.
/// * [`LogBuffer::dropped`] counts every entry evicted since the buffer was
///   created, which is what makes the degradation observable (audit 07, R7).
#[derive(Debug, Clone)]
pub struct LogBuffer {
    entries: VecDeque<AgentStepLog>,
    bytes: usize,
    dropped: usize,
    max_retained: usize,
    max_bytes: usize,
}

impl LogBuffer {
    /// A buffer with the default policy, pre-allocating the exact window so the
    /// `Vec` never over-allocates (audit 07, R2).
    pub fn new() -> Self {
        Self::with_policy(LogRetentionPolicy::default())
    }

    pub fn with_policy(policy: LogRetentionPolicy) -> Self {
        let max_retained = policy.max_retained.max(1);
        Self {
            entries: VecDeque::with_capacity(max_retained.min(MAX_RETAINED_LOGS_CEILING)),
            bytes: 0,
            dropped: 0,
            max_retained,
            max_bytes: policy.max_bytes.max(1),
        }
    }

    /// Append an entry, evicting the oldest ones until both the entry-count and
    /// byte budgets are satisfied again.
    pub fn push(&mut self, entry: AgentStepLog) {
        self.bytes += entry_size(&entry);
        self.entries.push_back(entry);
        self.evict_until_within_budget();
    }

    /// Drop oldest entries while the window exceeds its count budget, the byte
    /// budget, or both. A single entry larger than the byte budget is still
    /// evicted: the window must never keep data it cannot account for.
    fn evict_until_within_budget(&mut self) {
        while !self.entries.is_empty()
            && (self.entries.len() > self.max_retained || self.bytes > self.max_bytes)
        {
            if let Some(oldest) = self.entries.pop_front() {
                self.bytes = self.bytes.saturating_sub(entry_size(&oldest));
                self.dropped = self.dropped.saturating_add(1);
            }
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Number of entries evicted because they fell out of the window.
    pub fn dropped(&self) -> usize {
        self.dropped
    }

    /// Number of entries currently held in memory.
    pub fn retained(&self) -> usize {
        self.entries.len()
    }

    /// Total number of steps ever logged, retained plus dropped.
    pub fn total(&self) -> usize {
        self.entries.len() + self.dropped
    }

    /// Bytes currently charged against the retention budget.
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// Retained entries that a `max_emitted` budget would leave out.
    pub fn logs_omitted(&self, max_emitted: usize) -> usize {
        self.entries.len().saturating_sub(max_emitted)
    }

    /// Oldest retained entry, if any.
    pub fn front(&self) -> Option<&AgentStepLog> {
        self.entries.front()
    }

    /// Newest retained entry, if any.
    pub fn back(&self) -> Option<&AgentStepLog> {
        self.entries.back()
    }

    /// Iterate the retained window oldest-first.
    pub fn iter(&self) -> impl Iterator<Item = &AgentStepLog> {
        self.entries.iter()
    }

    /// Borrow the newest `limit` entries (the tail of the window), oldest-first.
    pub fn tail(&self, limit: usize) -> Vec<&AgentStepLog> {
        let skip = self.entries.len().saturating_sub(limit);
        self.entries.iter().skip(skip).collect()
    }

    /// Drop every retained entry, keeping the `dropped` counter.
    pub fn clear(&mut self) {
        self.bytes = 0;
        self.entries.clear();
    }
}

impl Default for LogBuffer {
    fn default() -> Self {
        Self::new()
    }
}

/// `LogBuffer` serializes as the plain retained array, so callers can embed a
/// snapshot where a `Vec<AgentStepLog>` used to be without reshaping the payload.
/// The eviction counters travel separately via [`LogStats`].
impl Serialize for LogBuffer {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.entries.serialize(serializer)
    }
}

/// Bytes charged for one entry: payload plus the inline `AgentStepLog` struct.
fn entry_size(entry: &AgentStepLog) -> usize {
    std::mem::size_of::<AgentStepLog>() + entry.command.len() + entry.output.len()
}

/// Truncate `value` so the *whole result* — truncation marker included — is at
/// most `budget` bytes, never splitting a UTF-8 code point (audit 07, F6).
///
/// The returned string is always valid UTF-8; when the marker alone would not
/// fit inside the budget the result degrades to an empty string rather than
/// exceeding the ceiling.
pub fn clamp_string(value: &str, budget: usize) -> String {
    if value.len() <= budget {
        return value.to_string();
    }

    // The marker length depends on the number of dropped bytes, so reserve room
    // for the widest plausible marker first and shrink the head until it fits.
    let mut dropped = value.len();
    loop {
        let marker = truncation_marker(dropped);
        if marker.len() >= budget {
            return String::new();
        }
        let head_budget = budget - marker.len();
        let cut = value.floor_char_boundary(head_budget);
        let out = format!("{}{}", &value[..cut], marker);
        if out.len() <= budget {
            return out;
        }
        // `cut` moved past a code point start; recompute with the real drop count.
        dropped = value.len() - cut;
    }
}

/// `... [N bytes truncated]` — the marker appended by [`clamp_string`].
fn truncation_marker(dropped: usize) -> String {
    format!("... [{dropped} bytes truncated]")
}

/// Build a bounded [`AgentStepLog`] entry: both text fields are clamped so the
/// retained payload is strictly bounded (audit 07, F6).
pub fn build_step_log(
    step: usize,
    command: &str,
    output: String,
    exit_code: Option<i32>,
) -> AgentStepLog {
    AgentStepLog {
        step,
        command: clamp_string(command, MAX_LOG_COMMAND_BYTES),
        output: clamp_string(&output, MAX_LOG_OUTPUT_BYTES),
        exit_code,
    }
}

/// Observability counters describing how much of a worker's step history is
/// actually visible (audit 07, R7).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize)]
pub struct LogStats {
    /// Total steps the worker executed.
    pub total_steps: usize,
    /// Entries currently held in memory.
    pub logs_retained: usize,
    /// Entries evicted by the retention window.
    pub logs_dropped: usize,
}

/// A rendered, degradation-aware view of a worker's step history.
#[derive(Debug, Clone, serde::Serialize)]
pub struct EmittedLogs {
    /// The emitted tail, oldest-first.
    pub logs: Vec<AgentStepLog>,
    /// How many retained entries were **not** emitted because of the emission
    /// budget.
    pub logs_omitted: usize,
    /// Present only when something is missing, so a consumer can never mistake
    /// the emitted tail for the full history.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logs_truncation_notice: Option<String>,
}

/// Render the tail of a log buffer for one response, bounded by
/// `WORKER_MAX_EMITTED_LOGS` (audit 07, R4).
pub fn emit_view(buffer: &LogBuffer, max_emitted: usize) -> EmittedLogs {
    emit_view_with(&buffer.tail(max_emitted), buffer.dropped(), buffer.len())
}

/// Assemble an [`EmittedLogs`] from a tail that is already materialised.
pub fn emit_view_with(tail: &[&AgentStepLog], dropped: usize, retained: usize) -> EmittedLogs {
    let omitted = retained.saturating_sub(tail.len());
    let mut notice_parts: Vec<String> = Vec::new();
    if dropped > 0 {
        notice_parts.push(format!(
            "{dropped} earlier log(s) evicted by the retention window"
        ));
    }
    if omitted > 0 {
        notice_parts.push(format!(
            "{omitted} retained log(s) omitted by the emission budget"
        ));
    }
    let logs_truncation_notice = if notice_parts.is_empty() {
        None
    } else {
        Some(format!(
            "{} (use `logs <worker_id>` for the full window)",
            notice_parts.join("; ")
        ))
    };

    EmittedLogs {
        logs: tail.iter().map(|e| (*e).clone()).collect(),
        logs_omitted: omitted,
        logs_truncation_notice,
    }
}

pub struct WorkerRecord {
    pub id: String,
    pub task: String,
    pub model: String,
    pub state: WorkerState,
    /// Bounded sliding window of step logs (audit 07, R1).
    pub logs: LogBuffer,
    pub pending_steer: Vec<String>,
    pub resume_tx: Option<tokio::sync::mpsc::Sender<String>>,
    pub handle: Option<JoinHandle<()>>,
}

impl WorkerRecord {
    fn fail(&mut self, error: impl Into<String>) {
        self.state = WorkerState::Failed {
            error: error.into(),
            step: self.state.step(),
            failed_at: unix_timestamp(),
        };
    }

    /// Unix timestamp at which this record became terminal, if it is terminal.
    pub fn terminal_at(&self) -> Option<u64> {
        match &self.state {
            WorkerState::Completed { completed_at, .. } => Some(*completed_at),
            WorkerState::Failed { failed_at, .. } => Some(*failed_at),
            WorkerState::Running { .. } | WorkerState::Paused { .. } => None,
        }
    }

    /// Step counters for the observability surface (audit 07, R7).
    pub fn log_stats(&self) -> LogStats {
        LogStats {
            total_steps: self.state.step().max(self.logs.total()),
            logs_retained: self.logs.retained(),
            logs_dropped: self.logs.dropped(),
        }
    }
}

/// Coarse lifecycle stage of a worker, used by progress polls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerPhase {
    Running,
    Paused,
    Completed,
    Failed,
}

/// Lightweight, allocation-cheap snapshot of a worker's progress.
///
/// Deliberately excludes the terminal payload (`diff`, `summary`,
/// `artifacts`) so that the 500 ms polling loops neither clone nor serialize
/// potentially multi-megabyte strings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerProgress {
    pub phase: WorkerPhase,
    /// Turn currently running (or the turn the worker paused at); for terminal
    /// phases this is the number of turns performed.
    pub step: usize,
    /// Last bash command summary while running.
    pub last_command: Option<String>,
    /// Escalated question while paused.
    pub question: Option<String>,
    /// `true` once the worker reached `Completed` or `Failed` and its full
    /// payload can be fetched once with `get_worker_state`.
    pub terminal: bool,
}

/// Result of a one-shot worker collection, detached from the live pool.
#[derive(Debug, Clone, serde::Serialize)]
pub struct CollectedWorker {
    pub id: String,
    pub task: String,
    pub model: String,
    pub state: WorkerState,
    /// The retained window, moved out of the pool (not a copy).
    pub logs: Vec<AgentStepLog>,
    /// Retained entries that were not part of `logs` because of the emission
    /// budget.
    pub logs_omitted: usize,
    /// Retained entries already evicted by the retention window.
    pub logs_dropped: usize,
    /// Human-readable explanation when the history is degraded.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logs_truncation_notice: Option<String>,
}

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

pub struct WorkerLaunchConfig {
    pub task: String,
    pub model: String,
    pub temperature: Option<f32>,
    pub repo_path: PathBuf,
    pub max_turns: usize,
    pub group: String,
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

#[derive(Clone)]
pub struct WorkerPool {
    semaphore: Arc<Semaphore>,
    bash_semaphore: Arc<Semaphore>,
    build_semaphore: Arc<Semaphore>,
    workers: Arc<RwLock<HashMap<String, WorkerRecord>>>,
    api_base: String,
    api_key: String,
    log_policy: LogRetentionPolicy,
    terminal_ttl: Duration,
}

impl WorkerPool {
    pub fn new(max_concurrent: usize, api_base: String, api_key: String) -> Self {
        let default_build_slots = std::thread::available_parallelism()
            .map(|n| (n.get() / 2).max(1))
            .unwrap_or(2);
        let build_slots = std::env::var("BASH_BUILD_LIMIT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default_build_slots);

        let default_bash_slots = max_concurrent.max(8);
        let bash_slots = std::env::var("BASH_CONCURRENT_LIMIT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default_bash_slots);

        let log_policy = LogRetentionPolicy::from_env();
        let terminal_ttl = Duration::from_secs(
            std::env::var("WORKER_TERMINAL_TTL_SECS")
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
                .filter(|&v| v > 0)
                .unwrap_or(DEFAULT_TERMINAL_TTL_SECS),
        );

        info!(
            bash_slots,
            build_slots,
            max_retained_logs = log_policy.max_retained,
            max_emitted_logs = log_policy.max_emitted,
            terminal_ttl_secs = terminal_ttl.as_secs(),
            "Bash and build execution semaphores initialized"
        );
        Self {
            semaphore: Arc::new(Semaphore::new(max_concurrent)),
            bash_semaphore: Arc::new(Semaphore::new(bash_slots)),
            build_semaphore: Arc::new(Semaphore::new(build_slots)),
            workers: Arc::new(RwLock::new(HashMap::new())),
            api_base,
            api_key,
            log_policy,
            terminal_ttl,
        }
    }

    /// The active step-log retention policy.
    pub fn log_policy(&self) -> LogRetentionPolicy {
        self.log_policy
    }

    /// Evict terminal worker records whose TTL expired, plus their registry
    /// entries. Fresh terminal records are deliberately kept so a subsequent
    /// `collect` / `wait: true` still finds them (audit 07, R3).
    pub async fn reap(&self) -> Vec<String> {
        let mut lock = self.workers.write().await;
        let ttl = self.terminal_ttl.as_secs();
        let expired = expired_terminal_ids(&lock, ttl);
        for id in &expired {
            lock.remove(id);
            remove_registry_entry(id);
        }
        if !expired.is_empty() {
            tracing::info!(
                count = expired.len(),
                ttl_secs = ttl,
                "Reaped terminal worker records"
            );
        }
        expired
    }

    /// Lazily reap terminal records so a long-running server that never calls
    /// [`WorkerPool::reap`] explicitly still bounds its residency.
    async fn prune_terminal_records_locked(
        &self,
        lock: &mut HashMap<String, WorkerRecord>,
    ) -> usize {
        let ttl = self.terminal_ttl.as_secs();
        let expired = expired_terminal_ids(lock, ttl);
        for id in &expired {
            lock.remove(id);
            remove_registry_entry(id);
        }
        if !expired.is_empty() {
            tracing::info!(
                count = expired.len(),
                ttl_secs = ttl,
                "Pruned expired terminal worker records on dispatch"
            );
        }
        expired.len()
    }

    pub async fn dispatch(
        &self,
        task: String,
        model: String,
        temperature: Option<f32>,
        repo_path: PathBuf,
        max_turns: usize,
        group: Option<String>,
    ) -> Result<String> {
        // F6: format the low 32 UUID bits directly instead of building (and
        // immediately discarding) a full hyphenated `String` per worker.
        let worker_id = format!("{:08x}", uuid::Uuid::new_v4().as_u128() as u32);
        let now = unix_timestamp();
        let resolved_group = group
            .or_else(|| extract_group(&task))
            .unwrap_or_else(|| "default".to_string());

        // H-6: the record keeps its own `String`s, so the original `task`/
        // `model` are cloned exactly once here and then *moved* into the launch
        // config instead of being cloned again below.
        let initial_record = WorkerRecord {
            id: worker_id.clone(),
            task: task.clone(),
            model: model.clone(),
            state: WorkerState::Running {
                step: 0,
                last_command: String::from("initializing"),
                started_at: now,
            },
            // Pre-size the retention window so the log buffer never
            // over-allocates (audit 07, R2).
            logs: LogBuffer::with_policy(self.log_policy),
            pending_steer: Vec::new(),
            resume_tx: None,
            handle: None,
        };

        save_registry_entry(&WorkerRegistryEntry {
            id: worker_id.clone(),
            pid: std::process::id(),
            task: task.clone(),
            model: model.clone(),
            status: "running".into(),
            step: 0,
            max_turns,
            last_command: "initializing".into(),
            question: None,
            started_at: now,
            updated_at: now,
            group: Some(resolved_group.clone()),
        });

        {
            // Prune stale terminal records *before* inserting, so a long-lived
            // server bounds residency even without the background reaper
            // (audit 07, R3).
            let mut lock = self.workers.write().await;
            self.prune_terminal_records_locked(&mut lock).await;
            lock.insert(worker_id.clone(), initial_record);
        }

        let pool = self.clone();
        let wid = worker_id.clone();
        let task_clone = task.clone();
        let model_clone = model.clone();
        let group_clone = resolved_group.clone();

        let config = WorkerLaunchConfig {
            task,
            model,
            temperature,
            repo_path,
            max_turns,
            group: resolved_group,
        };

        let join_handle = tokio::spawn(async move {
            if let Err(e) = pool.run_worker(wid.clone(), config).await {
                error!(worker = %wid, error = %e, "Worker failed with error");
                let mut lock = pool.workers.write().await;
                if let Some(w) = lock.get_mut(&wid) {
                    w.fail(e.to_string());
                }
                save_registry_entry(&WorkerRegistryEntry {
                    id: wid.clone(),
                    pid: std::process::id(),
                    task: task_clone,
                    model: model_clone,
                    status: "failed".into(),
                    step: 0,
                    max_turns,
                    last_command: format!("error: {e}"),
                    question: None,
                    started_at: now,
                    updated_at: unix_timestamp(),
                    group: Some(group_clone),
                });
            }
        });

        // Store handle for potential cancellation. The record was inserted
        // without a handle just above so the worker is visible immediately;
        // this second (short) write-guard attaches the handle as soon as the
        // task exists, minimising the window in which a concurrent `kill`/
        // `kill_all` could not yet abort the task.
        {
            let mut lock = self.workers.write().await;
            if let Some(w) = lock.get_mut(&worker_id) {
                w.handle = Some(join_handle);
            }
        }

        Ok(worker_id)
    }

    async fn run_worker(&self, worker_id: String, config: WorkerLaunchConfig) -> Result<()> {
        let WorkerLaunchConfig {
            task,
            model,
            temperature,
            repo_path,
            max_turns,
            group,
        } = config;

        let _permit = self.semaphore.acquire().await.context("Semaphore closed")?;
        info!(worker = %worker_id, model = %model, "Starting worker execution");

        let mut worktree = WorktreeGuard::new(&repo_path, &worker_id)?;
        let runner = AgentRunner::new(
            self.api_base.clone(),
            self.api_key.clone(),
            model.clone(),
            temperature,
        );

        let mut messages = vec![
            ChatMessage::text(Role::System, SYSTEM_PROMPT),
            ChatMessage::text(Role::User, format!("TASK:\n{}\n\nBegin by exploring the repository.", task)),
        ];

        let mut step = 0;
        let mut current_max_turns = max_turns;
        let mut consecutive_no_cmd = 0;
        let mut last_assistant_text = String::new();
        let started_at_ts = unix_timestamp();

        while step < current_max_turns {
            step += 1;

            // Inject any steering instructions queued by the orchestrator
            let steer_msgs: Vec<String> = {
                let mut lock = self.workers.write().await;
                if let Some(w) = lock.get_mut(&worker_id) {
                    std::mem::take(&mut w.pending_steer)
                } else {
                    Vec::new()
                }
            };

            for msg in steer_msgs {
                info!(worker = %worker_id, "Injected steering message into subagent turn");
                messages.push(ChatMessage::text(
                    Role::User,
                    format!("STEER / ORCHESTRATOR GUIDANCE:\n{}", msg),
                ));
            }

            // Proactive turn warning when approaching limit (at 5 and 2 turns remaining)
            let remaining = current_max_turns.saturating_sub(step);
            if remaining == 5 || remaining == 2 {
                info!(worker = %worker_id, step, current_max_turns, "Injecting proactive turn limit warning");
                messages.push(ChatMessage::text(
                    Role::User,
                    format!(
                        "TURN LIMIT WARNING: You have used {} of {} turns ({} remaining). If you need more turns to complete testing or refactoring, execute `echo \"REQUEST_TURNS: <number>\"` now. Otherwise, wrap up your changes and execute `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.",
                        step, current_max_turns, remaining
                    ),
                ));
            }

            // 1. Run LLM step with silent retry for empty / no-command responses
            let mut llm_resp = runner.run_step_llm(&messages).await?;

            if llm_resp.command.is_none() {
                info!(
                    worker = %worker_id,
                    "No command found (tool_calls or code block); discarding and silently retrying once without warning"
                );
                let retry_resp = runner.run_step_llm(&messages).await?;
                if retry_resp.command.is_some() {
                    llm_resp = retry_resp;
                }
            }

            let cmd_str = match llm_resp.command {
                Some(ref cmd) if cmd.contains("COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT") => {
                    info!(worker = %worker_id, step = step, "Worker requested completion");
                    if !llm_resp.content.trim().is_empty() {
                        last_assistant_text = llm_resp.content.clone();
                    }
                    break;
                }
                Some(ref cmd) => {
                    consecutive_no_cmd = 0;
                    if !llm_resp.content.trim().is_empty() {
                        last_assistant_text = llm_resp.content.clone();
                    }
                    cmd.clone()
                }
                None => {
                    info!(worker = %worker_id, step = step, "No bash command in response; prompting subagent directly");
                    messages.push(ChatMessage::text(
                        Role::Assistant,
                        if llm_resp.content.trim().is_empty() {
                            "I will execute a bash command.".into()
                        } else {
                            llm_resp.content
                        },
                    ));
                    messages.push(ChatMessage::text(
                        Role::User,
                        "ERROR: No bash command found. You MUST call the `bash` tool with your command.",
                    ));
                    if consecutive_no_cmd < 2 {
                        consecutive_no_cmd += 1;
                        step = step.saturating_sub(1);
                    }
                    continue;
                }
            };

            let cmd_summary = summarize_command(&cmd_str);

            // Update running state
            {
                let mut lock = self.workers.write().await;
                if let Some(w) = lock.get_mut(&worker_id)
                    && let WorkerState::Running {
                        step: ref mut s,
                        ref mut last_command,
                        ..
                    } = w.state
                {
                    *s = step;
                    *last_command = cmd_summary.clone();
                }
            }

            save_registry_entry(&WorkerRegistryEntry {
                id: worker_id.clone(),
                pid: std::process::id(),
                task: task.clone(),
                model: model.clone(),
                status: "running".into(),
                step,
                max_turns: current_max_turns,
                last_command: cmd_summary.clone(),
                question: None,
                started_at: started_at_ts,
                updated_at: unix_timestamp(),
                group: Some(group.clone()),
            });

            info!(worker = %worker_id, step = step, op = %cmd_summary, "Subagent step");

            let (output, code) = {
                let is_heavy = crate::agent::is_heavy_command(&cmd_str);
                let _build_permit = if is_heavy {
                    Some(
                        self.build_semaphore
                            .acquire()
                            .await
                            .context("Build semaphore closed")?,
                    )
                } else {
                    None
                };
                let _bash_permit = self
                    .bash_semaphore
                    .acquire()
                    .await
                    .context("Bash semaphore closed")?;
                runner.execute_bash(&worktree.path, &cmd_str).await?
            };

            // 2. Check for REQUEST_TURNS sentinel in the command itself
            if let Some(additional) = parse_request_turns(&cmd_str) {
                let old_max = current_max_turns;
                current_max_turns = (current_max_turns + additional).max(current_max_turns).min(500);
                info!(
                    worker = %worker_id,
                    requested = additional,
                    old_max,
                    new_max = current_max_turns,
                    "Subagent requested turn extension; granted"
                );
            }

            // 3. Check for ASK_ORCHESTRATOR sentinel in the command itself
            if let Some(question) = parse_ask_orchestrator(&cmd_str) {
                info!(
                    worker = %worker_id,
                    question = %question,
                    "Subagent paused waiting for orchestrator guidance"
                );
                let (tx, mut rx) = tokio::sync::mpsc::channel(1);
                let now = unix_timestamp();
                {
                    let mut lock = self.workers.write().await;
                    if let Some(w) = lock.get_mut(&worker_id) {
                        w.state = WorkerState::Paused {
                            question: question.clone(),
                            step,
                            paused_at: now,
                        };
                        w.resume_tx = Some(tx);
                    }
                }

                save_registry_entry(&WorkerRegistryEntry {
                    id: worker_id.clone(),
                    pid: std::process::id(),
                    task: task.clone(),
                    model: model.clone(),
                    status: "paused".into(),
                    step,
                    max_turns: current_max_turns,
                    last_command: cmd_summary.clone(),
                    question: Some(question.clone()),
                    started_at: started_at_ts,
                    updated_at: now,
                    group: Some(group.clone()),
                });

                if let Some(answer) = rx.recv().await {
                    info!(worker = %worker_id, "Worker resumed by orchestrator guidance");
                    {
                        let mut lock = self.workers.write().await;
                        if let Some(w) = lock.get_mut(&worker_id) {
                            w.state = WorkerState::Running {
                                step,
                                last_command: format!(
                                    "resumed: {}",
                                    summarize_command(&answer)
                                ),
                                started_at: now,
                            };
                            w.resume_tx = None;
                        }
                    }
                    messages.push(ChatMessage::text(
                        Role::User,
                        format!("ORCHESTRATOR RESPONSE / GUIDANCE:\n{}", answer),
                    ));
                }
            }

            if llm_resp.invalid_utf8_lines > 0 {
                info!(
                    worker = %worker_id,
                    step = step,
                    invalid_utf8_lines = llm_resp.invalid_utf8_lines,
                    "LLM stream contained non-UTF-8 frames; decoded lossily"
                );
            }

            let output_text = format!(
                "COMMAND OUTPUT (exit code: {}):\n```\n{}\n```",
                code.unwrap_or(-1),
                output
            );

            // The log entry is built *after* `output_text` so `output` is moved
            // rather than cloned (audit 07, F3), and both text fields are
            // clamped to a hard ceiling (audit 07, F6).
            let step_log = build_step_log(step, &cmd_summary, output, code);

            {
                let mut lock = self.workers.write().await;
                if let Some(w) = lock.get_mut(&worker_id) {
                    w.logs.push(step_log);
                }
            }

            if let (Some(tool_calls), Some(tc_id)) = (llm_resp.tool_calls, llm_resp.tool_call_id) {
                // OpenAI tool_calls protocol: assistant with tool_calls → tool response
                let content = if llm_resp.content.trim().is_empty() {
                    None
                } else {
                    Some(llm_resp.content)
                };
                messages.push(ChatMessage::assistant_with_tool_calls(content, tool_calls));
                messages.push(ChatMessage::tool_result(tc_id, &output_text));
            } else {
                // Fallback: code-block models use plain assistant + user messages
                let assistant_content = if llm_resp.content.trim().is_empty() {
                    "I will execute a bash command.".to_string()
                } else {
                    llm_resp.content
                };
                messages.push(ChatMessage::text(Role::Assistant, assistant_content));
                messages.push(ChatMessage::text(Role::User, output_text));
            }
        }

        let artifacts = worktree.sync_artifacts().unwrap_or_default();
        if !artifacts.is_empty() {
            info!(
                worker = %worker_id,
                count = artifacts.len(),
                "Synchronized worker artifacts to repo root"
            );
        }

        let diff = worktree.get_diff()?;
        let now = unix_timestamp();

        let summary = if !diff.trim().is_empty() {
            format!("Finished after {} turns. Produced git diff.", step)
        } else if !last_assistant_text.trim().is_empty() {
            last_assistant_text
        } else {
            format!("Finished after {} turns. Completed successfully.", step)
        };

        let branch = if !diff.trim().is_empty() {
            let commit_msg = format!(
                "worker({}): {}",
                worker_id,
                summary.lines().next().unwrap_or("")
            );
            worktree.commit_changes(&commit_msg).unwrap_or(None)
        } else {
            None
        };

        // The completion payload (diff/summary/artifacts/branch) is assembled
        // *before* the write-guard is taken: the critical section only performs
        // the O(1) move of the pre-built value into the record.
        let completed_state = WorkerState::Completed {
            turns: step,
            diff,
            summary,
            completed_at: now,
            artifacts,
            branch,
        };
        {
            let mut lock = self.workers.write().await;
            if let Some(w) = lock.get_mut(&worker_id) {
                w.state = completed_state;
            }
        }

        save_registry_entry(&WorkerRegistryEntry {
            id: worker_id.clone(),
            pid: std::process::id(),
            task: task.clone(),
            model: model.clone(),
            status: "completed".into(),
            step,
            max_turns: current_max_turns,
            last_command: "completed".into(),
            question: None,
            started_at: started_at_ts,
            updated_at: now,
            group: Some(group.clone()),
        });

        info!(worker = %worker_id, turns = step, "Worker completed successfully");
        Ok(())
    }

    pub async fn get_worker_state(&self, id: &str) -> Option<WorkerState> {
        self.workers.read().await.get(id).map(|w| w.state.clone())
    }

    /// Cheap snapshot of a worker's step history.
    ///
    /// Returns an `Arc` clone of the *bounded* buffer rather than a deep copy
    /// of every `String`, so a read never duplicates the whole history nor holds
    /// the shared `RwLock` while copying (audit 07, R5).
    pub async fn get_worker_logs(&self, id: &str) -> Option<Arc<LogBuffer>> {
        self.workers.read().await.get(id).map(|w| Arc::new(w.logs.clone()))
    }

    /// Retention counters for one worker, for the observability surface.
    pub async fn get_worker_log_stats(&self, id: &str) -> Option<LogStats> {
        self.workers.read().await.get(id).map(|w| w.log_stats())
    }

    /// Insert a synthetic record into the pool (test support).
    ///
    /// Exposed so integration tests can assert lock-discipline invariants
    /// (e.g. that `steer` performs its channel `send` *without* holding the
    /// write-guard) without spinning up a real LLM-backed worker.
    #[doc(hidden)]
    pub async fn __test_insert_worker(&self, record: WorkerRecord) {
        self.workers
            .write()
            .await
            .insert(record.id.clone(), record);
    }

    /// Lightweight poll used by the 500 ms progress loops.
    ///
    /// Clones only the small strings needed to render progress (`step`,
    /// `last_command` and the pause question) and never touches the potentially
    /// multi-megabyte `diff`, `summary` or `artifacts` of a completed worker.
    pub async fn worker_progress(&self, id: &str) -> Option<WorkerProgress> {
        let lock = self.workers.read().await;
        let w = lock.get(id)?;
        let progress = match &w.state {
            WorkerState::Running {
                step,
                last_command,
                ..
            } => WorkerProgress {
                phase: WorkerPhase::Running,
                step: *step,
                last_command: Some(last_command.clone()),
                question: None,
                terminal: false,
            },
            WorkerState::Paused { question, step, .. } => WorkerProgress {
                phase: WorkerPhase::Paused,
                step: *step,
                last_command: None,
                question: Some(question.clone()),
                terminal: false,
            },
            WorkerState::Completed { turns, .. } => WorkerProgress {
                phase: WorkerPhase::Completed,
                step: *turns,
                last_command: None,
                question: None,
                terminal: true,
            },
            WorkerState::Failed { step, .. } => WorkerProgress {
                phase: WorkerPhase::Failed,
                step: *step,
                last_command: None,
                question: None,
                terminal: true,
            },
        };
        drop(lock);
        Some(progress)
    }

    /// Number of recorded steps for a worker, without copying the log history.
    pub async fn worker_step_count(&self, id: &str) -> Option<usize> {
        self.workers.read().await.get(id).map(|w| w.logs.len())
    }

    /// Move a worker's log history out of the pool (terminal path).
    ///
    /// Taking the `Vec` is O(1); only the read-guard scope is exclusive of
    /// writers, and no per-step string is copied. The record stays registered so
    /// its final state can still be inspected with `get_worker_state`.
    pub async fn take_worker_logs(&self, id: &str) -> Option<LogBuffer> {
        let mut lock = self.workers.write().await;
        lock.get_mut(id).map(|w| std::mem::take(&mut w.logs))
    }

    /// Take a worker's record out of the pool and return it.
    ///
    /// Delegates to [`collect`](Self::collect) with bounded emission view.
    pub async fn take_worker(&self, id: &str) -> Option<CollectedWorker> {
        self.collect(id).await
    }

    pub async fn list_workers(&self) -> Vec<serde_json::Value> {
        let registry = load_all_registry_entries();
        if !registry.is_empty() {
            registry
                .into_iter()
                .map(|e| {
                    serde_json::json!({
                        "id": e.id,
                        "task": e.task,
                        "model": e.model,
                        "group": e.group.as_deref().unwrap_or("default"),
                        "state": {
                            "status": match e.status.as_str() {
                                "running" => "Running",
                                "completed" => "Completed",
                                "paused" => "Paused",
                                "failed" => "Failed",
                                _ => "Stopped",
                            },
                            "step": e.step,
                            "turns": e.step,
                            "last_command": e.last_command,
                            "pid": e.pid,
                            "started_at": e.started_at,
                        },
                        "total_steps": e.step,
                        // Registry rows are cross-process and carry no in-memory
                        // log buffer, so the retention counters are reported as
                        // 0/0 rather than being silently absent (audit 07, R7).
                        "logs_retained": 0,
                        "logs_dropped": 0,
                    })
                })
                .collect()
        } else {
            let lock = self.workers.read().await;
            lock.values()
                .map(|w| {
                    let stats = w.log_stats();
                    serde_json::json!({
                        "id": w.id,
                        "task": w.task,
                        "model": w.model,
                        "state": w.state.to_summary(),
                        "total_steps": stats.total_steps,
                        "logs_retained": stats.logs_retained,
                        "logs_dropped": stats.logs_dropped,
                    })
                })
                .collect()
        }
    }

    /// Deliver an orchestrator message to a worker.
    ///
    /// The `resume_tx` sender is extracted with `take()` while the write-guard is
    /// held and the guard is dropped *before* the `send().await` is performed:
    /// awaiting a full `mpsc` channel while holding a write-guard serialises the
    /// whole pool (`dispatch`, `kill`, `collect`, every state update and every
    /// read). A missing sender is now reported instead of silently dropping the
    /// guidance.
    /// Deliver an orchestrator message to a worker.
    ///
    /// The `resume_tx` sender is extracted with `take()` while the write-guard is
    /// held and the guard is dropped *before* the `send().await` is performed:
    /// awaiting a full `mpsc` channel while holding a write-guard serialises the
    /// whole pool (`dispatch`, `kill`, `collect`, every state update and every
    /// read). A missing sender is now reported instead of silently dropping the
    /// guidance.
    /// Deliver an orchestrator message to a worker.
    ///
    /// The `resume_tx` sender is extracted with `take()` while the write-guard is
    /// held and the guard is dropped *before* the `send().await` is performed:
    /// awaiting a full `mpsc` channel while holding a write-guard serialises the
    /// whole pool (`dispatch`, `kill`, `collect`, every state update and every
    /// read). A missing sender is now reported instead of silently dropping the
    /// guidance.
    pub async fn steer(&self, id: &str, message: String) -> Result<()> {
        let tx_opt = {
            let mut lock = self.workers.write().await;
            let w = lock
                .get_mut(id)
                .ok_or_else(|| anyhow::anyhow!("Worker not found: {id}"))?;
            match &w.state {
                WorkerState::Running { .. } => {
                    w.pending_steer.push(message);
                    return Ok(());
                }
                WorkerState::Paused { .. } => w.resume_tx.take(),
                _ => {
                    anyhow::bail!(
                        "Worker {} is not in a steerable state (running or paused)",
                        id
                    )
                }
            }
            // write-guard released here, before any `.await`
        };

        let Some(tx) = tx_opt else {
            anyhow::bail!(
                "Worker {} is paused but has no resume channel (already resumed)",
                id
            );
        };
        // Send without holding the lock: the worker needs the write-guard to
        // transition back to `Running` right after `rx.recv().await`.
        let _ = tx.send(message).await;
        Ok(())
    }

    pub async fn kill(&self, id: &str) -> bool {
        let mut lock = self.workers.write().await;
        if let Some(w) = lock.get_mut(id) {
            if let Some(handle) = w.handle.take() {
                handle.abort();
            }
            w.fail("Manually terminated by user/orchestrator");
            true
        } else {
            false
        }
    }

    /// Terminate every worker currently tracked by the pool.
    ///
    /// The map is walked once under a single write-guard. The loop contains no
    /// `.await`, so the critical section stays O(n) in the number of live
    /// workers (small, and bounded by the dispatch semaphore) and is never
    /// prolonged in wall-clock time.
    pub async fn kill_all(&self) -> usize {
        let mut lock = self.workers.write().await;
        let mut count = 0usize;
        for worker in lock.values_mut() {
            if !matches!(worker.state, WorkerState::Running { .. } | WorkerState::Paused { .. }) {
                continue;
            }
            if let Some(handle) = worker.handle.take() {
                handle.abort();
            }
            worker.fail("Server shutting down (received SIGINT)");
            count += 1;
        }
        count
    }

    /// Collect a worker's final result and release its in-memory resources.
    /// The retained window is *moved* out of the pool and then narrowed to the
    /// emission budget, so a single response can never serialize the full
    /// history (audit 07, R4). Both counters travel with the result so the
    /// degradation is visible to the orchestrator (audit 07, R7).
    pub async fn collect(&self, id: &str) -> Option<CollectedWorker> {
        let mut lock = self.workers.write().await;
        let record = lock.remove(id)?;
        tracing::info!(worker = %id, "Worker collected and evicted from pool");
        let dropped = record.logs.dropped();
        let view = emit_view(&record.logs, self.log_policy.max_emitted);
        Some(CollectedWorker {
            id: record.id,
            task: record.task,
            model: record.model,
            state: record.state,
            logs: view.logs,
            logs_omitted: view.logs_omitted,
            logs_dropped: dropped,
            logs_truncation_notice: view.logs_truncation_notice,
        })
    }
}

/// Spawn the background reaper that evicts expired terminal worker records.
///
/// Kept in the library so the server can start it from `run_stdio` without
/// depending on `main.rs`.
pub fn spawn_reaper(pool: WorkerPool) -> JoinHandle<()> {
    tokio::spawn(async move {
        let interval = std::time::Duration::from_secs(30);
        loop {
            tokio::time::sleep(interval).await;
            pool.reap().await;
        }
    })
}

/// Ids of `Completed`/`Failed` records that reached the terminal TTL (audit 07, R3).
///
/// * A `Running`/`Paused` record is *never* expired, whatever its age.
/// * Clock skew is absorbed by `saturating_sub`, so a timestamp in the future
///   (NTP jump, forged registry row) yields 0 and the record is kept.
/// * A fresh terminal record is kept, which is what keeps `collect` and
///   `wait: true` working after a worker finishes.
fn expired_terminal_ids(workers: &HashMap<String, WorkerRecord>, ttl_secs: u64) -> Vec<String> {
    let now = unix_timestamp();
    let mut ids: Vec<String> = workers
        .iter()
        .filter_map(|(id, record)| {
            record
                .terminal_at()
                .filter(|at| now.saturating_sub(*at) >= ttl_secs)
                .map(|_| id.clone())
        })
        .collect();
    ids.sort();
    ids
}

pub fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

pub fn summarize_command(cmd: &str) -> String {
    let first_line = cmd.lines().next().unwrap_or("").trim();
    if first_line.is_empty() {
        return "bash".to_string();
    }

    let mut out = String::with_capacity(40);
    let mut words = first_line.split_whitespace().take(4);

    if let Some(first) = words.next() {
        out.push_str(first);
        for word in words {
            out.push(' ');
            out.push_str(word);
        }
    }

    if out.len() > 40 {
        let cut = out.floor_char_boundary(37);
        out.truncate(cut);
        out.push_str("...");
    }

    out
}

pub fn parse_request_turns(cmd: &str) -> Option<usize> {
    let trimmed = cmd.trim();
    if (trimmed.starts_with("echo") || trimmed.starts_with("printf"))
        && let Some(pos) = trimmed.find("REQUEST_TURNS:")
    {
        let rest = &trimmed[pos + "REQUEST_TURNS:".len()..];
        let num_str: String = rest
            .chars()
            .skip_while(|c| c.is_whitespace())
            .take_while(|c| c.is_ascii_digit())
            .collect();
        if let Ok(n) = num_str.parse::<usize>()
            && n > 0
        {
            return Some(n);
        }
    }
    None
}

pub fn parse_ask_orchestrator(cmd: &str) -> Option<String> {
    let trimmed = cmd.trim();
    if (trimmed.starts_with("echo") || trimmed.starts_with("printf"))
        && let Some(pos) = trimmed.find("ASK_ORCHESTRATOR:")
    {
        let rest = &trimmed[pos + "ASK_ORCHESTRATOR:".len()..];
        let line = rest
            .lines()
            .next()
            .unwrap_or("")
            .trim()
            .trim_matches('"')
            .trim_matches('\'')
            .trim();
        if !line.is_empty() && line != "<your specific question>" && line != "<question>" {
            return Some(line.to_string());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_summarize_command_utf8() {
        let cmd = "echo 'esta_es_una_palabra_extremadamente_larga_con_ñ_y_acentos_para_superar_limite'";
        let summary = summarize_command(cmd);
        assert!(summary.ends_with("..."));

        // Multi-byte character exactly crossing byte 37
        let mut special = "a".repeat(36);
        special.push('€');
        special.push_str(" rest of command");
        let summary_special = summarize_command(&special);
        assert!(summary_special.ends_with("..."));
    }

    #[test]
    fn test_parse_request_turns() {
        assert_eq!(parse_request_turns("echo REQUEST_TURNS: 20"), Some(20));
        assert_eq!(parse_request_turns("printf 'REQUEST_TURNS: 15'"), Some(15));
        assert_eq!(parse_request_turns("cat file.rs"), None);
        assert_eq!(parse_request_turns("echo nothing"), None);
        assert_eq!(parse_request_turns("echo REQUEST_TURNS: 0"), None);
    }

    // ----------
    // LogBuffer retention (audit 07, R1/R2)
    // ----------

    fn entry(step: usize, out: &str) -> AgentStepLog {
        build_step_log(step, "cargo test", out.to_string(), Some(0))
    }

    fn policy(retained: usize, emitted: usize) -> LogRetentionPolicy {
        LogRetentionPolicy {
            max_retained: retained,
            max_bytes: retained * (MAX_LOG_OUTPUT_BYTES + MAX_LOG_COMMAND_BYTES),
            max_emitted: emitted,
        }
    }

    #[test]
    fn test_log_buffer_window_is_bounded() {
        let mut buf = LogBuffer::with_policy(policy(4, 4));
        for i in 0..50 {
            buf.push(entry(i, "ok"));
        }
        assert_eq!(buf.len(), 4, "entry count must never exceed the window");
        assert_eq!(buf.retained(), 4);
        assert_eq!(buf.dropped(), 46, "every evicted entry is counted");
        assert_eq!(buf.total(), 50, "retained + dropped == total steps");
        // The window is the *tail* of the history.
        let steps: Vec<usize> = buf.iter().map(|e| e.step).collect();
        assert_eq!(steps, vec![46, 47, 48, 49]);
    }

    #[test]
    fn test_log_buffer_reserves_capacity_for_the_window() {
        // R2: the backing store is pre-sized, so a full window never triggers a
        // reallocation (the old GeomGrow path wasted up to 41%).
        let buf = LogBuffer::with_policy(policy(128, 8));
        assert_eq!(
            buf.entries.capacity(),
            128,
            "capacity must be pre-reserved to the retention window"
        );
    }

    #[test]
    fn test_log_buffer_byte_budget_evicts_even_under_the_count_cap() {
        // A tiny byte budget with a generous count budget: the byte ceiling has
        // to win, otherwise the payload is still unbounded.
        let mut buf = LogBuffer::with_policy(LogRetentionPolicy {
            max_retained: 1000,
            max_bytes: 4 * 1024,
            max_emitted: 8,
        });
        for i in 0..20 {
            buf.push(entry(i, &"x".repeat(2048)));
        }
        assert!(
            buf.bytes() <= 4 * 1024,
            "byte budget exceeded: {}",
            buf.bytes()
        );
        assert!(
            buf.len() <= 2,
            "expected byte-driven eviction, got {}",
            buf.len()
        );
        assert!(buf.dropped() > 0);
    }

    #[test]
    fn test_log_buffer_empty_and_clear() {
        let mut buf = LogBuffer::new();
        assert!(buf.is_empty());
        assert!(buf.front().is_none());
        assert!(buf.back().is_none());
        buf.push(entry(1, "a"));
        assert!(buf.front().is_some());
        assert!(buf.back().is_some());
        buf.clear();
        assert!(buf.is_empty());
        assert_eq!(buf.bytes(), 0);
    }

    #[test]
    fn test_log_buffer_tail_returns_the_newest_entries() {
        let mut buf = LogBuffer::with_policy(policy(10, 10));
        for i in 0..7 {
            buf.push(entry(i, "ok"));
        }
        let tail: Vec<usize> = buf.tail(3).iter().map(|e| e.step).collect();
        assert_eq!(tail, vec![4, 5, 6]);
        // Asking for more than is retained yields the whole window.
        assert_eq!(buf.tail(100).len(), 7);
    }

    #[test]
    fn test_log_stats_track_total_retained_and_dropped() {
        let mut buf = LogBuffer::with_policy(policy(2, 2));
        for i in 0..5 {
            buf.push(entry(i, "ok"));
        }
        let stats = LogStats {
            total_steps: buf.total(),
            logs_retained: buf.retained(),
            logs_dropped: buf.dropped(),
        };
        assert_eq!(stats.total_steps, 5);
        assert_eq!(stats.logs_retained, 2);
        assert_eq!(stats.logs_dropped, 3);
    }

    #[test]
    fn test_log_buffer_serializes_as_a_plain_array() {
        let mut buf = LogBuffer::new();
        buf.push(entry(3, "ok"));
        let value = serde_json::to_value(&buf).unwrap();
        let arr = value
            .as_array()
            .expect("LogBuffer serializes as a JSON array");
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["step"], 3);
    }

    // ----------
    // clamp_string / build_step_log (audit 07, F6)
    // ----------

    #[test]
    fn test_clamp_string_leaves_short_values_untouched() {
        assert_eq!(clamp_string("hello", 64), "hello");
        assert_eq!(clamp_string("", 64), "");
    }

    #[test]
    fn test_clamp_string_charges_the_marker_against_the_budget() {
        // F6: the result is at most `budget` bytes -- the marker included.
        let long = "a".repeat(10_000);
        let clamped = clamp_string(&long, MAX_LOG_OUTPUT_BYTES);
        assert!(
            clamped.len() <= MAX_LOG_OUTPUT_BYTES,
            "clamped output was {} bytes",
            clamped.len()
        );
        assert!(clamped.contains("bytes truncated"));
        assert!(clamped.starts_with("aaaa"));
    }

    #[test]
    fn test_clamp_string_never_splits_a_code_point() {
        // '€' is 3 bytes: a 2047-byte budget must back off to a boundary.
        let mut s = "a".repeat(2045);
        s.push('€');
        s.push_str(&"b".repeat(1000));
        let clamped = clamp_string(&s, MAX_LOG_OUTPUT_BYTES);
        assert!(clamped.len() <= MAX_LOG_OUTPUT_BYTES);
        // The visible head is valid UTF-8 (it came from a str slice).
        assert!(clamped.contains('a'));
    }

    #[test]
    fn test_clamp_string_degrades_when_the_marker_does_not_fit() {
        // A budget smaller than the marker itself must not blow the ceiling.
        let clamped = clamp_string(&"x".repeat(100), 5);
        assert!(clamped.len() <= 5, "got {} bytes", clamped.len());
    }

    #[test]
    fn test_build_step_log_clamps_both_text_fields() {
        let log = build_step_log(7, &"c".repeat(500), "o".repeat(50_000), Some(3));
        assert!(log.command.len() <= MAX_LOG_COMMAND_BYTES);
        assert!(log.output.len() <= MAX_LOG_OUTPUT_BYTES);
        assert_eq!(log.step, 7);
        assert_eq!(log.exit_code, Some(3));
    }

    // ----------
    // emit_view (audit 07, R4/R7)
    // ----------

    #[test]
    fn test_emit_view_caps_the_payload_and_reports_what_is_missing() {
        let mut buf = LogBuffer::with_policy(policy(100, 10));
        for i in 0..60 {
            buf.push(entry(i, "ok"));
        }
        let view = emit_view(&buf, 10);
        assert_eq!(view.logs.len(), 10);
        assert_eq!(view.logs_omitted, 50);
        assert!(view.logs_truncation_notice.is_some());
        let notice = view.logs_truncation_notice.unwrap();
        assert!(
            notice.contains("50 retained log(s) omitted"),
            "got {notice}"
        );
        // The emitted tail is the newest 10.
        assert_eq!(view.logs.first().unwrap().step, 50);
        assert_eq!(view.logs.last().unwrap().step, 59);
    }

    #[test]
    fn test_emit_view_reports_evicted_entries_separately() {
        let mut buf = LogBuffer::with_policy(policy(5, 5));
        for i in 0..20 {
            buf.push(entry(i, "ok"));
        }
        let view = emit_view(&buf, 5);
        assert_eq!(view.logs_omitted, 0, "nothing retained was omitted");
        let notice = view
            .logs_truncation_notice
            .expect("eviction must be visible");
        assert!(notice.contains("15 earlier log(s) evicted"), "got {notice}");
    }

    #[test]
    fn test_emit_view_is_quiet_when_nothing_is_missing() {
        let mut buf = LogBuffer::new();
        buf.push(entry(1, "ok"));
        let view = emit_view(&buf, 40);
        assert_eq!(view.logs.len(), 1);
        assert_eq!(view.logs_omitted, 0);
        assert!(view.logs_truncation_notice.is_none());
        // ...and the notice is omitted from JSON entirely.
        let value = serde_json::to_value(&view).unwrap();
        assert!(value.get("logs_truncation_notice").is_none());
    }

    // ----------
    // LogRetentionPolicy::from_env (audit 07, R4 config)
    // ----------

    #[test]
    fn test_log_policy_defaults_and_ceilings() {
        let d = LogRetentionPolicy::default();
        assert_eq!(d.max_retained, DEFAULT_MAX_RETAINED_LOGS);
        assert_eq!(d.max_emitted, DEFAULT_MAX_EMITTED_LOGS);
        assert_eq!(
            d.max_bytes,
            DEFAULT_MAX_RETAINED_LOGS * worst_case_entry_bytes()
        );
        // The budget must actually cover a full window of worst-case entries.
        assert!(d.max_bytes >= d.max_retained * worst_case_entry_bytes());
        // A zero policy is coerced to something usable.
        let zero = LogBuffer::with_policy(LogRetentionPolicy {
            max_retained: 0,
            max_bytes: 0,
            max_emitted: 0,
        });
        let mut zero = zero;
        zero.push(entry(1, "ok"));
        assert_eq!(
            zero.len(),
            0,
            "a zero budget retains nothing rather than panicking"
        );
    }

    // ----------
    // WorkerState::step
    // ----------

    #[test]
    fn test_worker_state_step_covers_every_variant() {
        assert_eq!(
            WorkerState::Running {
                step: 4,
                last_command: "ls".into(),
                started_at: 0
            }
            .step(),
            4
        );
        assert_eq!(
            WorkerState::Paused {
                question: "?".into(),
                step: 9,
                paused_at: 0
            }
            .step(),
            9
        );
        assert_eq!(
            WorkerState::Completed {
                turns: 12,
                diff: String::new(),
                summary: String::new(),
                completed_at: 0,
                artifacts: Vec::new(),
                branch: None,
            }
            .step(),
            12
        );
        assert_eq!(
            WorkerState::Failed {
                error: "boom".into(),
                step: 3,
                failed_at: 0
            }
            .step(),
            3
        );
    }

    #[test]
    fn test_terminal_at_only_reports_terminal_states() {
        let running = WorkerState::Running {
            step: 0,
            last_command: String::new(),
            started_at: 0,
        };
        assert!(matches!(running, WorkerState::Running { .. }));
        let completed = WorkerState::Completed {
            turns: 1,
            diff: String::new(),
            summary: String::new(),
            completed_at: 1_700_000_000,
            artifacts: Vec::new(),
            branch: None,
        };
        let failed = WorkerState::Failed {
            error: "e".into(),
            step: 1,
            failed_at: 1_700_000_001,
        };
        assert!(!matches!(running, WorkerState::Completed { .. }));
        assert!(matches!(completed, WorkerState::Completed { .. }));
        assert!(matches!(failed, WorkerState::Failed { .. }));
    }

    fn record_with(state: WorkerState) -> WorkerRecord {
        WorkerRecord {
            id: "w".into(),
            task: "t".into(),
            model: "m".into(),
            state,
            logs: LogBuffer::new(),
            pending_steer: Vec::new(),
            resume_tx: None,
            handle: None,
        }
    }

    fn completed_at(when: u64) -> WorkerState {
        WorkerState::Completed {
            turns: 1,
            diff: String::new(),
            summary: String::new(),
            completed_at: when,
            artifacts: Vec::new(),
            branch: None,
        }
    }

    fn failed_at(when: u64) -> WorkerState {
        WorkerState::Failed {
            error: "boom".into(),
            step: 1,
            failed_at: when,
        }
    }

    // ----------
    // Terminal-record TTL (audit 07, R3)
    // ----------

    #[test]
    fn test_terminal_records_expire_after_the_ttl() {
        let now = unix_timestamp();
        let mut workers = HashMap::new();
        workers.insert("old-done".to_string(), record_with(completed_at(now - 400)));
        workers.insert("old-failed".to_string(), record_with(failed_at(now - 400)));
        workers.insert("fresh-done".to_string(), record_with(completed_at(now)));
        workers.insert(
            "running".to_string(),
            record_with(WorkerState::Running {
                step: 0,
                last_command: String::new(),
                started_at: now - 100_000,
            }),
        );
        workers.insert(
            "paused".to_string(),
            record_with(WorkerState::Paused {
                question: "?".into(),
                step: 1,
                paused_at: now - 100_000,
            }),
        );

        let expired = expired_terminal_ids(&workers, DEFAULT_TERMINAL_TTL_SECS);
        assert_eq!(
            expired,
            vec!["old-done".to_string(), "old-failed".to_string()],
            "only aged terminal records may be evicted"
        );
    }

    #[test]
    fn test_terminal_ttl_absorbs_clock_skew() {
        // A timestamp in the future (NTP jump, forged registry row) must not
        // evict the record: saturating_sub yields 0, which is below any TTL.
        let now = unix_timestamp();
        let mut workers = HashMap::new();
        workers.insert(
            "skewed".to_string(),
            record_with(completed_at(now + 10_000)),
        );
        assert!(
            expired_terminal_ids(&workers, DEFAULT_TERMINAL_TTL_SECS).is_empty(),
            "a future timestamp must never evict a record"
        );
    }

    #[test]
    fn test_terminal_ttl_of_zero_evicts_immediately() {
        let now = unix_timestamp();
        let mut workers = HashMap::new();
        workers.insert("done".to_string(), record_with(completed_at(now)));
        assert_eq!(expired_terminal_ids(&workers, 0).len(), 1);
    }

    #[test]
    fn test_terminal_ttl_keeps_everything_before_the_boundary() {
        let now = unix_timestamp();
        let mut workers = HashMap::new();
        workers.insert("edge".to_string(), record_with(completed_at(now)));
        assert!(
            expired_terminal_ids(&workers, 1).is_empty(),
            "a record younger than the TTL must survive so collect() still works"
        );
    }
    #[test]
    fn test_parse_ask_orchestrator() {
        assert_eq!(
            parse_ask_orchestrator("echo 'ASK_ORCHESTRATOR: should I delete old code?'"),
            Some("should I delete old code?".to_string())
        );
        assert_eq!(
            parse_ask_orchestrator("echo \"ASK_ORCHESTRATOR: is this ok?\""),
            Some("is this ok?".to_string())
        );
        assert_eq!(parse_ask_orchestrator("cat src/agent.rs"), None);
        assert_eq!(
            parse_ask_orchestrator("echo 'ASK_ORCHESTRATOR: <your specific question>'"),
            None
        );
        assert_eq!(parse_ask_orchestrator("ls -la"), None);
    }
}
