//! Worker pool: dispatch, tracking, steering, collection and reaping.
//!
//! Split by responsibility while keeping the historical
//! `mini_swe_mcp::pool::*` surface byte-for-byte identical through the
//! re-exports below:
//!
//! * [`buffer`] — bounded step-log retention window and emission view.
//! * [`state`] — worker lifecycle state, the in-memory record, progress views
//!   and the terminal-record TTL.
//! * [`registry`] — the on-disk JSON registry shared across processes.
//! * [`steer`] — the disk-backed steering mailbox, the cross-process delivery
//!   path for orchestrator guidance.
//! * [`runner`] — the agent execution loop and the orchestrator sentinels.
//! * [`clock`] — the shared wall-clock helper.
//! * [`admission`] — resource-aware admission control for heavy commands,
//!   replacing the fixed-width build semaphore.
//!
//! [`WorkerPool`] itself stays here: it owns the concurrency gates and the
//! worker map, and every operation on them (dispatch, collect, steer, kill,
//! reap) must stay in one place to keep the lock discipline auditable.
//!
//! A consolidator's reach is bounded by
//! [`check_consolidate_delegation`](registry::check_consolidate_delegation):
//! its owner's own workers, in its own group, and nothing else.

use anyhow::Result;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use tokio::sync::{RwLock, Semaphore, watch};
use tokio::task::JoinHandle;
use tracing::{error, info, warn};

pub mod admission;
mod buffer;
mod clock;
mod fair;
pub mod merge;
mod registry;
pub mod revision;
mod runner;
mod state;
mod steer;

pub use self::admission::{
    AdmissionController, AdmissionInputs, Blocked, Decision, HeavyPermit, HostSample, admit,
    jobs_for,
};
pub use self::buffer::{
    DEFAULT_MAX_EMITTED_LOGS, DEFAULT_MAX_RETAINED_LOGS, EmittedLogs, LogBuffer,
    LogRetentionPolicy, LogStats, MAX_EMITTED_LOGS_CEILING, MAX_LOG_COMMAND_BYTES,
    MAX_LOG_OUTPUT_BYTES, MAX_RETAINED_LOGS_CEILING, build_step_log, clamp_string, emit_view,
};
pub use self::clock::unix_timestamp;
pub(crate) use self::registry::recover_orphaned_workers;
pub use self::registry::{
    RegistryStatus, UNATTRIBUTED_OWNER, WorkerMeta, WorkerRegistryEntry, WorkerRole,
    check_consolidate_delegation, extract_group, load_all_registry_entries,
    load_all_registry_entries_in, load_registry_entries_read_only,
    load_registry_entries_read_only_in, load_registry_entry, load_registry_entry_in, registry_dir,
    registry_dir_in, registry_owner_label, remove_registry_entry, remove_registry_entry_in,
    save_registry_entry, save_registry_entry_in,
};
pub use self::revision::{
    CONTINUE_PREFIX, DEFAULT_REVISION_TURNS, MAX_AUTO_CONTINUES, REVISION_PREFIX, SteerOutcome,
    WorkerHistory, append_history_message, append_history_message_in, ensure_base_branch,
    history_log_path, history_log_path_in, history_path, history_path_in, is_replayable,
    load_worker_history, load_worker_history_in, load_worker_history_log,
    load_worker_history_log_in, prune_orphan_histories, prune_orphan_histories_in,
    remove_worker_history, remove_worker_history_in, save_worker_history, save_worker_history_in,
};

pub use self::merge::{MergeReport, MergeRequest, merge_worker, merge_worker_in};
pub use self::runner::RunConfig;
pub(crate) use self::runner::parse_shortstat;
pub use self::runner::{
    COMPLETION_SENTINEL, WorkerLaunchConfig, is_completion_request, parse_ask_orchestrator,
    parse_consolidate_merge, parse_request_turns, summarize_command,
};
pub use self::state::{
    CollectedWorker, DEFAULT_TERMINAL_TTL_SECS, WorkerMetrics, WorkerOwner, WorkerPhase,
    WorkerProgress, WorkerRecord, WorkerState,
};
pub use self::steer::{
    drain_steer_messages, drain_steer_messages_in, remove_steer_file, remove_steer_file_in,
    steer_path, steer_path_in, write_steer_message, write_steer_message_in,
};

use self::state::expired_terminal_ids;
use crate::manifest::ModelManifest;
use crate::worktree::{BranchMerge, ScratchRoot, WorktreeGuard};

/// Rebuild the conversation a request sends: compaction applied on top of the
/// append-only log.
///
/// The log keeps every message as it was pushed, so the compacted view is
/// derived here rather than stored, and a continuation replays the same shape
/// the live loop sent.
pub fn compact_for_request(
    messages: &[crate::agent::ChatMessage],
) -> Vec<crate::agent::ChatMessage> {
    let mut messages = messages.to_vec();
    crate::pool::runner::history::compact_history(&mut messages);
    messages
}

/// The observation one `CONSOLIDATE_MERGE` request returns to the model.
pub struct ConsolidateMerge {
    /// One compact line per requested id.
    pub observation: String,
    /// Whether any branch was merged, so the caller preserves the
    /// consolidator's branch instead of letting the guard delete it.
    pub integrated: bool,
}

/// Whether `id`'s registry row is live in a process other than this one.
///
/// The on-disk steer mailbox is only written for such a worker: a row whose
/// pid is dead (or is this process, which already holds the worker map) has
/// nobody to drain it, so the message continues the worker instead.
#[allow(dead_code)]
fn registry_row_live_elsewhere(id: &str) -> bool {
    load_registry_entry(id).is_some_and(|e| {
        e.status.is_live()
            && e.pid != std::process::id()
            && crate::worktree::is_process_alive(e.pid)
    })
}

impl WorkerPool {
    /// Whether `id`'s registry row is live in a process other than this one,
    /// read under this pool's scratch root.
    fn registry_row_live_elsewhere(&self, id: &str) -> bool {
        load_registry_entry_in(&self.scratch, id).is_some_and(|e| {
            e.status.is_live()
                && e.pid != std::process::id()
                && crate::worktree::is_process_alive(e.pid)
        })
    }
}

/// Guidance appended to every terminal payload and channel event.
///
/// Tells the orchestrator the review loop exists: the finished worker's branch
/// is still there, and `steer` with corrections resumes it in place.
pub fn next_step_for(branch: Option<&str>) -> String {
    match branch {
        Some(branch) => format!(
            "Review the diff (collect) and run the project's checks. To correct this worker -- or to continue any worker that stopped (failed, interrupted, killed) -- call steer <id> \"...\" with the concrete corrections; it continues on branch {branch} with its full context. Never dispatch a replacement for a stopped worker. Merge only when it is right."
        ),
        None => "Review the result and run the project's checks. To correct this worker -- or to continue any worker that stopped (failed, interrupted, killed) -- call steer <id> \"...\" with the concrete corrections; it continues with its full context. Never dispatch a replacement for a stopped worker. Merge only when it is right.".to_string(),
    }
}

/// The branch a terminal [`WorkerState`] finished on, if it kept one.
pub fn terminal_branch(state: &WorkerState) -> Option<String> {
    match state {
        WorkerState::Completed { branch, .. } => branch.clone(),
        WorkerState::Running { .. } | WorkerState::Paused { .. } | WorkerState::Failed { .. } => {
            None
        }
    }
}

/// Clears a worker's heavy-slot wait when the slot is granted or the wait is
/// abandoned (its worker killed, its step aborted).
pub struct BuildWait {
    pool: WorkerPool,
    id: String,
}

impl Drop for BuildWait {
    fn drop(&mut self) {
        let removed = self
            .pool
            .admission_waiting
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .remove(&self.id)
            .is_some();
        if removed {
            self.pool.notify_change();
        }
    }
}

/// Holds a worker's "bash command running" mark for the command's lifetime.
///
/// The stall detector reads it: a step whose command is still executing is not
/// idle, so a long `cargo test` or verify gate never looks like a stall.
pub struct CommandRun {
    pool: WorkerPool,
    id: String,
}

impl Drop for CommandRun {
    fn drop(&mut self) {
        let removed = self
            .pool
            .command_running
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .remove(&self.id)
            .is_some();
        if removed {
            self.pool.notify_change();
        }
    }
}

/// The word that stands for the caller's most recently dispatched worker.
pub const LAST_WORKER_ID: &str = "last";

/// Shortest prefix accepted for a worker id: below it a typo would be too
/// likely to name a different worker by accident.
pub const MIN_WORKER_ID_PREFIX: usize = 3;

/// How a caller-supplied worker reference resolved among its own workers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkerIdLookup {
    /// The full id to act on; every response renders this, never the prefix.
    Resolved(String),
    /// Nothing of the caller's matches. The needle is left as written so the
    /// verb keeps its own "not found" answer (and `kill` its `killed: false`).
    NotFound,
    /// A prefix shared by several of the caller's workers, sorted.
    Ambiguous(Vec<String>),
}

#[derive(Clone)]
pub struct WorkerPool {
    worker_slots: fair::FairScheduler,
    bash_semaphore: Arc<Semaphore>,
    /// Resource-aware gate for heavy commands: the slot count, the memory and
    /// load criteria and the job count all live here.
    admission: AdmissionController,
    /// Worker id -> heavy commands queued ahead of it while it waits for a
    /// build slot. Set while blocked in admission, cleared when granted.
    admission_waiting: Arc<std::sync::Mutex<HashMap<String, usize>>>,
    /// Worker id -> unix time its current bash command started. Set while
    /// `execute_bash` runs and cleared when it returns, so the stall detector
    /// can tell a long command from worker inactivity.
    command_running: Arc<std::sync::Mutex<HashMap<String, u64>>>,
    workers: Arc<RwLock<HashMap<String, WorkerRecord>>>,
    changes: watch::Sender<u64>,
    registry: Arc<std::sync::Mutex<registry::RegistryWriter>>,
    /// The scratch root every per-worker path is resolved under. Resolved
    /// once at construction so the pool never re-reads `SWE_TEMP_DIR`
    /// mid-run; cloned (not re-resolved) by every pool method.
    scratch: ScratchRoot,
    /// Checkout directory of every live worker. The `WorktreeGuard` stays the
    /// owner of the worktree itself; the pool only needs to know *where* a
    /// worker works so `kill` can commit what it leaves behind before the
    /// aborted task tears the checkout down.
    worktrees: Arc<RwLock<HashMap<String, PathBuf>>>,
    api_base: String,
    api_key: String,
    log_policy: LogRetentionPolicy,
    terminal_ttl: Duration,
    /// The model manifest this pool's workers resolve against. Shared with the
    /// MCP server so a dispatch and its worker never disagree on the catalog.
    manifest: Arc<ModelManifest>,
    /// Dispatch ordinal per worker id, so `last` names the most recently
    /// dispatched worker even when two share a whole-second `started_at`.
    dispatch_order: Arc<std::sync::Mutex<HashMap<String, u64>>>,
    next_dispatch_seq: Arc<AtomicU64>,
}

impl WorkerPool {
    pub fn new(max_concurrent: usize, api_base: String, api_key: String) -> Self {
        Self::with_scratch(max_concurrent, api_base, api_key, ScratchRoot::from_env())
    }

    /// [`WorkerPool::new`] under an explicit scratch root.
    ///
    /// Tests build one pool per temporary root, so in-process tests that run
    /// in parallel threads never share registry rows, mailboxes or history
    /// files. Production callers use [`WorkerPool::new`], which resolves the
    /// same root from the process environment once, at construction time.
    pub fn with_scratch(
        max_concurrent: usize,
        api_base: String,
        api_key: String,
        scratch: ScratchRoot,
    ) -> Self {
        // Heavy commands are dosed by the admission controller: a slot only
        // when the host can take another build, and a job count divided over
        // the builds already running.
        let admission = AdmissionController::from_env();

        // Each worker runs one command at a time, so with one slot per worker
        // this gate is inert; BASH_CONCURRENT_LIMIT opts into a tighter cap.
        // Heavy commands are throttled separately by the admission controller.
        let bash_slots = std::env::var("BASH_CONCURRENT_LIMIT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(max_concurrent);

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
            max_heavy = admission.max_heavy(),
            mem_reserve_mb = admission.reserve_mb(),
            build_mem_mb = admission.estimate_mb(),
            max_retained_logs = log_policy.max_retained,
            max_emitted_logs = log_policy.max_emitted,
            terminal_ttl_secs = terminal_ttl.as_secs(),
            worker_slots = max_concurrent,
            "Worker slots, bash semaphore and heavy-command admission controller initialized"
        );
        let registry = Arc::new(std::sync::Mutex::new(registry::RegistryWriter::new(
            scratch.clone(),
        )));
        Self {
            worker_slots: fair::FairScheduler::new(max_concurrent),
            bash_semaphore: Arc::new(Semaphore::new(bash_slots)),
            admission,
            admission_waiting: Arc::new(std::sync::Mutex::new(HashMap::new())),
            command_running: Arc::new(std::sync::Mutex::new(HashMap::new())),
            workers: Arc::new(RwLock::new(HashMap::new())),
            changes: watch::channel(0).0,
            registry,
            worktrees: Arc::new(RwLock::new(HashMap::new())),
            api_base,
            api_key,
            log_policy,
            terminal_ttl,
            manifest: Arc::new(ModelManifest::default()),
            dispatch_order: Arc::new(std::sync::Mutex::new(HashMap::new())),
            next_dispatch_seq: Arc::new(AtomicU64::new(1)),
            scratch,
        }
    }

    /// The scratch root this pool resolves every per-worker path under.
    /// The pool's heavy-command admission controller.
    ///
    /// Shared by clone, so a caller (the merge gate) reserves host budget from
    /// the same controller every worker's heavy step does instead of starting
    /// a second, unaware one.
    pub fn admission(&self) -> AdmissionController {
        self.admission.clone()
    }

    pub fn scratch_root(&self) -> &ScratchRoot {
        &self.scratch
    }

    /// Subscribe before reading state so a concurrent change cannot be missed.
    pub fn subscribe_changes(&self) -> watch::Receiver<u64> {
        self.changes.subscribe()
    }

    /// How many workers are executing a bash command right now.
    ///
    /// A planned handover reads it: a command in flight is work a stop must not
    /// interrupt, while a live worker between commands is exactly what the
    /// graceful shutdown checkpoints and the next daemon continues.
    pub fn commands_running(&self) -> usize {
        self.command_running
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .len()
    }

    /// Mark `id` as queued for a heavy build slot behind `queued` requests.
    /// The returned guard clears the state when the slot is granted or the
    /// wait is abandoned, so a killed worker leaves no stale wait behind.
    pub fn wait_for_build_slot(&self, id: &str, queued: usize) -> BuildWait {
        self.admission_waiting
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .insert(id.to_string(), queued);
        self.notify_change();
        BuildWait {
            pool: self.clone(),
            id: id.to_string(),
        }
    }

    /// Publish that `id`'s bash command started now; the guard clears the mark
    /// when the command returns or its future is dropped.
    pub fn command_running(&self, id: &str) -> CommandRun {
        self.command_running
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .insert(id.to_string(), unix_timestamp());
        self.notify_change();
        CommandRun {
            pool: self.clone(),
            id: id.to_string(),
        }
    }

    fn notify_change(&self) {
        self.changes
            .send_modify(|generation| *generation = generation.wrapping_add(1));
    }

    /// Stamp `id` as the newest dispatch, so `last` can name it.
    fn record_dispatch(&self, id: &str) {
        let ordinal = self.next_dispatch_seq.fetch_add(1, Ordering::Relaxed);
        self.dispatch_order
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .insert(id.to_string(), ordinal);
    }

    /// All worker progress and lifecycle mutations notify under the write lock.
    async fn update_worker(&self, id: &str, update: impl FnOnce(&mut WorkerRecord)) {
        let mut workers = self.workers.write().await;
        if let Some(worker) = workers.get_mut(id) {
            update(worker);
            self.notify_change();
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn save_status(
        &self,
        meta: &WorkerMeta,
        model: &str,
        status: RegistryStatus,
        step: usize,
        max_turns: usize,
        last_command: &str,
        question: Option<String>,
    ) {
        self.registry
            .lock()
            .expect("registry lock poisoned")
            .save(meta.entry(model, status, step, max_turns, last_command, question));
    }

    /// Attach the model manifest this pool's workers resolve against (the
    /// built-in catalog until one is attached). The MCP server reads it back
    /// from the pool, so both share one `Arc`.
    pub fn with_manifest(mut self, manifest: Arc<ModelManifest>) -> Self {
        self.manifest = manifest;
        self
    }

    /// The model manifest this pool's workers resolve against.
    pub fn manifest(&self) -> &ModelManifest {
        &self.manifest
    }

    /// Clone of the shared manifest `Arc`, for the MCP server to share.
    pub(crate) fn manifest_arc(&self) -> Arc<ModelManifest> {
        self.manifest.clone()
    }

    /// The active step-log retention policy.
    pub fn log_policy(&self) -> LogRetentionPolicy {
        self.log_policy
    }

    /// Evict terminal worker records whose TTL expired, plus their registry
    /// entries. Fresh terminal records are deliberately kept so a subsequent
    /// `collect` / `wait: true` still finds them (audit 07, R3).
    pub async fn reap(&self) -> Vec<String> {
        let expired = {
            let mut lock = self.workers.write().await;
            self.reap_locked(&mut lock)
        };
        self.forget_worktrees(&expired).await;
        expired
    }

    /// Drop the worktree paths of workers whose records are gone.
    ///
    /// The worktree map is keyed exactly like the worker map, so a record
    /// leaving the pool is what retires its checkout path.
    async fn forget_worktrees(&self, ids: &[String]) {
        if ids.is_empty() {
            return;
        }
        self.worktrees
            .write()
            .await
            .retain(|id, _| !ids.iter().any(|gone| gone == id));
    }

    /// Shared eviction helper: removes expired terminal records from `lock`.
    fn reap_locked(&self, lock: &mut HashMap<String, WorkerRecord>) -> Vec<String> {
        let ttl = self.terminal_ttl.as_secs();
        let expired = expired_terminal_ids(lock, ttl);
        for id in &expired {
            lock.remove(id);
            remove_registry_entry_in(&self.scratch, id);
            self.registry
                .lock()
                .expect("registry lock poisoned")
                .remove(id);
        }
        if !expired.is_empty() {
            self.notify_change();
            tracing::info!(
                count = expired.len(),
                ttl_secs = ttl,
                "Reaped terminal worker records"
            );
        }
        expired
    }

    /// Spawn one worker owned by `owner`.
    ///
    /// The owner is the agent identity of the connection that dispatched it
    /// and is recorded on both the in-memory record and the registry row, so
    /// ownership survives a process restart (H-3).
    ///
    /// `client_env` is the dispatcher's ambient environment, already filtered
    /// by the sandbox's secret filter: it is layered on top of the canonical
    /// sandbox environment by the differential verify gate, so a suite that
    /// only passes in the orchestrator's shell is caught by the worker itself.
    #[allow(clippy::too_many_arguments)]
    pub async fn dispatch(
        &self,
        owner: String,
        task: String,
        model: String,
        temperature: Option<f32>,
        repo_path: PathBuf,
        max_turns: usize,
        group: Option<String>,
        review_after: Option<String>,
        network_offline: bool,
        verify: Option<String>,
        client_env: Vec<(String, String)>,
    ) -> Result<String> {
        self.dispatch_with_role(
            owner,
            task,
            model,
            temperature,
            repo_path,
            max_turns,
            group,
            review_after,
            network_offline,
            verify,
            client_env,
            WorkerRole::Worker,
        )
        .await
    }

    /// Dispatch with explicit authority; consolidators require an explicit group.
    #[allow(clippy::too_many_arguments)]
    pub async fn dispatch_with_role(
        &self,
        owner: String,
        task: String,
        model: String,
        temperature: Option<f32>,
        repo_path: PathBuf,
        max_turns: usize,
        group: Option<String>,
        review_after: Option<String>,
        network_offline: bool,
        verify: Option<String>,
        client_env: Vec<(String, String)>,
        role: WorkerRole,
    ) -> Result<String> {
        if role == WorkerRole::Consolidate && group.as_deref().is_none_or(|g| g.trim().is_empty()) {
            anyhow::bail!("role 'consolidate' requires 'group'");
        }
        // F6: format the low 32 UUID bits directly instead of building (and
        // immediately discarding) a full hyphenated `String` per worker.
        let worker_id = format!("{:08x}", uuid::Uuid::new_v4().as_u128() as u32);
        self.record_dispatch(&worker_id);
        let now = unix_timestamp();
        let resolved_group = group
            .or_else(|| extract_group(&task))
            .unwrap_or_else(|| "default".to_string());

        let repo_path_str = repo_path.to_string_lossy().to_string();
        let meta = WorkerMeta {
            id: worker_id.clone(),
            task: task.clone(),
            group: Some(resolved_group.clone()),
            role,
            repo_path: Some(repo_path_str.clone()),
            owner: owner.clone(),
            started_at: now,
            pid: std::process::id(),
            // Filled in by the phase loop; the dispatch itself measures nothing.
            metrics: WorkerMetrics::default(),
            revision: 0,
            auto_continues: 0,
        };

        let initial_record = WorkerRecord {
            id: worker_id.clone(),
            task: task.clone(),
            model: model.clone(),
            owner,
            state: WorkerState::Running {
                step: 0,
                last_command: String::from("initializing"),
                started_at: now,
            },
            metrics: WorkerMetrics::default(),
            // Pre-size the retention window so the log buffer never
            // over-allocates (audit 07, R2).
            logs: LogBuffer::with_policy(self.log_policy),
            pending_steer: Vec::new(),
            resume_tx: None,
            handle: None,
            revision: 0,
        };

        self.save_status(
            &meta,
            &model,
            RegistryStatus::Running,
            0,
            max_turns,
            "initializing",
            None,
        );

        // Prune stale terminal records *before* inserting, so a long-lived
        // server bounds residency even without the background reaper
        // (audit 07, R3).
        let expired = {
            let mut lock = self.workers.write().await;
            let expired = self.reap_locked(&mut lock);
            lock.insert(worker_id.clone(), initial_record);
            self.notify_change();
            expired
        };
        self.forget_worktrees(&expired).await;

        let pool = self.clone();
        let wid = worker_id.clone();
        let mut meta_for_fail = meta;
        let model_for_fail = model.clone();
        // Resolve the verify gate: an explicit empty string disables it, an
        // explicit command is used verbatim, and an absent argument auto-detects
        // from the repository layout.
        let verify = match verify {
            Some(cmd) if cmd.is_empty() => None,
            Some(cmd) => Some(cmd),
            None => detect_verify_command(&repo_path),
        };

        let config = WorkerLaunchConfig {
            task,
            model,
            temperature,
            repo_path,
            max_turns,
            review_after,
            network_offline,
            verify,
            client_env,
            resume_messages: None,
            resume_base_commit: None,
            resume_base_branch: None,
        };

        let join_handle = tokio::spawn(async move {
            // `meta_for_fail` is lent to the loop, so the counters it moved
            // before failing are still readable here.
            if let Err(e) = pool
                .run_worker(wid.clone(), config, &mut meta_for_fail)
                .await
            {
                error!(worker = %wid, error = %e, "Worker failed with error");
                pool.update_worker(&wid, |w| w.fail(e.to_string())).await;
                pool.save_status(
                    &meta_for_fail,
                    &model_for_fail,
                    RegistryStatus::Failed,
                    0,
                    max_turns,
                    &format!("error: {e}"),
                    None,
                );
            }
        });

        // Attach the handle in a second short write-guard: the record was
        // inserted without one so the worker is visible immediately, and this
        // minimises the window in which a concurrent `kill`/`kill_all` could
        // not yet abort the task.
        {
            let mut lock = self.workers.write().await;
            if let Some(w) = lock.get_mut(&worker_id) {
                w.handle = Some(join_handle);
            }
        }

        Ok(worker_id)
    }
    pub async fn get_worker_state(&self, id: &str) -> Option<WorkerState> {
        self.workers.read().await.get(id).map(|w| w.state.clone())
    }

    /// Whether `id` is a finished worker: terminal in this process, or
    /// terminal in the shared registry (e.g. after a hub restart).
    ///
    /// A cheap read used by `steer --wait` to pick the progress denominator of
    /// a revision before the relaunch overwrites the record.
    pub async fn is_terminal(&self, id: &str) -> bool {
        if let Some(state) = self.get_worker_state(id).await {
            if matches!(
                state,
                WorkerState::Completed { .. } | WorkerState::Failed { .. }
            ) {
                return true;
            }
            return false;
        }
        load_registry_entry_in(&self.scratch, id).is_some_and(|e| e.status.is_terminal())
    }

    /// Cheap snapshot of a worker's step history.
    ///
    /// Returns a clone of the *bounded* buffer, so a read never duplicates the
    /// history nor holds the `RwLock` while copying (audit 07, R5).
    pub async fn get_worker_logs(&self, id: &str) -> Option<LogBuffer> {
        self.workers.read().await.get(id).map(|w| w.logs.clone())
    }

    /// Insert a synthetic record into the pool (test support).
    ///
    /// Exposed so integration tests can assert lock-discipline invariants
    /// (e.g. that `steer` performs its channel `send` *without* holding the
    /// write-guard) without spinning up a real LLM-backed worker.
    #[doc(hidden)]
    pub async fn __test_insert_worker(&self, record: WorkerRecord) {
        self.record_dispatch(&record.id);
        self.workers.write().await.insert(record.id.clone(), record);
        self.notify_change();
    }

    /// Change a synthetic worker's state through the normal notification path.
    #[doc(hidden)]
    pub async fn __test_set_worker_state(&self, id: &str, state: WorkerState) {
        self.update_worker(id, |worker| worker.state = state).await;
    }

    /// Register a synthetic worker's worktree path (test support).
    ///
    /// The kill/shutdown checkpoint path only knows where to commit through
    /// [`WorkerPool::register_worktree`], which a real dispatch calls; a test
    /// that drives `kill` or `kill_all` without an LLM needs the same hook.
    #[doc(hidden)]
    pub async fn __test_register_worktree(&self, worker_id: &str, path: PathBuf) {
        self.register_worktree(worker_id, path).await;
    }

    /// Route one registry write through the coalescing writer (test support).
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn __test_save_status(
        &self,
        meta: &WorkerMeta,
        model: &str,
        status: RegistryStatus,
        step: usize,
        max_turns: usize,
        last_command: &str,
        question: Option<String>,
    ) {
        self.save_status(meta, model, status, step, max_turns, last_command, question);
    }

    /// Forget the last-write timestamp of `id`'s row (test support).
    ///
    /// Lets a test drive the coalescing writer past its throttle window
    /// without sleeping for it.
    #[doc(hidden)]
    pub fn __test_reset_registry_throttle(&self, id: &str) {
        self.registry
            .lock()
            .expect("registry lock poisoned")
            .reset_throttle(id);
    }

    /// Lightweight snapshot for progress waiters.
    ///
    /// Clones only the small strings needed to render progress and never the
    /// potentially multi-megabyte terminal payload.
    pub async fn worker_progress(&self, id: &str) -> Option<WorkerProgress> {
        let waiting_for_slot = self
            .admission_waiting
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .get(id)
            .copied();
        let command_started_at = self
            .command_running
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .get(id)
            .copied();
        let lock = self.workers.read().await;
        let w = lock.get(id)?;
        let progress = match &w.state {
            WorkerState::Running {
                step, last_command, ..
            } => WorkerProgress {
                phase: WorkerPhase::Running,
                step: *step,
                last_command: Some(last_command.clone()),
                question: None,
                waiting_for_slot,
                command_started_at,
            },
            WorkerState::Paused { question, step, .. } => WorkerProgress {
                phase: WorkerPhase::Paused,
                step: *step,
                last_command: None,
                question: Some(question.clone()),
                waiting_for_slot,
                command_started_at: None,
            },
            WorkerState::Completed { turns, .. } => WorkerProgress {
                phase: WorkerPhase::Completed,
                step: *turns,
                last_command: None,
                question: None,
                waiting_for_slot: None,
                command_started_at: None,
            },
            WorkerState::Failed { step, .. } => WorkerProgress {
                phase: WorkerPhase::Failed,
                step: *step,
                last_command: None,
                question: None,
                waiting_for_slot: None,
                command_started_at: None,
            },
        };
        drop(lock);
        Some(progress)
    }

    /// Workers this pool still holds as `Running` or `Paused`.
    ///
    /// The hub daemon's idle shutdown keys off this: a terminal record is kept
    /// for `collect`, so it must not hold the daemon open.
    pub async fn active_worker_count(&self) -> usize {
        let lock = self.workers.read().await;
        lock.values()
            .filter(|w| {
                matches!(
                    w.state,
                    WorkerState::Running { .. } | WorkerState::Paused { .. }
                )
            })
            .count()
    }

    /// Every worker the registry knows about, each with the agent that owns it.
    pub async fn list_workers(&self) -> Vec<serde_json::Value> {
        self.list_workers_matching(None).await
    }

    /// Only the workers owned by `owner` (H-3).
    ///
    /// The default view of `list`: an orchestrator sees its own workers and
    /// nobody else's, while [`WorkerPool::list_workers`] (behind
    /// `scope: "all"`) still reports every row together with its owner.
    pub async fn list_workers_of(&self, owner: &str) -> Vec<serde_json::Value> {
        self.list_workers_matching(Some(owner)).await
    }

    /// The list payload, optionally narrowed to one owner.
    ///
    /// The union of this process's records (the live view, with log counters)
    /// and the registry rows of every other worker (other processes, or rows
    /// whose record was already reaped here).
    async fn list_workers_matching(&self, owner: Option<&str>) -> Vec<serde_json::Value> {
        let mut rows: Vec<serde_json::Value> = Vec::new();
        let mut seen = std::collections::HashSet::new();
        {
            let lock = self.workers.read().await;
            for w in lock
                .values()
                .filter(|w| owner.is_none_or(|owner| w.owner == owner))
            {
                let stats = w.log_stats();
                seen.insert(w.id.clone());
                rows.push(serde_json::json!({
                    "id": w.id,
                    "task": w.task,
                    "model": w.model,
                    "owner": w.owner,
                    "state": w.state.to_summary(),
                    "total_steps": stats.total_steps,
                    "logs_retained": stats.logs_retained,
                    "logs_dropped": stats.logs_dropped,
                }));
            }
        }
        let registry = load_all_registry_entries_in(&self.scratch)
            .into_iter()
            .filter(|e| {
                !seen.contains(&e.id) && owner.is_none_or(|owner| e.owner.as_deref() == Some(owner))
            });
        rows.extend(registry.map(|e| {
            serde_json::json!({
                "id": e.id,
                "task": e.task,
                "model": e.model,
                "owner": registry_owner_label(&e),
                "group": e.group.as_deref().unwrap_or("default"),
                "state": {
                    "status": e.status.display_name(),
                    "step": e.step,
                    "turns": e.step,
                    "last_command": e.last_command,
                    "pid": e.pid,
                    "started_at": e.started_at,
                },
                "total_steps": e.step,
                // Registry rows are cross-process and carry no in-memory log
                // buffer, so the retention counters are reported as 0/0 rather
                // than being silently absent (audit 07, R7).
                "logs_retained": 0,
                "logs_dropped": 0,
            })
        }));
        rows
    }

    /// Who owns `id`, from the in-memory record or, for a worker that lives in
    /// another process, from its registry row.
    ///
    /// `None` means the pool and the registry both have no row for `id`: the
    /// caller keeps its own "not found" answer.
    pub async fn worker_owner(&self, id: &str) -> Option<WorkerOwner> {
        if let Some(owner) = self.workers.read().await.get(id).map(|w| w.owner.clone()) {
            return Some(WorkerOwner::Agent(owner));
        }
        // A finished worker the reaper dropped is still revisable, so its
        // saved conversation still names who may steer it.
        let owner = match load_registry_entry_in(&self.scratch, id) {
            Some(entry) => entry.owner,
            None => load_worker_history_in(&self.scratch, id).ok()?.owner,
        };
        Some(owner.map_or(WorkerOwner::Unattributed, WorkerOwner::Agent))
    }

    /// The registry row of `id`, from this process's records or the shared
    /// registry.
    ///
    /// The delegation check reads a target's owner, group and role from here:
    /// the row is written at dispatch and rewritten on every status change, so
    /// it is the one place all three facts are current together.
    pub async fn worker_row(&self, id: &str) -> Option<WorkerRegistryEntry> {
        load_registry_entry_in(&self.scratch, id)
    }

    /// Whether `id` finished as `completed`, here or in the shared registry.
    ///
    /// A consolidator may only integrate a worker that passed its own gate: a
    /// running or failed worker's branch is not a round contribution.
    pub async fn is_completed(&self, id: &str) -> bool {
        if let Some(state) = self.get_worker_state(id).await {
            return matches!(state, WorkerState::Completed { .. });
        }
        load_registry_entry_in(&self.scratch, id)
            .is_some_and(|entry| entry.status == RegistryStatus::Completed)
    }

    /// Integrate the named workers' branches into a consolidator's worktree.
    ///
    /// One `CONSOLIDATE_MERGE` request: each id is resolved, checked against
    /// [`check_consolidate_delegation`] and merged in order. The first conflict
    /// stops the run and leaves the rest skipped, so the model resolves one
    /// merge at a time, and the merges run on the harness through
    /// [`WorktreeGuard::merge_branch_at`] -- the sandbox holds no git
    /// credentials.
    ///
    /// Lives on the pool rather than in the turn engine because every fact it
    /// needs (the id resolver, the registry row, the completion status) is
    /// already here, and an integration test can then drive the whole loop
    /// without an LLM.
    pub async fn consolidate_merge(
        &self,
        actor: &WorkerMeta,
        worktree: &WorktreeGuard,
        ids: &[String],
    ) -> ConsolidateMerge {
        let mut lines = Vec::new();
        let mut integrated = false;
        let mut conflicted = false;
        for id in ids {
            if conflicted {
                lines.push(format!("{id} skipped (an earlier merge conflicted)"));
                continue;
            }
            let target = match self.resolve_worker_id(id, &actor.owner).await {
                Ok(target) => target,
                Err(e) => {
                    lines.push(format!("{id} refused: {e}"));
                    continue;
                }
            };
            let Some(entry) = self.worker_row(&target).await else {
                lines.push(format!("{id} refused: no such worker"));
                continue;
            };
            if let Err(reason) = check_consolidate_delegation(actor, &entry) {
                lines.push(format!("{id} refused: {reason}"));
                continue;
            }
            if !self.is_completed(&target).await {
                lines.push(format!("{id} refused: the worker is not completed"));
                continue;
            }
            let path = worktree.path.clone();
            let repo_root = worktree.repo_root.clone();
            let branch = worktree.branch.clone();
            let base_commit = worktree.base_commit.clone();
            let merged = tokio::task::spawn_blocking(move || {
                WorktreeGuard::merge_branch_at(&path, &repo_root, &branch, &base_commit, &target)
            })
            .await;
            match merged {
                Ok(Ok(BranchMerge::Merged { files })) => {
                    integrated = true;
                    lines.push(format!("{id} merged ({files} files)"));
                }
                Ok(Ok(BranchMerge::Conflicts { files })) => {
                    conflicted = true;
                    lines.push(format!(
                        "{id} conflict: {} (resolve the markers, then continue)",
                        files.join(", ")
                    ));
                }
                Ok(Err(e)) => lines.push(format!("{id} refused: {e}")),
                Err(e) => lines.push(format!("{id} refused: {e}")),
            }
        }
        ConsolidateMerge {
            observation: lines.join("\n"),
            integrated,
        }
    }

    /// Resolve a caller-supplied worker reference among `owner`'s own workers.
    ///
    /// Accepts the full id, `last` (the caller's most recently dispatched
    /// worker) and any unique prefix of at least [`MIN_WORKER_ID_PREFIX`]
    /// characters. The search spans this process's records and the shared
    /// registry but never leaves `owner`: another agent's worker is not a
    /// candidate, so a prefix that only matches one is `NotFound` rather than
    /// leaking its id. An exact id is taken as written whatever its owner,
    /// because ownership is the verb's business -- its refusal names the
    /// owning agent.
    pub async fn lookup_worker_id(&self, needle: &str, owner: &str) -> WorkerIdLookup {
        if needle == LAST_WORKER_ID {
            return self
                .last_dispatched_of(owner)
                .await
                .map_or(WorkerIdLookup::NotFound, WorkerIdLookup::Resolved);
        }
        // A prefix must never shadow a worker whose own id it exactly is.
        if self.worker_owner(needle).await.is_some() {
            return WorkerIdLookup::Resolved(needle.to_string());
        }
        if needle.chars().count() < MIN_WORKER_ID_PREFIX {
            return WorkerIdLookup::NotFound;
        }
        let mut matched: Vec<String> = self
            .owned_worker_ids(owner)
            .await
            .into_iter()
            .filter(|id| id.starts_with(needle))
            .collect();
        matched.sort();
        matched.dedup();
        match matched.len() {
            0 => WorkerIdLookup::NotFound,
            1 => WorkerIdLookup::Resolved(matched.remove(0)),
            _ => WorkerIdLookup::Ambiguous(matched),
        }
    }

    /// [`Self::lookup_worker_id`], refusing an ambiguous prefix with a message
    /// that lists only the caller's matching ids.
    ///
    /// One implementation for every caller that takes an id -- the MCP verbs
    /// and the `hub/watch` route the CLI uses -- so a prefix or `last` means
    /// the same thing everywhere.
    pub async fn resolve_worker_id(&self, needle: &str, owner: &str) -> Result<String> {
        match self.lookup_worker_id(needle, owner).await {
            WorkerIdLookup::Resolved(id) => Ok(id),
            // Nothing of the caller's matches: pass the needle through so the
            // verb keeps its own "not found" answer.
            WorkerIdLookup::NotFound => Ok(needle.to_string()),
            WorkerIdLookup::Ambiguous(ids) => anyhow::bail!(
                "worker id '{needle}' is ambiguous: matches {}",
                ids.join(", ")
            ),
        }
    }

    /// Ids of every worker `owner` dispatched, in this process and the registry.
    async fn owned_worker_ids(&self, owner: &str) -> Vec<String> {
        let mut ids: Vec<String> = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for worker in self.workers.read().await.values() {
            if worker.owner == owner && seen.insert(worker.id.clone()) {
                ids.push(worker.id.clone());
            }
        }
        for entry in load_all_registry_entries_in(&self.scratch) {
            if entry.owner.as_deref() == Some(owner) && seen.insert(entry.id.clone()) {
                ids.push(entry.id);
            }
        }
        ids
    }

    /// `owner`'s most recently dispatched worker.
    ///
    /// The in-process dispatch ordinal decides between workers that share a
    /// whole-second `started_at`; a registry-only row (a worker this process
    /// never dispatched, e.g. after a hub restart) falls back to its
    /// `started_at`.
    async fn last_dispatched_of(&self, owner: &str) -> Option<String> {
        let own: Vec<String> = self
            .workers
            .read()
            .await
            .values()
            .filter(|worker| worker.owner == owner)
            .map(|worker| worker.id.clone())
            .collect();
        let entries = load_all_registry_entries_in(&self.scratch);
        let started_at: HashMap<&str, u64> = entries
            .iter()
            .map(|entry| (entry.id.as_str(), entry.started_at))
            .collect();
        let ordinals = self
            .dispatch_order
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let mut best: Option<(u64, u64, String)> = None;
        let registry_ids = entries
            .iter()
            .filter(|entry| entry.owner.as_deref() == Some(owner))
            .map(|entry| &entry.id);
        for id in own.iter().chain(registry_ids) {
            let key = (
                started_at.get(id.as_str()).copied().unwrap_or(0),
                ordinals.get(id).copied().unwrap_or(0),
                id.clone(),
            );
            if best.as_ref().is_none_or(|current| &key > current) {
                best = Some(key);
            }
        }
        best.map(|(_, _, id)| id)
    }

    /// Ids of `owner`'s still-running workers, used to report a per-agent cap.
    pub async fn active_workers_of(&self, owner: &str) -> Vec<String> {
        let lock = self.workers.read().await;
        let mut ids: Vec<String> = lock
            .values()
            .filter(|w| {
                w.owner == owner
                    && matches!(
                        w.state,
                        WorkerState::Running { .. } | WorkerState::Paused { .. }
                    )
            })
            .map(|w| w.id.clone())
            .collect();
        drop(lock);
        ids.sort();
        ids
    }

    /// Deliver an orchestrator message to a worker.
    ///
    /// Three delivery paths, tried in order:
    ///
    /// 1. **In-process.** A `Running` worker queues the message on
    ///    `pending_steer` for its next turn; a `Paused` one is handed to its
    ///    resume channel.
    /// 2. **Revision.** A `Completed`/`Failed` worker (or a registry-only one,
    ///    e.g. after a hub restart) is relaunched on its preserved branch with
    ///    its saved conversation plus this message (see [`WorkerPool::revise`]).
    /// 3. **Cross-process mailbox.** The worker is owned by a different
    ///    `mini-swe-mcp` process (the normal `dispatch … --wait` case): the
    ///    message is appended atomically to its mailbox (see [`steer_path`]),
    ///    which the owner drains on every step.
    ///
    /// The `resume_tx` sender is extracted with `take()` under the write-guard
    /// and the guard is dropped *before* `send().await`: awaiting a full `mpsc`
    /// channel while holding the write-guard would serialise the whole pool.
    /// A missing sender is reported instead of silently dropping the guidance.
    ///
    /// `revision_turns` is the fresh turn budget of a revision (`None` takes
    /// [`DEFAULT_REVISION_TURNS`]); it is ignored for live workers.
    pub async fn steer(&self, id: &str, message: String) -> Result<SteerOutcome> {
        self.steer_with_budget(id, message, None).await
    }

    /// [`WorkerPool::steer`] with an explicit revision budget (the MCP `steer`
    /// `max_turns` argument): a finished worker restarts its loop from this
    /// many turns instead of [`DEFAULT_REVISION_TURNS`].
    ///
    /// Returns what it did, so the reply can only claim what happened: a live
    /// worker was `Queued` or `Resumed`, a stopped one was `Continuing` -- as
    /// a revision of its saved conversation, or cold when none survived.
    pub async fn steer_with_budget(
        &self,
        id: &str,
        message: String,
        revision_turns: Option<usize>,
    ) -> Result<SteerOutcome> {
        let tx_opt = {
            let mut lock = self.workers.write().await;
            let Some(w) = lock.get_mut(id) else {
                drop(lock);
                // Not in this process. The mailbox is only for a worker whose
                // registry row is live in another process: writing to one
                // nobody reads would silently swallow the guidance.
                if self.registry_row_live_elsewhere(id) {
                    // Queue it in the on-disk mailbox for the owning process
                    // (a blocking write, so outside the lock).
                    let root = self.scratch.clone();
                    let path = write_steer_message_in(&root, id, &message).map_err(|e| {
                        anyhow::anyhow!("Worker {id} is not in this process and its steering mailbox could not be written: {e}")
                    })?;
                    info!(
                        worker = %id,
                        path = %path.display(),
                        "Worker not in this process; steering message queued to mailbox"
                    );
                    return Ok(SteerOutcome::Queued);
                }
                // No live owner: the worker stopped for any reason -- completed,
                // failed, killed, or interrupted by a hub crash -- and is
                // continued here on its own id and branch.
                return self.continue_worker(id, message, revision_turns).await;
            };
            match &w.state {
                WorkerState::Running { .. } => {
                    w.pending_steer.push(message);
                    return Ok(SteerOutcome::Queued);
                }
                WorkerState::Paused { .. } => w.resume_tx.take(),
                WorkerState::Completed { .. } | WorkerState::Failed { .. } => {
                    // A finished worker cannot be resumed mid-turn -- it has no
                    // turn left -- so the message continues it below, outside
                    // the guard.
                    drop(lock);
                    return self.continue_worker(id, message, revision_turns).await;
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
        Ok(SteerOutcome::Resumed)
    }

    /// The commit a continuation measures its diff from: the recorded base,
    /// else the merge-base of the branch with the base branch.
    ///
    /// A history saved before base-branch tracking existed names no
    /// `base_branch`, and a legacy worker may name no `base_commit` either, so
    /// the merge-base with the branch the repo has checked out is the honest
    /// answer. `None` when neither can be resolved.
    pub(crate) async fn resolve_continuation_base(
        &self,
        repo_path: &std::path::Path,
        branch: &str,
        base_branch: Option<&str>,
    ) -> Option<String> {
        let base_branch = base_branch
            .map(str::to_string)
            .or_else(|| revision::detect_base_branch(repo_path))?;
        let repo = repo_path.to_path_buf();
        let base_branch = base_branch.clone();
        let branch = branch.to_string();
        tokio::task::spawn_blocking(move || {
            let out =
                crate::worktree::git(&repo, "merge-base", &["merge-base", &base_branch, &branch])
                    .ok()?;
            out.status
                .success()
                .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
        })
        .await
        .ok()
        .flatten()
        .filter(|s| !s.is_empty())
    }

    /// Id of every registry row left `interrupted` by a hub crash.
    ///
    /// The daemon asks this at startup to decide what to continue: a row is
    /// only a candidate when its conversation survived, because a cold
    /// continuation is the orchestrator's call, not an automatic one.
    pub async fn interrupted_workers(&self) -> Vec<String> {
        crate::pool::registry::interrupted_registry_entries_in(&self.scratch)
            .into_iter()
            .map(|e| e.id)
            .collect()
    }

    /// Automatic continuations still available for `id`, after counting the
    /// ones already spent.
    ///
    /// The counter lives in the history metadata (and the registry row), so it
    /// survives both a restart and a re-registration.
    pub async fn auto_continue_budget(&self, id: &str) -> usize {
        MAX_AUTO_CONTINUES.saturating_sub(self.auto_continues_spent(id).await)
    }

    /// Automatic continuations `id` has already spent.
    async fn auto_continues_spent(&self, id: &str) -> usize {
        crate::pool::load_worker_history_in(&self.scratch, id)
            .map(|h| h.auto_continues)
            .unwrap_or_else(|_| {
                crate::pool::load_registry_entry_in(&self.scratch, id)
                    .map(|e| e.auto_continues)
                    .unwrap_or(0)
            })
    }

    /// Count one automatic continuation against `id`'s cap.
    ///
    /// The counter is written to the history metadata (the durable store) and
    /// the registry row, so the daemon's next start sees the same count.
    pub async fn count_auto_continue(&self, id: &str) {
        let spent = self.auto_continues_spent(id).await + 1;
        let id_owned = id.to_string();
        let root = self.scratch.clone();
        let _ = tokio::task::spawn_blocking(move || {
            if let Ok(mut history) = crate::pool::load_worker_history_in(&root, &id_owned) {
                history.auto_continues = spent;
                let _ = crate::pool::save_worker_history_in(&root, &id_owned, &history);
            }
            if let Some(mut entry) = crate::pool::load_registry_entry_in(&root, &id_owned) {
                entry.auto_continues = spent;
                crate::pool::save_registry_entry_in(&root, &entry);
            }
        })
        .await;
    }

    /// Record where a running worker's worktree lives.
    ///
    /// Dropped again by [`WorkerPool::unregister_worktree`]; until then a
    /// `kill` knows where to commit the worker's uncommitted changes.
    pub(crate) async fn register_worktree(&self, worker_id: &str, path: PathBuf) {
        self.worktrees
            .write()
            .await
            .insert(worker_id.to_string(), path);
    }

    /// Forget a worker's worktree path once the worker is done with it.
    pub(crate) async fn unregister_worktree(&self, worker_id: &str) {
        self.worktrees.write().await.remove(worker_id);
    }

    /// Commit a worker's uncommitted worktree changes.
    ///
    /// Runs *before* the task is aborted, because aborting drops the worker's
    /// `WorktreeGuard`, which removes the checkout: a commit afterwards would
    /// find nothing left to save. Best effort by design — a kill reports
    /// whether it stopped the worker, never whether git agreed.
    async fn checkpoint_before_kill(&self, id: &str) {
        let Some(path) = self.worktrees.read().await.get(id).cloned() else {
            return;
        };
        if !path.is_dir() {
            return;
        }
        let worker_id = id.to_string();
        let committed = tokio::task::spawn_blocking(move || {
            WorktreeGuard::commit_all(
                &path,
                &format!("worker({worker_id}): checkpoint before kill"),
            )
        })
        .await;
        match committed {
            Ok(Ok(true)) => info!(
                worker = %id,
                "Committed the killed worker's uncommitted changes"
            ),
            // A clean worktree has nothing to preserve, which is not a problem.
            Ok(Ok(false)) => {}
            Ok(Err(e)) => warn!(
                worker = %id,
                error = %e,
                "Checkpoint before kill failed; the worktree still holds uncommitted changes"
            ),
            Err(e) => warn!(worker = %id, error = %e, "Checkpoint before kill could not run"),
        }
    }

    /// The registry row a killed worker must end on.
    ///
    /// A kill is a status transition like any other, so the row is written at
    /// once rather than coalesced: the monitor and crash recovery read these
    /// files, and a row left at `running` with a dead pid is only normalised
    /// to `stopped` by a reader that happens to look. `None` when the worker
    /// never wrote a row (a synthetic record), so nothing is invented.
    fn killed_entry(&self, worker: &WorkerRecord) -> Option<WorkerRegistryEntry> {
        let mut entry = self
            .registry
            .lock()
            .expect("registry lock poisoned")
            .entry(&worker.id)
            .cloned()?;
        entry.status = RegistryStatus::Failed;
        entry.step = worker.state.step();
        entry.last_command = match &worker.state {
            WorkerState::Failed { error, .. } => format!("error: {error}"),
            _ => return None,
        };
        entry.question = None;
        entry.metrics = worker.metrics;
        entry.updated_at = unix_timestamp();
        Some(entry)
    }

    /// Write the rows of the workers terminated in one pass, after the guard
    /// is gone: the registry row of a `kill` or of a hub shutdown.
    fn persist_kills(&self, entries: Vec<WorkerRegistryEntry>) {
        if entries.is_empty() {
            return;
        }
        let mut registry = self.registry.lock().expect("registry lock poisoned");
        for entry in entries {
            registry.save(entry);
        }
    }

    pub async fn kill(&self, id: &str) -> bool {
        self.checkpoint_before_kill(id).await;
        // The row is built under the guard and written after it is released:
        // a blocking file write must never sit inside a pool write-lock.
        let entry = {
            let mut lock = self.workers.write().await;
            let Some(w) = lock.get_mut(id) else {
                return false;
            };
            if let Some(handle) = w.handle.take() {
                handle.abort();
            }
            w.fail("Manually terminated by user/orchestrator");
            self.notify_change();
            self.killed_entry(w)
        };
        self.persist_kills(entry.into_iter().collect());
        true
    }

    /// Interrupt every live worker tracked by the pool, saving its work first.
    ///
    /// A hub shutdown is a *planned* exit, so a live worker ends `Interrupted`,
    /// not `Failed`: its uncommitted changes are committed onto its branch and
    /// the next daemon's recovery auto-continues it, exactly as it would after
    /// a crash. The transition is bounded — no running command is awaited, and
    /// each checkpoint commits once — and best effort, because a checkpoint
    /// that cannot run must never hold up the exit.
    pub async fn kill_all(&self) -> usize {
        // Commit each live worker's worktree before its abort drops the
        // `WorktreeGuard`, the same window `kill` uses. The ids are read under
        // a short guard so the checkpoints never run under a pool lock.
        let live: Vec<String> = {
            let lock = self.workers.read().await;
            lock.values()
                .filter(|worker| {
                    matches!(
                        worker.state,
                        WorkerState::Running { .. } | WorkerState::Paused { .. }
                    )
                })
                .map(|worker| worker.id.clone())
                .collect()
        };
        for id in &live {
            self.checkpoint_before_kill(id).await;
        }
        let (count, entries) = {
            let mut lock = self.workers.write().await;
            let mut count = 0usize;
            let mut entries = Vec::new();
            for worker in lock.values_mut() {
                if !matches!(
                    worker.state,
                    WorkerState::Running { .. } | WorkerState::Paused { .. }
                ) {
                    continue;
                }
                if let Some(handle) = worker.handle.take() {
                    handle.abort();
                }
                worker.fail(Self::shutdown_message(&worker.id));
                if let Some(entry) = self.interrupted_entry(worker) {
                    entries.push(entry);
                }
                self.notify_change();
                count += 1;
            }
            (count, entries)
        };
        self.persist_kills(entries);
        count
    }

    /// The `last_command` a hub shutdown leaves on an interrupted worker's row.
    ///
    /// Names the branch holding the salvaged work, so an operator sees where it
    /// went and why the worker can be continued.
    fn shutdown_message(id: &str) -> String {
        format!("hub stopped; work saved on branch worker-{id}")
    }

    /// The registry row a worker interrupted by a hub shutdown must end on.
    ///
    /// Like [`WorkerPool::killed_entry`], but a planned shutdown is not a
    /// failure: the row is `Interrupted` so the next daemon's recovery
    /// auto-continues it. `None` when the worker never wrote a row (a
    /// synthetic record), so nothing is invented.
    fn interrupted_entry(&self, worker: &WorkerRecord) -> Option<WorkerRegistryEntry> {
        let mut entry = self
            .registry
            .lock()
            .expect("registry lock poisoned")
            .entry(&worker.id)
            .cloned()?;
        entry.status = RegistryStatus::Interrupted;
        entry.step = worker.state.step();
        entry.last_command = Self::shutdown_message(&worker.id);
        entry.question = None;
        entry.metrics = worker.metrics;
        entry.updated_at = unix_timestamp();
        Some(entry)
    }

    /// Collect a worker's final result and release its in-memory resources.
    ///
    /// The retained window is *moved* out and narrowed to the emission budget,
    /// so one response never serializes the full history (audit 07, R4); the
    /// counters travel with the result so degradation stays visible (audit 07, R7).
    pub async fn collect(&self, id: &str) -> Option<CollectedWorker> {
        // The record is moved out and the write-guard released before the
        // worktree path is retired, so no `.await` runs under the guard.
        let record = {
            let mut lock = self.workers.write().await;
            let record = lock.remove(id)?;
            self.registry
                .lock()
                .expect("registry lock poisoned")
                .remove(id);
            self.notify_change();
            drop(lock);
            record
        };
        self.worktrees.write().await.remove(id);
        // The saved conversation stays: a collected worker is registry-only
        // from here on, and steering it must still revise it (same id, same
        // branch, full context). Only prune retires the history file.
        tracing::info!(worker = %id, "Worker collected and evicted from pool");
        let dropped = record.logs.dropped();
        let view = emit_view(&record.logs, self.log_policy.max_emitted);
        Some(CollectedWorker {
            id: record.id,
            task: record.task,
            model: record.model,
            owner: record.owner,
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

/// Auto-detect a sensible verify gate from the repository layout.
///
/// Returns `None` when no recognised build/test manifest is present, so a
/// worker on a plain repository is not forced through a gate that cannot run.
pub fn detect_verify_command(repo_path: &Path) -> Option<String> {
    if repo_path.join("Cargo.toml").is_file() {
        return Some("cargo build --all-targets && cargo test".to_string());
    }
    if repo_path.join("package.json").is_file() {
        // Only gate on `npm test` when the manifest actually declares a test
        // script; otherwise the gate would fail on a repo with no tests.
        if let Ok(raw) = std::fs::read_to_string(repo_path.join("package.json"))
            && let Ok(pkg) = serde_json::from_str::<serde_json::Value>(&raw)
            && pkg.get("scripts").and_then(|s| s.get("test")).is_some()
        {
            return Some("npm test".to_string());
        }
    }
    if repo_path.join("pyproject.toml").is_file() || repo_path.join("pytest.ini").is_file() {
        return Some("pytest -q".to_string());
    }
    // One manifest per ecosystem, in the order a polyglot repository is most
    // likely to mean: each probe is a single file-exists check, so adding an
    // ecosystem costs nothing when its files are absent.
    if repo_path.join("go.mod").is_file() {
        return Some("go test ./...".to_string());
    }
    if repo_path.join("pom.xml").is_file() {
        return Some("mvn -q test".to_string());
    }
    if repo_path.join("build.gradle").is_file() || repo_path.join("build.gradle.kts").is_file() {
        return Some("gradle test".to_string());
    }
    if repo_path.join("Makefile").is_file() || repo_path.join("makefile").is_file() {
        return Some("make test".to_string());
    }
    // A lockfile without a `package.json` test script still means npm: the
    // manifest probe above already declined a repo with no test script, so this
    // is the "the ecosystem is here, the gate is whatever npm runs" fallback.
    for manifest in [
        "package-lock.json",
        "pnpm-lock.yaml",
        "pnpm-workspace.yaml",
        "yarn.lock",
    ] {
        if repo_path.join(manifest).is_file() {
            return Some("npm test".to_string());
        }
    }
    for manifest in ["setup.py", "setup.cfg", "tox.ini", "requirements.txt"] {
        if repo_path.join(manifest).is_file() {
            return Some("pytest -q".to_string());
        }
    }
    if has_extension(repo_path, "csproj") || repo_path.join("global.json").is_file() {
        return Some("dotnet test".to_string());
    }
    None
}

/// Whether `dir` holds a file with the given extension.
fn has_extension(dir: &Path, extension: &str) -> bool {
    std::fs::read_dir(dir).is_ok_and(|entries| {
        entries
            .flatten()
            .any(|entry| entry.path().extension().is_some_and(|ext| ext == extension))
    })
}

#[cfg(test)]
mod verify_detection_tests {
    use super::detect_verify_command;
    use std::path::PathBuf;

    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("verify-detect-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create scratch dir");
        dir
    }

    #[test]
    fn a_cargo_manifest_selects_the_cargo_gate() {
        let dir = scratch("cargo");
        std::fs::write(dir.join("Cargo.toml"), "[package]\n").unwrap();
        assert_eq!(
            detect_verify_command(&dir).as_deref(),
            Some("cargo build --all-targets && cargo test")
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_package_json_without_a_test_script_is_not_gated() {
        let dir = scratch("npm-no-test");
        std::fs::write(dir.join("package.json"), r#"{"name":"x"}"#).unwrap();
        assert_eq!(detect_verify_command(&dir), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_package_json_with_a_test_script_selects_npm_test() {
        let dir = scratch("npm-test");
        std::fs::write(
            dir.join("package.json"),
            r#"{"name":"x","scripts":{"test":"jest"}}"#,
        )
        .unwrap();
        assert_eq!(detect_verify_command(&dir).as_deref(), Some("npm test"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_python_manifest_selects_pytest() {
        let dir = scratch("py");
        std::fs::write(dir.join("pyproject.toml"), "[project]\n").unwrap();
        assert_eq!(detect_verify_command(&dir).as_deref(), Some("pytest -q"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_bare_repository_is_not_gated() {
        let dir = scratch("bare");
        assert_eq!(detect_verify_command(&dir), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The gate is language-agnostic: every common ecosystem resolves to that
    /// ecosystem's own test command, and none of them assumes Rust.
    #[test]
    fn every_common_ecosystem_is_detected() {
        for (file, expected) in [
            ("go.mod", "go test ./..."),
            ("pom.xml", "mvn -q test"),
            ("build.gradle", "gradle test"),
            ("build.gradle.kts", "gradle test"),
            ("Makefile", "make test"),
            ("makefile", "make test"),
            ("setup.py", "pytest -q"),
            ("tox.ini", "pytest -q"),
            ("requirements.txt", "pytest -q"),
            ("package-lock.json", "npm test"),
            ("pnpm-lock.yaml", "npm test"),
            ("yarn.lock", "npm test"),
        ] {
            let dir = scratch(file);
            std::fs::write(dir.join(file), "{}\n").unwrap();
            assert_eq!(
                detect_verify_command(&dir).as_deref(),
                Some(expected),
                "{file} must select {expected}"
            );
            let _ = std::fs::remove_dir_all(&dir);
        }
    }

    #[test]
    fn a_dotnet_project_is_detected() {
        let dir = scratch("dotnet");
        std::fs::write(dir.join("app.csproj"), "<Project/>\n").unwrap();
        assert_eq!(detect_verify_command(&dir).as_deref(), Some("dotnet test"));
        let _ = std::fs::remove_dir_all(&dir);

        let dir = scratch("dotnet-global");
        std::fs::write(dir.join("global.json"), "{}\n").unwrap();
        assert_eq!(detect_verify_command(&dir).as_deref(), Some("dotnet test"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A repository that carries several ecosystems is gated on the first the
    /// detector recognises, so the choice is deterministic rather than
    /// dependent on directory iteration order.
    #[test]
    fn detection_is_deterministic_for_a_polyglot_repository() {
        let dir = scratch("polyglot");
        std::fs::write(dir.join("go.mod"), "module x\n").unwrap();
        std::fs::write(dir.join("pyproject.toml"), "[project]\n").unwrap();
        assert_eq!(
            detect_verify_command(&dir).as_deref(),
            Some("pytest -q"),
            "the python manifest is probed before go, and the order is fixed"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[cfg(test)]
mod consolidate_delegation_tests {
    use super::registry::{
        WorkerMeta, WorkerRegistryEntry, WorkerRole, check_consolidate_delegation,
    };
    use super::{RegistryStatus, WorkerMetrics};

    /// A consolidator's registry row, as its dispatch wrote it.
    fn consolidator(id: &str, owner: &str, group: &str) -> WorkerMeta {
        WorkerMeta {
            id: id.to_string(),
            task: "integrate the round".to_string(),
            group: Some(group.to_string()),
            role: WorkerRole::Consolidate,
            repo_path: None,
            owner: owner.to_string(),
            started_at: 0,
            pid: std::process::id(),
            revision: 0,
            auto_continues: 0,
            metrics: WorkerMetrics::default(),
        }
    }

    /// A target worker's registry row, as its own dispatch wrote it.
    fn target(id: &str, owner: &str, group: &str, role: WorkerRole) -> WorkerRegistryEntry {
        WorkerRegistryEntry {
            id: id.to_string(),
            pid: std::process::id(),
            task: "do the work".to_string(),
            model: "test".to_string(),
            status: RegistryStatus::Completed,
            step: 3,
            max_turns: 10,
            last_command: "completed".to_string(),
            question: None,
            started_at: 0,
            updated_at: 0,
            group: Some(group.to_string()),
            role,
            repo_path: None,
            owner: Some(owner.to_string()),
            metrics: WorkerMetrics::default(),
            base_branch: None,
            base_commit: None,
            revision: 0,
            auto_continues: 0,
        }
    }

    #[test]
    fn a_consolidator_integrates_its_owners_workers_in_its_group() {
        let actor = consolidator("c1", "agent-a", "round-1");
        assert_eq!(
            check_consolidate_delegation(
                &actor,
                &target("w1", "agent-a", "round-1", WorkerRole::Worker)
            ),
            Ok(())
        );
    }

    #[test]
    fn everything_outside_the_owner_and_group_is_refused_with_a_reason() {
        let actor = consolidator("c1", "agent-a", "round-1");
        for (label, entry) in [
            (
                "another agent's worker",
                target("w2", "agent-b", "round-1", WorkerRole::Worker),
            ),
            (
                "another group's worker",
                target("w3", "agent-a", "round-2", WorkerRole::Worker),
            ),
            (
                "itself",
                target("c1", "agent-a", "round-1", WorkerRole::Worker),
            ),
            (
                "another consolidator",
                target("c2", "agent-a", "round-1", WorkerRole::Consolidate),
            ),
            (
                "a row with no owner",
                target("w4", "agent-a", "round-1", WorkerRole::Worker).with_owner(None),
            ),
            (
                "a row with no group",
                target("w5", "agent-a", "round-1", WorkerRole::Worker).with_group(None),
            ),
        ] {
            let refused = check_consolidate_delegation(&actor, &entry);
            assert!(refused.is_err(), "{label} must be refused");
            assert!(
                refused.err().is_some_and(|reason| !reason.is_empty()),
                "{label} must be refused with a reason"
            );
        }
    }

    /// An ordinary worker holds no delegation at all, consolidator or not.
    #[test]
    fn an_ordinary_worker_may_not_integrate_anything() {
        let actor = consolidator("w6", "agent-a", "round-1").with_role(WorkerRole::Worker);
        assert_eq!(
            check_consolidate_delegation(
                &actor,
                &target("w1", "agent-a", "round-1", WorkerRole::Worker)
            ),
            Err("the caller is not a consolidator".to_string())
        );
    }

    impl WorkerRegistryEntry {
        fn with_owner(mut self, owner: Option<&str>) -> Self {
            self.owner = owner.map(str::to_string);
            self
        }

        fn with_group(mut self, group: Option<&str>) -> Self {
            self.group = group.map(str::to_string);
            self
        }
    }

    impl WorkerMeta {
        fn with_role(mut self, role: WorkerRole) -> Self {
            self.role = role;
            self
        }
    }
}
