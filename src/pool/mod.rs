//! Worker pool: dispatch, tracking, steering, collection and reaping.
//!
//! The package is split by responsibility while keeping the historical
//! `mini_swe_mcp::pool::*` surface byte-for-byte identical through the
//! re-exports below:
//!
//! * [`buffer`] — bounded step-log retention window and emission view.
//! * [`state`] — worker lifecycle state, the in-memory record, progress views
//!   and the terminal-record TTL.
//! * [`registry`] — the on-disk JSON registry shared across processes.
//! * [`runner`] — the agent execution loop and the orchestrator sentinels.
//! * [`clock`] — the shared wall-clock helper.
//!
//! [`WorkerPool`] itself stays here: it owns the concurrency semaphores and the
//! worker map, and every operation on them (dispatch, collect, steer, kill,
//! reap) must stay in one place to keep the lock discipline auditable.

use anyhow::Result;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{RwLock, Semaphore};
use tokio::task::JoinHandle;
use tracing::{error, info};

mod buffer;
mod clock;
mod registry;
mod runner;
mod state;

pub use self::buffer::{
    DEFAULT_MAX_EMITTED_LOGS, DEFAULT_MAX_RETAINED_LOGS, EmittedLogs, LogBuffer,
    LogRetentionPolicy, LogStats, MAX_EMITTED_LOGS_CEILING, MAX_LOG_COMMAND_BYTES,
    MAX_LOG_OUTPUT_BYTES, MAX_RETAINED_LOGS_CEILING, build_step_log, clamp_string, emit_view,
    emit_view_with, worst_case_entry_bytes,
};
pub use self::clock::unix_timestamp;
pub use self::registry::{
    WorkerRegistryEntry, extract_group, load_all_registry_entries, registry_dir,
    remove_registry_entry, save_registry_entry,
};
pub use self::runner::{
    WorkerLaunchConfig, parse_ask_orchestrator, parse_request_turns, summarize_command,
};
pub use self::state::{
    CollectedWorker, DEFAULT_TERMINAL_TTL_SECS, WorkerPhase, WorkerProgress, WorkerRecord,
    WorkerState,
};

use self::state::expired_terminal_ids;

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

    #[allow(clippy::too_many_arguments)]
    pub async fn dispatch(
        &self,
        task: String,
        model: String,
        temperature: Option<f32>,
        repo_path: PathBuf,
        max_turns: usize,
        group: Option<String>,
        review_after: Option<String>,
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

        let repo_path_str = repo_path.to_string_lossy().to_string();

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
            repo_path: Some(repo_path_str.clone()),
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
        let repo_path_for_fail = repo_path_str;

        let config = WorkerLaunchConfig {
            task,
            model,
            temperature,
            repo_path,
            max_turns,
            group: resolved_group,
            review_after,
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
                    repo_path: Some(repo_path_for_fail),
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
