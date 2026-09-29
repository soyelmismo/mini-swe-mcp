use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
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

/// Maximum number of output bytes retained per step in the worker log history.
const MAX_STEP_LOG_BYTES: usize = 2048;

pub struct WorkerRecord {
    pub id: String,
    pub task: String,
    pub model: String,
    pub state: WorkerState,
    pub logs: Vec<AgentStepLog>,
    pub pending_steer: Vec<String>,
    pub resume_tx: Option<tokio::sync::mpsc::Sender<String>>,
    pub handle: Option<JoinHandle<()>>,
}

impl WorkerRecord {
    fn fail(&mut self, error: impl Into<String>) {
        self.state = WorkerState::Failed {
            error: error.into(),
            step: self.logs.len(),
            failed_at: unix_timestamp(),
        };
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
    pub logs: Vec<AgentStepLog>,
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

        info!(
            bash_slots,
            build_slots, "Bash and build execution semaphores initialized"
        );
        Self {
            semaphore: Arc::new(Semaphore::new(max_concurrent)),
            bash_semaphore: Arc::new(Semaphore::new(bash_slots)),
            build_semaphore: Arc::new(Semaphore::new(build_slots)),
            workers: Arc::new(RwLock::new(HashMap::new())),
            api_base,
            api_key,
        }
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
            logs: Vec::new(),
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

        self.workers
            .write()
            .await
            .insert(worker_id.clone(), initial_record);

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

            // 2. Check for REQUEST_TURNS sentinel in command or output
            if let Some(additional) = parse_request_turns(&cmd_str, &output) {
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

            // 3. Check for ASK_ORCHESTRATOR sentinel in command or output
            if let Some(question) = parse_ask_orchestrator(&cmd_str, &output) {
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

            // The log keeps a single owned copy of the (truncated) output; the
            // conversation entry below is built from that same copy instead of
            // cloning the raw output a second time.
            let logged_output = if output.len() > MAX_STEP_LOG_BYTES {
                let cut = output.floor_char_boundary(MAX_STEP_LOG_BYTES);
                format!(
                    "{}... [{} bytes truncated]",
                    &output[..cut],
                    output.len() - cut
                )
            } else {
                output
            };

            let step_log = AgentStepLog {
                step,
                command: cmd_summary,
                output: logged_output,
                exit_code: code,
            };

            // Built from the log copy before the record is moved into the pool,
            // so the raw output is never copied twice.
            let output_text = format!(
                "COMMAND OUTPUT (exit code: {}):\n```\n{}\n```",
                code.unwrap_or(-1),
                step_log.output
            );

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

    pub async fn get_worker_logs(&self, id: &str) -> Option<Vec<AgentStepLog>> {
        self.workers.read().await.get(id).map(|w| w.logs.clone())
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
    pub async fn take_worker_logs(&self, id: &str) -> Option<Vec<AgentStepLog>> {
        let mut lock = self.workers.write().await;
        lock.get_mut(id).map(|w| std::mem::take(&mut w.logs))
    }

    /// Take a worker's record out of the pool and return it untouched.
    ///
    /// Used by `collect`-style flows that need the full terminal payload
    /// (state *and* logs) in one shot, so the record is moved rather than
    /// cloned field by field.
    pub async fn take_worker(&self, id: &str) -> Option<CollectedWorker> {
        let record = self.workers.write().await.remove(id)?;
        Some(CollectedWorker {
            id: record.id,
            task: record.task,
            model: record.model,
            state: record.state,
            logs: record.logs,
        })
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
                    })
                })
                .collect()
        } else {
            let lock = self.workers.read().await;
            lock.values()
                .map(|w| {
                    serde_json::json!({
                        "id": w.id,
                        "task": w.task,
                        "model": w.model,
                        "state": w.state.to_summary(),
                        "total_steps": w.logs.len(),
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
    pub async fn collect(&self, id: &str) -> Option<CollectedWorker> {
        let mut lock = self.workers.write().await;
        let record = lock.remove(id)?;
        tracing::info!(worker = %id, "Worker collected and evicted from pool");
        Some(CollectedWorker {
            id: record.id,
            task: record.task,
            model: record.model,
            state: record.state,
            logs: record.logs,
        })
    }
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

pub fn parse_request_turns(cmd: &str, _output: &str) -> Option<usize> {
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

pub fn parse_ask_orchestrator(cmd: &str, _output: &str) -> Option<String> {
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
        assert_eq!(parse_request_turns("echo REQUEST_TURNS: 20", ""), Some(20));
        assert_eq!(parse_request_turns("printf 'REQUEST_TURNS: 15'", ""), Some(15));
        assert_eq!(parse_request_turns("cat file.rs", "REQUEST_TURNS: 15"), None);
        assert_eq!(parse_request_turns("echo nothing", "normal output"), None);
        assert_eq!(parse_request_turns("echo REQUEST_TURNS: 0", ""), None);
    }

    #[test]
    fn test_parse_ask_orchestrator() {
        assert_eq!(
            parse_ask_orchestrator("echo 'ASK_ORCHESTRATOR: should I delete old code?'", ""),
            Some("should I delete old code?".to_string())
        );
        assert_eq!(
            parse_ask_orchestrator("echo \"ASK_ORCHESTRATOR: is this ok?\"", ""),
            Some("is this ok?".to_string())
        );
        assert_eq!(
            parse_ask_orchestrator("cat src/agent.rs", "echo 'ASK_ORCHESTRATOR: <your specific question>'"),
            None
        );
        assert_eq!(
            parse_ask_orchestrator("echo 'ASK_ORCHESTRATOR: <your specific question>'", ""),
            None
        );
        assert_eq!(parse_ask_orchestrator("ls -la", "total 12"), None);
    }
}
