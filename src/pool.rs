use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::{RwLock, Semaphore};
use tokio::task::JoinHandle;
use tracing::{error, info};

use crate::agent::{AgentRunner, AgentStepLog, ChatMessage, SYSTEM_PROMPT};
use crate::worktree::WorktreeGuard;

#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "state", content = "details")]
pub enum WorkerState {
    Running {
        step: usize,
        last_command: String,
        started_at: u64,
    },
    Completed {
        turns: usize,
        diff: String,
        summary: String,
        completed_at: u64,
    },
    Failed {
        error: String,
        step: usize,
        failed_at: u64,
    },
}

pub struct WorkerRecord {
    pub id: String,
    pub task: String,
    pub model: String,
    pub state: WorkerState,
    pub logs: Vec<AgentStepLog>,
    pub pending_steer: Vec<String>,
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

#[derive(Clone)]
pub struct WorkerPool {
    semaphore: Arc<Semaphore>,
    workers: Arc<RwLock<HashMap<String, WorkerRecord>>>,
    api_base: String,
    api_key: String,
}

impl WorkerPool {
    pub fn new(max_concurrent: usize, api_base: String, api_key: String) -> Self {
        Self {
            semaphore: Arc::new(Semaphore::new(max_concurrent)),
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
    ) -> Result<String> {
        let worker_id = uuid::Uuid::new_v4().to_string()[..8].to_string();
        let now = unix_timestamp();

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
            handle: None,
        };

        self.workers
            .write()
            .await
            .insert(worker_id.clone(), initial_record);

        let pool = self.clone();
        let wid = worker_id.clone();

        let join_handle = tokio::spawn(async move {
            if let Err(e) = pool
                .run_worker(wid.clone(), task, model, temperature, repo_path, max_turns)
                .await
            {
                error!(worker = %wid, error = %e, "Worker failed with error");
                let mut lock = pool.workers.write().await;
                if let Some(w) = lock.get_mut(&wid) {
                    w.fail(e.to_string());
                }
            }
        });

        // Store handle for potential cancellation
        if let Some(w) = self.workers.write().await.get_mut(&worker_id) {
            w.handle = Some(join_handle);
        }

        Ok(worker_id)
    }

    async fn run_worker(
        &self,
        worker_id: String,
        task: String,
        model: String,
        temperature: Option<f32>,
        repo_path: PathBuf,
        max_turns: usize,
    ) -> Result<()> {
        let _permit = self.semaphore.acquire().await.context("Semaphore closed")?;
        info!(worker = %worker_id, model = %model, "Starting worker execution");

        let worktree = WorktreeGuard::new(&repo_path, &worker_id)?;
        let runner = AgentRunner::new(
            self.api_base.clone(),
            self.api_key.clone(),
            model,
            temperature,
        );

        let mut messages = vec![
            ChatMessage {
                role: "system".into(),
                content: SYSTEM_PROMPT.into(),
            },
            ChatMessage {
                role: "user".into(),
                content: format!("TASK:\n{}\n\nBegin by exploring the repository.", task),
            },
        ];

        let mut step = 0;

        while step < max_turns {
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
                messages.push(ChatMessage {
                    role: "user".into(),
                    content: format!("STEER / ORCHESTRATOR GUIDANCE:\n{}", msg),
                });
            }

            let llm_reply = runner.run_step_llm(&messages).await?;
            let command = runner.extract_command(&llm_reply);

            let (cmd_str, is_finish) = match command {
                Some(ref cmd) if cmd.contains("COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT") => {
                    (cmd.clone(), true)
                }
                Some(cmd) => (cmd, false),
                None => (
                    "echo 'ERROR: No bash block found in previous response. You MUST output a ```bash block.'".into(),
                    false,
                ),
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

            if is_finish {
                info!(worker = %worker_id, step = step, "Worker requested completion");
                break;
            }

            info!(worker = %worker_id, step = step, op = %cmd_summary, "Subagent step");

            let (output, code) = runner.execute_bash(&worktree.path, &cmd_str).await?;

            let step_log = AgentStepLog {
                step,
                command: cmd_summary,
                output: if output.len() > 500 {
                    format!(
                        "{}... [{} bytes truncated]",
                        &output[..500],
                        output.len() - 500
                    )
                } else {
                    output.clone()
                },
                exit_code: code,
            };

            {
                let mut lock = self.workers.write().await;
                if let Some(w) = lock.get_mut(&worker_id) {
                    w.logs.push(step_log);
                }
            }

            messages.push(ChatMessage {
                role: "assistant".into(),
                content: llm_reply,
            });

            messages.push(ChatMessage {
                role: "user".into(),
                content: format!(
                    "COMMAND OUTPUT (exit code: {}):\n```\n{}\n```",
                    code.unwrap_or(-1),
                    output
                ),
            });
        }

        let diff = worktree.get_diff()?;
        let now = unix_timestamp();

        let mut lock = self.workers.write().await;
        if let Some(w) = lock.get_mut(&worker_id) {
            w.state = WorkerState::Completed {
                turns: step,
                diff,
                summary: format!("Finished after {} turns. Completed successfully.", step),
                completed_at: now,
            };
        }

        info!(worker = %worker_id, turns = step, "Worker completed successfully");
        Ok(())
    }

    pub async fn get_worker_state(&self, id: &str) -> Option<WorkerState> {
        self.workers.read().await.get(id).map(|w| w.state.clone())
    }

    pub async fn get_worker_logs(&self, id: &str) -> Option<Vec<AgentStepLog>> {
        self.workers.read().await.get(id).map(|w| w.logs.clone())
    }

    pub async fn list_workers(&self) -> Vec<serde_json::Value> {
        let lock = self.workers.read().await;
        lock.values()
            .map(|w| {
                serde_json::json!({
                    "id": w.id,
                    "task": w.task,
                    "model": w.model,
                    "state": w.state,
                    "total_steps": w.logs.len(),
                })
            })
            .collect()
    }

    pub async fn steer(&self, id: &str, message: String) -> Result<()> {
        let mut lock = self.workers.write().await;
        if let Some(w) = lock.get_mut(id) {
            match w.state {
                WorkerState::Running { .. } => {
                    w.pending_steer.push(message);
                    Ok(())
                }
                _ => anyhow::bail!("Worker {} is not in running state", id),
            }
        } else {
            anyhow::bail!("Worker not found: {}", id)
        }
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
}

fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn summarize_command(cmd: &str) -> String {
    let first_line = cmd.lines().next().unwrap_or("").trim();
    let words: Vec<&str> = first_line.split_whitespace().take(4).collect();
    let joined = words.join(" ");
    if joined.len() > 40 {
        format!("{}...", &joined[..37])
    } else if !joined.is_empty() {
        joined
    } else {
        "bash".to_string()
    }
}
