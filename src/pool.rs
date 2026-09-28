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
    pub handle: Option<JoinHandle<()>>,
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
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

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
            handle: None,
        };

        self.workers.write().await.insert(worker_id.clone(), initial_record);

        let pool = self.clone();
        let wid = worker_id.clone();

        let join_handle = tokio::spawn(async move {
            if let Err(e) = pool.run_worker(wid.clone(), task, model, temperature, repo_path, max_turns).await {
                error!(worker = %wid, error = %e, "Worker failed with error");
                let mut lock = pool.workers.write().await;
                if let Some(w) = lock.get_mut(&wid) {
                    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
                    w.state = WorkerState::Failed {
                        error: e.to_string(),
                        step: w.logs.len(),
                        failed_at: now,
                    };
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
        let runner = AgentRunner::new(self.api_base.clone(), self.api_key.clone(), model, temperature);

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

            // Update running state
            {
                let mut lock = self.workers.write().await;
                if let Some(w) = lock.get_mut(&worker_id) {
                    if let WorkerState::Running { step: ref mut s, ref mut last_command, .. } = w.state {
                        *s = step;
                        *last_command = cmd_str.clone();
                    }
                }
            }

            if is_finish {
                info!(worker = %worker_id, step = step, "Worker requested completion");
                break;
            }

            let (output, code) = runner.execute_bash(&worktree.path, &cmd_str).await?;

            let step_log = AgentStepLog {
                step,
                command: cmd_str,
                output: output.clone(),
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
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();

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

    pub async fn kill(&self, id: &str) -> bool {
        let mut lock = self.workers.write().await;
        if let Some(w) = lock.get_mut(id) {
            if let Some(handle) = w.handle.take() {
                handle.abort();
            }
            let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
            w.state = WorkerState::Failed {
                error: "Manually terminated by user/orchestrator".into(),
                step: w.logs.len(),
                failed_at: now,
            };
            true
        } else {
            false
        }
    }
}
