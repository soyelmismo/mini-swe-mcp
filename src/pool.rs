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
            WorkerState::Completed { turns, summary, completed_at, .. } => serde_json::json!({
                "status": "Completed",
                "turns": turns,
                "summary": summary,
                "completed_at": completed_at,
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

/// Result of a one-shot worker collection, detached from the live pool.
#[derive(Debug, Clone, serde::Serialize)]
pub struct CollectedWorker {
    pub id: String,
    pub task: String,
    pub model: String,
    pub state: WorkerState,
    pub logs: Vec<AgentStepLog>,
}

#[derive(Clone)]
pub struct WorkerPool {
    semaphore: Arc<Semaphore>,
    bash_semaphore: Arc<Semaphore>,
    workers: Arc<RwLock<HashMap<String, WorkerRecord>>>,
    api_base: String,
    api_key: String,
}

impl WorkerPool {
    pub fn new(max_concurrent: usize, api_base: String, api_key: String) -> Self {
        let default_slots = std::thread::available_parallelism()
            .map(|n| (n.get() / 2).max(1))
            .unwrap_or(2);
        let bash_slots = std::env::var("BASH_CONCURRENT_LIMIT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default_slots);
        info!(bash_slots, "Bash execution semaphore initialized");
        Self {
            semaphore: Arc::new(Semaphore::new(max_concurrent)),
            bash_semaphore: Arc::new(Semaphore::new(bash_slots)),
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
            resume_tx: None,
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
            ChatMessage::text("system", SYSTEM_PROMPT),
            ChatMessage::text("user", format!("TASK:\n{}\n\nBegin by exploring the repository.", task)),
        ];

        let mut step = 0;
        let mut current_max_turns = max_turns;
        let mut consecutive_no_cmd = 0;
        let mut last_assistant_text = String::new();

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
                    "user",
                    format!("STEER / ORCHESTRATOR GUIDANCE:\n{}", msg),
                ));
            }

            // Proactive turn warning when approaching limit (at 5 and 2 turns remaining)
            let remaining = current_max_turns.saturating_sub(step);
            if remaining == 5 || remaining == 2 {
                info!(worker = %worker_id, step, current_max_turns, "Injecting proactive turn limit warning");
                messages.push(ChatMessage::text(
                    "user",
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
                        "assistant",
                        if llm_resp.content.trim().is_empty() {
                            "I will execute a bash command.".into()
                        } else {
                            llm_resp.content
                        },
                    ));
                    messages.push(ChatMessage::text(
                        "user",
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

            info!(worker = %worker_id, step = step, op = %cmd_summary, "Subagent step");

            let (output, code) = {
                let _bash_permit = self.bash_semaphore.acquire().await
                    .context("Bash semaphore closed")?;
                runner.execute_bash(&worktree.path, &cmd_str).await?
            };

            // 2. Check for REQUEST_TURNS sentinel in command or output
            if let Some(additional) = parse_request_turns(&cmd_str, &output) {
                let old_max = current_max_turns;
                current_max_turns = (current_max_turns + additional).min(150);
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
                        "user",
                        format!("ORCHESTRATOR RESPONSE / GUIDANCE:\n{}", answer),
                    ));
                }
            }

            let step_log = AgentStepLog {
                step,
                command: cmd_summary,
                output: if output.len() > 2048 {
                    let cut = output.floor_char_boundary(2048);
                    format!(
                        "{}... [{} bytes truncated]",
                        &output[..cut],
                        output.len() - cut
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

            let output_text = format!(
                "COMMAND OUTPUT (exit code: {}):\n```\n{}\n```",
                code.unwrap_or(-1),
                output
            );

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
                messages.push(ChatMessage::text("assistant", assistant_content));
                messages.push(ChatMessage::text("user", output_text));
            }
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

        let mut lock = self.workers.write().await;
        if let Some(w) = lock.get_mut(&worker_id) {
            w.state = WorkerState::Completed {
                turns: step,
                diff,
                summary,
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
                    "state": w.state.to_summary(),
                    "total_steps": w.logs.len(),
                })
            })
            .collect()
    }

    pub async fn steer(&self, id: &str, message: String) -> Result<()> {
        let mut lock = self.workers.write().await;
        if let Some(w) = lock.get_mut(id) {
            match &mut w.state {
                WorkerState::Running { .. } => {
                    w.pending_steer.push(message);
                    Ok(())
                }
                WorkerState::Paused { .. } => {
                    if let Some(tx) = w.resume_tx.take() {
                        let _ = tx.send(message).await;
                    }
                    Ok(())
                }
                _ => anyhow::bail!("Worker {} is not in a steerable state (running or paused)", id),
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

    /// Terminate every worker currently tracked by the pool.
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

fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

pub fn summarize_command(cmd: &str) -> String {
    let first_line = cmd.lines().next().unwrap_or("").trim();
    let words: Vec<&str> = first_line.split_whitespace().take(4).collect();
    let joined = words.join(" ");
    if joined.len() > 40 {
        let cut = joined.floor_char_boundary(37);
        format!("{}...", &joined[..cut])
    } else if !joined.is_empty() {
        joined
    } else {
        "bash".to_string()
    }
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
