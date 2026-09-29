//! The primary worker execution loop: prompt assembly, stepping, and the
//! orchestrator control sentinels.
//!
//! [`run_worker`](run_worker) is the agent loop driven by a pool permit: it
//! builds the conversation, executes each bash command inside the worker
//! worktree, records the bounded step log, and honours the two control
//! sentinels ([`parse_request_turns`] and [`parse_ask_orchestrator`]).
//!
//! The package is split by responsibility, keeping the historical
//! `mini_swe_mcp::pool::{parse_ask_orchestrator, parse_request_turns,
//! summarize_command}` surface identical through the re-exports below:
//!
//! * [`sentinels`] — the pure parsers for the orchestrator control protocol
//!   plus the bounded command label.
//! * [`review`] — the independent multi-phase review auditor that runs after
//!   the implementation loop finishes.
//!
//! What stays here is only the implementer's turn loop, so that the sequence
//! "steer → warn → LLM step → bash → sentinel → record → next turn" is
//! readable end to end in one place.

use anyhow::{Context, Result};
use tracing::{info, warn};

use crate::agent::{AgentRunner, ChatMessage, Role, SYSTEM_PROMPT};
use crate::worktree::WorktreeGuard;

use super::buffer::build_step_log;
use super::registry::{WorkerRegistryEntry, save_registry_entry};
use super::state::WorkerState;
use super::{WorkerPool, unix_timestamp};
use self::pause::PauseRequest;
use self::review::ReviewPhase;

mod pause;
mod review;
mod sentinels;

pub use self::sentinels::{parse_ask_orchestrator, parse_request_turns, summarize_command};

/// Everything the execution loop needs to start one worker.
pub struct WorkerLaunchConfig {
    pub task: String,
    pub model: String,
    pub temperature: Option<f32>,
    pub repo_path: std::path::PathBuf,
    pub max_turns: usize,
    pub group: String,
    pub review_after: Option<String>,
}

impl WorkerPool {
    pub(super) async fn run_worker(
        &self,
        worker_id: String,
        config: WorkerLaunchConfig,
    ) -> Result<()> {
        let WorkerLaunchConfig {
            task,
            model,
            temperature,
            repo_path,
            max_turns,
            group,
            review_after,
        } = config;

        let repo_path_str = repo_path.to_string_lossy().to_string();
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

            // 1. Run LLM step with automatic pause and checkpoint on network/API failure
            let mut llm_resp = match runner.run_step_llm(&messages).await {
                Ok(resp) => resp,
                Err(e) => {
                    // Safe checkpoint of uncommitted worktree changes so work is never lost
                    let _ = worktree.commit_changes(&format!(
                        "worker({}): checkpoint step {} before pause (error: {})",
                        worker_id, step, e
                    ));

                    warn!(
                        worker = %worker_id,
                        step,
                        error = %e,
                        "LLM step failed after retries; pausing worker for orchestrator resume"
                    );

                    let question = format!(
                        "LLM API error (step {}): {}. Send steer/resume to retry.",
                        step, e
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
                        last_command: format!("paused_on_error: {e}"),
                        question: Some(question.clone()),
                        started_at: started_at_ts,
                        updated_at: now,
                        group: Some(group.clone()),
                        repo_path: Some(repo_path_str.clone()),
                    });

                    // Wait for orchestrator resume via steer
                    if let Some(resume_msg) = rx.recv().await {
                        info!(worker = %worker_id, msg = %resume_msg, "Worker resumed after error by orchestrator");
                        {
                            let mut lock = self.workers.write().await;
                            if let Some(w) = lock.get_mut(&worker_id) {
                                w.state = WorkerState::Running {
                                    step,
                                    last_command: format!("resumed: {}", summarize_command(&resume_msg)),
                                    started_at: unix_timestamp(),
                                };
                                w.resume_tx = None;
                            }
                        }
                        if !resume_msg.trim().is_empty() && resume_msg.trim() != "resume" {
                            messages.push(ChatMessage::text(
                                Role::User,
                                format!("ORCHESTRATOR GUIDANCE:\n{}", resume_msg),
                            ));
                        }
                        // Re-run the LLM step now that network/connectivity is restored
                        runner.run_step_llm(&messages).await?
                    } else {
                        return Err(e);
                    }
                }
            };

            if llm_resp.command.is_none() {
                info!(
                    worker = %worker_id,
                    "No command found (tool_calls or code block); discarding and silently retrying once without warning"
                );
                if let Ok(retry_resp) = runner.run_step_llm(&messages).await
                    && retry_resp.command.is_some()
                {
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
                repo_path: Some(repo_path_str.clone()),
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
                let answer = self
                    .pause_for_orchestrator(PauseRequest {
                        worker_id: &worker_id,
                        question: &question,
                        step,
                        max_turns: current_max_turns,
                        last_command: &cmd_summary,
                        task: &task,
                        model: &model,
                        group: &group,
                        repo_path_str: &repo_path_str,
                        started_at_ts,
                    })
                    .await?;

                if let Some(answer) = answer {
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


        // --- MULTI-PHASE REVIEW PIPELINE ---
        // The implementer's loop is done; hand off to the independent auditor
        // and fold its turns back into the single monotonic step counter.
        if let Some(reviewer_model) = review_after {
            step = self
                .run_review_phase(
                    &mut worktree,
                    ReviewPhase {
                        worker_id: worker_id.clone(),
                        reviewer_model,
                        temperature,
                        task: task.clone(),
                        max_turns,
                        current_max_turns,
                        group: group.clone(),
                        repo_path_str: repo_path_str.clone(),
                        started_at_ts,
                        step,
                    },
                )
                .await?;
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
            repo_path: Some(repo_path_str.clone()),
        });

        info!(worker = %worker_id, turns = step, "Worker completed successfully");
        Ok(())
    }
}
