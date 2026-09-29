//! Unified turn engine shared by the implementer and reviewer agent loops.
//!
//! Both [`run_worker`](super::run_worker) and
//! [`run_review_phase`](super::review::run_review_phase) drive the same
//! per-turn sequence — steer drain, LLM call, command extraction, semaphore
//! gated bash execution, history push, in-memory state and registry update.
//! The only differences are the label prefix, the steering prefix text,
//! whether the orchestrator control sentinels apply, and how an LLM API error
//! is handled. Those four knobs live in [`TurnConfig`]; everything else is
//! shared here so the two loops cannot drift.

use anyhow::{Context, Result};
use tracing::{info, warn};

use crate::agent::{AgentRunner, ChatMessage, LlmResponse, Role, ToolCall};
use crate::worktree::WorktreeGuard;

use super::super::buffer::build_step_log;
use super::super::registry::{RegistryStatus, WorkerMeta};
use super::super::state::WorkerState;
use super::super::steer::drain_steer_messages;
use super::super::WorkerPool;
use super::pause::PauseRequest;
use super::sentinels::{
    is_completion_request, parse_ask_orchestrator, parse_request_turns, summarize_command,
};

/// How the engine handles an LLM API error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum LlmErrorPolicy {
    /// Checkpoint, pause for orchestrator, and resume on steer.
    PauseForOrchestrator,
    /// Log and end the phase quietly.
    EndQuietly,
}

/// Per-phase configuration for the turn engine.
pub(super) struct TurnConfig<'a> {
    /// Prefix for the command label in the registry and step log.
    pub label_prefix: &'a str,
    /// Prefix for steering messages injected into the conversation.
    pub steer_prefix: &'a str,
    /// Whether REQUEST_TURNS / ASK_ORCHESTRATOR sentinels apply.
    pub apply_sentinels: bool,
    /// How an LLM API error is handled.
    pub llm_error_policy: LlmErrorPolicy,
    /// Registry status to record for this phase.
    pub status: RegistryStatus,
    /// Model name for the registry entry.
    pub model: &'a str,
    /// Combined turn budget reported in the registry.
    pub max_turns: usize,
}

/// Outcome of one turn.
pub(super) enum TurnOutcome {
    /// The completion sentinel was found; stop the loop.
    Completed,
    /// A command was executed; continue the loop.
    Continue,
    /// No command was found; history was updated; continue the loop.
    NoCommand,
    /// The reviewer hit an LLM error and should end quietly.
    EndReview,
}

/// Shared state for one turn of the agent loop.
pub(super) struct TurnEngine<'a> {
    pub pool: &'a WorkerPool,
    pub worktree: &'a mut WorktreeGuard,
    pub runner: &'a AgentRunner,
    pub worker_id: &'a str,
    pub task: &'a str,
    pub group: &'a str,
    pub repo_path_str: &'a str,
    pub started_at_ts: u64,
    pub meta: &'a WorkerMeta,
    pub messages: &'a mut Vec<ChatMessage>,
    pub step: &'a mut usize,
    pub current_max_turns: &'a mut usize,
    pub last_assistant_text: &'a mut String,
    pub consecutive_no_cmd: &'a mut usize,
}

impl<'a> TurnEngine<'a> {
    /// Run one turn of the agent loop.
    pub(super) async fn run_turn(&mut self, config: &TurnConfig<'_>) -> Result<TurnOutcome> {
        // --- Steering drain ---
        let mut steer_msgs = self.pool.take_pending_steer(self.worker_id).await;
        let remote = drain_steer_messages(self.worker_id);
        if !remote.is_empty() {
            info!(
                worker = %self.worker_id,
                count = remote.len(),
                "Drained cross-process steering messages from mailbox"
            );
            steer_msgs.extend(remote);
        }
        for msg in steer_msgs {
            info!(worker = %self.worker_id, "Injected steering message into subagent turn");
            self.messages.push(ChatMessage::text(
                Role::User,
                format!("{}{}", config.steer_prefix, msg),
            ));
        }

        // --- Proactive turn warning (implementer only) ---
        if config.apply_sentinels {
            let remaining = self.current_max_turns.saturating_sub(*self.step);
            if remaining == 5 || remaining == 2 {
                info!(
                    worker = %self.worker_id,
                    step = *self.step,
                    max_turns = *self.current_max_turns,
                    "Injecting proactive turn limit warning"
                );
                self.messages.push(ChatMessage::text(
                    Role::User,
                    format!(
                        "TURN LIMIT WARNING: You have used {} of {} turns ({} remaining). If you need more turns to complete testing or refactoring, execute `echo \"REQUEST_TURNS: <number>\"` now. Otherwise, wrap up your changes and execute `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.",
                        *self.step, *self.current_max_turns, remaining
                    ),
                ));
            }
        }

        // --- LLM call with error handling ---
        let llm_resp = match self.runner.run_step_llm(self.messages).await {
            Ok(resp) => resp,
            Err(e) => {
                match config.llm_error_policy {
                    LlmErrorPolicy::EndQuietly => {
                        warn!(
                            worker = %self.worker_id,
                            step = *self.step,
                            error = %e,
                            "Reviewer LLM step failed; completing review phase"
                        );
                        return Ok(TurnOutcome::EndReview);
                    }
                    LlmErrorPolicy::PauseForOrchestrator => {
                        // Safe checkpoint of uncommitted worktree changes so work is never lost
                        let _ = self.worktree.commit_changes(&format!(
                            "worker({}): checkpoint step {} before pause (error: {})",
                            self.worker_id, *self.step, e
                        ));

                        warn!(
                            worker = %self.worker_id,
                            step = *self.step,
                            error = %e,
                            "LLM step failed after retries; pausing worker for orchestrator resume"
                        );

                        let question = format!(
                            "LLM API error (step {}): {}. Send steer/resume to retry.",
                            *self.step, e
                        );

                        let answer = self
                            .pool
                            .pause_for_orchestrator(PauseRequest {
                                worker_id: self.worker_id,
                                question: &question,
                                step: *self.step,
                                max_turns: *self.current_max_turns,
                                last_command: &format!("paused_on_error: {e}"),
                                task: self.task,
                                model: config.model,
                                group: self.group,
                                repo_path_str: self.repo_path_str,
                                started_at_ts: self.started_at_ts,
                            })
                            .await?;

                        match answer {
                            Some(resume_msg) => {
                                info!(
                                    worker = %self.worker_id,
                                    msg = %resume_msg,
                                    "Worker resumed after error by orchestrator"
                                );
                                if !resume_msg.trim().is_empty() && resume_msg.trim() != "resume" {
                                    self.messages.push(ChatMessage::text(
                                        Role::User,
                                        format!("ORCHESTRATOR GUIDANCE:\n{}", resume_msg),
                                    ));
                                }
                                // Re-run the LLM step now that network/connectivity is restored
                                self.runner.run_step_llm(self.messages).await?
                            }
                            None => return Err(e),
                        }
                    }
                }
            }
        };

        self.process_response(config, llm_resp).await
    }

    /// Process an LLM response: extract the command, execute it, and update
    /// history, state, and registry.
    async fn process_response(
        &mut self,
        config: &TurnConfig<'_>,
        llm_resp: LlmResponse,
    ) -> Result<TurnOutcome> {
        // --- Command extraction ---
        let cmd_str = match llm_resp.command {
            Some(ref cmd) if is_completion_request(cmd) => {
                info!(
                    worker = %self.worker_id,
                    step = *self.step,
                    "Worker requested completion"
                );
                if !llm_resp.content.trim().is_empty() {
                    *self.last_assistant_text = llm_resp.content.clone();
                }
                return Ok(TurnOutcome::Completed);
            }
            Some(ref cmd) => {
                *self.consecutive_no_cmd = 0;
                if !llm_resp.content.trim().is_empty() {
                    *self.last_assistant_text = llm_resp.content.clone();
                }
                cmd.clone()
            }
            None => {
                info!(
                    worker = %self.worker_id,
                    step = *self.step,
                    "No bash command in response; prompting subagent directly"
                );
                self.push_no_command_history(
                    &llm_resp.content,
                    llm_resp.reasoning_content,
                    llm_resp.tool_calls,
                );
                if *self.consecutive_no_cmd < 2 {
                    *self.consecutive_no_cmd += 1;
                    *self.step = self.step.saturating_sub(1);
                }
                return Ok(TurnOutcome::NoCommand);
            }
        };

        let cmd_summary = summarize_command(&cmd_str);
        let label = format!("{}{}", config.label_prefix, cmd_summary);

        // --- In-memory state update ---
        {
            let mut lock = self.pool.workers.write().await;
            if let Some(w) = lock.get_mut(self.worker_id)
                && let WorkerState::Running {
                    step: ref mut s,
                    ref mut last_command,
                    ..
                } = w.state
            {
                *s = *self.step;
                *last_command = label.clone();
            }
        }

        // --- Registry update through WorkerMeta::save_status ---
        self.meta.save_status(
            config.model,
            config.status,
            *self.step,
            config.max_turns,
            &label,
            None,
        );

        info!(
            worker = %self.worker_id,
            step = *self.step,
            op = %cmd_summary,
            "Subagent step"
        );

        // --- Execute command with semaphores ---
        let (output, code) = {
            let is_heavy = crate::agent::is_heavy_command(&cmd_str);
            let _build_permit = if is_heavy {
                Some(
                    self.pool
                        .build_semaphore
                        .acquire()
                        .await
                        .context("Build semaphore closed")?,
                )
            } else {
                None
            };
            let _bash_permit = self
                .pool
                .bash_semaphore
                .acquire()
                .await
                .context("Bash semaphore closed")?;
            self.runner.execute_bash(&self.worktree.path, &cmd_str).await?
        };

        // --- Orchestrator control sentinels (implementer only) ---
        if config.apply_sentinels {
            // REQUEST_TURNS
            if let Some(additional) = parse_request_turns(&cmd_str) {
                let old_max = *self.current_max_turns;
                *self.current_max_turns =
                    (*self.current_max_turns + additional).max(*self.current_max_turns).min(500);
                info!(
                    worker = %self.worker_id,
                    requested = additional,
                    old_max,
                    new_max = *self.current_max_turns,
                    "Subagent requested turn extension; granted"
                );
            }

            // ASK_ORCHESTRATOR
            if let Some(question) = parse_ask_orchestrator(&cmd_str) {
                let answer = self
                    .pool
                    .pause_for_orchestrator(PauseRequest {
                        worker_id: self.worker_id,
                        question: &question,
                        step: *self.step,
                        max_turns: *self.current_max_turns,
                        last_command: &cmd_summary,
                        task: self.task,
                        model: config.model,
                        group: self.group,
                        repo_path_str: self.repo_path_str,
                        started_at_ts: self.started_at_ts,
                    })
                    .await?;

                if let Some(answer) = answer {
                    self.messages.push(ChatMessage::text(
                        Role::User,
                        format!("ORCHESTRATOR RESPONSE / GUIDANCE:\n{}", answer),
                    ));
                }
            }
        }

        if llm_resp.invalid_utf8_lines > 0 {
            info!(
                worker = %self.worker_id,
                step = *self.step,
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
        // rather than cloned, and both text fields are clamped to a hard
        // ceiling.
        let step_log = build_step_log(*self.step, &label, output, code);

        {
            let mut lock = self.pool.workers.write().await;
            if let Some(w) = lock.get_mut(self.worker_id) {
                w.logs.push(step_log);
            }
        }

        // --- History push ---
        if let (Some(tool_calls), Some(tc_id)) = (llm_resp.tool_calls, llm_resp.tool_call_id) {
            // OpenAI tool_calls protocol: assistant with tool_calls → tool response
            let content = if llm_resp.content.trim().is_empty() {
                None
            } else {
                Some(llm_resp.content)
            };
            let msg = ChatMessage::assistant_with_tool_calls(content, tool_calls)
                .with_reasoning_content(llm_resp.reasoning_content);
            self.messages.push(msg);
            self.messages.push(ChatMessage::tool_result(tc_id, &output_text));
        } else {
            // Fallback: code-block models use plain assistant + user messages
            let assistant_content = if llm_resp.content.trim().is_empty() {
                "I will execute a bash command.".to_string()
            } else {
                llm_resp.content
            };
            let msg = ChatMessage::text(Role::Assistant, assistant_content)
                .with_reasoning_content(llm_resp.reasoning_content);
            self.messages.push(msg);
            self.messages.push(ChatMessage::text(Role::User, output_text));
        }

        Ok(TurnOutcome::Continue)
    }

    /// Push the protocol-correct history for a response with no usable command.
    ///
    /// Rule a: tool_calls present but no parseable command → assistant with
    /// tool_calls + reasoning, then one tool_result per call id.
    /// Rule b: text only, no command → assistant text + reasoning, then user
    /// ERROR message.
    fn push_no_command_history(
        &mut self,
        content: &str,
        reasoning: Option<String>,
        tool_calls: Option<Vec<ToolCall>>,
    ) {
        if let Some(tool_calls) = tool_calls {
            let content = if content.trim().is_empty() {
                None
            } else {
                Some(content.to_string())
            };
            let msg = ChatMessage::assistant_with_tool_calls(content, tool_calls.clone())
                .with_reasoning_content(reasoning);
            self.messages.push(msg);
            for tc in tool_calls {
                let args = tc.function.arguments;
                let truncated = if args.len() > 200 {
                    let cut = args.floor_char_boundary(197);
                    format!("{}...", &args[..cut])
                } else {
                    args
                };
                self.messages.push(ChatMessage::tool_result(
                    tc.id,
                    format!(
                        "ERROR: could not parse a `command` from the bash tool arguments: {truncated}"
                    ),
                ));
            }
        } else {
            let assistant_content = if content.trim().is_empty() {
                "I will execute a bash command.".to_string()
            } else {
                content.to_string()
            };
            let msg = ChatMessage::text(Role::Assistant, assistant_content)
                .with_reasoning_content(reasoning);
            self.messages.push(msg);
            self.messages.push(ChatMessage::text(
                Role::User,
                "ERROR: No bash command found. You MUST call the `bash` tool with your command.",
            ));
        }
    }
}
