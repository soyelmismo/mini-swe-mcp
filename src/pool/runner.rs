//! The worker execution loop: prompt assembly, stepping, and the orchestrator
//! control sentinels.
//!
//! [`run_worker`](run_worker) is the agent loop driven by a pool permit: it
//! builds the conversation, executes each bash command inside the worker
//! worktree, records the bounded step log, and honours the two control
//! sentinels ([`parse_request_turns`] and [`parse_ask_orchestrator`]).

use anyhow::{Context, Result};
use tracing::info;

use crate::agent::{AgentRunner, ChatMessage, Role, SYSTEM_PROMPT};
use crate::worktree::WorktreeGuard;

use super::buffer::build_step_log;
use super::registry::{WorkerRegistryEntry, save_registry_entry};
use super::state::WorkerState;
use super::{WorkerPool, unix_timestamp};

/// Everything the execution loop needs to start one worker.
pub struct WorkerLaunchConfig {
    pub task: String,
    pub model: String,
    pub temperature: Option<f32>,
    pub repo_path: std::path::PathBuf,
    pub max_turns: usize,
    pub group: String,
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
}

#[cfg(test)]
mod tests {
    use super::{parse_ask_orchestrator, parse_request_turns, summarize_command};

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
