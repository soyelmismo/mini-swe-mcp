//! The multi-phase review auditor: an independent agent that re-runs the
//! quality gates over the implementation phase's work.
//!
//! Once the implementation loop stops, [`run_review_phase`] checkpoints the
//! worktree in git and then drives a *second* agent (a different model) over
//! the same worktree. The reviewer is deliberately not the implementation
//! agent: its contract is to inspect the diff, run the test suite and clippy,
//! fix whatever it finds, and only then emit the completion sentinel.
//!
//! It runs under its own turn budget ([`ReviewPhase::review_max_turns`],
//! resolved from the model manifest) and its own message history, so the
//! reviewer's context never mixes with the implementer's.

use anyhow::Result;
use tracing::{info, warn};

use crate::agent::{AgentRunner, ChatMessage, Role, SYSTEM_PROMPT};
use crate::worktree::WorktreeGuard;

use super::sentinels::summarize_command;
use super::super::WorkerPool;
use super::super::buffer::build_step_log;
use super::super::registry::{WorkerRegistryEntry, save_registry_entry};
use super::super::state::WorkerState;
use super::super::unix_timestamp;

/// Everything the review phase needs, and the step counter it hands back.
///
/// The counter flows out through [`ReviewPhaseOutcome::step`] because the
/// reviewer's turns are interleaved with the implementer's in a single
/// monotonic sequence that the registry and the completion record both report.
pub struct ReviewPhase {
    pub worker_id: String,
    pub reviewer_model: String,
    pub temperature: Option<f32>,
    pub task: String,
    /// The configured turn budget; `0` means "ask the manifest for the
    /// reviewer model's budget instead".
    pub max_turns: usize,
    /// The implementation loop's running turn budget, needed to report the
    /// combined `max_turns` while the worker is marked `reviewing`.
    pub current_max_turns: usize,
    pub group: String,
    pub repo_path_str: String,
    pub started_at_ts: u64,
    /// The implementer's turn counter entering the review phase.
    pub step: usize,
    /// The dispatch's declared network policy, applied to the reviewer's own
    /// bash steps too: a worker declared `offline` must not regain egress just
    /// because a second agent takes over the worktree.
    pub network_offline: bool,
}

/// The step counter after the review phase, for the caller to fold back into
/// its own loop state.
pub type ReviewPhaseOutcome = usize;

impl WorkerPool {
    /// Run the review auditor over the finished implementation.
    ///
    /// Returns the combined turn count. Errors propagate: a failed review bash
    /// command is a real failure, while a *failed LLM step* is not — the
    /// reviewer gives up quietly and the implementation result stands.
    pub(super) async fn run_review_phase(
        &self,
        worktree: &mut WorktreeGuard,
        review: ReviewPhase,
    ) -> Result<ReviewPhaseOutcome> {
        let ReviewPhase {
            worker_id,
            reviewer_model,
            temperature,
            task,
            max_turns,
            current_max_turns,
            group,
            repo_path_str,
            started_at_ts,
            mut step,
            network_offline,
        } = review;

    info!(
        worker = %worker_id,
        reviewer = %reviewer_model,
        "Implementation finished; starting multi-phase review pipeline"
    );

    // Checkpoint phase 1 implementation changes in git
    let _ = worktree.commit_changes(&format!(
        "worker({}): implementation phase completed (checkpoint)",
        worker_id
    ));

    let review_prompt = format!(
        "AUDIT & REVIEW PHASE:\nThe previous subagent implemented the following task:\n{}\n\n\
        YOUR OBJECTIVE AS THE INDEPENDENT REVIEWER:\n\
        1. Inspect changes: run `git status`, `git diff HEAD~1` (or `git log -1 -p`).\n\
        2. Run test suites and static checks (e.g. `cargo clippy --all-targets -- -D warnings`, `cargo test`, linters).\n\
        3. Fix any regressions, edge cases, dead code, orphan imports, or missed requirements.\n\
        4. When verified and 100% clean, execute:\n\
           echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
        task
    );

    let mut review_messages = vec![
        ChatMessage::text(Role::System, SYSTEM_PROMPT),
        ChatMessage::text(Role::User, review_prompt),
    ];

    let reviewer_runner = AgentRunner::new(
        self.api_base.clone(),
        self.api_key.clone(),
        reviewer_model.clone(),
        temperature,
    )
    .with_network_offline(network_offline);

    let manifest = crate::manifest::ModelManifest::load();
    let (_, _, reviewer_manifest_turns) = manifest.resolve_model(&reviewer_model);
    let review_max_turns = if max_turns > 0 {
        max_turns
    } else {
        reviewer_manifest_turns.unwrap_or(current_max_turns)
    };

    save_registry_entry(&WorkerRegistryEntry {
        id: worker_id.clone(),
        pid: std::process::id(),
        task: task.clone(),
        model: reviewer_model.clone(),
        status: "reviewing".into(),
        step,
        max_turns: current_max_turns + review_max_turns,
        last_command: "starting review phase".into(),
        question: None,
        started_at: started_at_ts,
        updated_at: unix_timestamp(),
        group: Some(group.clone()),
        repo_path: Some(repo_path_str.clone()),
    });

    let mut review_step = 0;

    while review_step < review_max_turns {
        review_step += 1;
        step += 1;

        let steer_msgs: Vec<String> = {
            let mut lock = self.workers.write().await;
            if let Some(w) = lock.get_mut(&worker_id) {
                std::mem::take(&mut w.pending_steer)
            } else {
                Vec::new()
            }
        };
        for msg in steer_msgs {
            review_messages.push(ChatMessage::text(
                Role::User,
                format!("ORCHESTRATOR GUIDANCE:\n{}", msg),
            ));
        }

        let mut llm_resp = match reviewer_runner.run_step_llm(&review_messages).await {
            Ok(resp) => resp,
            Err(e) => {
                warn!(
                    worker = %worker_id,
                    step,
                    error = %e,
                    "Reviewer LLM step failed; completing review phase"
                );
                return Ok(step);
            }
        };

        if llm_resp.command.is_none()
            && let Ok(retry_resp) = reviewer_runner.run_step_llm(&review_messages).await
            && retry_resp.command.is_some()
        {
            llm_resp = retry_resp;
        }

        let cmd_str = match llm_resp.command {
            Some(ref cmd) if cmd.contains("COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT") => {
                info!(
                    worker = %worker_id,
                    step = review_step,
                    "Reviewer completed and approved changes"
                );
                return Ok(step);
            }
            Some(ref cmd) => cmd.clone(),
            None => {
                review_messages.push(ChatMessage::text(
                    Role::User,
                    "ERROR: No bash command found. You MUST call the `bash` tool with your command.",
                ));
                continue;
            }
        };

        let cmd_summary = summarize_command(&cmd_str);
        let now = unix_timestamp();

        {
            let mut lock = self.workers.write().await;
            if let Some(w) = lock.get_mut(&worker_id) {
                w.state = WorkerState::Running {
                    step,
                    last_command: format!("[review] {}", cmd_summary),
                    started_at: now,
                };
            }
        }

        save_registry_entry(&WorkerRegistryEntry {
            id: worker_id.clone(),
            pid: std::process::id(),
            task: task.clone(),
            model: reviewer_model.clone(),
            status: "reviewing".into(),
            step,
            max_turns: current_max_turns + review_max_turns,
            last_command: format!("[review] {}", cmd_summary),
            question: None,
            started_at: started_at_ts,
            updated_at: now,
            group: Some(group.clone()),
            repo_path: Some(repo_path_str.clone()),
        });

        let (output, code) = reviewer_runner.execute_bash(&worktree.path, &cmd_str).await?;
        let output_text = format!(
            "COMMAND OUTPUT (exit code: {}):\n```\n{}\n```",
            code.unwrap_or(-1),
            output
        );

        let step_log = build_step_log(step, &format!("[review] {}", cmd_summary), output, code);
        {
            let mut lock = self.workers.write().await;
            if let Some(w) = lock.get_mut(&worker_id) {
                w.logs.push(step_log);
            }
        }

        if let (Some(tool_calls), Some(tc_id)) = (llm_resp.tool_calls, llm_resp.tool_call_id) {
            let content = if llm_resp.content.trim().is_empty() {
                None
            } else {
                Some(llm_resp.content)
            };
            review_messages.push(ChatMessage::assistant_with_tool_calls(content, tool_calls));
            review_messages.push(ChatMessage::tool_result(tc_id, &output_text));
        } else {
            let assistant_content = if llm_resp.content.trim().is_empty() {
                "I will execute a bash command.".to_string()
            } else {
                llm_resp.content
            };
            review_messages.push(ChatMessage::text(Role::Assistant, assistant_content));
            review_messages.push(ChatMessage::text(Role::User, output_text));
        }
    }

        Ok(step)
    }
}
