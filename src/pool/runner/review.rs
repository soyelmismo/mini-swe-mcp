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
use tracing::info;

use std::path::Path;

use crate::agent::{AgentRunner, ChatMessage, Role};
use crate::manifest::build_system_prompt;
use crate::worktree::WorktreeGuard;

use super::super::WorkerPool;
use super::super::registry::{RegistryStatus, WorkerMeta};
use super::turn::{LlmErrorPolicy, TurnConfig, TurnEngine, TurnOutcome};

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
            1. First run the full test and lint suite on the checkpoint (e.g. `cargo test --all-targets`, `cargo clippy --all-targets -- -D warnings`) and see it pass.\n\
            2. Inspect the whole diff since the base commit plus the working tree: run `git status`, `git diff HEAD~1` (or `git log -1 -p`) and `git diff`.\n\
            3. Fix real problems only: regressions, edge cases, dead code, orphan imports, or missed requirements.\n\
            4. Re-run the full test and lint suite and see it pass before completing.\n\
            5. When verified and 100% clean, execute:\n\
               echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
            task
        );

        let reviewer_runner = AgentRunner::new(
            self.api_base.clone(),
            self.api_key.clone(),
            reviewer_model.clone(),
            temperature,
        )
        .with_network_offline(network_offline);

        let manifest = self.manifest();
        let (_, _, reviewer_manifest_turns) = manifest.resolve_model(&reviewer_model);

        // The reviewer gets *its own* role memory, keyed by the reviewer alias, so
        // review lessons never bleed into the implementer's prompt (and vice versa).
        // `reviewer_model` may already be a resolved id (`combo:nerd`), so it is
        // mapped back to its alias first; an unknown id passes through unchanged and
        // simply finds no memory file.
        let reviewer_alias = manifest.alias_for_model(&reviewer_model);
        let mut review_messages = vec![
            ChatMessage::text(
                Role::System,
                build_system_prompt(Path::new(&repo_path_str), &reviewer_alias),
            ),
            ChatMessage::text(Role::User, review_prompt),
        ];
        let review_max_turns = if max_turns > 0 {
            max_turns
        } else {
            reviewer_manifest_turns.unwrap_or(current_max_turns)
        };

        let meta = WorkerMeta {
            id: worker_id.clone(),
            task: task.clone(),
            group: Some(group.clone()),
            repo_path: Some(repo_path_str.clone()),
            started_at: started_at_ts,
            pid: std::process::id(),
        };

        meta.save_status(
            &reviewer_model,
            RegistryStatus::Reviewing,
            step,
            current_max_turns + review_max_turns,
            "starting review phase",
            None,
        );

        let mut review_step = 0;
        let mut last_assistant_text = String::new();
        let mut consecutive_no_cmd = 0;
        let mut combined_max_turns = current_max_turns + review_max_turns;

        while review_step < review_max_turns {
            review_step += 1;
            step += 1;
            let max_turns_for_config = current_max_turns + review_max_turns;
            let turn_config = TurnConfig {
                label_prefix: "[review] ",
                steer_prefix: "ORCHESTRATOR GUIDANCE:\n",
                apply_sentinels: false,
                llm_error_policy: LlmErrorPolicy::EndQuietly,
                status: RegistryStatus::Reviewing,
                model: &reviewer_model,
                max_turns: max_turns_for_config,
            };
            let mut engine = TurnEngine {
                pool: self,
                worktree,
                runner: &reviewer_runner,
                worker_id: &worker_id,
                task: &task,
                group: &group,
                repo_path_str: &repo_path_str,
                started_at_ts,
                meta: &meta,
                messages: &mut review_messages,
                step: &mut step,
                current_max_turns: &mut combined_max_turns,
                last_assistant_text: &mut last_assistant_text,
                consecutive_no_cmd: &mut consecutive_no_cmd,
            };
            match engine.run_turn(&turn_config).await? {
                TurnOutcome::Completed => {
                    info!(
                        worker = %worker_id,
                        step = review_step,
                        "Reviewer completed and approved changes"
                    );
                    return Ok(step);
                }
                TurnOutcome::Continue | TurnOutcome::NoCommand => {}
                TurnOutcome::EndReview => return Ok(step),
            }
        }

        Ok(step)
    }
}
