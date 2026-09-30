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
//! * [`turn`] — the unified turn engine shared by both loops.
//!
//! What stays here is only the implementer's turn loop, so that the sequence
//! "steer → warn → LLM step → bash → sentinel → record → next turn" is
//! readable end to end in one place.

use anyhow::{Context, Result};
use tracing::info;

use crate::agent::{AgentRunner, ChatMessage, Role};
use crate::manifest::build_system_prompt;
use crate::worktree::WorktreeGuard;

use super::registry::{RegistryStatus, WorkerMeta};
use super::state::WorkerState;
use super::steer::remove_steer_file;
use super::{WorkerPool, unix_timestamp};
use self::review::ReviewPhase;
use self::turn::{LlmErrorPolicy, TurnConfig, TurnEngine, TurnOutcome};

mod pause;
mod review;
mod sentinels;
mod turn;

pub use self::sentinels::{
    COMPLETION_SENTINEL, is_completion_request, parse_ask_orchestrator, parse_request_turns,
    summarize_command,
};

/// Everything the execution loop needs to start one worker.
pub struct WorkerLaunchConfig {
    pub task: String,
    pub model: String,
    pub temperature: Option<f32>,
    pub repo_path: std::path::PathBuf,
    pub max_turns: usize,
    pub group: String,
    pub review_after: Option<String>,
    /// Declared network policy: `true` confines every bash step to an
    /// isolated network namespace (`network: "offline"` on the dispatch).
    pub network_offline: bool,
    /// Optional shell command run through the same bash path before a
    /// completion sentinel is honoured. `None` disables the gate.
    pub verify: Option<String>,
}

/// Deletes a worker's steering mailbox when the worker exits.
///
/// Held as a local `let _steer_cleanup` for the whole duration of
/// [`WorkerPool::run_worker`]: the loop returns from a dozen places (the
/// completion sentinel, a failed bash step, a cancelled task, a propagated
/// error), and a `remove_steer_file` call in each of them is exactly the kind
/// of duplication that rots. A `Drop` impl cannot be forgotten on a new early
/// return.
struct SteerFileGuard(String);

impl SteerFileGuard {
    fn new(worker_id: String) -> Self {
        Self(worker_id)
    }
}

impl Drop for SteerFileGuard {
    fn drop(&mut self) {
        remove_steer_file(&self.0);
    }
}

impl WorkerPool {
    /// Take the guidance queued in this process for `worker_id`, if any.
    ///
    /// The in-memory half of the step loop's steering injection; the cross-
    /// process half is the [`drain_steer_messages`] mailbox polled alongside it.
    ///
    /// Split out of the loop so the implementation loop and the review loop
    /// cannot drift: both must read the same queue under the same lock
    /// discipline, and a record that has since been collected (a `kill` raced
    /// the loop) simply yields nothing. Exposed on the public pool so the merge
    /// of both sources is testable without an LLM round-trip.
    #[doc(hidden)]
    pub async fn take_pending_steer(&self, worker_id: &str) -> Vec<String> {
        let mut lock = self.workers.write().await;
        match lock.get_mut(worker_id) {
            Some(w) => std::mem::take(&mut w.pending_steer),
            None => Vec::new(),
        }
    }

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
            network_offline,
            verify,
        } = config;

        let repo_path_str = repo_path.to_string_lossy().to_string();
        let _permit = self.semaphore.acquire().await.context("Semaphore closed")?;
        info!(worker = %worker_id, model = %model, "Starting worker execution");

        // Dropped on *every* exit path -- completion, error, cancellation -- so
        // a finished worker never leaves a mailbox behind for a future worker
        // reusing the id to inherit as phantom guidance.
        let _steer_cleanup = SteerFileGuard::new(worker_id.clone());

        let mut worktree = WorktreeGuard::new(&repo_path, &worker_id)?;
        let runner = AgentRunner::new(
            self.api_base.clone(),
            self.api_key.clone(),
            model.clone(),
            temperature,
        )
        .with_network_offline(network_offline);

        // The system prompt carries this role's persistent memory
        // (`.agents/memory/<alias>.md`) when the repository provides any, so a
        // dispatch starts from what previous runs of the same role learned
        // instead of from the static prompt alone.
        let manifest = self.manifest();
        let memory_alias = manifest.alias_for_model(&model);
        let system_prompt = build_system_prompt(&repo_path, &memory_alias);

        let mut messages = vec![
            ChatMessage::text(Role::System, system_prompt),
            ChatMessage::text(Role::User, format!("TASK:\n{}\n\nBegin by exploring the repository.", task)),
        ];

        let mut step = 0;
        let mut current_max_turns = max_turns;
        let mut consecutive_no_cmd = 0;
        let mut last_assistant_text = String::new();
        let mut verify_failures = 0;
        let mut verified: Option<bool> = None;
        let started_at_ts = unix_timestamp();

        let meta = WorkerMeta {
            id: worker_id.clone(),
            task: task.clone(),
            group: Some(group.clone()),
            repo_path: Some(repo_path_str.clone()),
            started_at: started_at_ts,
            pid: std::process::id(),
        };

        while step < current_max_turns {
            step += 1;
            let max_turns_for_config = current_max_turns;
            let turn_config = TurnConfig {
                label_prefix: "",
                steer_prefix: "STEER / ORCHESTRATOR GUIDANCE:\n",
                apply_sentinels: true,
                llm_error_policy: LlmErrorPolicy::PauseForOrchestrator,
                status: RegistryStatus::Running,
                model: &model,
                max_turns: max_turns_for_config,
            };
            let mut engine = TurnEngine {
                pool: self,
                worktree: &mut worktree,
                runner: &runner,
                worker_id: &worker_id,
                task: &task,
                group: &group,
                repo_path_str: &repo_path_str,
                started_at_ts,
                meta: &meta,
                messages: &mut messages,
                step: &mut step,
                current_max_turns: &mut current_max_turns,
                last_assistant_text: &mut last_assistant_text,
                consecutive_no_cmd: &mut consecutive_no_cmd,
                verify: verify.as_deref(),
                verify_failures: &mut verify_failures,
            };
            match engine.run_turn(&turn_config).await? {
                TurnOutcome::Completed { verified: v } => {
                    verified = v;
                    break;
                }
                TurnOutcome::Continue | TurnOutcome::NoCommand => {}
                TurnOutcome::EndReview => unreachable!("implementer never ends review quietly"),
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
                        network_offline,
                    },
                )
                .await?;
        }

        let artifacts = worktree.sync_artifacts();
        if !artifacts.is_empty() {
            info!(
                worker = %worker_id,
                count = artifacts.len(),
                "Synchronized worker artifacts to repo root"
            );
        }

        let diff = worktree.get_diff()?;
        let now = unix_timestamp();

        let task_headline = task
            .lines()
            .map(str::trim)
            .find(|l| !l.is_empty())
            .unwrap_or("completed task");

        let agent_summary = last_assistant_text.trim();
        let summary = if !agent_summary.is_empty() {
            agent_summary.to_string()
        } else if !diff.trim().is_empty() {
            format!("{task_headline} (produced diff in {step} turns)")
        } else {
            format!("{task_headline} (completed in {step} turns)")
        };

        let branch = if !diff.trim().is_empty() {
            let commit_subject = if !agent_summary.is_empty() {
                let first_line = agent_summary.lines().next().unwrap_or(task_headline).trim();
                let stripped = first_line.trim_start_matches('#').trim();
                if stripped.is_empty() {
                    task_headline
                } else {
                    stripped
                }
            } else {
                task_headline
            };
            let clean_subject = if commit_subject.len() > 72 {
                let cut = commit_subject.floor_char_boundary(69);
                format!("{}...", &commit_subject[..cut])
            } else {
                commit_subject.to_string()
            };
            let commit_msg = format!("worker({worker_id}): {clean_subject}");
            worktree.commit_changes(&commit_msg).unwrap_or(None)
        } else {
            None
        };

        // A worker that exhausted its verification budget completes anyway but
        // is flagged: both the completion summary and the registry last_command
        // must say so, so the harness never mistakes it for a clean pass.
        let (summary, last_command) = if verified == Some(false) {
            (
                format!("{summary} (completed with failing verification)"),
                "completed with failing verification".to_string(),
            )
        } else {
            (summary, "completed".to_string())
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
            verified,
        };
        {
            let mut lock = self.workers.write().await;
            if let Some(w) = lock.get_mut(&worker_id) {
                w.state = completed_state;
            }
        }

        meta.save_status(
            &model,
            RegistryStatus::Completed,
            step,
            current_max_turns,
            &last_command,
            None,
        );

        info!(worker = %worker_id, turns = step, "Worker completed successfully");
        Ok(())
    }
}
