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
//! "steer → warn → checkpoint → stagnation → LLM step → repeat check → bash →
//! sentinel → record → next turn" is readable end to end in one place.

use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::{Context, Result};
use tracing::{info, warn};

use crate::agent::{AgentRunner, ChatMessage, Role};
use crate::manifest::build_system_prompt;
use crate::worktree::{FileFingerprint, WorktreeGuard};

use super::registry::{RegistryStatus, WorkerMeta};
use super::state::WorkerState;
use super::steer::remove_steer_file;
use super::revision::{WorkerHistory, append_history_message};
use super::{WorkerPool, unix_timestamp};
use self::review::ReviewPhase;
use self::turn::{
    LlmErrorPolicy, ProgressWatch, TurnConfig, TurnEngine, TurnOutcome, shortstat_of,
};

pub(crate) mod history;
mod pause;
mod review;
mod sentinels;
mod turn;
pub(crate) use self::turn::parse_shortstat;

pub use self::sentinels::{
    COMPLETION_SENTINEL, is_completion_request, parse_ask_orchestrator, parse_request_turns,
    summarize_command,
};

/// Read-only half of [`WorkerLaunchConfig`] for the phase loop: the caller owns
/// the worktree and the conversation, so a failure anywhere still leaves both
/// available for history persistence.
pub struct RunConfig<'a> {
    pub task: &'a str,
    pub model: &'a str,
    pub temperature: Option<f32>,
    pub max_turns: usize,
    pub review_after: Option<String>,
    pub network_offline: bool,
    pub verify: Option<&'a str>,
    pub repo_path_str: &'a str,
}

/// Everything the execution loop needs to start one worker.
pub struct WorkerLaunchConfig {
    pub task: String,
    pub model: String,
    pub temperature: Option<f32>,
    pub repo_path: std::path::PathBuf,
    pub max_turns: usize,
    pub review_after: Option<String>,
    /// Declared network policy: `true` confines every bash step to an
    /// isolated network namespace (`network: "offline"` on the dispatch).
    pub network_offline: bool,
    /// Optional shell command run through the same bash path before a
    /// completion sentinel is honoured. `None` disables the gate.
    pub verify: Option<String>,
    /// Conversation a revision continues: the finished worker's history plus
    /// the orchestrator's revision request. `None` starts a fresh dispatch,
    /// which builds its own system prompt and task message.
    pub resume_messages: Option<Vec<ChatMessage>>,
    /// Base commit of the run that produced `resume_messages`. `Some` makes the
    /// worktree re-attach to the worker's preserved branch instead of creating
    /// a fresh one, so a revision keeps its id, its branch and its checkpoints.
    pub resume_base_commit: Option<String>,
    pub resume_base_branch: Option<String>,
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

    /// Run one worker to completion.
    ///
    /// `meta` is the dispatch's registry row, lent in so the phase loop can
    /// move the health counters on it at every status write; the caller reads
    /// them back if the worker fails.
    pub(super) async fn run_worker(
        &self,
        worker_id: String,
        config: WorkerLaunchConfig,
        meta: &mut WorkerMeta,
    ) -> Result<()> {
        let WorkerLaunchConfig {
            task,
            model,
            temperature,
            repo_path,
            max_turns,
            review_after,
            network_offline,
            verify,
            resume_messages,
            resume_base_commit,
            resume_base_branch,
        } = config;

        let repo_path_str = repo_path.to_string_lossy().to_string();
        let _permit = self.worker_slots.acquire(&meta.owner).await;
        let revision = resume_base_commit.is_some();
        info!(worker = %worker_id, model = %model, revision, "Starting worker execution");

        // Dropped on *every* exit path -- completion, error, cancellation -- so
        // a finished worker never leaves a mailbox behind for a future worker
        // reusing the id to inherit as phantom guidance.
        let _steer_cleanup = SteerFileGuard::new(worker_id.clone());

        // A revision re-attaches to the branch the previous run committed to,
        // so the worker keeps its id, its checkpoints and its diff base; a
        // fresh dispatch creates the branch instead. Checkout shells out to
        // git and walks the tree, so it runs off the runtime thread.
        let resume_base_commit = resume_base_commit.clone();
        let repo_path_owned = repo_path.clone();
        let worker_id_owned = worker_id.clone();
        let mut worktree = tokio::task::spawn_blocking(move || match &resume_base_commit {
            Some(base) => {
                let mut guard = WorktreeGuard::reopen(&repo_path_owned, &worker_id_owned, base)?;
                guard.base_branch = resume_base_branch;
                Ok(guard)
            },
            None => WorktreeGuard::new(&repo_path_owned, &worker_id_owned),
        })
        .await
        .context("Worktree checkout task failed")??;
        // A kill must not lose what this worker leaves uncommitted, and the
        // guard that owns the checkout dies with the task a kill aborts, so the
        // pool keeps the path and commits through it (see `WorkerPool::kill`).
        self.register_worktree(&worker_id, worktree.path.clone()).await;

        // The system prompt carries this role's persistent memory
        // (`.agents/memory/<alias>.md`) when the repository provides any, so a
        // dispatch starts from what previous runs of the same role learned
        // instead of from the static prompt alone.
        let manifest = self.manifest();
        let memory_alias = manifest.alias_for_model(&model);
        let system_prompt = build_system_prompt(&repo_path, &memory_alias);

        // A revision replays the finished worker's conversation (system prompt,
        // task, every assistant turn with its reasoning and every tool result)
        // and appends nothing here: the revision request is already its last
        // user message, so the model continues exactly where it left off.
        let mut messages = match resume_messages {
            Some(replayed) => replayed,
            None => vec![
                ChatMessage::text(Role::System, system_prompt),
                ChatMessage::text(
                    Role::User,
                    format!("TASK:\n{}\n\nBegin by exploring the repository.", task),
                ),
            ],
        };

        // The opening messages are the log's first lines, so a worker that dies
        // before its first turn still leaves a continuable conversation behind.
        let opening_meta = WorkerHistory {
            task: task.clone(),
            group: meta.group.clone(),
            model: model.clone(),
            temperature,
            repo_path: repo_path_str.clone(),
            base_commit: worktree.base_commit.clone(),
            base_branch: worktree.base_branch.clone(),
            branch: worktree.branch.clone(),
            network_offline,
            verify: verify.clone(),
            max_turns,
            review_after: review_after.clone(),
            revision: meta.revision,
            auto_continues: meta.auto_continues,
            owner: Some(meta.owner.clone()),
            messages: Vec::new(),
        };
        if !super::revision::history_log_path(&worker_id).exists()
            && let Err(e) = Self::append_history_messages(&worker_id, &opening_meta, &messages)
        {
            warn!(
                worker = %worker_id,
                error = %e,
                "Could not persist the worker conversation; this worker can no longer be revised"
            );
        }

        // The conversation is durable one line per message (see
        // `TurnEngine::push_message`), so nothing is rewritten here: a crash may
        // lose only the in-flight turn.
        self.run_phases(
            &worker_id,
            &RunConfig {
                task: &task,
                model: &model,
                temperature,
                max_turns,
                review_after: review_after.clone(),
                network_offline,
                verify: verify.as_deref(),
                repo_path_str: &repo_path_str,
            },
            meta,
            &mut worktree,
            &mut messages,
        )
        .await
    }

    /// The implementer loop, the review phase and the completion payload.
    ///
    /// Append `messages` to `worker_id`'s history log, creating it with the
    /// metadata line when it does not exist yet.
    fn append_history_messages(
        worker_id: &str,
        meta: &WorkerHistory,
        messages: &[ChatMessage],
    ) -> anyhow::Result<()> {
        for msg in messages {
            append_history_message(worker_id, meta, msg)?;
        }
        Ok(())
    }

    /// Split from [`WorkerPool::run_worker`] so the caller owns the worktree and
    /// the conversation: whatever happens in here -- a completion sentinel, a
    /// failed bash step, a cancelled task -- the caller still holds both and can
    /// persist them.
    async fn run_phases(
        &self,
        worker_id: &str,
        config: &RunConfig<'_>,
        meta: &mut WorkerMeta,
        worktree: &mut WorktreeGuard,
        messages: &mut Vec<ChatMessage>,
    ) -> Result<()> {
        let task = config.task.to_string();
        let model = config.model.to_string();
        let temperature = config.temperature;
        let max_turns = config.max_turns;
        let review_after = config.review_after.clone();
        let network_offline = config.network_offline;
        let verify = config.verify.map(|v| v.to_string());
        let repo_path_str = config.repo_path_str.to_string();

        let runner = AgentRunner::new(
            self.api_base.clone(),
            self.api_key.clone(),
            model.clone(),
            temperature,
        )
        .with_network_offline(network_offline);

        let mut step = 0;
        let mut current_max_turns = max_turns;
        let mut consecutive_no_cmd = 0;
        let mut last_assistant_text = String::new();
        let mut watch = ProgressWatch::default();
        let mut verified: Option<bool> = None;

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
                task: &task,
                temperature,
                review_after: review_after.as_deref(),
                network_offline,
            };
            let mut engine = TurnEngine {
                pool: self,
                worktree,
                runner: &runner,
                worker_id,
                meta,
                messages,
                unsaved_messages: Vec::new(),
                step: &mut step,
                current_max_turns: &mut current_max_turns,
                last_assistant_text: &mut last_assistant_text,
                consecutive_no_cmd: &mut consecutive_no_cmd,
                verify: verify.as_deref(),
                dispatch_max_turns: max_turns,
                watch: &mut watch,
            };
            match engine.run_turn(&turn_config).await? {
                TurnOutcome::Completed { verified: v } => {
                    verified = v;
                    break;
                }
                TurnOutcome::Continue | TurnOutcome::NoCommand => {}
                TurnOutcome::EndReview => unreachable!("implementer never ends review quietly"),
            }
            // One line per message, flushed at the turn boundary: a crash can
            // only lose the turn that was in flight.
            engine.flush_history_log(&turn_config).await;
        }

        // --- MULTI-PHASE REVIEW PIPELINE ---
        // The implementer's loop is done; hand off to the independent auditor
        // and fold its turns back into the single monotonic step counter.
        if let Some(reviewer_model) = review_after {
            step = self
                .run_review_phase(
                    worktree,
                    ReviewPhase {
                        worker_id: worker_id.to_string(),
                        reviewer_model,
                        temperature,
                        task: task.clone(),
                        max_turns,
                        current_max_turns,
                        repo_path_str: repo_path_str.clone(),
                        step,
                        network_offline,
                        meta,
                    },
                )
                .await?;
        }

        // The artifact sync, the final diff and the final commit all shell
        // out to git and walk files, so the whole tail runs off the runtime
        // thread on owned copies of the guard's paths.
        let task_headline: String = task
            .lines()
            .map(str::trim)
            .find(|l| !l.is_empty())
            .unwrap_or("completed task")
            .to_string();
        let agent_summary = last_assistant_text.trim().to_string();
        let path = worktree.path.clone();
        let repo_root = worktree.repo_root.clone();
        let base_commit = worktree.base_commit.clone();
        let base_branch = worktree.base_branch.clone();
        let branch = worktree.branch.clone();
        let seeded = worktree.seeded();
        let metrics_path = path.clone();
        let metrics_base = base_commit.clone();
        let metrics_base_branch = base_branch.clone();
        let (artifacts, diff, summary, branch, now) = tokio::task::spawn_blocking(move || {
            finalize_worktree(FinalizeInput {
                path,
                repo_root,
                base_commit,
                base_branch,
                branch,
                seeded,
                task_headline,
                agent_summary,
                step,
            })
        })
        .await
        .context("Worktree finalization task failed")??;
        worktree.preserve_branch = worktree.preserve_branch || branch.is_some();
        if !artifacts.is_empty() {
            info!(
                worker = %worker_id,
                count = artifacts.len(),
                "Synchronized worker artifacts to repo root"
            );
        }

        // Health counters measured once, at the end: the turn total the record
        // reports and the size of the diff it produced. `git` is a blocking
        // subprocess, so the shortstat sample is taken off the runtime.
        meta.metrics.turns_used = step;
        if let Some((files, insertions, deletions)) =
            shortstat_of(&metrics_path, &metrics_base, metrics_base_branch.as_deref()).await
        {
            meta.metrics.diff_files = files;
            meta.metrics.diff_insertions = insertions;
            meta.metrics.diff_deletions = deletions;
        }


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
        // The completion carries the revision that produced it: a fresh
        // dispatch completes at zero, a revised worker at its attempt number.
        let revision = self
            .workers
            .read()
            .await
            .get(worker_id)
            .map(|w| w.revision)
            .unwrap_or(0);
        let completed_state = WorkerState::Completed {
            turns: step,
            diff,
            summary,
            completed_at: now,
            artifacts,
            branch,
            verified,
            metrics: meta.metrics,
            revision,
        };
        self.update_worker(worker_id, |w| w.state = completed_state)
            .await;

        self.save_status(
            meta,
            &model,
            RegistryStatus::Completed,
            step,
            current_max_turns,
            &last_command,
            None,
        );

        self.unregister_worktree(worker_id).await;
        info!(worker = %worker_id, turns = step, "Worker completed successfully");
        Ok(())
    }
}

/// Everything the worker's final tail needs, owned so it can cross into a
/// `spawn_blocking` thread.
struct FinalizeInput {
    path: PathBuf,
    repo_root: PathBuf,
    base_commit: String,
    base_branch: Option<String>,
    branch: String,
    seeded: BTreeMap<String, FileFingerprint>,
    task_headline: String,
    agent_summary: String,
    step: usize,
}

/// Sync the worker's artifacts, take its final diff and commit it.
///
/// The worker's finished output: synced artifacts, final diff, completion
/// summary, the branch the commit landed on (`None` when there was nothing to
/// commit) and the completion timestamp.
type FinalizedWork = (Vec<String>, String, String, Option<String>, u64);

/// Sync the worker's artifacts, take its final diff and commit it.
fn finalize_worktree(input: FinalizeInput) -> Result<FinalizedWork> {
    let FinalizeInput {
        path,
        repo_root,
        base_commit,
        base_branch,
        branch,
        seeded,
        task_headline,
        agent_summary,
        step,
    } = input;
    if WorktreeGuard::merge_in_progress_at(&path)? {
        anyhow::bail!("Base merge is unresolved; worker cannot complete");
    }
    let artifacts = WorktreeGuard::sync_artifacts_at(&path, &repo_root, &seeded);
    let diff = WorktreeGuard::diff_with_base_at(&path, &base_commit, base_branch.as_deref())?;
    let summary = if !agent_summary.is_empty() {
        agent_summary.to_string()
    } else if !diff.trim().is_empty() {
        format!("{task_headline} (produced diff in {step} turns)")
    } else {
        format!("{task_headline} (completed in {step} turns)")
    };
    let committed = {
        let commit_subject: &str = if !agent_summary.is_empty() {
            let first_line = agent_summary.lines().next().unwrap_or(&task_headline).trim();
            let stripped = first_line.trim_start_matches('#').trim();
            if stripped.is_empty() {
                &task_headline
            } else {
                stripped
            }
        } else {
            &task_headline
        };
        let clean_subject = if commit_subject.len() > 72 {
            let cut = commit_subject.floor_char_boundary(69);
            format!("{}...", &commit_subject[..cut])
        } else {
            commit_subject.to_string()
        };
        let commit_msg = format!("worker({branch}): {clean_subject}");
        WorktreeGuard::commit_changes_at(&path, &repo_root, &branch, &base_commit, &commit_msg)?
    };
    Ok((artifacts, diff, summary, committed, unix_timestamp()))
}
