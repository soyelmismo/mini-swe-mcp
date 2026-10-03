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

pub use self::context_pack::{
    PACK_CAP_BYTES, context_pack, extract_identifiers, extract_paths, outline_file,
};
use self::review::ReviewPhase;
pub use self::review::{
    ReviewMode, SecurityReviewOutcome, SecurityScope, approved_merged_branches, parse_findings,
    plan_review, review_prompt, scope_for,
};
use self::turn::{
    LlmErrorPolicy, ProgressWatch, TurnConfig, TurnEngine, TurnOutcome, shortstat_of,
};
use super::registry::{RegistryStatus, WorkerMeta, WorkerRole};
use super::revision::{WorkerHistory, append_history_message_in};
use super::state::{TURN_BUDGET_EXHAUSTED, WorkerState};
use super::steer::remove_steer_file_in;
use super::{WorkerPool, unix_timestamp};
use crate::worktree::ScratchRoot;

pub(crate) mod context_pack;
pub(crate) mod divergent;
pub(crate) mod history;
mod pause;
mod review;
mod sentinels;
mod turn;
pub(crate) use self::turn::parse_shortstat;

pub(crate) use self::sentinels::strip_markup;

pub use self::sentinels::{
    COMPLETION_SENTINEL, CONSOLIDATE_WAIT_DEFAULT_SECS, CONSOLIDATE_WAIT_MAX_SECS,
    HARNESS_WAIT_PREFIX, REPORT_FOLLOWUP, is_completion_request, parse_ask_orchestrator,
    parse_consolidate_merge, parse_consolidate_steer, parse_consolidate_wait, parse_kill_job,
    parse_report, parse_request_turns, parse_wait_job, summarize_command, summary_line,
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
    /// The dispatcher's ambient environment for the differential verify run.
    pub client_env: &'a [(String, String)],
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
    /// The dispatcher's ambient environment, filtered by the sandbox's secret
    /// filter. The differential verify gate layers it on top of the canonical
    /// sandbox environment for the second run of the verify command.
    pub client_env: Vec<(String, String)>,
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
struct SteerFileGuard {
    root: ScratchRoot,
    worker_id: String,
}

impl SteerFileGuard {
    fn new(root: ScratchRoot, worker_id: String) -> Self {
        Self { root, worker_id }
    }
}

impl Drop for SteerFileGuard {
    fn drop(&mut self) {
        remove_steer_file_in(&self.root, &self.worker_id);
    }
}

/// Stops a worker's background jobs when the worker ends.
///
/// Held for the whole of [`WorkerPool::run_phases`], which returns from a dozen
/// places: a job that outlived its worker would keep running with the build slot
/// and the build dir the command was admitted with. A `Drop` impl cannot be
/// forgotten on a new early return, and it runs before the worktree guard is
/// dropped, while the directories the process-group sweep matches on still
/// exist.
struct JobGuard<'a> {
    pool: &'a WorkerPool,
    worker_id: &'a str,
}

impl Drop for JobGuard<'_> {
    fn drop(&mut self) {
        let stopped = self.pool.end_worker_jobs(self.worker_id);
        if stopped > 0 {
            info!(
                worker = %self.worker_id,
                stopped, "Stopped the background jobs of a finished worker"
            );
        }
    }
}

/// The worker's opening user message: the task, then -- when a completion
/// verify is configured -- the exact command the gate will run, then the
/// bounded context pack ([`context_pack`]) built from the task text against
/// `root` (the worker's checkout).
///
/// The gate reuses an identical passing run on an unchanged tree (see
/// `TurnEngine::reusable_verify_step`), but only when the worker ran exactly
/// the verify string. Naming it here is what lets the worker's own last check
/// be the run the gate reuses instead of paying for a second full run.
pub fn opening_task_message(task: &str, verify: Option<&str>, root: &std::path::Path) -> String {
    let mut message = format!("TASK:\n{task}\n\nBegin by exploring the repository.");
    if let Some(verify) = verify.filter(|v| !v.is_empty()) {
        message.push_str(&format!(
            "\n\nCompletion gate: `{verify}`. Run exactly this command as your last check; an identical passing run on the same tree is reused."
        ));
    }
    // The bounded context pack: the paths the task names and the symbols it
    // quotes, so the worker starts editing instead of re-discovering them
    // over its first dozen read-only turns. Empty when the task names
    // nothing, so the message is unchanged for free-form tasks.
    if let Some(pack) = context_pack(task, root) {
        message.push_str(&format!("\n\n{pack}"));
    }
    message
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
            client_env,
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
        let _steer_cleanup = SteerFileGuard::new(self.scratch.clone(), worker_id.clone());

        // A revision re-attaches to the branch the previous run committed to,
        // so the worker keeps its id, its checkpoints and its diff base; a
        // fresh dispatch creates the branch instead. Checkout shells out to
        // git and walks the tree, so it runs off the runtime thread.
        let resume_base_commit = resume_base_commit.clone();
        let repo_path_owned = repo_path.clone();
        let worker_id_owned = worker_id.clone();
        let scratch = self.scratch.clone();
        let round_base =
            super::steer::read_source(&scratch, &worker_id).and_then(|source| source.round_base);
        let (mut worktree, initial_sync) =
            tokio::task::spawn_blocking(move || match &resume_base_commit {
                Some(base) => {
                    let mut guard = WorktreeGuard::reopen_in(
                        &scratch,
                        &repo_path_owned,
                        &worker_id_owned,
                        base,
                    )?;
                    guard.base_branch = resume_base_branch;
                    let sync = match round_base {
                        Some(base) => WorktreeGuard::sync_round_base_at(
                            &guard.path,
                            &guard.repo_root,
                            &guard.branch,
                            &guard.base_commit,
                            &base,
                        ),
                        None => WorktreeGuard::sync_base_at(
                            &guard.path,
                            &guard.repo_root,
                            &guard.branch,
                            &guard.base_commit,
                            guard.base_branch.as_deref(),
                        ),
                    }?;
                    Ok((guard, sync))
                }
                None => WorktreeGuard::new_in(&scratch, &repo_path_owned, &worker_id_owned)
                    .map(|guard| (guard, crate::worktree::BaseSync::Unchanged)),
            })
            .await
            .context("Worktree checkout task failed")??;
        // A kill must not lose what this worker leaves uncommitted, and the
        // guard that owns the checkout dies with the task a kill aborts, so the
        // pool keeps the path and commits through it (see `WorkerPool::kill`).
        self.register_worktree(&worker_id, worktree.path.clone())
            .await;
        // The branch this worker cut is now real, so the row names the
        // commit it points at. A worker that never commits still has a
        // branch, and a continuation that finds it pruned recreates it
        // from exactly this commit.
        self.record_base_commit(&worker_id, &worktree.base_commit)
            .await;

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
                    opening_task_message(&task, verify.as_deref(), &worktree.path),
                ),
            ],
        };

        // The opening messages are the log's first lines, so a worker that dies
        // before its first turn still leaves a continuable conversation behind.
        let opening_meta = WorkerHistory {
            task: task.clone(),
            group: meta.group.clone(),
            role: meta.role,
            model: model.clone(),
            temperature,
            repo_path: repo_path_str.clone(),
            base_commit: worktree.base_commit.clone(),
            base_branch: worktree.base_branch.clone(),
            branch: worktree.branch.clone(),
            network_offline,
            verify: verify.clone(),
            client_env: client_env.clone(),
            max_turns,
            review_after: review_after.clone(),
            revision: meta.revision,
            auto_continues: meta.auto_continues,
            owner: Some(meta.owner.clone()),
            messages: Vec::new(),
        };
        if !super::revision::history_log_path_in(&self.scratch, &worker_id).exists()
            && let Err(e) = self.append_history_messages(&worker_id, &opening_meta, &messages)
        {
            warn!(
                worker = %worker_id,
                error = %e,
                "Could not persist the worker conversation; this worker can no longer be revised"
            );
        }

        if let crate::worktree::BaseSync::Conflicts { branch, files } = initial_sync {
            let notice = ChatMessage::text(
                Role::User,
                format!(
                    "BASE INTEGRATION pending with {branch}. Remaining conflicted files: {}. Resolve the markers and request completion; the harness will re-check them.",
                    files.join(", ")
                ),
            );
            append_history_message_in(&self.scratch, &worker_id, &opening_meta, &notice)?;
            messages.push(notice);
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
                client_env: &client_env,
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
        &self,
        worker_id: &str,
        meta: &WorkerHistory,
        messages: &[ChatMessage],
    ) -> anyhow::Result<()> {
        for msg in messages {
            append_history_message_in(&self.scratch, worker_id, meta, msg)?;
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
        let client_env = config.client_env.to_vec();
        let repo_path_str = config.repo_path_str.to_string();

        let runner = AgentRunner::new(
            self.api_base.clone(),
            self.api_key.clone(),
            model.clone(),
            temperature,
        )
        .with_network_offline(network_offline)
        // A command that outlives its budget becomes one of this worker's
        // background jobs, so the runner needs the worker's job table.
        .with_jobs(self.job_handle(worker_id));
        let _jobs = JobGuard {
            pool: self,
            worker_id,
        };

        let mut step = 0;
        let mut current_max_turns = max_turns;
        let mut consecutive_no_cmd = 0;
        let mut last_assistant_text = String::new();
        let mut watch = ProgressWatch::default();
        let mut verified: Option<bool> = None;
        // Whether the implementer's loop ended on the completion sentinel
        // rather than by running out of turns.
        let mut completed = false;
        // The completion report and the one follow-up it may cost live across
        // turns: a verify failure replays the completion turn, and the report
        // the worker already wrote must survive that replay.
        let mut report: Option<crate::pool::WorkerReport> = None;
        let mut report_asked = false;
        let mut report_text = String::new();
        // A consolidator's per-worker verdicts live across turns like the
        // report: the completion turn is replayed on a verify failure, and the
        // verdicts it already gave must survive that replay.
        let mut verdicts: Option<crate::pool::WorkerVerdicts> = None;

        let mut auto_extended = false;
        loop {
            while step < current_max_turns {
                step += 1;
                let max_turns_for_config = current_max_turns;
                let turn_config = TurnConfig {
                    label_prefix: "",
                    steer_prefix: "STEER / ORCHESTRATOR GUIDANCE:\n",
                    apply_sentinels: true,
                    // A consolidator reviews, merges, steers and waits; it is not
                    // paid to edit, so the read-only escalation would pause it for
                    // doing its job. An ordinary implementer keeps the guard.
                    read_only_exempt: meta.role == WorkerRole::Consolidate,
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
                    client_env: &client_env,
                    dispatch_max_turns: max_turns,
                    watch: &mut watch,
                    report: &mut report,
                    report_asked: &mut report_asked,
                    report_text: &mut report_text,
                    verdicts: &mut verdicts,
                };
                match engine.run_turn(&turn_config).await? {
                    TurnOutcome::Completed { verified: v } => {
                        // Flush the completion turn too: a reused verify pushes its
                        // disclosure note here, and a crash must not lose it.
                        engine.flush_history_log(&turn_config).await;
                        verified = v;
                        completed = true;
                        break;
                    }
                    TurnOutcome::Continue | TurnOutcome::NoCommand => {}
                    TurnOutcome::EndReview => unreachable!("implementer never ends review quietly"),
                }
                // One line per message, flushed at the turn boundary: a crash can
                // only lose the turn that was in flight.
                engine.flush_history_log(&turn_config).await;
            }

            if completed || auto_extended {
                break;
            }

            // The budget ran out before the completion sentinel: decide on one
            // automatic extension, or fall through to the exhausted branch.
            let summary = watch.progress_summary(step);
            match turn::grant_extension(
                &summary,
                auto_extended,
                turn::auto_extension_budget(max_turns),
            ) {
                Some(n) => {
                    auto_extended = true;
                    current_max_turns += n;
                    meta.metrics.auto_extensions_granted += 1;
                    messages.push(ChatMessage::text(
                        Role::User,
                        format!(
                            "Budget extended once by {n} turns: finish now (gate, REPORT, completion sentinel)"
                        ),
                    ));
                    info!(
                        worker = %worker_id,
                        turns = n,
                        "Budget extended once by the automatic extension"
                    );
                }
                None => break,
            }
        }

        // --- MULTI-PHASE REVIEW PIPELINE ---
        // The implementer's loop is done; hand off to the independent auditor
        // and fold its turns back into the single monotonic step counter.
        //
        // The mode is decided here: a requested review runs as asked, but a
        // diff that touches a declared sensitive path is always upgraded to
        // the adversarial security review, and a sensitive diff with no
        // requested review triggers it on its own. This is the harness's
        // focused adversarial pass for the paths the repository declared.
        let mut patterns = crate::manifest::sensitive_paths(std::path::Path::new(&repo_path_str));
        patterns.extend(self.manifest().sensitive_paths.iter().cloned());
        patterns.sort();
        patterns.dedup();
        let requested = review_after.as_deref().map(ReviewMode::parse_model);
        // What this run has to be audited over: everything since the base, or
        // only what came after the commit an earlier security review approved.
        // A consolidator's own commits are what its security review covers: the
        // worker branches it merged were reviewed at their own approved commits.
        let merged_branches: Vec<String> = match meta.role {
            super::registry::WorkerRole::Consolidate => {
                super::load_registry_entry_in(&self.scratch, worker_id)
                    .map(|entry| {
                        let scratch = &self.scratch;
                        // Excluding a merged branch from this audit is a claim
                        // that it was reviewed at its own approved commit;
                        // `integrated` proves only that the merge happened, so
                        // the rule checks the approval itself.
                        self::review::approved_merged_branches(&entry.integrated, |id| {
                            super::load_registry_entry_in(scratch, id)
                                .and_then(|worker| worker.security_approved_commit)
                        })
                    })
                    .unwrap_or_default()
            }
            _ => Vec::new(),
        };
        let scope = self::review::security_scope(
            worktree,
            meta.role,
            meta.security_approved_commit.clone(),
            &merged_branches,
        )
        .await;
        // Nothing changed since the last approval: the audit that already
        // stands covers this run, so the security review is skipped rather than
        // repeated. The generic quality review is untouched by this: it is not a
        // security gate, and a revision asks for it the same way it asked before.
        let security_skip = scope.skip_log();
        let touched = match &scope {
            self::review::SecurityScope::Full => {
                self::review::touched_files(
                    &worktree.path,
                    &worktree.base_commit,
                    worktree.base_branch.as_deref(),
                )
                .await
            }
            // An incremental scope is probed over what it covers only: the files
            // of the commits after the last approval, or -- for a consolidator --
            // the files its own commits touched. A path an approved review already
            // covered is not sensitive again.
            incremental => incremental.reviewed_files(&worktree.path).await,
        };
        let sensitive: Vec<String> = touched
            .into_iter()
            .filter(|path| crate::manifest::matches_sensitive(path, &patterns))
            .collect();
        // The rule that picks the review lives in `plan_review`, next to the
        // scope that decides the skip, so the mode a requested review ends up
        // running in cannot drift from the one the rule states.
        // The automatic trigger audits on the manifest's strongest tier, and
        // falls back to the implementer's own model when the manifest marks none.
        let strongest = self
            .manifest()
            .strongest_alias()
            .map(|alias| self.manifest().resolve_model(alias).0)
            .unwrap_or_else(|| model.clone());
        let review_plan = self::review::plan_review(
            security_skip.is_some(),
            requested,
            !sensitive.is_empty(),
            &strongest,
        );
        if let Some(reason) = security_skip {
            info!(worker = %worker_id, "{reason}");
        }
        if let Some((reviewer_model, mode)) = review_plan {
            let outcome = self
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
                        mode,
                        verify: verify.clone(),
                        sensitive,
                        scope: scope.clone(),
                    },
                )
                .await?;
            step = outcome.step;
            // The reviewer's own completion stands in for the implementer's:
            // a run is finished only when some phase emitted the sentinel.
            completed |= outcome.completed;
            if let Some(security) = outcome.security {
                meta.security_review = Some(security);
                // The commit this review approved rides the registry row, so a
                // later revision reviews from here instead of re-auditing the
                // whole diff since the base commit.
                meta.security_approved_commit = self::review::head_commit_of(&worktree.path).await;
            }
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
        // The report's `done:` line is the summary every consumer reads; the
        // last chat message stays the fallback for a worker that never wrote
        // one, so the commit subject is never "Now I'll make the edits.". The
        // fallback skips the block marker and a consolidator's per-worker
        // verdicts: those are protocol, so a round never headlines as the bare
        // word "REPORT".
        let agent_summary = report
            .as_ref()
            .map(|r| r.done.trim().to_string())
            .filter(|done| !done.is_empty())
            .or_else(|| summary_line(&last_assistant_text).map(str::to_string))
            .unwrap_or_default();
        let path = worktree.path.clone();
        let repo_root = worktree.repo_root.clone();
        let base_commit = worktree.base_commit.clone();
        let base_branch = worktree.base_branch.clone();
        let branch = worktree.branch.clone();
        let seeded = worktree.seeded();
        let metrics_path = path.clone();
        let metrics_base = base_commit.clone();
        let metrics_base_branch = base_branch.clone();
        let (artifacts, diff, summary, branch, head_commit, now) =
            tokio::task::spawn_blocking(move || {
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

        // The payload (diff/summary/artifacts/branch) is assembled *before* the
        // write-guard is taken: the critical section only performs the O(1)
        // move of the pre-built value into the record. The revision that
        // produced it rides along: a fresh dispatch is at zero, a revised
        // worker at its attempt number.
        let revision = self
            .workers
            .read()
            .await
            .get(worker_id)
            .map(|w| w.revision)
            .unwrap_or(0);

        if !completed {
            // The turn budget ran out before any phase emitted the completion
            // sentinel. The work is checkpointed on the branch exactly like a
            // completion, but the run is stopped, not done, and never verified.
            let summary = if summary.trim().is_empty() {
                format!("Stopped after {step} turns: {TURN_BUDGET_EXHAUSTED} before completion")
            } else {
                format!(
                    "{summary}\n\nStopped after {step} turns: {TURN_BUDGET_EXHAUSTED} before completion"
                )
            };
            let exhausted_state = WorkerState::Exhausted {
                turns: step,
                diff,
                summary,
                stopped_at: now,
                artifacts,
                branch,
                metrics: meta.metrics,
                revision,
                report: report.clone(),
                verdicts: verdicts.clone(),
            };
            self.update_worker(worker_id, |w| w.state = exhausted_state)
                .await;
            meta.report = report;
            // An exhausted consolidator reports the workers it had reached a
            // verdict on before the budget ran out; the row carries them like
            // the report it already writes here.
            meta.verdicts = verdicts;
            self.save_status(
                meta,
                &model,
                RegistryStatus::Exhausted,
                step,
                current_max_turns,
                TURN_BUDGET_EXHAUSTED,
                None,
            );
            self.unregister_worktree(worker_id).await;
            info!(
                worker = %worker_id,
                turns = step,
                "Worker exhausted its turn budget without completing"
            );
            return Ok(());
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
            report: report.clone(),
            verdicts: verdicts.clone(),
        };
        self.update_worker(worker_id, |w| w.state = completed_state)
            .await;

        // The report and its verification verdict travel with the meta so the
        // terminal row carries them: the in-memory record is evicted after its
        // TTL, the row is not.
        meta.report = report;
        meta.verified = verified;
        meta.verdicts = verdicts;
        self.save_status(
            meta,
            &model,
            RegistryStatus::Completed,
            step,
            current_max_turns,
            &last_command,
            None,
        );
        // Remember where the branch ended, so a continuation can recreate it
        // after a merge prunes it within the retired grace period.
        self.record_head_commit(worker_id, head_commit).await;

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
/// commit), the head commit of that branch (`None` alongside a missing branch)
/// and the completion timestamp.
type FinalizedWork = (
    Vec<String>,
    String,
    String,
    Option<String>,
    Option<String>,
    u64,
);

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
            let first_line = agent_summary
                .lines()
                .next()
                .unwrap_or(&task_headline)
                .trim();
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
    let head_commit = committed.as_ref().and_then(|branch| {
        crate::worktree::git(
            &repo_root,
            "rev-parse",
            &["rev-parse", &format!("refs/heads/{branch}")],
        )
        .ok()
        .filter(|out| out.status.success())
        .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string())
        .filter(|sha| !sha.is_empty())
    });
    Ok((
        artifacts,
        diff,
        summary,
        committed,
        head_commit,
        unix_timestamp(),
    ))
}
