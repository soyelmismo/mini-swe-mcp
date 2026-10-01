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
//!
//! The engine also carries the four guards that keep a worker honest, all of
//! them stateless per turn and driven by [`ProgressWatch`] plus the turn
//! counter: a command byte-identical to the previous turn's is answered
//! instead of re-run (and three of those in a row park the worker on the
//! orchestrator), a worktree that stops changing gets a "make the edit or
//! escalate" nudge, `REQUEST_TURNS` may only add half the dispatch's budget,
//! and every 20 turns the worktree is checkpoint-committed so a kill or a
//! crash cannot lose the work.

use std::path::Path;

use anyhow::{Context, Result};
use tracing::{info, warn};

use crate::agent::{AgentRunner, ChatMessage, LlmResponse, Role, ToolCall};
use crate::manifest::MAX_TURNS_LIMIT;
use crate::worktree::{BaseSync, BranchMerge, WorktreeGuard, git};

use super::super::WorkerPool;
use super::super::buffer::build_step_log;
use super::super::registry::{
    RegistryStatus, WorkerMeta, WorkerRole, check_consolidate_delegation,
};
use super::super::revision::{WorkerHistory, append_history_message_in};
use super::super::state::WorkerState;
use super::super::steer::drain_steer_messages_in;
use super::history::compact_history;
use super::pause::PauseRequest;
use super::sentinels::{
    COMPLETION_SENTINEL, is_completion_request, parse_ask_orchestrator, parse_consolidate_merge,
    parse_request_turns, summarize_command,
};

/// Prefix used by both tool results and code-block command output messages.
pub(super) const COMMAND_OUTPUT_PREFIX: &str = "COMMAND OUTPUT (exit code: ";

/// Prefix of verification feedback; user-role feedback remains an instruction.
pub(super) const VERIFICATION_OUTPUT_PREFIX: &str = "VERIFICATION FAILED (exit ";

/// Follow-up when the model produced no executable bash command.
pub(super) const NO_COMMAND_NUDGE: &str =
    "ERROR: No bash command found. You MUST call the `bash` tool with your command.";

/// Turns between automatic checkpoint commits, so work left behind by a kill
/// or a crash is never more than this old.
const AUTO_CHECKPOINT_TURNS: usize = 20;

/// Turns between repository samples of the stagnation detector.
const STAGNATION_SAMPLE_TURNS: usize = 10;

/// Turns without a repository change that force the "stop exploring" nudge.
const STAGNATION_TURNS_LIMIT: usize = 30;

/// Consecutive blocked repetitions of one command before the worker is parked
/// on the orchestrator instead of being told to try something else.
const REPEAT_BLOCK_LIMIT: usize = 3;

/// Answer handed to the model that re-issues the command of the turn before.
const REPEAT_REFUSAL: &str = "You already ran this exact command; its output has not changed (see above). Take a different action.";

/// Nudge injected after a worker has explored long enough without changing
/// anything: the answer to a stuck agent is a decision, not another turn.
fn stagnation_nudge() -> String {
    format!(
        "No change to the repository in the last {STAGNATION_TURNS_LIMIT} turns. Stop exploring: make the edit, or ASK_ORCHESTRATOR if blocked."
    )
}

/// Turns a worker may self-grant through `REQUEST_TURNS`: half of the budget
/// its dispatch was given, never past the manifest ceiling. Without the bound
/// a confused model walks itself from 150 to 500 turns with nobody watching.
fn extension_budget(dispatch_max_turns: usize) -> usize {
    (dispatch_max_turns / 2).min(MAX_TURNS_LIMIT)
}

/// Fingerprint of the worktree's uncommitted work: the HEAD id plus
/// `git diff --stat HEAD`, so a commit alone counts as progress.
///
/// `None` when git could not answer, so a failed sample is never read as
/// "the repository did not change".
fn repository_sample(path: &Path) -> Option<String> {
    let head = git(path, "rev-parse HEAD", &["rev-parse", "HEAD"]).ok()?;
    let stat = git(path, "diff --stat HEAD", &["diff", "--stat", "HEAD"]).ok()?;
    if !head.status.success() || !stat.status.success() {
        return None;
    }
    Some(format!(
        "{}\n{}",
        String::from_utf8_lossy(&head.stdout).trim(),
        String::from_utf8_lossy(&stat.stdout).trim()
    ))
}

/// Read a `git diff --shortstat` line as `(files, insertions, deletions)`.
///
/// Git pluralises by count (`1 file changed`) and omits a section entirely when
/// it is zero (`2 files changed, 3 insertions(+)`), so each count is taken from
/// the part naming it and a missing part is zero. `None` when the line carries
/// no count at all, so an empty or unreadable diff is never reported as a
/// measured one.
pub(crate) fn parse_shortstat(line: &str) -> Option<(usize, usize, usize)> {
    let (mut files, mut insertions, mut deletions) = (None, None, None);
    for part in line.split(',') {
        let part = part.trim();
        let (digits, rest) = match part.find(|c: char| !c.is_ascii_digit()) {
            Some(end) => part.split_at(end),
            None => (part, ""),
        };
        let Ok(count) = digits.parse::<usize>() else {
            continue;
        };
        if rest.contains("file") {
            files = Some(count);
        } else if rest.contains("insertion") {
            insertions = Some(count);
        } else if rest.contains("deletion") {
            deletions = Some(count);
        }
    }
    let files = files.or(insertions).or(deletions)?;
    Some((files, insertions.unwrap_or(0), deletions.unwrap_or(0)))
}

/// Size of the worker's final diff, taken with the same git helper the
/// detectors sample the repository with.
///
/// The base commit is the one the worktree was created from, so checkpoint
/// commits along the way and the still-uncommitted tail are counted together.
/// `git` is a blocking subprocess, so the sample runs off the runtime thread,
/// and `None` means "not measured", never "measured as empty".
pub(super) async fn shortstat_of(
    path: &Path,
    base: &str,
    base_branch: Option<&str>,
) -> Option<(usize, usize, usize)> {
    let path = path.to_path_buf();
    let base = if base.is_empty() {
        "HEAD".to_string()
    } else {
        base.to_string()
    };
    let base_branch = base_branch.map(str::to_string);
    let output = tokio::task::spawn_blocking(move || {
        let base = WorktreeGuard::diff_base_at(&path, &base, base_branch.as_deref())?;
        git(&path, "diff --shortstat", &["diff", "--shortstat", &base])
    })
    .await
    .ok()?
    .ok()?;
    if !output.status.success() {
        return None;
    }
    parse_shortstat(&String::from_utf8_lossy(&output.stdout))
}

/// Run harness git off the runtime thread before a completion can reach verification.
async fn sync_base_for_completion(worktree: &WorktreeGuard) -> Result<BaseSync> {
    let path = worktree.path.clone();
    let repo_root = worktree.repo_root.clone();
    let branch = worktree.branch.clone();
    let base_commit = worktree.base_commit.clone();
    let base_branch = worktree.base_branch.clone();
    tokio::task::spawn_blocking(move || {
        WorktreeGuard::sync_base_at(
            &path,
            &repo_root,
            &branch,
            &base_commit,
            base_branch.as_deref(),
        )
    })
    .await
    .context("Base integration task failed")?
}

/// Cross-turn state of the two loop detectors.
///
/// Owned by the phase loop and lent to every turn, because `TurnEngine` is
/// rebuilt once per turn and a detector that lived in it would reset each time.
#[derive(Default)]
pub(super) struct ProgressWatch {
    /// The command submitted by the previous turn, trimmed.
    last_command: Option<String>,
    /// Consecutive turns that ended in a blocked repetition of it.
    repeat_blocks: usize,
    /// Last repository sample taken by the stagnation detector.
    last_sample: Option<String>,
    /// Turns elapsed since that sample last changed.
    unchanged_turns: usize,
}

impl ProgressWatch {
    /// Record `command` as the latest one and report how many consecutive
    /// turns it repeats, or `None` when it is a fresh command.
    fn register_command(&mut self, command: &str) -> Option<usize> {
        let command = command.trim();
        if self.last_command.as_deref() == Some(command) {
            self.repeat_blocks += 1;
            Some(self.repeat_blocks)
        } else {
            self.repeat_blocks = 0;
            self.last_command = Some(command.to_string());
            None
        }
    }

    /// Record a repository sample and return the turns the repository has been
    /// unchanged for. A sample that could not be taken is ignored rather than
    /// counted as "no change".
    fn record_sample(&mut self, sample: Option<String>) -> usize {
        if let Some(sample) = sample {
            if self.last_sample.as_deref() == Some(sample.as_str()) {
                self.unchanged_turns += STAGNATION_SAMPLE_TURNS;
            } else {
                self.unchanged_turns = 0;
            }
            self.last_sample = Some(sample);
        }
        self.unchanged_turns
    }
}

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
    /// Dispatch task: the history file carries it so a revision resumes the
    /// same work.
    pub task: &'a str,
    /// Sampling temperature of the dispatch, replayed by a revision.
    pub temperature: Option<f32>,
    /// Reviewer model of the dispatch, replayed by a revision.
    pub review_after: Option<&'a str>,
    /// Declared network policy of the dispatch, replayed by a revision.
    pub network_offline: bool,
}

/// Outcome of one turn.
pub(super) enum TurnOutcome {
    /// The completion sentinel was found; stop the loop. `verified` is
    /// `Some(true)` when the verify gate passed, `Some(false)` when it was
    /// exhausted after repeated failures, and `None` when no gate was set.
    Completed { verified: Option<bool> },
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
    /// The worker's registry row: it carries the worker's identity *and* the
    /// run's health counters, so every guard below moves its counter here, at
    /// the point it fires, and the row is written from the same struct.
    pub meta: &'a mut WorkerMeta,
    pub messages: &'a mut Vec<ChatMessage>,
    /// Messages pushed since the last [`Self::flush_history_log`], waiting to
    /// be appended to the durable log.
    pub unsaved_messages: Vec<ChatMessage>,
    pub step: &'a mut usize,
    pub current_max_turns: &'a mut usize,
    pub last_assistant_text: &'a mut String,
    pub consecutive_no_cmd: &'a mut usize,
    /// Optional shell command run through the same bash path before a
    /// completion sentinel is honoured. `None` disables the gate.
    pub verify: Option<&'a str>,
    /// The dispatcher's ambient environment, filtered by the sandbox's secret
    /// filter. Layered on top of the canonical sandbox environment for the
    /// differential verify run, so a suite that only passes in the
    /// orchestrator's shell is caught by the worker itself.
    pub client_env: &'a [(String, String)],
    /// The budget the dispatch was given, the base the self-grant cap is
    /// measured from (`current_max_turns` moves as the worker extends it).
    pub dispatch_max_turns: usize,
    /// Loop and stagnation detector state, shared across turns.
    pub watch: &'a mut ProgressWatch,
}

impl<'a> TurnEngine<'a> {
    /// Run one turn of the agent loop.
    pub(super) async fn run_turn(&mut self, config: &TurnConfig<'_>) -> Result<TurnOutcome> {
        // --- Steering drain ---
        let mut steer_msgs = self.pool.take_pending_steer(self.worker_id).await;
        let remote = drain_steer_messages_in(&self.pool.scratch, self.worker_id);
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
            self.push_message(ChatMessage::text(
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
                self.push_message(ChatMessage::text(
                    Role::User,
                    format!(
                        "TURN LIMIT WARNING: You have used {} of {} turns ({} remaining). If you need more turns to complete testing or refactoring, execute `echo \"REQUEST_TURNS: <number>\"` now. Otherwise, wrap up your changes and execute `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.",
                        *self.step, *self.current_max_turns, remaining
                    ),
                ));
            }
        }

        // --- Automatic checkpoint (both phases) ---
        if *self.step > 0 && (*self.step).is_multiple_of(AUTO_CHECKPOINT_TURNS) {
            self.checkpoint().await;
            self.persist_checkpoint_history(config).await;
        }

        // --- Stagnation detector (implementer only, like the sentinels) ---
        if config.apply_sentinels {
            self.check_stagnation().await;
        }

        // --- LLM call with error handling ---
        // A provider outage is waited out (up to `outage_patience`) before the
        // orchestrator is asked, and a resume retries the step rather than
        // failing the worker on the next error.
        let mut outage_waited = std::time::Duration::ZERO;
        let llm_resp = loop {
            compact_history(self.messages);
            let e = match self.runner.run_step_llm(self.messages).await {
                Ok(resp) => break resp,
                Err(e) => e,
            };
            if crate::agent::retry::is_llm_unavailable(&e)
                && outage_waited < crate::agent::retry::outage_patience()
            {
                let delay = crate::agent::retry::OUTAGE_RETRY_INTERVAL;
                warn!(
                    worker = %self.worker_id,
                    step = *self.step,
                    waited_secs = outage_waited.as_secs(),
                    error = %e,
                    "LLM provider unavailable; waiting before retrying the step"
                );
                tokio::time::sleep(delay).await;
                outage_waited += delay;
                continue;
            }
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
                    // Safe checkpoint of uncommitted worktree changes so work is never lost.
                    // `commit_changes` shells out to git, so it runs off the runtime thread.
                    {
                        let path = self.worktree.path.clone();
                        let message = format!(
                            "worker({}): checkpoint step {} before pause (error: {})",
                            self.worker_id, *self.step, e
                        );
                        let committed = tokio::task::spawn_blocking(move || {
                            WorktreeGuard::commit_all(&path, &message)
                        })
                        .await
                        .unwrap_or(Ok(false));
                        if committed.unwrap_or(false) {
                            self.worktree.preserve_branch = true;
                        }
                    }

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
                            model: config.model,
                            meta: self.meta,
                        })
                        .await?;

                    let Some(resume_msg) = answer else {
                        return Err(e);
                    };
                    info!(
                        worker = %self.worker_id,
                        msg = %resume_msg,
                        "Worker resumed after error by orchestrator"
                    );
                    if !resume_msg.trim().is_empty() && resume_msg.trim() != "resume" {
                        self.push_message(ChatMessage::text(
                            Role::User,
                            format!("ORCHESTRATOR GUIDANCE:\n{}", resume_msg),
                        ));
                    }
                    outage_waited = std::time::Duration::ZERO;
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
                return self.handle_completion(&llm_resp).await;
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
        // One write-guard moves the step counter, refreshes the cached metrics
        // and bumps the change generation, so a waiter parked in
        // `await_worker_result_until` wakes on this step instead of on the next
        // 500 ms tick. Refreshing the cached metrics in the same critical
        // section keeps a kill racing this turn reporting the counters to it.
        self.pool
            .update_worker(self.worker_id, |w| {
                w.metrics = self.meta.metrics;
                if let WorkerState::Running {
                    step: ref mut s,
                    ref mut last_command,
                    ..
                } = w.state
                {
                    *s = *self.step;
                    *last_command = label.clone();
                }
            })
            .await;

        // --- Registry update, coalesced by the pool's writer ---
        self.pool.save_status(
            self.meta,
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

        // --- Repetition detector ---
        // Re-issuing the identical command is not progress: running it again
        // burns a turn and returns the output the history already carries, so
        // the command is answered without being executed.
        if let Some(blocks) = self.watch.register_command(&cmd_str) {
            self.meta.metrics.repeat_blocks += 1;
            warn!(
                worker = %self.worker_id,
                step = *self.step,
                op = %cmd_summary,
                consecutive = blocks,
                "Blocked a repeated command; its output is already in the history"
            );
            if blocks >= REPEAT_BLOCK_LIMIT {
                return self
                    .park_on_loop(config, &llm_resp, &cmd_summary, blocks)
                    .await;
            }
            self.push_exchange(
                llm_resp.content,
                llm_resp.reasoning_content,
                llm_resp.tool_calls.zip(llm_resp.tool_call_id),
                REPEAT_REFUSAL.to_string(),
            );
            return Ok(TurnOutcome::Continue);
        }

        // --- Execute command with semaphores ---
        let (output, code) = self.run_gated(&cmd_str).await?;

        // --- Consolidator merge request (harness side, never bash) ---
        // The sandbox holds no git credentials, so the merge runs here, on the
        // harness, through the same machinery the base sync uses. Only a
        // consolidator sees this verb; an ordinary worker's identical command
        // stays plain bash.
        if self.meta.role == WorkerRole::Consolidate
            && let Some(ids) = parse_consolidate_merge(&cmd_str)
        {
            let observation = self.consolidate_merge(&ids).await;
            let output_text = format!("{COMMAND_OUTPUT_PREFIX}0):\n```\n{observation}\n```");
            let step_log = build_step_log(*self.step, &label, observation, Some(0));
            {
                let mut lock = self.pool.workers.write().await;
                if let Some(w) = lock.get_mut(self.worker_id) {
                    w.logs.push(step_log);
                }
            }
            self.push_exchange(
                llm_resp.content,
                llm_resp.reasoning_content,
                llm_resp.tool_calls.zip(llm_resp.tool_call_id),
                output_text,
            );
            return Ok(TurnOutcome::Continue);
        }

        // --- Orchestrator control sentinels (implementer only) ---
        if config.apply_sentinels {
            // REQUEST_TURNS, bounded by the self-grant budget: a worker may
            // add at most half of the budget its dispatch was given, so no
            // model can walk itself to the manifest ceiling unattended.
            if let Some(additional) = parse_request_turns(&cmd_str) {
                let budget = extension_budget(self.dispatch_max_turns);
                let granted = self
                    .current_max_turns
                    .saturating_sub(self.dispatch_max_turns);
                let requested_max = self
                    .dispatch_max_turns
                    .saturating_add(granted)
                    .saturating_add(additional)
                    .min(MAX_TURNS_LIMIT);
                if requested_max <= self.dispatch_max_turns + budget {
                    let old_max = *self.current_max_turns;
                    *self.current_max_turns = requested_max;
                    self.meta.metrics.extensions_granted += additional;
                    info!(
                        worker = %self.worker_id,
                        requested = additional,
                        old_max,
                        new_max = *self.current_max_turns,
                        granted_total = granted + additional,
                        budget,
                        "Subagent requested turn extension; granted"
                    );
                } else {
                    self.meta.metrics.extensions_refused += 1;
                    warn!(
                        worker = %self.worker_id,
                        requested = additional,
                        granted_total = granted,
                        budget,
                        "Refused turn extension beyond the self-grant budget"
                    );
                    self.push_message(ChatMessage::text(
                        Role::User,
                        format!(
                            "TURN EXTENSION REFUSED: {additional} more turns would take you to {requested_max}, past the {budget} turns you may self-grant on a {} turn budget. Wrap up your changes and execute `echo {COMPLETION_SENTINEL}`, or execute `echo \"ASK_ORCHESTRATOR: what is blocking you?\"` if you need a decision.",
                            self.dispatch_max_turns
                        ),
                    ));
                }
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
                        model: config.model,
                        meta: self.meta,
                    })
                    .await?;

                if let Some(answer) = answer {
                    self.push_message(ChatMessage::text(
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
            "{COMMAND_OUTPUT_PREFIX}{}):\n```\n{}\n```",
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

        self.push_exchange(
            llm_resp.content,
            llm_resp.reasoning_content,
            llm_resp.tool_calls.zip(llm_resp.tool_call_id),
            output_text,
        );

        Ok(TurnOutcome::Continue)
    }

    /// Merge the named workers' branches into this consolidator's worktree.
    ///
    /// Each id is resolved, checked against the delegation rule and merged in
    /// order; the first conflict stops the run and leaves the rest skipped, so
    /// the model resolves one merge at a time. A worker the pool and the
    /// registry do not both know is refused rather than silently ignored.
    async fn consolidate_merge(&mut self, ids: &[String]) -> String {
        let mut lines = Vec::new();
        let mut conflicted = false;
        for id in ids {
            if conflicted {
                lines.push(format!("{id} skipped (an earlier merge conflicted)"));
                continue;
            }
            let target = match self.pool.resolve_worker_id(id, &self.meta.owner).await {
                Ok(target) => target,
                Err(e) => {
                    lines.push(format!("{id} refused: {e}"));
                    continue;
                }
            };
            let Some(entry) = self.pool.worker_row(&target).await else {
                lines.push(format!("{id} refused: no such worker"));
                continue;
            };
            if let Err(reason) = check_consolidate_delegation(self.meta, &entry) {
                lines.push(format!("{id} refused: {reason}"));
                continue;
            }
            if !self.pool.is_completed(&target).await {
                lines.push(format!("{id} refused: the worker is not completed"));
                continue;
            }
            let path = self.worktree.path.clone();
            let repo_root = self.worktree.repo_root.clone();
            let branch = self.worktree.branch.clone();
            let base_commit = self.worktree.base_commit.clone();
            let merged = tokio::task::spawn_blocking(move || {
                WorktreeGuard::merge_branch_at(&path, &repo_root, &branch, &base_commit, &target)
            })
            .await;
            match merged {
                Ok(Ok(BranchMerge::Merged { files })) => {
                    // The branch now carries another worker's commits, so it
                    // must survive this guard's cleanup.
                    self.worktree.preserve_branch = true;
                    lines.push(format!("{id} merged ({files} files)"));
                }
                Ok(Ok(BranchMerge::Conflicts { files })) => {
                    conflicted = true;
                    lines.push(format!(
                        "{id} conflict: {} (resolve the markers, then continue)",
                        files.join(", ")
                    ));
                }
                Ok(Err(e)) => lines.push(format!("{id} refused: {e}")),
                Err(e) => lines.push(format!("{id} refused: {e}")),
            }
        }
        lines.join("\n")
    }

    /// Handle a completion sentinel: run the verify gate (if any) and either
    /// complete or push the failure back to the model for another turn.
    async fn handle_completion(&mut self, llm_resp: &LlmResponse) -> Result<TurnOutcome> {
        info!(
            worker = %self.worker_id,
            step = *self.step,
            "Worker requested completion"
        );
        if !llm_resp.content.trim().is_empty() {
            *self.last_assistant_text = llm_resp.content.clone();
        }

        let merged = match sync_base_for_completion(self.worktree).await? {
            BaseSync::Unchanged => None,
            BaseSync::Merged { branch } => {
                self.worktree.preserve_branch = true;
                Some(branch)
            }
            BaseSync::Conflicts { branch, files } => {
                self.worktree.preserve_branch = true;
                let refusal = if files.is_empty() {
                    format!(
                        "COMPLETION REFUSED: base {branch} was merged, but the merge is still in progress. Resolve any hidden conflicts (for example with `git status` and `git diff`), make every file compile and pass tests, then request completion again. Do not run git commit; the harness concludes the merge it started."
                    )
                } else {
                    format!(
                        "COMPLETION REFUSED: base {branch} was merged, but the merge is still in progress. Resolve the conflict markers (<<<<<<<) in: {}. Keep both sides' intent, remove every marker, then request completion again. Do not run git commit; the harness stages your resolutions and creates the merge commit.",
                        files.join(", ")
                    )
                };
                // One exchange, like a verify failure: the completion turn is
                // replayed, so the next request carries no dangling tool_call.
                self.push_exchange(
                    llm_resp.content.clone(),
                    llm_resp.reasoning_content.clone(),
                    llm_resp
                        .tool_calls
                        .clone()
                        .zip(llm_resp.tool_call_id.clone()),
                    refusal,
                );
                return Ok(TurnOutcome::Continue);
            }
        };

        let Some(verify) = self.verify.filter(|v| !v.is_empty()) else {
            return Ok(TurnOutcome::Completed { verified: None });
        };

        self.meta.metrics.verify_runs += 1;
        // The side-effect baseline is taken before the gate runs, so both the
        // canonical run and the divergent run are audited against it.
        let gate_baseline = super::divergent::snapshot(&self.worktree.repo_root);
        let (output, code) = self.run_gated(verify).await?;

        let exit = code.unwrap_or(-1);
        if exit == 0 {
            return self
                .finish_verified_completion(llm_resp, verify, &gate_baseline)
                .await;
        }

        // Verification failed. Record a step log and push the output back to
        // the model so it can fix the problems before completing again. After
        // three failed verifications the worker completes anyway, flagged
        // unverified.
        self.meta.metrics.verify_failures += 1;
        if self.meta.metrics.verify_failures >= 3 {
            return Ok(TurnOutcome::Completed {
                verified: Some(false),
            });
        }
        let label = format!("[verify] {}", summarize_command(verify));
        let step_log = build_step_log(*self.step, &label, output.clone(), code);
        {
            let mut lock = self.pool.workers.write().await;
            if let Some(w) = lock.get_mut(self.worker_id) {
                w.logs.push(step_log);
            }
        }
        let integration = merged.map(|branch| format!(
            " Base {branch} was merged before this check; verification ran on the integrated tree."
        )).unwrap_or_default();
        let output_text = format!(
            "{VERIFICATION_OUTPUT_PREFIX}{exit}) - fix these problems before completing:{integration}\n{output}"
        );
        // The completion turn is replayed with the same rules as a command
        // turn, so the next request never carries a dangling tool_call.
        self.push_exchange(
            llm_resp.content.clone(),
            llm_resp.reasoning_content.clone(),
            llm_resp
                .tool_calls
                .clone()
                .zip(llm_resp.tool_call_id.clone()),
            output_text,
        );
        Ok(TurnOutcome::Continue)
    }

    /// Finish a completion whose canonical verify run passed: audit what it
    /// left behind, then re-run the same command in the divergent environment.
    ///
    /// Variant B runs only after A passed and only at completion, so a worker
    /// that is still iterating pays no second-verify cost. Either refusal
    /// replays the completion turn, exactly like a verify failure, so the next
    /// request never carries a dangling tool_call.
    async fn finish_verified_completion(
        &mut self,
        llm_resp: &LlmResponse,
        verify: &str,
        baseline: &super::divergent::SideEffectBaseline,
    ) -> Result<TurnOutcome> {
        let repo_root = self.worktree.repo_root.clone();
        let worktree_path = self.worktree.path.clone();
        let worker_id = self.worker_id.to_string();

        // Side-effect audit of the canonical run, against the pre-gate
        // baseline: the suite must leave the repository, its refs and its
        // processes exactly as it found them.
        let effects = super::divergent::audit(&repo_root, &worktree_path, &worker_id, baseline);
        if !effects.is_empty() {
            super::divergent::cleanup(&repo_root, &effects);
            let refusal = super::divergent::side_effect_refusal(&effects);
            self.push_exchange(
                llm_resp.content.clone(),
                llm_resp.reasoning_content.clone(),
                llm_resp
                    .tool_calls
                    .clone()
                    .zip(llm_resp.tool_call_id.clone()),
                refusal,
            );
            return Ok(TurnOutcome::Continue);
        }

        // Variant B: the same command in the divergent environment. Disabled
        // by the operator, or skipped when there is nothing to diverge on.
        if !super::divergent::enabled() {
            return Ok(TurnOutcome::Completed {
                verified: Some(true),
            });
        }
        let divergent_env =
            super::divergent::divergent_environment(&worktree_path, self.client_env);
        tracing::info!(
            worker = %self.worker_id,
            names = ?super::divergent::divergent_names(&divergent_env),
            "Running divergent verify variant B"
        );
        let (output_b, code_b) = self
            .run_gated_with_env(verify, divergent_env.clone())
            .await?;
        tracing::info!(
            worker = %self.worker_id,
            exit = ?code_b,
            output = %crate::agent::sandbox::truncate_output(&output_b),
            "Divergent verify variant B finished"
        );

        // The audit covers both runs: variant B must clean up after itself too.
        let effects_b = super::divergent::audit(&repo_root, &worktree_path, &worker_id, baseline);
        if !effects_b.is_empty() {
            super::divergent::cleanup(&repo_root, &effects_b);
            let refusal = super::divergent::side_effect_refusal(&effects_b);
            self.push_exchange(
                llm_resp.content.clone(),
                llm_resp.reasoning_content.clone(),
                llm_resp
                    .tool_calls
                    .clone()
                    .zip(llm_resp.tool_call_id.clone()),
                refusal,
            );
            return Ok(TurnOutcome::Continue);
        }

        let exit_b = code_b.unwrap_or(-1);
        if exit_b == 0 {
            return Ok(TurnOutcome::Completed {
                verified: Some(true),
            });
        }
        let differing = super::divergent::divergent_names(&divergent_env);
        let refusal = super::divergent::divergence_refusal(verify, &differing, code_b, &output_b);
        self.push_exchange(
            llm_resp.content.clone(),
            llm_resp.reasoning_content.clone(),
            llm_resp
                .tool_calls
                .clone()
                .zip(llm_resp.tool_call_id.clone()),
            refusal,
        );
        Ok(TurnOutcome::Continue)
    }

    /// Park a worker that re-issued one command [`REPEAT_BLOCK_LIMIT`] times in
    /// a row. The implementer asks the orchestrator for a different action; the
    /// reviewer ends its phase, since there is nobody to steer a review.
    async fn park_on_loop(
        &mut self,
        config: &TurnConfig<'_>,
        llm_resp: &LlmResponse,
        cmd_summary: &str,
        blocks: usize,
    ) -> Result<TurnOutcome> {
        if !config.apply_sentinels {
            warn!(
                worker = %self.worker_id,
                step = *self.step,
                op = %cmd_summary,
                "Reviewer is looping on the same command; ending the review phase"
            );
            return Ok(TurnOutcome::EndReview);
        }

        self.meta.metrics.loop_pauses += 1;
        let question = format!(
            "Repetition loop: `{cmd_summary}` was blocked {blocks} turns in a row (byte-identical to the previous turn's command, output unchanged). The worker is not making progress; guide it to a different action."
        );
        let answer = self
            .pool
            .pause_for_orchestrator(PauseRequest {
                worker_id: self.worker_id,
                question: &question,
                step: *self.step,
                max_turns: *self.current_max_turns,
                last_command: cmd_summary,
                model: config.model,
                meta: self.meta,
            })
            .await?;

        // The blocked turn is still answered, so the history the model sees
        // next carries both the refusal and the orchestrator's guidance.
        self.push_exchange(
            llm_resp.content.clone(),
            llm_resp.reasoning_content.clone(),
            llm_resp
                .tool_calls
                .clone()
                .zip(llm_resp.tool_call_id.clone()),
            REPEAT_REFUSAL.to_string(),
        );
        if let Some(answer) = answer
            && !answer.trim().is_empty()
            && answer.trim() != "resume"
        {
            self.push_message(ChatMessage::text(
                Role::User,
                format!("ORCHESTRATOR GUIDANCE:\n{answer}"),
            ));
        }
        Ok(TurnOutcome::Continue)
    }

    /// Commit whatever the worker has uncommitted, so a kill or a crash never
    /// costs more than one checkpoint interval of work.
    ///
    /// `commit_changes` shells out to git, so it runs off the runtime thread.
    async fn checkpoint(&mut self) {
        let message = format!(
            "worker({}): auto-checkpoint step {}",
            self.worker_id, *self.step
        );
        let path = self.worktree.path.clone();
        let repo_root = self.worktree.repo_root.clone();
        let branch = self.worktree.branch.clone();
        let base_commit = self.worktree.base_commit.clone();
        let committed = tokio::task::spawn_blocking(move || {
            WorktreeGuard::commit_changes_at(&path, &repo_root, &branch, &base_commit, &message)
        })
        .await
        .unwrap_or(Ok(None));
        match committed {
            Ok(Some(kept)) => {
                self.worktree.preserve_branch = true;
                info!(
                    worker = %self.worker_id,
                    step = *self.step,
                    branch = %kept,
                    "Committed worker checkpoint"
                )
            }
            Ok(None) => {}
            Err(e) => warn!(
                worker = %self.worker_id,
                step = *self.step,
                error = %e,
                "Checkpoint commit failed; the worktree still holds uncommitted changes"
            ),
        }
    }

    /// Persist the conversation at an auto-checkpoint, so a killed hub leaves a
    /// revisable history log behind instead of only the branch.
    ///
    /// The log is append-only and already holds every message pushed up to the
    /// previous turn boundary, so a checkpoint is a flush of what this turn
    /// added: one line per message, never a whole-file rewrite.
    async fn persist_checkpoint_history(&mut self, config: &TurnConfig<'_>) {
        self.flush_history_log(config).await;
    }

    /// Sample the worktree every [`STAGNATION_SAMPLE_TURNS`] turns and tell a
    /// worker that stopped changing anything to make the edit or escalate.
    async fn check_stagnation(&mut self) {
        if *self.step == 0 || !(*self.step).is_multiple_of(STAGNATION_SAMPLE_TURNS) {
            return;
        }
        let path = self.worktree.path.clone();
        // `git` is a blocking subprocess, so it must not run on a runtime thread.
        let sample = tokio::task::spawn_blocking(move || repository_sample(&path))
            .await
            .ok()
            .flatten();
        if self.watch.record_sample(sample) < STAGNATION_TURNS_LIMIT {
            return;
        }
        self.meta.metrics.stagnation_nudges += 1;
        warn!(
            worker = %self.worker_id,
            step = *self.step,
            unchanged_turns = self.watch.unchanged_turns,
            "Repository unchanged for too long; nudging the worker to stop exploring"
        );
        self.messages
            .push(ChatMessage::text(Role::User, stagnation_nudge()));
    }

    /// Run `command` through the worker's semaphores: heavy commands are
    /// admitted by the resource-aware controller, every command takes a bash
    /// slot.
    ///
    /// A granted heavy command also carries the job count the controller
    /// divided over the builds already running, which rides on a runner clone
    /// for this one command; a light command keeps the default parallelism.
    /// [`run_gated`] with an environment overlay layered on top of the
    /// sanitized environment (see [`AgentRunner::with_extra_env`]).
    async fn run_gated_with_env(
        &mut self,
        command: &str,
        extra_env: Vec<(String, String)>,
    ) -> Result<(String, Option<i32>)> {
        let heavy = crate::agent::is_heavy_command(command);
        let build_permit = if heavy {
            Some(self.pool.admission.acquire().await)
        } else {
            None
        };
        let _bash_permit = self
            .pool
            .bash_semaphore
            .acquire()
            .await
            .context("Bash semaphore closed")?;
        let mut runner = self.runner.clone();
        if let Some(permit) = &build_permit {
            runner = runner.with_build_jobs(permit.jobs());
        }
        // The overlay is applied after the job count, so a divergent
        // environment can never be dropped by a builder-chain reorder.
        runner = runner.with_extra_env(extra_env);
        runner.build_target_dir = if heavy {
            self.worktree.build_dir().await
        } else {
            self.worktree.leased_build_dir().map(Path::to_path_buf)
        };
        let _running = self.pool.command_running(self.worker_id);
        runner.execute_bash(&self.worktree.path, command).await
    }

    async fn run_gated(&mut self, command: &str) -> Result<(String, Option<i32>)> {
        let heavy = crate::agent::is_heavy_command(command);
        let build_permit = if heavy {
            // A queued heavy command is not worker inactivity: publish the wait
            // (with the requests ahead of it) so the stall detector skips it.
            let _waiting = self
                .pool
                .wait_for_build_slot(self.worker_id, self.pool.admission.waiting() + 1);
            Some(self.pool.admission.acquire().await)
        } else {
            None
        };
        let _bash_permit = self
            .pool
            .bash_semaphore
            .acquire()
            .await
            .context("Bash semaphore closed")?;
        let mut runner = self.runner.clone();
        if let Some(permit) = &build_permit {
            // The admission slot still doses CPU, but it no longer picks the
            // directory: the worker leases one build dir for its whole lifetime.
            runner = runner.with_build_jobs(permit.jobs());
        }
        // The worker's exclusive build dir: a heavy command leases one on
        // first use, and a light command reuses it when there is one (and
        // otherwise builds in the worktree, as before).
        runner.build_target_dir = if heavy {
            self.worktree.build_dir().await
        } else {
            self.worktree.leased_build_dir().map(Path::to_path_buf)
        };
        let _running = self.pool.command_running(self.worker_id);
        runner.execute_bash(&self.worktree.path, command).await
    }

    /// Record one executed exchange in the history: an assistant turn that
    /// called a tool is answered by a `tool` message with that call's id; a
    /// code-block (prose) turn is answered by a user message. The assistant
    /// turn always carries the response's reasoning.
    fn push_exchange(
        &mut self,
        content: String,
        reasoning: Option<String>,
        tool_call: Option<(Vec<ToolCall>, String)>,
        output_text: String,
    ) {
        if let Some((tool_calls, tc_id)) = tool_call {
            let content = (!content.trim().is_empty()).then_some(content);
            let msg = ChatMessage::assistant_with_tool_calls(content, tool_calls)
                .with_reasoning_content(reasoning);
            self.push_message(msg);
            self.push_message(ChatMessage::tool_result(tc_id, output_text));
        } else {
            let content = if content.trim().is_empty() {
                "I will execute a bash command.".to_string()
            } else {
                content
            };
            let msg = ChatMessage::text(Role::Assistant, content).with_reasoning_content(reasoning);
            self.push_message(msg);
            self.push_message(ChatMessage::text(Role::User, output_text));
        }
    }

    /// Append one message to the live conversation *and* to the durable
    /// append-only log, so a crash costs at most the in-flight turn.
    ///
    /// The append is buffered and flushed by [`Self::flush_history_log`] at the
    /// end of the turn, keeping the blocking write off the async runtime while
    /// still writing one line per message.
    fn push_message(&mut self, msg: ChatMessage) {
        self.messages.push(msg.clone());
        self.unsaved_messages.push(msg);
    }

    /// Write the messages buffered by [`Self::push_message`] to the worker's
    /// append-only history log.
    ///
    /// The metadata line is written with the first message, so a log always
    /// opens with the facts a continuation needs. A failed append warns and
    /// carries on: the checkpoint snapshot still covers the whole conversation.
    pub(super) async fn flush_history_log(&mut self, config: &TurnConfig<'_>) {
        if self.unsaved_messages.is_empty() {
            return;
        }
        let meta = self.history_meta(config);
        let pending = std::mem::take(&mut self.unsaved_messages);
        let worker_id = self.worker_id.to_string();
        let root = self.pool.scratch.clone();
        let step = *self.step;
        if let Err(e) = tokio::task::spawn_blocking(move || {
            for msg in &pending {
                append_history_message_in(&root, &worker_id, &meta, msg)?;
            }
            Ok::<(), anyhow::Error>(())
        })
        .await
        .unwrap_or_else(|e| Err(anyhow::anyhow!("history append task failed: {e}")))
        {
            warn!(
                worker = %self.worker_id,
                step,
                error = %e,
                "Incremental history append failed; the worker continues without it"
            );
        }
    }

    /// The metadata line of this run's history log.
    ///
    /// The dispatch facts (`task`, `temperature`, `review_after`,
    /// `network_offline`) are replayed from the turn config rather than kept
    /// twice, so a continuation describes the run it continues.
    fn history_meta(&self, config: &TurnConfig<'_>) -> WorkerHistory {
        WorkerHistory {
            task: config.task.to_string(),
            group: self.meta.group.clone(),
            role: self.meta.role,
            model: config.model.to_string(),
            temperature: config.temperature,
            repo_path: self
                .meta
                .repo_path
                .clone()
                .unwrap_or_else(|| self.worktree.repo_root.to_string_lossy().to_string()),
            base_commit: self.worktree.base_commit.clone(),
            base_branch: self.worktree.base_branch.clone(),
            branch: self.worktree.branch.clone(),
            network_offline: config.network_offline,
            verify: self.verify.map(str::to_string),
            client_env: self.client_env.to_vec(),
            max_turns: config.max_turns,
            review_after: config.review_after.map(str::to_string),
            revision: self.meta.revision,
            auto_continues: self.meta.auto_continues,
            owner: Some(self.meta.owner.clone()),
            messages: Vec::new(),
        }
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
            self.push_message(msg);
            for tc in tool_calls {
                let args = tc.function.arguments;
                let truncated = if args.len() > 200 {
                    let cut = args.floor_char_boundary(197);
                    format!("{}...", &args[..cut])
                } else {
                    args
                };
                self.push_message(ChatMessage::tool_result(
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
            self.push_message(msg);
            self.push_message(ChatMessage::text(Role::User, NO_COMMAND_NUDGE));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        MAX_TURNS_LIMIT, ProgressWatch, REPEAT_BLOCK_LIMIT, STAGNATION_SAMPLE_TURNS,
        extension_budget, parse_shortstat,
    };

    #[test]
    fn self_grant_budget_is_half_the_dispatch_budget() {
        assert_eq!(extension_budget(150), 75);
        assert_eq!(extension_budget(4), 2);
        // A budget too small to halve cannot be extended at all.
        assert_eq!(extension_budget(1), 0);
        assert_eq!(extension_budget(0), 0);
    }

    #[test]
    fn self_grant_budget_never_passes_the_manifest_ceiling() {
        assert_eq!(
            extension_budget(MAX_TURNS_LIMIT * 4),
            MAX_TURNS_LIMIT,
            "an absurd dispatch budget is still clamped to the ceiling"
        );
        assert!(
            extension_budget(usize::MAX) <= MAX_TURNS_LIMIT,
            "a saturating dispatch budget must not overflow the ceiling"
        );
    }

    #[test]
    fn only_a_byte_identical_repeat_counts_as_a_repetition() {
        let mut watch = ProgressWatch::default();
        assert_eq!(watch.register_command("sed -n '1,5p' f"), None);
        assert_eq!(
            watch.register_command("  sed -n '1,5p' f  "),
            Some(1),
            "surrounding whitespace is not a different command"
        );
        assert_eq!(
            watch.register_command("sed -n '6,9p' f"),
            None,
            "a different command is progress, not a repetition"
        );
        assert_eq!(
            watch.register_command("sed -n '6,9p' f"),
            Some(1),
            "the block count is per command, not per worker"
        );
    }

    #[test]
    fn a_repeated_command_reaches_the_park_limit() {
        let mut watch = ProgressWatch::default();
        assert_eq!(watch.register_command("ls -la"), None);
        for expected in 1..=REPEAT_BLOCK_LIMIT {
            assert_eq!(watch.register_command("ls -la"), Some(expected));
        }
    }

    #[test]
    fn an_unchanged_repository_accumulates_turns_until_the_nudge() {
        let mut watch = ProgressWatch::default();
        assert_eq!(
            watch.record_sample(Some("head-a\nstat".to_string())),
            0,
            "the first sample has nothing to compare against"
        );
        assert_eq!(
            watch.record_sample(Some("head-a\nstat".to_string())),
            STAGNATION_SAMPLE_TURNS
        );
        assert_eq!(
            watch.record_sample(Some("head-a\nstat".to_string())),
            2 * STAGNATION_SAMPLE_TURNS
        );
        assert_eq!(
            watch.record_sample(Some("head-b\nstat".to_string())),
            0,
            "any change, including a commit, resets the streak"
        );
    }

    #[test]
    fn a_shortstat_line_yields_files_insertions_and_deletions() {
        assert_eq!(
            parse_shortstat(" 5 files changed, 120 insertions(+), 340 deletions(-)"),
            Some((5, 120, 340))
        );
        assert_eq!(
            parse_shortstat(" 1 file changed, 2 insertions(+)"),
            Some((1, 2, 0)),
            "git omits a zero section and singularises the rest"
        );
        assert_eq!(parse_shortstat(""), None);
        assert_eq!(parse_shortstat(" no diff "), None);
    }

    #[test]
    fn a_sample_that_could_not_be_taken_is_never_read_as_no_change() {
        let mut watch = ProgressWatch::default();
        assert_eq!(watch.record_sample(None), 0);
        assert_eq!(
            watch.record_sample(None),
            0,
            "a failed git call must not push a stuck worker towards the nudge"
        );
    }
}
