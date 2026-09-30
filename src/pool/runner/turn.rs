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

use std::borrow::Cow;
use std::path::Path;

use anyhow::{Context, Result};
use tracing::{info, warn};

use crate::agent::{AgentRunner, ChatMessage, LlmResponse, Role, ToolCall};
use crate::manifest::MAX_TURNS_LIMIT;
use crate::worktree::{WorktreeGuard, git};

use super::super::buffer::build_step_log;
use super::super::registry::{RegistryStatus, WorkerMeta};
use super::super::state::WorkerState;
use super::super::steer::drain_steer_messages;
use super::super::WorkerPool;
use super::pause::PauseRequest;
use super::sentinels::{
    COMPLETION_SENTINEL, is_completion_request, parse_ask_orchestrator, parse_request_turns,
    summarize_command,
};

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
const REPEAT_REFUSAL: &str =
    "You already ran this exact command; its output has not changed (see above). Take a different action.";

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

/// Commit `path` like [`WorktreeGuard::commit_changes`], off the runtime
/// thread: stage everything, commit when dirty, and still report the branch
/// when it already carries commits beyond `base_commit`.
pub(super) fn commit_all_preserving(
    path: &Path,
    repo_root: &Path,
    branch: &str,
    base_commit: &str,
    message: &str,
) -> anyhow::Result<Option<String>> {
    if WorktreeGuard::commit_all(path, message)? {
        return Ok(Some(branch.to_string()));
    }
    if !base_commit.is_empty()
        && crate::worktree::git(
            repo_root,
            "rev-list",
            &[
                "rev-list",
                "--count",
                &format!("{base_commit}..{branch}"),
            ],
        )
        .map(|o| {
            String::from_utf8_lossy(&o.stdout)
                .trim()
                .parse::<u64>()
                .unwrap_or(0)
                > 0
        })
        .unwrap_or(false)
    {
        return Ok(Some(branch.to_string()));
    }
    Ok(None)
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
pub(super) fn parse_shortstat(line: &str) -> Option<(usize, usize, usize)> {
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
pub(super) async fn shortstat_of(path: &Path, base: &str) -> Option<(usize, usize, usize)> {
    let path = path.to_path_buf();
    let base = if base.is_empty() {
        "HEAD".to_string()
    } else {
        base.to_string()
    };
    let output = tokio::task::spawn_blocking(move || {
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
    pub step: &'a mut usize,
    pub current_max_turns: &'a mut usize,
    pub last_assistant_text: &'a mut String,
    pub consecutive_no_cmd: &'a mut usize,
    /// Optional shell command run through the same bash path before a
    /// completion sentinel is honoured. `None` disables the gate.
    pub verify: Option<&'a str>,
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

        // --- Automatic checkpoint (both phases) ---
        if *self.step > 0 && (*self.step).is_multiple_of(AUTO_CHECKPOINT_TURNS) {
            self.checkpoint().await;
        }

        // --- Stagnation detector (implementer only, like the sentinels) ---
        if config.apply_sentinels {
            self.check_stagnation().await;
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
                        // Safe checkpoint of uncommitted worktree changes so work is never lost.
                        // `commit_changes` shells out to git, so it runs off the runtime thread.
                        {
                            let path = self.worktree.path.clone();
                            let message = format!(
                                "worker({}): checkpoint step {} before pause (error: {})",
                                self.worker_id, *self.step, e
                            );
                            let committed =
                                tokio::task::spawn_blocking(move || {
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
                return self.park_on_loop(config, &llm_resp, &cmd_summary, blocks).await;
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

        // --- Orchestrator control sentinels (implementer only) ---
        if config.apply_sentinels {
            // REQUEST_TURNS, bounded by the self-grant budget: a worker may
            // add at most half of the budget its dispatch was given, so no
            // model can walk itself to the manifest ceiling unattended.
            if let Some(additional) = parse_request_turns(&cmd_str) {
                let budget = extension_budget(self.dispatch_max_turns);
                let granted = self.current_max_turns.saturating_sub(self.dispatch_max_turns);
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
                    self.messages.push(ChatMessage::text(
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

        self.push_exchange(
            llm_resp.content,
            llm_resp.reasoning_content,
            llm_resp.tool_calls.zip(llm_resp.tool_call_id),
            output_text,
        );

        Ok(TurnOutcome::Continue)
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

        let Some(verify) = self.verify.filter(|v| !v.is_empty()) else {
            return Ok(TurnOutcome::Completed { verified: None });
        };

        self.meta.metrics.verify_runs += 1;
        let (output, code) = self.run_gated(verify).await?;

        let exit = code.unwrap_or(-1);
        if exit == 0 {
            return Ok(TurnOutcome::Completed { verified: Some(true) });
        }

        // Verification failed. Record a step log and push the output back to
        // the model so it can fix the problems before completing again. After
        // three failed verifications the worker completes anyway, flagged
        // unverified.
        self.meta.metrics.verify_failures += 1;
        if self.meta.metrics.verify_failures >= 3 {
            return Ok(TurnOutcome::Completed { verified: Some(false) });
        }
        let label = format!("[verify] {}", summarize_command(verify));
        let step_log = build_step_log(*self.step, &label, output.clone(), code);
        {
            let mut lock = self.pool.workers.write().await;
            if let Some(w) = lock.get_mut(self.worker_id) {
                w.logs.push(step_log);
            }
        }
        let output_text = format!(
            "VERIFICATION FAILED (exit {exit}) - fix these problems before completing:\n{output}"
        );
        // The completion turn is replayed with the same rules as a command
        // turn, so the next request never carries a dangling tool_call.
        self.push_exchange(
            llm_resp.content.clone(),
            llm_resp.reasoning_content.clone(),
            llm_resp.tool_calls.clone().zip(llm_resp.tool_call_id.clone()),
            output_text,
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
            llm_resp.tool_calls.clone().zip(llm_resp.tool_call_id.clone()),
            REPEAT_REFUSAL.to_string(),
        );
        if let Some(answer) = answer
            && !answer.trim().is_empty()
            && answer.trim() != "resume"
        {
            self.messages.push(ChatMessage::text(
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
            commit_all_preserving(&path, &repo_root, &branch, &base_commit, &message)
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
    async fn run_gated(&self, command: &str) -> Result<(String, Option<i32>)> {
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
        let runner = match &build_permit {
            Some(permit) => Cow::Owned(self.runner.clone().with_build_jobs(permit.jobs())),
            None => Cow::Borrowed(self.runner),
        };
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
            self.messages.push(msg);
            self.messages.push(ChatMessage::tool_result(tc_id, output_text));
        } else {
            let content = if content.trim().is_empty() {
                "I will execute a bash command.".to_string()
            } else {
                content
            };
            let msg = ChatMessage::text(Role::Assistant, content).with_reasoning_content(reasoning);
            self.messages.push(msg);
            self.messages.push(ChatMessage::text(Role::User, output_text));
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
        assert_eq!(watch.record_sample(Some("head-a\nstat".to_string())), STAGNATION_SAMPLE_TURNS);
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
