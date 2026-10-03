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
//! escalate" nudge -- earned earlier, once and never fatally, by a worker whose
//! dispatch already named the files to edit and that only reads anyway --
//! `REQUEST_TURNS` may only add half the dispatch's budget,
//! and every 20 turns the worktree is checkpoint-committed so a kill or a
//! crash cannot lose the work.
//!
//! The read-only detector is the one guard with three steps, because a worker
//! that ignored two nudges will ignore a third: it first demands the edit,
//! then hands back the plan its own task spells out (see [`edit_plan`]), and
//! finally parks the worker on the orchestrator instead of paying for more
//! turns of reading. It is a guard against an *implementer* that reads instead
//! of writing, so it is armed only for that role: a consolidator and the
//! review phases both make their progress without editing (see
//! [`TurnConfig::read_only_exempt`]). The turns the harness answers itself --
//! the consolidator verbs and a background-job wait -- are progress the
//! worktree sample cannot see, so they restart the streak rather than
//! lengthening it ([`ProgressWatch::note_harness_progress`]).

use std::collections::BTreeMap;
use std::path::Path;
use std::time::Duration;

use anyhow::{Context, Result};
use tracing::{info, warn};

use crate::agent::exec::tree_fingerprint;
use crate::agent::{AgentRunner, ChatMessage, LlmResponse, Role, ToolCall};
use crate::manifest::MAX_TURNS_LIMIT;
use crate::worktree::{BaseSync, WorktreeGuard, git};

use super::super::WorkerPool;
use super::super::admission::AdmissionClass;
use super::super::buffer::build_step_log;
use super::super::registry::{RegistryStatus, WorkerMeta, WorkerRole};
use super::super::revision::{WorkerHistory, append_history_message_in};
use super::super::state::{WorkerReport, WorkerState};
use super::super::steer::drain_steer_messages_in;
use super::history::compact_history;
use super::pause::PauseRequest;
use super::sentinels::{
    COMPLETION_SENTINEL, REPORT_FIELD_BYTES, REPORT_FOLLOWUP, is_completion_request,
    parse_ask_orchestrator, parse_consolidate_merge, parse_consolidate_steer,
    parse_consolidate_wait, parse_consolidator_verdicts, parse_kill_job, parse_report,
    parse_request_turns, parse_wait_job, summarize_command,
};

/// Prefix used by both tool results and code-block command output messages.
pub(super) const COMMAND_OUTPUT_PREFIX: &str = "COMMAND OUTPUT (exit code: ";

/// Prefix of verification feedback; user-role feedback remains an instruction.
pub(super) const VERIFICATION_OUTPUT_PREFIX: &str = "VERIFICATION FAILED (exit ";

/// Follow-up when the model produced no executable bash command.
pub(super) const NO_COMMAND_NUDGE: &str =
    "ERROR: No bash command found. You MUST call the `bash` tool with your command.";

/// Prefix of a tool output an isolation guard produced instead of running the
/// command, paired with the short rule identifier the audit log carries.
const ISOLATION_BLOCK_PREFIXES: &[(&str, &str)] = &[
    (
        "COMMAND BLOCKED BY INTERCEPTOR:",
        "destructive_command_interceptor",
    ),
    (
        "COMMAND BLOCKED BY WORKTREE GUARDRAIL:",
        "worktree_guardrail",
    ),
    (
        "BLOCKED: the sandbox could not be prepared",
        "sandbox_unavailable",
    ),
];

/// Bound on the `reason` field of an audit line: the guard's own message is
/// short, but a bound keeps a future message from carrying a command's text.
const AUDIT_REASON_BYTES: usize = 200;

/// Classify a command output an isolation guard produced instead of running
/// the command.
///
/// The guards answer in-band -- the model has to be able to recover -- so the
/// turn engine recognises the refusal by its fixed prefix and reports the
/// guard's own reason, bounded, never the command or its environment.
fn isolation_block(output: &str) -> Option<(&'static str, String)> {
    let (prefix, rule) = ISOLATION_BLOCK_PREFIXES
        .iter()
        .find(|(prefix, _)| output.starts_with(prefix))?;
    let reason = output[prefix.len()..]
        .lines()
        .find(|line| !line.trim().is_empty())
        .unwrap_or("")
        .trim();
    let mut reason = reason.to_string();
    if reason.len() > AUDIT_REASON_BYTES {
        reason.truncate(
            reason
                .char_indices()
                .nth(AUDIT_REASON_BYTES)
                .map_or(reason.len(), |(i, _)| i),
        );
    }
    Some((rule, reason))
}

/// Turns between automatic checkpoint commits, so work left behind by a kill
/// or a crash is never more than this old.
const AUTO_CHECKPOINT_TURNS: usize = 20;

/// Cap on the assistant text scanned for a REPORT block. A block is at most the
/// four [`REPORT_FIELD_BYTES`] fields plus the command that carries it, so the
/// newest slice always holds a whole block while a worker that never writes one
/// cannot grow the scan buffer past this bound.
const REPORT_SCAN_BYTES: usize = REPORT_FIELD_BYTES * 5;

/// Turns between repository samples of the stagnation detector.
const STAGNATION_SAMPLE_TURNS: usize = 10;

/// Turns without a repository change that force the "stop exploring" nudge.
const STAGNATION_TURNS_LIMIT: usize = 30;

/// Consecutive read-only turns after which a dispatch that already names a
/// file to edit is told to write that edit instead of reading one more file.
const READ_ONLY_NUDGE_TURNS: usize = 15;

/// Environment override of the first read-only nudge threshold, in turns.
const READ_ONLY_NUDGE_ENV: &str = "POOL_READ_ONLY_NUDGE_TURNS";

/// Environment override of the threshold that carries the plan; defaults to
/// twice the first threshold.
const READ_ONLY_ESCALATE_ENV: &str = "POOL_READ_ONLY_ESCALATE_TURNS";

/// Environment override of the threshold that parks the worker on the
/// orchestrator; defaults to three times the first threshold.
const READ_ONLY_PAUSE_ENV: &str = "POOL_READ_ONLY_PAUSE_TURNS";

/// Files the edit plan names at most, so the nudge carrying it stays one
/// sentence however path-heavy the dispatch is.
const EDIT_PLAN_FILES: usize = 6;

/// Identifiers the edit plan attaches to one file at most.
const EDIT_PLAN_IDENTIFIERS: usize = 3;

/// Words one backticked identifier may carry at most, which keeps a quoted
/// sentence of prose out of the plan.
const EDIT_PLAN_IDENTIFIER_WORDS: usize = 3;

/// Bytes one identifier of the edit plan may occupy.
const EDIT_PLAN_IDENTIFIER_BYTES: usize = 48;

/// Bytes one path of the edit plan may occupy. Bounding the number of files is
/// not enough on its own: a token with a file extension can be arbitrarily
/// long, and the nudge that carries the plan stays one sentence only if each
/// name in it is bounded too.
const EDIT_PLAN_PATH_BYTES: usize = 120;

/// Bytes of the dispatch the pause question quotes back, so a pause carries
/// the spec without pasting a whole dispatch into the orchestrator's terminal.
const TASK_QUESTION_BYTES: usize = 240;

/// Distinct commands of a read-only streak the pause question quotes back, so
/// the orchestrator sees what the worker spent its turns on.
const READ_ONLY_RECENT_COMMANDS: usize = 4;

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

/// Thresholds of the read-only detector: the consecutive read-only turns that
/// earn each of its three steps.
#[derive(Debug, Clone, Copy)]
struct ReadOnlyThresholds {
    /// Consecutive read-only turns before the first "write the edit now" nudge.
    first: usize,
    /// Consecutive read-only turns before the nudge that carries the plan.
    plan: usize,
    /// Consecutive read-only turns before the worker is parked on the
    /// orchestrator to be told what to do.
    pause: usize,
}

/// The read-only detector's defaults: the first nudge at
/// [`READ_ONLY_NUDGE_TURNS`], the plan at twice that and the pause at three
/// times.
fn named_file_defaults() -> ReadOnlyThresholds {
    ReadOnlyThresholds {
        first: READ_ONLY_NUDGE_TURNS,
        plan: READ_ONLY_NUDGE_TURNS.saturating_mul(2),
        pause: READ_ONLY_NUDGE_TURNS.saturating_mul(3),
    }
}

/// Read-only thresholds for `task`, or `None` when the dispatch names no file:
/// a worker that still has to find the code is exploring, and the stagnation
/// detector keeps its existing timing for it.
///
/// The thresholds themselves are overridable through the environment, so the
/// read-only budget can be tuned without a rebuild.
fn read_only_thresholds(task: &str) -> Option<ReadOnlyThresholds> {
    if !task_names_files(task) {
        return None;
    }
    let defaults = named_file_defaults();
    Some(ReadOnlyThresholds {
        first: env_threshold(READ_ONLY_NUDGE_ENV).unwrap_or(defaults.first),
        plan: env_threshold(READ_ONLY_ESCALATE_ENV).unwrap_or(defaults.plan),
        pause: env_threshold(READ_ONLY_PAUSE_ENV).unwrap_or(defaults.pause),
    })
}

/// Parse an environment override: a positive turn count, or `None` for an
/// unset, unparsable or zero value, so a bad override keeps the default rather
/// than nudging a worker on its first turn.
fn parse_threshold(raw: &str) -> Option<usize> {
    raw.trim().parse::<usize>().ok().filter(|turns| *turns > 0)
}

/// The environment override `name`, when it holds a usable turn count.
fn env_threshold(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .as_deref()
        .and_then(parse_threshold)
}

/// Whether the dispatch names a file or a path to edit, so the worker can act
/// on the spec instead of searching for it.
///
/// A token counts as a path when it carries a dotted file extension or a
/// directory separator; prose tokens with neither do not.
fn task_names_files(task: &str) -> bool {
    task.split_whitespace().any(is_path_token)
}

/// Whether `token` reads as a path or a file name: it carries a dotted file
/// extension or a directory separator, and prose tokens with neither do not.
fn is_path_token(token: &str) -> bool {
    let token = token
        .trim_matches(|c: char| !c.is_ascii_alphanumeric() && !"./_-".contains(c))
        .trim_end_matches('.');
    if token.len() < 3 {
        return false;
    }
    if token.contains('/') {
        return true;
    }
    match token.rsplit_once('.') {
        Some((stem, ext)) => {
            !stem.is_empty()
                && (2..=6).contains(&ext.len())
                && ext.chars().all(|c| c.is_ascii_alphabetic())
        }
        None => false,
    }
}

/// One file of the edit plan: the path the task named, and the identifiers
/// quoted near it, so the plan points at a function rather than a file.
#[derive(Debug, Clone, PartialEq, Eq)]
struct EditPlanEntry {
    path: String,
    identifiers: Vec<String>,
}

/// The edit plan read back out of the dispatch: the files it names and the
/// `fn`/type identifiers written next to them, in the order the task wrote
/// them.
///
/// Pure and language-agnostic: a task in any language is mined for the paths
/// and backticked identifiers it happens to carry.
fn edit_plan(task: &str) -> Vec<EditPlanEntry> {
    let mut entries: Vec<EditPlanEntry> = Vec::new();
    // The file a backticked identifier belongs to: the last one the task named
    // before it, so `fn foo` attaches to the file it was written next to.
    let mut owner: Option<usize> = None;
    // Identifiers quoted before any file was named, waiting for the first file
    // the task writes afterwards.
    let mut pending: Vec<String> = Vec::new();
    // Alternating prose and quoted spans: a backticked name may carry spaces
    // (`fn check_read_only`), so it is read between the backticks rather than
    // token by token.
    let mut quoted = false;
    for span in task.split('`') {
        if quoted {
            quoted = false;
            let span = span.trim();
            // A dispatch that writes its paths in backticks names files just as
            // plainly as one that writes them bare, so a quoted path is a file
            // of the plan in its own right -- never an identifier of the file
            // named before it.
            if let Some(path) = quoted_path(span) {
                owner = note_path(&mut entries, &path, &mut pending);
                continue;
            }
            if let Some(index) = owner.filter(|_| !entries.is_empty()) {
                let entry = entries.get_mut(index).expect("owner is in range");
                if entry.identifiers.len() < EDIT_PLAN_IDENTIFIERS && is_identifier(span) {
                    entry.identifiers.push(span.to_string());
                }
            } else if pending.len() < EDIT_PLAN_IDENTIFIERS && is_identifier(span) {
                pending.push(span.to_string());
            }
            continue;
        }
        quoted = true;
        // Prose is scanned in the order it is written, so the plan lists the
        // files as the task listed them.
        for token in span.split_whitespace() {
            let Some(path) = path_of(token) else { continue };
            owner = note_path(&mut entries, &path, &mut pending);
        }
    }
    entries
}

/// Record `path` as a file of the plan and report the entry it belongs to.
///
/// Files are kept in the order the task named them and capped, so a path past
/// the cap is named in no entry; identifiers quoted before a file waited for
/// it, because a task names the function and then the file it lives in as often
/// as the other way round.
fn note_path(
    entries: &mut Vec<EditPlanEntry>,
    path: &str,
    pending: &mut Vec<String>,
) -> Option<usize> {
    if let Some(index) = entries.iter().position(|e| e.path == path) {
        return Some(index);
    }
    if entries.len() >= EDIT_PLAN_FILES {
        return None;
    }
    entries.push(EditPlanEntry {
        path: path.to_string(),
        identifiers: std::mem::take(pending),
    });
    Some(entries.len() - 1)
}

/// The path a backticked span names, or `None` when it names none.
///
/// A span carrying whitespace is a quoted sentence or a multi-word name, not a
/// path, however much of it reads like one.
fn quoted_path(span: &str) -> Option<String> {
    if span.is_empty() || span.chars().any(char::is_whitespace) {
        return None;
    }
    path_of(span)
}

/// The path a whitespace token names, or `None` when it names none. Sentence
/// punctuation around the token is not part of the path, so `src/a.rs,` and
/// `src/a.rs.` are both read as `src/a.rs`.
fn path_of(token: &str) -> Option<String> {
    let token = token
        .trim_matches(|c: char| !c.is_ascii_alphanumeric() && !"./_-".contains(c))
        .trim_end_matches('.');
    (token.len() <= EDIT_PLAN_PATH_BYTES && is_path_token(token)).then(|| token.to_string())
}

/// Whether a backticked span reads as an identifier -- `fn x`, a type name, a
/// command -- rather than as a sentence of prose: it is at most a few words
/// long and every character is one an identifier, a call or a flag is written
/// with, so a quoted sentence of the dispatch stays out of the plan.
///
/// Alphanumerics are Unicode, not ASCII: the extractor is language-agnostic,
/// and a name like `vérifier` is as ordinary in a dispatch as `verify`.
fn is_identifier(text: &str) -> bool {
    let text = text.trim();
    if text.is_empty() || text.len() > EDIT_PLAN_IDENTIFIER_BYTES {
        return false;
    }
    let words: Vec<&str> = text.split_whitespace().collect();
    words.len() <= EDIT_PLAN_IDENTIFIER_WORDS
        && words.iter().all(|word| {
            word.chars()
                .all(|c| c.is_alphanumeric() || "_-.:,()[]<>*&'!/=+".contains(c))
        })
}

/// The plan half of the second nudge: the files and identifiers the task itself
/// named, ready to act on. A dispatch that names a file the extractor cannot
/// read still gets the demand, so the nudge is never a sentence with a hole in
/// it.
fn edit_plan_text(entries: &[EditPlanEntry]) -> String {
    if entries.is_empty() {
        return "Edit now. Open the file the task names and Write the first change in your next command.".to_string();
    }
    let files = entries
        .iter()
        .map(|entry| {
            if entry.identifiers.is_empty() {
                entry.path.clone()
            } else {
                format!("{} ({})", entry.path, entry.identifiers.join(", "))
            }
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!("Edit now. Files the task names: {files}. Write the first change in your next command.")
}

/// The read-only detector's verdict for one turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReadOnlyNudge {
    /// The first nudge of this streak, at the first threshold.
    First { read_only_turns: usize },
    /// The nudge that carries the task's own plan, at the second threshold.
    Plan { read_only_turns: usize },
    /// The last step: park the worker on the orchestrator, at the third.
    Pause { read_only_turns: usize },
}

/// Consecutive read-only turns and the nudges that streak has already earned.
///
/// Fed one repository sample per turn -- the same sample the stagnation
/// detector compares against -- so "read-only" means "the worktree did not
/// change", not "the command looked like a read".
#[derive(Default)]
struct ReadOnlyStreak {
    /// Last sample seen: the baseline the next one is compared against.
    last_sample: Option<String>,
    /// Consecutive turns whose command left the worktree unchanged.
    read_only_turns: usize,
    /// The first nudge has been sent for this streak.
    nudged: bool,
    /// The plan-carrying nudge has been sent for this streak.
    planned: bool,
    /// The orchestrator pause has been sent for this streak.
    paused: bool,
    /// Summaries of the distinct commands this streak has run, oldest first,
    /// so a pause question can say what the worker spent its turns on.
    recent_commands: Vec<String>,
}

impl ReadOnlyStreak {
    /// Remember one command of the streak, keeping the newest few distinct
    /// summaries: the pause question quotes them back, and an unbounded list
    /// would grow with every turn a worker only reads.
    ///
    /// Folded in by [`ProgressWatch::register_command`], the one place the
    /// repetition detector sees each command too, rather than at execution
    /// time.
    fn note_command(&mut self, command: &str) {
        let command = command.trim();
        if command.is_empty() {
            return;
        }
        let summary = summarize_command(command);
        if let Some(index) = self.recent_commands.iter().position(|c| *c == summary) {
            // Move it to the newest slot instead of duplicating it: a loop
            // over the same two reads stays a two-command summary.
            self.recent_commands.remove(index);
        } else {
            while self.recent_commands.len() >= READ_ONLY_RECENT_COMMANDS {
                self.recent_commands.remove(0);
            }
        }
        self.recent_commands.push(summary);
    }

    /// Start the streak over without a new repository sample: a turn the
    /// harness answered itself is progress the sample cannot see, so the
    /// counters and the nudges this streak earned are dropped and the next
    /// unchanged turn is counted as the first of a fresh streak.
    fn restart_streak(&mut self) {
        self.read_only_turns = 0;
        self.nudged = false;
        self.planned = false;
        self.paused = false;
        self.recent_commands.clear();
    }

    /// What this streak read, as one clause for the pause question.
    fn read_summary(&self) -> String {
        if self.recent_commands.is_empty() {
            return "no command recorded".to_string();
        }
        self.recent_commands.join("; ")
    }
}

impl ReadOnlyStreak {
    /// Fold one turn's repository sample in, remember the command the turn
    /// ran, and report the nudge, if any, the streak has earned. Each of the
    /// three thresholds fires once per streak and in order, so a streak of
    /// unchanged worktree reaches the plan and then the pause no matter where
    /// its thresholds sit.
    ///
    /// `None` is a sample git could not answer: it is not evidence of progress,
    /// so it neither extends nor resets the streak.
    fn record(
        &mut self,
        sample: Option<String>,
        limits: ReadOnlyThresholds,
    ) -> Option<ReadOnlyNudge> {
        let sample = sample?;
        if self.last_sample.as_deref() != Some(sample.as_str()) {
            self.read_only_turns = 0;
            self.nudged = false;
            self.planned = false;
            self.paused = false;
            self.recent_commands.clear();
            self.last_sample = Some(sample);
            return None;
        }
        self.read_only_turns += 1;
        if !self.nudged && self.read_only_turns >= limits.first {
            self.nudged = true;
            return Some(ReadOnlyNudge::First {
                read_only_turns: self.read_only_turns,
            });
        }
        if self.nudged && !self.planned && self.read_only_turns >= limits.plan {
            self.planned = true;
            return Some(ReadOnlyNudge::Plan {
                read_only_turns: self.read_only_turns,
            });
        }
        if self.planned && !self.paused && self.read_only_turns >= limits.pause {
            self.paused = true;
            return Some(ReadOnlyNudge::Pause {
                read_only_turns: self.read_only_turns,
            });
        }
        None
    }
}

/// First read-only nudge: name the streak, which counts turns spent reading,
/// and demand the edit or a question.
fn read_only_nudge_text(read_only_turns: usize) -> String {
    format!(
        "You have read {read_only_turns} files; write the first edit now, or ASK_ORCHESTRATOR what is missing."
    )
}

/// The second nudge of a read-only streak: the demand, plus the plan the task
/// itself spells out, because a worker that kept reading past the first nudge
/// is not short of permission but of a next step.
fn read_only_plan_text(read_only_turns: usize, plan: &str) -> String {
    format!("Still no edit after {read_only_turns} read-only turns. {plan}")
}

/// The last step of a read-only streak: the question parked on the orchestrator,
/// naming what the worker read and that it has not edited, so the decision is
/// taken off the worker's own turn budget.
fn read_only_pause_question(read_only_turns: usize, task: &str, read: &str) -> String {
    format!(
        "No edit after {read_only_turns} read-only turns. Read so far: {read}. Task: {}. The worker is still exploring and has not written a change; it was handed the plan its task names. Decide: point it at the first edit, or steer it elsewhere.",
        summarized_task(task)
    )
}

/// The task trimmed to one line of [`TASK_QUESTION_BYTES`], so a pause
/// question carries the spec without pasting a whole dispatch into the
/// orchestrator's terminal.
fn summarized_task(task: &str) -> String {
    // The question supplies its own punctuation, so a trailing full stop on the
    // dispatch would make a doubled one.
    let one_line = task
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .trim_end_matches('.')
        .to_string();
    if one_line.len() <= TASK_QUESTION_BYTES {
        return one_line;
    }
    let cut = one_line.floor_char_boundary(TASK_QUESTION_BYTES.saturating_sub(3));
    format!("{}...", &one_line[..cut])
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
async fn sync_base_for_completion(
    worktree: &WorktreeGuard,
    round_base: Option<String>,
) -> Result<BaseSync> {
    let path = worktree.path.clone();
    let repo_root = worktree.repo_root.clone();
    let branch = worktree.branch.clone();
    let base_commit = worktree.base_commit.clone();
    let base_branch = worktree.base_branch.clone();
    tokio::task::spawn_blocking(move || {
        if let Some(round_base) = round_base {
            return WorktreeGuard::sync_round_base_at(
                &path,
                &repo_root,
                &branch,
                &base_commit,
                &round_base,
            );
        }
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

/// A recorded successful run of a heavy command, so the completion gate can
/// reuse it on an unchanged tree instead of recompiling and re-testing.
struct VerifySuccess {
    /// Worktree fingerprint unchanged across the successful run.
    fingerprint: String,
    /// Step the run succeeded on, named in the "verify reused" log.
    step: usize,
}

/// Keep at most one record per permitted turn.
const VERIFY_SUCCESS_LIMIT: usize = MAX_TURNS_LIMIT;

/// Cross-turn state of the two loop detectors, plus the last successful run
/// of each heavy command the completion gate can reuse.
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
    /// The read-only streak detector, fed the same per-turn sample.
    read_only: ReadOnlyStreak,
    /// Last successful run of each heavy command, keyed by the exact
    /// command string. The completion gate reuses an entry when the same
    /// command is issued as the verify gate on an unchanged tree.
    verify_success: BTreeMap<String, VerifySuccess>,
}

impl ProgressWatch {
    /// Record `command` as the latest one and report how many consecutive
    /// turns it repeats, or `None` when it is a fresh command.
    fn register_command(&mut self, command: &str) -> Option<usize> {
        let command = command.trim();
        // The read-only detector is folded at the top of the next turn, so this
        // turn's command is noted here: a pause raised on that turn has to
        // report what the worker actually read.
        self.read_only.note_command(command);
        if self.last_command.as_deref() == Some(command) {
            self.repeat_blocks += 1;
            Some(self.repeat_blocks)
        } else {
            self.repeat_blocks = 0;
            self.last_command = Some(command.to_string());
            None
        }
    }

    /// Record a turn the harness answered itself -- a consolidator verb, a
    /// background-job wait -- as progress, for every role. Neither changes
    /// the worktree, so without this they would read to the detector as one
    /// more turn spent looking instead of the work the turn actually did.
    fn note_harness_progress(&mut self) {
        self.read_only.restart_streak();
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

    /// Remember a successful run of a heavy `command` on the tree
    /// fingerprinted `fingerprint`, at `step`. Only the most recent success
    /// per command is kept; a new command past the cap evicts the oldest.
    fn record_verify_success(&mut self, command: String, fingerprint: String, step: usize) {
        if !self.verify_success.contains_key(&command)
            && self.verify_success.len() >= VERIFY_SUCCESS_LIMIT
            && let Some(oldest) = self
                .verify_success
                .iter()
                .min_by_key(|(_, r)| r.step)
                .map(|(k, _)| k.clone())
        {
            self.verify_success.remove(&oldest);
        }
        self.verify_success
            .insert(command, VerifySuccess { fingerprint, step });
    }

    /// The step a recorded successful run of `command` can be reused from
    /// when the tree still fingerprints to `current`, or `None` when the
    /// command never succeeded here or succeeded on a tree that has since
    /// changed. A reused run is only sound when the command *and* the tree
    /// both match, so anything else re-runs.
    fn reusable_verify_step(&self, command: &str, current: &str) -> Option<usize> {
        let recorded = self.verify_success.get(command)?;
        (recorded.fingerprint == current).then_some(recorded.step)
    }

    /// Drop any recorded success of `command`: a later failure means an
    /// earlier pass no longer describes the tree, so the gate re-runs it.
    fn invalidate_verify_success(&mut self, command: &str) {
        self.verify_success.remove(command);
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
    /// Whether the read-only nudge/plan/pause escalation is armed at all.
    /// It is a guard against an implementer that reads instead of writing, so
    /// a consolidator and the review phases set this: reviewing, merging,
    /// steering, waiting and running the gate are their job, and a long
    /// read-only streak there is the work itself, not a stuck worker.
    pub read_only_exempt: bool,
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
    /// The structured report of the completion turn, once one has been parsed.
    /// Written here rather than returned so a completion that is refused (a
    /// verify failure, a base merge) keeps the report it already gave.
    pub report: &'a mut Option<WorkerReport>,
    /// Whether the worker has already been asked once for a missing report.
    /// One ask per run: a second would let a confused model trade turns for
    /// completions that never carry one.
    pub report_asked: &'a mut bool,
    /// The newest [`REPORT_SCAN_BYTES`] of the completion sequence's assistant
    /// texts: the completion turn, the one follow-up it may cost, and later
    /// replays. The REPORT block can be split across them -- the prose of one
    /// turn and the bash command of another -- so they are scanned together
    /// rather than one by one, but the buffer never grows past its bound.
    pub report_text: &'a mut String,
}

/// Append one assistant message to the completion scan buffer: its prose, then
/// its bash command with `\n` escapes unfolded. A block written as
/// `printf 'REPORT\ndone: ...'` carries its line breaks as that two-character
/// escape, so unfolding lets the same parser see it.
fn append_report_text(buffer: &mut String, llm_resp: &LlmResponse) -> Option<WorkerReport> {
    buffer.push_str(&llm_resp.content);
    buffer.push('\n');
    if let Some(command) = &llm_resp.command {
        let command = command.replace("\\n", "\n");
        buffer.push_str(&command);
        buffer.push('\n');
    }
    // Parse before trimming: a block that arrives just before an oversized
    // command is still seen. Then keep only the newest slice, at a character
    // boundary, so a worker that never writes a block cannot grow this second
    // conversation buffer without bound.
    let parsed = parse_report(buffer);
    if buffer.len() > REPORT_SCAN_BYTES {
        let cut = buffer.ceil_char_boundary(buffer.len() - REPORT_SCAN_BYTES);
        buffer.drain(..cut);
    }
    parsed
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

        // --- Change detectors (implementer only, like the sentinels) ---
        if config.apply_sentinels {
            self.check_changes(config).await?;
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
        // The REPORT block is not guaranteed to sit in the same assistant
        // message as the sentinel: the completion turn can carry the sentinel
        // while the block arrives in the one follow-up, or both are written
        // into the bash command that requests completion. Every assistant
        // text from the completion sequence onward is scanned together.
        if config.apply_sentinels
            && self.report.is_none()
            && (*self.report_asked
                || llm_resp
                    .command
                    .as_deref()
                    .is_some_and(is_completion_request))
        {
            // The scan returns the block the moment it appears, before the
            // buffer trims, so a block seen before a long stretch of
            // sentinel-less turns cannot be lost to the bound.
            if let Some(parsed) = append_report_text(self.report_text, &llm_resp) {
                *self.report = Some(parsed);
            }
        }

        // --- Command extraction ---
        let cmd_str = match llm_resp.command {
            Some(ref cmd) if is_completion_request(cmd) => {
                // Only the implementer's completion carries the report the
                // orchestrator reads: the reviewer's sentinel approves the
                // audit, and asking it for a report would cost a turn for a
                // payload nobody stores.
                return self
                    .handle_completion(&llm_resp, config.apply_sentinels)
                    .await;
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
        //
        // The label goes to the step unless the pool itself is holding a
        // harness-side wait as this worker's command in flight. `status` then
        // shows what the worker is really doing, and the stall detector reads
        // the wait as work instead of as a step that has gone idle. The
        // question is asked of the pool's own record of the label, never of
        // `last_command`: that field holds model-written text, and a command
        // beginning with the wait's name would otherwise own the label for
        // the rest of the worker's life, every later step reading the same
        // unchanged label and skipping its own write again.
        let wait_owns_label = self.pool.harness_wait_in_flight(self.worker_id);
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
                    if !wait_owns_label {
                        *last_command = label.clone();
                    }
                }
            })
            .await;

        // --- Registry update, coalesced by the pool's writer ---
        // The durable row records the step's own command whatever the wait
        // label shows, so a restart never adopts a wait label as the worker's
        // last command.
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

        // --- Background job sentinels ---
        // Handled before the repetition detector: waiting on a job is the
        // sanctioned alternative to sleep-polling, so a second `WAIT_JOB` for
        // the same job is a legitimate follow-up rather than a repeated
        // command, and neither sentinel runs a bash step.
        if config.apply_sentinels {
            if let Some(job) = parse_kill_job(&cmd_str) {
                let (output, code) = self.stop_job(job);
                return self
                    .record_harness_result(&llm_resp, &label, output, code)
                    .await;
            }
            if let Some(job) = parse_wait_job(&cmd_str) {
                let (output, code) = self.wait_on_job(job).await;
                return self
                    .record_harness_result(&llm_resp, &label, output, code)
                    .await;
            }
        }

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
        let (output, code) = self
            .run_gated(&cmd_str, AdmissionClass::Exploratory)
            .await?;

        // --- Consolidator verbs (harness side, never bash) ---
        // The sandbox holds no git credentials, so these run here, on the
        // harness, through the same machinery the orchestrator uses. Only a
        // consolidator sees them; an ordinary worker's identical command stays
        // plain bash.
        if self.meta.role == WorkerRole::Consolidate {
            // A merge integrates the group's finished branches into this
            // worktree, so the branch must survive the guard's cleanup.
            if let Some(ids) = parse_consolidate_merge(&cmd_str) {
                let merged = self
                    .pool
                    .consolidate_merge(self.meta, self.worktree, &ids)
                    .await;
                return self
                    .consolidator_reply(&label, merged.observation, merged.integrated, &llm_resp)
                    .await;
            }
            // A steer routes a failure or a conflict back to the worker that
            // owns it, exactly as the orchestrator's own steer would.
            if let Some((id, message)) = parse_consolidate_steer(&cmd_str) {
                let observation = self.pool.consolidate_steer(self.meta, &id, message).await;
                return self
                    .consolidator_reply(&label, observation, false, &llm_resp)
                    .await;
            }
            // A wait blocks until the group stops, spending no turn on it.
            if let Some((ids, timeout)) = parse_consolidate_wait(&cmd_str) {
                let observation = self.pool.consolidate_wait(self.meta, &ids, timeout).await;
                return self
                    .consolidator_reply(&label, observation, false, &llm_resp)
                    .await;
            }
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

        self.record_command_result(&llm_resp, &label, output, code)
            .await
    }

    /// Record one turn the harness answered itself: the job sentinels and the
    /// consolidator verbs. Such a turn is progress for the read-only detector
    /// -- it never touches the worktree -- and its answer reaches the history
    /// exactly as an executed command's does.
    async fn record_harness_result(
        &mut self,
        llm_resp: &LlmResponse,
        label: &str,
        output: String,
        code: Option<i32>,
    ) -> Result<TurnOutcome> {
        self.watch.note_harness_progress();
        self.record_command_result(llm_resp, label, output, code)
            .await
    }

    /// Record one answered turn: the tool result the model sees, the bounded
    /// step log and the durable history append.
    ///
    /// Shared by an executed command and by the job sentinels, which answer a
    /// turn without running bash, so all three reach the history the same way.
    async fn record_command_result(
        &mut self,
        llm_resp: &LlmResponse,
        label: &str,
        output: String,
        code: Option<i32>,
    ) -> Result<TurnOutcome> {
        let output_text = format!(
            "{COMMAND_OUTPUT_PREFIX}{}):\n```\n{}\n```",
            code.unwrap_or(-1),
            output
        );

        // The log entry is built *after* `output_text` so `output` is moved
        // rather than cloned, and both text fields are clamped to a hard
        // ceiling.
        let step_log = build_step_log(*self.step, label, output, code);

        {
            let mut lock = self.pool.workers.write().await;
            if let Some(w) = lock.get_mut(self.worker_id) {
                w.logs.push(step_log);
            }
        }

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

    /// Answer a harness-mediated consolidator verb: the observation stands in
    /// for the bash output, so the step is recorded as an ordinary exit-0 step
    /// and the loop continues without spending a sandbox turn on it.
    ///
    /// `preserve` keeps the consolidator's branch past this guard's cleanup,
    /// which only a merge that landed other workers' commits needs.
    async fn consolidator_reply(
        &mut self,
        label: &str,
        observation: String,
        preserve: bool,
        llm_resp: &LlmResponse,
    ) -> Result<TurnOutcome> {
        if preserve {
            self.worktree.preserve_branch = true;
        }
        self.record_harness_result(llm_resp, label, observation, Some(0))
            .await
    }

    /// Block on background job `job` for one `WAIT_JOB` budget.
    ///
    /// The worker counts as running a command for the whole wait, so the stall
    /// detector and the watch views see a live step rather than an idle one. No
    /// bash slot and no admission permit is taken: waiting runs nothing.
    async fn wait_on_job(&self, job: u64) -> (String, Option<i32>) {
        let _running = self.pool.command_running(self.worker_id);
        let limit = Duration::from_secs(crate::agent::jobs::wait_job_secs());
        match self.runner.wait_job(job, limit).await {
            Some(wait) => wait.report(job),
            None => (
                format!("No job {job} is running; it already ended or never existed."),
                Some(1),
            ),
        }
    }

    /// Stop background job `job`.
    fn stop_job(&self, job: u64) -> (String, Option<i32>) {
        if self.runner.kill_job(job) {
            (
                format!("Job {job} stopped; its process group was killed."),
                Some(0),
            )
        } else {
            (
                format!("No job {job} is running; it already ended or never existed."),
                Some(1),
            )
        }
    }

    /// Handle a completion sentinel: run the verify gate (if any) and either
    /// complete or push the failure back to the model for another turn.
    ///
    /// A verify command whose last run passed on the current tree is reused
    /// instead of re-run: variant A is skipped and the reuse is disclosed to
    /// the model, while variant B and the side-effect audit still run.
    async fn handle_completion(
        &mut self,
        llm_resp: &LlmResponse,
        require_report: bool,
    ) -> Result<TurnOutcome> {
        info!(
            worker = %self.worker_id,
            step = *self.step,
            "Worker requested completion"
        );
        if !llm_resp.content.trim().is_empty() {
            *self.last_assistant_text = llm_resp.content.clone();
        }

        // The completion turn must carry a REPORT block, which the response
        // scanner stores the moment it appears. A worker that omitted one is
        // asked exactly once, before any git work: the answer is what the
        // orchestrator reads, so it is worth one turn and never more.
        match (require_report, self.report.is_some()) {
            (true, true) => {}
            (true, false) if !*self.report_asked => {
                *self.report_asked = true;
                info!(
                    worker = %self.worker_id,
                    step = *self.step,
                    "Completion carried no REPORT block; asking once"
                );
                self.push_exchange(
                    llm_resp.content.clone(),
                    llm_resp.reasoning_content.clone(),
                    llm_resp
                        .tool_calls
                        .clone()
                        .zip(llm_resp.tool_call_id.clone()),
                    REPORT_FOLLOWUP.to_string(),
                );
                // A report-only retry is protocol repair, not another work
                // turn; allow it even at the dispatch's final turn.
                *self.step = self.step.saturating_sub(1);
                return Ok(TurnOutcome::Continue);
            }
            // Already asked once, or a phase whose completion carries no
            // report: accept it and fall back to the summary the harness has
            // always derived from the last message.
            _ => {}
        }

        // A consolidator's closing report is also the round's bookkeeping: the
        // workers it reports as `fixed`, plus the ones it steered and left
        // stopped, are absorbed by it and retire with its branch. Recorded
        // here, at the completion turn, so the record exists before anything
        // merges the consolidator.
        if self.meta.role == WorkerRole::Consolidate {
            let fixed: Vec<String> = parse_consolidator_verdicts(&llm_resp.content)
                .into_iter()
                .filter(|(_, verdict)| *verdict == "fixed")
                .map(|(id, _)| id)
                .collect();
            self.pool
                .record_consolidator_absorbed(self.meta, &fixed)
                .await;
        }

        let round_base = super::super::steer::read_source(&self.pool.scratch, self.worker_id)
            .and_then(|source| source.round_base);
        let merged = match sync_base_for_completion(self.worktree, round_base).await? {
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

        // Reuse a recorded successful run of this exact command on an
        // unchanged tree: the worker already ran the project's gates, so
        // re-running variant A would only recompile and re-test the same
        // bytes. Variant B and the side-effect audit still run -- B is what
        // A cannot prove. Anything different (another command, a changed
        // file, no recorded pass) runs variant A as usual.
        let reused_from = self.reusable_verify_step(verify).await;
        let (output, code) = match reused_from {
            Some(step) => {
                info!(
                    worker = %self.worker_id,
                    from_step = step,
                    "verify reused from step {step}"
                );
                (String::new(), Some(0))
            }
            None => self.run_gated(verify, AdmissionClass::Completion).await?,
        };

        let exit = code.unwrap_or(-1);
        if exit == 0 {
            return self
                .finish_verified_completion(llm_resp, verify, &gate_baseline, reused_from)
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

    /// Disclose a reused variant A to the model on a successful completion, so
    /// it never reads a reused pass as a fresh one. Flushed at the completion
    /// boundary; a refusal carries the same note in its message instead.
    fn push_reuse_note(&mut self, reused_from: Option<usize>) {
        if let Some(step) = reused_from {
            self.push_message(ChatMessage::text(
                Role::User,
                format!(
                    "verify reused from step {step}: canonical variant A was not rerun (identical command and tree). Variant B and the side-effect audit still ran and passed."
                ),
            ));
        }
    }

    /// Finish a completion whose canonical verify passed -- freshly or reused:
    /// audit what it left behind, then re-run the same command in the
    /// divergent environment.
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
        reused_from: Option<usize>,
    ) -> Result<TurnOutcome> {
        let repo_root = self.worktree.repo_root.clone();
        let worktree_path = self.worktree.path.clone();
        let worker_id = self.worker_id.to_string();

        // When variant A was reused, every refusal must say so: the model
        // must not read a reused pass as a fresh one.
        let reuse_note = match reused_from {
            Some(step) => format!(
                "verify reused from step {step}: the canonical verify (variant A) was not rerun (identical command and tree); only variant B and the side-effect audit ran this time.\n\n"
            ),
            None => String::new(),
        };

        // Side-effect audit of the canonical run, against the pre-gate
        // baseline: the suite must leave the repository, its refs and its
        // processes exactly as it found them.
        let effects = super::divergent::audit(&repo_root, &worktree_path, &worker_id, baseline);
        if !effects.is_empty() {
            super::divergent::cleanup(&repo_root, &effects);
            let refusal = format!(
                "{reuse_note}{}",
                super::divergent::side_effect_refusal(&effects)
            );
            self.note_isolation_block(
                "side_effect_audit",
                &super::divergent::side_effect_summary(&effects),
                verify,
            );
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
            self.push_reuse_note(reused_from);
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
            .run_gated_with_env(verify, AdmissionClass::Completion, divergent_env.clone())
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
            let refusal = format!(
                "{reuse_note}{}",
                super::divergent::side_effect_refusal(&effects_b)
            );
            self.note_isolation_block(
                "side_effect_audit",
                &super::divergent::side_effect_summary(&effects_b),
                verify,
            );
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
            self.push_reuse_note(reused_from);
            return Ok(TurnOutcome::Completed {
                verified: Some(true),
            });
        }
        let differing = super::divergent::divergent_names(&divergent_env);
        let refusal = format!(
            "{reuse_note}{}",
            super::divergent::divergence_refusal(verify, &differing, code_b, &output_b)
        );
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

    /// One repository sample this turn, the fingerprint both change detectors
    /// compare against their own previous one. `None` when git could not
    /// answer, so a failed sample is never read as "nothing changed".
    async fn sample_repository(&self) -> Option<String> {
        let path = self.worktree.path.clone();
        // `git` is a blocking subprocess, so it must not run on a runtime thread.
        tokio::task::spawn_blocking(move || repository_sample(&path))
            .await
            .ok()
            .flatten()
    }

    /// Sample the worktree once and feed both change detectors with it.
    ///
    /// The sample is taken every turn when the read-only detector is active
    /// (a dispatch that names a file to edit); otherwise it keeps the
    /// [`STAGNATION_SAMPLE_TURNS`] cadence it has today, so a dispatch that
    /// names no file pays nothing new.
    async fn check_changes(&mut self, config: &TurnConfig<'_>) -> Result<()> {
        // An exempt phase never walks the escalation, and so also keeps the
        // every-ten-turns sample cadence the stagnation detector has always
        // had: it pays nothing for a detector that would never fire.
        let read_only = if config.read_only_exempt {
            None
        } else {
            read_only_thresholds(config.task)
        };
        let stagnation_due = *self.step > 0 && (*self.step).is_multiple_of(STAGNATION_SAMPLE_TURNS);
        if read_only.is_none() && !stagnation_due {
            return Ok(());
        }
        let sample = self.sample_repository().await;
        self.check_stagnation(sample.clone());
        if let Some(limits) = read_only {
            self.check_read_only(config, sample, limits).await?;
        }
        Ok(())
    }

    /// Tell a worker that stopped changing anything to make the edit or
    /// escalate, sampled every [`STAGNATION_SAMPLE_TURNS`] turns.
    fn check_stagnation(&mut self, sample: Option<String>) {
        if *self.step == 0 || !(*self.step).is_multiple_of(STAGNATION_SAMPLE_TURNS) {
            return;
        }
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

    /// Tell a worker whose worktree has not changed for a run of turns to make
    /// its first edit, hand it the plan its own task spells out when the first
    /// nudge is ignored, and park it on the orchestrator when that is ignored
    /// too.
    ///
    /// Only called for a dispatch whose spec already named the files to edit,
    /// where a long read-only streak means the worker is stuck rather than
    /// still looking for the code, so the streak is counted per turn rather
    /// than per [`STAGNATION_SAMPLE_TURNS`] window. The first two steps only
    /// inject guidance; the third hands the decision to the orchestrator,
    /// because a worker that has already ignored two nudges will not read a
    /// third.
    async fn check_read_only(
        &mut self,
        config: &TurnConfig<'_>,
        sample: Option<String>,
        limits: ReadOnlyThresholds,
    ) -> Result<()> {
        let Some(nudge) = self.watch.read_only.record(sample, limits) else {
            return Ok(());
        };
        self.meta.metrics.stagnation_nudges += 1;
        let read_only_turns = self.watch.read_only.read_only_turns;
        warn!(
            worker = %self.worker_id,
            step = *self.step,
            read_only_turns,
            first_threshold = limits.first,
            "Worker has only been reading; escalating past the nudge"
        );
        match nudge {
            ReadOnlyNudge::First { read_only_turns } => {
                let text = read_only_nudge_text(read_only_turns);
                self.push_message(ChatMessage::text(Role::User, text));
            }
            ReadOnlyNudge::Plan { read_only_turns } => {
                let plan = edit_plan_text(&edit_plan(config.task));
                self.push_message(ChatMessage::text(
                    Role::User,
                    read_only_plan_text(read_only_turns, &plan),
                ));
            }
            ReadOnlyNudge::Pause { read_only_turns } => {
                let question = read_only_pause_question(
                    read_only_turns,
                    config.task,
                    &self.watch.read_only.read_summary(),
                );
                self.pause_on_read_only(config, &question).await?;
            }
        }
        Ok(())
    }

    /// Park a worker whose second nudge was ignored and wait for the
    /// orchestrator's decision, then hand that decision to the worker as
    /// guidance for the turn after it.
    async fn pause_on_read_only(&mut self, config: &TurnConfig<'_>, question: &str) -> Result<()> {
        self.meta.metrics.loop_pauses += 1;
        let answer = self
            .pool
            .pause_for_orchestrator(PauseRequest {
                worker_id: self.worker_id,
                question,
                step: *self.step,
                max_turns: *self.current_max_turns,
                last_command: &format!("paused_read_only: {question}"),
                model: config.model,
                meta: self.meta,
            })
            .await?;
        let Some(answer) = answer else {
            return Ok(());
        };
        info!(
            worker = %self.worker_id,
            step = *self.step,
            msg = %answer,
            "Worker resumed from read-only pause by orchestrator guidance"
        );
        if !answer.trim().is_empty() && answer.trim() != "resume" {
            self.push_message(ChatMessage::text(
                Role::User,
                format!("ORCHESTRATOR GUIDANCE:\n{answer}"),
            ));
        }
        Ok(())
    }

    /// Record one isolation denial in the hub log and the worker's counters.
    ///
    /// The refusal otherwise lives only in the worker's conversation, which is
    /// deleted when the worker is retired; the audit line is what survives.
    /// The command is summarised, never quoted in full, and no environment
    /// value is logged.
    fn note_isolation_block(&mut self, rule: &str, reason: &str, command: &str) {
        self.meta.metrics.isolation_blocks += 1;
        warn!(
            target: "audit",
            worker = %self.worker_id,
            owner = %self.meta.owner,
            rule = %rule,
            reason = %reason,
            command = %summarize_command(command),
            "isolation block: command refused by a guard"
        );
    }

    /// Run through resource admission and the bash semaphore, retaining both
    /// permits until execution finishes or is cancelled.
    async fn run_gated(
        &mut self,
        command: &str,
        class: AdmissionClass,
    ) -> Result<(String, Option<i32>)> {
        let heavy = crate::agent::is_heavy_command(command);
        // Capture the tree the command is about to run on, before taking a
        // slot: a pass is only reusable when the command leaves the tree
        // exactly as it found it. Only the canonical run is recorded -- the
        // divergent variant B must not certify or clear a canonical pass.
        let before = if heavy {
            self.current_fingerprint().await
        } else {
            None
        };
        // Invalidate before execution so errors and cancellation cannot preserve a stale pass.
        self.watch.invalidate_verify_success(command);
        let result = self.run_gated_with_env(command, class, Vec::new()).await;
        if heavy {
            self.update_verify_success(command, &result, before).await;
        }
        result
    }

    /// Shared execution path for exploratory commands and both completion
    /// variants; the overlay wins over the sanitized environment defaults.
    async fn run_gated_with_env(
        &mut self,
        command: &str,
        class: AdmissionClass,
        extra_env: Vec<(String, String)>,
    ) -> Result<(String, Option<i32>)> {
        let heavy = crate::agent::is_heavy_command(command);
        // A completion-class command is a harness gate -- the completion verify
        // or its divergent variant -- not the model's own work. It must never
        // become a background job nobody waits on, and its budget is the
        // absolute job ceiling rather than the step timeout, so a slow gate
        // simply takes longer and its real exit code decides.
        let gate = matches!(class, AdmissionClass::Completion);
        let build_permit = if heavy {
            // A queued command is not inactivity, including completion gates.
            let _waiting = self
                .pool
                .wait_for_build_slot(self.worker_id, self.pool.admission.waiting() + 1);
            Some(self.pool.admission.acquire(class).await)
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
        if gate {
            runner = runner
                .without_job_conversion()
                .with_command_timeout(crate::agent::jobs::job_max_secs());
        }
        if let Some(permit) = &build_permit {
            runner = runner.with_build_jobs(permit.jobs());
        }
        runner = runner.with_extra_env(extra_env);
        // Heavy commands lease the worker's build dir on first use; light
        // commands reuse it without allocating another lease.
        runner.build_target_dir = if heavy {
            self.worktree.build_dir().await
        } else {
            self.worktree.leased_build_dir().map(Path::to_path_buf)
        };
        let _running = self.pool.command_running(self.worker_id);
        let (output, code) = runner.execute_bash(&self.worktree.path, command).await?;
        // An isolation guard answered instead of running the command: record
        // it in the hub log and the worker's counters, so the refusal outlives
        // the worker's history.
        if let Some((rule, reason)) = isolation_block(&output) {
            self.note_isolation_block(rule, &reason, command);
        }
        // A command that outlived its budget is now a background job, and the
        // build slot it was admitted under moves into the job: a job never
        // outlives the admission it was granted.
        if let Some(job) = runner.take_last_job_id()
            && let Some(permit) = build_permit
        {
            runner.attach_job_guard(job, Box::new(permit));
        }
        Ok((output, code))
    }

    /// Only an exit-zero run on a stable source tree certifies reusable content.
    async fn update_verify_success(
        &mut self,
        command: &str,
        result: &Result<(String, Option<i32>)>,
        before: Option<String>,
    ) {
        if matches!(result, Ok((_, Some(0))))
            && let Some(before) = before
            && self.current_fingerprint().await.as_ref() == Some(&before)
        {
            self.watch
                .record_verify_success(command.to_string(), before, *self.step);
        }
    }

    async fn current_fingerprint(&self) -> Option<String> {
        let path = self.worktree.path.clone();
        tokio::task::spawn_blocking(move || tree_fingerprint(&path))
            .await
            .ok()
            .flatten()
    }

    /// The step a recorded successful run of `verify` can be reused from on
    /// the current tree, or `None` to run variant A. The fingerprint is only
    /// taken when the command has a recorded success, so a worker that never
    /// ran its gate pays nothing here.
    async fn reusable_verify_step(&self, verify: &str) -> Option<usize> {
        self.watch.verify_success.get(verify)?;
        self.watch
            .reusable_verify_step(verify, &self.current_fingerprint().await?)
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
        EDIT_PLAN_FILES, EDIT_PLAN_PATH_BYTES, LlmResponse, MAX_TURNS_LIMIT, ProgressWatch,
        READ_ONLY_NUDGE_TURNS, REPEAT_BLOCK_LIMIT, REPORT_SCAN_BYTES, ReadOnlyNudge,
        ReadOnlyStreak, ReadOnlyThresholds, STAGNATION_SAMPLE_TURNS, TASK_QUESTION_BYTES,
        append_report_text, edit_plan, edit_plan_text, extension_budget, isolation_block,
        named_file_defaults, parse_shortstat, parse_threshold, read_only_nudge_text,
        read_only_pause_question, read_only_plan_text, read_only_thresholds, summarized_task,
        task_names_files,
    };

    /// A response with no tool call and no reasoning, for scan-buffer tests.
    fn scanned(content: &str, command: Option<&str>) -> LlmResponse {
        LlmResponse {
            content: content.to_string(),
            reasoning_content: None,
            command: command.map(str::to_string),
            tool_calls: None,
            tool_call_id: None,
            invalid_utf8_lines: 0,
        }
    }

    /// A guard's in-band refusal is classified by its prefix, with the guard's
    /// own reason bounded; anything else is not a block.
    #[test]
    fn a_guard_refusal_is_classified_by_its_prefix() {
        let (rule, reason) = isolation_block(
            "COMMAND BLOCKED BY WORKTREE GUARDRAIL:\noutside the worktree\nPlease run ...",
        )
        .expect("a guardrail refusal is a block");
        assert_eq!(rule, "worktree_guardrail");
        assert_eq!(reason, "outside the worktree");

        let (rule, reason) = isolation_block(
            "BLOCKED: the sandbox could not be prepared (no landlock); the command was not run.",
        )
        .expect("a sandbox refusal is a block");
        assert_eq!(rule, "sandbox_unavailable");
        assert_eq!(reason, "(no landlock); the command was not run.");

        assert!(
            isolation_block("COMMAND OUTPUT (exit code: 0)\nhello").is_none(),
            "a normal command output is not a block"
        );
        assert!(isolation_block("").is_none());
    }
    #[test]
    fn the_report_scan_buffer_is_bounded_to_its_newest_bytes() {
        let long = "界".repeat(4096);
        let mut buffer = String::new();
        for _ in 0..64 {
            append_report_text(&mut buffer, &scanned(&long, Some(&"y".repeat(4096))));
            assert!(
                buffer.len() <= REPORT_SCAN_BYTES,
                "the scan buffer grew past its bound: {}",
                buffer.len()
            );
        }
        // The newest message is what the scan returns, however much bounded
        // text came before it.
        let report = append_report_text(
            &mut buffer,
            &scanned(
                "REPORT\ndone: bounded scan\nfiles: src/a.rs\ntests: cargo test: passed\nrisks: none\n",
                None,
            ),
        )
        .expect("the newest block must survive the bound");
        assert_eq!(report.done, "bounded scan");
        assert!(buffer.len() <= REPORT_SCAN_BYTES);
    }

    #[test]
    fn a_report_block_split_across_messages_still_parses() {
        let mut buffer = String::new();
        assert_eq!(
            append_report_text(&mut buffer, &scanned("REPORT\n", None)),
            None
        );
        let report = append_report_text(
            &mut buffer,
            &scanned("done: split capture\nfiles: src/a.rs\n", None),
        )
        .expect("a block split across turns must parse");
        assert_eq!(report.done, "split capture");
        assert_eq!(report.files, "src/a.rs");
    }

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

    /// The read-only detector over a sequence of samples: each of its three
    /// steps lands on its threshold turn exactly once, an edit resets the
    /// streak, and the next streak starts over from the first nudge.
    #[test]
    fn a_read_only_streak_walks_its_three_steps_once_each() {
        let limits = ReadOnlyThresholds {
            first: 3,
            plan: 6,
            pause: 9,
        };
        let mut streak = ReadOnlyStreak::default();
        let same = || Some("head-a\nstat".to_string());
        // The first sample only fixes the baseline the next one is compared to.
        assert_eq!(streak.record(same(), limits), None);
        for turns in 1..limits.first {
            assert_eq!(
                streak.record(same(), limits),
                None,
                "no nudge after only {turns} read-only turns"
            );
        }
        assert_eq!(
            streak.record(same(), limits),
            Some(ReadOnlyNudge::First { read_only_turns: 3 })
        );
        // The first nudge is sent once, not on every following turn.
        assert_eq!(streak.record(same(), limits), None);
        assert_eq!(streak.record(same(), limits), None);
        assert_eq!(
            streak.record(same(), limits),
            Some(ReadOnlyNudge::Plan { read_only_turns: 6 }),
            "the second threshold carries the plan"
        );
        for turns in 7..limits.pause {
            assert_eq!(
                streak.record(same(), limits),
                None,
                "no pause after only {turns} read-only turns"
            );
        }
        assert_eq!(
            streak.record(same(), limits),
            Some(ReadOnlyNudge::Pause { read_only_turns: 9 }),
            "the third threshold parks the worker on the orchestrator"
        );
        assert_eq!(
            streak.record(same(), limits),
            None,
            "the pause is sent once per streak"
        );
        // An edit resets the streak, and the next streak nudges again: the
        // sample taken right after the edit is the new baseline.
        assert_eq!(
            streak.record(Some("head-a\nedit".to_string()), limits),
            None
        );
        // The first of these fixes the post-edit baseline; the rest are the
        // start of the new streak, still short of the threshold.
        for _ in 1..=limits.first {
            assert_eq!(streak.record(same(), limits), None);
        }
        assert_eq!(
            streak.record(same(), limits),
            Some(ReadOnlyNudge::First { read_only_turns: 3 }),
            "after an edit the streak starts over and nudges again"
        );
    }

    /// A turn the harness answered itself is progress the worktree sample
    /// cannot see: it restarts the streak rather than lengthening it, so a
    /// consolidator that spends its turns on `CONSOLIDATE_WAIT` is never
    /// nudged for "not editing" -- whatever nudges the streak had earned go
    /// with it.
    #[test]
    fn a_harness_answered_turn_is_progress_not_another_read_only_turn() {
        let limits = ReadOnlyThresholds {
            first: 3,
            plan: 6,
            pause: 9,
        };
        let mut watch = ProgressWatch::default();
        let same = || Some("head-a\nstat".to_string());
        // One sample past the baseline is one read-only turn, so this reaches
        // the streak's first threshold -- and earns the first nudge.
        for _ in 0..=limits.first {
            watch.read_only.record(same(), limits);
        }
        assert_eq!(watch.read_only.read_only_turns, limits.first);

        // A consolidator verb or a job wait leaves the worktree untouched, so
        // nothing in the sample moves: the turn must be counted as progress.
        watch.note_harness_progress();
        assert_eq!(watch.read_only.read_only_turns, 0);

        // The turns after it are therefore the start of a fresh streak, which
        // is still short of the plan threshold the old streak had earned.
        for turns in 1..limits.first {
            assert_eq!(
                watch.read_only.record(same(), limits),
                None,
                "the streak must start over: no plan after only {turns} read-only turns"
            );
        }
        assert_eq!(
            watch.read_only.record(same(), limits),
            Some(ReadOnlyNudge::First { read_only_turns: 3 }),
            "the fresh streak still walks its steps, from the first threshold"
        );
    }

    /// The first read-only nudge names the streak and offers the worker the
    /// two ways out.
    #[test]
    fn the_first_read_only_nudge_offers_an_edit_or_a_question() {
        let first = read_only_nudge_text(15);
        assert!(first.contains("read 15 files"), "got {first:?}");
        assert!(first.contains("write the first edit now"), "got {first:?}");
        assert!(first.contains("ASK_ORCHESTRATOR"), "got {first:?}");
    }

    /// The pause question is the third step's whole output: it names the streak,
    /// quotes back what the worker read, and hands the decision to the
    /// orchestrator instead of letting it read on.
    #[test]
    fn the_read_only_pause_question_hands_the_decision_to_the_orchestrator() {
        let question = read_only_pause_question(
            45,
            "Fix the guard in src/pool/runner/turn.rs",
            "grep -rn nudge src/; sed -n 1,80p src/pool/runner/turn.rs",
        );
        assert!(question.contains("45 read-only turns"), "got {question:?}");
        assert!(
            question.contains("grep -rn nudge src/"),
            "the orchestrator is not told what the worker read: {question:?}"
        );
        assert!(
            question.contains("has not written a change"),
            "got {question:?}"
        );
        assert!(
            question.contains("src/pool/runner/turn.rs"),
            "got {question:?}"
        );
    }

    /// A sample git could not answer is not evidence of progress: it neither
    /// extends nor resets a read-only streak.
    #[test]
    fn an_unreadable_sample_leaves_a_read_only_streak_alone() {
        let limits = ReadOnlyThresholds {
            first: 2,
            plan: 4,
            pause: 6,
        };
        let mut streak = ReadOnlyStreak::default();
        assert_eq!(streak.record(Some("a".to_string()), limits), None);
        assert_eq!(streak.record(None, limits), None);
        assert_eq!(
            streak.record(Some("a".to_string()), limits),
            None,
            "the streak is at one turn, not two: the failed sample did not count"
        );
        assert_eq!(
            streak.record(Some("a".to_string()), limits),
            Some(ReadOnlyNudge::First { read_only_turns: 2 })
        );
    }

    /// The plan is read out of the dispatch the way a dispatch is written:
    /// a path, then the identifiers quoted next to it, in order and deduped.
    #[test]
    fn the_edit_plan_names_the_files_and_identifiers_the_task_writes() {
        let task = "TASK: [harness-E27] In src/pool/runner/turn.rs, add the plan \
                    to the second read-only nudge; the detector is `record` and \
                    the text is `read_only_nudge_text`. Then update tests/pool_test.rs.";
        let plan = edit_plan(task);
        assert_eq!(plan.len(), 2, "got {plan:?}");
        assert_eq!(plan[0].path, "src/pool/runner/turn.rs");
        assert_eq!(plan[0].identifiers, ["record", "read_only_nudge_text"]);
        assert_eq!(plan[1].path, "tests/pool_test.rs");
        assert!(plan[1].identifiers.is_empty(), "got {plan:?}");
    }

    /// A dispatch writes its paths in backticks about as often as bare, and a
    /// quoted path names a file of the plan exactly as a bare one does -- it is
    /// never mistaken for an identifier of the file named before it.
    #[test]
    fn a_backticked_path_names_a_file_of_the_plan() {
        let plan = edit_plan("edit `src/a.rs` and `app/main.py`, then `src/a.rs` again");
        assert_eq!(
            plan.iter().map(|e| e.path.as_str()).collect::<Vec<_>>(),
            ["src/a.rs", "app/main.py"],
            "a quoted path must be a file, deduplicated and in the order written"
        );
        assert!(
            plan.iter().all(|e| e.identifiers.is_empty()),
            "a quoted path is a file, not an identifier: {plan:?}"
        );
        // A quoted path still owns the identifiers written after it.
        let owned = edit_plan("touch `src/a.rs` and `fn helper`");
        assert_eq!(owned[0].path, "src/a.rs");
        assert_eq!(owned[0].identifiers, ["fn helper"]);
        // A quoted span carrying whitespace is a name or a sentence, not a
        // path, so it stays an identifier.
        let named = edit_plan("call `src/a.rs` from `fn main`");
        assert_eq!(named[0].identifiers, ["fn main"]);
    }

    /// A plan is bounded per name as well as per file count: an arbitrarily
    /// long token with a file extension cannot drag a wall of text into the
    /// nudge, and a name outside ASCII is read the same way an ASCII one is.
    #[test]
    fn the_edit_plan_bounds_each_name_and_reads_non_ascii_names() {
        // Bounded per file count, but each name is long enough to matter.
        let long_name = format!("src/{}.rs", "a".repeat(400));
        let plan = edit_plan(&format!("edit {long_name}"));
        assert!(
            plan.is_empty(),
            "a path past the byte cap must not enter the plan: {} bytes",
            long_name.len()
        );
        let just_over = format!("src/{}.rs", "b".repeat(EDIT_PLAN_PATH_BYTES));
        assert!(
            just_over.len() > EDIT_PLAN_PATH_BYTES,
            "the fixture must actually exceed the cap"
        );
        assert!(edit_plan(&format!("edit {just_over}")).is_empty());

        // A non-ASCII path and identifier are ordinary names, not prose.
        let unicode = edit_plan("mettre à jour `src/données.rs` : `fn vérifier`");
        assert_eq!(unicode[0].path, "src/données.rs");
        assert_eq!(unicode[0].identifiers, ["fn vérifier"]);
    }

    /// The extractor is language-agnostic: it mines whatever paths and
    /// backticked names a task carries, and bounds both, so a path-heavy or a
    /// prose-heavy dispatch cannot grow the nudge without limit.
    #[test]
    fn the_edit_plan_is_bounded_and_language_agnostic() {
        let many: String = (0..20).map(|i| format!("edit src/mod{i}.rs ")).collect();
        let plan = edit_plan(&many);
        assert_eq!(plan.len(), EDIT_PLAN_FILES, "got {} entries", plan.len());
        // Names beyond the per-file cap are dropped, not all of them kept.
        let many_names = "edit a.rs: `one` `two` `three` `four` `five`";
        assert_eq!(
            edit_plan(many_names)[0].identifiers,
            ["one", "two", "three"]
        );
        // A task names the function before the file it lives in as often as the
        // other way round, so a leading identifier attaches to the file after it.
        let leading = edit_plan("add the plan to `fn check_read_only` in src/lib.rs.");
        assert_eq!(leading[0].path, "src/lib.rs");
        assert_eq!(leading[0].identifiers, ["fn check_read_only"]);
        // A quoted sentence of prose is not an identifier.
        let prose = "fix a.rs: `the parser drops the last token when it sees one`";
        assert!(
            edit_plan(prose)[0].identifiers.is_empty(),
            "prose leaked into the plan: {:?}",
            edit_plan(prose)
        );
        // A task that names identifiers but no file has no plan to hand back.
        assert!(edit_plan("change `read_only_nudge_text` to be louder").is_empty());
        // Every language's file names come out the same way.
        let multi = "tweak src/a.rs, app/main.py and web/index.ts";
        assert_eq!(
            edit_plan(multi)
                .iter()
                .map(|e| e.path.as_str())
                .collect::<Vec<_>>(),
            ["src/a.rs", "app/main.py", "web/index.ts"]
        );
    }

    /// The second nudge carries the plan in the form the task asked for, and
    /// an identifier rides along with the file it was written next to.
    #[test]
    fn the_plan_nudge_reads_as_one_editable_sentence() {
        let plan = edit_plan("edit src/pool/runner/turn.rs, `fn check_read_only`");
        let text = read_only_plan_text(30, &edit_plan_text(&plan));
        assert!(text.contains("Edit now."), "got {text:?}");
        assert!(
            text.contains("Files the task names: src/pool/runner/turn.rs (fn check_read_only)."),
            "got {text:?}"
        );
        assert!(
            text.ends_with("Write the first change in your next command."),
            "got {text:?}"
        );
        assert!(text.contains("30 read-only turns"), "got {text:?}");
    }

    /// A task whose paths the extractor cannot read still gets a whole second
    /// nudge, not a sentence with an empty list in it.
    #[test]
    fn an_unreadable_task_still_gets_a_whole_plan_nudge() {
        let text = read_only_plan_text(30, &edit_plan_text(&edit_plan("fix the failing test")));
        assert!(!text.contains("task names: ."), "got {text:?}");
        assert!(text.contains("Edit now."), "got {text:?}");
        assert!(
            text.ends_with("Write the first change in your next command."),
            "got {text:?}"
        );
    }

    /// The dispatch quoted into a pause question is trimmed to a length a
    /// terminal can show, and a short one is left whole.
    #[test]
    fn a_pause_question_quotes_a_bounded_dispatch() {
        assert_eq!(summarized_task("fix src/a.rs"), "fix src/a.rs");
        // A dispatch ending in a full stop must not double it: the question
        // supplies its own punctuation.
        assert_eq!(summarized_task("fix src/a.rs."), "fix src/a.rs");
        let long: String = std::iter::repeat_n("word", 200)
            .collect::<Vec<_>>()
            .join(" ");
        let trimmed = summarized_task(&long);
        assert!(
            trimmed.len() <= TASK_QUESTION_BYTES,
            "got {}",
            trimmed.len()
        );
        assert!(trimmed.ends_with("..."), "got {trimmed:?}");
    }

    /// A dispatch that already names the files to edit gets the read-only
    /// detector and its earlier budget; one that names none leaves the detector
    /// off, so the existing stagnation timing stands for it.
    #[test]
    fn only_a_task_that_names_files_gets_the_read_only_detector() {
        let defaults = named_file_defaults();
        assert_eq!(defaults.first, READ_ONLY_NUDGE_TURNS);
        assert_eq!(defaults.plan, 2 * READ_ONLY_NUDGE_TURNS);
        assert_eq!(defaults.pause, 3 * READ_ONLY_NUDGE_TURNS);
        assert!(read_only_thresholds("edit src/pool/runner/turn.rs, see `Cargo.toml`").is_some());
        assert!(read_only_thresholds("fix the failing test").is_none());
    }

    #[test]
    fn only_a_path_or_a_file_extension_counts_as_naming_a_file() {
        assert!(task_names_files("rewrite src/lib.rs"));
        assert!(task_names_files("update `Cargo.toml` and models.yaml"));
        assert!(task_names_files("the bug is in src/pool"));
        assert!(!task_names_files("fix the failing test"));
        assert!(!task_names_files("e.g. rewrite the parser"));
        assert!(!task_names_files("parse v1.2 output"));
    }

    #[test]
    fn only_a_positive_turn_count_is_a_usable_override() {
        assert_eq!(parse_threshold("15"), Some(15));
        assert_eq!(parse_threshold(" 12 "), Some(12));
        assert_eq!(
            parse_threshold("0"),
            None,
            "a zero threshold would nudge on the first sample"
        );
        assert_eq!(parse_threshold("nonsense"), None);
        assert_eq!(parse_threshold(""), None);
    }

    /// The completion gate reuses a verify run only when the command *and*
    /// the tree both match a recorded success; anything else re-runs.
    mod verify_reuse {
        use super::ProgressWatch;
        use crate::agent::exec::tree_fingerprint;
        use std::path::{Path, PathBuf};

        fn scratch(tag: &str) -> PathBuf {
            let dir = std::env::temp_dir().join(format!(
                "turn-verify-{tag}-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).expect("create scratch dir");
            dir
        }

        fn git(dir: &Path, args: &[&str]) {
            let out = std::process::Command::new("git")
                .current_dir(dir)
                .args(args)
                .output()
                .unwrap_or_else(|e| panic!("git {args:?}: {e}"));
            assert!(
                out.status.success(),
                "git {args:?} failed: {}",
                String::from_utf8_lossy(&out.stderr)
            );
        }

        /// A committed one-file repository, the smallest tree a worker runs in.
        fn repo(tag: &str) -> PathBuf {
            let dir = scratch(tag);
            git(&dir, &["init", "-b", "master"]);
            git(&dir, &["config", "user.name", "t"]);
            git(&dir, &["config", "user.email", "t@localhost"]);
            std::fs::write(dir.join("seed.txt"), "seed\n").unwrap();
            git(&dir, &["add", "seed.txt"]);
            git(&dir, &["commit", "-m", "baseline"]);
            dir
        }

        /// An identical command on an identical tree is reused from the step
        /// it passed on, and the fingerprint is stable across calls.
        #[test]
        fn an_identical_command_and_tree_is_reused() {
            let dir = repo("reuse");
            let fp = tree_fingerprint(&dir).expect("fingerprint the seeded tree");
            let mut watch = ProgressWatch::default();
            watch.record_verify_success("cargo test".to_string(), fp.clone(), 4);
            assert_eq!(
                watch.reusable_verify_step("cargo test", &fp),
                Some(4),
                "the same command on the same tree must be reused"
            );
            assert_eq!(
                tree_fingerprint(&dir).as_deref(),
                Some(fp.as_str()),
                "an unchanged tree must fingerprint identically"
            );
        }

        /// A single changed file moves the fingerprint, so the gate re-runs.
        #[test]
        fn a_changed_file_forces_a_rerun() {
            let dir = repo("changed");
            let before = tree_fingerprint(&dir).expect("fingerprint before the edit");
            let mut watch = ProgressWatch::default();
            watch.record_verify_success("cargo test".to_string(), before.clone(), 4);

            std::fs::write(dir.join("seed.txt"), "seed\nmore\n").unwrap();
            let after = tree_fingerprint(&dir).expect("fingerprint after the edit");
            assert_ne!(before, after, "a tracked edit must move the fingerprint");
            assert_eq!(
                watch.reusable_verify_step("cargo test", &after),
                None,
                "a changed tree must re-run the gate"
            );
        }

        /// A different command has no recorded success, so the gate re-runs.
        #[test]
        fn a_different_command_forces_a_rerun() {
            let dir = repo("other-cmd");
            let fp = tree_fingerprint(&dir).expect("fingerprint the seeded tree");
            let mut watch = ProgressWatch::default();
            watch.record_verify_success("cargo test".to_string(), fp.clone(), 4);
            assert_eq!(
                watch.reusable_verify_step("cargo build", &fp),
                None,
                "a command that never succeeded here must re-run"
            );
        }

        /// A command that only ever failed is never recorded, so the gate
        /// re-runs it rather than reusing a pass that never happened.
        #[test]
        fn a_failed_run_is_never_reused() {
            let dir = repo("failed");
            let fp = tree_fingerprint(&dir).expect("fingerprint the seeded tree");
            let watch = ProgressWatch::default();
            assert_eq!(
                watch.reusable_verify_step("cargo test", &fp),
                None,
                "a command with no recorded success must re-run"
            );
        }

        /// A pass followed by a later failure of the same command on the
        /// same tree must not be reused: the failure drops the earlier pass.
        #[test]
        fn a_later_failure_invalidates_an_earlier_pass() {
            let dir = repo("stale");
            let fp = tree_fingerprint(&dir).expect("fingerprint the seeded tree");
            let mut watch = ProgressWatch::default();
            watch.record_verify_success("cargo test".to_string(), fp.clone(), 4);
            assert_eq!(
                watch.reusable_verify_step("cargo test", &fp),
                Some(4),
                "the pass is reusable before the failure"
            );
            // The same command later fails on the unchanged tree.
            watch.invalidate_verify_success("cargo test");
            assert_eq!(
                watch.reusable_verify_step("cargo test", &fp),
                None,
                "a command that has since failed must re-run"
            );
        }

        /// The success map is bounded: a new command past the cap evicts the
        /// oldest entry rather than growing without limit.
        #[test]
        fn the_success_map_is_bounded() {
            let mut watch = ProgressWatch::default();
            for step in 0..super::super::VERIFY_SUCCESS_LIMIT + 4 {
                watch.record_verify_success(format!("cmd-{step}"), "fp".to_string(), step);
            }
            // The oldest entries were evicted; the newest survives.
            assert_eq!(
                watch.reusable_verify_step("cmd-0", "fp"),
                None,
                "the oldest entry must have been evicted"
            );
            let newest = super::super::VERIFY_SUCCESS_LIMIT + 3;
            assert_eq!(
                watch.reusable_verify_step(&format!("cmd-{newest}"), "fp"),
                Some(newest),
                "the newest entry must survive"
            );
        }
    }
}
