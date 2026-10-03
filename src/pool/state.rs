//! Worker lifecycle state, the in-memory record, and the cheap read views.
//!
//! The types here are the *data* half of the pool: what a worker looks like
//! while it runs, what a caller sees when it polls, and when a terminal record
//! is old enough to be evicted.

use std::collections::HashMap;
use tokio::task::JoinHandle;

use crate::agent::AgentStepLog;
use crate::agent::jobs::JobStatus;

use super::buffer::{LogBuffer, LogStats};
use super::unix_timestamp;

/// Per-worker health counters, recorded while the run happens.
///
/// A summary alone cannot grade a worker: a run that needed 150 turns, was
/// refused three turn extensions and burned four turns on a repetition loop
/// completes with the same payload as a clean one. Each counter is moved by the
/// turn engine at the exact point its guard fires — never re-derived from the
/// log window afterwards — so two workers of the same task can be compared.
///
/// Every field defaults to zero, so a registry row written before these
/// counters existed still parses, and a worker that never moved one still
/// reports as "nothing measured" instead of as a healthy zero.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct WorkerMetrics {
    /// Turns performed by the implementer and reviewer loops together.
    pub turns_used: usize,
    /// Extra turns a `REQUEST_TURNS` sentinel was granted.
    pub extensions_granted: usize,
    /// `REQUEST_TURNS` asks past the self-grant budget that were refused.
    pub extensions_refused: usize,
    /// Automatic budget extensions granted once at the turn limit.
    pub auto_extensions_granted: usize,
    /// Commands answered by the repetition detector instead of being run.
    pub repeat_blocks: usize,
    /// "Stop exploring" nudges the stagnation detector injected.
    pub stagnation_nudges: usize,
    /// Times the repetition limit parked the worker on the orchestrator.
    pub loop_pauses: usize,
    /// Verification-gate runs.
    pub verify_runs: usize,
    /// Verification-gate runs that exited non-zero.
    pub verify_failures: usize,
    /// Files touched by the final diff.
    pub diff_files: usize,
    /// Lines added by the final diff.
    pub diff_insertions: usize,
    /// Lines removed by the final diff.
    pub diff_deletions: usize,
    /// Commands an isolation guard refused to run: a worktree-guardrail or
    /// interceptor block, a sandbox that could not be prepared, or a
    /// completion side-effect audit that found leftovers.
    pub isolation_blocks: usize,
    /// Turns the model answered without a tool call, which the engine had to
    /// answer with the tool-contract reminder instead of a command.
    pub no_command_turns: usize,
    /// Times the no-command guard parked the worker on the orchestrator: the
    /// model stopped calling tools, or its reasoning came back degenerate for
    /// turn after turn.
    pub no_command_pauses: usize,
}

impl WorkerMetrics {
    /// Whether any counter was ever moved off zero.
    ///
    /// An all-zero struct is what a registry row written before these counters
    /// existed carries, so a view must render nothing for it rather than a
    /// reassuring line of zeros.
    pub fn is_recorded(&self) -> bool {
        *self != Self::default()
    }

    /// Compact `3 repeats, 1 nudge` cell for the monitor's stacked row.
    pub fn repeat_nudge_cell(&self) -> String {
        format!(
            "{} repeat{}, {} nudge{}",
            self.repeat_blocks,
            if self.repeat_blocks == 1 { "" } else { "s" },
            self.stagnation_nudges,
            if self.stagnation_nudges == 1 { "" } else { "s" },
        )
    }
}

/// The structured report a worker writes in its completion turn.
///
/// A completion used to be read off the first line of the worker's last chat
/// message, which is whatever the model happened to say last ("Now I'll make
/// the edits."), so every consumer had to shell out to `git diff` to learn
/// what the run actually did. The four lines below are the contract instead:
/// the worker states them before the completion sentinel and the harness
/// carries them from the completion turn to the registry row, so a `status`,
/// `collect` or `watch` event answers without a second call.
///
/// Every field is optional and bounded: a worker that omits a line still
/// completes (the harness falls back to today's summary), and a report is
/// never allowed to grow into a second document.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct WorkerReport {
    /// One line: what changed.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub done: String,
    /// Paths changed, comma-separated.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub files: String,
    /// The commands run and their result, one line.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub tests: String,
    /// Security, contract or behaviour risks, or `none`.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub risks: String,
}

impl WorkerReport {
    /// Whether the report says anything at all.
    pub fn is_empty(&self) -> bool {
        self.done.is_empty()
            && self.files.is_empty()
            && self.tests.is_empty()
            && self.risks.is_empty()
    }
}

/// A consolidator's per-worker verdicts, kept beside the round's headline.
///
/// The consolidator procedure asks for one line per worker it integrated
/// (`REPORT <id> approved|returned|fixed: <one line>`) plus a `RISK:` line for
/// anything that touches the sandbox, governance or identity. Without them the
/// orchestrator only sees the round's one-line `done:`, and the per-worker
/// detail is in the consolidator's history JSONL -- which is why it is recorded
/// here, on the completed state and on the registry row, instead.
///
/// Bounded by construction: at most [`VERDICT_BYTES`] bytes survive, and the
/// lines are kept in the order the consolidator wrote them, so a reader sees
/// the same verdicts in the same order they were made.
///
/// It serializes as the flat array of lines a consumer renders (the per-worker
/// verdicts, then the risks), which is what the notification text shows and
/// what the compact views carry; the two groups stay available on the value
/// for code that wants them separately.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Deserialize)]
#[serde(from = "WorkerVerdictsWire")]
pub struct WorkerVerdicts {
    /// One `REPORT <id> <verdict>: <line>` per worker, in the written order.
    pub workers: Vec<String>,
    /// The consolidator's `RISK:` lines, verbatim, in the written order.
    pub risks: Vec<String>,
}

/// The on-the-wire shape of a [`WorkerVerdicts`]: one flat array of the lines
/// a notification shows, which is also the wire shape this type parses back.
#[derive(serde::Deserialize)]
#[serde(untagged)]
enum WorkerVerdictsWire {
    /// The compact views carry the lines as an array.
    Lines(Vec<String>),
    /// The two-group form, for a payload that keeps them apart.
    Grouped {
        #[serde(default)]
        workers: Vec<String>,
        #[serde(default)]
        risks: Vec<String>,
    },
}

impl From<WorkerVerdictsWire> for WorkerVerdicts {
    fn from(wire: WorkerVerdictsWire) -> Self {
        // The budget is enforced on the way *in* as well as on the way out: a
        // registry row is plain JSON in a directory another local user can
        // write, and the type promises at most `VERDICT_BYTES` survive. Without
        // this an oversized array off disk is read whole and rendered whole
        // into the completion event, the review payload and every watch view.
        match wire {
            WorkerVerdictsWire::Lines(lines) => {
                let (workers, risks) = lines
                    .into_iter()
                    .partition(|line| !line.starts_with("RISK:"));
                bounded(workers, risks)
            }
            WorkerVerdictsWire::Grouped { workers, risks } => bounded(workers, risks),
        }
    }
}

/// Charge `workers` then `risks` against [`VERDICT_BYTES`], dropping whatever
/// does not fit and naming the count, exactly as [`parse_verdict_lines`] does
/// when the value is built. One rule, so a stored payload and a read-back one
/// are bounded the same way.
fn bounded(workers: Vec<String>, risks: Vec<String>) -> WorkerVerdicts {
    let mut out = WorkerVerdicts::default();
    // Charged from the first line, so a truncated payload can still afford the
    // notice that says it is truncated.
    let mut spent = 0usize;
    let mut dropped = 0usize;
    let mut push = |line: String, into: fn(&mut WorkerVerdicts) -> &mut Vec<String>| {
        if spent + line.len() + 1 > VERDICT_BYTES - TRUNCATION_NOTICE_BYTES {
            dropped += 1;
            return;
        }
        spent += line.len() + 1;
        into(&mut out).push(line);
    };
    for line in workers {
        push(line, |v| &mut v.workers);
    }
    for line in risks {
        push(line, |v| &mut v.risks);
    }
    if dropped > 0 {
        out.risks
            .push(format!("... [{dropped} more verdict lines dropped]"));
    }
    out
}

impl serde::Serialize for WorkerVerdicts {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.lines().serialize(serializer)
    }
}

/// Byte budget for the whole of a [`WorkerVerdicts`] payload.
///
/// A round reports one line per worker it integrated, so a big round is the
/// case that matters: the lines are kept until the budget is spent and the
/// overflow is dropped with a marker line, never an unbounded registry row.
pub const VERDICT_BYTES: usize = 4096;

/// Budget held back from [`VERDICT_BYTES`] for the truncation notice, so a
/// round that hit the ceiling can always say so.
const TRUNCATION_NOTICE_BYTES: usize = 64;

impl WorkerVerdicts {
    /// Whether the consolidator recorded anything at all.
    pub fn is_empty(&self) -> bool {
        self.workers.is_empty() && self.risks.is_empty()
    }

    /// Append one harness-derived risk line, charged against the same
    /// [`VERDICT_BYTES`] budget the parsed lines are.
    ///
    /// The consolidator completion path adds its own lines (a worker whose
    /// commits never reached the round) after the model's, so the bound has to
    /// be enforced here too: the type promises at most [`VERDICT_BYTES`] survive,
    /// whichever path built the value.
    pub fn push_risk_bounded(&mut self, line: String) {
        let spent: usize = self
            .workers
            .iter()
            .chain(self.risks.iter())
            .map(|line| line.len() + 1)
            .sum();
        // Charged the way `parse_verdict_lines` charges a line: the newline
        // that joins it to the next one, and the notice held back for the
        // truncation marker.
        if spent + line.len() + 1 > VERDICT_BYTES - TRUNCATION_NOTICE_BYTES {
            return;
        }
        self.risks.push(line);
    }

    /// The lines a notification shows, one per line: the per-worker verdicts
    /// then the risks.
    pub fn lines(&self) -> Vec<&str> {
        self.workers
            .iter()
            .chain(self.risks.iter())
            .map(String::as_str)
            .collect()
    }
}

/// Collect a consolidator's per-worker `REPORT` lines and `RISK:` lines from
/// its closing message, bounded to [`VERDICT_BYTES`].
///
/// Lines are kept verbatim (markup stripped by the caller) in the order they
/// were written, because the order is the round's history. Once the budget is
/// spent the remaining lines are dropped and a trailing marker names how many,
/// so a truncated round reads as truncated instead of quietly short.
pub fn parse_verdict_lines(message: &str) -> WorkerVerdicts {
    let mut verdicts = WorkerVerdicts::default();
    // The notice is charged against the budget from the first line, so a
    // truncated round can always afford to say that it is truncated.
    let notice_room = TRUNCATION_NOTICE_BYTES;
    let mut spent = 0usize;
    let mut dropped = 0usize;
    let mut push = |line: String, into: fn(&mut WorkerVerdicts) -> &mut Vec<String>| {
        if spent + line.len() + 1 > VERDICT_BYTES - notice_room {
            dropped += 1;
            return;
        }
        spent += line.len() + 1;
        into(&mut verdicts).push(line);
    };
    for line in message.lines() {
        // The same peel the authoritative verdict parser applies, so a line the
        // harness absorbed a worker on is the line the round displays. A
        // consolidator that wraps its verdicts in a markdown bullet is read
        // identically by both, and a round never renders as having no verdicts
        // for the workers it did act on.
        let line = crate::pool::runner::strip_markup(line);
        if line.is_empty() {
            continue;
        }
        if is_risk_line(&line) {
            push(risk_line(&line), |v| &mut v.risks);
        } else if is_per_worker_verdict(&line) {
            push(line, |v| &mut v.workers);
        }
    }
    if dropped > 0 {
        // The count is fixed-width so the notice can never itself be the thing
        // that does not fit.
        verdicts
            .risks
            .push(format!("... [{dropped} more verdict lines dropped]"));
    }
    verdicts
}

/// Whether `line` is a consolidator's `RISK:` line.
fn is_risk_line(line: &str) -> bool {
    line.split_once(':')
        .is_some_and(|(head, _)| head.trim().eq_ignore_ascii_case("RISK"))
}

/// The `RISK:` line without its marker, so every stored risk reads the same way.
fn risk_line(line: &str) -> String {
    match line.split_once(':') {
        Some((_, rest)) => format!("RISK: {}", rest.trim()),
        None => line.to_string(),
    }
}

/// Whether `line` is a consolidator's per-worker verdict, `REPORT <id> <v>:`.
///
/// The id is what separates a verdict from the round's own block marker, and
/// the verdict must be one of the three the procedure names, so the block's
/// `REPORT` / `done:` / `risks:` lines are never mistaken for one.
fn is_per_worker_verdict(line: &str) -> bool {
    let Some(rest) = line.strip_prefix("REPORT ") else {
        return false;
    };
    let mut words = rest.split_whitespace();
    let Some(id) = words.next() else {
        return false;
    };
    if id.is_empty()
        || !id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
    {
        return false;
    }
    matches!(
        words.next().map(|verdict| verdict.trim_end_matches(':')),
        Some("approved" | "APPROVED" | "returned" | "RETURNED" | "fixed" | "FIXED")
    )
}

/// One file's share of a diff: the path, the lines added and the lines removed.
///
/// The completion diff is already in memory when a worker finishes, so the
/// per-file split is read out of it once and carried next to the totals: a
/// consumer that wants to know *what* changed should not have to shell out to
/// `git diff --stat` to find out.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct FileStat {
    pub path: String,
    pub insertions: usize,
    pub deletions: usize,
}

impl FileStat {
    /// Lines this file accounts for, the churn the top-N ordering sorts on.
    pub fn churn(&self) -> usize {
        self.insertions + self.deletions
    }
}

/// A path as git spells it in a diff header, without the `a/`/`b/` prefix or a
/// leading `./`, so a caller's `--file src/a.rs` matches what git printed.
pub fn normalize_diff_path(path: &str) -> String {
    let path = path
        .strip_prefix("b/")
        .or_else(|| path.strip_prefix("a/"))
        .unwrap_or(path)
        .trim_matches('"');
    path.strip_prefix("./").unwrap_or(path).to_string()
}

/// How many files a per-file diff line names before it starts counting the rest.
pub const TOP_FILE_LIMIT: usize = 8;

/// `a.rs (+3 -2), b.rs (+1 -1), +2 more`: the biggest-churn files of a diff,
/// then how many were left out.
///
/// Ordered by churn and then by path so the line is deterministic, and bounded
/// by [`TOP_FILE_LIMIT`] so it can never grow with the size of the diff.
pub fn churn_line(stats: &[FileStat]) -> String {
    let mut ordered: Vec<&FileStat> = stats.iter().collect();
    ordered.sort_by(|a, b| b.churn().cmp(&a.churn()).then_with(|| a.path.cmp(&b.path)));
    let shown = ordered.len().min(TOP_FILE_LIMIT);
    let mut line = ordered[..shown]
        .iter()
        .map(|stat| format!("{} (+{} -{})", stat.path, stat.insertions, stat.deletions))
        .collect::<Vec<_>>()
        .join(", ");
    let more = ordered.len().saturating_sub(shown);
    if more > 0 {
        line.push_str(&format!(", +{more} more"));
    }
    line
}

/// One section of a unified diff: the path it is about plus its raw body.
struct DiffSection {
    /// Path from the `+++` header, or from the `diff --git` line.
    path: String,
    /// Path from the `---` header, which is the only one a deleted file has.
    minus: String,
    body: String,
}

impl DiffSection {
    /// The path the section is about: the "after" side when the file still
    /// exists, the "before" side when it does not.
    fn path(&self) -> &str {
        if self.path.is_empty() {
            &self.minus
        } else {
            &self.path
        }
    }
}

/// Split a unified diff into one `(path, section)` pair per file.
///
/// The path comes from the `+++`/`---` headers, which precede every hunk, so a
/// removed line that happens to start with `--` can never be mistaken for one.
/// A section with neither header — a binary file, a mode-only change — falls
/// back to the paths its `diff --git` line names.
pub fn diff_sections_of(diff: &str) -> Vec<(String, String)> {
    let mut sections: Vec<(String, String)> = Vec::new();
    let mut current: Option<DiffSection> = None;
    for line in diff.lines() {
        if let Some(header) = line.strip_prefix("diff --git ") {
            if let Some(section) = current.take() {
                sections.push((section.path().to_string(), section.body));
            }
            current = Some(DiffSection {
                path: diff_git_path(header).unwrap_or_default(),
                minus: String::new(),
                body: format!("{line}\n"),
            });
            continue;
        }
        let Some(section) = current.as_mut() else {
            continue;
        };
        section.body.push_str(line);
        section.body.push('\n');
        if let Some(found) = line.strip_prefix("--- ").and_then(diff_header_path) {
            section.minus = found;
        } else if let Some(found) = line.strip_prefix("+++ ").and_then(diff_header_path) {
            section.path = found;
        }
    }
    if let Some(section) = current.take() {
        sections.push((section.path().to_string(), section.body));
    }
    sections
        .into_iter()
        .filter(|(path, _)| !path.is_empty())
        .collect()
}

/// The path a `diff --git a/<path> b/<path>` line names on its "before" side.
fn diff_git_path(header: &str) -> Option<String> {
    let split = header.rfind(" b/")?;
    let path = &header[..split];
    Some(normalize_diff_path(path.strip_prefix("a/").unwrap_or(path)))
}

/// The path a `--- `/`+++ ` header names, or `None` for `/dev/null`.
fn diff_header_path(header: &str) -> Option<String> {
    let path = header.trim();
    (path != "/dev/null").then(|| normalize_diff_path(path))
}

/// Per-file `(path, insertions, deletions)` of a unified diff.
///
/// Only hunk lines are counted, and a hunk starts at its `@@` header, so the
/// `---`/`+++` headers and an added line that itself starts with `+` are never
/// counted as content.
pub fn file_stats_of_diff(diff: &str) -> Vec<FileStat> {
    diff_sections_of(diff)
        .into_iter()
        .map(|(path, section)| {
            let mut insertions = 0;
            let mut deletions = 0;
            let mut in_hunks = false;
            for line in section.lines() {
                if line.starts_with("@@") {
                    in_hunks = true;
                } else if in_hunks && line.starts_with('+') {
                    insertions += 1;
                } else if in_hunks && line.starts_with('-') {
                    deletions += 1;
                }
            }
            FileStat {
                path,
                insertions,
                deletions,
            }
        })
        .collect()
}

/// Whether a requested path names the file a diff section is about.
///
/// Exact after normalisation, or a whole-component suffix of it, so `--file
/// a.rs` still finds `src/a.rs`.
pub fn same_diff_path(requested: &str, actual: &str) -> bool {
    let requested = normalize_diff_path(requested);
    let actual = normalize_diff_path(actual);
    actual == requested
        || (!requested.is_empty()
            && actual.len() > requested.len()
            && actual.ends_with(&requested)
            && actual[..actual.len() - requested.len()].ends_with('/'))
}

/// The typed reason an [`WorkerState::Exhausted`] records. A worker that ran
/// out of turns stopped without completing; its caller must continue it with a
/// fresh budget rather than treat the branch as done.
pub const TURN_BUDGET_EXHAUSTED: &str = "turn_budget_exhausted";

/// Most artifact paths a compact view embeds before the tail is reported as a
/// count. A worker that synced hundreds of files must not make every `status`
/// or `list` answer large.
pub const ARTIFACT_PREVIEW: usize = 5;

/// Split a worker's artifacts into the paths a compact view carries and the
/// total count, so a long tail is reported as a number instead of being
/// dropped silently.
pub fn compact_artifacts(artifacts: &[String]) -> (Vec<String>, usize) {
    (
        artifacts.iter().take(ARTIFACT_PREVIEW).cloned().collect(),
        artifacts.len(),
    )
}

#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "state", content = "details")]
pub enum WorkerState {
    Running {
        step: usize,
        last_command: String,
        started_at: u64,
    },
    Paused {
        question: String,
        step: usize,
        paused_at: u64,
    },
    Completed {
        turns: usize,
        diff: String,
        summary: String,
        completed_at: u64,
        #[serde(default)]
        artifacts: Vec<String>,
        #[serde(default)]
        branch: Option<String>,
        #[serde(default)]
        verified: Option<bool>,
        #[serde(default)]
        metrics: WorkerMetrics,
        /// Times this worker was revised after finishing. A fresh dispatch
        /// completes at zero; every revision bumps it, so the orchestrator can
        /// tell the first answer from a corrected one.
        #[serde(default)]
        revision: usize,
        /// The structured report of the completion turn, `None` when the
        /// worker never supplied one (the summary stays the fallback).
        #[serde(default)]
        report: Option<WorkerReport>,
        /// The per-worker `REPORT` lines and `RISK:` lines of a consolidator's
        /// closing message. `None` for an ordinary worker, which has one
        /// verdict to give -- its own -- already in `report`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        verdicts: Option<WorkerVerdicts>,
    },
    Failed {
        error: String,
        step: usize,
        failed_at: u64,
        /// What the run had measured before it died; a worker killed before
        /// its first turn reports the all-zero default.
        #[serde(default)]
        metrics: WorkerMetrics,
        /// Same counter as on [`WorkerState::Completed`]: a failed worker that
        /// was itself a revision reports which attempt died.
        #[serde(default)]
        revision: usize,
    },
    /// The worker spent its whole turn budget without emitting the completion
    /// sentinel. Its work is checkpointed on its branch exactly like a
    /// completion, but it never verified, and it must be read as stopped, not
    /// done: the caller continues it with a fresh budget.
    Exhausted {
        turns: usize,
        diff: String,
        summary: String,
        stopped_at: u64,
        #[serde(default)]
        artifacts: Vec<String>,
        #[serde(default)]
        branch: Option<String>,
        #[serde(default)]
        metrics: WorkerMetrics,
        /// Same counter as on [`WorkerState::Completed`]: an exhausted worker
        /// that was itself a revision reports which attempt ran out of turns.
        #[serde(default)]
        revision: usize,
        /// The structured report of the last turn, when the worker supplied
        /// one; the summary stays the fallback.
        #[serde(default)]
        report: Option<WorkerReport>,
        /// Same counter as on [`WorkerState::Completed`]: a consolidator that
        /// ran out of turns mid-round still reported the workers it had
        /// already reached a verdict on.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        verdicts: Option<WorkerVerdicts>,
    },
}

impl WorkerState {
    /// Turn this state corresponds to, for every variant.
    pub fn step(&self) -> usize {
        match self {
            WorkerState::Running { step, .. } | WorkerState::Paused { step, .. } => *step,
            WorkerState::Completed { turns, .. } => *turns,
            WorkerState::Failed { step, .. } => *step,
            WorkerState::Exhausted { turns, .. } => *turns,
        }
    }

    /// The variant name, matching the serde tag so a compact projection and
    /// the full serialization agree.
    pub fn name(&self) -> &'static str {
        match self {
            WorkerState::Running { .. } => "Running",
            WorkerState::Paused { .. } => "Paused",
            WorkerState::Completed { .. } => "Completed",
            WorkerState::Failed { .. } => "Failed",
            WorkerState::Exhausted { .. } => "Exhausted",
        }
    }

    pub fn to_summary(&self) -> serde_json::Value {
        match self {
            WorkerState::Running {
                step,
                last_command,
                started_at,
            } => serde_json::json!({
                "status": "Running",
                "step": step,
                "last_command": last_command,
                "started_at": started_at,
            }),
            WorkerState::Paused {
                question,
                step,
                paused_at,
            } => serde_json::json!({
                "status": "Paused",
                "step": step,
                "question": question,
                "paused_at": paused_at,
            }),
            WorkerState::Completed {
                turns,
                summary,
                completed_at,
                artifacts,
                branch,
                verified,
                metrics,
                revision,
                report,
                verdicts,
                diff,
            } => {
                let (artifacts, artifacts_total) = compact_artifacts(artifacts);
                serde_json::json!({
                    "status": "Completed",
                    "turns": turns,
                    "summary": summary,
                    "completed_at": completed_at,
                    "artifacts": artifacts,
                    "artifacts_total": artifacts_total,
                    "branch": branch,
                    "verified": verified,
                    "metrics": metrics,
                    "revision": revision,
                    "report": report,
                    // The round's per-worker verdicts ride in the compact view
                    // too: this projection is what `list` and a cold `status`
                    // read, and they are the round's detail.
                    "verdicts": verdicts,
                    "per_file": file_stats_of_diff(diff),
                })
            }
            WorkerState::Failed {
                error,
                step,
                failed_at,
                metrics,
                revision,
            } => serde_json::json!({
                "status": "Failed",
                "step": step,
                "error": error,
                "failed_at": failed_at,
                "metrics": metrics,
                "revision": revision,
            }),
            WorkerState::Exhausted {
                turns,
                summary,
                stopped_at,
                artifacts,
                branch,
                metrics,
                revision,
                report,
                verdicts,
                diff,
            } => {
                let (artifacts, artifacts_total) = compact_artifacts(artifacts);
                serde_json::json!({
                    "status": "Exhausted",
                    "turns": turns,
                    "summary": summary,
                    "stopped_at": stopped_at,
                    "artifacts": artifacts,
                    "artifacts_total": artifacts_total,
                    "branch": branch,
                    "metrics": metrics,
                    "revision": revision,
                    "report": report,
                    "verdicts": verdicts,
                    "reason": TURN_BUDGET_EXHAUSTED,
                    "per_file": file_stats_of_diff(diff),
                })
            }
        }
    }
}

pub struct WorkerRecord {
    pub id: String,
    pub task: String,
    pub model: String,
    /// Agent identity that dispatched this worker, and the only identity
    /// allowed to steer, kill, collect or wait on it (H-3).
    pub owner: String,
    pub state: WorkerState,
    /// Cache of the phase loop's [`WorkerMetrics`], refreshed on every state
    /// write, so a kill, a crash or a server shutdown can report what the run
    /// had measured when the loop's own copy went away with the task.
    pub metrics: WorkerMetrics,
    /// Bounded sliding window of step logs (audit 07, R1).
    pub logs: LogBuffer,
    pub pending_steer: Vec<String>,
    pub resume_tx: Option<tokio::sync::mpsc::Sender<String>>,
    pub handle: Option<JoinHandle<()>>,
    /// Times this worker was steered after reaching a terminal state. A fresh
    /// dispatch starts at zero; every revision bumps it, so the orchestrator
    /// can tell the first answer from a corrected one in the payloads and the
    /// channel events.
    pub revision: usize,
}

impl WorkerRecord {
    pub(super) fn fail(&mut self, error: impl Into<String>) {
        // A revision that dies keeps its number: the failure payload says
        // which attempt died, not just that something did.
        let revision = match &self.state {
            WorkerState::Completed { revision, .. }
            | WorkerState::Failed { revision, .. }
            | WorkerState::Exhausted { revision, .. } => *revision,
            WorkerState::Running { .. } | WorkerState::Paused { .. } => self.revision,
        };
        self.state = WorkerState::Failed {
            error: error.into(),
            step: self.state.step(),
            failed_at: unix_timestamp(),
            metrics: self.metrics,
            revision,
        };
    }

    /// Unix timestamp when this record became terminal, if it is terminal.
    pub fn terminal_at(&self) -> Option<u64> {
        match &self.state {
            WorkerState::Completed { completed_at, .. } => Some(*completed_at),
            WorkerState::Failed { failed_at, .. } => Some(*failed_at),
            WorkerState::Exhausted { stopped_at, .. } => Some(*stopped_at),
            WorkerState::Running { .. } | WorkerState::Paused { .. } => None,
        }
    }

    /// Step counters for the observability surface (audit 07, R7).
    pub fn log_stats(&self) -> LogStats {
        LogStats {
            total_steps: self.state.step().max(self.logs.total()),
            logs_retained: self.logs.retained(),
            logs_dropped: self.logs.dropped(),
        }
    }
}

/// Coarse lifecycle stage of a worker, used by progress polls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerPhase {
    Running,
    Paused,
    Completed,
    Failed,
    Exhausted,
}

/// Lightweight, allocation-cheap snapshot of a worker's progress.
///
/// Excludes the terminal payload (`diff`, `summary`, `artifacts`) so the
/// 500 ms polling loops neither clone nor serialize multi-megabyte strings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerProgress {
    pub phase: WorkerPhase,
    /// Turn currently running (or the turn the worker paused at); for terminal
    /// phases this is the number of turns performed.
    pub step: usize,
    /// Last bash command summary while running.
    pub last_command: Option<String>,
    /// Escalated question while paused.
    pub question: Option<String>,
    /// Queued heavy commands ahead of this worker while it waits for a build
    /// slot; `None` when it is not waiting. Time spent here is not a stall.
    pub waiting_for_slot: Option<usize>,
    /// Unix time this worker's current bash command started, while one is
    /// executing; `None` when no command is in flight. A step whose command is
    /// still running is not worker inactivity.
    pub command_started_at: Option<u64>,
    /// Background jobs this worker still has running: commands that outlived
    /// their budget and were continued instead of killed.
    pub jobs: Vec<JobStatus>,
}

/// Who a worker belongs to, as the pool and the registry record it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkerOwner {
    /// The agent identity that dispatched it.
    Agent(String),
    /// A row written before ownership was tracked: readable by anyone, but
    /// mutable by nobody short of an admin connection.
    Unattributed,
}

/// Result of a one-shot worker collection, detached from the live pool.
#[derive(Debug, Clone, serde::Serialize)]
pub struct CollectedWorker {
    pub id: String,
    pub task: String,
    pub model: String,
    /// The agent that owned the worker, carried through so a collected
    /// payload names whose worker it was.
    pub owner: String,
    pub state: WorkerState,
    /// The retained window, moved out of the pool (not a copy).
    pub logs: Vec<AgentStepLog>,
    /// Retained entries that were not part of `logs` because of the emission
    /// budget.
    pub logs_omitted: usize,
    /// Retained entries already evicted by the retention window.
    pub logs_dropped: usize,
    /// Human-readable explanation when the history is degraded.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logs_truncation_notice: Option<String>,
}

/// Default age (seconds) after which a `Completed`/`Failed` worker record is
/// evicted from the pool.
///
/// This bounds *memory* only. The record's registry row and its saved
/// conversation outlive the eviction, because a finished worker stays
/// steerable for as long as its branch does (see
/// [`DEFAULT_TERMINAL_RETENTION_SECS`]).
pub const DEFAULT_TERMINAL_TTL_SECS: u64 = 300;

/// Default age (seconds) after which a terminal worker's registry row and its
/// saved conversation are retired, even though its branch still exists.
///
/// A week: long enough that an orchestrator reviewing, gating or merging hours
/// or days later still finds the worker continuable, short enough that a
/// scratch root cannot grow without bound. Overridable through
/// `WORKER_RETENTION_SECS`.
pub const DEFAULT_TERMINAL_RETENTION_SECS: u64 = 7 * 24 * 60 * 60;

/// The retention an operator configured, or [`DEFAULT_TERMINAL_RETENTION_SECS`].
///
/// Read from the environment on each call rather than cached on the pool: the
/// sweeps that apply it are free functions with no pool at hand, and a
/// non-positive or unparseable value falls back to the default rather than to
/// "retire everything now".
pub fn terminal_retention_secs() -> u64 {
    std::env::var("WORKER_RETENTION_SECS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(DEFAULT_TERMINAL_RETENTION_SECS)
}

/// Whether a row last written at `updated_at` is past `retention_secs`.
///
/// A row that never recorded an age (`updated_at` of zero, written before the
/// field existed) is *not* expired: an unknown age must not be read as an
/// ancient one. Clock skew is absorbed by `saturating_sub`, exactly as
/// `expired_terminal_ids` absorbs it.
pub fn retention_expired(updated_at: u64, retention_secs: u64, now: u64) -> bool {
    updated_at != 0 && now.saturating_sub(updated_at) >= retention_secs
}

/// Default grace period (seconds) a worker's row and saved conversation are
/// kept after its branch disappears.
///
/// A merged worker branch is pruned on the next dispatch, but the orchestrator
/// may still revert that merge (a failing batch gate) and continue the worker,
/// so its row and conversation outlive the branch for this long before the
/// orphan sweep retires them. Overridable through `WORKER_RETIRED_GRACE_SECS`.
pub const DEFAULT_WORKER_RETIRED_GRACE_SECS: u64 = 24 * 60 * 60;

/// The retired grace an operator configured, or
/// [`DEFAULT_WORKER_RETIRED_GRACE_SECS`].
///
/// Read from the environment on each call, exactly like
/// [`terminal_retention_secs`]: a non-positive or unparseable value falls back
/// to the default rather than to "retire everything now".
pub fn worker_retired_grace_secs() -> u64 {
    std::env::var("WORKER_RETIRED_GRACE_SECS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(DEFAULT_WORKER_RETIRED_GRACE_SECS)
}

/// Whether a row last written at `updated_at` is still inside the grace period
/// during which it survives its branch's disappearance.
///
/// A row that never recorded an age (`updated_at` of zero, written before the
/// field existed) is *kept*: an unknown age must not be read as an ancient one,
/// exactly as [`retention_expired`]. A zero `grace_secs` retires immediately.
pub fn within_retired_grace(updated_at: u64, grace_secs: u64, now: u64) -> bool {
    updated_at == 0 || now.saturating_sub(updated_at) < grace_secs
}

/// Ids of `Completed`/`Failed` records that reached the terminal TTL (audit 07, R3).
///
/// * A `Running`/`Paused` record is *never* expired, whatever its age.
/// * Clock skew is absorbed by `saturating_sub`: a future timestamp (NTP jump,
///   forged registry row) yields 0 and the record is kept.
/// * A fresh terminal record is kept, which keeps `collect` and `wait: true`
///   working after a worker finishes.
pub(super) fn expired_terminal_ids(
    workers: &HashMap<String, WorkerRecord>,
    ttl_secs: u64,
) -> Vec<String> {
    let now = unix_timestamp();
    let mut ids: Vec<String> = workers
        .iter()
        .filter_map(|(id, record)| {
            record
                .terminal_at()
                .filter(|at| now.saturating_sub(*at) >= ttl_secs)
                .map(|_| id.clone())
        })
        .collect();
    ids.sort();
    ids
}

#[cfg(test)]
mod tests {
    use super::super::buffer::LogBuffer;
    use super::super::unix_timestamp;
    use super::{
        DEFAULT_TERMINAL_RETENTION_SECS, DEFAULT_TERMINAL_TTL_SECS, FileStat, WorkerMetrics,
        WorkerRecord, WorkerState, churn_line, expired_terminal_ids, file_stats_of_diff,
        retention_expired,
    };
    use std::collections::HashMap;

    // ----------
    // WorkerState::step
    // ----------

    #[test]
    fn test_worker_state_step_covers_every_variant() {
        assert_eq!(
            WorkerState::Running {
                step: 4,
                last_command: "ls".into(),
                started_at: 0
            }
            .step(),
            4
        );
        assert_eq!(
            WorkerState::Paused {
                question: "?".into(),
                step: 9,
                paused_at: 0
            }
            .step(),
            9
        );
        assert_eq!(
            WorkerState::Completed {
                turns: 12,
                diff: String::new(),
                summary: String::new(),
                completed_at: 0,
                artifacts: Vec::new(),
                branch: None,
                verified: None,
                metrics: WorkerMetrics::default(),
                revision: 0,
                report: None,
                verdicts: None,
            }
            .step(),
            12
        );
        assert_eq!(
            WorkerState::Failed {
                error: "boom".into(),
                step: 3,
                failed_at: 0,
                metrics: WorkerMetrics::default(),
                revision: 0,
            }
            .step(),
            3
        );
        assert_eq!(
            WorkerState::Exhausted {
                turns: 7,
                diff: String::new(),
                summary: String::new(),
                stopped_at: 0,
                artifacts: Vec::new(),
                branch: None,
                metrics: WorkerMetrics::default(),
                revision: 0,
                report: None,
                verdicts: None,
            }
            .step(),
            7
        );
    }

    #[test]
    fn churn_line_names_the_top_files_then_counts_the_rest() {
        let mut stats = Vec::new();
        for index in 0..11 {
            stats.push(FileStat {
                path: format!("src/f{index}.rs"),
                insertions: index,
                deletions: 0,
            });
        }
        let line = churn_line(&stats);
        assert_eq!(
            line,
            "src/f10.rs (+10 -0), src/f9.rs (+9 -0), src/f8.rs (+8 -0), \
             src/f7.rs (+7 -0), src/f6.rs (+6 -0), src/f5.rs (+5 -0), \
             src/f4.rs (+4 -0), src/f3.rs (+3 -0), +3 more"
        );
        assert!(churn_line(&[]).is_empty(), "no diff, no line");
    }

    #[test]
    fn a_unified_diff_is_read_into_per_file_stats() {
        let diff = concat!(
            "diff --git a/one.rs b/one.rs\n",
            "index 111..222 100644\n",
            "--- a/one.rs\n",
            "+++ b/one.rs\n",
            "@@ -1,3 +1,4 @@\n",
            " context\n",
            "-removed\n",
            "++added line that starts with a plus\n",
            "--removed line that starts with two dashes\n",
            "diff --git a/two.rs b/two.rs\n",
            "new file mode 100644\n",
            "index 000..333\n",
            "--- /dev/null\n",
            "+++ b/two.rs\n",
            "@@ -0,0 +1,2 @@\n",
            "+first\n",
            "+second\n",
        );
        assert_eq!(
            file_stats_of_diff(diff),
            vec![
                FileStat {
                    path: "one.rs".to_string(),
                    insertions: 1,
                    deletions: 2,
                },
                FileStat {
                    path: "two.rs".to_string(),
                    insertions: 2,
                    deletions: 0,
                },
            ]
        );
    }

    #[test]
    fn test_terminal_at_only_reports_terminal_states() {
        let running = WorkerState::Running {
            step: 0,
            last_command: String::new(),
            started_at: 0,
        };
        assert!(matches!(running, WorkerState::Running { .. }));
        let completed = WorkerState::Completed {
            turns: 1,
            diff: String::new(),
            summary: String::new(),
            completed_at: 1_700_000_000,
            artifacts: Vec::new(),
            branch: None,
            verified: None,
            metrics: WorkerMetrics::default(),
            revision: 0,
            report: None,
            verdicts: None,
        };
        let failed = WorkerState::Failed {
            error: "e".into(),
            step: 1,
            failed_at: 1_700_000_001,
            metrics: WorkerMetrics::default(),
            revision: 0,
        };
        let exhausted = exhausted_at(1_700_000_002);
        assert!(!matches!(running, WorkerState::Completed { .. }));
        assert!(matches!(completed, WorkerState::Completed { .. }));
        assert!(matches!(failed, WorkerState::Failed { .. }));
        assert!(matches!(exhausted, WorkerState::Exhausted { .. }));
        assert_eq!(record_with(exhausted).terminal_at(), Some(1_700_000_002));
    }

    fn record_with(state: WorkerState) -> WorkerRecord {
        WorkerRecord {
            id: "w".into(),
            task: "t".into(),
            model: "m".into(),
            owner: "test-owner".into(),
            state,
            metrics: WorkerMetrics::default(),
            logs: LogBuffer::new(),
            pending_steer: Vec::new(),
            resume_tx: None,
            handle: None,
            revision: 0,
        }
    }

    fn completed_at(when: u64) -> WorkerState {
        WorkerState::Completed {
            turns: 1,
            diff: String::new(),
            summary: String::new(),
            completed_at: when,
            artifacts: Vec::new(),
            branch: None,
            verified: None,
            metrics: WorkerMetrics::default(),
            revision: 0,
            report: None,
            verdicts: None,
        }
    }

    fn failed_at(when: u64) -> WorkerState {
        WorkerState::Failed {
            error: "boom".into(),
            step: 1,
            failed_at: when,
            metrics: WorkerMetrics::default(),
            revision: 0,
        }
    }

    fn exhausted_at(when: u64) -> WorkerState {
        WorkerState::Exhausted {
            turns: 1,
            diff: String::new(),
            summary: String::new(),
            stopped_at: when,
            artifacts: Vec::new(),
            branch: None,
            metrics: WorkerMetrics::default(),
            revision: 0,
            report: None,
            verdicts: None,
        }
    }

    // ----------
    // Terminal-record TTL (audit 07, R3)
    // ----------

    #[test]
    fn test_terminal_records_expire_after_the_ttl() {
        let now = unix_timestamp();
        let mut workers = HashMap::new();
        workers.insert("old-done".to_string(), record_with(completed_at(now - 400)));
        workers.insert("old-failed".to_string(), record_with(failed_at(now - 400)));
        workers.insert(
            "old-exhausted".to_string(),
            record_with(exhausted_at(now - 400)),
        );
        workers.insert("fresh-done".to_string(), record_with(completed_at(now)));
        workers.insert(
            "running".to_string(),
            record_with(WorkerState::Running {
                step: 0,
                last_command: String::new(),
                started_at: now - 100_000,
            }),
        );
        workers.insert(
            "paused".to_string(),
            record_with(WorkerState::Paused {
                question: "?".into(),
                step: 1,
                paused_at: now - 100_000,
            }),
        );

        let expired = expired_terminal_ids(&workers, DEFAULT_TERMINAL_TTL_SECS);
        assert_eq!(
            expired,
            vec![
                "old-done".to_string(),
                "old-exhausted".to_string(),
                "old-failed".to_string()
            ],
            "only aged terminal records may be evicted"
        );
    }

    #[test]
    fn test_retention_expires_only_a_row_that_recorded_its_age() {
        let now = unix_timestamp();
        // Aged past the retention: the row goes even though its branch lives.
        assert!(retention_expired(
            now - DEFAULT_TERMINAL_RETENTION_SECS - 1,
            DEFAULT_TERMINAL_RETENTION_SECS,
            now
        ));
        // Inside the retention: the row stays, so an orchestrator hours or
        // days later still finds the worker continuable.
        assert!(!retention_expired(
            now - DEFAULT_TERMINAL_RETENTION_SECS + 60,
            DEFAULT_TERMINAL_RETENTION_SECS,
            now
        ));
        // A row written before the field existed carries no age at all, and an
        // unknown age must not be read as an ancient one.
        assert!(!retention_expired(0, DEFAULT_TERMINAL_RETENTION_SECS, now));
        // Clock skew is absorbed exactly as the TTL absorbs it.
        assert!(!retention_expired(
            now + 10_000,
            DEFAULT_TERMINAL_RETENTION_SECS,
            now
        ));
    }

    #[test]
    fn test_terminal_ttl_absorbs_clock_skew() {
        // A timestamp in the future (NTP jump, forged registry row) must not
        // evict the record: saturating_sub yields 0, which is below any TTL.
        let now = unix_timestamp();
        let mut workers = HashMap::new();
        workers.insert(
            "skewed".to_string(),
            record_with(completed_at(now + 10_000)),
        );
        assert!(
            expired_terminal_ids(&workers, DEFAULT_TERMINAL_TTL_SECS).is_empty(),
            "a future timestamp must never evict a record"
        );
    }

    #[test]
    fn test_terminal_ttl_of_zero_evicts_immediately() {
        let now = unix_timestamp();
        let mut workers = HashMap::new();
        workers.insert("done".to_string(), record_with(completed_at(now)));
        assert_eq!(expired_terminal_ids(&workers, 0).len(), 1);
    }

    #[test]
    fn test_terminal_ttl_keeps_everything_before_the_boundary() {
        let now = unix_timestamp();
        let mut workers = HashMap::new();
        workers.insert("edge".to_string(), record_with(completed_at(now)));
        assert!(
            expired_terminal_ids(&workers, 1).is_empty(),
            "a record younger than the TTL must survive so collect() still works"
        );
    }
}
