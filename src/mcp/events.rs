//! Worker events pushed into the orchestrator session as `claude/channel`
//! notifications.
//!
//! Claude Code's research-preview `claude/channel` extension lets an MCP server
//! push text into the host session at any time: the server declares
//! `capabilities.experimental["claude/channel"]` plus an `instructions` string
//! in its initialize result (see `super::protocol::INITIALIZE_RESULT`), then
//! sends `notifications/claude/channel` frames, which the client renders as
//! `<channel source="mini-swe" …>text</channel>`. A session that did not opt
//! the server in (`claude --dangerously-load-development-channels
//! server:mini-swe`) drops such a notification silently, so emitting one is
//! always safe.
//!
//! [`spawn_event_stream`] is the producer: every [`POLL_INTERVAL`] it diffs the
//! worker state — this process's pool plus the shared on-disk registry, so a
//! worker owned by another `mini-swe-mcp` process (a CLI dispatch) is reported
//! too — and emits one notification per transition into a state the
//! orchestrator has to act on: paused (`needs_input`), `completed` and
//! `failed`. A worker that was already terminal when the task started is
//! seeded into the first snapshot instead of being diffed against an empty one,
//! so starting a server next to a hundred finished workers stays silent; one
//! already paused on a question is still announced, since it needs an answer.
//!
//! The notification text is rendered for a model, not for a log: the diff
//! between two snapshots is pure ([`diff_events`]), and only that pure part
//! decides *what* to say; rendering ([`ChannelEvent`], [`channel_frame`]) and
//! the polling loop are kept apart so the decision is testable on its own.

use std::collections::BTreeMap;
use std::time::Duration;

use serde_json::json;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use crate::pool::{
    RegistryStatus, WorkerMetrics, WorkerPhase, WorkerPool, WorkerRegistryEntry, WorkerState,
    clamp_string,
};

/// How often the event task re-reads the worker state.
///
/// Two seconds is coarse enough that a fleet of workers costs a handful of
/// small registry reads per minute, and fine enough that a paused worker is
/// reported long before the orchestrator gives up on it.
const POLL_INTERVAL: Duration = Duration::from_secs(2);

/// JSON-RPC method of a channel notification (research preview).
const CHANNEL_METHOD: &str = "notifications/claude/channel";

/// Byte budget for text copied out of a worker into a notification.
///
/// The notification is read by a model in a session that also holds the
/// dispatch and its result: quoting a whole agent summary would crowd out the
/// one line that matters.
const QUOTE_BYTES: usize = 1024;

/// The lifecycle transition a notification reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventKind {
    /// A worker parked on a question it escalated to the orchestrator.
    NeedsInput,
    /// A worker finished and is waiting to be reviewed.
    Completed,
    /// A worker died and wants an inspection.
    Failed,
}

impl EventKind {
    /// The wire name: the `meta.event` value and the `event="…"` attribute the
    /// client wraps the content in.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NeedsInput => "needs_input",
            Self::Completed => "completed",
            Self::Failed => "failed",
        }
    }
}

/// The part of a terminal worker a notification quotes.
///
/// The registry row is written by the worker itself and carries counters only,
/// so every field here is optional: a cross-process notification describes
/// *that* a worker finished and tells the model which verb to call, while a
/// notification about this process's own worker also carries the payload.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Outcome {
    /// First line of the completion summary.
    pub summary: Option<String>,
    /// Whether the run passed its verify gate.
    pub verified: Option<bool>,
    /// `3 files, +40 -12`, when the run measured a diff.
    pub diff_stat: Option<String>,
    /// Why the run died.
    pub error: Option<String>,
}

/// One worker's state as the event task sees it, reduced to what a notification
/// needs.
///
/// `worker_id` is always filled in by the producer; the remaining fields
/// describe what was observed, which is why the derived `Default` (an empty
/// worker that is not reporting anything) is meaningful on its own.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WorkerView {
    /// The worker's id, which is also the key it is stored under.
    pub worker_id: String,
    /// The transition to report, or `None` while the worker is live.
    pub event: Option<EventKind>,
    /// Owning group, `"default"` when no group was recorded.
    pub group: String,
    /// Model the worker resolved against.
    pub model: String,
    /// Observed status: `running`, `paused`, `reviewing`, `completed`, …
    pub status: String,
    /// The escalated question of a paused worker.
    pub question: Option<String>,
    /// Terminal payload, read only for a worker that is about to be reported.
    pub outcome: Outcome,
}

/// One tick's view of every known worker, keyed by worker id.
///
/// A `BTreeMap` rather than a `HashMap` so both the snapshot and the events
/// derived from it come out in a stable order.
pub type WorkerSnapshot = BTreeMap<String, WorkerView>;

/// A notification to push into the session: the text the model reads plus the
/// facts the client turns into `<channel …>` attributes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChannelEvent {
    /// Worker the event is about.
    pub worker_id: String,
    /// What happened to it.
    pub kind: EventKind,
    /// Owning group, for context the orchestrator did not have to remember.
    pub group: String,
    /// Model the worker ran on.
    pub model: String,
    /// Status observed in the same tick as the event.
    pub status: String,
    /// The rendered message.
    pub content: String,
}

/// The events for the transitions between two snapshots, ordered by worker id.
///
/// A worker is reported when it holds an event the previous snapshot did not —
/// an unknown worker, or a different event than last time. That is what makes
/// one transition produce exactly one notification: the next tick sees the
/// same `(worker_id, event)` pair and stays silent. A worker that disappeared
/// between the two ticks (collected, reaped, registry row pruned) is absent
/// from `current` and is never reported again.
pub fn diff_events(previous: &WorkerSnapshot, current: &WorkerSnapshot) -> Vec<ChannelEvent> {
    current
        .values()
        .filter_map(|view| {
            let kind = view.event?;
            let already = previous.get(&view.worker_id).and_then(|was| was.event) == Some(kind);
            (!already).then(|| ChannelEvent {
                worker_id: view.worker_id.clone(),
                kind,
                group: view.group.clone(),
                model: view.model.clone(),
                status: view.status.clone(),
                content: render(view, kind),
            })
        })
        .collect()
}

/// Render one event as a newline-terminated JSON-RPC frame.
///
/// Every `meta` key is a fixed identifier and every value a string, which is
/// what the channel protocol asks for; `None` means the frame could not be
/// serialized, and the event is dropped rather than sent malformed.
pub fn channel_frame(event: &ChannelEvent) -> Option<String> {
    let frame = json!({
        "jsonrpc": "2.0",
        "method": CHANNEL_METHOD,
        "params": {
            "content": event.content.clone(),
            "meta": {
                "event": event.kind.as_str(),
                "worker_id": event.worker_id.clone(),
                "group": event.group.clone(),
                "model": event.model.clone(),
                "status": event.status.clone(),
            }
        }
    });
    let mut frame = serde_json::to_string(&frame).ok()?;
    frame.push('\n');
    Some(frame)
}

/// The text the model reads: what happened, then the exact verb to answer it
/// with. The verb is repeated in every notification because a `wait: false`
/// dispatch may be the only trace of the worker left in the session.
fn render(view: &WorkerView, kind: EventKind) -> String {
    let header = format!(
        "Worker {} is {} (model {}, group {}).",
        view.worker_id, view.status, view.model, view.group
    );
    let (body, verb) = match kind {
        EventKind::NeedsInput => (
            format!(
                "Question: {}",
                quote(view.question.as_deref().unwrap_or("(none recorded)"))
            ),
            format!(
                "Answer with the worker tool: action \"steer\", worker_id \"{}\", message \"<your answer>\".",
                view.worker_id
            ),
        ),
        EventKind::Completed => {
            let mut body = String::new();
            if let Some(summary) = &view.outcome.summary {
                body.push_str(&format!("Summary: {summary}\n"));
            }
            if let Some(verified) = view.outcome.verified {
                body.push_str(&format!(
                    "Verified: {}\n",
                    if verified { "yes" } else { "no" }
                ));
            }
            if let Some(diff) = &view.outcome.diff_stat {
                body.push_str(&format!("Diff: {diff}"));
            }
            (
                if body.is_empty() {
                    "Completed with no recorded summary.".to_string()
                } else {
                    body.trim_end().to_string()
                },
                format!(
                    "Review it with the worker tool: action \"collect\", worker_id \"{}\".",
                    view.worker_id
                ),
            )
        }
        EventKind::Failed => (
            format!(
                "Error: {}",
                quote(view.outcome.error.as_deref().unwrap_or("(none recorded)"))
            ),
            format!(
                "Inspect it with the worker tool: action \"status\" (then \"logs\"), worker_id \"{}\".",
                view.worker_id
            ),
        ),
    };
    format!("{header}\n{body}\n{verb}")
}

/// Spawn the background task that pushes worker events into the session.
///
/// Frames go out on the same outbound channel as the responses, so a
/// notification can never land inside a response frame. The returned handle is
/// aborted when the stdio loop ends.
pub(super) fn spawn_event_stream(pool: WorkerPool, tx: mpsc::Sender<String>) -> JoinHandle<()> {
    tokio::spawn(async move {
        // Seed the first snapshot with the workers that are already terminal,
        // so a server starting next to finished workers does not replay their
        // history. A worker already paused on a question is left out of the
        // seed on purpose: it still needs an answer, so the first tick
        // announces it.
        let mut previous = snapshot(&pool, &WorkerSnapshot::new()).await;
        previous.retain(|_, view| view.event != Some(EventKind::NeedsInput));
        loop {
            tokio::time::sleep(POLL_INTERVAL).await;
            let current = snapshot(&pool, &previous).await;
            for event in diff_events(&previous, &current) {
                let Some(frame) = channel_frame(&event) else {
                    continue;
                };
                if tx.send(frame).await.is_err() {
                    // The client is gone; there is nobody left to notify.
                    return;
                }
            }
            previous = current;
        }
    })
}

/// Read every known worker once.
///
/// The registry is the only cross-process view, so it supplies both the id set
/// and the workers another `mini-swe-mcp` process owns; this process's pool is
/// then read for the payload a registry row cannot carry. The loader skips a
/// row it cannot parse, so a half-written or foreign registry file is skipped
/// here too instead of taking the loop down.
async fn snapshot(pool: &WorkerPool, reported: &WorkerSnapshot) -> WorkerSnapshot {
    let mut current = WorkerSnapshot::new();
    for entry in crate::pool::load_all_registry_entries() {
        current.insert(entry.id.clone(), registry_view(&entry));
    }

    // A dispatch writes its registry row before the record becomes visible in
    // the pool, so the registry's id set is already complete and the in-memory
    // read below only enriches it.
    for id in current.keys().cloned().collect::<Vec<String>>() {
        let Some(progress) = pool.worker_progress(&id).await else {
            // Registry-only worker: owned by another process, nothing to enrich.
            continue;
        };
        let Some(view) = current.get_mut(&id) else {
            continue;
        };
        view.status = phase_status(progress.phase).to_string();
        view.event = match progress.phase {
            WorkerPhase::Running => None,
            WorkerPhase::Paused => Some(EventKind::NeedsInput),
            WorkerPhase::Completed => Some(EventKind::Completed),
            WorkerPhase::Failed => Some(EventKind::Failed),
        };
        if let Some(question) = progress.question {
            view.question = Some(question);
        }
        // The terminal payload costs a full state clone — a completed worker's
        // diff is megabytes — so it is read once per transition: a worker
        // whose event was already reported keeps what the registry described.
        if view.event.is_some()
            && reported.get(&id).and_then(|was| was.event) != view.event
            && let Some(state) = pool.get_worker_state(&id).await
        {
            view.outcome = outcome_of(&state);
        }
    }
    current
}

/// A worker as the on-disk registry describes it.
fn registry_view(entry: &WorkerRegistryEntry) -> WorkerView {
    WorkerView {
        worker_id: entry.id.clone(),
        event: match entry.status {
            RegistryStatus::Paused => Some(EventKind::NeedsInput),
            RegistryStatus::Completed => Some(EventKind::Completed),
            RegistryStatus::Failed => Some(EventKind::Failed),
            RegistryStatus::Running | RegistryStatus::Reviewing | RegistryStatus::Stopped => None,
        },
        group: entry
            .group
            .clone()
            .unwrap_or_else(|| String::from("default")),
        model: entry.model.clone(),
        status: registry_status(entry.status).to_string(),
        question: entry.question.clone(),
        outcome: Outcome {
            error: registry_error(entry),
            diff_stat: diff_stat(&entry.metrics),
            ..Outcome::default()
        },
    }
}

/// The failure reason a registry row carries.
///
/// The failure write is the only path that records a status row for a dead
/// worker, and it stores the reason in `last_command` as `error: <reason>`.
fn registry_error(entry: &WorkerRegistryEntry) -> Option<String> {
    if entry.status != RegistryStatus::Failed {
        return None;
    }
    let reason = entry.last_command.trim();
    let reason = reason.strip_prefix("error:").unwrap_or(reason).trim();
    (!reason.is_empty()).then(|| quote(reason))
}

/// The terminal payload of an in-memory worker.
fn outcome_of(state: &WorkerState) -> Outcome {
    match state {
        WorkerState::Completed {
            summary,
            verified,
            metrics,
            ..
        } => Outcome {
            summary: first_line(summary),
            verified: *verified,
            diff_stat: diff_stat(metrics),
            error: None,
        },
        WorkerState::Failed { error, metrics, .. } => Outcome {
            error: (!error.trim().is_empty()).then(|| quote(error)),
            diff_stat: diff_stat(metrics),
            ..Outcome::default()
        },
        WorkerState::Running { .. } | WorkerState::Paused { .. } => Outcome::default(),
    }
}

/// `3 files, +40 -12`, or `None` when the run measured no diff at all.
fn diff_stat(metrics: &WorkerMetrics) -> Option<String> {
    let touched = metrics.diff_files + metrics.diff_insertions + metrics.diff_deletions;
    (touched > 0).then(|| {
        format!(
            "{} file{}, +{} -{}",
            metrics.diff_files,
            if metrics.diff_files == 1 { "" } else { "s" },
            metrics.diff_insertions,
            metrics.diff_deletions
        )
    })
}

/// The first line of a summary: an agent writes a heading and a body, and only
/// the heading belongs in a notification.
fn first_line(summary: &str) -> Option<String> {
    let line = summary.lines().next().unwrap_or_default().trim();
    (!line.is_empty()).then(|| quote(line))
}

/// Bound text copied out of a worker into a notification.
fn quote(text: &str) -> String {
    clamp_string(text, QUOTE_BYTES)
}

/// The lower-case status name of a registry row, as it goes into `meta.status`.
fn registry_status(status: RegistryStatus) -> &'static str {
    match status {
        RegistryStatus::Running => "running",
        RegistryStatus::Paused => "paused",
        RegistryStatus::Reviewing => "reviewing",
        RegistryStatus::Completed => "completed",
        RegistryStatus::Failed => "failed",
        RegistryStatus::Stopped => "stopped",
    }
}

/// The lower-case status name of an in-memory worker.
fn phase_status(phase: WorkerPhase) -> &'static str {
    match phase {
        WorkerPhase::Running => "running",
        WorkerPhase::Paused => "paused",
        WorkerPhase::Completed => "completed",
        WorkerPhase::Failed => "failed",
    }
}
