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
//! The daemon uses one pool watcher and a bounded replay buffer. Connections
//! receive only their owner's events, unless they explicitly announce admin.
//! Terminal workers present at daemon startup are seeded silently; new events
//! remain available in memory even while their owner is disconnected.
//!
//! The notification text is rendered for a model, not for a log: the diff
//! between two snapshots is pure ([`diff_events`]), and only that pure part
//! decides *what* to say; rendering ([`ChannelEvent`], [`channel_frame`]) and
//! the polling loop are kept apart so the decision is testable on its own.

use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;
use std::time::Duration;

use serde_json::json;
use tokio::sync::{Mutex, mpsc, watch};
use tokio::task::JoinHandle;

use crate::pool::{
    RegistryStatus, WorkerMetrics, WorkerPhase, WorkerPool, WorkerRegistryEntry, WorkerState,
    clamp_string,
};

/// How often the event task re-reads the registry for workers it does not own.
///
/// This tick only exists for a worker owned by another `mini-swe-mcp` process
/// (a CLI dispatch) or `MINI_SWE_NO_DAEMON` mode: a worker of this process
/// wakes the task through the pool's change subscription. Thirty seconds is
/// coarse enough that a fleet of workers costs a handful of small registry
/// reads per minute, and fine enough that a paused worker is still reported
/// long before the orchestrator gives up on it.
const FALLBACK_INTERVAL: Duration = Duration::from_secs(30);

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
    /// Branch the finished worker committed to (`worker-<id>`), when known.
    /// Carried so the completion guidance can name the branch the revision
    /// resumes on.
    pub branch: Option<String>,
    /// Times the worker was revised after finishing.
    pub revision: usize,
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
    render_event(view, kind)
}

/// Test seam for the notification text: the decision (what to say) is pure and
/// asserted without standing up the polling loop.
#[doc(hidden)]
pub fn render_for_test(view: &WorkerView, kind: EventKind) -> String {
    render_event(view, kind)
}

/// The text the model reads, shared by the producer and the test seam.
fn render_event(view: &WorkerView, kind: EventKind) -> String {
    let header = if view.revision > 0 {
        format!(
            "Worker {} is {} (model {}, group {}, revision {}).",
            view.worker_id, view.status, view.model, view.group, view.revision
        )
    } else {
        format!(
            "Worker {} is {} (model {}, group {}).",
            view.worker_id, view.status, view.model, view.group
        )
    };
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
                body.push_str(&format!("Diff: {diff}\n"));
            }
            if body.trim().is_empty() {
                body.push_str("Completed with no recorded summary.\n");
            }
            body.push_str(&crate::pool::next_step_for(view.branch.as_deref()));
            (
                body.trim_end().to_string(),
                format!(
                    "Review it with the worker tool: action \"collect\", worker_id \"{}\".",
                    view.worker_id
                ),
            )
        }
        EventKind::Failed => {
            let body = format!(
                "Error: {}\n{}",
                quote(view.outcome.error.as_deref().unwrap_or("(none recorded)")),
                crate::pool::next_step_for(view.branch.as_deref()),
            );
            (
            body,
            format!(
                "Inspect it with the worker tool: action \"status\" (then \"logs\"), worker_id \"{}\".",
                view.worker_id
            ),
        )
        }
    };
    format!("{header}\n{body}\n{verb}")
}

/// Spawn the background task that pushes worker events into the session.
///
/// Frames go out on the same outbound channel as the responses, so a
/// notification can never land inside a response frame. The returned handle is
/// aborted when the stdio loop ends.
pub(super) fn spawn_event_stream(
    pool: WorkerPool,
    tx: mpsc::Sender<String>,
    mut context: watch::Receiver<super::server::ConnectionContext>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut changes = pool.subscribe_changes();
        let mut previous = snapshot(&pool, &WorkerSnapshot::new()).await;
        previous.retain(|_, view| view.event != Some(EventKind::NeedsInput));
        loop {
            tokio::select! {
                _ = changes.changed() => {}
                _ = context.changed() => {}
                _ = tokio::time::sleep(FALLBACK_INTERVAL) => {}
            }
            let current = snapshot(&pool, &previous).await;
            let ctx = context.borrow().clone();
            for event in diff_events(&previous, &current) {
                if owns(&pool, &ctx, &event.worker_id).await
                    && let Some(frame) = channel_frame(&event)
                    && tx.send(frame).await.is_err()
                {
                    return;
                }
            }
            previous = current;
        }
    })
}

async fn owns(pool: &WorkerPool, ctx: &super::server::ConnectionContext, id: &str) -> bool {
    ctx.is_admin() || pool.worker_owner(id).await
        == Some(crate::pool::WorkerOwner::Agent(ctx.agent()))
}

/// Latest events survive disconnected owners, but never retain more than 100 workers.
#[derive(Default)]
pub(super) struct EventRouter {
    latest: VecDeque<(Option<String>, ChannelEvent)>,
    connections: BTreeMap<u64, (String, bool, mpsc::Sender<String>)>,
}

impl EventRouter {
    pub(super) async fn register(&mut self, ctx: &super::server::ConnectionContext, tx: mpsc::Sender<String>) {
        let agent = ctx.agent();
        if self.connections.get(&ctx.id).is_some_and(|(old, admin, _)| old == &agent && *admin == ctx.is_admin()) {
            return;
        }
        for (owner, event) in &self.latest {
            if (ctx.is_admin() || owner.as_deref() == Some(agent.as_str()))
                && let Some(frame) = channel_frame(event)
                && tx.send(frame).await.is_err()
            {
                return;
            }
        }
        self.connections.insert(ctx.id, (agent, ctx.is_admin(), tx));
    }

    pub(super) fn remove(&mut self, id: u64) {
        self.connections.remove(&id);
    }

    fn publish(&mut self, owner: Option<String>, event: ChannelEvent) {
        self.latest.retain(|(_, old)| old.worker_id != event.worker_id);
        if self.latest.len() == 100 {
            self.latest.pop_front();
        }
        if let Some(frame) = channel_frame(&event) {
            self.connections.retain(|_, (agent, admin, tx)| {
                if *admin || owner.as_deref() == Some(agent.as_str()) {
                    // A stalled connection must not block the pool watcher.
                    tx.try_send(frame.clone()).is_ok()
                } else {
                    !tx.is_closed()
                }
            });
        }
        self.latest.push_back((owner, event));
    }
}

pub(super) async fn spawn_hub_events(pool: WorkerPool, router: Arc<Mutex<EventRouter>>) -> JoinHandle<()> {
    let mut changes = pool.subscribe_changes();
    let mut previous = snapshot(&pool, &WorkerSnapshot::new()).await;
    previous.retain(|_, view| view.event != Some(EventKind::NeedsInput));
    tokio::spawn(async move {
        loop {
            let current = snapshot(&pool, &previous).await;
            for event in diff_events(&previous, &current) {
                let owner = match pool.worker_owner(&event.worker_id).await {
                    Some(crate::pool::WorkerOwner::Agent(owner)) => Some(owner),
                    _ => None,
                };
                router.lock().await.publish(owner, event);
            }
            previous = current;
            tokio::select! {
                _ = changes.changed() => {}
                _ = tokio::time::sleep(FALLBACK_INTERVAL) => {}
            }
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

    // Include in-memory workers even when their registry row was removed.
    for row in pool.list_workers().await {
        if let Some(id) = row["id"].as_str() {
            current.entry(id.to_string()).or_insert_with(|| WorkerView {
                worker_id: id.to_string(),
                model: row["model"].as_str().unwrap_or_default().to_string(),
                group: "default".to_string(),
                ..WorkerView::default()
            });
        }
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
            // The in-memory state names the branch the registry row cannot, so
            // the completion guidance points at the branch a revision resumes.
            view.branch = crate::pool::terminal_branch(&state);
            view.revision = match &state {
                WorkerState::Completed { revision, .. }
                | WorkerState::Failed { revision, .. } => *revision,
                WorkerState::Running { .. } | WorkerState::Paused { .. } => 0,
            };
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
        // A registry row names no branch, so the guidance falls back to the
        // branch-less wording; the in-memory enrichment below fills in the
        // real branch when this process owns the worker.
        branch: None,
        revision: 0,
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


#[cfg(test)]
mod router_tests {
    use super::*;
    use crate::mcp::ConnectionContext;

    fn event(id: &str, kind: EventKind) -> ChannelEvent {
        ChannelEvent {
            worker_id: id.to_string(), kind, group: "default".to_string(),
            model: "test".to_string(), status: kind.as_str().to_string(), content: "test".to_string(),
        }
    }

    #[tokio::test]
    async fn replay_is_bounded_latest_per_worker_and_owner_scoped() {
        let mut router = EventRouter::default();
        for id in 0..101 {
            router.publish(Some("a".to_string()), event(&id.to_string(), EventKind::Completed));
        }
        router.publish(Some("a".to_string()), event("1", EventKind::Failed));
        assert_eq!(router.latest.len(), 100);
        assert!(!router.latest.iter().any(|(_, event)| event.worker_id == "0"));
        let (tx, mut rx) = mpsc::channel(128);
        let mut ctx = ConnectionContext::hub_connection(1);
        ctx.agent_id = Some("b".to_string());
        router.register(&ctx, tx.clone()).await;
        assert!(rx.try_recv().is_err());
        ctx.agent_id = Some("a".to_string());
        router.register(&ctx, tx).await;
        let mut frames = Vec::new();
        while let Ok(frame) = rx.try_recv() {
            frames.push(serde_json::from_str::<serde_json::Value>(&frame).unwrap());
        }
        assert_eq!(frames.len(), 100);
        assert_eq!(frames.last().unwrap()["params"]["meta"]["worker_id"], "1");
        assert_eq!(frames.last().unwrap()["params"]["meta"]["event"], "failed");
    }
}
