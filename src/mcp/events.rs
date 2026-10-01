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
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use serde_json::json;
use tokio::sync::{Mutex, mpsc, watch};
use tokio::task::JoinHandle;

use crate::pool::{
    FileStat, LogBuffer, RegistryStatus, WorkerMetrics, WorkerPhase, WorkerPool,
    WorkerRegistryEntry, WorkerReport, WorkerState, clamp_string, file_stats_of_diff,
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
    /// A worker spent its turn budget without completing. Its branch is
    /// checkpointed, but it is stopped, not done.
    Exhausted,
}

impl EventKind {
    /// The wire name: the `meta.event` value and the `event="…"` attribute the
    /// client wraps the content in.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NeedsInput => "needs_input",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Exhausted => "exhausted",
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
    /// The structured report of the completion turn, when the worker wrote one.
    pub report: Option<WorkerReport>,
    /// The completion diff split per file, biggest churn first.
    pub per_file: Vec<FileStat>,
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
    /// Turns the worker has performed, so an exhausted worker's continuation
    /// can name the budget it spent.
    pub turns: usize,
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
    /// Full completion report and per-file stats, independent of the text view.
    pub report: Option<WorkerReport>,
    pub per_file: Vec<FileStat>,
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
                report: view.outcome.report.clone(),
                per_file: view.outcome.per_file.clone(),
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
            "report": event.report,
            "per_file": event.per_file,
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
            // The report's `done:` line is the headline; the verification flag
            // rides on it so the body stays five lines even with a per-file
            // diff and a risk note.
            let verified = view.outcome.verified.map_or_else(String::new, |ok| {
                format!(" | Verified: {}", if ok { "yes" } else { "no" })
            });
            let headline = view
                .outcome
                .report
                .as_ref()
                .map(|report| report.done.trim())
                .filter(|done| !done.is_empty())
                .or(view.outcome.summary.as_deref());
            match headline {
                Some(done) => body.push_str(&format!("Done: {done}{verified}\n")),
                None if verified.is_empty() => {
                    body.push_str("Completed with no recorded summary.\n")
                }
                None => body.push_str(&format!("Completed{verified}\n")),
            }
            if let Some(diff) = &view.outcome.diff_stat {
                body.push_str(&format!("Diff: {diff}\n"));
            }
            let files = crate::pool::churn_line(&view.outcome.per_file);
            if !files.is_empty() {
                body.push_str(&format!("files: {files}\n"));
            }
            if let Some(risks) = view
                .outcome
                .report
                .as_ref()
                .map(|report| report.risks.trim())
                .filter(|risks| !risks.is_empty() && !risks.eq_ignore_ascii_case("none"))
            {
                body.push_str(&format!("risks: {risks}\n"));
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
        EventKind::Exhausted => {
            let mut body = String::new();
            if let Some(diff) = &view.outcome.diff_stat {
                body.push_str(&format!("Diff: {diff}\n"));
            }
            body.push_str(&crate::pool::exhausted_next_step(
                &view.worker_id,
                view.turns,
                view.branch.as_deref(),
            ));
            (
                body.trim_end().to_string(),
                format!(
                    "Continue it with the worker tool: action \"steer\", worker_id \"{}\", message \"continue\".",
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
                    && !try_deliver(&tx, frame)
                {
                    return;
                }
            }
            previous = current;
        }
    })
}

async fn owns(pool: &WorkerPool, ctx: &super::server::ConnectionContext, id: &str) -> bool {
    ctx.is_admin()
        || pool.worker_owner(id).await == Some(crate::pool::WorkerOwner::Agent(ctx.agent()))
}

/// Latest events survive disconnected owners, but never retain more than 100 workers.
/// One agent's live watch: who holds it and what to tell a second caller.
struct ActiveWatch {
    token: u64,
    connection: u64,
    pid: Option<u32>,
    since: u64,
}

/// Identity -> the single watch that may be active for it.
///
/// A hub connection owns its slot for the connection's lifetime, released from
/// [`EventRouter::remove`]; an MCP `watch` call owns it for the call's lifetime
/// through a [`WatchGuard`]. A closed or cancelled watch frees its own slot, so
/// no stale lock outlives its watcher.
#[derive(Default)]
pub(super) struct WatchRegistry {
    slots: std::sync::Mutex<BTreeMap<String, ActiveWatch>>,
    next_token: AtomicU64,
}

impl WatchRegistry {
    fn lock(&self) -> std::sync::MutexGuard<'_, BTreeMap<String, ActiveWatch>> {
        self.slots
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
    }

    /// Whether `identity` holds a watch slot right now.
    fn has(&self, identity: &str) -> bool {
        self.lock().contains_key(identity)
    }

    fn busy(active: &ActiveWatch) -> anyhow::Error {
        let held = match active.pid {
            Some(pid) => format!("pid {pid}, since {}", active.since),
            None => format!("since {}", active.since),
        };
        anyhow::anyhow!(
            "a watch is already running for your session ({held}); it will deliver the next event - do not start another"
        )
    }

    fn insert(&self, identity: &str, connection: u64, pid: Option<u32>) -> anyhow::Result<u64> {
        let mut slots = self.lock();
        if let Some(active) = slots.get(identity) {
            return Err(Self::busy(active));
        }
        let token = self.next_token.fetch_add(1, Ordering::Relaxed);
        slots.insert(
            identity.to_string(),
            ActiveWatch {
                token,
                connection,
                pid,
                since: crate::pool::unix_timestamp(),
            },
        );
        Ok(token)
    }

    /// Claim `identity` for a hub connection, re-entrant so that connection's
    /// own repeated polls keep the same slot instead of locking themselves out.
    fn claim(&self, identity: &str, connection: u64, pid: Option<u32>) -> anyhow::Result<()> {
        if let Some(active) = self.lock().get(identity) {
            anyhow::ensure!(active.connection == connection, "{}", Self::busy(active));
            return Ok(());
        }
        self.insert(identity, connection, pid)?;
        Ok(())
    }

    fn release_token(&self, identity: &str, token: u64) {
        let mut slots = self.lock();
        if slots
            .get(identity)
            .is_some_and(|active| active.token == token)
        {
            slots.remove(identity);
        }
    }

    fn release_connection(&self, connection: u64) {
        self.lock()
            .retain(|_, active| active.connection != connection);
    }
}

/// Holds one identity's watch slot for the lifetime of an MCP `watch` call.
pub(super) struct WatchGuard {
    registry: Arc<WatchRegistry>,
    identity: String,
    token: u64,
}

impl Drop for WatchGuard {
    fn drop(&mut self) {
        self.registry.release_token(&self.identity, self.token);
    }
}

/// File name of the persisted per-owner acknowledged watch positions.
const WATCH_ACKS_FILE: &str = "watch_acks.json";
/// How many owners the persisted store keeps, least recently
/// acknowledged evicted first.
const MAX_ACK_OWNERS: usize = 1024;
/// How many workers one owner's persisted store keeps, least recently
/// acknowledged evicted first.
const MAX_ACK_WORKERS: usize = 4096;
/// Snapshot key marking a terminal worker whose branch is merged into its
/// base or no longer exists, so its event must never be replayed.
const BRANCH_GONE_OR_MERGED: &str = "branch_gone_or_merged";

/// The revision and kind of the last event an owner acknowledged for a worker.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
struct AckPosition {
    revision: u64,
    event: String,
}

/// One acknowledged worker and how recently its position was written.
#[derive(Debug, Clone)]
struct AckEntry {
    position: AckPosition,
    /// Bumped on every `record` for that key, so eviction is by true
    /// recency and never by key order.
    recency: u64,
}

/// Per-owner acknowledged watch positions, persisted so the first watch after a
/// daemon restart does not replay events the owner already saw.
///
/// The in-memory maps carry a monotonically increasing `recency` stamp per
/// entry; eviction is by least-recently-acknowledged, so the store keeps the
/// positions the owner is still likely to re-acknowledge. `path` is `None`
/// for the stdio router and for unit tests, which keeps the store entirely in
/// memory; the hub daemon points it at `<hub dir>/watch_acks.json`.
///
/// Eviction is deterministic (lowest recency stamp first, ties broken by key)
/// but is not part of the persisted format: a restart re-derives recency from
/// the order positions are next written, never from the file.
#[derive(Default)]
struct AckStore {
    path: Option<PathBuf>,
    positions: BTreeMap<String, BTreeMap<String, AckEntry>>,
    /// Monotonic stamp handed out by `record`, comparing entries by
    /// recency across every owner and worker.
    clock: u64,
}

impl AckStore {
    /// Load the store under `dir`. A missing or unreadable file is an empty
    /// store, never an error: losing the cache costs a replay, not correctness.
    fn load(&mut self, dir: &Path) {
        let path = dir.join(WATCH_ACKS_FILE);
        let loaded: BTreeMap<String, BTreeMap<String, AckPosition>> =
            std::fs::read_to_string(&path)
                .ok()
                .and_then(|text| serde_json::from_str(&text).ok())
                .unwrap_or_default();
        // A freshly loaded store has no recency information: every entry
        // starts equally stale, and the next `record` stamps it fresh.
        self.positions = loaded
            .into_iter()
            .map(|(owner, workers)| {
                let workers = workers
                    .into_iter()
                    .map(|(wid, position)| {
                        (
                            wid,
                            AckEntry {
                                position,
                                recency: 0,
                            },
                        )
                    })
                    .collect();
                (owner, workers)
            })
            .collect();
        self.trim();
        self.path = Some(path);
    }

    /// Whether `owner` already acknowledged exactly this `(revision, kind)` for
    /// `wid`, so it must not be replayed.
    fn acknowledged(&self, owner: &str, wid: &str, revision: u64, event: &str) -> bool {
        self.positions
            .get(owner)
            .and_then(|workers| workers.get(wid))
            .is_some_and(|entry| {
                entry.position.revision == revision && entry.position.event == event
            })
    }

    /// Remember `owner`'s newest acknowledged position for `wid` and persist it.
    fn record(&mut self, owner: &str, wid: &str, revision: u64, event: &str) {
        self.clock += 1;
        let recency = self.clock;
        self.positions
            .entry(owner.to_string())
            .or_default()
            .insert(
                wid.to_string(),
                AckEntry {
                    position: AckPosition {
                        revision,
                        event: event.to_string(),
                    },
                    recency,
                },
            );
        self.trim();
        self.persist();
    }

    /// Keep the store bounded, evicting the least recently acknowledged
    /// owner, then the least recently acknowledged worker of every owner.
    fn trim(&mut self) {
        while self.positions.len() > MAX_ACK_OWNERS {
            let Some(victim) = self
                .positions
                .iter()
                .map(|(owner, workers)| {
                    let oldest = workers.values().map(|e| e.recency).min().unwrap_or(0);
                    (owner.clone(), oldest)
                })
                .min_by_key(|(owner, oldest)| (*oldest, owner.clone()))
                .map(|(owner, _)| owner)
            else {
                break;
            };
            self.positions.remove(&victim);
        }
        for workers in self.positions.values_mut() {
            while workers.len() > MAX_ACK_WORKERS {
                let Some(victim) = workers
                    .iter()
                    .map(|(wid, entry)| (wid.clone(), entry.recency))
                    .min_by_key(|(wid, recency)| (*recency, wid.clone()))
                    .map(|(wid, _)| wid)
                else {
                    break;
                };
                workers.remove(&victim);
            }
        }
    }

    /// Write the store atomically at 0600. A failure is logged, never fatal.
    fn persist(&self) {
        use std::os::unix::fs::PermissionsExt;
        let Some(path) = &self.path else {
            return;
        };
        // The recency stamps are process-local bookkeeping, so the
        // persisted file carries only the compact position map.
        let positions: BTreeMap<_, BTreeMap<_, _>> = self
            .positions
            .iter()
            .map(|(owner, workers)| {
                let workers = workers
                    .iter()
                    .map(|(wid, entry)| (wid, &entry.position))
                    .collect();
                (owner, workers)
            })
            .collect();
        let Ok(text) = serde_json::to_string(&positions) else {
            return;
        };
        let tmp = path.with_extension("json.tmp");
        if std::fs::write(&tmp, text.as_bytes()).is_ok() {
            let _ = std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o600));
            if std::fs::rename(&tmp, path).is_err() {
                let _ = std::fs::remove_file(&tmp);
            }
        }
    }
}

#[derive(Default)]
pub(super) struct EventRouter {
    latest: VecDeque<(Option<String>, ChannelEvent)>,
    connections: BTreeMap<u64, (String, bool, mpsc::Sender<String>)>,
    watch_current: crate::cli::watch::Snapshot,
    watch_reported: crate::cli::watch::Snapshot,
    /// Worker id -> sequence of its last event the owner saw through an
    /// interactive verb. The sequence stays out of the payload so the
    /// published and JSON event shapes are unchanged.
    seen: BTreeMap<String, u64>,
    watch_history: BTreeMap<String, WatchHistory>,
    /// Per-owner acknowledged positions, persisted across a daemon restart.
    acks: AckStore,
    watches: Arc<WatchRegistry>,
    sequence: u64,
}

impl EventRouter {
    fn watch_channel_frame(&self, event: &ChannelEvent) -> Option<String> {
        let mut frame: serde_json::Value = serde_json::from_str(&channel_frame(event)?).ok()?;
        if let Some(payload) = self.watch_reported.get(&event.worker_id) {
            frame["params"]["payload"] = payload.clone();
        }
        Some(frame.to_string() + "\n")
    }

    pub(super) fn register(
        &mut self,
        ctx: &super::server::ConnectionContext,
        tx: mpsc::Sender<String>,
    ) {
        let agent = ctx.agent();
        if self
            .connections
            .get(&ctx.id)
            .is_some_and(|(old, admin, _)| old == &agent && *admin == ctx.is_admin())
        {
            return;
        }
        for (owner, event) in &self.latest {
            if (ctx.is_admin() || owner.as_deref() == Some(agent.as_str()))
                && let Some(frame) = self.watch_channel_frame(event)
                && !try_deliver(&tx, frame)
            {
                return;
            }
        }
        self.connections.insert(ctx.id, (agent, ctx.is_admin(), tx));
    }

    pub(super) fn remove(&mut self, id: u64) {
        self.connections.remove(&id);
        self.watches.release_connection(id);
    }

    /// Reserve `identity`'s one watch slot for an MCP `watch` call. The returned
    /// guard frees it when the call returns or is cancelled.
    pub(super) fn begin_watch(
        &self,
        identity: &str,
        connection: u64,
        pid: Option<u32>,
    ) -> anyhow::Result<WatchGuard> {
        let token = self.watches.insert(identity, connection, pid)?;
        Ok(WatchGuard {
            registry: Arc::clone(&self.watches),
            identity: identity.to_string(),
            token,
        })
    }

    /// Whether `identity` already has a watch running, so a dispatch or steer
    /// answer can drop the "start this" line the caller has already acted on.
    pub(super) fn has_watch(&self, identity: &str) -> bool {
        self.watches.has(identity)
    }

    /// Load the persisted acknowledged positions from the hub directory. Called
    /// once at daemon start, before the watcher observes any worker.
    pub(super) fn load_ack_store(&mut self, dir: &Path) {
        self.acks.load(dir);
    }

    fn publish(&mut self, owner: Option<String>, event: ChannelEvent) {
        self.latest
            .retain(|(_, old)| old.worker_id != event.worker_id);
        if self.latest.len() == 100 {
            self.latest.pop_front();
        }
        if let Some(frame) = self.watch_channel_frame(&event) {
            self.connections.retain(|_, (agent, admin, tx)| {
                if *admin || owner.as_deref() == Some(agent.as_str()) {
                    // A stalled connection must not block the pool watcher.
                    try_deliver(tx, frame.clone())
                } else {
                    !tx.is_closed()
                }
            });
        }
        self.latest.push_back((owner, event));
    }
}

fn try_deliver(tx: &mpsc::Sender<String>, frame: String) -> bool {
    match tx.try_send(frame) {
        Ok(()) => true,
        Err(mpsc::error::TrySendError::Full(_)) => {
            tracing::debug!("Dropping worker event for a stalled connection");
            true
        }
        Err(mpsc::error::TrySendError::Closed(_)) => false,
    }
}

pub(super) async fn spawn_hub_events(
    pool: WorkerPool,
    router: Arc<Mutex<EventRouter>>,
) -> JoinHandle<()> {
    let mut changes = pool.subscribe_changes();
    let mut previous = snapshot(&pool, &WorkerSnapshot::new()).await;
    previous.retain(|_, view| view.event != Some(EventKind::NeedsInput));
    tokio::spawn(async move {
        loop {
            let current = snapshot(&pool, &previous).await;
            {
                let mut guard = router.lock().await;
                let views = watch_snapshot(&pool).await;
                guard.observe_watch(views);
            }
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
                _ = tokio::time::sleep(Duration::from_secs(1)) => {}
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
    for entry in crate::pool::load_all_registry_entries_in(pool.scratch_root()) {
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
        view.turns = progress.step;
        view.event = match progress.phase {
            WorkerPhase::Running => None,
            WorkerPhase::Paused => Some(EventKind::NeedsInput),
            WorkerPhase::Completed => Some(EventKind::Completed),
            WorkerPhase::Failed => Some(EventKind::Failed),
            WorkerPhase::Exhausted => Some(EventKind::Exhausted),
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
                | WorkerState::Failed { revision, .. }
                | WorkerState::Exhausted { revision, .. } => *revision,
                WorkerState::Running { .. } | WorkerState::Paused { .. } => 0,
            };
        }
    }
    for (id, view) in &mut current {
        if view.event == Some(EventKind::NeedsInput) && pool.question_for_consolidator(id) {
            view.event = None;
        } else if matches!(
            view.event,
            Some(EventKind::Completed | EventKind::Failed | EventKind::Exhausted)
        ) && pool.steered_by_live_consolidator(id).await
        {
            // The consolidator that steered this worker is blocked in
            // `CONSOLIDATE_WAIT` on exactly this stop, so the owner's watch
            // stays quiet until the round is over (or the consolidator died):
            // the owner sees the worker again once the source names no live
            // consolidator.
            view.event = None;
        }
    }
    current
}

/// Lines a failed `[verify]` run is worth reading: failure markers and the
/// newest context, bounded to 40 lines and 4 KiB.
///
/// A verify run prints thousands of passing-test lines, so the raw tail is both
/// huge and uninformative. Marked lines (`FAILED`, `panicked`, `error`,
/// `warning:`, `assertion`) win the budget: when none of the last 40 lines
/// carries one, the most recent marked lines are shown instead of the oldest
/// tail lines, and the byte cap drops unmarked lines first.
pub(super) fn verify_tail(output: &str) -> String {
    const MAX_LINES: usize = 40;
    const MAX_BYTES: usize = 4096;
    const MARKERS: [&str; 5] = ["FAILED", "panicked", "error", "warning:", "assertion"];
    fn marked(line: &str) -> bool {
        MARKERS.iter().any(|marker| line.contains(marker))
    }
    let lines: Vec<&str> = output.lines().collect();
    let mut window: Vec<&str> = lines.iter().rev().take(MAX_LINES).rev().copied().collect();
    if !window.iter().any(|line| marked(line)) {
        let mut failures: Vec<&str> = lines.iter().filter(|line| marked(line)).copied().collect();
        failures.reverse();
        failures.truncate(MAX_LINES);
        let kept = failures.len();
        if kept > 0 {
            let context = window.len().saturating_sub(MAX_LINES - kept);
            let mut chosen = failures;
            chosen.extend_from_slice(&window[window.len() - context..]);
            window = chosen;
        }
    }
    let mut total: usize = window.iter().map(|line| line.len() + 1).sum();
    let mut index = 0;
    while total > MAX_BYTES && index < window.len() {
        if marked(window[index]) {
            index += 1;
        } else {
            total -= window[index].len() + 1;
            window.remove(index);
        }
    }
    let mut text = window.join("\n");
    if text.len() > MAX_BYTES {
        let cut = text.len() - MAX_BYTES;
        let start = text
            .char_indices()
            .find(|(at, _)| *at >= cut)
            .map(|(at, _)| at)
            .unwrap_or(text.len());
        text = text[start..].to_string();
    }
    text
}

/// The failure-focused tail of the newest `[verify]` step in `logs`, when the
/// window holds one.
pub(super) fn verify_tail_of(logs: &[&crate::agent::AgentStepLog]) -> Option<String> {
    logs.iter()
        .rev()
        .find(|log| log.command.starts_with("[verify]"))
        .map(|log| verify_tail(&log.output))
}

/// Attach the newest `[verify]` log tail to a view that is not verified.
///
/// A worker whose earlier verify run failed can still pass its completion
/// gate: showing that stale run would read as a current failure, so only an
/// unverified worker (whose tail explains the live failure) carries a tail.
fn attach_verify_tail(view: &mut serde_json::Value, logs: &LogBuffer) {
    if view["verified"] != true {
        view["verify_output_tail"] = json!(verify_tail_of(&logs.tail(1000)));
    }
}

/// A worker as the on-disk registry describes it.
fn registry_view(entry: &WorkerRegistryEntry) -> WorkerView {
    WorkerView {
        worker_id: entry.id.clone(),
        event: match entry.status {
            RegistryStatus::Paused => Some(EventKind::NeedsInput),
            RegistryStatus::Completed => Some(EventKind::Completed),
            RegistryStatus::Failed => Some(EventKind::Failed),
            RegistryStatus::Exhausted => Some(EventKind::Exhausted),
            // Interrupted is terminal for listing but continuable, so it is
            // not a terminal event: the worker is expected back.
            RegistryStatus::Interrupted => None,
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
            diff_stat: diff_stat(&entry.metrics).or_else(|| {
                // Zero metrics on a terminal row mean the worktree was already
                // gone when the row was sampled, not that the branch is empty.
                let (files, insertions, deletions) = crate::cli::watch::branch_diff_stat(entry)?;
                stat_text(files, insertions, deletions)
            }),
            // The row carries the report and its verification verdict so a
            // worker whose in-memory record was already evicted still says
            // what it did and whether it verified.
            verified: entry.verified,
            report: entry.report.clone(),
            summary: entry
                .report
                .as_ref()
                .and_then(|report| first_line(&report.done))
                .or_else(|| first_line(&entry.last_command)),
            ..Outcome::default()
        },
        // A registry row names no branch, so the guidance falls back to the
        // branch-less wording; the in-memory enrichment below fills in the
        // real branch when this process owns the worker.
        branch: None,
        revision: 0,
        turns: entry.step,
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
            diff,
            report,
            ..
        } => Outcome {
            summary: first_line(summary),
            verified: *verified,
            diff_stat: diff_stat(metrics),
            error: None,
            report: report.clone(),
            per_file: file_stats_of_diff(diff),
        },
        WorkerState::Failed { error, metrics, .. } => Outcome {
            error: (!error.trim().is_empty()).then(|| quote(error)),
            diff_stat: diff_stat(metrics),
            ..Outcome::default()
        },
        WorkerState::Exhausted {
            summary,
            metrics,
            diff,
            report,
            ..
        } => Outcome {
            summary: first_line(summary),
            verified: None,
            diff_stat: diff_stat(metrics),
            error: None,
            report: report.clone(),
            per_file: file_stats_of_diff(diff),
        },
        WorkerState::Running { .. } | WorkerState::Paused { .. } => Outcome::default(),
    }
}

/// `3 files, +40 -12`, or `None` when the run measured no diff at all.
fn diff_stat(metrics: &WorkerMetrics) -> Option<String> {
    stat_text(
        metrics.diff_files,
        metrics.diff_insertions,
        metrics.diff_deletions,
    )
}

/// [`diff_stat`] from raw counts, so a branch-derived diff renders identically.
fn stat_text(files: usize, insertions: usize, deletions: usize) -> Option<String> {
    (files + insertions + deletions > 0).then(|| {
        format!(
            "{files} file{}, +{insertions} -{deletions}",
            if files == 1 { "" } else { "s" }
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
        RegistryStatus::Exhausted => "exhausted",
        RegistryStatus::Stopped => "stopped",
        RegistryStatus::Interrupted => "interrupted",
    }
}

/// The lower-case status name of an in-memory worker.
fn phase_status(phase: WorkerPhase) -> &'static str {
    match phase {
        WorkerPhase::Running => "running",
        WorkerPhase::Paused => "paused",
        WorkerPhase::Completed => "completed",
        WorkerPhase::Failed => "failed",
        WorkerPhase::Exhausted => "exhausted",
    }
}

#[cfg(test)]
mod router_tests {
    use super::*;
    use crate::mcp::ConnectionContext;

    fn event(id: &str, kind: EventKind) -> ChannelEvent {
        ChannelEvent {
            worker_id: id.to_string(),
            kind,
            group: "default".to_string(),
            model: "test".to_string(),
            status: kind.as_str().to_string(),
            content: "test".to_string(),
            report: None,
            per_file: Vec::new(),
        }
    }

    #[test]
    fn full_channels_do_not_unregister_live_connections() {
        let mut router = EventRouter::default();
        let (tx, mut rx) = mpsc::channel(1);
        let mut ctx = ConnectionContext::hub_connection(1);
        ctx.agent_id = Some("a".to_string());
        router.register(&ctx, tx);
        router.publish(Some("a".to_string()), event("first", EventKind::Completed));
        router.publish(
            Some("a".to_string()),
            event("dropped", EventKind::Completed),
        );
        assert_eq!(router.connections.len(), 1);
        let first = rx.try_recv().unwrap();
        assert!(first.contains("first"));
        router.publish(Some("a".to_string()), event("next", EventKind::Failed));
        assert!(rx.try_recv().unwrap().contains("next"));
        drop(rx);
        router.publish(Some("a".to_string()), event("closed", EventKind::Failed));
        assert!(router.connections.is_empty());
    }

    #[test]
    fn replay_to_a_full_channel_never_blocks_registration() {
        let mut router = EventRouter::default();
        router.publish(Some("a".to_string()), event("first", EventKind::Completed));
        router.publish(Some("a".to_string()), event("second", EventKind::Failed));
        let (tx, mut rx) = mpsc::channel(1);
        tx.try_send("already full".to_string()).unwrap();
        let mut ctx = ConnectionContext::hub_connection(1);
        ctx.agent_id = Some("a".to_string());
        router.register(&ctx, tx);
        assert_eq!(router.connections.len(), 1);
        assert_eq!(rx.try_recv().unwrap(), "already full");
        router.publish(Some("a".to_string()), event("new", EventKind::Failed));
        assert!(rx.try_recv().unwrap().contains("new"));
        let (tx, rx) = mpsc::channel(1);
        drop(rx);
        ctx.id = 2;
        router.register(&ctx, tx);
        assert!(!router.connections.contains_key(&2));
    }

    #[tokio::test]
    async fn replay_is_bounded_latest_per_worker_and_owner_scoped() {
        let mut router = EventRouter::default();
        for id in 0..101 {
            router.publish(
                Some("a".to_string()),
                event(&id.to_string(), EventKind::Completed),
            );
        }
        router.publish(Some("a".to_string()), event("1", EventKind::Failed));
        assert_eq!(router.latest.len(), 100);
        assert!(
            !router
                .latest
                .iter()
                .any(|(_, event)| event.worker_id == "0")
        );
        let (tx, mut rx) = mpsc::channel(128);
        let mut ctx = ConnectionContext::hub_connection(1);
        ctx.agent_id = Some("b".to_string());
        router.register(&ctx, tx.clone());
        assert!(rx.try_recv().is_err());
        ctx.agent_id = Some("a".to_string());
        router.register(&ctx, tx);
        let mut frames = Vec::new();
        while let Ok(frame) = rx.try_recv() {
            frames.push(serde_json::from_str::<serde_json::Value>(&frame).unwrap());
        }
        assert_eq!(frames.len(), 100);
        assert_eq!(frames.last().unwrap()["params"]["meta"]["worker_id"], "1");
        assert_eq!(frames.last().unwrap()["params"]["meta"]["event"], "failed");
    }
}

/// An identity retains at most 100 unacknowledged events. Identity storage is
/// capped too; inactive identities are evicted oldest-first with their cursor.
#[derive(Default)]
struct WatchHistory {
    pending: VecDeque<serde_json::Value>,
    cursor: u64,
    dropped: u64,
    last_sequence: u64,
}

/// Whether a terminal worker's branch is merged into its base or no longer
/// exists, so its event must never be replayed.
///
/// A merge deletes the worker branch and keeps the registry row, so after a
/// daemon restart the row still reads `completed` and would otherwise replay.
///
/// Suppression requires *proof*: the repository must be readable and the
/// branch must be provably absent (`git show-ref` fails with exit `1`, the
/// "ref not found" code) or provably contained in the base branch
/// (`git merge-base --is-ancestor` succeeds). Any other outcome -- the git
/// process failed to run, the path is not a repository (exit `128`), or the
/// base ref cannot be resolved -- is *not* suppression: a terminal event that
/// could not be verified stays visible, never silently dropped.
fn branch_replay_suppressed(entry: &WorkerRegistryEntry) -> bool {
    /// `git` exit code for "the ref is not there", the one outcome that
    /// proves absence. Every other code means the probe itself failed.
    const REF_NOT_FOUND: i32 = 1;
    if !matches!(
        entry.status,
        RegistryStatus::Completed | RegistryStatus::Failed | RegistryStatus::Exhausted
    ) {
        return false;
    }
    let Some(repo) = entry
        .repo_path
        .as_deref()
        .map(Path::new)
        .filter(|path| path.is_dir())
    else {
        return false;
    };
    let branch = format!("worker-{}", entry.id);
    // A missing worktree probe is not a gone branch: `show-ref` distinguishes
    // the two by exit code, `1` meaning the ref is genuinely absent.
    let branch_gone = matches!(
        crate::worktree::git(
            repo,
            "show-ref --verify --quiet",
            &["show-ref", "--verify", "--quiet", &format!("refs/heads/{branch}")],
        ),
        Ok(output) if output.status.code() == Some(REF_NOT_FOUND)
    );
    if branch_gone {
        return true;
    }
    let Some(base) = entry
        .base_branch
        .as_deref()
        .filter(|base| !base.is_empty())
    else {
        return false;
    };
    // Only a successful ancestry check proves the branch landed in the
    // base. A failed probe (unreadable repo, unresolved base) is not proof.
    matches!(
        crate::worktree::git(
            repo,
            "merge-base --is-ancestor",
            &["merge-base", "--is-ancestor", &branch, base],
        ),
        Ok(output) if output.status.success()
    )
}

/// Read bounded watch facts in the daemon, without collecting the worker.
async fn watch_snapshot(pool: &WorkerPool) -> crate::cli::watch::Snapshot {
    use crate::cli::watch::{enrich_state, registry_snapshot};
    let now = crate::pool::unix_timestamp();
    let mut views: crate::cli::watch::Snapshot =
        crate::pool::load_all_registry_entries_in(pool.scratch_root())
            .iter()
            .map(|entry| {
                let mut view = registry_snapshot(entry, now);
                if branch_replay_suppressed(entry) {
                    view[BRANCH_GONE_OR_MERGED] = json!(true);
                }
                (entry.id.clone(), view)
            })
            .collect();
    for row in pool.list_workers().await {
        let Some(id) = row["id"].as_str() else {
            continue;
        };
        let view = views.entry(id.to_string()).or_insert_with(|| json!({
            "worker_id":id,"model":row["model"],"owner":row["owner"],"group":"default",
            "task":clamp_string(row["task"].as_str().unwrap_or("").lines().next().unwrap_or(""),500),
            "branch":null,"revision":0,"max_turns":0,"metrics":WorkerMetrics::default(),
            "elapsed":0,"last_step_at":now,"last_ops":[],"question":null
        }));
        view["owner"] = row["owner"].clone();
        if let Some(progress) = pool.worker_progress(id).await {
            view["step"] = json!(progress.step);
            view["turns"] = json!(progress.step);
            view["status"] = json!(phase_status(progress.phase));
            view["question"] = json!(progress.question);
            // A queued build slot is shown as its own state, never as a stall.
            view["waiting_for_slot"] = json!(progress.waiting_for_slot);
            // A command in flight keeps the worker out of the stall detector;
            // publish the mark only while there is one.
            if let Some(started) = progress.command_started_at {
                view["command_started_at"] = json!(started);
            }
            // A command that outlived its budget keeps running as a background
            // job the worker waits on, so it belongs in the status view.
            if !progress.jobs.is_empty() {
                view["jobs"] = json!(
                    progress
                        .jobs
                        .iter()
                        .map(|job| job.label())
                        .collect::<Vec<_>>()
                );
            }
            // list_workers supplies a summary without cloning the multi-megabyte diff.
            if progress.phase != WorkerPhase::Running {
                let details = &row["state"];
                for key in [
                    "summary", "verified", "branch", "revision", "metrics", "error", "report",
                    "per_file",
                ] {
                    if let Some(value) = details.get(key) {
                        view[key] = value
                            .as_str()
                            .map_or_else(|| value.clone(), |text| json!(clamp_string(text, 1500)));
                    }
                }
                // The pure state projection remains shared with tests; paused
                // progress carries exactly the fields the projection needs.
                if progress.phase == WorkerPhase::Paused {
                    enrich_state(
                        view,
                        &WorkerState::Paused {
                            question: view["question"].as_str().unwrap_or("").to_string(),
                            step: progress.step,
                            paused_at: now,
                        },
                    );
                }
            }
        }
        if let Some(logs) = pool.get_worker_logs(id).await {
            view["last_ops"] = json!(
                logs.tail(5)
                    .iter()
                    .map(|log| clamp_string(&log.command, 256))
                    .collect::<Vec<_>>()
            );
            attach_verify_tail(view, &logs);
        }
    }
    for (id, view) in &mut views {
        if pool.question_for_consolidator(id) {
            view["question_for_consolidator"] = json!(true);
        }
        if pool.steered_by_live_consolidator(id).await {
            view["steered_by_consolidator"] = json!(true);
        }
    }
    views
}

impl EventRouter {
    fn history(&mut self, owner: &str) -> &mut WatchHistory {
        if !self.watch_history.contains_key(owner)
            && self.watch_history.len() >= 1024
            && let Some(oldest) = self
                .watch_history
                .iter()
                .min_by_key(|(_, h)| h.last_sequence)
                .map(|(k, _)| k.clone())
        {
            self.watch_history.remove(&oldest);
        }
        self.watch_history.entry(owner.to_string()).or_default()
    }

    fn observe_watch(&mut self, mut views: crate::cli::watch::Snapshot) {
        let now = crate::pool::unix_timestamp();
        for (id, view) in &mut views {
            crate::cli::watch::progress_clock(view, self.watch_current.get(id), now);
            // A resumed worker may ask the same question at the same turn again.
            if self
                .watch_current
                .get(id)
                .is_some_and(|old| old["status"] != view["status"])
            {
                self.watch_reported.remove(id);
                self.seen.remove(id);
            }
            if view["status"] == "paused" && view["question_for_consolidator"] == true {
                self.watch_reported.remove(id);
                for history in self.watch_history.values_mut() {
                    history.pending.retain(|event| {
                        !(event["worker_id"] == *id && event["event"] == "needs_input")
                    });
                }
                continue;
            }
            // A consolidator that steered this worker waits on its stop; every
            // event of the worker -- completed, failed, exhausted, stalled --
            // belongs to that consolidator, not the owner's watch. Skipping one
            // also drops what is queued, so a duplicate never leaks through.
            if view["steered_by_consolidator"] == true {
                self.watch_reported.remove(id);
                self.seen.remove(id);
                for history in self.watch_history.values_mut() {
                    history.pending.retain(|event| event["worker_id"] != *id);
                }
                continue;
            }
            // A terminal worker whose branch is merged into its base or already
            // gone has no work left to report: replaying it after a restart
            // would tell the owner to review work that is already landed.
            if view[BRANCH_GONE_OR_MERGED] == true {
                self.watch_reported.remove(id);
                self.seen.remove(id);
                for history in self.watch_history.values_mut() {
                    history.pending.retain(|event| event["worker_id"] != *id);
                }
                continue;
            }
            if let Some(mut event) =
                crate::cli::watch::select_event(view, self.watch_reported.get(id), now)
            {
                // The owner already acknowledged this exact transition before
                // this daemon started, so a restart must not replay it as a
                // missed event.
                if self.acks.acknowledged(
                    event["owner"].as_str().unwrap_or("unattributed"),
                    id,
                    event["revision"].as_u64().unwrap_or(0),
                    event["event"].as_str().unwrap_or(""),
                ) {
                    continue;
                }
                // One transition is one event, whichever view describes it:
                // the live snapshot and the registry row of the same revision
                // share the key `(worker id, revision, kind)` and must not be
                // delivered twice. A stall is an episode rather than a
                // transition, so it is never keyed this way.
                if event["event"] != "stalled"
                    && self.watch_reported.get(id).is_some_and(|old| {
                        old["event"] == event["event"] && old["revision"] == event["revision"]
                    })
                {
                    continue;
                }
                self.sequence += 1;
                event["sequence"] = json!(self.sequence);
                self.watch_reported.insert(id.clone(), event.clone());
                let owner = event["owner"]
                    .as_str()
                    .unwrap_or("unattributed")
                    .to_string();
                let sequence = self.sequence;
                let history = self.history(&owner);
                if history.pending.len() == 100 {
                    history.pending.pop_front();
                    history.dropped += 1;
                }
                history.last_sequence = sequence;
                history.pending.push_back(event.clone());
                let frame = json!({"jsonrpc":"2.0","method":CHANNEL_METHOD,"params":{
                    "content":crate::cli::watch::render(&event),"payload":event,
                    "meta":{"worker_id":id,"event":event["event"].as_str().unwrap_or(""),"owner":owner}
                }}).to_string() + "\n";
                if event["event"] == "stalled" {
                    self.connections.retain(|_, (agent, admin, tx)| {
                        if *admin || *agent == owner {
                            try_deliver(tx, frame.clone())
                        } else {
                            !tx.is_closed()
                        }
                    });
                }
            }
        }
        self.watch_reported.retain(|id, _| views.contains_key(id));
        self.seen.retain(|id, _| views.contains_key(id));
        self.watch_current = views;
    }

    fn watch_reply(
        &mut self,
        ctx: &super::server::ConnectionContext,
        params: &serde_json::Value,
    ) -> anyhow::Result<serde_json::Value> {
        use std::collections::BTreeSet;
        let ids: BTreeSet<String> = params["worker_ids"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect();
        let initial = params["initial"].as_bool().unwrap_or(false);
        let group = params["group"].as_str();
        let owner = ctx.agent();
        // Unattributed legacy rows are the admin's alone: an agent that happens
        // to be named "unattributed" must not inherit them by accident.
        let allowed = |v: &serde_json::Value| {
            ctx.is_admin() || (v["owner"] == owner && v["owner"] != "unattributed")
        };
        for id in &ids {
            let known = self.watch_current.get(id).or_else(|| {
                self.watch_history
                    .values()
                    .flat_map(|h| &h.pending)
                    .find(|v| v["worker_id"] == *id)
            });
            if let Some(v) = known {
                anyhow::ensure!(
                    allowed(v),
                    "worker {id} belongs to agent {}",
                    v["owner"].as_str().unwrap_or("unattributed")
                );
            } else if initial {
                anyhow::bail!("Worker not found: {id}");
            }
        }
        let watching: BTreeSet<String> = self
            .watch_current
            .values()
            .filter(|v| {
                allowed(v)
                    && crate::cli::watch::matches(v, &ids, group)
                    && matches!(
                        v["status"].as_str(),
                        Some("running" | "paused" | "reviewing")
                    )
            })
            .filter_map(|v| v["worker_id"].as_str().map(str::to_string))
            .collect();
        let mut events = Vec::new();
        for (agent, history) in &self.watch_history {
            if !ctx.is_admin() && *agent != owner {
                continue;
            }
            for v in &history.pending {
                if !crate::cli::watch::matches(v, &ids, group) {
                    continue;
                }
                let mut event = v.clone();
                if v["event"] == "stalled"
                    && let Some(current) = v["worker_id"]
                        .as_str()
                        .and_then(|id| self.watch_current.get(id))
                {
                    if !matches!(current["status"].as_str(), Some("running" | "reviewing")) {
                        continue;
                    }
                    let now = crate::pool::unix_timestamp();
                    for (key, value) in current.as_object().into_iter().flatten() {
                        event[key] = value.clone();
                    }
                    event["time_since_last_step"] =
                        json!(now.saturating_sub(current["last_step_at"].as_u64().unwrap_or(now)));
                    event["commands"] = json!(crate::cli::watch::commands(&event));
                }
                event["missed"] = json!(initial);
                event["dropped_events"] = json!(history.dropped);
                events.push(event);
            }
        }
        events.sort_by_key(|v| v["sequence"].as_u64());
        // At most one watch per identity: a hub poll with anything to watch
        // claims the caller's slot, refusing a second connection. An MCP action
        // already holds the slot through its guard, so its own claim is a no-op.
        if !watching.is_empty() || !events.is_empty() {
            self.watches.claim(&owner, ctx.id, ctx.pid)?;
        }
        // An explicit terminal id is reported immediately even if another
        // watch already acknowledged its transition.
        if initial {
            for id in &ids {
                if !events.iter().any(|v| v["worker_id"] == *id)
                    && let Some(v) = self.watch_reported.get(id).filter(|v| {
                        matches!(v["event"].as_str(), Some("completed" | "failed"))
                            && self.seen.get(id).copied() != v["sequence"].as_u64()
                            && allowed(v)
                            && crate::cli::watch::matches(v, &ids, group)
                    })
                {
                    events.push(v.clone());
                }
            }
        }
        Ok(json!({"watching":watching,"events":events}))
    }

    /// Forget `owner`'s queued events for `wid` and mark its last reported
    /// event seen.
    ///
    /// An interaction (status, logs, collect, kill, steer) is the owner
    /// looking at the worker directly, so a later watch must not replay those
    /// events as "while you were not watching".
    pub(super) fn mark_seen(&mut self, owner: &str, wid: &str) {
        if let Some(history) = self.watch_history.get_mut(owner) {
            history.pending.retain(|v| v["worker_id"] != wid);
        }
        if let Some(event) = self.watch_reported.get(wid) {
            if let Some(sequence) = event["sequence"].as_u64() {
                self.seen.insert(wid.to_string(), sequence);
            }
            // An interaction is the owner looking at the worker directly, so
            // persist the position: a restart must not replay it either.
            self.acks.record(
                owner,
                wid,
                event["revision"].as_u64().unwrap_or(0),
                event["event"].as_str().unwrap_or(""),
            );
        }
    }

    fn acknowledge_watch(&mut self, ctx: &super::server::ConnectionContext, sequence: u64) {
        let owner = ctx.agent();
        // Collect first: the pending deque is borrowed mutably below, while the
        // acknowledged positions are recorded on `self` afterwards.
        let mut acknowledged = Vec::new();
        for (agent, history) in &mut self.watch_history {
            if !ctx.is_admin() && *agent != owner {
                continue;
            }
            let mut matched = false;
            history.pending.retain(|v| {
                if v["sequence"].as_u64() != Some(sequence) {
                    return true;
                }
                matched = true;
                acknowledged.push((
                    agent.clone(),
                    v["worker_id"].as_str().unwrap_or("").to_string(),
                    v["revision"].as_u64().unwrap_or(0),
                    v["event"].as_str().unwrap_or("").to_string(),
                ));
                false
            });
            if matched {
                history.cursor = history.cursor.max(sequence);
                history.dropped = 0;
            }
        }
        for (agent, wid, revision, event) in acknowledged {
            self.acks.record(&agent, &wid, revision, &event);
        }
    }
}

pub(super) async fn watch_request(
    pool: &WorkerPool,
    router: &Arc<Mutex<EventRouter>>,
    ctx: &super::server::ConnectionContext,
    mut params: serde_json::Value,
    ack: bool,
) -> anyhow::Result<serde_json::Value> {
    if ack {
        router
            .lock()
            .await
            .acknowledge_watch(ctx, params["sequence"].as_u64().unwrap_or(0));
        return Ok(json!({}));
    }
    // Resolve each id (prefix, or `last`) before the snapshot, so the
    // `hub/watch` route the CLI uses accepts them exactly like the MCP
    // `watch` action does. The caller's ownership is what scopes the search.
    if params.get("worker_ids").is_some() {
        let mut resolved = Vec::new();
        for needle in params["worker_ids"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|v| v.as_str())
        {
            resolved.push(pool.resolve_worker_id(needle, &ctx.agent()).await?);
        }
        params["worker_ids"] = json!(resolved);
    }
    let snapshot = watch_snapshot(pool).await;
    let mut guard = router.lock().await;
    guard.observe_watch(snapshot);
    guard.watch_reply(ctx, &params)
}

#[cfg(test)]
mod watch_stall_regression_tests {
    use super::*;

    fn view(step: usize) -> serde_json::Value {
        json!({"worker_id":"stall-probe", "owner":"owner", "status":"running",
            "step":step, "revision":0, "last_step_at":crate::pool::unix_timestamp().saturating_sub(601),
            "metrics":crate::pool::WorkerMetrics::default()})
    }

    #[test]
    fn acknowledged_stall_is_not_replayed_by_an_explicit_initial_watch() {
        let mut router = EventRouter::default();
        router.observe_watch([("stall-probe".into(), view(160))].into());
        let mut ctx = super::super::server::ConnectionContext::hub_connection(1);
        ctx.agent_id = Some("owner".into());
        let params = json!({"worker_ids":["stall-probe"], "initial":true});
        let first = router.watch_reply(&ctx, &params).unwrap();
        assert_eq!(first["events"][0]["event"], "stalled");
        let sequence = first["events"][0]["sequence"].as_u64().unwrap();
        router.acknowledge_watch(&ctx, sequence);
        assert_eq!(router.watch_history["owner"].cursor, sequence);
        let second = router.watch_reply(&ctx, &params).unwrap();
        assert!(second["events"].as_array().unwrap().is_empty(), "{second}");
    }

    #[test]
    fn pending_stall_renders_the_current_step_instead_of_its_queued_snapshot() {
        let mut router = EventRouter::default();
        router.observe_watch([("stall-probe".into(), view(160))].into());
        router.watch_current.insert("stall-probe".into(), view(162));
        let mut ctx = super::super::server::ConnectionContext::hub_connection(1);
        ctx.agent_id = Some("owner".into());
        let reply = router
            .watch_reply(&ctx, &json!({"worker_ids":["stall-probe"], "initial":true}))
            .unwrap();
        assert_eq!(reply["events"][0]["step"], 162);
    }
}

#[cfg(test)]
mod mark_seen_tests {
    use super::*;

    fn completed(id: &str) -> serde_json::Value {
        json!({"worker_id":id, "owner":"owner", "status":"completed", "step":2,
            "revision":0, "verified":true, "branch":format!("worker-{id}"),
            "metrics":crate::pool::WorkerMetrics::default()})
    }

    /// An interactive verb is the owner looking at the worker, so its queued
    /// event must not come back as "while you were not watching".
    #[test]
    fn an_interaction_drops_the_pending_event_and_its_initial_replay() {
        let mut router = EventRouter::default();
        router.observe_watch([("w".to_string(), completed("w"))].into());
        let mut ctx = super::super::server::ConnectionContext::hub_connection(1);
        ctx.agent_id = Some("owner".into());
        let params = json!({"worker_ids":[], "initial":true});
        let first = router.watch_reply(&ctx, &params).unwrap();
        assert_eq!(first["events"][0]["event"], "completed", "{first}");
        assert_eq!(
            router.watch_history["owner"].pending.len(),
            1,
            "no ack, so the event is still queued"
        );

        router.mark_seen("owner", "w");
        assert!(router.watch_history["owner"].pending.is_empty());
        let second = router.watch_reply(&ctx, &params).unwrap();
        assert!(second["events"].as_array().unwrap().is_empty(), "{second}");
        // The explicit-id path replays the last reported terminal event; it
        // must respect the seen mark too.
        let explicit = router
            .watch_reply(&ctx, &json!({"worker_ids":["w"], "initial":true}))
            .unwrap();
        assert!(
            explicit["events"].as_array().unwrap().is_empty(),
            "{explicit}"
        );
    }
}

#[cfg(test)]
mod verify_tail_tests {
    use super::verify_tail;

    /// A verify run's noise: one line per passing test.
    fn passing(count: usize) -> String {
        (0..count)
            .map(|i| format!("test suite::case_{i} ... ok"))
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn a_huge_passing_tail_is_cut_to_forty_lines_and_four_kib() {
        let output = passing(400);
        let tail = verify_tail(&output);
        assert!(tail.lines().count() <= 40, "{} lines", tail.lines().count());
        assert!(tail.len() <= 4096, "{} bytes", tail.len());
        assert!(
            tail.contains("case_399"),
            "the newest lines are kept: {tail}"
        );
    }

    #[test]
    fn failure_lines_survive_a_wall_of_passing_noise() {
        let output = format!(
            "{}\n---- verify stdout ----\nthread 'main' panicked at src/lib.rs:1\nassertion failed\n{}",
            passing(200),
            passing(200)
        );
        let tail = verify_tail(&output);
        assert!(tail.lines().count() <= 40, "{} lines", tail.lines().count());
        assert!(tail.len() <= 4096, "{} bytes", tail.len());
        assert!(
            tail.contains("panicked") && tail.contains("assertion failed"),
            "{tail}"
        );
    }

    #[test]
    fn a_failure_marker_inside_the_window_is_kept() {
        let output = format!("{}\nFAILED test suite::boom\n{}", passing(20), passing(20));
        let tail = verify_tail(&output);
        assert!(tail.contains("FAILED test suite::boom"), "{tail}");
        assert!(tail.lines().count() <= 40, "{} lines", tail.lines().count());
    }

    #[test]
    fn a_giant_marker_line_keeps_its_tail_within_four_kib() {
        let output = format!("error: {}\n{}", "x".repeat(9000), passing(50));
        let tail = verify_tail(&output);
        assert!(tail.len() <= 4096, "{} bytes", tail.len());
        assert!(
            tail.contains("xxxx"),
            "the marked line survives: {} bytes",
            tail.len()
        );
        assert!(
            !tail.contains("case_0"),
            "passing noise gives up the budget first"
        );
    }
}

/// Once the completion gate passes, an earlier failing `[verify]` run must not
/// resurface as a current failure in the watch payload.
#[cfg(test)]
mod verify_tail_attachment_tests {
    use super::attach_verify_tail;
    use crate::agent::AgentStepLog;
    use crate::pool::LogBuffer;
    use serde_json::json;

    fn logs_with_failed_verify() -> LogBuffer {
        let mut logs = LogBuffer::new();
        logs.push(AgentStepLog {
            step: 1,
            command: "[verify] cargo test".to_string(),
            output: "Command timed out after 600s and was terminated.".to_string(),
            exit_code: Some(1),
        });
        logs
    }

    fn completed_view(verified: bool) -> serde_json::Value {
        json!({"worker_id":"w", "event":"completed", "status":"completed",
            "verified":verified, "metrics":{"verify_failures":1}})
    }

    #[test]
    fn a_completed_worker_hides_a_stale_failing_verify_tail_after_the_gate_passes() {
        let mut view = completed_view(true);
        attach_verify_tail(&mut view, &logs_with_failed_verify());
        assert!(
            view.get("verify_output_tail").is_none(),
            "a verified worker must not carry a stale verify tail: {view}"
        );
        assert_eq!(
            view["metrics"]["verify_failures"], 1,
            "the failure counter stays as is"
        );
    }

    #[test]
    fn an_unverified_worker_carries_the_failing_verify_tail() {
        let mut view = completed_view(false);
        attach_verify_tail(&mut view, &logs_with_failed_verify());
        let tail = view["verify_output_tail"].as_str().unwrap_or_default();
        assert!(tail.contains("Command timed out after 600s"), "{view}");
    }
}

/// A registry row reduced to a view must carry the completion's persisted
/// verdict, so an evicted or restarted worker's event still reports it.
#[cfg(test)]
mod registry_verified_tests {
    use super::{EventKind, registry_view};
    use crate::pool::{RegistryStatus, WorkerMeta, WorkerRegistryEntry};

    fn completed_row(verified: Option<bool>) -> WorkerRegistryEntry {
        // Built through the same constructor a real write uses, so the fixture
        // picks up fields the current build adds to a row without a literal.
        let meta = WorkerMeta {
            task: "persist the verdict".to_string(),
            pid: 0,
            verified,
            ..WorkerMeta::test_meta("w-registry", "owner")
        };
        meta.entry(
            "test-model",
            RegistryStatus::Completed,
            5,
            10,
            "all gates green",
            None,
        )
    }

    #[test]
    fn a_completed_rows_view_carries_the_persisted_verdict() {
        let view = registry_view(&completed_row(Some(true)));
        assert_eq!(view.event, Some(EventKind::Completed));
        assert_eq!(view.outcome.verified, Some(true));
    }

    #[test]
    fn a_row_without_a_verdict_stays_unknown() {
        assert_eq!(registry_view(&completed_row(None)).outcome.verified, None);
    }
}
#[cfg(test)]
mod event_dedup_tests;
