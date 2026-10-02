//! A deterministic model-based (state-machine) test of the hub's event
//! delivery.
//!
//! Every point fix in [`super`] shipped with a narrow test: dedupe by
//! `(worker, revision, kind)` (E16), steered workers routed to their live
//! consolidator (E14/E22), per-owner acks persisted across a daemon restart
//! (E23), retired workers tombstoned (E28/F7), the `--all` round watch (E24/E30)
//! and missed-event replay. This module drives all of them together through the
//! same [`EventRouter`] a live daemon uses, over a seeded random walk, and
//! checks the invariants that must hold after *every* step.
//!
//! The harness keeps a small reference model of what each owner must receive:
//! one *episode* per terminal transition, keyed by `(worker, revision, kind)`
//! and carrying the state the router is expected to be in. The router's own
//! pending queue is compared to the model's pending set after every observation,
//! so a lost event and a duplicate event are both caught, not only the ones a
//! hand-written scenario would name.
//!
//! Invariants checked after every step:
//! (a) an owner never receives another owner's events;
//! (b) a `(worker, revision, kind)` is never delivered again once acknowledged,
//!     across restarts too;
//! (c) a retired worker is never delivered again;
//! (d) while a live consolidator steers a worker, its terminal event reaches no
//!     owner watch, and it is delivered once the consolidator stops;
//! (e) an `--all` watch yields at most one round event, and only when every
//!     selected worker stopped (or earlier for needs_input/failed);
//! (f) no event is lost: every terminal transition not acked and not suppressed
//!     by (c)/(d) is eventually delivered.
//!
//! On a violation the seed and a minimal step trace are printed.

use super::*;
use crate::mcp::server::ConnectionContext;
use crate::pool::{RegistryStatus, WorkerMetrics, WorkerRegistryEntry};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

/// Seeds the walk is repeated over. Each seed is an independent world.
const SEEDS: u64 = 300;
/// Steps per seed. The whole suite must stay well under 20 s.
const STEPS: usize = 200;

// ---------------------------------------------------------------------------
// Seeded PRNG (splitmix64): no new dependency, deterministic across platforms.
// ---------------------------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x1234_5678_9ABC_DEF0)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        assert!(n > 0, "an empty choice set has no element");
        (self.next_u64() % n as u64) as usize
    }

    fn chance(&mut self, percent: u64) -> bool {
        self.next_u64() % 100 < percent
    }
}

// ---------------------------------------------------------------------------
// The world: owners, groups, workers and the consolidators that steer them.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Status {
    Running,
    Paused,
    Completed,
    Failed,
    Exhausted,
}

impl Status {
    fn name(self) -> &'static str {
        match self {
            Status::Running => "running",
            Status::Paused => "paused",
            Status::Completed => "completed",
            Status::Failed => "failed",
            Status::Exhausted => "exhausted",
        }
    }

    /// The watch event a view of this status produces, or `None` while live.
    fn kind(self) -> Option<&'static str> {
        match self {
            Status::Running => None,
            Status::Paused => Some("needs_input"),
            Status::Completed => Some("completed"),
            Status::Failed => Some("failed"),
            Status::Exhausted => Some("exhausted"),
        }
    }

    fn registry(self) -> RegistryStatus {
        match self {
            Status::Running => RegistryStatus::Running,
            Status::Paused => RegistryStatus::Paused,
            Status::Completed => RegistryStatus::Completed,
            Status::Failed => RegistryStatus::Failed,
            Status::Exhausted => RegistryStatus::Exhausted,
        }
    }
}

#[derive(Clone)]
struct Worker {
    id: String,
    owner: String,
    group: String,
    /// Bumped on every terminal transition, so each episode is unique.
    revision: usize,
    status: Status,
    /// Index of the consolidator that steered this worker, if any.
    steered_by: Option<usize>,
    /// The question of a paused worker was handed to a consolidator.
    question_to_consolidator: bool,
    /// The branch is merged or gone: the event must never be replayed.
    branch_gone: bool,
    retired: bool,
    step: usize,
}

#[derive(Clone)]
struct Consolidator {
    live: bool,
}

struct World {
    owners: Vec<String>,
    groups: Vec<String>,
    workers: Vec<Worker>,
    consolidators: Vec<Consolidator>,
}

impl World {
    fn new(rng: &mut Rng) -> Self {
        let owner_count = 2 + rng.below(2); // 2..=3 owners
        let owners: Vec<String> = (0..owner_count).map(|i| format!("owner-{i}")).collect();
        let group_count = 1 + rng.below(2); // 1..=2 groups
        let groups: Vec<String> = (0..group_count).map(|i| format!("g{i}")).collect();
        let consolidators = vec![Consolidator { live: false }, Consolidator { live: false }];
        let mut workers = Vec::new();
        for owner in &owners {
            let count = 2 + rng.below(2); // 2..=3 workers per owner
            for j in 0..count {
                let group = groups[rng.below(groups.len())].clone();
                workers.push(Worker {
                    id: format!("{owner}-w{j}"),
                    owner: owner.clone(),
                    group,
                    revision: 0,
                    status: Status::Running,
                    steered_by: None,
                    question_to_consolidator: false,
                    branch_gone: false,
                    retired: false,
                    step: 0,
                });
            }
        }
        World {
            owners,
            groups,
            workers,
            consolidators,
        }
    }

    fn steered_live(&self, w: &Worker) -> bool {
        w.steered_by
            .is_some_and(|c| self.consolidators[c].live)
    }

    fn question_for_consolidator(&self, w: &Worker) -> bool {
        w.question_to_consolidator && w.status == Status::Paused
    }

    fn live_workers(&self) -> Vec<usize> {
        (0..self.workers.len())
            .filter(|&i| !self.workers[i].retired)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// The reference model: one episode per terminal transition.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum EpState {
    /// Deliverable and not yet acknowledged: the owner's queue holds it.
    Pending,
    /// Suppressed because a live consolidator steers the worker.
    Steered,
    /// Suppressed because the question went to a consolidator.
    Question,
    /// Permanently suppressed (retired, or the branch is gone).
    Suppressed,
    /// Delivered and acknowledged: never delivered again.
    Acked,
}

#[derive(Clone)]
struct Episode {
    owner: String,
    worker: String,
    revision: usize,
    kind: String,
    state: EpState,
    /// Whether the owner ever received it (liveness, invariant f).
    delivered: bool,
}

#[derive(Default)]
struct Model {
    /// Keyed by `(worker, revision, kind)`; a revision is unique per worker.
    episodes: BTreeMap<(String, usize, String), Episode>,
    /// The last transition the router reported for a worker, mirroring
    /// `EventRouter::watch_reported` so `mark_seen` can be modelled.
    reported: BTreeMap<String, (usize, String)>,
}

impl Model {
    fn add_episode(&mut self, w: &Worker) {
        let Some(kind) = w.status.kind() else {
            return;
        };
        self.episodes.insert(
            (w.id.clone(), w.revision, kind.to_string()),
            Episode {
                owner: w.owner.clone(),
                worker: w.id.clone(),
                revision: w.revision,
                kind: kind.to_string(),
                state: EpState::Pending,
                delivered: false,
            },
        );
    }

    /// Recompute every episode's state from the world, keeping acknowledgments.
    ///
    /// The router queues a transition once it observes it and keeps it queued
    /// until it is acknowledged or removed by a suppression rule; a worker that
    /// merely resumes does not drop the completion its owner has not read. The
    /// model mirrors that: an episode removed by steering or by a question
    /// routed to a consolidator returns to the queue only while the view still
    /// shows the same transition, while a queued one survives a resume.
    fn reconcile(&mut self, world: &World) {
        for w in &world.workers {
            let current_kind = w.status.kind();
            for ((wid, rev, kind), ep) in self.episodes.iter_mut() {
                if wid != &w.id || ep.state == EpState::Acked {
                    continue;
                }
                if w.retired || w.branch_gone {
                    ep.state = EpState::Suppressed;
                    continue;
                }
                if world.steered_live(w) {
                    ep.state = EpState::Steered;
                    continue;
                }
                if world.question_for_consolidator(w) && kind == "needs_input" {
                    ep.state = EpState::Question;
                    continue;
                }
                let current = w.revision == *rev && current_kind == Some(kind.as_str());
                ep.state = if current || ep.state == EpState::Pending {
                    EpState::Pending
                } else {
                    EpState::Suppressed
                };
            }
        }
        self.reported.clear();
        for w in &world.workers {
            let Some(kind) = w.status.kind() else {
                continue;
            };
            if let Some(ep) = self.episodes.get(&(w.id.clone(), w.revision, kind.to_string()))
                && matches!(ep.state, EpState::Pending | EpState::Acked)
            {
                self.reported
                    .insert(w.id.clone(), (w.revision, kind.to_string()));
            }
        }
    }

    fn pending(&self) -> BTreeSet<(String, String, usize, String)> {
        self.episodes
            .values()
            .filter(|ep| ep.state == EpState::Pending)
            .map(|ep| {
                (
                    ep.owner.clone(),
                    ep.worker.clone(),
                    ep.revision,
                    ep.kind.clone(),
                )
            })
            .collect()
    }

    fn ack(&mut self, worker: &str, revision: usize, kind: &str) {
        if let Some(ep) = self.episodes.get_mut(&(worker.to_string(), revision, kind.to_string())) {
            ep.state = EpState::Acked;
        }
    }

    /// A restart rebuilds the queue from the current views alone: a transition
    /// no view describes any more cannot be re-derived, so it is gone.
    fn on_restart(&mut self, world: &World) {
        for ((wid, rev, kind), ep) in self.episodes.iter_mut() {
            if ep.state != EpState::Pending {
                continue;
            }
            let current = world.workers.iter().find(|w| &w.id == wid).is_some_and(|w| {
                !w.retired
                    && !w.branch_gone
                    && w.revision == *rev
                    && w.status.kind() == Some(kind.as_str())
            });
            if !current {
                ep.state = EpState::Suppressed;
            }
        }
    }

    /// Model [`EventRouter::mark_seen`]: drop the worker's queue and persist the
    /// position of its last reported transition.
    fn mark_seen(&mut self, worker: &str) {
        let reported = self.reported.get(worker).cloned();
        for ((wid, _, _), ep) in self.episodes.iter_mut() {
            if wid == worker && ep.state == EpState::Pending {
                ep.state = EpState::Acked;
            }
        }
        if let Some((revision, kind)) = reported {
            self.ack(worker, revision, &kind);
        }
    }
}

// ---------------------------------------------------------------------------
// Views: the live pool record and the registry row are two sources of the same
// state, and the router must treat them as one transition.
// ---------------------------------------------------------------------------

fn live_view(w: &Worker, world: &World, now: u64) -> Value {
    let kind = w.status.kind();
    json!({
        "worker_id": w.id,
        "owner": w.owner,
        "group": w.group,
        "model": "test",
        "status": w.status.name(),
        "step": w.step,
        "turns": w.step,
        "revision": w.revision,
        "branch": format!("worker-{}", w.id),
        "question": if w.status == Status::Paused { json!("why?") } else { Value::Null },
        "last_step_at": now,
        "metrics": WorkerMetrics::default(),
        "verified": w.status == Status::Completed,
        "summary": if kind.is_some() { json!("Done.") } else { Value::Null },
        "error": if w.status == Status::Failed { json!("boom") } else { Value::Null },
        "steered_by_consolidator": world.steered_live(w),
        "question_for_consolidator": world.question_for_consolidator(w),
        "branch_gone_or_merged": w.branch_gone,
    })
}

/// The registry row the same state leaves behind: a second source that names
/// the same `(worker, revision, kind)` and must not be a second delivery.
fn row_view(w: &Worker, world: &World, now: u64) -> Value {
    let entry: WorkerRegistryEntry = serde_json::from_value(json!({
        "id": w.id,
        "pid": std::process::id(),
        "task": "model walk",
        "model": "test",
        "status": w.status.registry(),
        "step": w.step,
        "max_turns": 100,
        "last_command": "done",
        "question": if w.status == Status::Paused { json!("why?") } else { Value::Null },
        "started_at": 0,
        "updated_at": now,
        "revision": w.revision,
        "owner": w.owner,
        "group": w.group,
        "verified": w.status == Status::Completed,
        "metrics": {"diff_files": 1, "diff_insertions": 1, "diff_deletions": 0},
    }))
    .expect("a registry row round-trips");
    let mut view = crate::cli::watch::registry_snapshot(&entry, now);
    view["steered_by_consolidator"] = json!(world.steered_live(w));
    view["question_for_consolidator"] = json!(world.question_for_consolidator(w));
    view["branch_gone_or_merged"] = json!(w.branch_gone);
    view
}

// ---------------------------------------------------------------------------
// The harness: the router, the reference model and the scratch hub dir.
// ---------------------------------------------------------------------------

/// Removes the per-seed scratch directory when the harness is dropped, so a
/// failing assertion still cleans up.
struct Cleanup(PathBuf);

impl Drop for Cleanup {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

struct Harness {
    seed: u64,
    world: World,
    model: Model,
    router: EventRouter,
    dir: PathBuf,
    _cleanup: Cleanup,
    ctxs: BTreeMap<String, ConnectionContext>,
    trace: Vec<String>,
    /// `(owner, worker, revision, kind)` already acknowledged.
    acked: BTreeSet<(String, String, usize, String)>,
    /// A stale view of a just-retired worker, fed once to test the tombstone.
    stale: Vec<Value>,
}

impl Harness {
    fn new(seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let world = World::new(&mut rng);
        let dir = std::env::temp_dir().join(format!(
            "mcp-events-model-{}-{seed}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("scratch hub dir");
        let mut router = EventRouter::default();
        router.load_ack_store(&dir);
        let mut ctxs = BTreeMap::new();
        for (i, owner) in world.owners.iter().enumerate() {
            let mut ctx = ConnectionContext::hub_connection(i as u64 + 1);
            ctx.agent_id = Some(owner.clone());
            ctxs.insert(owner.clone(), ctx);
        }
        Harness {
            seed,
            world,
            model: Model::default(),
            router,
            dir: dir.clone(),
            _cleanup: Cleanup(dir),
            ctxs,
            trace: Vec::new(),
            acked: BTreeSet::new(),
            stale: Vec::new(),
        }
    }

    fn trace_detail(&mut self, line: String) {
        if let Some(last) = self.trace.last_mut() {
            last.push_str("\n");
            last.push_str(&line);
        }
    }

    fn violation(&self, msg: String, step: usize) -> ! {
        let start = self.trace.len().saturating_sub(12);
        let trace = self.trace[start..].join("\n");
        let mut world_dump = String::new();
        for w in &self.world.workers {
            world_dump.push_str(&format!(
                "\n  {} owner={} group={} status={} rev={} steered={:?} live_cons={:?} q2c={} gone={} retired={} step={}",
                w.id,
                w.owner,
                w.group,
                w.status.name(),
                w.revision,
                w.steered_by,
                w.steered_by
                    .map(|c| self.world.consolidators[c].live),
                w.question_to_consolidator,
                w.branch_gone,
                w.retired,
                w.step,
            ));
        }
        let mut eps = String::new();
        for ((wid, rev, kind), ep) in &self.model.episodes {
            eps.push_str(&format!("\n  {wid} rev {rev} {kind}: {:?} delivered={}", ep.state, ep.delivered));
        }
        panic!(
            "model violation at seed {} step {step}: {msg}\nrecent trace:\n{trace}\nworld:{world_dump}\nmodel:{eps}\nrouter reported: {:?}\nrouter seen: {:?}\nrouter acks: {:?}",
            self.seed,
            self.router.watch_reported.iter().map(|(k, v)| (k.clone(), v["event"].clone(), v["revision"].clone(), v["sequence"].clone())).collect::<Vec<_>>(),
            self.router.seen,
            self.router.acks.positions.iter().map(|(o, ws)| (o.clone(), ws.iter().map(|(w, e)| (w.clone(), e.position.revision, e.position.event.clone())).collect::<Vec<_>>())).collect::<Vec<_>>(),
        );
    }

    // -- world mutations ----------------------------------------------------

    fn mutate_status(&mut self, rng: &mut Rng) {
        let running: Vec<usize> = self
            .world
            .live_workers()
            .into_iter()
            .filter(|&i| self.world.workers[i].status == Status::Running)
            .collect();
        if running.is_empty() {
            return;
        }
        let idx = running[rng.below(running.len())];
        let status = match rng.below(4) {
            0 => Status::Completed,
            1 => Status::Failed,
            2 => Status::Paused,
            _ => Status::Exhausted,
        };
        let id = self.world.workers[idx].id.clone();
        let rev = self.world.workers[idx].revision;
        self.transition(idx, status);
        self.trace_detail(format!("  {id} -> {} rev {rev}", status.name()));
    }

    fn transition(&mut self, idx: usize, status: Status) {
        let w = &mut self.world.workers[idx];
        w.revision += 1;
        w.status = status;
        w.step += 1;
        let w = w.clone();
        self.model.add_episode(&w);
    }

    fn mutate_steer(&mut self, rng: &mut Rng) {
        let live_cons: Vec<usize> = (0..self.world.consolidators.len())
            .filter(|&c| self.world.consolidators[c].live)
            .collect();
        if live_cons.is_empty() {
            return;
        }
        let candidates = self.world.live_workers();
        if candidates.is_empty() {
            return;
        }
        let idx = candidates[rng.below(candidates.len())];
        let cons = live_cons[rng.below(live_cons.len())];
        let w = &mut self.world.workers[idx];
        // Steering a stopped worker resumes it; steering a running one only
        // re-points it at a live consolidator. The revision moves when the
        // run finishes, not when it is steered.
        let id = w.id.clone();
        let was = w.status.name();
        w.status = Status::Running;
        w.steered_by = Some(cons);
        w.step += 1;
        self.trace_detail(format!("  {id} {was} -> running steered_by={cons}"));
    }

    fn mutate_consolidator(&mut self, rng: &mut Rng) {
        let c = rng.below(self.world.consolidators.len());
        self.world.consolidators[c].live = !self.world.consolidators[c].live;
        let live = self.world.consolidators[c].live;
        self.trace_detail(format!("  consolidator {c} live={live}"));
    }

    fn mutate_retire(&mut self, rng: &mut Rng) {
        let live = self.world.live_workers();
        if live.is_empty() {
            return;
        }
        let idx = live[rng.below(live.len())];
        let now = crate::pool::unix_timestamp();
        let view = live_view(&self.world.workers[idx], &self.world, now);
        self.world.workers[idx].retired = true;
        let id = self.world.workers[idx].id.clone();
        self.router.forget_worker(&id);
        self.stale.push(view);
        self.trace_detail(format!("  retired {id}"));
    }

    fn mutate_branch_gone(&mut self, rng: &mut Rng) {
        let live = self.world.live_workers();
        if live.is_empty() {
            return;
        }
        let idx = live[rng.below(live.len())];
        self.world.workers[idx].branch_gone = true;
        let id = self.world.workers[idx].id.clone();
        self.trace_detail(format!("  branch gone {id}"));
    }

    /// Resume a stopped worker without a consolidator, the way a plain
    /// `steer` after a completion does: the revision is unchanged until the
    /// worker finishes again.
    fn mutate_resume(&mut self, rng: &mut Rng) {
        let stopped: Vec<usize> = self
            .world
            .live_workers()
            .into_iter()
            .filter(|&i| self.world.workers[i].status != Status::Running)
            .collect();
        if stopped.is_empty() {
            return;
        }
        let idx = stopped[rng.below(stopped.len())];
        let id = self.world.workers[idx].id.clone();
        let owner = self.world.workers[idx].owner.clone();
        let was = self.world.workers[idx].status.name();
        // A production `steer` marks the worker seen before it resumes, so the
        // queued terminal event is acknowledged first.
        let reported = self.model.reported.get(&id).cloned();
        self.model.mark_seen(&id);
        if let Some((revision, kind)) = reported {
            self.acked.insert((owner.clone(), id.clone(), revision, kind));
        }
        self.router.mark_seen(&owner, &id);
        let w = &mut self.world.workers[idx];
        w.status = Status::Running;
        w.steered_by = None;
        w.question_to_consolidator = false;
        w.step += 1;
        self.trace_detail(format!("  resume {id} {was} -> running"));
    }

    fn mutate_question(&mut self, rng: &mut Rng) {
        let paused: Vec<usize> = self
            .world
            .live_workers()
            .into_iter()
            .filter(|&i| self.world.workers[i].status == Status::Paused)
            .collect();
        if paused.is_empty() {
            return;
        }
        let idx = paused[rng.below(paused.len())];
        let w = &mut self.world.workers[idx];
        w.question_to_consolidator = !w.question_to_consolidator;
        let id = w.id.clone();
        let q = w.question_to_consolidator;
        self.trace_detail(format!("  question_to_consolidator {id}={q}"));
    }

    // -- observation --------------------------------------------------------

    fn build_snapshot(&self, now: u64, use_row: bool) -> crate::cli::watch::Snapshot {
        let mut snap = crate::cli::watch::Snapshot::new();
        for w in &self.world.workers {
            if w.retired {
                continue;
            }
            let view = if use_row {
                row_view(w, &self.world, now)
            } else {
                live_view(w, &self.world, now)
            };
            snap.insert(w.id.clone(), view);
        }
        for view in &self.stale {
            if let Some(id) = view["worker_id"].as_str() {
                snap.insert(id.to_string(), view.clone());
            }
        }
        snap
    }

    fn observe(&mut self, step: usize, use_row: bool) {
        let now = crate::pool::unix_timestamp();
        let snap = self.build_snapshot(now, use_row);
        self.router.observe_watch(snap);
        self.stale.clear();
        let got = router_pending(&self.router);
        let want = self.model.pending();
        if got != want {
            self.violation(
                format!("pending queue mismatch\n  router: {got:?}\n  model:  {want:?}"),
                step,
            );
        }
    }

    // -- watches ------------------------------------------------------------

    fn episode_matches(
        &self,
        worker: &str,
        ids: &BTreeSet<String>,
        groups: &BTreeSet<String>,
    ) -> bool {
        if !ids.is_empty() && !ids.contains(worker) {
            return false;
        }
        if !groups.is_empty() {
            let group = self
                .world
                .workers
                .iter()
                .find(|w| w.id == worker)
                .map(|w| w.group.as_str())
                .unwrap_or("");
            if !groups.contains(group) {
                return false;
            }
        }
        true
    }

    fn plain_watch(
        &mut self,
        owner: &str,
        ids: &BTreeSet<String>,
        groups: &BTreeSet<String>,
        ack: bool,
        step: usize,
    ) {
        let ctx = self.ctxs[owner].clone();
        let params = json!({
            "worker_ids": ids.iter().cloned().collect::<Vec<_>>(),
            "group": groups.iter().cloned().collect::<Vec<_>>(),
            "initial": false,
        });
        let reply = self
            .router
            .watch_reply(&ctx, &params)
            .expect("a plain watch answers");
        let events = reply["events"].as_array().cloned().unwrap_or_default();
        let expected: BTreeSet<(String, String, usize, String)> = self
            .model
            .pending()
            .into_iter()
            .filter(|(o, w, _, _)| o == owner && self.episode_matches(w, ids, groups))
            .collect();
        let got: BTreeSet<(String, String, usize, String)> = events.iter().map(event_key).collect();
        self.trace_detail(format!(
            "  watch {owner} ids={ids:?} groups={groups:?} ack={ack} -> got={got:?}"
        ));
        if expected != got {
            self.violation(
                format!("plain watch mismatch\n  expected: {expected:?}\n  got:      {got:?}"),
                step,
            );
        }
        for view in &events {
            self.record_delivery(owner, view, step);
            if ack {
                let sequence = view["sequence"].as_u64().expect("a sequence");
                self.router.acknowledge_watch(&ctx, sequence);
                let key = event_key(view);
                self.model.ack(&key.1, key.2, &key.3);
                self.acked.insert(key);
            }
        }
    }

    fn round_watch(&mut self, owner: &str, group: &str, step: usize) {
        let ctx = self.ctxs[owner].clone();
        let params = json!({
            "worker_ids": [],
            "group": group,
            "initial": false,
            "all": true,
        });
        let reply = self
            .router
            .watch_reply(&ctx, &params)
            .expect("a round watch answers");
        let events = reply["events"].as_array().cloned().unwrap_or_default();
        self.trace_detail(format!(
            "  round {owner} group={group} -> {} events",
            events.len()
        ));
        if events.len() > 1 {
            self.violation(
                format!("a round watch returned {} events", events.len()),
                step,
            );
        }
        let Some(event) = events.first() else {
            return;
        };
        self.check_round(event, owner, group, step);
        let workers: Vec<String> = event["workers"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|w| w["worker_id"].as_str().map(str::to_string))
            .collect();
        for wid in workers {
            let reported = self.model.reported.get(&wid).cloned();
            self.model.mark_seen(&wid);
            if let Some((revision, kind)) = reported {
                self.acked
                    .insert((owner.to_string(), wid.clone(), revision, kind));
            }
        }
    }

    fn check_round(&self, event: &Value, owner: &str, group: &str, step: usize) {
        let status = event["status"].as_str().unwrap_or("");
        let workers = event["workers"].as_array().cloned().unwrap_or_default();
        if workers.is_empty() {
            self.violation("a round event named no workers".to_string(), step);
        }
        for view in &workers {
            let wid = view["worker_id"].as_str().unwrap_or("");
            let Some(worker) = self.world.workers.iter().find(|w| w.id == wid) else {
                self.violation(format!("a round event named unknown worker {wid}"), step);
            };
            if worker.owner != owner {
                self.violation(format!("a round event leaked worker {wid} of {}", worker.owner), step);
            }
            if worker.group != group {
                self.violation(format!("a round event leaked group {} into {group}", worker.group), step);
            }
            if self.world.steered_live(worker) {
                self.violation(format!("a round event named steered worker {wid}"), step);
            }
            if self.world.question_for_consolidator(worker) {
                self.violation(format!("a round event named a consolidator-owned worker {wid}"), step);
            }
            let outcome = view["outcome"].as_str().unwrap_or("");
            if status == "stopped" && matches!(outcome, "running" | "needs_input" | "stalled") {
                self.violation(
                    format!("a round reported stopped while {wid} is {outcome}"),
                    step,
                );
            }
        }
        if status == "attention"
            && !workers.iter().any(|view| {
                matches!(
                    view["outcome"].as_str(),
                    Some("needs_input" | "failed" | "stalled")
                )
            })
        {
            self.violation(
                "a round reported attention with no worker needing it".to_string(),
                step,
            );
        }
    }

    fn record_delivery(&mut self, watcher: &str, view: &Value, step: usize) {
        let key = event_key(view);
        let (owner, worker, revision, kind) = key.clone();
        if owner != watcher {
            self.violation(
                format!("owner {watcher} received {owner}'s event {key:?}"),
                step,
            );
        }
        if self.acked.contains(&key) {
            self.violation(format!("re-delivered after ack: {key:?}"), step);
        }
        if let Some(w) = self.world.workers.iter().find(|w| w.id == worker) {
            if w.retired {
                self.violation(format!("a retired worker was delivered: {key:?}"), step);
            }
            if self.world.steered_live(w) {
                self.violation(
                    format!("a steered worker's event reached its owner: {key:?}"),
                    step,
                );
            }
        }
        if let Some(ep) = self
            .model
            .episodes
            .get_mut(&(worker, revision, kind))
        {
            ep.delivered = true;
        }
    }

    fn restart(&mut self) {
        let mut fresh = EventRouter::default();
        fresh.load_ack_store(&self.dir);
        self.router = fresh;
        self.stale.clear();
        self.model.on_restart(&self.world);
    }

    fn finish(&mut self) {
        for owner in self.world.owners.clone() {
            let ids = BTreeSet::new();
            let groups = BTreeSet::new();
            self.plain_watch(&owner, &ids, &groups, false, STEPS);
        }
        let undelivered: Vec<(String, String, usize, String)> = self
            .model
            .episodes
            .values()
            .filter(|ep| ep.state == EpState::Pending && !ep.delivered)
            .map(|ep| {
                (
                    ep.owner.clone(),
                    ep.worker.clone(),
                    ep.revision,
                    ep.kind.clone(),
                )
            })
            .collect();
        if !undelivered.is_empty() {
            self.violation(
                format!("pending events never delivered: {undelivered:?}"),
                STEPS,
            );
        }
    }
}

fn event_key(view: &Value) -> (String, String, usize, String) {
    (
        view["owner"].as_str().unwrap_or("").to_string(),
        view["worker_id"].as_str().unwrap_or("").to_string(),
        view["revision"].as_u64().unwrap_or(0) as usize,
        view["event"].as_str().unwrap_or("").to_string(),
    )
}

fn router_pending(router: &EventRouter) -> BTreeSet<(String, String, usize, String)> {
    let mut set = BTreeSet::new();
    for history in router.watch_history.values() {
        for view in &history.pending {
            set.insert(event_key(view));
        }
    }
    set
}

fn run_seed(seed: u64) {
    let mut harness = Harness::new(seed);
    let mut rng = Rng::new(seed);
    for step in 0..STEPS {
        let action = rng.below(100);
        let described = match action {
            0..=34 => {
                harness.mutate_status(&mut rng);
                "status"
            }
            35..=44 => {
                harness.mutate_steer(&mut rng);
                "steer"
            }
            45..=54 => {
                harness.mutate_consolidator(&mut rng);
                "consolidator"
            }
            55..=59 => {
                harness.mutate_retire(&mut rng);
                "retire"
            }
            60..=64 => {
                harness.mutate_branch_gone(&mut rng);
                "branch_gone"
            }
            65..=69 => {
                harness.mutate_question(&mut rng);
                "question"
            }
            70..=74 => {
                harness.mutate_resume(&mut rng);
                "resume"
            }
            _ => "noop",
        };
        harness.model.reconcile(&harness.world);
        harness.trace.push(format!("step {step}: {described}"));
        let use_row = rng.chance(15);
        harness.observe(step, use_row);

        match rng.below(100) {
            0..=44 => {
                let owner = harness.world.owners[rng.below(harness.world.owners.len())].clone();
                let ids = if rng.chance(30) {
                    let owned: Vec<String> = harness
                        .world
                        .workers
                        .iter()
                        .filter(|w| w.owner == owner)
                        .map(|w| w.id.clone())
                        .collect();
                    if owned.is_empty() {
                        BTreeSet::new()
                    } else {
                        [owned[rng.below(owned.len())].clone()].into_iter().collect()
                    }
                } else {
                    BTreeSet::new()
                };
                let groups = if rng.chance(30) {
                    let g = harness.world.groups[rng.below(harness.world.groups.len())].clone();
                    [g].into_iter().collect()
                } else {
                    BTreeSet::new()
                };
                let ack = rng.chance(60);
                harness.plain_watch(&owner, &ids, &groups, ack, step);
            }
            45..=59 => {
                let owner = harness.world.owners[rng.below(harness.world.owners.len())].clone();
                let group = harness.world.groups[rng.below(harness.world.groups.len())].clone();
                harness.round_watch(&owner, &group, step);
            }
            60..=69 => {
                let live = harness.world.live_workers();
                if !live.is_empty() {
                    let idx = live[rng.below(live.len())];
                    let wid = harness.world.workers[idx].id.clone();
                    let owner = harness.world.workers[idx].owner.clone();
                    let reported = harness.model.reported.get(&wid).cloned();
                    harness.model.mark_seen(&wid);
                    harness.trace_detail(format!(
                        "  mark_seen {owner} {wid} reported={reported:?}"
                    ));
                    if let Some((revision, kind)) = reported {
                        harness.acked.insert((owner.clone(), wid.clone(), revision, kind));
                    }
                    harness.router.mark_seen(&owner, &wid);
                }
            }
            70..=74 => {
                harness.trace_detail("  restart".to_string());
                harness.restart()
            }
            _ => {}
        }
    }
    harness.finish();
}

#[test]
fn event_delivery_model_holds_for_every_seed() {
    for seed in 0..SEEDS {
        run_seed(seed);
    }
}

/// A plain acknowledgement must suppress the same transition in `--all`, too.
#[test]
fn plain_ack_does_not_replay_as_a_round() {
    let mut harness = Harness::new(SEEDS + 1);
    harness.transition(0, Status::Completed);
    let owner = harness.world.workers[0].owner.clone();
    let id = harness.world.workers[0].id.clone();
    let group = harness.world.workers[0].group.clone();
    harness.model.reconcile(&harness.world);
    harness.observe(0, false);
    let ctx = harness.ctxs[&owner].clone();
    let reply = harness.router.watch_reply(&ctx, &json!({"worker_ids":[id]})).unwrap();
    harness.router.acknowledge_watch(&ctx, reply["events"][0]["sequence"].as_u64().unwrap());
    let round = harness.router.watch_reply(&ctx, &json!({"all":true,"worker_ids":[id],"group":group})).unwrap();
    assert_eq!(round["events"], json!([]), "an acknowledged completion is not a fresh round: {round}");
}
