//! A retired worker's event is never replayed, acknowledged or not.
//!
//! Retirement drops the watch acknowledgements of a worker whose branch is
//! gone: the event can never fire again, so its position would only pin a key
//! in a bounded store. The router's own replay state has to go with them. If a
//! snapshot taken before the retirement still describes the worker, the router
//! reads its last event as an unseen transition and queues it again, and the
//! next watch answers with a "While you were not watching" completion for work
//! that is already in the base branch.
//!
//! Driven through [`EventRouter`] directly: the retirement and the stale
//! snapshot both happen under the router's own lock, which is what a merge does
//! and what the recovery sweep does, so the two are ordered here as well.
use super::*;

const OWNER: &str = "retire-owner";

/// A worker that completed at `revision` and is still described by the view.
fn completed(id: &str, revision: usize) -> serde_json::Value {
    json!({"worker_id":id, "owner":OWNER, "group":"round", "model":"test",
        "status":"completed", "step":4, "turns":4, "revision":revision,
        "verified":true, "summary":"Done.", "branch":format!("worker-{id}"),
        "question":null, "last_step_at":crate::pool::unix_timestamp(),
        "metrics":WorkerMetrics::default()})
}

fn view(id: &str, revision: usize) -> crate::cli::watch::Snapshot {
    [(id.to_string(), completed(id, revision))].into()
}

fn agent(connection: u64) -> super::super::server::ConnectionContext {
    let mut ctx = super::super::server::ConnectionContext::hub_connection(connection);
    ctx.agent_id = Some(OWNER.to_string());
    ctx
}

fn params() -> serde_json::Value {
    json!({"worker_ids":[], "initial":true})
}

fn queued(router: &EventRouter) -> Vec<serde_json::Value> {
    let mut events: Vec<_> = router
        .watch_history
        .values()
        .flat_map(|history| history.pending.iter().cloned())
        .collect();
    events.sort_by_key(|event| event["sequence"].as_u64());
    events
}

/// Delivered and acknowledged, then retired: the completion must not come back
/// even though the retirement removed the very acknowledgement that used to
/// suppress it.
#[test]
fn an_acknowledged_completion_is_not_replayed_after_retirement() {
    let mut router = EventRouter::default();
    router.observe_watch(view("w-ack", 1));
    let ctx = agent(1);
    let first = router
        .watch_reply(&ctx, &params())
        .expect("the watch answers");
    let events = first["events"].as_array().expect("events array");
    assert_eq!(events.len(), 1, "the completion is delivered once: {first}");
    assert_eq!(events[0]["event"], "completed", "{first}");
    router.acknowledge_watch(
        &ctx,
        first["events"][0]["sequence"].as_u64().expect("a sequence"),
    );
    // The worker is still there and still completed: without an event of its
    // own it delivers nothing, which is the state the retirement starts from.
    router.observe_watch(view("w-ack", 1));
    assert!(
        queued(&router).is_empty(),
        "acknowledged, so nothing replays"
    );
    assert!(
        router.acks.acknowledged(OWNER, "w-ack", 1, "completed"),
        "the watch recorded the position"
    );

    router.forget_worker("w-ack");
    assert!(
        !router.acks.acknowledged(OWNER, "w-ack", 1, "completed"),
        "retirement drops the position with the worker"
    );

    // A snapshot taken before the retirement still describes the worker: the
    // router must not read that as a transition it has never seen.
    router.observe_watch(view("w-ack", 1));
    assert!(
        queued(&router).is_empty(),
        "a retired worker has no new event: {:?}",
        queued(&router)
    );
    let after = router
        .watch_reply(&ctx, &params())
        .expect("the watch answers");
    assert!(
        after["events"].as_array().is_some_and(Vec::is_empty),
        "a retired worker is never replayed: {after}"
    );
}

/// The worker was retired before its owner read the event. There is nothing
/// left to deliver -- the branch is in the base branch -- so the event is gone
/// rather than queued for a later session.
#[test]
fn an_unacknowledged_event_is_not_replayed_after_retirement() {
    let mut router = EventRouter::default();
    router.observe_watch(view("w-unacked", 1));
    assert_eq!(queued(&router).len(), 1, "the event is queued first");

    router.forget_worker("w-unacked");
    assert!(
        queued(&router).is_empty(),
        "retirement drops the queued event"
    );
    let ctx = agent(1);
    let after = router
        .watch_reply(&ctx, &params())
        .expect("the watch answers");
    assert!(
        after["events"].as_array().is_some_and(Vec::is_empty),
        "an unacknowledged event of a retired worker is gone: {after}"
    );
    // A stale snapshot taken before the retirement names the worker again.
    // The explicit-id path replays the last reported terminal event, so it is
    // the second way the same completion could come back: a retired worker is
    // unknown, never a completion to review.
    router.observe_watch(view("w-unacked", 1));
    let explicit = router
        .watch_reply(&ctx, &json!({"worker_ids":["w-unacked"], "initial":true}))
        .expect_err("a retired worker is not found by an explicit id");
    assert!(
        explicit.to_string().contains("Worker not found"),
        "the explicit id must not resurrect a retired worker: {explicit}"
    );
}

/// The tombstone only has to outlive the snapshots that still describe the
/// worker. Once no view names it, the branch, the row and the record are gone
/// and the id can never come back, so holding it would cost memory for a worker
/// that cannot fire an event again.
#[test]
fn a_tombstone_is_released_once_no_view_describes_the_worker() {
    let mut router = EventRouter::default();
    router.observe_watch(view("w-gone", 1));
    router.forget_worker("w-gone");
    assert_eq!(router.retired.len(), 1, "the retirement left a tombstone");

    // The snapshot that raced the retirement still names the worker: the
    // tombstone has to survive this, and the event still must not come back.
    router.observe_watch(view("w-gone", 1));
    assert_eq!(
        router.retired.len(),
        1,
        "a view that still names the worker keeps the tombstone"
    );
    assert!(queued(&router).is_empty(), "and still no event");

    // The next snapshot is the truth: the worker is gone, so the tombstone goes
    // with it and leaves the router holding nothing for a dead id.
    router.observe_watch(crate::cli::watch::Snapshot::new());
    assert!(
        router.retired.is_empty(),
        "a retired worker no view describes releases its tombstone"
    );
}
