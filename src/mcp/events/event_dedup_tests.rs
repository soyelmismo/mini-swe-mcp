//! One transition is delivered once, whichever source observes it.
//!
//! The live pool snapshot describes a finished worker with its verified
//! summary and its revision; the registry row of the same worker describes the
//! same revision with less detail. The router keys a reported event by
//! `(worker id, revision, event kind)`, so the row-derived view is not a second
//! delivery, while a new revision or a different kind still is.

use super::*;
use crate::pool::WorkerRegistryEntry;

/// The in-memory view of a worker that just completed at `revision`.
fn live_completion(id: &str, revision: usize, step: usize) -> serde_json::Value {
    json!({
        "worker_id": id, "owner": "owner", "status": "completed", "step": step,
        "turns": step, "revision": revision, "verified": true, "summary": "Done.",
        "branch": format!("worker-{id}"), "question": null,
        "metrics": WorkerMetrics::default(),
    })
}

/// The registry row the same completion leaves behind.
///
/// Built from JSON, not a struct literal: the registry gains optional columns
/// over time, and every other reader parses the row from disk, so the fixture
/// exercises that path and does not have to name a column added later.
fn row(id: &str, revision: usize, step: usize) -> WorkerRegistryEntry {
    serde_json::from_value(json!({
        "id": id, "pid": std::process::id(), "task": "deliver once",
        "model": "test", "status": "completed", "step": step, "max_turns": 10,
        "last_command": "completed", "question": null, "started_at": 0,
        "updated_at": 0, "revision": revision,
    }))
    .expect("a registry row round-trips")
}

/// Every event the router has queued, in sequence order.
fn queued(router: &EventRouter) -> Vec<serde_json::Value> {
    let mut events: Vec<_> = router
        .watch_history
        .values()
        .flat_map(|history| history.pending.iter().cloned())
        .collect();
    events.sort_by_key(|event| event["sequence"].as_u64());
    events
}

/// The row-derived view of a finished worker names the revision it finished
/// at, so the router recognises it as the event it already reported.
#[test]
fn a_live_completion_and_its_registry_row_are_one_event() {
    let now = crate::pool::unix_timestamp();
    let mut router = EventRouter::default();
    router.observe_watch([("w1".to_string(), live_completion("w1", 1, 4))].into());
    assert_eq!(queued(&router).len(), 1, "the live completion is delivered");

    // The in-memory record was reaped a few seconds later; the registry row
    // now describes the same revision of the same completion.
    let row_view = crate::cli::watch::registry_snapshot(&row("w1", 1, 4), now);
    router.observe_watch([("w1".to_string(), row_view)].into());
    let events = queued(&router);
    assert_eq!(
        events.len(),
        1,
        "the row view is not a second delivery: {events:?}"
    );
    assert_eq!(
        events[0]["verified"],
        json!(true),
        "the live payload is what the owner keeps"
    );
}

/// A continuation advanced the worker: the same event kind at a new revision is
/// a new event.
#[test]
fn a_new_revision_of_the_same_worker_is_a_new_event() {
    let now = crate::pool::unix_timestamp();
    let mut router = EventRouter::default();
    router.observe_watch([("w2".to_string(), live_completion("w2", 1, 4))].into());
    router.observe_watch(
        [(
            "w2".to_string(),
            crate::cli::watch::registry_snapshot(&row("w2", 1, 4), now),
        )]
        .into(),
    );
    assert_eq!(queued(&router).len(), 1, "the duplicate is suppressed");

    router.observe_watch(
        [(
            "w2".to_string(),
            crate::cli::watch::registry_snapshot(&row("w2", 2, 2), now),
        )]
        .into(),
    );
    let events = queued(&router);
    assert_eq!(events.len(), 2, "revision 2 is a new event: {events:?}");
    assert_eq!(events[1]["revision"], json!(2));
}

/// The registry row is written before the final turn counter moves, so its step
/// can lag the live view: the key ignores the step.
#[test]
fn a_row_view_that_lags_the_step_is_still_the_same_event() {
    let now = crate::pool::unix_timestamp();
    let mut router = EventRouter::default();
    router.observe_watch([("w3".to_string(), live_completion("w3", 1, 4))].into());
    let row_view = crate::cli::watch::registry_snapshot(&row("w3", 1, 3), now);
    router.observe_watch([("w3".to_string(), row_view)].into());
    assert_eq!(
        queued(&router).len(),
        1,
        "a lagging row view is not a second delivery"
    );
}
