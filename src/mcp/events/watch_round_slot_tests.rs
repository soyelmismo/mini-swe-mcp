//! The `--all` round watch and the identity's single watch slot.
//!
//! A round must reserve its watch slot *before* it acknowledges the
//! workers it folds in. Acknowledge first and a second connection's
//! round watch is admitted instead, consumes and acknowledges the
//! transitions, and the watch that legitimately held the slot never
//! receives its round.
//!
//! A second connection never takes the slot over: a request the running
//! watch already follows is answered "covered", and a broader one widens
//! the stored selection to the union while leaving every pending event
//! with the connection that holds the slot.
//!
//! Driven through [`EventRouter`] directly: a live daemon may push a
//! notification to the first connection between the two calls, which
//! would make the outcome depend on scheduling rather than on the
//! acknowledgement order under test.
use super::*;

const OWNER: &str = "round-owner";
const GROUP: &str = "round";

fn running(id: &str, step: usize) -> serde_json::Value {
    json!({
        "worker_id": id, "owner": OWNER, "group": GROUP, "model": "test",
        "status": "running", "step": step, "revision": 0,
        "branch": format!("worker-{id}"),
        "last_step_at": crate::pool::unix_timestamp(),
        "metrics": crate::pool::WorkerMetrics::default(),
    })
}

fn completed(id: &str, step: usize) -> serde_json::Value {
    json!({
        "worker_id": id, "owner": OWNER, "group": GROUP, "model": "test",
        "status": "completed", "step": step, "revision": 0,
        "branch": format!("worker-{id}"),
        "last_step_at": crate::pool::unix_timestamp(),
        "verified": true, "summary": "Fixed.", "report": serde_json::Value::Null,
        "metrics": crate::pool::WorkerMetrics::default(),
    })
}

fn views(worker: fn(&str, usize) -> serde_json::Value, step: usize) -> crate::cli::watch::Snapshot {
    [
        ("w-1".to_string(), worker("w-1", step)),
        ("w-2".to_string(), worker("w-2", step)),
    ]
    .into_iter()
    .collect()
}

fn agent(connection: u64) -> super::super::server::ConnectionContext {
    let mut ctx = super::super::server::ConnectionContext::hub_connection(connection);
    ctx.agent_id = Some(OWNER.to_string());
    ctx
}

fn all() -> serde_json::Value {
    json!({"worker_ids": [], "group": GROUP, "initial": false, "all": true})
}

/// The second watch is refused, and the refused call left the round for
/// the connection that holds the slot.
#[test]
fn a_refused_round_watch_does_not_consume_the_round() {
    let mut router = EventRouter::default();
    // The workers run, so the first watch reserves the slot without
    // reporting anything.
    router.observe_watch(views(running, 1));
    let first = agent(1);
    let reserved = router
        .watch_reply(&first, &all())
        .expect("the first watch reserves the slot");
    assert!(
        reserved["events"]
            .as_array()
            .is_some_and(|events| events.is_empty()),
        "a running round must wait, not report: {reserved}"
    );
    assert!(router.watches.has(OWNER), "the first watch holds the slot");

    // Both workers stop: the round is ready, with real transitions to
    // acknowledge.
    router.observe_watch(views(completed, 2));

    // A second connection running the same round watch is covered: it
    // answers in-band, without consuming anything.
    let second = agent(2);
    let covered = router
        .watch_reply(&second, &all())
        .expect("a covered watch answers in-band");
    assert!(
        covered["covered"]["selection"].as_str().is_some(),
        "the covered watch must name the running selection: {covered}"
    );
    assert!(
        covered["events"]
            .as_array()
            .is_some_and(|events| events.is_empty()),
        "a covered watch carries no events: {covered}"
    );

    // The covered watch did not acknowledge anything, so the connection
    // that holds the slot still receives the whole round.
    let round = router
        .watch_reply(&first, &all())
        .expect("the first watch still answers");
    let events = round["events"].as_array().expect("events array");
    assert_eq!(events.len(), 1, "one consolidated round: {round}");
    assert_eq!(events[0]["event"], "round", "{round}");
    assert_eq!(
        events[0]["workers"].as_array().map(Vec::len),
        Some(2),
        "both workers are listed: {round}"
    );
}

/// Acknowledging the round marks its workers seen: a later plain watch
/// of the same ids has nothing left to replay.
#[test]
fn an_acknowledged_round_is_not_replayed_to_its_own_connection() {
    let mut router = EventRouter::default();
    router.observe_watch(views(completed, 2));
    let ctx = agent(1);
    let round = router.watch_reply(&ctx, &all()).expect("the round answers");
    assert_eq!(round["events"].as_array().map(Vec::len), Some(1), "{round}");

    let plain = router
        .watch_reply(
            &ctx,
            &json!({"worker_ids": ["w-1", "w-2"], "group": GROUP, "initial": true}),
        )
        .expect("a plain watch answers");
    assert!(
        plain["events"]
            .as_array()
            .is_some_and(|events| events.is_empty()),
        "the round already acknowledged its workers: {plain}"
    );
}

/// A broader round watch of a second connection widens the running one: the
/// stored selection becomes the union, the widening reaches the running
/// connection, and its pending round survives.
#[test]
fn a_broader_round_watch_widens_the_running_one() {
    let mut router = EventRouter::default();
    router.observe_watch(views(running, 1));
    let first = agent(1);
    let reserved = router
        .watch_reply(&first, &all())
        .expect("the first watch reserves the slot");
    assert!(
        reserved["events"]
            .as_array()
            .is_some_and(|events| events.is_empty()),
        "a running round must wait, not report: {reserved}"
    );

    // A second connection asks for every group of the caller: it is widened
    // to the union, and the widening rides on the listed connection.
    let second = agent(2);
    let wider = json!({"worker_ids": [], "group": [], "initial": false, "all": true});
    let widened = router
        .watch_reply(&second, &wider)
        .expect("a broader watch widens the running one");
    let widened = &widened["widened"];
    assert!(
        widened["selection"].as_str().is_some(),
        "the widening must name the union: {widened}"
    );
    assert_eq!(widened["all"], json!(true), "round mode is kept: {widened}");
    assert!(
        widened["group"].as_array().is_some_and(Vec::is_empty),
        "an unfiltered union follows every group: {widened}"
    );

    // The union is what the running watch now follows, so its round still
    // arrives - and the widening did not consume or acknowledge anything.
    router.observe_watch(views(completed, 2));
    let round = router
        .watch_reply(&first, &wider)
        .expect("the widened watch still answers");
    let events = round["events"].as_array().expect("events array");
    assert_eq!(events.len(), 1, "one consolidated round: {round}");
    assert_eq!(events[0]["event"], "round", "{round}");
}

/// The widened filter is pushed to the connection that already watches, so a
/// running process can adopt it without reconnecting.
#[test]
fn a_widening_is_pushed_to_the_running_connection() {
    let mut router = EventRouter::default();
    let (tx, mut rx) = tokio::sync::mpsc::channel(4);
    let mut ctx = agent(1);
    ctx.pid = Some(4321);
    router.register(&ctx, tx);
    router.observe_watch(views(running, 1));
    let first = agent(1);
    router.watch_reply(&first, &all()).expect("the first watch");
    assert!(rx.try_recv().is_err(), "nothing was pushed yet");

    let second = agent(2);
    let wider = json!({"worker_ids": [], "group": [], "initial": false, "all": true});
    router
        .watch_reply(&second, &wider)
        .expect("a broader watch widens the running one");

    let frame = rx.try_recv().expect("the running connection is told");
    let frame: serde_json::Value = serde_json::from_str(frame.trim()).expect("JSON frame");
    assert_eq!(frame["method"], json!(WATCH_WIDEN_METHOD), "{frame}");
    assert_eq!(frame["params"]["all"], json!(true), "{frame}");
    assert!(
        frame["params"]["group"]
            .as_array()
            .is_some_and(Vec::is_empty),
        "the push carries the union: {frame}"
    );
}
