//! Launch cancellation and isolation do not require workers or network waits.
use super::*;

#[tokio::test]
async fn launch_does_not_block_unrelated_dispatch_and_cancellation_releases_claim() {
    let dir = std::env::temp_dir().join(format!("auto-claim-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir(&dir).unwrap();
    let store = AutoConsolidate::open(dir.clone()).unwrap();
    let round = Round {
        owner: "owner".into(),
        group: "group".into(),
        repo: dir.clone(),
        model: None,
        verify: None,
        generation: 0,
        consumed: false,
        baseline: Vec::new(),
        consolidators: Vec::new(),
    };
    store.record(round.clone(), true).unwrap();
    let launch = store.claim(&round).unwrap();
    assert!(store.claim(&round).is_none());
    let dispatch = tokio::time::timeout(
        std::time::Duration::from_millis(100),
        store.dispatch_guard(vec![
            ("another-owner".into(), "group".into()),
            ("owner".into(), "another-group".into()),
        ]),
    )
    .await;
    let same_round = tokio::time::timeout(
        std::time::Duration::from_millis(100),
        store.dispatch_guard(vec![(round.owner.clone(), round.group.clone())]),
    )
    .await;
    drop(launch);
    let retry = tokio::time::timeout(
        std::time::Duration::from_millis(100),
        store.dispatch_guard(vec![(round.owner.clone(), round.group.clone())]),
    )
    .await;
    std::fs::remove_dir_all(&dir).unwrap();
    assert!(
        same_round.is_err(),
        "Same round dispatch raced its claimed launch"
    );
    assert!(retry.is_ok(), "Cancelled launch left dispatch blocked");
    drop(retry);
    assert!(
        dispatch.is_ok(),
        "A waiting launch blocked an unrelated dispatch"
    );
    drop(dispatch);
    assert!(
        store.claim(&round).is_some(),
        "Cancelled launch leaked its claim"
    );
}

/// The amend is scoped to one pending round of the caller, and it lands on disk.
///
/// Two refusals are load-bearing and neither is visible from the outside: an
/// edit behind the daemon's back is overwritten by the next write, so the
/// supported path has to be the one that persists, and a consumed round must
/// not be amendable because its consolidator is already running with the
/// settings it was given.
#[tokio::test]
async fn amend_is_owner_scoped_pending_only_and_persisted() {
    let dir = std::env::temp_dir().join(format!("auto-amend-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    let store = AutoConsolidate::open(dir.clone()).unwrap();
    let pending = Round {
        owner: "owner".into(),
        group: "group".into(),
        repo: dir.clone(),
        model: None,
        verify: Some("cargo build".into()),
        generation: 0,
        consumed: false,
        baseline: Vec::new(),
        consolidators: Vec::new(),
    };
    store.record(pending.clone(), true).unwrap();

    // Another owner's round is not addressable at all: the row is looked up by
    // (owner, group), so there is nothing to name.
    let Err(foreign) = store.amend("other-owner", "group", None, Some(Some("cargo test"))) else {
        panic!("another owner's round must not be amendable");
    };
    assert!(
        foreign.to_string().contains("other-owner"),
        "the refusal must name the caller that has no such round: {foreign}"
    );

    // A pending round takes the amendment, and the durable file carries it --
    // the property a hand edit of that file lacks.
    let Ok(amended) = store.amend("owner", "group", None, Some(Some("cargo test --all"))) else {
        panic!("a pending round is amendable");
    };
    assert_eq!(amended.verify.as_deref(), Some("cargo test --all"));
    let stored: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("auto-consolidate.json")).unwrap()).unwrap();
    assert_eq!(stored[0]["verify"], "cargo test --all", "{stored}");

    // An omitted half keeps its value; an empty `verify` clears the gate.
    store
        .amend("owner", "group", Some(Some("ninja")), None)
        .unwrap();
    let reread = store.candidates();
    assert_eq!(reread.len(), 1, "the amend must not open or drop a round");
    assert_eq!(reread[0].verify.as_deref(), Some("cargo test --all"));
    assert_eq!(reread[0].model.as_deref(), Some("ninja"));
    store.amend("owner", "group", None, Some(Some(""))).unwrap();
    assert_eq!(store.candidates()[0].verify.as_deref(), Some(""));

    // Once consumed the settings are spent: the round leaves the candidate set
    // and an amend is refused by name rather than silently changing a row no
    // consolidator will read.
    let round = store.candidates()[0].clone();
    let launch = store.claim(&round).expect("claim the round");
    store.consume(&round).unwrap();
    drop(launch);
    let Err(consumed) = store.amend("owner", "group", None, Some(Some("cargo test"))) else {
        panic!("a consumed round must not be amendable");
    };
    assert!(
        consumed.to_string().contains("consumed"),
        "the refusal must say why: {consumed}"
    );
    let persisted: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("auto-consolidate.json")).unwrap()).unwrap();
    assert_eq!(
        persisted[0]["consumed"], true,
        "a refused amend must not write: {persisted}"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

/// The round41 window: a registry row that reads `Completed` while this
/// process is still running the worker it describes must not be treated as a
/// settled round.
///
/// The row is a cross-process view, and the round scheduler reads only rows.
/// Between the implementer's completion and the review phase's first write
/// (and for any racing or coalesced write), a row can read settled for a worker
/// this process is still driving through its review phase; starting the round
/// then dispatches a consolidator whose manifest lists that worker as *not
/// ready*, so the round merges only part of the group and closes without it.
/// The tick therefore cross-checks the pool's live state: a worker this process
/// still runs is not settled, whatever its row says.
///
/// Fails without the gate: the row alone satisfies the tick's precondition, so
/// the round would start.
#[tokio::test]
async fn a_completed_row_under_a_live_worker_does_not_settle_the_round() {
    let dir = std::env::temp_dir().join(format!("auto-live-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    let root = crate::worktree::ScratchRoot::new(dir.clone());
    let pool = crate::pool::WorkerPool::with_scratch(
        4,
        "http://127.0.0.1:1".into(),
        String::new(),
        root.clone(),
    );
    let server = crate::mcp::McpServer::new(pool.clone(), "integrator".into());

    let mut meta = crate::pool::WorkerMeta::test_meta("9c8264b1", "owner");
    meta.group = Some("round41".into());
    // A settled row is pruned on load unless the worker still has a worktree or
    // a branch, so the fixture gives it one: that is the shape a real completed
    // worker leaves behind, and without it the row would be deleted before the
    // scheduler ever saw it.
    std::fs::create_dir_all(root.join("swe-wt-9c8264b1")).unwrap();
    // The durable row reads `Completed` -- the settled status the tick keys on.
    pool.__test_save_status(
        &meta,
        "ninja",
        crate::pool::RegistryStatus::Completed,
        40,
        40,
        "completed",
        None,
    );
    // The pool itself is still running the worker: it is in its review phase,
    // which the row has not caught up with (and, before the row write, cannot).
    pool.__test_insert_worker(crate::pool::WorkerRecord {
        id: "9c8264b1".into(),
        task: "implement the thing".into(),
        model: "ninja".into(),
        owner: "owner".into(),
        state: crate::pool::WorkerState::Running {
            step: 41,
            last_command: "reviewing".into(),
            started_at: 0,
        },
        metrics: Default::default(),
        logs: crate::pool::LogBuffer::default(),
        pending_steer: Vec::new(),
        resume_tx: None,
        handle: None,
        revision: 0,
    })
    .await;

    let round = Round {
        owner: "owner".into(),
        group: "round41".into(),
        repo: dir.clone(),
        model: None,
        verify: None,
        generation: 0,
        consumed: false,
        baseline: Vec::new(),
        consolidators: Vec::new(),
    };
    let entries = crate::pool::load_all_registry_entries_in(&root);
    let members: Vec<_> = entries
        .iter()
        .filter(|e| {
            e.owner.as_deref() == Some(&round.owner) && e.group.as_deref() == Some(&round.group)
        })
        .collect();
    assert_eq!(members.len(), 1, "the round must own the one worker");
    assert_eq!(
        members[0].status,
        crate::pool::RegistryStatus::Completed,
        "the fixture must reproduce the stale settled row"
    );

    // This is the scheduler's own decision, the one the tick takes before it
    // claims and starts a round.
    assert!(
        matches!(
            server.round_decision(&round, &entries).await,
            crate::mcp::auto_consolidate::RoundDecision::Wait
        ),
        "a worker this process is still running is not settled: the round would \
         start a consolidator whose manifest lists 9c8264b1 as not ready"
    );

    // Once the run itself ends the worker, the row and the live state agree and
    // the round may start; the gate must not deadlock a settled round.
    pool.__test_set_worker_state(
        "9c8264b1",
        crate::pool::WorkerState::Completed {
            turns: 42,
            diff: String::new(),
            summary: "done".into(),
            completed_at: 0,
            artifacts: Vec::new(),
            branch: None,
            verified: Some(true),
            metrics: Default::default(),
            revision: 0,
            report: None,
            verdicts: None,
        },
    )
    .await;
    assert!(
        matches!(
            server.round_decision(&round, &entries).await,
            crate::mcp::auto_consolidate::RoundDecision::Start
        ),
        "a finished worker settles its round"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

/// A worker paused on an orchestrator question is not settled either.
///
/// `Running` is only one of the two non-terminal states the pool holds: a
/// worker that stopped to ask its orchestrator a question sits in `Paused`
/// until it is answered, and its durable row can lag the same way a running
/// worker's does. Treating only `Running` as live would let a round start on a
/// row that reads settled while the pool is still holding the worker open.
///
/// Fails without the `Paused` arm: the record is not `Running`, so the round
/// would start.
#[tokio::test]
async fn a_paused_worker_under_a_completed_row_does_not_settle_the_round() {
    let dir = std::env::temp_dir().join(format!("auto-paused-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    let root = crate::worktree::ScratchRoot::new(dir.clone());
    let pool = crate::pool::WorkerPool::with_scratch(
        4,
        "http://127.0.0.1:1".into(),
        String::new(),
        root.clone(),
    );
    let server = crate::mcp::McpServer::new(pool.clone(), "integrator".into());

    let mut meta = crate::pool::WorkerMeta::test_meta("9c8264b1", "owner");
    meta.group = Some("round41".into());
    // Keeps the settled row from being pruned on load, as above.
    std::fs::create_dir_all(root.join("swe-wt-9c8264b1")).unwrap();
    pool.__test_save_status(
        &meta,
        "ninja",
        crate::pool::RegistryStatus::Completed,
        40,
        40,
        "completed",
        None,
    );
    pool.__test_insert_worker(crate::pool::WorkerRecord {
        id: "9c8264b1".into(),
        task: "implement the thing".into(),
        model: "ninja".into(),
        owner: "owner".into(),
        state: crate::pool::WorkerState::Paused {
            question: "which approach?".into(),
            step: 41,
            paused_at: 0,
        },
        metrics: Default::default(),
        logs: crate::pool::LogBuffer::default(),
        pending_steer: Vec::new(),
        resume_tx: None,
        handle: None,
        revision: 0,
    })
    .await;

    let round = Round {
        owner: "owner".into(),
        group: "round41".into(),
        repo: dir.clone(),
        model: None,
        verify: None,
        generation: 0,
        consumed: false,
        baseline: Vec::new(),
        consolidators: Vec::new(),
    };
    let entries = crate::pool::load_all_registry_entries_in(&root);
    let members: Vec<_> = entries
        .iter()
        .filter(|e| {
            e.owner.as_deref() == Some(&round.owner) && e.group.as_deref() == Some(&round.group)
        })
        .collect();
    assert_eq!(
        members.len(),
        1,
        "the settled row must survive the load for this fixture to mean anything"
    );
    assert_eq!(
        members[0].status,
        crate::pool::RegistryStatus::Completed,
        "the fixture must reproduce the stale settled row"
    );

    assert!(
        matches!(
            server.round_decision(&round, &entries).await,
            crate::mcp::auto_consolidate::RoundDecision::Wait
        ),
        "a worker the pool still holds open is not settled, paused or not: the \
         round would start a consolidator whose manifest lists 9c8264b1 as not \
         ready and close without it"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}
