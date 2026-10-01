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
    store
        .amend("owner", "group", None, Some(Some("")))
        .unwrap();
    assert_eq!(store.candidates()[0].verify.as_deref(), Some(""));

    // Once consumed the settings are spent: the round leaves the candidate set
    // and an amend is refused by name rather than silently changing a row no
    // consolidator will read.
    let _launch = store.claim(&store.candidates()[0]).expect("claim the round");
    store.consume(&store.candidates()[0]).unwrap();
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
