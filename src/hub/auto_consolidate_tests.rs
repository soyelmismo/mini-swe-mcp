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
