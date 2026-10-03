//! The compact `status` contract: a small payload that still tells an
//! orchestrator what a worker is doing.
//!
//! These tests live apart from the shared CLI and pool suites because they pin
//! the two halves of the status work together: the text view of a
//! running/paused/completed worker, and the size and artifact scope of a
//! completed worker's payload.

mod common;

use common::{IsolatedPool, TempDir};
use mini_swe_mcp::cli::format::format_status;
use mini_swe_mcp::mcp::{ConnectionContext, McpServer};
use mini_swe_mcp::pool::{LogBuffer, WorkerMetrics, WorkerRecord, WorkerState};
use mini_swe_mcp::worktree::{ScratchRoot, WorktreeGuard};
use serde_json::{Value, json};

fn v(s: &str) -> Value {
    serde_json::from_str(s).expect("fixture must be valid JSON")
}

fn lines(text: &str) -> usize {
    text.lines().count()
}

#[test]
fn running_status_text_shows_the_step_the_clock_and_the_command_in_flight() {
    let text = format_status(&v(
        r#"{"worker_id":"w","state":{"state":"Running","details":{
             "step":2,"turns":2,"max_turns":30,"started_at":1000,"elapsed":45,
             "last_command":"cargo test --all-targets","command_started_at":1030,
             "command_elapsed":12,"metrics":{"turns_used":2,"repeat_blocks":1}}}}"#,
    ));
    for expected in [
        "State: Running",
        "Step 2/30",
        "elapsed 45s",
        "Command: cargo test --all-targets (running for 12s)",
        "Health: 2 turns, +0/-0 ext, 1 repeat, 0 nudges",
    ] {
        assert!(text.contains(expected), "missing {expected:?} in:\n{text}");
    }
    assert!(lines(&text) <= 8, "status must stay compact:\n{text}");
}

#[test]
fn running_status_text_names_the_build_slot_when_it_waits() {
    let text = format_status(&v(
        r#"{"worker_id":"w","state":{"state":"Running","details":{
             "step":1,"max_turns":20,"elapsed":5,"last_command":"cargo build",
             "waiting_for_slot":3}}}"#,
    ));
    assert!(
        text.contains("Waiting for a build slot (3 ahead)"),
        "the queued build slot must be visible:\n{text}"
    );
}

#[test]
fn paused_status_text_surfaces_the_question() {
    let text = format_status(&v(
        r#"{"worker_id":"w","state":{"state":"Paused","details":{
             "step":3,"max_turns":20,"elapsed":120,"question":"Ship the migration?"}}}"#,
    ));
    assert!(text.contains("State: Paused"), "{text}");
    assert!(text.contains("Step 3/20"), "{text}");
    assert!(text.contains("Question: Ship the migration?"), "{text}");
}

#[test]
fn completed_status_text_shows_verified_approval_revision_and_health() {
    let text = format_status(&v(
        r#"{"worker_id":"w","state":{"state":"Completed","details":{
             "turns":9,"step":9,"max_turns":20,"elapsed":320,"summary":"fixed parser",
             "verified":true,"revision":2,"metrics":{"turns_used":9,"repeat_blocks":2,
             "verify_runs":1,"verify_failures":1}}},
           "approved":{"at":1700000000,"note":"looks good"}}"#,
    ));
    for expected in [
        "State: Completed",
        "Step 9/20",
        "elapsed 320s",
        "Summary: fixed parser",
        "Verified: yes",
        "Revision: 2",
        "Approved: 1700000000 (looks good)",
        "Health: 9 turns, +0/-0 ext, 2 repeats, 0 nudges, verify 1/1 failed",
    ] {
        assert!(text.contains(expected), "missing {expected:?} in:\n{text}");
    }
    assert!(lines(&text) <= 8, "status must stay compact:\n{text}");
}

#[test]
fn failed_and_exhausted_status_text_name_the_reason() {
    let failed = format_status(&v(
        r#"{"worker_id":"w","state":{"state":"Failed","details":{
             "step":5,"max_turns":10,"elapsed":60,"error":"boom","revision":1}}}"#,
    ));
    assert!(failed.contains("Error: boom"), "{failed}");

    let exhausted = format_status(&v(
        r#"{"worker_id":"w","state":{"state":"Exhausted","details":{
             "turns":30,"step":30,"max_turns":30,"elapsed":900,"summary":"ran out",
             "reason":"turn_budget_exhausted","revision":0}}}"#,
    ));
    assert!(
        exhausted.contains("Reason: turn_budget_exhausted"),
        "{exhausted}"
    );
}

/// A completed worker's `status` stays small even with a multi-hundred-kilobyte
/// diff: the diff belongs to `collect`/`review`, and the artifact list is
/// capped. The pre-existing artifacts the sync seeded into the worktree are not
/// the worker's output and must not be reported at all.
#[tokio::test]
async fn completed_status_payload_is_small_and_lists_only_changed_artifacts() {
    let scratch = TempDir::new_in_tmp("status-payload");
    let repo = scratch.subdir("repo");
    common::git(&repo, &["init", "-q", "-b", "master"]);
    common::git(&repo, &["config", "user.name", "mini-swe-test"]);
    common::git(&repo, &["config", "user.email", "test@localhost"]);
    std::fs::write(repo.join("README.md"), "# status payload\n").unwrap();
    for i in 0..111 {
        let dir = if i % 2 == 0 {
            repo.join("audits")
        } else {
            repo.join(".agents")
        };
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(format!("pre_{i}.md")), format!("seed {i}\n")).unwrap();
    }
    common::git(&repo, &["add", "-A"]);
    common::git(
        &repo,
        &["commit", "-q", "-m", "seed pre-existing artifacts"],
    );

    // The worktree lives under the same temporary root as the repo, so the
    // test never resolves or touches the ambient real scratch.
    let root = ScratchRoot::new(scratch.subdir("scratch"));
    let id = format!(
        "status-payload-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let guard = WorktreeGuard::new_in(&root, &repo, &id).expect("worktree creation failed");
    std::fs::create_dir_all(guard.path.join("audits")).unwrap();
    std::fs::write(guard.path.join("audits/worker_new.md"), "new\n").unwrap();

    let changed = guard.sync_artifacts();
    assert_eq!(
        changed,
        vec!["audits/worker_new.md".to_string()],
        "only the worker's own artifact may be reported, not the seeded ones: {changed:?}"
    );

    // The runner stores the diff and the artifact list; a large diff must not
    // reach the status answer.
    let big_diff = format!("diff --git a/big.rs b/big.rs\n{}", "+x\n".repeat(60_000));
    let state = WorkerState::Completed {
        turns: 40,
        diff: big_diff,
        summary: "did the thing".to_string(),
        completed_at: 5_000,
        artifacts: changed,
        branch: Some(format!("worker-{id}")),
        verified: Some(true),
        metrics: WorkerMetrics::default(),
        revision: 2,
        report: None
    verdicts: None,,
    };

    let owned = IsolatedPool::new(4, "status-payload");
    owned.pool.__test_insert_worker(record(&id, state)).await;
    let server = McpServer::new(owned.pool.clone(), "ninja".to_string());
    let ctx = ConnectionContext {
        agent_id: Some("tester".to_string()),
        ..ConnectionContext::hub_connection(1)
    };
    let status = server
        .execute_tool_for("worker", json!({"action": "status", "worker_id": id}), &ctx)
        .await
        .expect("status of the completed worker");

    let details = &status["state"]["details"];
    assert!(
        details.get("diff").is_none(),
        "status must not embed the diff: {details}"
    );
    assert_eq!(details["artifacts"], json!(["audits/worker_new.md"]));
    assert_eq!(details["artifacts_total"], json!(1));
    let encoded = serde_json::to_string(&status).expect("serialize the status answer");
    assert!(
        encoded.len() < 2048,
        "status payload grew past 2 KB ({} bytes): {encoded}",
        encoded.len()
    );

    // `list` reads the same completion through its summary: it must stay small
    // too, even though it never carried the diff.
    let listing = server
        .execute_tool_for("worker", json!({"action": "list"}), &ctx)
        .await
        .expect("list");
    let encoded = serde_json::to_string(&listing).expect("serialize the list answer");
    assert!(
        encoded.len() < 2048,
        "list payload grew past 2 KB ({} bytes): {encoded}",
        encoded.len()
    );
    assert_eq!(
        listing["workers"][0]["state"]["artifacts"],
        json!(["audits/worker_new.md"])
    );
    drop(guard);
}

fn record(id: &str, state: WorkerState) -> WorkerRecord {
    WorkerRecord {
        id: id.to_string(),
        task: "t".to_string(),
        model: "m".to_string(),
        owner: "tester".to_string(),
        state,
        metrics: WorkerMetrics::default(),
        logs: LogBuffer::new(),
        pending_steer: Vec::new(),
        resume_tx: None,
        handle: None,
        revision: 2,
    }
}
