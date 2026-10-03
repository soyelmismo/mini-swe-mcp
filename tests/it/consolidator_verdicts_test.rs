//! A consolidator's per-worker verdicts are recorded and shown, not grepped.
//!
//! The consolidator procedure asks for one line per integrated worker
//! (`REPORT <id> approved|returned|fixed: <one line>`) and a `RISK:` line for
//! anything touching the sandbox, governance or identity. Reading them used to
//! mean grepping the consolidator's history JSONL, because the round's
//! completion event carried the one-line headline and nothing else. The
//! verdicts now live on the completed state and the registry row, and are
//! rendered in the completion event and in `review <consolidator>` -- bounded,
//! so a long round cannot turn a notification into a transcript.

use crate::common;
use mini_swe_mcp::cli::watch;
use mini_swe_mcp::mcp::{EventKind, Outcome, WorkerView, render_for_test};
use mini_swe_mcp::pool::{
    COMPLETION_SENTINEL, RegistryStatus, VERDICT_BYTES, WorkerMeta, WorkerPool,
    WorkerRegistryEntry, WorkerReport, WorkerRole, WorkerState, WorkerVerdicts,
    load_registry_entry_in, parse_verdict_lines, save_registry_entry_in,
};
use mini_swe_mcp::worktree::ScratchRoot;

/// The owner of every worker in this file.
const OWNER: &str = "verdict-agent";

/// A consolidator closing message with a standard block, per-worker verdicts
/// and the risks it flagged.
fn consolidator_message() -> String {
    [
        "REPORT",
        "done: integrated 2 branches, gate green",
        "files: src/a.rs, src/b.rs",
        "tests: cargo test: passed",
        "risks: none",
        "",
        "REPORT 2a9aaca3 approved: the parser change stands",
        "REPORT 41b0fde1 fixed: resolved the interaction in src/b.rs",
        "RISK: the sandbox policy edit touched src/agent/sandbox.rs",
    ]
    .join("\n")
}

/// The verdicts of [`consolidator_message`], in the order they were written.
fn expected_verdicts() -> WorkerVerdicts {
    WorkerVerdicts {
        workers: vec![
            "REPORT 2a9aaca3 approved: the parser change stands".to_string(),
            "REPORT 41b0fde1 fixed: resolved the interaction in src/b.rs".to_string(),
        ],
        risks: vec!["RISK: the sandbox policy edit touched src/agent/sandbox.rs".to_string()],
    }
}

/// A consolidator row in `status`, carrying `verdicts`.
fn meta_with_verdicts(id: &str, verdicts: Option<WorkerVerdicts>) -> WorkerMeta {
    WorkerMeta {
        task: "integrate the round".to_string(),
        group: Some("round-1".to_string()),
        role: WorkerRole::Consolidate,
        report: Some(WorkerReport {
            done: "integrated 2 branches, gate green".to_string(),
            files: "src/a.rs, src/b.rs".to_string(),
            tests: "cargo test: passed".to_string(),
            risks: "none".to_string(),
        }),
        verdicts,
        ..WorkerMeta::test_meta(id, OWNER)
    }
}

/// The completion event a consolidator's view produces for `verdicts`.
fn completion_event(verdicts: Option<WorkerVerdicts>) -> String {
    let mut view = WorkerView {
        worker_id: "c1".to_string(),
        event: Some(EventKind::Completed),
        status: "completed".to_string(),
        branch: Some("worker-c1".to_string()),
        outcome: Outcome {
            summary: Some("integrated 2 branches, gate green".to_string()),
            verified: Some(true),
            report: Some(WorkerReport {
                done: "integrated 2 branches, gate green".to_string(),
                files: "src/a.rs, src/b.rs".to_string(),
                tests: "cargo test: passed".to_string(),
                risks: "none".to_string(),
            }),
            verdicts,
            ..Default::default()
        },
        ..Default::default()
    };
    view.group = "round-1".to_string();
    render_for_test(&view, EventKind::Completed)
}

#[test]
fn a_consolidator_final_message_yields_its_verdicts_and_risks_in_order() {
    let verdicts = parse_verdict_lines(&consolidator_message());
    assert_eq!(verdicts, expected_verdicts());
    assert!(!verdicts.is_empty());
}

/// The round's own block is not a per-worker verdict: its `REPORT` marker and
/// `risks:` line must not be recorded as one, or every round would read as
/// having a worker named `risks`.
#[test]
fn the_round_block_is_not_mistaken_for_a_verdict() {
    let verdicts =
        parse_verdict_lines("REPORT\ndone: integrated\nrisks: none\nREPORT\nREPORT risK: hmm");
    assert!(verdicts.is_empty(), "{verdicts:?}");
    assert_eq!(
        parse_verdict_lines("Nothing to report.").workers,
        Vec::<String>::new()
    );
}

/// A message with no per-worker lines yields nothing, so an ordinary worker's
/// completion carries no verdict block at all.
#[test]
fn a_message_without_per_worker_lines_yields_nothing() {
    assert!(parse_verdict_lines("REPORT\ndone: fixed\nrisks: none").is_empty());
    assert!(parse_verdict_lines("").is_empty());
}

/// The rendered event shows the compact headline first and then one line per
/// worker, with the risks after them.
#[test]
fn the_completion_event_shows_the_verdicts_after_the_headline() {
    let text = completion_event(Some(expected_verdicts()));
    let headline = text
        .find("Done: integrated 2 branches")
        .expect("the headline leads");
    let first = text
        .find("REPORT 2a9aaca3 approved: the parser change stands")
        .expect("the first verdict is shown");
    let second = text
        .find("REPORT 41b0fde1 fixed: resolved the interaction in src/b.rs")
        .expect("the second verdict is shown");
    let risk = text
        .find("RISK: the sandbox policy edit touched src/agent/sandbox.rs")
        .expect("the risk is shown");
    assert!(
        headline < first && first < second && second < risk,
        "the compact block must lead and the lines follow in order: {text}"
    );
    assert!(
        text.lines().count() >= 4,
        "each verdict gets its own line: {text}"
    );
}

/// The compact event is what every consumer reads, so it stays bounded: a
/// round reporting thousands of workers must not produce a transcript.
#[test]
fn the_completion_event_stays_bounded() {
    // A round big enough that the ceiling is the only thing that stops it.
    let huge: String = (0..4000)
        .map(|i| {
            format!("REPORT w{i:05} approved: integrated branch {i} with the full gate green\n")
        })
        .collect();
    let verdicts = parse_verdict_lines(&(String::from("REPORT\ndone: all\nrisks: none\n") + &huge));
    let payload = serde_json::to_string(&verdicts).expect("verdicts serialize");
    assert!(
        payload.len() <= VERDICT_BYTES + 128,
        "the stored payload must stay bounded, got {} bytes for {} lines",
        payload.len(),
        verdicts.lines().len()
    );
    assert!(
        verdicts.lines().len() < 4000,
        "a 4000-line round must be truncated: {} lines",
        verdicts.lines().len()
    );
    assert!(
        verdicts
            .lines()
            .iter()
            .any(|line| line.contains("verdict lines dropped")),
        "a truncated round says so: {:?}",
        verdicts.risks
    );
}

/// The truncation notice counts what it dropped, so a bounded payload never
/// reads as a complete round.
#[test]
fn a_truncated_round_names_what_it_dropped() {
    let message: String = (0..500)
        .map(|i| format!("REPORT w{i:05} approved: branch {i} integrated cleanly\n"))
        .collect();
    let verdicts = parse_verdict_lines(&message);
    let notice = verdicts
        .risks
        .last()
        .expect("a truncated round carries a notice");
    assert!(notice.contains("verdict lines dropped"), "{notice}");
    let dropped: usize = notice
        .split('[')
        .nth(1)
        .and_then(|rest| rest.split(' ').next())
        .and_then(|n| n.parse().ok())
        .expect("the notice names a count");
    assert_eq!(dropped + verdicts.workers.len(), 500, "{notice}");
}

/// The registry row is the cross-process record, so the verdicts a row carries
/// are what a cold reader (`status`, `review` after eviction) can show.
#[tokio::test]
async fn the_row_carries_the_verdicts_and_a_cold_reader_renders_them() {
    let scratch = common::TempDir::new_in_tmp("verdict-row");
    let root = scratch.path().to_path_buf();
    let pool = WorkerPool::with_scratch(
        1,
        "http://127.0.0.1:1".to_string(),
        "test".to_string(),
        ScratchRoot::new(root.clone()),
    );
    let verdicts = expected_verdicts();
    let meta = meta_with_verdicts("c1", Some(verdicts.clone()));
    let row = meta.entry(
        "test-model",
        RegistryStatus::Completed,
        6,
        12,
        "cargo test",
        None,
    );
    save_registry_entry_in(&ScratchRoot::new(root.clone()), &row);
    drop(pool);

    let entry = load_registry_entry_in(&ScratchRoot::new(root), "c1").expect("the row survives");
    assert_eq!(entry.verdicts.as_ref(), Some(&verdicts));

    // The same row rendered as a compact view -- what a `watch`/`status` read
    // of another process sees -- carries the lines as an array.
    let view = watch::registry_snapshot(&entry, entry.updated_at);
    let lines = view["verdicts"]
        .as_array()
        .expect("the view carries the verdict lines as an array");
    let shown: Vec<&str> = lines.iter().filter_map(|line| line.as_str()).collect();
    assert_eq!(
        shown,
        vec![
            "REPORT 2a9aaca3 approved: the parser change stands",
            "REPORT 41b0fde1 fixed: resolved the interaction in src/b.rs",
            "RISK: the sandbox policy edit touched src/agent/sandbox.rs",
        ],
        "{view}"
    );
}

/// A worker that never reported a verdict carries none: the field is absent,
/// not an empty block a reader has to interpret.
#[tokio::test]
async fn a_worker_without_verdicts_carries_none() {
    let scratch = common::TempDir::new_in_tmp("verdict-none");
    let root = scratch.path().to_path_buf();
    let meta = meta_with_verdicts("c2", None);
    let row = meta.entry(
        "test-model",
        RegistryStatus::Completed,
        1,
        4,
        "cargo test",
        None,
    );
    save_registry_entry_in(&ScratchRoot::new(root.clone()), &row);
    let entry = load_registry_entry_in(&ScratchRoot::new(root), "c2").expect("the row exists");
    assert!(entry.verdicts.is_none());
    let json = serde_json::to_value(&entry).expect("the row serializes");
    assert!(
        json.get("verdicts").is_none(),
        "an absent verdict is omitted, not null: {json}"
    );
}

/// A row written before the field existed still reads: the default is "no
/// verdicts recorded", not a parse failure that loses the whole row.
#[test]
fn a_row_without_the_field_still_reads() {
    let row: WorkerRegistryEntry = serde_json::from_value(serde_json::json!({
        "id": "c3",
        "pid": 1,
        "task": "integrate",
        "model": "test-model",
        "status": "completed",
        "step": 2,
        "max_turns": 8,
        "last_command": "cargo test",
        "started_at": 1,
        "updated_at": 2,
    }))
    .expect("an older row still parses");
    assert!(row.verdicts.is_none());
    assert_eq!(row.id, "c3");
}

/// The finished state carries the verdicts next to the round's headline, so
/// the in-memory record and the row agree.
#[test]
fn the_completed_state_carries_the_verdicts_beside_the_report() {
    let verdicts = expected_verdicts();
    let state = WorkerState::Completed {
        turns: 6,
        diff: String::new(),
        summary: "integrated 2 branches".to_string(),
        completed_at: 1_700_000_000,
        artifacts: Vec::new(),
        branch: Some("worker-c1".to_string()),
        verified: Some(true),
        metrics: Default::default(),
        revision: 0,
        report: Some(WorkerReport {
            done: "integrated 2 branches, gate green".to_string(),
            files: String::new(),
            tests: String::new(),
            risks: "none".to_string(),
        }),
        verdicts: Some(verdicts.clone()),
    };
    let json = serde_json::to_value(&state).expect("state serializes");
    let details = json
        .get("details")
        .expect("a tagged state keeps its payload under `details`");
    let back: Option<WorkerVerdicts> = serde_json::from_value(
        details
            .get("verdicts")
            .cloned()
            .expect("verdicts serialize"),
    )
    .expect("the verdict payload deserializes");
    assert_eq!(back.as_ref(), Some(&verdicts));
}

/// `review <consolidator>` is the verb the orchestrator reaches for after the
/// completion event, so it carries the same per-worker verdicts: the JSON view
/// as an array and the rendered text as one line per worker.
#[tokio::test]
async fn review_of_a_consolidator_carries_its_verdicts() {
    let owned = common::IsolatedPool::new(2, "verdict-review");
    let scratch = common::TempDir::new_in_tmp("verdict-review-repo");
    common::git(scratch.path(), &["init", "-q", "-b", "main", "."]);
    common::git(scratch.path(), &["config", "user.name", "verdict-test"]);
    common::git(
        scratch.path(),
        &["config", "user.email", "verdict@localhost"],
    );
    std::fs::write(scratch.path().join("seed.txt"), "seed\n").expect("seed");
    common::git(scratch.path(), &["add", "-A"]);
    common::git(scratch.path(), &["commit", "-qm", "seed"]);

    let verdicts = expected_verdicts();
    let meta = meta_with_verdicts("c-review", Some(verdicts.clone()));
    save_registry_entry_in(
        &owned.root(),
        &meta.entry(
            "test-model",
            RegistryStatus::Completed,
            6,
            12,
            "cargo test",
            None,
        ),
    );
    owned
        .pool
        .__test_insert_worker(consolidator_record(
            "c-review",
            scratch.path().to_path_buf(),
            Some(verdicts.clone()),
        ))
        .await;

    let server = mini_swe_mcp::mcp::McpServer::new(owned.pool.clone(), "test-model".into());
    let ctx = mini_swe_mcp::mcp::ConnectionContext {
        agent_id: Some(OWNER.to_string()),
        ..mini_swe_mcp::mcp::ConnectionContext::stdio()
    };
    let payload = server
        .execute_tool_for(
            "worker",
            serde_json::json!({ "action": "review", "worker_id": "c-review", "diff": "none" }),
            &ctx,
        )
        .await
        .expect("the owner may review its own consolidator");

    let lines: Vec<&str> = payload["verdicts"]
        .as_array()
        .unwrap_or_else(|| panic!("the payload must carry the verdict lines: {payload}"))
        .iter()
        .filter_map(|line| line.as_str())
        .collect();
    assert_eq!(lines, expected_verdicts().lines(), "{payload}");

    let text = mini_swe_mcp::cli::format::format_output("review", &payload);
    for line in expected_verdicts().lines() {
        assert!(
            text.contains(line),
            "the rendered review must show {line}: {text}"
        );
    }
    assert!(
        text.find("Summary:").unwrap() < text.find("REPORT 2a9aaca3").unwrap(),
        "the compact headline leads: {text}"
    );
}

/// A consolidator record as the phase loop leaves it: the completed state, the
/// report and the verdicts, on a branch `review` can measure.
fn consolidator_record(
    id: &str,
    repo_path: std::path::PathBuf,
    verdicts: Option<WorkerVerdicts>,
) -> mini_swe_mcp::pool::WorkerRecord {
    let mut meta = meta_with_verdicts(id, verdicts.clone());
    meta.repo_path = Some(repo_path.to_string_lossy().into_owned());
    mini_swe_mcp::pool::WorkerRecord {
        id: id.to_string(),
        task: "integrate the round".to_string(),
        model: "test-model".to_string(),
        owner: OWNER.to_string(),
        state: WorkerState::Completed {
            turns: 6,
            diff: String::new(),
            summary: "integrated 2 branches, gate green".to_string(),
            completed_at: 1_700_000_000,
            artifacts: Vec::new(),
            branch: Some(format!("worker-{id}")),
            verified: Some(true),
            metrics: Default::default(),
            revision: 0,
            report: Some(WorkerReport {
                done: "integrated 2 branches, gate green".to_string(),
                files: "src/a.rs, src/b.rs".to_string(),
                tests: "cargo test: passed".to_string(),
                risks: "none".to_string(),
            }),
            verdicts,
        },
        metrics: Default::default(),
        logs: mini_swe_mcp::pool::LogBuffer::new(),
        pending_steer: Vec::new(),
        resume_tx: None,
        handle: None,
        revision: 0,
    }
}

/// The whole path, not just the parser: a consolidator dispatched against a
/// scripted model closes with the round's block and its per-worker verdicts,
/// and the finished state and the registry row both carry them. This is the
/// signal the task is about -- before it, the verdicts existed only in the
/// consolidator's history log.
#[tokio::test]
async fn a_consolidator_run_stores_its_verdicts_on_the_state_and_the_row() {
    use tokio::io::AsyncWriteExt;
    use tokio::net::TcpListener;

    // One scripted completion: the round's block, then one line per worker and
    // the risks, in the same message.
    let content = [
        "REPORT",
        "done: integrated 2 branches, gate green",
        "files: src/a.rs, src/b.rs",
        "tests: cargo test: passed",
        "risks: none",
        "REPORT 2a9aaca3 approved: the parser change stands",
        "REPORT 41b0fde1 fixed: resolved the interaction in src/b.rs",
        "RISK: the sandbox policy edit touched src/agent/sandbox.rs",
    ]
    .join("\n");
    let command = format!("echo {COMPLETION_SENTINEL}");
    let body = serde_json::json!({
        "choices": [{
            "delta": {
                "content": content,
                "tool_calls": [{
                    "index": 0,
                    "id": "call-0",
                    "function": {
                        "name": "bash",
                        "arguments": serde_json::json!({ "command": command }).to_string()
                    }
                }]
            }
        }]
    });
    let frame = format!("data: {body}\n\n");
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let base_url = format!("http://{}", listener.local_addr().expect("addr"));
    let server = tokio::spawn(async move {
        while let Ok((mut socket, _)) = listener.accept().await {
            let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n";
            if socket.write_all(head.as_bytes()).await.is_err() {
                continue;
            }
            if socket.write_all(frame.as_bytes()).await.is_err() {
                continue;
            }
            let _ = socket.flush().await;
            let _ = socket.shutdown().await;
        }
    });

    let repo = common::TempDir::new_in_tmp("verdict-run-repo");
    common::git(repo.path(), &["init", "-q", "-b", "main", "."]);
    common::git(repo.path(), &["config", "user.name", "verdict-run"]);
    common::git(repo.path(), &["config", "user.email", "run@localhost"]);
    std::fs::write(repo.path().join("seed.txt"), "seed\n").expect("seed");
    common::git(repo.path(), &["add", "-A"]);
    common::git(repo.path(), &["commit", "-qm", "seed"]);

    let scratch = common::TempDir::new_in_tmp("verdict-run-pool");
    let pool = WorkerPool::with_scratch(
        2,
        base_url,
        "test-key".to_string(),
        ScratchRoot::new(scratch.path().to_path_buf()),
    );
    let id = pool
        .dispatch_with_role(
            OWNER.to_string(),
            "integrate the round".to_string(),
            "test-model".to_string(),
            None,
            repo.path().to_path_buf(),
            4,
            Some("round-1".to_string()),
            None,
            false,
            None,
            Vec::new(),
            WorkerRole::Consolidate,
        )
        .await
        .expect("dispatch");

    let state = tokio::time::timeout(std::time::Duration::from_secs(30), async {
        let mut changes = pool.subscribe_changes();
        loop {
            if let Some(state) = pool.get_worker_state(&id).await
                && matches!(
                    state,
                    WorkerState::Completed { .. } | WorkerState::Failed { .. }
                )
            {
                return state;
            }
            changes.changed().await.expect("pool notification");
        }
    })
    .await
    .expect("the consolidator finishes");

    server.abort();
    let WorkerState::Completed {
        verdicts, report, ..
    } = &state
    else {
        panic!("the consolidator must complete, got {state:?}");
    };
    assert_eq!(
        verdicts.as_ref(),
        Some(&expected_verdicts()),
        "the run must store the round's verdicts on the completed state"
    );
    assert_eq!(
        report.as_ref().map(|report| report.done.as_str()),
        Some("integrated 2 branches, gate green"),
        "the compact headline is unchanged"
    );

    let entry = load_registry_entry_in(pool.scratch_root(), &id).expect("the terminal row exists");
    assert_eq!(
        entry.verdicts.as_ref(),
        Some(&expected_verdicts()),
        "the registry row carries the verdicts too"
    );
    assert_eq!(
        entry.report.as_ref().map(|report| report.done.as_str()),
        Some("integrated 2 branches, gate green"),
    );
}

/// A verdict the harness acts on and a verdict it displays must be the same
/// line. The absorption path already peels markdown before reading a verdict
/// (`parse_consolidator_verdicts`), so a consolidator that wraps its verdicts
/// in a bullet has its workers absorbed; without the same peel on the recorded
/// path the round then displays as having decided nothing about them.
#[test]
fn a_markdown_wrapped_verdict_is_shown_as_the_harness_read_it() {
    let message = [
        "REPORT",
        "done: integrated 1 branch",
        "risks: none",
        "- REPORT 2a9aaca3 fixed: applied the correction on its branch",
        "> REPORT 41b0fde1 approved: the parser change stands",
    ]
    .join("\n");

    let verdicts = parse_verdict_lines(&message);
    assert_eq!(
        verdicts.workers,
        vec![
            "REPORT 2a9aaca3 fixed: applied the correction on its branch".to_string(),
            "REPORT 41b0fde1 approved: the parser change stands".to_string(),
        ],
        "a bullet or a quote is markdown, not part of the verdict: {verdicts:?}"
    );

    // Each stored line is the bare verdict line, so re-reading the recorded
    // payload yields the same verdicts: what the round displays is what a
    // consumer can act on, not markdown it has to strip again.
    let text = completion_event(Some(verdicts));
    for line in [
        "REPORT 2a9aaca3 fixed: applied the correction on its branch",
        "REPORT 41b0fde1 approved: the parser change stands",
    ] {
        assert!(
            text.lines().any(|shown| shown == line),
            "the completion event shows the bare verdict line: {text}"
        );
    }
}

/// The budget is a property of the type, not only of the parser that builds
/// it. A registry row is plain JSON in a directory another local user can
/// write, so an oversized array read back off disk must be bounded exactly like
/// one written by a run -- otherwise the promise the type makes ("at most
/// `VERDICT_BYTES` survive", "never an unbounded registry row") holds on only
/// one of the two paths.
#[test]
fn an_oversized_row_is_bounded_when_it_is_read_back() {
    let huge: Vec<String> = (0..2_000)
        .map(|i| format!("REPORT w{i:05} approved: {}", "y".repeat(400)))
        .collect();

    let row = serde_json::json!({
        "id": "c-huge",
        "pid": 1,
        "task": "integrate the round",
        "model": "test-model",
        "status": "completed",
        "step": 3,
        "max_turns": 8,
        "last_command": "cargo test",
        "started_at": 1,
        "updated_at": 2,
        "verdicts": huge,
    });
    let entry: mini_swe_mcp::pool::WorkerRegistryEntry =
        serde_json::from_value(row).expect("the row parses");
    let verdicts = entry.verdicts.expect("the row carries verdicts");

    let rendered: usize = verdicts.lines().iter().map(|line| line.len() + 1).sum();
    assert!(
        rendered <= VERDICT_BYTES,
        "a read-back row stays inside the budget: {rendered} > {VERDICT_BYTES}"
    );
    // Bounded is not silent: the reader is told lines went missing.
    assert!(
        verdicts
            .lines()
            .iter()
            .any(|line| line.contains("verdict lines dropped")),
        "a truncated row says so: {:?}",
        verdicts.lines()
    );
}
