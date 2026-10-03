//! Structured completion reports: the parser contract, the one follow-up the
//! harness asks for, and what a `watch` event renders.
//!
//! A completion used to be read off the first line of the worker's last chat
//! message, so an orchestrator had to shell out to `git diff --stat` to learn
//! what a finished worker had done. The worker now writes a `REPORT` block
//! before the sentinel, the harness asks once when it is missing, and the
//! completion event carries the report plus the per-file diff.

mod common;

use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::Mutex;

use mini_swe_mcp::cli::watch;
use mini_swe_mcp::pool::{COMPLETION_SENTINEL, WorkerPool, WorkerState};
use mini_swe_mcp::worktree::ScratchRoot;

const TEST_OWNER: &str = "report-agent";

/// One scripted SSE body: the frames of a single provider turn.
type ScriptedTurn = Vec<String>;

#[derive(Clone)]
struct CapturedRequests(Arc<Mutex<Vec<Value>>>);

impl CapturedRequests {
    async fn all(&self) -> Vec<Value> {
        self.0.lock().await.clone()
    }
}

/// A loopback server that answers one `POST /chat/completions` per connection
/// with the next scripted turn, recording every request body it saw.
struct ScriptedSseServer {
    base_url: String,
    requests: CapturedRequests,
    task: tokio::task::JoinHandle<()>,
}

impl Drop for ScriptedSseServer {
    fn drop(&mut self) {
        self.task.abort();
    }
}

impl ScriptedSseServer {
    async fn spawn(turns: Vec<ScriptedTurn>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr: SocketAddr = listener.local_addr().expect("addr");
        let requests = CapturedRequests(Arc::new(Mutex::new(Vec::new())));
        let sink = requests.clone();
        let script = Arc::new(std::sync::Mutex::new(turns.into_iter()));
        let task = tokio::spawn(async move {
            loop {
                let Ok((mut socket, _)) = listener.accept().await else {
                    return;
                };
                let turn = script.lock().expect("script lock").next();
                let sink = sink.clone();
                tokio::spawn(async move {
                    let body = read_http_request(&mut socket).await;
                    if let Some(body) = body
                        && let Ok(value) = serde_json::from_str::<Value>(&body)
                    {
                        sink.0.lock().await.push(value);
                    }
                    write_sse(&mut socket, turn.as_deref()).await;
                });
            }
        });
        Self {
            base_url: format!("http://{addr}"),
            requests,
            task,
        }
    }

    /// A turn whose tool call runs `command`, with `content` as the prose the
    /// model wrote alongside it — which is where a REPORT block lives.
    fn turn(call_id: &str, content: &str, command: &str) -> ScriptedTurn {
        let arguments = json!({ "command": command }).to_string();
        vec![frame(&json!({
            "choices": [{
                "delta": {
                    "content": content,
                    "tool_calls": [{
                        "index": 0,
                        "id": call_id,
                        "function": { "name": "bash", "arguments": arguments }
                    }]
                }
            }]
        }))]
    }

    /// A completion turn whose prose carries no REPORT block.
    fn bare_completion(call_id: &str) -> ScriptedTurn {
        Self::turn(
            call_id,
            "Now I'll make the edits.",
            &format!("echo {COMPLETION_SENTINEL}"),
        )
    }

    /// A turn whose prose carries `content` and no bash command.
    fn content_only(content: &str) -> ScriptedTurn {
        vec![frame(&json!({
            "choices": [{ "delta": { "content": content } }]
        }))]
    }

    /// A completion turn whose prose carries a REPORT block.
    fn reported_completion(call_id: &str) -> ScriptedTurn {
        Self::turn(
            call_id,
            "REPORT\ndone: Fixed the parser\nfiles: src/a.rs, src/b.rs\ntests: cargo test: 4 passed\nrisks: none",
            &format!("echo {COMPLETION_SENTINEL}"),
        )
    }
}

fn frame(value: &Value) -> String {
    format!("data: {value}\n\n")
}

async fn write_sse(socket: &mut tokio::net::TcpStream, turn: Option<&[String]>) {
    let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n";
    if socket.write_all(head.as_bytes()).await.is_err() {
        return;
    }
    if let Some(chunks) = turn {
        for chunk in chunks {
            if socket.write_all(chunk.as_bytes()).await.is_err() {
                return;
            }
        }
    }
    let _ = socket.flush().await;
}

async fn read_http_request(socket: &mut tokio::net::TcpStream) -> Option<String> {
    let mut data: Vec<u8> = Vec::new();
    let mut probe = [0u8; 4096];
    while !data.windows(4).any(|w| w == b"\r\n\r\n") {
        match socket.read(&mut probe).await {
            Ok(0) | Err(_) => return None,
            Ok(n) => data.extend_from_slice(&probe[..n]),
        }
    }
    let head_end = data.windows(4).position(|w| w == b"\r\n\r\n")? + 4;
    let head = String::from_utf8_lossy(&data[..head_end]).to_string();
    let content_length: usize = head
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.trim()
                .eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse().ok())?
        })
        .unwrap_or(0);
    while data.len() - head_end < content_length {
        match socket.read(&mut probe).await {
            Ok(0) | Err(_) => break,
            Ok(n) => data.extend_from_slice(&probe[..n]),
        }
    }
    Some(String::from_utf8_lossy(&data[head_end..head_end + content_length]).to_string())
}

/// A git repository with one commit, the way a dispatch finds it.
fn repo(root: &Path, tag: &str) -> std::path::PathBuf {
    let repo = root.join(tag);
    std::fs::create_dir_all(&repo).expect("mkdir");
    common::git(&repo, &["init", "-q", "-b", "main", "."]);
    common::git(&repo, &["config", "user.name", "report-test"]);
    common::git(&repo, &["config", "user.email", "report@localhost"]);
    std::fs::write(repo.join("seed.txt"), "seed\n").expect("seed");
    common::git(&repo, &["add", "-A"]);
    common::git(&repo, &["commit", "-qm", "seed"]);
    repo
}

/// Dispatch one worker against `server` and wait for it to finish.
async fn dispatch_and_wait(
    server: &ScriptedSseServer,
    repo: &std::path::Path,
    max_turns: usize,
) -> (common::TempDir, WorkerPool, String, WorkerState) {
    let scratch = common::TempDir::new_in_tmp("report-pool");
    let root = scratch.path().to_path_buf();
    let pool = WorkerPool::with_scratch(
        1,
        server.base_url.clone(),
        "test-key".to_string(),
        ScratchRoot::new(root),
    );
    let worker_id = pool
        .dispatch(
            TEST_OWNER.to_string(),
            "write a structured report".to_string(),
            "test-model".to_string(),
            None,
            repo.to_path_buf(),
            max_turns,
            Some("report".to_string()),
            None,
            false,
            None,
            Vec::new(),
        )
        .await
        .expect("dispatch");
    for _ in 0..600 {
        if let Some(state) = pool.get_worker_state(&worker_id).await
            && matches!(
                state,
                WorkerState::Completed { ..
                verdicts: None, } | WorkerState::Failed { .. }
            )
        {
            return (scratch, pool, worker_id, state);
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("worker {worker_id} never finished");
}

/// A completion with no report costs exactly one follow-up turn, and the
/// report the worker then writes is what the finished state carries.
#[tokio::test]
async fn a_completion_without_a_report_is_asked_once() {
    let dir = common::TempDir::new_in_tmp("report-ask");
    let repo = repo(dir.path(), "ask");
    let server = ScriptedSseServer::spawn(vec![
        ScriptedSseServer::bare_completion("call_1"),
        ScriptedSseServer::reported_completion("call_2"),
    ])
    .await;

    let (_scratch, pool, id, state) = dispatch_and_wait(&server, &repo, 1).await;

    let WorkerState::Completed {
        report, summary, ..
    } = state
    else {
        panic!("the worker must complete, got {state:?}");
    };
    let report = report.expect("the follow-up must have produced a report");
    let entry = mini_swe_mcp::pool::load_registry_entry_in(pool.scratch_root(), &id)
        .expect("the terminal row must exist");
    assert_eq!(entry.report.as_ref(), Some(&report));
    assert_eq!(report.done, "Fixed the parser");
    assert_eq!(report.files, "src/a.rs, src/b.rs");
    assert_eq!(report.tests, "cargo test: 4 passed");
    assert_eq!(report.risks, "none");
    assert_eq!(
        summary, "Fixed the parser",
        "the report's done line is the summary every consumer reads"
    );

    let requests = server.requests.all().await;
    assert_eq!(
        requests.len(),
        2,
        "one completion, one follow-up: {requests:?}"
    );
    let follow_up = requests[1]["messages"]
        .as_array()
        .and_then(|messages| messages.last())
        .and_then(|message| message["content"].as_str())
        .unwrap_or_default();
    assert!(
        follow_up.contains("REPORT") && follow_up.contains("completion sentinel"),
        "the follow-up must ask for the block and the sentinel, got {follow_up:?}"
    );
}

/// The exact losing sequence: a completion with no block, the one follow-up,
/// and the block written in the prose of a turn that carries no command (the
/// sentinel only arrives in the turn after it). The block must survive to the
/// completed state.
#[tokio::test]
async fn a_report_in_a_follow_up_turn_before_the_sentinel_is_stored() {
    let dir = common::TempDir::new_in_tmp("report-late");
    let repo = repo(dir.path(), "late");
    let server = ScriptedSseServer::spawn(vec![
        ScriptedSseServer::bare_completion("call_1"),
        ScriptedSseServer::content_only(
            "REPORT\ndone: Fixed the parser\nfiles: src/a.rs, src/b.rs\ntests: cargo test: 4 passed\nrisks: none",
        ),
        ScriptedSseServer::turn(
            "call_3",
            "All gates pass. Final status and sentinel:",
            &format!("echo {COMPLETION_SENTINEL}"),
        ),
    ])
    .await;

    let (_scratch, _pool, _id, state) = dispatch_and_wait(&server, &repo, 10).await;

    let WorkerState::Completed {
        report, summary, ..
    } = state
    else {
        panic!("the worker must complete, got {state:?}");
    };
    let report = report.expect("the follow-up turn's block must be stored");
    assert_eq!(report.done, "Fixed the parser");
    assert_eq!(report.files, "src/a.rs, src/b.rs");
    assert_eq!(summary, "Fixed the parser");
}

/// A block written inside the bash command that also carries the sentinel,
/// with the line breaks spelled as `\n` the way `printf` takes them, is the
/// answer the harness must store.
#[tokio::test]
async fn a_report_delivered_inside_the_sentinel_command_is_stored() {
    let dir = common::TempDir::new_in_tmp("report-command");
    let repo = repo(dir.path(), "command");
    let command = format!(
        "printf 'REPORT\\ndone: From the command\\nfiles: src/x.rs\\ntests: cargo test: 4 passed\\nrisks: none\\n' && echo {COMPLETION_SENTINEL}"
    );
    let server = ScriptedSseServer::spawn(vec![ScriptedSseServer::turn(
        "call_1",
        "Now I'll make the edits.",
        &command,
    )])
    .await;

    let (_scratch, _pool, _id, state) = dispatch_and_wait(&server, &repo, 5).await;

    let WorkerState::Completed {
        report, summary, ..
    } = state
    else {
        panic!("the worker must complete, got {state:?}");
    };
    let report = report.expect("the block inside the sentinel command must be stored");
    assert_eq!(report.done, "From the command");
    assert_eq!(report.files, "src/x.rs");
    assert_eq!(report.risks, "none");
    assert_eq!(summary, "From the command");
}

/// A worker that still writes no report after being asked completes anyway,
/// with today's summary as the fallback.
#[tokio::test]
async fn a_second_reportless_completion_falls_back_to_the_summary() {
    let dir = common::TempDir::new_in_tmp("report-fallback");
    let repo = repo(dir.path(), "fallback");
    let server = ScriptedSseServer::spawn(vec![
        ScriptedSseServer::bare_completion("call_1"),
        ScriptedSseServer::bare_completion("call_2"),
    ])
    .await;

    let (_scratch, _pool, _id, state) = dispatch_and_wait(&server, &repo, 10).await;

    let WorkerState::Completed {
        report, summary, ..
    } = state
    else {
        panic!("the worker must complete, got {state:?}");
    };
    assert!(report.is_none(), "no block was ever written: {report:?}");
    assert_eq!(summary, "Now I'll make the edits.");

    let requests = server.requests.all().await;
    assert_eq!(
        requests.len(),
        2,
        "the worker is asked once, never twice: {requests:?}"
    );
}

/// The compact completion event names what changed, the files that changed and
/// the risks, within five lines.
#[test]
fn the_compact_completion_event_carries_done_files_and_risks() {
    let event = json!({
        "worker_id": "w-report",
        "event": "completed",
        "summary": "## Summary",
        "verified": true,
        "report": {
            "done": "Fix completion reporting",
            "files": "src/a.rs",
            "tests": "cargo test: passed",
            "risks": "Changes completion feedback"
        },
        "diff_stat": {"files": 1, "insertions": 3, "deletions": 2},
        "per_file": [{"path": "src/a.rs", "insertions": 3, "deletions": 2}],
        "step": 2,
        "max_turns": 5,
        "elapsed": 1,
        "task": "report probe"
    });
    let text = watch::render(&event);
    assert!(text.contains("Fix completion reporting"), "{text}");
    assert!(text.contains("files: src/a.rs (+3 -2)"), "{text}");
    assert!(
        text.contains("risks: Changes completion feedback"),
        "{text}"
    );
    assert!(text.lines().count() <= 5, "{text}");
}

/// A report that says `risks: none` shows no risk line at all.
#[test]
fn a_report_without_risks_shows_no_risk_line() {
    let event = json!({
        "worker_id": "w-clean",
        "event": "completed",
        "report": {
            "done": "Fix the parser",
            "files": "src/a.rs",
            "tests": "cargo test: passed",
            "risks": "none"
        },
        "per_file": [{"path": "src/a.rs", "insertions": 3, "deletions": 2}],
        "step": 1,
        "max_turns": 5,
        "elapsed": 1,
        "task": "probe"
    });
    let text = watch::render(&event);
    assert!(!text.contains("risks:"), "{text}");
    assert!(text.contains("Fix the parser"), "{text}");
}

/// A stalled worker's compact line names the health counter that moved, so the
/// orchestrator does not have to ask why it fired.
#[test]
fn a_stalled_event_names_the_counter_that_moved() {
    let event = json!({
        "worker_id": "w-stall",
        "event": "stalled",
        "time_since_last_step": 0,
        "moved_counters": ["stagnation_nudges=3"],
        "step": 4,
        "max_turns": 10,
        "elapsed": 30,
        "task": "probe"
    });
    let text = watch::render(&event);
    assert!(
        text.contains("no step for 0s | stagnation_nudges=3"),
        "{text}"
    );
    assert!(text.lines().count() <= 5, "{text}");
}

#[test]
fn json_channel_and_watch_events_keep_the_full_report_and_stats() {
    use mini_swe_mcp::mcp::{EventKind, WorkerSnapshot, WorkerView, channel_frame, diff_events};
    use mini_swe_mcp::pool::{WorkerMetrics, WorkerReport};
    let report = WorkerReport {
        done: "Fix the parser".into(),
        files: "src/a.rs".into(),
        tests: "cargo test: 4 passed".into(),
        risks: "Changes completion feedback".into(),
    };
    let state = WorkerState::Completed {
        turns: 1,
        diff: "diff --git a/src/a.rs b/src/a.rs\n--- a/src/a.rs\n+++ b/src/a.rs\n@@ -1 +1 @@\n-old\n+new\n".into(),
        summary: report.done.clone(),
        completed_at: 1,
        artifacts: vec![],
        branch: Some("worker-json".into()),
        verified: Some(true),
        metrics: WorkerMetrics::default(),
        revision: 0,
        report: Some(report.clone())
    verdicts: None,,
    };
    let mut view = json!({"worker_id":"json", "task":"probe"});
    watch::enrich_state(&mut view, &state);
    let event = watch::select_event(&view, None, 1).expect("completed event");
    let wire: Value = serde_json::from_str(&serde_json::to_string(&event).unwrap()).unwrap();
    assert_eq!(wire["report"], json!(report));
    assert_eq!(
        wire["per_file"],
        json!([{"path":"src/a.rs","insertions":1,"deletions":1}])
    );
    let summary = state.to_summary();
    assert_eq!(summary["report"], wire["report"]);
    assert_eq!(summary["per_file"], wire["per_file"]);
    let mut current = WorkerSnapshot::new();
    current.insert(
        "json".into(),
        WorkerView {
            worker_id: "json".into(),
            event: Some(EventKind::Completed),
            outcome: mini_swe_mcp::mcp::Outcome {
                report: Some(report),
                per_file: mini_swe_mcp::pool::file_stats_of_diff(match &state {
                    WorkerState::Completed { diff, ..
                    verdicts: None, } => diff,
                    _ => unreachable!(),
                }),
                ..Default::default()
            },
            ..Default::default()
        },
    );
    let events = diff_events(&WorkerSnapshot::new(), &current);
    let frame: Value = serde_json::from_str(&channel_frame(&events[0]).unwrap()).unwrap();
    assert_eq!(frame["params"]["report"], wire["report"]);
    assert_eq!(frame["params"]["per_file"], wire["per_file"]);
}

#[tokio::test]
async fn report_survives_eviction_in_status_review_and_collect() {
    use mini_swe_mcp::mcp::{ConnectionContext, McpServer};
    use mini_swe_mcp::pool::{
        LogBuffer, RegistryStatus, WorkerRecord, WorkerRegistryEntry, WorkerReport,
        save_registry_entry_in,
    };
    let owned = common::IsolatedPool::new(1, "report-evict");
    let report = WorkerReport {
        done: "Fix the parser".into(),
        files: "src/a.rs".into(),
        tests: "cargo test: 4 passed".into(),
        risks: "none".into(),
    };
    let entry = WorkerRegistryEntry {
        task: "probe".into(),
        status: RegistryStatus::Completed,
        step: 1,
        max_turns: 1,
        last_command: report.done.clone(),
        updated_at: mini_swe_mcp::pool::unix_timestamp(),
        report: Some(report.clone()),
        ..WorkerRegistryEntry::test_row("evicted-report", TEST_OWNER)
    };
    save_registry_entry_in(&owned.root(), &entry);
    owned
        .pool
        .__test_insert_worker(WorkerRecord {
            id: entry.id.clone(),
            task: entry.task.clone(),
            model: entry.model.clone(),
            owner: TEST_OWNER.into(),
            state: WorkerState::Completed {
                turns: 1,
                diff: String::new(),
                summary: report.done.clone(),
                completed_at: 0,
                artifacts: vec![],
                branch: None,
                verified: None,
                metrics: entry.metrics,
                revision: 0,
                report: Some(report.clone())
            verdicts: None,,
            },
            metrics: entry.metrics,
            logs: LogBuffer::new(),
            pending_steer: vec![],
            resume_tx: None,
            handle: None,
            revision: 0,
        })
        .await;
    assert_eq!(owned.pool.reap().await, vec![entry.id.clone()]);
    let server = McpServer::new(owned.pool.clone(), "test".into());
    let ctx = ConnectionContext {
        agent_id: Some(TEST_OWNER.into()),
        ..ConnectionContext::hub_connection(7)
    };
    for action in ["status", "review", "collect"] {
        let payload = server
            .execute_tool_for(
                "worker",
                json!({"action":action,"worker_id":entry.id}),
                &ctx,
            )
            .await
            .unwrap();
        let actual = if action == "status" {
            &payload["state"]["details"]["report"]
        } else {
            &payload["report"]
        };
        assert_eq!(actual, &json!(report), "{action}: {payload}");
    }
}
