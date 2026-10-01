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
}

impl ScriptedSseServer {
    async fn spawn(turns: Vec<ScriptedTurn>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr: SocketAddr = listener.local_addr().expect("addr");
        let requests = CapturedRequests(Arc::new(Mutex::new(Vec::new())));
        let sink = requests.clone();
        let script = Arc::new(std::sync::Mutex::new(turns.into_iter()));
        tokio::spawn(async move {
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
) -> (WorkerPool, String, WorkerState) {
    let scratch = common::TempDir::new_in_tmp("report-pool");
    let root = scratch.path().to_path_buf();
    std::mem::forget(scratch);
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
                WorkerState::Completed { .. } | WorkerState::Failed { .. }
            )
        {
            return (pool, worker_id, state);
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

    let (_pool, _id, state) = dispatch_and_wait(&server, &repo, 10).await;

    let WorkerState::Completed {
        report, summary, ..
    } = state
    else {
        panic!("the worker must complete, got {state:?}");
    };
    let report = report.expect("the follow-up must have produced a report");
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

    let (_pool, _id, state) = dispatch_and_wait(&server, &repo, 10).await;

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
