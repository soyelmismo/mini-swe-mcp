//! The read-only escalation, end to end: a worker whose dispatch names the
//! files to edit, and that only reads anyway, walks the detector's three steps
//! -- demand the edit, hand back the plan the task spells out, then park on the
//! orchestrator -- instead of spending its whole budget on reads.
//!
//! The thresholds are the defaults the pool ships (15, 30 and 45 read-only
//! turns), so these tests read no environment and no other test can observe
//! their effect: they pay for that with a turn budget sized to reach all three.

mod common;

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::time::Duration;

use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Mutex;

use mini_swe_mcp::pool::{COMPLETION_SENTINEL, WorkerMetrics, WorkerPool, WorkerState};

/// Owner recorded for the worker this test dispatches: the escalation is what
/// is under test, not the per-agent ownership check.
const TEST_OWNER: &str = "test-agent";

/// The read-only turn the third step fires on: the default pause threshold is
/// three times the default nudge threshold of 15. The tests below read these
/// defaults rather than overriding them through the environment, so they leave
/// no process-global state behind for another test to observe.
const PAUSE_TURN: usize = 45;

/// The dispatch under test. It names a file and a function, which is what arms
/// the read-only detector and what the plan half of the escalation quotes back.
const TASK: &str = "Add the plan to `fn check_read_only` in src/lib.rs.";

/// The three read-only thresholds, pulled down so the whole escalation runs
/// A read-only turn: a command that leaves the worktree exactly as it found it,
/// so the detector keeps counting the streak.
fn read_turn(n: usize) -> String {
    format!("sed -n '1,{}p' README.md", n % 3 + 1)
}

/// An editing turn: a command that grows the file, so the worktree sample
/// changes and the read-only streak starts over.
fn edit_turn(n: usize) -> String {
    format!("printf 'x{n}\\n' >> lib.rs")
}

// ----------
// Scripted SSE server
// ----------

/// A loopback server that answers one scripted turn per connection and captures
/// every request body, so a test can read the conversation the worker was given.
struct ScriptedServer {
    base_url: String,
    requests: Arc<Mutex<Vec<Value>>>,
}

impl ScriptedServer {
    /// Answer each turn with the next scripted body, and once the script runs
    /// out keep editing with a distinct command: a worker steered past its
    /// script must still get an answer every turn instead of hanging on a
    /// silent socket.
    async fn spawn(turns: Vec<String>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind the scripted server on loopback");
        let addr = listener.local_addr().expect("local addr");
        let requests = Arc::new(Mutex::new(Vec::new()));
        let captured = requests.clone();
        tokio::spawn(async move {
            let mut next = 0usize;
            loop {
                let Ok((mut socket, _)) = listener.accept().await else {
                    return;
                };
                let requests = captured.clone();
                let turn = match turns.get(next) {
                    Some(turn) => turn.clone(),
                    // Past the script the worker keeps editing, one distinct
                    // command per turn: the fallback never repeats, so the
                    // repetition guard is not what the test measures.
                    None => bash_turn(&format!("call_tail_{next}"), &edit_turn(next)),
                };
                next += 1;
                tokio::spawn(async move {
                    if let Some(body) = read_request(&mut socket).await
                        && let Ok(value) = serde_json::from_str::<Value>(&body)
                    {
                        requests.lock().await.push(value);
                    }
                    write_sse(&mut socket, &turn).await;
                });
            }
        });
        Self {
            base_url: format!("http://{addr}"),
            requests,
        }
    }

    async fn requests(&self) -> Vec<Value> {
        self.requests.lock().await.clone()
    }
}

/// One SSE turn: a tool call running `command`.
fn bash_turn(call_id: &str, command: &str) -> String {
    let arguments = json!({ "command": command }).to_string();
    format!(
        "data: {}\n\n",
        json!({
            "choices": [{
                "delta": {
                    "content": "",
                    "tool_calls": [{
                        "index": 0,
                        "id": call_id,
                        "function": { "name": "bash", "arguments": arguments }
                    }]
                }
            }]
        })
    )
}

/// The completion turn, carrying the REPORT block the system prompt requires.
/// The completion turn. Its prose carries the REPORT block the system prompt
/// requires, so the scripted worker is a compliant one and the harness never
/// spends a turn asking for a report it would not get.
fn completion_turn(call_id: &str) -> String {
    let report = "REPORT\ndone: scripted completion\nfiles: lib.rs\ntests: none\nrisks: none";
    let arguments = json!({ "command": format!("echo {COMPLETION_SENTINEL}") }).to_string();
    format!(
        "data: {}\n\n",
        json!({
            "choices": [{
                "delta": {
                    "content": report,
                    "tool_calls": [{
                        "index": 0,
                        "id": call_id,
                        "function": { "name": "bash", "arguments": arguments }
                    }]
                }
            }]
        })
    )
}

async fn write_sse(socket: &mut TcpStream, turn: &str) {
    let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n";
    if socket.write_all(head.as_bytes()).await.is_err() {
        return;
    }
    if socket.write_all(turn.as_bytes()).await.is_err() {
        return;
    }
    let _ = socket.flush().await;
    let _ = socket.write_all(b"data: [DONE]\n\n").await;
    let _ = socket.flush().await;
    let _ = socket.shutdown().await;
}

async fn read_request(socket: &mut TcpStream) -> Option<String> {
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

// ----------
// Scratch repository
// ----------

/// A throwaway git repository the worker is dispatched against.
struct TestRepo {
    dir: PathBuf,
}

impl TestRepo {
    fn new(tag: &str) -> Self {
        let dir = std::env::temp_dir().join(common::unique_suffix(&format!("read-only-{tag}")));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create scratch repo");
        let dir = dir.canonicalize().expect("canonicalize scratch repo");
        run_git(&dir, &["init", "-b", "master"]);
        run_git(&dir, &["config", "user.name", "mini-swe-test"]);
        run_git(&dir, &["config", "user.email", "test@localhost"]);
        std::fs::write(dir.join("README.md"), "# scratch\n").expect("seed file");
        std::fs::write(dir.join("lib.rs"), "fn main() {}\n").expect("seed file");
        run_git(&dir, &["add", "."]);
        run_git(&dir, &["commit", "-m", "baseline"]);
        Self { dir }
    }

    fn path(&self) -> &Path {
        &self.dir
    }
}

impl Drop for TestRepo {
    fn drop(&mut self) {
        // A worker leases build directories keyed by this repo's hash; they are
        // filed next to the scratch base, so removing the repo has to take them.
        mini_swe_mcp::cache::remove_build_dir_leases(&self.dir);
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn run_git(dir: &Path, args: &[&str]) {
    let output = Command::new("git")
        .current_dir(dir)
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("git {args:?}: {e}"));
    assert!(
        output.status.success(),
        "git {args:?} failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

// ----------
// Worker helpers
// ----------

async fn wait_for_paused(pool: &WorkerPool, worker_id: &str) -> Option<String> {
    for _ in 0..600 {
        match pool.get_worker_state(worker_id).await {
            Some(WorkerState::Paused { question, .. }) => return Some(question),
            Some(WorkerState::Completed { .. } | WorkerState::Failed { .. }) => return None,
            _ => {}
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    None
}

async fn wait_for_terminal(pool: &WorkerPool, worker_id: &str) -> WorkerState {
    // The last state seen is reported on timeout: "never finished" is only
    // actionable when it says what the worker was doing instead.
    let mut last = None;
    for _ in 0..600 {
        if let Some(state) = pool.get_worker_state(worker_id).await {
            match state {
                WorkerState::Completed { .. }
                | WorkerState::Failed { .. }
                | WorkerState::Exhausted { .. } => return state,
                other => last = Some(format!("{other:?}")),
            }
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("worker {worker_id} did not reach a terminal state; last: {last:?}");
}

fn metrics_of(state: &WorkerState) -> WorkerMetrics {
    match state {
        WorkerState::Completed { metrics, .. } => *metrics,
        other => panic!("expected a completed worker, got {other:?}"),
    }
}

/// Every user message of `request`, in order.
fn user_messages(request: &Value) -> Vec<String> {
    request["messages"]
        .as_array()
        .expect("messages array")
        .iter()
        .filter(|m| m["role"] == json!("user"))
        .map(|m| m["content"].as_str().unwrap_or_default().to_string())
        .collect()
}

async fn dispatch(pool: &WorkerPool, repo: &Path, max_turns: usize) -> String {
    pool.dispatch(
        TEST_OWNER.to_string(),
        TASK.to_string(),
        "test-model".to_string(),
        None,
        repo.to_path_buf(),
        max_turns,
        Some("read-only-escalation".to_string()),
        None,
        false,
        None,
        Vec::new(),
    )
    .await
    .expect("dispatch the worker")
}

/// A worker that only reads past the first nudge is handed the plan its own
/// task spells out, and a worker that ignores that too is parked on the
/// orchestrator with a question naming what it read.
#[tokio::test]
async fn an_ignored_nudge_carries_the_plan_and_then_pauses_the_worker() {
    let repo = TestRepo::new("escalate");
    // The default thresholds nudge at 15 read-only turns, carry the plan at 30
    // and park at 45, so the script reads past all three and leaves turns over
    // for the worker to answer the orchestrator.
    let mut turns: Vec<String> = (1..=PAUSE_TURN + 4)
        .map(|n| bash_turn(&format!("call_{n}"), &read_turn(n)))
        .collect();
    turns.push(completion_turn("call_done"));
    let server = ScriptedServer::spawn(turns).await;

    let scratch = common::TempDir::new_in_tmp("read-only-pool");
    let pool = WorkerPool::with_scratch(
        1,
        server.base_url.clone(),
        "test-key".to_string(),
        mini_swe_mcp::worktree::ScratchRoot::new(scratch.path()),
    );
    let _scratch = scratch;
    let worker_id = dispatch(&pool, repo.path(), PAUSE_TURN + 12).await;

    let question = wait_for_paused(&pool, &worker_id)
        .await
        .expect("a worker that ignores the plan must park on the orchestrator");
    assert!(
        question.contains("read-only turns"),
        "the pause must name the streak, got {question:?}"
    );
    assert!(
        question.contains("has not written a change"),
        "the pause must say the worker has not edited, got {question:?}"
    );
    assert!(
        question.contains("sed -n"),
        "the pause must summarise what the worker read, got {question:?}"
    );
    assert!(
        question.contains("src/lib.rs"),
        "the pause must quote the task, got {question:?}"
    );

    // The plan reached the model before the pause: it is a user message of an
    // earlier turn, naming the file and the function the task named.
    let requests = server.requests().await;
    let plan_seen = requests.iter().any(|request| {
        user_messages(request)
            .iter()
            .any(|m| m.contains("Edit now.") && m.contains("src/lib.rs (fn check_read_only)"))
    });
    assert!(
        plan_seen,
        "the second nudge must carry the plan the task names"
    );

    // The orchestrator decides, and the worker carries on from its answer.
    pool.steer(&worker_id, "write the edit in lib.rs now".to_string())
        .await
        .expect("steer the paused worker");
    let state = wait_for_terminal(&pool, &worker_id).await;
    assert!(
        matches!(state, WorkerState::Completed { .. }),
        "the worker must finish once the orchestrator guides it, got {state:?}"
    );
    let metrics = metrics_of(&state);
    assert!(
        metrics.stagnation_nudges >= 3,
        "all three steps must be counted, got {metrics:?}"
    );
    assert_eq!(
        metrics.loop_pauses, 1,
        "the read-only pause is counted once, got {metrics:?}"
    );
}

/// The first nudge alone still gets the worker editing, and a dispatch that
/// names no file never reaches any of the three steps.
#[tokio::test]
async fn a_worker_that_edits_after_the_nudge_never_reaches_the_pause() {
    let repo = TestRepo::new("edits");
    // Two edits: enough for the worktree sample to change twice, so the
    // read-only streak never reaches even its first threshold.
    let server = ScriptedServer::spawn(vec![
        bash_turn("call_1", &edit_turn(1)),
        bash_turn("call_2", &edit_turn(2)),
        completion_turn("call_done"),
    ])
    .await;

    let scratch = common::TempDir::new_in_tmp("read-only-pool");
    let pool = WorkerPool::with_scratch(
        1,
        server.base_url.clone(),
        "test-key".to_string(),
        mini_swe_mcp::worktree::ScratchRoot::new(scratch.path()),
    );
    let _scratch = scratch;
    let worker_id = dispatch(&pool, repo.path(), 20).await;

    let state = wait_for_terminal(&pool, &worker_id).await;
    let metrics = match &state {
        WorkerState::Completed { metrics, .. } => *metrics,
        other => panic!("a worker that edits must finish, got {other:?}"),
    };
    assert_eq!(
        metrics.loop_pauses, 0,
        "a worker that edits is never parked, got {metrics:?}"
    );
}
