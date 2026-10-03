//! The degeneracy guard, end to end.
//!
//! One real run answered ~55 consecutive turns with a `reasoning_content` of
//! exactly 32 `!` characters, no tool call, and the harness answered every one
//! of them with the same `NO_COMMAND_NUDGE` -- a loop nothing escalated out of
//! until the orchestrator steered it by hand.
//!
//! These tests script that shape against the public `pool` API and assert on
//! what the worker was sent and where it ended up:
//!
//! * the replayed reasoning leaves the next request once it goes degenerate;
//! * the worker is told to think about the last output before acting;
//! * it is parked on the orchestrator once it keeps happening;
//! * a long legitimate reasoning block full of repeated characters never
//!   triggers any of it (the `common::FakeLlm` script is that control).

use crate::common;
use serde_json::{Value, json};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Mutex;

use mini_swe_mcp::pool::{WorkerMetrics, WorkerPool, WorkerState};

/// Owner recorded for the workers these tests dispatch: the guard is what is
/// under test here, not the per-agent ownership check.
const TEST_OWNER: &str = "test-agent";

/// The value the observed run produced on ~55 consecutive turns.
const OBSERVED: &str = "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!";

/// A reasoning block a worker legitimately writes: a long diff-quoting plan
/// with a divider, an indented block and a table rule -- all runs of one
/// character, none of them filler.
fn legitimate_reasoning(turn: usize) -> String {
    let mut text = format!("Turn {turn}: the test failed, so let me read the diff.\n");
    for row in 0..12 {
        text.push_str(&format!("    {row:>3} | let value = calculate({row});\n"));
    }
    text.push_str(&"-".repeat(72));
    text.push_str("\n===== next step =====\n");
    text.push_str("Rerun the suite and compare the failure against the baseline.\n");
    text
}

/// One scripted SSE body: whatever the model emits on one turn.
type ScriptedTurn = Vec<String>;

fn frame(value: &Value) -> String {
    format!("data: {value}\n\n")
}

/// A loopback server that answers the next scripted body per connection and
/// captures every request, so a test can read the exact conversation the
/// provider received.
struct ScriptedServer {
    base_url: String,
    requests: Arc<Mutex<Vec<Value>>>,
}

impl ScriptedServer {
    async fn spawn(turns: Vec<ScriptedTurn>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind loopback");
        let addr: SocketAddr = listener.local_addr().expect("local addr");
        let requests = Arc::new(Mutex::new(Vec::new()));
        let script = Arc::new(std::sync::Mutex::new(turns.into_iter()));
        let sink = requests.clone();
        tokio::spawn(async move {
            loop {
                let Ok((mut socket, _)) = listener.accept().await else {
                    return;
                };
                // Past the script the worker is done being tested: keep
                // answering the completion sentinel so a stray extra turn ends
                // the run instead of hanging on a silent socket.
                let turn = script
                    .lock()
                    .expect("script lock")
                    .next()
                    .unwrap_or_else(|| {
                        ScriptedServer::completion_turn("call_tail", "Wrapping up.")
                    });
                let sink = sink.clone();
                tokio::spawn(async move {
                    if let Some(body) = read_http_request(&mut socket).await
                        && let Ok(value) = serde_json::from_str::<Value>(&body)
                    {
                        sink.lock().await.push(value);
                    }
                    write_sse(&mut socket, Some(turn.as_slice())).await;
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

    /// A turn with a tool call running `command`, and `reasoning` beside it.
    fn bash_turn(call_id: &str, command: &str, reasoning: &str) -> ScriptedTurn {
        let arguments = json!({ "command": command }).to_string();
        vec![
            frame(&json!({
                "choices": [{"delta": {
                    "reasoning_content": reasoning,
                    "content": "",
                    "tool_calls": [{
                        "index": 0,
                        "id": call_id,
                        "function": { "name": "bash", "arguments": arguments }
                    }]
                }}]
            })),
            frame(&json!({"choices": [{"delta": {}, "finish_reason": "tool_calls"}], })),
        ]
    }

    /// The degenerate turn of the observed run: prose, the filler reasoning,
    /// and no tool call at all.
    fn degenerate_turn(reasoning: &str) -> ScriptedTurn {
        vec![frame(&json!({
            "choices": [{"delta": {
                "reasoning_content": reasoning,
                "content": "I will execute a bash command."
            }}]
        }))]
    }

    /// The completion turn, with the REPORT block the system prompt requires.
    fn completion_turn(call_id: &str, reasoning: &str) -> ScriptedTurn {
        let report = "REPORT\ndone: scripted completion\nfiles: lib.rs\ntests: none\nrisks: none";
        Self::bash_turn(
            call_id,
            "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
            reasoning,
        )
        .into_iter()
        .map(|f| f.replace("\"content\":\"\"", &format!("\"content\":{:?}", report)))
        .collect()
    }
}

async fn write_sse(socket: &mut TcpStream, turn: Option<&[String]>) {
    let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n";
    if socket.write_all(head.as_bytes()).await.is_err() {
        return;
    }
    if let Some(chunks) = turn {
        for chunk in chunks {
            if socket.write_all(chunk.as_bytes()).await.is_err() {
                return;
            }
            let _ = socket.flush().await;
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        let _ = socket.write_all(b"data: [DONE]\n\n").await;
    }
    let _ = socket.flush().await;
    let _ = socket.shutdown().await;
}

async fn read_http_request(socket: &mut TcpStream) -> Option<String> {
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

struct TestRepo {
    dir: PathBuf,
}

impl TestRepo {
    fn new(tag: &str) -> Self {
        let dir = std::env::temp_dir().join(common::unique_suffix(&format!("degenerate-{tag}")));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create scratch repo");
        let dir = dir.canonicalize().expect("canonicalize scratch repo");
        let git = |args: &[&str]| {
            let output = Command::new("git")
                .current_dir(&dir)
                .args(args)
                .output()
                .expect("run git");
            assert!(
                output.status.success(),
                "git {args:?} failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        };
        git(&["init", "-b", "master"]);
        git(&["config", "user.name", "mini-swe-test"]);
        git(&["config", "user.email", "test@localhost"]);
        std::fs::write(dir.join("lib.rs"), "fn main() {}\n").expect("seed file");
        git(&["add", "."]);
        git(&["commit", "-m", "baseline"]);
        Self { dir }
    }

    fn path(&self) -> &Path {
        &self.dir
    }
}

impl Drop for TestRepo {
    fn drop(&mut self) {
        mini_swe_mcp::cache::remove_build_dir_leases(&self.dir);
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

// ----------
// Pool helpers
// ----------

async fn dispatch(pool: &WorkerPool, repo: &Path, max_turns: usize) -> String {
    pool.dispatch(
        TEST_OWNER.to_string(),
        "exercise the degeneracy guard".to_string(),
        "test-model".to_string(),
        None,
        repo.to_path_buf(),
        max_turns,
        Some("degenerate".to_string()),
        None,
        false,
        None,
        Vec::new(),
    )
    .await
    .expect("dispatch the worker")
}

fn pool_for(base_url: &str, tag: &str) -> (WorkerPool, common::TempDir) {
    let scratch = common::TempDir::new_in_tmp(tag);
    let pool = WorkerPool::with_scratch(
        1,
        base_url.to_string(),
        "test-key".to_string(),
        mini_swe_mcp::worktree::ScratchRoot::new(scratch.path()),
    );
    (pool, scratch)
}

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
        WorkerState::Completed { metrics, .. }
        | WorkerState::Exhausted { metrics, .. }
        | WorkerState::Failed { metrics, .. } => *metrics,
        other => panic!("expected a finished worker, got {other:?}"),
    }
}

fn messages_of(request: &Value) -> Vec<Value> {
    request["messages"]
        .as_array()
        .expect("messages array")
        .to_vec()
}

/// Every non-empty `reasoning_content` an assistant turn of `request` carries.
fn replayed_reasoning(request: &Value) -> Vec<String> {
    messages_of(request)
        .into_iter()
        .filter(|m| m["role"] == json!("assistant"))
        .filter_map(|m| m["reasoning_content"].as_str().map(str::to_string))
        .filter(|r| !r.is_empty())
        .collect()
}

/// Every user message of `request`, in order.
fn user_messages(request: &Value) -> Vec<String> {
    messages_of(request)
        .into_iter()
        .filter(|m| m["role"] == json!("user"))
        .map(|m| m["content"].as_str().unwrap_or_default().to_string())
        .collect()
}

// ----------
// Tests
// ----------

/// The observed shape: a tool-calling turn, then the same filler reasoning on
/// every turn after it.
///
/// What must happen is what the run never got: the filler must leave the
/// request (it is the value the model has been echoing back), the worker must
/// be told to think about the last output, and it must end on the orchestrator
/// rather than looping to its budget.
#[tokio::test]
async fn degenerate_reasoning_is_dropped_nudged_and_escalated() {
    let repo = TestRepo::new("escalate");
    let mut script = vec![ScriptedServer::bash_turn(
        "call_1",
        "printf 'first\\n' > lib.rs",
        "Let me start by writing the file.",
    )];
    // Four degenerate turns is where the guard parks the worker; the ones
    // after the steer are answered by the completion fallback, so a guard that
    // did not fire would show up as a run that never paused.
    for _ in 0..4 {
        script.push(ScriptedServer::degenerate_turn(OBSERVED));
    }
    script.push(ScriptedServer::completion_turn(
        "call_done",
        "The orchestrator asked for the completion, so I will finish.",
    ));
    let server = ScriptedServer::spawn(script).await;
    let (pool, _scratch) = pool_for(&server.base_url, "degenerate-pool");
    let worker_id = dispatch(&pool, repo.path(), 30).await;

    let question = wait_for_paused(&pool, &worker_id)
        .await
        .expect("a worker whose reasoning stayed degenerate must park on the orchestrator");
    assert!(
        question.contains("Degenerate reasoning"),
        "the pause must name the problem, got {question:?}"
    );
    assert!(
        question.contains("4 consecutive"),
        "the pause must name the streak, got {question:?}"
    );
    assert!(
        question.contains(OBSERVED),
        "the pause must quote the value the model keeps sending, got {question:?}"
    );

    let requests = server.requests().await;
    // The request after the first degenerate turn must not carry the filler
    // back to the provider: that replay is what kept the loop alive.
    let after_filler = requests
        .iter()
        .skip(2)
        .find(|request| replayed_reasoning(request).iter().any(|r| r.contains('!')));
    assert!(
        after_filler.is_none(),
        "the degenerate reasoning must be dropped from the request: {after_filler:?}"
    );
    // And the worker must have been told what to do instead.
    let nudged = requests.iter().any(|request| {
        user_messages(request)
            .iter()
            .any(|m| m.contains("think step by step about the last command output"))
    });
    assert!(
        nudged,
        "the worker must be told to reason about the last output"
    );

    // The orchestrator decides, and the worker finishes from its guidance.
    pool.steer(&worker_id, "run the suite and finish".to_string())
        .await
        .expect("steer the paused worker");
    let state = wait_for_terminal(&pool, &worker_id).await;
    assert!(
        matches!(state, WorkerState::Completed { .. }),
        "the worker must finish once guided, got {state:?}"
    );
    let metrics = metrics_of(&state);
    assert!(
        metrics.no_command_turns >= 4,
        "the turns without a tool call must be counted, got {metrics:?}"
    );
    assert_eq!(
        metrics.loop_pauses, 1,
        "the degeneracy pause is counted once, got {metrics:?}"
    );
}

/// The control: a worker whose reasoning is a long legitimate block full of
/// repeated characters must keep it, must never be nudged, and must never
/// pause. A detector that fired here would strip real reasoning from healthy
/// runs.
#[tokio::test]
async fn a_legitimate_reasoning_block_never_triggers_the_guard() {
    let repo = TestRepo::new("control");
    let script: Vec<ScriptedTurn> = (0..6)
        .map(|turn| {
            ScriptedServer::bash_turn(
                &format!("call_{turn}"),
                &format!("printf 'x{turn}\\n' >> lib.rs"),
                &legitimate_reasoning(turn),
            )
        })
        .chain(std::iter::once(ScriptedServer::completion_turn(
            "call_done",
            &legitimate_reasoning(99),
        )))
        .collect();
    let server = ScriptedServer::spawn(script).await;
    let (pool, _scratch) = pool_for(&server.base_url, "degenerate-control-pool");
    let worker_id = dispatch(&pool, repo.path(), 30).await;

    let state = wait_for_terminal(&pool, &worker_id).await;
    assert!(
        matches!(state, WorkerState::Completed { .. }),
        "a healthy worker must finish on its own, got {state:?}"
    );
    let metrics = metrics_of(&state);
    assert_eq!(
        metrics.no_command_turns, 0,
        "no turn was without a command, got {metrics:?}"
    );
    assert_eq!(
        (metrics.no_command_pauses, metrics.loop_pauses),
        (0, 0),
        "nothing may pause a healthy worker, got {metrics:?}"
    );

    let requests = server.requests().await;
    assert!(
        !requests.is_empty(),
        "the worker must have talked to the provider"
    );
    // The reasoning survives: it is real content, so it is replayed intact.
    let last = requests.last().expect("at least one request");
    assert!(
        replayed_reasoning(last)
            .iter()
            .any(|r| r.contains("the test failed, so let me read the diff")),
        "legitimate reasoning must be replayed, got {:?}",
        replayed_reasoning(last)
    );
    assert!(
        !requests.iter().any(|request| user_messages(request)
            .iter()
            .any(|m| m.contains("think step by step"))),
        "a healthy worker must never be nudged"
    );
}

/// A reply the provider cut short is not a model refusal: the pause has to say
/// so, because the fix is a shorter history rather than a nudge.
#[tokio::test]
async fn a_truncated_reply_is_named_as_such_when_the_worker_parks() {
    let repo = TestRepo::new("truncated");
    let mut script = vec![ScriptedServer::bash_turn(
        "call_1",
        "printf 'first\\n' > lib.rs",
        "Writing the file first.",
    )];
    // The provider cuts every reply short with `finish_reason: "length"`, so
    // the turn is incomplete however the content reads: the guard must name
    // the truncation as the mechanism, not read it as a model refusal.
    for _ in 0..6 {
        script.push(vec![frame(&json!({
            "choices": [{"delta": {"content": "I will execute a bash command."},
                         "finish_reason": "length"}]
        }))]);
    }
    script.push(ScriptedServer::completion_turn(
        "call_done",
        "Finishing now.",
    ));
    let server = ScriptedServer::spawn(script).await;
    let (pool, _scratch) = pool_for(&server.base_url, "degenerate-truncated-pool");
    let worker_id = dispatch(&pool, repo.path(), 30).await;

    let question = wait_for_paused(&pool, &worker_id)
        .await
        .expect("a worker that never calls a tool must park on the orchestrator");
    assert!(
        question.contains("stopped calling tools"),
        "the pause must name the mechanism, got {question:?}"
    );
    assert!(
        question.contains("finish_reason=length/content_filter"),
        "the pause must name the truncation as the cause, got {question:?}"
    );

    pool.steer(&worker_id, "call the bash tool".to_string())
        .await
        .expect("steer the paused worker");
    let state = wait_for_terminal(&pool, &worker_id).await;
    assert!(
        matches!(state, WorkerState::Completed { .. }),
        "the worker must finish once guided, got {state:?}"
    );
}
