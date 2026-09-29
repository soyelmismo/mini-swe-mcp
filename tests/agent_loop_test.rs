//! Protocol-correct history for the shared agent turn engine.
//!
//! The implementer and the review auditor used to be two copies of the same
//! loop, and both got the "the model produced no usable command" case wrong:
//! the first response was silently discarded (a hidden retry issued while
//! `messages` was still the pre-turn state), so a thinking-mode provider lost
//! the `reasoning_content` it requires to be replayed and the agent looped on
//! a command it had never been shown. The single engine now keeps the turn:
//!
//! * a `tool_calls` turn whose arguments carry no `command` is replayed as
//!   `assistant(tool_calls)` + one `tool` message per call id, so no
//!   unanswered `tool_calls` ever reach the provider;
//! * a prose-only turn is replayed as `assistant` + the `ERROR:` user message;
//! * every assistant message derived from an LLM response carries its
//!   `reasoning_content`.
//!
//! The server below is scripted: turn 1 answers with an unparseable tool call
//! plus reasoning, turn 2 with the completion sentinel. The captured turn-2
//! request is the proof that turn 1 reached the model intact.

mod common;

use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::time::Duration;

use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::Mutex;

use mini_swe_mcp::pool::{COMPLETION_SENTINEL, WorkerPool, WorkerState};

/// A tool call the engine cannot turn into a command: valid JSON, but no
/// `command` key, so `BashArgs` fails to deserialize and `command` is `None`.
const UNPARSEABLE_ARGS: &str = r#"{"not_command":"ls -la"}"#;

/// The reasoning the scripted turn 1 emits. Thinking-mode providers reject a
/// follow-up request whose history dropped it, so its presence in the turn-2
/// body is the assertion that matters.
const TURN_ONE_REASONING: &str = "Let me look at the repository first.";

/// One canned SSE body, per turn.
type ScriptedTurn = Vec<String>;

// ----------
// Scripted SSE server
// ----------

/// Captures every request body the agent sends, so a test can assert on the
/// exact conversation the provider would have received.
#[derive(Clone)]
struct CapturedRequests(Arc<Mutex<Vec<Value>>>);

impl CapturedRequests {
    async fn all(&self) -> Vec<Value> {
        self.0.lock().await.clone()
    }

}

/// A loopback HTTP server that answers `POST /chat/completions` with the next
/// scripted SSE body, one connection per turn.
///
/// A fixed script (rather than a per-model map) is deliberate: the turn
/// sequence is what these tests assert on, and a request that arrives out of
/// band must not silently shift the script.
struct ScriptedSseServer {
    base_url: String,
    requests: CapturedRequests,
}

impl ScriptedSseServer {
    /// Serve `turns` in order, then answer any further request with 500 so an
    /// unexpected extra turn fails loudly instead of hanging.
    async fn spawn(turns: Vec<ScriptedTurn>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind loopback");
        let addr: SocketAddr = listener.local_addr().expect("local addr");

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
                    let request = read_http_request(&mut socket).await;
                    if let Some(body) = request
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

    /// The turn a model issues when it is done: the sentinel as the final
    /// `echo` of the command, which is what `is_completion_request` accepts.
    fn completion_turn(call_id: &str) -> ScriptedTurn {
        let arguments = json!({ "command": format!("echo {COMPLETION_SENTINEL}") }).to_string();
        vec![frame(&json!({
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": call_id,
                        "function": { "name": "bash", "arguments": arguments }
                    }]
                }
            }]
        }))]
    }

    /// A turn whose tool call cannot yield a command, plus reasoning content.
    fn unparseable_turn(call_ids: &[&str], reasoning: &str) -> ScriptedTurn {
        let tool_calls: Vec<Value> = call_ids
            .iter()
            .enumerate()
            .map(|(i, id)| {
                json!({
                    "index": i,
                    "id": id,
                    "function": { "name": "bash", "arguments": UNPARSEABLE_ARGS }
                })
            })
            .collect();
        vec![frame(&json!({
            "choices": [{
                "delta": {
                    "reasoning_content": reasoning,
                    "tool_calls": tool_calls
                }
            }]
        }))]
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
            let _ = socket.flush().await;
            // Push a real TCP boundary so reqwest sees separate chunks.
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        let _ = socket.write_all(b"data: [DONE]\n\n").await;
    }
    let _ = socket.flush().await;
    let _ = socket.shutdown().await;
}

/// Read one HTTP request head plus its `Content-Length` body.
async fn read_http_request(socket: &mut tokio::net::TcpStream) -> Option<String> {
    let mut data: Vec<u8> = Vec::new();
    let mut probe = [0u8; 4096];

    while !data.windows(4).any(|w| w == b"\r\n\r\n") {
        match socket.read(&mut probe).await {
            Ok(0) | Err(_) => return None,
            Ok(n) => data.extend_from_slice(&probe[..n]),
        }
    }

    let head_end = data
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .expect("head terminator present")
        + 4;

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

/// A throwaway git repository the worker can be dispatched against.
struct TestRepo {
    dir: PathBuf,
}

impl TestRepo {
    fn new(tag: &str) -> Self {
        let dir = std::env::temp_dir().join(common::unique_suffix(&format!("agent-loop-{tag}")));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create scratch repo");
        let dir = dir.canonicalize().expect("canonicalize scratch repo");
        run_git(&dir, &["init", "-b", "master"]);
        run_git(&dir, &["config", "user.name", "mini-swe-test"]);
        run_git(&dir, &["config", "user.email", "test@localhost"]);
        std::fs::write(dir.join("README.md"), "# scratch\n").expect("seed file");
        run_git(&dir, &["add", "README.md"]);
        run_git(&dir, &["commit", "-m", "baseline"]);
        Self { dir }
    }

    fn path(&self) -> &Path {
        &self.dir
    }
}

impl Drop for TestRepo {
    fn drop(&mut self) {
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
// Request-body assertions
// ----------

fn messages_of(request: &Value) -> Vec<Value> {
    request["messages"]
        .as_array()
        .expect("request carries a messages array")
        .clone()
}

/// The tool messages of `request`, keyed by `tool_call_id`.
fn tool_results(request: &Value) -> HashMap<String, String> {
    messages_of(request)
        .into_iter()
        .filter(|m| m["role"] == json!("tool"))
        .map(|m| {
            (
                m["tool_call_id"].as_str().expect("tool_call_id").to_string(),
                m["content"].as_str().unwrap_or_default().to_string(),
            )
        })
        .collect()
}

/// The assistant turn that advertised `tool_calls`.
fn assistant_tool_call_turn(request: &Value) -> Option<Value> {
    messages_of(request).into_iter().find(|m| {
        m["role"] == json!("assistant") && m.get("tool_calls").is_some_and(|tc| !tc.is_null())
    })
}

async fn wait_for_terminal(pool: &WorkerPool, worker_id: &str) -> WorkerState {
    for _ in 0..600 {
        if let Some(state) = pool.get_worker_state(worker_id).await {
            match state {
                WorkerState::Completed { .. } | WorkerState::Failed { .. } => return state,
                WorkerState::Running { .. } | WorkerState::Paused { .. } => {}
            }
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("worker {worker_id} did not reach a terminal state");
}

/// Dispatch one worker against `base_url` and wait for it to finish.
async fn dispatch_and_wait(
    base_url: &str,
    repo: &Path,
    max_turns: usize,
    review_after: Option<String>,
) -> (WorkerPool, String, WorkerState) {
    let pool = WorkerPool::new(1, base_url.to_string(), "test-key".to_string());
    let worker_id = pool
        .dispatch(
            "exercise the agent loop".to_string(),
            "test-model".to_string(),
            None,
            repo.to_path_buf(),
            max_turns,
            Some("agent-loop".to_string()),
            review_after,
            false,
        )
        .await
        .expect("dispatch the worker");
    let state = wait_for_terminal(&pool, &worker_id).await;
    (pool, worker_id, state)
}

// ----------
// Implementer phase
// ----------

/// Turn 1 emits a `bash` tool call whose arguments carry no `command`.
///
/// The engine must keep that turn: replaying it (with its reasoning) plus one
/// `tool` message per call id is the only conversation turn 2 accepts, and it
/// is what stops the model from looping on an attempt it never saw.
#[tokio::test]
async fn implementer_replays_the_unparseable_tool_call_turn_with_its_reasoning() {
    let repo = TestRepo::new("implementer");
    let server = ScriptedSseServer::spawn(vec![
        ScriptedSseServer::unparseable_turn(&["call_bad_1"], TURN_ONE_REASONING),
        ScriptedSseServer::completion_turn("call_done"),
    ])
    .await;

    let (pool, worker_id, state) =
        dispatch_and_wait(&server.base_url, repo.path(), 5, None).await;

    match state {
        WorkerState::Completed { .. } => {}
        other => panic!("worker must complete on the sentinel turn, got {other:?}"),
    }

    let requests = server.requests.all().await;
    assert_eq!(
        requests.len(),
        2,
        "exactly one request per turn; the hidden retry is gone: {requests:?}"
    );

    let turn_two = &requests[1];

    // Rule a: the assistant turn is kept, with its call and its reasoning.
    let assistant = assistant_tool_call_turn(turn_two)
        .expect("turn 1's assistant tool_calls turn must be replayed to turn 2");
    let call_ids: Vec<String> = assistant["tool_calls"]
        .as_array()
        .expect("tool_calls array")
        .iter()
        .map(|tc| tc["id"].as_str().expect("call id").to_string())
        .collect();
    assert_eq!(
        call_ids,
        vec!["call_bad_1".to_string()],
        "the call of the dropped turn must be replayed"
    );
    assert_eq!(
        assistant["reasoning_content"], json!(TURN_ONE_REASONING),
        "thinking-mode providers reject a follow-up whose history dropped the reasoning"
    );

    // One tool message per call id, so no `tool_calls` is left unanswered.
    let results = tool_results(turn_two);
    assert_eq!(
        results.len(),
        1,
        "one tool result per call id, got {results:?}"
    );
    for id in &call_ids {
        let content = results
            .get(id)
            .unwrap_or_else(|| panic!("no tool result for {id}: {results:?}"));
        assert!(
            content.contains("could not parse a `command` from the bash tool arguments"),
            "the tool result must explain the parse failure, got {content:?}"
        );
        assert!(
            content.contains(UNPARSEABLE_ARGS),
            "the tool result must quote the arguments the model sent, got {content:?}"
        );
    }

    let _ = pool.kill(&worker_id).await;
}

/// A prose-only turn (no `tool_calls` at all) is rule b: the assistant text is
/// kept with its reasoning, followed by the `ERROR:` user message. Dropping the
/// assistant turn here is the bug that made the reviewer loop forever.
#[tokio::test]
async fn implementer_replays_a_prose_only_turn_before_the_error_message() {
    let repo = TestRepo::new("prose");
    let server = ScriptedSseServer::spawn(vec![
        vec![frame(&json!({
            "choices": [{
                "delta": {
                    "reasoning_content": TURN_ONE_REASONING,
                    "content": "I should look around first."
                }
            }]
        }))],
        ScriptedSseServer::completion_turn("call_done"),
    ])
    .await;

    let (_pool, _worker_id, state) =
        dispatch_and_wait(&server.base_url, repo.path(), 5, None).await;

    assert!(
        matches!(state, WorkerState::Completed { .. }),
        "worker must complete, got {state:?}"
    );

    let requests = server.requests.all().await;
    assert_eq!(requests.len(), 2, "got {requests:?}");

    let messages = messages_of(&requests[1]);
    let assistant = messages
        .iter()
        .find(|m| m["role"] == json!("assistant"))
        .expect("the prose assistant turn must survive into turn 2");
    assert_eq!(assistant["content"], json!("I should look around first."));
    assert_eq!(assistant["reasoning_content"], json!(TURN_ONE_REASONING));

    let assistant_at = messages.iter().position(|m| m == assistant).expect("index");
    let error_at = messages
        .iter()
        .position(|m| {
            m["role"] == json!("user")
                && m["content"]
                    .as_str()
                    .is_some_and(|c| c.contains("No bash command found"))
        })
        .expect("the ERROR user message must follow the assistant turn");
    assert!(
        error_at > assistant_at,
        "the assistant turn must precede its error, got {messages:?}"
    );
}

/// Unparseable arguments are truncated before they are echoed back, so a model
/// streaming megabytes of arguments cannot inflate the history.
#[tokio::test]
async fn the_parse_error_quotes_at_most_two_hundred_argument_bytes() {
    let repo = TestRepo::new("truncate");
    let long_args = "x".repeat(4_000);
    let escaped = serde_json::to_string(&json!({ "not_command": long_args })).expect("encode");
    let server = ScriptedSseServer::spawn(vec![
        vec![frame(&json!({
            "choices": [{
                "delta": {
                    "reasoning_content": TURN_ONE_REASONING,
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_long",
                        "function": { "name": "bash", "arguments": escaped }
                    }]
                }
            }]
        }))],
        ScriptedSseServer::completion_turn("call_done"),
    ])
    .await;

    let (_pool, _worker_id, state) =
        dispatch_and_wait(&server.base_url, repo.path(), 5, None).await;
    assert!(
        matches!(state, WorkerState::Completed { .. }),
        "worker must complete, got {state:?}"
    );

    let requests = server.requests.all().await;
    let results = tool_results(&requests[1]);
    let content = results.get("call_long").expect("tool result for call_long");
    let quoted = content
        .split_once("arguments: ")
        .expect("prefix before the quoted arguments")
        .1;
    assert!(
        quoted.contains("xxx"),
        "the model must see its own arguments, got {quoted:.80}"
    );
    assert!(
        quoted.ends_with("..."),
        "a long argument blob must be truncated with an ellipsis, got tail {:?}",
        &quoted[quoted.len().saturating_sub(16)..]
    );
    assert!(
        quoted.len() < 300,
        "the quoted arguments must stay bounded, got {} bytes",
        quoted.len()
    );
}

// ----------
// Reviewer phase (review_after)
// ----------

/// The reviewer is driven by the *same* engine, so it inherits rule a. The
/// script therefore serves both phases: two implementer turns, then the
/// reviewer's unparseable turn and its completion.
#[tokio::test]
async fn reviewer_replays_the_unparseable_turn_with_its_reasoning() {
    let repo = TestRepo::new("reviewer");
    let server = ScriptedSseServer::spawn(vec![
        ScriptedSseServer::completion_turn("call_impl_done"),
        ScriptedSseServer::unparseable_turn(&["call_review_bad"], TURN_ONE_REASONING),
        ScriptedSseServer::completion_turn("call_review_done"),
    ])
    .await;

    let (pool, worker_id, state) = dispatch_and_wait(
        &server.base_url,
        repo.path(),
        5,
        Some("test-reviewer-model".to_string()),
    )
    .await;

    assert!(
        matches!(state, WorkerState::Completed { .. }),
        "worker must complete after review, got {state:?}"
    );

    let requests = server.requests.all().await;
    assert_eq!(
        requests.len(),
        3,
        "one implementer turn, then two reviewer turns: {requests:?}"
    );

    let review_turn_two = &requests[2];
    assert_eq!(
        review_turn_two["model"], json!("test-reviewer-model"),
        "the third request is the reviewer's, on its own model"
    );

    let assistant = assistant_tool_call_turn(review_turn_two)
        .expect("the reviewer's unparseable turn must be replayed to its next turn");
    assert_eq!(assistant["reasoning_content"], json!(TURN_ONE_REASONING));
    assert_eq!(
        assistant["tool_calls"][0]["id"], json!("call_review_bad"),
        "the reviewer's own call must be replayed: {assistant}"
    );

    let results = tool_results(review_turn_two);
    let content = results
        .get("call_review_bad")
        .expect("a tool result for the reviewer's call");
    assert!(
        content.contains("could not parse a `command` from the bash tool arguments"),
        "the reviewer must be told why its call produced no command, got {content:?}"
    );

    let _ = pool.kill(&worker_id).await;
}
