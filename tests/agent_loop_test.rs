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

use mini_swe_mcp::pool::{COMPLETION_SENTINEL, WorkerMetrics, WorkerPool, WorkerState};

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

    /// A turn whose tool call runs `command` in the worker worktree.
    fn bash_turn(call_id: &str, command: &str) -> ScriptedTurn {
        let arguments = json!({ "command": command }).to_string();
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

    /// The turn a model issues when it is done: the sentinel as the final
    /// `echo` of the command, which is what `is_completion_request` accepts.
    fn completion_turn(call_id: &str) -> ScriptedTurn {
        Self::bash_turn(call_id, &format!("echo {COMPLETION_SENTINEL}"))
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

/// Trimmed stdout of one `git` invocation, for assertions on branch history.
fn git_capture(dir: &Path, args: &[&str]) -> String {
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
    String::from_utf8_lossy(&output.stdout).trim().to_string()
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

/// The question a paused worker is parked on, or `None` if it never pauses.
async fn wait_for_paused(pool: &WorkerPool, worker_id: &str) -> Option<String> {
    for _ in 0..600 {
        if let Some(WorkerState::Paused { question, .. }) = pool.get_worker_state(worker_id).await {
            return Some(question);
        }
        if matches!(
            pool.get_worker_state(worker_id).await,
            Some(WorkerState::Completed { .. } | WorkerState::Failed { .. })
        ) {
            return None;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    None
}

/// Every user message of `request`, in order.
fn user_messages(request: &Value) -> Vec<String> {
    messages_of(request)
        .into_iter()
        .filter(|m| m["role"] == json!("user"))
        .map(|m| m["content"].as_str().unwrap_or_default().to_string())
        .collect()
}

/// The health counters a completed run reported.
///
/// The counters are what an orchestrator grades a worker on, so the tests that
/// exercise a guard assert on the number the guard moved.
fn metrics_of(state: &WorkerState) -> WorkerMetrics {
    match state {
        WorkerState::Completed { metrics, .. } => *metrics,
        other => panic!("expected a completed worker, got {other:?}"),
    }
}

/// Dispatch one worker against `base_url` and wait for it to finish.
async fn dispatch_and_wait(
    base_url: &str,
    repo: &Path,
    max_turns: usize,
    review_after: Option<String>,
    verify: Option<String>,
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
            verify,
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
        dispatch_and_wait(&server.base_url, repo.path(), 5, None, None).await;

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
        dispatch_and_wait(&server.base_url, repo.path(), 5, None, None).await;

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
        dispatch_and_wait(&server.base_url, repo.path(), 5, None, None).await;
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
        None,
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

// ----------
// Verify gate (selfimprove-I1)
// ----------

/// A verify gate that passes lets the worker complete immediately, flagged
/// verified.
#[tokio::test]
async fn verify_gate_passes_and_worker_completes() {
    let repo = TestRepo::new("verify-pass");
    let server = ScriptedSseServer::spawn(vec![
        ScriptedSseServer::completion_turn("call_done"),
    ])
    .await;

    let (_pool, _worker_id, state) =
        dispatch_and_wait(&server.base_url, repo.path(), 5, None, Some("true".to_string())).await;

    match state {
        WorkerState::Completed { verified, .. } => {
            assert_eq!(verified, Some(true), "a passing gate must verify the worker");
        }
        other => panic!("worker must complete, got {other:?}"),
    }
}

/// A gate that fails then passes lets the worker fix and complete: the first
/// completion is rejected with a VERIFICATION FAILED message, and the second
/// completion passes the gate.
#[tokio::test]
async fn verify_gate_fails_then_passes_after_fix_turn() {
    let repo = TestRepo::new("verify-fix");
    // The verify command fails on the first run (count 1) and passes on the
    // second (count 2), so the worker must get a fix turn before completing.
    let verify = "if [ -f verify_count ]; then c=$(cat verify_count); else c=0; fi; c=$((c+1)); echo $c > verify_count; [ $c -ge 2 ]";
    let server = ScriptedSseServer::spawn(vec![
        ScriptedSseServer::completion_turn("call_done_1"),
        ScriptedSseServer::completion_turn("call_done_2"),
    ])
    .await;

    let (_pool, _worker_id, state) =
        dispatch_and_wait(&server.base_url, repo.path(), 5, None, Some(verify.to_string())).await;

    match state {
        WorkerState::Completed { verified, .. } => {
            assert_eq!(verified, Some(true), "the worker must pass after the fix turn");
        }
        other => panic!("worker must complete, got {other:?}"),
    }

    let metrics = metrics_of(&state);
    assert_eq!(
        (metrics.verify_runs, metrics.verify_failures),
        (2, 1),
        "both gate runs and the single failure must be counted, got {metrics:?}"
    );

    // The model must have been told the verification failed, and the completion
    // turn must stay answered: a `tool_calls` turn is replayed with a tool
    // result rather than left dangling.
    let requests = server.requests.all().await;
    assert_eq!(requests.len(), 2, "one fix turn then completion: {requests:?}");
    let results = tool_results(&requests[1]);
    let content = results
        .get("call_done_1")
        .unwrap_or_else(|| panic!("the rejected completion call must be answered, got {results:?}"));
    assert!(
        content.contains("VERIFICATION FAILED"),
        "the tool result must explain the failure, got {content:?}"
    );
    assert!(
        content.contains("fix these problems before completing"),
        "the tool result must tell the model what to do, got {content:?}"
    );
}

/// Three failed verifications exhaust the budget and the worker completes
/// flagged unverified, with the summary saying so.
#[tokio::test]
async fn verify_gate_exhausts_after_three_failures() {
    let repo = TestRepo::new("verify-exhaust");
    let server = ScriptedSseServer::spawn(vec![
        ScriptedSseServer::completion_turn("call_done_1"),
        ScriptedSseServer::completion_turn("call_done_2"),
        ScriptedSseServer::completion_turn("call_done_3"),
    ])
    .await;

    let (_pool, _worker_id, state) =
        dispatch_and_wait(&server.base_url, repo.path(), 5, None, Some("exit 1".to_string())).await;

    match state {
        WorkerState::Completed { verified, ref summary, .. } => {
            assert_eq!(verified, Some(false), "three failures must flag the worker unverified");
            assert!(
                summary.contains("failing verification"),
                "the summary must say the verification failed, got {summary:?}"
            );
        }
        other => panic!("worker must complete, got {other:?}"),
    }

    let metrics = metrics_of(&state);
    assert_eq!(
        (metrics.verify_runs, metrics.verify_failures),
        (3, 3),
        "the exhausted budget must be visible in the counters, got {metrics:?}"
    );
}

// ----------
// Loop, turn budget and work preservation (selfimprove-I3)
// ----------

/// The command a looping worker used to re-issue dozens of times.
const REPEATED_COMMAND: &str = "sed -n '1,5p' README.md";

/// The exact answer a blocked repetition gets instead of a second run.
const REPEAT_REFUSAL: &str = "You already ran this exact command; its output has not changed (see above). Take a different action.";

/// A command byte-identical to the one before it is answered, not run: its
/// output is already in the history, so running it again only burns a turn.
#[tokio::test]
async fn a_repeated_command_is_answered_without_being_executed() {
    let repo = TestRepo::new("repeat");
    let server = ScriptedSseServer::spawn(vec![
        ScriptedSseServer::bash_turn("call_1", REPEATED_COMMAND),
        ScriptedSseServer::bash_turn("call_2", REPEATED_COMMAND),
        ScriptedSseServer::bash_turn("call_3", "ls -la"),
        ScriptedSseServer::completion_turn("call_done"),
    ])
    .await;

    let (_pool, _worker_id, state) =
        dispatch_and_wait(&server.base_url, repo.path(), 5, None, None).await;
    assert!(
        matches!(state, WorkerState::Completed { .. }),
        "worker must complete after the repetition, got {state:?}"
    );

    let metrics = metrics_of(&state);
    assert_eq!(metrics.turns_used, 4, "every scripted turn must be counted");
    assert_eq!(
        metrics.repeat_blocks, 1,
        "the one repeated command must be counted, got {metrics:?}"
    );
    assert_eq!(metrics.loop_pauses, 0, "one repetition does not park a worker");

    let requests = server.requests.all().await;
    assert_eq!(requests.len(), 4, "got {requests:?}");

    // Turn 1 really ran: its tool result is the command output.
    let first = tool_results(&requests[1]);
    let first_output = first.get("call_1").expect("a tool result for call_1");
    assert!(
        first_output.contains("COMMAND OUTPUT"),
        "the first turn must have run its command, got {first_output:?}"
    );

    // Turn 2 repeated it byte for byte: the model is told why, and the answer
    // is not a command output, which is what "was not executed" looks like.
    let second = tool_results(&requests[2]);
    let refusal = second.get("call_2").expect("a tool result for call_2");
    assert_eq!(
        refusal, REPEAT_REFUSAL,
        "a repeated command must be refused with the loop answer, got {refusal:?}"
    );

    // A different command is progress again: the block is not sticky.
    let third = tool_results(&requests[3]);
    let resumed = third.get("call_3").expect("a tool result for call_3");
    assert!(
        resumed.contains("COMMAND OUTPUT"),
        "the next distinct command must run, got {resumed:?}"
    );
}

/// Three blocked repetitions in a row park the worker on the orchestrator,
/// because a model that will not change course needs a human decision.
#[tokio::test]
async fn three_blocked_repetitions_park_the_worker_for_the_orchestrator() {
    let repo = TestRepo::new("repeat-park");
    let mut script: Vec<ScriptedTurn> = (1..=4)
        .map(|n| ScriptedSseServer::bash_turn(&format!("call_{n}"), REPEATED_COMMAND))
        .collect();
    script.push(ScriptedSseServer::completion_turn("call_done"));
    let server = ScriptedSseServer::spawn(script).await;

    let pool = WorkerPool::new(1, server.base_url.clone(), "test-key".to_string());
    let worker_id = pool
        .dispatch(
            "loop forever".to_string(),
            "test-model".to_string(),
            None,
            repo.path().to_path_buf(),
            10,
            Some("agent-loop".to_string()),
            None,
            false,
            None,
        )
        .await
        .expect("dispatch the worker");

    let question = wait_for_paused(&pool, &worker_id)
        .await
        .expect("a looping worker must park on the orchestrator");
    assert!(
        question.contains("Repetition loop") && question.contains("sed"),
        "the pause must name the loop and the repeated command, got {question:?}"
    );

    pool.steer(&worker_id, "stop re-reading the file; make the edit".to_string())
        .await
        .expect("steer the paused worker");
    let state = wait_for_terminal(&pool, &worker_id).await;
    assert!(
        matches!(state, WorkerState::Completed { .. }),
        "the worker must finish once the orchestrator guides it, got {state:?}"
    );

    let metrics = metrics_of(&state);
    assert_eq!(
        metrics.repeat_blocks, 3,
        "all three blocked repetitions must be counted, got {metrics:?}"
    );
    assert_eq!(
        metrics.loop_pauses, 1,
        "hitting the limit parks the worker exactly once, got {metrics:?}"
    );
}

/// A `REQUEST_TURNS` inside the self-grant budget extends the loop.
#[tokio::test]
async fn a_turn_extension_within_the_budget_extends_the_loop() {
    let repo = TestRepo::new("turns-grant");
    let server = ScriptedSseServer::spawn(vec![
        ScriptedSseServer::bash_turn("call_rq", "echo REQUEST_TURNS: 2"),
        ScriptedSseServer::bash_turn("call_1", "echo one"),
        ScriptedSseServer::bash_turn("call_2", "echo two"),
        ScriptedSseServer::bash_turn("call_3", "echo three"),
        ScriptedSseServer::bash_turn("call_4", "echo four"),
        ScriptedSseServer::completion_turn("call_done"),
    ])
    .await;

    // Budget 4, so half of it (2 turns) may be self-granted: 6 turns in total.
    let (_pool, _worker_id, state) =
        dispatch_and_wait(&server.base_url, repo.path(), 4, None, None).await;
    assert!(
        matches!(state, WorkerState::Completed { .. }),
        "worker must complete, got {state:?}"
    );

    let requests = server.requests.all().await;
    assert_eq!(
        requests.len(),
        6,
        "the grant must lift the loop from 4 to 6 turns: {requests:?}"
    );
    for (i, request) in requests.iter().enumerate() {
        assert!(
            !user_messages(request)
                .iter()
                .any(|m| m.contains("TURN EXTENSION REFUSED")),
            "a grant within budget must not be refused, request {i}: {request}"
        );
    }

    let metrics = metrics_of(&state);
    assert_eq!(
        metrics.extensions_granted, 2,
        "the two self-granted turns must be counted, got {metrics:?}"
    );
    assert_eq!(metrics.extensions_refused, 0);
}

/// A `REQUEST_TURNS` past the budget is refused, and the model is told to wrap
/// up or escalate instead of asking again.
#[tokio::test]
async fn a_turn_extension_beyond_the_budget_is_refused() {
    let repo = TestRepo::new("turns-refuse");
    let server = ScriptedSseServer::spawn(vec![
        ScriptedSseServer::bash_turn("call_rq", "echo \"REQUEST_TURNS: 40\""),
        ScriptedSseServer::bash_turn("call_1", "echo one"),
        ScriptedSseServer::completion_turn("call_done"),
    ])
    .await;

    // Budget 4, so 40 more turns is far past the 2 turns it may self-grant.
    let (_pool, _worker_id, state) =
        dispatch_and_wait(&server.base_url, repo.path(), 4, None, None).await;
    assert!(
        matches!(state, WorkerState::Completed { .. }),
        "worker must complete on the sentinel turn, got {state:?}"
    );

    let requests = server.requests.all().await;
    assert_eq!(
        requests.len(),
        3,
        "a refused extension must not extend the loop: {requests:?}"
    );

    let refusal = user_messages(&requests[1])
        .into_iter()
        .find(|m| m.contains("TURN EXTENSION REFUSED"))
        .expect("the refusal must be injected as a user message");
    assert!(
        refusal.contains(COMPLETION_SENTINEL) && refusal.contains("ASK_ORCHESTRATOR"),
        "the refusal must offer both ways out, got {refusal:?}"
    );

    let metrics = metrics_of(&state);
    assert_eq!(
        metrics.extensions_refused, 1,
        "the refused ask must be counted, got {metrics:?}"
    );
    assert_eq!(metrics.extensions_granted, 0);
}

/// Every twentieth turn a dirty worktree is checkpoint-committed, so the work
/// survives a kill or a crash without waiting for the completion sentinel.
#[tokio::test]
async fn every_twenty_turns_a_dirty_worktree_is_checkpointed() {
    let repo = TestRepo::new("checkpoint");
    let script: Vec<ScriptedTurn> = (1..=25)
        .map(|turn| match turn {
            // Dirty the worktree on turn 19, so turn 20 has something to commit.
            19 => ScriptedSseServer::bash_turn("call_19", "echo checkpoint > note.txt"),
            25 => ScriptedSseServer::completion_turn("call_done"),
            _ => {
                ScriptedSseServer::bash_turn(&format!("call_{turn}"), &format!("echo turn {turn}"))
            }
        })
        .collect();
    let server = ScriptedSseServer::spawn(script).await;

    let (_pool, worker_id, state) =
        dispatch_and_wait(&server.base_url, repo.path(), 25, None, None).await;
    assert!(
        matches!(state, WorkerState::Completed { .. }),
        "worker must complete, got {state:?}"
    );

    let metrics = metrics_of(&state);
    assert_eq!(
        (metrics.diff_files, metrics.diff_insertions, metrics.diff_deletions),
        (1, 1, 0),
        "the final diff must be measured against the worker's base commit, got {metrics:?}"
    );
    assert_eq!(metrics.turns_used, 25, "every scripted turn must be counted");

    let branch = format!("worker-{worker_id}");
    let subjects = git_capture(repo.path(), &["log", "--format=%s", &branch]);
    assert!(
        subjects.contains(&format!("worker({worker_id}): auto-checkpoint step 20")),
        "turn 20 must have checkpointed the worktree, branch log was:\n{subjects}"
    );
    assert_eq!(
        git_capture(repo.path(), &["show", &format!("{branch}:note.txt")]),
        "checkpoint",
        "the checkpointed commit must carry the uncommitted file"
    );
}

/// A kill commits whatever the worker left uncommitted *before* aborting it:
/// the aborted task drops its worktree guard, so a later commit would find
/// nothing to save.
#[tokio::test]
async fn killing_a_worker_checkpoints_its_uncommitted_work() {
    let repo = TestRepo::new("kill-checkpoint");
    let server = ScriptedSseServer::spawn(vec![
        ScriptedSseServer::bash_turn("call_1", "echo preserved > kept.txt"),
        ScriptedSseServer::bash_turn("call_2", "sleep 30"),
        ScriptedSseServer::completion_turn("call_done"),
    ])
    .await;

    let pool = WorkerPool::new(1, server.base_url.clone(), "test-key".to_string());
    let worker_id = pool
        .dispatch(
            "leave work behind".to_string(),
            "test-model".to_string(),
            None,
            repo.path().to_path_buf(),
            5,
            Some("agent-loop".to_string()),
            None,
            false,
            None,
        )
        .await
        .expect("dispatch the worker");

    // Turn 1 has written its file and turn 2 is in flight, so the worktree is
    // dirty and the worker is still alive to be killed.
    let mut reached_second_turn = false;
    for _ in 0..600 {
        if pool
            .worker_progress(&worker_id)
            .await
            .is_some_and(|p| p.step >= 2)
        {
            reached_second_turn = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert!(reached_second_turn, "the worker never reached its second turn");

    assert!(pool.kill(&worker_id).await, "kill must find the worker");
    let state = wait_for_terminal(&pool, &worker_id).await;
    assert!(
        matches!(state, WorkerState::Failed { .. }),
        "a killed worker must fail, got {state:?}"
    );

    let branch = format!("worker-{worker_id}");
    let subject = git_capture(repo.path(), &["log", "-1", "--format=%s", &branch]);
    assert!(
        subject.contains("checkpoint before kill"),
        "the kill must have committed the worktree, branch head was {subject:?}"
    );
    assert_eq!(
        git_capture(repo.path(), &["show", &format!("{branch}:kept.txt")]),
        "preserved",
        "the killed worker's uncommitted file must survive on its branch"
    );
}

/// A worktree that has not changed for 30 sampled turns gets told to make the
/// edit or escalate, instead of exploring until the budget runs out.
#[tokio::test]
async fn a_worker_that_stops_changing_anything_is_told_to_stop_exploring() {
    let repo = TestRepo::new("stagnation");
    // 40 read-only turns: the samples at turns 10, 20, 30 and 40 all agree,
    // so the streak reaches 30 turns on the fourth one.
    let script: Vec<ScriptedTurn> = (1..=40)
        .map(|turn| {
            ScriptedSseServer::bash_turn(&format!("call_{turn}"), &format!("echo turn {turn}"))
        })
        .chain(std::iter::once(ScriptedSseServer::completion_turn("call_done")))
        .collect();
    let server = ScriptedSseServer::spawn(script).await;

    let (_pool, _worker_id, state) =
        dispatch_and_wait(&server.base_url, repo.path(), 41, None, None).await;
    assert!(
        matches!(state, WorkerState::Completed { .. }),
        "worker must complete, got {state:?}"
    );

    let requests = server.requests.all().await;
    assert_eq!(requests.len(), 41, "got {} requests", requests.len());

    let nudge = |request: &Value| {
        user_messages(request)
            .into_iter()
            .find(|m| m.contains("No change to the repository in the last 30 turns"))
    };
    assert!(
        nudge(&requests[29]).is_none(),
        "30 turns had not passed yet on turn 30: {:?}",
        user_messages(&requests[29])
    );
    let injected = nudge(&requests[39]).unwrap_or_else(|| {
        panic!(
            "turn 40 must inject the stagnation nudge, got {:?}",
            user_messages(&requests[39])
        )
    });
    assert!(
        injected.contains("Stop exploring") && injected.contains("ASK_ORCHESTRATOR"),
        "the nudge must offer both ways out, got {injected:?}"
    );

    let metrics = metrics_of(&state);
    assert_eq!(
        metrics.stagnation_nudges, 1,
        "the single nudge must be counted, got {metrics:?}"
    );
    assert_eq!(metrics.turns_used, 41, "implementer and reviewer turns together");
}
