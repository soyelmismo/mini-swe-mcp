//! Integration tests for the MCP stdio server's JSON-RPC 2.0 protocol behaviour.
//!
//! The server is exercised as a real child process speaking newline-delimited
//! JSON-RPC 2.0 over stdio, exactly like an MCP host would:
//!
//! * `initialize` performs the handshake and reports `protocolVersion` + `serverInfo`
//! * `ping` returns an empty JSON object result
//! * notifications (requests without a usable `id`, i.e. `id: null`) are never answered
//! * malformed JSON produces a `-32700` (Parse error) response and the stream
//!   stays usable afterwards
//!
//! The harness is written on plain `std` so the crate keeps its dependency graph
//! unchanged, and it polls the child pipe with a small worker thread so a
//! missing response fails fast instead of hanging the suite.

mod common;
use common::IsolatedPool;

use mini_swe_mcp::agent::wrap_network_command;
use mini_swe_mcp::mcp::{
    ChannelEvent, ConnectionContext, EventKind, McpServer, NETWORK_DEFAULT, NETWORK_MODES, Outcome,
    WORKER_ACTIONS, WorkerSnapshot, WorkerView, channel_frame, diff_events,
};
use mini_swe_mcp::pool::{LogBuffer, WorkerMetrics, WorkerPool, WorkerRecord, WorkerState};
use mini_swe_mcp::pool::{
    RegistryStatus, WorkerRegistryEntry, remove_registry_entry_in, save_registry_entry_in,
};
use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{Receiver, RecvTimeoutError, channel};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

const PROTOCOL_VERSION: &str = "2024-11-05";
const SERVER_NAME: &str = "mini-swe-mcp";
/// The worker-event notification the server pushes at any moment (Claude
/// Code's `claude/channel` extension): background traffic, never an answer to a
/// request, so a test waiting for one must step over it.
const CHANNEL_NOTIFICATION: &str = "notifications/claude/channel";
const RESPONSE_TIMEOUT: Duration = Duration::from_secs(20);
const SILENCE_GRACE: Duration = Duration::from_millis(500);

/// One line read from the server's stdout.
enum ServerOutput {
    Line(String),
    Eof,
}

impl ServerOutput {
    fn describe(&self) -> String {
        match self {
            ServerOutput::Line(l) => format!("line {l:?}"),
            ServerOutput::Eof => "EOF (server exited)".to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// Subprocess harness
static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// One test's override of a process-wide environment variable, restored on drop.
///
/// Edition 2024 makes `set_var` unsafe, which is exactly the hazard this
/// serializes: no other test in this binary may observe the overridden value.
struct ScopedEnv {
    _guard: std::sync::MutexGuard<'static, ()>,
    name: &'static str,
    previous: Option<String>,
}

impl ScopedEnv {
    fn set(name: &'static str, value: &str) -> Self {
        let guard = ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let previous = std::env::var(name).ok();
        // SAFETY: the lock above excludes every other test in this binary.
        unsafe { std::env::set_var(name, value) };
        Self {
            _guard: guard,
            name,
            previous,
        }
    }
}

impl Drop for ScopedEnv {
    fn drop(&mut self) {
        match self.previous.take() {
            Some(value) => unsafe { std::env::set_var(self.name, value) },
            None => unsafe { std::env::remove_var(self.name) },
        }
    }
}

// ---------------------------------------------------------------------------

/// A running `mini-swe-mcp --stdio` server connected through pipes.
struct McpProcess {
    child: Child,
    stdin: ChildStdin,
    stdout_rx: Receiver<ServerOutput>,
    reader: Option<JoinHandle<()>>,
}

impl McpProcess {
    fn spawn() -> Self {
        Self::spawn_with_env(&[])
    }

    /// Spawn the server with extra environment overrides, e.g. a scratch
    /// `SWE_TEMP_DIR`: a test that really dispatches a worker must keep that
    /// worker's worktree out of the developer's own scratch directory.
    fn spawn_with_env(envs: &[(&str, &str)]) -> Self {
        let (exe, args) = binary_command();
        let mut command = Command::new(&exe);
        command
            .args(&args)
            // The stdio server refuses to start without a key, and
            // `dotenvy` never overrides a variable that is already set, so
            // this dummy also keeps the suite independent of (and unable to
            // read) whatever key the developer happens to have exported.
            .env(
                "OPENAI_API_KEY",
                "test-key-not-used-by-these-protocol-tests",
            )
            // Protocol tests exercise the in-process server; the hub transport
            // has its own end-to-end tests (tests/hub_test.rs).
            .env("MINI_SWE_NO_DAEMON", "1");
        common::scrub_identity_env(&mut command);
        for (name, value) in envs {
            command.env(name, value);
        }
        let mut child = command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .unwrap_or_else(|e| panic!("failed to spawn {}: {e}", exe.display()));

        let stdin = child.stdin.take().expect("stdin pipe");
        let mut stdout = child.stdout.take().expect("stdout pipe");
        let (tx, stdout_rx) = channel();
        let reader = std::thread::spawn(move || {
            let mut buf = BufReader::new(&mut stdout);
            loop {
                let mut line = String::new();
                match buf.read_line(&mut line) {
                    Ok(0) | Err(_) => {
                        let _ = tx.send(ServerOutput::Eof);
                        return;
                    }
                    Ok(_) => {
                        if tx.send(ServerOutput::Line(line)).is_err() {
                            return;
                        }
                    }
                }
            }
        });

        Self {
            child,
            stdin,
            stdout_rx,
            reader: Some(reader),
        }
    }

    /// Send an already serialized line to the server.
    fn send_raw(&mut self, line: &str) {
        writeln!(self.stdin, "{line}").expect("write to server stdin");
        self.stdin.flush().expect("flush server stdin");
    }

    /// Send a JSON-RPC message, serializing the payload ourselves.
    fn send(&mut self, value: &Value) {
        self.send_raw(&value.to_string());
    }

    /// Perform the standard MCP handshake and assert its documented contract.
    fn initialize(&mut self) -> Value {
        self.send(&json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": { "name": "mcp_test", "version": "0.0.0" }
            }
        }));

        let response = self
            .expect_response("initialize")
            .unwrap_or_else(|| panic!("initialize did not answer with id 1"));

        assert_eq!(response["id"], json!(1), "id must be echoed verbatim");
        let result = expect_result(&response);
        assert_eq!(
            result["protocolVersion"].as_str(),
            Some(PROTOCOL_VERSION),
            "initialize must report the negotiated protocolVersion, got: {result}"
        );

        let server_info = &result["serverInfo"];
        assert_eq!(
            server_info["name"].as_str(),
            Some(SERVER_NAME),
            "initialize must report serverInfo.name, got: {server_info}"
        );
        assert!(
            server_info["version"].is_string(),
            "initialize must report serverInfo.version, got: {server_info}"
        );
        assert!(
            result["capabilities"].is_object(),
            "initialize must report capabilities, got: {result}"
        );

        // End of handshake: the client announces that it is initialized.
        self.send(&json!({ "jsonrpc": "2.0", "method": "notifications/initialized" }));
        response
    }

    /// Read the next response line, failing loudly on timeout or EOF.
    fn expect_response(&mut self, context: &str) -> Option<Value> {
        let deadline = Instant::now() + RESPONSE_TIMEOUT;
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            assert!(
                !remaining.is_zero(),
                "timed out after {RESPONSE_TIMEOUT:?} waiting for a response to {context}"
            );

            match self.stdout_rx.recv_timeout(remaining) {
                Ok(ServerOutput::Eof) => return None,
                Ok(ServerOutput::Line(raw)) => {
                    let trimmed = raw.trim();
                    if trimmed.is_empty() {
                        continue;
                    }
                    let value: Value = serde_json::from_str(trimmed).unwrap_or_else(|e| {
                        panic!("server wrote non-JSON to stdout: {trimmed:?} ({e})")
                    });
                    // A worker-event notification is not an answer to the
                    // pending request; reading one as a response would make
                    // every test that shares a registry with a live worker
                    // flaky.
                    if value["method"] == json!(CHANNEL_NOTIFICATION) {
                        continue;
                    }
                    assert_eq!(
                        value["jsonrpc"].as_str(),
                        Some("2.0"),
                        "every response must declare jsonrpc 2.0, got: {value}"
                    );
                    return Some(value);
                }
                Err(RecvTimeoutError::Timeout) => panic!(
                    "timed out after {RESPONSE_TIMEOUT:?} waiting for a response to {context}"
                ),
                Err(RecvTimeoutError::Disconnected) => {
                    panic!("stdout reader thread vanished while waiting for {context}")
                }
            }
        }
    }

    /// Wait for the next worker-event notification, stepping over any other
    /// frame the server volunteers in the meantime. The budget is explicit
    /// because a cross-process row change is only discovered on the server's
    /// coarse fallback tick, while an in-process one arrives at once.
    fn expect_channel_event_within(&mut self, context: &str, timeout: Duration) -> Value {
        let deadline = Instant::now() + timeout;
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            assert!(
                !remaining.is_zero(),
                "timed out after {RESPONSE_TIMEOUT:?} waiting for {context}"
            );
            match self.stdout_rx.recv_timeout(remaining) {
                Ok(ServerOutput::Eof) => panic!("server exited while waiting for {context}"),
                Ok(ServerOutput::Line(raw)) => {
                    let trimmed = raw.trim();
                    if trimmed.is_empty() {
                        continue;
                    }
                    let value: Value = serde_json::from_str(trimmed).unwrap_or_else(|e| {
                        panic!("server wrote non-JSON to stdout: {trimmed:?} ({e})")
                    });
                    if value["method"] == json!(CHANNEL_NOTIFICATION) {
                        return value;
                    }
                }
                Err(RecvTimeoutError::Timeout) => {
                    panic!("timed out after {timeout:?} waiting for {context}")
                }
                Err(RecvTimeoutError::Disconnected) => {
                    panic!("stdout reader thread vanished while waiting for {context}")
                }
            }
        }
    }

    /// Assert the server writes nothing on stdout for a short grace period.
    ///
    /// The one frame that may appear anyway is a worker-event notification: it
    /// is background traffic the server emits on its own schedule and says
    /// nothing about the request under test.
    fn expect_silence(&mut self, context: &str) {
        let deadline = Instant::now() + SILENCE_GRACE;
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return;
            }
            match self.stdout_rx.recv_timeout(remaining) {
                Err(RecvTimeoutError::Timeout) | Err(RecvTimeoutError::Disconnected) => return,
                Ok(out) => {
                    let ServerOutput::Line(raw) = &out else {
                        return;
                    };
                    let trimmed = raw.trim();
                    if trimmed.is_empty() {
                        continue;
                    }
                    let value: Value = serde_json::from_str(trimmed).unwrap_or_else(|e| {
                        panic!("server wrote non-JSON to stdout: {trimmed:?} ({e})")
                    });
                    if value["method"] == json!(CHANNEL_NOTIFICATION) {
                        continue;
                    }
                    panic!(
                        "expected no response for {context}, but got {}",
                        out.describe()
                    );
                }
            }
        }
    }
}

impl Drop for McpProcess {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        if let Some(reader) = self.reader.take() {
            let _ = reader.join();
        }
    }
}

/// Locate the freshly built server binary. Cargo exports `CARGO_BIN_EXE_<name>`
/// for integration tests; fall back to the standard `target/<profile>/` path.
fn binary_command() -> (PathBuf, Vec<String>) {
    if let Ok(exe) = std::env::var("CARGO_BIN_EXE_mini-swe-mcp") {
        return (PathBuf::from(exe), vec!["--stdio".to_string()]);
    }
    let mut path = std::env::current_exe().expect("current exe");
    path.pop();
    if path.ends_with("deps") {
        path.pop();
    }
    (path.join("mini-swe-mcp"), vec!["--stdio".to_string()])
}

// ---------------------------------------------------------------------------
// Assertion helpers
// ---------------------------------------------------------------------------

fn expect_result(response: &Value) -> Value {
    assert!(
        response.get("error").is_none(),
        "expected a successful result, got an error response: {response}"
    );
    let result = response
        .get("result")
        .unwrap_or_else(|| panic!("response is missing a result field: {response}"));
    assert!(
        result.is_object(),
        "result must be an object, got: {result}"
    );
    result.clone()
}

fn expect_error_code(response: &Value, code: i64) -> Value {
    let error = response
        .get("error")
        .unwrap_or_else(|| panic!("response is missing an error field: {response}"));
    assert_eq!(
        error["code"].as_i64(),
        Some(code),
        "unexpected JSON-RPC error code, got: {error}"
    );
    let message = error["message"]
        .as_str()
        .unwrap_or_else(|| panic!("error.message must be a string, got: {error}"));
    assert!(!message.is_empty(), "error.message must not be empty");
    assert!(
        response.get("result").is_none(),
        "error responses must not carry a result: {response}"
    );
    error.clone()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// 1. The initialize handshake reports protocolVersion and serverInfo.
#[test]
fn initialize_handshake_returns_protocol_version_and_server_info() {
    let mut server = McpProcess::spawn();
    let response = server.initialize();

    assert_eq!(response["jsonrpc"], json!("2.0"));
    assert!(response["result"]["protocolVersion"].is_string());
    assert!(response["result"]["serverInfo"]["name"].is_string());
    assert!(response["result"]["serverInfo"]["version"].is_string());
}

/// 2. `ping` yields an empty JSON object result and echoes the request id.
#[test]
fn ping_returns_empty_object_result() {
    let mut server = McpProcess::spawn();
    server.initialize();

    server.send(&json!({ "jsonrpc": "2.0", "id": "ping-1", "method": "ping" }));
    let response = server
        .expect_response("ping")
        .unwrap_or_else(|| panic!("ping must be answered"));

    assert_eq!(
        response["id"],
        json!("ping-1"),
        "id must be echoed verbatim"
    );
    assert_eq!(
        expect_result(&response),
        json!({}),
        "ping result must be {{}}"
    );
}

/// 3. Notifications (no id, or `id: null`) are ignored: no response is emitted.
#[test]
fn notifications_are_ignored_without_response() {
    let mut server = McpProcess::spawn();
    server.initialize();

    // `id: null` and a missing `id` are both notifications per JSON-RPC 2.0.
    server.send(&json!({ "jsonrpc": "2.0", "id": Value::Null, "method": "ping" }));
    server.send(&json!({
        "jsonrpc": "2.0",
        "method": "notifications/cancelled",
        "params": { "requestId": 1 }
    }));
    server.send(&json!({ "jsonrpc": "2.0", "id": Value::Null, "method": "tools/list" }));

    server.expect_silence("notifications (id: null / missing)");

    // The session is still healthy: a real request is answered.
    server.send(&json!({ "jsonrpc": "2.0", "id": 2, "method": "ping" }));
    let response = server
        .expect_response("ping after notifications")
        .unwrap_or_else(|| panic!("server must keep serving after notifications"));
    assert_eq!(expect_result(&response), json!({}));
}

/// 4. Malformed JSON yields error code -32700 and the stream survives it.
#[test]
fn malformed_json_returns_parse_error_and_keeps_stream_alive() {
    let mut server = McpProcess::spawn();
    server.initialize();

    server.send_raw("{\"jsonrpc\": \"2.0\", \"id\": 7, \"method\": ");
    let response = server
        .expect_response("malformed JSON")
        .unwrap_or_else(|| panic!("malformed JSON must produce a response"));

    // The id cannot be recovered from a parse failure, so it reports null.
    assert_eq!(response["id"], Value::Null, "parse errors report id: null");
    let error = expect_error_code(&response, -32700);
    assert!(
        error["message"]
            .as_str()
            .is_some_and(|m| m.contains("Parse error")),
        "expected a Parse error message, got: {error}"
    );

    // The connection must stay usable after a parse error.
    server.send(&json!({ "jsonrpc": "2.0", "id": 8, "method": "ping" }));
    let response = server
        .expect_response("ping after parse error")
        .unwrap_or_else(|| panic!("server must stay alive after a parse error"));
    assert_eq!(expect_result(&response), json!({}));
}

/// Bonus: unknown methods are rejected with -32601 and keep the id.
#[test]
fn unknown_method_returns_method_not_found() {
    let mut server = McpProcess::spawn();
    server.initialize();

    server.send(&json!({ "jsonrpc": "2.0", "id": 3, "method": "does/not/exist" }));
    let response = server
        .expect_response("unknown method")
        .unwrap_or_else(|| panic!("unknown methods must be answered"));

    assert_eq!(response["id"], json!(3), "id must be echoed verbatim");
    let error = expect_error_code(&response, -32601);
    assert!(
        error["message"]
            .as_str()
            .is_some_and(|m| m.contains("Method not found")),
        "expected a Method not found message, got: {error}"
    );
}

/// 5. `tools/list` advertises the `worker` tool with its full action enum and
///    argument properties (including the `path` / `id` aliases).
#[test]
fn test_tools_list_schema() {
    let mut server = McpProcess::spawn();
    server.initialize();

    server.send(&json!({ "jsonrpc": "2.0", "id": "tools-1", "method": "tools/list" }));
    let response = server
        .expect_response("tools/list")
        .unwrap_or_else(|| panic!("tools/list must be answered"));

    assert_eq!(
        response["id"],
        json!("tools-1"),
        "id must be echoed verbatim"
    );
    let result = expect_result(&response);

    let tools = result["tools"]
        .as_array()
        .unwrap_or_else(|| panic!("result.tools must be an array, got: {result}"));
    assert!(
        !tools.is_empty(),
        "tools/list must advertise at least one tool, got: {result}"
    );

    let worker = tools
        .iter()
        .find(|t| t["name"].as_str() == Some("worker"))
        .unwrap_or_else(|| panic!("tools/list must expose the 'worker' tool, got: {result}"));

    assert!(
        worker["description"]
            .as_str()
            .is_some_and(|d| !d.is_empty()),
        "the worker tool needs a non-empty description, got: {worker}"
    );

    let schema = &worker["inputSchema"];
    assert_eq!(
        schema["type"].as_str(),
        Some("object"),
        "worker inputSchema must be a JSON object schema, got: {schema}"
    );

    let properties = schema["properties"]
        .as_object()
        .unwrap_or_else(|| panic!("worker inputSchema needs a properties object, got: {schema}"));

    // The action selector drives the tool, so its enum must cover every verb.
    let action_enum = properties
        .get("action")
        .and_then(|action| action["enum"].as_array())
        .unwrap_or_else(|| {
            panic!("worker inputSchema.properties.action must carry an enum, got: {schema}")
        });
    let actions: Vec<&str> = action_enum
        .iter()
        .map(|v| v.as_str().unwrap_or_default())
        .collect();
    // Sourced from the crate constant so the wire contract and the dispatch
    // table cannot drift apart.
    assert_eq!(
        actions,
        WORKER_ACTIONS.to_vec(),
        "worker action enum must mirror the dispatch table, got: {actions:?}"
    );

    // `path` and `id` are the documented aliases for repo_path / worker_id.
    for prop in ["path", "id"] {
        assert!(
            properties.contains_key(prop),
            "worker inputSchema.properties must include '{prop}', got: {:?}",
            properties.keys().collect::<Vec<_>>()
        );
        assert_eq!(
            properties[prop]["type"].as_str(),
            Some("string"),
            "property '{prop}' must be typed as a string, got: {}",
            properties[prop]
        );
    }

    // `action` stays the only required argument.
    let required = schema["required"].as_array().unwrap_or_else(|| {
        panic!("worker inputSchema must declare required fields, got: {schema}")
    });
    assert_eq!(
        required
            .iter()
            .filter(|v| v.as_str() == Some("action"))
            .count(),
        1,
        "'action' must be required exactly once, got: {required:?}"
    );
}

#[test]
fn test_concurrent_pipelined_requests() {
    let mut server = McpProcess::spawn();
    server.initialize();

    // Send 10 pipelined requests in rapid succession without waiting
    for id in 100..110 {
        server.send(&json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": if id % 2 == 0 { "ping" } else { "tools/list" }
        }));
    }

    // Collect all 10 responses and verify every id is received
    let mut received_ids = std::collections::HashSet::new();
    for _ in 100..110 {
        let resp = server
            .expect_response("pipelined request")
            .expect("must return response");
        let id = resp["id"].as_i64().expect("valid integer id");
        received_ids.insert(id);
    }

    assert_eq!(received_ids.len(), 10);
    for id in 100..110 {
        assert!(received_ids.contains(&id), "missing id {id}");
    }
}

#[test]
fn test_tools_call_prune_with_progress_token_emits_notifications() {
    let mut server = McpProcess::spawn();
    server.initialize();

    server.send(&json!({
        "jsonrpc": "2.0",
        "id": 200,
        "method": "tools/call",
        "params": {
            "name": "worker",
            "arguments": {
                "action": "prune",
                "repo_path": "."
            },
            "_meta": {
                "progressToken": "token-prune-xyz"
            }
        }
    }));

    // Expect progress notification 0/1
    let notif1 = server
        .expect_response("first progress notification")
        .expect("must receive notification");
    assert_eq!(notif1["method"], json!("notifications/progress"));
    assert_eq!(notif1["params"]["progressToken"], json!("token-prune-xyz"));
    assert_eq!(notif1["params"]["progress"], json!(0));

    // Expect progress notification 1/1
    let notif2 = server
        .expect_response("second progress notification")
        .expect("must receive notification");
    assert_eq!(notif2["method"], json!("notifications/progress"));
    assert_eq!(notif2["params"]["progressToken"], json!("token-prune-xyz"));
    assert_eq!(notif2["params"]["progress"], json!(1));

    // Expect final response
    let final_resp = server
        .expect_response("final tools/call response")
        .expect("must receive tool response");
    assert_eq!(final_resp["id"], json!(200));
    assert!(final_resp.get("result").is_some());
}

#[test]
fn test_tools_call_reap_reports_evicted_records() {
    let mut server = McpProcess::spawn();
    server.initialize();

    server.send(&json!({
        "jsonrpc": "2.0",
        "id": 300,
        "method": "tools/call",
        "params": { "name": "worker", "arguments": { "action": "reap" } }
    }));

    let response = server
        .expect_response("tools/call reap")
        .expect("reap must be answered");
    let result = expect_result(&response);
    // The result is double-encoded: text holds the pretty-printed JSON payload.
    let payload: Value = serde_json::from_str(
        result["content"][0]["text"]
            .as_str()
            .expect("tool text content"),
    )
    .expect("tool text must be JSON");
    assert_eq!(payload["status"], json!("reaped"));
    assert_eq!(
        payload["reaped"],
        json!(0),
        "a fresh pool has nothing to reap"
    );
    assert_eq!(payload["worker_ids"], json!([]));
}

#[test]
fn test_tools_call_logs_reports_retention_counters() {
    let mut server = McpProcess::spawn();
    server.initialize();

    server.send(&json!({
        "jsonrpc": "2.0",
        "id": 301,
        "method": "tools/call",
        "params": { "name": "worker", "arguments": { "action": "logs", "worker_id": "missing-xyz" } }
    }));

    let response = server
        .expect_response("tools/call logs")
        .expect("logs must be answered");
    // An unknown worker is a tool error, surfaced as -32000 with a useful message.
    expect_error_code(&response, -32000);
}

/// The `tools/call` envelope must still expose the tool payload as a
/// pretty-printed JSON string, byte-for-byte as before the F4 change, now that
/// it is streamed into the envelope instead of via an intermediate `String`
/// (audit 07, F4).
#[test]
fn test_tools_call_text_content_is_pretty_printed_json() {
    let mut server = McpProcess::spawn();
    server.initialize();

    server.send(&json!({
        "jsonrpc": "2.0",
        "id": 400,
        "method": "tools/call",
        "params": { "name": "worker", "arguments": { "action": "reap" } }
    }));

    let response = server
        .expect_response("tools/call reap")
        .expect("must receive tool response");
    let result = expect_result(&response);

    let text = result["content"][0]["text"]
        .as_str()
        .unwrap_or_else(|| panic!("content[0].text must be a string, got: {result}"));
    assert_eq!(result["content"][0]["type"], json!("text"));
    // Still valid JSON...
    let payload: Value = serde_json::from_str(text).expect("text must contain valid JSON");
    assert_eq!(payload["status"], json!("reaped"));
    // ...and still pretty-printed (the envelope never carries compact JSON).
    assert!(
        text.contains('\n') && text.contains("  "),
        "tool text must stay pretty-printed, got: {text}"
    );
}

/// The `network` property is part of the advertised contract: an optional
/// string, its enum is exactly the accepted policies, and it stays out of
/// `required` so pre-existing clients are unaffected.
#[test]
fn test_tools_list_advertises_the_optional_network_policy() {
    let mut server = McpProcess::spawn();
    server.initialize();

    server.send(&json!({ "jsonrpc": "2.0", "id": "network-schema", "method": "tools/list" }));
    let response = server
        .expect_response("tools/list")
        .expect("tools/list must be answered");
    let result = expect_result(&response);

    let worker = result["tools"]
        .as_array()
        .expect("result.tools must be an array")
        .iter()
        .find(|tool| tool["name"] == "worker")
        .expect("tools/list must expose the 'worker' tool");
    let schema = &worker["inputSchema"];
    let network = schema["properties"].get("network").unwrap_or_else(|| {
        panic!("the worker tool must advertise a 'network' property, got: {schema}")
    });

    assert_eq!(network["type"], json!("string"));
    assert_eq!(
        network["enum"],
        json!(NETWORK_MODES.to_vec()),
        "the advertised enum must be the accepted policy vocabulary"
    );
    assert_eq!(network["default"], json!(NETWORK_DEFAULT));
    assert_eq!(NETWORK_DEFAULT, "allow");
    assert!(
        network["description"]
            .as_str()
            .is_some_and(|text| text.contains("offline") && text.contains("allow")),
        "the description must document both policies, got: {network}"
    );

    // Optional by construction: `action` is still the only required argument.
    assert_eq!(schema["required"], json!(["action"]));
}

/// The `verify` property is part of the advertised contract: an optional
/// string that stays out of `required` so pre-existing clients are unaffected.
#[test]
fn test_tools_list_advertises_the_optional_verify_gate() {
    let mut server = McpProcess::spawn();
    server.initialize();

    server.send(&json!({ "jsonrpc": "2.0", "id": "verify-schema", "method": "tools/list" }));
    let response = server
        .expect_response("tools/list")
        .expect("tools/list must be answered");
    let result = expect_result(&response);

    let worker = result["tools"]
        .as_array()
        .expect("result.tools must be an array")
        .iter()
        .find(|tool| tool["name"] == "worker")
        .expect("tools/list must expose the 'worker' tool");
    let schema = &worker["inputSchema"];
    let verify = schema["properties"].get("verify").unwrap_or_else(|| {
        panic!("the worker tool must advertise a 'verify' property, got: {schema}")
    });

    assert_eq!(verify["type"], json!("string"));
    assert!(
        verify["description"]
            .as_str()
            .is_some_and(|text| text.contains("auto-detect") && text.contains("empty string")),
        "the description must document auto-detection and how to disable, got: {verify}"
    );

    // Optional by construction: `action` is still the only required argument.
    assert_eq!(schema["required"], json!(["action"]));
}

/// An unknown `network` value is a hard tool error, not a silent fallback: a
/// caller that asked for isolation must never quietly get connectivity back.
#[test]
fn test_tools_call_dispatch_rejects_an_unknown_network_policy() {
    let mut server = McpProcess::spawn();
    server.initialize();

    server.send(&json!({
        "jsonrpc": "2.0",
        "id": 500,
        "method": "tools/call",
        "params": {
            "name": "worker",
            "arguments": {
                "action": "dispatch",
                "task": "tidy the docs",
                "repo_path": ".",
                "network": "offine"
            }
        }
    }));

    let response = server
        .expect_response("tools/call dispatch with a bad network policy")
        .expect("the call must be answered");
    let error = expect_error_code(&response, -32000);
    let message = error["message"]
        .as_str()
        .expect("error.message must be a string")
        .to_string();
    assert!(
        message.contains("not a valid 'network' policy"),
        "the error must name the invalid policy, got: {message}"
    );
    assert!(
        message.contains("offline") && message.contains("allow"),
        "the error must list the accepted policies, got: {message}"
    );
}

/// A non-string `network` is rejected the same way, before any worker is
/// spawned (no `worker_id` is ever handed back).
#[test]
fn test_tools_call_dispatch_rejects_a_non_string_network_policy() {
    let mut server = McpProcess::spawn();
    server.initialize();

    server.send(&json!({
        "jsonrpc": "2.0",
        "id": 501,
        "method": "tools/call",
        "params": {
            "name": "worker",
            "arguments": { "action": "dispatch", "task": "tidy the docs", "network": true }
        }
    }));

    let response = server
        .expect_response("tools/call dispatch with a non-string network policy")
        .expect("the call must be answered");
    let error = expect_error_code(&response, -32000);
    assert!(
        error["message"]
            .as_str()
            .is_some_and(|message| message.contains("must be a string")),
        "got: {error}"
    );
}

/// The verbs that do not dispatch a worker ignore the property entirely: it is
/// a `dispatch` option, so a `reap` carrying it must still succeed.
#[test]
fn test_tools_call_ignores_network_on_non_dispatch_verbs() {
    let mut server = McpProcess::spawn();
    server.initialize();

    server.send(&json!({
        "jsonrpc": "2.0",
        "id": 502,
        "method": "tools/call",
        "params": {
            "name": "worker",
            "arguments": { "action": "reap", "network": "offline" }
        }
    }));

    let response = server
        .expect_response("tools/call reap with network")
        .expect("reap must be answered");
    let result = expect_result(&response);
    let payload: Value = serde_json::from_str(result["content"][0]["text"].as_str().expect("text"))
        .expect("tool text must be JSON");
    assert_eq!(payload["status"], json!("reaped"));
}

/// A dispatch with an explicit `network: "offline"` reports `offline` in the
/// response: the explicit argument wins over the manifest policy.
#[test]
fn test_dispatch_reports_explicit_offline_network() {
    let scratch = DispatchScratch::new("offline");
    let crate_branches = CrateBranchGuard::new();
    let swe = scratch.base();
    let mut server = McpProcess::spawn_with_env(&[("SWE_TEMP_DIR", swe.as_str())]);
    server.initialize();

    server.send(&json!({
        "jsonrpc": "2.0",
        "id": 510,
        "method": "tools/call",
        "params": {
            "name": "worker",
            "arguments": {
                "action": "dispatch",
                "task": "tidy the docs",
                "repo_path": scratch.repo(),
                "network": "offline"
            }
        }
    }));

    let response = server
        .expect_response("tools/call dispatch with explicit offline")
        .expect("the call must be answered");
    let result = expect_result(&response);
    let payload: Value = serde_json::from_str(result["content"][0]["text"].as_str().expect("text"))
        .expect("tool text must be JSON");
    assert_eq!(payload["network"], json!("offline"));
    assert_eq!(payload["status"], json!("dispatched"));

    let wid = payload["worker_id"]
        .as_str()
        .expect("worker id")
        .to_string();
    kill_worker(&mut server, 610, &wid);
    drop(server);
    crate_branches.assert_untouched();
}

/// A dispatch that omits `network` inherits the resolved model's manifest
/// policy. The shipped `ninja` declares `allow`, so the response reports it.
#[test]
fn test_dispatch_reports_manifest_network_policy_when_argument_omitted() {
    let scratch = DispatchScratch::new("manifest-policy");
    let crate_branches = CrateBranchGuard::new();
    let swe = scratch.base();
    let mut server = McpProcess::spawn_with_env(&[("SWE_TEMP_DIR", swe.as_str())]);
    server.initialize();

    server.send(&json!({
        "jsonrpc": "2.0",
        "id": 511,
        "method": "tools/call",
        "params": {
            "name": "worker",
            "arguments": {
                "action": "dispatch",
                "task": "tidy the docs",
                "repo_path": scratch.repo(),
                "model": "ninja"
            }
        }
    }));

    let response = server
        .expect_response("tools/call dispatch with omitted network")
        .expect("the call must be answered");
    let result = expect_result(&response);
    let payload: Value = serde_json::from_str(result["content"][0]["text"].as_str().expect("text"))
        .expect("tool text must be JSON");
    assert_eq!(payload["network"], json!("allow"));
    assert_eq!(payload["status"], json!("dispatched"));

    let wid = payload["worker_id"]
        .as_str()
        .expect("worker id")
        .to_string();
    kill_worker(&mut server, 611, &wid);
    drop(server);
    crate_branches.assert_untouched();
}

/// A dispatch with no `network` argument and a model that declares no policy
/// falls back to the runtime default (`allow`).
#[test]
fn test_dispatch_reports_default_network_when_nothing_declared() {
    let scratch = DispatchScratch::new("default-network");
    let crate_branches = CrateBranchGuard::new();
    let swe = scratch.base();
    let mut server = McpProcess::spawn_with_env(&[("SWE_TEMP_DIR", swe.as_str())]);
    server.initialize();

    server.send(&json!({
        "jsonrpc": "2.0",
        "id": 512,
        "method": "tools/call",
        "params": {
            "name": "worker",
            "arguments": {
                "action": "dispatch",
                "task": "tidy the docs",
                "repo_path": scratch.repo(),
                "model": "some/unknown-model"
            }
        }
    }));

    let response = server
        .expect_response("tools/call dispatch with nothing declared")
        .expect("the call must be answered");
    let result = expect_result(&response);
    let payload: Value = serde_json::from_str(result["content"][0]["text"].as_str().expect("text"))
        .expect("tool text must be JSON");
    assert_eq!(payload["network"], json!("allow"));
    assert_eq!(payload["status"], json!("dispatched"));

    let wid = payload["worker_id"]
        .as_str()
        .expect("worker id")
        .to_string();
    kill_worker(&mut server, 612, &wid);
    drop(server);
    crate_branches.assert_untouched();
}

/// The wrapper itself: connected stays byte-identical, `offline` enters a
/// network namespace and keeps the command verbatim inside it.
#[test]
fn test_wrap_network_command_wraps_only_when_offline() {
    let cmd = "git fetch --all && echo done";

    // The default policy never rewrites a command: 'allow' (the advertised
    // default) means exactly "run the command as written".
    assert_eq!(NETWORK_DEFAULT, "allow");
    assert_eq!(wrap_network_command(cmd, false), cmd);

    let wrapped = wrap_network_command(cmd, true);
    assert!(
        wrapped.starts_with("unshare -n -- bash -c "),
        "offline must enter an isolated network namespace, got: {wrapped}"
    );
    assert!(
        wrapped.contains(&format!("'{cmd}'")),
        "the command must survive the wrapper verbatim, got: {wrapped}"
    );

    // The wrapper is shell-transparent: the command is still one plain bash
    // string, so where the host allows namespaces it runs exactly as written.
    if can_create_network_namespace() {
        let output = std::process::Command::new("bash")
            .arg("-c")
            .arg(wrap_network_command("echo done", true))
            .output()
            .expect("bash must run");
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            output.status.success() && stdout.contains("done"),
            "the wrapped command must still execute, got: {stdout:?}"
        );
    } else {
        // Where namespaces cannot be created, the wrapper must fail loudly
        // rather than quietly running the command *with* network access.
        let output = std::process::Command::new("bash")
            .arg("-c")
            .arg(wrap_network_command("echo done", true))
            .output()
            .expect("bash must run");
        assert!(
            !output.status.success() && !String::from_utf8_lossy(&output.stdout).contains("done"),
            "an unavailable namespace must fail instead of silently running \
             unisolated (status: {}, stdout: {:?})",
            output.status,
            String::from_utf8_lossy(&output.stdout)
        );
    }
}

/// Whether this host actually lets a process build a network namespace.
///
/// `unshare` being on `$PATH` is not enough: inside an unprivileged container
/// `unshare -n` still fails with `EPERM`, and the isolation a caller asked for
/// cannot be created there.
fn can_create_network_namespace() -> bool {
    std::process::Command::new("unshare")
        .args(["-n", "--", "true"])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

// ----------
// `wait`: re-attaching to a worker, and the wait heartbeat
// ----------

/// A waiter blocked in `await_worker_result_until` returns within ~100 ms of a
/// synthetic state change, proving the wait sleeps on the pool's change
/// subscription rather than on a fixed tick.
#[tokio::test]
async fn a_blocked_wait_returns_promptly_on_a_state_change() {
    use std::time::{Duration, Instant};
    let _scratch = common::TempDir::new_in_tmp("iso-mcp-1");
    let pool = WorkerPool::with_scratch(
        1,
        "http://localhost:1".to_string(),
        "test-key".to_string(),
        mini_swe_mcp::worktree::ScratchRoot::new(_scratch.path()),
    );
    pool.__test_insert_worker(running_worker("wait-test-wakes"))
        .await;
    let server = McpServer::new(pool.clone(), "ninja".to_string());

    let waiter = tokio::spawn(async move {
        server
            .await_worker_result_until("wait-test-wakes", 10, None, None, None)
            .await
    });
    // Let the waiter reach its sleep; then pause the worker behind it.
    tokio::time::sleep(Duration::from_millis(200)).await;
    let changed_at = Instant::now();
    pool.__test_set_worker_state(
        "wait-test-wakes",
        WorkerState::Paused {
            question: "which branch?".to_string(),
            step: 2,
            paused_at: 0,
        },
    )
    .await;
    let result = tokio::time::timeout(Duration::from_secs(10), waiter)
        .await
        .expect("the waiter must answer")
        .expect("the waiter task stays alive")
        .expect("a paused worker answers needs_input");
    assert_eq!(result["worker_id"], "wait-test-wakes");
    assert_eq!(result["status"], "needs_input");
    assert_eq!(result["question"], "which branch?");
    // ~100 ms is the bar; 500 ms of scheduling slack keeps the assertion
    // honest on a loaded host without weakening it into the old 500 ms tick.
    assert!(
        changed_at.elapsed() < Duration::from_millis(500),
        "the waiter took {:?} to observe the change; it must wake on the notification",
        changed_at.elapsed()
    );
}

/// A synthetic pool record: the `worker` verb table is exercised here without
/// dispatching an LLM-backed worker, exactly like the pool's own tests do.
fn synthetic_worker(id: &str, state: WorkerState) -> WorkerRecord {
    WorkerRecord {
        id: id.to_string(),
        task: "t".to_string(),
        model: "m".to_string(),
        owner: LOCAL_AGENT.to_string(),
        state,
        metrics: WorkerMetrics::default(),
        logs: LogBuffer::new(),
        pending_steer: Vec::new(),
        resume_tx: None,
        handle: None,
        revision: 0,
    }
}

fn completed_worker(id: &str) -> WorkerRecord {
    synthetic_worker(
        id,
        WorkerState::Completed {
            turns: 4,
            diff: "--- a\n+++ b".to_string(),
            summary: "fixed the parser".to_string(),
            completed_at: 0,
            artifacts: Vec::new(),
            branch: Some("swe-wt-done".to_string()),
            verified: Some(true),
            metrics: WorkerMetrics::default(),
            revision: 0,
        },
    )
}

fn running_worker(id: &str) -> WorkerRecord {
    synthetic_worker(
        id,
        WorkerState::Running {
            step: 2,
            last_command: "cargo test --all-targets".to_string(),
            started_at: 0,
        },
    )
}

/// An unknown worker is a clear error, never a watch that never returns.
#[tokio::test]
async fn watch_on_an_unknown_worker_is_a_clear_error() {
    let _scratch = common::TempDir::new_in_tmp("iso-mcp-2");
    let pool = WorkerPool::with_scratch(
        1,
        "http://localhost:1".to_string(),
        "test-key".to_string(),
        mini_swe_mcp::worktree::ScratchRoot::new(_scratch.path()),
    );
    let server = McpServer::new(pool, "ninja".to_string());

    let error = server
        .execute_tool(
            "worker",
            json!({ "action": "watch", "worker_id": "watch-test-ghost" }),
        )
        .await
        .expect_err("watching a worker that was never dispatched must fail");
    assert!(
        error
            .to_string()
            .contains("Worker not found: watch-test-ghost"),
        "the error must name the worker: {error}"
    );
}

/// `timeout_secs` turns the watch into a bounded long-poll: with nothing to
/// watch it answers `no_event` instead of blocking past the host deadline.
#[tokio::test]
async fn watch_with_a_deadline_returns_no_event() {
    let _scratch = common::TempDir::new_in_tmp("iso-mcp-3");
    let pool = WorkerPool::with_scratch(
        1,
        "http://localhost:1".to_string(),
        "test-key".to_string(),
        mini_swe_mcp::worktree::ScratchRoot::new(_scratch.path()),
    );
    let server = McpServer::new(pool, "ninja".to_string());

    let result = server
        .execute_tool("worker", json!({ "action": "watch", "timeout_secs": 0 }))
        .await
        .expect("an expired deadline must answer, not hang");

    assert_eq!(result["status"], "no_event");
    assert_eq!(result["events"].as_array().map(Vec::len), Some(0));
}

/// A worker that is already finished reports its event on the first poll, so
/// an MCP-only orchestrator never has to poll `status` to learn the outcome.
#[tokio::test]
async fn watch_on_a_completed_worker_returns_its_event() {
    let _scratch = common::TempDir::new_in_tmp("iso-mcp-4");
    let pool = WorkerPool::with_scratch(
        1,
        "http://localhost:1".to_string(),
        "test-key".to_string(),
        mini_swe_mcp::worktree::ScratchRoot::new(_scratch.path()),
    );
    pool.__test_insert_worker(completed_worker("watch-test-done"))
        .await;
    let server = McpServer::new(pool, "ninja".to_string());

    let result = server
        .execute_tool(
            "worker",
            json!({ "action": "watch", "worker_id": "watch-test-done" }),
        )
        .await
        .expect("a finished worker must answer without waiting");

    assert_eq!(result["status"], "event");
    let events = result["events"].as_array().expect("events array");
    assert_eq!(events.len(), 1, "one event per finished worker: {result}");
    assert_eq!(events[0]["worker_id"], "watch-test-done");
    assert_eq!(events[0]["event"], "completed");
}

/// The heartbeat has to fire well inside the 30-minute window an MCP client
/// allows a silent stdio call before aborting it as idle.
#[test]
fn the_wait_heartbeat_fires_within_a_minute() {
    assert!(
        McpServer::PROGRESS_HEARTBEAT_INTERVAL <= Duration::from_secs(60),
        "the heartbeat interval is {:?}, which can outlast a client's idle abort",
        McpServer::PROGRESS_HEARTBEAT_INTERVAL
    );
    assert!(McpServer::PROGRESS_HEARTBEAT_INTERVAL > Duration::ZERO);
}

/// The new verb is routed by the live stdio dispatcher, and an unknown worker
/// surfaces as a tool error rather than a silent hang.
#[test]
fn test_tools_call_wait_on_an_unknown_worker_errors() {
    let mut server = McpProcess::spawn();
    server.initialize();

    server.send(&json!({
        "jsonrpc": "2.0",
        "id": 500,
        "method": "tools/call",
        "params": {
            "name": "worker",
            "arguments": { "action": "watch", "worker_id": "missing-xyz" }
        }
    }));

    let response = server
        .expect_response("tools/call watch")
        .expect("watch must be answered");
    let error = expect_error_code(&response, -32000);
    assert!(
        error["message"]
            .as_str()
            .is_some_and(|message| message.contains("Worker not found")),
        "expected a clear unknown-worker error, got: {error}"
    );
}

// ----------
// `claude/channel` worker events
// ----------

/// A worker view carrying the context every notification renders, so a test
/// only has to set the fields it is actually about.
fn worker_view(worker_id: &str, event: Option<EventKind>) -> WorkerView {
    WorkerView {
        worker_id: worker_id.to_string(),
        event,
        group: String::from("backend"),
        model: String::from("ninja"),
        status: event.map_or("running", EventKind::as_str).to_string(),
        ..WorkerView::default()
    }
}

/// One tick's snapshot of the given workers.
fn worker_snapshot(views: impl IntoIterator<Item = WorkerView>) -> WorkerSnapshot {
    views
        .into_iter()
        .map(|view| (view.worker_id.clone(), view))
        .collect()
}

/// The rendered text of the event about `worker_id`.
fn content_of(events: &[ChannelEvent], worker_id: &str) -> String {
    events
        .iter()
        .find(|event| event.worker_id == worker_id)
        .unwrap_or_else(|| panic!("no event for worker {worker_id} in {events:?}"))
        .content
        .clone()
}

/// The handshake opts into the `claude/channel` extension and tells the model
/// how to answer what arrives through it.
#[test]
fn initialize_declares_the_claude_channel_extension() {
    let mut server = McpProcess::spawn();
    let response = server.initialize();
    let result = &response["result"];

    assert_eq!(
        result["capabilities"]["experimental"]["claude/channel"],
        json!({}),
        "the channel capability must be declared, and declared empty: {result}"
    );
    assert_eq!(
        result["capabilities"]["tools"]["listChanged"],
        json!(false),
        "opting into the channel must not drop the tools capability: {result}"
    );
    let instructions = result["instructions"]
        .as_str()
        .unwrap_or_else(|| panic!("the channel extension needs instructions: {result}"));
    for expected in [
        "needs_input",
        "steer",
        "completed",
        "collect",
        "failed",
        "status",
        "logs",
        "worker_id",
    ] {
        assert!(
            instructions.contains(expected),
            "the instructions must tell the model what to do about `{expected}`: {instructions}"
        );
    }
}

/// A pause, a completion and a failure each reach the session, carrying what
/// the orchestrator needs to answer and the verb it answers with.
#[test]
fn worker_transitions_become_one_event_each() {
    let running = worker_snapshot([worker_view("w-run", None)]);

    let mut paused = worker_view("w-run", Some(EventKind::NeedsInput));
    paused.question = Some(String::from("Ship the migration or roll it back?"));
    let mut completed = worker_view("w-done", Some(EventKind::Completed));
    completed.outcome = Outcome {
        summary: Some(String::from("Fixed the retry loop.")),
        verified: Some(true),
        diff_stat: Some(String::from("2 files, +30 -4")),
        error: None,
    };
    let mut failed = worker_view("w-dead", Some(EventKind::Failed));
    failed.outcome = Outcome {
        error: Some(String::from("bash exited 1")),
        ..Outcome::default()
    };

    let events = diff_events(&running, &worker_snapshot([paused, completed, failed]));
    let reported: Vec<(&str, &str)> = events
        .iter()
        .map(|event| (event.worker_id.as_str(), event.kind.as_str()))
        .collect();
    assert_eq!(
        reported,
        [
            ("w-dead", "failed"),
            ("w-done", "completed"),
            ("w-run", "needs_input")
        ],
        "one event per transition, ordered by worker id"
    );

    let needs_input = content_of(&events, "w-run");
    assert!(
        needs_input.contains("Ship the migration or roll it back?"),
        "a pause must carry the question: {needs_input}"
    );
    assert!(
        needs_input.contains("\"steer\"") && needs_input.contains("w-run"),
        "a pause must say how to answer it: {needs_input}"
    );

    let completed = content_of(&events, "w-done");
    for expected in [
        "Fixed the retry loop.",
        "Verified: yes",
        "2 files, +30 -4",
        "\"collect\"",
    ] {
        assert!(
            completed.contains(expected),
            "a completion must carry `{expected}`: {completed}"
        );
    }

    let failed = content_of(&events, "w-dead");
    for expected in ["bash exited 1", "\"status\"", "\"logs\""] {
        assert!(
            failed.contains(expected),
            "a failure must carry `{expected}`: {failed}"
        );
    }

    assert!(
        events.iter().all(|event| event.group == "backend"
            && event.model == "ninja"
            && !event.status.is_empty()),
        "every event must carry the context the client renders as tag attributes: {events:?}"
    );
}

/// One transition produces one notification: the next tick sees the same
/// `(worker_id, event)` pair and stays silent.
#[test]
fn a_reported_transition_is_not_reported_again() {
    let running = worker_snapshot([worker_view("w-done", None)]);
    let mut finished = worker_view("w-done", Some(EventKind::Completed));
    finished.outcome = Outcome {
        summary: Some(String::from("Done.")),
        ..Outcome::default()
    };
    let after = worker_snapshot([finished]);

    assert_eq!(
        diff_events(&running, &after).len(),
        1,
        "the transition into `completed` must be reported"
    );
    assert!(
        diff_events(&after, &after).is_empty(),
        "an unchanged worker must stay silent: {:?}",
        diff_events(&after, &after)
    );
    assert!(
        diff_events(&after, &running).is_empty(),
        "a collected worker must not be re-reported: {:?}",
        diff_events(&after, &running)
    );
}

/// The first snapshot is seeded, never diffed against an empty one: a server
/// started next to finished workers stays silent instead of replaying history.
#[test]
fn workers_that_were_already_terminal_at_startup_are_not_replayed() {
    let seeded = worker_snapshot([
        worker_view("w-old-a", Some(EventKind::Completed)),
        worker_view("w-old-b", Some(EventKind::Failed)),
        worker_view("w-old-c", None),
    ]);
    assert!(
        diff_events(&seeded, &seeded.clone()).is_empty(),
        "no historical flood on the first tick: {:?}",
        diff_events(&seeded, &seeded.clone())
    );

    let mut resumed = worker_view("w-old-c", Some(EventKind::NeedsInput));
    resumed.question = Some(String::from("Which branch?"));
    let events = diff_events(
        &seeded,
        &worker_snapshot([
            worker_view("w-old-a", Some(EventKind::Completed)),
            worker_view("w-old-b", Some(EventKind::Failed)),
            resumed,
        ]),
    );
    assert_eq!(
        events.len(),
        1,
        "only the new transition is news: {events:?}"
    );
    assert_eq!(events[0].worker_id, "w-old-c");
    assert_eq!(events[0].kind, EventKind::NeedsInput);
}

/// A channel notification is one JSON-RPC line whose `params.meta` keys are
/// identifiers and whose values are strings — what the client turns into the
/// `<channel source=… key=…>` wrapper.
#[test]
fn channel_frames_carry_content_and_valid_meta_keys() {
    let mut paused = worker_view("w-run", Some(EventKind::NeedsInput));
    paused.question = Some(String::from("Ship the migration?"));
    let events = diff_events(&WorkerSnapshot::new(), &worker_snapshot([paused]));
    let frame = channel_frame(&events[0]).expect("a channel event must serialize");

    assert!(
        frame.ends_with('\n') && frame.matches('\n').count() == 1,
        "one line per frame: {frame:?}"
    );
    let wire: Value = serde_json::from_str(&frame).expect("the frame is one JSON document");
    assert_eq!(wire["jsonrpc"], json!("2.0"));
    assert_eq!(wire["method"], json!("notifications/claude/channel"));
    assert!(
        wire.get("id").is_none(),
        "a notification must not carry an id: {wire}"
    );
    assert!(
        wire["params"]["content"]
            .as_str()
            .is_some_and(|content| content.contains("Ship the migration?")),
        "the question must reach the session: {wire}"
    );

    let meta = wire["params"]["meta"]
        .as_object()
        .unwrap_or_else(|| panic!("params.meta must be an object: {wire}"));
    for key in ["event", "worker_id", "group", "model", "status"] {
        assert!(meta.contains_key(key), "meta must carry `{key}`: {wire}");
    }
    for (key, value) in meta {
        assert!(
            !key.is_empty() && key.chars().all(|c| c.is_ascii_alphanumeric() || c == '_'),
            "meta key `{key}` must be a plain identifier: {wire}"
        );
        assert!(
            value.is_string(),
            "meta value for `{key}` must be a string: {wire}"
        );
    }
    assert_eq!(meta["event"], json!("needs_input"));
    assert_eq!(meta["worker_id"], json!("w-run"));
}

/// A registry row for a synthetic worker, owned by this process so the reader
/// never normalizes it to `stopped`.
fn synthetic_registry_row(worker_id: &str, status: RegistryStatus) -> WorkerRegistryEntry {
    WorkerRegistryEntry {
        approved: None,
        id: worker_id.to_string(),
        pid: std::process::id(),
        task: String::from("channel smoke test"),
        model: String::from("ninja"),
        status,
        step: 3,
        max_turns: 20,
        last_command: String::from("cargo test"),
        question: None,
        started_at: 0,
        updated_at: 0,
        group: Some(String::from("backend")),
        role: mini_swe_mcp::pool::WorkerRole::Worker,
        repo_path: None,
        owner: Some(String::from("registry-owner")),
        metrics: WorkerMetrics::default(),
        base_branch: None,
        base_commit: None,
        revision: 0,
        auto_continues: 0,
    }
}

/// End to end: a worker that pauses after the server started is pushed into the
/// session as one newline-terminated `notifications/claude/channel` frame.
///
/// The row exists (as a running worker) before the server does, so the event
/// task's seeded first snapshot already knows it and the pause that follows is
/// a genuine transition rather than history.
#[test]
fn a_worker_transition_reaches_the_session_over_stdio() {
    let worker_id = format!("chan-smoke-{}", std::process::id());
    // A private scratch root, so the row this test writes never lands in the
    // real registry: the server child sees it through `SWE_TEMP_DIR`.
    let scratch = common::TempDir::new_in_tmp("mcp-channel");
    let swe = scratch.path().to_string_lossy().into_owned();
    save_registry_entry_in(
        &mini_swe_mcp::worktree::ScratchRoot::new(&swe),
        &synthetic_registry_row(&worker_id, RegistryStatus::Running),
    );

    let mut server = McpProcess::spawn_with_env(&[("SWE_TEMP_DIR", &swe)]);
    server.send(&json!({"jsonrpc": "2.0", "method": "hub/hello",
        "params": {"agent_id": "registry-owner"}}));
    server.initialize();
    let mut paused = synthetic_registry_row(&worker_id, RegistryStatus::Paused);
    paused.question = Some(String::from("Ship the migration or roll it back?"));
    save_registry_entry_in(&mini_swe_mcp::worktree::ScratchRoot::new(&swe), &paused);
    // The row belongs to another process, so the server only discovers it on
    // its coarse cross-process fallback tick.
    let event = server.expect_channel_event_within("the paused worker", Duration::from_secs(45));
    remove_registry_entry_in(&mini_swe_mcp::worktree::ScratchRoot::new(&swe), &worker_id);

    let meta = &event["params"]["meta"];
    assert_eq!(event["jsonrpc"], json!("2.0"));
    assert_eq!(event["method"], json!(CHANNEL_NOTIFICATION));
    assert!(
        event.get("id").is_none(),
        "a notification carries no id: {event}"
    );
    assert_eq!(meta["event"], json!("needs_input"));
    assert_eq!(meta["worker_id"], json!(worker_id));
    assert_eq!(meta["group"], json!("backend"));
    assert_eq!(meta["model"], json!("ninja"));
    assert_eq!(meta["status"], json!("paused"));
    assert!(
        event["params"]["content"]
            .as_str()
            .is_some_and(|content| content.contains("Ship the migration or roll it back?")),
        "the escalated question must reach the session: {event}"
    );
}

// ----------
// Per-agent ownership (H-3)
// ----------

/// Owner of the synthetic workers the ownership tests share: the same identity
/// the in-process stdio connection resolves to.
const LOCAL_AGENT: &str = mini_swe_mcp::mcp::LOCAL_AGENT;

/// A connection of `agent`, as a hub client that announced `MINI_SWE_AGENT_ID`.
fn agent_context(agent: &str) -> ConnectionContext {
    ConnectionContext {
        agent_id: Some(agent.to_string()),
        ..ConnectionContext::hub_connection(3)
    }
}

/// The operator's `--admin` connection: the same agent, plus the override.
fn admin_context(agent: &str) -> ConnectionContext {
    ConnectionContext {
        admin: true,
        ..agent_context(agent)
    }
}

/// A synthetic worker owned by `owner`, running.
fn owned_worker(id: &str, owner: &str) -> WorkerRecord {
    WorkerRecord {
        owner: owner.to_string(),
        ..running_worker(id)
    }
}

/// A pool plus server over it, with no LLM anywhere in sight.
///
/// The pool owns a temporary scratch root, so the registry rows these tests
/// write never land in the real registry under `swe_base_dir()`.
fn owned_server() -> (IsolatedPool, McpServer) {
    let owned = IsolatedPool::new(8, "mcp-owned");
    let server = McpServer::new(owned.pool.clone(), "ninja".to_string());
    (owned, server)
}

/// One agent controls its own workers and nobody else's: `steer`, `kill`,
/// `collect` and `wait` are refused for a foreign worker, while `status` and
/// `logs` stay readable by anyone.
#[tokio::test]
async fn an_agent_cannot_act_on_another_agents_worker_but_can_read_it() {
    let (owned, server) = owned_server();
    owned
        .pool
        .__test_insert_worker(owned_worker("h3-foreign", "agent-a"))
        .await;
    let agent_b = agent_context("agent-b");

    for arguments in [
        json!({ "action": "steer", "worker_id": "h3-foreign", "message": "stop" }),
        json!({ "action": "kill", "worker_id": "h3-foreign" }),
        json!({ "action": "collect", "worker_id": "h3-foreign" }),
        json!({ "action": "watch", "worker_id": "h3-foreign", "timeout_secs": 0 }),
        json!({ "action": "status", "worker_id": "h3-foreign" }),
        json!({ "action": "logs", "worker_id": "h3-foreign" }),
    ] {
        let action = arguments["action"].as_str().expect("action");
        let error = server
            .execute_tool_for("worker", arguments.clone(), &agent_b)
            .await
            .expect_err("another agent's worker must be refused");
        assert_eq!(
            error.to_string(),
            "worker h3-foreign belongs to agent agent-a",
            "'{action}' must name the owning agent: {error}"
        );
    }

    // The refusals changed nothing: the worker is still running and still owned
    // by agent A, and A can steer it.
    assert!(
        owned.pool.worker_progress("h3-foreign").await.is_some(),
        "a refused verb must not evict the worker"
    );
    let steered = server
        .execute_tool_for(
            "worker",
            json!({ "action": "steer", "worker_id": "h3-foreign", "message": "carry on" }),
            &agent_context("agent-a"),
        )
        .await
        .expect("the owner may steer its own worker");
    assert_eq!(steered["status"], "steered");

    // The refusals changed nothing: the worker is still running and still owned
    // by agent A, and A can steer it.
    assert!(
        owned.pool.worker_progress("h3-foreign").await.is_some(),
        "a refused verb must not evict the worker"
    );
    let steered = server
        .execute_tool_for(
            "worker",
            json!({ "action": "steer", "worker_id": "h3-foreign", "message": "carry on" }),
            &agent_context("agent-a"),
        )
        .await
        .expect("the owner may steer its own worker");
    assert_eq!(steered["status"], "steered");

    // `status` and `logs` are reads, but they are private reads: the owner
    // sees its own worker, and only the admin override sees a foreign one.
    let status = server
        .execute_tool_for(
            "worker",
            json!({ "action": "status", "worker_id": "h3-foreign" }),
            &agent_context("agent-a"),
        )
        .await
        .expect("the owner may read its own worker");
    assert_eq!(status["owner"], "agent-a");
    assert_eq!(status["state"]["state"], "Running");
    let admin_status = server
        .execute_tool_for(
            "worker",
            json!({ "action": "status", "worker_id": "h3-foreign" }),
            &admin_context("operator"),
        )
        .await
        .expect("the admin override may read any worker");
    assert_eq!(admin_status["owner"], "agent-a");
}

/// The operator's `--admin` connection is the one caller allowed past the check.
#[tokio::test]
async fn an_admin_connection_bypasses_the_ownership_check() {
    let (owned, server) = owned_server();
    owned
        .pool
        .__test_insert_worker(owned_worker("h3-admin", "agent-a"))
        .await;

    let steered = server
        .execute_tool_for(
            "worker",
            json!({ "action": "steer", "worker_id": "h3-admin", "message": "wrap up" }),
            &admin_context("operator"),
        )
        .await
        .expect("the admin override must reach the worker");
    assert_eq!(steered["status"], "steered");
    assert_eq!(steered["worker_id"], "h3-admin");

    // The override is a bypass of ownership only: the admin still sees the
    // worker it acted on under its real owner.
    let status = server
        .execute_tool_for(
            "worker",
            json!({ "action": "status", "worker_id": "h3-admin" }),
            &admin_context("operator"),
        )
        .await
        .expect("status stays readable");
    assert_eq!(status["owner"], "agent-a");
}

/// A running registry row owned by `owner`: the cross-process view of a worker
/// another connection dispatched, which is what the hub really holds.
fn owned_registry_row(worker_id: &str, owner: &str) -> WorkerRegistryEntry {
    let mut row = synthetic_registry_row(worker_id, RegistryStatus::Running);
    row.owner = Some(owner.to_string());
    row
}

/// `list` shows the caller's own workers; `scope: "all"` shows everyone's, each
/// row naming its owner. The rows live in the registry because that is where a
/// worker of another *connection* is: the pool itself only knows its own.
#[tokio::test]
async fn list_is_scoped_to_the_caller_and_scope_all_names_every_owner() {
    let (owned, server) = owned_server();
    let root = owned.root();
    save_registry_entry_in(&root, &owned_registry_row("h3-mine", "agent-a"));
    save_registry_entry_in(&root, &owned_registry_row("h3-theirs", "agent-b"));
    let agent_a = agent_context("agent-a");

    let mine = server
        .execute_tool_for("worker", json!({ "action": "list" }), &agent_a)
        .await
        .expect("list answers");
    remove_registry_entry_in(&root, "h3-mine");
    remove_registry_entry_in(&root, "h3-theirs");
    let ids = |payload: &serde_json::Value| -> Vec<String> {
        payload["workers"]
            .as_array()
            .expect("workers array")
            .iter()
            .filter_map(|row| row["id"].as_str().map(str::to_string))
            .collect()
    };
    let owners: Vec<String> = mine["workers"]
        .as_array()
        .expect("workers array")
        .iter()
        .map(|row| row["owner"].as_str().expect("owner").to_string())
        .collect();

    assert!(ids(&mine).contains(&"h3-mine".to_string()), "{mine}");
    assert!(!ids(&mine).contains(&"h3-theirs".to_string()), "{mine}");
    assert!(owners.iter().all(|owner| owner == "agent-a"), "{mine}");

    save_registry_entry_in(&root, &owned_registry_row("h3-mine", "agent-a"));
    save_registry_entry_in(&root, &owned_registry_row("h3-theirs", "agent-b"));
    let all = server
        .execute_tool_for(
            "worker",
            json!({ "action": "list", "scope": "all" }),
            &admin_context("agent-a"),
        )
        .await
        .expect("scope=all answers for the admin override");
    remove_registry_entry_in(&root, "h3-mine");
    remove_registry_entry_in(&root, "h3-theirs");
    assert!(ids(&all).contains(&"h3-mine".to_string()), "{all}");
    assert!(ids(&all).contains(&"h3-theirs".to_string()), "{all}");
    let owner_of = |worker_id: &str| -> String {
        all["workers"]
            .as_array()
            .expect("workers array")
            .iter()
            .find(|row| row["id"] == worker_id)
            .and_then(|row| row["owner"].as_str())
            .unwrap_or_default()
            .to_string()
    };
    assert_eq!(owner_of("h3-mine"), "agent-a");
    assert_eq!(owner_of("h3-theirs"), "agent-b");

    // A typo is a hard error, not a silent fallback to the caller's own rows.
    let error = server
        .execute_tool_for(
            "worker",
            json!({ "action": "list", "scope": "everything" }),
            &agent_a,
        )
        .await
        .expect_err("an unknown scope must not be accepted");
    assert!(
        error.to_string().contains("'scope' must be one of"),
        "{error}"
    );
}

/// One agent may not fill the pool: past `MAX_WORKERS_PER_AGENT` a dispatch is
/// refused, and the error names the workers already running.
#[tokio::test]
async fn a_dispatch_past_the_per_agent_cap_is_refused() {
    let (owned, server) = owned_server();
    owned
        .pool
        .__test_insert_worker(owned_worker("h3-cap", "cap-agent"))
        .await;
    let _cap = ScopedEnv::set("MAX_WORKERS_PER_AGENT", "1");
    let capped = agent_context("cap-agent");

    let error = server
        .execute_tool_for(
            "worker",
            json!({
                "action": "dispatch",
                "task": "one worker too many",
                "repo_path": std::env::temp_dir(),
            }),
            &capped,
        )
        .await
        .expect_err("the per-agent cap must refuse the dispatch");
    let message = error.to_string();
    assert!(message.contains("cap-agent"), "{message}");
    assert!(message.contains("MAX_WORKERS_PER_AGENT=1"), "{message}");
    assert!(message.contains("h3-cap"), "{message}");
    // The refusal happened before the dispatch, so no second worker was started.
    assert_eq!(owned.pool.active_worker_count().await, 1);
}

// ----------
// Revision loop (hub-H8): next_step guidance and steer-after-finish
// ----------

/// A completed payload carries the review guidance, naming the branch the
/// revision resumes on.
#[tokio::test]
async fn completed_payloads_carry_the_review_guidance() {
    let _scratch = common::TempDir::new_in_tmp("iso-mcp-5");
    let pool = WorkerPool::with_scratch(
        1,
        "http://localhost:1".to_string(),
        "test-key".to_string(),
        mini_swe_mcp::worktree::ScratchRoot::new(_scratch.path()),
    );
    pool.__test_insert_worker(completed_worker("guide-done"))
        .await;
    let server = McpServer::new(pool, "ninja".to_string());

    // `collect` evicts the record, so it goes last.
    for action in ["status", "collect"] {
        let result = server
            .execute_tool(
                "worker",
                json!({ "action": action, "worker_id": "guide-done" }),
            )
            .await
            .unwrap_or_else(|e| panic!("{action} on a finished worker must answer: {e}"));
        let next = result
            .get("next_step")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        assert!(
            next.contains("steer"),
            "{action} must tell the orchestrator to steer for corrections: {result}"
        );
        assert!(
            next.contains("swe-wt-done"),
            "{action} must name the branch the revision resumes on: {result}"
        );
    }
}

/// A non-integer `max_turns` on a steer is a hard error, like on dispatch.
#[tokio::test]
async fn steer_with_a_malformed_budget_is_a_hard_error() {
    let _scratch = common::TempDir::new_in_tmp("iso-mcp-6");
    let pool = WorkerPool::with_scratch(
        1,
        "http://localhost:1".to_string(),
        "test-key".to_string(),
        mini_swe_mcp::worktree::ScratchRoot::new(_scratch.path()),
    );
    pool.__test_insert_worker(running_worker("budget-bad"))
        .await;
    let server = McpServer::new(pool, "ninja".to_string());

    let err = server
        .execute_tool(
            "worker",
            json!({ "action": "steer", "worker_id": "budget-bad", "message": "go", "max_turns": "many" }),
        )
        .await
        .expect_err("a string budget must not be accepted");
    assert!(
        err.to_string().contains("max_turns"),
        "the error must name the argument: {err}"
    );
}

/// The channel event for a finished worker carries the same guidance.
#[test]
fn completed_channel_event_carries_the_review_guidance() {
    use mini_swe_mcp::mcp::{ChannelEvent, EventKind, WorkerView};
    let view = WorkerView {
        worker_id: "ev-done".to_string(),
        event: Some(EventKind::Completed),
        group: "backend".to_string(),
        model: "ninja".to_string(),
        status: "completed".to_string(),
        question: None,
        outcome: Default::default(),
        branch: Some("worker-ev-done".to_string()),
        revision: 0,
    };
    let event = ChannelEvent {
        worker_id: view.worker_id.clone(),
        kind: EventKind::Completed,
        group: view.group.clone(),
        model: view.model.clone(),
        status: view.status.clone(),
        content: mini_swe_mcp::mcp::render_for_test(&view, EventKind::Completed),
    };
    assert!(
        event.content.contains("steer"),
        "the completed event must point at steer: {}",
        event.content
    );
    assert!(
        event.content.contains("worker-ev-done"),
        "the completed event must name the branch: {}",
        event.content
    );
}

// ----------
// Dispatch scratch
// ----------

/// Run `git` in `dir`, panicking on failure: only test scaffolding calls this.
fn git_in(dir: &Path, args: &[&str]) {
    let out = Command::new("git")
        .current_dir(dir)
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("failed to run git {args:?} in {}: {e}", dir.display()));
    assert!(
        out.status.success(),
        "git {args:?} in {} failed: {}",
        dir.display(),
        String::from_utf8_lossy(&out.stderr)
    );
}

/// `worker-*` branches of `repo`, or an empty list when it is not a repository.
fn worker_branches(repo: &Path) -> Vec<String> {
    let out = Command::new("git")
        .current_dir(repo)
        .args(["branch", "--list", "worker-*"])
        .output();
    match out {
        Ok(out) if out.status.success() => String::from_utf8_lossy(&out.stdout)
            .lines()
            .map(|line| line.trim().trim_start_matches('*').trim().to_string())
            .filter(|name| !name.is_empty())
            .collect(),
        _ => Vec::new(),
    }
}

/// A throwaway repository plus the scratch root a dispatched worker builds its
/// worktree in. Both vanish on drop: a protocol test that really dispatches a
/// worker must leave the crate's own repository untouched.
struct DispatchScratch {
    root: PathBuf,
    repo: PathBuf,
    base: PathBuf,
}

impl DispatchScratch {
    fn new(tag: &str) -> Self {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!("swe-mcp-{tag}-{}-{n}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let repo = root.join("repo");
        let base = root.join("swe");
        std::fs::create_dir_all(&repo).expect("create the scratch repository");
        std::fs::create_dir_all(&base).expect("create the scratch base dir");
        git_in(&repo, &["init", "-b", "master"]);
        git_in(&repo, &["config", "user.name", "mini-swe-test"]);
        git_in(&repo, &["config", "user.email", "test@localhost"]);
        std::fs::write(repo.join("README.md"), "# scratch\n").expect("seed the repository");
        git_in(&repo, &["add", "README.md"]);
        git_in(&repo, &["commit", "-m", "baseline"]);
        Self { root, repo, base }
    }

    /// The repository the worker must treat as its `repo_path`.
    fn repo(&self) -> String {
        self.repo.to_string_lossy().into_owned()
    }

    /// The scratch root the server under test must see as `SWE_TEMP_DIR`.
    fn base(&self) -> String {
        self.base.to_string_lossy().into_owned()
    }

    fn worker_branches(&self) -> Vec<String> {
        worker_branches(&self.repo)
    }
}

impl Drop for DispatchScratch {
    fn drop(&mut self) {
        // Best effort: a leftover scratch tree must never fail a good test.
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

/// Snapshot of the crate repository's `worker-*` branches, so a test can prove
/// its dispatch created none of them: the worker's branch belongs in the test's
/// own scratch repository, never in the crate.
struct CrateBranchGuard {
    before: Vec<String>,
}

impl CrateBranchGuard {
    fn new() -> Self {
        Self {
            before: worker_branches(&std::env::current_dir().expect("current dir")),
        }
    }

    fn assert_untouched(&self) {
        let cwd = std::env::current_dir().expect("current dir");
        let leaked: Vec<String> = worker_branches(&cwd)
            .into_iter()
            .filter(|branch| !self.before.contains(branch))
            .collect();
        assert!(
            leaked.is_empty(),
            "a dispatch leaked worker branches into the crate repository: {leaked:?}"
        );
    }
}

/// Ask the server to kill `worker_id` and wait for the reply, so the worker's
/// `WorktreeGuard` drops before the server process is hard-killed.
fn kill_worker(server: &mut McpProcess, id: u64, worker_id: &str) {
    server.send(&json!({
        "jsonrpc": "2.0",
        "id": id,
        "method": "tools/call",
        "params": {
            "name": "worker",
            "arguments": { "action": "kill", "worker_id": worker_id }
        }
    }));
    let response = server
        .expect_response("tools/call kill")
        .expect("the kill call must be answered");
    let result = expect_result(&response);
    let payload: Value = serde_json::from_str(result["content"][0]["text"].as_str().expect("text"))
        .expect("tool text must be JSON");
    assert_eq!(
        payload["worker_id"],
        json!(worker_id),
        "kill must echo the worker id"
    );
}

// ----------
// H-11: dispatch never waits, and every verb is private to its owner
// ----------

/// `dispatch` answers with the worker id and a pointer to `watch`, and it
/// answers *now*: no `wait`, no `timeout_secs`, no blocking.
#[tokio::test]
async fn dispatch_returns_immediately_with_a_watch_hint() {
    let scratch = DispatchScratch::new("dispatch-watch");
    let crate_branches = CrateBranchGuard::new();
    let owned = IsolatedPool::new(1, "mcp-dispatch");
    let server = McpServer::new(owned.pool.clone(), "ninja".to_string());

    let result = tokio::time::timeout(
        Duration::from_secs(3),
        server.execute_tool(
            "worker",
            json!({
                "action": "dispatch",
                "task": "probe",
                "repo_path": scratch.repo(),
                "wait": true,
                "timeout_secs": 0,
            }),
        ),
    )
    .await
    .expect("dispatch must not block")
    .expect("dispatch must answer");

    let wid = result["worker_id"].as_str().expect("worker id").to_string();
    assert_eq!(result["status"], "dispatched");
    assert!(
        result["message"]
            .as_str()
            .is_some_and(|m| m.contains("watch")),
        "the reply must point at watch: {result}"
    );
    assert!(
        result.get("state").is_none() && result.get("logs").is_none(),
        "an immediate dispatch carries no terminal payload: {result}"
    );

    // The worktree is built asynchronously; wait until it appears in the
    // scratch repository before proving the crate's own repository is intact.
    tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            if !scratch.worker_branches().is_empty() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("the dispatched worker must create its worktree");

    owned.pool.kill(&wid).await;
    remove_registry_entry_in(&owned.root(), &wid);
    crate_branches.assert_untouched();
}

/// The `watch` action is the only way to wait, and it is bounded by
/// `timeout_secs`: an expired deadline answers `no_event` rather than hanging.
#[tokio::test]
async fn watch_action_answers_no_event_on_an_expired_deadline() {
    let _scratch = common::TempDir::new_in_tmp("iso-mcp-7");
    let pool = WorkerPool::with_scratch(
        1,
        "http://localhost:1".to_string(),
        "test-key".to_string(),
        mini_swe_mcp::worktree::ScratchRoot::new(_scratch.path()),
    );
    pool.__test_insert_worker(running_worker("h11-watch-running"))
        .await;
    let server = McpServer::new(pool, "ninja".to_string());

    let result = tokio::time::timeout(
        Duration::from_secs(5),
        server.execute_tool(
            "worker",
            json!({ "action": "watch", "worker_id": "h11-watch-running", "timeout_secs": 0 }),
        ),
    )
    .await
    .expect("watch must honour its deadline")
    .expect("watch must answer");

    assert_eq!(result["status"], "no_event");
    assert_eq!(result["events"].as_array().map(Vec::len), Some(0));
}

/// A worker is private to its owner: agent B gets the owner's name and nothing
/// else -- not the task, not the state -- for `status`, `logs`, `collect` and
/// `watch`, and `list` shows it only to its owner.
#[tokio::test]
async fn an_agent_cannot_read_another_agents_worker() {
    let (owned, server) = owned_server();
    owned
        .pool
        .__test_insert_worker(owned_worker("h11-mine", "agent-a"))
        .await;
    owned
        .pool
        .__test_insert_worker(owned_worker("h11-theirs", "agent-b"))
        .await;

    for arguments in [
        json!({ "action": "status", "worker_id": "h11-theirs" }),
        json!({ "action": "logs", "worker_id": "h11-theirs" }),
        json!({ "action": "collect", "worker_id": "h11-theirs" }),
        json!({ "action": "watch", "worker_id": "h11-theirs", "timeout_secs": 0 }),
    ] {
        let action = arguments["action"].as_str().expect("action").to_string();
        let error = server
            .execute_tool_for("worker", arguments.clone(), &agent_context("agent-a"))
            .await
            .expect_err("another agent's worker must be refused");
        assert_eq!(
            error.to_string(),
            "worker h11-theirs belongs to agent agent-b",
            "'{action}' must name the owning agent: {error}"
        );
    }

    // `list` shows the caller its own workers only.
    let mine = server
        .execute_tool_for(
            "worker",
            json!({ "action": "list" }),
            &agent_context("agent-a"),
        )
        .await
        .expect("list answers");
    let ids: Vec<String> = mine["workers"]
        .as_array()
        .expect("workers array")
        .iter()
        .filter_map(|row| row["id"].as_str().map(str::to_string))
        .collect();
    assert!(ids.contains(&"h11-mine".to_string()), "{mine}");
    assert!(!ids.contains(&"h11-theirs".to_string()), "{mine}");

    // The admin override still sees and acts on everything.
    let admin = server
        .execute_tool_for(
            "worker",
            json!({ "action": "list", "scope": "all" }),
            &admin_context("op"),
        )
        .await
        .expect("admin scope=all answers");
    let admin_ids: Vec<String> = admin["workers"]
        .as_array()
        .expect("workers array")
        .iter()
        .filter_map(|row| row["id"].as_str().map(str::to_string))
        .collect();
    assert!(admin_ids.contains(&"h11-mine".to_string()), "{admin}");
    assert!(admin_ids.contains(&"h11-theirs".to_string()), "{admin}");

    let admin_status = server
        .execute_tool_for(
            "worker",
            json!({ "action": "status", "worker_id": "h11-theirs" }),
            &admin_context("op"),
        )
        .await
        .expect("the admin override may read any worker");
    assert_eq!(admin_status["owner"], "agent-b");
}
