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

use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{Receiver, RecvTimeoutError, channel};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

const PROTOCOL_VERSION: &str = "2024-11-05";
const SERVER_NAME: &str = "mini-swe-mcp";
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
        let (exe, args) = binary_command();
        let mut child = Command::new(&exe)
            .args(&args)
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

    /// Assert the server writes nothing on stdout for a short grace period.
    fn expect_silence(&mut self, context: &str) {
        match self.stdout_rx.recv_timeout(SILENCE_GRACE) {
            Err(RecvTimeoutError::Timeout) => {}
            Ok(out) => panic!("expected no response for {context}, but got {}", out.describe()),
            Err(RecvTimeoutError::Disconnected) => {}
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
    (
        path.join("mini-swe-mcp"),
        vec!["--stdio".to_string()],
    )
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
    assert!(result.is_object(), "result must be an object, got: {result}");
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

    assert_eq!(response["id"], json!("ping-1"), "id must be echoed verbatim");
    assert_eq!(expect_result(&response), json!({}), "ping result must be {{}}");
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
    for expected in [
        "prune",
        "dispatch",
        "status",
        "steer",
        "collect",
        "list",
        "kill",
        "manifest",
    ] {
        assert!(
            actions.contains(&expected),
            "worker action enum must contain '{expected}', got: {actions:?}"
        );
    }

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
    let required = schema["required"]
        .as_array()
        .unwrap_or_else(|| panic!("worker inputSchema must declare required fields, got: {schema}"));
    assert_eq!(
        required.iter().filter(|v| v.as_str() == Some("action")).count(),
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
    assert_eq!(
        notif1["params"]["progressToken"],
        json!("token-prune-xyz")
    );
    assert_eq!(notif1["params"]["progress"], json!(0));

    // Expect progress notification 1/1
    let notif2 = server
        .expect_response("second progress notification")
        .expect("must receive notification");
    assert_eq!(notif2["method"], json!("notifications/progress"));
    assert_eq!(
        notif2["params"]["progressToken"],
        json!("token-prune-xyz")
    );
    assert_eq!(notif2["params"]["progress"], json!(1));

    // Expect final response
    let final_resp = server
        .expect_response("final tools/call response")
        .expect("must receive tool response");
    assert_eq!(final_resp["id"], json!(200));
    assert!(final_resp.get("result").is_some());
}
