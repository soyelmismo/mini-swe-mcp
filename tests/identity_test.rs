//! One identity per agent session: the host process, shared by the agent's MCP
//! connection and its shell commands.
//!
//! The daemon runs in-process on a scratch hub directory, so no test needs an
//! LLM or the developer's real hub: the workers are inserted straight into the
//! pool and every assertion goes through the same Unix socket a thin client
//! would dial.

mod common;

use mini_swe_mcp::hub::{HubConfig, HubPaths, HubServer};
use mini_swe_mcp::manifest::ModelManifest;
use mini_swe_mcp::mcp::{CLI_CLIENT_NAME, McpServer};
use mini_swe_mcp::pool::{LogBuffer, WorkerMetrics, WorkerPool, WorkerRecord, WorkerState};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;

static TAG: AtomicU64 = AtomicU64::new(0);

/// A scratch hub directory, removed when the test ends.
fn scratch_dir() -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "swe-identity-test-{}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock before epoch")
            .as_nanos(),
        TAG.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create scratch hub dir");
    dir
}

/// A server backed by a pool that answers handshake verbs without an LLM.
fn server() -> Arc<McpServer> {
    let pool = WorkerPool::new(4, "http://localhost:1".to_string(), "test-key".to_string())
        .with_manifest(Arc::new(ModelManifest::default()));
    Arc::new(McpServer::new(pool, "test-model".to_string()))
}

/// Wait until `path` accepts a connection, or panic.
async fn wait_for_socket(path: &Path) {
    for _ in 0..100 {
        if UnixStream::connect(path).await.is_ok() {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    panic!("hub socket {} never came up", path.display());
}

/// A synthetic running worker owned by `owner`, as if that agent had
/// dispatched it.
fn owned_worker(id: &str, owner: &str) -> WorkerRecord {
    WorkerRecord {
        id: id.to_string(),
        task: "t".to_string(),
        model: "m".to_string(),
        owner: owner.to_string(),
        state: WorkerState::Running {
            step: 2,
            last_command: "cargo test".to_string(),
            started_at: 0,
        },
        metrics: WorkerMetrics::default(),
        logs: LogBuffer::new(),
        pending_steer: Vec::new(),
        resume_tx: None,
        handle: None,
        revision: 0,
    }
}

/// A hub client: enough of the wire to handshake and call the `worker` tool.
struct Client {
    reader: BufReader<tokio::net::unix::OwnedReadHalf>,
    writer: tokio::net::unix::OwnedWriteHalf,
    next_id: u64,
}

impl Client {
    async fn connect(socket: &Path) -> Self {
        let stream = UnixStream::connect(socket)
            .await
            .expect("connect to hub socket");
        let (reader, writer) = stream.into_split();
        Self {
            reader: BufReader::new(reader),
            writer,
            next_id: 1,
        }
    }

    /// Send one request and return the whole reply envelope, error included.
    async fn request(&mut self, method: &str, params: serde_json::Value) -> serde_json::Value {
        let id = self.next_id;
        self.next_id += 1;
        let frame =
            serde_json::json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params});
        self.writer
            .write_all(format!("{frame}\n").as_bytes())
            .await
            .expect("write frame");
        self.writer.flush().await.expect("flush frame");
        loop {
            let mut line = String::new();
            self.reader
                .read_line(&mut line)
                .await
                .expect("read response");
            assert!(!line.is_empty(), "hub closed the connection");
            let reply: serde_json::Value =
                serde_json::from_str(line.trim()).expect("response is JSON");
            if reply.get("id") == Some(&serde_json::json!(id)) {
                return reply;
            }
        }
    }

    /// Send one notification, e.g. the `hub/hello` handshake.
    async fn notify(&mut self, method: &str, params: serde_json::Value) {
        let frame = serde_json::json!({"jsonrpc": "2.0", "method": method, "params": params});
        self.writer
            .write_all(format!("{frame}\n").as_bytes())
            .await
            .expect("write notification");
        self.writer.flush().await.expect("flush notification");
    }

    /// The handshake a client makes: `initialize` names it, `hub/hello` names
    /// the agent it speaks for.
    async fn handshake(
        &mut self,
        client_name: &str,
        agent_id: Option<&str>,
        host_id: Option<&str>,
    ) {
        let reply = self
            .request(
                "initialize",
                serde_json::json!({
                    "protocolVersion": "2024-11-05", "capabilities": {},
                    "clientInfo": {"name": client_name, "version": "test"}
                }),
            )
            .await;
        assert_eq!(reply["result"]["protocolVersion"], "2024-11-05");
        self.notify(
            "hub/hello",
            serde_json::json!({"agent_id": agent_id, "host_id": host_id, "pid": 1}),
        )
        .await;
    }

    /// One `worker` tool call: the decoded payload, or the error message.
    async fn worker(&mut self, arguments: serde_json::Value) -> Result<serde_json::Value, String> {
        let reply = self
            .request(
                "tools/call",
                serde_json::json!({"name": "worker", "arguments": arguments}),
            )
            .await;
        if let Some(message) = reply.get("error").and_then(|e| e["message"].as_str()) {
            return Err(message.to_string());
        }
        let result = &reply["result"];
        let text = result["content"][0]["text"]
            .as_str()
            .unwrap_or_else(|| panic!("worker tool result text in {result}"))
            .to_string();
        serde_json::from_str(&text).map_err(|error| error.to_string())
    }
}

/// A daemon on a scratch directory, stopped and removed when it drops.
struct Daemon {
    socket: PathBuf,
    server: Arc<McpServer>,
    task: tokio::task::JoinHandle<()>,
    dir: PathBuf,
}

impl Daemon {
    async fn start() -> Self {
        let dir = scratch_dir();
        let socket = dir.join("hub.sock");
        let server = server();
        let daemon = HubServer::new(
            server.clone(),
            HubConfig::new(HubPaths::new(dir.clone()), 60),
        );
        let task = tokio::spawn(async move {
            let _ = daemon.run().await;
        });
        wait_for_socket(&socket).await;
        Self {
            socket,
            server,
            task,
            dir,
        }
    }
}

impl Drop for Daemon {
    fn drop(&mut self) {
        self.task.abort();
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// A shell command of this session names the test process as its host, not the
/// shell it ran under: the walk steps over `sh` and stops at the first process
/// that is left.
#[test]
fn a_shell_command_names_the_test_process_as_its_host() {
    let exe = common::binary_path();
    let output = Command::new("sh")
        .arg("-c")
        // A trailing command keeps the shell alive as the parent: `sh -c 'cmd'`
        // execs a lone command, and then there would be no shell to skip.
        .arg(format!("{} whoami; true", exe.display()))
        .env("MINI_SWE_NO_DAEMON", "1")
        .output()
        .unwrap_or_else(|e| panic!("failed to run {} whoami: {e}", exe.display()));
    assert!(
        output.status.success(),
        "whoami must exit 0: {}",
        common::stderr_of(&output)
    );
    let stdout = common::stdout_of(&output);
    let expected = common::host_of_this_process();
    assert!(
        stdout.contains(&format!("agent {expected}")),
        "the host is this test process, not the shell: {stdout}"
    );
    assert!(
        !stdout.contains("host:sh:"),
        "a shell is never the host: {stdout}"
    );
    assert!(
        stdout.contains("skipping sh"),
        "whoami must say how the identity was derived: {stdout}"
    );
}

/// `MINI_SWE_AGENT_ID` still outranks the host process.
#[test]
fn an_explicit_agent_id_outranks_the_host_of_a_shell_command() {
    let exe = common::binary_path();
    let output = Command::new("sh")
        .arg("-c")
        .arg(format!("{} whoami", exe.display()))
        .env("MINI_SWE_NO_DAEMON", "1")
        .env("MINI_SWE_AGENT_ID", "orchestrator-7")
        .output()
        .unwrap_or_else(|e| panic!("failed to run {} whoami: {e}", exe.display()));
    let stdout = common::stdout_of(&output);
    assert_eq!(
        stdout.lines().next(),
        Some("agent orchestrator-7"),
        "the override wins: {stdout}"
    );
    assert!(
        stdout.contains("MINI_SWE_AGENT_ID"),
        "whoami must say the override answered: {stdout}"
    );
}

/// The MCP connection an agent dispatches over and the `mini-swe-mcp` calls its
/// shell makes are one agent: both hello the same host process, so the second
/// sees and steers the worker the first dispatched.
#[tokio::test]
async fn an_mcp_connection_and_a_cli_call_of_one_host_share_workers() {
    let daemon = Daemon::start().await;
    let host = "host:claude:4242:9182734";
    let pool = daemon.server.pool();
    pool.__test_insert_worker(owned_worker("shared-1", host))
        .await;

    // The agent's MCP connection: it dispatched the worker.
    let mut connection = Client::connect(&daemon.socket).await;
    connection.handshake("claude-code", None, Some(host)).await;
    // The agent's shell: the same host, announced by the CLI's own hello.
    let mut shell = Client::connect(&daemon.socket).await;
    shell.handshake(CLI_CLIENT_NAME, None, Some(host)).await;

    for caller in [&mut connection, &mut shell] {
        let listed = caller
            .worker(serde_json::json!({"action": "list"}))
            .await
            .expect("the caller lists its own workers");
        let ids: Vec<&str> = listed["workers"]
            .as_array()
            .expect("workers array")
            .iter()
            .filter_map(|row| row["id"].as_str())
            .collect();
        assert!(
            ids.contains(&"shared-1"),
            "the caller must see the worker its host dispatched: {ids:?}"
        );
        // `watch` is the verb the agent's shell blocks on; the owner may wait
        // on the worker, and `steer` proves it controls it.
        caller
            .worker(
                serde_json::json!({"action": "watch", "worker_id": "shared-1", "timeout_secs": 0}),
            )
            .await
            .expect("the owner may watch its own worker");
        let steered = caller
            .worker(serde_json::json!({"action": "steer", "worker_id": "shared-1", "message": "carry on"}))
            .await
            .expect("the owner may steer its own worker");
        assert_eq!(steered["status"], "steered");
    }
}

/// Two host processes are two agents: neither sees nor steers the other's
/// workers, whatever client name they connect under.
#[tokio::test]
async fn two_host_identities_cannot_see_each_others_workers() {
    let daemon = Daemon::start().await;
    let pool = daemon.server.pool();
    pool.__test_insert_worker(owned_worker("mine-1", "host:claude:4242:9182734"))
        .await;
    pool.__test_insert_worker(owned_worker("theirs-1", "host:opencode:5253:9182735"))
        .await;

    let mut mine = Client::connect(&daemon.socket).await;
    mine.handshake("claude-code", None, Some("host:claude:4242:9182734"))
        .await;
    let mut theirs = Client::connect(&daemon.socket).await;
    theirs
        .handshake("claude-code", None, Some("host:opencode:5253:9182735"))
        .await;

    let listed = mine
        .worker(serde_json::json!({"action": "list"}))
        .await
        .expect("the caller lists its own workers");
    let ids: Vec<&str> = listed["workers"]
        .as_array()
        .expect("workers array")
        .iter()
        .filter_map(|row| row["id"].as_str())
        .collect();
    assert_eq!(ids, ["mine-1"], "only the caller's own worker: {ids:?}");

    for action in ["steer", "kill", "collect"] {
        let error = mine
            .worker(
                serde_json::json!({"action": action, "worker_id": "theirs-1", "message": "stop"}),
            )
            .await
            .expect_err("another host's worker must be refused");
        assert!(
            error.contains("belongs to agent host:opencode:5253:9182735"),
            "'{action}' must name the owning agent: {error}"
        );
    }
}

/// An explicit `MINI_SWE_AGENT_ID` still wins over the host process, so an
/// operator can pin one session to a name of their choosing.
#[tokio::test]
async fn an_explicit_agent_id_outranks_the_host_identity() {
    let daemon = Daemon::start().await;
    let host = "host:claude:4242:9182734";
    let pool = daemon.server.pool();
    pool.__test_insert_worker(owned_worker("pinned-1", "orchestrator-7"))
        .await;

    let mut pinned = Client::connect(&daemon.socket).await;
    pinned
        .handshake("claude-code", Some("orchestrator-7"), Some(host))
        .await;
    let mut host_named = Client::connect(&daemon.socket).await;
    host_named.handshake("claude-code", None, Some(host)).await;

    pinned
        .worker(serde_json::json!({"action": "steer", "worker_id": "pinned-1", "message": "go on"}))
        .await
        .expect("the explicit identity owns the worker");
    let error = host_named
        .worker(serde_json::json!({"action": "steer", "worker_id": "pinned-1", "message": "go on"}))
        .await
        .expect_err("the host identity does not own it");
    assert!(error.contains("belongs to agent orchestrator-7"), "{error}");
}
