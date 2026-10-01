//! Agent identity separates the sessions that share one host process.
//!
//! A host like opencode v2 runs a tab per session inside ONE process, over ONE
//! shared MCP connection, so the host alone would merge them: these tests pin
//! the session half of the identity — `_meta.sessionID` per call, the session
//! variables in the environment, and the watch token a shell that knows neither
//! presents instead.
//!
//! The daemon runs in-process on a scratch hub directory (mode 0700, so the
//! client side can resolve a token through `hub_dir()`), so no test needs an
//! LLM or the developer's real hub: the workers are inserted straight into the
//! pool and every assertion goes through the same Unix socket a thin client
//! would dial.

mod common;

use mini_swe_mcp::hub::{HubConfig, HubEndpoint, HubPaths, HubServer, WatchTokens};
use mini_swe_mcp::manifest::ModelManifest;
use mini_swe_mcp::mcp::{CLI_CLIENT_NAME, McpServer};
use mini_swe_mcp::pool::{LogBuffer, WorkerMetrics, WorkerPool, WorkerRecord, WorkerState};
use serde_json::{Value, json};
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

static TAG: AtomicU64 = AtomicU64::new(0);

/// One host process, as the identity names it.
const HOST: &str = "host:opencode:730:12";

/// The two sessions of that one host process.
const TAB_A: &str = "host:opencode:730:12/session:tab-a";
const TAB_B: &str = "host:opencode:730:12/session:tab-b";

/// A scratch hub directory, removed when the test ends.
///
/// Mode 0700: the client side resolves a watch token through `hub_dir()`, which
/// refuses a directory that is group or world accessible.
fn scratch_dir() -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "swe-session-test-{}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock before epoch")
            .as_nanos(),
        TAG.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create scratch hub dir");
    std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700))
        .expect("restrict scratch hub dir to 0700");
    dir
}

/// A server backed by a pool that answers handshake verbs without an LLM.
fn server() -> Arc<McpServer> {
    let _scratch = common::TempDir::new_in_tmp("iso-session-1");
    let pool = WorkerPool::with_scratch(
        4,
        "http://localhost:1".to_string(),
        "test-key".to_string(),
        mini_swe_mcp::worktree::ScratchRoot::new(_scratch.path()),
    )
    .with_manifest(Arc::new(ModelManifest::default()));
    Arc::new(McpServer::new(pool, "test-model".to_string()))
}

/// Wait until `path` accepts a connection, or panic.
/// Wait until `endpoint` accepts a connection, or panic.
///
/// The endpoint is derived exactly as the product's client derives it: a deep
/// scratch directory moves the hub onto a short fallback directory or onto a
/// Linux abstract socket, and polling the filesystem path alone would wait for
/// a file that never appears.
async fn wait_for_endpoint(endpoint: &HubEndpoint) {
    for _ in 0..100 {
        if mini_swe_mcp::hub::connect_endpoint(endpoint).await.is_ok() {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    panic!("hub endpoint {endpoint:?} never came up");
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
    async fn connect(endpoint: &HubEndpoint) -> Self {
        let stream = mini_swe_mcp::hub::connect_endpoint(endpoint)
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
    async fn request(&mut self, method: &str, params: Value) -> Value {
        let id = self.next_id;
        self.next_id += 1;
        let frame = json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params});
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
            let reply: Value = serde_json::from_str(line.trim()).expect("response is JSON");
            if reply.get("id") == Some(&json!(id)) {
                return reply;
            }
        }
    }

    /// Send one notification, e.g. the `hub/hello` handshake.
    async fn notify(&mut self, method: &str, params: Value) {
        let frame = json!({"jsonrpc": "2.0", "method": method, "params": params});
        self.writer
            .write_all(format!("{frame}\n").as_bytes())
            .await
            .expect("write notification");
        self.writer.flush().await.expect("flush notification");
    }

    /// The handshake a client makes: `initialize` names it, `hub/hello` names
    /// the agent it speaks for.
    ///
    /// `session` is the session the *connection* speaks for; `meta_session` is
    /// the `_meta.sessionID` a single call carries. `watch_token` is the token
    /// a shell that cannot know its session presents instead.
    async fn handshake(
        &mut self,
        client_name: &str,
        agent_id: Option<&str>,
        host_id: Option<&str>,
        session: Option<&str>,
        watch_token: Option<&str>,
    ) {
        let reply = self
            .request(
                "initialize",
                json!({
                    "protocolVersion": "2024-11-05", "capabilities": {},
                    "clientInfo": {"name": client_name, "version": "test"}
                }),
            )
            .await;
        assert_eq!(reply["result"]["protocolVersion"], "2024-11-05");
        self.notify(
            "hub/hello",
            json!({
                "agent_id": agent_id,
                "host_id": host_id,
                "session_id": session,
                "watch_token": watch_token,
                "pid": 1,
            }),
        )
        .await;
    }

    /// One `worker` tool call, optionally carrying the session of this one call
    /// in `_meta.sessionID` the way opencode v2 does.
    async fn worker(
        &mut self,
        arguments: Value,
        meta_session: Option<&str>,
    ) -> Result<Value, String> {
        let mut params = json!({"name": "worker", "arguments": arguments});
        if let Some(session) = meta_session {
            params["_meta"] = json!({"sessionID": session});
        }
        let reply = self.request("tools/call", params).await;
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

    /// The worker ids a `list` answers with.
    async fn listed_ids(&mut self, meta_session: Option<&str>) -> Vec<String> {
        let listed = self
            .worker(json!({"action": "list"}), meta_session)
            .await
            .expect("the caller lists its own workers");
        listed["workers"]
            .as_array()
            .expect("workers array")
            .iter()
            .filter_map(|row| row["id"].as_str().map(str::to_string))
            .collect()
    }
}

/// A daemon on a scratch directory, stopped and removed when it drops.
struct Daemon {
    endpoint: HubEndpoint,
    server: Arc<McpServer>,
    task: Option<tokio::task::JoinHandle<()>>,
    dir: PathBuf,
    /// Set by [`Daemon::stop`]: the directory outlives this daemon, so the
    /// caller (a restart) owns it and its cleanup.
    detached: bool,
}

impl Daemon {
    /// Start a daemon on a fresh scratch directory.
    async fn start() -> Self {
        Self::start_in(scratch_dir()).await
    }

    /// Start a daemon on `dir`: the seam a restart is tested through, since the
    /// tokens must survive it.
    async fn start_in(dir: PathBuf) -> Self {
        let paths = HubPaths::new(dir.to_path_buf());
        let endpoint = paths.endpoint();
        let server = server();
        let daemon = HubServer::new(server.clone(), HubConfig::new(paths, 60));
        let task = tokio::spawn(async move {
            let _ = daemon.run().await;
        });
        wait_for_endpoint(&endpoint).await;
        Self {
            endpoint,
            server,
            task: Some(task),
            dir,
            detached: false,
        }
    }

    /// Stop the daemon and wait for it to let go of the hub lock, leaving the
    /// directory in place: the seam a restart is tested through, since the
    /// tokens have to survive it.
    async fn stop(&mut self) {
        self.detached = true;
        if let Some(task) = self.task.take() {
            task.abort();
            let _ = task.await;
        }
    }
}

impl Drop for Daemon {
    fn drop(&mut self) {
        if let Some(task) = self.task.take() {
            task.abort();
        }
        if !self.detached {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }
}

/// The `watch_command` a steer answer carries, and the token inside it.
async fn watch_token_of(reply: &Value) -> String {
    let command = reply["watch_command"]
        .as_str()
        .unwrap_or_else(|| panic!("a steer answer carries watch_command: {reply}"))
        .to_string();
    assert!(
        command.starts_with("MINI_SWE_WATCH_TOKEN=") && command.ends_with(" mini-swe-mcp watch"),
        "the command waits on this caller's workers: {command}"
    );
    command
        .strip_prefix("MINI_SWE_WATCH_TOKEN=")
        .and_then(|rest| rest.strip_suffix(" mini-swe-mcp watch"))
        .expect("token between the assignment and the command")
        .to_string()
}

/// Two calls on ONE connection with different `_meta.sessionID` are two
/// sessions of one host: each owns its own workers and cannot see the other's.
#[tokio::test]
async fn two_sessions_of_one_connection_own_different_workers() {
    let daemon = Daemon::start().await;
    let pool = daemon.server.pool();
    pool.__test_insert_worker(owned_worker("tab-a-1", TAB_A))
        .await;
    pool.__test_insert_worker(owned_worker("tab-b-1", TAB_B))
        .await;

    // One connection, one host, two sessions: opencode v2's tabs.
    let mut connection = Client::connect(&daemon.endpoint).await;
    connection
        .handshake("opencode", None, Some(HOST), None, None)
        .await;

    assert_eq!(
        connection.listed_ids(Some("tab-a")).await,
        ["tab-a-1"],
        "the call that names tab-a sees only tab-a's worker"
    );
    assert_eq!(
        connection.listed_ids(Some("tab-b")).await,
        ["tab-b-1"],
        "the next call on the same connection names tab-b and sees only its own"
    );
    assert_eq!(
        connection.listed_ids(None).await,
        Vec::<String>::new(),
        "a call that names no session is the bare host, which owns neither"
    );

    for (session, mine, theirs, theirs_owner) in [
        ("tab-a", "tab-a-1", "tab-b-1", TAB_B),
        ("tab-b", "tab-b-1", "tab-a-1", TAB_A),
    ] {
        connection
            .worker(
                json!({"action": "steer", "worker_id": mine, "message": "carry on"}),
                Some(session),
            )
            .await
            .unwrap_or_else(|e| panic!("{session} must steer its own worker: {e}"));
        let error = connection
            .worker(
                json!({"action": "steer", "worker_id": theirs, "message": "stop"}),
                Some(session),
            )
            .await
            .expect_err("the other session's worker must be refused");
        assert!(
            error.contains(&format!("belongs to agent {theirs_owner}")),
            "'{session}' must be told who owns the worker: {error}"
        );
    }
}

/// The session is never cached on the connection: the same connection keeps
/// answering for whichever session the call names.
#[tokio::test]
async fn a_session_is_read_per_call_and_never_cached() {
    let daemon = Daemon::start().await;
    let pool = daemon.server.pool();
    pool.__test_insert_worker(owned_worker("tab-a-1", TAB_A))
        .await;

    let mut connection = Client::connect(&daemon.endpoint).await;
    connection
        .handshake("opencode", None, Some(HOST), Some("tab-a"), None)
        .await;

    // The connection announced tab-a, so a call with no `_meta` is tab-a ...
    assert_eq!(connection.listed_ids(None).await, ["tab-a-1"]);
    // ... and a call that names tab-b is tab-b, without relearning anything.
    assert_eq!(
        connection.listed_ids(Some("tab-b")).await,
        Vec::<String>::new()
    );
    // ... and the connection is still tab-a afterwards.
    assert_eq!(connection.listed_ids(None).await, ["tab-a-1"]);
}

/// A watch token lets a shell that cannot know its session act as the session
/// that dispatched the worker; a wrong or absent token does not.
#[tokio::test]
async fn a_watch_token_acts_as_the_dispatching_session() {
    let daemon = Daemon::start().await;
    let pool = daemon.server.pool();
    pool.__test_insert_worker(owned_worker("tab-a-1", TAB_A))
        .await;

    // The session that dispatched: it steers its worker and is handed the
    // command that waits on it.
    let mut session = Client::connect(&daemon.endpoint).await;
    session
        .handshake("opencode", None, Some(HOST), None, None)
        .await;
    let steered = session
        .worker(
            json!({"action": "steer", "worker_id": "tab-a-1", "message": "carry on"}),
            Some("tab-a"),
        )
        .await
        .expect("the session may steer its own worker");
    assert_eq!(steered["worker_id"], "tab-a-1");
    let token = watch_token_of(&steered).await;

    // The shell: no session variable reaches it, only the token.
    let mut shell = Client::connect(&daemon.endpoint).await;
    shell
        .handshake(CLI_CLIENT_NAME, None, None, None, Some(&token))
        .await;
    assert_eq!(
        shell.listed_ids(None).await,
        ["tab-a-1"],
        "the token makes the shell exactly the dispatching session"
    );
    shell
        .worker(
            json!({"action": "steer", "worker_id": "tab-a-1", "message": "again"}),
            None,
        )
        .await
        .expect("the shell may steer the worker its token names");
    let error = shell
        .worker(json!({"action": "list", "scope": "all"}), None)
        .await
        .expect_err("a token is never the operator's admin override");
    assert!(
        error.contains("admin"),
        "the token must not lift the ownership check: {error}"
    );

    // A wrong token is not an identity: the caller falls back to its own host.
    let mut wrong = Client::connect(&daemon.endpoint).await;
    wrong
        .handshake(
            CLI_CLIENT_NAME,
            None,
            Some(HOST),
            None,
            Some(&"0".repeat(32)),
        )
        .await;
    assert_eq!(
        wrong.listed_ids(None).await,
        Vec::<String>::new(),
        "an unknown token leaves the caller its own host identity"
    );

    // An absent token leaves the caller its own host identity too.
    let mut bare = Client::connect(&daemon.endpoint).await;
    bare.handshake(CLI_CLIENT_NAME, None, Some(HOST), None, None)
        .await;
    assert_eq!(bare.listed_ids(None).await, Vec::<String>::new());
}

/// The tokens live in the hub directory, mode 0600, so they survive a daemon
/// restart — and are never listed to anyone.
#[tokio::test]
async fn watch_tokens_survive_a_daemon_restart() {
    let dir = scratch_dir();
    let token = {
        let mut daemon = Daemon::start_in(dir.clone()).await;
        let pool = daemon.server.pool();
        pool.__test_insert_worker(owned_worker("tab-a-1", TAB_A))
            .await;
        let mut session = Client::connect(&daemon.endpoint).await;
        session
            .handshake("opencode", None, Some(HOST), None, None)
            .await;
        let steered = session
            .worker(
                json!({"action": "steer", "worker_id": "tab-a-1", "message": "carry on"}),
                Some("tab-a"),
            )
            .await
            .expect("the session may steer its own worker");
        let token = watch_token_of(&steered).await;

        let path = HubPaths::new(dir.clone()).watch_tokens();
        let mode = std::fs::metadata(&path)
            .expect("the token store is a file in the hub directory")
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(mode, 0o600, "the token store is private to its user");
        let stored = std::fs::read_to_string(&path).expect("read the token store");
        assert!(stored.contains(&token), "the token is stored: {stored}");
        assert!(
            !stored.contains(TAB_B),
            "only the identities that asked are stored: {stored}"
        );
        daemon.stop().await;
        token
    };

    // A fresh daemon on the same directory: the token still names the session.
    let restarted = Daemon::start_in(dir.clone()).await;
    let pool = restarted.server.pool();
    pool.__test_insert_worker(owned_worker("tab-a-1", TAB_A))
        .await;
    let mut shell = Client::connect(&restarted.endpoint).await;
    shell
        .handshake(CLI_CLIENT_NAME, None, None, None, Some(&token))
        .await;
    assert_eq!(
        shell.listed_ids(None).await,
        ["tab-a-1"],
        "the token survived the restart"
    );
    assert_eq!(
        WatchTokens::new(dir.clone()).identity_of(&token).as_deref(),
        Some(TAB_A),
        "the store on disk still maps the token to the session"
    );
    assert_eq!(
        WatchTokens::new(dir).identity_of(&"f".repeat(32)),
        None,
        "an unknown token maps to nothing"
    );
}

/// `MINI_SWE_AGENT_ID` still outranks everything, the watch token included.
#[tokio::test]
async fn an_explicit_agent_id_outranks_the_session_and_the_token() {
    let daemon = Daemon::start().await;
    let pool = daemon.server.pool();
    pool.__test_insert_worker(owned_worker("pinned-1", "orchestrator-7"))
        .await;

    let mut session = Client::connect(&daemon.endpoint).await;
    session
        .handshake("opencode", None, Some(HOST), None, None)
        .await;
    let steered = session
        .worker(
            json!({"action": "steer", "worker_id": "pinned-1", "message": "go on"}),
            Some("tab-a"),
        )
        .await
        .expect_err("the session does not own the pinned worker");
    assert!(
        steered.contains("belongs to agent orchestrator-7"),
        "{steered}"
    );
    let token = {
        // The pinned agent dispatches, so the token it is handed is its own.
        let mut pinned = Client::connect(&daemon.endpoint).await;
        pinned
            .handshake("opencode", Some("orchestrator-7"), Some(HOST), None, None)
            .await;
        pinned
            .worker(
                json!({"action": "steer", "worker_id": "pinned-1", "message": "go on"}),
                Some("tab-a"),
            )
            .await
            .expect("the pinned agent owns the worker");
        let steered = pinned
            .worker(
                json!({"action": "steer", "worker_id": "pinned-1", "message": "again"}),
                Some("tab-a"),
            )
            .await
            .expect("the pinned agent may steer again");
        watch_token_of(&steered).await
    };

    // The token names the pinned agent, and the override outranks the token.
    let mut shell = Client::connect(&daemon.endpoint).await;
    shell
        .handshake(
            CLI_CLIENT_NAME,
            Some("orchestrator-7"),
            None,
            None,
            Some(&token),
        )
        .await;
    shell
        .worker(
            json!({"action": "steer", "worker_id": "pinned-1", "message": "go on"}),
            None,
        )
        .await
        .expect("the override wins over the token");
    let mut token_only = Client::connect(&daemon.endpoint).await;
    token_only
        .handshake(CLI_CLIENT_NAME, None, None, None, Some(&token))
        .await;
    assert_eq!(
        token_only.listed_ids(None).await,
        ["pinned-1"],
        "the token alone is the pinned agent"
    );
}

/// A host with no session information keeps today's host identity: the shell
/// commands and the MCP connection of one host still share their workers.
#[tokio::test]
async fn a_host_with_no_session_keeps_its_host_identity() {
    let daemon = Daemon::start().await;
    let pool = daemon.server.pool();
    pool.__test_insert_worker(owned_worker("host-1", HOST))
        .await;
    pool.__test_insert_worker(owned_worker("tab-a-1", TAB_A))
        .await;

    let mut connection = Client::connect(&daemon.endpoint).await;
    connection
        .handshake("opencode", None, Some(HOST), None, None)
        .await;
    let mut shell = Client::connect(&daemon.endpoint).await;
    shell
        .handshake(CLI_CLIENT_NAME, None, Some(HOST), None, None)
        .await;

    for caller in [&mut connection, &mut shell] {
        assert_eq!(
            caller.listed_ids(None).await,
            ["host-1"],
            "the bare host owns only what the bare host dispatched"
        );
        caller
            .worker(
                json!({"action": "steer", "worker_id": "host-1", "message": "carry on"}),
                None,
            )
            .await
            .expect("the host may steer its own worker");
        let error = caller
            .worker(
                json!({"action": "steer", "worker_id": "tab-a-1", "message": "stop"}),
                None,
            )
            .await
            .expect_err("a session of that host is a different agent");
        assert!(
            error.contains(&format!("belongs to agent {TAB_A}")),
            "{error}"
        );
    }
}

/// Run `mini-swe-mcp whoami` with `vars` set and every other identity variable
/// cleared, so the environment a test means is the environment the child sees.
fn whoami_with(vars: &[(&str, &str)], hub_dir: Option<&Path>) -> String {
    let exe = common::binary_path();
    let mut command = Command::new(&exe);
    command.arg("whoami").env("MINI_SWE_NO_DAEMON", "1");
    common::scrub_identity_env(&mut command);
    for (var, value) in vars {
        command.env(var, value);
    }
    if let Some(dir) = hub_dir {
        command.env("SWE_HUB_DIR", dir);
    }
    let output = command
        .output()
        .unwrap_or_else(|e| panic!("failed to run {} whoami: {e}", exe.display()));
    assert!(
        output.status.success(),
        "whoami must exit 0: {}",
        common::stderr_of(&output)
    );
    common::stdout_of(&output)
}

/// The session variables split one host into one identity per session, and the
/// first one that is set wins.
#[test]
fn the_session_variables_split_one_host_into_sessions() {
    let host = common::host_of_this_process();
    for (var, value) in [
        ("CLAUDE_CODE_SESSION_ID", "cc-session-1"),
        ("OPENCODE_SESSION_ID", "oc-session-2"),
        ("MINI_SWE_SESSION_ID", "generic-3"),
    ] {
        let stdout = whoami_with(&[(var, value)], None);
        assert_eq!(
            stdout.lines().next(),
            Some(format!("agent {host}/session:{value}").as_str()),
            "{var} must qualify the host: {stdout}"
        );
        assert!(
            stdout.contains("host process") && stdout.contains(&format!("session {value}")),
            "whoami must say how the session was derived: {stdout}"
        );
    }

    let first = whoami_with(
        &[
            ("CLAUDE_CODE_SESSION_ID", "cc-session-1"),
            ("OPENCODE_SESSION_ID", "oc-session-2"),
            ("MINI_SWE_SESSION_ID", "generic-3"),
        ],
        None,
    );
    assert_eq!(
        first.lines().next(),
        Some(format!("agent {host}/session:cc-session-1").as_str()),
        "the first variable that is set wins: {first}"
    );
}

/// A shell with no session variable keeps the host identity it has today.
#[test]
fn a_shell_with_no_session_variable_keeps_its_host_identity() {
    let host = common::host_of_this_process();
    let stdout = whoami_with(&[], None);
    assert_eq!(
        stdout.lines().next(),
        Some(format!("agent {host}").as_str()),
        "no session, so the host alone: {stdout}"
    );
    assert!(
        stdout.contains("derived from host process"),
        "whoami must say the host answered, with no session in it: {stdout}"
    );
}

/// A watch token in the environment is the caller's identity, resolved through
/// the hub directory the daemon keeps it in.
#[test]
fn a_watch_token_in_the_environment_names_the_dispatching_session() {
    let dir = scratch_dir();
    let store = WatchTokens::new(dir.clone());
    let token = store
        .token_for(TAB_A)
        .expect("mint a token for the session");
    assert_eq!(store.identity_of(&token).as_deref(), Some(TAB_A));

    let stdout = whoami_with(&[("MINI_SWE_WATCH_TOKEN", &token)], Some(&dir));
    assert_eq!(
        stdout.lines().next(),
        Some(format!("agent {TAB_A}").as_str()),
        "the token is the identity: {stdout}"
    );
    assert!(
        stdout.contains("MINI_SWE_WATCH_TOKEN"),
        "whoami must say the token answered: {stdout}"
    );

    // An unknown token is not an identity, so the host answers instead.
    let unknown = whoami_with(&[("MINI_SWE_WATCH_TOKEN", &"a".repeat(32))], Some(&dir));
    assert_eq!(
        unknown.lines().next(),
        Some(format!("agent {}", common::host_of_this_process()).as_str()),
        "an unknown token falls through to the host: {unknown}"
    );
}
