//! Hub daemon: one socket, one pool, two clients at once.
//!
//! The daemon is started in-process on a scratch `SWE_HUB_DIR` with a short
//! idle window, so no test needs an LLM or a spawned binary: every assertion
//! goes through the same Unix socket a thin client would dial.

mod common;

use mini_swe_mcp::hub::{HubConfig, HubPaths, HubServer, hub_dir};
use mini_swe_mcp::manifest::ModelManifest;
use mini_swe_mcp::mcp::McpServer;
use mini_swe_mcp::pool::WorkerPool;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;

static TAG: AtomicU64 = AtomicU64::new(0);

/// A scratch hub directory, removed when the test ends.
fn scratch_dir() -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "swe-hub-test-{}-{}-{}",
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

/// A server backed by a pool that can answer handshake verbs without an LLM.
fn server() -> Arc<McpServer> {
    let pool =
        WorkerPool::new(4, "http://localhost:1".to_string(), "test-key".to_string())
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

    /// Send one request and return its decoded `result`.
    async fn call(&mut self, method: &str) -> serde_json::Value {
        self.request(method, serde_json::json!({}))
            .await["result"]
            .clone()
    }

    /// Send one request and return the whole reply envelope, error included.
    async fn request(&mut self, method: &str, params: serde_json::Value) -> serde_json::Value {
        let id = self.next_id;
        self.next_id += 1;
        let frame =
            serde_json::json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params});
        self.writer
            .write_all(format!("{}\n", frame).as_bytes())
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
            // Event notifications carry no `id`; the answer carries ours.
            if reply.get("id") == Some(&serde_json::json!(id)) {
                return reply;
            }
        }
    }

    /// Send a notification, e.g. the `hub/hello` handshake.
    async fn notify(&mut self, method: &str, params: serde_json::Value) {
        let frame = serde_json::json!({"jsonrpc": "2.0", "method": method, "params": params});
        self.writer
            .write_all(format!("{}\n", frame).as_bytes())
            .await
            .expect("write notification");
        self.writer.flush().await.expect("flush notification");
    }

    /// The handshake: the client name becomes part of the connection identity.
    async fn initialize(&mut self, client: &str) {
        let reply = self
            .request(
                "initialize",
                serde_json::json!({
                    "protocolVersion": "2024-11-05", "capabilities": {},
                    "clientInfo": {"name": client, "version": "test"}
                }),
            )
            .await;
        assert_eq!(reply["result"]["protocolVersion"], "2024-11-05");
    }

    /// One `worker` tool call: the decoded payload, or the error message.
    async fn worker(
        &mut self,
        arguments: serde_json::Value,
    ) -> Result<serde_json::Value, String> {
        let reply = self
            .request(
                "tools/call",
                serde_json::json!({"name": "worker", "arguments": arguments}),
            )
            .await;
        if let Some(message) = reply.get("error").and_then(|e| e["message"].as_str()) {
            return Err(message.to_string());
        }
        let text = reply["result"]["content"][0]["text"]
            .as_str()
            .expect("worker tool result text")
            .to_string();
        Ok(serde_json::from_str(&text).expect("worker payload is JSON"))
    }
}

/// Two concurrent clients share one daemon and one pool.
#[tokio::test]
async fn two_clients_share_one_daemon() {
    let dir = scratch_dir();
    let config = HubConfig::new(
        hub_paths_for_test(&dir),
        60,
    );
    let daemon = HubServer::new(server(), config);
    let task = tokio::spawn(async move { daemon.run().await });

    let socket = dir.join("hub.sock");
    wait_for_socket(&socket).await;
    let (mut a, mut b) = tokio::join!(Client::connect(&socket), Client::connect(&socket));
    let (init_a, init_b) = tokio::join!(
        a.call("initialize"),
        b.call("initialize"),
    );
    assert_eq!(init_a["protocolVersion"], "2024-11-05");
    assert_eq!(init_b["protocolVersion"], "2024-11-05");
    let (tools, ping) = tokio::join!(a.call("tools/list"), b.call("ping"));
    assert!(tools["tools"].is_array());
    assert_eq!(ping, serde_json::json!({}));

    // Dropping both clients closes the connections; stopping the daemon task
    // is what ends the test daemon, which must remove its socket.
    drop(a);
    drop(b);
    task.abort();
    let _ = task.await;
    let _ = std::fs::remove_dir_all(&dir);
}

/// A second daemon on a locked directory reports `false` and leaves the first.
#[tokio::test]
async fn second_daemon_defers_to_the_lock_holder() {
    let dir = scratch_dir();
    let first = HubServer::new(
        server(),
        HubConfig::new(hub_paths_for_test(&dir), 60),
    );
    let running = Arc::new(tokio::sync::Mutex::new(false));
    let flag = running.clone();
    let task = tokio::spawn(async move {
        let held = first.run().await.expect("first daemon runs");
        *flag.lock().await = held;
    });
    wait_for_socket(&dir.join("hub.sock")).await;
    let second = HubServer::new(
        server(),
        HubConfig::new(hub_paths_for_test(&dir), 60),
    );
    assert!(!second.run().await.expect("lock query runs"));

    task.abort();
    let _ = task.await;
    let _ = std::fs::remove_dir_all(&dir);
}

/// An idle daemon exits by itself and removes its socket.
#[tokio::test]
async fn idle_daemon_removes_its_socket() {
    let dir = scratch_dir();
    let daemon = HubServer::new(
        server(),
        HubConfig::new(hub_paths_for_test(&dir), 1),
    );
    let socket = dir.join("hub.sock");
    let probe = socket.clone();
    let task = tokio::spawn(async move { daemon.run().await });
    wait_for_socket(&probe).await;
    let held = tokio::time::timeout(std::time::Duration::from_secs(15), task)
        .await
        .expect("daemon exits while idle")
        .expect("daemon task joins")
        .expect("daemon runs");
    assert!(held);
    assert!(!socket.exists(), "idle shutdown removes hub.sock");
    assert!(dir.join("hub.log").is_file(), "daemon records its lifecycle in hub.log");
    let _ = std::fs::remove_dir_all(&dir);
}

/// A group/world accessible hub directory is refused.
#[tokio::test]
async fn world_writable_hub_dir_is_refused() {
    let dir = scratch_dir();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o777))
            .expect("chmod scratch dir");
    }
    let before = std::env::var_os("SWE_HUB_DIR");
    unsafe { std::env::set_var("SWE_HUB_DIR", &dir) };
    let refused = hub_dir().is_err();
    match before {
        Some(v) => unsafe { std::env::set_var("SWE_HUB_DIR", v) },
        None => unsafe { std::env::remove_var("SWE_HUB_DIR") },
    }
    assert!(refused, "a world-writable hub dir must be refused");
    let _ = std::fs::remove_dir_all(&dir);
}

/// Build hub paths for a scratch directory without touching global state.
fn hub_paths_for_test(dir: &Path) -> HubPaths {
    HubPaths::new(dir.to_path_buf())
}

/// Thin-client lifecycle over the real binary: the first CLI call auto-starts
/// the daemon, the second reuses it, the `--stdio` proxy answers the handshake
/// verbs, and the escape hatch never creates a socket.
#[test]
fn thin_clients_autostart_reuse_and_proxy_through_the_hub() {
    use std::io::{BufRead, BufReader as StdBufReader, Write};
    use std::process::{Command, Stdio};

    let exe = common::binary_path();
    let hub = common::TempDir::new_in_tmp("hub-thin");
    let hub_dir = hub.path().to_path_buf();
    let _reaper = DaemonReaper(hub_dir.clone());
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&hub_dir, std::fs::Permissions::from_mode(0o700))
            .expect("restrict the hub dir to 0700");
    }
    // Worktrees and registry rows of the probe dispatch stay in the scratch dir.
    let swe = hub.subdir("swe");
    let envs = |cmd: &mut Command| {
        cmd.env("SWE_HUB_DIR", &hub_dir)
            .env("SWE_TEMP_DIR", &swe)
            .env("OPENAI_API_KEY", "test-key-not-used-by-list")
            .env("ENV_FILE", format!("{}/.env.does-not-exist", env!("CARGO_MANIFEST_DIR")))
            .env("MODELS_FILE", format!("{}/models.yaml", env!("CARGO_MANIFEST_DIR")));
    };

    // First CLI call auto-starts the daemon and answers through it.
    let mut first = Command::new(&exe);
    envs(&mut first);
    let out = first
        .args(["list", "--json"])
        .output()
        .expect("first CLI call runs");
    assert!(out.status.success(), "first CLI call must succeed: {}", String::from_utf8_lossy(&out.stderr));
    assert!(hub_dir.join("hub.sock").exists(), "first call auto-starts the daemon");
    let first_log = std::fs::read_to_string(hub_dir.join("hub.log")).expect("daemon writes hub.log");
    assert!(first_log.contains("listening"), "daemon records startup: {first_log}");

    // Second CLI call reuses the same daemon: no second listener starts.
    let mut second = Command::new(&exe);
    envs(&mut second);
    let out = second
        .args(["list", "--json"])
        .output()
        .expect("second CLI call runs");
    assert!(out.status.success(), "second CLI call must succeed: {}", String::from_utf8_lossy(&out.stderr));
    let second_log = std::fs::read_to_string(hub_dir.join("hub.log")).expect("daemon log persists");
    // Both CLI connections reached the daemon; the socket path stayed put and
    // no second listener line was appended (stderr lines also say "listening").
    assert!(
        second_log.matches("Hub daemon listening").count() <= 1,
        "the second call must reuse the daemon: {second_log}"
    );

    // The `--stdio` proxy answers the handshake verbs through the same daemon.
    let mut proxy = Command::new(&exe);
    envs(&mut proxy);
    let mut child = proxy
        .arg("--stdio")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn the stdio proxy");
    let mut stdin = child.stdin.take().expect("proxy stdin");
    let stdout = child.stdout.take().expect("proxy stdout");
    let mut lines = StdBufReader::new(stdout).lines();
    for (id, method) in [(1, "initialize"), (2, "tools/list")] {
        writeln!(stdin, "{{\"jsonrpc\":\"2.0\",\"id\":{id},\"method\":\"{method}\"}}")
            .expect("write a proxy frame");
    }
    stdin.flush().expect("flush proxy frames");
    // Requests are served concurrently, so the replies may arrive in either
    // order: collect one response per request (skipping notifications, which
    // carry no id) and check the set.
    let mut seen = String::new();
    let mut responses = 0;
    while responses < 2 {
        match lines.next() {
            Some(Ok(line)) if line.contains("\"id\"") => {
                responses += 1;
                seen.push_str(&line);
            }
            Some(Ok(_)) => {}
            Some(Err(e)) => panic!("reading the proxy reply: {e}"),
            None => break,
        }
    }
    for expected in ["\"protocolVersion\":\"2024-11-05\"", "\"tools\""] {
        assert!(seen.contains(expected), "proxy must answer with {expected}: {seen}");
    }
    drop(stdin);
    let _ = child.wait();

    // A dispatch against an unreachable API base records the worker in the hub
    // and survives the CLI process exiting (no LLM call is awaited).
    let mut dispatch = Command::new(&exe);
    envs(&mut dispatch);
    let out = dispatch
        .args(["dispatch", "thin-client lifecycle probe", "--json"])
        .env("OPENAI_API_BASE", "http://127.0.0.1:1")
        .output()
        .expect("dispatch runs");
    assert!(out.status.success(), "dispatch must be accepted: {}", String::from_utf8_lossy(&out.stderr));
    let payload: serde_json::Value =
        serde_json::from_str(String::from_utf8_lossy(&out.stdout).trim())
            .expect("dispatch prints JSON");
    let wid = payload["worker_id"].as_str().expect("dispatch names the worker").to_string();
    assert_eq!(
        payload["owner"], "cli",
        "a CLI dispatch is owned by the stable `cli` identity: {payload}"
    );
    let mut status = Command::new(&exe);
    envs(&mut status);
    let out = status
        .args(["status", &wid, "--json"])
        .output()
        .expect("status runs");
    assert!(
        out.status.success(),
        "the worker record must survive the dispatching CLI exiting: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    // That identity is stable across invocations, so a *second* CLI process
    // still sees (and could steer) the worker the first one dispatched.
    let mut listing = Command::new(&exe);
    envs(&mut listing);
    let out = listing
        .args(["list", "--json"])
        .output()
        .expect("second CLI list runs");
    assert!(out.status.success(), "list must succeed: {}", String::from_utf8_lossy(&out.stderr));
    let listed: serde_json::Value =
        serde_json::from_str(String::from_utf8_lossy(&out.stdout).trim())
            .expect("list prints JSON");
    let row = listed["workers"]
        .as_array()
        .expect("workers array")
        .iter()
        .find(|row| row["id"] == wid.as_str())
        .unwrap_or_else(|| panic!("the `cli` identity must keep its workers across invocations: {listed}"));
    assert_eq!(row["owner"], "cli");

    // The escape hatch never creates a socket.
    let bare = common::TempDir::new_in_tmp("hub-no-daemon");
    let mut local = Command::new(&exe);
    local
        .env("SWE_HUB_DIR", bare.path())
        .env("MINI_SWE_NO_DAEMON", "1")
        .env("OPENAI_API_KEY", "test-key-not-used-by-list")
        .env("ENV_FILE", format!("{}/.env.does-not-exist", env!("CARGO_MANIFEST_DIR")))
        .env("MODELS_FILE", format!("{}/models.yaml", env!("CARGO_MANIFEST_DIR")));
    let out = local
        .args(["list", "--json"])
        .output()
        .expect("local CLI call runs");
    assert!(out.status.success(), "escape-hatch call must succeed: {}", String::from_utf8_lossy(&out.stderr));
    assert!(!bare.path().join("hub.sock").exists(), "MINI_SWE_NO_DAEMON=1 creates no socket");
}

/// Terminates every daemon that logged into `hub_dir` when dropped.
///
/// Auto-started daemons are detached (`setsid`) and a daemon with running
/// workers never idles out, so a test must stop the ones it caused or they
/// outlive the suite (and hold its output pipe open).
struct DaemonReaper(std::path::PathBuf);

impl Drop for DaemonReaper {
    fn drop(&mut self) {
        let log = std::fs::read_to_string(self.0.join("hub.log")).unwrap_or_default();
        let pids: std::collections::BTreeSet<i32> = log
            .split_whitespace()
            .filter_map(|word| word.strip_prefix("pid=")?.parse().ok())
            .collect();
        for pid in pids {
            // SAFETY: `kill` takes plain integers; a stale pid only yields ESRCH.
            unsafe { libc::kill(pid, libc::SIGTERM) };
        }
    }
}

/// Two clients with different working directories and no `repo_path` each get
/// their own repository: the daemon resolves a relative path against the
/// caller's `cwd` from `hub/hello`, not against its own.
#[test]
fn a_relative_repo_path_resolves_against_the_callers_cwd() {
    use std::process::Command;

    let exe = common::binary_path();
    let hub = common::TempDir::new_in_tmp("hub-cwd");
    let _reaper = DaemonReaper(hub.path().to_path_buf());
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(hub.path(), std::fs::Permissions::from_mode(0o700))
            .expect("restrict the hub dir to 0700");
    }
    let (repo_a, repo_b) = (hub.subdir("repo-a"), hub.subdir("repo-b"));
    // A shared registry scratch dir, so both clients' worker rows land in one
    // place the daemon (which inherits `SWE_TEMP_DIR`) can be read back from.
    let swe = hub.subdir("swe");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&swe, std::fs::Permissions::from_mode(0o700))
            .expect("restrict the scratch dir to 0700");
    }
    for repo in [&repo_a, &repo_b] {
        let _ = Command::new("git").args(["init", "-q"]).current_dir(repo).output();
        let _ = Command::new("git")
            .args(["commit", "-q", "--allow-empty", "-m", "seed"])
            .current_dir(repo)
            .env("GIT_AUTHOR_NAME", "t").env("GIT_AUTHOR_EMAIL", "t@t")
            .env("GIT_COMMITTER_NAME", "t").env("GIT_COMMITTER_EMAIL", "t@t")
            .output();
    }

    // A dead API base keeps the dispatch from an LLM call: the worker record
    // still names the repository the daemon resolved for this caller.
    let dispatch = |repo: &std::path::Path| -> String {
        let out = Command::new(&exe)
            .current_dir(repo)
            .args(["dispatch", "cwd probe", "--json"])
            .env("SWE_HUB_DIR", hub.path())
            .env("SWE_TEMP_DIR", &swe)
            .env("OPENAI_API_KEY", "test-key-not-used-by-dispatch")
            .env("OPENAI_API_BASE", "http://127.0.0.1:1")
            .env("ENV_FILE", format!("{}/.env.does-not-exist", env!("CARGO_MANIFEST_DIR")))
            .env("MODELS_FILE", format!("{}/models.yaml", env!("CARGO_MANIFEST_DIR")))
            .output()
            .expect("dispatch runs");
        assert!(out.status.success(), "dispatch must be accepted: {}", String::from_utf8_lossy(&out.stderr));
        let payload: serde_json::Value =
            serde_json::from_str(String::from_utf8_lossy(&out.stdout).trim()).expect("dispatch prints JSON");
        payload["worker_id"].as_str().expect("dispatch names the worker").to_string()
    };

    let (a, b) = (dispatch(&repo_a), dispatch(&repo_b));
    // The registry row the daemon wrote is the record of which repository the
    // caller's `cwd` resolved to.
    let record = |wid: &str| -> serde_json::Value {
        let path = swe.join("swe-registry").join(format!("{wid}.json"));
        let raw = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {wid} row: {e}"));
        serde_json::from_str(&raw).expect("registry row is JSON")
    };

    let (seen_a, seen_b) = (record(&a), record(&b));
    assert_ne!(a, b, "each dispatch is its own worker");
    assert_eq!(
        seen_a["repo_path"].as_str(),
        Some(repo_a.to_string_lossy().as_ref()),
        "the first caller's cwd must win: {seen_a}"
    );
    assert_eq!(
        seen_b["repo_path"].as_str(),
        Some(repo_b.to_string_lossy().as_ref()),
        "the second caller's cwd must win: {seen_b}"
    );
}

/// Ownership over the wire (H-3): the daemon refuses to let one agent steer,
/// kill or collect another agent's worker, keeps `status` readable by everyone,
/// and honours the operator's `admin` hello.
#[tokio::test]
async fn a_hub_connection_only_controls_its_own_workers() {
    use mini_swe_mcp::mcp::ConnectionContext;
    use mini_swe_mcp::pool::{LogBuffer, WorkerMetrics, WorkerRecord, WorkerState};

    // The daemon serves the very pool this test fills, so ownership is decided
    // against a record that is definitely there (no LLM, no registry row).
    let pool = WorkerPool::new(4, "http://localhost:1".to_string(), "test-key".to_string())
        .with_manifest(std::sync::Arc::new(ModelManifest::default()));
    pool.__test_insert_worker(WorkerRecord {
        id: "h3-hub-worker".to_string(),
        task: "owned by agent-a".to_string(),
        model: "test-model".to_string(),
        owner: "agent-a".to_string(),
        state: WorkerState::Running {
            step: 1,
            last_command: "cargo test".to_string(),
            started_at: 0,
        },
        metrics: WorkerMetrics::default(),
        logs: LogBuffer::new(),
        pending_steer: Vec::new(),
        resume_tx: None,
        handle: None,
        revision: 0,
    })
    .await;
    let server = Arc::new(McpServer::new(pool, "test-model".to_string()));

    let dir = scratch_dir();
    let daemon = HubServer::new(server, HubConfig::new(hub_paths_for_test(&dir), 60));
    let task = tokio::spawn(async move { daemon.run().await });
    let socket = dir.join("hub.sock");
    wait_for_socket(&socket).await;

    // Agent B: an ordinary orchestrator, identified by its hello.
    let mut b = Client::connect(&socket).await;
    b.initialize("other-host").await;
    b.notify("hub/hello", serde_json::json!({"agent_id": "agent-b"})).await;
    for arguments in [
        serde_json::json!({"action": "steer", "worker_id": "h3-hub-worker", "message": "stop"}),
        serde_json::json!({"action": "kill", "worker_id": "h3-hub-worker"}),
        serde_json::json!({"action": "collect", "worker_id": "h3-hub-worker"}),
        serde_json::json!({"action": "wait", "worker_id": "h3-hub-worker", "timeout_secs": 0}),
    ] {
        let action = arguments["action"].as_str().expect("action").to_string();
        let error = b
            .worker(arguments)
            .await
            .expect_err("another agent's worker must be refused")
            ;
        assert_eq!(
            error,
            "worker h3-hub-worker belongs to agent agent-a",
            "'{action}' must name the owning agent: {error}"
        );
    }
    // The refusals left the worker alone, and reading it stays open.
    let status = b
        .worker(serde_json::json!({"action": "status", "worker_id": "h3-hub-worker"}))
        .await
        .expect("status is readable by every agent");
    assert_eq!(status["owner"], "agent-a");
    assert_eq!(status["state"]["state"], "Running");

    // The owner may act on it, and the CLI's stable identity is `cli` whatever
    // its connection id (H-3), so it never sees another agent's worker either.
    let mut a = Client::connect(&socket).await;
    a.initialize("other-host").await;
    a.notify("hub/hello", serde_json::json!({"agent_id": "agent-a"})).await;
    let steered = a
        .worker(serde_json::json!({"action": "steer", "worker_id": "h3-hub-worker", "message": "go"}))
        .await
        .expect("the owner may steer its own worker");
    assert_eq!(steered["status"], "steered");

    let mut cli = Client::connect(&socket).await;
    cli.initialize(mini_swe_mcp::mcp::CLI_CLIENT_NAME).await;
    let listed = cli
        .worker(serde_json::json!({"action": "list"}))
        .await
        .expect("list answers");
    let ids: Vec<&str> = listed["workers"]
        .as_array()
        .expect("workers array")
        .iter()
        .filter_map(|row| row["id"].as_str())
        .collect();
    assert!(!ids.contains(&"h3-hub-worker"), "the CLI must not list it: {listed}");
    let all = cli
        .worker(serde_json::json!({"action": "list", "scope": "all"}))
        .await
        .expect("scope=all answers");
    let row = all["workers"]
        .as_array()
        .expect("workers array")
        .iter()
        .find(|row| row["id"] == "h3-hub-worker")
        .unwrap_or_else(|| panic!("scope=all must list it: {all}"));
    assert_eq!(row["owner"], "agent-a");

    // The operator's `--admin`: the same connection, with the override in hello.
    let mut operator = Client::connect(&socket).await;
    operator.initialize(mini_swe_mcp::mcp::CLI_CLIENT_NAME).await;
    operator
        .notify(
            "hub/hello",
            serde_json::json!({"agent_id": "agent-b", "admin": true}),
        )
        .await;
    let killed = operator
        .worker(serde_json::json!({"action": "kill", "worker_id": "h3-hub-worker"}))
        .await
        .expect("the admin override must bypass the ownership check");
    assert_eq!(killed["killed"], true);
    assert_eq!(
        ConnectionContext::hub_connection(1).agent(),
        "connection#1",
        "an unannounced connection is scoped to itself"
    );

    drop(a);
    drop(b);
    drop(cli);
    drop(operator);
    task.abort();
    let _ = task.await;
    let _ = std::fs::remove_dir_all(&dir);
}

/// Crash recovery: a dead row and its dirty checkout become a failed row with
/// the salvage message, the uncommitted file lands on `worker-<id>`, and the
/// saved history survives for a later revision.
#[test]
fn daemon_recovers_an_orphaned_worker_on_startup() {
    use mini_swe_mcp::pool::{RegistryStatus, WorkerRegistryEntry};
    use std::process::{Command, Stdio};

    fn git(repo: &Path, args: &[&str]) {
        let out = Command::new("git")
            .current_dir(repo)
            .args(args)
            .output()
            .unwrap_or_else(|_| panic!("git {args:?} failed to run"));
        assert!(out.status.success(), "git {args:?}: {}", String::from_utf8_lossy(&out.stderr));
    }

    let root = scratch_dir();
    let hub = root.join("hub");
    let swe = root.join("swe");
    let repo = root.join("repo");
    for dir in [&hub, &swe, &repo] {
        std::fs::create_dir_all(dir).expect("create scratch dir");
    }
    let mut dead = Command::new("true").spawn().expect("spawn short-lived owner");
    let dead_pid = dead.id();
    dead.wait().expect("reap owner");
    assert!(!mini_swe_mcp::worktree::is_process_alive(dead_pid));
    let _reaper = DaemonReaper(hub.clone());
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&hub, std::fs::Permissions::from_mode(0o700))
            .expect("restrict hub dir");
        std::fs::set_permissions(&swe, std::fs::Permissions::from_mode(0o700))
            .expect("restrict swe dir");
    }
    // A repo with one commit, plus a registered worktree carrying an
    // uncommitted file on `worker-<orphan>`.
    let wid = "orphan1";
    git(&repo, &["init", "-b", "master"]);
    git(&repo, &["config", "user.name", "t"]);
    git(&repo, &["config", "user.email", "t@t"]);
    std::fs::write(repo.join("base.txt"), "base\n").expect("seed the repo");
    git(&repo, &["add", "base.txt"]);
    git(&repo, &["commit", "-m", "seed"]);
    let branch = format!("worker-{wid}");
    let checkout = swe.join(format!("swe-wt-{wid}"));
    git(&repo, &[
        "worktree", "add", "-b", &branch, &checkout.to_string_lossy(), "HEAD",
    ]);
    std::fs::write(checkout.join("dirty.txt"), "unsaved\n").expect("dirty the checkout");
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock before epoch")
        .as_secs();
    let row = serde_json::json!({
        "id": wid, "pid": dead_pid, "task": "t", "model": "ninja", "status": "running",
        "step": 3, "max_turns": 10, "last_command": "cargo test",
        "started_at": now - 60, "updated_at": now - 60,
    });
    std::fs::create_dir_all(swe.join("swe-registry")).expect("create the registry");
    std::fs::write(swe.join("swe-registry").join(format!("{wid}.json")), row.to_string())
        .expect("write the orphan row");
    let history = serde_json::json!({"worker": wid});
    std::fs::write(swe.join(format!("swe-wt-{wid}.history.json")), history.to_string())
        .expect("write the history file");

    let mut daemon = Command::new(common::binary_path())
        .arg("daemon")
        .env("SWE_HUB_DIR", &hub)
        .env("SWE_TEMP_DIR", &swe)
        .env("TMPDIR", &swe)
        .env("HUB_IDLE_SECS", "1")
        .env("ENV_FILE", root.join("absent.env"))
        .env("MODELS_FILE", format!("{}/models.yaml", env!("CARGO_MANIFEST_DIR")))
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn recovery daemon");
    for _ in 0..100 {
        if std::os::unix::net::UnixStream::connect(hub.join("hub.sock")).is_ok() {
            break;
        }
        assert!(daemon.try_wait().expect("probe daemon").is_none(), "daemon exited early");
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    assert!(hub.join("hub.sock").exists(), "daemon must listen after recovery");
    let entry: WorkerRegistryEntry = serde_json::from_slice(
        &std::fs::read(swe.join("swe-registry").join(format!("{wid}.json")))
            .expect("the orphan row survives recovery"),
    ).expect("registry JSON");
    assert_eq!(entry.status, RegistryStatus::Failed, "the orphan row must be failed");
    let expected = format!("hub restarted; work salvaged on branch worker-{wid}");
    assert_eq!(entry.last_command, expected, "the owner must see the salvage branch");
    let log = Command::new("git")
        .current_dir(&repo)
        .args(["log", &branch, "--oneline"])
        .output()
        .expect("read the worker branch");
    assert!(
        String::from_utf8_lossy(&log.stdout).contains("salvaged uncommitted work"),
        "the uncommitted file must be committed on {branch}"
    );
    let show = Command::new("git")
        .current_dir(&repo)
        .args(["show", &format!("{branch}:dirty.txt")])
        .output()
        .expect("read the salvaged file");
    assert_eq!(String::from_utf8_lossy(&show.stdout).trim(), "unsaved");
    assert!(
        swe.join(format!("swe-wt-{wid}.history.json")).is_file(),
        "recovery must keep the history file for revision"
    );
    let registered = Command::new("git")
        .current_dir(&repo)
        .args(["worktree", "list", "--porcelain"])
        .output()
        .expect("list worktrees");
    assert!(
        !String::from_utf8_lossy(&registered.stdout).contains(checkout.to_string_lossy().as_ref()),
        "the orphan checkout must be released so steer can reopen the branch"
    );

    for _ in 0..100 {
        if daemon.try_wait().expect("probe idle shutdown").is_some() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    assert!(daemon.try_wait().expect("probe exit").is_some(), "daemon must idle out");
    let log = std::fs::read_to_string(hub.join("hub.log")).expect("read recovery log");
    assert!(log.contains("recovered 1 orphaned workers"), "{log}");
    let _ = std::fs::remove_dir_all(&root);
}

/// SIGKILL bypasses the worker's exit path: only checkpoint history makes the
/// salvaged worker revisable when the next daemon starts.
#[tokio::test]
async fn a_checkpointed_worker_survives_hub_sigkill_and_revision() {
    use std::process::Stdio;
    use tokio::io::AsyncReadExt;
    use tokio::net::TcpListener;

    let root = common::TempDir::new_in_tmp("hub-checkpoint-revision");
    // Unix socket paths are bounded by sockaddr_un, unlike scratch paths.
    let hub = std::env::temp_dir().join(format!("h5-{}", std::process::id()));
    std::fs::create_dir_all(&hub).unwrap();
    let swe = root.subdir("swe");
    let repo = root.subdir("repo");
    for dir in [&hub, &swe] {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700)).unwrap();
    }
    common::git(&repo, &["init", "-b", "master"]);
    common::git(&repo, &["config", "user.name", "test"]);
    common::git(&repo, &["config", "user.email", "test@localhost"]);
    common::git(&repo, &["commit", "--allow-empty", "-m", "seed"]);

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let api_base = format!("http://{}/v1", listener.local_addr().unwrap());
    let llm = tokio::spawn(async move {
        for turn in 1..=21 {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut bytes = Vec::new();
            let header_end = loop {
                let mut byte = [0];
                socket.read_exact(&mut byte).await.unwrap();
                bytes.push(byte[0]);
                if bytes.ends_with(b"\r\n\r\n") { break bytes.len(); }
            };
            let headers = String::from_utf8_lossy(&bytes).to_lowercase();
            let length: usize = headers.lines().find_map(|line| {
                line.strip_prefix("content-length:")?.trim().parse().ok()
            }).unwrap();
            bytes.resize(header_end + length, 0);
            socket.read_exact(&mut bytes[header_end..]).await.unwrap();
            if turn == 20 {
                // Leave the post-checkpoint LLM call unfinished until SIGKILL.
                continue;
            }
            let command = match turn {
                19 => "echo checkpoint > kept.txt".to_string(),
                21 => "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT".to_string(),
                _ => format!("echo turn {turn}"),
            };
            let chunk = serde_json::json!({"choices":[{"delta":{
                "content": format!("```bash\n{command}\n```"),
            }}]});
            let body = format!("data: {chunk}\n\ndata: [DONE]\n\n");
            socket.write_all(format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()
            ).as_bytes()).await.unwrap();
        }
    });
    let command = || {
        let mut cmd = tokio::process::Command::new(common::binary_path());
        cmd.env("SWE_HUB_DIR", &hub).env("SWE_TEMP_DIR", &swe).env("TMPDIR", &swe)
            .env_remove("MINI_SWE_NO_DAEMON")
            .env("OPENAI_API_BASE", &api_base).env("OPENAI_API_KEY", "test-key")
            .env("ENV_FILE", root.path().join("absent.env"))
            .env("MODELS_FILE", format!("{}/models.yaml", env!("CARGO_MANIFEST_DIR")))
            .env("HUB_IDLE_SECS", "60");
        cmd
    };
    let mut first = command().arg("daemon").stdout(Stdio::null()).stderr(Stdio::null())
        .kill_on_drop(true).spawn().unwrap();
    wait_for_socket(&hub.join("hub.sock")).await;
    let out = command().args(["dispatch", "checkpoint recovery", "--repo", repo.to_str().unwrap(),
        "--model", "test-model", "--max-turns", "30", "--verify", "", "--json"])
        .output().await.unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let dispatched: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let wid = dispatched["worker_id"].as_str().unwrap();
    let history_path = swe.join(format!("swe-wt-{wid}.history.json"));
    for _ in 0..400 {
        if history_path.is_file() { break; }
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    }
    let checkpoint: mini_swe_mcp::pool::WorkerHistory = serde_json::from_slice(
        &std::fs::read(&history_path).expect("checkpoint must save history before exit")
    ).unwrap();
    assert!(mini_swe_mcp::pool::is_replayable(&checkpoint.messages));
    assert!(checkpoint.messages.len() >= 40, "must retain exchanges before turn 20");
    assert_eq!(checkpoint.owner.as_deref(), Some("cli"));
    assert_eq!(checkpoint.verify, None);
    first.kill().await.unwrap();
    first.wait().await.unwrap();

    let mut second = command().arg("daemon").stdout(Stdio::null()).stderr(Stdio::null())
        .kill_on_drop(true).spawn().unwrap();
    wait_for_socket(&hub.join("hub.sock")).await;
    let recovered: mini_swe_mcp::pool::WorkerRegistryEntry = serde_json::from_slice(
        &std::fs::read(swe.join("swe-registry").join(format!("{wid}.json"))).unwrap()
    ).unwrap();
    assert_eq!(recovered.status, mini_swe_mcp::pool::RegistryStatus::Failed);
    assert_eq!(recovered.last_command, format!("hub restarted; work salvaged on branch worker-{wid}"));
    let out = tokio::time::timeout(std::time::Duration::from_secs(15),
        command().args(["steer", wid, "resume after crash", "--max-turns", "2", "--wait", "--json"])
            .output()).await.expect("revision must finish").unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert!(String::from_utf8_lossy(&out.stdout).contains(&checkpoint.branch));
    assert_eq!(common::git(&repo, &["show", &format!("{}:kept.txt", checkpoint.branch)]).trim(), "checkpoint");
    second.kill().await.unwrap();
    second.wait().await.unwrap();
    llm.await.unwrap();
    std::fs::remove_dir_all(&hub).unwrap();
}
