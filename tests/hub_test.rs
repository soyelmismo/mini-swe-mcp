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
