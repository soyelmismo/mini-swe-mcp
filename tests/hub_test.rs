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
use tokio::net::{TcpListener, UnixStream};

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
    let _scratch = common::TempDir::new_in_tmp("iso-hub-1");
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
async fn wait_for_socket(path: &Path) {
    for _ in 0..100 {
        if UnixStream::connect(path).await.is_ok() {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    panic!("hub socket {} never came up", path.display());
}

/// Wait until the worker's registry row reports `status`, or return the last
/// row read at the deadline.
///
/// The hub recovers orphans before it binds its socket, but a child of the
/// killed hub can keep the previous listener alive, so [`wait_for_socket`] may
/// return while the row still carries its pre-crash status: poll for the write.
async fn wait_for_registry_status(
    registry: &Path,
    wid: &str,
    status: mini_swe_mcp::pool::RegistryStatus,
) -> mini_swe_mcp::pool::WorkerRegistryEntry {
    let path = registry.join(format!("{wid}.json"));
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    loop {
        let row: mini_swe_mcp::pool::WorkerRegistryEntry =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        if row.status == status || std::time::Instant::now() >= deadline {
            return row;
        }
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    }
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
        self.request(method, serde_json::json!({})).await["result"].clone()
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
    let config = HubConfig::new(hub_paths_for_test(&dir), 60);
    let daemon = HubServer::new(server(), config);
    let task = tokio::spawn(async move { daemon.run().await });

    let socket = mini_swe_mcp::hub::HubPaths::new(dir.to_path_buf()).socket();
    wait_for_socket(&socket).await;
    let (mut a, mut b) = tokio::join!(Client::connect(&socket), Client::connect(&socket));
    let (init_a, init_b) = tokio::join!(a.call("initialize"), b.call("initialize"),);
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
    let first = HubServer::new(server(), HubConfig::new(hub_paths_for_test(&dir), 60));
    let running = Arc::new(tokio::sync::Mutex::new(false));
    let flag = running.clone();
    let task = tokio::spawn(async move {
        let held = first.run().await.expect("first daemon runs");
        *flag.lock().await = held;
    });
    wait_for_socket(&mini_swe_mcp::hub::HubPaths::new(dir.to_path_buf()).socket()).await;
    let second = HubServer::new(server(), HubConfig::new(hub_paths_for_test(&dir), 60));
    assert!(!second.run().await.expect("lock query runs"));

    task.abort();
    let _ = task.await;
    let _ = std::fs::remove_dir_all(&dir);
}

/// Teardown removes `hub.sock` before it drops `hub.lock`; a starter that
/// arrives in that window must wait for the lock instead of concluding that a
/// hub is already running.
#[tokio::test]
async fn a_starting_daemon_waits_out_a_shutting_down_predecessor() {
    let dir = scratch_dir();
    let paths = hub_paths_for_test(&dir);
    let lock = hold_hub_lock(&paths.lock());
    // The predecessor has already removed its socket.
    assert!(!paths.socket().exists(), "no socket while shutting down");

    let daemon = HubServer::new(server(), HubConfig::new(hub_paths_for_test(&dir), 60));
    let task = tokio::spawn(async move { daemon.run().await });

    // The predecessor releases the lock half a second later.
    let release = tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        drop(lock);
    });

    let socket = mini_swe_mcp::hub::HubPaths::new(dir.to_path_buf()).socket();
    let probe = socket.clone();
    tokio::time::timeout(std::time::Duration::from_secs(8), wait_for_socket(&probe))
        .await
        .expect("the starter must outwait the lock holder");
    release.await.expect("release task joins");
    assert!(socket.exists(), "the replacement daemon must be listening");

    task.abort();
    let _ = task.await;
    let _ = std::fs::remove_dir_all(&dir);
}

/// An idle daemon exits by itself and removes its socket.
#[tokio::test]
async fn idle_daemon_removes_its_socket() {
    let dir = scratch_dir();
    let daemon = HubServer::new(server(), HubConfig::new(hub_paths_for_test(&dir), 1));
    let socket = mini_swe_mcp::hub::HubPaths::new(dir.to_path_buf()).socket();
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
    assert!(
        dir.join("hub.log").is_file(),
        "daemon records its lifecycle in hub.log"
    );
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

/// Take the exclusive `flock` on `path`, standing in for a daemon that has not
/// finished its teardown; the lock is released when the returned file drops.
fn hold_hub_lock(path: &Path) -> std::fs::File {
    use std::os::unix::io::AsRawFd;
    let file = std::fs::OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(path)
        .expect("open hub lock");
    // SAFETY: `flock` takes the fd and an operation flag; no pointers.
    let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    assert_eq!(
        rc,
        0,
        "test must take the hub lock: {}",
        std::io::Error::last_os_error()
    );
    file
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
    // The probe dispatch passes `--repo`: without it `repo_path` resolves to
    // the cwd (the crate directory) and the worker's `worker-*` branch is
    // created in the crate's own repository.
    let repo = hub.subdir("repo");
    common::git(&repo, &["init", "-b", "master"]);
    common::git(&repo, &["config", "user.name", "test"]);
    common::git(&repo, &["config", "user.email", "test@localhost"]);
    common::git(&repo, &["commit", "--allow-empty", "-m", "seed"]);
    let envs = |cmd: &mut Command| {
        cmd.env("SWE_HUB_DIR", &hub_dir)
            .env("SWE_TEMP_DIR", &swe)
            .env("OPENAI_API_KEY", "test-key-not-used-by-list")
            .env(
                "ENV_FILE",
                format!("{}/.env.does-not-exist", env!("CARGO_MANIFEST_DIR")),
            )
            .env(
                "MODELS_FILE",
                format!("{}/models.yaml", env!("CARGO_MANIFEST_DIR")),
            );
    };

    // First CLI call auto-starts the daemon and answers through it.
    let mut first = common::binary_command(&exe);
    envs(&mut first);
    let out = first
        .args(["list", "--json"])
        .output()
        .expect("first CLI call runs");
    assert!(
        out.status.success(),
        "first CLI call must succeed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        mini_swe_mcp::hub::HubPaths::new(hub_dir.to_path_buf())
            .socket()
            .exists(),
        "first call auto-starts the daemon"
    );
    let first_log =
        std::fs::read_to_string(hub_dir.join("hub.log")).expect("daemon writes hub.log");
    assert!(
        first_log.contains("listening"),
        "daemon records startup: {first_log}"
    );

    // Second CLI call reuses the same daemon: no second listener starts.
    let mut second = common::binary_command(&exe);
    envs(&mut second);
    let out = second
        .args(["list", "--json"])
        .output()
        .expect("second CLI call runs");
    assert!(
        out.status.success(),
        "second CLI call must succeed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let second_log = std::fs::read_to_string(hub_dir.join("hub.log")).expect("daemon log persists");
    // Both CLI connections reached the daemon; the socket path stayed put and
    // no second listener line was appended (stderr lines also say "listening").
    assert!(
        second_log.matches("Hub daemon listening").count() <= 1,
        "the second call must reuse the daemon: {second_log}"
    );

    // The `--stdio` proxy answers the handshake verbs through the same daemon.
    let mut proxy = common::binary_command(&exe);
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
        writeln!(
            stdin,
            "{{\"jsonrpc\":\"2.0\",\"id\":{id},\"method\":\"{method}\"}}"
        )
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
        assert!(
            seen.contains(expected),
            "proxy must answer with {expected}: {seen}"
        );
    }
    drop(stdin);
    let _ = child.wait();

    // A dispatch against an unreachable API base records the worker in the hub
    // and survives the CLI process exiting (no LLM call is awaited).
    let mut dispatch = common::binary_command(&exe);
    envs(&mut dispatch);
    let out = dispatch
        .args([
            "dispatch",
            "thin-client lifecycle probe",
            "--json",
            "--repo",
        ])
        .arg(&repo)
        .env("OPENAI_API_BASE", "http://127.0.0.1:1")
        .output()
        .expect("dispatch runs");
    assert!(
        out.status.success(),
        "dispatch must be accepted: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let payload: serde_json::Value =
        serde_json::from_str(String::from_utf8_lossy(&out.stdout).trim())
            .expect("dispatch prints JSON");
    let wid = payload["worker_id"]
        .as_str()
        .expect("dispatch names the worker")
        .to_string();
    assert_eq!(
        payload["owner"],
        common::host_of_this_process(),
        "a CLI dispatch is owned by the host process it ran under: {payload}"
    );
    let mut status = common::binary_command(&exe);
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
    let mut listing = common::binary_command(&exe);
    envs(&mut listing);
    let out = listing
        .args(["list", "--json"])
        .output()
        .expect("second CLI list runs");
    assert!(
        out.status.success(),
        "list must succeed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let listed: serde_json::Value =
        serde_json::from_str(String::from_utf8_lossy(&out.stdout).trim())
            .expect("list prints JSON");
    let row = listed["workers"]
        .as_array()
        .expect("workers array")
        .iter()
        .find(|row| row["id"] == wid.as_str())
        .unwrap_or_else(|| {
            panic!("the host identity must keep its workers across invocations: {listed}")
        });
    assert_eq!(row["owner"], common::host_of_this_process());

    // The escape hatch never creates a socket.
    let bare = common::TempDir::new_in_tmp("hub-no-daemon");
    let mut local = common::binary_command(&exe);
    local
        .env("SWE_HUB_DIR", bare.path())
        .env("MINI_SWE_NO_DAEMON", "1")
        .env("OPENAI_API_KEY", "test-key-not-used-by-list")
        .env(
            "ENV_FILE",
            format!("{}/.env.does-not-exist", env!("CARGO_MANIFEST_DIR")),
        )
        .env(
            "MODELS_FILE",
            format!("{}/models.yaml", env!("CARGO_MANIFEST_DIR")),
        );
    let out = local
        .args(["list", "--json"])
        .output()
        .expect("local CLI call runs");
    assert!(
        out.status.success(),
        "escape-hatch call must succeed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        !mini_swe_mcp::hub::HubPaths::new(bare.path().to_path_buf())
            .socket()
            .exists(),
        "MINI_SWE_NO_DAEMON=1 creates no socket"
    );
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
        let _ = Command::new("git")
            .args(["init", "-q"])
            .current_dir(repo)
            .output();
        let _ = Command::new("git")
            .args(["commit", "-q", "--allow-empty", "-m", "seed"])
            .current_dir(repo)
            .env("GIT_AUTHOR_NAME", "t")
            .env("GIT_AUTHOR_EMAIL", "t@t")
            .env("GIT_COMMITTER_NAME", "t")
            .env("GIT_COMMITTER_EMAIL", "t@t")
            .output();
    }

    // A dead API base keeps the dispatch from an LLM call: the worker record
    // still names the repository the daemon resolved for this caller.
    let dispatch = |repo: &std::path::Path| -> String {
        let out = common::binary_command(&exe)
            .current_dir(repo)
            .args(["dispatch", "cwd probe", "--json"])
            .env("SWE_HUB_DIR", hub.path())
            .env("SWE_TEMP_DIR", &swe)
            .env("OPENAI_API_KEY", "test-key-not-used-by-dispatch")
            .env("OPENAI_API_BASE", "http://127.0.0.1:1")
            .env(
                "ENV_FILE",
                format!("{}/.env.does-not-exist", env!("CARGO_MANIFEST_DIR")),
            )
            .env(
                "MODELS_FILE",
                format!("{}/models.yaml", env!("CARGO_MANIFEST_DIR")),
            )
            .output()
            .expect("dispatch runs");
        assert!(
            out.status.success(),
            "dispatch must be accepted: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let payload: serde_json::Value =
            serde_json::from_str(String::from_utf8_lossy(&out.stdout).trim())
                .expect("dispatch prints JSON");
        payload["worker_id"]
            .as_str()
            .expect("dispatch names the worker")
            .to_string()
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
    let _scratch = common::TempDir::new_in_tmp("iso-hub-2");
    let pool = WorkerPool::with_scratch(
        4,
        "http://localhost:1".to_string(),
        "test-key".to_string(),
        mini_swe_mcp::worktree::ScratchRoot::new(_scratch.path()),
    )
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
    let socket = mini_swe_mcp::hub::HubPaths::new(dir.to_path_buf()).socket();
    wait_for_socket(&socket).await;

    // Agent B: an ordinary orchestrator, identified by its hello.
    let mut b = Client::connect(&socket).await;
    b.initialize("other-host").await;
    b.notify("hub/hello", serde_json::json!({"agent_id": "agent-b"}))
        .await;
    for arguments in [
        serde_json::json!({"action": "steer", "worker_id": "h3-hub-worker", "message": "stop"}),
        serde_json::json!({"action": "kill", "worker_id": "h3-hub-worker"}),
        serde_json::json!({"action": "collect", "worker_id": "h3-hub-worker"}),
        serde_json::json!({"action": "status", "worker_id": "h3-hub-worker"}),
        serde_json::json!({"action": "logs", "worker_id": "h3-hub-worker"}),
        serde_json::json!({"action": "watch", "worker_id": "h3-hub-worker", "timeout_secs": 0}),
    ] {
        let action = arguments["action"].as_str().expect("action").to_string();
        let error = b
            .worker(arguments)
            .await
            .expect_err("another agent's worker must be refused");
        assert_eq!(
            error, "worker h3-hub-worker belongs to agent agent-a",
            "'{action}' must name the owning agent: {error}"
        );
    }
    // The owner may act on it, and the CLI's stable identity is `cli` whatever
    // its connection id (H-3), so it never sees another agent's worker either.
    let mut a = Client::connect(&socket).await;
    a.initialize("other-host").await;
    a.notify("hub/hello", serde_json::json!({"agent_id": "agent-a"}))
        .await;
    let steered = a
        .worker(
            serde_json::json!({"action": "steer", "worker_id": "h3-hub-worker", "message": "go"}),
        )
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
    assert!(
        !ids.contains(&"h3-hub-worker"),
        "the CLI must not list it: {listed}"
    );
    let error = cli
        .worker(serde_json::json!({"action": "list", "scope": "all"}))
        .await
        .expect_err("scope=all requires admin");
    assert!(error.contains("admin"), "{error}");

    // The operator's `--admin`: the same connection, with the override in hello.
    let mut operator = Client::connect(&socket).await;
    operator
        .initialize(mini_swe_mcp::mcp::CLI_CLIENT_NAME)
        .await;
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
        assert!(
            out.status.success(),
            "git {args:?}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    let root = scratch_dir();
    let hub = root.join("hub");
    let swe = root.join("swe");
    let repo = root.join("repo");
    for dir in [&hub, &swe, &repo] {
        std::fs::create_dir_all(dir).expect("create scratch dir");
    }
    let mut dead = Command::new("true")
        .spawn()
        .expect("spawn short-lived owner");
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
    git(
        &repo,
        &[
            "worktree",
            "add",
            "-b",
            &branch,
            &checkout.to_string_lossy(),
            "HEAD",
        ],
    );
    std::fs::write(checkout.join("dirty.txt"), "unsaved\n").expect("dirty the checkout");
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock before epoch")
        .as_secs();
    let row = serde_json::json!({
        "id": wid, "pid": dead_pid, "task": "t", "model": "ninja", "status": "running",
        "step": 3, "max_turns": 10, "last_command": "cargo test",
        "started_at": now - 60, "updated_at": now - 60,
        // Real rows name their repository; without it the daemon's own
        // registry readers look for the branch in the wrong repo.
        "repo_path": repo,
    });
    std::fs::create_dir_all(swe.join("swe-registry")).expect("create the registry");
    std::fs::write(
        swe.join("swe-registry").join(format!("{wid}.json")),
        row.to_string(),
    )
    .expect("write the orphan row");
    let history = serde_json::json!({"worker": wid});
    std::fs::write(
        swe.join(format!("swe-wt-{wid}.history.json")),
        history.to_string(),
    )
    .expect("write the history file");

    let mut daemon = common::binary_command(&common::binary_path())
        .arg("daemon")
        .env("SWE_HUB_DIR", &hub)
        .env("SWE_TEMP_DIR", &swe)
        .env("TMPDIR", &swe)
        .env("HUB_IDLE_SECS", "1")
        // Auto-resume off, so the row keeps the status recovery gave it.
        .env("HUB_AUTO_RESUME", "0")
        .env("ENV_FILE", root.join("absent.env"))
        .env(
            "MODELS_FILE",
            format!("{}/models.yaml", env!("CARGO_MANIFEST_DIR")),
        )
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn recovery daemon");
    for _ in 0..100 {
        if std::os::unix::net::UnixStream::connect(
            mini_swe_mcp::hub::HubPaths::new(hub.to_path_buf()).socket(),
        )
        .is_ok()
        {
            break;
        }
        assert!(
            daemon.try_wait().expect("probe daemon").is_none(),
            "daemon exited early"
        );
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    assert!(
        mini_swe_mcp::hub::HubPaths::new(hub.to_path_buf())
            .socket()
            .exists(),
        "daemon must listen after recovery"
    );
    let entry: WorkerRegistryEntry = serde_json::from_slice(
        &std::fs::read(swe.join("swe-registry").join(format!("{wid}.json")))
            .expect("the orphan row survives recovery"),
    )
    .expect("registry JSON");
    // Interrupted, not failed: the worker stopped because the hub did, so it is
    // terminal for listing but continuable with `steer`.
    assert_eq!(
        entry.status,
        RegistryStatus::Interrupted,
        "the orphan row must be interrupted"
    );
    let expected = format!("hub restarted; work salvaged on branch worker-{wid}");
    assert_eq!(
        entry.last_command, expected,
        "the owner must see the salvage branch"
    );
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
    assert!(
        daemon.try_wait().expect("probe exit").is_some(),
        "daemon must idle out"
    );
    let log = std::fs::read_to_string(hub.join("hub.log")).expect("read recovery log");
    assert!(log.contains("recovered 1 orphaned workers"), "{log}");
    let _ = std::fs::remove_dir_all(&root);
}

/// SIGKILL bypasses the worker's exit path: only checkpoint history makes the
/// salvaged worker revisable when the next daemon starts.
#[tokio::test]
async fn a_checkpointed_worker_survives_hub_sigkill_and_revision() {
    use std::process::Stdio;

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
    let llm = tokio::spawn(serve_checkpoint_revision_script(listener));
    let command = || {
        let mut cmd = tokio::process::Command::new(common::binary_path());
        cmd.env("SWE_HUB_DIR", &hub)
            .env("SWE_TEMP_DIR", &swe)
            .env("TMPDIR", &swe)
            .env_remove("MINI_SWE_NO_DAEMON")
            .env("OPENAI_API_BASE", &api_base)
            .env("OPENAI_API_KEY", "test-key")
            .env("ENV_FILE", root.path().join("absent.env"))
            .env(
                "MODELS_FILE",
                format!("{}/models.yaml", env!("CARGO_MANIFEST_DIR")),
            )
            .env("HUB_IDLE_SECS", "60");
        common::scrub_identity_env(cmd.as_std_mut());
        cmd
    };
    let mut first = command()
        .arg("daemon")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .unwrap();
    wait_for_socket(&mini_swe_mcp::hub::HubPaths::new(hub.to_path_buf()).socket()).await;
    let out = command()
        .args([
            "dispatch",
            "checkpoint recovery",
            "--repo",
            repo.to_str().unwrap(),
            "--model",
            "test-model",
            "--max-turns",
            "30",
            "--verify",
            "",
            "--json",
        ])
        .output()
        .await
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let dispatched: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let wid = dispatched["worker_id"].as_str().unwrap();
    // The durable store is the append-only log, written one line per message.
    let history_path = swe.join(format!("swe-wt-{wid}.history.jsonl"));
    // The log is seeded with the opening messages and then grows one line per
    // message, so wait for the checkpoint's worth of exchanges rather than for
    // the file to appear.
    for _ in 0..400 {
        let lines = std::fs::read_to_string(&history_path)
            .map(|raw| raw.lines().filter(|l| !l.trim().is_empty()).count())
            .unwrap_or(0);
        if lines >= 40 {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    }
    // Read the log directly: it lives under this test's `SWE_TEMP_DIR`, which
    // the test process itself does not share with the daemon that wrote it.
    let raw =
        std::fs::read_to_string(&history_path).expect("checkpoint must save history before exit");
    let mut lines = raw.lines().filter(|l| !l.trim().is_empty());
    let mut checkpoint: mini_swe_mcp::pool::WorkerHistory =
        serde_json::from_str(lines.next().expect("the log has a metadata line")).unwrap();
    for line in lines {
        checkpoint
            .messages
            .push(serde_json::from_str(line).expect("one message per line"));
    }
    assert!(mini_swe_mcp::pool::is_replayable(&checkpoint.messages));
    assert!(
        checkpoint.messages.len() >= 40,
        "must retain exchanges before turn 20"
    );
    assert_eq!(
        checkpoint.owner.as_deref(),
        Some(common::host_of_this_process().as_str())
    );
    assert_eq!(checkpoint.verify, None);
    first.kill().await.unwrap();
    first.wait().await.unwrap();

    let mut second = command()
        .arg("daemon")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .env("HUB_AUTO_RESUME", "0")
        .kill_on_drop(true)
        .spawn()
        .unwrap();
    wait_for_socket(&mini_swe_mcp::hub::HubPaths::new(hub.to_path_buf()).socket()).await;
    let recovered = wait_for_registry_status(
        &swe.join("swe-registry"),
        wid,
        mini_swe_mcp::pool::RegistryStatus::Interrupted,
    )
    .await;
    // Interrupted, not failed: the worker stopped because the hub did, and its
    // conversation survived, so the steer below continues it.
    assert_eq!(
        recovered.status,
        mini_swe_mcp::pool::RegistryStatus::Interrupted
    );
    assert_eq!(
        recovered.last_command,
        format!("hub restarted; work salvaged on branch worker-{wid}")
    );
    let out = tokio::time::timeout(
        std::time::Duration::from_secs(15),
        command()
            .args([
                "steer",
                wid,
                "resume after crash",
                "--max-turns",
                "2",
                "--json",
            ])
            .output(),
    )
    .await
    .expect("steer must return")
    .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let watched = tokio::time::timeout(
        std::time::Duration::from_secs(15),
        command()
            .args(["watch", wid, "--timeout", "10", "--json"])
            .output(),
    )
    .await
    .expect("revision watch must finish")
    .unwrap();
    assert!(
        watched.status.success(),
        "{}",
        String::from_utf8_lossy(&watched.stderr)
    );
    assert!(String::from_utf8_lossy(&watched.stdout).contains(&checkpoint.branch));
    assert_eq!(
        common::git(&repo, &["show", &format!("{}:kept.txt", checkpoint.branch)]).trim(),
        "checkpoint"
    );
    second.kill().await.unwrap();
    second.wait().await.unwrap();
    tokio::time::timeout(std::time::Duration::from_secs(5), llm)
        .await
        .expect("LLM completed all turns")
        .unwrap();
    std::fs::remove_dir_all(&hub).unwrap();
}

/// Script the fake LLM for `a_checkpointed_worker_survives_hub_sigkill_and_revision`:
/// turn 19 writes the checkpoint marker, turn 20 is deliberately left unanswered
/// so the hub can be SIGKILLed mid-call, and turn 21 finishes the worker.
async fn serve_checkpoint_revision_script(listener: TcpListener) {
    for turn in 1..=21 {
        let (mut socket, _) = listener.accept().await.unwrap();
        // A SIGKILL can drop the connection mid-request: skip that turn and keep
        // accepting, or the restarted worker would find no LLM left.
        if common::fake_llm::read_request(&mut socket).await.is_none() {
            continue;
        }
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
        // A client killed mid-response is not an error: the next turn's
        // connection is the one the worker needs.
        let _ = socket.write_all(format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()
        ).as_bytes()).await;
    }
}

/// A hub killed mid-request closes its socket before the fake LLM has read the
/// request whole. The script must skip that connection and move on, so the retry
/// from the restarted worker still gets the closing turn.
#[tokio::test]
async fn the_revision_script_answers_after_a_connection_closed_mid_request() {
    use tokio::io::AsyncReadExt;
    use tokio::net::TcpStream;

    async fn ask(addr: std::net::SocketAddr) -> String {
        let mut stream = TcpStream::connect(addr).await.unwrap();
        stream
            .write_all(
                b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}",
            )
            .await
            .unwrap();
        let mut reply = String::new();
        stream.read_to_string(&mut reply).await.unwrap();
        reply
    }

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(serve_checkpoint_revision_script(listener));

    for turn in 1..=18 {
        assert!(ask(addr).await.contains(&format!("echo turn {turn}")));
    }
    assert!(ask(addr).await.contains("kept.txt"));

    // Turn 20's request is in flight when the hub dies: half a head, then the
    // socket drops.
    let mut torn = TcpStream::connect(addr).await.unwrap();
    torn.write_all(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n")
        .await
        .unwrap();
    drop(torn);

    // The resumed worker's retry is answered with the closing turn.
    assert!(
        ask(addr)
            .await
            .contains("COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")
    );
    server.abort();
}

fn event_worker(id: &str, owner: &str) -> mini_swe_mcp::pool::WorkerRecord {
    use mini_swe_mcp::pool::{LogBuffer, WorkerMetrics, WorkerRecord, WorkerState};
    WorkerRecord {
        id: id.to_string(),
        task: "event probe".to_string(),
        model: "test".to_string(),
        owner: owner.to_string(),
        state: WorkerState::Running {
            step: 1,
            last_command: "test".to_string(),
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

async fn next_event(client: &mut Client) -> serde_json::Value {
    let mut line = String::new();
    tokio::time::timeout(
        std::time::Duration::from_secs(3),
        client.reader.read_line(&mut line),
    )
    .await
    .expect("event arrives")
    .expect("read event");
    let event: serde_json::Value = serde_json::from_str(&line).expect("event JSON");
    assert_eq!(event["method"], "notifications/claude/channel");
    event
}

#[tokio::test]
async fn events_are_owner_scoped_and_replayed_after_hello() {
    use mini_swe_mcp::pool::{WorkerMetrics, WorkerState};
    let scratch = common::TempDir::new_in_tmp("hub-pool");
    let pool = WorkerPool::with_scratch(
        4,
        "http://localhost:1".to_string(),
        "test-key".to_string(),
        mini_swe_mcp::worktree::ScratchRoot::new(scratch.path()),
    );
    let _scratch = scratch;
    pool.__test_insert_worker(event_worker("h4-live", "h4-a"))
        .await;
    pool.__test_insert_worker(event_worker("h4-late", "h4-late-owner"))
        .await;
    let dir = scratch_dir();
    let daemon = HubServer::new(
        Arc::new(McpServer::new(pool.clone(), "test".to_string())),
        HubConfig::new(hub_paths_for_test(&dir), 60),
    );
    let task = tokio::spawn(async move { daemon.run().await });
    let socket = mini_swe_mcp::hub::HubPaths::new(dir.to_path_buf()).socket();
    wait_for_socket(&socket).await;
    let mut a = Client::connect(&socket).await;
    let mut b = Client::connect(&socket).await;
    a.request("hub/hello", serde_json::json!({"agent_id": "h4-a"}))
        .await;
    b.request("hub/hello", serde_json::json!({"agent_id": "h4-b"}))
        .await;
    pool.__test_set_worker_state(
        "h4-live",
        WorkerState::Paused {
            question: "continue?".to_string(),
            step: 1,
            paused_at: 0,
        },
    )
    .await;
    let event = next_event(&mut a).await;
    assert_eq!(event["params"]["meta"]["worker_id"], "h4-live");
    let mut line = String::new();
    assert!(
        tokio::time::timeout(
            std::time::Duration::from_millis(100),
            b.reader.read_line(&mut line)
        )
        .await
        .is_err()
    );
    pool.__test_set_worker_state(
        "h4-late",
        WorkerState::Failed {
            error: "late failure".to_string(),
            step: 2,
            failed_at: 1,
            metrics: WorkerMetrics::default(),
            revision: 0,
        },
    )
    .await;
    // Observe a later transition to establish that the watcher ran while the
    // late owner still had no connection.
    pool.__test_set_worker_state(
        "h4-live",
        WorkerState::Failed {
            error: "failure".to_string(),
            step: 2,
            failed_at: 1,
            metrics: WorkerMetrics::default(),
            revision: 0,
        },
    )
    .await;
    next_event(&mut a).await;
    let mut late = Client::connect(&socket).await;
    late.request(
        "hub/hello",
        serde_json::json!({"agent_id": "h4-late-owner"}),
    )
    .await;
    let queued = next_event(&mut late).await;
    assert_eq!(queued["params"]["meta"]["worker_id"], "h4-late");
    assert_eq!(queued["params"]["meta"]["event"], "failed");
    let mut admin = Client::connect(&socket).await;
    admin
        .request("hub/hello", serde_json::json!({"admin": true}))
        .await;
    // An admin sees every agent's events, including real workers in this
    // host's registry, so look for this test's event among the first few.
    let mut admin_saw = Vec::new();
    for _ in 0..20 {
        let event = next_event(&mut admin).await;
        let id = event["params"]["meta"]["worker_id"].clone();
        admin_saw.push(id.clone());
        if id == "h4-late" {
            break;
        }
    }
    assert_eq!(
        admin_saw.last(),
        Some(&serde_json::json!("h4-late")),
        "{admin_saw:?}"
    );
    drop(a);
    drop(b);
    drop(late);
    drop(admin);
    task.abort();
    let _ = task.await;
    let _ = std::fs::remove_dir_all(dir);
}

#[tokio::test]
async fn shutdown_refuses_running_and_paused_workers_then_stops_when_idle() {
    use mini_swe_mcp::pool::{WorkerMetrics, WorkerState};
    let scratch = common::TempDir::new_in_tmp("hub-pool");
    let pool = WorkerPool::with_scratch(
        4,
        "http://localhost:1".to_string(),
        "test-key".to_string(),
        mini_swe_mcp::worktree::ScratchRoot::new(scratch.path()),
    );
    let _scratch = scratch;
    pool.__test_insert_worker(event_worker("h4-busy", "h4-owner"))
        .await;
    let dir = scratch_dir();
    let daemon = HubServer::new(
        Arc::new(McpServer::new(pool.clone(), "test".to_string())),
        HubConfig::new(hub_paths_for_test(&dir), 60),
    );
    let task = tokio::spawn(async move { daemon.run().await });
    let socket = mini_swe_mcp::hub::HubPaths::new(dir.to_path_buf()).socket();
    wait_for_socket(&socket).await;
    let mut client = Client::connect(&socket).await;
    let hello = client
        .request("hub/hello", serde_json::json!({"version": "99.0.0"}))
        .await;
    assert_eq!(hello["result"]["version"], env!("CARGO_PKG_VERSION"));
    assert_eq!(hello["result"]["build"]["id"], env!("MINI_SWE_BUILD_ID"));
    assert_eq!(
        hello["result"]["build"]["ts"],
        env!("MINI_SWE_BUILD_TS").parse::<u64>().unwrap()
    );
    assert_eq!(hello["result"]["busy"], true);
    assert_eq!(
        client.request("hub/shutdown", serde_json::json!({})).await["error"]["message"],
        "Hub is busy"
    );
    pool.__test_set_worker_state(
        "h4-busy",
        WorkerState::Paused {
            question: "wait".to_string(),
            step: 1,
            paused_at: 0,
        },
    )
    .await;
    assert_eq!(
        client.request("hub/shutdown", serde_json::json!({})).await["error"]["message"],
        "Hub is busy"
    );
    pool.__test_set_worker_state(
        "h4-busy",
        WorkerState::Failed {
            error: "done".to_string(),
            step: 1,
            failed_at: 1,
            metrics: WorkerMetrics::default(),
            revision: 0,
        },
    )
    .await;
    assert_eq!(
        client.request("hub/shutdown", serde_json::json!({})).await["result"]["busy"],
        false
    );
    assert!(
        tokio::time::timeout(std::time::Duration::from_secs(3), task)
            .await
            .unwrap()
            .unwrap()
            .unwrap()
    );
    assert!(!socket.exists());
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn newer_cli_replaces_an_idle_daemon() {
    let exe = common::binary_path();
    let hub = common::TempDir::new_in_tmp("hub-version");
    let _reaper = DaemonReaper(hub.path().to_path_buf());
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(hub.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
    let invoke = |fake: bool| {
        let mut cmd = common::binary_command(&exe);
        cmd.args(["list", "--json"])
            .env("SWE_HUB_DIR", hub.path())
            .env("SWE_TEMP_DIR", hub.subdir("swe"))
            .env("ENV_FILE", "/nonexistent-mini-swe-env");
        if fake {
            cmd.env("MINI_SWE_FAKE_VERSION", "99.0.0");
        }
        let output = cmd.output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    };
    invoke(false);
    invoke(true);
    let log = std::fs::read_to_string(hub.path().join("hub.log")).unwrap();
    assert_eq!(
        log.lines()
            .filter(|line| line.ends_with(" listening"))
            .count(),
        2,
        "{log}"
    );
    assert!(log.lines().any(|line| line.ends_with(" stopped")), "{log}");
}

/// A newer client that asked the idle hub to stop waits for the predecessor to
/// drop `hub.lock`, not merely for `hub.sock` to vanish, before it spawns the
/// replacement daemon. The predecessor here removes its socket but keeps the
/// lock for half a second.
#[test]
fn a_replacing_client_waits_for_the_predecessor_to_release_the_lock() {
    use std::io::{BufRead, BufReader, Write};
    use std::os::unix::net::UnixListener;

    let exe = common::binary_path();
    let hub = common::TempDir::new_in_tmp("hub-lockwait");
    let _reaper = DaemonReaper(hub.path().to_path_buf());
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(hub.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
    let socket = mini_swe_mcp::hub::HubPaths::new(hub.path().to_path_buf()).socket();
    let lock_path = hub.path().join("hub.lock");
    let listener = UnixListener::bind(&socket).expect("bind the fake predecessor");
    let fake = std::thread::spawn(move || {
        let lock = hold_hub_lock(&lock_path);
        {
            let (stream, _) = listener.accept().expect("accept the client");
            let mut writer = stream.try_clone().expect("clone the stream");
            for line in BufReader::new(stream).lines() {
                let Ok(line) = line else { break };
                let frame: serde_json::Value =
                    serde_json::from_str(&line).expect("client sends JSON");
                let method = frame["method"].as_str().unwrap_or_default().to_string();
                let Some(id) = frame.get("id") else { continue };
                // An older release with an idle pool: the client supersedes it.
                let reply = match &method[..] {
                    "hub/hello" => serde_json::json!({"jsonrpc": "2.0", "id": id,
                        "result": {"version": "0.0.9", "busy": false}}),
                    "hub/shutdown" => serde_json::json!({"jsonrpc": "2.0", "id": id,
                        "result": {"busy": false}}),
                    _ => serde_json::json!({"jsonrpc": "2.0", "id": id, "result": {}}),
                };
                writeln!(writer, "{reply}").expect("reply to the client");
                if method == "hub/shutdown" {
                    break;
                }
            }
        }
        // Teardown: the socket goes first, the lock lags behind.
        std::fs::remove_file(&socket).expect("remove the predecessor socket");
        std::thread::sleep(std::time::Duration::from_millis(500));
        drop(lock);
    });

    let mut cmd = common::binary_command(&exe);
    cmd.args(["list", "--json"])
        .env("SWE_HUB_DIR", hub.path())
        .env("SWE_TEMP_DIR", hub.subdir("swe"))
        .env("ENV_FILE", "/nonexistent-mini-swe-env");
    let out = cmd.output().expect("run the CLI");
    fake.join().expect("fake predecessor thread");
    assert!(
        out.status.success(),
        "the client must outwait the predecessor: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let listed: serde_json::Value =
        serde_json::from_str(String::from_utf8_lossy(&out.stdout).trim())
            .expect("list prints JSON");
    assert!(listed["workers"].is_array(), "{listed}");
    let log = std::fs::read_to_string(hub.path().join("hub.log")).unwrap_or_default();
    assert_eq!(
        log.lines()
            .filter(|line| line.ends_with(" listening"))
            .count(),
        1,
        "{log}"
    );
}

/// Start a hub with `list --json`, then call it again as a client whose build
/// clock is shifted by `build_skew_nanos` from the daemon's own; returns the hub
/// log. The shift is an hour: the stamps only have to be ordered, not dated.
async fn run_list_twice_with_client_build_skew(build_skew_nanos: i128) -> String {
    use tokio::process::Command;

    async fn invoke(exe: &Path, hub: &Path, swe: &Path, fake_build_ts: Option<String>) {
        let mut cmd = Command::new(exe);
        cmd.args(["list", "--json"])
            .env("SWE_HUB_DIR", hub)
            .env("SWE_TEMP_DIR", swe)
            .env("ENV_FILE", "/nonexistent-mini-swe-env");
        if let Some(ts) = fake_build_ts {
            cmd.env("MINI_SWE_FAKE_BUILD_TS", ts);
        }
        common::scrub_identity_env(cmd.as_std_mut());
        let output = cmd.output().await.unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let exe = common::binary_path();
    let hub = common::TempDir::new_in_tmp("hub-build");
    let _reaper = DaemonReaper(hub.path().to_path_buf());
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(hub.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
    let swe = hub.subdir("swe");
    invoke(&exe, hub.path(), &swe, None).await;

    // The daemon's own clock is the reference, whatever this test binary was
    // itself compiled with.
    let mut probe =
        Client::connect(&mini_swe_mcp::hub::HubPaths::new(hub.path().to_path_buf()).socket()).await;
    let hello = probe
        .request("hub/hello", serde_json::json!({"agent_id": "build-probe"}))
        .await;
    let daemon_ts = hello["result"]["build"]["ts"]
        .as_u64()
        .expect("the daemon reports its build clock") as i128;
    drop(probe);

    let fake = daemon_ts + build_skew_nanos;
    invoke(&exe, hub.path(), &swe, Some(fake.to_string())).await;
    std::fs::read_to_string(hub.path().join("hub.log")).unwrap()
}

/// A rebuild moves no release version, so the build clock is the only thing
/// that tells a client the idle daemon it dialed was built before it.
#[tokio::test]
async fn a_rebuilt_client_replaces_an_idle_daemon_at_the_same_version() {
    let log = run_list_twice_with_client_build_skew(3_600_000_000_000).await;
    assert_eq!(
        log.lines()
            .filter(|line| line.ends_with(" listening"))
            .count(),
        2,
        "{log}"
    );
    assert!(log.lines().any(|line| line.ends_with(" stopped")), "{log}");
}

/// The reverse skew must not cost a healthy daemon its workers.
#[tokio::test]
async fn an_older_client_build_keeps_the_running_daemon() {
    let log = run_list_twice_with_client_build_skew(-3_600_000_000_000).await;
    assert_eq!(
        log.lines()
            .filter(|line| line.ends_with(" listening"))
            .count(),
        1,
        "{log}"
    );
    assert!(!log.lines().any(|line| line.ends_with(" stopped")), "{log}");
}

#[tokio::test]
async fn newer_clients_warn_once_and_keep_a_busy_daemon() {
    let dir = scratch_dir();
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700)).unwrap();
    let scratch = common::TempDir::new_in_tmp("hub-pool");
    let pool = WorkerPool::with_scratch(
        4,
        "http://localhost:1".to_string(),
        "test-key".to_string(),
        mini_swe_mcp::worktree::ScratchRoot::new(scratch.path()),
    );
    let _scratch = scratch;
    pool.__test_insert_worker(event_worker("h4-version-busy", "owner"))
        .await;
    let daemon = HubServer::new(
        Arc::new(McpServer::new(pool, "test".to_string())),
        HubConfig::new(hub_paths_for_test(&dir), 60),
    );
    let task = tokio::spawn(async move { daemon.run().await });
    wait_for_socket(&mini_swe_mcp::hub::HubPaths::new(dir.to_path_buf()).socket()).await;
    let mut command = tokio::process::Command::new(common::binary_path());
    command
        .args(["list", "--json"])
        .env("SWE_HUB_DIR", &dir)
        .env("MINI_SWE_FAKE_VERSION", "99.0.0")
        .env("ENV_FILE", "/nonexistent-mini-swe-env");
    common::scrub_identity_env(command.as_std_mut());
    let output = command.output().await.unwrap();
    assert!(output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert_eq!(
        stderr
            .lines()
            .filter(|line| line.contains("is newer than hub"))
            .count(),
        1,
        "{stderr}"
    );
    assert!(
        mini_swe_mcp::hub::HubPaths::new(dir.to_path_buf())
            .socket()
            .exists()
    );

    let mut proxy = tokio::process::Command::new(common::binary_path());
    proxy
        .arg("--stdio")
        .env("SWE_HUB_DIR", &dir)
        .env("MINI_SWE_FAKE_VERSION", "99.0.0")
        .env("ENV_FILE", "/nonexistent-mini-swe-env")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true);
    common::scrub_identity_env(proxy.as_std_mut());
    let mut child = proxy.spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());
    stdin
        .write_all(b"{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"ping\"}\n")
        .await
        .unwrap();
    let mut line = String::new();
    tokio::time::timeout(
        std::time::Duration::from_secs(3),
        stdout.read_line(&mut line),
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&line).unwrap()["result"],
        serde_json::json!({})
    );
    drop(stdin);
    let output = child.wait_with_output().await.unwrap();
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert_eq!(
        stderr
            .lines()
            .filter(|line| line.contains("is newer than hub"))
            .count(),
        1,
        "{stderr}"
    );
    task.abort();
    let _ = task.await;
    let _ = std::fs::remove_dir_all(dir);
}

#[tokio::test]
async fn a_blocked_steer_does_not_delay_shutdowns_answer() {
    use mini_swe_mcp::pool::WorkerState;
    let scratch = common::TempDir::new_in_tmp("hub-pool");
    let pool = WorkerPool::with_scratch(
        4,
        "http://localhost:1".to_string(),
        "test-key".to_string(),
        mini_swe_mcp::worktree::ScratchRoot::new(scratch.path()),
    );
    let _scratch = scratch;
    pool.__test_insert_worker(event_worker("h4-blocked", "h4-owner"))
        .await;
    let dir = scratch_dir();
    let daemon = HubServer::new(
        Arc::new(McpServer::new(pool.clone(), "test".to_string())),
        HubConfig::new(hub_paths_for_test(&dir), 60),
    );
    let task = tokio::spawn(async move { daemon.run().await });
    let socket = mini_swe_mcp::hub::HubPaths::new(dir.to_path_buf()).socket();
    wait_for_socket(&socket).await;
    let mut waiter = Client::connect(&socket).await;
    waiter
        .notify("hub/hello", serde_json::json!({"agent_id": "h4-owner"}))
        .await;
    let steered = tokio::time::timeout(
        std::time::Duration::from_secs(3),
        waiter.worker(
            serde_json::json!({"action": "steer", "worker_id": "h4-blocked", "message": "go"}),
        ),
    )
    .await
    .expect("steer returns immediately")
    .expect("steer succeeds");
    assert_eq!(steered["status"], "steered");
    let mut closer = Client::connect(&socket).await;
    closer
        .notify("hub/hello", serde_json::json!({"agent_id": "h4-owner"}))
        .await;
    let reply = tokio::time::timeout(
        std::time::Duration::from_secs(3),
        closer.request("hub/shutdown", serde_json::json!({})),
    )
    .await
    .expect("shutdown answers promptly");
    assert_eq!(reply["error"]["message"], "Hub is busy");
    pool.__test_set_worker_state(
        "h4-blocked",
        WorkerState::Failed {
            error: "done".to_string(),
            step: 1,
            failed_at: 1,
            metrics: mini_swe_mcp::pool::WorkerMetrics::default(),
            revision: 0,
        },
    )
    .await;
    task.abort();
    let _ = task.await;
    let _ = std::fs::remove_dir_all(dir);
}

/// A client newer than the running hub still works when that hub predates the
/// version handshake: `hub/hello` as a request is answered "Method not found",
/// and the client falls back to the notification form instead of failing.
#[test]
fn a_client_degrades_gracefully_against_a_pre_handshake_hub() {
    use std::io::{BufRead, BufReader, Write};
    use std::os::unix::net::UnixListener;

    let hub = common::TempDir::new_in_tmp("hub-legacy");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(hub.path(), std::fs::Permissions::from_mode(0o700))
            .expect("restrict the hub dir to 0700");
    }
    let listener =
        UnixListener::bind(mini_swe_mcp::hub::HubPaths::new(hub.path().to_path_buf()).socket())
            .expect("bind the fake hub");
    let fake = std::thread::spawn(move || {
        let (stream, _) = listener.accept().expect("accept the client");
        let mut writer = stream.try_clone().expect("clone the stream");
        let mut methods = Vec::new();
        for line in BufReader::new(stream).lines() {
            let Ok(line) = line else { break };
            let frame: serde_json::Value = serde_json::from_str(&line).expect("client sends JSON");
            let method = frame["method"].as_str().unwrap_or_default().to_string();
            let reply = match (&method[..], frame.get("id")) {
                (_, None) => None,
                ("hub/hello", Some(id)) => Some(serde_json::json!({"jsonrpc": "2.0", "id": id,
                    "error": {"code": -32601, "message": "Method not found: hub/hello"}})),
                ("tools/call", Some(id)) => Some(serde_json::json!({"jsonrpc": "2.0", "id": id,
                    "result": {"content": [{"type": "text", "text": "{\"workers\":[]}"}]}})),
                (_, Some(id)) => {
                    Some(serde_json::json!({"jsonrpc": "2.0", "id": id, "result": {}}))
                }
            };
            methods.push((method.clone(), frame.get("id").is_some()));
            if let Some(reply) = reply {
                writeln!(writer, "{reply}").expect("reply to the client");
            }
            if method == "tools/call" {
                break;
            }
        }
        methods
    });

    let out = common::binary_command(&common::binary_path())
        .args(["list", "--json"])
        .env("SWE_HUB_DIR", hub.path())
        .env("OPENAI_API_KEY", "test-key-not-used")
        .env("ENV_FILE", hub.path().join("absent.env"))
        .output()
        .expect("run the CLI");
    assert!(
        out.status.success(),
        "the CLI must work against an old hub: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(String::from_utf8_lossy(&out.stderr).contains("predates the version handshake"));
    let methods = fake.join().expect("fake hub thread");
    assert!(
        methods.contains(&("hub/hello".to_string(), false)),
        "the identity must still be announced as a notification: {methods:?}"
    );
}

/// A hub whose `hub/hello` reply carries no `build` predates build ids, so the
/// client decides on the release alone: an equal release is left alone, an
/// older one is still asked to step aside.
#[test]
fn a_hub_without_a_build_field_falls_back_to_the_release() {
    /// Run `list --json` against a fake hub answering `hub/hello` with
    /// `version` and no `build`; return the methods that hub received and the
    /// params of the client's hello.
    fn list_against_fake_hub(version: &str) -> (Vec<(String, bool)>, serde_json::Value) {
        use std::io::{BufRead, BufReader, Write};
        use std::os::unix::net::UnixListener;

        let hub = common::TempDir::new_in_tmp("hub-nobuild");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(hub.path(), std::fs::Permissions::from_mode(0o700))
                .expect("restrict the hub dir to 0700");
        }
        let version = version.to_string();
        let listener =
            UnixListener::bind(mini_swe_mcp::hub::HubPaths::new(hub.path().to_path_buf()).socket())
                .expect("bind the fake hub");
        let fake = std::thread::spawn(move || {
            let (stream, _) = listener.accept().expect("accept the client");
            let mut writer = stream.try_clone().expect("clone the stream");
            let mut methods = Vec::new();
            let mut hello_params = serde_json::Value::Null;
            for line in BufReader::new(stream).lines() {
                let Ok(line) = line else { break };
                let frame: serde_json::Value =
                    serde_json::from_str(&line).expect("client sends JSON");
                let method = frame["method"].as_str().unwrap_or_default().to_string();
                methods.push((method.clone(), frame.get("id").is_some()));
                if method == "hub/hello" {
                    hello_params = frame["params"].clone();
                }
                let Some(id) = frame.get("id") else { continue };
                // No `build`: this reply is shaped like a pre-build-id hub's.
                let reply = match &method[..] {
                    "hub/hello" => serde_json::json!({"jsonrpc": "2.0", "id": id,
                        "result": {"version": version.as_str(), "busy": false}}),
                    // Refuse the shutdown as a busy hub would, so the client
                    // keeps going instead of waiting for a teardown it cannot see.
                    "hub/shutdown" => serde_json::json!({"jsonrpc": "2.0", "id": id,
                        "error": {"code": -32000, "message": "Hub is busy"}}),
                    "tools/call" => serde_json::json!({"jsonrpc": "2.0", "id": id,
                        "result": {"content": [{"type": "text", "text": "{\"workers\":[]}"}]}}),
                    _ => serde_json::json!({"jsonrpc": "2.0", "id": id, "result": {}}),
                };
                writeln!(writer, "{reply}").expect("reply to the client");
                if method == "tools/call" {
                    break;
                }
            }
            (methods, hello_params)
        });

        let out = common::binary_command(&common::binary_path())
            .args(["list", "--json"])
            .env("SWE_HUB_DIR", hub.path())
            .env("OPENAI_API_KEY", "test-key-not-used")
            .env("ENV_FILE", hub.path().join("absent.env"))
            .output()
            .expect("run the CLI");
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        fake.join().expect("fake hub thread")
    }

    // A reply without a build is still asked for the release of the client,
    // which must announce its own build in `hub/hello` either way.
    let (methods, hello) = list_against_fake_hub(env!("CARGO_PKG_VERSION"));
    assert_eq!(hello["version"], env!("CARGO_PKG_VERSION"));
    assert!(
        hello["build"]["id"]
            .as_str()
            .is_some_and(|id| !id.is_empty()),
        "{hello}"
    );
    assert!(hello["build"]["ts"].is_u64(), "{hello}");
    assert!(
        !methods.contains(&("hub/shutdown".to_string(), true)),
        "{methods:?}"
    );

    // An older release still loses the argument on the semver fallback.
    let (methods, _) = list_against_fake_hub("0.0.9");
    assert!(
        methods.contains(&("hub/shutdown".to_string(), true)),
        "{methods:?}"
    );
}

/// A hub directory deep enough that `<dir>/hub.sock` exceeds the Unix socket
/// path limit (108 bytes) still works: the socket moves to a short private
/// fallback directory that the daemon and the client both derive.
#[test]
fn a_deep_hub_dir_still_gets_a_working_socket() {
    let base = common::TempDir::new_in_tmp("hub-deep");
    let deep = base
        .path()
        .join("a-rather-long-directory-name-to-push-the-socket-path")
        .join("past-the-unix-socket-path-limit-of-one-hundred-and-eight-bytes");
    std::fs::create_dir_all(&deep).expect("create the deep hub dir");
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&deep, std::fs::Permissions::from_mode(0o700)).unwrap();
    }
    assert!(
        deep.join("hub.sock").as_os_str().len() > 108,
        "the test path must be too long"
    );
    let out = common::binary_command(&common::binary_path())
        .args(["list", "--json"])
        .env("SWE_HUB_DIR", &deep)
        .env("SWE_TEMP_DIR", base.subdir("swe"))
        .env("HUB_IDLE_SECS", "1")
        .env("ENV_FILE", "/nonexistent-mini-swe-env")
        .output()
        .expect("run the CLI");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
}
