//! Planned handover: a busy daemon stops at the first quiet moment.
//!
//! A rebuilt binary could only replace an idle hub, so a daemon with workers
//! running all day never picked up the newer build. `hub/handover` lets a newer
//! client ask a busy daemon to step aside: the daemon keeps serving until no
//! worker is executing a command and no heavy admission permit is held, then
//! takes the graceful shutdown path (checkpoint, `Interrupted`, auto-continue
//! by the next daemon). A deadline bounds a daemon that never goes quiet, and
//! every client that outlives the cut reconnects instead of exiting.
//!
//! The daemon runs in-process on a scratch hub directory, exactly as
//! `tests/hub_test.rs` starts it, so no test needs an LLM: the only worker state
//! is a synthetic record and a held command mark.

mod common;

use mini_swe_mcp::hub::{HubConfig, HubPaths, HubServer};
use mini_swe_mcp::mcp::McpServer;
use mini_swe_mcp::pool::{WorkerPool, WorkerState};
use serde_json::{Value, json};
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;

/// A build clock far in the future, so this client supersedes any daemon.
const NEWER_BUILD_TS: u64 = 4_000_000_000_000_000_000;

/// One JSON-RPC client over the hub socket.
struct Client {
    reader: BufReader<UnixStream>,
    next_id: u64,
}

impl Client {
    /// Dial `socket`, announcing the identity the way a thin client does.
    async fn connect(socket: &std::path::Path) -> Self {
        let stream = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if let Ok(stream) = UnixStream::connect(socket).await {
                    return stream;
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .expect("hub socket came up");
        let mut client = Self {
            reader: BufReader::new(stream),
            next_id: 1,
        };
        client.hello().await;
        client
    }

    /// The `hub/hello` handshake, as a notification: it is what makes this
    /// connection the newer client the daemon is allowed to hand over to.
    async fn hello(&mut self) {
        self.notify(
            "hub/hello",
            json!({"agent_id": "handover-test", "version": "99.0.0",
                   "build": {"id": "test-newer", "ts": NEWER_BUILD_TS}}),
        )
        .await;
    }

    async fn notify(&mut self, method: &str, params: Value) {
        self.get_mut()
            .write_all(
                format!(
                    "{}\n",
                    json!({"jsonrpc": "2.0", "method": method, "params": params})
                )
                .as_bytes(),
            )
            .await
            .expect("write notification");
        self.get_mut().flush().await.expect("flush notification");
    }

    fn get_mut(&mut self) -> &mut UnixStream {
        self.reader.get_mut()
    }

    /// Send one request and return its whole reply envelope.
    async fn request(&mut self, method: &str, params: Value) -> Value {
        let id = self.next_id;
        self.next_id += 1;
        self.get_mut()
            .write_all(
                format!(
                    "{}\n",
                    json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params})
                )
                .as_bytes(),
            )
            .await
            .expect("write request");
        self.get_mut().flush().await.expect("flush request");
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let mut line = String::new();
                self.reader.read_line(&mut line).await.expect("read reply");
                assert!(!line.is_empty(), "hub closed before replying to {method}");
                let frame: Value = serde_json::from_str(line.trim()).expect("reply is JSON");
                if frame["id"] == json!(id) {
                    return frame;
                }
            }
        })
        .await
        .expect("hub replied in time")
    }

    /// `hub/handover` with the deadline this test wants.
    async fn handover(&mut self, deadline_secs: u64) -> Value {
        self.request(
            "hub/handover",
            json!({"version": "99.0.0", "deadline_secs": deadline_secs}),
        )
        .await
    }
}

/// A synthetic live worker, so the pool counts as busy without an LLM behind it.
fn running_worker(id: &str) -> mini_swe_mcp::pool::WorkerRecord {
    use mini_swe_mcp::pool::{LogBuffer, WorkerMetrics, WorkerRecord};
    WorkerRecord {
        id: id.to_string(),
        task: "handover probe".to_string(),
        model: "test".to_string(),
        owner: "handover-test".to_string(),
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
    }
}

/// A daemon on a scratch hub directory, plus the pool it serves.
fn daemon_on(hub: &common::TempDir, pool: WorkerPool) -> (HubPaths, HubServer) {
    let paths = HubPaths::new(hub.path().to_path_buf());
    let daemon = HubServer::new(
        Arc::new(McpServer::new(pool, "test".to_string())),
        HubConfig::new(paths.clone(), 60),
    );
    (paths, daemon)
}

/// Wait until the daemon's registry row for `id` reports `status`.
async fn wait_for_row(
    root: &mini_swe_mcp::worktree::ScratchRoot,
    id: &str,
    status: mini_swe_mcp::pool::RegistryStatus,
) {
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    loop {
        let row = mini_swe_mcp::pool::load_registry_entry_in(root, id);
        if row.is_some_and(|row| row.status == status) || std::time::Instant::now() >= deadline {
            return;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}

/// A busy daemon given a handover keeps serving while a command is marked
/// running, and stops as soon as the mark clears — leaving the live worker
/// `Interrupted` for the next daemon to continue.
#[tokio::test]
async fn a_handover_waits_for_the_command_to_clear() {
    let hub = common::TempDir::new_in_tmp("handover");
    let isolated = common::IsolatedPool::new(2, "handover-wait");
    let pool = isolated.pool.clone();
    pool.__test_save_status(
        &meta("handover-wait"),
        "test",
        mini_swe_mcp::pool::RegistryStatus::Running,
        1,
        10,
        "cargo test",
        None,
    );
    pool.__test_insert_worker(running_worker("handover-wait"))
        .await;
    let (paths, daemon) = daemon_on(&hub, pool.clone());
    let task = tokio::spawn(async move { daemon.run().await });

    let mut client = Client::connect(&paths.socket()).await;
    // The mark stands in for a worker executing a command: the quiet moment the
    // handover waits for cannot arrive while it is held.
    let running = pool.command_running("handover-wait");
    let reply = client.handover(900).await;
    assert_eq!(reply["result"]["busy"], true, "{reply}");
    assert_eq!(reply["result"]["deadline_secs"], 900, "{reply}");
    // Repeated requests are idempotent: the second one neither re-arms the
    // watcher nor moves the deadline.
    let repeat = client.handover(900).await;
    assert_eq!(repeat["result"], reply["result"], "{repeat}");
    assert!(
        !task.is_finished(),
        "a command in flight must hold the handover"
    );
    // The daemon keeps serving while it waits.
    assert_eq!(
        client.request("ping", json!({})).await["result"],
        json!({}),
        "the daemon must keep serving during a pending handover"
    );

    drop(running);
    assert!(
        tokio::time::timeout(Duration::from_secs(5), task)
            .await
            .expect("the daemon stops once the command clears")
            .expect("the daemon task joins")
            .expect("the daemon shuts down cleanly"),
    );
    assert!(!paths.socket().exists(), "teardown removes the socket");
    wait_for_row(
        &isolated.root(),
        "handover-wait",
        mini_swe_mcp::pool::RegistryStatus::Interrupted,
    )
    .await;
    let row = mini_swe_mcp::pool::load_registry_entry_in(&isolated.root(), "handover-wait")
        .expect("the interrupted row survives");
    assert_eq!(row.status, mini_swe_mcp::pool::RegistryStatus::Interrupted);
    assert!(
        row.last_command.contains("work saved on branch"),
        "the graceful path must checkpoint the worker: {:?}",
        row.last_command
    );
}

#[tokio::test]
async fn handover_waits_for_a_heavy_permit() {
    use mini_swe_mcp::pool::admission::AdmissionClass;
    let hub = common::TempDir::new_in_tmp("handover-heavy");
    let isolated = common::IsolatedPool::new(2, "handover-heavy");
    let permit = isolated
        .pool
        .admission()
        .acquire(AdmissionClass::Completion)
        .await;
    let (paths, daemon) = daemon_on(&hub, isolated.pool.clone());
    let task = tokio::spawn(async move { daemon.run().await });
    let mut client = Client::connect(&paths.socket()).await;
    assert_eq!(client.handover(900).await["result"]["pending"], true);
    let deadline = tokio::time::Instant::now() + Duration::from_millis(400);
    while tokio::time::Instant::now() < deadline {
        assert_eq!(client.request("ping", json!({})).await["result"], json!({}));
        tokio::task::yield_now().await;
    }
    assert!(!task.is_finished());
    drop(permit);
    assert!(
        tokio::time::timeout(Duration::from_secs(5), task)
            .await
            .unwrap()
            .unwrap()
            .unwrap()
    );
}

/// The deadline hands over even while a command is still running, so a daemon
/// busy all day still picks up the newer build.
#[tokio::test]
async fn the_deadline_hands_over_while_a_command_runs() {
    let hub = common::TempDir::new_in_tmp("handover-deadline");
    let isolated = common::IsolatedPool::new(2, "handover-deadline");
    let pool = isolated.pool.clone();
    pool.__test_save_status(
        &meta("handover-deadline"),
        "test",
        mini_swe_mcp::pool::RegistryStatus::Running,
        1,
        10,
        "cargo test",
        None,
    );
    pool.__test_insert_worker(running_worker("handover-deadline"))
        .await;
    let (paths, daemon) = daemon_on(&hub, pool.clone());
    let task = tokio::spawn(async move { daemon.run().await });

    let mut client = Client::connect(&paths.socket()).await;
    let running = pool.command_running("handover-deadline");
    let reply = client.handover(1).await;
    assert_eq!(reply["result"]["deadline_secs"], 1, "{reply}");
    assert!(!task.is_finished(), "the deadline has not passed yet");
    assert!(
        tokio::time::timeout(Duration::from_secs(5), task)
            .await
            .expect("the deadline ends the wait")
            .expect("the daemon task joins")
            .expect("the daemon shuts down cleanly"),
    );
    // The worker is still interrupted, not failed: the graceful path ran even
    // though no quiet moment ever arrived.
    drop(running);
    assert!(!paths.socket().exists());
    assert!(
        pool.interrupted_workers()
            .await
            .contains(&"handover-deadline".to_string()),
        "the deadline path must still checkpoint live workers"
    );
}

/// A client that is not newer than the daemon may not ask it to hand over.
#[tokio::test]
async fn an_older_client_cannot_ask_for_a_handover() {
    let hub = common::TempDir::new_in_tmp("handover-older");
    let isolated = common::IsolatedPool::new(2, "handover-older");
    let (paths, daemon) = daemon_on(&hub, isolated.pool.clone());
    let task = tokio::spawn(async move { daemon.run().await });

    let mut client = Client::connect(&paths.socket()).await;
    // The same release and an older build clock: the daemon keeps its place.
    client
        .notify(
            "hub/hello",
            json!({"agent_id": "handover-test", "version": "0.0.1",
                   "build": {"id": "test-older", "ts": 1}}),
        )
        .await;
    let reply = client.handover(900).await;
    assert!(
        reply["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("not newer")),
        "{reply}"
    );
    assert!(
        !task.is_finished(),
        "an older client must not stop the daemon"
    );
    assert_eq!(
        client.request("ping", json!({})).await["result"],
        json!({}),
        "the daemon keeps serving"
    );
    task.abort();
    let _ = task.await;
}

/// The stdio proxy follows a daemon that goes away: it reconnects to the
/// replacement with the same identity and keeps answering the MCP client.
#[tokio::test]
async fn the_stdio_proxy_reconnects_to_a_replacement_daemon() {
    use tokio::io::{AsyncBufReadExt as _, BufReader as TokioBufReader};
    use tokio::process::{ChildStdin, ChildStdout};

    let exe = common::binary_path();
    let hub = common::TempDir::new_in_tmp("handover-proxy");
    let hub_dir = hub.path().to_path_buf();
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&hub_dir, std::fs::Permissions::from_mode(0o700)).unwrap();
    }
    let swe = hub.subdir("swe");
    let _reaper = Reaper(hub_dir.clone());

    let mut proxy = tokio::process::Command::new(&exe);
    proxy
        .arg("--stdio")
        .env_remove("MINI_SWE_NO_DAEMON")
        .env("SWE_HUB_DIR", &hub_dir)
        .env("SWE_TEMP_DIR", &swe)
        .env("ENV_FILE", "/nonexistent-mini-swe-env")
        .env("OPENAI_API_KEY", "test-key-not-used-by-the-handover-test")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true);
    common::scrub_identity_env(proxy.as_std_mut());
    let mut child = proxy.spawn().expect("spawn the stdio proxy");
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = TokioBufReader::new(child.stdout.take().unwrap());

    /// Write one request frame to the proxy's stdin.
    async fn send(stdin: &mut ChildStdin, id: u64, method: &str, params: Value) {
        stdin
            .write_all(
                format!(
                    "{}\n",
                    json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params})
                )
                .as_bytes(),
            )
            .await
            .expect("write the request");
        stdin.flush().await.expect("flush the request");
    }

    /// Read the reply carrying `id`, whatever else the proxy forwards meanwhile.
    async fn read_reply(stdout: &mut TokioBufReader<ChildStdout>, id: u64) -> Value {
        let mut line = String::new();
        tokio::time::timeout(Duration::from_secs(10), stdout.read_line(&mut line))
            .await
            .unwrap_or_else(|_| panic!("the proxy did not answer id {id}"))
            .expect("read the reply");
        assert!(!line.is_empty(), "the proxy closed its stdout");
        let frame: Value = serde_json::from_str(line.trim()).expect("reply is JSON");
        assert_eq!(frame["id"], json!(id), "{frame}");
        frame
    }

    // The first request auto-starts the daemon through the proxy.
    send(&mut stdin, 1, "initialize", json!({})).await;
    let initialize = read_reply(&mut stdout, 1).await;
    assert_eq!(
        initialize["result"]["protocolVersion"], "2024-11-05",
        "{initialize}"
    );
    send(&mut stdin, 2, "tools/list", json!({})).await;
    let tools = read_reply(&mut stdout, 2).await;
    assert!(tools["result"]["tools"].is_array(), "{tools}");

    // A watch with nothing to watch blocks until its own deadline, so it is
    // still in flight when the daemon is killed underneath it.
    send(
        &mut stdin,
        3,
        "tools/call",
        json!({"name": "worker", "arguments": {"action": "watch", "timeout_secs": 30}}),
    )
    .await;
    // A later ping proves the preceding watch reached the daemon before the cut.
    send(&mut stdin, 5, "ping", json!({})).await;
    assert_eq!(read_reply(&mut stdout, 5).await["result"], json!({}));

    // Kill the daemon the proxy is talking to: the socket goes stale and the
    // proxy's connection is cut, exactly as a handover cuts it.
    let pid = daemon_pid(&hub_dir).expect("the auto-started daemon logged its pid");
    // SAFETY: `kill` takes plain integers; a stale pid only yields ESRCH.
    assert_eq!(
        unsafe { libc::kill(pid, libc::SIGKILL) },
        0,
        "kill the daemon"
    );

    // The request that was in flight at the cut is answered once, with an error
    // that says to retry it: its reply died with the old daemon.
    let cut = read_reply(&mut stdout, 3).await;
    assert_eq!(cut["error"]["code"], -32000, "{cut}");
    assert!(
        cut["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("retry")),
        "{cut}"
    );

    // The proxy reconnects — auto-starting the replacement daemon itself — and
    // keeps serving the same MCP client.
    send(&mut stdin, 4, "tools/list", json!({})).await;
    let replayed = read_reply(&mut stdout, 4).await;
    assert!(
        replayed["result"]["tools"].is_array(),
        "the proxy must answer after the daemon went away: {replayed}"
    );
    let log = std::fs::read_to_string(hub_dir.join("hub.log")).unwrap_or_default();
    assert_eq!(
        log.lines()
            .filter(|line| line.ends_with(" listening"))
            .count(),
        2,
        "a replacement daemon must have started: {log}"
    );
}

/// Wait until the hub log holds at least `count` lines mentioning `event`.
async fn wait_for_log(hub_dir: &std::path::Path, event: &str, count: usize) {
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    loop {
        let seen = std::fs::read_to_string(hub_dir.join("hub.log"))
            .map(|log| {
                log.lines()
                    .filter(|line| {
                        if event == "listening" {
                            line.ends_with(" listening")
                        } else {
                            line.contains(event)
                        }
                    })
                    .count()
            })
            .unwrap_or(0);
        if seen >= count || std::time::Instant::now() >= deadline {
            assert!(seen >= count, "hub.log never showed {count} x {event}");
            return;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}

/// The CLI `watch` follows a daemon that goes away instead of ending: it
/// reconnects to the replacement and keeps reporting the workers it watches.
#[tokio::test]
async fn the_cli_watch_survives_a_daemon_restart() {
    let exe = common::binary_path();
    let hub = common::TempDir::new_in_tmp("handover-watch");
    let hub_dir = hub.path().to_path_buf();
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&hub_dir, std::fs::Permissions::from_mode(0o700)).unwrap();
    }
    let isolated = common::IsolatedPool::new(2, "handover-watch");
    let swe = isolated.root().path().to_path_buf();
    let _reaper = Reaper(hub_dir.clone());

    // A live worker owned by the watching agent, standing in for one the
    // orchestrator dispatched: its pid is a real sleeper, so the daemon's
    // recovery leaves it running rather than interrupting it.
    let mut sleeper = tokio::process::Command::new("sleep")
        .arg("60")
        .kill_on_drop(true)
        .spawn()
        .expect("spawn the stand-in worker process");
    let mut meta = meta("handover-watch");
    meta.pid = sleeper.id().unwrap();
    isolated.pool.__test_save_status(
        &meta,
        "test",
        mini_swe_mcp::pool::RegistryStatus::Running,
        1,
        10,
        "probe",
        None,
    );

    let mut daemon = common::binary_command(&exe);
    daemon
        .arg("daemon")
        .env("SWE_HUB_DIR", &hub_dir)
        .env("SWE_TEMP_DIR", &swe)
        .env("HUB_AUTO_RESUME", "0")
        .env("ENV_FILE", "/nonexistent-mini-swe-env")
        .env("OPENAI_API_KEY", "test-key-not-used-by-the-handover-test")
        .stdout(std::process::Stdio::null())
        .stderr(
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(hub_dir.join("hub.log"))
                .unwrap(),
        );
    let mut daemon = tokio::process::Command::from(daemon)
        .spawn()
        .expect("start the daemon");
    wait_for_log(&hub_dir, "listening", 1).await;

    let mut watch = common::binary_command(&exe);
    watch
        .args(["watch", "--json"])
        .env_remove("MINI_SWE_NO_DAEMON")
        .env("SWE_HUB_DIR", &hub_dir)
        .env("SWE_TEMP_DIR", &swe)
        .env("MINI_SWE_AGENT_ID", "handover-test")
        .env("ENV_FILE", "/nonexistent-mini-swe-env")
        .env("OPENAI_API_KEY", "test-key-not-used-by-the-handover-test")
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    let mut watch = tokio::process::Command::from(watch)
        .kill_on_drop(true)
        .spawn()
        .expect("start the watch");
    // The watch is watching once the daemon serves its second connection.
    wait_for_log(&hub_dir, "Serving MCP connection", 1).await;

    // Kill the daemon the watch is talking to, exactly as a handover cuts it.
    let pid = daemon_pid(&hub_dir).expect("the daemon logged its pid");
    // SAFETY: `kill` takes plain integers; a stale pid only yields ESRCH.
    assert_eq!(
        unsafe { libc::kill(pid, libc::SIGKILL) },
        0,
        "kill the daemon"
    );
    let _ = daemon.wait().await;

    // The watch follows the daemon: it reconnects to the replacement the hub
    // auto-starts, rather than ending with the connection it lost.
    wait_for_log(&hub_dir, "listening", 2).await;
    assert!(
        watch.try_wait().expect("poll the watch").is_none(),
        "the watch must survive the daemon going away"
    );

    // And it is still watching: an event the replacement daemon reports reaches
    // the same CLI process.
    std::fs::create_dir_all(isolated.root().join("swe-wt-handover-watch")).unwrap();
    isolated
        .pool
        .__test_reset_registry_throttle("handover-watch");
    isolated.pool.__test_save_status(
        &meta,
        "test",
        mini_swe_mcp::pool::RegistryStatus::Failed,
        2,
        10,
        "probe ended",
        None,
    );
    let output = tokio::time::timeout(Duration::from_secs(15), watch.wait_with_output())
        .await
        .expect("the watch ends once its worker is terminal")
        .expect("the watch exits");
    assert!(
        output.status.success(),
        "the watch reported its last event: {output:?}"
    );
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    assert!(
        stdout.contains("handover-watch"),
        "the reconnected watch must still print events: {stdout}"
    );
    let _ = sleeper.kill().await;
    let _ = sleeper.wait().await;
}

#[tokio::test]
async fn daemon_respawns_itself_after_handover_without_a_client() {
    let hub = common::TempDir::new_in_tmp("handover-respawn");
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(hub.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
    }
    let swe = hub.subdir("swe");
    let _reaper = Reaper(hub.path().to_path_buf());
    let mut command = tokio::process::Command::new(common::binary_path());
    command
        .arg("daemon")
        .env("SWE_HUB_DIR", hub.path())
        .env("SWE_TEMP_DIR", &swe)
        .env("ENV_FILE", hub.path().join("absent.env"))
        .env("OPENAI_API_KEY", "unused")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .kill_on_drop(true);
    common::scrub_identity_env(command.as_std_mut());
    let mut daemon = command.spawn().unwrap();
    let paths = HubPaths::new(hub.path().to_path_buf());
    let mut client = Client::connect(&paths.socket()).await;
    assert_eq!(client.handover(900).await["result"]["pending"], true);
    drop(client);
    assert!(
        tokio::time::timeout(Duration::from_secs(5), daemon.wait())
            .await
            .unwrap()
            .unwrap()
            .success()
    );
    // No client connects until the replacement has already announced listening.
    wait_for_log(hub.path(), "listening", 2).await;
    let mut replacement = Client::connect(&paths.socket()).await;
    assert!(replacement.request("tools/list", json!({})).await["result"]["tools"].is_array());
}

/// The pid of the daemon the hub log last reported listening.
fn daemon_pid(hub_dir: &std::path::Path) -> Option<i32> {
    let log = std::fs::read_to_string(hub_dir.join("hub.log")).ok()?;
    log.lines()
        .filter(|line| line.ends_with(" listening"))
        .filter_map(|line| {
            line.split_whitespace()
                .find_map(|word| word.strip_prefix("pid=")?.parse().ok())
        })
        .next_back()
}

/// Kill every daemon the hub log names, so a test leaves none behind.
struct Reaper(std::path::PathBuf);

impl Drop for Reaper {
    fn drop(&mut self) {
        let Ok(log) = std::fs::read_to_string(self.0.join("hub.log")) else {
            return;
        };
        let pids: std::collections::BTreeSet<i32> = log
            .split_whitespace()
            .filter_map(|word| word.strip_prefix("pid=")?.parse().ok())
            .collect();
        for pid in &pids {
            // SAFETY: `kill` takes plain integers; a stale pid only yields ESRCH.
            unsafe { libc::kill(*pid, libc::SIGTERM) };
        }
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        while std::time::Instant::now() < deadline
            && pids.iter().any(|pid| (unsafe { libc::kill(*pid, 0) }) == 0)
        {
            std::thread::sleep(Duration::from_millis(20));
        }
    }
}

/// The registry row a live worker needs for the graceful path to checkpoint it.
fn meta(id: &str) -> mini_swe_mcp::pool::WorkerMeta {
    mini_swe_mcp::pool::WorkerMeta {
        id: id.to_string(),
        task: "handover probe".to_string(),
        group: None,
        role: mini_swe_mcp::pool::WorkerRole::Worker,
        repo_path: None,
        owner: "handover-test".to_string(),
        started_at: 0,
        pid: std::process::id(),
        revision: 0,
        auto_continues: 0,
        metrics: mini_swe_mcp::pool::WorkerMetrics::default(),
    }
}
