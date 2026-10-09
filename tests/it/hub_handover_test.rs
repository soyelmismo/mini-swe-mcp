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

use crate::common;
use mini_swe_mcp::hub::{HubConfig, HubPaths, HubServer};
use mini_swe_mcp::mcp::McpServer;
use mini_swe_mcp::pool::{RegistryStatus, WorkerPool, WorkerState};
use mini_swe_mcp::worktree::{ScratchRoot, WorktreeGuard};
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
    // Own the short socket fallback directory a too-deep hub directory moves
    // its socket to; a SIGKILLed daemon never runs its own cleanup.
    let _fallback = common::fallback_socket_dir(&hub_dir);
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

    // A request the daemon cannot answer stays in flight when it is killed
    // underneath it. No tool call blocks any more -- a host aborts one that
    // outlives its own deadline, and the abort is where an event would be lost --
    // so the daemon is stopped instead: it reads nothing, so the requests sit
    // in the socket unanswered until the cut.
    let pid = daemon_pid(&hub_dir).expect("the auto-started daemon logged its pid");
    // SAFETY: `kill` takes plain integers; a stale pid only yields ESRCH.
    assert_eq!(
        unsafe { libc::kill(pid, libc::SIGSTOP) },
        0,
        "stop the daemon"
    );
    // A stopped process never handles the SIGTERM the test Reaper sends, so a
    // failure between here and the kill below would otherwise leave this daemon
    // stopped for good: the guard owns it from the moment it is stopped.
    let _stopped = StoppedDaemon(pid);
    // The stop is delivered asynchronously, and a daemon that has not stopped
    // yet would answer the request instead of leaving it in flight, so wait
    // for the state rather than assuming it.
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while process_state(pid).as_deref() != Some("T") {
        assert!(
            std::time::Instant::now() < deadline,
            "the daemon never stopped (state {:?})",
            process_state(pid)
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    send(
        &mut stdin,
        3,
        "tools/call",
        json!({"name": "worker", "arguments": {"action": "watch"}}),
    )
    .await;
    send(&mut stdin, 4, "tools/list", json!({})).await;

    // Kill the daemon the proxy is talking to: the socket goes stale and the
    // proxy's connection is cut, exactly as a handover cuts it. The guard kills
    // it again on the way out, where a dead pid only yields ESRCH.
    // SAFETY: as above.
    assert_eq!(
        unsafe { libc::kill(pid, libc::SIGKILL) },
        0,
        "kill the daemon"
    );

    // The requests that were in flight at the cut are each answered once, with
    // an error that says to retry them: their replies died with the old daemon.
    for id in [3, 4] {
        let cut = read_reply(&mut stdout, id).await;
        assert_eq!(cut["error"]["code"], -32000, "{cut}");
        assert!(
            cut["error"]["message"]
                .as_str()
                .is_some_and(|message| message.contains("retry")),
            "{cut}"
        );
    }

    // The proxy reconnects -- auto-starting the replacement daemon itself -- and
    // keeps serving the same MCP client.
    send(&mut stdin, 6, "tools/list", json!({})).await;
    let replayed = read_reply(&mut stdout, 6).await;
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
///
/// The event is produced by a real daemon process the test (or its watch)
/// spawns; on a loaded host its exec and bind can take many seconds, so the
/// deadline carries margin. The wait returns as soon as the count is reached,
/// so an unloaded run is unaffected.
async fn wait_for_log(hub_dir: &std::path::Path, event: &str, count: usize) {
    wait_for_log_within(hub_dir, event, count, Duration::from_secs(30)).await
}

/// [`wait_for_log`] with an explicit deadline, for a daemon whose replacement
/// another process (the watch) auto-starts: the product's own reconnect budget
/// bounds how long that chase may take, so waiting longer than a direct spawn
/// would need still fails the moment the product itself would give up.
async fn wait_for_log_within(
    hub_dir: &std::path::Path,
    event: &str,
    count: usize,
    budget: Duration,
) {
    let deadline = std::time::Instant::now() + budget;
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
    // Own the short socket fallback directory a too-deep hub directory moves
    // its socket to; a SIGKILLed daemon never runs its own cleanup.
    let _fallback = common::fallback_socket_dir(&hub_dir);
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
    // auto-starts, rather than ending with the connection it lost. The
    // replacement is started by the watch's own reconnect chase, so it is
    // given the product's reconnect budget rather than a direct spawn's.
    wait_for_log_within(
        &hub_dir,
        "listening",
        2,
        Duration::from_secs(mini_swe_mcp::hub::DEFAULT_RECONNECT_SECS),
    )
    .await;
    // Listening only means the socket is bound; recovery is what puts the
    // salvaged worker back in the pool. Writing the terminal status before
    // the replacement daemon has recovered the row would leave the watch
    // nothing owned to watch, and it would rightly end with no event.
    wait_for_log_within(
        &hub_dir,
        "recovered",
        2,
        Duration::from_secs(mini_swe_mcp::hub::DEFAULT_RECONNECT_SECS),
    )
    .await;
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
    // Own the short socket fallback directory a too-deep hub directory moves
    // its socket to; a SIGKILLed daemon never runs its own cleanup.
    let _fallback = common::fallback_socket_dir(hub.path());
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

/// A daemon this test stopped with SIGSTOP, killed again on the way out.
///
/// SIGSTOP is not something a process recovers from, and a stopped one never
/// handles the SIGTERM the hub `Reaper` sends, so it cannot be the one to clean
/// this up: whatever happens between the stop and the explicit kill, the guard
/// still takes the daemon down.
struct StoppedDaemon(i32);

impl Drop for StoppedDaemon {
    fn drop(&mut self) {
        // SAFETY: `kill` takes plain integers; a stale pid only yields ESRCH.
        unsafe { libc::kill(self.0, libc::SIGKILL) };
    }
}

/// The scheduler state of `pid` from `/proc`, or `None` once it is gone.
fn process_state(pid: i32) -> Option<String> {
    let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    // The comm field is parenthesised and may hold spaces, so the state is the
    // first word after the last `)`.
    stat.rsplit_once(')')
        .map(|(_, rest)| rest.split_whitespace().next().unwrap_or("?").to_string())
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
        task: "handover probe".to_string(),
        ..mini_swe_mcp::pool::WorkerMeta::test_meta(id, "handover-test")
    }
}

// ---------------------------------------------------------------------------
// OTA-style handover: the daemon notices its own rebuilt executable.
// ---------------------------------------------------------------------------

/// Replace every occurrence of `from` with `to` in place; the two are the
/// same length, so the file's offsets do not shift.
fn replace_all(bytes: &mut [u8], from: &[u8], to: &[u8]) -> usize {
    let mut count = 0;
    let mut i = 0;
    while i + from.len() <= bytes.len() {
        if &bytes[i..i + from.len()] == from {
            bytes[i..i + from.len()].copy_from_slice(to);
            count += 1;
            i += from.len();
        } else {
            i += 1;
        }
    }
    count
}

/// Patch the build identity embedded in a copy of the binary, so the daemon
/// watching that path sees a *different, newer* build when it is replaced.
///
/// The build id and clock are string literals compiled into the binary, so
/// overwriting them in place (same length) changes what `--build-id` prints
/// without disturbing the file's structure.
fn patch_build_id(exe: &std::path::Path, new_id: &str, new_ts: &str) {
    let old_id = env!("MINI_SWE_BUILD_ID");
    let old_ts = env!("MINI_SWE_BUILD_TS");
    assert_eq!(
        new_id.len(),
        old_id.len(),
        "build id length must be preserved"
    );
    assert_eq!(
        new_ts.len(),
        old_ts.len(),
        "build ts length must be preserved"
    );
    let mut bytes = std::fs::read(exe).expect("read the binary copy");
    let n_id = replace_all(&mut bytes, old_id.as_bytes(), new_id.as_bytes());
    let n_ts = replace_all(&mut bytes, old_ts.as_bytes(), new_ts.as_bytes());
    assert!(n_id > 0, "the build id literal must be in the binary");
    assert!(n_ts > 0, "the build ts literal must be in the binary");
    std::fs::write(exe, &bytes).expect("write the patched binary");
}

/// A build id different from this build's, same length.
fn different_build_id() -> String {
    let id = format!("{:016x}", 0xdead_beef_cafe_babeu64);
    assert_ne!(id, env!("MINI_SWE_BUILD_ID"));
    id
}

/// A build clock later than this build's, same length.
fn newer_build_ts() -> String {
    let old: u64 = env!("MINI_SWE_BUILD_TS").parse().unwrap();
    let new = old + 1_000_000_000_000_000_000;
    let s = new.to_string();
    assert_eq!(s.len(), env!("MINI_SWE_BUILD_TS").len());
    s
}

/// Spawn the daemon from a *copy* of the binary in a temp dir, so the test can
/// replace that copy with a different build without touching the real one.
fn spawn_daemon_from(exe: &std::path::Path, hub: &common::TempDir) -> tokio::process::Child {
    let swe = hub.subdir("swe");
    let mut command = tokio::process::Command::new(exe);
    command
        .arg("daemon")
        .env("SWE_HUB_DIR", hub.path())
        .env("SWE_TEMP_DIR", &swe)
        .env("ENV_FILE", hub.path().join("absent.env"))
        .env("OPENAI_API_KEY", "unused")
        .env("MINI_SWE_HUB_EXE_POLL_MS", "50")
        .env("MINI_SWE_HUB_EXE_STABLE_MS", "150")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .kill_on_drop(true);
    common::scrub_identity_env(command.as_std_mut());
    // A concurrent `exec` of the same inode can make the kernel refuse the
    // spawn with `ETXTBSY`; retry briefly rather than blame the test for a
    // transient of a 40 MB binary being paged in.
    for _ in 0..40 {
        match command.spawn() {
            Ok(child) => return child,
            Err(e) if e.raw_os_error() == Some(libc::ETXTBSY) => {
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(e) => panic!("spawn the daemon from {}: {e}", exe.display()),
        }
    }
    panic!("spawning {} kept hitting ETXTBSY", exe.display())
}

/// Replace the daemon's executable with a different, newer build, and return
/// the new build id.
fn replace_exe_with_newer_build(exe: &std::path::Path) -> String {
    let new_id = different_build_id();
    let new_ts = newer_build_ts();
    let patched = exe.with_extension("new");
    std::fs::copy(exe, &patched).unwrap();
    patch_build_id(&patched, &new_id, &new_ts);
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&patched, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    std::fs::rename(&patched, exe).unwrap();
    new_id
}

/// Replacing the daemon's executable with a different, complete build arms a
/// handover without any client call: the daemon notices the new binary on its
/// own, hands over, and the replacement comes up.
#[tokio::test]
async fn a_rebuilt_executable_arms_a_handover_without_a_client() {
    let hub = common::TempDir::new_in_tmp("handover-auto");
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(hub.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
    }
    let _fallback = common::fallback_socket_dir(hub.path());
    let _reaper = Reaper(hub.path().to_path_buf());

    // The daemon runs from a copy, so the test can replace it.
    let exe_dir = hub.subdir("exe");
    let exe = exe_dir.join("mini-swe-mcp");
    std::fs::copy(common::binary_path(), &exe).unwrap();

    let mut daemon = spawn_daemon_from(&exe, &hub);
    let paths = HubPaths::new(hub.path().to_path_buf());
    wait_for_log(hub.path(), "listening", 1).await;

    // Replace the copy with a different, newer build. The patched binary
    // must report the new identity through the flag the daemon probes with.
    let new_id = replace_exe_with_newer_build(&exe);
    let probe = std::process::Command::new(&exe)
        .arg("--build-id")
        .output()
        .expect("run the patched binary");
    assert!(probe.status.success(), "the patched build must run");
    let identity: Value = serde_json::from_slice(&probe.stdout).expect("--build-id prints JSON");
    assert_eq!(identity["id"], new_id.as_str(), "{identity}");

    // The daemon arms the handover by itself and stops; the replacement
    // daemon (the patched build) comes up without any client call.
    wait_for_log(
        hub.path(),
        "executable changed: arming handover to build",
        1,
    )
    .await;
    let log = std::fs::read_to_string(hub.path().join("hub.log")).unwrap_or_default();
    assert!(
        log.contains(&format!("arming handover to build {new_id}")),
        "the log must name the new build: {log}"
    );
    assert!(
        tokio::time::timeout(Duration::from_secs(10), daemon.wait())
            .await
            .unwrap()
            .unwrap()
            .success(),
        "the daemon must stop for the handover"
    );
    wait_for_log(hub.path(), "listening", 2).await;
    let mut replacement = Client::connect(&paths.socket()).await;
    assert!(
        replacement.request("tools/list", json!({})).await["result"]["tools"].is_array(),
        "the replacement daemon must serve"
    );
}

/// A half-written or unexecutable file at the executable path does not arm a
/// handover: the daemon only hands over to a build that is complete and runs.
#[tokio::test]
async fn a_half_written_or_unexecutable_file_does_not_arm() {
    let hub = common::TempDir::new_in_tmp("handover-auto-neg");
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(hub.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
    }
    let _fallback = common::fallback_socket_dir(hub.path());
    let _reaper = Reaper(hub.path().to_path_buf());
    let exe_dir = hub.subdir("exe");
    let exe = exe_dir.join("mini-swe-mcp");
    std::fs::copy(common::binary_path(), &exe).unwrap();

    let mut daemon = spawn_daemon_from(&exe, &hub);
    let paths = HubPaths::new(hub.path().to_path_buf());
    wait_for_log(hub.path(), "listening", 1).await;

    // A half-written file (a cargo write in progress, or a truncated copy).
    let half = exe_dir.join("half");
    let bytes = std::fs::read(&exe).unwrap();
    std::fs::write(&half, &bytes[..bytes.len() / 2]).unwrap();
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&half, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    std::fs::rename(&half, &exe).unwrap();

    // An unexecutable file.
    let not_exec = exe_dir.join("not-exec");
    std::fs::copy(common::binary_path(), &not_exec).unwrap();
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&not_exec, std::fs::Permissions::from_mode(0o644)).unwrap();
    }
    std::fs::rename(&not_exec, &exe).unwrap();

    // Neither arms a handover, and the daemon keeps serving.
    tokio::time::sleep(Duration::from_millis(2000)).await;
    let log = std::fs::read_to_string(hub.path().join("hub.log")).unwrap_or_default();
    assert!(
        !log.contains("arming handover"),
        "a broken build must not arm: {log}"
    );
    assert!(
        daemon.try_wait().unwrap().is_none(),
        "the daemon must keep running"
    );
    let mut client = Client::connect(&paths.socket()).await;
    assert_eq!(
        client.request("ping", json!({})).await["result"],
        json!({}),
        "the daemon must keep serving"
    );

    // A complete replacement still arms the handover: the refusals above were
    // the completeness checks, not a watcher that is dead altogether.
    let new_id = replace_exe_with_newer_build(&exe);
    wait_for_log(hub.path(), &format!("arming handover to build {new_id}"), 1).await;
}

/// `HUB_AUTO_HANDOVER=0` disables the watch: a rebuilt executable does not
/// arm a handover.
#[tokio::test]
async fn auto_handover_is_disabled_by_env() {
    let hub = common::TempDir::new_in_tmp("handover-auto-off");
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(hub.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
    }
    let _fallback = common::fallback_socket_dir(hub.path());
    let _reaper = Reaper(hub.path().to_path_buf());
    let exe_dir = hub.subdir("exe");
    let exe = exe_dir.join("mini-swe-mcp");
    std::fs::copy(common::binary_path(), &exe).unwrap();

    let swe = hub.subdir("swe");
    let mut command = tokio::process::Command::new(&exe);
    command
        .arg("daemon")
        .env("SWE_HUB_DIR", hub.path())
        .env("SWE_TEMP_DIR", &swe)
        .env("ENV_FILE", hub.path().join("absent.env"))
        .env("OPENAI_API_KEY", "unused")
        .env("HUB_AUTO_HANDOVER", "0")
        .env("MINI_SWE_HUB_EXE_POLL_MS", "50")
        .env("MINI_SWE_HUB_EXE_STABLE_MS", "150")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .kill_on_drop(true);
    common::scrub_identity_env(command.as_std_mut());
    let mut daemon = command.spawn().unwrap();
    wait_for_log(hub.path(), "listening", 1).await;

    // Replace with a different, newer build.
    replace_exe_with_newer_build(&exe);

    tokio::time::sleep(Duration::from_millis(2500)).await;
    let log = std::fs::read_to_string(hub.path().join("hub.log")).unwrap_or_default();
    assert!(
        !log.contains("arming handover"),
        "HUB_AUTO_HANDOVER=0 must disable the watch: {log}"
    );
    assert!(
        daemon.try_wait().unwrap().is_none(),
        "the daemon must keep running"
    );
}

// ----------
// Strict teardown ordering across a handover.
// ----------

/// A git repository a dispatch can cut a worker worktree from.
fn dispatch_repo(tag: &str) -> common::TempDir {
    let repo = common::TempDir::new_in_tmp(tag);
    common::git(repo.path(), &["init", "-b", "master"]);
    common::git(repo.path(), &["config", "user.name", "handover"]);
    common::git(repo.path(), &["config", "user.email", "handover@test"]);
    std::fs::write(repo.path().join("lib.rs"), "// base\n").unwrap();
    common::git(repo.path(), &["add", "."]);
    common::git(repo.path(), &["commit", "-m", "base"]);
    repo
}

/// Wait until `id` reaches a terminal state in `pool`, or panic.
async fn wait_for_terminal(pool: &WorkerPool, id: &str) -> WorkerState {
    let deadline = std::time::Instant::now() + Duration::from_secs(30);
    loop {
        if let Some(state) = pool.get_worker_state(id).await
            && !matches!(
                state,
                WorkerState::Running { .. } | WorkerState::Paused { .. }
            )
        {
            return state;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "worker {id} never reached a terminal state"
        );
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}

/// A handover with an interrupted worker whose teardown is artificially slowed
/// must not let the replacement daemon touch that worker's worktree before the
/// old daemon has finished.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_handover_waits_for_the_predecessors_worker_teardown() {
    let hub = common::TempDir::new_in_tmp("handover-order");
    let repo = dispatch_repo("handover-order-repo");
    let scratch = common::TempDir::new_in_tmp("handover-order-root");
    let root = ScratchRoot::new(scratch.path());

    // The old daemon: a real pool over the shared scratch root.
    let llm = common::fake_llm::FakeLlm::spawn("true", "sleep 2").await;
    let old_pool =
        WorkerPool::with_scratch(1, llm.base_url().to_string(), "k".to_string(), root.clone());
    let (paths, old_daemon) = daemon_on(&hub, old_pool.clone());
    let old_task = tokio::spawn(async move { old_daemon.run().await });

    // One worker, interrupted mid-turn with a live worktree: turn one runs
    // `true`, turn two the slow `sleep 2` the handover cuts short.
    let id = old_pool
        .dispatch(
            "handover-test".to_string(),
            "slow worker".to_string(),
            "test-model".to_string(),
            None,
            repo.path().to_path_buf(),
            10,
            None,
            None,
            false,
            None,
            Vec::new(),
        )
        .await
        .expect("dispatch the worker");
    let worktree = root.join(format!("swe-wt-{id}"));
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    loop {
        if worktree.is_dir()
            && let Some(WorkerState::Running { last_command, .. }) =
                old_pool.get_worker_state(&id).await
            && last_command.contains("sleep")
        {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "the worker never reached its slow command"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    // Slow the old daemon's worktree teardown: its guard drop now sleeps two
    // seconds after the abort, so a replacement that does not wait recreates
    // the worktree while the old one is still being removed.
    WorktreeGuard::__test_set_teardown_delay(&id, Duration::from_secs(2));

    let mut client = Client::connect(&paths.socket()).await;

    // Deadline one: a command is in flight, so the handover happens at the
    // deadline instead of waiting for it.
    let started = std::time::Instant::now();
    let reply = client.handover(1).await;
    assert_eq!(reply["result"]["deadline_secs"], 1, "{reply}");

    // The old daemon stops accepting before it tears the workers down, so a
    // client that finds the hub gone starts the replacement while the old one
    // still holds the lock. Wait for that moment, then start the replacement.
    let socket_gone = std::time::Instant::now() + Duration::from_secs(10);
    while paths.socket().exists() {
        assert!(
            std::time::Instant::now() < socket_gone,
            "the old daemon never stopped accepting connections"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    let new_pool =
        WorkerPool::with_scratch(1, llm.base_url().to_string(), "k".to_string(), root.clone());
    let (new_paths, new_daemon) = daemon_on(&hub, new_pool.clone());
    let new_task = tokio::spawn(async move { new_daemon.run().await });

    // The old daemon finishes only after its delayed teardown completes.
    let old_done = tokio::time::timeout(Duration::from_secs(20), old_task)
        .await
        .expect("the old daemon stops")
        .expect("the old daemon task joins")
        .expect("the old daemon shuts down cleanly");
    assert!(old_done);
    assert!(
        started.elapsed() >= Duration::from_secs(2),
        "the old daemon must hold the lock until its delayed teardown finishes, took {:?}",
        started.elapsed()
    );

    // The replacement must not have touched the interrupted worker's worktree
    // before the old daemon finished: the old teardown removed it, and the new
    // daemon has not recreated it yet.
    assert!(
        !worktree.exists(),
        "the replacement daemon touched the interrupted worker's worktree before \
         the old daemon finished teardown"
    );
    let log = std::fs::read_to_string(paths.log()).unwrap_or_default();
    let stopped = log.find("stopped").expect("the old daemon logs 'stopped'");
    let resumed = log.find("auto-continued");
    assert!(
        resumed.is_none_or(|at| at > stopped),
        "recovery must run after the old daemon's teardown:\n{log}"
    );

    // The replacement waited for the lock, then recovered and continued the
    // interrupted worker to completion.
    let state = wait_for_terminal(&new_pool, &id).await;
    assert!(
        matches!(state, WorkerState::Completed { .. }),
        "the replacement must continue the interrupted worker, got {state:?}"
    );
    WorktreeGuard::__test_clear_teardown_delay(&id);
    new_task.abort();
    let _ = new_task.await;
    assert!(!new_paths.socket().exists(), "teardown removes the socket");
}

/// A teardown that blocks past the shutdown wait must not hang the handover:
/// the old daemon releases the lock on the bound, and the replacement waits on
/// the worker's `.teardown` marker instead of racing its worktree.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn a_teardown_past_the_wait_releases_the_lock_and_the_replacement_waits() {
    let hub = common::TempDir::new_in_tmp("handover-bound");
    let repo = dispatch_repo("handover-bound-repo");
    let scratch = common::TempDir::new_in_tmp("handover-bound-root");
    let root = ScratchRoot::new(scratch.path());

    // The old daemon: its shutdown wait is one second, far less than the
    // teardown it is about to block on.
    let llm = common::fake_llm::FakeLlm::spawn("true", "sleep 2").await;
    let old_pool =
        WorkerPool::with_scratch(1, llm.base_url().to_string(), "k".to_string(), root.clone())
            .with_teardown_wait(Duration::from_secs(1));
    let (paths, old_daemon) = daemon_on(&hub, old_pool.clone());
    let old_task = tokio::spawn(async move { old_daemon.run().await });

    let id = old_pool
        .dispatch(
            "handover-test".to_string(),
            "slow worker".to_string(),
            "test-model".to_string(),
            None,
            repo.path().to_path_buf(),
            10,
            None,
            None,
            false,
            None,
            Vec::new(),
        )
        .await
        .expect("dispatch the worker");
    let worktree = root.join(format!("swe-wt-{id}"));
    let marker = root.join(format!("swe-wt-{id}.teardown"));
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    loop {
        if worktree.is_dir()
            && let Some(WorkerState::Running { last_command, .. }) =
                old_pool.get_worker_state(&id).await
            && last_command.contains("sleep")
        {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "the worker never reached its slow command"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    // Block the teardown for longer than the shutdown wait.
    WorktreeGuard::__test_set_teardown_delay(&id, Duration::from_secs(5));

    let mut client = Client::connect(&paths.socket()).await;
    let started = std::time::Instant::now();
    let reply = client.handover(1).await;
    assert_eq!(reply["result"]["deadline_secs"], 1, "{reply}");

    // The old daemon stops accepting first; start the replacement while the old
    // one is still inside its bounded (and here, exceeded) teardown wait.
    let socket_gone = std::time::Instant::now() + Duration::from_secs(10);
    while paths.socket().exists() {
        assert!(
            std::time::Instant::now() < socket_gone,
            "the old daemon never stopped accepting connections"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    let new_pool =
        WorkerPool::with_scratch(1, llm.base_url().to_string(), "k".to_string(), root.clone())
            // Long enough to outlast the blocked teardown and continue the worker.
            .with_teardown_wait(Duration::from_secs(30));
    let (new_paths, new_daemon) = daemon_on(&hub, new_pool.clone());
    let new_task = tokio::spawn(async move { new_daemon.run().await });

    // The shutdown wait is bounded: the old daemon releases the lock well
    // before the five-second teardown finishes, instead of hanging on it.
    let old_done = tokio::time::timeout(Duration::from_secs(10), old_task)
        .await
        .expect("the old daemon stops")
        .expect("the old daemon task joins")
        .expect("the old daemon shuts down cleanly");
    assert!(old_done);
    assert!(
        started.elapsed() < Duration::from_secs(4),
        "a teardown past the wait must not block shutdown, took {:?}",
        started.elapsed()
    );
    // The teardown is still running, so the old process left its marker and the
    // replacement has not recreated the worktree.
    assert!(
        marker.exists(),
        "the blocked teardown must leave its marker"
    );
    // The worktree directory's identity, so a replacement that silently
    // recreated it (the H18 remove/create race) is caught even if its worker
    // record never reaches `Running`.
    let inode = |path: &std::path::Path| -> Option<(u64, u64)> {
        use std::os::unix::fs::MetadataExt;
        std::fs::metadata(path).ok().map(|m| (m.dev(), m.ino()))
    };
    let original = inode(&worktree);
    assert!(
        original.is_some(),
        "the worktree must exist while it is torn down"
    );
    let settle = std::time::Instant::now() + Duration::from_millis(1500);
    while std::time::Instant::now() < settle {
        assert!(
            !matches!(
                new_pool.get_worker_state(&id).await,
                Some(WorkerState::Running { .. })
            ),
            "the replacement must not start the worker while the marker exists"
        );
        let row = mini_swe_mcp::pool::load_registry_entry_in(&root, &id)
            .expect("the interrupted row survives");
        assert_eq!(
            row.status,
            RegistryStatus::Interrupted,
            "the replacement recorded the worker while the marker exists"
        );
        assert_eq!(
            inode(&worktree),
            original,
            "the replacement recreated the worktree while the marker exists"
        );
        assert!(marker.exists(), "the marker must outlive the wait window");
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    // Once the old teardown finishes it removes its marker, and only then does
    // the replacement continue the worker to completion.
    let state = wait_for_terminal(&new_pool, &id).await;
    assert!(
        matches!(state, WorkerState::Completed { .. }),
        "the replacement must continue the worker once the marker is gone, got {state:?}"
    );
    assert!(!marker.exists(), "the teardown must remove its marker");
    let _ = new_paths;
    new_task.abort();
    let _ = new_task.await;
}
