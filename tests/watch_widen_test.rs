//! A second, broader `watch` widens the running one instead of being refused.
//!
//! One watch runs per session: the first `watch --group A --all` holds the
//! slot, and a second `watch --all` of the same session folds its selection
//! into the running one. The running watch keeps its connection, its place and
//! its pending events, and from then on answers for the union - here, a round
//! event for group B. A request the running watch already covers answers
//! "already covered" instead, and another owner's workers never appear.

mod common;

use mini_swe_mcp::hub::{HubConfig, HubPaths, HubServer};
use mini_swe_mcp::mcp::McpServer;
use mini_swe_mcp::pool::{
    LogBuffer, RegistryStatus, WorkerMeta, WorkerMetrics, WorkerPool, WorkerRecord, WorkerState,
};
use serde_json::json;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;

const OWNER: &str = "agent-a";
const OTHER: &str = "agent-b";
const GROUP_A: &str = "round-a";
const GROUP_B: &str = "round-b";

/// A worker record owned by `owner`, in the state the test hands it.
fn record_owned(id: &str, owner: &str, state: WorkerState) -> WorkerRecord {
    WorkerRecord {
        id: id.to_string(),
        task: "widen probe".to_string(),
        model: "test".to_string(),
        owner: owner.to_string(),
        state,
        metrics: WorkerMetrics::default(),
        logs: LogBuffer::new(),
        pending_steer: Vec::new(),
        resume_tx: None,
        handle: None,
        revision: 0,
    }
}

fn running() -> WorkerState {
    WorkerState::Running {
        step: 1,
        last_command: "ls".to_string(),
        started_at: 0,
    }
}

fn completed() -> WorkerState {
    WorkerState::Completed {
        turns: 2,
        diff: String::new(),
        summary: "Fixed.".to_string(),
        completed_at: 0,
        artifacts: Vec::new(),
        branch: Some("worker-x".to_string()),
        verified: Some(true),
        metrics: WorkerMetrics::default(),
        revision: 0,
        report: None,
    }
}

/// The registry row that puts `id` in `group` for `owner`.
fn meta_in(id: &str, owner: &str, group: &str) -> WorkerMeta {
    WorkerMeta {
        group: Some(group.to_string()),
        owner: owner.to_string(),
        ..WorkerMeta::test_meta(id, owner)
    }
}

async fn add_running_in(pool: &WorkerPool, id: &str, owner: &str, group: &str) {
    pool.__test_insert_worker(record_owned(id, owner, running()))
        .await;
    pool.__test_save_status(
        &meta_in(id, owner, group),
        "test",
        RegistryStatus::Running,
        1,
        10,
        "ls",
        None,
    );
}

async fn set_completed(pool: &WorkerPool, id: &str) {
    pool.__test_set_worker_state(id, completed()).await;
}

/// The hub paths a daemon under test binds, plus ownership of the short socket
/// fallback directory that a hub directory too deep for `sun_path` moves its
/// socket into.
fn paths(dir: &Path) -> (HubPaths, Option<common::TempDir>) {
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700)).expect("0700");
    let fallback = common::fallback_socket_dir(dir);
    (HubPaths::new(dir.to_path_buf()), fallback)
}

async fn wait_for_socket(path: &Path) {
    for _ in 0..100 {
        if UnixStream::connect(path).await.is_ok() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("hub socket never came up");
}

/// A minimal JSON-RPC client, so a test can send the same `hub/watch` frames
/// the CLI does without spawning a binary.
struct Raw {
    reader: BufReader<tokio::net::unix::OwnedReadHalf>,
    writer: tokio::net::unix::OwnedWriteHalf,
    next_id: u64,
}

impl Raw {
    async fn connect(socket: &Path) -> Self {
        let (reader, writer) = UnixStream::connect(socket)
            .await
            .expect("connect")
            .into_split();
        Self {
            reader: BufReader::new(reader),
            writer,
            next_id: 1,
        }
    }

    async fn request(&mut self, method: &str, params: serde_json::Value) -> serde_json::Value {
        let id = self.next_id;
        self.next_id += 1;
        self.writer
            .write_all(
                format!(
                    "{}\n",
                    json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params})
                )
                .as_bytes(),
            )
            .await
            .expect("write");
        self.writer.flush().await.expect("flush");
        loop {
            let mut line = String::new();
            let read = self.reader.read_line(&mut line).await.expect("read");
            assert!(read > 0, "hub closed the connection");
            let reply: serde_json::Value = serde_json::from_str(line.trim()).expect("JSON");
            if reply.get("id") == Some(&json!(id)) {
                return reply;
            }
        }
    }
}

/// One running hub over one worker per group, plus a connected `OWNER` client.
struct Harness {
    pool: WorkerPool,
    _isolated: common::IsolatedPool,
    hub: common::TempDir,
    client: Raw,
    task: tokio::task::JoinHandle<()>,
}

async fn harness() -> Harness {
    let isolated = common::IsolatedPool::new(4, "watch-widen");
    add_running_in(&isolated.pool, "wa-1", OWNER, GROUP_A).await;
    add_running_in(&isolated.pool, "wb-1", OWNER, GROUP_B).await;
    add_running_in(&isolated.pool, "other-1", OTHER, GROUP_B).await;
    let server = Arc::new(McpServer::new(isolated.pool.clone(), "test".to_string()));
    // `SWE_HUB_DIR` is what a CLI child reads; the daemon binds it directly.
    let hub = common::TempDir::new_in_tmp("watch-widen-hub");
    let (paths, _fallback) = paths(hub.path());
    let socket = paths.socket();
    let daemon = HubServer::new(server, HubConfig::new(paths, 60));
    let task = tokio::spawn(async move {
        let _ = daemon.run().await;
    });
    wait_for_socket(&socket).await;
    let mut client = Raw::connect(&socket).await;
    client
        .request("hub/hello", json!({"agent_id": OWNER, "pid": 4242}))
        .await;
    Harness {
        pool: isolated.pool.clone(),
        _isolated: isolated,
        hub,
        client,
        task,
    }
}

/// The `hub/watch` parameters a CLI `watch --group <g> --all` sends.
fn watch_group_all(group: &str) -> serde_json::Value {
    json!({"worker_ids": [], "group": [group], "initial": false, "all": true})
}

/// The `hub/watch` parameters a CLI `watch --all` sends: every live group.
fn watch_every_all() -> serde_json::Value {
    json!({"worker_ids": [], "group": [], "initial": false, "all": true})
}

/// Poll `hub/watch` until it answers a round event, or fail at the deadline.
async fn wait_for_round(client: &mut Raw, request: serde_json::Value) -> Vec<serde_json::Value> {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let reply = client.request("hub/watch", request.clone()).await;
        let events = reply["result"]["events"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        if !events.is_empty() {
            assert!(
                events.iter().all(|event| event["event"] == "round"),
                "`--all` must only ever answer the consolidated round: {reply}"
            );
            return events;
        }
        assert!(
            Instant::now() < deadline,
            "no round event within the deadline: {reply}"
        );
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}

/// A running `--group A --all` watch, widened by `--all`, then delivers a
/// round event for group B: the running watch follows the union from then on.
#[tokio::test]
async fn a_broader_second_watch_widens_the_running_one() {
    let mut harness = harness().await;

    // The first watch follows group A only, and waits while it runs.
    let reply = harness
        .client
        .request("hub/watch", watch_group_all(GROUP_A))
        .await;
    assert_eq!(reply["result"]["events"], json!([]), "{reply}");

    // A second connection of the same session asks for every group: it must
    // not take the slot over. It answers in-band with the union and exits,
    // while the running watch keeps its connection.
    let mut second =
        Raw::connect(&mini_swe_mcp::hub::HubPaths::new(harness.hub.path().to_path_buf()).socket())
            .await;
    second
        .request("hub/hello", json!({"agent_id": OWNER}))
        .await;
    let reply = second.request("hub/watch", watch_every_all()).await;
    assert!(reply.get("error").is_none(), "{reply}");
    assert_eq!(reply["result"]["events"], json!([]), "{reply}");
    let widened = &reply["result"]["widened"];
    assert!(
        widened["selection"].as_str().is_some(),
        "the widening names the union: {reply}"
    );
    assert!(
        !widened["selection"]
            .as_str()
            .unwrap_or("")
            .contains(GROUP_A)
            || widened["all"] == json!(true),
        "the union keeps round mode: {reply}"
    );

    // Group B stops in full: the running watch - on its own connection -
    // answers with that round, listing its own worker alone.
    set_completed(&harness.pool, "wb-1").await;
    let events = wait_for_round(&mut harness.client, watch_every_all()).await;
    assert_eq!(events.len(), 1, "{events:?}");
    assert_eq!(events[0]["group"], GROUP_B, "{events:?}");
    let workers = events[0]["workers"].as_array().expect("workers");
    assert_eq!(
        workers
            .iter()
            .map(|w| w["worker_id"].as_str())
            .collect::<Vec<_>>(),
        vec![Some("wb-1")],
        "another owner's workers never appear: {events:?}"
    );

    harness.task.abort();
    let _ = harness.task.await;
}

/// A request the running watch already covers exits 0 with the covered line:
/// it answers in-band, carrying nothing, and the running watch is untouched.
#[tokio::test]
async fn a_covered_second_watch_answers_in_band() {
    let mut harness = harness().await;

    let reply = harness.client.request("hub/watch", watch_every_all()).await;
    assert_eq!(reply["result"]["events"], json!([]), "{reply}");

    // The same session asks for a group the running watch already follows.
    let mut second =
        Raw::connect(&mini_swe_mcp::hub::HubPaths::new(harness.hub.path().to_path_buf()).socket())
            .await;
    second
        .request("hub/hello", json!({"agent_id": OWNER, "pid": 7777}))
        .await;
    let reply = second.request("hub/watch", watch_group_all(GROUP_A)).await;
    assert!(reply.get("error").is_none(), "{reply}");
    assert_eq!(reply["result"]["events"], json!([]), "{reply}");
    assert!(
        reply["result"]["covered"]["selection"].as_str().is_some(),
        "a covered watch names the running selection: {reply}"
    );
    assert_eq!(
        reply["result"]["covered"]["pid"],
        json!(4242),
        "a covered watch names the running watch: {reply}"
    );

    // The running watch is untouched: group A still waits on its own
    // connection, and answers once it stops.
    set_completed(&harness.pool, "wa-1").await;
    let events = wait_for_round(&mut harness.client, watch_every_all()).await;
    assert_eq!(events.len(), 1, "{events:?}");
    assert_eq!(events[0]["group"], GROUP_A, "{events:?}");

    harness.task.abort();
    let _ = harness.task.await;
}

/// The running watch's pending event survives the widening: nothing is lost
/// when the filter grows.
#[tokio::test]
async fn a_pending_event_survives_the_widening() {
    let mut harness = harness().await;

    // A plain watch of group A queues the completion without delivering it.
    let plain = json!({"worker_ids": [], "group": [GROUP_A], "initial": false});
    let reply = harness.client.request("hub/watch", plain.clone()).await;
    assert_eq!(reply["result"]["events"], json!([]), "{reply}");
    set_completed(&harness.pool, "wa-1").await;
    tokio::time::sleep(Duration::from_millis(1500)).await;

    // A second connection widens the running watch to every group.
    let mut second =
        Raw::connect(&mini_swe_mcp::hub::HubPaths::new(harness.hub.path().to_path_buf()).socket())
            .await;
    second
        .request("hub/hello", json!({"agent_id": OWNER}))
        .await;
    let reply = second.request("hub/watch", plain.clone()).await;
    // A plain request of the same breadth is covered, not widened.
    assert!(
        reply["result"]["covered"].is_object() || reply["result"]["widened"].is_object(),
        "the second plain watch must not take the slot: {reply}"
    );

    // The pending completion is still there on the running connection.
    let reply = harness.client.request("hub/watch", plain).await;
    let events = reply["result"]["events"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    assert!(
        events.iter().any(|event| event["worker_id"] == "wa-1"),
        "the pending event survives the widening: {reply}"
    );

    harness.task.abort();
    let _ = harness.task.await;
}

/// Another owner's workers never appear, however wide the union grows.
#[tokio::test]
async fn a_widened_watch_stays_inside_its_caller_ownership() {
    let mut harness = harness().await;

    let reply = harness
        .client
        .request("hub/watch", watch_group_all(GROUP_A))
        .await;
    assert_eq!(reply["result"]["events"], json!([]), "{reply}");

    let mut second =
        Raw::connect(&mini_swe_mcp::hub::HubPaths::new(harness.hub.path().to_path_buf()).socket())
            .await;
    second
        .request("hub/hello", json!({"agent_id": OWNER}))
        .await;
    let reply = second.request("hub/watch", watch_every_all()).await;
    assert!(reply.get("error").is_none(), "{reply}");

    // Group B holds the caller's worker and another owner's worker; only the
    // caller's is ever listed.
    set_completed(&harness.pool, "wb-1").await;
    set_completed(&harness.pool, "other-1").await;
    let events = wait_for_round(&mut harness.client, watch_every_all()).await;
    assert_eq!(events[0]["group"], GROUP_B, "{events:?}");
    let workers = events[0]["workers"].as_array().expect("workers");
    assert!(
        workers.iter().all(|w| w["worker_id"] != "other-1"),
        "another owner's workers never appear: {events:?}"
    );
    assert!(
        workers.iter().any(|w| w["worker_id"] == "wb-1"),
        "the caller's own worker is listed: {events:?}"
    );

    harness.task.abort();
    let _ = harness.task.await;
}

/// The CLI's own rule: a running `watch --group A --all` is widened by a later
/// `watch --all` (exit 0, one line naming the union), and a request it already
/// covers answers "already covered" with exit 0 instead of the old exit 5.
#[test]
fn the_second_cli_watch_widens_or_reports_covered() {
    let hub = common::TempDir::new_in_tmp("widen-cli-hub");
    let swe = common::TempDir::new_in_tmp("widen-cli-swe");
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("runtime");
    let owner = common::host_of_this_process();
    let hub_path = hub.path().to_path_buf();
    let socket = HubPaths::new(hub_path.clone()).socket();
    let task = rt.spawn(async move {
        let isolated = common::IsolatedPool::new(4, "widen-cli");
        add_running_in(&isolated.pool, "wa-1", &owner, GROUP_A).await;
        add_running_in(&isolated.pool, "wb-1", &owner, GROUP_B).await;
        let server = Arc::new(McpServer::new(isolated.pool.clone(), "test".to_string()));
        let (paths, _fallback) = paths(&hub_path);
        let daemon = HubServer::new(server, HubConfig::new(paths, 60));
        let wait = tokio::spawn(async move {
            let _ = daemon.run().await;
        });
        let _keep = isolated;
        let _ = wait.await;
    });
    rt.block_on(wait_for_socket(&socket));

    let spawn_watch = |args: &[&str]| {
        common::binary_command(&common::binary_path())
            .args(args)
            .env("SWE_HUB_DIR", hub.path())
            .env("SWE_TEMP_DIR", swe.path())
            .env("TMPDIR", swe.path())
            .env("OPENAI_API_KEY", "test-key-not-used")
            .env("ENV_FILE", "/nonexistent-mini-swe-env")
            .env(
                "MODELS_FILE",
                concat!(env!("CARGO_MANIFEST_DIR"), "/models.yaml"),
            )
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .expect("spawn watch")
    };

    let mut running = spawn_watch(&["watch", "--group", GROUP_A, "--all", "--timeout", "30"]);
    std::thread::sleep(Duration::from_millis(1500));

    // A broader request: widened, exit 0, one line naming the union.
    let widened = spawn_watch(&["watch", "--all", "--timeout", "5"]);
    let out = widened.wait_with_output().expect("the widened watch exits");
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    assert_eq!(
        out.status.code(),
        Some(0),
        "{stdout}{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        stdout.contains("widened the running watch (pid ") && stdout.contains(" to: "),
        "{stdout}"
    );

    // A request the running watch now covers: exit 0, the covered line.
    let covered = spawn_watch(&["watch", "--group", GROUP_A, "--all", "--timeout", "5"]);
    let out = covered.wait_with_output().expect("the covered watch exits");
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    assert_eq!(
        out.status.code(),
        Some(0),
        "{stdout}{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        stdout.contains("already covered by the running watch (pid "),
        "{stdout}"
    );

    let _ = running.kill();
    let _ = running.wait();
    task.abort();
    let _ = rt.block_on(task);
}
