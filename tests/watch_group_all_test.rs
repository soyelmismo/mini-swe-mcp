//! `watch --all`: one consolidated event for a whole round.
//!
//! A group watch keeps waiting while any selected worker runs, and answers
//! once, when the last one stops or as soon as one needs the orchestrator
//! (a question or a failure). The individual transitions it folds in are
//! acknowledged, so a later plain watch does not replay them.

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
const GROUP: &str = "round";

/// A worker record owned by `OWNER`, in the state the test hands it.
fn record(id: &str, state: WorkerState) -> WorkerRecord {
    WorkerRecord {
        id: id.to_string(),
        task: "round probe".to_string(),
        model: "test".to_string(),
        owner: OWNER.to_string(),
        state,
        metrics: WorkerMetrics::default(),
        logs: LogBuffer::new(),
        pending_steer: Vec::new(),
        resume_tx: None,
        handle: None,
        revision: 0,
    }
}

/// The registry row that puts `id` in [`GROUP`], so a `--group` watch finds it.
fn meta(id: &str) -> WorkerMeta {
    WorkerMeta {
        group: Some(GROUP.to_string()),
        ..WorkerMeta::test_meta(id, OWNER)
    }
}

fn running() -> WorkerState {
    WorkerState::Running {
        step: 1,
        last_command: "ls".to_string(),
        started_at: 0,
    }
}

async fn add_running(pool: &WorkerPool, id: &str) {
    pool.__test_insert_worker(record(id, running())).await;
    pool.__test_save_status(
        &meta(id),
        "test",
        RegistryStatus::Running,
        1,
        10,
        "ls",
        None,
    );
}

async fn set_completed(pool: &WorkerPool, id: &str) {
    pool.__test_set_worker_state(
        id,
        WorkerState::Completed {
            turns: 2,
            diff: String::new(),
            summary: "Fixed.".to_string(),
            completed_at: 0,
            artifacts: Vec::new(),
            branch: Some(format!("worker-{id}")),
            verified: Some(true),
            metrics: WorkerMetrics::default(),
            revision: 0,
            report: None,
        },
    )
    .await;
}

async fn set_paused(pool: &WorkerPool, id: &str, question: &str) {
    pool.__test_set_worker_state(
        id,
        WorkerState::Paused {
            question: question.to_string(),
            step: 2,
            paused_at: 0,
        },
    )
    .await;
}

/// The `hub/watch` parameters a CLI `watch --group <g> --all` sends.
fn watch_all() -> serde_json::Value {
    json!({"worker_ids": [], "group": GROUP, "initial": false, "all": true})
}

fn paths(dir: &Path) -> HubPaths {
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700)).expect("0700");
    HubPaths::new(dir.to_path_buf())
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

/// One running hub over `pool`, plus a connected `OWNER` client.
struct Harness {
    pool: WorkerPool,
    _isolated: common::IsolatedPool,
    hub: common::TempDir,
    client: Raw,
    task: tokio::task::JoinHandle<()>,
}

async fn harness(ids: &[&str]) -> Harness {
    let isolated = common::IsolatedPool::new(4, "watch-all");
    for id in ids {
        add_running(&isolated.pool, id).await;
    }
    let server = Arc::new(McpServer::new(isolated.pool.clone(), "test".to_string()));
    let hub = common::TempDir::new_in_tmp("watch-all-hub");
    let socket = paths(hub.path()).socket();
    let daemon = HubServer::new(server, HubConfig::new(paths(hub.path()), 60));
    let task = tokio::spawn(async move {
        let _ = daemon.run().await;
    });
    wait_for_socket(&socket).await;
    let mut client = Raw::connect(&socket).await;
    client
        .request("hub/hello", json!({"agent_id": OWNER}))
        .await;
    Harness {
        pool: isolated.pool.clone(),
        _isolated: isolated,
        hub,
        client,
        task,
    }
}

/// Poll `hub/watch --all` until it answers a round, or fail at the deadline.
async fn wait_for_round(client: &mut Raw) -> Vec<serde_json::Value> {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let reply = client.request("hub/watch", watch_all()).await;
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

/// Three workers, one round: the event only appears once the last one stops,
/// and it lists each worker on its own compact line.
#[tokio::test]
async fn one_round_event_after_the_last_worker_stops() {
    let mut harness = harness(&["w-1", "w-2", "w-3"]).await;

    // While everyone runs the round waits.
    let reply = harness.client.request("hub/watch", watch_all()).await;
    assert_eq!(reply["result"]["events"], json!([]), "{reply}");

    // Two stops are not the whole round yet.
    set_completed(&harness.pool, "w-1").await;
    set_completed(&harness.pool, "w-2").await;
    let reply = harness.client.request("hub/watch", watch_all()).await;
    assert_eq!(
        reply["result"]["events"],
        json!([]),
        "a partly stopped round must keep waiting: {reply}"
    );

    set_completed(&harness.pool, "w-3").await;
    let events = wait_for_round(&mut harness.client).await;
    assert_eq!(events.len(), 1, "one event for the whole round: {events:?}");
    let event = &events[0];
    assert_eq!(event["event"], "round", "{event}");
    assert_eq!(event["group"], GROUP, "{event}");
    let workers = event["workers"].as_array().expect("workers array");
    assert_eq!(workers.len(), 3, "every selected worker is listed: {event}");
    for (worker, id) in workers.iter().zip(["w-1", "w-2", "w-3"]) {
        assert_eq!(worker["worker_id"], id, "{event}");
        assert_eq!(worker["outcome"], "completed", "{event}");
        assert_eq!(worker["verified"], true, "{event}");
        assert_eq!(worker["done"], "Fixed.", "{event}");
    }
    // The compact content is the one line per worker an orchestrator reads.
    assert_eq!(
        event["content"],
        json!(
            "w-1 completed verified:yes Fixed.\nw-2 completed verified:yes Fixed.\nw-3 completed verified:yes Fixed."
        ),
        "{event}"
    );

    harness.task.abort();
    let _ = harness.task.await;
    drop(harness.hub);
}

/// A paused worker escalates early: the round answers while its siblings still
/// run, because the question is what needs the orchestrator.
#[tokio::test]
async fn a_paused_worker_returns_the_round_early() {
    let harness = harness(&["w-1", "w-2", "w-3"]).await;
    set_paused(&harness.pool, "w-2", "Which fixture should I use?").await;

    let mut client = harness.client;
    let events = wait_for_round(&mut client).await;
    assert_eq!(events.len(), 1, "{events:?}");
    let event = &events[0];
    assert_eq!(event["status"], "attention", "{event}");
    let workers = event["workers"].as_array().expect("workers array");
    let paused = workers
        .iter()
        .find(|worker| worker["worker_id"] == "w-2")
        .expect("the paused worker is listed");
    assert_eq!(paused["outcome"], "needs_input", "{paused}");
    assert_eq!(paused["done"], "Which fixture should I use?", "{paused}");
    // The siblings were still running when the round answered.
    assert!(
        workers
            .iter()
            .filter(|worker| worker["outcome"] == "running")
            .count()
            == 2,
        "the round answers while the siblings run: {event}"
    );

    harness.task.abort();
    let _ = harness.task.await;
    drop(harness.hub);
}

/// The round acknowledges the workers it folds in, so a later plain watch
/// finds nothing left to replay.
#[tokio::test]
async fn a_reported_round_never_replays() {
    let harness = harness(&["w-1", "w-2", "w-3"]).await;
    for id in ["w-1", "w-2", "w-3"] {
        set_completed(&harness.pool, id).await;
    }

    let mut client = harness.client;
    let events = wait_for_round(&mut client).await;
    assert_eq!(events.len(), 1, "{events:?}");

    // The same `--all` call sees no fresh transition and waits.
    let reply = client.request("hub/watch", watch_all()).await;
    assert_eq!(
        reply["result"]["events"],
        json!([]),
        "a reported round must not replay: {reply}"
    );

    // A plain watch of the same ids is clear too.
    let reply = client
        .request(
            "hub/watch",
            json!({"worker_ids": ["w-1", "w-2", "w-3"], "group": GROUP, "initial": true, "all": false}),
        )
        .await;
    assert_eq!(
        reply["result"]["events"],
        json!([]),
        "the round acknowledged its workers: {reply}"
    );

    harness.task.abort();
    let _ = harness.task.await;
    drop(harness.hub);
}

/// The MCP `watch` action reaches the same consolidated round through
/// `all: true`, with no hub in between.
#[tokio::test]
async fn the_mcp_watch_action_returns_the_round_with_all_true() {
    let isolated = common::IsolatedPool::new(4, "watch-all-mcp");
    for id in ["w-1", "w-2", "w-3"] {
        add_running(&isolated.pool, id).await;
        set_completed(&isolated.pool, id).await;
    }
    let server = McpServer::new(isolated.pool.clone(), "test".to_string());
    let ctx = mini_swe_mcp::mcp::ConnectionContext {
        agent_id: Some(OWNER.to_string()),
        ..mini_swe_mcp::mcp::ConnectionContext::hub_connection(3)
    };

    let result = server
        .execute_tool_for(
            "worker",
            json!({"action": "watch", "all": true, "group": GROUP, "timeout_secs": 5}),
            &ctx,
        )
        .await
        .expect("the round watch must answer");

    assert_eq!(result["status"], "event", "{result}");
    let events = result["events"].as_array().expect("events array");
    assert_eq!(events.len(), 1, "one event for the round: {result}");
    assert_eq!(events[0]["event"], "round", "{result}");
    assert_eq!(
        events[0]["workers"].as_array().map(Vec::len),
        Some(3),
        "{result}"
    );
}

/// The CLI `--all` flag reaches the same round through the no-daemon polling
/// path, over a group whose workers have already stopped.
#[test]
fn the_cli_all_flag_reports_a_stopped_group() {
    let exe = common::binary_path();
    let hub = common::TempDir::new_in_tmp("watch-all-cli-hub");
    let swe = common::TempDir::new_in_tmp("watch-all-cli-swe");
    std::fs::create_dir_all(swe.path().join("swe-registry")).expect("registry dir");
    for id in ["w-1", "w-2", "w-3"] {
        // A terminal row survives only with its worktree, so the round
        // has its preserved branch to read the diff stat from.
        let row = format!(
            r#"{{"id":"{id}","pid":{},"task":"t","model":"m","status":"completed","step":2,"max_turns":10,"last_command":"done","started_at":1,"updated_at":2,"owner":"{OWNER}","group":"{GROUP}"}}"#,
            std::process::id()
        );
        std::fs::write(
            swe.path().join("swe-registry").join(format!("{id}.json")),
            row,
        )
        .expect("row");
        std::fs::create_dir_all(swe.path().join(format!("swe-wt-{id}")))
            .expect("preserved worktree");
    }

    let output = common::binary_command(&exe)
        .args(["watch", "--group", GROUP, "--all", "--json"])
        .env("SWE_HUB_DIR", hub.path())
        .env("SWE_TEMP_DIR", swe.path())
        .env("TMPDIR", swe.path())
        .env("MINI_SWE_NO_DAEMON", "1")
        .env("MINI_SWE_AGENT_ID", OWNER)
        .env("ENV_FILE", "/nonexistent-mini-swe-env")
        .env("OPENAI_API_KEY", "test-key-not-used")
        .env(
            "MODELS_FILE",
            concat!(env!("CARGO_MANIFEST_DIR"), "/models.yaml"),
        )
        .output()
        .expect("run the CLI watch");

    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    assert_eq!(
        output.status.code(),
        Some(0),
        "stdout: {stdout}\nstderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let event: serde_json::Value =
        serde_json::from_str(stdout.lines().next().unwrap_or_default()).expect("round JSON");
    assert_eq!(event["event"], "round", "{stdout}");
    assert_eq!(
        event["workers"].as_array().map(Vec::len),
        Some(3),
        "{stdout}"
    );
}

/// `--all` without a group or ids is refused instead of silently watching
/// every worker the caller owns.
#[test]
fn the_cli_all_flag_needs_a_group_or_ids() {
    let args = [
        "mini-swe-mcp".to_string(),
        "watch".to_string(),
        "--all".to_string(),
    ];
    let error = mini_swe_mcp::cli::watch::Options::parse(&args)
        .err()
        .expect("an unselective --all must be refused");
    assert!(
        error
            .to_string()
            .contains("--all needs --group or explicit worker ids"),
        "{error}"
    );
}

/// Two agents, one group: a round must never report — or
/// acknowledge — a worker that belongs to the other agent.
#[tokio::test]
async fn a_round_stays_inside_its_caller_ownership() {
    let isolated = common::IsolatedPool::new(4, "watch-all-own");
    for id in ["own-1", "other-1"] {
        let owner = if id.starts_with("own-") {
            OWNER
        } else {
            "agent-b"
        };
        pool_add_running_owned(&isolated.pool, id, owner).await;
        set_completed_owned(&isolated.pool, id, owner).await;
    }
    let server = McpServer::new(isolated.pool.clone(), "test".to_string());
    let ctx = mini_swe_mcp::mcp::ConnectionContext {
        agent_id: Some(OWNER.to_string()),
        ..mini_swe_mcp::mcp::ConnectionContext::hub_connection(3)
    };

    let result = server
        .execute_tool_for(
            "worker",
            json!({"action": "watch", "all": true, "group": GROUP, "timeout_secs": 5}),
            &ctx,
        )
        .await
        .expect("the round watch must answer");

    assert_eq!(result["status"], "event", "{result}");
    let events = result["events"].as_array().expect("events array");
    assert_eq!(events.len(), 1, "{result}");
    let workers = events[0]["workers"].as_array().expect("workers");
    assert_eq!(
        workers
            .iter()
            .map(|w| w["worker_id"].as_str())
            .collect::<Vec<_>>(),
        vec![Some("own-1")],
        "only the caller's worker is listed: {result}"
    );

    // The round must not consume the other agent's event: agent-b's
    // plain watch still replays its own worker's completion.
    let other_ctx = mini_swe_mcp::mcp::ConnectionContext {
        agent_id: Some("agent-b".to_string()),
        ..mini_swe_mcp::mcp::ConnectionContext::hub_connection(4)
    };
    let replay = server
        .execute_tool_for(
            "worker",
            json!({"action": "watch", "worker_id": "other-1", "timeout_secs": 5}),
            &other_ctx,
        )
        .await
        .expect("agent-b's watch must answer");
    assert_eq!(replay["status"], "event", "{replay}");
    assert_eq!(
        replay["events"][0]["worker_id"], "other-1",
        "the round must not mark the other agent's worker seen: {replay}"
    );
}

/// Insert `id` as a running worker of `owner` (the harness helper
/// fixes the owner to `OWNER`).
async fn pool_add_running_owned(pool: &WorkerPool, id: &str, owner: &str) {
    let mut meta = meta(id);
    meta.owner = owner.to_string();
    let mut record = record(id, running());
    record.owner = owner.to_string();
    pool.__test_insert_worker(record).await;
    pool.__test_save_status(&meta, "test", RegistryStatus::Running, 1, 10, "ls", None);
}

/// Move `id` — owned by `owner` — to completed, preserving the row.
async fn set_completed_owned(pool: &WorkerPool, id: &str, owner: &str) {
    let mut state = WorkerState::Completed {
        turns: 2,
        diff: String::new(),
        summary: "Owned.".to_string(),
        completed_at: 0,
        artifacts: Vec::new(),
        branch: Some(format!("worker-{id}")),
        verified: Some(false),
        metrics: WorkerMetrics::default(),
        revision: 0,
        report: None,
    };
    if let WorkerState::Completed { summary, .. } = &mut state {
        *summary = format!("Owned by {owner}.");
    }
    pool.__test_set_worker_state(id, state).await;
}
