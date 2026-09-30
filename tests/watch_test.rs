//! `watch` integration: immediate event, timeout, nothing-to-watch, governance.

mod common;

use mini_swe_mcp::hub::{HubConfig, HubPaths, HubServer};
use mini_swe_mcp::manifest::ModelManifest;
use mini_swe_mcp::mcp::McpServer;
use mini_swe_mcp::pool::{LogBuffer, WorkerMetrics, WorkerPool, WorkerRecord, WorkerState};
use std::path::Path;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;

fn paths(dir: &Path) -> HubPaths {
    HubPaths::new(dir.to_path_buf())
}

fn record(id: &str, owner: &str, state: WorkerState) -> WorkerRecord {
    WorkerRecord {
        id: id.to_string(),
        task: "watch probe".to_string(),
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

async fn wait_for_socket(path: &Path) {
    for _ in 0..100 {
        if UnixStream::connect(path).await.is_ok() {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    panic!("hub socket never came up");
}

struct Raw {
    reader: BufReader<tokio::net::unix::OwnedReadHalf>,
    writer: tokio::net::unix::OwnedWriteHalf,
    next_id: u64,
}

impl Raw {
    async fn connect(socket: &Path) -> Self {
        let (reader, writer) = UnixStream::connect(socket).await.expect("connect").into_split();
        Self { reader: BufReader::new(reader), writer, next_id: 1 }
    }

    async fn request(&mut self, method: &str, params: serde_json::Value) -> serde_json::Value {
        let id = self.next_id;
        self.next_id += 1;
        self.writer
            .write_all(format!("{}\n", serde_json::json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params})).as_bytes())
            .await
            .expect("write");
        self.writer.flush().await.expect("flush");
        loop {
            let mut line = String::new();
            self.reader.read_line(&mut line).await.expect("read");
            let reply: serde_json::Value = serde_json::from_str(line.trim()).expect("JSON");
            if reply.get("id") == Some(&serde_json::json!(id)) {
                return reply;
            }
        }
    }

    async fn cli(&mut self, command: &[&str]) -> serde_json::Value {
        self.request("hub/hello", serde_json::json!({})).await;
        self.request("initialize", serde_json::json!({"protocolVersion": "2024-11-05", "capabilities": {},
            "clientInfo": {"name": mini_swe_mcp::mcp::CLI_CLIENT_NAME, "version": "test"}})).await;
        let reply = self
            .request("tools/call", serde_json::json!({"name": "worker", "arguments": {"action": command[0]}}))
            .await;
        let text = reply["result"]["content"][0]["text"].as_str().unwrap_or("").to_string();
        serde_json::from_str(&text).unwrap_or(serde_json::json!({}))
    }
}

async fn pool_with(records: Vec<WorkerRecord>) -> Arc<McpServer> {
    let pool = WorkerPool::new(4, "http://localhost:1".to_string(), "test-key".to_string())
        .with_manifest(Arc::new(ModelManifest::default()));
    for record in records {
        pool.__test_insert_worker(record).await;
    }
    Arc::new(McpServer::new(pool, "test".to_string()))
}

#[tokio::test]
async fn agent_b_cannot_watch_agent_a_worker_and_missed_events_replay_to_owner() {
    let _ = ();
    let dir = common::TempDir::new_in_tmp("watch-governance");
    let daemon = HubServer::new(
        pool_with(vec![record("w-watch", "agent-a", WorkerState::Running {
            step: 1,
            last_command: "test".to_string(),
            started_at: 0,
        })]),
        HubConfig::new(paths(dir.path()), 60),
    );
    let task = tokio::spawn(async move { daemon.run().await });
    wait_for_socket(&dir.path().join("hub.sock")).await;

    let mut a = Raw::connect(&dir.path().join("hub.sock")).await;
    let reply = a
        .request("hub/watch", serde_json::json!({"worker_ids": ["w-watch"], "group": null, "initial": true}))
        .await;
    assert_eq!(reply["error"]["message"], "worker w-watch belongs to agent agent-a");
    drop(a);

    let mut owner = Raw::connect(&dir.path().join("hub.sock")).await;
    owner.request("hub/hello", serde_json::json!({"agent_id": "agent-a"})).await;
    let reply = owner
        .request("hub/watch", serde_json::json!({"worker_ids": ["w-watch"], "group": null, "initial": true}))
        .await;
    assert!(reply["result"]["watching"].as_array().unwrap().contains(&serde_json::json!("w-watch")), "{reply:?}");

    let mut b = Raw::connect(&dir.path().join("hub.sock")).await;
    b.request("hub/hello", serde_json::json!({"agent_id": "agent-b"})).await;
    let reply = b
        .request("hub/watch", serde_json::json!({"worker_ids": [], "group": null, "initial": true}))
        .await;
    assert_eq!(reply["result"]["watching"], serde_json::json!([]));
    assert_eq!(reply["result"]["events"], serde_json::json!([]));

    let mut admin = Raw::connect(&dir.path().join("hub.sock")).await;
    admin.request("hub/hello", serde_json::json!({"agent_id": "agent-b", "admin": true})).await;
    let reply = admin
        .request("hub/watch", serde_json::json!({"worker_ids": ["w-watch"], "group": null, "initial": true}))
        .await;
    assert!(reply.get("error").is_none(), "{reply:?}");
    assert!(reply["result"]["watching"].as_array().unwrap().contains(&serde_json::json!("w-watch")), "{reply:?}");

    task.abort();
    let _ = task.await;
}

#[tokio::test]
async fn completed_worker_is_reported_immediately_with_missed_marker() {
    let dir = common::TempDir::new_in_tmp("watch-missed");
    let daemon = HubServer::new(
        pool_with(vec![record("w-done", "agent-a", WorkerState::Completed {
            turns: 2,
            diff: String::new(),
            summary: "Fixed.".to_string(),
            completed_at: 0,
            artifacts: Vec::new(),
            branch: Some("worker-w-done".to_string()),
            verified: Some(true),
            metrics: WorkerMetrics::default(),
            revision: 0,
        })]),
        HubConfig::new(paths(dir.path()), 60),
    );
    let task = tokio::spawn(async move { daemon.run().await });
    wait_for_socket(&dir.path().join("hub.sock")).await;

    let mut owner = Raw::connect(&dir.path().join("hub.sock")).await;
    owner.request("hub/hello", serde_json::json!({"agent_id": "agent-a"})).await;
    // Let the daemon's 1 s watch loop observe the terminal worker first.
    tokio::time::sleep(std::time::Duration::from_millis(1500)).await;
    let reply = owner
        .request("hub/watch", serde_json::json!({"worker_ids": [], "group": null, "initial": true}))
        .await;
    let event = &reply["result"]["events"][0];
    assert_eq!(event["worker_id"], "w-done");
    assert_eq!(event["event"], "completed");
    assert_eq!(event["missed"], true);
    assert_eq!(event["branch"], "worker-w-done");
    assert!(event["actions"].as_array().map(|actions| actions.iter().any(|action| action.as_str().unwrap_or("").contains("mini-swe-mcp steer"))).unwrap_or(false), "{event:?}");
    assert!(event["next_step"].as_str().unwrap_or("").contains("worker-w-done"), "{event:?}");

    // Acknowledging removes the backlog; the next watch has nothing to replay.
    owner.request("hub/watch/ack", serde_json::json!({"sequence": event["sequence"]})).await;
    let reply = owner
        .request("hub/watch", serde_json::json!({"worker_ids": ["w-done"], "group": null, "initial": false}))
        .await;
    assert_eq!(reply["result"]["events"], serde_json::json!([]));

    let mut b = Raw::connect(&dir.path().join("hub.sock")).await;
    b.request("hub/hello", serde_json::json!({"agent_id": "agent-b"})).await;
    let reply = b
        .request("hub/watch", serde_json::json!({"worker_ids": [], "group": null, "initial": true}))
        .await;
    assert_eq!(reply["result"]["events"], serde_json::json!([]), "A must never replay to B");

    task.abort();
    let _ = task.await;
}

#[test]
fn watch_cli_exits_2_on_timeout_and_3_when_nothing_to_watch() {
    let exe = common::binary_path();
    let hub = common::TempDir::new_in_tmp("watch-cli-hub");
    let swe = common::TempDir::new_in_tmp("watch-cli-swe");
    std::fs::create_dir_all(swe.path().join("swe-registry")).expect("registry dir");
    let run = |args: &[&str]| {
        std::process::Command::new(&exe)
            .args(args)
            .env("SWE_HUB_DIR", hub.path())
            .env("SWE_TEMP_DIR", swe.path())
            .env("TMPDIR", swe.path())
            .env("MINI_SWE_NO_DAEMON", "1")
            .env("ENV_FILE", "/nonexistent-mini-swe-env")
            .env("MODELS_FILE", concat!(env!("CARGO_MANIFEST_DIR"), "/models.yaml"))
            .output()
            .unwrap_or_else(|e| panic!("run {args:?}: {e}"))
    };
    let output = run(&["watch", "--timeout", "1"]);
    assert_eq!(output.status.code(), Some(3), "{}", String::from_utf8_lossy(&output.stdout));
    let completed = format!(r#"{{"id":"w-cli","pid":{},"task":"t","model":"m","status":"completed","step":2,"max_turns":10,"last_command":"done","started_at":1,"updated_at":2}}"#, std::process::id());
    std::fs::write(swe.path().join("swe-registry").join("w-cli.json"), completed).expect("row");
    let output = run(&["watch", "w-cli"]);
    assert_eq!(output.status.code(), Some(0), "{}", String::from_utf8_lossy(&output.stderr));
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    assert!(stdout.contains("w-cli") && stdout.contains("completed") && stdout.contains("mini-swe-mcp steer"), "{stdout}");
    let _ = std::fs::remove_dir_all(swe.path().join("swe-registry"));
    let output = run(&["watch", "w-cli", "--timeout", "1"]);
    assert_eq!(output.status.code(), Some(3), "{}", String::from_utf8_lossy(&output.stdout));
}

#[test]
fn tool_description_carries_the_orchestrator_guidelines() {
    let manifest = ModelManifest::default();
    let server = McpServer::new(
        WorkerPool::new(1, "http://localhost:1".to_string(), "k".to_string()).with_manifest(Arc::new(manifest)),
        "m".to_string(),
    );
    let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build().expect("runtime");
    let text = serde_json::to_string(&server.tools_list()).expect("list");
    for needle in ["ONE focused concern", "mini-swe-mcp watch", "steer", "merge only when it is right"] {
        assert!(text.contains(needle), "tool schema must carry the guidelines ({needle} missing)");
    }
}
