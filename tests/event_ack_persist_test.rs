//! Persisted per-owner watch acks: a daemon restart must not replay an event
//! the owner already acknowledged, and a worker whose branch is merged into its
//! base or already gone must never be replayed.

mod common;

use mini_swe_mcp::hub::{HubConfig, HubPaths, HubServer};
use mini_swe_mcp::manifest::ModelManifest;
use mini_swe_mcp::mcp::McpServer;
use mini_swe_mcp::pool::{
    LogBuffer, RegistryStatus, WorkerMetrics, WorkerPool, WorkerRecord, WorkerRegistryEntry,
    WorkerState, save_registry_entry_in,
};
use mini_swe_mcp::worktree::ScratchRoot;
use serde_json::json;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;

fn paths(dir: &Path) -> HubPaths {
    std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700)).expect("0700");
    HubPaths::new(dir.to_path_buf())
}

fn record(id: &str, owner: &str, state: WorkerState) -> WorkerRecord {
    WorkerRecord {
        id: id.to_string(),
        task: "ack persist probe".to_string(),
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

fn completed(id: &str, summary: &str) -> WorkerState {
    WorkerState::Completed {
        turns: 1,
        diff: String::new(),
        summary: summary.to_string(),
        completed_at: 0,
        artifacts: Vec::new(),
        branch: Some(format!("worker-{id}")),
        verified: Some(true),
        metrics: WorkerMetrics::default(),
        revision: 0,
        report: None
    verdicts: None,,
    }
}

async fn pool_with(records: Vec<WorkerRecord>, scratch: &Path) -> Arc<McpServer> {
    let pool = WorkerPool::with_scratch(
        4,
        "http://localhost:1".to_string(),
        "test-key".to_string(),
        ScratchRoot::new(scratch),
    )
    .with_manifest(Arc::new(ModelManifest::default()));
    for record in records {
        pool.__test_insert_worker(record).await;
    }
    Arc::new(McpServer::new(pool, "test".to_string()))
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
            self.reader.read_line(&mut line).await.expect("read");
            let reply: serde_json::Value = serde_json::from_str(line.trim()).expect("JSON");
            if reply.get("id") == Some(&json!(id)) {
                return reply;
            }
        }
    }
}

async fn hello_and_watch(socket: &Path, agent: &str) -> serde_json::Value {
    let mut owner = Raw::connect(socket).await;
    owner.request("hub/hello", json!({"agent_id": agent})).await;
    owner
        .request(
            "hub/watch",
            json!({"worker_ids": [], "group": null, "initial": true}),
        )
        .await
}

/// A published acknowledgment survives the daemon that recorded it: the same
/// worker is silent after a restart instead of replaying as "missed".
#[tokio::test]
async fn acknowledged_event_is_not_replayed_after_restart() {
    let hub = common::TempDir::new_in_tmp("ack-hub");
    let scratch = common::TempDir::new_in_tmp("ack-pool");
    let socket = HubPaths::new(hub.path().to_path_buf()).socket();

    let server = pool_with(
        vec![record("w-ack", "agent-ack", completed("w-ack", "done"))],
        scratch.path(),
    )
    .await;
    let daemon = HubServer::new(server, HubConfig::new(paths(hub.path()), 60));
    let task = tokio::spawn(async move { daemon.run().await });
    wait_for_socket(&socket).await;

    let mut owner = Raw::connect(&socket).await;
    owner
        .request("hub/hello", json!({"agent_id": "agent-ack"}))
        .await;
    let reply = owner
        .request(
            "hub/watch",
            json!({"worker_ids": [], "group": null, "initial": true}),
        )
        .await;
    let event = &reply["result"]["events"][0];
    assert_eq!(event["worker_id"], "w-ack", "{reply:?}");
    assert_eq!(event["event"], "completed", "{reply:?}");
    owner
        .request("hub/watch/ack", json!({"sequence": event["sequence"]}))
        .await;
    task.abort();
    let _ = task.await;

    // Persisted 0600, named after the hub directory.
    let store = hub.path().join("watch_acks.json");
    let mode = std::fs::metadata(&store)
        .expect("acknowledging must persist the store")
        .permissions()
        .mode()
        & 0o777;
    assert_eq!(
        mode, 0o600,
        "the ack store must not be group/world readable"
    );

    // A restart sees the same worker; the persisted ack must keep it silent.
    let server = pool_with(
        vec![record("w-ack", "agent-ack", completed("w-ack", "done"))],
        scratch.path(),
    )
    .await;
    let daemon = HubServer::new(server, HubConfig::new(paths(hub.path()), 60));
    let task = tokio::spawn(async move { daemon.run().await });
    wait_for_socket(&socket).await;
    let reply = hello_and_watch(&socket, "agent-ack").await;
    assert_eq!(
        reply["result"]["events"],
        json!([]),
        "an acknowledged event must not replay after a restart: {reply:?}"
    );
    task.abort();
    let _ = task.await;
}

/// The control for the test above: without an acknowledgment the terminal
/// event is still replayed after a restart.
#[tokio::test]
async fn unacknowledged_event_is_replayed_after_restart() {
    let hub = common::TempDir::new_in_tmp("noack-hub");
    let scratch = common::TempDir::new_in_tmp("noack-pool");
    let socket = HubPaths::new(hub.path().to_path_buf()).socket();

    let server = pool_with(
        vec![record("w-open", "agent-noack", completed("w-open", "done"))],
        scratch.path(),
    )
    .await;
    let daemon = HubServer::new(server, HubConfig::new(paths(hub.path()), 60));
    let task = tokio::spawn(async move { daemon.run().await });
    wait_for_socket(&socket).await;
    let reply = hello_and_watch(&socket, "agent-noack").await;
    assert_eq!(reply["result"]["events"][0]["worker_id"], "w-open");
    task.abort();
    let _ = task.await;

    let server = pool_with(
        vec![record("w-open", "agent-noack", completed("w-open", "done"))],
        scratch.path(),
    )
    .await;
    let daemon = HubServer::new(server, HubConfig::new(paths(hub.path()), 60));
    let task = tokio::spawn(async move { daemon.run().await });
    wait_for_socket(&socket).await;
    let reply = hello_and_watch(&socket, "agent-noack").await;
    assert_eq!(
        reply["result"]["events"][0]["worker_id"], "w-open",
        "an unacknowledged event must still be replayed: {reply:?}"
    );
    task.abort();
    let _ = task.await;
}

fn init_repo(dir: &Path) {
    common::git(dir, &["init", "-b", "main"]);
    common::git(dir, &["config", "user.name", "test"]);
    common::git(dir, &["config", "user.email", "test@localhost"]);
    std::fs::write(dir.join("file.txt"), "base\n").unwrap();
    common::git(dir, &["add", "."]);
    common::git(dir, &["commit", "-m", "base"]);
}

fn completed_row(id: &str, owner: &str, repo: &Path) -> WorkerRegistryEntry {
    let mut row = WorkerRegistryEntry::test_row(id, owner);
    row.status = RegistryStatus::Completed;
    row.repo_path = Some(repo.to_string_lossy().into_owned());
    row.base_branch = Some("main".to_string());
    row.verified = Some(true);
    row
}

/// A worker whose branch was merged into base or deleted has nothing left to
/// review; only the still-open branch is replayed.
#[tokio::test]
async fn merged_or_gone_worker_event_is_not_replayed() {
    let hub = common::TempDir::new_in_tmp("merge-hub");
    let scratch = common::TempDir::new_in_tmp("merge-pool");
    let repo = common::TempDir::new_in_tmp("merge-repo");
    init_repo(repo.path());

    // w-open: a branch with work the base does not have.
    common::git(repo.path(), &["checkout", "-b", "worker-w-open"]);
    std::fs::write(repo.path().join("work.txt"), "work\n").unwrap();
    common::git(repo.path(), &["add", "."]);
    common::git(repo.path(), &["commit", "-m", "worker work"]);
    common::git(repo.path(), &["checkout", "main"]);
    // w-merged: the branch still exists but is already contained in main.
    common::git(repo.path(), &["branch", "worker-w-merged", "main"]);
    // w-gone: the branch was deleted, as merge cleanup does.
    common::git(repo.path(), &["branch", "worker-w-gone", "main"]);
    common::git(repo.path(), &["branch", "-D", "worker-w-gone"]);

    let server = pool_with(vec![], scratch.path()).await;
    for id in ["w-open", "w-merged", "w-gone"] {
        save_registry_entry_in(
            server.pool().scratch_root(),
            &completed_row(id, "agent-m", repo.path()),
        );
    }
    let daemon = HubServer::new(server, HubConfig::new(paths(hub.path()), 60));
    let task = tokio::spawn(async move { daemon.run().await });
    let socket = HubPaths::new(hub.path().to_path_buf()).socket();
    wait_for_socket(&socket).await;

    let reply = hello_and_watch(&socket, "agent-m").await;
    let ids: Vec<&str> = reply["result"]["events"]
        .as_array()
        .expect("events array")
        .iter()
        .filter_map(|event| event["worker_id"].as_str())
        .collect();
    assert!(
        ids.contains(&"w-open"),
        "open branch must replay: {reply:?}"
    );
    assert!(
        !ids.contains(&"w-merged"),
        "merged branch must not replay: {reply:?}"
    );
    assert!(
        !ids.contains(&"w-gone"),
        "gone branch must not replay: {reply:?}"
    );
    task.abort();
    let _ = task.await;
}
