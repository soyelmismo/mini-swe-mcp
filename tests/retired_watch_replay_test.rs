//! A retired worker's event is never replayed, whatever the owner saw (events-F7).
//!
//! The reported failure: a consolidator's completion was delivered by `watch`
//! and read; the orchestrator then merged it, which retired the worker, its
//! branch, its registry row, its history and its watch acks; the next `watch`
//! replayed the same completion under "While you were not watching".
//!
//! The window is a snapshot read before the router's lock. A `watch` that
//! resolved its snapshot, then a retirement that landed in between, then the
//! `observe_watch` of the now-stale snapshot: the router saw a completed worker
//! it had no acknowledged position for, because the retirement dropped the
//! position with the worker, and queued the completion again. So this test
//! races a real `watch` against a real retirement and asserts the invariant the
//! race can break: once the worker is retired, no completion of it is delivered
//! again, whether it was acknowledged before or never read at all.
//!
//! `discard` is the retirement `merge` performs -- both go through the one
//! `retire_and_forget` that forgets the worker's replay state -- and unlike a
//! merge it needs no repository, so the test states the event contract without a
//! git fixture. The hub, the pool and the socket are given their own temporary
//! location and passed to the code under test, so nothing here can see or retire
//! another test's state.

mod common;

use mini_swe_mcp::hub::{HubConfig, HubPaths, HubServer};
use mini_swe_mcp::manifest::ModelManifest;
use mini_swe_mcp::mcp::McpServer;
use mini_swe_mcp::pool::{
    LogBuffer, WorkerMetrics, WorkerPool, WorkerRecord, WorkerState,
};
use mini_swe_mcp::worktree::ScratchRoot;
use serde_json::json;
use std::path::Path;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;

fn paths(dir: &Path) -> HubPaths {
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700)).expect("0700");
    HubPaths::new(dir.to_path_buf())
}

fn completed(id: &str) -> WorkerState {
    WorkerState::Completed {
        turns: 1,
        diff: String::new(),
        summary: "consolidated the round".to_string(),
        completed_at: 0,
        artifacts: Vec::new(),
        branch: Some(format!("worker-{id}")),
        verified: Some(true),
        metrics: WorkerMetrics::default(),
        revision: 0,
        report: None,
    }
}

fn record(id: &str, owner: &str) -> WorkerRecord {
    WorkerRecord {
        id: id.to_string(),
        task: "retired replay probe".to_string(),
        model: "test".to_string(),
        owner: owner.to_string(),
        state: completed(id),
        metrics: WorkerMetrics::default(),
        logs: LogBuffer::new(),
        pending_steer: Vec::new(),
        resume_tx: None,
        handle: None,
        revision: 0,
    }
}

async fn pool_with(id: &str, owner: &str, scratch: &Path) -> Arc<McpServer> {
    let pool = WorkerPool::with_scratch(
        4,
        "http://localhost:1".to_string(),
        "test-key".to_string(),
        ScratchRoot::new(scratch),
    )
    .with_manifest(Arc::new(ModelManifest::default()));
    pool.__test_insert_worker(record(id, owner)).await;
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

/// A hub connection speaking JSON-RPC over the unix socket.
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

    /// The owner establishes its identity, so a watch is scoped to it.
    async fn hello(&mut self, agent: &str) {
        self.request("hub/hello", json!({"agent_id": agent})).await;
    }

    async fn watch(&mut self) -> serde_json::Value {
        self.request(
            "hub/watch",
            json!({"worker_ids": [], "group": null, "initial": true}),
        )
        .await
    }

    async fn discard(&mut self, id: &str) -> serde_json::Value {
        self.request(
            "tools/call",
            json!({"name": "worker", "arguments": {"action": "discard", "worker_id": id}}),
        )
        .await
    }
}

/// The events a watch answered with.
fn events(reply: &serde_json::Value) -> &[serde_json::Value] {
    reply["result"]["events"]
        .as_array()
        .unwrap_or_else(|| panic!("a watch answer carries events: {reply:?}"))
}

/// The completions a watch delivered, by worker id.
fn completions(reply: &serde_json::Value) -> Vec<String> {
    events(reply)
        .iter()
        .filter(|event| event["event"] == "completed")
        .map(|event| event["worker_id"].as_str().unwrap_or("").to_string())
        .collect()
}

/// A completed worker is delivered once, then acknowledged: a later watch is
/// silent while the worker is still there. This is the control that makes the
/// two tests below assertions about retirement rather than about watching.
#[tokio::test]
async fn an_acknowledged_completion_of_a_live_worker_is_not_replayed() {
    let hub = common::TempDir::new_in_tmp("f7-live-hub");
    let scratch = common::TempDir::new_in_tmp("f7-live-pool");
    let socket = HubPaths::new(hub.path().to_path_buf()).socket();

    let server = pool_with("c4f708d1", "agent-f7", scratch.path()).await;
    let daemon = HubServer::new(server, HubConfig::new(paths(hub.path()), 60));
    let task = tokio::spawn(async move { daemon.run().await });
    wait_for_socket(&socket).await;

    let mut owner = Raw::connect(&socket).await;
    owner.hello("agent-f7").await;
    let first = owner.watch().await;
    assert_eq!(
        completions(&first),
        vec!["c4f708d1".to_string()],
        "the completion is delivered once: {first:?}"
    );
    owner
        .request(
            "hub/watch/ack",
            json!({"sequence": events(&first)[0]["sequence"]}),
        )
        .await;

    let second = owner.watch().await;
    assert!(
        completions(&second).is_empty(),
        "an acknowledged event of a live worker is silent: {second:?}"
    );
    task.abort();
    let _ = task.await;
}

/// The reported failure, end to end and under the race that caused it: watches
/// and the retirement run at the same time, and once the worker is retired no
/// completion of it may be delivered again.
///
/// The first watch establishes the delivered-and-acknowledged state of the
/// report. The retirement then races a burst of watches, which is how the stale
/// snapshot reaches the router: each of them resolved its view before the
/// retirement and observed it after. Every answer from the race on is checked,
/// because the replay this forbids is delivered by whichever watch observes the
/// stale snapshot -- not necessarily the one after the retirement returned.
#[tokio::test]
async fn an_acknowledged_completion_is_not_replayed_after_the_worker_is_retired() {
    let hub = common::TempDir::new_in_tmp("f7-hub");
    let scratch = common::TempDir::new_in_tmp("f7-pool");
    let socket = HubPaths::new(hub.path().to_path_buf()).socket();

    let server = pool_with("c4f708d1", "agent-f7", scratch.path()).await;
    let daemon = HubServer::new(server, HubConfig::new(paths(hub.path()), 60));
    let task = tokio::spawn(async move { daemon.run().await });
    wait_for_socket(&socket).await;

    let mut owner = Raw::connect(&socket).await;
    owner.hello("agent-f7").await;
    let first = owner.watch().await;
    assert_eq!(
        completions(&first),
        vec!["c4f708d1".to_string()],
        "the completion is delivered once: {first:?}"
    );
    owner
        .request(
            "hub/watch/ack",
            json!({"sequence": events(&first)[0]["sequence"]}),
        )
        .await;

    // Watches racing the retirement, each on its own connection: an identity
    // holds one watch slot, so a single connection would be refused while it
    // holds one. These are the calls whose snapshot can outlive the retirement.
    let racers: Vec<_> = (0..4)
        .map(|_| {
            let socket = socket.clone();
            async move {
                let mut racer = Raw::connect(&socket).await;
                racer.hello("agent-f7").await;
                racer
            }
        })
        .collect();
    let mut racers = futures_join_all(racers).await;

    let retired = owner.discard("c4f708d1").await;
    assert!(
        retired["result"].is_object(),
        "the retirement must succeed: {retired:?}"
    );

    // Whichever racer observed the snapshot taken before the retirement, the
    // completion must not come back from any of them.
    for racer in &mut racers {
        let reply = racer.watch().await;
        assert!(
            completions(&reply).is_empty(),
            "a retired worker is never replayed: {reply:?}"
        );
    }
    let after = owner.watch().await;
    assert!(
        completions(&after).is_empty(),
        "a retired worker is never replayed: {after:?}"
    );
    task.abort();
    let _ = task.await;
}

/// A worker retired before its owner read the event is gone by design: the
/// branch is in the base branch, so there is nothing left to review and the
/// event must not be queued for whoever watches next.
#[tokio::test]
async fn an_unacknowledged_completion_is_not_replayed_after_retirement() {
    let hub = common::TempDir::new_in_tmp("f7-open-hub");
    let scratch = common::TempDir::new_in_tmp("f7-open-pool");
    let socket = HubPaths::new(hub.path().to_path_buf()).socket();

    let server = pool_with("d6eaf62a", "agent-f7", scratch.path()).await;
    let daemon = HubServer::new(server, HubConfig::new(paths(hub.path()), 60));
    let task = tokio::spawn(async move { daemon.run().await });
    wait_for_socket(&socket).await;

    let mut owner = Raw::connect(&socket).await;
    owner.hello("agent-f7").await;
    // No watch and no acknowledgment: the event is queued and never read.
    let retired = owner.discard("d6eaf62a").await;
    assert!(
        retired["result"].is_object(),
        "the retirement must succeed: {retired:?}"
    );

    let watch = owner.watch().await;
    assert!(
        completions(&watch).is_empty(),
        "an unacknowledged event of a retired worker is gone: {watch:?}"
    );
    task.abort();
    let _ = task.await;
}
