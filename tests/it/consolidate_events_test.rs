//! A consolidator that steered a worker owns its lifecycle events until the
//! round is over.
//!
//! E14 routes a steered worker's *question* to the consolidator through the
//! steer source; the same rule applies to its completion. While that source
//! names a live consolidator the owner's watch stays quiet, and once the
//! consolidator has finished the stopped worker is visible to its owner again.

use crate::common::IsolatedPool;
use mini_swe_mcp::hub::{HubConfig, HubPaths, HubServer};
use mini_swe_mcp::mcp::McpServer;
use mini_swe_mcp::pool::{
    RegistryStatus, WorkerMeta, WorkerPool, WorkerRole, save_registry_entry_in,
};
use mini_swe_mcp::worktree::ScratchRoot;
use serde_json::json;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;

/// The owner of every worker in this test.
const OWNER: &str = "owner";

/// The registry row a worker of `role` in the round leaves behind.
fn write_row(root: &ScratchRoot, id: &str, role: WorkerRole, status: RegistryStatus) {
    let meta = WorkerMeta {
        task: "exercise consolidate routing".to_string(),
        group: Some("round-1".to_string()),
        role,
        ..WorkerMeta::test_meta(id, OWNER)
    };
    save_registry_entry_in(
        root,
        &meta.entry("test-model", status, 3, 10, "cargo test", None),
    );
}

/// The steer source that `consolidate_steer` writes for a steered worker.
///
/// The file is what production reads back; writing it directly keeps this test
/// to the routing rule and out of a real consolidator dispatch.
fn write_steer_source(root: &ScratchRoot, worker: &str, consolidator: &str) {
    let source = serde_json::json!({"consolidator": consolidator, "round_base": null});
    std::fs::write(
        root.join(format!("swe-wt-{worker}.steer-source")),
        serde_json::to_vec(&source).unwrap(),
    )
    .unwrap();
}

/// The `hub/watch` parameters the owner's shell watch polls with: the same
/// wire the `watch` action used to sit on, and the one that still delivers.
fn watch_args(worker: &str) -> serde_json::Value {
    json!({"worker_ids": [worker], "group": [], "initial": true, "all": false})
}

/// A minimal JSON-RPC client, so this test polls the same `hub/watch` frames
/// the shell watch does without spawning a binary.
struct Watch {
    reader: BufReader<tokio::net::unix::OwnedReadHalf>,
    writer: tokio::net::unix::OwnedWriteHalf,
    next_id: u64,
}

impl Watch {
    async fn connect(socket: &std::path::Path) -> Self {
        let (reader, writer) = UnixStream::connect(socket)
            .await
            .expect("connect to the test daemon")
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
            assert!(
                self.reader.read_line(&mut line).await.expect("read") > 0,
                "the daemon closed the connection"
            );
            let reply: serde_json::Value = serde_json::from_str(line.trim()).expect("JSON");
            if reply.get("id") == Some(&json!(id)) {
                return reply;
            }
        }
    }
}

#[tokio::test]
async fn a_steered_completion_waits_for_the_consolidator_to_finish() {
    let harness = IsolatedPool::new(2, "consolidate-events");
    let root = harness.root();
    // A terminal registry row survives the loader only while its worktree
    // still exists; the event router is the reader that keeps it.
    std::fs::create_dir_all(root.join("swe-wt-w1")).unwrap();
    write_row(&root, "w1", WorkerRole::Worker, RegistryStatus::Completed);
    write_row(
        &root,
        "c1",
        WorkerRole::Consolidate,
        RegistryStatus::Running,
    );
    write_steer_source(&root, "w1", "c1");

    let server = McpServer::new(WorkerPool::clone(&harness.pool), "test-model".into());
    let hub = crate::common::TempDir::new_in_tmp("consolidate-events-hub");
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(hub.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
    }
    let socket = HubPaths::new(hub.path().to_path_buf()).socket();
    let daemon = HubServer::new(
        Arc::new(server),
        HubConfig::new(HubPaths::new(hub.path().to_path_buf()), 60),
    );
    let task = tokio::spawn(async move {
        let _ = daemon.run().await;
    });
    for _ in 0..100 {
        if UnixStream::connect(&socket).await.is_ok() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    let mut watch = Watch::connect(&socket).await;
    watch.request("hub/hello", json!({"agent_id": OWNER})).await;

    let hidden = watch.request("hub/watch", watch_args("w1")).await;
    assert!(
        !hidden.to_string().contains("completed"),
        "a completion steered to a live consolidator must not wake the owner: {hidden}"
    );

    // The consolidator finished, so its steer source names no live owner and
    // normal delivery resumes: the stopped worker is visible again.
    write_row(
        &root,
        "c1",
        WorkerRole::Consolidate,
        RegistryStatus::Completed,
    );
    let shown = watch.request("hub/watch", watch_args("w1")).await;
    assert!(
        shown.to_string().contains("completed"),
        "the stopped worker must be visible once its consolidator is gone: {shown}"
    );

    task.abort();
    let _ = task.await;
}
