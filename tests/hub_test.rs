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
        let id = self.next_id;
        self.next_id += 1;
        let frame = serde_json::json!({"jsonrpc": "2.0", "id": id, "method": method});
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
                return reply["result"].clone();
            }
        }
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
    HubPaths::for_test(dir.to_path_buf())
}
