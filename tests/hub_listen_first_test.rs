//! Hub startup: the daemon serves connections before slow recovery finishes.
//!
//! It used to run `recover_orphaned_workers` (a git salvage of every orphaned
//! worktree) before it bound `hub.sock`, so a client that auto-started it gave
//! up after five seconds. It now binds and accepts first, recovers
//! concurrently, and holds worker-state requests behind a recovery gate until
//! the pool is recovered.

mod common;

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;

/// How long the recovery test hook holds recovery open.
///
/// Long enough that a served handshake and the `listening` log line provably
/// happen before recovery ends, and short enough to keep the test quick.
const RECOVERY_DELAY_MS: u64 = 4000;

/// The socket both this test and a client of `hub` dial.
fn socket_for(hub: &Path) -> PathBuf {
    mini_swe_mcp::hub::HubPaths::new(hub.to_path_buf()).socket()
}

/// Wait until `path` accepts a connection, or panic.
async fn wait_for_socket(path: &Path) {
    for _ in 0..200 {
        if UnixStream::connect(path).await.is_ok() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    panic!("hub socket {} never came up", path.display());
}

/// A minimal JSON-RPC client over the hub socket.
struct Client {
    reader: BufReader<tokio::net::unix::OwnedReadHalf>,
    writer: tokio::net::unix::OwnedWriteHalf,
    next_id: u64,
}

impl Client {
    async fn connect(socket: &Path) -> Self {
        let stream = UnixStream::connect(socket)
            .await
            .expect("connect to the hub socket");
        let (reader, writer) = stream.into_split();
        Self {
            reader: BufReader::new(reader),
            writer,
            next_id: 1,
        }
    }

    /// Send one request and return the whole reply envelope.
    async fn request(&mut self, method: &str, params: serde_json::Value) -> serde_json::Value {
        let id = self.next_id;
        self.next_id += 1;
        let frame =
            serde_json::json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params});
        self.writer
            .write_all(format!("{frame}\n").as_bytes())
            .await
            .expect("write frame");
        self.writer.flush().await.expect("flush frame");
        loop {
            let mut line = String::new();
            self.reader.read_line(&mut line).await.expect("read reply");
            assert!(!line.is_empty(), "hub closed the connection");
            let reply: serde_json::Value =
                serde_json::from_str(line.trim()).expect("reply is JSON");
            // Worker-event notifications carry no `id`; the reply carries ours.
            if reply.get("id") == Some(&serde_json::json!(id)) {
                return reply;
            }
        }
    }
}

/// A client that connects while recovery is still running gets its handshake
/// answered, and its first worker-state request sees the recovered pool.
#[tokio::test]
async fn a_client_is_served_before_startup_recovery_finishes() {
    let root = common::TempDir::new_in_tmp("lsn-first");
    let hub = root.subdir("hub");
    let swe = root.subdir("swe");
    {
        use std::os::unix::fs::PermissionsExt;
        for dir in [&hub, &swe] {
            std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700))
                .expect("restrict scratch dir");
        }
    }

    // A dead predecessor left one running row behind, so recovery has a row to
    // rewrite; the id is owned by the client identity below so `list` shows it.
    let wid = "orphan1";
    let agent = "listen-first";
    let mut dead = std::process::Command::new("true")
        .spawn()
        .expect("spawn a short-lived owner");
    let dead_pid = dead.id();
    dead.wait().expect("reap the owner");
    assert!(!mini_swe_mcp::worktree::is_process_alive(dead_pid));
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock before the epoch")
        .as_secs();
    let row = serde_json::json!({
        "id": wid, "pid": dead_pid, "task": "t", "model": "ninja", "status": "running",
        "step": 1, "max_turns": 10, "last_command": "cargo test",
        "started_at": now - 60, "updated_at": now - 60,
        "owner": agent,
    });
    std::fs::create_dir_all(swe.join("swe-registry")).expect("create the registry");
    std::fs::write(
        swe.join("swe-registry").join(format!("{wid}.json")),
        row.to_string(),
    )
    .expect("write the orphan row");

    let mut command = tokio::process::Command::new(common::binary_path());
    command
        .arg("daemon")
        .env("SWE_HUB_DIR", &hub)
        .env("SWE_TEMP_DIR", &swe)
        .env("TMPDIR", &swe)
        .env("HUB_IDLE_SECS", "60")
        // Auto-resume off, so the row keeps the status recovery gave it.
        .env("HUB_AUTO_RESUME", "0")
        // Hold recovery open long enough to observe the handshake racing it.
        .env(
            "MINI_SWE_HUB_RECOVERY_DELAY_MS",
            RECOVERY_DELAY_MS.to_string(),
        )
        .env(
            "OPENAI_API_KEY",
            "test-key-not-used-by-the-listen-first-test",
        )
        .env("ENV_FILE", root.path().join("absent.env"))
        .env(
            "MODELS_FILE",
            format!("{}/models.yaml", env!("CARGO_MANIFEST_DIR")),
        )
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .kill_on_drop(true);
    common::scrub_identity_env(command.as_std_mut());
    let mut daemon = command.spawn().expect("spawn the hub daemon");

    let socket = socket_for(&hub);
    wait_for_socket(&socket).await;

    let mut client = Client::connect(&socket).await;
    let hello = client
        .request("hub/hello", serde_json::json!({"agent_id": agent}))
        .await;
    assert!(
        hello["result"]["version"].is_string(),
        "the handshake must be answered: {hello}"
    );

    // The handshake was answered while recovery was still running: the daemon
    // has logged that it is listening but not yet that it recovered the row.
    let log = std::fs::read_to_string(hub.join("hub.log")).expect("read hub.log");
    assert!(
        log.contains("listening"),
        "daemon must log listening: {log}"
    );
    assert!(
        !log.contains("recovered 1 orphaned workers"),
        "recovery must still be running when the handshake is answered: {log}"
    );

    // A worker-state request waits for recovery, then sees the recovered row.
    let list = client
        .request(
            "tools/call",
            serde_json::json!({"name": "worker", "arguments": {"action": "list"}}),
        )
        .await;
    let text = list["result"]["content"][0]["text"]
        .as_str()
        .unwrap_or_else(|| panic!("list payload missing: {list}"));
    let payload: serde_json::Value = serde_json::from_str(text).expect("list payload is JSON");
    let worker = payload["workers"]
        .as_array()
        .expect("workers array")
        .iter()
        .find(|w| w["id"] == wid)
        .unwrap_or_else(|| panic!("recovered worker {wid} must be listable: {payload}"));
    assert_eq!(
        worker["state"]["status"], "Interrupted",
        "list must see the recovered row, not the pre-recovery one: {worker}"
    );

    let _ = daemon.kill().await;
}
