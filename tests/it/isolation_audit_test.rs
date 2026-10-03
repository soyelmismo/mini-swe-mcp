//! Every isolation denial reaches the hub log and the worker's counters.
//!
//! A guardrail block, a sandbox that could not be prepared and a side-effect
//! audit refusal are otherwise answered only in the worker's conversation,
//! which is deleted when the worker is retired. The tests dispatch a worker
//! whose command the interceptor refuses and one whose commands all run, and
//! check the `audit` WARN line and the `isolation_blocks` counter for the
//! first and their absence for the second.

use crate::common;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use serde_json::json;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

use mini_swe_mcp::pool::{COMPLETION_SENTINEL, WorkerMetrics, WorkerPool, WorkerState};

const TEST_OWNER: &str = "test-agent";
const TASK: &str = "Add a note to lib.rs.";

// ----------
// Tracing capture
// ----------

/// A writer that appends everything the subscriber formats into a shared
/// buffer, so a test can assert on the log lines the pool emitted.
#[derive(Clone)]
struct SharedWriter(Arc<Mutex<Vec<u8>>>);

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for SharedWriter {
    type Writer = SharedWriter;

    fn make_writer(&'a self) -> Self::Writer {
        self.clone()
    }
}

impl std::io::Write for SharedWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().expect("log buffer").extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Install the capturing subscriber once for the test binary and return the
/// buffer it writes to.
fn capture_logs() -> Arc<Mutex<Vec<u8>>> {
    static ONCE: std::sync::OnceLock<Arc<Mutex<Vec<u8>>>> = std::sync::OnceLock::new();
    ONCE.get_or_init(|| {
        let buffer = Arc::new(Mutex::new(Vec::new()));
        let subscriber = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::WARN)
            .with_ansi(false)
            .with_writer(SharedWriter(buffer.clone()))
            .finish();
        tracing::subscriber::set_global_default(subscriber)
            .expect("install the capturing subscriber");
        buffer
    })
    .clone()
}

fn logs(buffer: &Arc<Mutex<Vec<u8>>>) -> String {
    String::from_utf8_lossy(&buffer.lock().expect("log buffer")).to_string()
}

// ----------
// Scripted SSE server
// ----------

/// A loopback server that answers one scripted turn per connection.
async fn spawn_server(turns: Vec<String>) -> String {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind the scripted server on loopback");
    let addr = listener.local_addr().expect("local addr");
    tokio::spawn(async move {
        let mut next = 0usize;
        loop {
            let Ok((mut socket, _)) = listener.accept().await else {
                return;
            };
            let turn = turns
                .get(next)
                .cloned()
                .unwrap_or_else(|| bash_turn("call_tail", "true"));
            next += 1;
            tokio::spawn(async move {
                if read_request(&mut socket).await.is_some() {
                    write_sse(&mut socket, &turn).await;
                }
            });
        }
    });
    format!("http://{addr}")
}

/// One SSE turn: a tool call running `command`.
fn bash_turn(call_id: &str, command: &str) -> String {
    let arguments = json!({ "command": command }).to_string();
    format!(
        "data: {}\n\n",
        json!({
            "choices": [{
                "delta": {
                    "content": "",
                    "tool_calls": [{
                        "index": 0,
                        "id": call_id,
                        "function": { "name": "bash", "arguments": arguments }
                    }]
                }
            }]
        })
    )
}

/// The completion turn, with the REPORT block the system prompt requires.
fn completion_turn(call_id: &str) -> String {
    let report = "REPORT\ndone: scripted completion\nfiles: lib.rs\ntests: none\nrisks: none";
    let arguments = json!({ "command": format!("echo {COMPLETION_SENTINEL}") }).to_string();
    format!(
        "data: {}\n\n",
        json!({
            "choices": [{
                "delta": {
                    "content": report,
                    "tool_calls": [{
                        "index": 0,
                        "id": call_id,
                        "function": { "name": "bash", "arguments": arguments }
                    }]
                }
            }]
        })
    )
}

async fn write_sse(socket: &mut TcpStream, turn: &str) {
    let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n";
    if socket.write_all(head.as_bytes()).await.is_err() {
        return;
    }
    if socket.write_all(turn.as_bytes()).await.is_err() {
        return;
    }
    let _ = socket.flush().await;
    let _ = socket.write_all(b"data: [DONE]\n\n").await;
    let _ = socket.flush().await;
    let _ = socket.shutdown().await;
}

async fn read_request(socket: &mut TcpStream) -> Option<String> {
    let mut data: Vec<u8> = Vec::new();
    let mut probe = [0u8; 4096];
    while !data.windows(4).any(|w| w == b"\r\n\r\n") {
        match socket.read(&mut probe).await {
            Ok(0) | Err(_) => return None,
            Ok(n) => data.extend_from_slice(&probe[..n]),
        }
    }
    let head_end = data.windows(4).position(|w| w == b"\r\n\r\n")? + 4;
    let head = String::from_utf8_lossy(&data[..head_end]).to_string();
    let content_length: usize = head
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.trim()
                .eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse().ok())?
        })
        .unwrap_or(0);
    while data.len() - head_end < content_length {
        match socket.read(&mut probe).await {
            Ok(0) | Err(_) => break,
            Ok(n) => data.extend_from_slice(&probe[..n]),
        }
    }
    Some(String::from_utf8_lossy(&data[head_end..head_end + content_length]).to_string())
}

// ----------
// Scratch repository
// ----------

struct TestRepo {
    dir: PathBuf,
}

impl TestRepo {
    fn new(tag: &str) -> Self {
        let dir = std::env::temp_dir().join(common::unique_suffix(&format!("isolation-{tag}")));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create scratch repo");
        let dir = dir.canonicalize().expect("canonicalize scratch repo");
        run_git(&dir, &["init", "-b", "master"]);
        run_git(&dir, &["config", "user.name", "mini-swe-test"]);
        run_git(&dir, &["config", "user.email", "test@localhost"]);
        std::fs::write(dir.join("README.md"), "# scratch\n").expect("seed file");
        std::fs::write(dir.join("lib.rs"), "fn main() {}\n").expect("seed file");
        run_git(&dir, &["add", "."]);
        run_git(&dir, &["commit", "-m", "baseline"]);
        Self { dir }
    }

    fn path(&self) -> &Path {
        &self.dir
    }
}

impl Drop for TestRepo {
    fn drop(&mut self) {
        mini_swe_mcp::cache::remove_build_dir_leases(&self.dir);
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn run_git(dir: &Path, args: &[&str]) {
    let output = Command::new("git")
        .current_dir(dir)
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("git {args:?}: {e}"));
    assert!(
        output.status.success(),
        "git {args:?} failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

// ----------
// Worker helpers
// ----------

async fn dispatch(pool: &WorkerPool, repo: &Path) -> String {
    pool.dispatch(
        TEST_OWNER.to_string(),
        TASK.to_string(),
        "test-model".to_string(),
        None,
        repo.to_path_buf(),
        10,
        None,
        None,
        false,
        None,
        Vec::new(),
    )
    .await
    .expect("dispatch the worker")
}

async fn wait_for_terminal(pool: &WorkerPool, worker_id: &str) -> WorkerState {
    let mut last = None;
    for _ in 0..600 {
        if let Some(state) = pool.get_worker_state(worker_id).await {
            match state {
                WorkerState::Completed { .. }
                | WorkerState::Failed { .. }
                | WorkerState::Exhausted { .. } => return state,
                other => last = Some(format!("{other:?}")),
            }
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("worker {worker_id} did not reach a terminal state; last: {last:?}");
}

fn metrics_of(state: &WorkerState) -> WorkerMetrics {
    match state {
        WorkerState::Completed { metrics, .. } => *metrics,
        other => panic!("expected a completed worker, got {other:?}"),
    }
}

/// A blocked command is logged as an audit line and counted; a clean run is
/// neither.
#[tokio::test]
async fn a_guardrail_block_is_logged_and_counted() {
    let buffer = capture_logs();

    // Worker one: its first command is refused by the interceptor, then it
    // completes with a harmless one.
    let blocked_repo = TestRepo::new("blocked");
    let blocked_url = spawn_server(vec![
        bash_turn("call_1", "rm -rf /"),
        bash_turn("call_2", "printf edited >> lib.rs"),
        completion_turn("call_done"),
    ])
    .await;
    let scratch = common::TempDir::new_in_tmp("isolation-pool");
    let pool = WorkerPool::with_scratch(
        1,
        blocked_url,
        "test-key".to_string(),
        mini_swe_mcp::worktree::ScratchRoot::new(scratch.path()),
    );
    let _scratch = scratch;
    let blocked_id = dispatch(&pool, blocked_repo.path()).await;
    let blocked_metrics = metrics_of(&wait_for_terminal(&pool, &blocked_id).await);

    assert_eq!(
        blocked_metrics.isolation_blocks, 1,
        "the refused command must be counted, got {blocked_metrics:?}"
    );
    let line = wait_for_audit_line(&buffer, &blocked_id)
        .await
        .expect("the refusal must reach the hub log");
    assert!(
        line.contains("rule=destructive_command_interceptor"),
        "{line}"
    );
    assert!(line.contains(&format!("worker={blocked_id}")), "{line}");
    assert!(line.contains(&format!("owner={TEST_OWNER}")), "{line}");
    assert!(line.contains("reason="), "{line}");
    assert!(
        line.contains("command=rm -rf /"),
        "the command summary must be bounded, got {line}"
    );
    assert!(
        !line.contains("rm -rf / --no-preserve-root"),
        "the full command must not be logged, got {line}"
    );

    // Worker two: every command runs, so nothing is logged or counted.
    let clean_repo = TestRepo::new("clean");
    let clean_url = spawn_server(vec![
        bash_turn("call_1", "printf edited >> lib.rs"),
        completion_turn("call_done"),
    ])
    .await;
    let scratch = common::TempDir::new_in_tmp("isolation-pool-clean");
    let pool = WorkerPool::with_scratch(
        1,
        clean_url,
        "test-key".to_string(),
        mini_swe_mcp::worktree::ScratchRoot::new(scratch.path()),
    );
    let _scratch = scratch;
    let clean_id = dispatch(&pool, clean_repo.path()).await;
    let clean_metrics = metrics_of(&wait_for_terminal(&pool, &clean_id).await);
    assert_eq!(
        clean_metrics.isolation_blocks, 0,
        "a clean run must not be counted, got {clean_metrics:?}"
    );
    assert!(
        !logs(&buffer).contains(&format!("worker={clean_id}")),
        "a clean run must not produce an audit line"
    );
}

/// Poll for the audit line of `worker_id`, so the test does not race the
/// subscriber's writer.
async fn wait_for_audit_line(buffer: &Arc<Mutex<Vec<u8>>>, worker_id: &str) -> Option<String> {
    for _ in 0..100 {
        let line = logs(buffer)
            .lines()
            .find(|line| line.contains("isolation block") && line.contains(worker_id))
            .map(str::to_string);
        if line.is_some() {
            return line;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    None
}
