//! The differential verify gate and the side-effect audit around it.
//!
//! The harness runs the worker's verify gate in a sandbox with a scrubbed
//! environment, while the code is later verified elsewhere - the
//! orchestrator's shell, CI, a developer machine. A suite that depends on
//! ambient state passes inside and fails outside, and the worker never sees
//! the failure. So when the gate passes in the canonical environment the same
//! command runs once more in a divergent one (the dispatcher's filtered
//! ambient variables, a fresh `HOME`/`TMPDIR`, a shifted `TZ`); a suite that
//! fails there is refused with the variable named.
//!
//! Every suite below is a tiny shell script, so no language or test runner is
//! assumed: the gate is whatever command the dispatch chose, replayed
//! verbatim.

use crate::common;
use mini_swe_mcp::pool::{COMPLETION_SENTINEL, WorkerPool, WorkerState};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

/// Owner recorded for the workers these tests dispatch: the gate is what is
/// under test here, not the per-agent ownership check.
const TEST_OWNER: &str = "divergent-verify";

/// A throwaway git repository the worker can be dispatched against.
struct TestRepo {
    dir: PathBuf,
}

impl TestRepo {
    fn new(tag: &str) -> Self {
        let dir = std::env::temp_dir().join(common::unique_suffix(&format!("divergent-{tag}")));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create scratch repo");
        let dir = dir.canonicalize().expect("canonicalize scratch repo");
        run_git(&dir, &["init", "-b", "master"]);
        run_git(&dir, &["config", "user.name", "mini-swe-test"]);
        run_git(&dir, &["config", "user.email", "test@localhost"]);
        std::fs::write(dir.join("README.md"), "# scratch\n").expect("seed file");
        run_git(&dir, &["add", "README.md"]);
        run_git(&dir, &["commit", "-m", "baseline"]);
        Self { dir }
    }

    fn path(&self) -> &Path {
        &self.dir
    }
}

impl Drop for TestRepo {
    fn drop(&mut self) {
        // A worker leases build directories keyed by this repo's hash; they are
        // filed next to the scratch base, so removing the repo has to take them.
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

/// A loopback LLM that answers every turn with the completion sentinel.
///
/// The worker runs no commands of its own: the verify gate is the only thing
/// that executes, which is exactly what these tests exercise.
struct SentinelServer {
    base_url: String,
}

impl SentinelServer {
    async fn spawn() -> Self {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind loopback");
        let addr = listener.local_addr().expect("local addr");
        tokio::spawn(async move {
            loop {
                let Ok((mut socket, _)) = listener.accept().await else {
                    return;
                };
                tokio::spawn(async move {
                    use tokio::io::{AsyncReadExt, AsyncWriteExt};
                    let mut data: Vec<u8> = Vec::new();
                    let mut probe = [0u8; 4096];
                    while !data.windows(4).any(|w| w == b"\r\n\r\n") {
                        match socket.read(&mut probe).await {
                            Ok(0) | Err(_) => return,
                            Ok(n) => data.extend_from_slice(&probe[..n]),
                        }
                    }
                    let head_end = data
                        .windows(4)
                        .position(|w| w == b"\r\n\r\n")
                        .expect("head terminator present")
                        + 4;
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
                    let arguments =
                        serde_json::json!({ "command": format!("echo {COMPLETION_SENTINEL}") })
                            .to_string();
                    let body = format!(
                        "data: {{\"choices\":[{{\"delta\":{{\"tool_calls\":[{{\"index\":0,\"id\":\"call_done\",\"function\":{{\"name\":\"bash\",\"arguments\":{arguments}}}}}]}}}}]}}\n\ndata: [DONE]\n\n",
                        arguments = serde_json::to_string(&arguments).expect("quote arguments"),
                    );
                    let head = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\nContent-Length: {}\r\n\r\n",
                        body.len()
                    );
                    let _ = socket.write_all(head.as_bytes()).await;
                    let _ = socket.write_all(body.as_bytes()).await;
                    let _ = socket.flush().await;
                    let _ = socket.shutdown().await;
                });
            }
        });
        Self {
            base_url: format!("http://{addr}"),
        }
    }
}

async fn wait_for_terminal(pool: &WorkerPool, worker_id: &str) -> WorkerState {
    for _ in 0..600 {
        if let Some(state) = pool.get_worker_state(worker_id).await {
            match state {
                WorkerState::Completed { .. }
                | WorkerState::Failed { .. }
                | WorkerState::Exhausted { .. } => return state,
                WorkerState::Running { .. } | WorkerState::Paused { .. } => {}
            }
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("worker {worker_id} did not reach a terminal state");
}

/// Wait until the refusal `needle` appears in the worker's durable history
/// log, or panic after the deadline. A refused completion replays the turn
/// with the refusal as the `tool` answer to the completion call, so the log
/// carries it even while the worker keeps running.
async fn wait_for_refusal(pool: &WorkerPool, worker_id: &str, needle: &str) -> String {
    for _ in 0..600 {
        let path = mini_swe_mcp::pool::revision::history_log_path(worker_id);
        if let Ok(raw) = std::fs::read_to_string(&path) {
            for line in raw.lines() {
                let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else {
                    continue;
                };
                if let Some(content) = value["content"].as_str()
                    && content.contains(needle)
                {
                    return content.to_string();
                }
            }
        }
        if let Some(state) = pool.get_worker_state(worker_id).await
            && matches!(
                state,
                WorkerState::Completed { .. } | WorkerState::Failed { .. }
            )
        {
            let path = mini_swe_mcp::pool::revision::history_log_path(worker_id);
            let raw = std::fs::read_to_string(&path).unwrap_or_default();
            panic!(
                "worker {worker_id} finished before refusing with {needle:?}: {state:?}\n--- log ---\n{raw}"
            );
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("worker {worker_id} never refused with {needle:?}");
}

/// Dispatch one worker with an explicit verify command and ambient snapshot.
async fn dispatch_verify(
    base_url: &str,
    repo: &Path,
    verify: &str,
    client_env: Vec<(String, String)>,
) -> (WorkerPool, String) {
    let pool = WorkerPool::new(1, base_url.to_string(), "test-key".to_string());
    let worker_id = pool
        .dispatch(
            TEST_OWNER.to_string(),
            "exercise the divergent gate".to_string(),
            "test-model".to_string(),
            None,
            repo.to_path_buf(),
            6,
            Some("divergent".to_string()),
            None,
            false,
            Some(verify.to_string()),
            client_env,
        )
        .await
        .expect("dispatch the worker");
    (pool, worker_id)
}

/// A suite that fails when an ambient variable is set is refused, with the
/// variable named.
#[tokio::test]
async fn an_ambient_variable_failure_is_refused_with_the_variable_named() {
    let repo = TestRepo::new("ambient");
    let server = SentinelServer::spawn().await;
    // Passes in the canonical environment (the variable is absent there) and
    // fails in variant B (where the dispatcher's value is layered on top).
    let verify = "test -z \"$SWE_DIVERGENT_PROBE_VAR\"";
    let (pool, worker_id) = dispatch_verify(
        &server.base_url,
        repo.path(),
        verify,
        vec![("SWE_DIVERGENT_PROBE_VAR".to_string(), "set".to_string())],
    )
    .await;
    let refusal = wait_for_refusal(&pool, &worker_id, "clean environment").await;
    assert!(
        refusal.contains("orchestrator's environment")
            && refusal.contains("SWE_DIVERGENT_PROBE_VAR"),
        "the refusal must name the differing variable, got {refusal:?}"
    );
    let _ = pool.kill(&worker_id).await;
}

/// A suite that depends on `TZ` is refused.
#[tokio::test]
async fn a_timezone_dependent_suite_is_refused() {
    let repo = TestRepo::new("tz");
    let server = SentinelServer::spawn().await;
    // The canonical environment keeps the host's own zone; variant B shifts it
    // to one of the two far zones, so the suite passes in A and fails in B.
    let verify = "test \"$TZ\" != \"Pacific/Kiritimati\" && test \"$TZ\" != \"Etc/GMT+12\"";
    let (pool, worker_id) =
        dispatch_verify(&server.base_url, repo.path(), verify, Vec::new()).await;
    let refusal = wait_for_refusal(&pool, &worker_id, "clean environment").await;
    assert!(
        refusal.contains("HOME/TMPDIR/TZ differ"),
        "the refusal must state that HOME/TMPDIR/TZ differ, got {refusal:?}"
    );
    let _ = pool.kill(&worker_id).await;
}

/// A hermetic suite passes both variants and completes verified.
#[tokio::test]
async fn a_hermetic_suite_completes_verified() {
    let repo = TestRepo::new("hermetic");
    let server = SentinelServer::spawn().await;
    let (pool, worker_id) = dispatch_verify(
        &server.base_url,
        repo.path(),
        "echo hermetic-ok",
        vec![("SWE_DIVERGENT_PLAIN".to_string(), "set".to_string())],
    )
    .await;
    let state = wait_for_terminal(&pool, &worker_id).await;
    match state {
        WorkerState::Completed { .. } => {}
        other => panic!("a hermetic suite must complete, got {other:?}"),
    }
}

/// The disable switch is unit-tested in the crate: it reads a process-global
/// variable, so exercising it here would race the sibling tests that expect
/// variant B to run.
/// Secrets from the client environment never reach the worker: a variable
/// matching the secret filter is absent in variant B.
#[tokio::test]
async fn secrets_never_reach_variant_b() {
    // The filter is the unit under test here: the snapshot drops the secret
    // before it is ever stored, so variant B cannot see it.
    let snapshot = mini_swe_mcp::agent::ambient_environment_snapshot();
    assert!(
        !snapshot
            .iter()
            .any(|(name, _)| mini_swe_mcp::agent::is_secret_name(name)),
        "the ambient snapshot must never carry a secret name"
    );
    let decoded = mini_swe_mcp::hub::decode_ambient_env(&serde_json::json!([
        {"name": "SWE_DIVERGENT_PLAIN_CHECK", "value": "yes"},
        {"name": "SWE_DIVERGENT_API_TOKEN_CHECK", "value": "must-not-travel"},
    ]));
    assert!(
        decoded
            .iter()
            .any(|(name, _)| name == "SWE_DIVERGENT_PLAIN_CHECK"),
        "plain variables must survive the handshake decode"
    );
    assert!(
        !decoded
            .iter()
            .any(|(name, _)| name == "SWE_DIVERGENT_API_TOKEN_CHECK"),
        "secret names must be dropped by the daemon-side decode as well"
    );
    // A credential can hide in a *value* under an innocent name: the daemon
    // applies the same value filter, so a tampered frame cannot smuggle one in.
    let decoded = mini_swe_mcp::hub::decode_ambient_env(&serde_json::json!([
        {"name": "SWE_DIVERGENT_URL_CHECK", "value": "postgres://app:hunter2@db.internal:5432/prod"},
        {"name": "SWE_DIVERGENT_PROXY_CHECK", "value": "http://user:s3cret@proxy.internal:8080"},
        {"name": "SWE_DIVERGENT_BLOB_CHECK", "value": "-----BEGIN RSA PRIVATE KEY-----"},
    ]));
    assert!(
        !decoded
            .iter()
            .any(|(name, _)| name == "SWE_DIVERGENT_URL_CHECK"),
        "a credentialed URL must not survive the daemon-side decode"
    );
    assert!(
        !decoded
            .iter()
            .any(|(name, _)| name == "SWE_DIVERGENT_BLOB_CHECK"),
        "a PEM block must not survive the daemon-side decode"
    );
    assert_eq!(
        decoded
            .iter()
            .find(|(name, _)| name == "SWE_DIVERGENT_PROXY_CHECK")
            .map(|(_, value)| value.as_str()),
        None,
        "a non-proxy name carrying proxy credentials is dropped whole"
    );

    // And end to end: a suite that fails when the secret is visible passes,
    // because variant B never sees it.
    let repo = TestRepo::new("secret");
    let server = SentinelServer::spawn().await;
    let verify = "test -z \"$SWE_DIVERGENT_API_TOKEN_CHECK\"";
    let (pool, worker_id) = dispatch_verify(
        &server.base_url,
        repo.path(),
        verify,
        vec![(
            "SWE_DIVERGENT_API_TOKEN_CHECK".to_string(),
            "must-not-travel".to_string(),
        )],
    )
    .await;
    let state = wait_for_terminal(&pool, &worker_id).await;
    match state {
        WorkerState::Completed { .. } => {}
        other => panic!("a secret must never reach variant B, got {other:?}"),
    }
}
