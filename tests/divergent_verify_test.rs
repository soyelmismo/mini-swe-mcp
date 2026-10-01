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

mod common;

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use mini_swe_mcp::pool::{COMPLETION_SENTINEL, WorkerPool, WorkerState};

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
                WorkerState::Completed { .. } | WorkerState::Failed { .. } => return state,
                WorkerState::Running { .. } | WorkerState::Paused { .. } => {}
            }
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("worker {worker_id} did not reach a terminal state");
}

/// The user messages of the last request the worker made: a refused
/// completion replays the turn with the refusal as the newest one.
async fn last_user_messages(pool: &WorkerPool, worker_id: &str) -> Vec<String> {
    let logs = pool.get_worker_logs(worker_id).await.unwrap_or_default();
    let _ = logs;
    // The refusal is pushed into the conversation, which the history log
    // carries: read it back from the durable log.
    let path = mini_swe_mcp::pool::revision::history_log_path(worker_id);
    let raw = std::fs::read_to_string(&path).unwrap_or_default();
    raw.lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .filter(|value| value["role"] == serde_json::json!("user"))
        .filter_map(|value| value["content"].as_str().map(str::to_string))
        .collect()
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
    let state = wait_for_terminal(&pool, &worker_id).await;
    assert!(
        matches!(state, WorkerState::Running { .. } | WorkerState::Paused { .. }),
        "the worker must be refused, not completed, got {state:?}"
    );
    let users = last_user_messages(&pool, &worker_id).await;
    let refusal = users.last().cloned().unwrap_or_default();
    assert!(
        refusal.contains("clean environment")
            && refusal.contains("orchestrator's environment")
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
    // The canonical environment sets no TZ (UTC); variant B shifts it far.
    let verify = "test \"$(date +%Z)\" = \"UTC\"";
    let (pool, worker_id) = dispatch_verify(&server.base_url, repo.path(), verify, Vec::new()).await;
    let state = wait_for_terminal(&pool, &worker_id).await;
    assert!(
        matches!(state, WorkerState::Running { .. } | WorkerState::Paused { .. }),
        "the worker must be refused, not completed, got {state:?}"
    );
    let users = last_user_messages(&pool, &worker_id).await;
    let refusal = users.last().cloned().unwrap_or_default();
    assert!(
        refusal.contains("clean environment") && refusal.contains("HOME/TMPDIR/TZ differ"),
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

/// A suite that creates a git branch in the repo is refused, and the branch
/// is removed.
#[tokio::test]
async fn a_suite_that_creates_a_branch_is_refused_and_cleaned_up() {
    let repo = TestRepo::new("branch");
    let server = SentinelServer::spawn().await;
    let verify = "git branch worker-leftover-branch && echo branched";
    let (pool, worker_id) = dispatch_verify(&server.base_url, repo.path(), verify, Vec::new()).await;
    let state = wait_for_terminal(&pool, &worker_id).await;
    assert!(
        matches!(state, WorkerState::Running { .. } | WorkerState::Paused { .. }),
        "the worker must be refused, not completed, got {state:?}"
    );
    let users = last_user_messages(&pool, &worker_id).await;
    let refusal = users.last().cloned().unwrap_or_default();
    assert!(
        refusal.contains("must clean up") && refusal.contains("refs/heads/worker-leftover-branch"),
        "the refusal must name the created ref, got {refusal:?}"
    );
    let output = Command::new("git")
        .current_dir(repo.path())
        .args(["for-each-ref", "--format=%(refname)"])
        .output()
        .expect("list refs");
    let refs = String::from_utf8_lossy(&output.stdout);
    assert!(
        !refs.contains("refs/heads/worker-leftover-branch"),
        "the harness must remove the branch the suite created: {refs}"
    );
    let _ = pool.kill(&worker_id).await;
}

/// `WORKER_DIVERGENT_VERIFY=0` skips variant B: the same ambient-dependent
/// suite now completes.
#[tokio::test]
async fn disabling_variant_b_lets_an_ambient_suite_complete() {
    let repo = TestRepo::new("disabled");
    let server = SentinelServer::spawn().await;
    unsafe {
        std::env::set_var("WORKER_DIVERGENT_VERIFY", "0");
    }
    let verify = "test -z \"$SWE_DIVERGENT_DISABLED_VAR\"";
    let (pool, worker_id) = dispatch_verify(
        &server.base_url,
        repo.path(),
        verify,
        vec![("SWE_DIVERGENT_DISABLED_VAR".to_string(), "set".to_string())],
    )
    .await;
    let state = wait_for_terminal(&pool, &worker_id).await;
    unsafe {
        std::env::remove_var("WORKER_DIVERGENT_VERIFY");
    }
    match state {
        WorkerState::Completed { .. } => {}
        other => panic!("with variant B disabled the suite must complete, got {other:?}"),
    }
}

/// Secrets from the client environment never reach the worker: a variable
/// matching the secret filter is absent in variant B.
#[tokio::test]
async fn secrets_never_reach_variant_b() {
    // The filter is the unit under test here: the snapshot drops the secret
    // before it is ever stored, so variant B cannot see it.
    let snapshot = mini_swe_mcp::agent::ambient_environment_snapshot();
    assert!(
        !snapshot.iter().any(|(name, _)| mini_swe_mcp::agent::is_secret_name(name)),
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

    // And end to end: a suite that fails when the secret is visible passes,
    // because variant B never sees it.
    let repo = TestRepo::new("secret");
    let server = SentinelServer::spawn().await;
    let verify = "test -z \"$SWE_DIVERGENT_API_TOKEN_CHECK\"";
    let (pool, worker_id) = dispatch_verify(
        &server.base_url,
        repo.path(),
        verify,
        vec![("SWE_DIVERGENT_API_TOKEN_CHECK".to_string(), "must-not-travel".to_string())],
    )
    .await;
    let state = wait_for_terminal(&pool, &worker_id).await;
    match state {
        WorkerState::Completed { .. } => {}
        other => panic!("a secret must never reach variant B, got {other:?}"),
    }
}
