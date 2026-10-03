//! User-declared review modes (`review_modes:` in `models.yaml`).
//!
//! The manifest may declare its own auditors the same way it declares models:
//! each entry carries a `checklist` (focus instructions appended to the common
//! review frame) and an optional default `model`. Built-in modes `quality`
//! and `security` keep their prompts and can be overridden by declaring the
//! same name. `--review-after <model>:<mode>` accepts any declared mode; an
//! unknown mode is a dispatch error listing the available ones.

mod common;

use mini_swe_mcp::manifest::ModelManifest;
use mini_swe_mcp::pool::{ReviewMode, review_prompt};

// ----------
// Parsing
// ----------

fn modes_manifest(yaml: &str) -> ModelManifest {
    serde_yaml::from_str(yaml).expect("review-modes manifest must parse")
}

#[test]
fn a_manifest_without_review_modes_parses_to_empty() {
    let manifest = modes_manifest("models:\n  solo:\n    id: combo:solo\n");
    assert!(manifest.review_modes.is_empty());
    assert!(manifest.validate().is_empty());
}

#[test]
fn a_declared_mode_parses_its_checklist_and_model() {
    let manifest = modes_manifest(
        "models:\n  nerd:\n    id: combo:nerd\nreview_modes:\n  perf:\n    checklist: Check for N+1 queries.\n    model: nerd\n",
    );
    let def = manifest.review_mode("perf").expect("perf mode");
    assert_eq!(def.checklist, "Check for N+1 queries.");
    assert_eq!(def.model.as_deref(), Some("nerd"));
    assert!(manifest.validate().is_empty());
}

#[test]
fn a_declared_mode_without_a_model_has_none() {
    let manifest = modes_manifest(
        "models:\n  nerd:\n    id: combo:nerd\nreview_modes:\n  style:\n    checklist: Check naming.\n",
    );
    let def = manifest.review_mode("style").expect("style mode");
    assert_eq!(def.model, None);
    assert!(manifest.validate().is_empty());
}

// ----------
// Validation
// ----------

#[test]
fn an_empty_checklist_warns_and_is_dropped_by_normalize() {
    let manifest = modes_manifest(
        "models:\n  nerd:\n    id: combo:nerd\nreview_modes:\n  empty:\n    checklist: \"   \"\n",
    );
    let warnings = manifest.validate();
    assert!(
        warnings
            .iter()
            .any(|w| w.contains("empty") && w.contains("checklist")),
        "an empty checklist must warn: {warnings:?}"
    );
    let normalized = manifest.normalize();
    assert!(
        normalized.review_mode("empty").is_none(),
        "an empty checklist must be dropped by normalize"
    );
}

// ----------
// Mode resolution
// ----------

#[test]
fn parse_with_manifest_resolves_a_custom_mode() {
    let manifest = modes_manifest(
        "models:\n  nerd:\n    id: combo:nerd\nreview_modes:\n  perf:\n    checklist: Check for N+1 queries.\n    model: nerd\n",
    );
    let (model, mode) =
        ReviewMode::parse_with_manifest("some-reviewer:perf", &manifest).expect("perf mode");
    assert_eq!(model, "some-reviewer");
    assert_eq!(mode.name, "perf");
    assert_eq!(mode.checklist.as_deref(), Some("Check for N+1 queries."));
}

#[test]
fn parse_with_manifest_uses_the_mode_default_for_an_empty_model() {
    let manifest = modes_manifest(
        "models:\n  nerd:\n    id: combo:nerd\nreview_modes:\n  perf:\n    checklist: Check for N+1 queries.\n    model: nerd\n",
    );
    let (model, mode) = ReviewMode::parse_with_manifest(":perf", &manifest).expect("perf mode");
    assert_eq!(model, "nerd");
    assert_eq!(mode.name, "perf");
}

#[test]
fn parse_with_manifest_keeps_colon_model_ids_as_quality() {
    let manifest = ModelManifest::default();
    // `combo:nerd` is a model id, not a mode: the suffix names no mode and
    // the whole string is a known model.
    let (model, mode) = ReviewMode::parse_with_manifest("combo:nerd", &manifest).expect("model id");
    assert_eq!(model, "combo:nerd");
    assert_eq!(mode.name, "quality");
    assert_eq!(mode.checklist, None);
}

#[test]
fn an_unknown_mode_is_refused_with_the_available_ones() {
    let manifest = modes_manifest(
        "models:\n  nerd:\n    id: combo:nerd\nreview_modes:\n  perf:\n    checklist: Check.\n",
    );
    let err = ReviewMode::parse_with_manifest("reviewer:nope", &manifest)
        .expect_err("an unknown mode must be refused");
    let message = err.to_string();
    assert!(message.contains("unknown review mode"), "{message}");
    assert!(message.contains("nope"), "{message}");
    for available in ["quality", "security", "perf"] {
        assert!(
            message.contains(available),
            "{message} must list {available}"
        );
    }
}

// ----------
// The prompt
// ----------

#[test]
fn a_custom_mode_appends_its_checklist_to_the_review_frame() {
    let mode = ReviewMode {
        name: "perf".to_string(),
        checklist: Some("Check for N+1 queries.".to_string()),
    };
    let prompt = review_prompt(&mode, "speed up", Some("make check"), &[]);
    assert!(
        prompt.contains("REVIEW PHASE (perf)"),
        "a custom mode names itself: {prompt}"
    );
    assert!(
        prompt.contains("Check for N+1 queries."),
        "the checklist is appended: {prompt}"
    );
    assert!(
        prompt.contains("make check"),
        "the dispatch gate is still run: {prompt}"
    );
    assert!(
        prompt.contains("COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"),
        "the completion contract is kept: {prompt}"
    );
}

#[test]
fn overriding_security_replaces_its_checklist() {
    let manifest = modes_manifest(
        "models:\n  nerd:\n    id: combo:nerd\nreview_modes:\n  security:\n    checklist: Custom hostile review.\n",
    );
    let mode = ReviewMode::resolve_declared("security", &manifest);
    assert_eq!(mode.name, "security");
    assert_eq!(mode.checklist.as_deref(), Some("Custom hostile review."));
    assert!(
        mode.is_security(),
        "an overridden security is still the security review"
    );
    let prompt = review_prompt(&mode, "task", Some("make check"), &[]);
    assert!(
        prompt.contains("Custom hostile review."),
        "the override replaces the prompt: {prompt}"
    );
    assert!(
        !prompt.contains("ADVERSARIAL SECURITY REVIEW PHASE"),
        "the built-in prompt is replaced: {prompt}"
    );
}

// ----------
// End-to-end: the phase loop uses the mode's checklist and reviewer
// ----------

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Mutex;

use mini_swe_mcp::pool::{WorkerPool, WorkerState};
use mini_swe_mcp::worktree::ScratchRoot;

const TEST_OWNER: &str = "test-agent";
const COMPLETION_SENTINEL: &str = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT";

/// A loopback HTTP server that answers `POST /chat/completions` with the
/// next scripted SSE body and captures every request body.
struct ScriptedSseServer {
    base_url: String,
    requests: Arc<Mutex<Vec<Value>>>,
}

impl ScriptedSseServer {
    async fn spawn(turns: Vec<Vec<String>>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind loopback");
        let addr: SocketAddr = listener.local_addr().expect("local addr");
        let requests = Arc::new(Mutex::new(Vec::new()));
        let sink = requests.clone();
        let script = Arc::new(std::sync::Mutex::new(turns.into_iter()));
        tokio::spawn(async move {
            loop {
                let Ok((mut socket, _)) = listener.accept().await else {
                    return;
                };
                let turn = script.lock().expect("script lock").next();
                let sink = sink.clone();
                tokio::spawn(async move {
                    let request = read_http_request(&mut socket).await;
                    if let Some(body) = request
                        && let Ok(value) = serde_json::from_str::<Value>(&body)
                    {
                        sink.lock().await.push(value);
                    }
                    write_sse(&mut socket, turn.as_deref()).await;
                });
            }
        });
        Self {
            base_url: format!("http://{addr}"),
            requests,
        }
    }

    fn turn(call_id: &str, content: &str, command: &str) -> Vec<String> {
        let arguments = json!({ "command": command }).to_string();
        vec![frame(&json!({
            "choices": [{
                "delta": {
                    "content": content,
                    "tool_calls": [{
                        "index": 0,
                        "id": call_id,
                        "function": { "name": "bash", "arguments": arguments }
                    }]
                }
            }]
        }))]
    }

    fn completion_turn(call_id: &str, content: &str) -> Vec<String> {
        Self::turn(call_id, content, &format!("echo {COMPLETION_SENTINEL}"))
    }
}

fn frame(value: &Value) -> String {
    format!("data: {value}\n\n")
}

async fn write_sse(socket: &mut TcpStream, turn: Option<&[String]>) {
    let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n";
    if socket.write_all(head.as_bytes()).await.is_err() {
        return;
    }
    if let Some(chunks) = turn {
        for chunk in chunks {
            if socket.write_all(chunk.as_bytes()).await.is_err() {
                return;
            }
            let _ = socket.flush().await;
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        let _ = socket.write_all(b"data: [DONE]\n\n").await;
    }
    let _ = socket.flush().await;
    let _ = socket.shutdown().await;
}

async fn read_http_request(socket: &mut TcpStream) -> Option<String> {
    let mut data: Vec<u8> = Vec::new();
    let mut probe = [0u8; 4096];
    while !data.windows(4).any(|w| w == b"\r\n\r\n") {
        match socket.read(&mut probe).await {
            Ok(0) | Err(_) => return None,
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
    Some(String::from_utf8_lossy(&data[head_end..head_end + content_length]).to_string())
}

struct TestRepo {
    dir: PathBuf,
}

impl TestRepo {
    fn new(tag: &str) -> Self {
        let dir = common::process_temp_dir(&format!("review-modes-{tag}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create scratch repo");
        let dir = dir.canonicalize().expect("canonicalize scratch repo");
        common::git(&dir, &["init", "-b", "master"]);
        common::git(&dir, &["config", "user.name", "mini-swe-test"]);
        common::git(&dir, &["config", "user.email", "test@localhost"]);
        std::fs::write(dir.join("README.md"), "# scratch\n").expect("seed file");
        common::git(&dir, &["add", "README.md"]);
        common::git(&dir, &["commit", "-m", "baseline"]);
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

fn modes_pool(base_url: &str, scratch: &common::TempDir) -> WorkerPool {
    let manifest: ModelManifest = serde_yaml::from_str(
        "models:\n  test-model:\n    id: test-model\nreview_modes:\n  perf:\n    checklist: Check for N+1 queries and missing indexes.\n    model: test-model\n",
    )
    .expect("modes manifest must parse");
    WorkerPool::with_scratch(
        1,
        base_url.to_string(),
        "test-key".to_string(),
        ScratchRoot::new(scratch.path()),
    )
    .with_manifest(Arc::new(manifest))
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

async fn review_prompt_of(server: &ScriptedSseServer) -> Option<String> {
    for request in server.requests.lock().await.iter() {
        for message in request["messages"].as_array().unwrap_or(&Vec::new()) {
            if message["role"] == json!("user") {
                let content = message["content"].as_str().unwrap_or_default();
                if content.contains("REVIEW PHASE") {
                    return Some(content.to_string());
                }
            }
        }
    }
    None
}

/// A custom mode selected by suffix runs its checklist on the requested model.
#[tokio::test]
async fn a_custom_mode_uses_its_checklist_and_reviewer() {
    let repo = TestRepo::new("custom");
    let server = ScriptedSseServer::spawn(vec![
        ScriptedSseServer::turn("call_write", "", "echo changed > src/ordinary.rs"),
        ScriptedSseServer::turn(
            "call_impl",
            "REPORT\ndone: impl\nrisks: none",
            &format!("echo {COMPLETION_SENTINEL}"),
        ),
        ScriptedSseServer::completion_turn("call_review", "REPORT\ndone: reviewed\nrisks: none"),
    ])
    .await;

    let scratch = common::TempDir::new_in_tmp("review-modes-custom");
    let pool = modes_pool(&server.base_url, &scratch);
    let worker_id = pool
        .dispatch(
            TEST_OWNER.to_string(),
            "exercise the custom mode".to_string(),
            "test-model".to_string(),
            None,
            repo.path().to_path_buf(),
            5,
            Some("review-modes".to_string()),
            Some("test-reviewer:perf".to_string()),
            false,
            None,
            Vec::new(),
        )
        .await
        .expect("dispatch the worker");
    let state = wait_for_terminal(&pool, &worker_id).await;

    let requests = server.requests.lock().await.clone();
    assert_eq!(
        requests.len(),
        3,
        "write, implementer, and the custom review"
    );
    assert_eq!(
        requests[2]["model"],
        json!("test-reviewer"),
        "the suffix selects the mode, not the model"
    );
    let prompt = review_prompt_of(&server).await.expect("a review prompt");
    assert!(
        prompt.contains("REVIEW PHASE (perf)"),
        "the custom mode names itself: {prompt}"
    );
    assert!(
        prompt.contains("Check for N+1 queries and missing indexes."),
        "the review runs the mode's checklist: {prompt}"
    );

    let WorkerState::Completed { .. } = &state else {
        panic!("worker must complete, got {state:?}")
    };
    let _ = pool.kill(&worker_id).await;
}

/// An unknown mode is a dispatch error, not a worker failure.
#[tokio::test]
async fn an_unknown_mode_is_a_dispatch_error() {
    let repo = TestRepo::new("unknown");
    let server = ScriptedSseServer::spawn(vec![]).await;

    let scratch = common::TempDir::new_in_tmp("review-modes-unknown");
    let pool = modes_pool(&server.base_url, &scratch);
    let err = pool
        .dispatch(
            TEST_OWNER.to_string(),
            "exercise the unknown mode".to_string(),
            "test-model".to_string(),
            None,
            repo.path().to_path_buf(),
            5,
            Some("review-modes".to_string()),
            Some("test-reviewer:nope".to_string()),
            false,
            None,
            Vec::new(),
        )
        .await
        .expect_err("an unknown mode must be a dispatch error");
    let message = err.to_string();
    assert!(message.contains("unknown review mode"), "{message}");
    assert!(message.contains("nope"), "{message}");
    assert!(
        message.contains("perf"),
        "{message} must list the declared mode"
    );
}

// ----------
// Triggers
// ----------

#[test]
fn a_declared_mode_parses_its_triggers() {
    let manifest = modes_manifest(
        "models:\n  nerd:\n    id: combo:nerd\nreview_modes:\n  perf:\n    checklist: Check.\n    triggers:\n      - src/hot/**\n      - src/perf.rs\n",
    );
    let def = manifest.review_mode("perf").expect("perf mode");
    assert_eq!(def.triggers, vec!["src/hot/**".to_string(), "src/perf.rs".to_string()]);
    assert!(manifest.validate().is_empty());
}

#[test]
fn an_invalid_trigger_glob_warns_and_is_dropped_by_normalize() {
    let manifest = modes_manifest(
        "models:\n  nerd:\n    id: combo:nerd\nreview_modes:\n  perf:\n    checklist: Check.\n    triggers:\n      - src/good/**\n      - \"   \"\n",
    );
    let warnings = manifest.validate();
    assert!(
        warnings.iter().any(|w| w.contains("invalid trigger glob")),
        "an invalid glob must warn: {warnings:?}"
    );
    let normalized = manifest.normalize();
    let def = normalized.review_mode("perf").expect("perf mode");
    assert_eq!(def.triggers, vec!["src/good/**".to_string()], "the invalid glob is dropped");
}

/// A manifest-declared mode whose `triggers` match the diff runs automatically.
#[tokio::test]
async fn a_custom_trigger_fires_its_mode() {
    let repo = TestRepo::new("trigger");
    let server = ScriptedSseServer::spawn(vec![
        ScriptedSseServer::turn("call_write", "", "mkdir -p src/hot && echo changed > src/hot/mod.rs"),
        ScriptedSseServer::turn(
            "call_impl",
            "REPORT\ndone: impl\nrisks: none",
            &format!("echo {COMPLETION_SENTINEL}"),
        ),
        ScriptedSseServer::completion_turn("call_review", "REPORT\ndone: reviewed\nrisks: none"),
    ])
    .await;

    let scratch = common::TempDir::new_in_tmp("review-modes-trigger");
    let manifest: ModelManifest = serde_yaml::from_str(
        "models:\n  test-model:\n    id: test-model\nreview_modes:\n  perf:\n    checklist: Check for N+1 queries.\n    triggers:\n      - src/hot/**\n",
    )
    .expect("modes manifest must parse");
    let pool = WorkerPool::with_scratch(
        1,
        server.base_url.clone(),
        "test-key".to_string(),
        ScratchRoot::new(scratch.path()),
    )
    .with_manifest(Arc::new(manifest));
    let worker_id = pool
        .dispatch(
            TEST_OWNER.to_string(),
            "exercise the custom trigger".to_string(),
            "test-model".to_string(),
            None,
            repo.path().to_path_buf(),
            5,
            Some("review-modes".to_string()),
            None,
            false,
            None,
            Vec::new(),
        )
        .await
        .expect("dispatch the worker");
    let state = wait_for_terminal(&pool, &worker_id).await;

    let requests = server.requests.lock().await.clone();
    assert_eq!(
        requests.len(),
        3,
        "write, implementer, and the triggered review"
    );
    // The custom mode has no default model, so the implementer's model runs it.
    assert_eq!(requests[2]["model"], json!("test-model"));
    let prompt = review_prompt_of(&server).await.expect("a review prompt");
    assert!(
        prompt.contains("REVIEW PHASE (perf)"),
        "the trigger runs the perf mode: {prompt}"
    );
    assert!(
        prompt.contains("Check for N+1 queries."),
        "the checklist is used: {prompt}"
    );

    let WorkerState::Completed { .. } = &state else {
        panic!("worker must complete, got {state:?}")
    };
    let _ = pool.kill(&worker_id).await;
}

/// Several modes whose triggers match run as successive review phases in
/// sorted name order.
#[tokio::test]
async fn several_triggered_modes_run_successively_in_sorted_order() {
    let repo = TestRepo::new("several");
    let server = ScriptedSseServer::spawn(vec![
        ScriptedSseServer::turn("call_write", "", "mkdir -p src/hot && echo changed > src/hot/mod.rs"),
        ScriptedSseServer::turn(
            "call_impl",
            "REPORT\ndone: impl\nrisks: none",
            &format!("echo {COMPLETION_SENTINEL}"),
        ),
        ScriptedSseServer::completion_turn("call_review_a", "REPORT\ndone: a\nrisks: none"),
        ScriptedSseServer::completion_turn("call_review_b", "REPORT\ndone: b\nrisks: none"),
    ])
    .await;

    let scratch = common::TempDir::new_in_tmp("review-modes-several");
    let manifest: ModelManifest = serde_yaml::from_str(
        "models:\n  test-model:\n    id: test-model\nreview_modes:\n  zebra:\n    checklist: Zebra check.\n    triggers:\n      - src/hot/**\n  alpha:\n    checklist: Alpha check.\n    triggers:\n      - src/hot/**\n",
    )
    .expect("modes manifest must parse");
    let pool = WorkerPool::with_scratch(
        1,
        server.base_url.clone(),
        "test-key".to_string(),
        ScratchRoot::new(scratch.path()),
    )
    .with_manifest(Arc::new(manifest));
    let worker_id = pool
        .dispatch(
            TEST_OWNER.to_string(),
            "exercise several triggers".to_string(),
            "test-model".to_string(),
            None,
            repo.path().to_path_buf(),
            5,
            Some("review-modes".to_string()),
            None,
            false,
            None,
            Vec::new(),
        )
        .await
        .expect("dispatch the worker");
    let state = wait_for_terminal(&pool, &worker_id).await;

    let requests = server.requests.lock().await.clone();
    assert_eq!(requests.len(), 4, "write, implementer, and two reviews");
    // The two reviews run in sorted name order: alpha before zebra.
    let prompts: Vec<String> = requests
        .iter()
        .skip(2)
        .map(|req| {
            req["messages"]
                .as_array()
                .unwrap_or(&Vec::new())
                .iter()
                .filter_map(|m| {
                    if m["role"] == json!("user") {
                        m["content"].as_str().map(str::to_string)
                    } else {
                        None
                    }
                })
                .find(|c| c.contains("REVIEW PHASE"))
                .unwrap_or_default()
        })
        .collect();
    let alpha = prompts
        .iter()
        .position(|p| p.contains("REVIEW PHASE (alpha)"))
        .expect("alpha phase ran");
    let zebra = prompts
        .iter()
        .position(|p| p.contains("REVIEW PHASE (zebra)"))
        .expect("zebra phase ran");
    assert!(alpha < zebra, "phases run in sorted order: {prompts:?}");

    let WorkerState::Completed { .. } = &state else {
        panic!("worker must complete, got {state:?}")
    };
    let _ = pool.kill(&worker_id).await;
}
