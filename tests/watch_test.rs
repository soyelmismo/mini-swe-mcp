//! `watch` integration: immediate event, timeout, nothing-to-watch, governance.

mod common;

use mini_swe_mcp::cli::watch;
use mini_swe_mcp::hub::{HubConfig, HubPaths, HubServer};
use mini_swe_mcp::manifest::ModelManifest;
use mini_swe_mcp::mcp::McpServer;
use mini_swe_mcp::pool::{LogBuffer, WorkerMetrics, WorkerPool, WorkerRecord, WorkerState};
use mini_swe_mcp::pool::{RegistryStatus, WorkerRegistryEntry};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;

/// Point this process's registry at a scratch directory.
///
/// The daemon under test runs *inside* the test process, so it reads the
/// registry through this process's environment. Without this it would report
/// whatever workers the host's real registry happens to hold, which is both
/// flaky and a leak of unrelated state into the assertions.
fn isolate_registry() -> PathBuf {
    static REGISTRY: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
    REGISTRY
        .get_or_init(|| {
            // The directory must outlive the test that created it: the daemon
            // keeps reading it for as long as the test binary runs.
            let dir = common::TempDir::new_in_tmp("watch-registry");
            let path = dir.path().to_path_buf();
            std::mem::forget(dir);
            // SAFETY: `OnceLock` runs this closure exactly once and blocks every
            // other caller until it returns, so no thread observes a half-set
            // environment.
            unsafe { std::env::set_var("SWE_TEMP_DIR", &path) };
            path
        })
        .clone()
}

fn paths(dir: &Path) -> HubPaths {
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700)).expect("0700");
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
}

async fn pool_with(records: Vec<WorkerRecord>) -> Arc<McpServer> {
    let scratch = common::TempDir::new_in_tmp("watch-pool");
    let pool = WorkerPool::with_scratch(
        4,
        "http://localhost:1".to_string(),
        "test-key".to_string(),
        mini_swe_mcp::worktree::ScratchRoot::new(scratch.path()),
    )
    .with_manifest(Arc::new(ModelManifest::default()));
    let _scratch = scratch;
    for record in records {
        pool.__test_insert_worker(record).await;
    }
    Arc::new(McpServer::new(pool, "test".to_string()))
}

#[tokio::test]
async fn agent_b_cannot_watch_agent_a_worker_and_missed_events_replay_to_owner() {
    isolate_registry();
    let dir = common::TempDir::new_in_tmp("wg");
    let server = pool_with(vec![record(
        "w-watch",
        "agent-a",
        WorkerState::Running {
            step: 1,
            last_command: "test".to_string(),
            started_at: 0,
        },
    )])
    .await;
    let daemon = HubServer::new(server, HubConfig::new(paths(dir.path()), 60));
    let task = tokio::spawn(async move { daemon.run().await });
    wait_for_socket(&dir.path().join("hub.sock")).await;

    let mut a = Raw::connect(&dir.path().join("hub.sock")).await;
    let reply = a
        .request(
            "hub/watch",
            serde_json::json!({"worker_ids": ["w-watch"], "group": null, "initial": true}),
        )
        .await;
    assert_eq!(
        reply["error"]["message"],
        "worker w-watch belongs to agent agent-a"
    );
    drop(a);

    let mut owner = Raw::connect(&dir.path().join("hub.sock")).await;
    owner
        .request("hub/hello", serde_json::json!({"agent_id": "agent-a"}))
        .await;
    let reply = owner
        .request(
            "hub/watch",
            serde_json::json!({"worker_ids": ["w-watch"], "group": null, "initial": true}),
        )
        .await;
    assert!(
        reply["result"]["watching"]
            .as_array()
            .unwrap()
            .contains(&serde_json::json!("w-watch")),
        "{reply:?}"
    );

    let mut b = Raw::connect(&dir.path().join("hub.sock")).await;
    b.request("hub/hello", serde_json::json!({"agent_id": "agent-b"}))
        .await;
    let reply = b
        .request(
            "hub/watch",
            serde_json::json!({"worker_ids": [], "group": null, "initial": true}),
        )
        .await;
    assert_eq!(reply["result"]["watching"], serde_json::json!([]));
    assert_eq!(reply["result"]["events"], serde_json::json!([]));

    let mut admin = Raw::connect(&dir.path().join("hub.sock")).await;
    admin
        .request(
            "hub/hello",
            serde_json::json!({"agent_id": "agent-b", "admin": true}),
        )
        .await;
    let reply = admin
        .request(
            "hub/watch",
            serde_json::json!({"worker_ids": ["w-watch"], "group": null, "initial": true}),
        )
        .await;
    assert!(reply.get("error").is_none(), "{reply:?}");
    assert!(
        reply["result"]["watching"]
            .as_array()
            .unwrap()
            .contains(&serde_json::json!("w-watch")),
        "{reply:?}"
    );

    task.abort();
    let _ = task.await;
}

#[tokio::test]
async fn completed_worker_is_reported_immediately_with_missed_marker() {
    isolate_registry();
    let dir = common::TempDir::new_in_tmp("wm");
    let server = pool_with(vec![record(
        "w-done",
        "agent-a",
        WorkerState::Completed {
            turns: 2,
            diff: String::new(),
            summary: "Fixed.".to_string(),
            completed_at: 0,
            artifacts: Vec::new(),
            branch: Some("worker-w-done".to_string()),
            verified: Some(true),
            metrics: WorkerMetrics::default(),
            revision: 0,
        },
    )])
    .await;
    let daemon = HubServer::new(server, HubConfig::new(paths(dir.path()), 60));
    let task = tokio::spawn(async move { daemon.run().await });
    wait_for_socket(&dir.path().join("hub.sock")).await;

    let mut owner = Raw::connect(&dir.path().join("hub.sock")).await;
    owner
        .request("hub/hello", serde_json::json!({"agent_id": "agent-a"}))
        .await;
    // Let the daemon's 1 s watch loop observe the terminal worker first.
    tokio::time::sleep(std::time::Duration::from_millis(1500)).await;
    let reply = owner
        .request(
            "hub/watch",
            serde_json::json!({"worker_ids": [], "group": null, "initial": true}),
        )
        .await;
    let event = &reply["result"]["events"][0];
    assert_eq!(event["worker_id"], "w-done");
    assert_eq!(event["event"], "completed");
    assert_eq!(event["missed"], true);
    assert_eq!(event["branch"], "worker-w-done");
    assert!(
        event["commands"]
            .as_array()
            .map(|actions| actions
                .iter()
                .any(|action| action.as_str().unwrap_or("").contains("mini-swe-mcp steer")))
            .unwrap_or(false),
        "{event:?}"
    );
    assert!(
        event["next_step"]
            .as_str()
            .unwrap_or("")
            .contains("worker-w-done"),
        "{event:?}"
    );

    // Acknowledging removes the backlog; the next watch has nothing to replay.
    owner
        .request(
            "hub/watch/ack",
            serde_json::json!({"sequence": event["sequence"]}),
        )
        .await;
    let reply = owner
        .request(
            "hub/watch",
            serde_json::json!({"worker_ids": ["w-done"], "group": null, "initial": false}),
        )
        .await;
    assert_eq!(reply["result"]["events"], serde_json::json!([]));

    let mut b = Raw::connect(&dir.path().join("hub.sock")).await;
    b.request("hub/hello", serde_json::json!({"agent_id": "agent-b"}))
        .await;
    let reply = b
        .request(
            "hub/watch",
            serde_json::json!({"worker_ids": [], "group": null, "initial": true}),
        )
        .await;
    assert_eq!(
        reply["result"]["events"],
        serde_json::json!([]),
        "A must never replay to B"
    );

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
        common::binary_command(&exe)
            .args(args)
            .env("SWE_HUB_DIR", hub.path())
            .env("SWE_TEMP_DIR", swe.path())
            .env("TMPDIR", swe.path())
            .env("MINI_SWE_NO_DAEMON", "1")
            .env("ENV_FILE", "/nonexistent-mini-swe-env")
            .env("OPENAI_API_KEY", "test-key-not-used")
            .env(
                "MODELS_FILE",
                concat!(env!("CARGO_MANIFEST_DIR"), "/models.yaml"),
            )
            .output()
            .unwrap_or_else(|e| panic!("run {args:?}: {e}"))
    };
    // Nothing to watch: no Running/Paused worker of this identity.
    let output = run(&["watch", "--timeout", "1"]);
    assert_eq!(
        output.status.code(),
        Some(3),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    let completed = format!(
        r#"{{"id":"w-cli","pid":{},"task":"t","model":"m","status":"completed","step":2,"max_turns":10,"last_command":"done","started_at":1,"updated_at":2,"owner":"{}"}}"#,
        std::process::id(),
        common::host_of_this_process()
    );
    std::fs::write(
        swe.path().join("swe-registry").join("w-cli.json"),
        completed,
    )
    .expect("row");
    std::fs::create_dir_all(swe.path().join("swe-wt-w-cli")).expect("preserved worktree");
    let output = run(&["watch", "w-cli"]);
    assert_eq!(
        output.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    assert!(
        stdout.contains("w-cli")
            && stdout.contains("completed")
            && stdout.contains("mini-swe-mcp steer"),
        "{stdout}"
    );
    let _ = std::fs::remove_dir_all(swe.path().join("swe-registry"));
    // The registry-only row carries no branch, so the guidance falls back to
    // the branch-less wording instead of naming a branch that is not there.
    assert!(stdout.contains("Review the result"), "{stdout}");
    // The same worker, now gone: still nothing to watch.
    std::fs::create_dir_all(swe.path().join("swe-registry")).unwrap();
    let now = mini_swe_mcp::pool::unix_timestamp();
    let running = serde_json::json!({"id":"w-cli","pid":std::process::id(),"task":"t","model":"m","status":"running","step":1,"max_turns":10,"last_command":"test","started_at":now,"updated_at":now,"owner":common::host_of_this_process()});
    std::fs::write(
        swe.path().join("swe-registry/w-cli.json"),
        running.to_string(),
    )
    .unwrap();
    let output = run(&["watch", "w-cli", "--timeout", "1"]);
    assert_eq!(
        output.status.code(),
        Some(2),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("no event"));
}

#[test]
fn tool_description_stays_short_and_points_at_the_help_topics() {
    let manifest = ModelManifest::default();
    let server = McpServer::new(
        WorkerPool::with_scratch(
            1,
            "http://localhost:1".to_string(),
            "k".to_string(),
            mini_swe_mcp::worktree::ScratchRoot::new(
                common::TempDir::new_in_tmp("watch-tools").path(),
            ),
        )
        .with_manifest(Arc::new(manifest)),
        "m".to_string(),
    );
    let text = serde_json::to_string(&server.tools_list()).expect("list");
    for needle in [
        "mini-swe-mcp watch",
        "mini-swe-mcp help <topic>",
        "own workers",
        "no_event",
        "steer",
    ] {
        assert!(
            text.contains(needle),
            "tool schema must carry the calling rules ({needle} missing)"
        );
    }
    // The long orchestrator guidelines moved to `mini-swe-mcp help <topic>`.
    assert!(
        !text.contains("many workers at once is the intended use"),
        "the payload must not carry the long-form guidelines: {text}"
    );
}

/// The real binary against an in-process daemon: the immediate event, the
/// timeout and the ownership refusal all go through the hub transport.
///
/// The daemon needs its own worker thread: the test body blocks on the child
/// process, and a current-thread runtime would never let the daemon answer it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_binary_watches_through_the_hub() {
    isolate_registry();
    let hub = common::TempDir::new_in_tmp("watch-hub-cli");
    let swe = common::TempDir::new_in_tmp("watch-hub-swe");
    let server = pool_with(vec![
        record(
            "w-hub",
            &common::host_of_this_process(),
            WorkerState::Completed {
                turns: 3,
                diff: String::new(),
                summary: "Fixed the parser.".to_string(),
                completed_at: 0,
                artifacts: Vec::new(),
                branch: Some("worker-w-hub".to_string()),
                verified: Some(true),
                metrics: WorkerMetrics::default(),
                revision: 0,
            },
        ),
        // Another agent's live worker: never watchable, never leaked.
        record(
            "w-foreign",
            "other",
            WorkerState::Running {
                step: 1,
                last_command: "cargo test".to_string(),
                started_at: 0,
            },
        ),
    ])
    .await;
    let daemon = HubServer::new(server, HubConfig::new(paths(hub.path()), 60));
    let task = tokio::spawn(async move { daemon.run().await });
    wait_for_socket(&hub.path().join("hub.sock")).await;
    let run = |args: &[&str]| {
        common::binary_command(&common::binary_path())
            .args(args)
            .env("SWE_HUB_DIR", hub.path())
            .env("SWE_TEMP_DIR", swe.path())
            .env("TMPDIR", swe.path())
            .env("OPENAI_API_KEY", "test-key-not-used")
            .env("ENV_FILE", "/nonexistent-mini-swe-env")
            .env(
                "MODELS_FILE",
                concat!(env!("CARGO_MANIFEST_DIR"), "/models.yaml"),
            )
            .output()
            .unwrap_or_else(|e| panic!("run {args:?}: {e}"))
    };
    let output = run(&["watch", "w-hub", "--timeout", "10"]);
    assert_eq!(
        output.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    assert!(
        stdout.contains("w-hub") && stdout.contains("completed"),
        "{stdout}"
    );
    assert!(stdout.contains("While you were not watching:"), "{stdout}");
    assert!(stdout.contains("worker-w-hub"), "{stdout}");
    // The other agent's worker is refused, and never leaks into the output.
    let output = run(&["watch", "w-foreign", "--timeout", "5"]);
    assert_eq!(
        output.status.code(),
        Some(4),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("belongs to agent other"));
    let output = run(&["watch", "--timeout", "3"]);
    assert_eq!(
        output.status.code(),
        Some(3),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    task.abort();
    let _ = task.await;
}

#[test]
fn one_watch_call_replays_every_missed_event() {
    let dir = common::TempDir::new_in_tmp("watch-replay");
    let registry = dir.path().join("swe-registry");
    std::fs::create_dir_all(&registry).expect("create the scratch registry");
    // The registry prunes a terminal row with neither a worktree nor a branch,
    // so each missed worker leaves the branch its torn-down worktree pushed.
    let repo = dir.subdir("repo");
    common::git(&repo, &["init", "-q", "-b", "main"]);
    common::git(&repo, &["config", "user.email", "test@example.invalid"]);
    common::git(&repo, &["config", "user.name", "test"]);
    std::fs::write(repo.join("base.txt"), "base\n").expect("write the base file");
    common::git(&repo, &["add", "-A"]);
    common::git(&repo, &["commit", "-qm", "base"]);
    for id in ["w-first", "w-second"] {
        let branch = format!("worker-{id}");
        common::git(&repo, &["checkout", "-q", "-b", &branch, "main"]);
        std::fs::write(repo.join(format!("{id}.txt")), format!("{id}\n")).expect("write");
        common::git(&repo, &["add", "-A"]);
        common::git(&repo, &["commit", "-qm", &branch]);
        common::git(&repo, &["checkout", "-q", "main"]);
        let row = serde_json::json!({
            "id": id, "pid": std::process::id(), "task": "t", "model": "test",
            "status": "completed", "step": 2, "max_turns": 10,
            "last_command": "cargo test", "started_at": 1, "updated_at": 1_700_000_000u64,
            "owner": "agent-a", "repo_path": repo.to_string_lossy(),
        })
        .to_string();
        std::fs::write(registry.join(format!("{id}.json")), row).expect("write the row");
    }

    let exe = common::binary_path();
    let output = common::binary_command(&exe)
        .args(["--json", "watch", "w-first", "w-second"])
        .current_dir(dir.path())
        .env("MINI_SWE_NO_DAEMON", "1")
        .env("SWE_TEMP_DIR", dir.path())
        .env("MINI_SWE_AGENT_ID", "agent-a")
        .env("HOME", dir.path())
        .env("XDG_CONFIG_HOME", dir.path().join(".config"))
        .env("ENV_FILE", dir.path().join(".env.does-not-exist"))
        .env_remove("OPENAI_API_KEY")
        .output()
        .unwrap_or_else(|e| panic!("failed to run {}: {e}", exe.display()));
    let stdout = common::stdout_of(&output);
    assert!(
        output.status.success(),
        "one call must exit 0: {stdout}{}",
        common::stderr_of(&output)
    );
    let replayed: Vec<String> = stdout
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .filter_map(|event| event["worker_id"].as_str().map(str::to_string))
        .collect();
    assert!(
        replayed.contains(&"w-first".to_string()) && replayed.contains(&"w-second".to_string()),
        "one call must replay every missed event: {stdout}"
    );
}

/// A no-arg MCP `watch` follows a worker dispatched after it started: its set
/// is re-evaluated on every poll instead of frozen at the first one.
#[tokio::test]
async fn a_no_arg_watch_action_follows_late_dispatches() {
    isolate_registry();
    let pool = WorkerPool::new(4, "http://localhost:1".to_string(), "test-key".to_string())
        .with_manifest(Arc::new(ModelManifest::default()));
    pool.__test_insert_worker(record(
        "w-mcp-first",
        mini_swe_mcp::mcp::LOCAL_AGENT,
        WorkerState::Running {
            step: 1,
            last_command: "cargo test".to_string(),
            started_at: 0,
        },
    ))
    .await;
    let server = Arc::new(McpServer::new(pool.clone(), "test".to_string()));

    let watch = tokio::spawn({
        let server = server.clone();
        async move {
            server
                .execute_tool(
                    "worker",
                    serde_json::json!({"action": "watch", "timeout_secs": 10}),
                )
                .await
        }
    });
    // Let the watch resolve its initial set to w-mcp-first.
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    // The late dispatch: only its completion should wake the watch.
    pool.__test_insert_worker(record(
        "w-mcp-second",
        mini_swe_mcp::mcp::LOCAL_AGENT,
        WorkerState::Completed {
            turns: 2,
            diff: String::new(),
            summary: "late worker done".to_string(),
            completed_at: 0,
            artifacts: Vec::new(),
            branch: Some("worker-w-mcp-second".to_string()),
            verified: Some(true),
            metrics: WorkerMetrics::default(),
            revision: 0,
        },
    ))
    .await;

    let result = tokio::time::timeout(std::time::Duration::from_secs(8), watch)
        .await
        .expect("a late dispatch must wake the no-arg watch")
        .expect("the watch task stays alive")
        .expect("the watch answers");
    assert_eq!(result["status"], "event", "{result}");
    let events = result["events"].as_array().expect("events array");
    assert!(
        events
            .iter()
            .any(|event| event["worker_id"] == "w-mcp-second"),
        "the late worker must be reported: {result}"
    );
}

/// The backgrounded `mini-swe-mcp watch` the orchestrator runs must follow a
/// worker dispatched after it started, not the set it saw first.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_no_arg_watch_through_the_hub_follows_late_dispatches() {
    isolate_registry();
    let hub = common::TempDir::new_in_tmp("watch-late-hub");
    let swe = common::TempDir::new_in_tmp("watch-late-swe");
    let pool = WorkerPool::new(4, "http://localhost:1".to_string(), "test-key".to_string())
        .with_manifest(Arc::new(ModelManifest::default()));
    let owner = common::host_of_this_process();
    pool.__test_insert_worker(record(
        "w-late-first",
        &owner,
        WorkerState::Running {
            step: 1,
            last_command: "cargo test".to_string(),
            started_at: 0,
        },
    ))
    .await;
    let server = Arc::new(McpServer::new(pool.clone(), "test".to_string()));
    let daemon = HubServer::new(server, HubConfig::new(paths(hub.path()), 60));
    let task = tokio::spawn(async move { daemon.run().await });
    wait_for_socket(&hub.path().join("hub.sock")).await;

    let child = common::binary_command(&common::binary_path())
        .args(["--json", "watch", "--timeout", "20"])
        .env("SWE_HUB_DIR", hub.path())
        .env("SWE_TEMP_DIR", swe.path())
        .env("TMPDIR", swe.path())
        .env("OPENAI_API_KEY", "test-key-not-used")
        .env("ENV_FILE", "/nonexistent-mini-swe-env")
        .env(
            "MODELS_FILE",
            concat!(env!("CARGO_MANIFEST_DIR"), "/models.yaml"),
        )
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("spawn the backgrounded watch");

    // Let the watch resolve its initial set to w-late-first.
    tokio::time::sleep(std::time::Duration::from_millis(1200)).await;
    pool.__test_insert_worker(record(
        "w-late-second",
        &owner,
        WorkerState::Completed {
            turns: 2,
            diff: String::new(),
            summary: "late worker done".to_string(),
            completed_at: 0,
            artifacts: Vec::new(),
            branch: Some("worker-w-late-second".to_string()),
            verified: Some(true),
            metrics: WorkerMetrics::default(),
            revision: 0,
        },
    ))
    .await;

    let output = tokio::time::timeout(
        std::time::Duration::from_secs(15),
        tokio::task::spawn_blocking(move || child.wait_with_output()),
    )
    .await
    .expect("the backgrounded watch must wake for the late dispatch")
    .expect("the wait task stays alive")
    .expect("child output");
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    assert_eq!(
        output.status.code(),
        Some(0),
        "{stdout}{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let reported: Vec<String> = stdout
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .filter_map(|event| event["worker_id"].as_str().map(str::to_string))
        .collect();
    assert!(
        reported.contains(&"w-late-second".to_string()),
        "the late worker must be reported: {stdout}"
    );

    task.abort();
    let _ = task.await;
}

/// A no-arg registry-polling watch follows a worker registered after it
/// started, and `--group` still filters late arrivals.
#[test]
fn a_no_arg_polling_watch_follows_late_dispatches_under_a_group() {
    let dir = common::TempDir::new_in_tmp("watch-late-poll");
    let registry = dir.subdir("swe-registry");
    let owner = common::host_of_this_process();
    let now = mini_swe_mcp::pool::unix_timestamp();
    let row = |id: &str, status: &str, group: &str| {
        serde_json::json!({
            "id": id, "pid": std::process::id(), "task": "watch probe", "model": "test",
            "status": status, "step": 2, "max_turns": 10, "last_command": "cargo test",
            "started_at": 1, "updated_at": now, "owner": owner, "group": group,
        })
        .to_string()
    };
    std::fs::write(
        registry.join("w-first.json"),
        row("w-first", "running", "g1"),
    )
    .expect("first row");

    let child = common::binary_command(&common::binary_path())
        .args(["--json", "watch", "--group", "g1", "--timeout", "15"])
        .env("MINI_SWE_NO_DAEMON", "1")
        .env("SWE_TEMP_DIR", dir.path())
        .env("TMPDIR", dir.path())
        .env("ENV_FILE", "/nonexistent-mini-swe-env")
        .env_remove("OPENAI_API_KEY")
        .env(
            "MODELS_FILE",
            concat!(env!("CARGO_MANIFEST_DIR"), "/models.yaml"),
        )
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("spawn the watch");

    // Let the watch resolve its initial set to w-first.
    std::thread::sleep(std::time::Duration::from_millis(1500));
    // A late worker in another group is never watched.
    std::fs::write(
        registry.join("w-other.json"),
        row("w-other", "completed", "g2"),
    )
    .expect("other row");
    std::thread::sleep(std::time::Duration::from_millis(1200));
    // A late worker in the watched group is reported; its worktree keeps the
    // terminal row from being pruned.
    let _ = dir.subdir("swe-wt-w-late");
    std::fs::write(
        registry.join("w-late.json"),
        row("w-late", "completed", "g1"),
    )
    .expect("late row");

    let output = child.wait_with_output().expect("the watch must exit");
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    assert_eq!(
        output.status.code(),
        Some(0),
        "{stdout}{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let reported: Vec<String> = stdout
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .filter_map(|event| event["worker_id"].as_str().map(str::to_string))
        .collect();
    assert!(reported.contains(&"w-late".to_string()), "{stdout}");
    assert!(!reported.contains(&"w-other".to_string()), "{stdout}");
}

/// One watch per identity: a second connection of the same agent is refused
/// and named the first, another agent watches at the same time, and the slot
/// frees when the holder disconnects.
#[tokio::test]
async fn the_daemon_allows_one_watch_per_identity() {
    isolate_registry();
    let dir = common::TempDir::new_in_tmp("wg-one-daemon");
    let server = pool_with(vec![
        record(
            "w-one-a",
            "agent-a",
            WorkerState::Running {
                step: 1,
                last_command: "t".to_string(),
                started_at: 0,
            },
        ),
        record(
            "w-one-b",
            "agent-b",
            WorkerState::Running {
                step: 1,
                last_command: "t".to_string(),
                started_at: 0,
            },
        ),
    ])
    .await;
    let daemon = HubServer::new(server, HubConfig::new(paths(dir.path()), 60));
    let task = tokio::spawn(async move { daemon.run().await });
    let socket = dir.path().join("hub.sock");
    wait_for_socket(&socket).await;
    let watch = serde_json::json!({"worker_ids": [], "group": null, "initial": true});

    let mut a1 = Raw::connect(&socket).await;
    a1.request(
        "hub/hello",
        serde_json::json!({"agent_id": "agent-a", "pid": 1111}),
    )
    .await;
    let reply = a1.request("hub/watch", watch.clone()).await;
    assert!(reply.get("error").is_none(), "{reply:?}");

    // A second connection of the same identity is refused, naming the first.
    let mut a2 = Raw::connect(&socket).await;
    a2.request("hub/hello", serde_json::json!({"agent_id": "agent-a"}))
        .await;
    let reply = a2.request("hub/watch", watch.clone()).await;
    let message = reply["error"]["message"].as_str().unwrap_or_default();
    assert!(message.contains("a watch is already running"), "{reply:?}");
    assert!(message.contains("pid 1111"), "{reply:?}");

    // A different identity watches at the same time.
    let mut b = Raw::connect(&socket).await;
    b.request("hub/hello", serde_json::json!({"agent_id": "agent-b"}))
        .await;
    let reply = b.request("hub/watch", watch.clone()).await;
    assert!(reply.get("error").is_none(), "{reply:?}");
    assert!(
        reply["result"]["watching"]
            .as_array()
            .is_some_and(|ids| ids.contains(&serde_json::json!("w-one-b"))),
        "{reply:?}"
    );

    // Closing the holder frees the slot for the identity's next watch.
    drop(a1);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(3);
    let a3 = loop {
        assert!(
            std::time::Instant::now() < deadline,
            "the slot must free when its connection closes"
        );
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        let mut candidate = Raw::connect(&socket).await;
        candidate
            .request("hub/hello", serde_json::json!({"agent_id": "agent-a"}))
            .await;
        let reply = candidate.request("hub/watch", watch.clone()).await;
        if reply.get("error").is_none() {
            break candidate;
        }
    };
    let _ = a3; // Keep the reconnected watch alive until the daemon stops.

    task.abort();
    let _ = task.await;
}

/// The MCP `watch` action obeys the same one-watch-per-identity rule.
#[tokio::test]
async fn the_mcp_watch_action_allows_one_watch_per_identity() {
    isolate_registry();
    let pool = WorkerPool::new(4, "http://localhost:1".to_string(), "test-key".to_string())
        .with_manifest(Arc::new(ModelManifest::default()));
    pool.__test_insert_worker(record(
        "w-mcp-one",
        mini_swe_mcp::mcp::LOCAL_AGENT,
        WorkerState::Running {
            step: 1,
            last_command: "cargo test".to_string(),
            started_at: 0,
        },
    ))
    .await;
    let server = Arc::new(McpServer::new(pool, "test".to_string()));

    let first = tokio::spawn({
        let server = server.clone();
        async move {
            server
                .execute_tool(
                    "worker",
                    serde_json::json!({"action": "watch", "timeout_secs": 10}),
                )
                .await
        }
    });
    // Let the first watch reserve the identity's slot.
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;

    let error = server
        .execute_tool(
            "worker",
            serde_json::json!({"action": "watch", "timeout_secs": 10}),
        )
        .await
        .expect_err("a second watch for the same identity must be refused");
    assert!(
        error.to_string().contains("a watch is already running"),
        "{error}"
    );

    // A different identity may watch at the same time.
    let other = mini_swe_mcp::mcp::ConnectionContext {
        agent_id: Some("agent-other".to_string()),
        ..mini_swe_mcp::mcp::ConnectionContext::hub_connection(9)
    };
    let reply = server
        .execute_tool_for(
            "worker",
            serde_json::json!({"action": "watch", "timeout_secs": 0}),
            &other,
        )
        .await
        .expect("a different identity watches concurrently");
    assert_eq!(reply["status"], "no_event", "{reply}");

    // Cancelling the holder frees its slot.
    first.abort();
    let _ = first.await;
    let reply = server
        .execute_tool(
            "worker",
            serde_json::json!({"action": "watch", "timeout_secs": 0}),
        )
        .await
        .expect("the freed slot accepts a new watch");
    assert_eq!(reply["status"], "no_event", "{reply}");
}

/// The CLI maps a refused second watch of one session to exit code 5 and
/// accepts a fresh watch once the holder has exited.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_second_cli_watch_for_one_session_exits_five() {
    isolate_registry();
    let hub = common::TempDir::new_in_tmp("wg-cli-hub");
    let swe = common::TempDir::new_in_tmp("wg-cli-swe");
    let pool = WorkerPool::new(4, "http://localhost:1".to_string(), "test-key".to_string())
        .with_manifest(Arc::new(ModelManifest::default()));
    let owner = common::host_of_this_process();
    pool.__test_insert_worker(record(
        "w-cli-one",
        &owner,
        WorkerState::Running {
            step: 1,
            last_command: "cargo test".to_string(),
            started_at: 0,
        },
    ))
    .await;
    let server = Arc::new(McpServer::new(pool, "test".to_string()));
    let daemon = HubServer::new(server, HubConfig::new(paths(hub.path()), 60));
    let task = tokio::spawn(async move { daemon.run().await });
    wait_for_socket(&hub.path().join("hub.sock")).await;

    let spawn_watch = |timeout: &str| {
        common::binary_command(&common::binary_path())
            .args(["--json", "watch", "--timeout", timeout])
            .env("SWE_HUB_DIR", hub.path())
            .env("SWE_TEMP_DIR", swe.path())
            .env("TMPDIR", swe.path())
            .env("OPENAI_API_KEY", "test-key-not-used")
            .env("ENV_FILE", "/nonexistent-mini-swe-env")
            .env(
                "MODELS_FILE",
                concat!(env!("CARGO_MANIFEST_DIR"), "/models.yaml"),
            )
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .expect("spawn watch")
    };
    let mut first = spawn_watch("30");
    // Let the first watch reserve the session's slot.
    tokio::time::sleep(std::time::Duration::from_millis(1200)).await;

    let second = spawn_watch("5");
    let out = tokio::task::spawn_blocking(move || second.wait_with_output())
        .await
        .expect("the wait task stays alive")
        .expect("the second watch exits");
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    assert_eq!(
        out.status.code(),
        Some(5),
        "{stdout}{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(stdout.contains("a watch is already running"), "{stdout}");

    // Once the holder exits, the session may watch again.
    let _ = first.kill();
    let _ = first.wait();
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    let third = spawn_watch("2");
    let out = tokio::task::spawn_blocking(move || third.wait_with_output())
        .await
        .expect("the wait task stays alive")
        .expect("the third watch exits");
    assert_eq!(
        out.status.code(),
        Some(2),
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    task.abort();
    let _ = task.await;
}

/// A worker queued for a heavy build slot is not stalled, however long the
/// wait: the admission wait is its own state, and the stall detector skips it.
#[test]
fn a_queued_build_slot_is_never_a_stall() {
    let mut view = serde_json::json!({
        "worker_id": "w-slot",
        "owner": "local",
        "status": "running",
        "step": 4,
        "turns": 4,
        "revision": 0,
        "question": null,
        "branch": null,
        "metrics": {},
        "last_step_at": 400u64,
        "waiting_for_slot": 3,
    });
    let now = 1001; // 601 s since the last step, well past the stall threshold.
    assert!(
        watch::select_event(&view, None, now).is_none(),
        "a queued build slot must never be reported as stalled"
    );

    // Without the admission wait the same idle is a stall.
    view["waiting_for_slot"] = serde_json::json!(null);
    let event = watch::select_event(&view, None, now).expect("idle with no wait is a stall");
    assert_eq!(event["event"], "stalled");
}

/// Waiting for a build slot keeps the idle clock at zero, so granting the slot
/// starts a fresh episode instead of a stall the moment admission succeeds.
#[test]
fn a_build_slot_wait_keeps_the_idle_clock_at_zero() {
    let mut view = serde_json::json!({
        "step": 1,
        "revision": 0,
        "last_step_at": 10u64,
        "waiting_for_slot": 2,
    });
    let old = serde_json::json!({"step": 1, "revision": 0, "last_step_at": 10u64});
    watch::progress_clock(&mut view, Some(&old), 5000);
    assert_eq!(view["last_step_at"], serde_json::json!(5000));
}

/// The pool publishes a worker's queued build slot and clears it as soon as the
/// wait ends, so the exposed state tracks admission exactly.
#[tokio::test]
async fn the_pool_exposes_then_clears_a_build_slot_wait() {
    let pool = WorkerPool::new(4, "http://localhost:1".to_string(), "test-key".to_string());
    pool.__test_insert_worker(record(
        "w-slot-pool",
        "local",
        WorkerState::Running {
            step: 1,
            last_command: "cargo build".to_string(),
            started_at: 0,
        },
    ))
    .await;
    assert_eq!(
        pool.worker_progress("w-slot-pool")
            .await
            .unwrap()
            .waiting_for_slot,
        None
    );
    let wait = pool.wait_for_build_slot("w-slot-pool", 4);
    assert_eq!(
        pool.worker_progress("w-slot-pool")
            .await
            .unwrap()
            .waiting_for_slot,
        Some(4)
    );
    drop(wait);
    assert_eq!(
        pool.worker_progress("w-slot-pool")
            .await
            .unwrap()
            .waiting_for_slot,
        None
    );
}

#[test]
fn torn_down_worker_diff_stat_comes_from_its_branch() {
    let dir = common::TempDir::new_in_tmp("watch-branch");
    let repo = dir.subdir("repo");
    common::git(&repo, &["init", "-q", "-b", "main"]);
    common::git(&repo, &["config", "user.email", "test@example.invalid"]);
    common::git(&repo, &["config", "user.name", "test"]);
    std::fs::write(repo.join("base.txt"), "base\n").expect("write the base file");
    common::git(&repo, &["add", "-A"]);
    common::git(&repo, &["commit", "-qm", "base"]);
    common::git(&repo, &["checkout", "-q", "-b", "worker-w-gone"]);
    std::fs::write(repo.join("one.txt"), "one\n").expect("write");
    std::fs::write(repo.join("two.txt"), "two\n").expect("write");
    common::git(&repo, &["add", "-A"]);
    common::git(&repo, &["commit", "-qm", "work"]);
    // The worker's worktree is gone; the repo sits back on its base branch.
    common::git(&repo, &["checkout", "-q", "main"]);

    let entry = WorkerRegistryEntry {
        id: "w-gone".to_string(),
        pid: std::process::id(),
        task: "fix the parser".to_string(),
        model: "test".to_string(),
        status: RegistryStatus::Completed,
        step: 2,
        max_turns: 10,
        last_command: "cargo test".to_string(),
        question: None,
        started_at: 0,
        updated_at: 0,
        group: None,
        repo_path: Some(repo.to_string_lossy().to_string()),
        owner: Some("agent-a".to_string()),
        metrics: WorkerMetrics::default(),
        base_branch: Some("main".to_string()),
        base_commit: None,
        revision: 0,
        auto_continues: 0,
    };
    let now = 1_700_000_000;
    let view = watch::registry_snapshot(&entry, now);
    let event = watch::select_event(&view, None, now).expect("a completed worker yields an event");
    let text = watch::render(&event);
    assert!(text.contains("Diff: 2 files, +2 -0"), "{text}");
}

// ---------------------------------------------------------------------------
// An older hub that answers `hub/watch` without the fields this client reads
// must not look like an empty watch set: the client says so and polls the
// registry, where the same workers still report.
// ---------------------------------------------------------------------------

/// A hub that speaks the version handshake and `hub/watch`, but answers the
/// watch with the pre-`watching` reply shape.
async fn fake_old_hub(socket: PathBuf) {
    let _ = std::fs::remove_file(&socket);
    let listener = tokio::net::UnixListener::bind(&socket).expect("bind the fake hub");
    loop {
        let Ok((stream, _)) = listener.accept().await else {
            return;
        };
        tokio::spawn(async move {
            let (read, mut write) = stream.into_split();
            let mut reader = BufReader::new(read);
            loop {
                let mut line = String::new();
                if reader.read_line(&mut line).await.unwrap_or(0) == 0 {
                    return;
                }
                let Ok(request) = serde_json::from_str::<serde_json::Value>(line.trim()) else {
                    continue;
                };
                // Notifications carry no id and need no reply.
                let Some(id) = request.get("id").cloned() else {
                    continue;
                };
                let result = match request["method"].as_str() {
                    Some("hub/hello") => {
                        serde_json::json!({"version": "999.0.0", "build": null, "busy": false})
                    }
                    // The old shape: no `watching`, only `events`.
                    Some("hub/watch") => serde_json::json!({"events": []}),
                    _ => serde_json::json!({}),
                };
                let reply = serde_json::json!({"jsonrpc": "2.0", "id": id, "result": result});
                if write
                    .write_all(format!("{reply}\n").as_bytes())
                    .await
                    .is_err()
                {
                    return;
                }
                let _ = write.flush().await;
            }
        });
    }
}

fn run_watch_via(hub: &Path, swe: &Path, args: &[&str]) -> std::process::Output {
    common::binary_command(&common::binary_path())
        .args(args)
        .env("SWE_HUB_DIR", hub)
        .env("SWE_TEMP_DIR", swe)
        .env("TMPDIR", swe)
        .env("ENV_FILE", "/nonexistent-mini-swe-env")
        .env("OPENAI_API_KEY", "test-key-not-used")
        .env(
            "MODELS_FILE",
            concat!(env!("CARGO_MANIFEST_DIR"), "/models.yaml"),
        )
        .output()
        .expect("run watch")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_old_hub_watch_reply_falls_back_to_the_registry() {
    let hub = common::TempDir::new_in_tmp("watch-old-hub");
    let swe = common::TempDir::new_in_tmp("watch-old-swe");
    let registry = swe.subdir("swe-registry");
    let socket = paths(hub.path()).socket();
    let fake = tokio::spawn(fake_old_hub(socket.clone()));
    wait_for_socket(&socket).await;

    // Nothing in the registry: the fallback must still say why it stopped.
    let (hub_dir, swe_dir) = (hub.path().to_path_buf(), swe.path().to_path_buf());
    let output = tokio::task::spawn_blocking(move || {
        run_watch_via(&hub_dir, &swe_dir, &["watch", "--timeout", "1"])
    })
    .await
    .expect("the run task stays alive");
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    assert!(
        stderr.contains("falling back to registry polling"),
        "{stderr}"
    );
    assert_eq!(output.status.code(), Some(3), "{stdout}{stderr}");
    assert!(stdout.contains("nothing to watch"), "{stdout}");

    // A synthetic registry row is still reported by the polling path. Its
    // worktree keeps the terminal row from being pruned before the poll.
    let owner = common::host_of_this_process();
    let _ = swe.subdir("swe-wt-w-synth");
    std::fs::write(
        registry.join("w-synth.json"),
        serde_json::json!({
            "id": "w-synth", "pid": std::process::id(), "task": "watch probe",
            "model": "test", "status": "completed", "step": 2, "max_turns": 10,
            "last_command": "cargo test", "started_at": 1, "updated_at": 2,
            "owner": owner,
        })
        .to_string(),
    )
    .expect("synthetic row");
    let (hub_dir, swe_dir) = (hub.path().to_path_buf(), swe.path().to_path_buf());
    let output = tokio::task::spawn_blocking(move || {
        run_watch_via(&hub_dir, &swe_dir, &["watch", "w-synth"])
    })
    .await
    .expect("the run task stays alive");
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    assert_eq!(output.status.code(), Some(0), "{stdout}{stderr}");
    assert!(
        stdout.contains("w-synth") && stdout.contains("completed"),
        "{stdout}"
    );

    fake.abort();
    let _ = fake.await;
}

/// A watch whose workers disappear before any event fires must not end as a
/// silent success: it names why and exits non-zero.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_drained_polling_watch_never_exits_zero_silently() {
    let dir = common::TempDir::new_in_tmp("watch-drain");
    let registry = dir.subdir("swe-registry");
    let now = mini_swe_mcp::pool::unix_timestamp();
    std::fs::write(
        registry.join("w-drain.json"),
        serde_json::json!({
            "id": "w-drain", "pid": std::process::id(), "task": "watch probe",
            "model": "test", "status": "running", "step": 1, "max_turns": 10,
            "last_command": "cargo test", "started_at": 1, "updated_at": now,
            "owner": common::host_of_this_process(),
        })
        .to_string(),
    )
    .expect("running row");

    let child = common::binary_command(&common::binary_path())
        .args(["watch", "w-drain"])
        .env("MINI_SWE_NO_DAEMON", "1")
        .env("SWE_TEMP_DIR", dir.path())
        .env("TMPDIR", dir.path())
        .env("ENV_FILE", "/nonexistent-mini-swe-env")
        .env_remove("OPENAI_API_KEY")
        .env(
            "MODELS_FILE",
            concat!(env!("CARGO_MANIFEST_DIR"), "/models.yaml"),
        )
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("spawn the watch");
    // Let the first poll see the running worker, then tear it away.
    tokio::time::sleep(std::time::Duration::from_millis(1500)).await;
    std::fs::remove_file(registry.join("w-drain.json")).expect("remove the row");

    let output = tokio::time::timeout(
        std::time::Duration::from_secs(10),
        tokio::task::spawn_blocking(move || child.wait_with_output()),
    )
    .await
    .expect("a drained watch must end")
    .expect("the wait task stays alive")
    .expect("child output");
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    assert_ne!(
        output.status.code(),
        Some(0),
        "a watch that never printed an event must not exit 0: {stdout}"
    );
    assert!(!stdout.trim().is_empty(), "the exit must say why: {stdout}");
}
