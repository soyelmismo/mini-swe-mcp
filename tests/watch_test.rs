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
    let pool = WorkerPool::new(4, "http://localhost:1".to_string(), "test-key".to_string())
        .with_manifest(Arc::new(ModelManifest::default()));
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
fn tool_description_carries_the_orchestrator_guidelines() {
    let manifest = ModelManifest::default();
    let server = McpServer::new(
        WorkerPool::new(1, "http://localhost:1".to_string(), "k".to_string())
            .with_manifest(Arc::new(manifest)),
        "m".to_string(),
    );
    let text = serde_json::to_string(&server.tools_list()).expect("list");
    for needle in [
        "ONE focused concern",
        "mini-swe-mcp watch",
        "timeout_secs",
        "no_event",
        "push notifications",
        "steer",
        "merge only when it is right",
    ] {
        assert!(
            text.contains(needle),
            "tool schema must carry the guidelines ({needle} missing)"
        );
    }
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
