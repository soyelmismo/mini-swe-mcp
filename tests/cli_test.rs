//! Integration tests for the `mini-swe-mcp` CLI executable.

mod common;

use serde_json::json;
use std::path::PathBuf;
use std::process::Command;

fn binary_path() -> PathBuf {
    if let Ok(exe) = std::env::var("CARGO_BIN_EXE_mini-swe-mcp") {
        return PathBuf::from(exe);
    }
    let mut path = std::env::current_exe().expect("current exe");
    path.pop();
    if path.ends_with("deps") {
        path.pop();
    }
    path.join("mini-swe-mcp")
}

#[test]
fn test_cli_version_flag() {
    let exe = binary_path();
    for flag in ["--version", "-V"] {
        let output = common::binary_command(&exe)
            .env("MINI_SWE_NO_DAEMON", "1")
            .arg(flag)
            .output()
            .unwrap_or_else(|e| panic!("failed to run {}: {e}", exe.display()));

        assert!(output.status.success(), "flag {flag} failed");
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            stdout.contains(env!("CARGO_PKG_VERSION")),
            "version output missing package version for {flag}: {stdout}"
        );
        assert!(stdout.contains("mini-swe-mcp"));
    }
}

#[test]
fn test_cli_help_flag() {
    let exe = binary_path();
    for flag in ["--help", "-h"] {
        let output = common::binary_command(&exe)
            .env("MINI_SWE_NO_DAEMON", "1")
            .arg(flag)
            .output()
            .unwrap_or_else(|e| panic!("failed to run {}: {e}", exe.display()));

        assert!(output.status.success(), "flag {flag} failed");
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            stdout.contains("Usage: mini-swe-mcp"),
            "help output missing usage for {flag}: {stdout}"
        );
        assert!(stdout.contains("dispatch"));
        assert!(stdout.contains("prune"));
        // `watch` is the only blocking verb, and dispatch/steer never wait.
        assert!(
            !stdout.contains("[--wait]"),
            "dispatch/steer must no longer advertise --wait: {stdout}"
        );
        assert!(
            stdout.contains("watch [<worker_id>...]"),
            "help missing the watch usage: {stdout}"
        );
        // Long-form guidance moved to the `help <topic>` topics; `--help` only
        // indexes them.
        assert!(
            stdout.contains("mini-swe-mcp help <topic>"),
            "help must point at the topics: {stdout}"
        );
        for topic in ["workflow", "watch", "steer", "identity", "sandbox", "env"] {
            assert!(
                stdout.contains(topic),
                "help must list the '{topic}' topic: {stdout}"
            );
        }
        assert!(
            stdout.contains("watch [<worker_id>...]"),
            "help missing the watch usage: {stdout}"
        );
    }
}

/// `mini-swe-mcp help <topic>` prints the long-form guidance the MCP tool
/// description points at; an unknown topic is refused with the available list.
#[test]
fn test_cli_help_topics() {
    let exe = binary_path();
    for (topic, needle) in [
        ("workflow", "the default is a ROUND"),
        ("workflow", "watch --group <g> --all"),
        ("workflow", "--consolidate"),
        ("watch", "mini-swe-mcp watch"),
        ("steer", "worker-<id>"),
        ("identity", "own workers"),
        ("sandbox", "offline"),
        ("env", "OPENAI_API_KEY"),
    ] {
        let output = common::binary_command(&exe)
            .env("MINI_SWE_NO_DAEMON", "1")
            .args(["help", topic])
            .output()
            .unwrap_or_else(|e| panic!("failed to run {}: {e}", exe.display()));
        assert!(output.status.success(), "help {topic} failed");
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            stdout.contains(needle),
            "help {topic} must mention {needle}: {stdout}"
        );
    }

    let output = common::binary_command(&exe)
        .env("MINI_SWE_NO_DAEMON", "1")
        .args(["help", "nope"])
        .output()
        .unwrap_or_else(|e| panic!("failed to run {}: {e}", exe.display()));
    assert!(!output.status.success(), "an unknown topic must be refused");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("workflow"),
        "an unknown topic must list the topics: {stderr}"
    );
}

/// The `workflow` topic is the round workflow: the group dispatch, the cheap
/// worker gate, the `--all` wait for the round, the consolidator's report, and
/// the single branch that is merged. The rules an orchestrator is most likely
/// to break are the ones it must never break by hand.
#[test]
fn test_cli_help_workflow_topic_is_the_round_workflow() {
    let exe = binary_path();
    let output = common::binary_command(&exe)
        .env("MINI_SWE_NO_DAEMON", "1")
        .args(["help", "workflow"])
        .output()
        .unwrap_or_else(|e| panic!("failed to run {}: {e}", exe.display()));
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);

    for needle in [
        "ONE focused concern",
        "dispatch -f tasks.yaml --group <g> --consolidate",
        "CHEAP gate",
        "watch --group <g> --all",
        "one-line-per-worker report",
        "security-relevant parts",
        "merge <consolidator>",
        // The single-worker path stays, for the one-off task.
        "For a one-off task",
        "merge only when it is right",
        // Corrections go through steer, and the base branch stays put.
        "never edit its branch yourself",
        "Do not move the base branch while a round is consolidating",
    ] {
        assert!(
            stdout.contains(needle),
            "help workflow must mention {needle}: {stdout}"
        );
    }
}

/// The MCP-only agent cannot run the binary, so the `help` action must return
/// the same topic index and topic text the CLI prints, over the tool call.
#[test]
fn test_mcp_help_action_returns_the_index_and_a_topic() {
    let server = mini_swe_mcp::mcp::McpServer::new(
        mini_swe_mcp::pool::WorkerPool::new(1, "http://localhost:1".to_string(), "k".to_string()),
        "ninja".to_string(),
    );
    let index = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("a runtime")
        .block_on(server.execute_tool("worker", json!({ "action": "help" })))
        .expect("the help index");
    let topics: Vec<&str> = index["topics"]
        .as_array()
        .expect("a topics array")
        .iter()
        .map(|topic| topic.as_str().expect("topic names are strings"))
        .collect();
    assert_eq!(topics, mini_swe_mcp::cli::help::TOPICS.to_vec());

    let topic = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("a runtime")
        .block_on(server.execute_tool("worker", json!({ "action": "help", "topic": "workflow" })))
        .expect("the workflow topic");
    assert_eq!(topic["topic"], json!("workflow"));
    assert_eq!(
        topic["text"],
        json!(mini_swe_mcp::cli::help::topic_text("workflow").expect("workflow topic"))
    );

    // A typo is refused with the alternatives, never an empty answer.
    let error = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("a runtime")
        .block_on(server.execute_tool("worker", json!({ "action": "help", "topic": "nope" })))
        .expect_err("an unknown topic must be refused");
    assert!(error.to_string().contains("Unknown help topic"), "{error}");
}

/// `--version`/`-V` and `--help`/`-h` must be handled *before* any API-key
/// resolution, so they work even when `OPENAI_API_KEY` is absent from the
/// environment.
#[test]
fn test_cli_flags_without_api_key() {
    let exe = binary_path();
    let temp = std::env::temp_dir().join(format!("test-cli-nokey-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&temp);
    let _ = std::fs::create_dir_all(temp.join(".config"));

    let mut outputs = Vec::new();
    for flag in ["--version", "-V", "--help", "-h"] {
        let output = common::binary_command(&exe)
            .env("MINI_SWE_NO_DAEMON", "1")
            .current_dir(&temp)
            .arg(flag)
            .env_remove("OPENAI_API_KEY")
            .env("ENV_FILE", temp.join(".env.does-not-exist"))
            .env("XDG_CONFIG_HOME", temp.join(".config"))
            .env("HOME", &temp)
            .output()
            .unwrap_or_else(|e| panic!("failed to run {} {flag}: {e}", exe.display()));
        outputs.push((flag, output));
    }
    let _ = std::fs::remove_dir_all(&temp);

    for (flag, output) in outputs {
        assert!(
            output.status.success(),
            "flag {flag} must exit 0 without OPENAI_API_KEY, got {:?}; stderr: {}",
            output.status.code(),
            String::from_utf8_lossy(&output.stderr)
        );

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stdout.contains("mini-swe-mcp"),
            "output should identify the binary for {flag}: {stdout}"
        );
        assert!(
            !stdout.contains("Missing OPENAI_API_KEY")
                && !stderr.contains("Missing OPENAI_API_KEY"),
            "flag {flag} must not require OPENAI_API_KEY; stderr: {stderr}"
        );

        if flag == "--version" || flag == "-V" {
            assert!(
                stdout.contains(env!("CARGO_PKG_VERSION")),
                "version output missing package version for {flag}: {stdout}"
            );
        } else {
            assert!(
                stdout.contains("Usage: mini-swe-mcp"),
                "help output missing usage for {flag}: {stdout}"
            );
        }
    }
}

#[test]
fn test_cli_unknown_action() {
    let exe = binary_path();
    let output = common::binary_command(&exe)
        .env("MINI_SWE_NO_DAEMON", "1")
        .arg("nonexistent_action_xyz")
        .output()
        .unwrap_or_else(|e| panic!("failed to run {}: {e}", exe.display()));

    assert!(
        !output.status.success(),
        "unknown action should exit with non-zero code"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("Unknown action: nonexistent_action_xyz"),
        "stderr should mention unknown action: {stderr}"
    );
    assert!(
        stderr.contains("Available:"),
        "stderr should list available actions: {stderr}"
    );
    assert!(
        !stderr.contains("Did you mean"),
        "completely unknown action should not suggest anything: {stderr}"
    );
}

#[test]
fn test_cli_typo_suggestion() {
    let exe = binary_path();
    let output = common::binary_command(&exe)
        .env("MINI_SWE_NO_DAEMON", "1")
        .arg("statsu")
        .output()
        .unwrap_or_else(|e| panic!("failed to run {}: {e}", exe.display()));

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("Did you mean 'status'?"),
        "stderr should suggest 'status' for typo 'statsu': {stderr}"
    );
}

#[test]
fn test_cli_prefix_suggestion() {
    let exe = binary_path();
    let output = common::binary_command(&exe)
        .env("MINI_SWE_NO_DAEMON", "1")
        .arg("disp")
        .output()
        .unwrap_or_else(|e| panic!("failed to run {}: {e}", exe.display()));

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("Did you mean 'dispatch'?"),
        "stderr should suggest 'dispatch' for prefix 'disp': {stderr}"
    );
}

#[test]
fn test_cli_prune_action() {
    let exe = binary_path();
    let temp = std::env::temp_dir().join(format!("test-prune-cli-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&temp);
    let _ = Command::new("git")
        .args(["init"])
        .current_dir(&temp)
        .output();

    // 1. Plain text format (default)
    let output = common::binary_command(&exe)
        .env("MINI_SWE_NO_DAEMON", "1")
        .current_dir(&temp)
        .arg("prune")
        .output()
        .unwrap_or_else(|e| panic!("failed to run {}: {e}", exe.display()));

    assert!(output.status.success(), "prune plain text failed");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("✓ Stale worktrees and dead worker branches cleaned up"),
        "expected formatted plain text, got: {stdout}"
    );

    // 2. JSON format with --json flag
    let output_json = common::binary_command(&exe)
        .env("MINI_SWE_NO_DAEMON", "1")
        .current_dir(&temp)
        .args(["prune", "--json"])
        .output()
        .unwrap_or_else(|e| panic!("failed to run {}: {e}", exe.display()));

    let _ = std::fs::remove_dir_all(&temp);

    assert!(output_json.status.success(), "prune --json failed");
    let stdout_json = String::from_utf8_lossy(&output_json.stdout);
    let val: serde_json::Value =
        serde_json::from_str(stdout_json.trim()).expect("prune --json must return valid JSON");
    assert_eq!(val["status"], "pruned");
}

fn run_action(exe: &std::path::Path, args: &[&str]) -> std::process::Output {
    common::binary_command(exe)
        .args(args)
        .env("MINI_SWE_NO_DAEMON", "1")
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .env("OPENAI_API_KEY", "test-key-not-used-by-manifest-or-list")
        .env(
            "ENV_FILE",
            env!("CARGO_MANIFEST_DIR").to_owned() + "/.env.does-not-exist",
        )
        .env(
            "MODELS_FILE",
            env!("CARGO_MANIFEST_DIR").to_owned() + "/models.yaml",
        )
        .output()
        .unwrap_or_else(|e| panic!("failed to run {} {args:?}: {e}", exe.display()))
}

fn parse_json(action: &str, output: &std::process::Output) -> serde_json::Value {
    assert!(
        output.status.success(),
        "`{action}` must exit successfully, got {:?}\nstderr: {}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    serde_json::from_str(stdout.trim())
        .unwrap_or_else(|e| panic!("`{action}` must print valid JSON ({e}), got: {stdout}"))
}

#[test]
fn test_cli_manifest_plain_text() {
    let exe = binary_path();
    let output = run_action(&exe, &["manifest"]);
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        !stdout.trim().starts_with('{'),
        "manifest without --json must not be JSON: {stdout}"
    );
    assert!(stdout.contains("Default model: ninja"));
    assert!(stdout.contains("Models:"));
    assert!(stdout.contains("- ninja (id: combo:ninja"));
    assert!(stdout.contains("- nerd (id: combo:nerd"));
}

#[test]
fn test_cli_manifest_json() {
    let exe = binary_path();
    let output = run_action(&exe, &["manifest", "--json"]);
    let val = parse_json("manifest --json", &output);

    assert!(
        val.is_object(),
        "manifest output must be a JSON object, got: {val}"
    );

    let default_model = val
        .get("default_model")
        .unwrap_or_else(|| panic!("manifest output missing `default_model`: {val}"));
    assert_eq!(default_model.as_str(), Some("ninja"));

    let models = val
        .get("models")
        .unwrap_or_else(|| panic!("manifest output missing `models`: {val}"));
    let models = models
        .as_object()
        .unwrap_or_else(|| panic!("`models` must be a JSON object, got: {models}"));
    assert!(
        !models.is_empty(),
        "`models` must expose at least one model, got: {val}"
    );
}

#[test]
fn test_cli_list_plain_text() {
    let exe = binary_path();
    let output = run_action(&exe, &["list"]);
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("No active or recent workers found.") || stdout.contains("Workers ("),
        "expected empty list or worker listing, got: {stdout}"
    );
}

/// The per-worker health line: one compact summary of the counters a run
/// recorded, and nothing at all for a row written before they existed.
#[test]
fn test_cli_status_renders_the_health_line() {
    let exe = binary_path();
    let id = format!("{:08x}", std::process::id());
    let swe = std::env::temp_dir().join(format!("test-cli-health-{id}"));
    let registry = swe.join("swe-registry");
    std::fs::create_dir_all(&registry).expect("create the registry dir");
    // A completed run, with every counter moved.
    let measured_row = format!(
        r#"{{"id":"{id}","pid":1,"task":"t","model":"ninja","status":"completed","step":142,"max_turns":150,"last_command":"completed","started_at":1,"updated_at":2,"metrics":{{"turns_used":142,"extensions_granted":4,"extensions_refused":2,"repeat_blocks":3,"stagnation_nudges":1,"loop_pauses":1,"verify_runs":2,"verify_failures":1,"diff_files":5,"diff_insertions":120,"diff_deletions":340}}}}"#
    );
    std::fs::write(registry.join(format!("{id}.json")), measured_row)
        .expect("write the measured registry row");
    // A row from a build that recorded no counters at all.
    let legacy = format!("legacy{id}");
    let legacy_row = format!(
        r#"{{"id":"{legacy}","pid":1,"task":"t","model":"ninja","status":"completed","step":9,"max_turns":10,"last_command":"completed","started_at":1,"updated_at":2}}"#
    );
    std::fs::write(registry.join(format!("{legacy}.json")), legacy_row)
        .expect("write the legacy registry row");

    // The registry lives under `SWE_TEMP_DIR` and the hub must not share it:
    // orphaned registry rows with no worktree or branch are reaped on `list`,
    // so the hub gets its own scratch dir and `status` reads rows in-process.
    let status = |wid: &str| {
        let output = common::binary_command(&exe)
            .env("MINI_SWE_NO_DAEMON", "1")
            .args(["status", wid, "--admin"])
            .env("SWE_TEMP_DIR", &swe)
            .env("OPENAI_API_KEY", "test-key-not-used-by-status")
            .env(
                "ENV_FILE",
                env!("CARGO_MANIFEST_DIR").to_owned() + "/.env.does-not-exist",
            )
            .env(
                "MODELS_FILE",
                env!("CARGO_MANIFEST_DIR").to_owned() + "/models.yaml",
            )
            .output()
            .unwrap_or_else(|e| panic!("failed to run {}: {e}", exe.display()));
        assert!(output.status.success(), "`status {wid}` must succeed");
        String::from_utf8_lossy(&output.stdout).into_owned()
    };

    let measured = status(&id);
    assert!(
        measured.contains(
            "Health: 142 turns, +4/-2 ext, 3 repeats, 1 nudge, 1 loop pause, verify 1/2 failed, diff 5 files +120/-340"
        ),
        "status must render the health line, got:\n{measured}"
    );

    let unmeasured = status(&legacy);
    assert!(
        !unmeasured.contains("Health"),
        "a row without metrics must render no health line, got:\n{unmeasured}"
    );

    let _ = std::fs::remove_dir_all(&swe);
}

#[test]
fn test_cli_list_json() {
    let exe = binary_path();
    let output = run_action(&exe, &["list", "--json"]);
    let val = parse_json("list --json", &output);

    assert!(
        val.is_object(),
        "list output must be a JSON object, got: {val}"
    );

    let workers = val
        .get("workers")
        .unwrap_or_else(|| panic!("list output missing `workers`: {val}"));
    assert!(
        workers.is_array(),
        "`workers` must be a JSON array, got: {workers}"
    );
}

/// `watch <worker_id>` reaches the `worker` tool's `watch` verb: an unknown
/// worker therefore fails the way every other verb reports one, instead of
/// being swallowed by the argv mapper.
#[test]
fn test_cli_watch_on_an_unknown_worker_fails() {
    let exe = binary_path();
    let output = run_action(&exe, &["watch", "cli-wait-missing-xyz"]);
    assert!(
        !output.status.success(),
        "watching an unknown worker must exit non-zero"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("Worker not found: cli-wait-missing-xyz"),
        "stderr should name the missing worker: {stderr}"
    );
}

/// `status --line` is the Claude Code statusLine probe: one registry-only line
/// that never auto-starts the hub daemon.
#[test]
fn test_cli_status_line_never_autostarts_the_hub() {
    use std::time::{SystemTime, UNIX_EPOCH};
    let exe = binary_path();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock before epoch")
        .as_nanos();
    let root =
        std::env::temp_dir().join(format!("test-status-line-{}-{nanos}", std::process::id()));
    let hub = root.join("hub");
    let swe = root.join("swe");
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&hub).expect("create the scratch hub dir");
    std::fs::create_dir_all(swe.join("swe-registry")).expect("create the scratch registry");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&hub, std::fs::Permissions::from_mode(0o700))
            .expect("restrict the scratch hub dir to 0700");
        std::fs::set_permissions(&swe, std::fs::Permissions::from_mode(0o700))
            .expect("restrict the scratch swe dir to 0700");
    }
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock before epoch")
        .as_secs();
    // One live worker and one recent completion; the line names both buckets.
    let registry = |id: &str, status: &str, updated_at: u64| {
        format!(
            r#"{{"id":"{id}","pid":{},"task":"t","model":"ninja","status":"{status}","step":1,"max_turns":10,"last_command":"done","started_at":1,"updated_at":{updated_at}}}"#,
            std::process::id()
        )
    };
    std::fs::write(
        swe.join("swe-registry").join("line-live.json"),
        registry("line-live", "running", now),
    )
    .expect("write the live row");
    std::fs::write(
        swe.join("swe-registry").join("line-done.json"),
        registry("line-done", "completed", now),
    )
    .expect("write the done row");

    let run = |args: &[&str]| {
        common::binary_command(&exe)
            .args(args)
            .env("SWE_HUB_DIR", &hub)
            .env("SWE_TEMP_DIR", &swe)
            .env("TMPDIR", &swe)
            .env_remove("MINI_SWE_NO_DAEMON")
            .env_remove("OPENAI_API_KEY")
            .env(
                "ENV_FILE",
                env!("CARGO_MANIFEST_DIR").to_owned() + "/.env.does-not-exist",
            )
            .env(
                "MODELS_FILE",
                env!("CARGO_MANIFEST_DIR").to_owned() + "/models.yaml",
            )
            .output()
            .unwrap_or_else(|e| panic!("failed to run {} {args:?}: {e}", exe.display()))
    };

    let output = run(&["status", "--line"]);
    assert!(output.status.success(), "status --line must exit 0");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(
        stdout.trim(),
        "⚙ 1 running · 1 done",
        "unexpected status line: {stdout:?}"
    );
    assert!(
        !mini_swe_mcp::hub::HubPaths::new(hub.to_path_buf())
            .socket()
            .exists(),
        "status --line must never auto-start the daemon"
    );

    // Flag order must not matter, and an empty registry prints nothing.
    let output = run(&["--json", "status", "--line"]);
    assert!(output.status.success(), "flag order must not matter");
    let _ = std::fs::remove_dir_all(swe.join("swe-registry"));
    let output = run(&["status", "--line"]);
    assert!(output.status.success(), "empty status --line must exit 0");
    assert!(
        String::from_utf8_lossy(&output.stdout).trim().is_empty(),
        "an empty registry prints no line"
    );
    assert!(
        !mini_swe_mcp::hub::HubPaths::new(hub.to_path_buf())
            .socket()
            .exists(),
        "still no daemon must exist"
    );
    let _ = std::fs::remove_dir_all(&root);
}
