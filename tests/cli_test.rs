//! Integration tests for the `mini-swe-mcp` CLI executable.

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
        let output = Command::new(&exe)
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
        let output = Command::new(&exe)
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
    }
}

#[test]
fn test_cli_unknown_action() {
    let exe = binary_path();
    let output = Command::new(&exe)
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
}

#[test]
fn test_cli_prune_action() {
    let exe = binary_path();
    let temp = std::env::temp_dir().join(format!("test-prune-cli-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&temp);
    let _ = Command::new("git").args(["init"]).current_dir(&temp).output();

    let output = Command::new(&exe)
        .current_dir(&temp)
        .arg("prune")
        .output()
        .unwrap_or_else(|e| panic!("failed to run {}: {e}", exe.display()));

    let _ = std::fs::remove_dir_all(&temp);

    assert!(output.status.success(), "prune action failed");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let val: serde_json::Value =
        serde_json::from_str(stdout.trim()).expect("prune must return valid JSON");
    assert_eq!(val["status"], "ok");
}

fn run_action(exe: &std::path::Path, action: &str) -> std::process::Output {
    Command::new(exe)
        .arg(action)
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .env("OPENAI_API_KEY", "test-key-not-used-by-manifest-or-list")
        .env("ENV_FILE", env!("CARGO_MANIFEST_DIR").to_owned() + "/.env.does-not-exist")
        .env("MODELS_FILE", env!("CARGO_MANIFEST_DIR").to_owned() + "/models.yaml")
        .output()
        .unwrap_or_else(|e| panic!("failed to run {} {action}: {e}", exe.display()))
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
fn test_cli_manifest() {
    let exe = binary_path();
    let output = run_action(&exe, "manifest");
    let val = parse_json("manifest", &output);

    assert!(val.is_object(), "manifest output must be a JSON object, got: {val}");

    let default_model = val
        .get("default_model")
        .unwrap_or_else(|| panic!("manifest output missing `default_model`: {val}"));
    assert!(
        default_model.as_str().is_some_and(|m| !m.is_empty()),
        "`default_model` must be a non-empty string, got: {default_model}"
    );

    let models = val
        .get("models")
        .unwrap_or_else(|| panic!("manifest output missing `models`: {val}"));
    let models = models
        .as_object()
        .unwrap_or_else(|| panic!("`models` must be a JSON object, got: {models}"));
    assert!(!models.is_empty(), "`models` must expose at least one model, got: {val}");
}

#[test]
fn test_cli_list() {
    let exe = binary_path();
    let output = run_action(&exe, "list");
    let val = parse_json("list", &output);

    assert!(val.is_object(), "list output must be a JSON object, got: {val}");

    let workers = val
        .get("workers")
        .unwrap_or_else(|| panic!("list output missing `workers`: {val}"));
    assert!(workers.is_array(), "`workers` must be a JSON array, got: {workers}");
}
