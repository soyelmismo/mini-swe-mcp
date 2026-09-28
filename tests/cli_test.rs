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
