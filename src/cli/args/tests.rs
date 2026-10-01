use super::*;

pub(super) fn args(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| s.to_string()).collect()
}

#[test]
fn test_json_flag_is_detected_and_stripped() {
    let raw = args(&["mini-swe-mcp", "--json", "list"]);
    assert!(json_requested(&raw));
    let stripped = strip_json_flag(raw);
    assert_eq!(stripped, args(&["mini-swe-mcp", "list"]));
    assert!(!json_requested(&stripped));
}

/// `--quiet` is a rendering selector, not a tool argument: it is detected,
/// stripped before the positional parse, and never reaches the `worker` tool.
#[test]
fn test_quiet_flag_is_detected_and_stripped() {
    for flag in ["--quiet", "-q"] {
        let raw = args(&["mini-swe-mcp", "dispatch", flag, "task"]);
        assert!(quiet_requested(&raw), "{flag} must be detected");
        let stripped = strip_quiet_flag(raw);
        assert_eq!(stripped, args(&["mini-swe-mcp", "dispatch", "task"]));
        assert!(!quiet_requested(&stripped));

        let tool = tool_args("dispatch", &stripped, true).unwrap().unwrap();
        assert_eq!(tool["task"], "task");
        assert!(!tool.contains_key("quiet"), "{tool:?}");
    }
    assert!(!quiet_requested(&args(&["mini-swe-mcp", "dispatch", "task"])));
}

#[test]
fn test_stdio_and_action_detection() {
    assert!(stdio_requested(&args(&["mini-swe-mcp", "--stdio"])));
    assert!(!stdio_requested(&args(&["mini-swe-mcp", "list"])));
    assert_eq!(action_of(&args(&["mini-swe-mcp", "list"])), Some("list"));
    assert_eq!(action_of(&args(&["mini-swe-mcp", "--stdio"])), None);
    assert_eq!(action_of(&args(&["mini-swe-mcp"])), None);
}

/// `--admin` is a connection flag, not a positional: it must be stripped
/// before `status <id>` reads its worker id.
#[test]
fn test_admin_flag_is_detected_and_stripped() {
    let raw = args(&["mini-swe-mcp", "--admin", "status", "w1"]);
    assert!(admin_requested(&raw));
    let stripped = strip_admin_flag(raw);
    assert_eq!(stripped, args(&["mini-swe-mcp", "status", "w1"]));
    assert!(!admin_requested(&stripped));

    let plain = args(&["mini-swe-mcp", "status", "w1"]);
    assert!(!admin_requested(&plain));
}

#[test]
fn test_tool_args_maps_positional_arguments_for_simple_verbs() {
    let a = args(&["mini-swe-mcp", "status", "w1"]);
    let out = tool_args("status", &a, true).unwrap().unwrap();
    assert_eq!(out["action"], "status");
    assert_eq!(out["worker_id"], "w1");

    let s = args(&["mini-swe-mcp", "steer", "w1", "stop"]);
    let out = tool_args("steer", &s, true).unwrap().unwrap();
    assert_eq!(out["worker_id"], "w1");
    assert_eq!(out["message"], "stop");

    // A verb with no positional needs no extra arguments.
    let r = tool_args("reap", &args(&["mini-swe-mcp", "reap"]), true)
        .unwrap()
        .unwrap();
    assert_eq!(r.len(), 1);
}
