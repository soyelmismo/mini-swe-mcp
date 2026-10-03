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
    assert!(!quiet_requested(&args(&[
        "mini-swe-mcp",
        "dispatch",
        "task"
    ])));
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

/// `discard` targets one worker like `kill`, so the positional maps to the same
/// `worker_id` argument the handler reads.
#[test]
fn test_tool_args_maps_the_discard_target() {
    let a = args(&["mini-swe-mcp", "discard", "w1"]);
    let out = tool_args("discard", &a, true).unwrap().unwrap();
    assert_eq!(out["action"], "discard");
    assert_eq!(out["worker_id"], "w1");
}

/// An unquoted `steer <id> sal del loop!` arrives as several argv words, and
/// every one of them is the message: reading only the first word would drop
/// "del loop!" without a word of complaint, which is the bug this guards.
#[test]
fn test_steer_joins_every_unquoted_message_word() {
    let argv = args(&["mini-swe-mcp", "steer", "e32dcc08", "sal", "del", "loop!"]);
    let out = tool_args("steer", &argv, true).unwrap().unwrap();
    assert_eq!(out["worker_id"], "e32dcc08");
    assert_eq!(
        out["message"], "sal del loop!",
        "no message word may be dropped: {out:?}"
    );
}

/// A quoted message is one argv word already, so the join leaves it untouched,
/// and the flags beside it keep parsing.
#[test]
fn test_steer_keeps_a_quoted_message_and_its_flags_unchanged() {
    let argv = args(&[
        "mini-swe-mcp",
        "steer",
        "w1",
        "fix the edge case",
        "--max-turns",
        "30",
    ]);
    let out = tool_args("steer", &argv, true).unwrap().unwrap();
    assert_eq!(out["message"], "fix the edge case");
    assert_eq!(out["max_turns"], 30);
}

/// A flag may sit between words of the message: it and its value are consumed
/// by the flag parser, never folded into the text the worker reads.
#[test]
fn test_steer_parses_flags_interleaved_after_words() {
    let argv = args(&[
        "mini-swe-mcp",
        "steer",
        "w1",
        "please",
        "--max-turns",
        "7",
        "restart the loop",
    ]);
    let out = tool_args("steer", &argv, true).unwrap().unwrap();
    assert_eq!(out["max_turns"], 7);
    assert_eq!(
        out["message"], "please restart the loop",
        "flag and value stay out of the message: {out:?}"
    );
}

/// `dispatch <task>` is the same pattern: an unquoted multi-word task arrives
/// whole, and a value flag's word is not mistaken for task text.
#[test]
fn test_dispatch_joins_every_unquoted_task_word() {
    let argv = args(&[
        "mini-swe-mcp",
        "dispatch",
        "fix",
        "the",
        "flaky",
        "parse",
        "--model",
        "m1",
        "test",
    ]);
    let out = tool_args("dispatch", &argv, true).unwrap().unwrap();
    assert_eq!(
        out["task"], "fix the flaky parse test",
        "no task word may be dropped: {out:?}"
    );
    assert_eq!(out["model"], "m1");
}

/// `-f` reads every task from the file, so a stray word beside it has nowhere
/// to go; refuse it clearly rather than dispatch the file and drop the word.
#[test]
fn test_dispatch_batch_refuses_an_extra_word_beside_the_file() {
    let dir = std::env::temp_dir().join(format!("mini-swe-batch-extra-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let path = dir.join("tasks.yaml");
    std::fs::write(&path, "- task: only\n").expect("write batch file");

    let argv = args(&[
        "mini-swe-mcp",
        "dispatch",
        "-f",
        path.to_str().expect("utf8"),
        "and",
        "also",
        "this",
    ]);
    let error = tool_args("dispatch", &argv, true)
        .expect_err("a bare word beside -f must be refused, not dropped");
    let text = error.to_string();
    assert!(text.contains("-f"), "the refusal must name -f: {error}");

    // The documented `dispatch -f <file> --flag value` form stays silent.
    let ok = args(&[
        "mini-swe-mcp",
        "dispatch",
        "-f",
        path.to_str().expect("utf8"),
        "--group",
        "g1",
    ]);
    let out = tool_args("dispatch", &ok, true).unwrap().unwrap();
    assert_eq!(out["group"], "g1");
    assert!(out["tasks"].is_array(), "{out:?}");

    let _ = std::fs::remove_dir_all(&dir);
}
