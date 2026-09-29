//! argv → `worker` tool argument mapping for the CLI-only verbs.
//!
//! The binary never re-implements tool semantics: it only turns positional argv
//! into the same JSON object the MCP `tools/call` path would have sent, so both
//! callers share one validation and rendering implementation.

use anyhow::Result;
use serde_json::{Map, Value};

/// Dispatch usage line, shared by `--help` and the missing-task error.
pub const DISPATCH_USAGE: &str = "dispatch <task> [--model <model>] [--review-after <model>] [--repo <repo>] [--wait] [--max-turns <n>] [--group <group>] [--offline]";

/// Build the `worker` tool arguments for `action` from `cli_args` (argv minus
/// the program name and the `--json` flag).
///
/// Returns `Ok(None)` for verbs the binary handles itself (`monitor` /
/// `supervisor`) — the caller is expected to have dispatched those earlier — and
/// `Ok(Some(args))` for every verb that goes through the `worker` tool. An
/// unrecognised action is a hard error, so the caller can exit non-zero after
/// printing the "did you mean" hint.
pub fn tool_args(action: &str, cli_args: &[String], api_key_present: bool) -> Result<Option<Map<String, Value>>> {
    let mut tool_args = Map::new();
    tool_args.insert("action".into(), Value::String(action.to_string()));

    match action {
        "dispatch" => {
            if !api_key_present {
                anyhow::bail!("Missing OPENAI_API_KEY. Please provide it via environment variable or .env file.");
            }
            if cli_args.len() < 3 {
                eprintln!("Usage: mini-swe-mcp {DISPATCH_USAGE}");
                return Ok(None);
            }
            dispatch_args(cli_args, &mut tool_args);
        }
        "status" | "collect" | "logs" | "kill" => {
            if cli_args.len() > 2 {
                tool_args.insert("worker_id".into(), Value::String(cli_args[2].clone()));
            }
        }
        "steer" => {
            if cli_args.len() > 3 {
                tool_args.insert("worker_id".into(), Value::String(cli_args[2].clone()));
                tool_args.insert("message".into(), Value::String(cli_args[3].clone()));
            }
        }
        "monitor" | "supervisor" => return Ok(None),
        "manifest" | "list" | "reap" => {}
        _ => {
            let actions = crate::cli::available_actions();
            if let Some(suggestion) = crate::cli::suggest_action(action, &actions) {
                eprintln!(
                    "Unknown action: {action}. Did you mean '{suggestion}'?\nAvailable: {}",
                    actions.join(", ")
                );
            } else {
                eprintln!(
                    "Unknown action: {action}. Available: {}",
                    actions.join(", ")
                );
            }
            std::process::exit(1);
        }
    }

    Ok(Some(tool_args))
}

/// Fold the `dispatch` flags after the task into the tool arguments.
///
/// A value flag consumes the next word verbatim; one with nothing after it is
/// dropped rather than defaulted to an empty string.
fn dispatch_args(cli_args: &[String], tool_args: &mut Map<String, Value>) {
    tool_args.insert("task".into(), Value::String(cli_args[2].clone()));
    let mut i = 3;
    while i < cli_args.len() {
        match cli_args[i].as_str() {
            "--model" | "-m" => take_value(cli_args, &mut i, tool_args, "model"),
            "--review-after" => take_value(cli_args, &mut i, tool_args, "review_after"),
            "--repo" | "-r" => take_value(cli_args, &mut i, tool_args, "repo_path"),
            "--wait" | "-w" => {
                tool_args.insert("wait".into(), Value::Bool(true));
            }
            "--max-turns" | "-t" => {
                if i + 1 < cli_args.len() {
                    if let Ok(turns) = cli_args[i + 1].parse::<u64>() {
                        tool_args.insert("max_turns".into(), Value::Number(turns.into()));
                    }
                    i += 1;
                }
            }
            "--group" | "-g" if i + 1 < cli_args.len() => {
                tool_args.insert("group".into(), Value::String(cli_args[i + 1].clone()));
                i += 1;
            }
            // `--offline` is the CLI spelling of `network: "offline"`: the same
            // tool argument, so the policy has exactly one implementation.
            "--offline" => {
                tool_args.insert("network".into(), Value::String("offline".into()));
            }
            _ => {}
        }
        i += 1;
    }
}

/// Consume the value after a flag, if present.
fn take_value(cli_args: &[String], i: &mut usize, tool_args: &mut Map<String, Value>, key: &str) {
    if *i + 1 < cli_args.len() {
        tool_args.insert(key.into(), Value::String(cli_args[*i + 1].clone()));
        *i += 1;
    }
}

/// True when the operator asked for JSON instead of the plain-text views.
pub fn json_requested(raw_args: &[String]) -> bool {
    raw_args.iter().any(|arg| arg == "--json")
}

/// argv with the `--json` selector removed so positional parsing never trips
/// over the flag.
pub fn strip_json_flag(raw_args: Vec<String>) -> Vec<String> {
    raw_args.into_iter().filter(|arg| arg != "--json").collect()
}

/// True when the binary should serve MCP over stdio rather than run an action.
pub fn stdio_requested(cli_args: &[String]) -> bool {
    cli_args.iter().any(|arg| arg == "--stdio")
}

/// The action word selected on the command line, if any.
///
/// `None` means the binary was asked for nothing but a stdio server (`--stdio`
/// or no arguments at all).
pub fn action_of(cli_args: &[String]) -> Option<&str> {
    match cli_args {
        [_, action, ..] if action != "--stdio" => Some(action),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(v: &[&str]) -> Vec<String> {
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

    #[test]
    fn test_stdio_and_action_detection() {
        assert!(stdio_requested(&args(&["mini-swe-mcp", "--stdio"])));
        assert!(!stdio_requested(&args(&["mini-swe-mcp", "list"])));
        assert_eq!(action_of(&args(&["mini-swe-mcp", "list"])), Some("list"));
        assert_eq!(action_of(&args(&["mini-swe-mcp", "--stdio"])), None);
        assert_eq!(action_of(&args(&["mini-swe-mcp"])), None);
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

    #[test]
    fn test_tool_args_requires_an_api_key_only_for_dispatch() {
        let d = args(&["mini-swe-mcp", "dispatch", "task"]);
        assert!(tool_args("dispatch", &d, false).is_err());
        assert!(tool_args("dispatch", &d, true).unwrap().is_some());
        // Non-dispatch verbs never need the key.
        assert!(
            tool_args("list", &args(&["mini-swe-mcp", "list"]), false)
                .unwrap()
                .is_some()
        );
    }

    /// `--offline` is only a spelling of the tool's `network` property, so the
    /// CLI must produce exactly the argument the MCP path would send.
    #[test]
    fn test_dispatch_offline_flag_maps_to_the_network_property() {
        let without = tool_args(
            "dispatch",
            &args(&["mini-swe-mcp", "dispatch", "tidy docs"]),
            true,
        )
        .unwrap()
        .unwrap();
        assert!(
            !without.contains_key("network"),
            "an omitted flag must leave the default policy in place"
        );

        let with = tool_args(
            "dispatch",
            &args(&["mini-swe-mcp", "dispatch", "tidy docs", "--offline"]),
            true,
        )
        .unwrap()
        .unwrap();
        assert_eq!(with["network"], "offline");
        assert!(crate::mcp::NETWORK_MODES.contains(&with["network"].as_str().unwrap()));
    }

    #[test]
    fn test_tool_args_returns_none_for_a_taskless_dispatch() {
        let d = args(&["mini-swe-mcp", "dispatch"]);
        assert!(tool_args("dispatch", &d, true).unwrap().is_none());
    }

    #[test]
    fn test_tool_args_defers_the_cli_only_verbs() {
        for verb in ["monitor", "supervisor"] {
            assert!(
                tool_args(verb, &args(&["mini-swe-mcp", verb, "--once"]), true)
                    .unwrap()
                    .is_none(),
                "{verb} must be handled by the binary, not the worker tool"
            );
        }
    }

    #[test]
    fn test_dispatch_args_maps_every_flag_including_short_forms() {
        let cli_args = args(&[
            "mini-swe-mcp", "dispatch", "fix it",
            "--model", "m1", "--review-after", "m2", "--repo", "/tmp/r",
            "--wait", "--max-turns", "12", "--group", "g1",
            "-m", "m3", "-r", "/tmp/r2", "-g", "g2", "-t", "3", "-w",
        ]);
        let mut tool_args = Map::new();
        dispatch_args(&cli_args, &mut tool_args);

        assert_eq!(tool_args["task"], "fix it");
        assert_eq!(tool_args["model"], "m3"); // the last flag wins
        assert_eq!(tool_args["review_after"], "m2");
        assert_eq!(tool_args["repo_path"], "/tmp/r2");
        assert_eq!(tool_args["wait"], Value::Bool(true));
        assert_eq!(tool_args["max_turns"], 3);
        assert_eq!(tool_args["group"], "g2");
    }

    #[test]
    fn test_dispatch_args_ignores_unknown_flags_and_unparsable_turns() {
        let cli_args = args(&["mini-swe-mcp", "dispatch", "t", "--nope", "x", "-t", "nan"]);
        let mut tool_args = Map::new();
        dispatch_args(&cli_args, &mut tool_args);

        assert_eq!(tool_args["task"], "t");
        assert_eq!(tool_args.len(), 1, "unknown flags add nothing: {tool_args:?}");
    }

    #[test]
    fn test_dispatch_args_takes_the_next_word_verbatim_as_a_flag_value() {
        // Historical behaviour: a value flag consumes the next word whatever it
        // is, so `dispatch t --model --repo` sets model="--repo" and nothing else.
        let cli_args = args(&["mini-swe-mcp", "dispatch", "t", "--model", "--repo"]);
        let mut tool_args = Map::new();
        dispatch_args(&cli_args, &mut tool_args);

        assert_eq!(tool_args["model"], "--repo");
        assert!(!tool_args.contains_key("repo_path"), "{tool_args:?}");

        // A value flag with nothing after it is dropped, not defaulted.
        let trailing = args(&["mini-swe-mcp", "dispatch", "t", "--group"]);
        let mut tool_args = Map::new();
        dispatch_args(&trailing, &mut tool_args);
        assert_eq!(tool_args.len(), 1, "{tool_args:?}");
    }
}
