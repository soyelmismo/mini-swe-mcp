//! argv → `worker` tool argument mapping for the CLI-only verbs.
//!
//! The binary never re-implements tool semantics: it only turns positional argv
//! into the same JSON object the MCP `tools/call` path would have sent, so both
//! callers share one validation and rendering implementation.

use anyhow::Result;
use serde_json::{Map, Value};

/// Dispatch usage line, shared by `--help` and the missing-task error.
pub const DISPATCH_USAGE: &str = "dispatch <task> | dispatch -f <tasks.yaml> [--model <model>] [--review-after <model>] [--repo <repo>] [--max-turns <n>] [--group <group>] [--role <role>] [--offline] [--verify <cmd>] (task: ONE focused concern, scoped files, acceptance gate; -f runs a YAML/JSON list, '-' reads stdin)";

/// Build the `worker` tool arguments for `action` from `cli_args` (argv minus
/// the program name and the `--json` flag).
///
/// Returns `Ok(None)` when the verb was already answered here (a `dispatch`
/// without a task prints its usage) and `Ok(Some(args))` for every verb that
/// goes through the `worker` tool. An
/// unrecognised action is a hard error, so the caller can exit non-zero after
/// printing the "did you mean" hint.
pub fn tool_args(
    action: &str,
    cli_args: &[String],
    api_key_present: bool,
) -> Result<Option<Map<String, Value>>> {
    let mut tool_args = Map::new();
    tool_args.insert("action".into(), Value::String(action.to_string()));

    match action {
        "dispatch" => {
            if !api_key_present {
                anyhow::bail!(
                    "Missing OPENAI_API_KEY. Please provide it via environment variable or .env file."
                );
            }
            match batch_tasks(cli_args)? {
                Some((file_index, tasks)) => {
                    tool_args.insert("tasks".into(), tasks);
                    // Flags after the file are shared defaults for every entry;
                    // the file flag and its path are skipped, never parsed.
                    collect_dispatch_flags(
                        cli_args,
                        3,
                        &[file_index, file_index + 1],
                        &mut tool_args,
                    );
                }
                None => {
                    if cli_args.len() < 3 {
                        eprintln!("Usage: mini-swe-mcp {DISPATCH_USAGE}");
                        return Ok(None);
                    }
                    dispatch_args(cli_args, &mut tool_args)?;
                }
            }
        }
        "collect" | "kill" | "logs" | "review" | "status" | "merge" => {
            if action == "merge" && flag_index(cli_args, &["--approved"]).is_some() {
                tool_args.insert("approved".into(), Value::Bool(true));
                if let Some(i) = flag_index(cli_args, &["--group", "-g"]) {
                    let group = cli_args.get(i + 1).ok_or_else(|| anyhow::anyhow!("--group needs a value"))?;
                    tool_args.insert("group".into(), Value::String(group.clone()));
                }
            } else if cli_args.len() > 2 {
                tool_args.insert("worker_id".into(), Value::String(cli_args[2].clone()));
            }
            if action == "collect" {
                collect_diff_args(cli_args, &mut tool_args);
            }
            // `--no-delete` keeps the merged branch: the same tool argument
            // the MCP action reads, so the flag has one implementation.
            if flag_index(cli_args, &["--no-delete"]).is_some() {
                tool_args.insert("keep_branch".into(), Value::Bool(true));
            }
        }
        "steer" => {
            if cli_args.len() > 3 {
                tool_args.insert("worker_id".into(), Value::String(cli_args[2].clone()));
                tool_args.insert("message".into(), Value::String(cli_args[3].clone()));
                // `--max-turns <n>` on a steer is the fresh turn budget of a
                // revision (steering a finished worker); the same tool
                // argument `dispatch` uses, so the budget has one spelling.
                if let Some(mut i) = flag_index(cli_args, &["--max-turns", "-t"]) {
                    take_turns(cli_args, &mut i, &mut tool_args);
                }
            }
        }
        "list" => {
            // `--all` is the operator's view of the whole pool; without it the
            // CLI sees only the workers the `cli` identity owns.
            if flag_index(cli_args, &["--all"]).is_some() {
                tool_args.insert("scope".into(), Value::String("all".into()));
            }
        }
        "manifest" | "reap" | "prune" => {}
        _ => {
            let actions = crate::cli::available_actions();
            let msg = if let Some(suggestion) = crate::cli::suggest_action(action, &actions) {
                format!(
                    "Unknown action: {action}. Did you mean '{suggestion}'?\nAvailable: {}",
                    actions.join(", ")
                )
            } else {
                format!(
                    "Unknown action: {action}. Available: {}",
                    actions.join(", ")
                )
            };
            anyhow::bail!("{msg}");
        }
    }

    Ok(Some(tool_args))
}

/// Fold the `collect` diff selectors into the tool arguments.
///
/// `--full` asks for the whole diff and `--file <path>` (repeatable) narrows it
/// to the named files: the same tool arguments the MCP path sends, so the diff
/// scope has exactly one implementation.
fn collect_diff_args(cli_args: &[String], tool_args: &mut Map<String, Value>) {
    if cli_args.iter().any(|arg| arg == "--full") {
        tool_args.insert("full".into(), Value::Bool(true));
    }
    let mut files = Vec::new();
    let mut i = 0;
    while i < cli_args.len() {
        if cli_args[i] == "--file" && i + 1 < cli_args.len() {
            files.push(Value::String(cli_args[i + 1].clone()));
            i += 1;
        }
        i += 1;
    }
    if !files.is_empty() {
        tool_args.insert("files".into(), Value::Array(files));
    }
}

/// Fold the `dispatch` flags after the task into the tool arguments.
///
/// A value flag consumes the next word verbatim; one with nothing after it is
/// dropped rather than defaulted to an empty string.
fn dispatch_args(cli_args: &[String], tool_args: &mut Map<String, Value>) -> Result<()> {
    tool_args.insert("task".into(), Value::String(cli_args[2].clone()));
    collect_dispatch_flags(cli_args, 3, &[], tool_args);
    Ok(())
}

/// Fold the `dispatch` flags from `start` on into the tool arguments.
///
/// The flag table lives here once, so the single-task and batch forms cannot
/// drift apart. Positions in `skip` are left alone: batch dispatch uses it for
/// the `-f <file>` flag and its path, so neither is parsed as a flag or a task.
fn collect_dispatch_flags(
    cli_args: &[String],
    start: usize,
    skip: &[usize],
    tool_args: &mut Map<String, Value>,
) {
    let mut i = start;
    while i < cli_args.len() {
        if skip.contains(&i) {
            i += 1;
            continue;
        }
        match cli_args[i].as_str() {
            "--model" | "-m" => take_value(cli_args, &mut i, tool_args, "model"),
            "--review-after" => take_value(cli_args, &mut i, tool_args, "review_after"),
            "--repo" | "-r" => take_value(cli_args, &mut i, tool_args, "repo_path"),
            "--max-turns" | "-t" => take_turns(cli_args, &mut i, tool_args),
            "--group" | "-g" if i + 1 < cli_args.len() => {
                tool_args.insert("group".into(), Value::String(cli_args[i + 1].clone()));
                i += 1;
            }
            // `--offline` is the CLI spelling of `network: "offline"`: the same
            // tool argument, so the policy has exactly one implementation.
            "--offline" => {
                tool_args.insert("network".into(), Value::String("offline".into()));
            }
            "--verify" => take_value(cli_args, &mut i, tool_args, "verify"),
            // `--role <role>` selects the dispatch authority: the default
            // worker, or the round's consolidator.
            "--role" => take_value(cli_args, &mut i, tool_args, "role"),
            _ => {}
        }
        i += 1;
    }
}

/// Read the batch list named by `-f`/`--file`, returning the flag's position
/// and the parsed `tasks` array. `Ok(None)` means the flag was not passed.
///
/// `-` reads standard input, so `mini-swe-mcp dispatch -f - < tasks.yaml` works.
fn batch_tasks(cli_args: &[String]) -> Result<Option<(usize, Value)>> {
    let Some(index) = flag_index(cli_args, &["-f", "--file"]) else {
        return Ok(None);
    };
    let path = cli_args.get(index + 1).ok_or_else(|| {
        anyhow::anyhow!("-f/--file needs a path (use '-' to read the list from stdin)")
    })?;
    let text = if path == "-" {
        let mut text = String::new();
        std::io::Read::read_to_string(&mut std::io::stdin(), &mut text)
            .map_err(|e| anyhow::anyhow!("could not read the task list from stdin: {e}"))?;
        text
    } else {
        std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("could not read the task list {path}: {e}"))?
    };
    Ok(Some((index, parse_batch_tasks(&text)?)))
}

/// Parse a YAML or JSON task list into the `tasks` array of the `worker` tool.
///
/// YAML is a superset of JSON, so one parser reads both spellings.
pub fn parse_batch_tasks(text: &str) -> Result<Value> {
    let value: Value = serde_yaml::from_str(text)
        .map_err(|e| anyhow::anyhow!("task list is not valid YAML or JSON: {e}"))?;
    let Value::Array(tasks) = value else {
        anyhow::bail!("task list must be a YAML or JSON list of task objects");
    };
    if tasks.is_empty() {
        anyhow::bail!("task list must contain at least one task");
    }
    Ok(Value::Array(tasks))
}

/// Position of the first of `flags` in `cli_args`, if the operator passed one.
fn flag_index(cli_args: &[String], flags: &[&str]) -> Option<usize> {
    cli_args
        .iter()
        .position(|arg| flags.contains(&arg.as_str()))
}

/// Fold `--max-turns <n>` into the tool's `max_turns` argument.
///
/// Shared by `dispatch` and `steer`: the same budget argument, so a revision
/// started by steering a finished worker is spelled exactly like the dispatch
/// that preceded it. A malformed value is dropped rather than defaulted, which
/// is what the dispatch path already does.
fn take_turns(cli_args: &[String], i: &mut usize, tool_args: &mut Map<String, Value>) {
    if *i + 1 < cli_args.len()
        && let Ok(turns) = cli_args[*i + 1].parse::<u64>()
    {
        tool_args.insert("max_turns".into(), Value::Number(turns.into()));
        *i += 1;
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

/// True when the operator passed `--admin`.
///
/// The flag is the human operator's override: it is sent as `admin: true` in
/// the hub handshake, which lifts the per-agent ownership check for that one
/// connection (H-3).
pub fn admin_requested(raw_args: &[String]) -> bool {
    raw_args.iter().any(|arg| arg == "--admin")
}

/// argv with the `--admin` selector removed, for the same reason as
/// [`strip_json_flag`]: it is a flag, never a positional argument.
pub fn strip_admin_flag(raw_args: Vec<String>) -> Vec<String> {
    raw_args
        .into_iter()
        .filter(|arg| arg != "--admin")
        .collect()
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
    use serde_json::json;

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

    /// `list --all` is the CLI spelling of the tool's `scope` property.
    #[test]
    fn test_list_all_flag_maps_to_the_scope_property() {
        let scoped = tool_args("list", &args(&["mini-swe-mcp", "list", "--all"]), true)
            .unwrap()
            .unwrap();
        assert_eq!(scoped["action"], "list");
        assert_eq!(scoped["scope"], "all");

        let mine = tool_args("list", &args(&["mini-swe-mcp", "list"]), true)
            .unwrap()
            .unwrap();
        assert!(!mine.contains_key("scope"), "{mine:?}");
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

    /// `review <id>` is a read like `status`: one positional, no flags.
    #[test]
    fn test_review_maps_its_positional_worker_id() {
        let out = tool_args("review", &args(&["mini-swe-mcp", "review", "w1"]), true)
            .unwrap()
            .unwrap();
        assert_eq!(out["action"], "review");
        assert_eq!(out["worker_id"], "w1");
    }

    /// `--full` and `--file` are the CLI spelling of the `collect` tool
    /// arguments, so the diff scope has exactly one implementation.
    #[test]
    fn test_collect_flags_map_to_the_diff_scope_arguments() {
        let plain = tool_args("collect", &args(&["mini-swe-mcp", "collect", "w1"]), true)
            .unwrap()
            .unwrap();
        assert!(!plain.contains_key("full"), "{plain:?}");
        assert!(!plain.contains_key("files"), "{plain:?}");

        let full = tool_args(
            "collect",
            &args(&["mini-swe-mcp", "collect", "w1", "--full"]),
            true,
        )
        .unwrap()
        .unwrap();
        assert_eq!(full["full"], true);

        let files = tool_args(
            "collect",
            &args(&[
                "mini-swe-mcp",
                "collect",
                "w1",
                "--file",
                "src/a.rs",
                "--file",
                "b.rs",
            ]),
            true,
        )
        .unwrap()
        .unwrap();
        assert_eq!(files["files"], json!(["src/a.rs", "b.rs"]));
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

    /// `--verify <cmd>` is the CLI spelling of the tool's `verify` property.
    #[test]
    fn test_dispatch_verify_flag_maps_to_the_verify_property() {
        let without = tool_args(
            "dispatch",
            &args(&["mini-swe-mcp", "dispatch", "tidy docs"]),
            true,
        )
        .unwrap()
        .unwrap();
        assert!(
            !without.contains_key("verify"),
            "an omitted flag must leave auto-detection in place"
        );

        let with = tool_args(
            "dispatch",
            &args(&[
                "mini-swe-mcp",
                "dispatch",
                "tidy docs",
                "--verify",
                "cargo test --all-targets",
            ]),
            true,
        )
        .unwrap()
        .unwrap();
        assert_eq!(with["verify"], "cargo test --all-targets");
    }

    /// `--role <role>` is the CLI spelling of the tool's `role` property, and an
    /// omitted flag leaves the ordinary worker default in place.
    #[test]
    fn test_dispatch_role_flag_maps_to_the_role_property() {
        let without = tool_args(
            "dispatch",
            &args(&["mini-swe-mcp", "dispatch", "tidy docs"]),
            true,
        )
        .unwrap()
        .unwrap();
        assert!(
            !without.contains_key("role"),
            "an omitted flag must leave the worker default in place"
        );

        let with = tool_args(
            "dispatch",
            &args(&[
                "mini-swe-mcp",
                "dispatch",
                "integrate the round",
                "--group",
                "round-1",
                "--role",
                "consolidate",
            ]),
            true,
        )
        .unwrap()
        .unwrap();
        assert_eq!(with["role"], "consolidate");
        assert_eq!(with["group"], "round-1");
    }

    /// `--max-turns <n>` on `steer` is the revision's fresh turn budget: the
    /// same tool argument `dispatch` uses, so a revision is spelled exactly
    /// like the dispatch that preceded it.
    #[test]
    fn test_steer_max_turns_flag_maps_to_the_budget_property() {
        let plain = args(&["mini-swe-mcp", "steer", "w1", "fix the edge case"]);
        let without = tool_args("steer", &plain, true).unwrap().unwrap();
        assert!(
            !without.contains_key("max_turns"),
            "an omitted flag must not set a budget: {without:?}"
        );

        for flag in ["--max-turns", "-t"] {
            let flagged = args(&[
                "mini-swe-mcp",
                "steer",
                "w1",
                "fix the edge case",
                flag,
                "25",
            ]);
            let with = tool_args("steer", &flagged, true).unwrap().unwrap();
            assert_eq!(with["max_turns"], 25, "flag {flag}");
            assert_eq!(with["message"], "fix the edge case", "flag {flag}");
        }
    }

    #[test]
    fn test_tool_args_returns_none_for_a_taskless_dispatch() {
        let d = args(&["mini-swe-mcp", "dispatch"]);
        assert!(tool_args("dispatch", &d, true).unwrap().is_none());
    }

    #[test]
    fn test_dispatch_args_maps_every_flag_including_short_forms() {
        let cli_args = args(&[
            "mini-swe-mcp",
            "dispatch",
            "fix it",
            "--model",
            "m1",
            "--review-after",
            "m2",
            "--repo",
            "/tmp/r",
            "--max-turns",
            "12",
            "--group",
            "g1",
            "-m",
            "m3",
            "-r",
            "/tmp/r2",
            "-g",
            "g2",
            "-t",
            "3",
        ]);
        let mut tool_args = Map::new();
        dispatch_args(&cli_args, &mut tool_args).expect("valid flags");

        assert_eq!(tool_args["task"], "fix it");
        assert_eq!(tool_args["model"], "m3"); // the last flag wins
        assert_eq!(tool_args["review_after"], "m2");
        assert_eq!(tool_args["repo_path"], "/tmp/r2");
        assert!(
            !tool_args.contains_key("wait"),
            "dispatch never waits: {tool_args:?}"
        );
        assert_eq!(tool_args["max_turns"], 3);
        assert_eq!(tool_args["group"], "g2");
    }

    #[test]
    fn test_dispatch_args_ignores_unknown_flags_and_unparsable_turns() {
        let cli_args = args(&["mini-swe-mcp", "dispatch", "t", "--nope", "x", "-t", "nan"]);
        let mut tool_args = Map::new();
        dispatch_args(&cli_args, &mut tool_args).expect("valid flags");

        assert_eq!(tool_args["task"], "t");
        assert_eq!(
            tool_args.len(),
            1,
            "unknown flags add nothing: {tool_args:?}"
        );
    }

    #[test]
    fn test_dispatch_args_takes_the_next_word_verbatim_as_a_flag_value() {
        // Historical behaviour: a value flag consumes the next word whatever it
        // is, so `dispatch t --model --repo` sets model="--repo" and nothing else.
        let cli_args = args(&["mini-swe-mcp", "dispatch", "t", "--model", "--repo"]);
        let mut tool_args = Map::new();
        dispatch_args(&cli_args, &mut tool_args).expect("valid flags");

        assert_eq!(tool_args["model"], "--repo");
        assert!(!tool_args.contains_key("repo_path"), "{tool_args:?}");

        // A value flag with nothing after it is dropped, not defaulted.
        let trailing = args(&["mini-swe-mcp", "dispatch", "t", "--group"]);
        let mut tool_args = Map::new();
        dispatch_args(&trailing, &mut tool_args).expect("valid flags");
        assert_eq!(tool_args.len(), 1, "{tool_args:?}");
    }

    /// `dispatch -f <file>` turns the file's YAML list into the tool's `tasks`
    /// argument and leaves the single-task positional form untouched.
    #[test]
    fn test_dispatch_file_flag_builds_the_tasks_argument() {
        let dir = std::env::temp_dir().join(format!("mini-swe-batch-yaml-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("scratch dir");
        let path = dir.join("tasks.yaml");
        std::fs::write(
            &path,
            "- task: first\n  model: ninja\n- task: second\n  network: offline\n",
        )
        .expect("write batch file");

        let argv = args(&[
            "mini-swe-mcp",
            "dispatch",
            "-f",
            path.to_str().expect("utf8"),
        ]);
        let out = tool_args("dispatch", &argv, true).unwrap().unwrap();
        assert_eq!(out["action"], "dispatch");
        assert!(
            !out.contains_key("task"),
            "the batch form sets no positional task: {out:?}"
        );
        let tasks = out["tasks"].as_array().expect("tasks array");
        assert_eq!(tasks.len(), 2);
        assert_eq!(tasks[0]["task"], "first");
        assert_eq!(tasks[0]["model"], "ninja");
        assert_eq!(tasks[1]["task"], "second");
        assert_eq!(tasks[1]["network"], "offline");

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A JSON list is the same batch, and dispatch flags after the file are
    /// shared top-level defaults the file's own keys override.
    #[test]
    fn test_dispatch_file_flag_accepts_json_and_shared_defaults() {
        let dir = std::env::temp_dir().join(format!("mini-swe-batch-json-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("scratch dir");
        let path = dir.join("tasks.json");
        std::fs::write(&path, r#"[{"task":"a"},{"task":"b","network":"allow"}]"#)
            .expect("write batch file");

        let argv = args(&[
            "mini-swe-mcp",
            "dispatch",
            "-f",
            path.to_str().expect("utf8"),
            "--model",
            "nerd",
            "--offline",
        ]);
        let out = tool_args("dispatch", &argv, true).unwrap().unwrap();
        assert_eq!(
            out["model"], "nerd",
            "top-level flags are shared defaults: {out:?}"
        );
        assert_eq!(out["network"], "offline");
        let tasks = out["tasks"].as_array().expect("tasks array");
        assert_eq!(tasks.len(), 2);
        assert_eq!(tasks[0]["task"], "a");
        assert_eq!(tasks[1]["network"], "allow");

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A missing file, a non-list document and an empty list are hard errors,
    /// never a silent empty dispatch.
    #[test]
    fn test_dispatch_file_flag_rejects_bad_lists() {
        assert!(parse_batch_tasks("task: only-one\n").is_err());
        assert!(parse_batch_tasks("[]").is_err());
        assert!(parse_batch_tasks("not: [valid").is_err());

        let missing = tool_args(
            "dispatch",
            &args(&["mini-swe-mcp", "dispatch", "-f", "/nonexistent/tasks.yaml"]),
            true,
        );
        assert!(missing.is_err());

        let no_value = tool_args("dispatch", &args(&["mini-swe-mcp", "dispatch", "-f"]), true);
        assert!(no_value.is_err());
    }
}
