use super::*;

pub(super) fn build(
    cli_args: &[String],
    api_key_present: bool,
    tool_args: &mut Map<String, Value>,
) -> Result<Option<()>> {
    if !api_key_present {
        anyhow::bail!(
            "Missing OPENAI_API_KEY. Please provide it via environment variable or .env file."
        );
    }
    match batch_tasks(cli_args)? {
        Some((file_index, tasks)) => {
            // Every entry's task comes from the file, so a bare word beside
            // `-f <path>` has nowhere to go: refuse it rather than dispatch the
            // file and drop what the operator asked for.
            if let Some(extra) = extra_batch_word(cli_args, file_index) {
                anyhow::bail!(
                    "dispatch -f <{path}> takes flags only, but got the extra word {extra:?}; \
                     put the tasks in the file or dispatch one task without -f",
                    path = cli_args[file_index + 1]
                );
            }
            tool_args.insert("tasks".into(), tasks);
            // Flags after the file are shared defaults for every entry;
            // the file flag and its path are skipped, never parsed.
            collect_dispatch_flags(cli_args, 3, &[file_index, file_index + 1], tool_args);
        }
        None => {
            if cli_args.len() < 3 {
                eprintln!("Usage: mini-swe-mcp {DISPATCH_USAGE}");
                return Ok(None);
            }
            dispatch_args(cli_args, tool_args)?;
        }
    }

    Ok(Some(()))
}

/// Fold the `dispatch` flags after the task into the tool arguments.
///
/// The task is every word that is not a flag or a flag's value, so an unquoted
/// `dispatch fix the flaky test` arrives whole; a value flag consumes the next
/// word verbatim, and one with nothing after it is dropped rather than
/// defaulted to an empty string.
pub(super) fn dispatch_args(cli_args: &[String], tool_args: &mut Map<String, Value>) -> Result<()> {
    tool_args.insert(
        "task".into(),
        Value::String(join_words(cli_args, 2, &[], DISPATCH_VALUE_FLAGS)),
    );
    collect_dispatch_flags(cli_args, 3, &[], tool_args);
    Ok(())
}

/// The first bare word beside a batch dispatch's `-f <path>`, if there is one.
///
/// Only the words no flag claims are reported, so the documented
/// `dispatch -f tasks.yaml --group g` spellings stay silent.
fn extra_batch_word(cli_args: &[String], file_index: usize) -> Option<String> {
    let skip = [file_index, file_index + 1];
    let mut i = 3;
    while i < cli_args.len() {
        if !skip.contains(&i) && !cli_args[i].starts_with('-') {
            return Some(cli_args[i].clone());
        }
        if DISPATCH_VALUE_FLAGS.contains(&cli_args[i].as_str()) {
            i += 1;
        }
        i += 1;
    }
    None
}

/// The `dispatch` flags that consume the following word as their value.
///
/// Shared by the positional task and the flag fold, so a word that is one
/// flag's value can never leak into the other one.
pub(super) const DISPATCH_VALUE_FLAGS: &[&str] = &[
    "--model",
    "-m",
    "--review-after",
    "--repo",
    "-r",
    "--max-turns",
    "-t",
    "--group",
    "-g",
    "--verify",
    "--consolidate-verify",
    "--role",
];

/// Read the batch list named by `-f`/`--file`, returning the flag's position
/// and the parsed `tasks` array. `Ok(None)` means the flag was not passed.
///
/// `-` reads standard input, so `mini-swe-mcp dispatch -f - < tasks.yaml` works.
pub(super) fn batch_tasks(cli_args: &[String]) -> Result<Option<(usize, Value)>> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::args::tests::args;
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

        // An unknown flag is ignored but the word after it is still a task word,
        // so nothing the operator typed is silently dropped.
        assert_eq!(tool_args["task"], "t x");
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
