use super::*;

/// Fold the `collect` diff selectors into the tool arguments.
///
/// `--full` asks for the whole diff and `--file <path>` (repeatable) narrows it
/// to the named files: the same tool arguments the MCP path sends, so the diff
/// scope has exactly one implementation.
pub(super) fn collect_diff_args(cli_args: &[String], tool_args: &mut Map<String, Value>) {
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

pub(super) fn build(cli_args: &[String], tool_args: &mut Map<String, Value>) {
    target::worker_id(cli_args, tool_args);
    collect_diff_args(cli_args, tool_args);
    target::keep_branch(cli_args, tool_args);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::args::tests::args;
    use serde_json::json;
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
}
