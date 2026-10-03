use super::*;

pub(super) fn build(
    action: &str,
    cli_args: &[String],
    tool_args: &mut Map<String, Value>,
) -> Result<()> {
    if cli_args.len() > 2 {
        tool_args.insert("worker_id".into(), Value::String(cli_args[2].clone()));
    }
    // `approve <id> ["note"]`: the note is the same `message` argument
    // the MCP action reads, and every unquoted word after the id belongs to it.
    if action == "approve" && cli_args.len() > 3 {
        tool_args.insert(
            "message".into(),
            Value::String(join_words(cli_args, 3, &[], &[])),
        );
    }

    Ok(())
}

/// Fold `review --diff <scope>` into the tool arguments.
///
/// The scope is the same `diff` argument the MCP action reads, so the selector
/// has one implementation and an unknown value is refused by the handler.
pub(super) fn review_diff_arg(cli_args: &[String], tool_args: &mut Map<String, Value>) {
    if let Some(mut i) = flag_index(cli_args, &["--diff"]) {
        take_value(cli_args, &mut i, tool_args, "diff");
    }
}

pub(super) fn build_view(cli_args: &[String], tool_args: &mut Map<String, Value>) {
    target::worker_id(cli_args, tool_args);
    review_diff_arg(cli_args, tool_args);
    target::keep_branch(cli_args, tool_args);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::args::tests::args;
    /// `merge --approved [--group <g>]` is the batch form: no worker id, the
    /// whole round's approved workers instead.
    #[test]
    fn test_merge_approved_flag_maps_to_the_batch_arguments() {
        let batch = tool_args(
            "merge",
            &args(&["mini-swe-mcp", "merge", "--approved", "--group", "round-1"]),
            true,
        )
        .unwrap()
        .unwrap();
        assert_eq!(batch["action"], "merge");
        assert_eq!(batch["approved"], true);
        assert_eq!(batch["group"], "round-1");
        assert!(
            !batch.contains_key("worker_id"),
            "a batch names no single worker: {batch:?}"
        );

        let every_group = tool_args(
            "merge",
            &args(&["mini-swe-mcp", "merge", "--approved"]),
            true,
        )
        .unwrap()
        .unwrap();
        assert_eq!(every_group["approved"], true);
        assert!(!every_group.contains_key("group"), "{every_group:?}");

        // The single-worker form is unchanged.
        let one = tool_args("merge", &args(&["mini-swe-mcp", "merge", "w1"]), true)
            .unwrap()
            .unwrap();
        assert_eq!(one["worker_id"], "w1");
        assert!(!one.contains_key("approved"), "{one:?}");
    }

    /// `review <id>` is a read like `status`: one positional, plus the optional
    /// diff scope, which maps to the same `diff` argument the MCP action reads.
    #[test]
    fn test_review_maps_its_positional_worker_id() {
        let out = tool_args("review", &args(&["mini-swe-mcp", "review", "w1"]), true)
            .unwrap()
            .unwrap();
        assert_eq!(out["action"], "review");
        assert_eq!(out["worker_id"], "w1");
        assert!(
            !out.contains_key("diff"),
            "the default scope is implicit: {out:?}"
        );

        let scoped = tool_args(
            "review",
            &args(&["mini-swe-mcp", "review", "w1", "--diff", "all"]),
            true,
        )
        .unwrap()
        .unwrap();
        assert_eq!(scoped["worker_id"], "w1");
        assert_eq!(scoped["diff"], "all");
    }

    /// `approve <id> ["note"]` maps the id and the note to the same `message`
    /// argument the MCP action reads; `unapprove <id>` needs only the id.
    #[test]
    fn test_approve_maps_the_id_and_optional_note() {
        let bare = tool_args("approve", &args(&["mini-swe-mcp", "approve", "w1"]), true)
            .unwrap()
            .unwrap();
        assert_eq!(bare["action"], "approve");
        assert_eq!(bare["worker_id"], "w1");
        assert!(!bare.contains_key("message"), "{bare:?}");

        let noted = tool_args(
            "approve",
            &args(&["mini-swe-mcp", "approve", "w1", "looks right"]),
            true,
        )
        .unwrap()
        .unwrap();
        assert_eq!(noted["message"], "looks right");

        let back = tool_args(
            "unapprove",
            &args(&["mini-swe-mcp", "unapprove", "w1"]),
            true,
        )
        .unwrap()
        .unwrap();
        assert_eq!(back["action"], "unapprove");
        assert_eq!(back["worker_id"], "w1");
    }
}
