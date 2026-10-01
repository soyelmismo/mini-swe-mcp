use super::*;

pub(super) fn build(cli_args: &[String], tool_args: &mut Map<String, Value>) -> Result<()> {
    // `--all` is the operator's view of the whole pool; without it the
    // CLI sees only the workers the `cli` identity owns.
    if flag_index(cli_args, &["--all"]).is_some() {
        tool_args.insert("scope".into(), Value::String("all".into()));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::args::tests::args;
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
}
