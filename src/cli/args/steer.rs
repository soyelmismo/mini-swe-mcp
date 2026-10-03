use super::*;

/// Turn `steer <id> <message...>` into the tool arguments.
///
/// The message is every remaining word: an unquoted `steer <id> sal del loop!`
/// arrives as four argv words, and reading only the first one would drop the
/// rest of the instruction without a word of complaint. `--max-turns`/`-t` is
/// the one flag, and it may sit anywhere after the id.
pub(super) fn build(cli_args: &[String], tool_args: &mut Map<String, Value>) -> Result<()> {
    if cli_args.len() > 3 {
        tool_args.insert("worker_id".into(), Value::String(cli_args[2].clone()));
        let message = join_words(cli_args, 3, &[], &["--max-turns", "-t"]);
        if message.is_empty() {
            anyhow::bail!("steer needs a message: mini-swe-mcp steer <worker_id> <message>");
        }
        tool_args.insert("message".into(), Value::String(message));
        // `--max-turns <n>` on a steer is the fresh turn budget of a
        // revision (steering a finished worker); the same tool
        // argument `dispatch` uses, so the budget has one spelling.
        if let Some(mut i) = flag_index(cli_args, &["--max-turns", "-t"]) {
            take_turns(cli_args, &mut i, tool_args);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::args::tests::args;
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
}
