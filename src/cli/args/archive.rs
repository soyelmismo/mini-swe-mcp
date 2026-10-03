use super::*;

/// Fold the `archive` flags into the tool arguments.
///
/// `--last` is a bound, not a filter: it keeps the newest N of whatever
/// survives the other filters, so it applies after `--group`. A `--last 0` is
/// refused rather than answered with nothing, because "the last zero workers"
/// is never what an operator meant to ask.
pub(super) fn build(cli_args: &[String], tool_args: &mut Map<String, Value>) -> Result<()> {
    let mut i = 2;
    while i < cli_args.len() {
        match cli_args[i].as_str() {
            "--group" | "-g" => take_value(cli_args, &mut i, tool_args, "group"),
            "--last" | "-n" => {
                let raw = cli_args
                    .get(i + 1)
                    .ok_or_else(|| anyhow::anyhow!("--last needs a count"))?;
                let count: u64 = raw
                    .parse()
                    .map_err(|_| anyhow::anyhow!("--last needs a count, got {raw:?}"))?;
                if count == 0 {
                    anyhow::bail!("--last needs at least 1");
                }
                tool_args.insert("last".into(), Value::from(count));
                i += 1;
            }
            _ => {}
        }
        i += 1;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::args::tests::args;

    fn built(argv: &[&str]) -> Map<String, Value> {
        tool_args("archive", &args(argv), true)
            .unwrap()
            .expect("archive is a dispatchable action")
    }

    #[test]
    fn test_archive_group_and_last_map_to_the_tool_properties() {
        let tool_args = built(&["mini-swe-mcp", "archive", "--group", "round-1", "--last", "3"]);
        assert_eq!(tool_args["action"], "archive");
        assert_eq!(tool_args["group"], "round-1");
        assert_eq!(tool_args["last"], 3);
    }

    #[test]
    fn test_archive_without_filters_carries_neither() {
        let tool_args = built(&["mini-swe-mcp", "archive"]);
        assert_eq!(tool_args["action"], "archive");
        assert!(!tool_args.contains_key("group"), "{tool_args:?}");
        assert!(!tool_args.contains_key("last"), "{tool_args:?}");
    }

    #[test]
    fn test_archive_refuses_a_last_of_zero_or_a_nonsense_one() {
        for argv in [
            &["mini-swe-mcp", "archive", "--last", "0"][..],
            &["mini-swe-mcp", "archive", "--last", "many"][..],
            &["mini-swe-mcp", "archive", "--last"][..],
        ] {
            let error = tool_args("archive", &args(argv), true)
                .expect_err("--last 0, a non-count or a missing count must be refused");
            assert!(
                error.to_string().contains("--last"),
                "the refusal must name the flag: {error}"
            );
        }
    }
}
