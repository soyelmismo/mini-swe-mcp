use super::*;

pub(super) fn build(cli_args: &[String], tool_args: &mut Map<String, Value>) -> Result<()> {
    consolidate_args(cli_args, tool_args)?;

    Ok(())
}

/// Fold the `consolidate` flags into the tool arguments.
///
/// `--group` is the one required flag: a consolidator integrates exactly one
/// group's round, so a missing one is a hard error instead of a dispatch that
/// would have to guess. The rest are the defaults that differ from a plain
/// dispatch -- the model, the full gate and the turn budget -- and every one of
/// them is optional because the hub computes a sensible value.
///
/// `--set` picks the other meaning of the same verb: amend the pending round's
/// auto-consolidation settings (`--model` and/or `--verify`) through the hub
/// rather than dispatching its consolidator now.
pub(super) fn consolidate_args(
    cli_args: &[String],
    tool_args: &mut Map<String, Value>,
) -> Result<()> {
    let mut i = 2;
    while i < cli_args.len() {
        match cli_args[i].as_str() {
            "--group" | "-g" => take_value(cli_args, &mut i, tool_args, "group"),
            "--model" | "-m" => take_value(cli_args, &mut i, tool_args, "model"),
            "--verify" => take_value(cli_args, &mut i, tool_args, "verify"),
            "--max-turns" | "-t" => take_turns(cli_args, &mut i, tool_args),
            // `--set` amends the pending round's settings instead of
            // dispatching its consolidator; `--model`/`--verify` say which.
            "--set" => {
                tool_args.insert("set".into(), Value::Bool(true));
            }
            _ => {}
        }
        i += 1;
    }
    if !tool_args.contains_key("group") {
        anyhow::bail!("Usage: mini-swe-mcp {CONSOLIDATE_USAGE}");
    }
    Ok(())
}
