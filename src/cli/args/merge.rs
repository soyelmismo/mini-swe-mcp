use super::*;

pub(super) fn build(cli_args: &[String], tool_args: &mut Map<String, Value>) -> Result<()> {
    if flag_index(cli_args, &["--approved"]).is_some() {
        tool_args.insert("approved".into(), Value::Bool(true));
        if let Some(i) = flag_index(cli_args, &["--group", "-g"]) {
            let group = cli_args
                .get(i + 1)
                .ok_or_else(|| anyhow::anyhow!("--group needs a value"))?;
            tool_args.insert("group".into(), Value::String(group.clone()));
        }
    } else {
        target::worker_id(cli_args, tool_args);
    }
    target::keep_branch(cli_args, tool_args);
    if flag_index(cli_args, &["--force"]).is_some() {
        tool_args.insert("force".into(), Value::Bool(true));
    }
    Ok(())
}
