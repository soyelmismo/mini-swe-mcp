use super::*;

pub(super) fn worker_id(cli_args: &[String], tool_args: &mut Map<String, Value>) {
    if cli_args.len() > 2 {
        tool_args.insert("worker_id".into(), Value::String(cli_args[2].clone()));
    }
}

pub(super) fn keep_branch(cli_args: &[String], tool_args: &mut Map<String, Value>) {
    // `--no-delete` keeps the merged branch: the same tool argument
    // the MCP action reads, so the flag has one implementation.
    if flag_index(cli_args, &["--no-delete"]).is_some() {
        tool_args.insert("keep_branch".into(), Value::Bool(true));
    }
}

pub(super) fn build(cli_args: &[String], tool_args: &mut Map<String, Value>) {
    worker_id(cli_args, tool_args);
    keep_branch(cli_args, tool_args);
}
