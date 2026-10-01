//! argv → `worker` tool argument mapping for the CLI-only verbs.
//!
//! The binary never re-implements tool semantics: it only turns positional argv
//! into the same JSON object the MCP `tools/call` path would have sent, so both
//! callers share one validation and rendering implementation.

use anyhow::Result;
use serde_json::{Map, Value};

mod collect;
mod consolidate;
mod dispatch;
mod list;
mod merge;
mod review;
mod steer;
mod target;

pub use dispatch::parse_batch_tasks;

/// Dispatch usage line, shared by `--help` and the missing-task error.
pub const DISPATCH_USAGE: &str = "dispatch <task> | dispatch -f <tasks.yaml> [--model <model>] [--review-after <model>] [--repo <repo>] [--max-turns <n>] [--group <group>] [--role <role>] [--offline] [--verify <cmd>] (task: ONE focused concern, scoped files, acceptance gate; -f runs a YAML/JSON list, '-' reads stdin)";

/// Consolidate usage line, shared by `--help` and the missing-group error.
pub const CONSOLIDATE_USAGE: &str =
    "consolidate --group <group> [--model <model>] [--verify <cmd>] [--max-turns <n>]";

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
            if dispatch::build(cli_args, api_key_present, &mut tool_args)?.is_none() {
                return Ok(None);
            }
        }
        "collect" => collect::build(cli_args, &mut tool_args),
        "review" => review::build_view(cli_args, &mut tool_args),
        "merge" => merge::build(cli_args, &mut tool_args)?,
        "kill" | "logs" | "status" => target::build(cli_args, &mut tool_args),
        "consolidate" => consolidate::build(cli_args, &mut tool_args)?,
        "steer" => steer::build(cli_args, &mut tool_args)?,
        "approve" | "unapprove" => review::build(action, cli_args, &mut tool_args)?,
        "list" => list::build(cli_args, &mut tool_args)?,
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
mod tests;
