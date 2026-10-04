use super::*;

pub fn format_dispatch(val: &serde_json::Value) -> String {
    if let Some(workers) = val.get("workers").and_then(|v| v.as_array()) {
        return format_batch_dispatch(val, workers);
    }
    let wid = val.get("worker_id").and_then(|v| v.as_str()).unwrap_or("");
    if val.get("status").and_then(|v| v.as_str()) == Some("dispatched") {
        let mut out = format!(
            "✓ Worker {wid} dispatched in background.\nUse 'mini-swe-mcp status {wid}' to check progress."
        );
        out.push_str(&watch_command_line(val));
        out
    } else if val.get("status").and_then(|v| v.as_str()) == Some("still_running") {
        // A bounded wait hands the worker back unfinished; say so instead of
        // implying it finished.
        let step = val.get("step").and_then(|v| v.as_u64()).unwrap_or(0);
        let last_command = val
            .get("last_command")
            .and_then(|v| v.as_str())
            .unwrap_or("no command reported");
        format!(
            "Worker {wid} is still running (step {step}): {last_command}\nCall 'mini-swe-mcp wait {wid}' again to keep waiting."
        )
    } else {
        let health = health_line(val);
        let mut out = format!("✓ Worker {wid} finished.\n");
        if let Some(state) = val.get("state") {
            let state_name = state.get("state").and_then(|v| v.as_str()).unwrap_or("");
            let details = state
                .get("details")
                .or_else(|| state.get("Completed"))
                .or_else(|| state.get("Failed"));

            if state_name == "Completed" || state.get("Completed").is_some() {
                if let Some(turns) = details
                    .and_then(|d| d.get("turns"))
                    .and_then(|v| v.as_u64())
                {
                    out.push_str(&format!("Turns: {turns}\n"));
                }
                if let Some(summary) = details
                    .and_then(|d| d.get("summary"))
                    .and_then(|v| v.as_str())
                {
                    out.push_str(&format!("Summary: {summary}\n"));
                }
                if let Some(d) = details {
                    push_verified_line(&mut out, d.get("verified"));
                }
                if let Some(branch) = details
                    .and_then(|d| d.get("branch"))
                    .and_then(|v| v.as_str())
                {
                    out.push_str(&format!("Branch: {branch}\n"));
                }
                if let Some(artifacts) = details
                    .and_then(|d| d.get("artifacts"))
                    .and_then(|v| v.as_array())
                    && !artifacts.is_empty()
                {
                    let list: Vec<&str> = artifacts.iter().filter_map(|a| a.as_str()).collect();
                    out.push_str(&format!("Preserved Artifacts: {}\n", list.join(", ")));
                }
                if let Some(health) = &health {
                    out.push_str(&format!("{health}\n"));
                }
                if let Some(diff) = details.and_then(|d| d.get("diff")).and_then(|v| v.as_str())
                    && !diff.trim().is_empty()
                {
                    out.push_str(&format!("\nDiff:\n{diff}\n"));
                }
                if let Some(next) = val.get("next_step").and_then(|v| v.as_str())
                    && !next.trim().is_empty()
                {
                    out.push_str(&format!("\nNext step: {next}\n"));
                }
            } else if state_name == "Failed" || state.get("Failed").is_some() {
                out.push_str("State: Failed\n");
                if let Some(err) = details
                    .and_then(|d| d.get("error"))
                    .and_then(|v| v.as_str())
                {
                    out.push_str(&format!("Error: {err}\n"));
                }
                if let Some(health) = &health {
                    out.push_str(&format!("{health}\n"));
                }
            }
        }
        out.trim_end().to_string()
    }
}

/// `dispatch` with `tasks`: one line per entry, including the error an entry
/// that never started reported, so a batch never hides a partial failure.
pub(super) fn format_batch_dispatch(
    val: &serde_json::Value,
    workers: &[serde_json::Value],
) -> String {
    let dispatched = val.get("dispatched").and_then(|v| v.as_u64()).unwrap_or(0);
    let failed = val.get("failed").and_then(|v| v.as_u64()).unwrap_or(0);
    let mut out = format!("✓ Batch dispatch: {dispatched} started, {failed} failed.");
    for worker in workers {
        let index = worker.get("index").and_then(|v| v.as_u64()).unwrap_or(0);
        if let Some(wid) = worker.get("worker_id").and_then(|v| v.as_str()) {
            out.push_str(&format!(
                "\n  - Task {index}: worker {wid} dispatched in background."
            ));
        } else if let Some(error) = worker.get("error").and_then(|v| v.as_str()) {
            out.push_str(&format!("\n  - Task {index} failed: {error}"));
        }
    }
    out.push_str(&watch_command_line(val));
    out
}

/// The split a `--quiet` dispatch prints: the started worker ids on stdout, one
/// per line, and the error each entry that never started reported on stderr.
///
/// Keeping the two apart here, in a pure formatter, is what lets the binary
/// keep stdout a clean id list for a script while the failure stays visible.
#[derive(Debug)]
pub struct QuietDispatch {
    /// The `worker_id` of every entry that started, in payload order.
    pub worker_ids: Vec<String>,
    /// The error of every entry that never started, in payload order.
    pub errors: Vec<String>,
    /// The watch command to wait on these workers with.
    ///
    /// Empty exactly when no worker started, so the caller can gate the
    /// reminder on "something is running" alone and never on the command's
    /// own emptiness: a caller with no token store still needs the reminder.
    pub watch_command: String,
}

/// `dispatch --quiet`: the ids to print, and the entry errors to report.
///
/// A single dispatch carries `worker_id`; a batch carries `workers`, whose
/// entries hold either `worker_id` or `error`.
pub fn format_dispatch_quiet(val: &serde_json::Value) -> QuietDispatch {
    let mut worker_ids = Vec::new();
    let mut errors = Vec::new();
    if let Some(workers) = val.get("workers").and_then(|v| v.as_array()) {
        for worker in workers {
            if let Some(wid) = worker.get("worker_id").and_then(|v| v.as_str()) {
                worker_ids.push(wid.to_string());
            } else if let Some(error) = worker.get("error").and_then(|v| v.as_str()) {
                errors.push(error.to_string());
            }
        }
    } else if let Some(wid) = val.get("worker_id").and_then(|v| v.as_str()) {
        worker_ids.push(wid.to_string());
    }
    // A dispatch that started nothing has nothing to wait on, so the reminder
    // is withheld rather than pointing at a watch with no worker behind it.
    let watch_command = match worker_ids.is_empty() {
        true => String::new(),
        false => quiet_watch_command(val),
    };
    QuietDispatch {
        worker_ids,
        errors,
        watch_command,
    }
}

/// The watch command a `--quiet` dispatch reminder names.
///
/// Same precedence as the human-facing [`format_dispatch`]: the hub's
/// token-bound `watch_command` when it minted one, the round command for a
/// consolidated dispatch, and the plain `mini-swe-mcp watch` otherwise -- which
/// follows every worker the caller owns, so it is right even with no token and
/// no group.
fn quiet_watch_command(val: &serde_json::Value) -> String {
    if let Some(command) = val.get("watch_command").and_then(|v| v.as_str()) {
        return command.to_string();
    }
    match val.get("group").and_then(|v| v.as_str()) {
        Some(group) => {
            format!("mini-swe-mcp watch --group {} --all", shell_word(group))
        }
        None => "mini-swe-mcp watch".to_string(),
    }
}

/// `word` as one shell word: quoted only when it holds something a shell would
/// act on, so an ordinary group name stays the readable command the help text
/// shows and a crafted one cannot become a second command.
fn shell_word(word: &str) -> String {
    let safe = !word.is_empty()
        && word
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || "-_./:=@,+".contains(c));
    match safe {
        true => word.to_string(),
        false => crate::agent::exec::shell_quote(word),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::format::worker::tests::v;
    /// A batch dispatch renders one line per entry, including the entries that
    /// failed, so a partial failure is visible at a glance.
    #[test]
    fn test_format_dispatch_renders_a_batch() {
        let out = format_dispatch(&v(
            r#"{"workers":[{"index":0,"worker_id":"w1","network":"allow"},{"index":1,"error":"'task' is required"}],"dispatched":1,"failed":1}"#,
        ));
        assert!(out.contains("Batch dispatch: 1 started, 1 failed"), "{out}");
        assert!(out.contains("Task 0: worker w1 dispatched"), "{out}");
        assert!(out.contains("Task 1 failed: 'task' is required"), "{out}");
    }

    /// `--quiet` is a clean id list: one started id per line, in payload order,
    /// and the entries that failed go to the errors side, never into stdout.
    #[test]
    fn test_format_dispatch_quiet_lists_ids_and_splits_errors() {
        let single = format_dispatch_quiet(&v(r#"{"worker_id":"w1","status":"dispatched"}"#));
        assert_eq!(single.worker_ids, vec!["w1"]);
        assert!(single.errors.is_empty(), "{:?}", single.errors);

        let batch = format_dispatch_quiet(&v(
            r#"{"workers":[{"index":0,"worker_id":"w1"},{"index":1,"error":"'task' is required"},{"index":2,"worker_id":"w2"}]}"#,
        ));
        assert_eq!(batch.worker_ids, vec!["w1", "w2"]);
        assert_eq!(batch.errors, vec!["'task' is required"]);
    }

    /// A `--quiet` dispatch that started a worker always names a watch: the
    /// reminder is the whole point of the flag, so withholding it because the
    /// hub minted no token-bound command is exactly the failure it exists to
    /// prevent. It is withheld only when nothing started.
    #[test]
    fn a_started_worker_always_gets_a_watch_command() {
        // No token store (the in-process stdio server): still a usable command,
        // since a watch with no ids follows every worker the caller owns.
        let single = format_dispatch_quiet(&v(r#"{"worker_id":"w1","status":"dispatched"}"#));
        assert_eq!(
            single.watch_command, "mini-swe-mcp watch",
            "a token-less caller must still be told how to wait"
        );

        let batch = format_dispatch_quiet(&v(r#"{"workers":[{"index":0,"worker_id":"w1"}]}"#));
        assert_eq!(batch.watch_command, "mini-swe-mcp watch", "{batch:?}");

        // The hub's token-bound command wins: it is what binds the watch to
        // this caller's identity.
        let tokenized = format_dispatch_quiet(&v(
            r#"{"worker_id":"w1","watch_command":"MINI_SWE_WATCH_TOKEN=abc mini-swe-mcp watch"}"#,
        ));
        assert_eq!(
            tokenized.watch_command,
            "MINI_SWE_WATCH_TOKEN=abc mini-swe-mcp watch"
        );

        // A consolidated round waits for the whole round, not one worker.
        let round = format_dispatch_quiet(&v(
            r#"{"workers":[{"index":0,"worker_id":"w1"}],"group":"round-1"}"#,
        ));
        assert_eq!(
            round.watch_command, "mini-swe-mcp watch --group round-1 --all",
            "{round:?}"
        );

        // Nothing started: no watch command, so no reminder is printed.
        let nothing = format_dispatch_quiet(&v(
            r#"{"workers":[{"index":0,"error":"'task' is required"}]}"#,
        ));
        assert!(nothing.worker_ids.is_empty());
        assert!(
            nothing.watch_command.is_empty(),
            "no worker means no watch to remind about: {:?}",
            nothing.watch_command
        );
    }

    /// A group name is caller-supplied text that reaches the command line the
    /// reminder tells the operator to run, so it must be quoted: an unquoted
    /// name would let whoever named the group choose what that shell executes.
    #[test]
    fn a_hostile_group_name_stays_inside_one_shell_word() {
        let hostile = format_dispatch_quiet(&v(
            r#"{"workers":[{"index":0,"worker_id":"w1"}],"group":"r; touch /tmp/pwned"}"#,
        ));
        assert_eq!(
            hostile.watch_command, "mini-swe-mcp watch --group 'r; touch /tmp/pwned' --all",
            "{hostile:?}"
        );

        // An embedded quote must not close the quoting either: the quoted word
        // has to come back as the operator typed it, and nothing beside it.
        let quoting = format_dispatch_quiet(&v(
            r#"{"workers":[{"index":0,"worker_id":"w1"}],"group":"r'x"}"#,
        ));
        assert_eq!(
            quoting.watch_command,
            format!("mini-swe-mcp watch --group {} --all", "'r'\\''x'"),
            "{quoting:?}"
        );
    }

    #[test]
    fn test_format_dispatch_shows_the_watch_command_when_the_hub_minted_one() {
        let with_token = format_dispatch(&v(
            r#"{"worker_id":"w","status":"dispatched","watch_command":"MINI_SWE_WATCH_TOKEN=abc mini-swe-mcp watch"}"#,
        ));
        assert!(
            with_token.contains("\nTo wait for it: MINI_SWE_WATCH_TOKEN=abc mini-swe-mcp watch")
                && with_token
                    .contains("run it in the background as-is; run it again after each event"),
            "{with_token}"
        );
        let without = format_dispatch(&v(r#"{"worker_id":"w","status":"dispatched"}"#));
        assert!(!without.contains("To wait for it"), "{without}");
    }

    #[test]
    fn test_format_dispatch_reports_background_and_terminal_states() {
        let background = format_dispatch(&v(r#"{"worker_id":"w","status":"dispatched"}"#));
        assert!(background.contains("dispatched in background"));
        assert!(background.contains("status w"));

        let completed = format_dispatch(&v(
            r#"{"worker_id":"w","state":{"state":"Completed","details":{
                 "turns":9,"summary":"done","branch":"b","artifacts":["a.md"],"diff":"--- a"}}}"#,
        ));
        assert!(completed.starts_with("✓ Worker w finished.\n"));
        assert!(completed.contains("Turns: 9"));
        assert!(completed.contains("Branch: b"));
        assert!(completed.contains("Preserved Artifacts: a.md"));
        assert!(completed.contains("\nDiff:\n--- a"));

        let failed = format_dispatch(&v(
            r#"{"worker_id":"w","state":{"Failed":{"error":"exploded"}}}"#,
        ));
        assert_eq!(
            failed,
            "✓ Worker w finished.\nState: Failed\nError: exploded"
        );
    }
}
