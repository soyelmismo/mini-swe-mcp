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
