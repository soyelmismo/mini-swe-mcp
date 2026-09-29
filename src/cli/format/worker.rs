//! Plain-text renderers for the per-worker inspection verbs.
//!
//! Formatters behind `status`, `collect`, `logs`, `dispatch`, `steer`, `kill`
//! and `reap` — the actions that answer about one worker (or, for `reap`, about
//! a set of terminal workers) rather than about the system catalog.
//! [`log_counters_line`] is the shared counter/notice line that `collect` and
//! `logs` both append so step-log truncation is never silent (audit 07, R7).
//!
//! Every function here is a pure function over [`serde_json::Value`]: no I/O,
//! no state, no formatting knobs. That is what keeps them unit-testable and
//! keeps the CLI dispatch layer down to argument handling.

/// One-line summary of the step-log counters present in `val`.
pub fn log_counters_line(val: &serde_json::Value) -> String {
    let read = |key: &str| val.get(key).and_then(|v| v.as_u64());
    let total = read("total_steps");
    let emitted = val
        .get("logs")
        .and_then(|v| v.as_array())
        .map(|a| a.len() as u64);
    let retained = read("logs_retained")
        .or_else(|| emitted.map(|n| n.saturating_add(read("logs_omitted").unwrap_or(0))));
    let omitted = read("logs_omitted");
    let dropped = read("logs_dropped");
    let mut parts: Vec<String> = Vec::new();
    if let Some(total) = total {
        parts.push(format!("total_steps: {total}"));
    }
    if let Some(retained) = retained {
        parts.push(format!("retained: {retained}"));
    }
    if let Some(omitted) = omitted
        && omitted > 0
    {
        parts.push(format!("omitted: {omitted}"));
    }
    if let Some(dropped) = dropped
        && dropped > 0
    {
        parts.push(format!("dropped: {dropped}"));
    }
    if let Some(notice) = val.get("logs_truncation_notice").and_then(|v| v.as_str()) {
        parts.push(notice.to_string());
    }
    if parts.is_empty() {
        return "no step logs".to_string();
    }
    parts.join(" | ")
}

pub fn format_status(val: &serde_json::Value) -> String {
    let wid = val.get("worker_id").and_then(|v| v.as_str()).unwrap_or("");
    let mut out = format!("Worker: {wid}\n");
    if let Some(state) = val.get("state") {
        if let Some(status_str) = state.as_str() {
            out.push_str(&format!("State: {status_str}\n"));
        } else {
            let tag = state.get("state").and_then(|v| v.as_str());
            let details = state.get("details").and_then(|v| v.as_object());

            if let Some(t) = tag {
                out.push_str(&format!("State: {t}\n"));
                if let Some(d) = details {
                    if let Some(turns) = d.get("turns").and_then(|v| v.as_u64()) {
                        out.push_str(&format!("Turns: {turns}\n"));
                    }
                    if let Some(step) = d.get("step").and_then(|v| v.as_u64()) {
                        out.push_str(&format!("Step: {step}\n"));
                    }
                    if let Some(summary) = d.get("summary").and_then(|v| v.as_str()) {
                        out.push_str(&format!("Summary: {summary}\n"));
                    }
                    if let Some(err) = d.get("error").and_then(|v| v.as_str()) {
                        out.push_str(&format!("Error: {err}\n"));
                    }
                    if let Some(q) = d.get("question").and_then(|v| v.as_str()) {
                        out.push_str(&format!("Question: {q}\n"));
                    }
                }
            } else if let Some(obj) = state.as_object() {
                for (state_name, d) in obj {
                    out.push_str(&format!("State: {state_name}\n"));
                    if let Some(turns) = d.get("turns").and_then(|v| v.as_u64()) {
                        out.push_str(&format!("Turns: {turns}\n"));
                    }
                    if let Some(step) = d.get("step").and_then(|v| v.as_u64()) {
                        out.push_str(&format!("Step: {step}\n"));
                    }
                    if let Some(summary) = d.get("summary").and_then(|v| v.as_str()) {
                        out.push_str(&format!("Summary: {summary}\n"));
                    }
                    if let Some(err) = d.get("error").and_then(|v| v.as_str()) {
                        out.push_str(&format!("Error: {err}\n"));
                    }
                    if let Some(q) = d.get("question").and_then(|v| v.as_str()) {
                        out.push_str(&format!("Question: {q}\n"));
                    }
                }
            }
        }
    }
    out.trim_end().to_string()
}

pub fn format_collect(val: &serde_json::Value) -> String {
    let wid = val.get("worker_id").and_then(|v| v.as_str()).unwrap_or("");
    let diff = val
        .get("state")
        .and_then(|s| s.get("details").or_else(|| s.get("Completed")))
        .and_then(|c| c.get("diff"))
        .or_else(|| val.get("diff"))
        .and_then(|v| v.as_str())
        .unwrap_or("");

    let counters = log_counters_line(val);
    if diff.trim().is_empty() {
        format!("Worker {wid}: No git diff produced.\n{counters}")
    } else {
        format!("{diff}\n{counters}")
    }
}

/// Render the `logs` action: the bounded window plus the counters that make any
/// truncation visible instead of silent (audit 07, R7).
pub fn format_logs(val: &serde_json::Value) -> String {
    let wid = val.get("worker_id").and_then(|v| v.as_str()).unwrap_or("");
    let mut out = format!("Worker {wid} step logs\n");
    if let Some(entries) = val.get("logs").and_then(|v| v.as_array()) {
        for entry in entries {
            let step = entry.get("step").and_then(|v| v.as_u64()).unwrap_or(0);
            let command = entry.get("command").and_then(|v| v.as_str()).unwrap_or("");
            out.push_str(&format!("  [{step}] {command}\n"));
        }
    }
    out.push_str(&format!(
        "{}
",
        log_counters_line(val)
    ));
    out.trim_end().to_string()
}

pub fn format_reap(val: &serde_json::Value) -> String {
    let reaped = val.get("reaped").and_then(|v| v.as_u64()).unwrap_or(0);
    let ids = val
        .get("worker_ids")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        })
        .unwrap_or_default();
    if reaped == 0 {
        "✓ No expired terminal worker records to reap.".to_string()
    } else {
        format!("✓ Reaped {reaped} expired worker record(s): {ids}")
    }
}

pub fn format_dispatch(val: &serde_json::Value) -> String {
    let wid = val.get("worker_id").and_then(|v| v.as_str()).unwrap_or("");
    if val.get("status").and_then(|v| v.as_str()) == Some("dispatched") {
        format!("✓ Worker {wid} dispatched in background.\nUse 'mini-swe-mcp status {wid}' to check progress.")
    } else {
        let mut out = format!("✓ Worker {wid} finished.\n");
        if let Some(state) = val.get("state") {
            let state_name = state.get("state").and_then(|v| v.as_str()).unwrap_or("");
            let details = state
                .get("details")
                .or_else(|| state.get("Completed"))
                .or_else(|| state.get("Failed"));

            if state_name == "Completed" || state.get("Completed").is_some() {
                if let Some(turns) = details.and_then(|d| d.get("turns")).and_then(|v| v.as_u64()) {
                    out.push_str(&format!("Turns: {turns}\n"));
                }
                if let Some(summary) = details.and_then(|d| d.get("summary")).and_then(|v| v.as_str()) {
                    out.push_str(&format!("Summary: {summary}\n"));
                }
                if let Some(branch) = details.and_then(|d| d.get("branch")).and_then(|v| v.as_str()) {
                    out.push_str(&format!("Branch: {branch}\n"));
                }
                if let Some(artifacts) = details.and_then(|d| d.get("artifacts")).and_then(|v| v.as_array())
                    && !artifacts.is_empty()
                {
                    let list: Vec<&str> = artifacts.iter().filter_map(|a| a.as_str()).collect();
                    out.push_str(&format!("Preserved Artifacts: {}\n", list.join(", ")));
                }
                if let Some(diff) = details.and_then(|d| d.get("diff")).and_then(|v| v.as_str())
                    && !diff.trim().is_empty()
                {
                    out.push_str(&format!("\nDiff:\n{diff}\n"));
                }
            } else if state_name == "Failed" || state.get("Failed").is_some() {
                out.push_str("State: Failed\n");
                if let Some(err) = details.and_then(|d| d.get("error")).and_then(|v| v.as_str()) {
                    out.push_str(&format!("Error: {err}\n"));
                }
            }
        }
        out.trim_end().to_string()
    }
}

pub fn format_steer(val: &serde_json::Value) -> String {
    let wid = val.get("worker_id").and_then(|v| v.as_str()).unwrap_or("");
    let msg = val
        .get("message")
        .and_then(|v| v.as_str())
        .unwrap_or("Steering instruction queued");
    format!("✓ Worker {wid}: {msg}")
}

pub fn format_kill(val: &serde_json::Value) -> String {
    let wid = val.get("worker_id").and_then(|v| v.as_str()).unwrap_or("");
    let killed = val.get("killed").and_then(|v| v.as_bool()).unwrap_or(false);
    if killed {
        format!("✓ Worker {wid} terminated.")
    } else {
        format!("Worker {wid} was not running.")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(s: &str) -> serde_json::Value {
        serde_json::from_str(s).expect("fixture must be valid JSON")
    }

    #[test]
    fn test_format_status_covers_string_tagged_and_object_states() {
        let bare = format_status(&v(r#"{"worker_id":"w","state":"Running"}"#));
        assert_eq!(bare, "Worker: w\nState: Running");

        let tagged = format_status(&v(
            r#"{"worker_id":"w","state":{"state":"Paused","details":{"turns":4,"step":2,"summary":"s","question":"q"}}}"#,
        ));
        assert_eq!(tagged, "Worker: w\nState: Paused\nTurns: 4\nStep: 2\nSummary: s\nQuestion: q");

        let keyed = format_status(&v(
            r#"{"worker_id":"w","state":{"Failed":{"turns":1,"error":"e"}}}"#,
        ));
        assert_eq!(keyed, "Worker: w\nState: Failed\nTurns: 1\nError: e");
    }

    #[test]
    fn test_format_collect_prefers_the_diff_and_keeps_counters() {
        let with_diff = format_collect(&v(
            r#"{"worker_id":"w","state":{"details":{"diff":"--- a\n+++ b"}},"total_steps":2}"#,
        ));
        assert!(with_diff.starts_with("--- a\n+++ b\n"));
        assert!(with_diff.ends_with("total_steps: 2"));

        let without = format_collect(&v(r#"{"worker_id":"w","diff":"  "}"#));
        assert_eq!(without, "Worker w: No git diff produced.\nno step logs");
    }

    #[test]
    fn test_format_logs_numbers_steps_and_surfaces_truncation() {
        let out = format_logs(&v(
            r#"{"worker_id":"w","logs":[{"step":1,"command":"ls"},{"step":2,"command":"pwd"}],
                 "total_steps":9,"logs_omitted":4,"logs_dropped":1}"#,
        ));
        assert_eq!(
            out,
            "Worker w step logs\n  [1] ls\n  [2] pwd\ntotal_steps: 9 | retained: 6 | omitted: 4 | dropped: 1"
        );
    }

    #[test]
    fn test_log_counters_line_appends_an_explicit_truncation_notice() {
        assert_eq!(
            log_counters_line(&v(r#"{"logs_truncation_notice":"head of the window was evicted"}"#)),
            "head of the window was evicted"
        );
    }

    #[test]
    fn test_format_reap_lists_ids_only_when_something_was_reaped() {
        assert_eq!(
            format_reap(&v(r#"{"reaped":0}"#)),
            "✓ No expired terminal worker records to reap."
        );
        assert_eq!(
            format_reap(&v(r#"{"reaped":2,"worker_ids":["a","b",7]}"#)),
            "✓ Reaped 2 expired worker record(s): a, b"
        );
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
        assert_eq!(failed, "✓ Worker w finished.\nState: Failed\nError: exploded");
    }

    #[test]
    fn test_format_kill_and_steer_reflect_the_tool_answer() {
        assert_eq!(
            format_kill(&v(r#"{"worker_id":"w","killed":true}"#)),
            "✓ Worker w terminated."
        );
        assert_eq!(
            format_kill(&v(r#"{"worker_id":"w"}"#)),
            "Worker w was not running."
        );
        assert_eq!(
            format_steer(&v(r#"{"worker_id":"w"}"#)),
            "✓ Worker w: Steering instruction queued"
        );
    }
}
