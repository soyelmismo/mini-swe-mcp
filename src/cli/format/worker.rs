//! Plain-text renderers for the per-worker inspection verbs.
//!
//! Formatters behind `status`, `collect`, `logs`, `dispatch`, `steer`, `wait`,
//! `kill` and `reap` — the actions that answer about one worker (or, for
//! `reap`, about a set of terminal workers) rather than about the system
//! catalog. `format_dispatch` doubles as the worker-result view, because
//! `wait` and `steer --wait` answer with exactly that payload.
//! [`log_counters_line`] is the shared counter/notice line that `collect` and
//! `logs` both append so step-log truncation is never silent (audit 07, R7), and
//! [`health_line`] is the per-worker health line `status`, `collect` and
//! `dispatch --wait` append so a run can be graded, not just read.
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

/// The `metrics` object a payload carries, wherever it sits in the shape.
///
/// `status`, `collect` and `dispatch --wait` all report the worker's
/// `WorkerState`, whose metrics live inside the enum's `details`; the keyed
/// form (`{"Completed": {...}}`) used by a few payloads nests them one level
/// deeper, and a flattened payload carries them at the top.
fn metrics_of(val: &serde_json::Value) -> Option<&serde_json::Value> {
    let state = val.get("state");
    if let Some(metrics) = state.and_then(|s| s.get("metrics")) {
        return Some(metrics);
    }
    if let Some(metrics) = state
        .and_then(|s| s.get("details"))
        .and_then(|d| d.get("metrics"))
    {
        return Some(metrics);
    }
    if let Some(metrics) = ["Completed", "Failed", "Running", "Paused"]
        .iter()
        .find_map(|tag| state.and_then(|s| s.get(tag)).and_then(|d| d.get("metrics")))
    {
        return Some(metrics);
    }
    val.get("metrics")
}

/// One compact line of per-worker health, or `None` when there is nothing to
/// report.
///
/// Omitted for a payload with no metrics and for an all-zero set, which is what
/// a registry row written before the counters existed carries: a view must not
/// dress an unmeasured run up as a healthy one.
pub fn health_line(val: &serde_json::Value) -> Option<String> {
    let metrics = metrics_of(val)?;
    let count = |key: &str| metrics.get(key).and_then(|v| v.as_u64()).unwrap_or(0);
    let (turns, granted, refused) = (count("turns_used"), count("extensions_granted"), count("extensions_refused"));
    let (repeats, nudges, pauses) = (count("repeat_blocks"), count("stagnation_nudges"), count("loop_pauses"));
    let (verify_runs, verify_failures) = (count("verify_runs"), count("verify_failures"));
    let (files, insertions, deletions) = (
        count("diff_files"),
        count("diff_insertions"),
        count("diff_deletions"),
    );
    if turns == 0
        && granted == 0
        && refused == 0
        && repeats == 0
        && nudges == 0
        && pauses == 0
        && verify_runs == 0
        && files == 0
        && insertions == 0
        && deletions == 0
    {
        return None;
    }

    let plural = |n: u64, word: &str| {
        format!("{n} {word}{}", if n == 1 { "" } else { "s" })
    };
    let mut parts = vec![
        plural(turns, "turn"),
        format!("+{granted}/-{refused} ext"),
        plural(repeats, "repeat"),
        plural(nudges, "nudge"),
    ];
    if pauses > 0 {
        parts.push(plural(pauses, "loop pause"));
    }
    if verify_runs > 0 {
        parts.push(format!("verify {verify_failures}/{verify_runs} failed"));
    }
    if files > 0 {
        parts.push(format!(
            "diff {} +{insertions}/-{deletions}",
            plural(files, "file")
        ));
    }
    Some(format!("Health: {}", parts.join(", ")))
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
                    push_verified_line(&mut out, d.get("verified"));
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
                    push_verified_line(&mut out, d.get("verified"));
                }
            }
        }
    }
    if let Some(health) = health_line(val) {
        out.push_str(&health);
        out.push('\n');
    }
    out.trim_end().to_string()
}

/// Append the verification outcome when the worker reports one.
///
/// A worker that exhausted its verification budget completes flagged
/// unverified, so the plain-text status must not present it as a clean pass.
fn push_verified_line(out: &mut String, verified: Option<&serde_json::Value>) {
    match verified {
        Some(serde_json::Value::Bool(true)) => out.push_str("Verified: yes\n"),
        Some(serde_json::Value::Bool(false)) => {
            out.push_str("Verified: no (completed with failing verification)\n");
        }
        _ => {}
    }
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
    let health = health_line(val).map(|h| format!("\n{h}")).unwrap_or_default();
    if diff.trim().is_empty() {
        format!("Worker {wid}: No git diff produced.\n{counters}{health}")
    } else {
        format!("{diff}\n{counters}{health}")
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

/// True when `val` is an awaited worker result rather than the plain
/// acknowledgement of a verb that queued something.
fn is_awaited_result(val: &serde_json::Value) -> bool {
    val.get("state").is_some()
        || val.get("status").and_then(|v| v.as_str()) == Some("still_running")
}

pub fn format_dispatch(val: &serde_json::Value) -> String {
    let wid = val.get("worker_id").and_then(|v| v.as_str()).unwrap_or("");
    if val.get("status").and_then(|v| v.as_str()) == Some("dispatched") {
        format!("✓ Worker {wid} dispatched in background.\nUse 'mini-swe-mcp status {wid}' to check progress.")
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
                if let Some(turns) = details.and_then(|d| d.get("turns")).and_then(|v| v.as_u64()) {
                    out.push_str(&format!("Turns: {turns}\n"));
                }
                if let Some(summary) = details.and_then(|d| d.get("summary")).and_then(|v| v.as_str()) {
                    out.push_str(&format!("Summary: {summary}\n"));
                }
                if let Some(d) = details {
                    push_verified_line(&mut out, d.get("verified"));
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
                if let Some(health) = &health {
                    out.push_str(&format!("{health}\n"));
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
                if let Some(health) = &health {
                    out.push_str(&format!("{health}\n"));
                }
            }
        }
        out.trim_end().to_string()
    }
}

/// `steer` action. A `steer --wait` answers with the awaited worker result
/// (the same payload `wait` returns), which is rendered by the shared
/// worker-result view instead of the one-line acknowledgement.
pub fn format_steer(val: &serde_json::Value) -> String {
    if is_awaited_result(val) {
        return format_dispatch(val);
    }
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

    /// `steer --wait` answers with the awaited result, so it must be rendered
    /// by the worker-result view rather than as a bare acknowledgement.
    #[test]
    fn test_format_steer_renders_an_awaited_result_as_the_worker_result() {
        let queued = format_steer(&v(r#"{"worker_id":"w","status":"steered","message":"queued"}"#));
        assert_eq!(queued, "✓ Worker w: queued");

        let awaited = format_steer(&v(
            r#"{"worker_id":"w","state":{"state":"Completed","details":{"turns":3,"summary":"s"}}}"#,
        ));
        assert!(
            awaited.contains("✓ Worker w finished.") && awaited.contains("Turns: 3"),
            "a waited-on steer must read like the dispatch result: {awaited}"
        );

        let bounded = format_steer(&v(
            r#"{"worker_id":"w","status":"still_running","step":4,"last_command":"cargo test"}"#,
        ));
        assert!(
            bounded.contains("Worker w is still running (step 4): cargo test")
                && bounded.contains("wait w"),
            "an expired deadline must not read as a finished worker: {bounded}"
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

    /// A worker that exhausted its verification budget must not be presented
    /// as a clean pass by the plain-text views.
    #[test]
    fn test_completed_views_surface_the_verification_outcome() {
        let verified = format_status(&v(
            r#"{"worker_id":"w","state":{"state":"Completed","details":{"turns":3,"verified":true}}}"#,
        ));
        assert!(verified.contains("Verified: yes"), "{verified}");

        let unverified = format_status(&v(
            r#"{"worker_id":"w","state":{"Completed":{"turns":3,"verified":false}}}"#,
        ));
        assert!(
            unverified.contains("Verified: no (completed with failing verification)"),
            "{unverified}"
        );

        let dispatched = format_dispatch(&v(
            r#"{"worker_id":"w","state":{"state":"Completed","details":{"turns":3,"verified":false}}}"#,
        ));
        assert!(
            dispatched.contains("Verified: no (completed with failing verification)"),
            "{dispatched}"
        );

        // An absent field must add nothing, so pre-existing payloads render
        // exactly as before.
        let unflagged = format_status(&v(
            r#"{"worker_id":"w","state":{"state":"Completed","details":{"turns":3}}}"#,
        ));
        assert!(!unflagged.contains("Verified"), "{unflagged}");
    }

    #[test]
    fn test_health_line_renders_every_measured_counter() {
        let line = health_line(&v(
            r#"{"worker_id":"w","state":{"state":"Completed","details":{"metrics":{
                 "turns_used":142,"extensions_granted":0,"extensions_refused":0,
                 "repeat_blocks":3,"stagnation_nudges":1,"loop_pauses":0,
                 "verify_runs":2,"verify_failures":1,
                 "diff_files":5,"diff_insertions":120,"diff_deletions":340}}}}"#,
        ))
        .expect("a measured run renders a health line");
        assert_eq!(
            line,
            "Health: 142 turns, +0/-0 ext, 3 repeats, 1 nudge, verify 1/2 failed, diff 5 files +120/-340"
        );
    }

    #[test]
    fn test_health_line_reports_the_rare_counters_only_when_they_happened() {
        let clean = health_line(&v(
            r#"{"state":{"details":{"metrics":{"turns_used":7,"diff_files":1,
                 "diff_insertions":4,"diff_deletions":0}}}}"#,
        ))
        .expect("a measured run renders a health line");
        assert_eq!(
            clean,
            "Health: 7 turns, +0/-0 ext, 0 repeats, 0 nudges, diff 1 file +4/-0"
        );

        let looping = health_line(&v(
            r#"{"state":{"Failed":{"metrics":{"turns_used":9,"extensions_granted":3,
                 "extensions_refused":1,"repeat_blocks":3,"loop_pauses":1,
                 "verify_runs":3,"verify_failures":3}}}}"#,
        ))
        .expect("a measured run renders a health line");
        assert_eq!(
            looping,
            "Health: 9 turns, +3/-1 ext, 3 repeats, 0 nudges, 1 loop pause, verify 3/3 failed"
        );
    }

    #[test]
    fn test_health_line_is_omitted_when_nothing_was_measured() {
        // No metrics at all: a payload from a build that did not record them.
        assert_eq!(health_line(&v(r#"{"worker_id":"w","state":"Running"}"#)), None);
        // Metrics present but untouched: a worker killed before its first turn.
        assert_eq!(health_line(&v(r#"{"state":{"details":{"metrics":{}}}}"#)), None);
        assert_eq!(
            health_line(&v(
                r#"{"state":{"state":"Failed","details":{"metrics":{"turns_used":0,"repeat_blocks":0}}}}"#
            )),
            None
        );
    }

    #[test]
    fn test_the_worker_views_append_the_health_line() {
        let payload = r#"{"worker_id":"w","state":{"state":"Completed","details":{
                       "turns":3,"summary":"done","diff":"--- a","total_steps":3,
                       "metrics":{"turns_used":3,"repeat_blocks":1}}}}"#;
        assert!(
            format_status(&v(payload)).ends_with("Health: 3 turns, +0/-0 ext, 1 repeat, 0 nudges"),
            "status must end with the health line: {}",
            format_status(&v(payload))
        );
        assert!(
            format_collect(&v(payload)).ends_with("Health: 3 turns, +0/-0 ext, 1 repeat, 0 nudges"),
            "collect must end with the health line: {}",
            format_collect(&v(payload))
        );
        let dispatched = format_dispatch(&v(payload));
        assert!(
            dispatched.contains("\nHealth: 3 turns, +0/-0 ext, 1 repeat, 0 nudges\n\nDiff:\n"),
            "dispatch --wait must report health before the diff: {dispatched}"
        );
        // A failed worker reports what it measured too.
        let failed = format_dispatch(&v(
            r#"{"worker_id":"w","state":{"Failed":{"error":"boom","metrics":{"turns_used":4}}}}"#,
        ));
        assert!(
            failed.ends_with("Health: 4 turns, +0/-0 ext, 0 repeats, 0 nudges"),
            "{failed}"
        );
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
