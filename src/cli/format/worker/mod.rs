//! Plain-text renderers for the per-worker inspection verbs.
//!
//! Formatters behind `status`, `collect`, `logs`, `dispatch`, `consolidate`,
//! `steer`, `wait`, `kill` and `reap` — the actions that answer about one worker (or, for
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

mod collect;
mod consolidate;
mod discard;
mod dispatch;
mod kill;
mod logs;
mod merge;
mod prune_reap;
mod review;
mod status;
mod steer;

pub use collect::format_collect;
pub use consolidate::format_consolidate;
pub use discard::format_discard;
pub use dispatch::{format_dispatch, format_dispatch_quiet};
pub use kill::format_kill;
pub use logs::format_logs;
pub use merge::format_merge;
pub use prune_reap::format_reap;
pub use review::format_review;
pub use status::format_status;
pub use steer::format_steer;

#[cfg(test)]
mod tests;

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
        .find_map(|tag| {
            state
                .and_then(|s| s.get(tag))
                .and_then(|d| d.get("metrics"))
        })
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
    let (turns, granted, refused) = (
        count("turns_used"),
        count("extensions_granted"),
        count("extensions_refused"),
    );
    let (repeats, nudges, pauses) = (
        count("repeat_blocks"),
        count("stagnation_nudges"),
        count("loop_pauses"),
    );
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

    let plural = |n: u64, word: &str| format!("{n} {word}{}", if n == 1 { "" } else { "s" });
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

/// True when `val` is an awaited worker result rather than the plain
/// acknowledgement of a verb that queued something.
fn is_awaited_result(val: &serde_json::Value) -> bool {
    val.get("state").is_some()
        || val.get("status").and_then(|v| v.as_str()) == Some("still_running")
}

/// The `watch_command` a dispatch or steer answer carried, as one line to run.
///
/// A shell cannot know its session, so the token in that command is what binds
/// it to the agent that dispatched: without it a `mini-swe-mcp watch` would
/// fall back to the bare host and miss the worker. Empty when the hub minted no
/// token, which is the in-process server's case.
fn watch_command_line(val: &serde_json::Value) -> String {
    match val.get("watch_command").and_then(|v| v.as_str()) {
        Some(command) => format!(
            "\nTo wait for it: {command} (run it in the background as-is; run it again after each event)"
        ),
        None => String::new(),
    }
}
