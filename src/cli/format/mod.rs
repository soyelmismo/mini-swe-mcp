//! Plain-text rendering of every `worker` tool payload the CLI prints.
//!
//! The MCP server answers in JSON; the CLI is human-facing, so every action has
//! a bespoke renderer here. They are pure functions over
//! [`serde_json::Value`] with no I/O, which is what makes them unit-testable
//! and keeps `main.rs` down to argument dispatch.
//!
//! [`format_output`] is the single entry point used by the binary: it maps an
//! action name to its renderer and falls back to pretty JSON for anything that
//! has no dedicated view. It lives here, next to the re-exports, because it is
//! the one piece of the package that spans both halves.
//!
//! The renderers themselves are split by what they describe:
//!
//! * `worker` — the per-worker inspection verbs: `status`, `collect`, `logs`,
//!   `dispatch`, `steer`, `kill`, `reap`, plus the shared `log_counters_line`
//!   helper that keeps step-log truncation visible (audit 07, R7) and the
//!   `health_line` that keeps a run's quality measurable.
//! * `catalog` — the system-catalog verbs: `manifest`, `list`, `prune`.
//!
//! Both submodules are private to the package and every formatter is
//! re-exported below, so the historical `cli::format::format_*` paths and the
//! binary's `cli::format::format_output` call keep working unchanged.

mod catalog;
mod worker;

pub use self::catalog::{format_list, format_manifest, format_prune};
pub use self::worker::{
    format_collect, format_dispatch, format_kill, format_logs, format_reap, format_status,
    format_steer, health_line, log_counters_line,
};

/// Render `val` for `action`, falling back to pretty JSON for actions with no
/// dedicated human-facing view.
pub fn format_output(action: &str, val: &serde_json::Value) -> String {
    match action {
        "manifest" => format_manifest(val),
        "list" => format_list(val),
        "prune" => format_prune(val),
        "status" => format_status(val),
        "collect" => format_collect(val),
        "logs" => format_logs(val),
        "reap" => format_reap(val),
        "dispatch" => format_dispatch(val),
        "steer" => format_steer(val),
        "kill" => format_kill(val),
        _ => serde_json::to_string_pretty(val).unwrap_or_default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(s: &str) -> serde_json::Value {
        serde_json::from_str(s).expect("fixture must be valid JSON")
    }

    #[test]
    fn test_format_output_dispatches_every_advertised_action() {
        let cases = [
            ("manifest", "Default model: x", r#"{"default_model":"x"}"#),
            ("list", "Workers (1):", r#"{"workers":[{"id":"w"}]}"#),
            ("prune", "✓", r#"{"message":"done"}"#),
            ("status", "Worker: w", r#"{"worker_id":"w"}"#),
            ("collect", "Worker w: No git diff produced.", r#"{"worker_id":"w"}"#),
            ("logs", "Worker w step logs", r#"{"worker_id":"w"}"#),
            ("reap", "✓ No expired", r#"{"reaped":0}"#),
            ("dispatch", "✓ Worker w finished.", r#"{"worker_id":"w"}"#),
            ("steer", "✓ Worker w:", r#"{"worker_id":"w","message":"go"}"#),
            ("kill", "Worker w was not running.", r#"{"worker_id":"w","killed":false}"#),
        ];
        for (action, needle, json) in cases {
            let out = format_output(action, &v(json));
            assert!(
                out.contains(needle),
                "{action} rendering missing {needle:?}: {out}"
            );
        }
    }

    #[test]
    fn test_format_output_falls_back_to_pretty_json() {
        let out = format_output("not-an-action", &v(r#"{"a":1}"#));
        assert_eq!(out, "{\n  \"a\": 1\n}");
    }

}
