use super::*;

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
    format!("✓ Worker {wid}: {msg}{}", watch_command_line(val))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::format::worker::tests::v;
    /// A steer answer carries the `watch_command` only while the caller has no
    /// watch of its own running: the hub drops the field otherwise, and the
    /// render must then omit the line.
    #[test]
    fn test_format_steer_shows_the_watch_command_only_when_the_hub_minted_one() {
        let with_token = format_steer(&v(
            r#"{"worker_id":"w","status":"steered","message":"queued","watch_command":"MINI_SWE_WATCH_TOKEN=abc mini-swe-mcp watch"}"#,
        ));
        assert!(
            with_token.contains("\nTo wait for it: MINI_SWE_WATCH_TOKEN=abc mini-swe-mcp watch")
                && with_token
                    .contains("run it in the background as-is; run it again after each event"),
            "{with_token}"
        );
        let without = format_steer(&v(
            r#"{"worker_id":"w","status":"steered","message":"queued"}"#,
        ));
        assert!(!without.contains("To wait for it"), "{without}");
    }

    /// `steer --wait` answers with the awaited result, so it must be rendered
    /// by the worker-result view rather than as a bare acknowledgement.
    #[test]
    fn test_format_steer_renders_an_awaited_result_as_the_worker_result() {
        let queued = format_steer(&v(
            r#"{"worker_id":"w","status":"steered","message":"queued"}"#,
        ));
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
}
