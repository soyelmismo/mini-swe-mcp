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
