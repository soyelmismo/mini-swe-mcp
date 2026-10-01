use super::*;

pub(super) fn v(s: &str) -> serde_json::Value {
    serde_json::from_str(s).expect("fixture must be valid JSON")
}

#[test]
fn test_log_counters_line_appends_an_explicit_truncation_notice() {
    assert_eq!(
        log_counters_line(&v(
            r#"{"logs_truncation_notice":"head of the window was evicted"}"#
        )),
        "head of the window was evicted"
    );
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
    assert_eq!(
        health_line(&v(r#"{"worker_id":"w","state":"Running"}"#)),
        None
    );
    // Metrics present but untouched: a worker killed before its first turn.
    assert_eq!(
        health_line(&v(r#"{"state":{"details":{"metrics":{}}}}"#)),
        None
    );
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
fn test_finished_views_render_the_next_step_guidance() {
    let next = "Review the diff (collect) and run the project's checks.";
    let payload = serde_json::json!({
        "worker_id": "w",
        "state": {"state": "Completed", "details": {"turns": 3, "summary": "done"}},
        "next_step": next,
    })
    .to_string();
    assert!(
        format_status(&v(&payload)).ends_with(&format!("Next step: {next}")),
        "status must end with the guidance: {}",
        format_status(&v(&payload))
    );
    assert!(
        format_collect(&v(&payload)).ends_with(&format!("Next step: {next}")),
        "collect must end with the guidance: {}",
        format_collect(&v(&payload))
    );
    assert!(
        format_dispatch(&v(&payload)).contains(&format!("Next step: {next}")),
        "dispatch --wait must carry the guidance: {}",
        format_dispatch(&v(&payload))
    );
    // A payload without guidance renders exactly as before.
    assert!(
        !format_status(&v(r#"{"worker_id":"w","state":"Running"}"#)).contains("Next step"),
        "a running worker has no next step"
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
