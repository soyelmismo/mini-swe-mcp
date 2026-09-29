//! Unit tests for the per-entry payload clamping.
//!
//! Compiled only under `cfg(test)`. These lock the audit-07 / F6 guarantee: a
//! stored field is at most its budget *including* the truncation marker, no UTF-8
//! code point is ever split, and a budget too small for the marker degrades to
//! an empty string rather than blowing the ceiling.

use super::*;

// ----------
// clamp_string / build_step_log (audit 07, F6)
// ----------

#[test]
fn test_clamp_string_leaves_short_values_untouched() {
    assert_eq!(clamp_string("hello", 64), "hello");
    assert_eq!(clamp_string("", 64), "");
}

#[test]
fn test_clamp_string_charges_the_marker_against_the_budget() {
    // F6: the result is at most `budget` bytes -- the marker included.
    let long = "a".repeat(10_000);
    let clamped = clamp_string(&long, MAX_LOG_OUTPUT_BYTES);
    assert!(
        clamped.len() <= MAX_LOG_OUTPUT_BYTES,
        "clamped output was {} bytes",
        clamped.len()
    );
    assert!(clamped.contains("bytes truncated"));
    assert!(clamped.starts_with("aaaa"));
}

#[test]
fn test_clamp_string_never_splits_a_code_point() {
    // '€' is 3 bytes: a 2047-byte budget must back off to a boundary.
    let mut s = "a".repeat(2045);
    s.push('€');
    s.push_str(&"b".repeat(1000));
    let clamped = clamp_string(&s, MAX_LOG_OUTPUT_BYTES);
    assert!(clamped.len() <= MAX_LOG_OUTPUT_BYTES);
    // The visible head is valid UTF-8 (it came from a str slice).
    assert!(clamped.contains('a'));
}

#[test]
fn test_clamp_string_degrades_when_the_marker_does_not_fit() {
    // A budget smaller than the marker itself must not blow the ceiling.
    let clamped = clamp_string(&"x".repeat(100), 5);
    assert!(clamped.len() <= 5, "got {} bytes", clamped.len());
}

#[test]
fn test_build_step_log_clamps_both_text_fields() {
    let log = build_step_log(7, &"c".repeat(500), "o".repeat(50_000), Some(3));
    assert!(log.command.len() <= MAX_LOG_COMMAND_BYTES);
    assert!(log.output.len() <= MAX_LOG_OUTPUT_BYTES);
    assert_eq!(log.step, 7);
    assert_eq!(log.exit_code, Some(3));
}
