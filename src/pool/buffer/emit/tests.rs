//! Unit tests for the degradation-aware emission view.
//!
//! Compiled only under `cfg(test)`. These pin what a single MCP response is
//! allowed to carry: the emitted tail is capped by the emission budget,
//! `logs_omitted` matches the retained entries left behind, evicted entries are
//! reported separately from omitted ones, and the truncation notice is present
//! exactly when something is missing (audit 07, R4/R7).

use super::super::{
    build_step_log, LogRetentionPolicy, MAX_LOG_COMMAND_BYTES, MAX_LOG_OUTPUT_BYTES,
};
use super::*;

use crate::agent::AgentStepLog;

fn entry(step: usize, out: &str) -> AgentStepLog {
    build_step_log(step, "cargo test", out.to_string(), Some(0))
}

fn policy(retained: usize, emitted: usize) -> LogRetentionPolicy {
    LogRetentionPolicy {
        max_retained: retained,
        max_bytes: retained * (MAX_LOG_OUTPUT_BYTES + MAX_LOG_COMMAND_BYTES),
        max_emitted: emitted,
    }
}

// ----------
// LogStats (audit 07, R7)
// ----------

#[test]
fn test_log_stats_track_total_retained_and_dropped() {
    let mut buf = LogBuffer::with_policy(policy(2, 2));
    for i in 0..5 {
        buf.push(entry(i, "ok"));
    }
    let stats = LogStats {
        total_steps: buf.total(),
        logs_retained: buf.retained(),
        logs_dropped: buf.dropped(),
    };
    assert_eq!(stats.total_steps, 5);
    assert_eq!(stats.logs_retained, 2);
    assert_eq!(stats.logs_dropped, 3);
}

// ----------
// emit_view (audit 07, R4/R7)
// ----------

#[test]
fn test_emit_view_caps_the_payload_and_reports_what_is_missing() {
    let mut buf = LogBuffer::with_policy(policy(100, 10));
    for i in 0..60 {
        buf.push(entry(i, "ok"));
    }
    let view = emit_view(&buf, 10);
    assert_eq!(view.logs.len(), 10);
    assert_eq!(view.logs_omitted, 50);
    assert!(view.logs_truncation_notice.is_some());
    let notice = view.logs_truncation_notice.unwrap();
    assert!(
        notice.contains("50 retained log(s) omitted"),
        "got {notice}"
    );
    // The emitted tail is the newest 10.
    assert_eq!(view.logs.first().unwrap().step, 50);
    assert_eq!(view.logs.last().unwrap().step, 59);
}

#[test]
fn test_emit_view_reports_evicted_entries_separately() {
    let mut buf = LogBuffer::with_policy(policy(5, 5));
    for i in 0..20 {
        buf.push(entry(i, "ok"));
    }
    let view = emit_view(&buf, 5);
    assert_eq!(view.logs_omitted, 0, "nothing retained was omitted");
    let notice = view
        .logs_truncation_notice
        .expect("eviction must be visible");
    assert!(notice.contains("15 earlier log(s) evicted"), "got {notice}");
}

#[test]
fn test_emit_view_is_quiet_when_nothing_is_missing() {
    let mut buf = LogBuffer::new();
    buf.push(entry(1, "ok"));
    let view = emit_view(&buf, 40);
    assert_eq!(view.logs.len(), 1);
    assert_eq!(view.logs_omitted, 0);
    assert!(view.logs_truncation_notice.is_none());
    // ...and the notice is omitted from JSON entirely.
    let value = serde_json::to_value(&view).unwrap();
    assert!(value.get("logs_truncation_notice").is_none());
}
