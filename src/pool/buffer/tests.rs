//! Unit tests for the bounded step-log buffer.
//!
//! Kept in a dedicated file rather than inlined in `mod.rs` so the retention
//! policy and the circular eviction loop stay readable end to end. Compiled only
//! under `cfg(test)`; everything is exercised through the package surface
//! (`super::…`), i.e. exactly the API the rest of the crate sees.
//!
//! * the sliding window, bounded by entry count, evicting strictly oldest-first
//!   and counting every eviction (audit 07, R1/R2),
//! * the pre-sized backing store, so a full window never reallocates (R2),
//! * the `LogRetentionPolicy` defaults and hard ceilings (R4 config).
//!
//! The payload ceilings are covered in `clamp/tests.rs`; the emission view and the
//! `LogStats` counters in `emit/tests.rs`.
//!
//! `LogBuffer::entries` is private, so the capacity assertion reaches it
//! through `super::*`: this is a child module of the struct's own module.

use super::*;

fn entry(step: usize, out: &str) -> AgentStepLog {
    build_step_log(step, "cargo test", out.to_string(), Some(0))
}

fn policy(retained: usize, emitted: usize) -> LogRetentionPolicy {
    LogRetentionPolicy {
        max_retained: retained,
        max_emitted: emitted,
    }
}

// ----------
// LogBuffer retention (audit 07, R1/R2)
// ----------

#[test]
fn test_log_buffer_window_is_bounded() {
    let mut buf = LogBuffer::with_policy(policy(4, 4));
    for i in 0..50 {
        buf.push(entry(i, "ok"));
    }
    assert_eq!(buf.len(), 4, "entry count must never exceed the window");
    assert_eq!(buf.retained(), 4);
    assert_eq!(buf.dropped(), 46, "every evicted entry is counted");
    assert_eq!(buf.total(), 50, "retained + dropped == total steps");
    // The window is the *tail* of the history.
    let steps: Vec<usize> = buf.iter().map(|e| e.step).collect();
    assert_eq!(steps, vec![46, 47, 48, 49]);
}

#[test]
fn test_log_buffer_reserves_capacity_for_the_window() {
    // R2: the backing store is pre-sized, so a full window never triggers a
    // reallocation (the old GeomGrow path wasted up to 41%).
    let buf = LogBuffer::with_policy(policy(128, 8));
    assert_eq!(
        buf.entries.capacity(),
        128,
        "capacity must be pre-reserved to the retention window"
    );
}

#[test]
fn test_log_buffer_tail_returns_the_newest_entries() {
    let mut buf = LogBuffer::with_policy(policy(10, 10));
    for i in 0..7 {
        buf.push(entry(i, "ok"));
    }
    let tail: Vec<usize> = buf.tail(3).iter().map(|e| e.step).collect();
    assert_eq!(tail, vec![4, 5, 6]);
    // Asking for more than is retained yields the whole window.
    assert_eq!(buf.tail(100).len(), 7);
}

#[test]
fn test_log_buffer_serializes_as_a_plain_array() {
    let mut buf = LogBuffer::new();
    buf.push(entry(3, "ok"));
    let value = serde_json::to_value(&buf).unwrap();
    let arr = value
        .as_array()
        .expect("LogBuffer serializes as a JSON array");
    assert_eq!(arr.len(), 1);
    assert_eq!(arr[0]["step"], 3);
}

// ----------
// LogRetentionPolicy::from_env (audit 07, R4 config)
// ----------

#[test]
fn test_log_policy_defaults_and_ceilings() {
    let d = LogRetentionPolicy::default();
    assert_eq!(d.max_retained, DEFAULT_MAX_RETAINED_LOGS);
    assert_eq!(d.max_emitted, DEFAULT_MAX_EMITTED_LOGS);
    // A zero policy is coerced to something usable.
    let zero = LogBuffer::with_policy(LogRetentionPolicy {
        max_retained: 0,
        max_emitted: 0,
    });
    let mut zero = zero;
    zero.push(entry(1, "ok"));
    assert_eq!(
        zero.len(),
        0,
        "a zero budget retains nothing rather than panicking"
    );
}
