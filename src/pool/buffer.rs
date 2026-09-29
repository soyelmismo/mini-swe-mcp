//! Bounded step-log retention and emission for worker histories.
//!
//! Everything here exists to keep per-worker memory O(1) in the number of turns
//! and a single MCP response bounded, while making the degradation observable
//! (see `audits/opt_07_step_log_memory.md`).

use serde::Serialize;
use std::collections::VecDeque;

use crate::agent::AgentStepLog;

// ---------------------------------------------------------------------------
// Step-log retention policy (audit 07 — R1/R2/F1/F2/F6)
// ---------------------------------------------------------------------------

/// Default number of step-log entries retained **per worker**.
///
/// The buffer is a sliding window, so memory per worker is O(1) in the number
/// of turns instead of O(n).
pub const DEFAULT_MAX_RETAINED_LOGS: usize = 200;
/// Hard ceiling for [`DEFAULT_MAX_RETAINED_LOGS`]; larger `WORKER_MAX_RETAINED_LOGS`
/// values are clamped to this so a misconfiguration cannot reintroduce the
/// unbounded growth the audit flagged.
pub const MAX_RETAINED_LOGS_CEILING: usize = 1000;

/// Default number of step-log entries inlined into a single MCP response.
pub const DEFAULT_MAX_EMITTED_LOGS: usize = 40;
/// Hard ceiling for [`DEFAULT_MAX_EMITTED_LOGS`].
pub const MAX_EMITTED_LOGS_CEILING: usize = 500;

/// Hard cap on the `output` field of a retained [`AgentStepLog`].
///
/// Unlike the previous implementation the truncation marker is *charged
/// against* this budget, so the stored value is `<= MAX_LOG_OUTPUT_BYTES`
/// rather than "budget + marker" (audit 07, F6).
pub const MAX_LOG_OUTPUT_BYTES: usize = 2048;

/// Hard cap on the `command` field of a retained [`AgentStepLog`].
pub const MAX_LOG_COMMAND_BYTES: usize = 64;

/// Worst-case charged cost of one retained entry: the two text fields plus the
/// inline `AgentStepLog` struct. Used to derive the per-worker byte budget from
/// the entry-count window.
pub fn worst_case_entry_bytes() -> usize {
    std::mem::size_of::<AgentStepLog>() + MAX_LOG_COMMAND_BYTES + MAX_LOG_OUTPUT_BYTES
}

/// How a worker's step-log history is retained and how much of it is emitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LogRetentionPolicy {
    /// Maximum number of entries kept in the window.
    pub max_retained: usize,
    /// Total byte budget for the retained payloads (both fields combined).
    pub max_bytes: usize,
    /// Maximum number of entries inlined into one MCP response.
    pub max_emitted: usize,
}

impl Default for LogRetentionPolicy {
    fn default() -> Self {
        Self {
            max_retained: DEFAULT_MAX_RETAINED_LOGS,
            max_bytes: DEFAULT_MAX_RETAINED_LOGS * worst_case_entry_bytes(),
            max_emitted: DEFAULT_MAX_EMITTED_LOGS,
        }
    }
}

impl LogRetentionPolicy {
    /// Build a policy from the environment, clamping every value to its ceiling
    /// and falling back to the defaults for zero / non-numeric input.
    pub fn from_env() -> Self {
        let default = Self::default();
        let max_retained = std::env::var("WORKER_MAX_RETAINED_LOGS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&v| v > 0)
            .map(|v| v.min(MAX_RETAINED_LOGS_CEILING))
            .unwrap_or(default.max_retained);
        let max_emitted = std::env::var("WORKER_MAX_EMITTED_LOGS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&v| v > 0)
            .map(|v| v.min(MAX_EMITTED_LOGS_CEILING))
            .unwrap_or(default.max_emitted);
        Self {
            max_retained,
            max_bytes: max_retained * worst_case_entry_bytes(),
            max_emitted,
        }
    }
}

/// A bounded, append-only-with-eviction step-log history.
///
/// # Invariants
///
/// * `entries.len() <= policy.max_retained` and `bytes <= policy.max_bytes`
///   hold after every [`LogBuffer::push`].
/// * Entries are evicted strictly oldest-first, so the retained window is the
///   *tail* of the worker's history.
/// * [`LogBuffer::dropped`] counts every entry evicted since the buffer was
///   created, which is what makes the degradation observable (audit 07, R7).
#[derive(Debug, Clone)]
pub struct LogBuffer {
    entries: VecDeque<AgentStepLog>,
    bytes: usize,
    dropped: usize,
    max_retained: usize,
    max_bytes: usize,
}

impl LogBuffer {
    /// A buffer with the default policy, pre-allocating the exact window so the
    /// `Vec` never over-allocates (audit 07, R2).
    pub fn new() -> Self {
        Self::with_policy(LogRetentionPolicy::default())
    }

    pub fn with_policy(policy: LogRetentionPolicy) -> Self {
        let max_retained = policy.max_retained.max(1);
        Self {
            entries: VecDeque::with_capacity(max_retained.min(MAX_RETAINED_LOGS_CEILING)),
            bytes: 0,
            dropped: 0,
            max_retained,
            max_bytes: policy.max_bytes.max(1),
        }
    }

    /// Append an entry, evicting the oldest ones until both the entry-count and
    /// byte budgets are satisfied again.
    pub fn push(&mut self, entry: AgentStepLog) {
        self.bytes += entry_size(&entry);
        self.entries.push_back(entry);
        self.evict_until_within_budget();
    }

    /// Drop oldest entries while the window exceeds its count budget, the byte
    /// budget, or both. A single entry larger than the byte budget is still
    /// evicted: the window must never keep data it cannot account for.
    fn evict_until_within_budget(&mut self) {
        while !self.entries.is_empty()
            && (self.entries.len() > self.max_retained || self.bytes > self.max_bytes)
        {
            if let Some(oldest) = self.entries.pop_front() {
                self.bytes = self.bytes.saturating_sub(entry_size(&oldest));
                self.dropped = self.dropped.saturating_add(1);
            }
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Number of entries evicted because they fell out of the window.
    pub fn dropped(&self) -> usize {
        self.dropped
    }

    /// Number of entries currently held in memory.
    pub fn retained(&self) -> usize {
        self.entries.len()
    }

    /// Total number of steps ever logged, retained plus dropped.
    pub fn total(&self) -> usize {
        self.entries.len() + self.dropped
    }

    /// Bytes currently charged against the retention budget.
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// Retained entries that a `max_emitted` budget would leave out.
    pub fn logs_omitted(&self, max_emitted: usize) -> usize {
        self.entries.len().saturating_sub(max_emitted)
    }

    /// Oldest retained entry, if any.
    pub fn front(&self) -> Option<&AgentStepLog> {
        self.entries.front()
    }

    /// Newest retained entry, if any.
    pub fn back(&self) -> Option<&AgentStepLog> {
        self.entries.back()
    }

    /// Iterate the retained window oldest-first.
    pub fn iter(&self) -> impl Iterator<Item = &AgentStepLog> {
        self.entries.iter()
    }

    /// Borrow the newest `limit` entries (the tail of the window), oldest-first.
    pub fn tail(&self, limit: usize) -> Vec<&AgentStepLog> {
        let skip = self.entries.len().saturating_sub(limit);
        self.entries.iter().skip(skip).collect()
    }

    /// Drop every retained entry, keeping the `dropped` counter.
    pub fn clear(&mut self) {
        self.bytes = 0;
        self.entries.clear();
    }
}

impl Default for LogBuffer {
    fn default() -> Self {
        Self::new()
    }
}

/// `LogBuffer` serializes as the plain retained array, so callers can embed a
/// snapshot where a `Vec<AgentStepLog>` used to be without reshaping the payload.
/// The eviction counters travel separately via [`LogStats`].
impl Serialize for LogBuffer {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.entries.serialize(serializer)
    }
}

/// Bytes charged for one entry: payload plus the inline `AgentStepLog` struct.
fn entry_size(entry: &AgentStepLog) -> usize {
    std::mem::size_of::<AgentStepLog>() + entry.command.len() + entry.output.len()
}

/// Truncate `value` so the *whole result* — truncation marker included — is at
/// most `budget` bytes, never splitting a UTF-8 code point (audit 07, F6).
///
/// The returned string is always valid UTF-8; when the marker alone would not
/// fit inside the budget the result degrades to an empty string rather than
/// exceeding the ceiling.
pub fn clamp_string(value: &str, budget: usize) -> String {
    if value.len() <= budget {
        return value.to_string();
    }

    // The marker length depends on the number of dropped bytes, so reserve room
    // for the widest plausible marker first and shrink the head until it fits.
    let mut dropped = value.len();
    loop {
        let marker = truncation_marker(dropped);
        if marker.len() >= budget {
            return String::new();
        }
        let head_budget = budget - marker.len();
        let cut = value.floor_char_boundary(head_budget);
        let out = format!("{}{}", &value[..cut], marker);
        if out.len() <= budget {
            return out;
        }
        // `cut` moved past a code point start; recompute with the real drop count.
        dropped = value.len() - cut;
    }
}

/// `... [N bytes truncated]` — the marker appended by [`clamp_string`].
fn truncation_marker(dropped: usize) -> String {
    format!("... [{dropped} bytes truncated]")
}

/// Build a bounded [`AgentStepLog`] entry: both text fields are clamped so the
/// retained payload is strictly bounded (audit 07, F6).
pub fn build_step_log(
    step: usize,
    command: &str,
    output: String,
    exit_code: Option<i32>,
) -> AgentStepLog {
    AgentStepLog {
        step,
        command: clamp_string(command, MAX_LOG_COMMAND_BYTES),
        output: clamp_string(&output, MAX_LOG_OUTPUT_BYTES),
        exit_code,
    }
}

/// Observability counters describing how much of a worker's step history is
/// actually visible (audit 07, R7).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize)]
pub struct LogStats {
    /// Total steps the worker executed.
    pub total_steps: usize,
    /// Entries currently held in memory.
    pub logs_retained: usize,
    /// Entries evicted by the retention window.
    pub logs_dropped: usize,
}

/// A rendered, degradation-aware view of a worker's step history.
#[derive(Debug, Clone, serde::Serialize)]
pub struct EmittedLogs {
    /// The emitted tail, oldest-first.
    pub logs: Vec<AgentStepLog>,
    /// How many retained entries were **not** emitted because of the emission
    /// budget.
    pub logs_omitted: usize,
    /// Present only when something is missing, so a consumer can never mistake
    /// the emitted tail for the full history.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logs_truncation_notice: Option<String>,
}

/// Render the tail of a log buffer for one response, bounded by
/// `WORKER_MAX_EMITTED_LOGS` (audit 07, R4).
pub fn emit_view(buffer: &LogBuffer, max_emitted: usize) -> EmittedLogs {
    emit_view_with(&buffer.tail(max_emitted), buffer.dropped(), buffer.len())
}

/// Assemble an [`EmittedLogs`] from a tail that is already materialised.
pub fn emit_view_with(tail: &[&AgentStepLog], dropped: usize, retained: usize) -> EmittedLogs {
    let omitted = retained.saturating_sub(tail.len());
    let mut notice_parts: Vec<String> = Vec::new();
    if dropped > 0 {
        notice_parts.push(format!(
            "{dropped} earlier log(s) evicted by the retention window"
        ));
    }
    if omitted > 0 {
        notice_parts.push(format!(
            "{omitted} retained log(s) omitted by the emission budget"
        ));
    }
    let logs_truncation_notice = if notice_parts.is_empty() {
        None
    } else {
        Some(format!(
            "{} (use `logs <worker_id>` for the full window)",
            notice_parts.join("; ")
        ))
    };

    EmittedLogs {
        logs: tail.iter().map(|e| (*e).clone()).collect(),
        logs_omitted: omitted,
        logs_truncation_notice,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ----------
    // LogBuffer retention (audit 07, R1/R2)
    // ----------

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
    fn test_log_buffer_byte_budget_evicts_even_under_the_count_cap() {
        // A tiny byte budget with a generous count budget: the byte ceiling has
        // to win, otherwise the payload is still unbounded.
        let mut buf = LogBuffer::with_policy(LogRetentionPolicy {
            max_retained: 1000,
            max_bytes: 4 * 1024,
            max_emitted: 8,
        });
        for i in 0..20 {
            buf.push(entry(i, &"x".repeat(2048)));
        }
        assert!(
            buf.bytes() <= 4 * 1024,
            "byte budget exceeded: {}",
            buf.bytes()
        );
        assert!(
            buf.len() <= 2,
            "expected byte-driven eviction, got {}",
            buf.len()
        );
        assert!(buf.dropped() > 0);
    }

    #[test]
    fn test_log_buffer_empty_and_clear() {
        let mut buf = LogBuffer::new();
        assert!(buf.is_empty());
        assert!(buf.front().is_none());
        assert!(buf.back().is_none());
        buf.push(entry(1, "a"));
        assert!(buf.front().is_some());
        assert!(buf.back().is_some());
        buf.clear();
        assert!(buf.is_empty());
        assert_eq!(buf.bytes(), 0);
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

    // ----------
    // LogRetentionPolicy::from_env (audit 07, R4 config)
    // ----------

    #[test]
    fn test_log_policy_defaults_and_ceilings() {
        let d = LogRetentionPolicy::default();
        assert_eq!(d.max_retained, DEFAULT_MAX_RETAINED_LOGS);
        assert_eq!(d.max_emitted, DEFAULT_MAX_EMITTED_LOGS);
        assert_eq!(
            d.max_bytes,
            DEFAULT_MAX_RETAINED_LOGS * worst_case_entry_bytes()
        );
        // The budget must actually cover a full window of worst-case entries.
        assert!(d.max_bytes >= d.max_retained * worst_case_entry_bytes());
        // A zero policy is coerced to something usable.
        let zero = LogBuffer::with_policy(LogRetentionPolicy {
            max_retained: 0,
            max_bytes: 0,
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
}
