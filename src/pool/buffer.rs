//! Bounded step-log retention for worker histories.
//!
//! Everything here exists to keep per-worker memory O(1) in the number of turns
//! and a single MCP response bounded, while making the degradation observable
//! (see `audits/opt_07_step_log_memory.md`).
//!
//! This module holds the retention window and its circular eviction:
//!
//! * the constants — per-field byte ceilings plus the env-overridable window
//!   and emission sizes, each with a hard ceiling so a misconfiguration cannot
//!   reintroduce the unbounded growth the audit flagged;
//! * [`LogRetentionPolicy`] — how much history is kept and how much of it is
//!   emitted;
//! * [`LogBuffer`] — the sliding window itself: a pre-sized `VecDeque` that
//!   evicts strictly oldest-first until both budgets hold, counting every
//!   eviction so the degradation is observable rather than silent.
//!
//! The two bounding steps around that window live in sibling modules:
//! [`clamp`] bounds how large a single entry can be ([`clamp_string`],
//! [`build_step_log`]) and [`emit`] bounds what goes on the wire
//! ([`EmittedLogs`], [`LogStats`], [`emit_view`]). Both re-export their items
//! here, so the `mini_swe_mcp::pool::*` surface is unchanged. Unit tests live
//! in [`tests`], [`clamp::tests`] and [`emit::tests`].

mod clamp;
mod emit;

#[cfg(test)]
mod tests;

pub use self::clamp::{build_step_log, clamp_string};
pub use self::emit::{emit_view, emit_view_with, EmittedLogs, LogStats};

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
