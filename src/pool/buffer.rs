//! Bounded step-log retention for worker histories.
//!
//! Keeps per-worker memory O(1) in turns and each MCP response bounded, while
//! making degradation observable (see `audits/opt_07_step_log_memory.md`).
//!
//! * constants — per-field byte ceilings plus env-overridable window and
//!   emission sizes, each with a hard ceiling so misconfiguration cannot
//!   reintroduce the unbounded growth the audit flagged;
//! * [`LogRetentionPolicy`] — how much history is kept and emitted;
//! * [`LogBuffer`] — a pre-sized `VecDeque` sliding window that evicts
//!   strictly oldest-first until the count budget holds, counting every
//!   eviction.
//!
//! Sibling modules bound the two steps around that window: [`clamp`] bounds a
//! single entry ([`clamp_string`], [`build_step_log`]) and [`emit`] bounds what
//! goes on the wire ([`EmittedLogs`], [`LogStats`], [`emit_view`]). Both
//! re-export here, so the `mini_swe_mcp::pool::*` surface is unchanged. Tests
//! live in [`tests`], [`clamp::tests`] and [`emit::tests`].

mod clamp;
mod emit;

#[cfg(test)]
mod tests;

pub use self::clamp::{build_step_log, clamp_string};
pub use self::emit::{EmittedLogs, LogStats, emit_view};

use serde::Serialize;
use std::collections::VecDeque;

use crate::agent::AgentStepLog;
use crate::config::env_parse;

// ----------
// Step-log retention policy (audit 07 — R1/R2/F1/F2/F6)
// ----------

/// Default step-log entries retained **per worker**.
///
/// Sliding window keeps memory per worker O(1) in turns, not O(n).
pub const DEFAULT_MAX_RETAINED_LOGS: usize = 200;
/// Ceiling for [`DEFAULT_MAX_RETAINED_LOGS`]; larger `WORKER_MAX_RETAINED_LOGS`
/// values clamp here so misconfiguration cannot reintroduce unbounded growth.
pub const MAX_RETAINED_LOGS_CEILING: usize = 1000;

/// Default step-log entries inlined into one MCP response.
pub const DEFAULT_MAX_EMITTED_LOGS: usize = 40;
/// Ceiling for [`DEFAULT_MAX_EMITTED_LOGS`].
pub const MAX_EMITTED_LOGS_CEILING: usize = 500;

/// Hard cap on the `output` field of a retained [`AgentStepLog`].
///
/// The truncation marker is *charged against* this budget, so the stored value
/// is `<= MAX_LOG_OUTPUT_BYTES`, not "budget + marker" (audit 07, F6).
pub const MAX_LOG_OUTPUT_BYTES: usize = 2048;

/// Hard cap on the `command` field of a retained [`AgentStepLog`].
pub const MAX_LOG_COMMAND_BYTES: usize = 64;

/// How a worker's step-log history is retained and how much is emitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LogRetentionPolicy {
    /// Maximum entries kept in the window.
    pub max_retained: usize,
    /// Maximum entries inlined into one MCP response.
    pub max_emitted: usize,
}

impl Default for LogRetentionPolicy {
    fn default() -> Self {
        Self {
            max_retained: DEFAULT_MAX_RETAINED_LOGS,
            max_emitted: DEFAULT_MAX_EMITTED_LOGS,
        }
    }
}

impl LogRetentionPolicy {
    /// Build a policy from the environment, clamping each value to its ceiling
    /// and falling back to defaults for zero / non-numeric input.
    pub fn from_env() -> Self {
        let default = Self::default();
        let max_retained = env_parse::<usize>("WORKER_MAX_RETAINED_LOGS")
            .filter(|&v| v > 0)
            .map(|v| v.min(MAX_RETAINED_LOGS_CEILING))
            .unwrap_or(default.max_retained);
        let max_emitted = env_parse::<usize>("WORKER_MAX_EMITTED_LOGS")
            .filter(|&v| v > 0)
            .map(|v| v.min(MAX_EMITTED_LOGS_CEILING))
            .unwrap_or(default.max_emitted);
        Self {
            max_retained,
            max_emitted,
        }
    }
}

/// A bounded, append-only-with-eviction step-log history.
///
/// # Invariants
///
/// * `entries.len() <= policy.max_retained` holds after every
///   [`LogBuffer::push`].
/// * Entries evict strictly oldest-first, so the retained window is the *tail*
///   of the worker's history.
/// * [`LogBuffer::dropped`] counts every eviction since creation, making the
///   degradation observable (audit 07, R7).
#[derive(Debug, Clone)]
pub struct LogBuffer {
    entries: VecDeque<AgentStepLog>,
    dropped: usize,
    max_retained: usize,
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
            dropped: 0,
            max_retained,
        }
    }

    /// Append an entry, evicting the oldest until the entry-count budget is
    /// satisfied again. Every entry is already clamped to its per-field
    /// ceiling by [`build_step_log`], so the count limit always binds first.
    pub fn push(&mut self, entry: AgentStepLog) {
        self.entries.push_back(entry);
        self.evict_until_within_budget();
    }

    /// Drop oldest entries while the window exceeds its count budget.
    fn evict_until_within_budget(&mut self) {
        while self.entries.len() > self.max_retained {
            self.entries.pop_front();
            self.dropped = self.dropped.saturating_add(1);
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Entries evicted because they fell out of the window.
    pub fn dropped(&self) -> usize {
        self.dropped
    }

    /// Entries currently held in memory.
    pub fn retained(&self) -> usize {
        self.entries.len()
    }

    /// Total steps ever logged, retained plus dropped.
    pub fn total(&self) -> usize {
        self.entries.len() + self.dropped
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
}

impl Default for LogBuffer {
    fn default() -> Self {
        Self::new()
    }
}

/// `LogBuffer` serializes as the plain retained array, so callers can embed a
/// snapshot where a `Vec<AgentStepLog>` used to be without reshaping the
/// payload. Eviction counters travel separately via [`LogStats`].
impl Serialize for LogBuffer {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.entries.serialize(serializer)
    }
}
