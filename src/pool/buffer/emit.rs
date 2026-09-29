//! The degradation-aware emission view over a retained step-log window.
//!
//! The buffer bounds what is *held*; this module bounds what is *put on the
//! wire*. A response inlines only the newest `max_emitted` entries and carries
//! an explicit [`EmittedLogs::logs_truncation_notice`] whenever anything is
//! missing, so a consumer can never mistake the emitted tail for the full
//! history (audit 07, R4/R7). [`LogStats`] is the same idea as counters.

#[cfg(test)]
mod tests;

use crate::agent::AgentStepLog;

use super::LogBuffer;

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
    /// Retained entries **not** emitted because of the emission budget.
    pub logs_omitted: usize,
    /// Present only when something is missing, so a consumer can never mistake
    /// the emitted tail for the full history.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logs_truncation_notice: Option<String>,
}

/// Render the tail of a log buffer for one response, bounded by
/// `WORKER_MAX_EMITTED_LOGS` (audit 07, R4).
pub fn emit_view(buffer: &LogBuffer, max_emitted: usize) -> EmittedLogs {
    let tail = buffer.tail(max_emitted);
    let omitted = buffer.len().saturating_sub(tail.len());
    let mut notice_parts: Vec<String> = Vec::new();
    if buffer.dropped() > 0 {
        notice_parts.push(format!(
            "{} earlier log(s) evicted by the retention window",
            buffer.dropped()
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
