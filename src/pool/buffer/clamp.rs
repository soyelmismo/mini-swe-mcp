//! Per-entry payload clamping: the byte ceilings that make one retained step
//! log bounded.
//!
//! [`LogBuffer`](super::LogBuffer) bounds *how many* entries are held; this
//! module bounds *how large* each one can be. The truncation marker is charged
//! against the budget, so a stored value is `<= budget` rather than
//! "budget + marker" — the content-preserving truncation of audit 07 / F6 made
//! the 2 KiB ceiling a floor, never a cap.

#[cfg(test)]
mod tests;

use crate::agent::AgentStepLog;

use super::{MAX_LOG_COMMAND_BYTES, MAX_LOG_OUTPUT_BYTES};

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
