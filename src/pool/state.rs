//! Worker lifecycle state, the in-memory record, and the cheap read views.
//!
//! The types here are the *data* half of the pool: what a worker looks like
//! while it runs, what a caller sees when it polls, and when a terminal record
//! is old enough to be evicted.

use std::collections::HashMap;
use tokio::task::JoinHandle;

use crate::agent::AgentStepLog;

use super::buffer::{LogBuffer, LogStats};
use super::unix_timestamp;

#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "state", content = "details")]
pub enum WorkerState {
    Running {
        step: usize,
        last_command: String,
        started_at: u64,
    },
    Paused {
        question: String,
        step: usize,
        paused_at: u64,
    },
    Completed {
        turns: usize,
        diff: String,
        summary: String,
        completed_at: u64,
        #[serde(default)]
        artifacts: Vec<String>,
        #[serde(default)]
        branch: Option<String>,
    },
    Failed {
        error: String,
        step: usize,
        failed_at: u64,
    },
}

impl WorkerState {
    /// Turn this state corresponds to, for every variant.
    pub fn step(&self) -> usize {
        match self {
            WorkerState::Running { step, .. } | WorkerState::Paused { step, .. } => *step,
            WorkerState::Completed { turns, .. } => *turns,
            WorkerState::Failed { step, .. } => *step,
        }
    }

    pub fn to_summary(&self) -> serde_json::Value {
        match self {
            WorkerState::Running { step, last_command, started_at } => serde_json::json!({
                "status": "Running",
                "step": step,
                "last_command": last_command,
                "started_at": started_at,
            }),
            WorkerState::Paused { question, step, paused_at } => serde_json::json!({
                "status": "Paused",
                "step": step,
                "question": question,
                "paused_at": paused_at,
            }),
            WorkerState::Completed { turns, summary, completed_at, artifacts, branch, .. } => serde_json::json!({
                "status": "Completed",
                "turns": turns,
                "summary": summary,
                "completed_at": completed_at,
                "artifacts": artifacts,
                "branch": branch,
            }),
            WorkerState::Failed { error, step, failed_at } => serde_json::json!({
                "status": "Failed",
                "step": step,
                "error": error,
                "failed_at": failed_at,
            }),
        }
    }
}

pub struct WorkerRecord {
    pub id: String,
    pub task: String,
    pub model: String,
    pub state: WorkerState,
    /// Bounded sliding window of step logs (audit 07, R1).
    pub logs: LogBuffer,
    pub pending_steer: Vec<String>,
    pub resume_tx: Option<tokio::sync::mpsc::Sender<String>>,
    pub handle: Option<JoinHandle<()>>,
}

impl WorkerRecord {
    pub(super) fn fail(&mut self, error: impl Into<String>) {
        self.state = WorkerState::Failed {
            error: error.into(),
            step: self.state.step(),
            failed_at: unix_timestamp(),
        };
    }

    /// Unix timestamp when this record became terminal, if it is terminal.
    pub fn terminal_at(&self) -> Option<u64> {
        match &self.state {
            WorkerState::Completed { completed_at, .. } => Some(*completed_at),
            WorkerState::Failed { failed_at, .. } => Some(*failed_at),
            WorkerState::Running { .. } | WorkerState::Paused { .. } => None,
        }
    }

    /// Step counters for the observability surface (audit 07, R7).
    pub fn log_stats(&self) -> LogStats {
        LogStats {
            total_steps: self.state.step().max(self.logs.total()),
            logs_retained: self.logs.retained(),
            logs_dropped: self.logs.dropped(),
        }
    }
}

/// Coarse lifecycle stage of a worker, used by progress polls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerPhase {
    Running,
    Paused,
    Completed,
    Failed,
}

/// Lightweight, allocation-cheap snapshot of a worker's progress.
///
/// Excludes the terminal payload (`diff`, `summary`, `artifacts`) so the
/// 500 ms polling loops neither clone nor serialize multi-megabyte strings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerProgress {
    pub phase: WorkerPhase,
    /// Turn currently running (or the turn the worker paused at); for terminal
    /// phases this is the number of turns performed.
    pub step: usize,
    /// Last bash command summary while running.
    pub last_command: Option<String>,
    /// Escalated question while paused.
    pub question: Option<String>,
    /// `true` once the worker reached `Completed` or `Failed` and its full
    /// payload can be fetched once with `get_worker_state`.
    pub terminal: bool,
}

/// Result of a one-shot worker collection, detached from the live pool.
#[derive(Debug, Clone, serde::Serialize)]
pub struct CollectedWorker {
    pub id: String,
    pub task: String,
    pub model: String,
    pub state: WorkerState,
    /// The retained window, moved out of the pool (not a copy).
    pub logs: Vec<AgentStepLog>,
    /// Retained entries that were not part of `logs` because of the emission
    /// budget.
    pub logs_omitted: usize,
    /// Retained entries already evicted by the retention window.
    pub logs_dropped: usize,
    /// Human-readable explanation when the history is degraded.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logs_truncation_notice: Option<String>,
}

/// Default age (seconds) after which a `Completed`/`Failed` worker record is
/// evicted from the pool.
pub const DEFAULT_TERMINAL_TTL_SECS: u64 = 300;

/// Ids of `Completed`/`Failed` records that reached the terminal TTL (audit 07, R3).
///
/// * A `Running`/`Paused` record is *never* expired, whatever its age.
/// * Clock skew is absorbed by `saturating_sub`: a future timestamp (NTP jump,
///   forged registry row) yields 0 and the record is kept.
/// * A fresh terminal record is kept, which keeps `collect` and `wait: true`
///   working after a worker finishes.
pub(super) fn expired_terminal_ids(
    workers: &HashMap<String, WorkerRecord>,
    ttl_secs: u64,
) -> Vec<String> {
    let now = unix_timestamp();
    let mut ids: Vec<String> = workers
        .iter()
        .filter_map(|(id, record)| {
            record
                .terminal_at()
                .filter(|at| now.saturating_sub(*at) >= ttl_secs)
                .map(|_| id.clone())
        })
        .collect();
    ids.sort();
    ids
}

#[cfg(test)]
mod tests {
    use super::super::buffer::LogBuffer;
    use super::super::unix_timestamp;
    use super::{DEFAULT_TERMINAL_TTL_SECS, WorkerRecord, WorkerState, expired_terminal_ids};
    use std::collections::HashMap;

    // ----------
    // WorkerState::step
    // ----------

    #[test]
    fn test_worker_state_step_covers_every_variant() {
        assert_eq!(
            WorkerState::Running {
                step: 4,
                last_command: "ls".into(),
                started_at: 0
            }
            .step(),
            4
        );
        assert_eq!(
            WorkerState::Paused {
                question: "?".into(),
                step: 9,
                paused_at: 0
            }
            .step(),
            9
        );
        assert_eq!(
            WorkerState::Completed {
                turns: 12,
                diff: String::new(),
                summary: String::new(),
                completed_at: 0,
                artifacts: Vec::new(),
                branch: None,
            }
            .step(),
            12
        );
        assert_eq!(
            WorkerState::Failed {
                error: "boom".into(),
                step: 3,
                failed_at: 0
            }
            .step(),
            3
        );
    }

    #[test]
    fn test_terminal_at_only_reports_terminal_states() {
        let running = WorkerState::Running {
            step: 0,
            last_command: String::new(),
            started_at: 0,
        };
        assert!(matches!(running, WorkerState::Running { .. }));
        let completed = WorkerState::Completed {
            turns: 1,
            diff: String::new(),
            summary: String::new(),
            completed_at: 1_700_000_000,
            artifacts: Vec::new(),
            branch: None,
        };
        let failed = WorkerState::Failed {
            error: "e".into(),
            step: 1,
            failed_at: 1_700_000_001,
        };
        assert!(!matches!(running, WorkerState::Completed { .. }));
        assert!(matches!(completed, WorkerState::Completed { .. }));
        assert!(matches!(failed, WorkerState::Failed { .. }));
    }

    fn record_with(state: WorkerState) -> WorkerRecord {
        WorkerRecord {
            id: "w".into(),
            task: "t".into(),
            model: "m".into(),
            state,
            logs: LogBuffer::new(),
            pending_steer: Vec::new(),
            resume_tx: None,
            handle: None,
        }
    }

    fn completed_at(when: u64) -> WorkerState {
        WorkerState::Completed {
            turns: 1,
            diff: String::new(),
            summary: String::new(),
            completed_at: when,
            artifacts: Vec::new(),
            branch: None,
        }
    }

    fn failed_at(when: u64) -> WorkerState {
        WorkerState::Failed {
            error: "boom".into(),
            step: 1,
            failed_at: when,
        }
    }

    // ----------
    // Terminal-record TTL (audit 07, R3)
    // ----------

    #[test]
    fn test_terminal_records_expire_after_the_ttl() {
        let now = unix_timestamp();
        let mut workers = HashMap::new();
        workers.insert("old-done".to_string(), record_with(completed_at(now - 400)));
        workers.insert("old-failed".to_string(), record_with(failed_at(now - 400)));
        workers.insert("fresh-done".to_string(), record_with(completed_at(now)));
        workers.insert(
            "running".to_string(),
            record_with(WorkerState::Running {
                step: 0,
                last_command: String::new(),
                started_at: now - 100_000,
            }),
        );
        workers.insert(
            "paused".to_string(),
            record_with(WorkerState::Paused {
                question: "?".into(),
                step: 1,
                paused_at: now - 100_000,
            }),
        );

        let expired = expired_terminal_ids(&workers, DEFAULT_TERMINAL_TTL_SECS);
        assert_eq!(
            expired,
            vec!["old-done".to_string(), "old-failed".to_string()],
            "only aged terminal records may be evicted"
        );
    }

    #[test]
    fn test_terminal_ttl_absorbs_clock_skew() {
        // A timestamp in the future (NTP jump, forged registry row) must not
        // evict the record: saturating_sub yields 0, which is below any TTL.
        let now = unix_timestamp();
        let mut workers = HashMap::new();
        workers.insert(
            "skewed".to_string(),
            record_with(completed_at(now + 10_000)),
        );
        assert!(
            expired_terminal_ids(&workers, DEFAULT_TERMINAL_TTL_SECS).is_empty(),
            "a future timestamp must never evict a record"
        );
    }

    #[test]
    fn test_terminal_ttl_of_zero_evicts_immediately() {
        let now = unix_timestamp();
        let mut workers = HashMap::new();
        workers.insert("done".to_string(), record_with(completed_at(now)));
        assert_eq!(expired_terminal_ids(&workers, 0).len(), 1);
    }

    #[test]
    fn test_terminal_ttl_keeps_everything_before_the_boundary() {
        let now = unix_timestamp();
        let mut workers = HashMap::new();
        workers.insert("edge".to_string(), record_with(completed_at(now)));
        assert!(
            expired_terminal_ids(&workers, 1).is_empty(),
            "a record younger than the TTL must survive so collect() still works"
        );
    }
}
