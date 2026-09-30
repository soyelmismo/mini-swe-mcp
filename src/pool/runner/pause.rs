//! The suspend/resume handshake between a worker and the orchestrator.
//!
//! A worker never blocks on the orchestrator by accident: it blocks only when
//! it emitted the `ASK_ORCHESTRATOR` sentinel, and it unblocks only when a
//! `steer`/`resume` arrives through the worker's resume channel. Parking the
//! handshake here keeps [`PauseRequest`] the one description of "a worker is
//! waiting, here is why, here is where to answer it", so the in-memory state
//! and the on-disk registry entry can never disagree about the question.

use anyhow::Result;
use tracing::info;

use super::sentinels::summarize_command;
use super::super::WorkerPool;
use super::super::registry::{RegistryStatus, WorkerMeta};
use super::super::state::WorkerState;
use super::super::unix_timestamp;

/// Everything needed to park a worker on an orchestrator question and wait.
///
/// `last_command` is the command that carried the sentinel, so the registry
/// keeps showing what the worker was doing while it waits. `meta` is the
/// worker's registry row, so a pause is written with the same identity -- and
/// the same health counters -- as every other status update of the run.
pub struct PauseRequest<'a> {
    pub worker_id: &'a str,
    pub question: &'a str,
    pub step: usize,
    pub max_turns: usize,
    pub last_command: &'a str,
    pub model: &'a str,
    pub meta: &'a WorkerMeta,
}

impl WorkerPool {
    /// Park the worker on `question` and block until the orchestrator answers.
    ///
    /// Returns the answer once one arrives. If the channel closes instead —
    /// the worker was killed or reaped — this returns `None` and the caller
    /// carries on, so a dead orchestrator cannot strand a worker mid-loop.
    pub(super) async fn pause_for_orchestrator(
        &self,
        req: PauseRequest<'_>,
    ) -> Result<Option<String>> {
        let PauseRequest {
            worker_id,
            question,
            step,
            max_turns,
            last_command,
            model,
            meta,
        } = req;

        let worker_id = worker_id.to_string();
        info!(
            worker = %worker_id,
            question = %question,
            "Subagent paused waiting for orchestrator guidance"
        );
        let (tx, mut rx) = tokio::sync::mpsc::channel(1);
        let now = unix_timestamp();
        {
            let mut lock = self.workers.write().await;
            if let Some(w) = lock.get_mut(&worker_id) {
                w.state = WorkerState::Paused {
                    question: question.to_string(),
                    step,
                    paused_at: now,
                };
                w.resume_tx = Some(tx);
            }
        }

        meta.save_status(
            model,
            RegistryStatus::Paused,
            step,
            max_turns,
            last_command,
            Some(question.to_string()),
        );

        let Some(answer) = rx.recv().await else {
            return Ok(None);
        };
        info!(worker = %worker_id, "Worker resumed by orchestrator guidance");
        {
            let mut lock = self.workers.write().await;
            if let Some(w) = lock.get_mut(&worker_id) {
                w.state = WorkerState::Running {
                    step,
                    last_command: format!("resumed: {}", summarize_command(&answer)),
                    started_at: now,
                };
                w.resume_tx = None;
            }
        }
        Ok(Some(answer))
    }
}
