//! The hub's automatic-round scheduler shares the manual consolidate handler.
//! Durable baselines delimit rounds without timestamps; a registry consolidator
//! outside that baseline closes the round even after a crash between dispatch
//! and the consumed write. Dispatch guards cover a whole batch, not each entry.
use super::server::{ConnectionContext, McpServer};
use crate::hub::auto_consolidate::{AutoConsolidate, Round};
use crate::pool::{RegistryStatus, WorkerRole, load_all_registry_entries_in};
use anyhow::Result;
use serde_json::{Value, json};
use std::path::PathBuf;
use std::sync::Arc;

/// What the scheduler makes of one round.
///
/// The tick needs three answers, not one: start it, leave it for now, or retire
/// it. A round whose consolidator already ran is *retired* -- it is consumed on
/// disk so the scheduler stops reconsidering it, which is different from
/// waiting for one more worker.
pub(crate) enum RoundDecision {
    /// Every worker has settled and none of them is still being run here.
    Start,
    /// Not yet: a worker is still running, or its row is behind the run.
    Wait,
    /// Already consolidated; consume the round so it stops being a candidate.
    Retire,
}

/// Whether `args` asks for the group to be consolidated when it stops.
///
/// `false` is the absence of the request, not a request for no round: it is what
/// a caller that spelled the flag out sends, and [`McpServer::validate_auto_consolidate`]
/// takes it as a no-op. Every decision that follows the flag reads it through
/// here -- the cheap worker gate, the round the hub records, and the round a
/// `--quiet` caller is told to wait on -- so a `false` can never be mistaken for
/// a round that will get a consolidator.
pub(crate) fn consolidate_requested(args: &Value) -> bool {
    matches!(
        args.get("consolidate"),
        Some(Value::Bool(true) | Value::String(_))
    )
}

impl McpServer {
    pub(super) fn auto_store(&self) -> Option<Arc<AutoConsolidate>> {
        self.auto_consolidate.lock().unwrap().clone()
    }

    pub(super) fn validate_auto_consolidate(&self, args: &Value) -> Result<()> {
        match args.get("consolidate") {
            None | Some(Value::Bool(false)) => {}
            Some(Value::Bool(true)) | Some(Value::String(_)) => {
                anyhow::ensure!(
                    self.auto_store().is_some(),
                    "Automatic consolidation requires the hub daemon"
                );
                if let Some(Value::String(model)) = args.get("consolidate") {
                    anyhow::ensure!(
                        !model.trim().is_empty(),
                        "consolidate model must not be empty"
                    );
                }
            }
            _ => anyhow::bail!("consolidate must be a boolean or model alias"),
        }
        let verify = args.get("consolidate_verify");
        anyhow::ensure!(
            verify.is_none_or(Value::is_string),
            "consolidate_verify must be a string"
        );
        // The gate is stored verbatim on the round and run much later by the
        // auto-dispatched consolidator, with no one left to notice that a
        // mangled quote left it unrunnable. Parse-check it here, while the
        // caller is still in the loop that can fix the argument.
        if let Some(cmd) = verify.and_then(Value::as_str) {
            crate::pool::validate_verify_command(cmd, "consolidate_verify")?;
        }
        Ok(())
    }

    pub(super) async fn record_auto_dispatch(
        &self,
        args: &Value,
        owner: &str,
        id: &str,
    ) -> Result<()> {
        let Some(store) = self.auto_store() else {
            return Ok(());
        };
        let entries = load_all_registry_entries_in(self.pool.scratch_root());
        let Some(worker) = entries.iter().find(|e| e.id == id) else {
            anyhow::bail!("Dispatched worker has no registry row");
        };
        let group = worker.group.clone().unwrap_or_else(|| "default".into());
        let members: Vec<_> = entries
            .iter()
            .filter(|e| e.owner.as_deref() == Some(owner) && e.group.as_deref() == Some(&group))
            .collect();
        store.record(
            Round {
                owner: owner.into(),
                group,
                repo: PathBuf::from(worker.repo_path.as_deref().unwrap_or(".")),
                model: args
                    .get("consolidate")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                verify: args
                    .get("consolidate_verify")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                generation: 0,
                consumed: false,
                baseline: members
                    .iter()
                    .filter(|e| e.role == WorkerRole::Worker && e.id != id)
                    .map(|e| e.id.clone())
                    .collect(),
                consolidators: members
                    .iter()
                    .filter(|e| e.role == WorkerRole::Consolidate)
                    .map(|e| e.id.clone())
                    .collect(),
            },
            consolidate_requested(args),
        )
    }

    /// Start the daemon's scheduler over its private durable hub directory.
    /// The returned task must be aborted with the daemon's other background tasks.
    pub async fn start_auto_consolidate(
        &self,
        dir: PathBuf,
    ) -> Result<tokio::task::JoinHandle<()>> {
        let store = AutoConsolidate::open(dir)?;
        *self.auto_consolidate.lock().unwrap() = Some(store.clone());
        let server = self.clone();
        let mut changes = self.pool.subscribe_changes();
        Ok(tokio::spawn(async move {
            server.recovery_wait().await;
            loop {
                if let Err(error) = server.auto_consolidate_tick(&store).await {
                    tracing::warn!(%error, "Automatic consolidation deferred");
                }
                tokio::select! {
                    _ = changes.changed() => {},
                    _ = tokio::time::sleep(std::time::Duration::from_secs(1)) => {},
                }
            }
        }))
    }

    /// Decide whether this round may start now.
    ///
    /// Both facts are cross-process-visible, so neither alone decides:
    ///
    /// * the *rows* say who has settled durably. A row is rewritten on every
    ///   status change, but it is a file: a racing or coalesced write can leave
    ///   it reading a settled status for a worker that is not, and only the
    ///   run itself can fix that.
    /// * this process's own *live records* say what it is still driving. A
    ///   worker the pool still runs is not settled whatever its row says -- the
    ///   window between the implementer's completion and the review phase's
    ///   first write lives exactly there, and a round started in it would
    ///   dispatch a consolidator whose manifest lists that worker as *not
    ///   ready*, merging the rest of the group and closing without it.
    pub(crate) async fn round_decision(
        &self,
        round: &Round,
        entries: &[crate::pool::WorkerRegistryEntry],
    ) -> RoundDecision {
        let members: Vec<&crate::pool::WorkerRegistryEntry> = entries
            .iter()
            .filter(|e| {
                e.owner.as_deref() == Some(&round.owner) && e.group.as_deref() == Some(&round.group)
            })
            .collect();
        // A round whose consolidator already ran, or is running, is settled or
        // being settled: neither is a reason to start another one.
        if members
            .iter()
            .any(|e| e.role == WorkerRole::Consolidate && !round.consolidators.contains(&e.id))
        {
            return RoundDecision::Retire;
        }
        if members
            .iter()
            .any(|e| e.role == WorkerRole::Consolidate && e.status.is_live())
        {
            return RoundDecision::Wait;
        }
        let workers: Vec<&crate::pool::WorkerRegistryEntry> = members
            .iter()
            .copied()
            .filter(|e| e.role == WorkerRole::Worker && !round.baseline.contains(&e.id))
            .collect();
        // Nothing finished yet, or something is still running, failed or
        // paused: the round is not the scheduler's to close.
        if !workers
            .iter()
            .any(|e| e.status == RegistryStatus::Completed)
            || workers.iter().any(|e| {
                !matches!(
                    e.status,
                    RegistryStatus::Completed | RegistryStatus::Stopped
                )
            })
        {
            return RoundDecision::Wait;
        }
        // The rows all read settled; the pool still driving one of these
        // workers means a row is behind the run, so the round waits for the
        // terminal write instead of racing it. Every non-terminal record counts,
        // not just `Running`: a worker paused on an orchestrator question is
        // just as unsettled as one running, and its row can lag the same way.
        for worker in &workers {
            if matches!(
                self.pool.get_worker_state(&worker.id).await,
                Some(
                    crate::pool::WorkerState::Running { .. }
                        | crate::pool::WorkerState::Paused { .. }
                )
            ) {
                return RoundDecision::Wait;
            }
        }
        RoundDecision::Start
    }

    async fn auto_consolidate_tick(&self, store: &Arc<AutoConsolidate>) -> Result<()> {
        for round in store.candidates() {
            let entries = load_all_registry_entries_in(self.pool.scratch_root());
            match self.round_decision(&round, &entries).await {
                RoundDecision::Start => {}
                RoundDecision::Wait => continue,
                // Its consolidator already ran: consume the round so the
                // scheduler stops offering it.
                RoundDecision::Retire => {
                    store.consume(&round)?;
                    continue;
                }
            }
            let Some(_claim) = store.claim(&round) else {
                continue;
            };
            // Same owner, manifest, model defaults and full gate as the manual action.
            let mut ctx = ConnectionContext::stdio();
            ctx.agent_id = Some(round.owner.clone());
            let mut args = json!({"group": round.group, "repo_path": round.repo});
            if let Some(model) = &round.model {
                args["model"] = json!(model);
            }
            if let Some(verify) = &round.verify {
                args["verify"] = json!(verify);
            }
            self.handle_consolidate(&args, None, None, &ctx).await?;
            store.consume(&round)?;
        }
        Ok(())
    }
}
