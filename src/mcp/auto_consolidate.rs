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
            matches!(
                args.get("consolidate"),
                Some(Value::Bool(true) | Value::String(_))
            ),
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

    async fn auto_consolidate_tick(&self, store: &Arc<AutoConsolidate>) -> Result<()> {
        for round in store.candidates() {
            let entries = load_all_registry_entries_in(self.pool.scratch_root());
            let members: Vec<_> = entries
                .iter()
                .filter(|e| {
                    e.owner.as_deref() == Some(&round.owner)
                        && e.group.as_deref() == Some(&round.group)
                })
                .collect();
            if members
                .iter()
                .any(|e| e.role == WorkerRole::Consolidate && !round.consolidators.contains(&e.id))
            {
                store.consume(&round)?;
                continue;
            }
            if members
                .iter()
                .any(|e| e.role == WorkerRole::Consolidate && e.status.is_live())
            {
                continue;
            }
            let workers: Vec<_> = members
                .iter()
                .filter(|e| e.role == WorkerRole::Worker && !round.baseline.contains(&e.id))
                .collect();
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
                continue;
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
