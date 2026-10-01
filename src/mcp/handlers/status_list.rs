use super::*;

impl McpServer {
    /// `status` action: the caller's own view of one worker's step and
    /// progress. A foreign worker id is refused with its owner, never its
    /// task or state.
    pub(super) async fn handle_status(
        &self,
        args: &Value,
        ctx: &crate::mcp::server::ConnectionContext,
    ) -> Result<Value> {
        let resolved = self.resolve_worker_id(args, "status", ctx).await?;
        let wid = resolved.as_str();
        self.require_owner(wid, ctx).await?;
        if let Some(state) = self.pool.get_worker_state(wid).await {
            // A finished worker's status carries the same review guidance as
            // the wait payload, so polling the status is enough to learn the
            // loop exists.
            let next_step = match &state {
                crate::pool::WorkerState::Completed { .. }
                | crate::pool::WorkerState::Failed { .. } => Some(crate::pool::next_step_for(
                    crate::pool::terminal_branch(&state).as_deref(),
                )),
                _ => None,
            };
            let approved = crate::pool::load_registry_entry_in(self.pool.scratch_root(), wid)
                .and_then(|entry| entry.approved);
            Ok(json!({
                "worker_id": wid,
                "owner": self.owner_of(wid).await,
                "state": state,
                "approved": approved,
                "next_step": next_step,
            }))
        } else if let Some(entry) =
            crate::pool::load_registry_entry_in(self.pool.scratch_root(), wid)
        {
            let state_name = entry.status.display_name();
            // A registry-only terminal row (collected worker, restarted hub)
            // carries the same review guidance as the live path.
            let next_step = entry
                .status
                .is_terminal()
                .then(|| crate::pool::next_step_for(None));
            Ok(json!({
                "worker_id": wid,
                "owner": crate::pool::registry_owner_label(&entry),
                "task": entry.task,
                "model": entry.model,
                "state": {
                    "state": state_name,
                    "details": {
                        "status": state_name,
                        "step": entry.step,
                        "turns": entry.step,
                        "summary": entry.last_command.clone(),
                        "report": entry.report,
                        "verified": entry.verified,
                        "error": if entry.status == crate::pool::RegistryStatus::Failed { Some(entry.last_command) } else { None },
                        "question": entry.question,
                        "pid": entry.pid,
                        "started_at": entry.started_at,
                        "metrics": entry.metrics,
                    }
                },
                "approved": entry.approved,
                "next_step": next_step,
            }))
        } else {
            anyhow::bail!("Worker not found: {wid}")
        }
    }

    /// `list` is scoped to the caller's own workers; `scope: "all"` widens it
    /// to every agent's and needs the admin override (H-3).
    pub(super) async fn handle_list(
        &self,
        args: &Value,
        ctx: &crate::mcp::server::ConnectionContext,
    ) -> Result<Value> {
        let workers = if Self::lists_every_agent(args, ctx)? {
            self.pool.list_workers().await
        } else {
            self.pool.list_workers_of(&ctx.agent()).await
        };
        Ok(json!({ "workers": workers }))
    }

    /// Whether the caller asked for every agent's workers, and may have them.
    ///
    /// Only the two documented values are accepted: a typo is a hard error
    /// rather than a silent fallback to the caller's own workers, which would
    /// look like a pool with nobody else's runs in it. `scope: "all"` is the
    /// admin override, so a non-admin caller gets a refusal rather than a
    /// truncated list it would read as "nobody else is running".
    pub(super) fn lists_every_agent(
        args: &Value,
        ctx: &crate::mcp::server::ConnectionContext,
    ) -> Result<bool> {
        match args.get("scope") {
            None => Ok(false),
            Some(scope) => match scope.as_str() {
                Some(scope) if crate::mcp::schema::LIST_SCOPES.contains(&scope) => {
                    if scope == crate::mcp::schema::LIST_SCOPE_ALL && !ctx.is_admin() {
                        anyhow::bail!("'scope' \"all\" requires the admin override (--admin)");
                    }
                    Ok(scope == crate::mcp::schema::LIST_SCOPE_ALL)
                }
                _ => anyhow::bail!(
                    "'scope' must be one of: {}",
                    crate::mcp::schema::LIST_SCOPES.join(", ")
                ),
            },
        }
    }
}

pub(in crate::mcp) const WORKER_ID_DESCRIPTION: &str = "Target worker (alias 'id'): a unique 3+ char prefix or 'last'; required for verbs that target one.";
