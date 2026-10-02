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
                crate::pool::WorkerState::Exhausted { turns, .. } => {
                    Some(crate::pool::exhausted_next_step(
                        wid,
                        *turns,
                        crate::pool::terminal_branch(&state).as_deref(),
                    ))
                }
                _ => None,
            };
            let entry = crate::pool::load_registry_entry_in(self.pool.scratch_root(), wid);
            // The build-slot wait and the command in flight live on the live
            // progress, not on the state, so read them once for the compact
            // projection.
            let progress = self.pool.worker_progress(wid).await;
            let approved = entry.as_ref().and_then(|entry| entry.approved.clone());
            Ok(json!({
                "worker_id": wid,
                "owner": self.owner_of(wid).await,
                // The full state carries the multi-megabyte diff and every
                // artifact path; the projection below keeps exactly what an
                // orchestrator reads and leaves the diff to collect/review.
                "state": {
                    "state": state.name(),
                    "details": compact_status_details(
                        &state,
                        entry.as_ref(),
                        progress.as_ref(),
                        crate::pool::unix_timestamp(),
                    ),
                },
                "approved": approved,
                "next_step": next_step,
            }))
        } else if let Some(entry) =
            crate::pool::load_registry_entry_in(self.pool.scratch_root(), wid)
        {
            let state_name = entry.status.display_name();
            // A registry-only terminal row (collected worker, restarted hub)
            // carries the same review guidance as the live path.
            let branch = format!("worker-{wid}");
            let next_step = if entry.status == crate::pool::RegistryStatus::Exhausted {
                Some(crate::pool::exhausted_next_step(
                    wid,
                    entry.step,
                    Some(branch.as_str()),
                ))
            } else if entry.status.is_terminal() {
                Some(crate::pool::next_step_for(None))
            } else {
                None
            };
            // A terminal row stopped the clock at its last write; a live row is
            // still running, so it is measured to now.
            let elapsed = if entry.status.is_terminal() {
                entry.updated_at.saturating_sub(entry.started_at)
            } else {
                crate::pool::unix_timestamp().saturating_sub(entry.started_at)
            };
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
                        "max_turns": entry.max_turns,
                        "revision": entry.revision,
                        "summary": entry.last_command.clone(),
                        "report": entry.report,
                        "verified": entry.verified,
                        "error": if entry.status == crate::pool::RegistryStatus::Failed { Some(entry.last_command) } else { None },
                        "question": entry.question,
                        "pid": entry.pid,
                        "started_at": entry.started_at,
                        "elapsed": elapsed,
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

pub(in crate::mcp) const WORKER_ID_DESCRIPTION: &str = "Target worker (alias 'id'): 3+ char prefix or 'last'; required by targeting verbs.";

/// The compact `details` object behind a live worker's `status`.
///
/// Only what an orchestrator reads: the step and its budget, the elapsed time
/// and the command in flight (or the build slot it waits for), plus the fields
/// a terminal state carries. The diff is deliberately absent -- `collect` and
/// `review` serve it -- and the artifact list is capped to a preview plus a
/// count, so a worker that synced hundreds of files stays small.
fn compact_status_details(
    state: &crate::pool::WorkerState,
    entry: Option<&crate::pool::WorkerRegistryEntry>,
    progress: Option<&crate::pool::WorkerProgress>,
    now: u64,
) -> Value {
    let mut details = serde_json::Map::new();
    details.insert("step".into(), json!(state.step()));
    details.insert("turns".into(), json!(state.step()));
    if let Some(max_turns) = entry.map(|entry| entry.max_turns) {
        details.insert("max_turns".into(), json!(max_turns));
    }
    // Prefer the registry row's dispatch time: it is the one clock that also
    // answers for a worker this process only knows through its row.
    let started_at = entry
        .map(|entry| entry.started_at)
        .or_else(|| state_timestamp(state));
    if let Some(started_at) = started_at {
        let elapsed = match state_end(state) {
            Some(end) => end.saturating_sub(started_at),
            None => now.saturating_sub(started_at),
        };
        details.insert("started_at".into(), json!(started_at));
        details.insert("elapsed".into(), json!(elapsed));
    }
    if let Some(revision) = state_revision(state).or_else(|| entry.map(|entry| entry.revision)) {
        details.insert("revision".into(), json!(revision));
    }
    if let Some(metrics) = state_metrics(state).or_else(|| entry.map(|entry| entry.metrics)) {
        details.insert("metrics".into(), json!(metrics));
    }
    if let Some(progress) = progress {
        if let Some(command) = progress.last_command.as_deref() {
            details.insert("last_command".into(), json!(command));
        }
        if let Some(waiting) = progress.waiting_for_slot {
            details.insert("waiting_for_slot".into(), json!(waiting));
        }
        if let Some(started) = progress.command_started_at {
            details.insert("command_started_at".into(), json!(started));
            details.insert("command_elapsed".into(), json!(now.saturating_sub(started)));
        }
        if let Some(question) = progress.question.as_deref() {
            details.insert("question".into(), json!(question));
        }
    }
    match state {
        crate::pool::WorkerState::Running { last_command, .. } => {
            details
                .entry("last_command".to_string())
                .or_insert_with(|| json!(last_command));
        }
        crate::pool::WorkerState::Paused { question, .. } => {
            details
                .entry("question".to_string())
                .or_insert_with(|| json!(question));
        }
        crate::pool::WorkerState::Completed {
            summary,
            artifacts,
            branch,
            verified,
            revision,
            report,
            ..
        } => {
            details.insert("summary".into(), json!(summary));
            insert_compact_artifacts(&mut details, artifacts);
            details.insert("branch".into(), json!(branch));
            details.insert("verified".into(), json!(verified));
            details.insert("revision".into(), json!(revision));
            // The report is a handful of fields, not the diff: it travels so a
            // live completion answers the same question its registry row does.
            details.insert("report".into(), json!(report));
        }
        crate::pool::WorkerState::Failed {
            error, revision, ..
        } => {
            details.insert("error".into(), json!(error));
            details.insert("revision".into(), json!(revision));
        }
        crate::pool::WorkerState::Exhausted {
            summary,
            artifacts,
            branch,
            revision,
            report,
            ..
        } => {
            details.insert("summary".into(), json!(summary));
            insert_compact_artifacts(&mut details, artifacts);
            details.insert("branch".into(), json!(branch));
            details.insert("revision".into(), json!(revision));
            details.insert("reason".into(), json!(crate::pool::TURN_BUDGET_EXHAUSTED));
            details.insert("report".into(), json!(report));
        }
    }
    Value::Object(details)
}

/// Cap the artifact list to a preview and report how many were left out, so a
/// large completion answer carries a count instead of every path.
fn insert_compact_artifacts(details: &mut serde_json::Map<String, Value>, artifacts: &[String]) {
    let (preview, total) = crate::pool::compact_artifacts(artifacts);
    details.insert("artifacts".into(), json!(preview));
    details.insert("artifacts_total".into(), json!(total));
}

/// The clock a non-running state records: dispatch time while live, completion
/// time once terminal.
fn state_timestamp(state: &crate::pool::WorkerState) -> Option<u64> {
    match state {
        crate::pool::WorkerState::Running { started_at, .. } => Some(*started_at),
        crate::pool::WorkerState::Paused { paused_at, .. } => Some(*paused_at),
        crate::pool::WorkerState::Completed { completed_at, .. } => Some(*completed_at),
        crate::pool::WorkerState::Failed { failed_at, .. } => Some(*failed_at),
        crate::pool::WorkerState::Exhausted { stopped_at, .. } => Some(*stopped_at),
    }
}

/// The moment a terminal state stopped the clock, so its elapsed time is the
/// run's length rather than the time since it finished.
fn state_end(state: &crate::pool::WorkerState) -> Option<u64> {
    match state {
        crate::pool::WorkerState::Running { .. } | crate::pool::WorkerState::Paused { .. } => None,
        _ => state_timestamp(state),
    }
}

/// The revision counter a state carries, when it has one.
fn state_revision(state: &crate::pool::WorkerState) -> Option<usize> {
    match state {
        crate::pool::WorkerState::Completed { revision, .. }
        | crate::pool::WorkerState::Failed { revision, .. }
        | crate::pool::WorkerState::Exhausted { revision, .. } => Some(*revision),
        crate::pool::WorkerState::Running { .. } | crate::pool::WorkerState::Paused { .. } => None,
    }
}

/// The metrics a state carries, when it has them.
fn state_metrics(state: &crate::pool::WorkerState) -> Option<crate::pool::WorkerMetrics> {
    match state {
        crate::pool::WorkerState::Completed { metrics, .. }
        | crate::pool::WorkerState::Failed { metrics, .. }
        | crate::pool::WorkerState::Exhausted { metrics, .. } => Some(*metrics),
        crate::pool::WorkerState::Running { .. } | crate::pool::WorkerState::Paused { .. } => None,
    }
}
