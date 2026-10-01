use super::*;

impl McpServer {
    /// `merge` action: land one finished worker's branch on its base branch.
    ///
    /// Owner-only, like every other per-worker verb. The whole sequence --
    /// trial merge, dirty check, gate, real merge, cleanup -- is one blocking
    /// unit in [`crate::pool::merge`], so it runs off the runtime thread and
    /// answers with a single payload the CLI renders as one line.
    pub(super) async fn handle_merge(
        &self,
        args: &Value,
        ctx: &crate::mcp::server::ConnectionContext,
    ) -> Result<Value> {
        // `merge --approved` lands a whole round with one gate: no worker id,
        // the caller's approved workers (optionally one group) instead.
        if args.get("approved").and_then(|v| v.as_bool()) == Some(true) {
            return self.merge_approved(args, ctx).await;
        }
        let wid = Self::get_worker_id(args, "merge")?;
        self.require_owner(wid, ctx).await?;
        // A worker this process owns carries its verify verdict in memory; a
        // cross-process caller has only the on-disk row, which names none, and
        // therefore re-runs the gate.
        let verified = match self.pool.get_worker_state(wid).await {
            Some(crate::pool::WorkerState::Completed { verified, .. }) => verified,
            _ => None,
        };
        let keep_branch = args
            .get("keep_branch")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let root = self.pool.scratch_root().clone();
        let admission = self.pool.admission();
        let worker_id = wid.to_string();
        let report = tokio::task::spawn_blocking(move || {
            crate::pool::merge_worker_in(
                &root,
                &crate::pool::MergeRequest {
                    worker_id: &worker_id,
                    verified,
                    keep_branch,
                    admission: Some(admission),
                },
            )
        })
        .await
        .map_err(|e| anyhow::anyhow!("merge task for worker {wid} failed: {e}"))??;
        // The backstop sweep, so a worker integrated by any other path (an older
        // merge, another session, a consolidator) is retired too.
        self.pool.sweep_retired_workers().await;
        Ok(json!({
            "worker_id": report.worker_id,
            "branch": report.branch,
            "base_branch": report.base_branch,
            "repo_path": report.repo_path,
            "commit": report.commit,
            "gate": if report.gate_ran { "ran" } else { "skipped" },
            "gate_command": report.gate_command,
            "branch_deleted": report.branch_deleted,
            "cleaned": report.cleaned,
        }))
    }

    /// Land every approved worker of the caller (optionally one group) with a
    /// single gate.
    ///
    /// The selection is owner-scoped exactly like `list`: another agent's
    /// approved workers are never in the batch, and the admin override is the
    /// only way to see every owner's. The whole sequence -- compose, one gate,
    /// one `--no-ff` merge per worker, cleanup -- is one blocking unit in
    /// [`crate::pool::merge`], so it runs off the runtime thread.
    pub(super) async fn merge_approved(
        &self,
        args: &Value,
        ctx: &crate::mcp::server::ConnectionContext,
    ) -> Result<Value> {
        let group = args
            .get("group")
            .and_then(|v| v.as_str())
            .map(str::to_string);
        let group_echo = group.clone();
        let owner = if ctx.is_admin() {
            None
        } else {
            Some(ctx.agent().to_string())
        };
        let root = self.pool.scratch_root().clone();
        let admission = self.pool.admission();
        let report = tokio::task::spawn_blocking(move || {
            crate::pool::merge_approved_in(
                &root,
                &crate::pool::MergeApprovedRequest {
                    owner: owner.as_deref(),
                    group: group.as_deref(),
                    admission: Some(admission),
                },
            )
        })
        .await
        .map_err(|e| anyhow::anyhow!("batch merge task failed: {e}"))??;
        self.pool.sweep_retired_workers().await;
        Ok(json!({
            "approved": true,
            "group": group_echo,
            "base_branch": report.base_branch,
            "repo_path": report.repo_path,
            "merged": report
                .merged
                .iter()
                .map(|m| json!({"worker_id": m.worker_id, "commit": m.commit}))
                .collect::<Vec<_>>(),
            "skipped": report
                .skipped
                .iter()
                .map(|s| json!({"worker_id": s.worker_id, "files": s.files, "steer": s.steer}))
                .collect::<Vec<_>>(),
            "gate_command": report.gate_command,
            "gate_duration_ms": report.gate_duration_ms,
            "cleaned": report.cleaned,
        }))
    }
}
