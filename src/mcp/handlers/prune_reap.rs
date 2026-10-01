use super::*;

impl McpServer {
    /// `reap` action: evict terminal worker records whose TTL expired.
    pub(super) async fn handle_reap(&self) -> Result<Value> {
        let reaped = self.pool.reap().await;
        Ok(json!({
            "status": "reaped",
            "reaped": reaped.len(),
            "worker_ids": reaped,
        }))
    }

    pub(super) async fn handle_prune(
        &self,
        args: &Value,
        token: Option<&Value>,
        tx: Option<&mpsc::Sender<String>>,
        ctx: &crate::mcp::server::ConnectionContext,
    ) -> Result<Value> {
        let repo_path = Self::get_repo_path(args, ctx);
        Self::emit_progress(
            tx,
            token,
            0,
            1,
            "Pruning stale worktrees and dead worker branches",
        )
        .await;
        // Both sweeps walk directories, shell out to git and salvage dead
        // worktrees, so they run off the runtime thread.
        let root = self.pool.scratch_root().clone();
        let pruned = tokio::task::spawn_blocking(move || {
            crate::worktree::prune_stale_worktrees_in(&repo_path, &root.base_dirs());
            crate::pool::prune_orphan_histories_in(&root, &repo_path);
        })
        .await;
        if pruned.is_err() {
            tracing::warn!("Worktree prune task could not run");
        }
        Self::emit_progress(tx, token, 1, 1, "Prune complete").await;
        Ok(json!({
            "status": "pruned",
            "message": "Stale worktrees and dead worker branches cleaned up"
        }))
    }
}
