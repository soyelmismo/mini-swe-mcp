use super::*;
use crate::pool::WorkerState;
use std::path::PathBuf;

impl McpServer {
    /// `discard <id>`: drop a stopped worker on purpose, with no merge.
    ///
    /// Owner-only, like `kill` (H-3): a worker belongs to the agent that
    /// dispatched it. This is the removal path for a worker that must go
    /// without ever landing -- a failed consolidator whose gate can never run,
    /// say -- so it deletes the branch, the row, the recorded history, the
    /// steering mailbox and steer-source, the pinned round base and the
    /// worktree leftovers in one command instead of by hand.
    ///
    /// A *running* or *paused* worker is refused, naming `kill` as the step
    /// that comes first: a discard deletes unmerged work with no gate at all,
    /// so it must never be a way to stop a worker that is still producing.
    /// Everything it removes goes through the one shared retirement
    /// ([`crate::pool::retire_worker_reporting`], pool-E28), so a discarded
    /// worker leaves exactly as little behind as a merged one.
    pub(super) async fn handle_discard(
        &self,
        args: &Value,
        ctx: &crate::mcp::server::ConnectionContext,
    ) -> Result<Value> {
        let resolved = self.resolve_worker_id(args, "discard", ctx).await?;
        let wid = resolved.as_str();
        self.require_owner(wid, ctx).await?;
        // Read this process's own record first: it is the only state that
        // distinguishes a live worker from a row a dead process left behind,
        // and it is the live one that must be killed first.
        match self.pool.get_worker_state(wid).await {
            Some(WorkerState::Running { .. }) => anyhow::bail!(
                "Worker {wid} is still running: kill it first with `kill {wid}`, then discard it"
            ),
            Some(WorkerState::Paused { .. }) => anyhow::bail!(
                "Worker {wid} is paused: kill it first with `kill {wid}`, then discard it"
            ),
            _ => {}
        }
        // The repository is the one thing a retirement cannot guess: it is what
        // holds the `worker-<id>` branch. A row names it, and so does the
        // history of a row that is already gone.
        let repo = self.discard_repo(wid).await;
        let root = self.pool.scratch_root().clone();
        let worker_id = wid.to_string();
        // The retirement shells out to git, so it runs off the runtime thread.
        let outcome = tokio::task::spawn_blocking(move || {
            crate::pool::retire_worker_reporting(
                &root,
                &worker_id,
                &crate::pool::RetireContext {
                    repo: repo.as_deref(),
                    // The persisted watch acknowledgements live in the hub
                    // directory, which only the hub daemon knows; the daemon's
                    // own sweep passes it, and `retire_and_forget` below drops
                    // the in-memory half from the router either way.
                    ack_dir: None,
                    keep_branch: false,
                },
            )
        })
        .await
        .map_err(|e| anyhow::anyhow!("discard task for worker {wid} failed: {e}"))?;
        // Its files are gone, so its live record, its event replay state and its
        // acknowledged watch positions go with them: `list` stops showing it and
        // a retired worker can never fire an event again.
        self.retire_and_forget(std::iter::once(wid.to_string())).await;
        Ok(json!({
            "worker_id": wid,
            "discarded": true,
            "branch_deleted": outcome.branch_deleted,
            "worktree_reclaimed": outcome.worktree_reclaimed,
            "row_removed": outcome.row_removed,
        }))
    }

    /// The repository that holds `wid`'s branch, from its row or its history.
    ///
    /// A worker with neither has nothing of its own left in a repository, and
    /// guessing one would risk deleting a branch of a repository this worker
    /// never touched.
    async fn discard_repo(&self, worker_id: &str) -> Option<PathBuf> {
        if let Some(row) = self.pool.worker_row(worker_id).await
            && let Some(repo) = row.repo_path
        {
            return Some(PathBuf::from(repo));
        }
        let root = self.pool.scratch_root().clone();
        let id = worker_id.to_string();
        tokio::task::spawn_blocking(move || {
            crate::pool::load_worker_history_in(&root, &id)
                .ok()
                .map(|history| PathBuf::from(history.repo_path))
        })
        .await
        .ok()
        .flatten()
    }
}
