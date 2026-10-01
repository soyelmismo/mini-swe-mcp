use super::*;

impl McpServer {
    pub(super) async fn handle_kill(
        &self,
        args: &Value,
        ctx: &crate::mcp::server::ConnectionContext,
    ) -> Result<Value> {
        let resolved = self.resolve_worker_id(args, "kill", ctx).await?;
        let wid = resolved.as_str();
        self.require_owner(wid, ctx).await?;
        let killed = self.pool.kill(wid).await;
        if killed {
            Ok(json!({ "worker_id": wid, "killed": true }))
        } else if let Some(entry) =
            crate::pool::load_registry_entry_in(self.pool.scratch_root(), wid)
            && crate::worktree::is_process_alive(entry.pid)
        {
            #[cfg(unix)]
            {
                // SAFETY: `entry.pid` is a foreign pid read from the registry;
                // `kill` only signals, never dereferences, so this is safe.
                unsafe {
                    libc::kill(entry.pid as libc::pid_t, libc::SIGTERM);
                }
            }
            Ok(json!({ "worker_id": wid, "killed": true }))
        } else {
            Ok(json!({ "worker_id": wid, "killed": false }))
        }
    }
}
