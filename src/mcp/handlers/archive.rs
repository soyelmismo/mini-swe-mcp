use super::*;

/// Description of the `group` property, next to the handler it belongs to.
pub(crate) const ARCHIVE_GROUP_DESCRIPTION: &str =
    "Only the retired workers of this round. Accepts a string or an array.";

/// Description of the `last` property.
pub(crate) const ARCHIVE_LAST_DESCRIPTION: &str =
    "Only the N most recently retired workers, newest first.";

impl McpServer {
    /// `archive`: the final REPORTs of workers this pool has already retired.
    ///
    /// Retirement deletes the row, the conversation and the branch, so a
    /// worker's REPORT is gone the moment it is merged; every retirement keeps
    /// one compact line of it in `<hub dir>/archive.jsonl`, and this reads it
    /// back.
    ///
    /// Owner-scoped exactly like `list`: a line names the agent whose worker
    /// retired, and another agent's report is a work product, not a pool
    /// statistic. The admin override is the only way to see every owner's.
    pub(super) async fn handle_archive(
        &self,
        args: &Value,
        ctx: &crate::mcp::server::ConnectionContext,
    ) -> Result<Value> {
        let owner = if ctx.is_admin() {
            None
        } else {
            Some(ctx.agent().to_string())
        };
        let group = args
            .get("group")
            .and_then(Value::as_str)
            .map(str::to_string);
        let last = args.get("last").and_then(Value::as_u64).map(|n| n as usize);
        let dir = crate::hub::hub_dir()?;
        // Reading two files off disk, so it runs off the runtime thread like
        // every other blocking step here.
        let requested_owner = owner.clone();
        let records = tokio::task::spawn_blocking(move || {
            crate::pool::archive::read_records(&dir, group.as_deref(), last)
        })
        .await
        .map_err(|e| anyhow::anyhow!("archive read failed: {e}"))??;
        // The filter is applied here rather than in the reader: one reader for
        // every caller, and the scoping rule lives with the handler that knows
        // who is asking. A line with no owner (a row written before ownership
        // was tracked) is unattributed and never shown to a non-admin caller.
        let entries: Vec<Value> = records
            .iter()
            .filter(|record| match &requested_owner {
                Some(owner) => record.owner == *owner,
                None => true,
            })
            .map(|record| {
                json!({
                    "worker_id": record.worker_id,
                    "owner": record.owner,
                    "group": record.group,
                    "task": record.task,
                    "status": record.status,
                    "verified": record.verified,
                    "report": record.report,
                    "commit": record.commit,
                    "retired_at": record.retired_at,
                    "reason": record.reason,
                })
            })
            .collect();
        Ok(json!({
            "owner": owner,
            "count": entries.len(),
            "entries": entries,
        }))
    }
}
