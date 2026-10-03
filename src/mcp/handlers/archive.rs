use super::*;

/// Description of the `last` property.
pub(crate) const ARCHIVE_LAST_DESCRIPTION: &str = "Only the N most recent.";

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
        // The same gate `list` uses, so the advertised `'list'/'archive' scope`
        // knob really works here: an admin asking for every owner's archive gets
        // it, and a non-admin asking for one is *refused* rather than silently
        // handed their own rows, which would read as "nobody else ever retired".
        let owner = if Self::lists_every_agent(args, ctx)? {
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
        // every other blocking step here. The owner filter goes *into* the
        // read, before `--last`: applied afterwards it would make `--last` a
        // window over the whole archive, and a busy hub's newest five lines
        // would be somebody else's, leaving this agent with an empty answer to
        // "my last five". A line with no owner (a row written before ownership
        // was tracked) is unattributed and never shown to a non-admin caller.
        let scoped_owner = owner.clone();
        let records = tokio::task::spawn_blocking(move || {
            crate::pool::archive::read_records(
                &dir,
                scoped_owner.as_deref(),
                group.as_deref(),
                last,
            )
        })
        .await
        .map_err(|e| anyhow::anyhow!("archive read failed: {e}"))??;
        let entries: Vec<Value> = records
            .iter()
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
