use super::*;

impl McpServer {
    /// `collect` action: the worker's final answer, with the diff summarised
    /// unless the caller asks for it.
    ///
    /// The default reply is the compact one — summary, verified, per-file diff
    /// stat and branch — because the full diff of a large task is what makes a
    /// review expensive. `full: true` restores the whole diff, and
    /// `files: [...]` narrows it to the named paths.
    pub(super) async fn handle_collect(
        &self,
        args: &Value,
        ctx: &crate::mcp::server::ConnectionContext,
    ) -> Result<Value> {
        let resolved = self.resolve_worker_id(args, "collect", ctx).await?;
        let wid = resolved.as_str();
        self.require_owner(wid, ctx).await?;
        let full = args.get("full").and_then(Value::as_bool).unwrap_or(false);
        let files = Self::get_diff_files(args, "collect")?;
        if let Some(collected) = self.pool.collect(wid).await {
            let log_view = LogView {
                logs: collected.logs,
                logs_omitted: collected.logs_omitted,
                logs_dropped: collected.logs_dropped,
                logs_truncation_notice: collected.logs_truncation_notice,
            };
            let (summary, verified, branch, report) = completed_fields(Some(&collected.state));
            // Collect ends the worker's reviewable life, so the guidance is
            // about the branch it leaves behind rather than a further steer.
            let next_step = crate::pool::next_step_for(branch.as_deref());
            let mut state = serde_json::to_value(&collected.state).unwrap_or_default();
            let diff = state
                .pointer("/details/diff")
                .and_then(|value| value.as_str())
                .unwrap_or_default()
                .to_owned();
            let stats = diff_file_stats(&diff);
            // The diff is the one field that can be arbitrarily large, so it
            // leaves the payload unless it was asked for — whole, or narrowed
            // to the files that were named.
            if let Some(details) = state.pointer_mut("/details").and_then(Value::as_object_mut) {
                let scoped = diff_of_files(&diff, &files);
                if !scoped.is_empty() {
                    details.insert("diff".to_string(), json!(scoped));
                } else if !full {
                    details.remove("diff");
                }
            }
            let mut result = json!({
                "worker_id": wid,
                "owner": collected.owner,
                "state": state,
                "summary": summary,
                "verified": verified,
                "branch": branch,
                "report": report,
                "diff_stat": diff_stat_value(&stats),
                "next_step": next_step,
            });
            if let serde_json::Value::Object(map) = &mut result {
                map.extend(log_view.as_map());
            }
            Ok(result)
        } else {
            anyhow::bail!("Worker not found: {wid}")
        }
    }
}
