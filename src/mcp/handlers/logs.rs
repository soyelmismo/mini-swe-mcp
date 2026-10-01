use super::*;

impl McpServer {
    /// Render the bounded tail of a live worker's step history plus the
    /// counters that make the degradation explicit.
    pub(in crate::mcp) async fn render_logs(&self, wid: &str) -> LogView {
        let Some(buffer) = self.pool.get_worker_logs(wid).await else {
            return LogView::default();
        };
        let view = emit_view(&buffer, self.pool.log_policy().max_emitted);
        LogView {
            logs: view.logs,
            logs_omitted: view.logs_omitted,
            logs_dropped: buffer.dropped(),
            logs_truncation_notice: view.logs_truncation_notice,
        }
    }

    /// `logs` action: inspect a live worker's retained history without
    /// collecting (and thus evicting) it. Same ownership rule as `status`.
    pub(in crate::mcp) async fn handle_logs(
        &self,
        args: &Value,
        ctx: &crate::mcp::server::ConnectionContext,
    ) -> Result<Value> {
        let resolved = self.resolve_worker_id(args, "logs", ctx).await?;
        let wid = resolved.as_str();
        self.require_owner(wid, ctx).await?;
        let Some(buffer) = self.pool.get_worker_logs(wid).await else {
            anyhow::bail!("Worker not found: {wid}")
        };
        let policy = self.pool.log_policy();
        let view = emit_view(&buffer, policy.max_emitted);
        let log_view = LogView {
            logs: view.logs,
            logs_omitted: view.logs_omitted,
            logs_dropped: buffer.dropped(),
            logs_truncation_notice: view.logs_truncation_notice,
        };
        let mut result = json!({
            "worker_id": wid,
            "state": self.pool.get_worker_state(wid).await,
            "total_steps": buffer.total(),
            "logs_retained": buffer.retained(),
            "retention": {
                "max_retained": policy.max_retained,
                "max_emitted": policy.max_emitted,
            },
        });
        if let serde_json::Value::Object(map) = &mut result {
            map.extend(log_view.as_map());
        }
        Ok(result)
    }
}

/// The four-field log view shared by `logs`, `collect` and `await_worker_result`.
///
/// [`LogView::as_map`] flattens the keys into the top level of the enclosing
/// response, exactly as before.
#[derive(Debug, Default, serde::Serialize)]
pub(in crate::mcp) struct LogView {
    pub(in crate::mcp) logs: Vec<crate::agent::AgentStepLog>,
    pub(in crate::mcp) logs_omitted: usize,
    pub(in crate::mcp) logs_dropped: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(in crate::mcp) logs_truncation_notice: Option<String>,
}

impl LogView {
    /// The four log keys as a JSON map, for embedding into a `json!` response.
    pub(in crate::mcp) fn as_map(&self) -> Map<String, Value> {
        let mut map = Map::new();
        map.insert(
            "logs".to_string(),
            serde_json::to_value(&self.logs).unwrap_or_default(),
        );
        map.insert("logs_omitted".to_string(), json!(self.logs_omitted));
        map.insert("logs_dropped".to_string(), json!(self.logs_dropped));
        map.insert(
            "logs_truncation_notice".to_string(),
            serde_json::to_value(&self.logs_truncation_notice).unwrap_or_default(),
        );
        map
    }
}
