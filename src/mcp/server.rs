//! The [`McpServer`] handle, its handshake, and the stdio run loop.
//!
//! The server owns no worker logic of its own: it holds the [`WorkerPool`], the
//! model manifest and the precomputed `tools/list` payload, then routes each
//! JSON-RPC request to the right response. Verb handling lives in
//! [`super::handlers`], the advertised contract in [`super::schema`] and the
//! envelope types in [`super::protocol`].

use anyhow::Result;
use serde_json::{Value, json};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::mpsc;
use tracing::{error, info};

use super::protocol::{JsonRpcRequest, JsonRpcResponse, PreSerializedResult};
use super::schema::build_tools_list;
use crate::manifest::ModelManifest;
use crate::pool::WorkerPool;

#[derive(Clone)]
pub struct McpServer {
    pub(super) pool: Arc<WorkerPool>,
    pub(super) default_model: String,
    pub(super) manifest: Arc<ModelManifest>,
    /// Precomputed, immutable `tools/list` result. The manifest is never mutated
    /// after construction, so the payload is byte-identical for the process
    /// lifetime and is cloned (an `Arc` memcpy) instead of rebuilt per request.
    pub(super) tools_list: Arc<Value>,
}

impl McpServer {
    pub fn new(pool: WorkerPool, default_model: String, manifest: ModelManifest) -> Self {
        let manifest = Arc::new(manifest);
        let tools_list = Arc::new(build_tools_list(&manifest));
        Self {
            pool: Arc::new(pool),
            default_model,
            manifest,
            tools_list,
        }
    }

    /// Serve MCP over stdin/stdout until the client closes the input.
    pub async fn run_stdio(&self) -> Result<()> {
        // Background reaper: bounds the memory held by terminal worker records
        // even when the orchestrator never calls `collect` (audit 07, R3).
        let reaper = crate::pool::spawn_reaper((*self.pool).clone());

        let stdin = tokio::io::stdin();
        let mut stdout = tokio::io::stdout();
        let mut reader = BufReader::new(stdin).lines();

        info!("Mini-SWE-MCP server listening on stdio");

        let (out_tx, mut out_rx) = mpsc::channel::<String>(128);

        // Dedicated background writer draining stdout messages
        let stdout_task = tokio::spawn(async move {
            while let Some(msg) = out_rx.recv().await {
                if let Err(e) = stdout.write_all(msg.as_bytes()).await {
                    error!(error = %e, "Failed writing to stdout");
                    break;
                }
                if let Err(e) = stdout.flush().await {
                    error!(error = %e, "Failed flushing stdout");
                    break;
                }
            }
        });

        while let Some(line) = reader.next_line().await? {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            let req: JsonRpcRequest = match serde_json::from_str(line) {
                Ok(r) => r,
                Err(e) => {
                    error!(error = %e, line = %line, "Malformed JSON-RPC request");
                    let resp = JsonRpcResponse::err(None, -32700, format!("Parse error: {e}"));
                    if let Ok(serialized) = serde_json::to_string(&resp) {
                        let _ = out_tx.send(serialized + "\n").await;
                    }
                    continue;
                }
            };

            // JSON-RPC 2.0: Server MUST NOT reply to notifications (requests without an ID)
            if req.id.is_none() {
                info!(method = %req.method, "Received notification");
                continue;
            }

            let server = self.clone();
            let tx = out_tx.clone();
            tokio::spawn(async move {
                let resp = server.handle_request(req, Some(tx.clone())).await;
                if let Ok(serialized) = serde_json::to_string(&resp) {
                    let _ = tx.send(serialized + "\n").await;
                }
            });
        }

        drop(out_tx);
        reaper.abort();
        let _ = stdout_task.await;

        Ok(())
    }

    /// Route one JSON-RPC request to its response envelope.
    async fn handle_request(
        &self,
        req: JsonRpcRequest,
        progress_tx: Option<mpsc::Sender<String>>,
    ) -> JsonRpcResponse {
        let id = req.id;
        match req.method.as_str() {
            "ping" => JsonRpcResponse::ok(id, json!({})),

            "initialize" => JsonRpcResponse::ok(
                id,
                json!({
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": { "listChanged": false }
                    },
                    "serverInfo": {
                        "name": "mini-swe-mcp",
                        "version": "0.1.0"
                    }
                }),
            ),

            // The schema is immutable for the process lifetime, so it is built
            // once in `McpServer::new` and only cloned here.
            "tools/list" => JsonRpcResponse::ok(id, (*self.tools_list).clone()),

            "tools/call" => {
                let params = req.params.unwrap_or_default();
                let tool_name = params.get("name").and_then(|v| v.as_str()).unwrap_or("");
                let arguments = params.get("arguments").cloned().unwrap_or(json!({}));

                let progress_token = params
                    .get("_meta")
                    .and_then(|m| m.get("progressToken"))
                    .or_else(|| arguments.get("_meta").and_then(|m| m.get("progressToken")))
                    .or_else(|| arguments.get("progressToken"))
                    .cloned();

                match self
                    .execute_tool_with_progress(tool_name, arguments, progress_token, progress_tx)
                    .await
                {
                    Ok(val) => JsonRpcResponse {
                        jsonrpc: "2.0",
                        id,
                        // Serialized straight into the envelope: no intermediate
                        // pretty-printed `String` for the payload (audit 07, F4).
                        result: Some(
                            serde_json::to_value(PreSerializedResult::text(val)).unwrap_or_else(
                                |_| json!({ "content": [{ "type": "text", "text": "{}" }] }),
                            ),
                        ),
                        error: None,
                    },
                    Err(e) => JsonRpcResponse::err(id, -32000, e.to_string()),
                }
            }

            _ => JsonRpcResponse::err(id, -32601, format!("Method not found: {}", req.method)),
        }
    }

    /// The precomputed `tools/list` result.
    ///
    /// Exposed so the advertised tool contract can be asserted in-process
    /// (unit tests, embedders) instead of only through a live stdio subprocess.
    pub fn tools_list(&self) -> Value {
        (*self.tools_list).clone()
    }

    /// Execute a tool call without progress notifications (CLI path).
    pub async fn execute_tool(&self, name: &str, args: Value) -> Result<Value> {
        self.execute_tool_with_progress(name, args, None, None)
            .await
    }

    /// Execute a tool call, optionally streaming `notifications/progress`.
    pub async fn execute_tool_with_progress(
        &self,
        name: &str,
        args: Value,
        progress_token: Option<Value>,
        progress_tx: Option<mpsc::Sender<String>>,
    ) -> Result<Value> {
        if name != "worker" {
            anyhow::bail!("Unknown tool: '{name}'. Only 'worker' is supported.");
        }

        let action = args.get("action").and_then(|v| v.as_str()).unwrap_or("");

        self.dispatch(action, &args, progress_token.as_ref(), progress_tx.as_ref())
            .await
    }

    /// Poll a worker until it finishes, fails, or pauses for orchestrator input.
    ///
    /// Returns the terminal payload:
    /// * `{ worker_id, state, logs }` when the worker reached `Completed`/`Failed`
    /// * `{ worker_id, status: "needs_input", question, step, message }` when the
    ///   worker paused waiting for steering.
    ///
    /// Progress notifications are emitted only when a `progress_token`/`tx` pair is
    /// supplied (i.e. the MCP stdio path); the plain-CLI path passes `None`, and the
    /// polling algorithm stays identical for both callers.
    pub async fn await_worker_result(
        &self,
        wid: &str,
        max_turns: usize,
        token: Option<&Value>,
        tx: Option<&mpsc::Sender<String>>,
    ) -> Result<Value> {
        let mut last_reported_step = 0;
        loop {
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
            // H-5: poll the lightweight progress snapshot. It never clones the
            // (potentially multi-megabyte) `diff`/`summary`/`artifacts` that a
            // `get_worker_state` clone would copy on every 500 ms tick.
            let Some(progress) = self.pool.worker_progress(wid).await else {
                continue;
            };
            match progress.phase {
                crate::pool::WorkerPhase::Running => {
                    let step = progress.step;
                    if step > last_reported_step {
                        last_reported_step = step;
                        let last_command = progress.last_command.as_deref().unwrap_or("");
                        Self::emit_progress(
                            tx,
                            token,
                            step,
                            max_turns,
                            format!("Step {step}/{max_turns}: {last_command}"),
                        )
                        .await;
                    }
                }
                crate::pool::WorkerPhase::Completed | crate::pool::WorkerPhase::Failed => {
                    Self::emit_progress(
                        tx,
                        token,
                        max_turns,
                        max_turns,
                        format!("Worker {wid} finished execution"),
                    )
                    .await;
                    let state = self.pool.get_worker_state(wid).await;
                    // Bounded emission: the response never carries the
                    // whole history, and the counters say so (audit 07,
                    // R4 / R7).
                    let (logs, logs_omitted, logs_dropped, logs_truncation_notice) =
                        self.render_logs(wid).await;
                    return Ok(json!({
                        "worker_id": wid,
                        "state": state,
                        "logs": logs,
                        "logs_omitted": logs_omitted,
                        "logs_dropped": logs_dropped,
                        "logs_truncation_notice": logs_truncation_notice,
                    }));
                }
                crate::pool::WorkerPhase::Paused => {
                    let step = progress.step;
                    let question = progress.question.as_deref().unwrap_or("");
                    Self::emit_progress(
                        tx,
                        token,
                        step,
                        max_turns,
                        format!("Worker {wid} paused: waiting for orchestrator steering"),
                    )
                    .await;
                    return Ok(json!({
                        "worker_id": wid,
                        "status": "needs_input",
                        "question": question,
                        "step": step,
                        "message": "Worker is paused waiting for orchestrator steering."
                    }));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcp::schema::WORKER_ACTIONS;
    use serde_json::json;

    fn server() -> McpServer {
        McpServer::new(
            WorkerPool::new(1, "http://localhost:1".to_string(), "test-key".to_string()),
            "ninja".to_string(),
            ModelManifest::default(),
        )
    }

    /// No verb may be advertised without a handler behind it.
    #[tokio::test]
    async fn every_advertised_action_is_dispatchable() {
        let server = server();
        for action in WORKER_ACTIONS {
            // Arguments are deliberately missing, so most verbs fail their own
            // validation; what matters is that the verb itself is recognised.
            let unknown = match server
                .execute_tool("worker", json!({ "action": action }))
                .await
            {
                Ok(_) => None,
                Err(error) => Some(error.to_string()),
            };
            assert!(
                !unknown
                    .as_deref()
                    .is_some_and(|error| error.contains("Unknown action or tool")),
                "'{action}' is advertised in the schema but not dispatched: {unknown:?}"
            );
        }
    }

    /// The payload is immutable, hence built once and only cloned afterwards.
    #[test]
    fn tools_list_is_precomputed_and_stable() {
        let server = server();
        let clone = server.clone();

        // Clones share the very same precomputed payload...
        assert!(Arc::ptr_eq(&server.tools_list, &clone.tools_list));
        // ...which already carries the manifest-derived model description.
        assert_eq!(
            server.tools_list()["tools"][0]["inputSchema"]["properties"]["model"]["description"],
            Value::String(server.manifest.build_tool_description())
        );
    }
}
