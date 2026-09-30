//! The [`McpServer`] handle, its handshake, and the stdio run loop.
//!
//! The server owns no worker logic: it holds the [`WorkerPool`], the model
//! manifest and the precomputed `tools/list` payload, then routes each JSON-RPC
//! request to the right response. Verb handling lives in [`super::handlers`],
//! the advertised contract in [`super::schema`], the envelope types in
//! [`super::protocol`].

use anyhow::Result;
use serde_json::{Value, json};
use std::borrow::Cow;
use std::sync::Arc;
use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::mpsc;
use tracing::{error, info, trace};

use super::protocol::{
    FrameRejection, INITIALIZE_RESULT, INTERNAL_ERROR_FRAME, JsonRpcRequest, JsonRpcResponse,
    MAX_FRAME_BYTES, code, parse_frame,
};
use super::schema::build_tools_list;
use crate::manifest::ModelManifest;
use crate::pool::WorkerPool;

#[derive(Clone)]
pub struct McpServer {
    pub(super) pool: Arc<WorkerPool>,
    pub(super) default_model: String,
    pub(super) manifest: Arc<ModelManifest>,
    /// Precomputed, immutable `tools/list` result. The manifest is never
    /// mutated after construction, so the payload is byte-identical for the
    /// process lifetime and is cloned (an `Arc` memcpy) instead of rebuilt.
    pub(super) tools_list: Arc<Value>,
}

impl McpServer {
    pub fn new(pool: WorkerPool, default_model: String) -> Self {
        // The pool owns the manifest (attached in `main.rs`); the server shares
        // the same `Arc` so a dispatch and its worker never disagree on the
        // catalog.
        let manifest = pool.manifest_arc();
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
        // even when the orchestrator never calls `collect`.
        let reaper = crate::pool::spawn_reaper((*self.pool).clone());

        let stdin = tokio::io::stdin();
        let mut stdout = tokio::io::stdout();
        let mut reader = BufReader::new(stdin);
        let mut input = Vec::new();

        info!("Mini-SWE-MCP server listening on stdio");

        let (out_tx, mut out_rx) = mpsc::channel::<String>(128);

        // Worker events: one `claude/channel` notification per state transition,
        // written through the same outbound channel as the responses so a
        // notification can never land inside a response frame.
        let events = super::events::spawn_event_stream((*self.pool).clone(), out_tx.clone());

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

        while let Some(oversized) = read_bounded_line(&mut reader, &mut input).await? {
            if oversized {
                let _ = out_tx
                    .send(FrameRejection::FrameTooLarge.into_frame())
                    .await;
                continue;
            }
            let line = std::str::from_utf8(&input)?.trim();
            if line.is_empty() {
                continue;
            }

            let server = self.clone();
            let tx = out_tx.clone();
            let owned_line = line.to_string();
            tokio::spawn(async move {
                let line = owned_line.as_str();
                let req = match parse_frame(line) {
                    Ok(req) => req,
                    Err(rejection) => {
                        error!("Rejected JSON-RPC frame: {rejection:?}");
                        let _ = tx.send(rejection.into_frame()).await;
                        return;
                    }
                };

                // JSON-RPC 2.0 §4.1: a Notification is a Request object without
                // an `id` (absent or `null`); the server MUST NOT reply to it.
                if req.id.is_none() {
                    trace!(method = %req.method, "Received notification");
                    return;
                }

                let response = server.handle_request(req, Some(tx.clone())).await;
                let frame = response
                    .to_frame()
                    .unwrap_or_else(|_| String::from(INTERNAL_ERROR_FRAME));
                let _ = tx.send(frame).await;
            });
        }

        drop(out_tx);
        reaper.abort();
        events.abort();
        let _ = stdout_task.await;

        Ok(())
    }

    /// Route one JSON-RPC request to its response envelope.
    async fn handle_request(
        &self,
        req: JsonRpcRequest,
        progress_tx: Option<mpsc::Sender<String>>,
    ) -> JsonRpcResponse {
        let id = req.id_or_null().map(|v| v.to_owned());
        match req.method.as_str() {
            "ping" => JsonRpcResponse::ok(id, json!({})),

            // The handshake document lives in a `static`; it is cloned out of
            // it per request.
            "initialize" => JsonRpcResponse::ok(id, (*INITIALIZE_RESULT).clone()),

            // The schema is immutable for the process lifetime, so it is built
            // once in `McpServer::new` and only cloned here.
            "tools/list" => JsonRpcResponse::ok(id, (*self.tools_list).clone()),

            "tools/call" => {
                let params = req.params.unwrap_or_default();
                let tool_name = params.get("name").and_then(|v| v.as_str()).unwrap_or("");
                let arguments = params.get("arguments").cloned().unwrap_or(json!({}));

                // The token is a client-supplied id: it is echoed verbatim into
                // the progress notifications.
                let progress_token = params
                    .get("_meta")
                    .and_then(|meta| meta.get("progressToken"))
                    .or_else(|| {
                        arguments
                            .get("_meta")
                            .and_then(|meta| meta.get("progressToken"))
                    })
                    .or_else(|| arguments.get("progressToken"))
                    .cloned();

                match self
                    .execute_tool_with_progress(tool_name, arguments, progress_token, progress_tx)
                    .await
                {
                    Ok(payload) => JsonRpcResponse::tool_call(id, payload),
                    Err(error) => {
                        JsonRpcResponse::err(id, code::SERVER_ERROR, Cow::Owned(error.to_string()))
                    }
                }
            }

            // `-32601` for every method the server does not expose, including
            // the MCP notifications a host may send us.
            _ => JsonRpcResponse::method_not_found(id, Cow::Owned(req.method.clone())),
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
    /// * `{ worker_id, state, logs }` on `Completed`/`Failed`
    /// * `{ worker_id, status: "needs_input", question, step, message }` when
    ///   paused waiting for steering.
    ///
    /// Progress notifications are emitted only when a `progress_token`/`tx`
    /// pair is supplied (the MCP stdio path); the plain-CLI path passes `None`,
    /// and the polling algorithm stays identical for both callers.
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
                    let log_view = self.render_logs(wid).await;
                    let mut result = json!({
                        "worker_id": wid,
                        "state": state,
                    });
                    if let serde_json::Value::Object(map) = &mut result {
                        map.extend(log_view.as_map());
                    }
                    return Ok(result);
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

/// Read one newline-delimited frame without letting a peer grow the input
/// buffer beyond the protocol limit. On overflow, discard through the next
/// newline so the following request stays in sync. `false` means the buffer
/// holds a complete line; `true` means the line was too long.
async fn read_bounded_line<R: AsyncBufRead + Unpin>(
    reader: &mut R,
    line: &mut Vec<u8>,
) -> std::io::Result<Option<bool>> {
    line.clear();
    let mut oversized = false;
    let mut seen = false;
    loop {
        let chunk = reader.fill_buf().await?;
        if chunk.is_empty() {
            return Ok(seen.then_some(oversized));
        }
        seen = true;
        let end = chunk.iter().position(|&b| b == b'\n');
        let count = end.unwrap_or(chunk.len());
        if !oversized {
            if count > MAX_FRAME_BYTES - line.len() {
                oversized = true;
                line.clear();
            } else {
                line.extend_from_slice(&chunk[..count]);
            }
        }
        reader.consume(count + usize::from(end.is_some()));
        if end.is_some() {
            return Ok(Some(oversized));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcp::schema::WORKER_ACTIONS;
    use serde_json::json;

    #[tokio::test]
    async fn oversized_stdio_line_is_discarded_without_losing_the_next_frame() {
        let valid = br#"{"id":1,"method":"ping"}"#;
        let mut input = vec![b'x'; MAX_FRAME_BYTES + 1];
        input.push(b'\n');
        input.extend_from_slice(valid);
        let mut reader = BufReader::new(input.as_slice());
        let mut line = Vec::new();

        assert_eq!(
            read_bounded_line(&mut reader, &mut line).await.unwrap(),
            Some(true)
        );
        assert!(line.is_empty(), "do not retain an oversized prefix");
        assert_eq!(
            read_bounded_line(&mut reader, &mut line).await.unwrap(),
            Some(false)
        );
        assert_eq!(line, valid, "the final unterminated frame is preserved");
        assert_eq!(
            read_bounded_line(&mut reader, &mut line).await.unwrap(),
            None
        );
    }

    #[tokio::test]
    async fn unterminated_oversized_line_is_rejected_once() {
        let input = vec![b'x'; MAX_FRAME_BYTES + 1];
        let mut reader = BufReader::new(input.as_slice());
        let mut line = Vec::new();
        assert_eq!(
            read_bounded_line(&mut reader, &mut line).await.unwrap(),
            Some(true)
        );
        assert!(line.is_empty());
        assert_eq!(
            read_bounded_line(&mut reader, &mut line).await.unwrap(),
            None
        );
    }

    #[tokio::test]
    async fn frame_ceiling_is_inclusive_for_the_stdio_reader() {
        let input = vec![b'x'; MAX_FRAME_BYTES];
        let mut reader = BufReader::new(input.as_slice());
        let mut line = Vec::new();
        assert_eq!(
            read_bounded_line(&mut reader, &mut line).await.unwrap(),
            Some(false)
        );
        assert_eq!(line.len(), MAX_FRAME_BYTES);
    }

    fn server() -> McpServer {
        McpServer::new(
            WorkerPool::new(1, "http://localhost:1".to_string(), "test-key".to_string()),
            "ninja".to_string(),
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
