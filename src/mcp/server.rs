//! The [`McpServer`] handle, its handshake, and the stdio run loop.
//!
//! The server owns no worker logic: it holds the [`WorkerPool`], the model
//! manifest and the precomputed `tools/list` payload, then routes each JSON-RPC
//! request to the right response. Verb handling lives in [`super::handlers`],
//! the advertised contract in [`super::schema`], the envelope types in
//! [`super::protocol`].
//!
//! [`McpServer::serve_connection`] is the one frame loop; [`McpServer::run_stdio`]
//! and the hub daemon's Unix sockets are both just transports feeding it.

use anyhow::Result;
use serde_json::{Value, json};
use std::borrow::Cow;
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncWrite, AsyncWriteExt, BufReader};
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

/// `clientInfo.name` the CLI sends in its `initialize` handshake.
pub const CLI_CLIENT_NAME: &str = "mini-swe-cli";

/// Owner identity shared by every CLI invocation, so a worker dispatched by one
/// `mini-swe-mcp list` is still steerable from the next one.
pub const CLI_AGENT: &str = "cli";

/// Owner identity of the in-process stdio server (`MINI_SWE_NO_DAEMON=1`), which
/// is its own only client.
pub const LOCAL_AGENT: &str = "local";

/// Owner identity of a hub connection that announced neither an agent id nor a
/// client name, qualified by the connection so two such clients never share
/// workers.
pub const ANONYMOUS_AGENT_PREFIX: &str = "connection";

/// Identity of one client connection serving MCP requests.
///
/// The hello metadata is descriptive except for `agent_id` and `admin`: the
/// first names the agent that owns this connection's workers, the second is the
/// human operator's override. Neither is authenticated — the hub accepts a
/// connection only from the same uid, which is what makes the override safe to
/// hand to a local client.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ConnectionContext {
    /// Connection id, unique per process for log correlation.
    pub id: u64,
    /// `MINI_SWE_AGENT_ID` from `hub/hello`, when the client sent one.
    pub agent_id: Option<String>,
    /// `clientInfo.name` from `initialize`, when the client sent one.
    pub client_name: Option<String>,
    /// `admin: true` from `hub/hello`: the operator may act on any worker.
    pub admin: bool,
    /// This connection is the in-process stdio server, not a hub client.
    pub local: bool,
    pub pid: Option<u32>,
    pub version: Option<String>,
    pub cwd: Option<std::path::PathBuf>,
}

impl ConnectionContext {
    /// Context for the stdio transport, which serves exactly one connection.
    pub fn stdio() -> Self {
        Self {
            id: 0,
            agent_id: None,
            client_name: None,
            admin: false,
            local: true,
            pid: None,
            version: None,
            cwd: None,
        }
    }

    /// Context for the `id`-th accepted hub connection (1-based).
    pub fn hub_connection(id: u64) -> Self {
        Self {
            id,
            agent_id: None,
            client_name: None,
            admin: false,
            local: false,
            pid: None,
            version: None,
            cwd: None,
        }
    }

    /// The agent identity that owns the workers this connection dispatches
    /// (H-3), in precedence order:
    ///
    /// 1. the `MINI_SWE_AGENT_ID` the client sent in `hub/hello`, so several
    ///    connections of one orchestrator share their workers;
    /// 2. `cli` for the CLI, whose identity is stable across invocations;
    /// 3. `<clientInfo.name>#<connection id>`, which keeps two clients of the
    ///    same host separate;
    /// 4. `local` for the in-process stdio server, and `connection#<id>` for a
    ///    hub client that announced nothing at all.
    pub fn agent(&self) -> String {
        if let Some(agent_id) = self.agent_id.as_deref().filter(|id| !id.is_empty()) {
            return agent_id.to_string();
        }
        match self.client_name.as_deref().filter(|name| !name.is_empty()) {
            Some(CLI_CLIENT_NAME) => CLI_AGENT.to_string(),
            Some(name) => format!("{name}#{}", self.id),
            None if self.local => LOCAL_AGENT.to_string(),
            None => format!("{ANONYMOUS_AGENT_PREFIX}#{}", self.id),
        }
    }

    /// Whether this connection may act on workers owned by other agents.
    pub fn is_admin(&self) -> bool {
        self.admin
    }
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
        // even when the orchestrator never calls `collect`. The hub daemon
        // starts its own once for every connection it serves.
        let reaper = crate::pool::spawn_reaper((*self.pool).clone());

        info!("Mini-SWE-MCP server listening on stdio");
        let served = self
            .serve_connection(
                BufReader::new(tokio::io::stdin()),
                tokio::io::stdout(),
                ConnectionContext::stdio(),
            )
            .await;
        reaper.abort();
        served
    }

    /// Serve one MCP connection over any byte transport.
    ///
    /// Frames are newline-delimited JSON-RPC; each request is handled on its
    /// own task while a single writer task owns `writer`, so a response can
    /// never interleave with another frame. Oversized frames are rejected with
    /// the shared `FrameTooLarge` frame and the reader resynchronizes on the
    /// next newline. Returns when the peer closes the input.
    pub async fn serve_connection<R, W>(
        &self,
        reader: R,
        writer: W,
        mut ctx: ConnectionContext,
    ) -> Result<()>
    where
        R: AsyncBufRead + Unpin,
        W: AsyncWrite + Unpin + Send + 'static,
    {
        let mut reader = reader;
        let mut writer = writer;
        let mut input = Vec::new();

        info!(connection = ctx.id, "Serving MCP connection");

        let (out_tx, mut out_rx) = mpsc::channel::<String>(128);

        // Worker events: one `claude/channel` notification per state transition,
        // written through the same outbound channel as the responses so a
        // notification can never land inside a response frame.
        let events = super::events::spawn_event_stream((*self.pool).clone(), out_tx.clone());

        // Dedicated background writer: the single owner of `writer`, so
        // frames from concurrent request tasks cannot interleave.
        let writer_task = tokio::spawn(async move {
            while let Some(msg) = out_rx.recv().await {
                if let Err(e) = writer.write_all(msg.as_bytes()).await {
                    error!(error = %e, "Failed writing to client");
                    break;
                }
                if let Err(e) = writer.flush().await {
                    error!(error = %e, "Failed flushing client stream");
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

            let req = match parse_frame(line) {
                Ok(req) => req,
                Err(rejection) => {
                    error!("Rejected JSON-RPC frame: {rejection:?}");
                    let _ = out_tx.send(rejection.into_frame()).await;
                    continue;
                }
            };
            // Process hello in wire order before snapshotting the next request's context.
            if req.id.is_none() {
                if req.method == "hub/hello" {
                    let params = req.params.unwrap_or_default();
                    ctx.agent_id = params["agent_id"].as_str().map(str::to_owned);
                    ctx.admin = params["admin"].as_bool().unwrap_or(false);
                    ctx.pid = params["pid"].as_u64().and_then(|pid| u32::try_from(pid).ok());
                    ctx.version = params["version"].as_str().map(str::to_owned);
                    ctx.cwd = params["cwd"].as_str().map(std::path::PathBuf::from)
                        .filter(|cwd| cwd.is_absolute());
                }
                trace!(method = %req.method, "Received notification");
                continue;
            }
            // The handshake's `clientInfo.name` is the connection's agent
            // identity, so it is read in wire order here too: a request sent
            // after `initialize` must already see the name (H-3).
            if req.method == "initialize" {
                ctx.client_name = req
                    .params
                    .as_ref()
                    .and_then(|params| params["clientInfo"]["name"].as_str())
                    .map(str::to_owned);
            }
            let server = self.clone();
            let tx = out_tx.clone();
            let ctx = ctx.clone();
            tokio::spawn(async move {
                let response = server.handle_request(req, ctx, Some(tx.clone())).await;
                let frame = response
                    .to_frame()
                    .unwrap_or_else(|_| String::from(INTERNAL_ERROR_FRAME));
                let _ = tx.send(frame).await;
            });
        }

        drop(out_tx);
        events.abort();
        let _ = writer_task.await;

        Ok(())
    }

    /// Route one JSON-RPC request to its response envelope.
    async fn handle_request(
        &self,
        req: JsonRpcRequest,
        ctx: ConnectionContext,
        progress_tx: Option<mpsc::Sender<String>>,
    ) -> JsonRpcResponse {
        trace!(connection = ctx.id, method = %req.method, "Dispatching JSON-RPC request");
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
                    .execute_tool_in_context(tool_name, arguments, progress_token, progress_tx, &ctx)
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

    /// The pool this server dispatches into.
    ///
    /// The hub daemon shares one server (hence one pool) across every
    /// connection and needs the pool for its own lifecycle decisions.
    pub fn pool(&self) -> &WorkerPool {
        &self.pool
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

    /// Execute a tool call on behalf of the connection `ctx` describes.
    ///
    /// Ownership (H-3) is decided from this context, so every caller other
    /// than the stdio transport — the CLI through the hub, a second
    /// orchestrator — reaches the verb table through here.
    pub async fn execute_tool_for(
        &self,
        name: &str,
        args: Value,
        ctx: &ConnectionContext,
    ) -> Result<Value> {
        self.execute_tool_in_context(name, args, None, None, ctx).await
    }

    /// Execute a tool call, optionally streaming `notifications/progress`.
    pub async fn execute_tool_with_progress(
        &self,
        name: &str,
        args: Value,
        progress_token: Option<Value>,
        progress_tx: Option<mpsc::Sender<String>>,
    ) -> Result<Value> {
        self.execute_tool_in_context(name, args, progress_token, progress_tx, &ConnectionContext::stdio()).await
    }

    async fn execute_tool_in_context(
        &self, name: &str, args: Value, progress_token: Option<Value>,
        progress_tx: Option<mpsc::Sender<String>>, ctx: &ConnectionContext,
    ) -> Result<Value> {
        if name != "worker" {
            anyhow::bail!("Unknown tool: '{name}'. Only 'worker' is supported.");
        }

        let action = args.get("action").and_then(|v| v.as_str()).unwrap_or("");

        self.dispatch(action, &args, progress_token.as_ref(), progress_tx.as_ref(), ctx)
            .await
    }

    /// Wait indefinitely for a worker's next event.
    ///
    /// Thin wrapper over [`McpServer::await_worker_result_until`] for the
    /// callers that have no client deadline of their own.
    pub async fn await_worker_result(
        &self,
        wid: &str,
        max_turns: usize,
        token: Option<&Value>,
        tx: Option<&mpsc::Sender<String>>,
    ) -> Result<Value> {
        self.await_worker_result_until(wid, max_turns, None, token, tx)
            .await
    }

    /// Wait on a worker until it finishes, fails, or pauses for orchestrator
    /// input.
    ///
    /// Returns the terminal payload:
    /// * `{ worker_id, state, logs }` on `Completed`/`Failed`
    /// * `{ worker_id, status: "needs_input", question, step, message }` when
    ///   paused waiting for steering
    /// * `{ worker_id, status: "still_running", step, last_command }` when the
    ///   optional `timeout` deadline expires first
    ///
    /// The deadline exists for hosts that abort a tool call of their own
    /// accord: rather than being cut off, the agent gets the worker's current
    /// step and can simply call again. `None` waits indefinitely.
    ///
    /// Progress notifications are emitted only when a `progress_token`/`tx`
    /// pair is supplied (the MCP stdio path); the plain-CLI path passes `None`,
    /// and the waiting algorithm stays identical for both callers. While the
    /// worker runs, the current step is re-sent as a heartbeat every
    /// [`PROGRESS_HEARTBEAT_INTERVAL`](Self::PROGRESS_HEARTBEAT_INTERVAL) so a
    /// step that never ends cannot be mistaken for an idle call.
    ///
    /// The wait itself is event-driven: the pool bumps a generation counter on
    /// every worker state change and this loop sleeps on that subscription, so
    /// a step, a pause, a resume or a terminal state is observed as it happens
    /// instead of on a fixed tick. The only coarse tick left is
    /// [`CROSS_PROCESS_TICK`](Self::CROSS_PROCESS_TICK), for a worker this
    /// process does not own (another `mini-swe-mcp` process, or
    /// `MINI_SWE_NO_DAEMON` mode): its state changes are invisible to the
    /// subscription, so the registry has to be re-read.
    pub async fn await_worker_result_until(
        &self,
        wid: &str,
        max_turns: usize,
        timeout: Option<std::time::Duration>,
        token: Option<&Value>,
        tx: Option<&mpsc::Sender<String>>,
    ) -> Result<Value> {
        // A deadline too large to represent is no deadline at all; `checked_add`
        // keeps an absurd `timeout_secs` from panicking the wait.
        let deadline = timeout.and_then(|timeout| std::time::Instant::now().checked_add(timeout));
        let mut last_reported_step = 0;
        let mut last_reported_at = std::time::Instant::now();
        let mut last_progress: Option<crate::pool::WorkerProgress> = None;
        // Subscribed before the first read, so a change landing between the
        // read and the sleep is still seen by `changed()`.
        let mut changes = self.pool.subscribe_changes();
        // A worker owned by another process never bumps this pool's counter, so
        // its wait falls back to re-reading the registry on a coarse tick.
        let tick = if self.pool.worker_progress(wid).await.is_some() {
            Self::HEARTBEAT_TICK
        } else {
            Self::CROSS_PROCESS_TICK
        };
        loop {
            // H-5: read the lightweight progress snapshot. It never clones the
            // (potentially multi-megabyte) `diff`/`summary`/`artifacts` that a
            // `get_worker_state` clone would copy on every tick.
            if let Some(progress) = self.pool.worker_progress(wid).await {
                last_progress = Some(progress);
            }
            let Some(progress) = last_progress.as_ref() else {
                Self::wait_for_change(&mut changes, tick).await;
                continue;
            };
            match progress.phase {
                crate::pool::WorkerPhase::Running => {
                    let step = progress.step;
                    let last_command = progress.last_command.as_deref().unwrap_or("");
                    if step > last_reported_step {
                        last_reported_step = step;
                        last_reported_at = std::time::Instant::now();
                        Self::emit_progress(
                            tx,
                            token,
                            step,
                            max_turns,
                            format!("Step {step}/{max_turns}: {last_command}"),
                        )
                        .await;
                    } else if last_reported_at.elapsed() >= Self::PROGRESS_HEARTBEAT_INTERVAL {
                        // Heartbeat: one step can outlast the client's idle
                        // timeout on its own, so the unchanged step is resent.
                        last_reported_at = std::time::Instant::now();
                        Self::emit_progress(
                            tx,
                            token,
                            step,
                            max_turns,
                            format!("Step {step}/{max_turns}: {last_command} (still running)"),
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
                        "owner": self.owner_of(wid).await,
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
                        "owner": self.owner_of(wid).await,
                        "status": "needs_input",
                        "question": question,
                        "step": step,
                        "message": "Worker is paused waiting for orchestrator steering."
                    }));
                }
            }

            if deadline.is_some_and(|deadline| std::time::Instant::now() >= deadline) {
                let step = progress.step;
                Self::emit_progress(
                    tx,
                    token,
                    step,
                    max_turns,
                    format!("Step {step}/{max_turns}: still running, call again to keep waiting"),
                )
                .await;
                return Ok(json!({
                    "worker_id": wid,
                    "owner": self.owner_of(wid).await,
                    "status": "still_running",
                    "step": step,
                    "last_command": progress.last_command,
                }));
            }
            Self::wait_for_change(&mut changes, tick).await;
        }
    }

    /// Sleep until the pool reports a worker state change, or `tick` elapses.
    ///
    /// The tick is a safety net rather than the primary wake-up: it bounds how
    /// long a missed notification can stall the wait, and it is clamped to the
    /// heartbeat interval so a step that never ends still reports in.
    async fn wait_for_change(changes: &mut tokio::sync::watch::Receiver<u64>, tick: Duration) {
        let wait = tick.min(Self::PROGRESS_HEARTBEAT_INTERVAL);
        tokio::select! {
            // A new generation means some worker moved; the caller re-reads the
            // snapshot it cares about and decides whether it was the one.
            _ = changes.changed() => {}
            _ = tokio::time::sleep(wait) => {}
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
