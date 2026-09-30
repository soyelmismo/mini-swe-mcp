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
use tokio::sync::{Mutex, RwLock, mpsc, watch};
use tracing::{error, info, trace};

use super::protocol::{
    FrameRejection, INITIALIZE_RESULT, INTERNAL_ERROR_FRAME, JsonRpcRequest, JsonRpcResponse,
    MAX_FRAME_BYTES, code, parse_frame,
};
use super::schema::build_tools_list;
use crate::hub::WatchTokens;
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
    pub(super) hub_events: Arc<Mutex<super::events::EventRouter>>,
    hub_enabled: Arc<std::sync::atomic::AtomicBool>,
    shutdown: watch::Sender<bool>,
    daemon_version: Arc<str>,
    /// `hub/hello` reply: this build's id and clock, stamped by `build.rs`, so a
    /// client can tell a rebuilt binary from the daemon already serving.
    daemon_build: Arc<Value>,
    hub_shutdown_gate: Arc<RwLock<bool>>,
}

/// `clientInfo.name` the CLI sends in its `initialize` handshake.
pub const CLI_CLIENT_NAME: &str = "mini-swe-cli";

/// Owner identity of a CLI invocation whose host process could not be named
/// (no readable `/proc` ancestry), so every such invocation still shares one
/// identity and a worker stays steerable from the next call.
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
    /// The agent's host process from `hub/hello`, when the client sent one:
    /// `host:<comm>:<pid>:<starttime>` (see [`crate::hub::identity`]). One
    /// host process is one agent session, so the MCP connection and the shell
    /// commands it spawned share their workers.
    pub host_id: Option<String>,
    /// The session inside the host process, from `hub/hello` or from the
    /// `_meta.sessionID` of the call being served (see
    /// [`crate::hub::identity::qualify`]). A host that runs several sessions —
    /// opencode v2's tabs, over one shared connection — is not one agent, so
    /// the session qualifies the host instead of being cached on the
    /// connection.
    pub session_id: Option<String>,
    /// The identity a valid `MINI_SWE_WATCH_TOKEN` named, when the client
    /// presented one in `hub/hello`: the caller is then exactly that identity.
    pub token_identity: Option<String>,
    /// `clientInfo.name` from `initialize`, when the client sent one.
    pub client_name: Option<String>,
    /// `admin: true` from `hub/hello`: the operator may act on any worker.
    pub admin: bool,
    /// This connection is the in-process stdio server, not a hub client.
    pub local: bool,
    pub pid: Option<u32>,
    pub version: Option<String>,
    pub cwd: Option<std::path::PathBuf>,
    /// The daemon's watch-token store, so a dispatch answer can carry the
    /// `watch_command` that binds a shell to this identity. `None` for the
    /// in-process stdio server, which has no hub directory to keep tokens in.
    pub watch_tokens: Option<Arc<WatchTokens>>,
}

impl ConnectionContext {
    /// Context for the stdio transport, which serves exactly one connection.
    pub fn stdio() -> Self {
        Self {
            id: 0,
            agent_id: None,
            host_id: None,
            session_id: None,
            token_identity: None,
            client_name: None,
            admin: false,
            local: true,
            pid: None,
            version: None,
            cwd: None,
            watch_tokens: None,
        }
    }

    /// Context for the `id`-th accepted hub connection (1-based).
    pub fn hub_connection(id: u64) -> Self {
        Self {
            id,
            agent_id: None,
            host_id: None,
            session_id: None,
            token_identity: None,
            client_name: None,
            admin: false,
            local: false,
            pid: None,
            version: None,
            cwd: None,
            watch_tokens: None,
        }
    }

    /// Install the daemon's watch-token store on this connection.
    pub fn with_watch_tokens(mut self, store: Arc<WatchTokens>) -> Self {
        self.watch_tokens = Some(store);
        self
    }

    /// The watch-token store this connection mints from, when it has one.
    pub fn token_store(&self) -> Option<&Arc<WatchTokens>> {
        self.watch_tokens.as_ref()
    }

    /// The agent identity that owns the workers this connection dispatches
    /// (H-3), in precedence order:
    ///
    /// 1. the `MINI_SWE_AGENT_ID` the client sent in `hub/hello`, so several
    ///    connections of one orchestrator share their workers;
    /// 2. the identity a valid `MINI_SWE_WATCH_TOKEN` named, which is how a
    ///    shell that cannot know its session acts as the session that
    ///    dispatched a worker;
    /// 3. the agent's host process qualified by the session running inside it,
    ///    both of which the client sent in `hub/hello`: one host is one
    ///    session, so the MCP connection and the shell commands it spawned
    ///    share their workers while two hosts — and two sessions of one host —
    ///    never do, and the identity survives a reconnect or a hub restart;
    /// 4. `cli` for a CLI whose host could not be named, whose identity is
    ///    therefore stable across invocations;
    /// 5. `<clientInfo.name>#<connection id>`, which keeps two clients of the
    ///    same host separate;
    /// 6. `local` for the in-process stdio server, and `connection#<id>` for a
    ///    hub client that announced nothing at all.
    pub fn agent(&self) -> String {
        if let Some(agent_id) = self.agent_id.as_deref().filter(|id| !id.is_empty()) {
            return agent_id.to_string();
        }
        if let Some(identity) = self.token_identity.as_deref().filter(|id| !id.is_empty()) {
            return identity.to_string();
        }
        if let Some(identity) =
            crate::hub::identity::qualify(self.host_id.as_deref(), self.session_id.as_deref())
        {
            return identity;
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

/// This binary's build identity in the shape `hub/hello` carries it: `id` names
/// the exact build, `ts` is its comparable clock (`hub::client` sends the same).
fn build_identity() -> Value {
    json!({
        "id": env!("MINI_SWE_BUILD_ID"),
        "ts": env!("MINI_SWE_BUILD_TS").parse::<u64>().unwrap_or_default(),
    })
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
            hub_events: Arc::new(Mutex::new(super::events::EventRouter::default())),
            hub_enabled: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            shutdown: watch::channel(false).0,
            daemon_version: Arc::from(env!("CARGO_PKG_VERSION")),
            daemon_build: Arc::new(build_identity()),
            hub_shutdown_gate: Arc::new(RwLock::new(false)),
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
        let hub = self.hub_enabled.load(std::sync::atomic::Ordering::Acquire);
        let (context_tx, context_rx) = watch::channel(ctx.clone());
        let events = (!hub).then(|| {
            super::events::spawn_event_stream((*self.pool).clone(), out_tx.clone(), context_rx)
        });

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

        let mut requested_shutdown = false;
        let served = async {
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
                // Identity changes and hub control requests are handled in wire order.
                let handshake = req.method == "hub/hello" || req.method == "initialize";
                if req.method == "hub/hello" {
                    let params = req.params.as_ref().cloned().unwrap_or_default();
                    ctx.agent_id = params["agent_id"].as_str().map(str::to_owned);
                    ctx.host_id = params["host_id"].as_str().map(str::to_owned);
                    ctx.session_id = params["session_id"]
                        .as_str()
                        .map(str::to_owned)
                        .filter(|session| !session.is_empty());
                    // A watch token is the caller's whole identity: it is what
                    // a shell that cannot know its session presents, so it
                    // outranks the host and never carries the operator's
                    // `admin` override with it.
                    let watch_token = params["watch_token"].as_str();
                    ctx.token_identity = watch_token
                        .and_then(|token| ctx.token_store().map(|store| (token, store)))
                        .and_then(|(token, store)| store.identity_of(token));
                    let operator_admin = params["admin"].as_bool().unwrap_or(false);
                    ctx.admin = operator_admin && ctx.token_identity.is_none();
                    ctx.pid = params["pid"]
                        .as_u64()
                        .and_then(|pid| u32::try_from(pid).ok());
                    ctx.version = params["version"].as_str().map(str::to_owned);
                    ctx.cwd = params["cwd"]
                        .as_str()
                        .map(std::path::PathBuf::from)
                        .filter(|cwd| cwd.is_absolute());
                } else if req.method == "initialize" {
                    ctx.client_name = req
                        .params
                        .as_ref()
                        .and_then(|params| params["clientInfo"]["name"].as_str())
                        .map(str::to_owned);
                }
                if handshake {
                    if req.id.is_some() {
                        let response = if req.method == "hub/hello" {
                            JsonRpcResponse::ok(
                                req.id_or_null().map(ToOwned::to_owned),
                                json!({
                                    "version": &*self.daemon_version,
                                    "build": &*self.daemon_build,
                                    "busy": self.pool.active_worker_count().await > 0,
                                }),
                            )
                        } else {
                            self.handle_request(req, ctx.clone(), None).await
                        };
                        let _ = out_tx.send(response.to_frame()?).await;
                    }
                    if hub
                        && (ctx.agent_id.is_some()
                            || ctx.host_id.is_some()
                            || ctx.client_name.is_some()
                            || ctx.is_admin())
                    {
                        self.hub_events.lock().await.register(&ctx, out_tx.clone());
                    }
                    context_tx.send_replace(ctx.clone());
                    continue;
                }
                if req.id.is_none() {
                    trace!(method = %req.method, "Received notification");
                    continue;
                }
                if hub && req.method == "hub/shutdown" {
                    let mut busy = self.pool.active_worker_count().await > 0;
                    if !busy {
                        // Serialize the idle decision with dispatch/revision admission.
                        let mut stopping = self.hub_shutdown_gate.write().await;
                        busy = self.pool.active_worker_count().await > 0;
                        if !busy {
                            *stopping = true;
                        }
                    }
                    let id = req.id_or_null().map(ToOwned::to_owned);
                    let response = if busy {
                        JsonRpcResponse::err(id, code::SERVER_ERROR, Cow::Borrowed("Hub is busy"))
                    } else {
                        JsonRpcResponse::ok(
                            id,
                            json!({"version": &*self.daemon_version, "busy": false}),
                        )
                    };
                    if !busy {
                        // Drain through the shutdown reply before the daemon closes sockets.
                        let _ = out_tx.send(response.to_frame()?).await;
                        requested_shutdown = true;
                        break;
                    }
                    let _ = out_tx.send(response.to_frame()?).await;
                    continue;
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

            Ok::<(), anyhow::Error>(())
        }
        .await;
        drop(out_tx);
        if let Some(events) = events {
            events.abort();
        }
        self.hub_events.lock().await.remove(ctx.id);
        let _ = writer_task.await;
        if requested_shutdown {
            self.shutdown.send_replace(true);
        }
        served
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
            "hub/watch" | "hub/watch/ack" => {
                match super::events::watch_request(
                    &self.pool,
                    &self.hub_events,
                    &ctx,
                    req.params.unwrap_or_default(),
                    req.method == "hub/watch/ack",
                )
                .await
                {
                    Ok(value) => JsonRpcResponse::ok(id, value),
                    Err(error) => {
                        JsonRpcResponse::err(id, code::SERVER_ERROR, Cow::Owned(error.to_string()))
                    }
                }
            }

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

                // The session this call belongs to, which is not necessarily
                // the connection's: opencode v2 sends `_meta.sessionID` on
                // every call of one shared connection, one per tab. It is read
                // per call and never cached on the connection, so the next call
                // of the same connection may name another session.
                let session = params
                    .get("_meta")
                    .and_then(|meta| meta.get("sessionID"))
                    .or_else(|| {
                        arguments
                            .get("_meta")
                            .and_then(|meta| meta.get("sessionID"))
                    })
                    .and_then(Value::as_str)
                    .map(str::to_owned)
                    .filter(|session| !session.is_empty());
                let call_ctx;
                let ctx = match session {
                    Some(session) => {
                        call_ctx = ConnectionContext {
                            session_id: Some(session),
                            ..ctx.clone()
                        };
                        &call_ctx
                    }
                    None => &ctx,
                };

                match self
                    .execute_tool_in_context(
                        tool_name,
                        arguments,
                        progress_token,
                        progress_tx,
                        ctx,
                    )
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

    /// Start the daemon's single watcher before accepting any connections.
    pub async fn start_hub_events(&self) -> tokio::task::JoinHandle<()> {
        self.hub_enabled
            .store(true, std::sync::atomic::Ordering::Release);
        super::events::spawn_hub_events((*self.pool).clone(), self.hub_events.clone()).await
    }

    /// Subscribe to an idle shutdown requested through the hub transport.
    pub fn subscribe_shutdown(&self) -> watch::Receiver<bool> {
        self.shutdown.subscribe()
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
        self.execute_tool_in_context(name, args, None, None, ctx)
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
        self.execute_tool_in_context(
            name,
            args,
            progress_token,
            progress_tx,
            &ConnectionContext::stdio(),
        )
        .await
    }

    async fn execute_tool_in_context(
        &self,
        name: &str,
        args: Value,
        progress_token: Option<Value>,
        progress_tx: Option<mpsc::Sender<String>>,
        ctx: &ConnectionContext,
    ) -> Result<Value> {
        if name != "worker" {
            anyhow::bail!("Unknown tool: '{name}'. Only 'worker' is supported.");
        }

        let action = args.get("action").and_then(|v| v.as_str()).unwrap_or("");

        self.dispatch(
            action,
            &args,
            progress_token.as_ref(),
            progress_tx.as_ref(),
            ctx,
        )
        .await
    }

    pub(super) async fn admit_worker(&self) -> Result<tokio::sync::RwLockReadGuard<'_, bool>> {
        let admission = self.hub_shutdown_gate.read().await;
        if *admission {
            anyhow::bail!("Hub is shutting down");
        }
        Ok(admission)
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
                    // The terminal payload tells the orchestrator what to do
                    // next: review the branch, and steer this same worker when
                    // it needs corrections.
                    let next_step = state.as_ref().map(|state| {
                        crate::pool::next_step_for(crate::pool::terminal_branch(state).as_deref())
                    });
                    let mut result = json!({
                        "worker_id": wid,
                        "owner": self.owner_of(wid).await,
                        "state": state,
                        "next_step": next_step,
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

    /// A `steer` answer is immediate, so the admission guard it took to queue
    /// the guidance must be released by the time the reply is built: holding it
    /// would wedge the pool's admission for every later dispatch.
    #[tokio::test]
    async fn steer_returns_immediately_and_releases_its_admission_guard() {
        use crate::pool::{LogBuffer, WorkerMetrics, WorkerRecord, WorkerState};
        let server = server();
        server
            .pool
            .__test_insert_worker(WorkerRecord {
                id: "h4-steer-gate".to_string(),
                task: "probe".to_string(),
                model: "test".to_string(),
                owner: LOCAL_AGENT.to_string(),
                state: WorkerState::Running {
                    step: 1,
                    last_command: "probe".to_string(),
                    started_at: 0,
                },
                metrics: WorkerMetrics::default(),
                logs: LogBuffer::new(),
                pending_steer: Vec::new(),
                resume_tx: None,
                handle: None,
                revision: 0,
            })
            .await;
        let reply = tokio::time::timeout(
            Duration::from_secs(3),
            server.execute_tool_with_progress(
                "worker",
                json!({"action": "steer", "worker_id": "h4-steer-gate", "message": "go"}),
                None,
                None,
            ),
        )
        .await
        .expect("steer must not block")
        .expect("steer must succeed");
        assert_eq!(reply["status"], "steered");
        assert!(
            server.hub_shutdown_gate.try_write().is_ok(),
            "steer must not retain admission"
        );
    }

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

    /// A `hub/hello` agent id wins: it is how several connections of one
    /// orchestrator end up sharing their workers.
    #[test]
    fn a_hello_agent_id_is_the_connection_identity() {
        let mut ctx = ConnectionContext::hub_connection(3);
        ctx.client_name = Some(CLI_CLIENT_NAME.to_string());
        ctx.agent_id = Some("orchestrator-7".to_string());
        assert_eq!(ctx.agent(), "orchestrator-7");
    }

    /// A host identity outranks the client name, so the MCP connection an agent
    /// dispatches over and the `mini-swe-mcp` calls its shell makes are one
    /// agent — while a second host is a second agent.
    #[test]
    fn one_host_identity_is_shared_by_a_connection_and_its_shell() {
        let host = "host:claude:4242:9182734";
        let mut connection = ConnectionContext::hub_connection(7);
        connection.client_name = Some("claude-code".to_string());
        connection.host_id = Some(host.to_string());
        let mut shell = ConnectionContext::hub_connection(8);
        shell.client_name = Some(CLI_CLIENT_NAME.to_string());
        shell.host_id = Some(host.to_string());
        assert_eq!(connection.agent(), host);
        assert_eq!(shell.agent(), host);

        let mut other_host = ConnectionContext::hub_connection(9);
        other_host.client_name = Some("claude-code".to_string());
        other_host.host_id = Some("host:opencode:5253:9182735".to_string());
        assert_ne!(other_host.agent(), connection.agent());
    }

    /// The explicit override still outranks the host identity.
    #[test]
    fn an_explicit_agent_id_outranks_the_host_identity() {
        let mut ctx = ConnectionContext::hub_connection(3);
        ctx.host_id = Some("host:claude:4242:9182734".to_string());
        ctx.agent_id = Some("orchestrator-7".to_string());
        assert_eq!(ctx.agent(), "orchestrator-7");
    }

    /// A CLI whose host process could not be named keeps the `cli` identity,
    /// which is what lets `mini-swe-mcp list` see the worker an earlier
    /// `dispatch` started.
    #[test]
    fn every_cli_connection_shares_the_cli_identity() {
        let mut first = ConnectionContext::hub_connection(1);
        first.client_name = Some(CLI_CLIENT_NAME.to_string());
        let mut second = ConnectionContext::hub_connection(99);
        second.client_name = Some(CLI_CLIENT_NAME.to_string());
        assert_eq!(first.agent(), CLI_AGENT);
        assert_eq!(second.agent(), CLI_AGENT);
    }

    /// Any other client is qualified by its connection id, so two orchestrators
    /// of the same host never share workers by accident.
    #[test]
    fn another_client_is_scoped_to_its_connection() {
        let mut ctx = ConnectionContext::hub_connection(12);
        ctx.client_name = Some("claude-code".to_string());
        assert_eq!(ctx.agent(), "claude-code#12");
    }

    /// The in-process stdio server is its own only client; a hub connection that
    /// announced nothing falls back to its connection id.
    #[test]
    fn an_unannounced_connection_falls_back_to_its_transport() {
        assert_eq!(ConnectionContext::stdio().agent(), LOCAL_AGENT);
        assert_eq!(
            ConnectionContext::hub_connection(4).agent(),
            format!("{ANONYMOUS_AGENT_PREFIX}#4")
        );
    }

    /// Admin is opt-in per connection: only an explicit `admin: true` in the
    /// hello lifts the ownership check.
    #[test]
    fn admin_is_opt_in() {
        assert!(!ConnectionContext::stdio().is_admin());
        let mut ctx = ConnectionContext::hub_connection(1);
        assert!(!ctx.is_admin());
        ctx.admin = true;
        assert!(ctx.is_admin());
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
