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
use tracing::{error, info, trace, warn};

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
    pub(super) auto_consolidate:
        Arc<std::sync::Mutex<Option<Arc<crate::hub::auto_consolidate::AutoConsolidate>>>>,
    pub(super) default_model: String,
    pub(super) manifest: Arc<ModelManifest>,
    /// Per-agent worker cap, `MAX_WORKERS_PER_AGENT`; `0` (the default) is
    /// unlimited. Resolved once, at construction, like every other
    /// environment-derived setting.
    pub(super) max_workers_per_agent: usize,
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
    /// Closed by the hub daemon while startup recovery runs; see
    /// [`RecoveryGate`].
    recovery: Arc<RecoveryGate>,
    /// Set while a newer client has asked this daemon to hand over: the
    /// instant the graceful stop is due, quiet moment or deadline.
    handover: Arc<std::sync::Mutex<Option<std::time::Instant>>>,
}

/// How often a pending handover looks for a quiet moment.
const HANDOVER_POLL: Duration = Duration::from_millis(200);

/// A one-way readiness gate for hub startup recovery.
///
/// Worker-state requests wait on it so a client that connected before the
/// daemon finished recovering orphaned workers observes the recovered pool
/// rather than a half-recovered one. It is open from construction: only the
/// hub daemon closes it (see [`McpServer::begin_recovery`]), so the stdio
/// transport and in-process callers never wait.
struct RecoveryGate {
    ready: watch::Sender<bool>,
}

impl RecoveryGate {
    /// A gate that is already open, so a non-daemon caller never waits.
    fn open() -> Self {
        Self {
            ready: watch::channel(true).0,
        }
    }

    /// Close the gate: worker-state requests wait until [`Self::finish`].
    fn begin(&self) {
        self.ready.send_replace(false);
    }

    /// Open the gate and release every waiter.
    fn finish(&self) {
        self.ready.send_replace(true);
    }

    /// Wait until the gate is open.
    async fn wait(&self) {
        let mut ready = self.ready.subscribe();
        let _ = ready.wait_for(|open| *open).await;
    }
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
    /// The client's build identity from `hub/hello`, which is what tells a
    /// rebuilt client from the daemon it dialed at the same release version.
    pub build: Option<Value>,
    pub cwd: Option<std::path::PathBuf>,
    /// The daemon's watch-token store, so a dispatch answer can carry the
    /// `watch_command` that binds a shell to this identity. `None` for the
    /// in-process stdio server, which has no hub directory to keep tokens in.
    pub watch_tokens: Option<Arc<WatchTokens>>,
    /// The dispatcher's ambient environment, filtered by the sandbox's secret
    /// filter on the client and again here. It is what the differential verify
    /// gate layers on top of the canonical sandbox environment, so a suite that
    /// only passes in the orchestrator's shell is caught by the worker itself.
    pub client_env: Vec<(String, String)>,
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
            build: None,
            cwd: None,
            watch_tokens: None,
            client_env: Vec::new(),
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
            build: None,
            cwd: None,
            watch_tokens: None,
            client_env: Vec::new(),
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
            auto_consolidate: Arc::new(std::sync::Mutex::new(None)),
            default_model,
            max_workers_per_agent: Self::max_workers_per_agent_from_env(),
            manifest,
            tools_list,
            hub_events: Arc::new(Mutex::new(super::events::EventRouter::default())),
            hub_enabled: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            shutdown: watch::channel(false).0,
            daemon_version: Arc::from(env!("CARGO_PKG_VERSION")),
            daemon_build: Arc::new(build_identity()),
            hub_shutdown_gate: Arc::new(RwLock::new(false)),
            recovery: Arc::new(RecoveryGate::open()),
            handover: Arc::new(std::sync::Mutex::new(None)),
        }
    }

    /// The per-agent worker cap this process resolves at construction.
    ///
    /// Read once, from the environment, the way the binary sets it up; a
    /// caller that needs a different cap names it with
    /// [`McpServer::with_max_workers_per_agent`] instead of setting the
    /// variable for the whole process.
    pub(in crate::mcp) fn max_workers_per_agent_from_env() -> usize {
        crate::config::env_parse("MAX_WORKERS_PER_AGENT").unwrap_or(0)
    }

    /// Name the per-agent worker cap this server enforces.
    ///
    /// `0` (the default) is unlimited. The binary keeps
    /// `MAX_WORKERS_PER_AGENT`; this setter is the seam an in-process caller
    /// uses to exercise the cap without mutating the environment every other
    /// test in the same process would inherit.
    pub fn with_max_workers_per_agent(mut self, cap: usize) -> Self {
        self.max_workers_per_agent = cap;
        self
    }

    /// Serve MCP over stdin/stdout until the client closes the input.
    pub async fn run_stdio(&self) -> Result<()> {
        // Background reaper: bounds the memory held by terminal worker records
        // even when the orchestrator never calls `collect`. The hub daemon
        // starts its own once for every connection it serves.
        let reaper = crate::pool::spawn_reaper((*self.pool).clone(), crate::hub::hub_dir().ok());

        info!("Mini-SWE-MCP server listening on stdio");
        // The stdio transport has no `hub/hello` handshake, so the ambient
        // snapshot is taken here, from this process, with the same filter.
        let mut stdio_ctx = ConnectionContext::stdio();
        stdio_ctx.client_env = crate::agent::env::ambient_environment_snapshot();
        let served = self
            .serve_connection(
                BufReader::new(tokio::io::stdin()),
                tokio::io::stdout(),
                stdio_ctx,
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

        // Cancelling a connection must also release its detached writer socket.
        struct WriterGuard(tokio::task::AbortHandle);
        impl Drop for WriterGuard {
            fn drop(&mut self) {
                self.0.abort();
            }
        }
        let _writer_guard = WriterGuard(writer_task.abort_handle());
        let mut requests = tokio::task::JoinSet::new();
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
                    ctx.build = params.get("build").cloned().filter(Value::is_object);
                    ctx.cwd = params["cwd"]
                        .as_str()
                        .map(std::path::PathBuf::from)
                        .filter(|cwd| cwd.is_absolute());
                    // The dispatcher's ambient environment, re-filtered here:
                    // the client already dropped credential-bearing names, and
                    // the daemon never trusts a handshake to have done it.
                    ctx.client_env = crate::hub::client::decode_ambient_env(
                        params.get("ambient_env").unwrap_or(&Value::Null),
                    );
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
                if hub && req.method == "hub/handover" {
                    // A newer client asks a busy daemon to step aside. The
                    // daemon keeps serving and stops at the first quiet
                    // moment, so nothing in flight is cut off; the deadline
                    // bounds a daemon that never goes quiet.
                    let id = req.id_or_null().map(ToOwned::to_owned);
                    let requested = req
                        .params
                        .as_ref()
                        .and_then(|params| params["deadline_secs"].as_u64());
                    let deadline = crate::hub::client::handover_deadline(requested);
                    let response = if !crate::hub::client::supersedes(
                        ctx.version.as_deref().unwrap_or(""),
                        ctx.build.as_ref().unwrap_or(&Value::Null),
                        &self.daemon_version,
                        &self.daemon_build,
                    ) {
                        JsonRpcResponse::err(
                            id,
                            code::SERVER_ERROR,
                            Cow::Borrowed("Client is not newer than hub"),
                        )
                    } else {
                        self.begin_handover(deadline);
                        JsonRpcResponse::ok(
                            id,
                            json!({
                                "version": &*self.daemon_version,
                                "pending": true,
                                "busy": self.pool.active_worker_count().await > 0,
                                "deadline_secs": deadline.as_secs(),
                            }),
                        )
                    };
                    let _ = out_tx.send(response.to_frame()?).await;
                    continue;
                }
                let server = self.clone();
                let tx = out_tx.clone();
                let ctx = ctx.clone();
                while requests.try_join_next().is_some() {}
                requests.spawn(async move {
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
        requests.abort_all();
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
        // Startup recovery runs concurrently with accepting connections, so a
        // request that reads worker state waits for it; handshakes and
        // liveness do not.
        if needs_recovery(&req) {
            self.recovery.wait().await;
        }
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
                    .execute_tool_in_context(tool_name, arguments, progress_token, progress_tx, ctx)
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
    ///
    /// `store_dir` is the hub directory whose persisted acknowledged watch
    /// positions are loaded first, so a restarted daemon does not replay events
    /// the owner already acknowledged. `None` (stdio) keeps the store in memory.
    pub async fn start_hub_events(
        &self,
        store_dir: Option<&std::path::Path>,
    ) -> tokio::task::JoinHandle<()> {
        self.hub_enabled
            .store(true, std::sync::atomic::Ordering::Release);
        if let Some(dir) = store_dir {
            self.hub_events.lock().await.load_ack_store(dir);
        }
        super::events::spawn_hub_events((*self.pool).clone(), self.hub_events.clone()).await
    }

    /// Subscribe to an idle shutdown requested through the hub transport.
    pub fn subscribe_shutdown(&self) -> watch::Receiver<bool> {
        self.shutdown.subscribe()
    }

    /// Retire a set of already-landed workers from every long-lived view.
    ///
    /// The one place the two halves of a retirement meet: each id is forgotten
    /// through the event router (so its queued events can never replay and its
    /// acknowledged positions leave memory and disk together) and its live
    /// record is dropped from the pool (so `list` stops showing it). Ids are
    /// de-duplicated first, because a merge reports its own round *and* the
    /// sweep reports what it reached, which overlap.
    pub(crate) async fn retire_and_forget<I>(&self, ids: I)
    where
        I: IntoIterator<Item = String>,
    {
        let mut ids: Vec<String> = ids.into_iter().collect();
        ids.sort();
        ids.dedup();
        for id in &ids {
            self.forget_retired_worker(id).await;
        }
        self.pool.forget_retired_workers(&ids).await;
    }

    /// Drop the watch acknowledgements of a retired `worker_id`.
    ///
    /// On the router's own lock, so the in-memory ack store and the file it
    /// persists cannot disagree: a worker that is gone can never fire an event
    /// again, and its position would otherwise be rewritten on the next
    /// `persist` from an unrelated owner.
    pub(crate) async fn forget_retired_worker(&self, worker_id: &str) {
        self.hub_events.lock().await.forget_worker(worker_id);
    }

    /// Close the recovery gate before accepting hub connections; see
    /// `RecoveryGate`.
    pub fn begin_recovery(&self) {
        self.recovery.begin();
    }

    pub(super) async fn recovery_wait(&self) {
        self.recovery.wait().await;
    }

    /// Open the recovery gate, releasing the worker-state requests that
    /// arrived while startup recovery ran.
    pub fn finish_recovery(&self) {
        self.recovery.finish();
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

    /// The build identity this daemon runs, which the executable watcher
    /// compares a replacement against.
    pub fn build_identity(&self) -> Value {
        self.daemon_build.as_ref().clone()
    }

    /// Arm the planned handover from outside a client connection.
    ///
    /// The executable watcher (H17) arms the same handover a newer client
    /// would, with the same deadline and the same idempotence.
    pub fn request_handover(&self, deadline: Duration) -> bool {
        self.begin_handover(deadline)
    }

    /// Whether a newer client has asked this daemon to hand over.
    ///
    /// The daemon reads it after teardown: a handover leaves the hub unserved
    /// until something dials it again, so the daemon that stepped aside starts
    /// its replacement itself instead of waiting for a client.
    pub fn handover_requested(&self) -> bool {
        self.handover
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .is_some()
    }

    /// Ask this daemon to hand over at the first quiet moment.
    ///
    /// Idempotent: a repeated request keeps the deadline the first one set and
    /// arms no second watcher. Returns whether this call armed the handover.
    fn begin_handover(&self, deadline: Duration) -> bool {
        let mut pending = self
            .handover
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        if pending.is_some() {
            return false;
        }
        *pending = Some(std::time::Instant::now() + deadline);
        drop(pending);
        let server = self.clone();
        tokio::spawn(async move { server.watch_handover().await });
        true
    }

    /// Whether the pool is at a quiet moment: no worker executing a command and
    /// no heavy admission permit held.
    ///
    /// A command in flight is work the graceful stop would interrupt, while a
    /// live worker between commands is what that stop checkpoints and the next
    /// daemon continues.
    fn handover_quiet(&self) -> bool {
        self.pool.commands_running() == 0 && self.pool.admission().running_heavy() == 0
    }

    /// Stop the daemon at the first quiet moment, or at the deadline.
    ///
    /// The deadline path hands over with a command still running: the graceful
    /// shutdown still checkpoints every live worker, it only interrupts work
    /// that never got its quiet moment.
    async fn watch_handover(&self) {
        let deadline = self
            .handover
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .expect("the deadline is set before the watcher starts");
        loop {
            tokio::time::sleep(HANDOVER_POLL).await;
            if self.handover_quiet() || std::time::Instant::now() >= deadline {
                break;
            }
        }
        if !self.handover_quiet() {
            warn!(
                "Hub handover deadline reached with a command still running; handing over anyway"
            );
        }
        info!("Hub handover: stopping for the newer build");
        // The same admission gate `hub/shutdown` takes, so a dispatch that
        // arrives while the daemon finishes is refused rather than interrupted.
        // `try_write` because a dispatch in flight holds the read side for its
        // whole turn, and waiting for it would outlast the deadline.
        if let Ok(mut stopping) = self.hub_shutdown_gate.try_write() {
            *stopping = true;
        }
        self.shutdown.send_replace(true);
    }

    pub(super) async fn admit_worker(&self) -> Result<tokio::sync::RwLockReadGuard<'_, bool>> {
        let admission = self.hub_shutdown_gate.read().await;
        if *admission || *self.shutdown.borrow() {
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
    /// `CROSS_PROCESS_TICK`, for a worker this process does not own (another
    /// `mini-swe-mcp` process, or `MINI_SWE_NO_DAEMON` mode): its state changes
    /// are invisible to the subscription, so the registry has to be re-read.
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
                crate::pool::WorkerPhase::Completed
                | crate::pool::WorkerPhase::Failed
                | crate::pool::WorkerPhase::Exhausted => {
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

/// Whether `req` reads worker state that startup recovery fills in.
///
/// `tools/call` covers every `worker` verb, and `hub/watch` streams worker
/// events, so both have to see the recovered pool. The handshake, `ping`,
/// `tools/list` and the protocol document do not.
fn needs_recovery(req: &JsonRpcRequest) -> bool {
    matches!(
        req.method.as_str(),
        "tools/call" | "hub/watch" | "hub/watch/ack"
    )
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

    /// A temporary scratch root a test's pool writes into, removed on drop.
    ///
    /// [`crate::test_support::TestScratch`] also drops the private
    /// `swe-tmp-*` / `swe-target-*` companions the sandbox derives from a
    /// worktree's leaf name, which a bare directory would leave behind.
    type ScratchDir = crate::test_support::TestScratch;

    /// A server whose pool files every row, mailbox and steer log under a
    /// temporary scratch root, together with the [`ScratchDir`] that removes
    /// it on drop.
    ///
    /// `WorkerPool::new` resolves the *real* scratch base, so a test that
    /// steers a worker writes this process's durable steer log and its nonce
    /// into the operator's `/var/tmp`. The test keeps the root alive for as
    /// long as the server lives.
    fn server() -> (ScratchDir, McpServer) {
        let root = ScratchDir::new("mcp-server");
        let pool = WorkerPool::with_scratch(
            1,
            "http://localhost:1".to_string(),
            "test-key".to_string(),
            crate::worktree::ScratchRoot::new(root.path()),
        );
        let server = McpServer::new(pool, "ninja".to_string());
        (root, server)
    }

    /// A `steer` answer is immediate, so the admission guard it took to queue
    /// the guidance must be released by the time the reply is built: holding it
    /// would wedge the pool's admission for every later dispatch.
    #[tokio::test]
    async fn steer_returns_immediately_and_releases_its_admission_guard() {
        use crate::pool::{LogBuffer, WorkerMetrics, WorkerRecord, WorkerState};
        let (_root, server) = server();
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
        let (_root, server) = server();
        for action in WORKER_ACTIONS {
            // Arguments are deliberately missing, so most verbs fail their own
            // validation; what matters is that the verb itself is recognised.
            // A no-arg `watch` with no timeout would wait for the next
            // dispatch, so it gets a zero deadline to stay a recognisability
            // probe.
            let arguments = if *action == "watch" {
                json!({ "action": action, "timeout_secs": 0 })
            } else {
                json!({ "action": action })
            };
            let unknown = match server.execute_tool("worker", arguments).await {
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
        let (_root, server) = server();
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
