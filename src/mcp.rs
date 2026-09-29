use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::borrow::Cow;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::mpsc;
use tracing::{error, info};

use crate::agent::AgentStepLog;
use crate::manifest::ModelManifest;
use crate::pool::{LogBuffer, WorkerPool, emit_view};

/// Every verb accepted by the single `worker` tool.
///
/// This is the *only* place the action list is spelled out: the `tools/list`
/// schema enum is derived from it, the dispatcher matches on it, and the CLI's
/// "did you mean …?" hint reuses it. Adding a verb therefore touches one
/// constant instead of several independent copies.
pub const WORKER_ACTIONS: &[&str] = &[
    "dispatch", "status", "steer", "collect", "logs", "list", "kill", "reap", "manifest", "prune",
];

/// Description of the `worker` tool itself.
const WORKER_TOOL_DESCRIPTION: &str = "Manage autonomous SWE mini-agents. Dispatches subagents in isolated Git worktrees, checks progress, injects steering instructions, retrieves git diffs, or inspects models.";

/// Where the `description` of an `inputSchema` property comes from.
enum DescriptionSource {
    /// A compile-time constant baked into [`WORKER_PROPERTIES`].
    Static(&'static str),
    /// Rendered at runtime; currently only the `model` property, whose text is
    /// produced by [`ModelManifest::build_tool_description`].
    Dynamic,
}

/// The `inputSchema` of the `worker` tool, expressed as data.
///
/// Each row is `(json_name, json_type, description)` — the same name the
/// handlers read back with `args.get(json_name)`. Building the schema from this
/// table keeps the advertised tool contract next to the dispatch table instead
/// of inlining it as a 60-line `json!` literal.
const WORKER_PROPERTIES: &[(&str, &str, DescriptionSource)] = &[
    (
        "action",
        "string",
        DescriptionSource::Static(
            "Action to perform: 'dispatch' (spawn subagent), 'status' (check step & progress), 'steer' (inject follow-up instruction), 'collect' (get final diff), 'logs' (inspect a live worker's bounded step history without collecting it), 'list' (list all workers), 'kill' (terminate worker), 'reap' (evict expired terminal worker records), 'manifest' (models catalog), 'prune' (clean stale worktrees). For unattended tracking, poll 'status' or pass wait:true; avoid short-interval busy-waiting.",
        ),
    ),
    (
        "task",
        "string",
        DescriptionSource::Static("Task description or bug to fix. Required for 'dispatch'."),
    ),
    (
        "repo_path",
        "string",
        DescriptionSource::Static(
            "Absolute path to repository root (alias: 'path'). Required for 'dispatch'.",
        ),
    ),
    (
        "path",
        "string",
        DescriptionSource::Static("Alias for repo_path."),
    ),
    (
        "model",
        "string",
        // Dynamic: renders the manifest's alias -> role catalogue.
        DescriptionSource::Dynamic,
    ),
    (
        "worker_id",
        "string",
        DescriptionSource::Static(
            "Target worker ID (alias: 'id'). Required for 'status', 'steer', 'collect', 'logs', and 'kill'.",
        ),
    ),
    (
        "id",
        "string",
        DescriptionSource::Static("Alias for worker_id."),
    ),
    (
        "message",
        "string",
        DescriptionSource::Static(
            "Steering guidance or follow-up instruction. Required for 'steer'.",
        ),
    ),
    (
        "wait",
        "boolean",
        DescriptionSource::Static(
            "If true, blocks until worker completes and returns final diff immediately. Optional for 'dispatch' (default: false). Recommended for unattended single-worker runs to avoid manual polling loops.",
        ),
    ),
    (
        "max_turns",
        "integer",
        DescriptionSource::Static("Maximum bash exploration turns (overrides manifest default)."),
    ),
    (
        "temperature",
        "number",
        DescriptionSource::Static("Model sampling temperature (overrides manifest default)."),
    ),
];

/// Render one table row as a JSON Schema property object.
fn property_schema(name: &str, json_type: &str, description: &str) -> Value {
    let mut schema = Map::new();
    schema.insert("type".to_string(), Value::String(json_type.to_string()));
    schema.insert(
        "description".to_string(),
        Value::String(description.to_string()),
    );
    if name == "action" {
        schema.insert(
            "enum".to_string(),
            Value::Array(
                WORKER_ACTIONS
                    .iter()
                    .map(|action| Value::String((*action).to_string()))
                    .collect(),
            ),
        );
    }
    if name == "max_turns" {
        schema.insert("minimum".to_string(), Value::from(1));
        schema.insert("maximum".to_string(), Value::from(crate::manifest::MAX_TURNS_LIMIT));
    }
    if name == "temperature" {
        schema.insert("minimum".to_string(), Value::from(*crate::manifest::TEMPERATURE_RANGE.start()));
        schema.insert("maximum".to_string(), Value::from(*crate::manifest::TEMPERATURE_RANGE.end()));
    }
    Value::Object(schema)
}

/// Build the whole `tools/list` result from the tables above.
///
/// Only the `model` description is dynamic (rendered from the model manifest);
/// every other key is a compile-time constant.
fn build_tools_list(manifest: &ModelManifest) -> Value {
    let model_description = manifest.build_tool_description();
    let mut properties = Map::new();
    for (name, json_type, source) in WORKER_PROPERTIES {
        let description = match source {
            DescriptionSource::Static(text) => Cow::Borrowed(*text),
            DescriptionSource::Dynamic => Cow::Borrowed(model_description.as_str()),
        };
        properties.insert(
            (*name).to_string(),
            property_schema(name, json_type, description.as_ref()),
        );
    }

    json!({
        "tools": [
            {
                "name": "worker",
                "description": WORKER_TOOL_DESCRIPTION,
                "inputSchema": {
                    "type": "object",
                    "properties": Value::Object(properties),
                    "required": ["action"]
                }
            }
        ]
    })
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct JsonRpcRequest {
    jsonrpc: String,
    id: Option<Value>,
    method: String,
    params: Option<Value>,
}

#[derive(Debug, Serialize)]
struct JsonRpcResponse {
    jsonrpc: &'static str,
    id: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<Value>,
}

/// A JSON-RPC `result` whose single text content is a tool payload.
///
/// `tools/call` embeds the tool payload as a JSON *string* inside the envelope
/// (`{"content":[{"type":"text","text":"<json>"}]}`). Building that with `json!`
/// requires pretty-printing the payload into a `String` first, which the outer
/// `to_string` then escapes and re-serializes — a second full materialization
/// of the same bytes. `PreSerializedResult` keeps the payload as a `Value` and
/// lets the envelope serializer write it directly, so the payload is
/// materialized once (audit 07, F4).
struct PreSerializedResult {
    content: [PreSerializedContent; 1],
}

struct PreSerializedContent {
    kind: &'static str,
    payload: Value,
}

impl Serialize for PreSerializedResult {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut st = serializer.serialize_struct("Result", 1)?;
        st.serialize_field("content", &self.content)?;
        st.end()
    }
}

impl Serialize for PreSerializedContent {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut st = serializer.serialize_struct("Content", 2)?;
        st.serialize_field("type", self.kind)?;
        // The `text` field is a JSON string; the payload is written into it
        // directly instead of via an intermediate `to_string_pretty` result.
        st.serialize_field("text", &PayloadAsText(&self.payload))?;
        st.end()
    }
}

/// Writes a `Value` into a serialized string field without a separate
/// `String` allocation for the envelope to copy.
struct PayloadAsText<'a>(&'a Value);

impl Serialize for PayloadAsText<'_> {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        serializer.collect_str(&PayloadWriter(self.0))
    }
}

/// Feeds a `Value` to `collect_str` through `Display`, so the payload is written
/// into the target string buffer in one pass.
///
/// Pretty-printing is preserved: MCP clients read this field as human-readable
/// tool output, and `collect_str` streams the formatting straight into the
/// envelope's own buffer, so the previous `to_string_pretty` `String` is gone.
struct PayloadWriter<'a>(&'a Value);

impl std::fmt::Display for PayloadWriter<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Render pretty JSON into an internal buffer using a `Write` sink that
        // forwards to the formatter, so the value is produced once and written
        // through `collect_str` without a second, escaped envelope copy.
        let mut sink = FmtSink { f };
        let mut ser = serde_json::Serializer::with_formatter(
            &mut sink,
            serde_json::ser::PrettyFormatter::new(),
        );
        self.0.serialize(&mut ser).map_err(|_| std::fmt::Error)
    }
}

/// Minimal `std::io::Write` adapter over a `fmt::Formatter`, used to drive
/// `serde_json`'s `PrettyFormatter` while `collect_str` streams into the
/// envelope's own buffer.
struct FmtSink<'a, 'b> {
    f: &'a mut std::fmt::Formatter<'b>,
}

impl std::io::Write for FmtSink<'_, '_> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let s = std::str::from_utf8(buf).map_err(|_| std::io::Error::other("non-utf8 json"))?;
        self.f
            .write_str(s)
            .map_err(|_| std::io::Error::other("fmt error"))?;
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl JsonRpcResponse {
    /// Successful JSON-RPC 2.0 response.
    fn ok(id: Option<Value>, result: Value) -> Self {
        Self {
            jsonrpc: "2.0",
            id,
            result: Some(result),
            error: None,
        }
    }

    /// JSON-RPC 2.0 error response carrying an application-level `code`.
    fn err(id: Option<Value>, code: i64, message: impl Into<String>) -> Self {
        Self {
            jsonrpc: "2.0",
            id,
            result: None,
            error: Some(json!({
                "code": code,
                "message": message.into()
            })),
        }
    }
}

#[derive(Clone)]
pub struct McpServer {
    pool: Arc<WorkerPool>,
    default_model: String,
    manifest: Arc<ModelManifest>,
    /// Precomputed, immutable `tools/list` result. The manifest is never mutated
    /// after construction, so the payload is byte-identical for the process
    /// lifetime and is cloned (an `Arc` memcpy) instead of rebuilt per request.
    tools_list: Arc<Value>,
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
                            serde_json::to_value(PreSerializedResult {
                                content: [PreSerializedContent { kind: "text", payload: val }],
                            })
                            .unwrap_or_else(|_| {
                                json!({ "content": [{ "type": "text", "text": "{}" }] })
                            }),
                        ),
                        error: None,
                    },
                    Err(e) => JsonRpcResponse::err(id, -32000, e.to_string()),
                }
            }

            _ => JsonRpcResponse::err(id, -32601, format!("Method not found: {}", req.method)),
        }
    }

    fn required_string<'a>(args: &'a Value, name: &str, action: &str) -> Result<&'a str> {
        args.get(name)
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| anyhow::anyhow!("'{name}' is required for action '{action}'"))
    }

    fn get_worker_id<'a>(args: &'a Value, action: &str) -> Result<&'a str> {
        args.get("worker_id")
            .or_else(|| args.get("id"))
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| anyhow::anyhow!("'worker_id' (or 'id') is required for action '{action}'"))
    }

    fn get_repo_path(args: &Value) -> PathBuf {
        let repo_path_str = args
            .get("repo_path")
            .or_else(|| args.get("path"))
            .and_then(|v| v.as_str())
            .unwrap_or(".");
        PathBuf::from(repo_path_str)
    }

    async fn emit_progress(
        tx: Option<&mpsc::Sender<String>>,
        token: Option<&Value>,
        progress: usize,
        total: usize,
        message: impl std::fmt::Display,
    ) {
        if let (Some(token), Some(tx)) = (token, tx) {
            let notif = json!({
                "jsonrpc": "2.0",
                "method": "notifications/progress",
                "params": {
                    "progressToken": token,
                    "progress": progress,
                    "total": total,
                    "message": message.to_string(),
                }
            });
            if let Ok(serialized) = serde_json::to_string(&notif) {
                let _ = tx.send(serialized + "\n").await;
            }
        }
    }

    /// The precomputed `tools/list` result.
    ///
    /// Exposed so the advertised tool contract can be asserted in-process
    /// (unit tests, embedders) instead of only through a live stdio subprocess.
    pub fn tools_list(&self) -> Value {
        (*self.tools_list).clone()
    }

    pub async fn execute_tool(&self, name: &str, args: Value) -> Result<Value> {
        self.execute_tool_with_progress(name, args, None, None).await
    }

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
        let token = progress_token.as_ref();
        let tx = progress_tx.as_ref();

        match action {
            "manifest" => self.handle_manifest(),
            "dispatch" => self.handle_dispatch(&args, token, tx).await,
            "status" => self.handle_status(&args).await,
            "collect" => self.handle_collect(&args).await,
            "logs" => self.handle_logs(&args).await,
            "reap" => self.handle_reap().await,
            "list" => self.handle_list().await,
            "kill" => self.handle_kill(&args).await,
            "steer" => self.handle_steer(&args).await,
            "prune" => self.handle_prune(&args, token, tx).await,
            _ => anyhow::bail!("Unknown action or tool: {action}"),
        }
    }

    fn handle_manifest(&self) -> Result<Value> {
        Ok(json!({
            "default_model": self.manifest.default,
            "models": self.manifest.models,
        }))
    }

    async fn handle_dispatch(
        &self,
        args: &Value,
        token: Option<&Value>,
        tx: Option<&mpsc::Sender<String>>,
    ) -> Result<Value> {
        let task = Self::required_string(args, "task", "dispatch")?.to_string();
        let repo_path = Self::get_repo_path(args);
        let requested_model = args
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or(&self.default_model);

        let (resolved_model, def_temp, def_turns) =
            self.manifest.resolve_model(requested_model);

        // The manifest is sanitized at load time and the request arguments are
        // sanitized here, so neither ingress can ship a value a provider would
        // reject (or a `0` turn budget that would strand the worker).
        let requested_temp = args
            .get("temperature")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32);
        let temperature = ModelManifest::sanitize_temperature(requested_temp.or(def_temp));

        let requested_turns = args
            .get("max_turns")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);
        let max_turns = ModelManifest::sanitize_max_turns(requested_turns, def_turns);

        let wait = args.get("wait").and_then(|v| v.as_bool()).unwrap_or(false);
        let group = args
            .get("group")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let wid = self
            .pool
            .dispatch(
                task,
                resolved_model,
                temperature,
                repo_path,
                max_turns,
                group,
            )
            .await?;

        Self::emit_progress(
            tx,
            token,
            0,
            max_turns,
            format!("Worker {wid} dispatched in isolated worktree"),
        )
        .await;

        if wait {
            self.wait_for_worker(&wid, max_turns, token, tx).await
        } else {
            Ok(json!({
                "worker_id": wid,
                "status": "dispatched",
                "message": "Worker is executing in isolated worktree in background"
            }))
        }
    }

    async fn wait_for_worker(
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

    async fn handle_status(&self, args: &Value) -> Result<Value> {
        let wid = Self::get_worker_id(args, "status")?;
        if let Some(state) = self.pool.get_worker_state(wid).await {
            Ok(json!({ "worker_id": wid, "state": state }))
        } else {
            let path = crate::pool::registry_dir().join(format!("{wid}.json"));
            if let Ok(content) = std::fs::read_to_string(&path)
                && let Ok(entry) = serde_json::from_str::<crate::pool::WorkerRegistryEntry>(&content)
            {
                let is_alive = crate::worktree::is_process_alive(entry.pid);
                let status = if !is_alive && (entry.status == "running" || entry.status == "paused") {
                    "stopped"
                } else {
                    &entry.status
                };
                let state_name = match status {
                    "running" => "Running",
                    "completed" => "Completed",
                    "paused" => "Paused",
                    "failed" => "Failed",
                    _ => "Stopped",
                };
                return Ok(json!({
                    "worker_id": wid,
                    "task": entry.task,
                    "model": entry.model,
                    "state": {
                        "state": state_name,
                        "details": {
                            "status": state_name,
                            "step": entry.step,
                            "turns": entry.step,
                            "summary": entry.last_command.clone(),
                            "error": if entry.status == "failed" { Some(entry.last_command) } else { None },
                            "question": entry.question,
                            "pid": entry.pid,
                            "started_at": entry.started_at,
                        }
                    }
                }));
            }
            anyhow::bail!("Worker not found: {wid}")
        }
    }

    /// Render the bounded tail of a live worker's step history together with the
    /// counters that make the degradation explicit (audit 07, R4 / R7).
    async fn render_logs(&self, wid: &str) -> (Vec<AgentStepLog>, usize, usize, Option<String>) {
        let Some(buffer): Option<std::sync::Arc<LogBuffer>> = self.pool.get_worker_logs(wid).await
        else {
            return (Vec::new(), 0, 0, None);
        };
        let view = emit_view(&buffer, self.pool.log_policy().max_emitted);
        (
            view.logs,
            view.logs_omitted,
            buffer.dropped(),
            view.logs_truncation_notice,
        )
    }

    /// `logs` action: inspect a live worker's retained history without
    /// collecting (and thus evicting) it.
    async fn handle_logs(&self, args: &Value) -> Result<Value> {
        let wid = Self::get_worker_id(args, "logs")?;
        let Some(buffer) = self.pool.get_worker_logs(wid).await else {
            anyhow::bail!("Worker not found: {wid}")
        };
        let policy = self.pool.log_policy();
        let (logs, logs_omitted, logs_dropped, logs_truncation_notice) = {
            let view = emit_view(&buffer, policy.max_emitted);
            (
                view.logs,
                view.logs_omitted,
                buffer.dropped(),
                view.logs_truncation_notice,
            )
        };
        Ok(json!({
            "worker_id": wid,
            "state": self.pool.get_worker_state(wid).await,
            "logs": logs,
            "total_steps": buffer.total(),
            "logs_retained": buffer.retained(),
            "logs_omitted": logs_omitted,
            "logs_dropped": logs_dropped,
            "logs_truncation_notice": logs_truncation_notice,
            "retention": {
                "max_retained": policy.max_retained,
                "max_bytes": policy.max_bytes,
                "max_emitted": policy.max_emitted,
            },
        }))
    }

    /// `reap` action: evict terminal worker records whose TTL expired.
    async fn handle_reap(&self) -> Result<Value> {
        let reaped = self.pool.reap().await;
        Ok(json!({
            "status": "reaped",
            "reaped": reaped.len(),
            "worker_ids": reaped,
        }))
    }

    async fn handle_collect(&self, args: &Value) -> Result<Value> {
        let wid = Self::get_worker_id(args, "collect")?;
        if let Some(collected) = self.pool.collect(wid).await {
            Ok(json!({
                "worker_id": wid,
                "state": collected.state,
                "logs": collected.logs,
                "logs_omitted": collected.logs_omitted,
                "logs_dropped": collected.logs_dropped,
                "logs_truncation_notice": collected.logs_truncation_notice,
            }))
        } else {
            anyhow::bail!("Worker not found: {wid}")
        }
    }

    async fn handle_list(&self) -> Result<Value> {
        let workers = self.pool.list_workers().await;
        Ok(json!({ "workers": workers }))
    }

    async fn handle_kill(&self, args: &Value) -> Result<Value> {
        let wid = Self::get_worker_id(args, "kill")?;
        let killed = self.pool.kill(wid).await;
        if killed {
            Ok(json!({ "worker_id": wid, "killed": true }))
        } else {
            let path = crate::pool::registry_dir().join(format!("{wid}.json"));
            if let Ok(content) = std::fs::read_to_string(&path)
                && let Ok(entry) = serde_json::from_str::<crate::pool::WorkerRegistryEntry>(&content)
                && crate::worktree::is_process_alive(entry.pid)
            {
                #[cfg(unix)]
                {
                    let _ = std::process::Command::new("kill")
                        .args(["-TERM", &entry.pid.to_string()])
                        .status();
                }
                return Ok(json!({ "worker_id": wid, "killed": true }));
            }
            Ok(json!({ "worker_id": wid, "killed": false }))
        }
    }

    async fn handle_steer(&self, args: &Value) -> Result<Value> {
        let wid = Self::get_worker_id(args, "steer")?;
        let message = Self::required_string(args, "message", "steer")?.to_string();
        self.pool.steer(wid, message).await?;
        Ok(json!({
            "worker_id": wid,
            "status": "steered",
            "message": "Steering instruction queued for next turn"
        }))
    }

    async fn handle_prune(
        &self,
        args: &Value,
        token: Option<&Value>,
        tx: Option<&mpsc::Sender<String>>,
    ) -> Result<Value> {
        let repo_path = Self::get_repo_path(args);
        Self::emit_progress(
            tx,
            token,
            0,
            1,
            "Pruning stale worktrees and dead worker branches",
        )
        .await;
        crate::worktree::prune_stale_worktrees(&repo_path);
        Self::emit_progress(tx, token, 1, 1, "Prune complete").await;
        Ok(json!({
            "status": "pruned",
            "message": "Stale worktrees and dead worker branches cleaned up"
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn server() -> McpServer {
        McpServer::new(
            WorkerPool::new(1, "http://localhost:1".to_string(), "test-key".to_string()),
            "ninja".to_string(),
            ModelManifest::default(),
        )
    }

    fn worker_schema(tools_list: &Value) -> &Value {
        tools_list["tools"]
            .as_array()
            .and_then(|tools| {
                tools
                    .iter()
                    .find(|tool| tool["name"] == "worker")
                    .map(|tool| &tool["inputSchema"])
            })
            .expect("tools/list must expose the 'worker' tool")
    }

    /// The advertised `action` enum is the dispatch table, not a second copy.
    #[test]
    fn action_enum_is_derived_from_the_dispatch_table() {
        let tools_list = build_tools_list(&ModelManifest::default());
        let schema = worker_schema(&tools_list);
        let actions: Vec<&str> = schema["properties"]["action"]["enum"]
            .as_array()
            .expect("the action property must carry an enum")
            .iter()
            .map(|value| value.as_str().expect("enum entries must be strings"))
            .collect();

        assert_eq!(actions, WORKER_ACTIONS.to_vec());
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

    /// The property table *is* the schema: every row is typed and documented.
    #[test]
    fn property_table_renders_every_row() {
        let tools_list = build_tools_list(&ModelManifest::default());
        let schema = worker_schema(&tools_list);
        let properties = schema["properties"]
            .as_object()
            .expect("the input schema must expose a properties object");

        assert_eq!(properties.len(), WORKER_PROPERTIES.len());
        for (name, json_type, _) in WORKER_PROPERTIES {
            let property = &properties[*name];
            assert_eq!(property["type"], *json_type, "wrong type for '{name}'");
            assert!(
                property["description"]
                    .as_str()
                    .is_some_and(|description| !description.is_empty()),
                "'{name}' needs a non-empty description"
            );
        }
        assert_eq!(schema["required"], json!(["action"]));
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

    /// The `ok` / `err` constructors emit spec-shaped JSON-RPC 2.0 envelopes.
    #[test]
    fn json_rpc_constructors_are_spec_shaped() {
        let ok = serde_json::to_value(JsonRpcResponse::ok(Some(json!(7)), json!({ "a": 1 })))
            .expect("ok responses must serialise");
        assert_eq!(
            ok,
            json!({ "jsonrpc": "2.0", "id": 7, "result": { "a": 1 } })
        );

        let err = serde_json::to_value(JsonRpcResponse::err(None, -32601, "nope"))
            .expect("error responses must serialise");
        assert_eq!(
            err,
            json!({
                "jsonrpc": "2.0",
                "id": null,
                "error": { "code": -32601, "message": "nope" }
            })
        );
    }
}
