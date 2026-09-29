use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::mpsc;
use tracing::{error, info};

use crate::manifest::ModelManifest;
use crate::pool::WorkerPool;

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

#[derive(Clone)]
pub struct McpServer {
    pool: Arc<WorkerPool>,
    default_model: String,
    manifest: Arc<ModelManifest>,
}

impl McpServer {
    pub fn new(pool: WorkerPool, default_model: String, manifest: ModelManifest) -> Self {
        Self {
            pool: Arc::new(pool),
            default_model,
            manifest: Arc::new(manifest),
        }
    }

    pub async fn run_stdio(&self) -> Result<()> {
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
                    let resp = JsonRpcResponse {
                        jsonrpc: "2.0",
                        id: None,
                        result: None,
                        error: Some(json!({
                            "code": -32700,
                            "message": format!("Parse error: {}", e)
                        })),
                    };
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
            "ping" => JsonRpcResponse {
                jsonrpc: "2.0",
                id,
                result: Some(json!({})),
                error: None,
            },

            "initialize" => JsonRpcResponse {
                jsonrpc: "2.0",
                id,
                result: Some(json!({
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": { "listChanged": false }
                    },
                    "serverInfo": {
                        "name": "mini-swe-mcp",
                        "version": "0.1.0"
                    }
                })),
                error: None,
            },

            "tools/list" => JsonRpcResponse {
                jsonrpc: "2.0",
                id,
                result: Some(json!({
                    "tools": [
                        {
                            "name": "worker",
                            "description": "Manage autonomous SWE mini-agents. Dispatches subagents in isolated Git worktrees, checks progress, injects steering instructions, retrieves git diffs, or inspects models.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "action": {
                                        "type": "string",
                                        "enum": ["dispatch", "status", "steer", "collect", "list", "kill", "manifest", "prune"],
                                        "description": "Action to perform: 'dispatch' (spawn subagent), 'status' (check step & progress), 'steer' (inject follow-up instruction), 'collect' (get final diff), 'list' (list all workers), 'kill' (terminate worker), 'manifest' (models catalog), 'prune' (clean stale worktrees)"
                                    },
                                    "task": {
                                        "type": "string",
                                        "description": "Task description or bug to fix. Required for 'dispatch'."
                                    },
                                    "repo_path": {
                                        "type": "string",
                                        "description": "Absolute path to repository root (alias: 'path'). Required for 'dispatch'."
                                    },
                                    "path": {
                                        "type": "string",
                                        "description": "Alias for repo_path."
                                    },
                                    "model": {
                                        "type": "string",
                                        "description": self.manifest.build_tool_description()
                                    },
                                    "worker_id": {
                                        "type": "string",
                                        "description": "Target worker ID (alias: 'id'). Required for 'status', 'steer', 'collect', and 'kill'."
                                    },
                                    "id": {
                                        "type": "string",
                                        "description": "Alias for worker_id."
                                    },
                                    "message": {
                                        "type": "string",
                                        "description": "Steering guidance or follow-up instruction. Required for 'steer'."
                                    },
                                    "wait": {
                                        "type": "boolean",
                                        "description": "If true, blocks until worker completes and returns final diff immediately. Optional for 'dispatch' (default: false)."
                                    },
                                    "max_turns": {
                                        "type": "integer",
                                        "minimum": 1,
                                        "maximum": crate::manifest::MAX_TURNS_LIMIT,
                                        "description": "Maximum bash exploration turns (overrides manifest default)."
                                    },
                                    "temperature": {
                                        "type": "number",
                                        "minimum": crate::manifest::TEMPERATURE_RANGE.start(),
                                        "maximum": crate::manifest::TEMPERATURE_RANGE.end(),
                                        "description": "Model sampling temperature (overrides manifest default)."
                                    }
                                },
                                "required": ["action"]
                            }
                        }
                    ]
                })),
                error: None,
            },

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
                        result: Some(json!({
                            "content": [
                                {
                                    "type": "text",
                                    "text": serde_json::to_string_pretty(&val).unwrap_or_default()
                                }
                            ]
                        })),
                        error: None,
                    },
                    Err(e) => JsonRpcResponse {
                        jsonrpc: "2.0",
                        id,
                        result: None,
                        error: Some(json!({
                            "code": -32000,
                            "message": e.to_string()
                        })),
                    },
                }
            }

            _ => JsonRpcResponse {
                jsonrpc: "2.0",
                id,
                result: None,
                error: Some(json!({
                    "code": -32601,
                    "message": format!("Method not found: {}", req.method)
                })),
            },
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
                    // H-3: the terminal payload is fetched exactly once, on the
                    // terminal path, instead of on every poll.
                    let state = self.pool.get_worker_state(wid).await;
                    let logs = self.pool.take_worker_logs(wid).await.unwrap_or_default();
                    return Ok(json!({
                        "worker_id": wid,
                        "state": state,
                        "logs": logs
                    }));
                }
                crate::pool::WorkerPhase::Paused => {
                    let step = progress.step;
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
                        "question": progress.question,
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

    async fn handle_collect(&self, args: &Value) -> Result<Value> {
        let wid = Self::get_worker_id(args, "collect")?;
        if let Some(collected) = self.pool.collect(wid).await {
            Ok(json!({
                "worker_id": wid,
                "state": collected.state,
                "logs": collected.logs
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
