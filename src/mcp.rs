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
                                        "description": "Maximum bash exploration turns (overrides manifest default)."
                                    },
                                    "temperature": {
                                        "type": "number",
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
            anyhow::bail!("Unknown tool: '{}'. Only 'worker' is supported.", name);
        }

        let action = args.get("action").and_then(|v| v.as_str()).unwrap_or("");

        match action {
            "manifest" => Ok(json!({
                "default_model": self.manifest.default,
                "models": self.manifest.models,
            })),

            "dispatch" => {
                let task = Self::required_string(&args, "task", action)?.to_string();
                let repo_path = Self::get_repo_path(&args);
                let requested_model = args
                    .get("model")
                    .and_then(|v| v.as_str())
                    .unwrap_or(&self.default_model);

                let (resolved_model, def_temp, def_turns) =
                    self.manifest.resolve_model(requested_model);

                let temperature = args
                    .get("temperature")
                    .and_then(|v| v.as_f64())
                    .map(|v| v as f32)
                    .or(def_temp);

                let max_turns = args
                    .get("max_turns")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize)
                    .or(def_turns)
                    .unwrap_or(100);

                let wait = args.get("wait").and_then(|v| v.as_bool()).unwrap_or(false);

                let wid = self
                    .pool
                    .dispatch(task, resolved_model, temperature, repo_path, max_turns)
                    .await?;

                if let (Some(token), Some(tx)) = (&progress_token, &progress_tx) {
                    let notif = json!({
                        "jsonrpc": "2.0",
                        "method": "notifications/progress",
                        "params": {
                            "progressToken": token,
                            "progress": 0,
                            "total": max_turns,
                            "message": format!("Worker {wid} dispatched in isolated worktree")
                        }
                    });
                    if let Ok(serialized) = serde_json::to_string(&notif) {
                        let _ = tx.send(serialized + "\n").await;
                    }
                }

                if wait {
                    let mut last_reported_step = 0;
                    // Poll until completed or failed
                    loop {
                        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                        if let Some(state) = self.pool.get_worker_state(&wid).await {
                            match state {
                                crate::pool::WorkerState::Running {
                                    step,
                                    ref last_command,
                                    ..
                                } => {
                                    if step > last_reported_step {
                                        last_reported_step = step;
                                        if let (Some(token), Some(tx)) =
                                            (&progress_token, &progress_tx)
                                        {
                                            let notif = json!({
                                                "jsonrpc": "2.0",
                                                "method": "notifications/progress",
                                                "params": {
                                                    "progressToken": token,
                                                    "progress": step,
                                                    "total": max_turns,
                                                    "message": format!("Step {step}/{max_turns}: {last_command}")
                                                }
                                            });
                                            if let Ok(serialized) = serde_json::to_string(&notif) {
                                                let _ = tx.send(serialized + "\n").await;
                                            }
                                        }
                                    }
                                }
                                crate::pool::WorkerState::Completed { .. }
                                | crate::pool::WorkerState::Failed { .. } => {
                                    if let (Some(token), Some(tx)) =
                                        (&progress_token, &progress_tx)
                                    {
                                        let notif = json!({
                                            "jsonrpc": "2.0",
                                            "method": "notifications/progress",
                                            "params": {
                                                "progressToken": token,
                                                "progress": max_turns,
                                                "total": max_turns,
                                                "message": format!("Worker {wid} finished execution")
                                            }
                                        });
                                        if let Ok(serialized) = serde_json::to_string(&notif) {
                                            let _ = tx.send(serialized + "\n").await;
                                        }
                                    }
                                    let logs =
                                        self.pool.get_worker_logs(&wid).await.unwrap_or_default();
                                    return Ok(json!({
                                        "worker_id": wid,
                                        "state": state,
                                        "logs": logs
                                    }));
                                }
                                crate::pool::WorkerState::Paused {
                                    ref question,
                                    step,
                                    ..
                                } => {
                                    if let (Some(token), Some(tx)) =
                                        (&progress_token, &progress_tx)
                                    {
                                        let notif = json!({
                                            "jsonrpc": "2.0",
                                            "method": "notifications/progress",
                                            "params": {
                                                "progressToken": token,
                                                "progress": step,
                                                "total": max_turns,
                                                "message": format!("Worker {wid} paused: waiting for orchestrator steering")
                                            }
                                        });
                                        if let Ok(serialized) = serde_json::to_string(&notif) {
                                            let _ = tx.send(serialized + "\n").await;
                                        }
                                    }
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
                } else {
                    Ok(json!({
                        "worker_id": wid,
                        "status": "dispatched",
                        "message": "Worker is executing in isolated worktree in background"
                    }))
                }
            }

            "status" => {
                let wid = Self::get_worker_id(&args, action)?;
                if let Some(state) = self.pool.get_worker_state(wid).await {
                    Ok(json!({
                        "worker_id": wid,
                        "state": state
                    }))
                } else {
                    anyhow::bail!("Worker not found: {}", wid)
                }
            }

            "collect" => {
                let wid = Self::get_worker_id(&args, action)?;
                if let Some(collected) = self.pool.collect(wid).await {
                    Ok(json!({
                        "worker_id": wid,
                        "state": collected.state,
                        "logs": collected.logs
                    }))
                } else {
                    anyhow::bail!("Worker not found: {}", wid)
                }
            }

            "list" => {
                let workers = self.pool.list_workers().await;
                Ok(json!({ "workers": workers }))
            }

            "kill" => {
                let wid = Self::get_worker_id(&args, action)?;
                let killed = self.pool.kill(wid).await;
                Ok(json!({ "worker_id": wid, "killed": killed }))
            }

            "steer" => {
                let wid = Self::get_worker_id(&args, action)?;
                let message = Self::required_string(&args, "message", action)?.to_string();
                self.pool.steer(wid, message).await?;
                Ok(json!({
                    "worker_id": wid,
                    "status": "steered",
                    "message": "Steering instruction queued for next turn"
                }))
            }

            "prune" => {
                let repo_path = Self::get_repo_path(&args);
                if let (Some(token), Some(tx)) = (&progress_token, &progress_tx) {
                    let notif = json!({
                        "jsonrpc": "2.0",
                        "method": "notifications/progress",
                        "params": {
                            "progressToken": token,
                            "progress": 0,
                            "total": 1,
                            "message": "Pruning stale worktrees and dead worker branches"
                        }
                    });
                    if let Ok(serialized) = serde_json::to_string(&notif) {
                        let _ = tx.send(serialized + "\n").await;
                    }
                }
                crate::worktree::prune_stale_worktrees(&repo_path);
                if let (Some(token), Some(tx)) = (&progress_token, &progress_tx) {
                    let notif = json!({
                        "jsonrpc": "2.0",
                        "method": "notifications/progress",
                        "params": {
                            "progressToken": token,
                            "progress": 1,
                            "total": 1,
                            "message": "Prune complete"
                        }
                    });
                    if let Ok(serialized) = serde_json::to_string(&notif) {
                        let _ = tx.send(serialized + "\n").await;
                    }
                }
                Ok(json!({
                    "status": "pruned",
                    "message": "Stale worktrees and dead worker branches cleaned up"
                }))
            }

            _ => anyhow::bail!("Unknown action or tool: {}", action),
        }
    }
}
