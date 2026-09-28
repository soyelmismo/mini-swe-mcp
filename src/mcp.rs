use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
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

        while let Some(line) = reader.next_line().await? {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            let req: JsonRpcRequest = match serde_json::from_str(line) {
                Ok(r) => r,
                Err(e) => {
                    error!(error = %e, line = %line, "Malformed JSON-RPC request");
                    continue;
                }
            };

            let resp = self.handle_request(req).await;
            let serialized = serde_json::to_string(&resp)? + "\n";
            stdout.write_all(serialized.as_bytes()).await?;
            stdout.flush().await?;
        }

        Ok(())
    }

    async fn handle_request(&self, req: JsonRpcRequest) -> JsonRpcResponse {
        let id = req.id;
        match req.method.as_str() {
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
                            "name": "dispatch_worker",
                            "description": "Spawn an autonomous SWE mini-agent in an isolated Git worktree to solve a coding task.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "task": {
                                        "type": "string",
                                        "description": "Detailed task description, bug to fix, or feature to implement"
                                    },
                                    "model": {
                                        "type": "string",
                                        "description": self.manifest.build_tool_description()
                                    },
                                    "repo_path": {
                                        "type": "string",
                                        "description": "Absolute path to repository root"
                                    },
                                    "max_turns": {
                                        "type": "integer",
                                        "description": "Maximum bash exploration turns (overrides manifest default)"
                                    },
                                    "temperature": {
                                        "type": "number",
                                        "description": "Model sampling temperature (overrides manifest default)"
                                    },
                                    "wait": {
                                        "type": "boolean",
                                        "description": "If true, blocks until the worker finishes and returns final diff immediately"
                                    }
                                },
                                "required": ["task", "repo_path"]
                            }
                        },
                        {
                            "name": "get_model_manifest",
                            "description": "Get the declarative catalog of available models, their specialized roles, and guidelines on when to use each.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {}
                            }
                        },
                        {
                            "name": "worker_status",
                            "description": "Check current step, last executed bash command, and status of a worker.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "worker_id": { "type": "string" }
                                },
                                "required": ["worker_id"]
                            }
                        },
                        {
                            "name": "collect_result",
                            "description": "Retrieve the final result, git diff patch, and command logs from a completed worker.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "worker_id": { "type": "string" }
                                },
                                "required": ["worker_id"]
                            }
                        },
                        {
                            "name": "list_workers",
                            "description": "List all active, completed, or failed workers in the pool.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {}
                            }
                        },
                        {
                            "name": "kill_worker",
                            "description": "Terminate a running worker subagent and clean up its git worktree.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "worker_id": { "type": "string" }
                                },
                                "required": ["worker_id"]
                            }
                        },
                        {
                            "name": "steer_worker",
                            "description": "Inject a steering instruction, correction, or follow-up guidance into a running worker. It will be injected directly into the subagent's prompt on its next turn.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "worker_id": {
                                        "type": "string",
                                        "description": "ID of the running worker"
                                    },
                                    "message": {
                                        "type": "string",
                                        "description": "Steering prompt or follow-up guidance for the subagent"
                                    }
                                },
                                "required": ["worker_id", "message"]
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

                match self.execute_tool(tool_name, arguments).await {
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

    async fn execute_tool(&self, name: &str, args: Value) -> Result<Value> {
        match name {
            "get_model_manifest" => {
                Ok(json!({
                    "default_model": self.manifest.default,
                    "models": self.manifest.models,
                }))
            }

            "dispatch_worker" => {
                let task = args.get("task").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let repo_path = PathBuf::from(
                    args.get("repo_path")
                        .and_then(|v| v.as_str())
                        .unwrap_or("."),
                );
                let requested_model = args
                    .get("model")
                    .and_then(|v| v.as_str())
                    .unwrap_or(&self.default_model);

                let (resolved_model, def_temp, def_turns) = self.manifest.resolve_model(requested_model);

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

                let wid = self.pool.dispatch(task, resolved_model, temperature, repo_path, max_turns).await?;

                if wait {
                    // Poll until completed or failed
                    loop {
                        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                        if let Some(state) = self.pool.get_worker_state(&wid).await {
                            match state {
                                crate::pool::WorkerState::Completed { .. }
                                | crate::pool::WorkerState::Failed { .. } => {
                                    let logs = self.pool.get_worker_logs(&wid).await.unwrap_or_default();
                                    return Ok(json!({
                                        "worker_id": wid,
                                        "state": state,
                                        "logs": logs
                                    }));
                                }
                                _ => {}
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

            "worker_status" => {
                let wid = args.get("worker_id").and_then(|v| v.as_str()).unwrap_or("");
                if let Some(state) = self.pool.get_worker_state(wid).await {
                    Ok(json!({
                        "worker_id": wid,
                        "state": state
                    }))
                } else {
                    anyhow::bail!("Worker not found: {}", wid)
                }
            }

            "collect_result" => {
                let wid = args.get("worker_id").and_then(|v| v.as_str()).unwrap_or("");
                if let Some(state) = self.pool.get_worker_state(wid).await {
                    let logs = self.pool.get_worker_logs(wid).await.unwrap_or_default();
                    Ok(json!({
                        "worker_id": wid,
                        "state": state,
                        "logs": logs
                    }))
                } else {
                    anyhow::bail!("Worker not found: {}", wid)
                }
            }

            "list_workers" => {
                let workers = self.pool.list_workers().await;
                Ok(json!({ "workers": workers }))
            }

            "kill_worker" => {
                let wid = args.get("worker_id").and_then(|v| v.as_str()).unwrap_or("");
                let killed = self.pool.kill(wid).await;
                Ok(json!({ "worker_id": wid, "killed": killed }))
            }

            "steer_worker" => {
                let wid = args.get("worker_id").and_then(|v| v.as_str()).unwrap_or("");
                let message = args
                    .get("message")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                self.pool.steer(wid, message).await?;
                Ok(json!({
                    "worker_id": wid,
                    "status": "steered",
                    "message": "Steering instruction queued for next turn"
                }))
            }

            _ => anyhow::bail!("Unknown tool: {}", name),
        }
    }
}
