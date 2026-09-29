//! Tool-call dispatch: the `worker` verb table and the handlers behind it.
//!
//! The `tools/call` request is turned into a [`Value`] payload here; the
//! JSON-RPC envelope around it is [`crate::mcp::protocol`]'s business and the
//! stdio plumbing is [`crate::mcp::server`]'s.

use anyhow::Result;
use serde_json::{Value, json};
use std::path::PathBuf;
use tokio::sync::mpsc;

use super::server::McpServer;
use crate::agent::AgentStepLog;
use crate::manifest::ModelManifest;
use crate::pool::{LogBuffer, emit_view};

impl McpServer {
    /// Shared argument extraction and progress reporting.
    ///
    /// Defined here next to the verbs that use them so the request-shape
    /// contract stays in one file.
    pub(super) fn required_string<'a>(
        args: &'a Value,
        name: &str,
        action: &str,
    ) -> Result<&'a str> {
        args.get(name)
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| anyhow::anyhow!("'{name}' is required for action '{action}'"))
    }

    pub(super) fn get_worker_id<'a>(args: &'a Value, action: &str) -> Result<&'a str> {
        args.get("worker_id")
            .or_else(|| args.get("id"))
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| {
                anyhow::anyhow!("'worker_id' (or 'id') is required for action '{action}'")
            })
    }

    pub(super) fn get_repo_path(args: &Value) -> PathBuf {
        let repo_path_str = args
            .get("repo_path")
            .or_else(|| args.get("path"))
            .and_then(|v| v.as_str())
            .unwrap_or(".");
        PathBuf::from(repo_path_str)
    }

    /// Send a `notifications/progress` frame when the caller supplied both a
    /// token and a channel (i.e. the MCP stdio path).
    pub(super) async fn emit_progress(
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

    /// Map a `tools/call` request to its handler.
    ///
    /// The verbs are exactly [`super::schema::WORKER_ACTIONS`]; `dispatch`,
    /// `prune` and `await_worker_result` may emit progress notifications, the
    /// remaining verbs answer immediately.
    pub(super) async fn dispatch(
        &self,
        action: &str,
        args: &Value,
        token: Option<&Value>,
        tx: Option<&mpsc::Sender<String>>,
    ) -> Result<Value> {
        match action {
            "manifest" => self.handle_manifest(),
            "dispatch" => self.handle_dispatch(args, token, tx).await,
            "status" => self.handle_status(args).await,
            "collect" => self.handle_collect(args).await,
            "logs" => self.handle_logs(args).await,
            "reap" => self.handle_reap().await,
            "list" => self.handle_list().await,
            "kill" => self.handle_kill(args).await,
            "steer" => self.handle_steer(args).await,
            "prune" => self.handle_prune(args, token, tx).await,
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

        let (resolved_model, def_temp, def_turns) = self.manifest.resolve_model(requested_model);

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
        let review_after = args
            .get("review_after")
            .and_then(|v| v.as_str())
            .map(|s| {
                let (resolved, _, _) = self.manifest.resolve_model(s);
                resolved
            });

        let wid = self
            .pool
            .dispatch(
                task,
                resolved_model,
                temperature,
                repo_path,
                max_turns,
                group,
                review_after,
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
            self.await_worker_result(&wid, max_turns, token, tx).await
        } else {
            Ok(json!({
                "worker_id": wid,
                "status": "dispatched",
                "message": "Worker is executing in isolated worktree in background"
            }))
        }
    }

    async fn handle_status(&self, args: &Value) -> Result<Value> {
        let wid = Self::get_worker_id(args, "status")?;
        if let Some(state) = self.pool.get_worker_state(wid).await {
            Ok(json!({ "worker_id": wid, "state": state }))
        } else {
            let path = crate::pool::registry_dir().join(format!("{wid}.json"));
            if let Ok(content) = std::fs::read_to_string(&path)
                && let Ok(entry) =
                    serde_json::from_str::<crate::pool::WorkerRegistryEntry>(&content)
            {
                let is_alive = crate::worktree::is_process_alive(entry.pid);
                let status = if !is_alive && (entry.status == "running" || entry.status == "paused")
                {
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
    pub(super) async fn render_logs(
        &self,
        wid: &str,
    ) -> (Vec<AgentStepLog>, usize, usize, Option<String>) {
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
                && let Ok(entry) =
                    serde_json::from_str::<crate::pool::WorkerRegistryEntry>(&content)
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
