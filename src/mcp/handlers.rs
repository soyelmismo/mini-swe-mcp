//! Tool-call dispatch: the `worker` verb table and the handlers behind it.
//!
//! Turns `tools/call` into a [`Value`] payload; the JSON-RPC envelope is
//! [`crate::mcp::protocol`]'s business and the stdio plumbing is
//! [`crate::mcp::server`]'s.

use anyhow::Result;
use serde_json::{Map, Value, json};
use std::path::PathBuf;
use tokio::sync::mpsc;

use super::server::McpServer;
use crate::manifest::{ModelManifest, NetworkPolicy};
use crate::pool::emit_view;

impl McpServer {
    /// Shared argument extraction and progress reporting, defined next to the
    /// verbs that use them so the request-shape contract stays in one file.
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

    /// Parse the explicit `network` argument of a `dispatch` call.
    ///
    /// Returns `None` when the argument is absent (the caller then falls back
    /// to the resolved model's manifest policy, then the runtime default).
    /// Parsing goes through [`NetworkPolicy::parse`] so the MCP vocabulary and
    /// the manifest vocabulary are one and the same. An unknown value is a hard
    /// error rather than a silent fallback: asking for isolation and getting
    /// connectivity (or the reverse) is worse than a rejected call.
    pub(super) fn get_network_offline(args: &Value, action: &str) -> Result<Option<bool>> {
        let Some(value) = args.get("network") else {
            return Ok(None);
        };
        let mode = value
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("'network' must be a string for action '{action}'"))?;
        match NetworkPolicy::parse(mode) {
            NetworkPolicy::Offline => Ok(Some(true)),
            NetworkPolicy::Allow => Ok(Some(false)),
            NetworkPolicy::Other(other) => anyhow::bail!(
                "'{other}' is not a valid 'network' policy for action '{action}'; \
                 expected one of: {}",
                super::schema::NETWORK_MODES.join(", ")
            ),
        }
    }

    /// Resolve the effective network policy of a `dispatch` call.
    ///
    /// An explicit `network` argument wins; otherwise the resolved model's
    /// manifest `policy.network` applies when declared; otherwise the runtime
    /// default ([`super::schema::NETWORK_DEFAULT`], `"allow"`).
    pub(super) fn resolve_network_policy(
        args: &Value,
        action: &str,
        manifest: &ModelManifest,
        resolved_model: &str,
    ) -> Result<bool> {
        if let Some(explicit) = Self::get_network_offline(args, action)? {
            return Ok(explicit);
        }
        Ok(matches!(
            manifest.network_policy(resolved_model),
            Some(NetworkPolicy::Offline)
        ))
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
    /// token and a channel (the MCP stdio path).
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
    /// Verbs are exactly [`super::schema::WORKER_ACTIONS`]; `dispatch`, `prune`
    /// and `await_worker_result` may emit progress notifications, the rest
    /// answer immediately.
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

        let network_offline =
            Self::resolve_network_policy(args, "dispatch", &self.manifest, &resolved_model)?;

        // Optional verify gate: an explicit string (possibly empty to disable)
        // is passed through; an absent argument lets the pool auto-detect.
        let verify = args
            .get("verify")
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
                review_after,
                network_offline,
                verify,
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
                "network": if network_offline { "offline" } else { super::schema::NETWORK_DEFAULT },
                "message": "Worker is executing in isolated worktree in background"
            }))
        }
    }

    async fn handle_status(&self, args: &Value) -> Result<Value> {
        let wid = Self::get_worker_id(args, "status")?;
        if let Some(state) = self.pool.get_worker_state(wid).await {
            Ok(json!({ "worker_id": wid, "state": state }))
        } else if let Some(entry) = crate::pool::load_registry_entry(wid) {
            let state_name = entry.status.display_name();
            Ok(json!({
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
                        "error": if entry.status == crate::pool::RegistryStatus::Failed { Some(entry.last_command) } else { None },
                        "question": entry.question,
                        "pid": entry.pid,
                        "started_at": entry.started_at,
                        "metrics": entry.metrics,
                    }
                }
            }))
        } else {
            anyhow::bail!("Worker not found: {wid}")
        }
    }

    /// Render the bounded tail of a live worker's step history plus the
    /// counters that make the degradation explicit.
    pub(super) async fn render_logs(&self, wid: &str) -> LogView {
        let Some(buffer) = self.pool.get_worker_logs(wid).await
        else {
            return LogView::default();
        };
        let view = emit_view(&buffer, self.pool.log_policy().max_emitted);
        LogView {
            logs: view.logs,
            logs_omitted: view.logs_omitted,
            logs_dropped: buffer.dropped(),
            logs_truncation_notice: view.logs_truncation_notice,
        }
    }

    /// `logs` action: inspect a live worker's retained history without
    /// collecting (and thus evicting) it.
    async fn handle_logs(&self, args: &Value) -> Result<Value> {
        let wid = Self::get_worker_id(args, "logs")?;
        let Some(buffer) = self.pool.get_worker_logs(wid).await else {
            anyhow::bail!("Worker not found: {wid}")
        };
        let policy = self.pool.log_policy();
        let view = emit_view(&buffer, policy.max_emitted);
        let log_view = LogView {
            logs: view.logs,
            logs_omitted: view.logs_omitted,
            logs_dropped: buffer.dropped(),
            logs_truncation_notice: view.logs_truncation_notice,
        };
        let mut result = json!({
            "worker_id": wid,
            "state": self.pool.get_worker_state(wid).await,
            "total_steps": buffer.total(),
            "logs_retained": buffer.retained(),
            "retention": {
                "max_retained": policy.max_retained,
                "max_emitted": policy.max_emitted,
            },
        });
        if let serde_json::Value::Object(map) = &mut result {
            map.extend(log_view.as_map());
        }
        Ok(result)
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
            let log_view = LogView {
                logs: collected.logs,
                logs_omitted: collected.logs_omitted,
                logs_dropped: collected.logs_dropped,
                logs_truncation_notice: collected.logs_truncation_notice,
            };
            let mut result = json!({
                "worker_id": wid,
                "state": collected.state,
            });
            if let serde_json::Value::Object(map) = &mut result {
                map.extend(log_view.as_map());
            }
            Ok(result)
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
        } else if let Some(entry) = crate::pool::load_registry_entry(wid)
            && crate::worktree::is_process_alive(entry.pid)
        {
            #[cfg(unix)]
            {
                // SAFETY: `entry.pid` is a foreign pid read from the registry;
                // `kill` only signals, never dereferences, so this is safe.
                unsafe {
                    libc::kill(entry.pid as libc::pid_t, libc::SIGTERM);
                }
            }
            Ok(json!({ "worker_id": wid, "killed": true }))
        } else {
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

/// The four-field log view shared by `logs`, `collect` and `await_worker_result`.
///
/// [`LogView::as_map`] flattens the keys into the top level of the enclosing
/// response, exactly as before.
#[derive(Debug, Default, serde::Serialize)]
pub(super) struct LogView {
    pub(super) logs: Vec<crate::agent::AgentStepLog>,
    pub(super) logs_omitted: usize,
    pub(super) logs_dropped: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) logs_truncation_notice: Option<String>,
}

impl LogView {
    /// The four log keys as a JSON map, for embedding into a `json!` response.
    pub(super) fn as_map(&self) -> Map<String, Value> {
        let mut map = Map::new();
        map.insert("logs".to_string(), serde_json::to_value(&self.logs).unwrap_or_default());
        map.insert("logs_omitted".to_string(), json!(self.logs_omitted));
        map.insert("logs_dropped".to_string(), json!(self.logs_dropped));
        map.insert(
            "logs_truncation_notice".to_string(),
            serde_json::to_value(&self.logs_truncation_notice).unwrap_or_default(),
        );
        map
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An omitted `network` yields `None`: the caller falls back to the
    /// manifest policy, then the runtime default.
    #[test]
    fn network_absent_yields_none() {
        let args = json!({ "action": "dispatch", "task": "t" });
        assert_eq!(
            McpServer::get_network_offline(&args, "dispatch").expect("absent is not an error"),
            None,
            "an omitted network policy must defer to the manifest/default"
        );
    }

    /// An explicit `offline` is the opt-in that turns isolation on, and
    /// `allow` is the explicit spelling of the default.
    #[test]
    fn network_offline_and_allow_are_both_accepted() {
        assert_eq!(
            McpServer::get_network_offline(&json!({ "network": "offline" }), "dispatch")
                .expect("offline must be accepted"),
            Some(true)
        );
        assert_eq!(
            McpServer::get_network_offline(&json!({ "network": "allow" }), "dispatch")
                .expect("allow must be accepted"),
            Some(false)
        );
    }

    /// An unknown policy (or a non-string) is rejected instead of silently
    /// falling back: a caller that asked for isolation must never silently get
    /// connectivity instead.
    #[test]
    fn an_unknown_network_policy_is_a_hard_error() {
        let err = McpServer::get_network_offline(&json!({ "network": "offine" }), "dispatch")
            .expect_err("a typo must not be accepted");
        assert!(err.to_string().contains("not a valid 'network' policy"), "{err}");

        let err = McpServer::get_network_offline(&json!({ "network": true }), "dispatch")
            .expect_err("a non-string network must not be accepted");
        assert!(err.to_string().contains("must be a string"), "{err}");
    }

    /// An explicit argument wins over the manifest policy.
    #[test]
    fn explicit_network_argument_wins_over_manifest() {
        let manifest = ModelManifest::default();
        // ninja declares `allow` in the built-in manifest.
        assert!(
            McpServer::resolve_network_policy(
                &json!({ "network": "offline" }),
                "dispatch",
                &manifest,
                "combo:ninja",
            )
            .expect("explicit offline must win"),
            "an explicit offline must override the manifest's allow"
        );
        assert!(
            !McpServer::resolve_network_policy(
                &json!({ "network": "allow" }),
                "dispatch",
                &manifest,
                "combo:ninja",
            )
            .expect("explicit allow must win"),
            "an explicit allow must override the manifest's allow"
        );
    }

    /// When the argument is omitted, the resolved model's manifest policy
    /// applies.
    #[test]
    fn manifest_policy_applies_when_argument_omitted() {
        let manifest = ModelManifest::default();
        // ninja declares `allow` in the built-in manifest.
        assert!(
            !McpServer::resolve_network_policy(
                &json!({}),
                "dispatch",
                &manifest,
                "combo:ninja",
            )
            .expect("manifest policy must apply"),
            "ninja's manifest policy is allow"
        );
    }

    /// A manifest that declares `offline` isolates the worker when the
    /// argument is omitted.
    #[test]
    fn manifest_offline_policy_isolates_when_argument_omitted() {
        let mut manifest = ModelManifest::default();
        manifest.models.insert(
            "sealed".to_string(),
            crate::manifest::ModelDefinition {
                id: "vendor:sealed".to_string(),
                role: None,
                temperature: None,
                max_turns: None,
                policy: Some(crate::manifest::ExecutionPolicy {
                    network: Some(NetworkPolicy::Offline),
                }),
            },
        );
        assert!(
            McpServer::resolve_network_policy(
                &json!({}),
                "dispatch",
                &manifest,
                "vendor:sealed",
            )
            .expect("manifest offline must apply"),
            "a model declaring offline must isolate the worker"
        );
    }

    /// When neither the argument nor the manifest declares a policy, the
    /// runtime default (`allow`) applies.
    #[test]
    fn runtime_default_applies_when_nothing_declared() {
        let manifest = ModelManifest::default();
        // An unknown model has no manifest entry, so no policy is declared.
        assert!(
            !McpServer::resolve_network_policy(
                &json!({}),
                "dispatch",
                &manifest,
                "some/unknown-model",
            )
            .expect("default must apply"),
            "an undeclared policy must fall back to allow"
        );
    }
}
