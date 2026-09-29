//! Tool-call dispatch: the `worker` verb table and the handlers behind it.
//!
//! Turns `tools/call` into a [`Value`] payload; the JSON-RPC envelope is
//! [`crate::mcp::protocol`]'s business and the stdio plumbing is
//! [`crate::mcp::server`]'s.

use anyhow::Result;
use serde_json::{Value, json};
use std::path::PathBuf;
use tokio::sync::mpsc;

use super::server::McpServer;
use crate::agent::AgentStepLog;
use crate::manifest::ModelManifest;
use crate::pool::{LogBuffer, emit_view};

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

    /// Resolve the declarative network policy of a `dispatch` call.
    ///
    /// Optional, defaulting to [`super::schema::NETWORK_DEFAULT`] (`"allow"`),
    /// so a client that never sends it keeps its prior behaviour. A value
    /// outside [`NETWORK_MODES`](super::schema::NETWORK_MODES) is a hard error
    /// rather than a silent fallback: asking for isolation and getting
    /// connectivity (or the reverse) is worse than a rejected call.
    pub(super) fn get_network_offline(args: &Value, action: &str) -> Result<bool> {
        let Some(value) = args.get("network") else {
            return Ok(false);
        };
        let mode = value
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("'network' must be a string for action '{action}'"))?;
        match mode {
            "offline" => Ok(true),
            mode if super::schema::NETWORK_MODES.contains(&mode) => Ok(false),
            other => anyhow::bail!(
                "'{other}' is not a valid 'network' policy for action '{action}';                  expected one of: {}",
                super::schema::NETWORK_MODES.join(", ")
            ),
        }
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
    ///
    /// Unthrottled: every call writes one frame. Rapid successive updates
    /// should route through [`ProgressThrottle`] via
    /// [`Self::emit_progress_throttled`] instead.
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

    /// Send a `notifications/progress` frame only when the previous one is at
    /// least [`PROGRESS_MIN_INTERVAL`] old, coalescing updates inside that
    /// window.
    ///
    /// The first frame is always allowed so the client observes the work start.
    /// Terminal notifications keep using the unthrottled [`Self::emit_progress`]
    /// so a completion is never swallowed.
    pub(super) async fn emit_progress_throttled(
        throttle: &mut ProgressThrottle,
        tx: Option<&mpsc::Sender<String>>,
        token: Option<&Value>,
        progress: usize,
        total: usize,
        message: impl std::fmt::Display,
    ) {
        if !throttle.should_emit() {
            return;
        }
        Self::emit_progress(tx, token, progress, total, message).await;
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

        let network_offline = Self::get_network_offline(args, "dispatch")?;

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
                    "reviewing" => "Reviewing",
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

    /// Render the bounded tail of a live worker's step history plus the
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

/// Minimum spacing between two `notifications/progress` frames from one
/// throttled emitter (today: the `await_worker_result` step loop).
///
/// A worker can finish several steps in far less than 100 ms; without a floor a
/// fast worker would flood stdout with a frame per step. Frames inside the
/// window are coalesced and the next one past it carries the then-current step,
/// so the client still converges on the true progress.
pub(super) const PROGRESS_MIN_INTERVAL: std::time::Duration =
    std::time::Duration::from_millis(100);

/// Per-invocation micro-throttle for progress notifications.
///
/// Holds the monotonic instant of the last frame written. The first
/// [`Self::should_emit`] returns `true` (priming the stream), thereafter a
/// frame is allowed only once [`PROGRESS_MIN_INTERVAL`] has elapsed. An
/// allocation-free `Instant` comparison, created fresh per `tools/call` so
/// unrelated requests never throttle each other.
pub(super) struct ProgressThrottle {
    last_emit: Option<tokio::time::Instant>,
}

impl ProgressThrottle {
    /// A throttle that has not emitted a frame yet.
    pub(super) fn new() -> Self {
        Self { last_emit: None }
    }

    /// Whether a progress frame may be written now, latching the clock.
    ///
    /// On `true` the clock advances to `now`, collapsing a burst of ticks
    /// inside one interval into a single frame.
    pub(super) fn should_emit(&mut self) -> bool {
        let now = tokio::time::Instant::now();
        match self.last_emit {
            Some(prev) if now.duration_since(prev) < PROGRESS_MIN_INTERVAL => false,
            _ => {
                self.last_emit = Some(now);
                true
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A burst of ticks inside one interval must collapse to a single frame:
    /// this is the whole point of the throttle (no stdio flooding).
    #[tokio::test]
    async fn rapid_ticks_are_coalesced_into_one_frame() {
        let (tx, mut rx) = mpsc::channel::<String>(64);
        let token = json!("burst-token");
        let mut throttle = ProgressThrottle::new();

        // Ten "steps" fired back-to-back, far inside the 100 ms window.
        for step in 1..=10 {
            McpServer::emit_progress_throttled(
                &mut throttle,
                Some(&tx),
                Some(&token),
                step,
                10,
                format!("Step {step}"),
            )
            .await;
        }

        // Only the priming frame got through.
        let emitted = rx.try_recv().expect("the first tick must be emitted");
        assert!(emitted.contains("notifications/progress"), "got: {emitted}");
        assert!(rx.try_recv().is_err(), "burst must not flood stdio");
    }

    /// Once the window elapses, progress must flow again, and the frame carries
    /// the then-current value so the client still converges.
    #[tokio::test]
    async fn progress_resumes_after_the_minimum_interval() {
        let (tx, mut rx) = mpsc::channel::<String>(64);
        let token = json!("paced-token");
        let mut throttle = ProgressThrottle::new();

        McpServer::emit_progress_throttled(&mut throttle, Some(&tx), Some(&token), 1, 5, "one")
            .await;
        // Inside the window: coalesced.
        McpServer::emit_progress_throttled(&mut throttle, Some(&tx), Some(&token), 2, 5, "two")
            .await;
        assert!(rx.try_recv().is_ok(), "first frame must be emitted");
        assert!(rx.try_recv().is_err(), "in-window frame must be dropped");

        // Past the window: emitted again.
        tokio::time::sleep(PROGRESS_MIN_INTERVAL + std::time::Duration::from_millis(20)).await;
        McpServer::emit_progress_throttled(&mut throttle, Some(&tx), Some(&token), 3, 5, "three")
            .await;

        let frame = rx.try_recv().expect("frame after the window must be emitted");
        assert!(frame.contains(r#""progress":3"#), "got: {frame}");
        assert!(rx.try_recv().is_err(), "only one frame per window");
    }

    /// The gate itself: first call passes, the rest inside the window do not.
    #[tokio::test]
    async fn should_emit_primes_once_then_gates_the_window() {
        let mut throttle = ProgressThrottle::new();
        assert!(throttle.should_emit(), "the first frame must prime the stream");
        for tick in 0..1000 {
            assert!(
                !throttle.should_emit(),
                "tick {tick} landed inside the window and must be coalesced"
            );
        }
        tokio::time::sleep(PROGRESS_MIN_INTERVAL + std::time::Duration::from_millis(20)).await;
        assert!(
            throttle.should_emit(),
            "after the window a frame must be allowed again"
        );
    }

    /// An omitted `network` must keep the pre-existing connected behaviour:
    /// the property is additive, so a client that never sends it is unaffected.
    #[test]
    fn network_defaults_to_allow_when_absent() {
        let args = json!({ "action": "dispatch", "task": "t" });
        assert!(
            !McpServer::get_network_offline(&args, "dispatch").expect("absent is not an error"),
            "an omitted network policy must not isolate the worker"
        );
    }

    /// An explicit `offline` is the opt-in that turns isolation on, and
    /// `allow` is the explicit spelling of the default.
    #[test]
    fn network_offline_and_allow_are_both_accepted() {
        assert!(
            McpServer::get_network_offline(&json!({ "network": "offline" }), "dispatch")
                .expect("offline must be accepted")
        );
        assert!(
            !McpServer::get_network_offline(&json!({ "network": "allow" }), "dispatch")
                .expect("allow must be accepted")
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

    /// The interval is the documented 100 ms floor.
    #[test]
    fn min_interval_is_100ms() {
        assert_eq!(PROGRESS_MIN_INTERVAL, std::time::Duration::from_millis(100));
    }

    /// Throttling is best-effort framing, never a tool failure: the channel is
    /// optional, and a missing token/channel simply emits nothing.
    #[tokio::test]
    async fn throttled_emit_is_a_noop_without_a_token_or_channel() {
        let mut throttle = ProgressThrottle::new();
        // No channel.
        McpServer::emit_progress_throttled(&mut throttle, None, Some(&json!("t")), 1, 1, "x").await;
        // No token.
        McpServer::emit_progress_throttled(&mut throttle, Some(&mpsc::channel(1).0), None, 1, 1, "x")
            .await;
    }
}
