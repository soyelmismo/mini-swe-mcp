//! Tool-call dispatch: the `worker` verb table and the handlers behind it.
//!
//! Turns `tools/call` into a [`Value`] payload; the JSON-RPC envelope is
//! [`crate::mcp::protocol`]'s business and the stdio plumbing is
//! [`crate::mcp::server`]'s.
//!
//! The verb table is also where per-agent ownership (H-3) is decided: the
//! connection context names the agent, every worker records the agent that
//! dispatched it, and the verbs with a side effect on a worker refuse anyone
//! but its owner (or the operator's `admin` connection).

use anyhow::Result;
use serde_json::{Map, Value, json};
use std::path::PathBuf;
use std::time::Duration;
use tokio::sync::mpsc;

use super::server::McpServer;
use crate::manifest::{ModelManifest, NetworkPolicy};
use crate::pool::{SteerOutcome, UNATTRIBUTED_OWNER, WorkerOwner, emit_view};

/// Owner label used when neither the pool nor the registry has a row.
const UNKNOWN_OWNER: &str = "unknown";

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

    /// Resolve the `worker_id`/`id` argument to the full id of one of the
    /// caller's own workers.
    ///
    /// Accepts the full id, `last` (the caller's most recently dispatched
    /// worker) and any unique prefix of at least three characters. An id
    /// nothing of the caller's matches is passed through so the verb answers
    /// with its own "not found"; an ambiguous prefix is refused here, listing
    /// only the caller's matching ids. The response always carries the full id.
    pub(super) async fn resolve_worker_id(
        &self,
        args: &Value,
        action: &str,
        ctx: &super::server::ConnectionContext,
    ) -> Result<String> {
        let needle = Self::get_worker_id(args, action)?;
        self.pool.resolve_worker_id(needle, &ctx.agent()).await
    }

    /// Parse the optional `timeout_secs` deadline of a blocking call.
    ///
    /// Clients disagree wildly on how long a tool call may run, so a caller
    /// that would rather re-poll than risk its own deadline passes the budget
    /// explicitly. An absent argument means "wait indefinitely", which is the
    /// historical behaviour; a non-integer one is a hard error rather than a
    /// silently dropped deadline, because dropping it is what makes the call
    /// hang.
    pub(super) fn get_timeout(args: &Value, action: &str) -> Result<Option<std::time::Duration>> {
        let Some(value) = args.get("timeout_secs") else {
            return Ok(None);
        };
        let secs = value.as_u64().ok_or_else(|| {
            anyhow::anyhow!("'timeout_secs' must be a non-negative integer for action '{action}'")
        })?;
        Ok(Some(std::time::Duration::from_secs(secs)))
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

    pub(super) fn get_repo_path(args: &Value, ctx: &super::server::ConnectionContext) -> PathBuf {
        let repo_path_str = args
            .get("repo_path")
            .or_else(|| args.get("path"))
            .and_then(|v| v.as_str())
            .unwrap_or(".");
        // A bare "." names the caller's own directory, not a child of it.
        if repo_path_str == "." {
            return ctx.cwd.clone().unwrap_or_else(|| PathBuf::from("."));
        }
        let path = PathBuf::from(repo_path_str);
        if path.is_relative()
            && let Some(cwd) = &ctx.cwd
        {
            cwd.join(path)
        } else {
            path
        }
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

    /// Longest gap between two progress notifications while a wait blocks.
    ///
    /// A worker can sit inside one long step (a slow build, a big test run)
    /// for far longer than a step change, and an MCP client aborts a stdio call
    /// that sends no notification at all as idle. The wait loop therefore
    /// re-emits the current step on this interval, well under the 30-minute
    /// idle timeout.
    pub const PROGRESS_HEARTBEAT_INTERVAL: std::time::Duration = std::time::Duration::from_secs(60);

    /// Coarse fallback tick for a worker this process does not own.
    ///
    /// Its state changes happen in another process, so no in-process
    /// notification can arrive and the registry has to be re-read. Thirty
    /// seconds is far coarser than the 500 ms poll this replaced, because the
    /// only thing it can still discover is a transition that already happened.
    pub(super) const CROSS_PROCESS_TICK: std::time::Duration = std::time::Duration::from_secs(30);

    /// Tick used while waiting on a worker this process owns.
    ///
    /// Such a worker wakes the wait through the change subscription, so this
    /// only has to be short enough to carry the heartbeat deadline.
    pub(super) const HEARTBEAT_TICK: std::time::Duration = std::time::Duration::from_secs(1);

    /// Map a `tools/call` request to its handler.
    ///
    /// Verbs are exactly [`super::schema::WORKER_ACTIONS`]; `dispatch`,
    /// `prune`, `wait` and a `steer` carrying `wait: true` may emit progress
    /// notifications, the rest answer immediately.
    pub(super) async fn dispatch(
        &self,
        action: &str,
        args: &Value,
        token: Option<&Value>,
        tx: Option<&mpsc::Sender<String>>,
        ctx: &super::server::ConnectionContext,
    ) -> Result<Value> {
        let result = match action {
            "manifest" => self.handle_manifest(),
            "dispatch" => self.handle_dispatch(args, token, tx, ctx).await,
            "status" => self.handle_status(args, ctx).await,
            "collect" => self.handle_collect(args, ctx).await,
            "review" => self.handle_review(args, ctx).await,
            "approve" => self.handle_approve(args, ctx).await,
            "unapprove" => self.handle_unapprove(args, ctx).await,
            "logs" => self.handle_logs(args, ctx).await,
            "reap" => self.handle_reap().await,
            "list" => self.handle_list(args, ctx).await,
            "kill" => self.handle_kill(args, ctx).await,
            "steer" => self.handle_steer(args, token, tx, ctx).await,
            "watch" => self.handle_watch(args, ctx).await,
            "prune" => self.handle_prune(args, token, tx, ctx).await,
            "merge" => self.handle_merge(args, ctx).await,
            _ => anyhow::bail!("Unknown action or tool: {action}"),
        };
        // Looking at or acting on a worker is the owner having seen it: drop
        // that worker's queued watch events so a later watch does not replay
        // them as "while you were not watching".
        if matches!(action, "status" | "logs" | "collect" | "kill" | "steer")
            && result.is_ok()
            && let Ok(wid) = Self::get_worker_id(args, action)
        {
            self.hub_events.lock().await.mark_seen(&ctx.agent(), wid);
        }
        result
    }

    /// Owner label of `wid` for a payload: the recorded agent, or a marker for
    /// a row that names none / a worker nothing knows about.
    pub(super) async fn owner_of(&self, wid: &str) -> String {
        match self.pool.worker_owner(wid).await {
            Some(WorkerOwner::Agent(owner)) => owner,
            Some(WorkerOwner::Unattributed) => UNATTRIBUTED_OWNER.to_string(),
            None => UNKNOWN_OWNER.to_string(),
        }
    }

    /// Refuse to act on a worker another agent owns (H-3).
    ///
    /// Applied to every verb with a side effect on the worker itself: `steer`,
    /// `kill`, `collect`, `wait` and the wait a `dispatch`/`steer` blocks on.
    /// `status` and `logs` stay open to every agent, and a worker neither the
    /// pool nor the registry knows is left to the verb's own "not found"
    /// answer.
    ///
    /// This is a coordination boundary between the agents sharing one hub, not
    /// an authentication one: any process of this user may name itself another
    /// agent, or ask for `admin`. What keeps a worker private to its user is
    /// still the hub daemon's `SO_PEERCRED` check.
    async fn require_owner(&self, wid: &str, ctx: &super::server::ConnectionContext) -> Result<()> {
        if ctx.is_admin() {
            return Ok(());
        }
        match self.pool.worker_owner(wid).await {
            Some(WorkerOwner::Agent(owner)) if owner != ctx.agent() => {
                anyhow::bail!("worker {wid} belongs to agent {owner}")
            }
            Some(WorkerOwner::Unattributed) => anyhow::bail!(
                "worker {wid} has no recorded owner (dispatched before ownership was tracked)"
            ),
            _ => Ok(()),
        }
    }

    /// The shell command that waits on *this* caller's workers.
    ///
    /// A shell cannot know its session — the agent's `bash` tool sees none of
    /// the session variables — so the token is what binds the command to the
    /// caller: the hub resolves it back to exactly this identity, never to
    /// `admin` and never to another agent's. `None` for a caller with no token
    /// store, which is the in-process stdio server: it has no hub, so its
    /// `watch` runs in the same process.
    fn watch_command(ctx: &super::server::ConnectionContext) -> Option<String> {
        let token = ctx.token_store()?.token_for(&ctx.agent())?;
        Some(format!("MINI_SWE_WATCH_TOKEN={token} mini-swe-mcp watch"))
    }

    /// Add `watch_command` to a dispatch or steer payload, when this caller has
    /// a token store to mint from.
    fn with_watch_command(payload: &mut Value, ctx: &super::server::ConnectionContext) {
        if let Some(command) = Self::watch_command(ctx) {
            payload["watch_command"] = json!(command);
        }
    }

    /// Per-agent worker cap, `MAX_WORKERS_PER_AGENT`; `0` (the default) is
    /// unlimited.
    pub(super) fn max_workers_per_agent() -> usize {
        crate::config::env_parse("MAX_WORKERS_PER_AGENT").unwrap_or(0)
    }

    /// Refuse a dispatch that would push `agent` past its cap, naming the
    /// workers already running so the caller can pick one to collect.
    async fn check_agent_cap(&self, agent: &str, cap: usize) -> Result<()> {
        if cap == 0 {
            return Ok(());
        }
        let running = self.pool.active_workers_of(agent).await;
        if running.len() < cap {
            return Ok(());
        }
        anyhow::bail!(
            "agent {agent} already has {} running workers (limit MAX_WORKERS_PER_AGENT={cap}): {}",
            running.len(),
            running.join(", ")
        )
    }

    fn handle_manifest(&self) -> Result<Value> {
        Ok(json!({
            "default_model": self.manifest.default,
            "models": self.manifest.models,
        }))
    }

    /// Dispatch one worker, or a batch when the call carries `tasks`.
    ///
    /// A batch is a list of `{task, model?, repo_path?, max_turns?, verify?,
    /// group?, network?}` objects; shared top-level dispatch values act as
    /// defaults for every entry. Each entry goes through the same single-task
    /// path, so one bad entry only answers with its own error.
    async fn handle_dispatch(
        &self,
        args: &Value,
        token: Option<&Value>,
        tx: Option<&mpsc::Sender<String>>,
        ctx: &super::server::ConnectionContext,
    ) -> Result<Value> {
        if args.get("tasks").is_some() {
            return self.handle_batch_dispatch(args, token, tx, ctx).await;
        }
        self.dispatch_one(args, token, tx, ctx).await
    }

    /// The dispatch properties a batch entry may set, and whose top-level value
    /// becomes the default for every entry.
    const BATCH_ENTRY_KEYS: &'static [&'static str] = &[
        "task",
        "model",
        "repo_path",
        "path",
        "max_turns",
        "temperature",
        "review_after",
        "group",
        "network",
        "verify",
    ];

    /// Effective arguments of one batch entry: the shared top-level dispatch
    /// values, overridden by the entry's own keys.
    fn batch_entry_args(shared: &Value, entry: &Value) -> Result<Map<String, Value>> {
        let entry = entry.as_object().ok_or_else(|| {
            anyhow::anyhow!("each 'tasks' entry must be an object such as {{task, model}}")
        })?;
        let mut merged = Map::new();
        for key in Self::BATCH_ENTRY_KEYS {
            if let Some(value) = entry.get(*key).or_else(|| shared.get(*key)) {
                merged.insert((*key).to_string(), value.clone());
            }
        }
        Ok(merged)
    }

    /// Dispatch every entry of a batch, one compact result per task.
    ///
    /// A bad entry reports its own error in its slot; it never aborts the
    /// others. Each slot is `{index, worker_id, network}` on success and
    /// `{index, error}` on failure.
    async fn handle_batch_dispatch(
        &self,
        args: &Value,
        token: Option<&Value>,
        tx: Option<&mpsc::Sender<String>>,
        ctx: &super::server::ConnectionContext,
    ) -> Result<Value> {
        let tasks = args
            .get("tasks")
            .and_then(Value::as_array)
            .ok_or_else(|| anyhow::anyhow!("'tasks' must be an array for action 'dispatch'"))?;
        if tasks.is_empty() {
            anyhow::bail!("'tasks' must contain at least one task for action 'dispatch'");
        }

        let mut workers = Vec::with_capacity(tasks.len());
        let mut dispatched = 0usize;
        let mut failed = 0usize;
        for (index, entry) in tasks.iter().enumerate() {
            let outcome = match Self::batch_entry_args(args, entry) {
                Ok(entry_args) => self
                    .dispatch_one(&Value::Object(entry_args), token, tx, ctx)
                    .await
                    .map(|payload| {
                        json!({
                            "index": index,
                            "worker_id": payload.get("worker_id").cloned().unwrap_or(Value::Null),
                            "network": payload
                                .get("network")
                                .cloned()
                                .unwrap_or_else(|| json!(super::schema::NETWORK_DEFAULT)),
                        })
                    }),
                Err(error) => Err(error),
            };
            match outcome {
                Ok(worker) => {
                    dispatched += 1;
                    workers.push(worker);
                }
                Err(error) => {
                    failed += 1;
                    workers.push(json!({ "index": index, "error": error.to_string() }));
                }
            }
        }

        let mut payload = json!({
            "workers": workers,
            "dispatched": dispatched,
            "failed": failed,
            "message": "Workers are executing in isolated worktrees in background. Use 'watch' (or mini-swe-mcp watch) to wait for them.",
        });
        Self::with_watch_command(&mut payload, ctx);
        Ok(payload)
    }

    /// Dispatch exactly one worker from a `dispatch` argument object.
    ///
    /// Shared by the single-task form and every batch entry, so both go through
    /// one validation path.
    async fn dispatch_one(
        &self,
        args: &Value,
        token: Option<&Value>,
        tx: Option<&mpsc::Sender<String>>,
        ctx: &super::server::ConnectionContext,
    ) -> Result<Value> {
        let task = Self::required_string(args, "task", "dispatch")?.to_string();
        let agent = ctx.agent();
        // Fairness gate: one agent may not fill the pool, so its dispatches
        // stop at `MAX_WORKERS_PER_AGENT` running workers (0 = unlimited).
        self.check_agent_cap(&agent, Self::max_workers_per_agent())
            .await?;
        let repo_path = Self::get_repo_path(args, ctx);
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

        let group = args
            .get("group")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let role = match args.get("role") {
            None => crate::pool::WorkerRole::Worker,
            Some(Value::String(role)) if role == "worker" => crate::pool::WorkerRole::Worker,
            Some(Value::String(role)) if role == "consolidate" => {
                crate::pool::WorkerRole::Consolidate
            }
            _ => anyhow::bail!("role must be 'worker' or 'consolidate'"),
        };
        if role == crate::pool::WorkerRole::Consolidate
            && group.as_deref().is_none_or(|g| g.trim().is_empty())
        {
            anyhow::bail!("role 'consolidate' requires 'group'");
        }
        let review_after = args.get("review_after").and_then(|v| v.as_str()).map(|s| {
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

        let admission = self.admit_worker().await?;
        let wid = self
            .pool
            .dispatch_with_role(
                agent.clone(),
                task,
                resolved_model,
                temperature,
                repo_path,
                max_turns,
                group,
                review_after,
                network_offline,
                verify,
                ctx.client_env.clone(),
                role,
            )
            .await?;
        drop(admission);

        Self::emit_progress(
            tx,
            token,
            0,
            max_turns,
            format!("Worker {wid} dispatched in isolated worktree"),
        )
        .await;

        // Dispatch never blocks: the worker id is the whole handle, and the
        // event the caller actually wants arrives through `watch`.
        let mut payload = json!({
            "worker_id": wid,
            "owner": agent,
            "status": "dispatched",
            "network": if network_offline { "offline" } else { super::schema::NETWORK_DEFAULT },
            "message": "Worker is executing in isolated worktree in background. Use 'watch' (or mini-swe-mcp watch) to wait for its next event."
        });
        Self::with_watch_command(&mut payload, ctx);
        Ok(payload)
    }

    /// `status` action: the caller's own view of one worker's step and
    /// progress. A foreign worker id is refused with its owner, never its
    /// task or state.
    async fn handle_status(
        &self,
        args: &Value,
        ctx: &super::server::ConnectionContext,
    ) -> Result<Value> {
        let resolved = self.resolve_worker_id(args, "status", ctx).await?;
        let wid = resolved.as_str();
        self.require_owner(wid, ctx).await?;
        if let Some(state) = self.pool.get_worker_state(wid).await {
            // A finished worker's status carries the same review guidance as
            // the wait payload, so polling the status is enough to learn the
            // loop exists.
            let next_step = match &state {
                crate::pool::WorkerState::Completed { .. }
                | crate::pool::WorkerState::Failed { .. } => Some(crate::pool::next_step_for(
                    crate::pool::terminal_branch(&state).as_deref(),
                )),
                _ => None,
            };
            Ok(json!({
                "worker_id": wid,
                "owner": self.owner_of(wid).await,
                "state": state,
                "next_step": next_step,
            }))
        } else if let Some(entry) =
            crate::pool::load_registry_entry_in(self.pool.scratch_root(), wid)
        {
            let state_name = entry.status.display_name();
            // A registry-only terminal row (collected worker, restarted hub)
            // carries the same review guidance as the live path.
            let next_step = entry
                .status
                .is_terminal()
                .then(|| crate::pool::next_step_for(None));
            Ok(json!({
                "worker_id": wid,
                "owner": crate::pool::registry_owner_label(&entry),
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
                },
                "next_step": next_step,
            }))
        } else {
            anyhow::bail!("Worker not found: {wid}")
        }
    }

    /// Render the bounded tail of a live worker's step history plus the
    /// counters that make the degradation explicit.
    pub(super) async fn render_logs(&self, wid: &str) -> LogView {
        let Some(buffer) = self.pool.get_worker_logs(wid).await else {
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
    /// collecting (and thus evicting) it. Same ownership rule as `status`.
    async fn handle_logs(
        &self,
        args: &Value,
        ctx: &super::server::ConnectionContext,
    ) -> Result<Value> {
        let resolved = self.resolve_worker_id(args, "logs", ctx).await?;
        let wid = resolved.as_str();
        self.require_owner(wid, ctx).await?;
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

    /// `collect` action: the worker's final answer, with the diff summarised
    /// unless the caller asks for it.
    ///
    /// The default reply is the compact one — summary, verified, per-file diff
    /// stat and branch — because the full diff of a large task is what makes a
    /// review expensive. `full: true` restores the whole diff, and
    /// `files: [...]` narrows it to the named paths.
    async fn handle_collect(
        &self,
        args: &Value,
        ctx: &super::server::ConnectionContext,
    ) -> Result<Value> {
        let resolved = self.resolve_worker_id(args, "collect", ctx).await?;
        let wid = resolved.as_str();
        self.require_owner(wid, ctx).await?;
        let full = args.get("full").and_then(Value::as_bool).unwrap_or(false);
        let files = Self::get_diff_files(args, "collect")?;
        if let Some(collected) = self.pool.collect(wid).await {
            let log_view = LogView {
                logs: collected.logs,
                logs_omitted: collected.logs_omitted,
                logs_dropped: collected.logs_dropped,
                logs_truncation_notice: collected.logs_truncation_notice,
            };
            let (summary, verified, branch) = completed_fields(Some(&collected.state));
            // Collect ends the worker's reviewable life, so the guidance is
            // about the branch it leaves behind rather than a further steer.
            let next_step = crate::pool::next_step_for(branch.as_deref());
            let mut state = serde_json::to_value(&collected.state).unwrap_or_default();
            let diff = state
                .pointer("/details/diff")
                .and_then(|value| value.as_str())
                .unwrap_or_default()
                .to_owned();
            let stats = diff_file_stats(&diff);
            // The diff is the one field that can be arbitrarily large, so it
            // leaves the payload unless it was asked for — whole, or narrowed
            // to the files that were named.
            if let Some(details) = state.pointer_mut("/details").and_then(Value::as_object_mut) {
                let scoped = diff_of_files(&diff, &files);
                if !scoped.is_empty() {
                    details.insert("diff".to_string(), json!(scoped));
                } else if !full {
                    details.remove("diff");
                }
            }
            let mut result = json!({
                "worker_id": wid,
                "owner": collected.owner,
                "state": state,
                "summary": summary,
                "verified": verified,
                "branch": branch,
                "diff_stat": diff_stat_value(&stats),
                "next_step": next_step,
            });
            if let serde_json::Value::Object(map) = &mut result {
                map.extend(log_view.as_map());
            }
            Ok(result)
        } else {
            anyhow::bail!("Worker not found: {wid}")
        }
    }

    /// `review` action: one compact view of a worker's branch, ending with the
    /// command that acts on it.
    ///
    /// Reviewing used to mean `status`, plus `collect` (the whole diff, however
    /// large), plus `logs`, plus a hand-run `git merge-tree`. This is the same
    /// answer in one bounded payload: the task's first line, what verification
    /// said, the per-file diff stat, the revision, and whether the branch still
    /// merges cleanly into the base branch tip. Read-only — unlike `collect` it
    /// never evicts the worker.
    async fn handle_review(
        &self,
        args: &Value,
        ctx: &super::server::ConnectionContext,
    ) -> Result<Value> {
        let wid = self.resolve_worker_id(args, "review", ctx).await?;
        self.require_owner(&wid, ctx).await?;
        let scope = Self::get_review_diff_scope(args)?;
        let state = self.pool.get_worker_state(&wid).await;
        let entry = crate::pool::load_registry_entry_in(self.pool.scratch_root(), &wid);
        if state.is_none() && entry.is_none() {
            anyhow::bail!("Worker not found: {wid}");
        }
        let (summary, verified, state_branch) = completed_fields(state.as_ref());
        let branch = state_branch.unwrap_or_else(|| format!("worker-{wid}"));
        // The registry row is the only cross-process record of where the
        // worker's repository is and which branch it integrates with.
        let repo = entry
            .as_ref()
            .and_then(|entry| entry.repo_path.as_deref())
            .map(std::path::Path::new)
            .filter(|path| path.is_dir())
            .map(std::path::Path::to_path_buf);
        let base_branch = entry
            .as_ref()
            .and_then(|entry| entry.base_branch.clone())
            .or_else(|| {
                repo.as_deref()
                    .and_then(crate::pool::revision::detect_base_branch)
            });
        // A live worker's diff is the exact text it produced; a collected one
        // has only its branch left, so its change is measured from that.
        let live_diff = match &state {
            Some(crate::pool::WorkerState::Completed { diff, .. }) => Some(diff.clone()),
            _ => None,
        };
        let probe = repo
            .clone()
            .zip(base_branch.clone())
            .map(|(repo, base)| (repo, base, branch.clone()));
        let (stats, summaries, merge, raw_diff) = match tokio::task::spawn_blocking(move || {
            let Some((repo, base, branch)) = probe else {
                // No repository recorded: a live worker still carries its own
                // diff, so the change can be classified even without git.
                return match live_diff {
                    Some(diff) => {
                        let summaries = diff_file_summaries(&diff);
                        let stats = diff_file_stats(&diff);
                        (stats, summaries, None, diff)
                    }
                    None => (Vec::new(), Vec::new(), None, String::new()),
                };
            };
            let merge = merge_check(&repo, &base, &branch);
            let text = match &live_diff {
                Some(diff) => diff.clone(),
                None => branch_diff(&repo, &base, &branch),
            };
            let stats = match &live_diff {
                Some(diff) => diff_file_stats(diff),
                None => branch_file_stats(&repo, &base, &branch),
            };
            let summaries = diff_file_summaries(&text);
            (stats, summaries, merge, text)
        })
        .await
        {
            Ok(probed) => probed,
            Err(err) => {
                tracing::warn!("review probe for worker {wid} could not run: {err}");
                (Vec::new(), Vec::new(), None, String::new())
            }
        };
        // A worker that never verified is exactly the one whose verify output
        // the orchestrator needs; a verified one has nothing to show.
        let verify_tail = if verified == Some(true) {
            None
        } else {
            self.pool
                .get_worker_logs(&wid)
                .await
                .and_then(|logs| super::events::verify_tail_of(&logs.tail(VERIFY_TAIL_STEPS)))
        };
        // Test files are summarised rather than shown, so the diff stays code
        // by default; `all` shows everything and `none` withholds it.
        let shown = match scope {
            ReviewDiffScope::None => String::new(),
            ReviewDiffScope::All => raw_diff,
            ReviewDiffScope::Code => diff_of_kind(&raw_diff, PathKind::Code),
        };
        let diff = match scope {
            ReviewDiffScope::None => Value::Null,
            _ => Value::String(crate::agent::truncate_output(&shown)),
        };
        Ok(json!({
            "worker_id": wid,
            "owner": self.owner_of(&wid).await,
            "task": first_line(entry.as_ref().map(|entry| entry.task.as_str()).unwrap_or_default()),
            "state": state_name(state.as_ref(), entry.as_ref()),
            "verified": verified,
            "verify_tail": verify_tail,
            "approved": entry.as_ref().and_then(|entry| entry.approved.clone()),
            "diff_scope": scope.name(),
            "diff": diff,
            "diff_stat": diff_stat_value(&stats),
            "test_files": test_files_value(&summaries),
            "docs": docs_value(&summaries),
            "summary": summary,
            "revision": revision_of(state.as_ref(), entry.as_ref()),
            "branch": branch,
            "merge": merge,
            "next_command": next_command(&wid, &branch, merge.as_ref()),
        }))
    }

    /// Parse the optional `diff` argument of `review`: which part of the change
    /// to show. Defaults to the code diff, which is what a reviewer reads.
    fn get_review_diff_scope(args: &Value) -> Result<ReviewDiffScope> {
        match args.get("diff") {
            None => Ok(ReviewDiffScope::Code),
            Some(value) => match value.as_str() {
                Some("code") => Ok(ReviewDiffScope::Code),
                Some("all") => Ok(ReviewDiffScope::All),
                Some("none") => Ok(ReviewDiffScope::None),
                _ => anyhow::bail!("'diff' must be one of: code, all, none"),
            },
        }
    }

    /// `approve` action: the orchestrator signs off a completed worker.
    ///
    /// Owner-only like `collect`. The verdict is written to the registry row so
    /// it outlives the in-memory record `collect` evicts; steering the worker
    /// into a new revision drops it, because a changed branch needs a new
    /// review.
    async fn handle_approve(
        &self,
        args: &Value,
        ctx: &super::server::ConnectionContext,
    ) -> Result<Value> {
        let wid = self.resolve_worker_id(args, "approve", ctx).await?;
        self.require_owner(&wid, ctx).await?;
        let note = args
            .get("message")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|note| !note.is_empty())
            .map(str::to_owned);
        let approved = self.pool.approve(&wid, note).await?;
        Ok(json!({
            "worker_id": wid,
            "status": "approved",
            "approved": approved,
        }))
    }

    /// `unapprove` action: withdraw a completed worker's approval.
    async fn handle_unapprove(
        &self,
        args: &Value,
        ctx: &super::server::ConnectionContext,
    ) -> Result<Value> {
        let wid = self.resolve_worker_id(args, "unapprove", ctx).await?;
        self.require_owner(&wid, ctx).await?;
        self.pool.unapprove(&wid).await?;
        Ok(json!({
            "worker_id": wid,
            "status": "unapproved",
            "approved": Value::Null,
        }))
    }

    /// Parse the optional `files` argument: the paths whose diff the caller
    /// wants.
    ///
    /// A non-array, or an array holding anything but a non-empty string, is a
    /// hard error: silently dropping it would answer with a diff the caller
    /// never asked for.
    pub(super) fn get_diff_files(args: &Value, action: &str) -> Result<Vec<String>> {
        let Some(value) = args.get("files") else {
            return Ok(Vec::new());
        };
        let items = value.as_array().ok_or_else(|| {
            anyhow::anyhow!("'files' must be an array of paths for action '{action}'")
        })?;
        items
            .iter()
            .map(|item| {
                item.as_str()
                    .filter(|path| !path.is_empty())
                    .map(str::to_owned)
                    .ok_or_else(|| {
                        anyhow::anyhow!("'files' must be an array of paths for action '{action}'")
                    })
            })
            .collect()
    }

    /// `list` is scoped to the caller's own workers; `scope: "all"` widens it
    /// to every agent's and needs the admin override (H-3).
    async fn handle_list(
        &self,
        args: &Value,
        ctx: &super::server::ConnectionContext,
    ) -> Result<Value> {
        let workers = if Self::lists_every_agent(args, ctx)? {
            self.pool.list_workers().await
        } else {
            self.pool.list_workers_of(&ctx.agent()).await
        };
        Ok(json!({ "workers": workers }))
    }

    /// Whether the caller asked for every agent's workers, and may have them.
    ///
    /// Only the two documented values are accepted: a typo is a hard error
    /// rather than a silent fallback to the caller's own workers, which would
    /// look like a pool with nobody else's runs in it. `scope: "all"` is the
    /// admin override, so a non-admin caller gets a refusal rather than a
    /// truncated list it would read as "nobody else is running".
    fn lists_every_agent(args: &Value, ctx: &super::server::ConnectionContext) -> Result<bool> {
        match args.get("scope") {
            None => Ok(false),
            Some(scope) => match scope.as_str() {
                Some(scope) if super::schema::LIST_SCOPES.contains(&scope) => {
                    if scope == super::schema::LIST_SCOPE_ALL && !ctx.is_admin() {
                        anyhow::bail!("'scope' \"all\" requires the admin override (--admin)");
                    }
                    Ok(scope == super::schema::LIST_SCOPE_ALL)
                }
                _ => anyhow::bail!(
                    "'scope' must be one of: {}",
                    super::schema::LIST_SCOPES.join(", ")
                ),
            },
        }
    }

    async fn handle_kill(
        &self,
        args: &Value,
        ctx: &super::server::ConnectionContext,
    ) -> Result<Value> {
        let resolved = self.resolve_worker_id(args, "kill", ctx).await?;
        let wid = resolved.as_str();
        self.require_owner(wid, ctx).await?;
        let killed = self.pool.kill(wid).await;
        if killed {
            Ok(json!({ "worker_id": wid, "killed": true }))
        } else if let Some(entry) =
            crate::pool::load_registry_entry_in(self.pool.scratch_root(), wid)
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

    /// `watch` action: block until one of the caller's own workers produces an
    /// event, replaying the ones it missed while it was away.
    ///
    /// This is the MCP-only orchestrator's equivalent of `mini-swe-mcp watch`:
    /// dispatch and steer never block, so this is the only way to wait. It is
    /// the same long-poll the CLI runs -- the router already filters every
    /// worker to its owner -- and it answers `status: "no_event"` when
    /// `timeout_secs` expires first, so a caller under a host deadline can
    /// simply call `watch` again.
    async fn handle_watch(
        &self,
        args: &Value,
        ctx: &super::server::ConnectionContext,
    ) -> Result<Value> {
        let mut ids: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        let needles = args["worker_ids"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .chain(args.get("worker_id").and_then(Value::as_str));
        for needle in needles {
            ids.insert(self.pool.resolve_worker_id(needle, &ctx.agent()).await?);
        }
        // Without explicit ids the watch follows every worker the caller owns,
        // re-checked on each poll, so a later dispatch joins automatically. An
        // explicit id set stays fixed for the whole call.
        let explicit = !ids.is_empty();
        let group = args.get("group").and_then(Value::as_str);
        let timeout = Self::get_timeout(args, "watch")?;
        // A named worker must exist and be the caller's own: watching a
        // foreign id is refused with its owner, never with its task or state.
        for id in &ids {
            self.require_owner(id, ctx).await?;
        }
        // One watch per identity: reserve this call's slot up front and hold it
        // until the call returns (or is cancelled), so a second watch is
        // refused instead of silently competing for the same events.
        let _slot = self
            .hub_events
            .lock()
            .await
            .begin_watch(&ctx.agent(), ctx.id, ctx.pid)?;
        let started = tokio::time::Instant::now();
        let mut changes = self.pool.subscribe_changes();
        let mut initial = true;
        let mut watched_any = false;
        loop {
            let reply = self.watch_poll(ctx, &ids, group, initial).await?;
            let events = reply["events"].as_array().cloned().unwrap_or_default();
            if !events.is_empty() {
                // Acknowledge what was delivered: the router's per-agent
                // backlog is bounded, and a caller that never acks would
                // eventually see `dropped_events` instead of its own history.
                for event in &events {
                    self.watch_ack(ctx, event["sequence"].as_u64().unwrap_or(0))
                        .await?;
                }
                return Ok(json!({
                    "status": "event",
                    "events": events,
                    "watching": reply["watching"].as_array().cloned().unwrap_or_default(),
                }));
            }
            let watching: Vec<Value> = reply["watching"].as_array().cloned().unwrap_or_default();
            if !watching.is_empty() {
                watched_any = true;
            }
            if initial && explicit {
                ids = watching
                    .iter()
                    .filter_map(|id| id.as_str().map(str::to_string))
                    .collect();
                if ids.is_empty() {
                    return Ok(json!({
                        "status": "no_event",
                        "events": [],
                        "watching": [],
                        "message": "nothing to watch",
                    }));
                }
            }
            initial = false;
            if watching.is_empty() && (explicit || watched_any) {
                return Ok(json!({"status": "no_event", "events": [], "watching": []}));
            }
            let left = match timeout {
                Some(t) => match t.checked_sub(started.elapsed()) {
                    Some(left) => left,
                    None => {
                        // The deadline expired. A no-id watch that never saw a
                        // worker reports nothing to watch; otherwise the caller
                        // just got no event and can call again.
                        let mut payload = json!({
                            "status": "no_event",
                            "events": [],
                            "watching": watching,
                        });
                        if !watched_any {
                            payload["message"] = json!("nothing to watch");
                        }
                        return Ok(payload);
                    }
                },
                None => Duration::from_secs(1),
            };
            // The tick bounds how long a missed wake-up can stall the poll;
            // the pool's change channel is what makes it prompt.
            tokio::select! {
                _ = changes.changed() => {}
                _ = tokio::time::sleep(left.min(Duration::from_secs(1))) => {}
            }
        }
    }

    /// One `hub/watch` request against this server's own event router.
    async fn watch_poll(
        &self,
        ctx: &super::server::ConnectionContext,
        ids: &std::collections::BTreeSet<String>,
        group: Option<&str>,
        initial: bool,
    ) -> Result<Value> {
        let params = json!({
            "worker_ids": ids.iter().collect::<Vec<_>>(),
            "group": group,
            "initial": initial,
        });
        super::events::watch_request(&self.pool, &self.hub_events, ctx, params, false).await
    }

    /// Acknowledge one delivered event so it leaves the caller's backlog.
    async fn watch_ack(
        &self,
        ctx: &super::server::ConnectionContext,
        sequence: u64,
    ) -> Result<Value> {
        super::events::watch_request(
            &self.pool,
            &self.hub_events,
            ctx,
            json!({ "sequence": sequence }),
            true,
        )
        .await
    }

    /// `steer` action: queue guidance for the worker's next turn.
    ///
    /// Steering a finished worker starts a revision on its preserved branch
    /// (same id, full context, fresh turn budget). The reply is immediate:
    /// `watch` is the only way to wait for the revision's next event.
    async fn handle_steer(
        &self,
        args: &Value,
        _token: Option<&Value>,
        _tx: Option<&mpsc::Sender<String>>,
        ctx: &super::server::ConnectionContext,
    ) -> Result<Value> {
        let resolved = self.resolve_worker_id(args, "steer", ctx).await?;
        let wid = resolved.as_str();
        self.require_owner(wid, ctx).await?;
        let message = Self::required_string(args, "message", "steer")?.to_string();
        // An explicit `max_turns` on a steer is the revision's fresh budget;
        // it is validated but otherwise ignored for live workers (whose loop
        // keeps its own budget). A non-integer value is a hard error, like on
        // dispatch, so a typo cannot silently become the default budget.
        let revision_turns = match args.get("max_turns") {
            None => None,
            Some(v) => Some(
                v.as_u64()
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "'max_turns' must be a non-negative integer for action 'steer'"
                        )
                    })
                    .map(|v| v as usize)
                    .and_then(|v| {
                        if v == 0 {
                            Err(anyhow::anyhow!(
                                "'max_turns' must be at least 1 for action 'steer'"
                            ))
                        } else {
                            Ok(v.min(crate::manifest::MAX_TURNS_LIMIT))
                        }
                    })?,
            ),
        };
        let admission = self.admit_worker().await?;
        let outcome = self
            .pool
            .steer_with_budget(wid, message, revision_turns)
            .await?;
        drop(admission);
        // The reply names exactly what happened: a live worker was steered or
        // resumed, a stopped one continued -- as a revision of its saved
        // conversation, or cold when none survived.
        if let SteerOutcome::Continuing { revision, cold } = outcome {
            let budget = self.revision_await_budget(revision_turns);
            let (status, message) = if cold {
                (
                    "continuing",
                    format!(
                        "Continuing worker {wid} on branch worker-{wid} with a fresh conversation (no saved history) and a fresh budget of {budget} turns"
                    ),
                )
            } else {
                (
                    "revising",
                    format!(
                        "Revision {revision} started on branch worker-{wid} with a fresh budget of {budget} turns"
                    ),
                )
            };
            let mut payload = json!({
                "worker_id": wid,
                "status": status,
                "message": format!("{message}. Use watch for the next event."),
            });
            Self::with_watch_command(&mut payload, ctx);
            return Ok(payload);
        }
        if matches!(outcome, SteerOutcome::Resumed) {
            let mut payload = json!({
                "worker_id": wid,
                "status": "resumed",
                "message": "Worker resumed with your steering instruction. Use watch for the next event."
            });
            Self::with_watch_command(&mut payload, ctx);
            return Ok(payload);
        }
        let mut payload = json!({
            "worker_id": wid,
            "status": "steered",
            "message": "Steering instruction queued for next turn. Use watch for the next event."
        });
        Self::with_watch_command(&mut payload, ctx);
        Ok(payload)
    }

    /// Fresh turn budget a revision started by steering a finished worker runs
    /// on: the explicit `max_turns`, or the default revision budget.
    fn revision_await_budget(&self, explicit: Option<usize>) -> usize {
        explicit.unwrap_or(crate::pool::DEFAULT_REVISION_TURNS)
    }

    async fn handle_prune(
        &self,
        args: &Value,
        token: Option<&Value>,
        tx: Option<&mpsc::Sender<String>>,
        ctx: &super::server::ConnectionContext,
    ) -> Result<Value> {
        let repo_path = Self::get_repo_path(args, ctx);
        Self::emit_progress(
            tx,
            token,
            0,
            1,
            "Pruning stale worktrees and dead worker branches",
        )
        .await;
        // Both sweeps walk directories, shell out to git and salvage dead
        // worktrees, so they run off the runtime thread.
        let root = self.pool.scratch_root().clone();
        let pruned = tokio::task::spawn_blocking(move || {
            crate::worktree::prune_stale_worktrees_in(&repo_path, &root.base_dirs());
            crate::pool::prune_orphan_histories_in(&root, &repo_path);
        })
        .await;
        if pruned.is_err() {
            tracing::warn!("Worktree prune task could not run");
        }
        Self::emit_progress(tx, token, 1, 1, "Prune complete").await;
        Ok(json!({
            "status": "pruned",
            "message": "Stale worktrees and dead worker branches cleaned up"
        }))
    }

    /// `merge` action: land one finished worker's branch on its base branch.
    ///
    /// Owner-only, like every other per-worker verb. The whole sequence --
    /// trial merge, dirty check, gate, real merge, cleanup -- is one blocking
    /// unit in [`crate::pool::merge`], so it runs off the runtime thread and
    /// answers with a single payload the CLI renders as one line.
    async fn handle_merge(
        &self,
        args: &Value,
        ctx: &super::server::ConnectionContext,
    ) -> Result<Value> {
        let wid = Self::get_worker_id(args, "merge")?;
        self.require_owner(wid, ctx).await?;
        // A worker this process owns carries its verify verdict in memory; a
        // cross-process caller has only the on-disk row, which names none, and
        // therefore re-runs the gate.
        let verified = match self.pool.get_worker_state(wid).await {
            Some(crate::pool::WorkerState::Completed { verified, .. }) => verified,
            _ => None,
        };
        let keep_branch = args
            .get("keep_branch")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let root = self.pool.scratch_root().clone();
        let admission = self.pool.admission();
        let worker_id = wid.to_string();
        let report = tokio::task::spawn_blocking(move || {
            crate::pool::merge_worker_in(
                &root,
                &crate::pool::MergeRequest {
                    worker_id: &worker_id,
                    verified,
                    keep_branch,
                    admission: Some(admission),
                },
            )
        })
        .await
        .map_err(|e| anyhow::anyhow!("merge task for worker {wid} failed: {e}"))??;
        Ok(json!({
            "worker_id": report.worker_id,
            "branch": report.branch,
            "base_branch": report.base_branch,
            "repo_path": report.repo_path,
            "commit": report.commit,
            "gate": if report.gate_ran { "ran" } else { "skipped" },
            "gate_command": report.gate_command,
            "branch_deleted": report.branch_deleted,
            "cleaned": report.cleaned,
        }))
    }
}

/// Which part of a worker's change `review` returns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReviewDiffScope {
    /// Only the code diff; test files are summarised and docs listed.
    Code,
    /// The whole diff, test and doc churn included.
    All,
    /// No diff at all.
    None,
}

impl ReviewDiffScope {
    /// The wire name of the scope.
    fn name(self) -> &'static str {
        match self {
            Self::Code => "code",
            Self::All => "all",
            Self::None => "none",
        }
    }
}

/// How many step logs a review looks back through for a failed verify.
const VERIFY_TAIL_STEPS: usize = 8;

/// The three fields only a completed worker carries: its summary, whether the
/// gate verified it, and the branch it leaves behind.
fn completed_fields(
    state: Option<&crate::pool::WorkerState>,
) -> (Option<String>, Option<bool>, Option<String>) {
    match state {
        Some(crate::pool::WorkerState::Completed {
            summary,
            verified,
            branch,
            ..
        }) => (Some(summary.clone()), *verified, branch.clone()),
        _ => (None, None, None),
    }
}

/// The first line of a task that says something: an agent writes a heading and
/// a body, and only the heading belongs in a compact view.
fn first_line(text: &str) -> String {
    text.lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .unwrap_or_default()
        .to_string()
}

/// The lifecycle name of a worker, from its live state when it still has one
/// and from its registry row otherwise.
fn state_name(
    state: Option<&crate::pool::WorkerState>,
    entry: Option<&crate::pool::WorkerRegistryEntry>,
) -> &'static str {
    match state {
        Some(crate::pool::WorkerState::Running { .. }) => "Running",
        Some(crate::pool::WorkerState::Paused { .. }) => "Paused",
        Some(crate::pool::WorkerState::Completed { .. }) => "Completed",
        Some(crate::pool::WorkerState::Failed { .. }) => "Failed",
        None => entry.map_or("Unknown", |entry| entry.status.display_name()),
    }
}

/// The revision a worker reached: the live state carries it, and a collected
/// worker's registry row is the only other record of it.
fn revision_of(
    state: Option<&crate::pool::WorkerState>,
    entry: Option<&crate::pool::WorkerRegistryEntry>,
) -> usize {
    match state {
        Some(crate::pool::WorkerState::Completed { revision, .. })
        | Some(crate::pool::WorkerState::Failed { revision, .. }) => *revision,
        _ => entry.map_or(0, |entry| entry.revision),
    }
}

/// The command that acts on this review: merge a clean branch, or send the
/// conflicts back to the worker that owns them.
fn next_command(wid: &str, branch: &str, merge: Option<&MergeCheck>) -> String {
    let Some(merge) = merge else {
        // No answer was possible, so the merge itself still has to be checked.
        return format!("git merge-tree --write-tree <base-branch> {branch}");
    };
    if merge.clean == Some(true) {
        return format!("git merge {branch}");
    }
    if !merge.conflicts.is_empty() {
        return format!(
            "mini-swe-mcp steer {wid} \"resolve the merge conflicts with {}: {}\"",
            merge.base_branch,
            merge.conflicts.join(", ")
        );
    }
    format!("git merge {branch}")
}

/// Whether `branch` still merges cleanly into the tip of `base_branch`.
///
/// `git merge-tree --write-tree` (git >= 2.38) performs the merge in memory: it
/// writes the resulting tree object and answers through its exit status, so the
/// probe touches no worktree, no index and no lock — whichever way it answers,
/// nothing on disk changes. Exit 0 is a clean merge; exit 1 lists the conflicted
/// files after the tree oid; anything else means git refused (an unknown option
/// on an older git, a ref that does not exist).
fn merge_check(repo: &std::path::Path, base_branch: &str, branch: &str) -> Option<MergeCheck> {
    let mut check = MergeCheck {
        base_branch: base_branch.to_string(),
        clean: None,
        conflicts: Vec::new(),
        error: None,
    };
    let output = match crate::worktree::git(
        repo,
        "merge-tree",
        &["merge-tree", "--write-tree", base_branch, branch],
    ) {
        Ok(output) => output,
        Err(err) => {
            check.error = Some(err.to_string());
            return Some(check);
        }
    };
    match output.status.code() {
        Some(0) => check.clean = Some(true),
        Some(1) => {
            check.clean = Some(false);
            // The conflicted file list follows the tree oid, one
            // `<mode> <oid> <stage>\t<path>` line per stage of a path, and
            // stops at the blank line that introduces the informational
            // messages. A path therefore appears once per stage it conflicts
            // in, so the list is deduplicated in the order git reported it.
            for path in String::from_utf8_lossy(&output.stdout)
                .lines()
                .skip(1)
                .take_while(|line| !line.is_empty())
                .filter_map(|line| line.rsplit('\t').next().map(str::to_string))
            {
                if !check.conflicts.contains(&path) {
                    check.conflicts.push(path);
                }
            }
        }
        _ => {
            check.error = Some(format!(
                "git merge-tree --write-tree failed: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            ));
        }
    }
    Some(check)
}

/// The answer to "does this branch still merge into the base branch tip?".
#[derive(Debug, serde::Serialize)]
pub(super) struct MergeCheck {
    pub(super) base_branch: String,
    /// `None` when git could not answer at all.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) clean: Option<bool>,
    pub(super) conflicts: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) error: Option<String>,
}

/// The merge-base commit `branch` shares with `base_branch`, or `None` when
/// git cannot name one.
fn merge_base(repo: &std::path::Path, base_branch: &str, branch: &str) -> Option<String> {
    let output =
        crate::worktree::git(repo, "merge-base", &["merge-base", base_branch, branch]).ok()?;
    if !output.status.success() {
        return None;
    }
    let base = String::from_utf8_lossy(&output.stdout).trim().to_string();
    (!base.is_empty()).then_some(base)
}

/// Per-file diff stat of `branch` against its merge-base with `base_branch`.
///
/// The branch is what the orchestrator merges, so it is measured against the
/// base tip it will land on — the same range `git diff --shortstat` reports.
fn branch_file_stats(repo: &std::path::Path, base_branch: &str, branch: &str) -> Vec<DiffFileStat> {
    let Some(base) = merge_base(repo, base_branch, branch) else {
        return Vec::new();
    };
    let range = format!("{base}...{branch}");
    let Ok(output) = crate::worktree::git(repo, "diff --numstat", &["diff", "--numstat", &range])
    else {
        return Vec::new();
    };
    if !output.status.success() {
        return Vec::new();
    }
    parse_numstat(&String::from_utf8_lossy(&output.stdout))
}

/// The whole unified diff of `branch` against its merge-base with
/// `base_branch`, so a collected worker's code diff can still be shown.
fn branch_diff(repo: &std::path::Path, base_branch: &str, branch: &str) -> String {
    let Some(base) = merge_base(repo, base_branch, branch) else {
        return String::new();
    };
    let range = format!("{base}...{branch}");
    match crate::worktree::git(repo, "diff", &["diff", &range]) {
        Ok(output) if output.status.success() => {
            String::from_utf8_lossy(&output.stdout).into_owned()
        }
        _ => String::new(),
    }
}

/// Read `git diff --numstat` output: `<added>\t<deleted>\t<path>` per line,
/// with `-` for a binary file, which counts as neither added nor deleted.
fn parse_numstat(text: &str) -> Vec<DiffFileStat> {
    text.lines()
        .filter_map(|line| {
            let mut fields = line.split('\t');
            let insertions = numstat_count(fields.next()?)?;
            let deletions = numstat_count(fields.next()?)?;
            let path = fields.next()?;
            Some(DiffFileStat {
                path: normalize_diff_path(path),
                insertions,
                deletions,
            })
        })
        .collect()
}

/// One `--numstat` count: a number, or `0` for the `-` a binary file carries.
fn numstat_count(field: &str) -> Option<usize> {
    Some(if field == "-" {
        0
    } else {
        field.parse::<usize>().ok()?
    })
}

/// One file's share of a diff, as `git diff --numstat` reports it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct DiffFileStat {
    pub(super) path: String,
    pub(super) insertions: usize,
    pub(super) deletions: usize,
}

/// A path as git spells it in a diff header, without the `a/`/`b/` prefix or a
/// leading `./`, so a caller's `--file src/a.rs` matches what git printed.
fn normalize_diff_path(path: &str) -> String {
    let path = path
        .strip_prefix("b/")
        .or_else(|| path.strip_prefix("a/"))
        .unwrap_or(path)
        .trim_matches('"');
    path.strip_prefix("./").unwrap_or(path).to_string()
}

/// Whether a requested path names the file a diff section is about.
///
/// Exact after normalisation, or a whole-component suffix of it, so `--file
/// a.rs` still finds `src/a.rs`.
fn same_diff_path(requested: &str, actual: &str) -> bool {
    let requested = normalize_diff_path(requested);
    let actual = normalize_diff_path(actual);
    actual == requested
        || (!requested.is_empty()
            && actual.len() > requested.len()
            && actual.ends_with(&requested)
            && actual[..actual.len() - requested.len()].ends_with('/'))
}

/// The path a `--- `/`+++ ` header names, or `None` for `/dev/null`.
fn diff_header_path(header: &str) -> Option<String> {
    let path = header.trim();
    (path != "/dev/null").then(|| normalize_diff_path(path))
}

/// One file's section of a diff, while it is still being read.
struct DiffSection {
    /// The path as the "after" side spells it.
    path: String,
    /// The path as the "before" side spells it, for a file that was deleted.
    minus: String,
    body: String,
}

impl DiffSection {
    /// The path the section is about: the "after" side when the file still
    /// exists, the "before" side when it does not.
    fn path(&self) -> &str {
        if self.path.is_empty() {
            &self.minus
        } else {
            &self.path
        }
    }
}

/// Split a unified diff into one `(path, section)` pair per file.
///
/// The path comes from the `+++`/`---` headers, which precede every hunk, so a
/// removed line that happens to start with `--` can never be mistaken for one.
/// A section with neither header — a binary file, a mode-only change — falls
/// back to the paths its `diff --git` line names.
fn diff_sections(diff: &str) -> Vec<(String, String)> {
    let mut sections: Vec<(String, String)> = Vec::new();
    let mut current: Option<DiffSection> = None;
    for line in diff.lines() {
        if let Some(header) = line.strip_prefix("diff --git ") {
            if let Some(section) = current.take() {
                sections.push((section.path().to_string(), section.body));
            }
            current = Some(DiffSection {
                path: diff_git_path(header).unwrap_or_default(),
                minus: String::new(),
                body: format!("{line}\n"),
            });
            continue;
        }
        let Some(section) = current.as_mut() else {
            continue;
        };
        section.body.push_str(line);
        section.body.push('\n');
        if let Some(found) = line.strip_prefix("--- ").and_then(diff_header_path) {
            section.minus = found;
        } else if let Some(found) = line.strip_prefix("+++ ").and_then(diff_header_path) {
            section.path = found;
        }
    }
    if let Some(section) = current.take() {
        sections.push((section.path().to_string(), section.body));
    }
    sections
        .into_iter()
        .filter(|(path, _)| !path.is_empty())
        .collect()
}

/// The path a `diff --git a/<path> b/<path>` line names on its "before" side.
fn diff_git_path(header: &str) -> Option<String> {
    let split = header.rfind(" b/")?;
    let path = &header[..split];
    Some(normalize_diff_path(path.strip_prefix("a/").unwrap_or(path)))
}

/// Which part of a change a path belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PathKind {
    Code,
    Test,
    Doc,
}

/// Directory components that mark a file as a test wherever it lives.
const TEST_DIRS: &[&str] = &["tests", "test", "__tests__", "spec"];
/// Filename tails that mark a file as a test whatever its directory.
const TEST_NAME_TAILS: &[&str] = &["_test.", "_spec.", ".test.", ".spec."];
/// Directory components that mark a file as documentation.
const DOC_DIRS: &[&str] = &["docs"];
/// Filename tails that mark a file as documentation.
const DOC_EXTENSIONS: &[&str] = &[".md", ".rst", ".txt"];

/// Classify one diff path as code, test or doc.
///
/// One small table, applied in order: a test directory or a `*_test.*` style
/// name wins over a documentation extension, so `tests/README.md` is a test.
fn classify_path(path: &str) -> PathKind {
    let file = path.rsplit('/').next().unwrap_or(path);
    let has_dir = |dir: &str| path.split('/').any(|part| part == dir);
    if TEST_DIRS.iter().any(|dir| has_dir(dir))
        || TEST_NAME_TAILS.iter().any(|tail| file.contains(tail))
        || (file.starts_with("test_") && file.ends_with(".py"))
    {
        return PathKind::Test;
    }
    if DOC_DIRS.iter().any(|dir| has_dir(dir))
        || DOC_EXTENSIONS.iter().any(|ext| file.ends_with(ext))
    {
        return PathKind::Doc;
    }
    PathKind::Code
}

/// Declarations that each start one test case, language-agnostically.
///
/// `#[test]`/`#[tokio::test]` are the Rust forms, `fn test_`/`def test_` the
/// function forms, `it(`/`test(`/`describe(` the JS ones and `@Test`/
/// `func Test` the JVM and Swift/Go ones.
const TEST_CASE_PATTERNS: &[&str] = &[
    "#[test]",
    "#[tokio::test]",
    "fn test_",
    "def test_",
    "it(",
    "test(",
    "describe(",
    "@Test",
    "func Test",
];

/// How many test cases one changed line declares.
fn count_test_cases(line: &str) -> usize {
    TEST_CASE_PATTERNS
        .iter()
        .filter(|pattern| contains_at_word_start(line, pattern))
        .count()
}

/// Whether `line` contains `pattern` at a word start, so `it(` does not match
/// inside an identifier such as `unit(`.
fn contains_at_word_start(line: &str, pattern: &str) -> bool {
    let mut from = 0;
    while let Some(at) = line[from..].find(pattern) {
        let idx = from + at;
        let boundary = idx == 0
            || !line[..idx]
                .chars()
                .next_back()
                .is_some_and(|c| c.is_alphanumeric() || c == '_');
        if boundary {
            return true;
        }
        from = idx + pattern.len();
    }
    false
}

/// One file's share of a diff, including the test cases its changed lines
/// declare and how the path classifies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct DiffFileSummary {
    pub(super) path: String,
    pub(super) insertions: usize,
    pub(super) deletions: usize,
    pub(super) added_cases: usize,
    pub(super) removed_cases: usize,
    pub(super) kind: PathKind,
}

/// Per-file summaries of a unified diff.
///
/// Only hunk lines are counted, and a hunk starts at its `@@` header, so the
/// `---`/`+++` headers and an added line that itself starts with `+` are never
/// mistaken for a change.
fn diff_file_summaries(diff: &str) -> Vec<DiffFileSummary> {
    diff_sections(diff)
        .into_iter()
        .map(|(path, section)| {
            let kind = classify_path(&path);
            let mut insertions = 0;
            let mut deletions = 0;
            let mut added_cases = 0;
            let mut removed_cases = 0;
            let mut in_hunks = false;
            for line in section.lines() {
                if line.starts_with("@@") {
                    in_hunks = true;
                } else if in_hunks && line.starts_with('+') {
                    insertions += 1;
                    added_cases += count_test_cases(line.get(1..).unwrap_or_default());
                } else if in_hunks && line.starts_with('-') {
                    deletions += 1;
                    removed_cases += count_test_cases(line.get(1..).unwrap_or_default());
                }
            }
            DiffFileSummary {
                path,
                insertions,
                deletions,
                added_cases,
                removed_cases,
                kind,
            }
        })
        .collect()
}

/// Per-file `(path, insertions, deletions)` of a unified diff.
fn diff_file_stats(diff: &str) -> Vec<DiffFileStat> {
    diff_file_summaries(diff)
        .into_iter()
        .map(|summary| DiffFileStat {
            path: summary.path,
            insertions: summary.insertions,
            deletions: summary.deletions,
        })
        .collect()
}

/// The sections of `diff` whose path classifies as `kind`, rejoined.
fn diff_of_kind(diff: &str, kind: PathKind) -> String {
    diff_sections(diff)
        .into_iter()
        .filter(|(path, _)| classify_path(path) == kind)
        .map(|(_, section)| section)
        .collect()
}

/// The `test_files` payload: one entry per test file, naming the test cases
/// its added and removed lines declare.
fn test_files_value(summaries: &[DiffFileSummary]) -> Value {
    Value::Array(
        summaries
            .iter()
            .filter(|summary| summary.kind == PathKind::Test)
            .map(|summary| {
                json!({
                    "path": summary.path,
                    "added_cases": summary.added_cases,
                    "removed_cases": summary.removed_cases,
                })
            })
            .collect(),
    )
}

/// The `docs` payload: one entry per documentation file with its +/ counts.
fn docs_value(summaries: &[DiffFileSummary]) -> Value {
    Value::Array(
        summaries
            .iter()
            .filter(|summary| summary.kind == PathKind::Doc)
            .map(|summary| {
                json!({
                    "path": summary.path,
                    "insertions": summary.insertions,
                    "deletions": summary.deletions,
                })
            })
            .collect(),
    )
}

/// The sections of `diff` that touch any of `files`, rejoined.
fn diff_of_files(diff: &str, files: &[String]) -> String {
    diff_sections(diff)
        .into_iter()
        .filter(|(path, _)| files.iter().any(|file| same_diff_path(file, path)))
        .map(|(_, section)| section)
        .collect()
}

/// The `diff_stat` payload: the totals plus the per-file counts.
fn diff_stat_value(stats: &[DiffFileStat]) -> Value {
    let insertions: usize = stats.iter().map(|stat| stat.insertions).sum();
    let deletions: usize = stats.iter().map(|stat| stat.deletions).sum();
    json!({
        "files": stats.len(),
        "insertions": insertions,
        "deletions": deletions,
        "per_file": stats
            .iter()
            .map(|stat| json!({
                "path": stat.path,
                "insertions": stat.insertions,
                "deletions": stat.deletions,
            }))
            .collect::<Vec<_>>(),
    })
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
        map.insert(
            "logs".to_string(),
            serde_json::to_value(&self.logs).unwrap_or_default(),
        );
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
    use crate::agent::AgentStepLog;

    /// An omitted `timeout_secs` means "wait indefinitely"; a non-integer one
    /// is a hard error, because a dropped deadline is the unbounded hang the
    /// argument exists to prevent.
    #[test]
    fn timeout_absent_is_unbounded_and_a_non_integer_is_rejected() {
        assert_eq!(
            McpServer::get_timeout(&json!({ "action": "wait" }), "wait")
                .expect("an absent deadline is not an error"),
            None
        );
        assert_eq!(
            McpServer::get_timeout(&json!({ "timeout_secs": 90 }), "wait")
                .expect("a whole number of seconds is accepted"),
            Some(std::time::Duration::from_secs(90))
        );
        let err = McpServer::get_timeout(&json!({ "timeout_secs": "90" }), "wait")
            .expect_err("a string deadline must not be accepted");
        assert!(
            err.to_string()
                .contains("'timeout_secs' must be a non-negative integer"),
            "{err}"
        );
    }

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
        assert!(
            err.to_string().contains("not a valid 'network' policy"),
            "{err}"
        );

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
            !McpServer::resolve_network_policy(&json!({}), "dispatch", &manifest, "combo:ninja",)
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
            McpServer::resolve_network_policy(&json!({}), "dispatch", &manifest, "vendor:sealed",)
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

    /// A diff split into sections keeps every file, and the counts come from the
    /// hunks alone: an added line that itself starts with `+` and a removed line
    /// that starts with `--` are content, not headers.
    #[test]
    fn diff_sections_count_hunk_lines_only() {
        let diff = concat!(
            "diff --git a/one.rs b/one.rs\n",
            "index 111..222 100644\n",
            "--- a/one.rs\n",
            "+++ b/one.rs\n",
            "@@ -1,3 +1,4 @@\n",
            " context\n",
            "-removed\n",
            "++added line that starts with a plus\n",
            "--removed line that starts with two dashes\n",
            "diff --git a/two.rs b/two.rs\n",
            "new file mode 100644\n",
            "index 000..333\n",
            "--- /dev/null\n",
            "+++ b/two.rs\n",
            "@@ -0,0 +1,2 @@\n",
            "+first\n",
            "+second\n",
        );
        let stats = diff_file_stats(diff);
        assert_eq!(
            stats,
            vec![
                DiffFileStat {
                    path: "one.rs".to_string(),
                    insertions: 1,
                    deletions: 2,
                },
                DiffFileStat {
                    path: "two.rs".to_string(),
                    insertions: 2,
                    deletions: 0,
                },
            ]
        );
        let stat = diff_stat_value(&stats);
        assert_eq!(stat["files"], 2);
        assert_eq!(stat["insertions"], 3);
        assert_eq!(stat["deletions"], 2);
        assert_eq!(stat["per_file"][1]["path"], "two.rs");
    }

    /// A binary file carries no line counts, but it is still a file that changed.
    #[test]
    fn a_binary_file_counts_as_a_file_with_no_lines() {
        let diff = concat!(
            "diff --git a/logo.png b/logo.png\n",
            "index 111..222 100644\n",
            "Binary files a/logo.png and b/logo.png differ\n",
        );
        assert_eq!(
            diff_file_stats(diff),
            vec![DiffFileStat {
                path: "logo.png".to_string(),
                insertions: 0,
                deletions: 0,
            }]
        );
    }

    /// `files` selects sections by path, and a bare name still finds the file
    /// inside a directory.
    #[test]
    fn diff_of_files_matches_a_path_or_a_component_suffix() {
        let diff = concat!(
            "diff --git a/src/a.rs b/src/a.rs\n",
            "--- a/src/a.rs\n",
            "+++ b/src/a.rs\n",
            "@@ -1 +1 @@\n",
            "-old\n",
            "+new\n",
            "diff --git a/b.rs b/b.rs\n",
            "--- a/b.rs\n",
            "+++ b/b.rs\n",
            "@@ -1 +1 @@\n",
            "-old\n",
            "+new\n",
        );
        // Six lines: the `diff --git` header, the two path headers, the hunk
        // header and the two changed lines.
        assert_eq!(
            diff_of_files(diff, &["src/a.rs".to_string()])
                .lines()
                .count(),
            6
        );
        assert_eq!(
            diff_of_files(diff, &["a.rs".to_string()]).lines().count(),
            6
        );
        assert_eq!(
            diff_of_files(diff, &["b.rs".to_string()]).lines().count(),
            6
        );
        assert!(diff_of_files(diff, &["nope.rs".to_string()]).is_empty());
    }

    /// `git diff --numstat` is read as the same per-file stat the in-memory diff
    /// produces, and a binary file (`-`) counts as neither.
    #[test]
    fn numstat_is_read_as_the_same_per_file_stat() {
        assert_eq!(
            parse_numstat("3\t1\tsrc/a.rs\n-\t-\tlogo.png\n"),
            vec![
                DiffFileStat {
                    path: "src/a.rs".to_string(),
                    insertions: 3,
                    deletions: 1,
                },
                DiffFileStat {
                    path: "logo.png".to_string(),
                    insertions: 0,
                    deletions: 0,
                },
            ]
        );
    }

    /// The next command acts on the merge answer: merge a clean branch, send the
    /// conflicts back to the worker that owns them.
    #[test]
    fn next_command_follows_the_merge_answer() {
        let clean = MergeCheck {
            base_branch: "master".to_string(),
            clean: Some(true),
            conflicts: Vec::new(),
            error: None,
        };
        assert_eq!(
            next_command("w1", "worker-w1", Some(&clean)),
            "git merge worker-w1"
        );

        let mut conflicting = clean;
        conflicting.clean = Some(false);
        conflicting.conflicts = vec!["a.rs".to_string(), "b.rs".to_string()];
        assert_eq!(
            next_command("w1", "worker-w1", Some(&conflicting)),
            "mini-swe-mcp steer w1 \"resolve the merge conflicts with master: a.rs, b.rs\""
        );

        let unknown = MergeCheck {
            base_branch: "master".to_string(),
            clean: None,
            conflicts: Vec::new(),
            error: Some("git merge-tree failed".to_string()),
        };
        assert_eq!(
            next_command("w1", "worker-w1", Some(&unknown)),
            "git merge worker-w1"
        );
        assert_eq!(
            next_command("w1", "worker-w1", None),
            "git merge-tree --write-tree <base-branch> worker-w1"
        );
    }

    /// A review carries the first line of the task and the tail of the verify
    /// that failed, never the whole log.
    #[test]
    fn the_review_view_is_bounded() {
        assert_eq!(first_line("Fix the parser\nand its docs"), "Fix the parser");
        assert_eq!(first_line("   \n  padded  "), "padded");
        assert_eq!(first_line("  \n\n"), "");

        let build = AgentStepLog {
            step: 1,
            command: "cargo build".to_string(),
            output: "warning: unused".to_string(),
            exit_code: Some(0),
        };
        let verify = AgentStepLog {
            step: 2,
            command: "[verify] cargo test".to_string(),
            output: "test a ... FAILED\nassertion failed".to_string(),
            exit_code: Some(101),
        };
        let logs = vec![&build, &verify];
        assert_eq!(
            crate::mcp::events::verify_tail_of(&logs).expect("a failed verify must be shown"),
            "test a ... FAILED\nassertion failed"
        );
        assert_eq!(crate::mcp::events::verify_tail_of(&logs[..1]), None);
    }
    /// The path classifier is one table: a test directory or a test-style name
    /// beats a documentation extension, so `tests/README.md` is a test.
    #[test]
    fn classify_path_sorts_code_tests_and_docs() {
        for (path, kind) in [
            ("src/parser.rs", PathKind::Code),
            ("src/main.py", PathKind::Code),
            ("tests/integration.rs", PathKind::Test),
            ("test/cli_test.go", PathKind::Test),
            ("app/__tests__/x.js", PathKind::Test),
            ("spec/models_spec.rb", PathKind::Test),
            ("src/serde_test.rs", PathKind::Test),
            ("scripts/test_smoke.py", PathKind::Test),
            ("web/widget.spec.ts", PathKind::Test),
            ("src/thing.test.ts", PathKind::Test),
            ("README.md", PathKind::Doc),
            ("docs/design.rst", PathKind::Doc),
            ("notes.txt", PathKind::Doc),
            ("tests/README.md", PathKind::Test),
        ] {
            assert_eq!(classify_path(path), kind, "wrong kind for {path}");
        }
    }

    /// The counter is language-agnostic: every documented declaration earns one
    /// case, and an identifier that merely contains a pattern earns none.
    #[test]
    fn count_test_cases_reads_every_declaration() {
        for line in [
            "#[test]",
            "    #[tokio::test]",
            "fn test_parses() {",
            "def test_parses(self):",
            "it(`adds`)",
            "test('adds', () => {})",
            "describe('parser', () => {",
            "    @Test",
            "func TestParse(t *testing.T) {",
        ] {
            assert_eq!(count_test_cases(line), 1, "missed a case in {line:?}");
        }
        for line in [
            "// a comment about tests",
            "fn parses() {",
            "let unit = 1;",
            "let submit = 2;",
        ] {
            assert_eq!(count_test_cases(line), 0, "false positive in {line:?}");
        }
    }

    /// Test-case churn is measured from the changed hunk lines only, and each
    /// summary keeps the path's classification.
    #[test]
    fn diff_summaries_count_cases_and_classify() {
        let diff = concat!(
            "diff --git a/src/a.rs b/src/a.rs\n",
            "--- a/src/a.rs\n",
            "+++ b/src/a.rs\n",
            "@@ -1 +1 @@\n",
            "-old\n",
            "+new\n",
            "diff --git a/tests/a_test.rs b/tests/a_test.rs\n",
            "--- a/tests/a_test.rs\n",
            "+++ b/tests/a_test.rs\n",
            "@@ -1,2 +1,3 @@\n",
            "-#[test]\n",
            "-fn test_old() {}\n",
            "+#[test]\n",
            "+fn test_new() {}\n",
            "+#[test]\n",
        );
        let summaries = diff_file_summaries(diff);
        assert_eq!(summaries.len(), 2);
        assert_eq!(summaries[0].kind, PathKind::Code);
        assert_eq!(summaries[0].added_cases, 0);
        let test = &summaries[1];
        assert_eq!(test.kind, PathKind::Test);
        assert_eq!(test.added_cases, 3);
        assert_eq!(test.removed_cases, 2);
        assert_eq!(diff_of_kind(diff, PathKind::Code).lines().count(), 6);
    }

    /// `review --diff` accepts exactly the three documented scopes and defaults
    /// to the code diff.
    #[test]
    fn review_diff_scope_defaults_to_code_and_rejects_typos() {
        assert_eq!(
            McpServer::get_review_diff_scope(&json!({})).expect("absent is the default"),
            ReviewDiffScope::Code
        );
        assert_eq!(
            McpServer::get_review_diff_scope(&json!({ "diff": "all" })).unwrap(),
            ReviewDiffScope::All
        );
        assert_eq!(
            McpServer::get_review_diff_scope(&json!({ "diff": "none" })).unwrap(),
            ReviewDiffScope::None
        );
        assert!(McpServer::get_review_diff_scope(&json!({ "diff": "everything" })).is_err());
    }
}
