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
use crate::pool::round::first_line;
use crate::pool::{SteerOutcome, UNATTRIBUTED_OWNER, WorkerOwner, emit_view, normalize_diff_path};

/// Owner label used when neither the pool nor the registry has a row.
const UNKNOWN_OWNER: &str = "unknown";

mod collect;
pub(super) mod consolidate;
pub(super) mod dispatch;
mod help;
mod kill;
mod logs;
mod manifest;
mod merge;
mod prune_reap;
pub(super) mod review;
pub(super) mod status_list;
pub(super) mod steer;
pub(super) mod watch;

use logs::LogView;
use review::*;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod watch_command_tests;

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
            "help" => self.handle_help(args),
            "consolidate" => self.handle_consolidate(args, token, tx, ctx).await,
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
        "consolidate",
        "consolidate_verify",
    ];

    /// How many not-ready workers a `consolidate` refusal names before it
    /// truncates, so the message stays a line rather than a listing.
    const MAX_REFUSED_WORKERS: usize = 8;

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
}
