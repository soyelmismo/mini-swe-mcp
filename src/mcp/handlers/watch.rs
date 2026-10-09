use super::*;
use crate::mcp::events::watch_key;

impl McpServer {
    /// The shell command that waits on *this* caller's workers.
    ///
    /// A shell cannot know its session — the agent's `bash` tool sees none of
    /// the session variables — so the token is what binds the command to the
    /// caller: the hub resolves it back to exactly this identity, never to
    /// `admin` and never to another agent's. `None` for a caller with no token
    /// store, which is the in-process stdio server: it has no hub, so its
    /// `watch` runs in the same process.
    pub(super) fn watch_command(ctx: &crate::mcp::server::ConnectionContext) -> Option<String> {
        let token = ctx.token_store()?.token_for(&ctx.agent())?;
        Some(format!("MINI_SWE_WATCH_TOKEN={token} mini-swe-mcp watch"))
    }

    /// Add `watch_command` to a dispatch or steer payload, when this caller has
    /// a token store to mint from and no watch of its own already running.
    ///
    /// One watch per identity is the hub's rule, so a caller that already has
    /// one does not need the command again: its watch will deliver the next
    /// event. Repeating it would only make every answer carry tokens the
    /// caller cannot spend.
    pub(super) async fn with_watch_command(
        &self,
        payload: &mut Value,
        ctx: &crate::mcp::server::ConnectionContext,
    ) {
        if self.hub_events.lock().await.has_watch(&watch_key(ctx)) {
            return;
        }
        if let Some(command) = Self::watch_command(ctx) {
            payload["watch_command"] = json!(command);
        }
    }

    /// `watch` action: name the shell command that waits, and answer at once.
    ///
    /// A tool call is not a place to wait. A host bounds every call with a
    /// deadline of its own and aborts the one that outlives it, and the abort
    /// that cuts a blocking watch off is not a place an event can be
    /// delivered: the caller is severed before it learns the worker needs it,
    /// and the hub has already handed the event to the connection that is
    /// being torn down. The shell watch has no such deadline — it is a
    /// background task that blocks until the next event, prints it and exits,
    /// and is re-armed after each one — so that is where the wait belongs.
    ///
    /// The call's own selection is translated into the equivalent shell
    /// arguments, so the answer names a command that waits for exactly what
    /// was asked for: the same ids or groups, `--all` for whole rounds, and
    /// the caller's `timeout_secs` as `--timeout`. The hub's watch-token
    /// prefix rides along when the caller has a token to spend, because the
    /// shell cannot know its session and a watch started as a different
    /// identity would follow someone else's workers.
    pub(super) async fn handle_watch(
        &self,
        args: &Value,
        ctx: &crate::mcp::server::ConnectionContext,
    ) -> Result<Value> {
        let command = Self::shell_watch_command(args, ctx);
        Ok(json!({
            "status": "use_shell",
            "events": [],
            "watching": [],
            "watch_command": command,
            "message": format!(
                "watch does not block over MCP: a tool call is cut off by the client's own \
                 timeout, and the abort that ends it cannot deliver the event it waited for. \
                 Run this in your shell instead: {command}. It blocks until the next event, \
                 prints it and exits; run it again after each event."
            ),
        }))
    }

    /// The shell watch command that waits for the selection `args` asked for.
    fn shell_watch_command(args: &Value, ctx: &crate::mcp::server::ConnectionContext) -> String {
        // Every argument the action accepts has a shell spelling, so the
        // translated command is the same watch the caller meant to run.
        let mut flags = String::new();
        let needles = args["worker_ids"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .chain(args.get("worker_id").and_then(Value::as_str));
        for needle in needles {
            flags.push(' ');
            flags.push_str(&crate::agent::exec::shell_word(needle));
        }
        for group in crate::mcp::events::watch_groups(args) {
            flags.push_str(" --group ");
            flags.push_str(&crate::agent::exec::shell_word(&group));
        }
        if args.get("all").and_then(Value::as_bool).unwrap_or(false) {
            flags.push_str(" --all");
        }
        // A deadline the caller spelled out is honoured, not dropped: it
        // bounds the shell watch the same way it would have bounded the call.
        if let Ok(Some(timeout)) = Self::get_timeout(args, "watch") {
            flags.push_str(&format!(" --timeout {}", timeout.as_secs()));
        }
        match Self::watch_command(ctx) {
            Some(tokenized) => format!("{tokenized}{flags}"),
            None => format!("mini-swe-mcp watch{flags}"),
        }
    }

    /// One `hub/watch` poll against this server's own event router.
    ///
    /// The in-process shape of the wire the shell watch speaks: same selection,
    /// same ownership filtering, same acks bookkeeping, and it answers at once
    /// with whatever the router holds for this caller. An embedder (and a test
    /// whose router it restarts in place) polls through here; a caller over the
    /// MCP protocol is handed the shell command instead, because a tool call is
    /// bounded by the client's own deadline.
    pub async fn watch_poll(
        &self,
        ctx: &crate::mcp::server::ConnectionContext,
        ids: &std::collections::BTreeSet<String>,
        groups: &std::collections::BTreeSet<String>,
        initial: bool,
        all: bool,
    ) -> Result<Value> {
        let params = json!({
            "worker_ids": ids.iter().collect::<Vec<_>>(),
            "group": groups,
            "initial": initial,
            "all": all,
        });
        crate::mcp::events::watch_request(&self.pool, &self.hub_events, ctx, params, false).await
    }
}

pub(in crate::mcp) const WORKER_IDS_DESCRIPTION: &str =
    "Worker IDs to watch as 'worker_id' prefixes; omitted: your own.";

pub(in crate::mcp) const GROUP_DESCRIPTION: &str = "Group of workers (watch, merge --approved).";

pub(in crate::mcp) const TIMEOUT_SECS_DESCRIPTION: &str =
    "Watch deadline secs; the MCP action answers with the shell command.";

pub(in crate::mcp) const ALL_DESCRIPTION: &str = "The whole round as one event.";
