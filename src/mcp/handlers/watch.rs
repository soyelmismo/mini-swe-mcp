use super::*;

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
        if self.hub_events.lock().await.has_watch(&ctx.agent()) {
            return;
        }
        if let Some(command) = Self::watch_command(ctx) {
            payload["watch_command"] = json!(command);
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
    pub(super) async fn handle_watch(
        &self,
        args: &Value,
        ctx: &crate::mcp::server::ConnectionContext,
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
        let all = args.get("all").and_then(Value::as_bool).unwrap_or(false);
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
            let reply = self.watch_poll(ctx, &ids, group, initial, all).await?;
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
            // A `--all` round keeps every selected id until it reports, so its
            // terminal workers are never pruned away mid-round.
            if initial && explicit && !all {
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
    pub(super) async fn watch_poll(
        &self,
        ctx: &crate::mcp::server::ConnectionContext,
        ids: &std::collections::BTreeSet<String>,
        group: Option<&str>,
        initial: bool,
        all: bool,
    ) -> Result<Value> {
        let params = json!({
            "worker_ids": ids.iter().collect::<Vec<_>>(),
            "group": group,
            "initial": initial,
            "all": all,
        });
        crate::mcp::events::watch_request(&self.pool, &self.hub_events, ctx, params, false).await
    }

    /// Acknowledge one delivered event so it leaves the caller's backlog.
    pub(super) async fn watch_ack(
        &self,
        ctx: &crate::mcp::server::ConnectionContext,
        sequence: u64,
    ) -> Result<Value> {
        crate::mcp::events::watch_request(
            &self.pool,
            &self.hub_events,
            ctx,
            json!({ "sequence": sequence }),
            true,
        )
        .await
    }
}

pub(in crate::mcp) const WORKER_IDS_DESCRIPTION: &str =
    "Worker IDs to watch (same prefixes as 'worker_id'). Omitted watches your own workers.";

pub(in crate::mcp) const GROUP_DESCRIPTION: &str =
    "Only workers of this group. For 'watch' and 'merge --approved'.";

pub(in crate::mcp) const TIMEOUT_SECS_DESCRIPTION: &str =
    "Deadline in seconds for the blocking 'watch'; on expiry {status:'no_event'}.";

pub(in crate::mcp) const ALL_DESCRIPTION: &str =
    "Watch a whole round: with 'group' (or explicit ids) one event when every selected worker stopped, or early when one needs input or fails.";
