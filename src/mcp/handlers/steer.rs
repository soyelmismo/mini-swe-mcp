use super::*;

impl McpServer {
    /// `steer` action: queue guidance for the worker's next turn.
    ///
    /// Steering a finished worker starts a revision on its preserved branch
    /// (same id, full context, fresh turn budget). The reply is immediate:
    /// `watch` is the only way to wait for the revision's next event.
    pub(super) async fn handle_steer(
        &self,
        args: &Value,
        _token: Option<&Value>,
        _tx: Option<&mpsc::Sender<String>>,
        ctx: &crate::mcp::server::ConnectionContext,
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
            self.with_watch_command(&mut payload, ctx).await;
            return Ok(payload);
        }
        if matches!(outcome, SteerOutcome::Resumed) {
            let mut payload = json!({
                "worker_id": wid,
                "status": "resumed",
                "message": "Worker resumed with your steering instruction. Use watch for the next event."
            });
            self.with_watch_command(&mut payload, ctx).await;
            return Ok(payload);
        }
        let mut payload = json!({
            "worker_id": wid,
            "status": "steered",
            "message": "Steering instruction queued for next turn. Use watch for the next event."
        });
        self.with_watch_command(&mut payload, ctx).await;
        Ok(payload)
    }

    /// Fresh turn budget a revision started by steering a finished worker runs
    /// on: the explicit `max_turns`, or the default revision budget.
    pub(super) fn revision_await_budget(&self, explicit: Option<usize>) -> usize {
        explicit.unwrap_or(crate::pool::DEFAULT_REVISION_TURNS)
    }
}

pub(in crate::mcp) const MESSAGE_DESCRIPTION: &str = "Correction or follow-up for 'steer': continues on the worker's own branch with full context ('max_turns' sets the budget) or a stopped worker. Never dispatch a replacement.";

pub(in crate::mcp) const MAX_TURNS_DESCRIPTION: &str = "Max bash turns (overrides the default); on 'steer', the budget when continuing a stopped worker.";
