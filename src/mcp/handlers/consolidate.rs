use super::*;

impl McpServer {
    /// `consolidate` action: dispatch the round's consolidator.
    ///
    /// The glue that makes a consolidated round hard to run wrong: the caller
    /// names a group, and the hub computes the round manifest (which of the
    /// caller's workers in that group finished with an unmerged branch, what
    /// each touched, and which files more than one of them touched), refuses
    /// when there is nothing to integrate, and embeds the manifest plus
    /// [`crate::agent::CONSOLIDATOR_INSTRUCTIONS`] in the consolidator's task.
    ///
    /// Defaults differ from a plain dispatch on purpose: the consolidator runs
    /// on the manifest's strongest tier when one is marked, and its gate is the
    /// project's *full* gate (the explicit `verify`, else the auto-detected
    /// one), because it is the only worker that runs the whole suite.
    ///
    /// The dispatch itself is delegated to [`Self::dispatch_one`], so the
    /// consolidator goes through exactly the validation, admission and launch
    /// path every other worker does.
    pub(in crate::mcp) async fn handle_consolidate(
        &self,
        args: &Value,
        token: Option<&Value>,
        tx: Option<&mpsc::Sender<String>>,
        ctx: &crate::mcp::server::ConnectionContext,
    ) -> Result<Value> {
        let group = Self::required_string(args, "group", "consolidate")?
            .trim()
            .to_string();
        if group.is_empty() {
            anyhow::bail!("'group' must not be empty for action 'consolidate'");
        }
        match args.get("set") {
            None | Some(Value::Bool(false)) => {}
            Some(Value::Bool(true)) => return self.amend_round(args, &group, ctx).await,
            Some(_) => anyhow::bail!("'set' must be a boolean for action 'consolidate'"),
        }
        let agent = ctx.agent();
        let repo_path = Self::get_repo_path(args, ctx);
        let manifest = self.pool.round_manifest(&agent, &group, &repo_path).await;
        if !manifest.has_ready() {
            // Name what the group does hold: the usual cause is a worker that
            // is still running, and the orchestrator needs to know which.
            let waiting = if manifest.not_ready.is_empty() {
                "every branch in it is already merged".to_string()
            } else {
                let listed: Vec<String> = manifest
                    .not_ready
                    .iter()
                    .take(Self::MAX_REFUSED_WORKERS)
                    .map(|worker| format!("{} ({})", worker.id, worker.state))
                    .collect();
                format!(
                    "{} not ready: {}{}",
                    manifest.not_ready.len(),
                    listed.join(", "),
                    if manifest.not_ready.len() > Self::MAX_REFUSED_WORKERS {
                        ", ..."
                    } else {
                        ""
                    }
                )
            };
            anyhow::bail!(
                "group '{group}' has no completed, unmerged worker of yours to consolidate: {waiting}"
            );
        }

        // The strongest tier when the manifest marks one, else the dispatch
        // default: integrating a round is the deepest job in the pool.
        let requested_model = args
            .get("model")
            .and_then(|v| v.as_str())
            .or_else(|| self.manifest.strongest_alias())
            .unwrap_or(&self.default_model);

        // The explicit gate wins; an absent one auto-detects, so a consolidator
        // never runs a cheaper subset than the project's own gate.
        let verify = match args.get("verify").and_then(|v| v.as_str()) {
            // An explicit empty string disables the gate, exactly as on dispatch.
            Some("") => None,
            Some(cmd) => Some(cmd.to_string()),
            None => crate::pool::detect_verify_command(&repo_path),
        };

        let mut dispatch = Map::new();
        dispatch.insert("action".into(), Value::String("dispatch".into()));
        dispatch.insert("role".into(), Value::String("consolidate".into()));
        dispatch.insert("group".into(), Value::String(group.clone()));
        dispatch.insert(
            "task".into(),
            Value::String(manifest.task_text(verify.as_deref())),
        );
        dispatch.insert("model".into(), Value::String(requested_model.to_string()));
        dispatch.insert("repo_path".into(), json!(repo_path));
        // Preserve an explicitly disabled gate rather than auto-detecting again.
        dispatch.insert("verify".into(), Value::String(verify.unwrap_or_default()));
        if let Some(turns) = args.get("max_turns").and_then(|v| v.as_u64()) {
            dispatch.insert("max_turns".into(), Value::Number(turns.into()));
        }
        let mut payload = self
            .dispatch_one(&Value::Object(dispatch), token, tx, ctx)
            .await?;
        if let Some(object) = payload.as_object_mut() {
            object.insert("group".into(), Value::String(group));
            object.insert("round".into(), Value::String(manifest.render()));
        }
        Ok(payload)
    }

    /// `consolidate` with `set: true`: amend the *pending* round's
    /// auto-consolidation settings instead of dispatching a consolidator.
    ///
    /// A round's gate is fixed at dispatch and spent much later by a
    /// consolidator nobody is watching, which is why a wrong one had to be
    /// discovered by hand: the daemon keeps the rounds in memory, so editing
    /// `<hub_dir>/auto-consolidate.json` was overwritten by the next write.
    /// This is the path that goes through the daemon and lands on disk.
    ///
    /// The scope is deliberately narrow: one pending round of the *caller*, by
    /// `(owner, group)`. Another agent's round is not addressable at all, and a
    /// consumed round is refused rather than edited, because its consolidator
    /// is already running with the settings it was given.
    ///
    /// An absent `model` or `verify` leaves that half alone; an empty
    /// `verify` clears the gate, which is the same spelling an empty gate has
    /// on a dispatch. A new gate is parse-checked here exactly as at dispatch,
    /// so the amend cannot install the unrunnable gate this verb exists to fix.
    async fn amend_round(
        &self,
        args: &Value,
        group: &str,
        ctx: &crate::mcp::server::ConnectionContext,
    ) -> Result<Value> {
        let store = self
            .auto_store()
            .ok_or_else(|| anyhow::anyhow!("Amending an automatic round requires the hub daemon"))?;
        // `None` leaves a half alone; `Some(None)` clears it. A non-string is
        // refused instead of dropped, so an amend cannot silently keep the
        // setting the caller meant to replace.
        let model: Option<Option<&str>> = args
            .get("model")
            .map(|value| {
                value
                    .as_str()
                    .map(Some)
                    .ok_or_else(|| {
                        anyhow::anyhow!("'model' must be a string for action 'consolidate'")
                    })
            })
            .transpose()?;
        let verify: Option<Option<&str>> = args
            .get("verify")
            .map(|value| {
                value
                    .as_str()
                    .map(Some)
                    .ok_or_else(|| {
                        anyhow::anyhow!("'verify' must be a string for action 'consolidate'")
                    })
            })
            .transpose()?;
        anyhow::ensure!(
            model.is_some() || verify.is_some(),
            "'set' amends a round's settings: pass 'model' and/or 'verify' to change, \
             or dispatch the consolidator without it"
        );
        if let Some(Some(gate)) = verify {
            crate::pool::validate_verify_command(gate, "verify")?;
        }
        let round = store.amend(&ctx.agent(), group, model, verify)?;
        Ok(json!({
            // `amended` distinguishes this answer from a dispatch, which shares
            // the action; the formatter keys its view on it.
            "amended": true,
            "group": round.group,
            "owner": round.owner,
            "model": round.model,
            "verify": round.verify,
            "generation": round.generation,
            "consumed": round.consumed,
            "message": format!(
                "Automatic consolidation round for group {} updated; the hub will run its \
                 consolidator with these settings.",
                round.group
            ),
        }))
    }
}

pub(in crate::mcp) const ROLE_DESCRIPTION: &str =
    "'consolidate': integrate this group's completed workers (requires 'group')";

pub(in crate::mcp) const SET_DESCRIPTION: &str =
    "'consolidate --set': amend the pending round instead of dispatching.";
