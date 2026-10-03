use super::*;

impl McpServer {
    /// Refuse a dispatch that would push `agent` past its cap, naming the
    /// workers already running so the caller can pick one to collect.
    pub(super) async fn check_agent_cap(&self, agent: &str, cap: usize) -> Result<()> {
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

    /// Dispatch one worker, or a batch when the call carries `tasks`.
    ///
    /// A batch is a list of `{task, model?, repo_path?, max_turns?, verify?,
    /// group?, network?}` objects; shared top-level dispatch values act as
    /// defaults for every entry. Each entry goes through the same single-task
    /// path, so one bad entry only answers with its own error.
    pub(super) async fn handle_dispatch(
        &self,
        args: &Value,
        token: Option<&Value>,
        tx: Option<&mpsc::Sender<String>>,
        ctx: &crate::mcp::server::ConnectionContext,
    ) -> Result<Value> {
        if args.get("tasks").is_some() {
            return self.handle_batch_dispatch(args, token, tx, ctx).await;
        }
        self.dispatch_one(args, token, tx, ctx).await
    }

    fn dispatch_round_key(args: &Value, owner: &str) -> (String, String) {
        let group = args
            .get("group")
            .and_then(Value::as_str)
            .map(str::to_string)
            .or_else(|| {
                args.get("task")
                    .and_then(Value::as_str)
                    .and_then(crate::pool::extract_group)
            })
            .unwrap_or_else(|| "default".into());
        (owner.into(), group)
    }

    /// Effective arguments of one batch entry: the shared top-level dispatch
    /// values, overridden by the entry's own keys.
    pub(super) fn batch_entry_args(shared: &Value, entry: &Value) -> Result<Map<String, Value>> {
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
    pub(super) async fn handle_batch_dispatch(
        &self,
        args: &Value,
        token: Option<&Value>,
        tx: Option<&mpsc::Sender<String>>,
        ctx: &crate::mcp::server::ConnectionContext,
    ) -> Result<Value> {
        let tasks = args
            .get("tasks")
            .and_then(Value::as_array)
            .ok_or_else(|| anyhow::anyhow!("'tasks' must be an array for action 'dispatch'"))?;
        if tasks.is_empty() {
            anyhow::bail!("'tasks' must contain at least one task for action 'dispatch'");
        }

        let keys = tasks
            .iter()
            .filter_map(|entry| Self::batch_entry_args(args, entry).ok())
            .map(|entry| Self::dispatch_round_key(&Value::Object(entry), &ctx.agent()))
            .collect();
        let store = self.auto_store();
        let _round_dispatch = match store {
            Some(store) => Some(store.dispatch_guard(keys).await),
            None => None,
        };
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
                                .unwrap_or_else(|| json!(crate::mcp::schema::NETWORK_DEFAULT)),
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
        self.with_watch_command(&mut payload, ctx).await;
        Ok(payload)
    }

    /// Dispatch exactly one worker from a `dispatch` argument object.
    ///
    /// Shared by the single-task form and every batch entry, so both go through
    /// one validation path.
    pub(super) async fn dispatch_one(
        &self,
        args: &Value,
        token: Option<&Value>,
        tx: Option<&mpsc::Sender<String>>,
        ctx: &crate::mcp::server::ConnectionContext,
    ) -> Result<Value> {
        let store = self.auto_store();
        let _round_dispatch = match store {
            Some(store) if args.get("role").and_then(Value::as_str) != Some("consolidate") => Some(
                store
                    .dispatch_guard(vec![Self::dispatch_round_key(args, &ctx.agent())])
                    .await,
            ),
            _ => None,
        };
        let task = Self::required_string(args, "task", "dispatch")?.to_string();
        let agent = ctx.agent();
        // Fairness gate: one agent may not fill the pool, so its dispatches
        // stop at `MAX_WORKERS_PER_AGENT` running workers (0 = unlimited).
        self.check_agent_cap(&agent, self.max_workers_per_agent)
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
        // `--review-after <model>[:<mode>]`: the mode suffix is split off
        // *before* the model is resolved, so an alias (`nerd:security`) still
        // resolves to its id and the suffix survives to the phase loop, which
        // re-parses it against the manifest. An unknown mode is a dispatch
        // error listing the available ones.
        let review_after = args
            .get("review_after")
            .and_then(|v| v.as_str())
            .map(|s| -> anyhow::Result<String> {
                let (model, mode) =
                    crate::pool::ReviewMode::parse_with_manifest(s, &self.manifest)?;
                // An empty model part uses the mode's default reviewer; the
                // phase loop resolves it against the implementer's model.
                let resolved = if model.trim().is_empty() {
                    model
                } else {
                    self.manifest.resolve_model(&model).0
                };
                if mode.is_security() && mode.checklist.is_none() {
                    Ok(format!(
                        "{resolved}:{}",
                        crate::pool::ReviewMode::SECURITY_SUFFIX
                    ))
                } else if mode.name.eq_ignore_ascii_case("quality") && mode.checklist.is_none() {
                    Ok(resolved)
                } else {
                    Ok(format!("{resolved}:{}", mode.name))
                }
            })
            .transpose()?;

        let network_offline =
            Self::resolve_network_policy(args, "dispatch", &self.manifest, &resolved_model)?;

        // Optional verify gate: an explicit string (possibly empty to disable)
        // is passed through verbatim and wins over every default. An absent
        // argument is auto-detected: the project's full gate, except on a
        // dispatch that asks to consolidate the round, whose workers get the
        // cheap static gate -- the consolidator (dispatched separately, with
        // `role: "consolidate"`) still runs the full one. A non-string is
        // refused here rather than dropped, so a caller who meant a gate never
        // silently gets a default one.
        let verify = match args.get("verify") {
            Some(value) => Some(
                value
                    .as_str()
                    .ok_or_else(|| {
                        anyhow::anyhow!("'verify' must be a string for action 'dispatch'")
                    })?
                    .to_string(),
            ),
            None if matches!(
                args.get("consolidate"),
                Some(Value::Bool(true) | Value::String(_))
            ) && args.get("role").and_then(Value::as_str) != Some("consolidate") =>
            {
                crate::pool::detect_cheap_verify_command(&repo_path)
            }
            None => None,
        };
        // Parse-checked where it enters: a gate stored verbatim is run by a
        // worker (or, for `consolidate_verify`, by a consolidator dispatched
        // long after anyone was watching), and an unparsable one fails there
        // with nobody left to fix it.
        if let Some(gate) = &verify {
            crate::pool::validate_verify_command(gate, "verify")?;
        }

        self.validate_auto_consolidate(args)?;
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

        if role == crate::pool::WorkerRole::Worker {
            self.record_auto_dispatch(args, &agent, &wid).await?;
        }

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
            "network": if network_offline { "offline" } else { crate::mcp::schema::NETWORK_DEFAULT },
            "message": "Worker is executing in isolated worktree in background. Use 'watch' (or mini-swe-mcp watch) to wait for its next event."
        });
        self.with_watch_command(&mut payload, ctx).await;
        Ok(payload)
    }
}

pub(in crate::mcp) const TASK_DESCRIPTION: &str =
    "ONE focused concern: files in scope, acceptance gate.";

pub(in crate::mcp) const TASKS_DESCRIPTION: &str = "Batch {task, model?, ...}; top-level defaults.";

pub(in crate::mcp) const REPO_PATH_DESCRIPTION: &str =
    "Repository root (alias: 'path'). Required for 'dispatch'.";

pub(in crate::mcp) const REVIEW_AFTER_DESCRIPTION: &str =
    "Reviewer `<model>:<mode>`; see `manifest`.";

pub(in crate::mcp) const AUTO_CONSOLIDATE_DESCRIPTION: &str =
    "Auto-consolidate the group when it stops: boolean or model.";

pub(in crate::mcp) const VERIFY_DESCRIPTION: &str = "Completion gate: auto-detect if omitted; empty string disables. On consolidate: Cheap for workers, full consolidator.";

pub(in crate::mcp) const NETWORK_DESCRIPTION: &str =
    "Network: 'offline' isolates every step (no egress); 'allow' (default) keeps it.";
