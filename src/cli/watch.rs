//! Actionable worker snapshots and the blocking CLI event consumer.
use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::time::Duration;

use crate::pool::{WorkerMetrics, WorkerRegistryEntry, WorkerState, clamp_string};
use anyhow::Result;
use serde_json::{Value, json};

pub type Snapshot = BTreeMap<String, Value>;

/// The workflow shared by CLI help and the MCP tool description.
pub const WORKFLOW: &str = "Write task as ONE focused concern with files in scope and an acceptance gate. Avoid parallel workers whose scopes share files. To wait, run `mini-swe-mcp watch` in the background: it blocks until an actionable event, prints it and exits, so the host CLI wakes you when it ends; missed events are replayed first. A watch with no worker ids follows every worker you own, including any dispatched after it starts (--group still filters). One watch runs per session: a second is refused (exit 5) so the first is the one the next event wakes. Claude Code sessions started with channels enabled also receive the same events as push notifications. An agent with no shell can call the 'watch' action instead, passing timeout_secs below its host's tool deadline and calling it again on no_event. After completion, review the diff and run the checks. Send every correction AND any merge conflict back to the same worker with steer: it resumes on its branch with full context. Do not edit its branch yourself; merge only when it is right.";

#[derive(Default)]
pub struct Options {
    pub ids: BTreeSet<String>,
    pub group: Option<String>,
    pub follow: bool,
    pub timeout: Option<Duration>,
}
impl Options {
    pub fn parse(args: &[String]) -> Result<Self> {
        let mut out = Self::default();
        let mut i = 2;
        while i < args.len() {
            match args[i].as_str() {
                "--follow" => out.follow = true,
                "--group" | "--timeout" => {
                    let flag = &args[i];
                    i += 1;
                    let value = args
                        .get(i)
                        .ok_or_else(|| anyhow::anyhow!("{flag} requires a value"))?;
                    if flag == "--group" {
                        out.group = Some(value.clone());
                    } else {
                        out.timeout = Some(Duration::from_secs(value.parse().map_err(|_| {
                            anyhow::anyhow!("--timeout expects a whole number of seconds")
                        })?));
                    }
                }
                flag if flag.starts_with('-') => anyhow::bail!("Unknown watch flag: {flag}"),
                id => {
                    out.ids.insert(id.to_string());
                }
            }
            i += 1;
        }
        Ok(out)
    }
}

/// `git diff --shortstat <merge-base>...worker-<id>` in the worker's repo.
///
/// The registry samples diff metrics only while the worktree lives, so a worker
/// whose tree was torn down would otherwise report "0 files, +0 -0" for commits
/// that are still on its branch. The branch is what the orchestrator merges, so
/// measure it against its merge-base with the checked-out base branch.
pub fn branch_diff_stat(entry: &WorkerRegistryEntry) -> Option<(usize, usize, usize)> {
    let repo = entry
        .repo_path
        .as_deref()
        .map(std::path::Path::new)
        .filter(|path| path.is_dir())?;
    let branch = format!("worker-{}", entry.id);
    let merge_base =
        crate::worktree::git(repo, "merge-base", &["merge-base", "HEAD", &branch]).ok()?;
    if !merge_base.status.success() {
        return None;
    }
    let base = String::from_utf8_lossy(&merge_base.stdout)
        .trim()
        .to_string();
    if base.is_empty() {
        return None;
    }
    let range = format!("{base}...{branch}");
    let output =
        crate::worktree::git(repo, "diff --shortstat", &["diff", "--shortstat", &range]).ok()?;
    if !output.status.success() {
        return None;
    }
    crate::pool::parse_shortstat(&String::from_utf8_lossy(&output.stdout))
}

pub fn registry_snapshot(entry: &WorkerRegistryEntry, now: u64) -> Value {
    let mut view = registry_snapshot_row(entry, now);
    // A torn-down worktree means the row's metrics were sampled while the worker
    // still lived: fall back to the branch it left behind, but never overwrite a
    // measured diff with a guess.
    if matches!(view["status"].as_str(), Some("completed" | "failed"))
        && view["metrics"]["diff_files"].as_u64().unwrap_or(0) == 0
        && view["metrics"]["diff_insertions"].as_u64().unwrap_or(0) == 0
        && view["metrics"]["diff_deletions"].as_u64().unwrap_or(0) == 0
        && let Some((files, insertions, deletions)) = branch_diff_stat(entry)
    {
        view["metrics"]["diff_files"] = json!(files);
        view["metrics"]["diff_insertions"] = json!(insertions);
        view["metrics"]["diff_deletions"] = json!(deletions);
    }
    view
}

fn registry_snapshot_row(entry: &WorkerRegistryEntry, now: u64) -> Value {
    json!({"worker_id":entry.id, "owner":entry.owner.as_deref().unwrap_or("unattributed"),
        "model":entry.model, "group":entry.group.as_deref().unwrap_or("default"),
        "status":match entry.status { crate::pool::RegistryStatus::Running=>"running", crate::pool::RegistryStatus::Paused=>"paused", crate::pool::RegistryStatus::Reviewing=>"reviewing", crate::pool::RegistryStatus::Completed=>"completed", crate::pool::RegistryStatus::Failed=>"failed", crate::pool::RegistryStatus::Stopped=>"stopped", crate::pool::RegistryStatus::Interrupted=>"interrupted" }.to_string(),
        "step":entry.step, "turns":entry.step, "max_turns":entry.max_turns,
        "elapsed":if entry.status.is_terminal() {entry.updated_at.saturating_sub(entry.started_at)} else {now.saturating_sub(entry.started_at)}, "last_step_at":entry.updated_at, "question":entry.question.clone(), "last_ops":[clamp_string(&entry.last_command, 256)],
        "metrics":entry.metrics, "branch":null, "revision":0, "summary":null,
        "task":clamp_string(entry.task.lines().next().unwrap_or(""), 500),
        "verified":null, "error":if entry.status == crate::pool::RegistryStatus::Failed {Some(clamp_string(&entry.last_command, 1500))} else {None}})
}

pub fn enrich_state(view: &mut Value, state: &WorkerState) {
    view["step"] = json!(state.step());
    view["turns"] = json!(state.step());
    match state {
        WorkerState::Running { .. } => view["status"] = json!("running"),
        WorkerState::Paused { question, .. } => {
            view["status"] = json!("paused");
            view["question"] = json!(clamp_string(question, 1500));
        }
        WorkerState::Completed {
            summary,
            verified,
            branch,
            revision,
            metrics,
            ..
        } => {
            view["status"] = json!("completed");
            view["summary"] = json!(clamp_string(summary, 1500));
            view["verified"] = json!(verified);
            view["branch"] = json!(branch);
            view["revision"] = json!(revision);
            view["metrics"] = json!(metrics);
        }
        WorkerState::Failed {
            error,
            revision,
            metrics,
            ..
        } => {
            view["status"] = json!("failed");
            view["error"] = json!(clamp_string(error, 1500));
            view["revision"] = json!(revision);
            view["metrics"] = json!(metrics);
        }
    }
}

/// Decide from state and the last reported health baseline, without I/O.
pub fn select_event(view: &Value, previous: Option<&Value>, now: u64) -> Option<Value> {
    let status = view["status"].as_str()?;
    // A worker queued for a heavy build slot is not idle in the worker sense:
    // the time is spent waiting on admission, so it can never be a stall.
    if view["waiting_for_slot"].is_number() && matches!(status, "running" | "reviewing") {
        return None;
    }
    let metrics: WorkerMetrics =
        serde_json::from_value(view["metrics"].clone()).unwrap_or_default();
    let baseline: WorkerMetrics = previous
        .and_then(|v| serde_json::from_value(v["metrics"].clone()).ok())
        .unwrap_or_default();
    let idle = now.saturating_sub(view["last_step_at"].as_u64().unwrap_or(now));
    let event = match status {
        "completed" | "failed" => status,
        "paused" => "needs_input",
        "running" | "reviewing"
            if idle >= 600
                || metrics.repeat_blocks.saturating_sub(baseline.repeat_blocks) >= 3
                || metrics
                    .stagnation_nudges
                    .saturating_sub(baseline.stagnation_nudges)
                    >= 3 =>
        {
            "stalled"
        }
        _ => return None,
    };
    if let Some(old) = previous {
        let old_event = old["event"].as_str().unwrap_or("");
        // Real progress ends the stall episode: the next idle period is a new
        // one and must be reported, not swallowed by the last suppression.
        if old_event == "stalled" && old["step"] != view["step"] && idle < 600 {
            return None;
        }
        if event != "stalled"
            && old_event == event
            && old["revision"] == view["revision"]
            && old["step"] == view["step"]
            && old["question"] == view["question"]
        {
            return None;
        }
        // Suppress the same stall until another blocking episode occurs.
        if event == "stalled"
            && old_event == event
            && metrics.repeat_blocks.saturating_sub(baseline.repeat_blocks) < 3
            && metrics
                .stagnation_nudges
                .saturating_sub(baseline.stagnation_nudges)
                < 3
        {
            return None;
        }
    }
    let mut payload = view.clone();
    payload["event"] = json!(event);
    payload["time_since_last_step"] = json!(idle);
    payload["diff_stat"] = json!({"files":metrics.diff_files,"insertions":metrics.diff_insertions,"deletions":metrics.diff_deletions});
    payload["next_step"] = json!(crate::pool::next_step_for(view["branch"].as_str()));
    payload["commands"] = json!(commands(&payload));
    Some(payload)
}
fn shell(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\\''"))
}
pub(crate) fn commands(v: &Value) -> Vec<String> {
    let id = shell(v["worker_id"].as_str().unwrap_or(""));
    let steer = format!("mini-swe-mcp steer {id} \"<concrete redirection or answer>\"");
    match v["event"].as_str().unwrap_or("") {
        "needs_input" => vec![steer],
        "stalled" => vec![steer, format!("mini-swe-mcp kill {id}")],
        _ => {
            let branch = v["branch"].as_str().map(shell);
            let mut out = vec![
                format!(
                    "git diff HEAD...{}",
                    branch.as_deref().unwrap_or("'<worker branch>'")
                ),
                v["verify_command"]
                    .as_str()
                    .filter(|s| !s.is_empty())
                    .unwrap_or("<run the acceptance checks>")
                    .to_string(),
                format!(
                    "mini-swe-mcp steer {id} \"<corrections or merge conflicts>\" --max-turns 60"
                ),
            ];
            if let Some(branch) = branch {
                out.push(format!("git merge {branch}"));
            }
            out
        }
    }
}

/// The heading a missed batch shares: a `watch` call prints it once, not once
/// per replayed event.
pub const MISSED_HEADING: &str = "While you were not watching:";

pub fn render(v: &Value) -> String {
    if v["missed"] == true {
        format!("{MISSED_HEADING}\n{}", render_event(v))
    } else {
        render_event(v)
    }
}

/// One event body, without the heading a whole missed batch shares.
fn render_event(v: &Value) -> String {
    let text = |key: &str| v[key].as_str().unwrap_or("unknown");
    // A registry row carries no branch, but every worker commits to its own
    // `worker-<id>` branch, so name that instead of admitting we do not know.
    let branch = v["branch"]
        .as_str()
        .map_or_else(|| format!("worker-{}", text("worker_id")), str::to_string);
    let mut out = String::new();
    if let Some(dropped) = v["dropped_events"].as_u64().filter(|n| *n > 0) {
        out.push_str(&format!(
            "{dropped} older events dropped (backlog limit 100).\n"
        ));
    }
    out.push_str(&format!("{}: {} | {} | owner {} | group {} | branch {} | revision {}\nStep {}/{} | elapsed {}s | {}\n", text("worker_id"), text("event"), text("model"), text("owner"), text("group"), branch, v["revision"], v["step"], v["max_turns"], v["elapsed"], text("task")));
    match text("event") {
        "completed" | "failed" => {
            // A registry-only row never ran the gate: stay silent rather than
            // printing a null the orchestrator would have to interpret.
            let verified = v["verified"]
                .as_bool()
                .map_or_else(String::new, |ok| format!("Verified: {ok} | "));
            out.push_str(&format!(
                "{verified}Diff: {} files, +{} -{}\n",
                v["diff_stat"]["files"], v["diff_stat"]["insertions"], v["diff_stat"]["deletions"]
            ));
            for key in ["summary", "error", "verify_output_tail", "next_step"] {
                if let Some(value) = v[key].as_str() {
                    out.push_str(&format!("{key}: {value}\n"));
                }
            }
        }
        "needs_input" => out.push_str(&format!("Question: {}\n", text("question"))),
        _ => {
            let metrics = &v["metrics"];
            let counters = [
                "repeat_blocks",
                "stagnation_nudges",
                "loop_pauses",
                "extensions_refused",
                "verify_failures",
            ]
            .iter()
            .filter_map(|key| {
                metrics[*key]
                    .as_u64()
                    .filter(|n| *n > 0)
                    .map(|n| format!("{key}={n}"))
            })
            .collect::<Vec<_>>()
            .join(", ");
            let ops = v["last_ops"]
                .as_array()
                .map(|ops| {
                    ops.iter()
                        .filter_map(|op| op.as_str())
                        .collect::<Vec<_>>()
                        .join("; ")
                })
                .unwrap_or_default();
            out.push_str(&format!(
                "No step for {}s | {} | Last 5 ops: {}\nSteer with a concrete redirection, or kill.\n",
                v["time_since_last_step"],
                if counters.is_empty() { "no health counter moved".to_string() } else { counters },
                ops
            ));
        }
    }
    if let Some(cmds) = v["commands"].as_array() {
        for cmd in cmds {
            out.push_str(cmd.as_str().unwrap_or(""));
            out.push('\n');
        }
    }
    out.trim_end().to_string()
}

/// Print one batch of events: the missed heading once, then every event body,
/// so a single `watch` call catches the caller up completely.
fn print_events(events: &[Value], json_output: bool, follow: bool) -> Result<()> {
    print_events_to(&mut std::io::stdout(), events, json_output, follow)
}

/// [`print_events`] against an explicit sink, so the batch layout is testable.
fn print_events_to(
    out: &mut impl Write,
    events: &[Value],
    json_output: bool,
    follow: bool,
) -> Result<()> {
    if json_output {
        for event in events {
            writeln!(out, "{}", serde_json::to_string(event)?)?;
        }
        out.flush()?;
        return Ok(());
    }
    if events.iter().any(|event| event["missed"] == true) {
        write!(out, "{MISSED_HEADING}{}", if follow { " | " } else { "\n" })?;
    }
    for event in events {
        let body = render_event(event);
        writeln!(
            out,
            "{}",
            if follow {
                body.replace('\n', " | ")
            } else {
                body
            }
        )?;
    }
    out.flush()?;
    Ok(())
}

/// Update the progress clock only when the turn changes, not on health writes.
pub fn progress_clock(view: &mut Value, old: Option<&Value>, now: u64) {
    // Time queued for a heavy build slot is not inactivity: keep the worker's
    // idle clock at zero while it waits, so granting the slot starts a fresh
    // episode instead of an immediate stall.
    if view["waiting_for_slot"].is_number() {
        view["last_step_at"] = json!(now);
        return;
    }
    if let Some(old) = old {
        view["last_step_at"] = if old["step"] == view["step"] && old["revision"] == view["revision"]
        {
            old["last_step_at"].clone()
        } else {
            json!(now)
        };
    }
}

/// A worker whose status can never produce another watch event.
fn terminal(view: &Value) -> bool {
    matches!(
        view["status"].as_str(),
        Some("completed" | "failed" | "stopped" | "interrupted")
    )
}

pub fn matches(view: &Value, ids: &BTreeSet<String>, group: Option<&str>) -> bool {
    (ids.is_empty()
        || view["worker_id"]
            .as_str()
            .is_some_and(|id| ids.contains(id)))
        && group.is_none_or(|g| view["group"] == g)
}

pub async fn run(args: &[String], json_output: bool, admin: bool) -> Result<i32> {
    let opts = Options::parse(args)?;
    if std::env::var("MINI_SWE_NO_DAEMON").ok().as_deref() == Some("1") {
        return polling(opts, json_output, admin).await;
    }
    let mut client = match crate::hub::HubClient::connect_as_admin(admin).await {
        Ok(client) => client,
        // A hub that predates `hub/watch` cannot stream events; the registry
        // poll sees the same workers, just a second late.
        Err(error) if error.to_string().contains("Method not found") => {
            eprintln!(
                "[mini-swe] The running hub predates 'hub/watch'; falling back to registry polling."
            );
            return polling(opts, json_output, admin).await;
        }
        Err(error) => return Err(error),
    };
    // With no explicit ids the set is dynamic: every worker this caller owns
    // is watched, so a dispatch made after the watch began joins automatically.
    // An explicit id set stays fixed for the whole watch.
    let explicit = !opts.ids.is_empty();
    let started = tokio::time::Instant::now();
    let mut ids = opts.ids.clone();
    let mut initial = true;
    let mut watched_any = false;
    loop {
        let response = match client
            .watch_snapshot(&ids, opts.group.as_deref(), initial)
            .await
        {
            Ok(value) => value,
            Err(error) if error.to_string().contains("a watch is already running") => {
                println!("{error}");
                return Ok(5);
            }
            Err(error) if error.to_string().contains("belongs to agent") => {
                println!("{error}");
                return Ok(4);
            }
            Err(error) if error.to_string().contains("Method not found") => {
                eprintln!(
                    "[mini-swe] The running hub predates 'hub/watch'; falling back to registry polling."
                );
                return polling(opts, json_output, admin).await;
            }
            Err(error) => return Err(error),
        };
        let watching: BTreeSet<String> = response["watching"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect();
        if explicit {
            ids = watching.clone();
        }
        if !watching.is_empty() {
            watched_any = true;
        }
        let events = response["events"].as_array().cloned().unwrap_or_default();
        if initial && explicit && ids.is_empty() && events.is_empty() {
            println!("nothing to watch");
            return Ok(3);
        }
        print_events(&events, json_output, opts.follow)?;
        for event in &events {
            client
                .watch_ack(event["sequence"].as_u64().unwrap_or(0))
                .await?;
        }
        // Every missed event came back in this one reply, so a non-following
        // caller leaves as soon as it has been caught up.
        if !events.is_empty() && !opts.follow {
            return Ok(0);
        }
        initial = false;
        if watching.is_empty() && (explicit || watched_any) {
            return Ok(0);
        }
        let wait = match opts.timeout {
            Some(timeout) => {
                let Some(left) = timeout.checked_sub(started.elapsed()) else {
                    if watched_any {
                        println!("no event");
                        return Ok(2);
                    }
                    println!("nothing to watch");
                    return Ok(3);
                };
                left.min(Duration::from_secs(1))
            }
            None => Duration::from_secs(1),
        };
        // Notifications wake the consumer promptly. The next snapshot request
        // repairs channel overflow from the owner's bounded, unacknowledged backlog.
        if let Ok(result) = tokio::time::timeout(wait, client.next_watch_notification()).await {
            result?;
        }
    }
}

async fn polling(opts: Options, json_output: bool, admin: bool) -> Result<i32> {
    // The same resolution the hub applies to this process: the operator's
    // override, then the host process, then the CLI's own identity.
    let owner = crate::hub::identity::identity(crate::mcp::CLI_AGENT).id;
    // With no explicit ids the set is dynamic: every owned worker joins, so one
    // dispatched after the watch began is followed too. An explicit id set is
    // fixed for the whole watch.
    let explicit = !opts.ids.is_empty();
    let started = tokio::time::Instant::now();
    let mut previous = Snapshot::new();
    let mut reported = Snapshot::new();
    let mut ids = opts.ids.clone();
    let mut ignored: BTreeSet<String> = BTreeSet::new();
    let mut initial = true;
    let mut watched_any = false;
    loop {
        let now = crate::pool::unix_timestamp();
        let mut current: Snapshot = crate::pool::load_all_registry_entries()
            .iter()
            .map(|e| (e.id.clone(), registry_snapshot(e, now)))
            .collect();
        if initial {
            for id in &ids {
                if let Some(v) = current.get(id) {
                    if !admin && v["owner"] != owner {
                        println!(
                            "worker {id} belongs to agent {}",
                            v["owner"].as_str().unwrap_or("unattributed")
                        );
                        return Ok(4);
                    }
                } else {
                    anyhow::bail!("Worker not found: {id}");
                }
            }
            // A worker already terminal when the watch began is not a late
            // dispatch: a no-arg watch must not replay it.
            if !explicit {
                ignored = current
                    .iter()
                    .filter(|(_, v)| {
                        (admin || v["owner"] == owner)
                            && matches(v, &opts.ids, opts.group.as_deref())
                            && terminal(v)
                    })
                    .map(|(id, _)| id.clone())
                    .collect();
            }
            initial = false;
        }
        // Every owned worker joins a no-id watch, whenever it was dispatched;
        // --group still filters the set.
        if !explicit {
            for (id, v) in &current {
                if !ignored.contains(id)
                    && (admin || v["owner"] == owner)
                    && matches(v, &opts.ids, opts.group.as_deref())
                {
                    ids.insert(id.clone());
                }
            }
        }
        current.retain(|id, v| {
            ids.contains(id)
                && (admin || v["owner"] == owner)
                && matches(v, &opts.ids, opts.group.as_deref())
        });
        let mut events = Vec::new();
        for (id, view) in &mut current {
            progress_clock(view, previous.get(id), now);
            if previous
                .get(id)
                .is_some_and(|old| old["status"] != view["status"])
            {
                reported.remove(id);
            }
            if let Some(event) = select_event(view, reported.get(id), now) {
                reported.insert(id.clone(), event.clone());
                events.push(event);
            }
        }
        print_events(&events, json_output, opts.follow)?;
        if !events.is_empty() && !opts.follow {
            return Ok(0);
        }
        if !current.is_empty() {
            watched_any = true;
        }
        ids.retain(|id| current.get(id).is_some_and(|v| !terminal(v)));
        if ids.is_empty() && (explicit || watched_any) {
            return Ok(0);
        }
        previous = current;
        if opts
            .timeout
            .is_some_and(|timeout| started.elapsed() >= timeout)
        {
            if watched_any {
                println!("no event");
                return Ok(2);
            }
            println!("nothing to watch");
            return Ok(3);
        }
        tokio::time::sleep(
            opts.timeout
                .map(|t| {
                    t.saturating_sub(started.elapsed())
                        .min(Duration::from_secs(1))
                })
                .unwrap_or(Duration::from_secs(1)),
        )
        .await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pool::WorkerMetrics;

    fn state(status: &str, step: usize, metrics: WorkerMetrics, last: u64, max: usize) -> Value {
        json!({"worker_id":"w","owner":"cli","model":"m","group":"g","branch":"worker-w","revision":0,
            "step":step,"turns":step,"max_turns":max,"elapsed":7,"last_step_at":last,"task":"fix it",
            "question":"go on?","summary":"Done.","verified":true,"error":null,"metrics":metrics,
            "status":status,"last_ops":["a"]})
    }

    #[test]
    fn every_actionable_event_is_selected_and_rendered() {
        let metrics = WorkerMetrics::default();
        assert_eq!(
            select_event(&state("completed", 3, metrics, 100, 10), None, 110).unwrap()["event"],
            "completed"
        );
        assert_eq!(
            select_event(&state("failed", 3, metrics, 100, 10), None, 110).unwrap()["event"],
            "failed"
        );
        assert_eq!(
            select_event(&state("paused", 2, metrics, 100, 10), None, 110).unwrap()["event"],
            "needs_input"
        );
        let mut repeated = metrics;
        repeated.repeat_blocks = 3;
        assert_eq!(
            select_event(
                &state("running", 2, repeated, 100, 10),
                Some(&state("running", 2, metrics, 100, 10)),
                110
            )
            .unwrap()["event"],
            "stalled"
        );
        assert_eq!(
            select_event(&state("running", 2, metrics, 0, 10), None, 700).unwrap()["event"],
            "stalled"
        );
        assert!(select_event(&state("running", 2, metrics, 100, 10), None, 110).is_none());
        let done = select_event(&state("completed", 3, metrics, 100, 10), None, 110).unwrap();
        assert!(select_event(&state("completed", 3, metrics, 100, 10), Some(&done), 110).is_none());
        let out = render(&done);
        assert!(
            out.contains("mini-swe-mcp steer") && out.contains("git diff"),
            "{out}"
        );
        let text = render(
            &select_event(
                &state("running", 2, repeated, 100, 10),
                Some(&state("running", 2, metrics, 100, 10)),
                110,
            )
            .unwrap(),
        );
        assert!(text.contains("mini-swe-mcp kill"), "{text}");
    }

    /// A stall is one episode: the worker taking another step while still idle
    /// must not re-deliver it with the step it has already passed.
    #[test]
    fn a_stall_is_reported_once_per_episode() {
        let metrics = WorkerMetrics::default();
        let first = select_event(&state("running", 160, metrics, 100, 162), None, 700)
            .expect("an idle running worker stalls");
        assert_eq!(first["event"], "stalled");
        assert_eq!(first["step"], 160);

        // The worker advanced but is still idle: same episode, no repeat.
        assert!(
            select_event(&state("running", 162, metrics, 100, 162), Some(&first), 700).is_none(),
            "a step change alone must not re-report the stall"
        );
        // The worker stepped again and is still idle: still the same episode.
        assert!(
            select_event(&state("running", 163, metrics, 100, 162), Some(&first), 700).is_none(),
            "the stall must not follow the worker step by step"
        );

        // A fresh episode -- the worker blocked three more times -- reports again.
        let mut blocked = metrics;
        blocked.repeat_blocks = 3;
        let again = select_event(&state("running", 164, blocked, 100, 162), Some(&first), 700)
            .expect("a new blocking episode reports again");
        assert_eq!(again["event"], "stalled");
        assert_eq!(again["step"], 164, "the payload carries the current step");
    }

    #[test]
    fn a_registry_only_row_names_the_branch_and_stays_silent_about_verification() {
        let mut view = state("completed", 3, WorkerMetrics::default(), 100, 10);
        view["branch"] = Value::Null;
        view["verified"] = Value::Null;
        let text = render(&select_event(&view, None, 110).expect("terminal row is actionable"));
        assert!(text.contains("branch worker-w"), "{text}");
        assert!(!text.contains("unknown"), "{text}");
        assert!(!text.contains("Verified"), "{text}");
        // A row that did run the gate still reports the outcome.
        let mut verified = view.clone();
        verified["verified"] = json!(false);
        let text = render(&select_event(&verified, None, 110).expect("terminal row is actionable"));
        assert!(text.contains("Verified: false"), "{text}");
    }
}

#[cfg(test)]
mod replay_batch_tests {
    use super::*;

    fn missed(id: &str, summary: &str) -> Value {
        json!({"worker_id": id, "event": "completed", "missed": true, "owner": "cli",
            "summary": summary, "verified": true, "diff_stat": {"files": 1, "insertions": 2, "deletions": 0}})
    }

    #[test]
    fn a_missed_batch_shares_one_heading_and_prints_every_event() {
        let batch = vec![missed("w-one", "First."), missed("w-two", "Second.")];
        let mut out = Vec::new();
        print_events_to(&mut out, &batch, false, false).expect("print");
        let text = String::from_utf8(out).expect("utf8");
        assert_eq!(text.matches(MISSED_HEADING).count(), 1, "{text}");
        assert!(
            text.contains("First.") && text.contains("Second."),
            "{text}"
        );
        assert!(text.contains("w-one") && text.contains("w-two"), "{text}");
    }

    #[test]
    fn a_mixed_batch_still_prints_the_heading_once() {
        let live = json!({"worker_id": "w-live", "event": "needs_input", "question": "go on?",
            "owner": "cli"});
        let mut out = Vec::new();
        print_events_to(&mut out, &[missed("w-old", "Late."), live], false, false).expect("print");
        let text = String::from_utf8(out).expect("utf8");
        assert_eq!(text.matches(MISSED_HEADING).count(), 1, "{text}");
        assert!(text.contains("Late.") && text.contains("go on?"), "{text}");
    }
}
