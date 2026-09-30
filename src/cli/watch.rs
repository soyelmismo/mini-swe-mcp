//! Actionable worker snapshots and the blocking CLI event consumer.
use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::time::Duration;

use anyhow::Result;
use serde_json::{Value, json};
use crate::pool::{WorkerMetrics, WorkerRegistryEntry, WorkerState, clamp_string};

pub type Snapshot = BTreeMap<String, Value>;

/// The workflow shared by CLI help and the MCP tool description.
pub const WORKFLOW: &str = "Write task as ONE focused concern with files in scope and an acceptance gate. Avoid parallel workers whose scopes share files. Use mini-swe-mcp watch to wait for events instead of polling status. After completion, review the diff and run the checks. Send every correction AND any merge conflict back to the same worker with steer: it resumes on its branch with full context. Do not edit its branch yourself; merge only when it is right.";

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
                    let value = args.get(i).ok_or_else(|| anyhow::anyhow!("{flag} requires a value"))?;
                    if flag == "--group" { out.group = Some(value.clone()); }
                    else { out.timeout = Some(Duration::from_secs(value.parse().map_err(|_| anyhow::anyhow!("--timeout expects a whole number of seconds"))?)); }
                }
                flag if flag.starts_with('-') => anyhow::bail!("Unknown watch flag: {flag}"),
                id => { out.ids.insert(id.to_string()); }
            }
            i += 1;
        }
        Ok(out)
    }
}

pub fn registry_snapshot(entry: &WorkerRegistryEntry, now: u64) -> Value {
    json!({"worker_id":entry.id, "owner":entry.owner.as_deref().unwrap_or("unattributed"),
        "model":entry.model, "group":entry.group.as_deref().unwrap_or("default"),
        "status":match entry.status { crate::pool::RegistryStatus::Running=>"running", crate::pool::RegistryStatus::Paused=>"paused", crate::pool::RegistryStatus::Reviewing=>"reviewing", crate::pool::RegistryStatus::Completed=>"completed", crate::pool::RegistryStatus::Failed=>"failed", crate::pool::RegistryStatus::Stopped=>"stopped" }.to_string(),
        "step":entry.step, "turns":entry.step, "max_turns":entry.max_turns,
        "elapsed":now.saturating_sub(entry.started_at), "last_step_at":entry.updated_at, "last_op":"", "question":entry.question.clone(), "last_ops":[clamp_string(&entry.last_command, 256)],
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
            view["status"] = json!("paused"); view["question"] = json!(clamp_string(question, 1500));
        }
        WorkerState::Completed { summary, verified, branch, revision, metrics, .. } => {
            view["status"] = json!("completed"); view["summary"] = json!(clamp_string(summary, 1500));
            view["verified"] = json!(verified); view["branch"] = json!(branch);
            view["revision"] = json!(revision); view["metrics"] = json!(metrics);
        }
        WorkerState::Failed { error, revision, metrics, .. } => {
            view["status"] = json!("failed"); view["error"] = json!(clamp_string(error, 1500));
            view["revision"] = json!(revision); view["metrics"] = json!(metrics);
        }
    }
}

/// Decide from state and the last reported health baseline, without I/O.
pub fn select_event(view: &Value, previous: Option<&Value>, now: u64) -> Option<Value> {
    let status = view["status"].as_str()?;
    let metrics: WorkerMetrics = serde_json::from_value(view["metrics"].clone()).unwrap_or_default();
    let old: WorkerMetrics = previous.and_then(|v| serde_json::from_value(v["metrics"].clone()).ok()).unwrap_or_default();
    let idle = now.saturating_sub(view["last_step_at"].as_u64().unwrap_or(now));
    let event = match status {
        "completed" | "failed" => status,
        "paused" => "needs_input",
        "running" | "reviewing" if idle >= 600 || metrics.repeat_blocks.saturating_sub(old.repeat_blocks) >= 3 || metrics.stagnation_nudges.saturating_sub(old.stagnation_nudges) >= 3 => "stalled",
        _ => return None,
    };
    if let Some(old) = previous {
        if event != "stalled" && old["event"] == event && old["revision"] == view["revision"] && old["step"] == view["step"] && old["question"] == view["question"] { return None; }
        if event == "stalled" && old["event"] == event && old["step"] == view["step"] && metrics.repeat_blocks.saturating_sub(old["metrics"]["repeat_blocks"].as_u64().unwrap_or(0) as usize) < 3 && metrics.stagnation_nudges.saturating_sub(old["metrics"]["stagnation_nudges"].as_u64().unwrap_or(0) as usize) < 3 { return None; }
    }
    let mut payload = view.clone();
    payload["event"] = json!(event);
    payload["time_since_last_step"] = json!(idle);
    payload["diff_stat"] = json!({"files":metrics.diff_files,"insertions":metrics.diff_insertions,"deletions":metrics.diff_deletions});
    payload["next_step"] = json!(crate::pool::next_step_for(view["branch"].as_str()));
    payload["commands"] = json!(commands(&payload));
    Some(payload)
}
fn shell(value: &str) -> String { format!("'{}'", value.replace('\'', "'\\''")) }
fn commands(v: &Value) -> Vec<String> {
    let id = shell(v["worker_id"].as_str().unwrap_or(""));
    let steer = format!("mini-swe-mcp steer {id} \"<concrete redirection or answer>\"");
    match v["event"].as_str().unwrap_or("") {
        "needs_input" => vec![steer],
        "stalled" => vec![steer, format!("mini-swe-mcp kill {id}")],
        _ => {
            let branch = v["branch"].as_str().map(shell);
            let mut out = vec![format!("git diff HEAD...{}", branch.as_deref().unwrap_or("'<worker branch>'")), "<run the acceptance checks>".to_string(), format!("mini-swe-mcp steer {id} \"<corrections or merge conflicts>\" --max-turns 60")];
            if let Some(branch) = branch { out.push(format!("git merge {branch}")); }
            out
        }
    }
}

pub fn render(v: &Value) -> String {
    let text = |key: &str| v[key].as_str().unwrap_or("unknown");
    let mut out = if v["missed"] == true { "While you were not watching:\n".to_string() } else { String::new() };
    if let Some(dropped) = v["dropped_events"].as_u64().filter(|n| *n > 0) { out.push_str(&format!("{dropped} older events dropped (backlog limit 100).\n")); }
    out.push_str(&format!("{}: {} | {} | owner {} | group {} | branch {} | revision {}\nStep {}/{} | elapsed {}s | {}\n", text("worker_id"), text("event"), text("model"), text("owner"), text("group"), text("branch"), v["revision"], v["step"], v["max_turns"], v["elapsed"], text("task")));
    match text("event") {
        "completed" | "failed" => {
            out.push_str(&format!("Verified: {} | Diff: {} files, +{} -{}\n", v["verified"], v["diff_stat"]["files"], v["diff_stat"]["insertions"], v["diff_stat"]["deletions"]));
            for key in ["summary", "error", "verify_output_tail", "next_step"] { if let Some(value) = v[key].as_str() { out.push_str(&format!("{key}: {value}\n")); } }
        }
        "needs_input" => out.push_str(&format!("Question: {}\n", text("question"))),
        _ => out.push_str(&format!("No step for {}s | counters {}\nLast 5 ops: {}\nSteer with a concrete redirection, or kill.\n", v["time_since_last_step"], v["metrics"], v["last_ops"])),
    }
    if let Some(cmds) = v["commands"].as_array() { for cmd in cmds { out.push_str(cmd.as_str().unwrap_or("")); out.push('\n'); } }
    out.trim_end().to_string()
}

fn print_event(event: &Value, json_output: bool, follow: bool) -> Result<()> {
    let output = if json_output { serde_json::to_string(event)? } else { render(event) };
    println!("{}", if follow && !json_output { output.replace('\n', " | ") } else { output });
    std::io::stdout().flush()?;
    Ok(())
}


/// Update the progress clock only when the turn changes, not on health writes.
pub fn progress_clock(view: &mut Value, old: Option<&Value>, now: u64) {
    if let Some(old) = old {
        view["last_step_at"] = if old["step"] == view["step"] && old["revision"] == view["revision"] {
            old["last_step_at"].clone()
        } else { json!(now) };
    }
}

pub fn matches(view: &Value, ids: &BTreeSet<String>, group: Option<&str>) -> bool {
    (ids.is_empty() || view["worker_id"].as_str().is_some_and(|id| ids.contains(id)))
        && group.is_none_or(|g| view["group"] == g)
}

pub async fn run(args: &[String], json_output: bool, admin: bool) -> Result<i32> {
    let opts = Options::parse(args)?;
    if std::env::var("MINI_SWE_NO_DAEMON").ok().as_deref() == Some("1") {
        return polling(opts, json_output, admin).await;
    }
    let mut client = crate::hub::HubClient::connect_as_admin(admin).await?;
    let started = tokio::time::Instant::now();
    let mut ids = opts.ids.clone();
    let mut initial = true;
    loop {
        let response = match client.watch_snapshot(&ids, opts.group.as_deref(), initial).await {
            Ok(value) => value,
            Err(error) if error.to_string().contains("belongs to agent") => { println!("{error}"); return Ok(4); }
            Err(error) => return Err(error),
        };
        ids = response["watching"].as_array().into_iter().flatten().filter_map(|v| v.as_str().map(str::to_string)).collect();
        let events = response["events"].as_array().cloned().unwrap_or_default();
        if initial && ids.is_empty() && events.is_empty() { println!("nothing to watch"); return Ok(3); }
        for event in events {
            print_event(&event, json_output, opts.follow)?;
            client.watch_ack(event["sequence"].as_u64().unwrap_or(0)).await?;
            if !opts.follow { return Ok(0); }
        }
        initial = false;
        if ids.is_empty() { return Ok(0); }
        let wait = match opts.timeout {
            Some(timeout) => {
                let Some(left) = timeout.checked_sub(started.elapsed()) else { println!("no event"); return Ok(2); };
                left.min(Duration::from_secs(1))
            }
            None => Duration::from_secs(1),
        };
        // Notifications wake the consumer promptly. The next snapshot request
        // repairs channel overflow from the owner's bounded, unacknowledged backlog.
        if let Ok(result) = tokio::time::timeout(wait, client.next_watch_notification()).await { result?; }
    }
}

async fn polling(opts: Options, json_output: bool, admin: bool) -> Result<i32> {
    let owner = std::env::var("MINI_SWE_AGENT_ID").ok().filter(|v| !v.is_empty()).unwrap_or_else(|| crate::mcp::CLI_AGENT.to_string());
    let started = tokio::time::Instant::now();
    let mut previous = Snapshot::new();
    let mut reported = Snapshot::new();
    let mut ids = opts.ids.clone();
    let mut initial = true;
    loop {
        let now = crate::pool::unix_timestamp();
        let mut current: Snapshot = crate::pool::load_all_registry_entries().iter().map(|e| (e.id.clone(), registry_snapshot(e, now))).collect();
        if initial {
            for id in &ids {
                if let Some(v) = current.get(id) {
                    if !admin && v["owner"] != owner { println!("worker {id} belongs to agent {}", v["owner"].as_str().unwrap_or("unattributed")); return Ok(4); }
                } else { anyhow::bail!("Worker not found: {id}"); }
            }
            if ids.is_empty() {
                ids = current.values().filter(|v| (admin || v["owner"] == owner) && matches(v, &opts.ids, opts.group.as_deref()) && matches!(v["status"].as_str(), Some("running" | "paused"))).filter_map(|v| v["worker_id"].as_str().map(str::to_string)).collect();
            }
            if ids.is_empty() { println!("nothing to watch"); return Ok(3); }
            initial = false;
        }
        current.retain(|id, v| ids.contains(id) && (admin || v["owner"] == owner) && matches(v, &ids, opts.group.as_deref()));
        for (id, view) in &mut current {
            progress_clock(view, previous.get(id), now);
            if let Some(event) = select_event(view, reported.get(id), now) {
                print_event(&event, json_output, opts.follow)?;
                reported.insert(id.clone(), event);
                if !opts.follow { return Ok(0); }
            }
        }
        ids.retain(|id| current.get(id).is_some_and(|v| !matches!(v["status"].as_str(), Some("completed" | "failed" | "stopped"))));
        if ids.is_empty() { return Ok(0); }
        previous = current;
        if opts.timeout.is_some_and(|timeout| started.elapsed() >= timeout) { println!("no event"); return Ok(2); }
        tokio::time::sleep(opts.timeout.map(|t| t.saturating_sub(started.elapsed()).min(Duration::from_secs(1))).unwrap_or(Duration::from_secs(1))).await;
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
        assert_eq!(select_event(&state("completed", 3, metrics, 100, 10), None, 110).unwrap()["event"], "completed");
        assert_eq!(select_event(&state("failed", 3, metrics, 100, 10), None, 110).unwrap()["event"], "failed");
        assert_eq!(select_event(&state("paused", 2, metrics, 100, 10), None, 110).unwrap()["event"], "needs_input");
        let mut repeated = metrics;
        repeated.repeat_blocks = 3;
        assert_eq!(
            select_event(&state("running", 2, repeated, 100, 10), Some(&state("running", 2, metrics, 100, 10)), 110).unwrap()["event"],
            "stalled"
        );
        assert_eq!(select_event(&state("running", 2, metrics, 0, 10), None, 700).unwrap()["event"], "stalled");
        assert!(select_event(&state("running", 2, metrics, 100, 10), None, 110).is_none());
        let done = select_event(&state("completed", 3, metrics, 100, 10), None, 110).unwrap();
        assert!(select_event(&state("completed", 3, metrics, 100, 10), Some(&done), 110).is_none());
        let out = render(&done);
        assert!(out.contains("mini-swe-mcp steer") && out.contains("git diff"), "{out}");
        let text = render(&select_event(&state("running", 2, repeated, 100, 10), Some(&state("running", 2, metrics, 100, 10)), 110).unwrap());
        assert!(text.contains("mini-swe-mcp kill"), "{text}");
    }
}
