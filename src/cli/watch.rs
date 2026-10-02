//! Actionable worker snapshots and the blocking CLI event consumer.
use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::time::Duration;

use crate::hub::client::{daemon_went_away, reconnect_deadline};
use crate::pool::{WorkerMetrics, WorkerRegistryEntry, WorkerState, clamp_string};
use anyhow::Result;
use serde_json::{Value, json};

pub type Snapshot = BTreeMap<String, Value>;

#[derive(Default)]
pub struct Options {
    pub ids: BTreeSet<String>,
    /// The rounds this watch follows: several `--group` flags are one watch,
    /// not several. An empty set means "every live group of the caller", which
    /// only `--all` may read that way.
    pub groups: BTreeSet<String>,
    pub follow: bool,
    /// Print the full event body (long next-step guidance, every command).
    /// `--json` stays complete regardless of this flag.
    pub verbose: bool,
    pub timeout: Option<Duration>,
    /// Wait for whole rounds: one event as soon as any selected round has
    /// fully stopped (that round's per-worker lines), or as soon as one worker
    /// needs input or fails. Without a group or ids it covers every live group
    /// of the caller.
    pub all: bool,
}
impl Options {
    pub fn parse(args: &[String]) -> Result<Self> {
        let mut out = Self::default();
        let mut i = 2;
        while i < args.len() {
            match args[i].as_str() {
                "--follow" => out.follow = true,
                "--verbose" => out.verbose = true,
                "--all" => out.all = true,
                "--group" | "--timeout" => {
                    let flag = &args[i];
                    i += 1;
                    let value = args
                        .get(i)
                        .ok_or_else(|| anyhow::anyhow!("{flag} requires a value"))?;
                    if flag == "--group" {
                        out.groups.insert(value.clone());
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
        // `--all` without a group or ids is not a refusal any more: it covers
        // every live group the caller owns.
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
    if matches!(
        view["status"].as_str(),
        Some("completed" | "failed" | "exhausted")
    ) && view["metrics"]["diff_files"].as_u64().unwrap_or(0) == 0
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
        "status":match entry.status { crate::pool::RegistryStatus::Running=>"running", crate::pool::RegistryStatus::Paused=>"paused", crate::pool::RegistryStatus::Reviewing=>"reviewing", crate::pool::RegistryStatus::Completed=>"completed", crate::pool::RegistryStatus::Failed=>"failed", crate::pool::RegistryStatus::Exhausted=>"exhausted", crate::pool::RegistryStatus::Stopped=>"stopped", crate::pool::RegistryStatus::Interrupted=>"interrupted" }.to_string(),
        "step":entry.step, "turns":entry.step, "max_turns":entry.max_turns,
        "elapsed":if entry.status.is_terminal() {entry.updated_at.saturating_sub(entry.started_at)} else {now.saturating_sub(entry.started_at)}, "last_step_at":entry.updated_at, "question":entry.question.clone(), "last_ops":[clamp_string(&entry.last_command, 256)],
        "metrics":entry.metrics, "branch":null, "revision":entry.revision, "summary":null,
        "task":clamp_string(entry.task.lines().next().unwrap_or(""), 500),
        "verified":entry.verified, "report":entry.report,
        "error":if entry.status == crate::pool::RegistryStatus::Failed {Some(clamp_string(&entry.last_command, 1500))} else {None}})
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
            diff,
            report,
            ..
        } => {
            view["status"] = json!("completed");
            view["summary"] = json!(clamp_string(summary, 1500));
            view["verified"] = json!(verified);
            view["branch"] = json!(branch);
            view["revision"] = json!(revision);
            view["metrics"] = json!(metrics);
            // The report rides on the view so a `watch` event can name what
            // changed, and the per-file split comes from the diff the state
            // already holds rather than a second `git diff --stat`.
            view["report"] = json!(report);
            view["per_file"] = json!(crate::pool::file_stats_of_diff(diff));
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
        WorkerState::Exhausted {
            summary,
            branch,
            revision,
            metrics,
            diff,
            report,
            ..
        } => {
            view["status"] = json!("exhausted");
            view["summary"] = json!(clamp_string(summary, 1500));
            // Never verified: an exhausted worker stopped before its gate ran.
            view["verified"] = json!(false);
            view["error"] = json!(crate::pool::TURN_BUDGET_EXHAUSTED);
            view["branch"] = json!(branch);
            view["revision"] = json!(revision);
            view["metrics"] = json!(metrics);
            view["report"] = json!(report);
            view["per_file"] = json!(crate::pool::file_stats_of_diff(diff));
        }
    }
}

/// The health counters that grew between two samples, as `key=delta`.
///
/// Only a counter that moved is named: a run that never nudged or blocked
/// anything must not read as if it had.
fn moved_counters_since(now: &WorkerMetrics, before: &WorkerMetrics) -> Vec<String> {
    const COUNTERS: [&str; 5] = [
        "repeat_blocks",
        "stagnation_nudges",
        "loop_pauses",
        "extensions_refused",
        "verify_failures",
    ];
    COUNTERS
        .iter()
        .filter_map(|key| {
            let delta = match *key {
                "repeat_blocks" => now.repeat_blocks.saturating_sub(before.repeat_blocks),
                "stagnation_nudges" => now
                    .stagnation_nudges
                    .saturating_sub(before.stagnation_nudges),
                "loop_pauses" => now.loop_pauses.saturating_sub(before.loop_pauses),
                "extensions_refused" => now
                    .extensions_refused
                    .saturating_sub(before.extensions_refused),
                _ => now.verify_failures.saturating_sub(before.verify_failures),
            };
            (delta > 0).then(|| format!("{key}={delta}"))
        })
        .collect()
}

/// Whether `view` is doing work the watch must not read as inactivity: a bash
/// command in flight, or a command that outlived its budget and keeps running
/// on as a background job the worker waits on.
///
/// The conversion itself lands exactly on the idle threshold: the executor's
/// mark clears the moment the job handle is returned, so reading that instant
/// as idle reports a stall for a step the worker is still making progress on.
/// A worker that goes idle for the whole threshold *after* the conversion has
/// no job left in flight and stalls as it should.
fn in_flight(view: &Value, status: &str) -> bool {
    matches!(status, "running" | "reviewing")
        && (view["command_started_at"].is_number() || has_live_job(view))
}

/// Whether the view lists a background job that has not been reaped yet.
///
/// A reaped job stays listed until the worker collects it, so a job line only
/// counts as activity while it says it is still running.
fn has_live_job(view: &Value) -> bool {
    view["jobs"]
        .as_array()
        .is_some_and(|jobs| jobs.iter().any(|job| job.as_str().is_some_and(running_job)))
}

/// Whether a `job <n>: <command> (running, 12s)` status line is a live job.
fn running_job(label: &str) -> bool {
    label.contains("(running,")
}

/// Decide from state and the last reported health baseline, without I/O.
pub fn select_event(view: &Value, previous: Option<&Value>, now: u64) -> Option<Value> {
    let status = view["status"].as_str()?;
    // A worker queued for a heavy build slot is not idle in the worker sense:
    // the time is spent waiting on admission, so it can never be a stall.
    if view["waiting_for_slot"].is_number() && matches!(status, "running" | "reviewing") {
        return None;
    }
    // A command that is still executing is work in flight, not inactivity: a
    // long `cargo test` or verify gate must never read as a stall. A command
    // that reached its budget and became a background job is the same command
    // from here: it keeps running in its own process group and the worker is
    // only waiting on it, so it is activity too.
    if in_flight(view, status) {
        return None;
    }
    let metrics: WorkerMetrics =
        serde_json::from_value(view["metrics"].clone()).unwrap_or_default();
    let baseline: WorkerMetrics = previous
        .and_then(|v| serde_json::from_value(v["metrics"].clone()).ok())
        .unwrap_or_default();
    let idle = now.saturating_sub(view["last_step_at"].as_u64().unwrap_or(now));
    let event = match status {
        "completed" | "failed" | "exhausted" => status,
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
    // The counters that moved since the last snapshot, so a compact stall
    // event can name why it fired instead of only how long it has been idle.
    payload["moved_counters"] = json!(moved_counters_since(&metrics, &baseline));
    // An exhausted worker is stopped, not done: its next step is to continue
    // it with a fresh budget, not to review and merge its branch.
    payload["next_step"] = json!(if view["status"] == json!("exhausted") {
        crate::pool::exhausted_next_step(
            view["worker_id"].as_str().unwrap_or(""),
            view["turns"].as_u64().unwrap_or(0) as usize,
            view["branch"].as_str(),
        )
    } else {
        crate::pool::next_step_for(view["branch"].as_str())
    });
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
        "exhausted" => vec![format!(
            "mini-swe-mcp steer {id} \"continue\" --max-turns {}",
            v["turns"].as_u64().unwrap_or(0).max(1)
        )],
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
    render_with(v, false)
}

/// The complete event body: the long next-step guidance and every suggested
/// command. `watch --verbose` and the `--json` payload keep this detail.
pub fn render_verbose(v: &Value) -> String {
    render_with(v, true)
}

/// One event, compact by default and complete under [`render_verbose`].
pub fn render_with(v: &Value, verbose: bool) -> String {
    if v["missed"] == true {
        format!("{MISSED_HEADING}\n{}", render_event(v, verbose))
    } else {
        render_event(v, verbose)
    }
}

/// One event body, without the heading a whole missed batch shares.
fn render_event(v: &Value, verbose: bool) -> String {
    if v["event"] == ROUND_EVENT {
        return v["content"]
            .as_str()
            .map(str::to_string)
            .unwrap_or_else(|| {
                v["workers"]
                    .as_array()
                    .map(|workers| {
                        workers
                            .iter()
                            .map(render_round_line)
                            .collect::<Vec<_>>()
                            .join("\n")
                    })
                    .unwrap_or_default()
            });
    }
    if verbose {
        return render_event_verbose(v);
    }
    let text = |key: &str| v[key].as_str().unwrap_or("unknown");
    // A registry row carries no branch, but every worker commits to its own
    // `worker-<id>` branch, so name that instead of admitting we do not know.
    let branch = v["branch"]
        .as_str()
        .map_or_else(|| format!("worker-{}", text("worker_id")), str::to_string);
    let event = text("event");
    let mut out = String::new();
    if let Some(dropped) = v["dropped_events"].as_u64().filter(|n| *n > 0) {
        out.push_str(&format!(
            "{dropped} older events dropped (backlog limit 100). | "
        ));
    }
    // The headline carries what decides the next move: the outcome, its
    // verification, its size, and a one-line summary.
    let verified = v["verified"]
        .as_bool()
        .map_or_else(String::new, |ok| format!(" | Verified: {ok}"));
    let diff = matches!(event, "completed" | "failed" | "exhausted").then(|| {
        format!(
            " | Diff: {} files, +{} -{}",
            v["diff_stat"]["files"], v["diff_stat"]["insertions"], v["diff_stat"]["deletions"]
        )
    });
    // A completed worker's headline is its report's `done:` line: the summary
    // the harness derives from the last chat message is only the fallback.
    let summary = match event {
        "completed" => report_done(v)
            .or_else(|| one_line(v["summary"].as_str()))
            .unwrap_or_else(|| "done".to_string()),
        "failed" => one_line(v["error"].as_str()).unwrap_or_else(|| "failed".to_string()),
        "exhausted" => one_line(v["summary"].as_str())
            .unwrap_or_else(|| "stopped: turn budget exhausted".to_string()),
        "needs_input" => {
            one_line(v["question"].as_str()).unwrap_or_else(|| "needs input".to_string())
        }
        _ => {
            let moved = moved_counters(v);
            if moved.is_empty() {
                format!("no step for {}s", v["time_since_last_step"])
            } else {
                format!(
                    "no step for {}s | {}",
                    v["time_since_last_step"],
                    moved.join(", ")
                )
            }
        }
    };
    out.push_str(&format!(
        "{} {}{}{} | {}\n",
        text("worker_id"),
        event,
        verified,
        diff.as_deref().unwrap_or(""),
        summary
    ));
    // The per-file split of the completion diff: the top files by churn, then
    // how many were left out. One line, so the body stays within five.
    if event == "completed" {
        let files = crate::pool::churn_line(&per_file_stats(v));
        if !files.is_empty() {
            out.push_str(&format!("files: {files}\n"));
        }
        if let Some(risks) = report_risks(v) {
            out.push_str(&format!("risks: {risks}\n"));
        }
    }
    out.push_str(&format!(
        "branch {} | step {}/{} | elapsed {}s | {}\n",
        branch,
        v["step"],
        v["max_turns"],
        v["elapsed"],
        text("task")
    ));
    if let Some(command) = primary_command(v) {
        out.push_str(&format!("$ {command}\n"));
    }
    out.trim_end().to_string()
}

/// The per-file diff a payload carries, as [`crate::pool::FileStat`]s.
fn per_file_stats(v: &Value) -> Vec<crate::pool::FileStat> {
    serde_json::from_value(v["per_file"].clone()).unwrap_or_default()
}

/// The `done:` line of a payload's report, when it wrote one.
fn report_done(v: &Value) -> Option<String> {
    one_line(v["report"]["done"].as_str())
}

/// The report's `risks:` line, unless it says there are none.
fn report_risks(v: &Value) -> Option<String> {
    let risks = one_line(v["report"]["risks"].as_str())?;
    (!risks.eq_ignore_ascii_case("none")).then_some(risks)
}

/// The health counters that moved since the previous snapshot, as `key=delta`.
///
/// A compact stall event has room for one line, so it names the counters that
/// fired rather than every counter the run ever moved.
fn moved_counters(v: &Value) -> Vec<String> {
    v["moved_counters"]
        .as_array()
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

/// First non-empty line of a payload text field, bounded for a one-line event.
fn one_line(value: Option<&str>) -> Option<String> {
    let line = value?.lines().next()?.trim();
    (!line.is_empty()).then(|| clamp_string(line, 200))
}

/// The one command a compact event suggests: the steer that resumes the
/// worker. The diff, verify and merge commands stay in the verbose body.
fn primary_command(v: &Value) -> Option<String> {
    commands(v).into_iter().find(|cmd| cmd.contains(" steer "))
}

/// The full event body, as `watch --verbose` and the JSON payload show it.
fn render_event_verbose(v: &Value) -> String {
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
        "completed" | "failed" | "exhausted" => {
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
            for key in ["done", "files", "tests", "risks"] {
                if let Some(value) = one_line(v["report"][key].as_str()) {
                    out.push_str(&format!("{key}: {value}\n"));
                }
            }
            let files = crate::pool::churn_line(&per_file_stats(v));
            if !files.is_empty() {
                out.push_str(&format!("files_stat: {files}\n"));
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
fn print_events(events: &[Value], json_output: bool, follow: bool, verbose: bool) -> Result<()> {
    print_events_to(&mut std::io::stdout(), events, json_output, follow, verbose)
}

/// [`print_events`] against an explicit sink, so the batch layout is testable.
fn print_events_to(
    out: &mut impl Write,
    events: &[Value],
    json_output: bool,
    follow: bool,
    verbose: bool,
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
        let body = render_event(event, verbose);
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
        Some("completed" | "failed" | "exhausted" | "stopped" | "interrupted")
    )
}

/// Whether `view` names a group of `groups`; an empty selection is every
/// group, which is what an `--all` watch without a `--group` flag selects.
fn in_groups(view: &Value, groups: &BTreeSet<String>) -> bool {
    groups.is_empty()
        || view["group"]
            .as_str()
            .is_some_and(|group| groups.contains(group))
}

/// Whether `view` is one of the workers this watch follows: its ids, when the
/// caller named any, and its group, when the caller named any.
pub fn matches(view: &Value, ids: &BTreeSet<String>, groups: &BTreeSet<String>) -> bool {
    (ids.is_empty()
        || view["worker_id"]
            .as_str()
            .is_some_and(|id| ids.contains(id)))
        && in_groups(view, groups)
}

/// The event kind a `--all` round emits: one consolidated event per round.
pub const ROUND_EVENT: &str = "round";

/// How long a `--all` round tolerates no step before one worker's stall is
/// worth the orchestrator's attention. A round is the consolidator's business,
/// so ordinary stalls stay out of its events.
pub const ROUND_STALL_SECS: u64 = 1200;

/// How many seconds a `--all` round worker has gone without a step.
///
/// A worker waiting for a build slot, or running a command, is not inactive:
/// same rules as [`select_event`], so a long gate never reads as a stall.
fn round_idle(v: &Value, now: u64) -> u64 {
    if v["waiting_for_slot"].is_number() || in_flight(v, v["status"].as_str().unwrap_or("running"))
    {
        return 0;
    }
    now.saturating_sub(v["last_step_at"].as_u64().unwrap_or(now))
}

/// The outcome label one worker carries in a `--all` round line.
fn round_outcome(v: &Value, now: u64) -> &'static str {
    match v["status"].as_str() {
        Some("completed") => "completed",
        Some("failed") => "failed",
        Some("exhausted") => "exhausted",
        Some("stopped") => "stopped",
        Some("interrupted") => "interrupted",
        Some("paused") => "needs_input",
        Some("running" | "reviewing") if round_idle(v, now) >= ROUND_STALL_SECS => "stalled",
        _ => "running",
    }
}

/// The `done:` headline a round line shows: the report's line, a summary, the
/// failure reason, or the escalated question.
fn round_done(v: &Value) -> Option<String> {
    match v["status"].as_str() {
        Some("completed") => report_done(v).or_else(|| one_line(v["summary"].as_str())),
        Some("failed") => one_line(v["error"].as_str()),
        Some("exhausted") => one_line(v["summary"].as_str()),
        Some("paused") => one_line(v["question"].as_str()),
        _ => None,
    }
}

/// One worker as a structured line of the consolidated round event.
fn round_line(v: &Value, now: u64) -> Value {
    json!({
        "worker_id": v["worker_id"].clone(),
        "outcome": round_outcome(v, now),
        "status": v["status"].clone(),
        "verified": v["verified"].clone(),
        "done": round_done(v),
        "question": v["question"].clone(),
        "branch": v["branch"].clone(),
        "time_since_last_step": round_idle(v, now),
    })
}

/// One worker's compact line in a `--all` round: id, outcome, verification and
/// the report's `done:` line (or the reason it needs the orchestrator).
pub fn render_round_line(w: &Value) -> String {
    let id = w["worker_id"].as_str().unwrap_or("?");
    let outcome = w["outcome"].as_str().unwrap_or("?");
    let verified = match w["verified"].as_bool() {
        Some(true) => "verified:yes",
        Some(false) => "verified:no",
        None => "verified:-",
    };
    match w["done"].as_str().filter(|done| !done.is_empty()) {
        Some(done) => format!("{id} {outcome} {verified} {done}"),
        None if outcome == "stalled" => format!(
            "{id} {outcome} {verified} no step for {}s",
            w["time_since_last_step"]
        ),
        None => format!("{id} {outcome} {verified}"),
    }
}

/// The consolidated `--all` event for the first round worth reporting, or
/// `None` while every round keeps waiting.
///
/// Each round is its own unit: it keeps waiting until all of its workers have
/// stopped (completed, failed, exhausted, killed or interrupted), and returns
/// early when one needs input, has failed, or (only past [`ROUND_STALL_SECS`])
/// has gone quiet long enough to matter. `fresh` reports whether a worker has
/// an event the caller has not acknowledged, so a reported round never replays.
///
/// Rounds are ranked, not merged: a stopped round outranks one that only needs
/// attention, and ties go to the first group by name so the answer is the same
/// on every poll. The event carries that one round's workers, which is what
/// lets an orchestrator with several rounds running read the result of the
/// first one to land.
pub fn round_event(
    current: &Snapshot,
    ids: &BTreeSet<String>,
    groups: &BTreeSet<String>,
    now: u64,
    fresh: impl Fn(&str) -> bool,
    allowed: impl Fn(&Value) -> bool,
) -> Option<Value> {
    let mut best: Option<Round> = None;
    for round in rounds(current, ids, groups, &allowed) {
        let Some(all_stopped) = round_ready(&round.workers, now, &fresh) else {
            continue;
        };
        // A finished round outranks an early return: it is the answer, while a
        // question only interrupts the wait when nothing else finished. Ties go
        // to the first group by name, so every poll ranks the rounds alike.
        let outranks = match &best {
            None => true,
            Some(best) => all_stopped && !best.all_stopped || best.group > round.group,
        };
        if outranks {
            best = Some(Round {
                all_stopped,
                group: round.group,
                workers: round.workers,
            });
        }
    }
    let best = best?;
    Some(round_payload(
        &best.workers,
        &best.group,
        now,
        best.all_stopped,
    ))
}

/// The caller's rounds, each one name and the workers of that group alone.
///
/// Grouping is what keeps the rounds apart: a stopped round can never be held
/// back by a sibling round that is still running. An empty group selection
/// means every live group of the caller, which is what `--all` without a
/// `--group` flag asks for.
fn rounds<'a>(
    current: &'a Snapshot,
    ids: &BTreeSet<String>,
    groups: &BTreeSet<String>,
    allowed: &impl Fn(&Value) -> bool,
) -> Vec<Round<'a>> {
    let selected = |v: &Value| {
        allowed(v)
            && matches(v, ids, groups)
            && v["steered_by_consolidator"] != true
            && v["question_for_consolidator"] != true
    };
    let mut by_group: BTreeMap<String, Vec<&Value>> = BTreeMap::new();
    for view in current.values().filter(|v| selected(v)) {
        if let Some(group) = view["group"].as_str() {
            by_group.entry(group.to_string()).or_default().push(view);
        }
    }
    // A named group that holds no worker of the caller is not a round: it would
    // otherwise answer as an empty one that is instantly stopped.
    if !groups.is_empty() {
        by_group.retain(|group, workers| {
            !workers.is_empty()
                || current
                    .values()
                    .any(|v| allowed(v) && v["group"].as_str().is_some_and(|name| name == group))
        });
    }
    by_group
        .into_iter()
        .map(|(group, workers)| Round {
            all_stopped: false,
            group,
            workers,
        })
        .collect()
}

/// One group of the snapshot, with the workers the round is made of.
struct Round<'a> {
    all_stopped: bool,
    group: String,
    workers: Vec<&'a Value>,
}

/// Whether the round is over, or `None` while it keeps waiting.
fn round_ready(selected: &[&Value], now: u64, fresh: &impl Fn(&str) -> bool) -> Option<bool> {
    if selected.is_empty() {
        return None;
    }
    let mut all_stopped = true;
    let mut attention = false;
    let mut fresh_any = false;
    for v in selected {
        let id = v["worker_id"].as_str().unwrap_or("");
        let is_fresh = fresh(id);
        fresh_any |= is_fresh;
        match v["status"].as_str() {
            Some("completed" | "exhausted" | "stopped" | "interrupted") => {}
            Some("failed") => attention |= is_fresh,
            Some("paused") => {
                all_stopped = false;
                attention |= is_fresh;
            }
            Some("running" | "reviewing") => {
                all_stopped = false;
                if is_fresh && round_idle(v, now) >= ROUND_STALL_SECS {
                    attention = true;
                }
            }
            _ => all_stopped = false,
        }
    }
    if !fresh_any || (!attention && !all_stopped) {
        return None;
    }
    Some(all_stopped)
}

/// Assemble the round event from the selected worker views.
fn round_payload(selected: &[&Value], group: &str, now: u64, all_stopped: bool) -> Value {
    let mut workers: Vec<Value> = selected.iter().map(|v| round_line(v, now)).collect();
    workers.sort_by(|a, b| a["worker_id"].as_str().cmp(&b["worker_id"].as_str()));
    let content = workers
        .iter()
        .map(render_round_line)
        .collect::<Vec<_>>()
        .join("\n");
    json!({
        "worker_id": group,
        "event": ROUND_EVENT,
        "status": if all_stopped { "stopped" } else { "attention" },
        "group": group,
        "workers": workers,
        "content": content,
    })
}

/// A hub that predates `hub/watch` cannot stream events at all.
const NO_WATCH_NOTICE: &str =
    "[mini-swe] The running hub predates 'hub/watch'; falling back to registry polling.";

/// A hub whose `hub/watch` reply predates the fields this client reads.
const OLD_WATCH_REPLY_NOTICE: &str = "[mini-swe] The running hub's 'hub/watch' reply lacks fields this client needs; falling back to registry polling.";

/// A `hub/watch` reply carries the fields this client reads: a missing
/// `watching` is an older hub, not a genuine empty watch set.
fn watch_reply_has_fields(response: &Value) -> bool {
    response.get("watching").is_some_and(Value::is_array)
        && response.get("events").is_some_and(Value::is_array)
}

/// Read the registry the way `MINI_SWE_NO_DAEMON=1` does, announcing why.
async fn registry_fallback(
    opts: Options,
    json_output: bool,
    admin: bool,
    notice: &str,
) -> Result<i32> {
    eprintln!("{notice}");
    polling(opts, json_output, admin).await
}

/// The reconnect budget this watch spends, read before the first dial so the
/// [`crate::hub::client::RECONNECT_DEADLINE_ENV`] override applies to the whole
/// watch and not to one attempt.
fn reconnect_budget() -> Duration {
    reconnect_deadline()
}

/// End a watch with no further event to show. Exit 0 is reserved for a watch
/// that already printed one; otherwise name why and exit non-zero.
fn end_watch(printed_event: bool, watched_any: bool) -> i32 {
    if printed_event {
        return 0;
    }
    if watched_any {
        println!("no event");
        2
    } else {
        println!("nothing to watch");
        3
    }
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
            return registry_fallback(opts, json_output, admin, NO_WATCH_NOTICE).await;
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
    let mut printed_event = false;
    let reconnects = reconnect_budget();
    'watch: loop {
        let response = match client
            .watch_snapshot(&ids, &opts.groups, initial, opts.all)
            .await
        {
            Ok(value) => value,
            Err(error) if daemon_went_away(&error) => {
                // The daemon restarted under this watch: follow it, and ask for
                // the missed events again on the new connection.
                client = follow(reconnects, admin).await?;
                // Ask the replacement for the events the daemon took with it.
                initial = true;
                ids = opts.ids.clone();
                continue;
            }
            Err(error) if error.to_string().contains("a watch is already running") => {
                println!("{error}");
                return Ok(5);
            }
            Err(error) if error.to_string().contains("belongs to agent") => {
                println!("{error}");
                return Ok(4);
            }
            Err(error) if error.to_string().contains("Method not found") => {
                return registry_fallback(opts, json_output, admin, NO_WATCH_NOTICE).await;
            }
            Err(error) => return Err(error),
        };
        if !watch_reply_has_fields(&response) {
            return registry_fallback(opts, json_output, admin, OLD_WATCH_REPLY_NOTICE).await;
        }
        let watching: BTreeSet<String> = response["watching"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect();
        // A `--all` round must keep every selected id until it reports, so its
        // terminal workers are never pruned away mid-round.
        if explicit && !opts.all {
            ids = watching.clone();
        }
        if !watching.is_empty() {
            watched_any = true;
        }
        let events = response["events"].as_array().cloned().unwrap_or_default();
        if initial && explicit && ids.is_empty() && events.is_empty() {
            return Ok(end_watch(false, false));
        }
        print_events(&events, json_output, opts.follow, opts.verbose)?;
        printed_event |= !events.is_empty();
        for event in &events {
            if let Err(error) = client
                .watch_ack(event["sequence"].as_u64().unwrap_or(0))
                .await
            {
                if daemon_went_away(&error) {
                    client = follow(reconnects, admin).await?;
                    initial = true;
                    ids = opts.ids.clone();
                    continue 'watch;
                }
                return Err(error);
            }
        }
        // Every missed event came back in this one reply, so a non-following
        // caller leaves as soon as it has been caught up.
        if !events.is_empty() && !opts.follow {
            return Ok(0);
        }
        initial = false;
        if watching.is_empty() && (explicit || watched_any) {
            return Ok(end_watch(printed_event, watched_any));
        }
        let wait = match opts.timeout {
            Some(timeout) => {
                let Some(left) = timeout.checked_sub(started.elapsed()) else {
                    return Ok(end_watch(printed_event, watched_any));
                };
                left.min(Duration::from_secs(1))
            }
            None => Duration::from_secs(1),
        };
        // Notifications wake the consumer promptly. The next snapshot request
        // repairs channel overflow from the owner's bounded, unacknowledged backlog.
        if let Ok(result) = tokio::time::timeout(wait, client.next_watch_notification()).await {
            match result {
                Ok(()) => {}
                Err(error) if daemon_went_away(&error) => {
                    client = follow(reconnects, admin).await?;
                    initial = true;
                    ids = opts.ids.clone();
                    continue;
                }
                Err(error) => return Err(error),
            }
        }
    }
}

/// Follow a daemon that went away, or give up once the budget is spent.
///
/// The budget belongs to the whole watch, not to one reconnect: a handover
/// spends a few fast failures on the replacement coming up, and a hub that never
/// comes back ends the watch with an explanation instead of an indefinitely
/// retried dial. Every dial re-announces the same identity, and the caller marks
/// its next snapshot `initial`, so the events the watch missed while the daemon
/// was down are replayed rather than lost.
async fn follow(budget: Duration, admin: bool) -> Result<crate::hub::HubClient> {
    crate::hub::client::reconnect_following(budget, || {
        crate::hub::HubClient::connect_as_admin(admin)
    })
    .await
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
    // One signature per worker already folded into a `--all` round, so the
    // same round is never emitted twice.
    let mut round_reported = Snapshot::new();
    let mut initial = true;
    let mut watched_any = false;
    let mut printed_event = false;
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
            // dispatch: a no-arg watch must not replay it. A `--all` round
            // must still report such a worker: it is the round's result.
            if !explicit && !opts.all {
                ignored = current
                    .iter()
                    .filter(|(_, v)| {
                        (admin || v["owner"] == owner)
                            && matches(v, &opts.ids, &opts.groups)
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
                    && matches(v, &opts.ids, &opts.groups)
                {
                    ids.insert(id.clone());
                }
            }
        }
        current.retain(|id, v| {
            ids.contains(id)
                && (admin || v["owner"] == owner)
                && matches(v, &opts.ids, &opts.groups)
        });
        if opts.all {
            let signature = |v: &Value| {
                json!({
                    "status": v["status"].clone(),
                    "revision": v["revision"].clone(),
                    "step": v["step"].clone(),
                    "question": v["question"].clone(),
                })
            };
            // The polling set is already ownership-filtered, so every
            // view here is the caller's own.
            let event = round_event(
                &current,
                &ids,
                &opts.groups,
                now,
                |id| {
                    current
                        .get(id)
                        .is_some_and(|v| round_reported.get(id) != Some(&signature(v)))
                },
                |_| true,
            );
            if let Some(event) = event {
                for (id, view) in &current {
                    if matches(view, &ids, &opts.groups) {
                        round_reported.insert(id.clone(), signature(view));
                    }
                }
                print_events(&[event], json_output, opts.follow, opts.verbose)?;
                printed_event = true;
                if !opts.follow {
                    return Ok(0);
                }
            }
            if !current.is_empty() {
                watched_any = true;
            }
            previous = current;
            if previous.is_empty() && (explicit || watched_any) {
                return Ok(end_watch(printed_event, watched_any));
            }
            if opts
                .timeout
                .is_some_and(|timeout| started.elapsed() >= timeout)
            {
                return Ok(end_watch(printed_event, watched_any));
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
            continue;
        }
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
        print_events(&events, json_output, opts.follow, opts.verbose)?;
        printed_event |= !events.is_empty();
        if !events.is_empty() && !opts.follow {
            return Ok(0);
        }
        if !current.is_empty() {
            watched_any = true;
        }
        ids.retain(|id| current.get(id).is_some_and(|v| !terminal(v)));
        if ids.is_empty() && (explicit || watched_any) {
            return Ok(end_watch(printed_event, watched_any));
        }
        previous = current;
        if opts
            .timeout
            .is_some_and(|timeout| started.elapsed() >= timeout)
        {
            return Ok(end_watch(printed_event, watched_any));
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
            out.contains("mini-swe-mcp steer") && !out.contains("git diff"),
            "the compact body suggests one interaction: {out}"
        );
        assert!(
            render_verbose(&done).contains("git diff"),
            "verbose keeps every command"
        );
        let stalled = select_event(
            &state("running", 2, repeated, 100, 10),
            Some(&state("running", 2, metrics, 100, 10)),
            110,
        )
        .unwrap();
        let text = render(&stalled);
        assert!(
            text.contains("mini-swe-mcp steer") && !text.contains("mini-swe-mcp kill"),
            "the compact body suggests one interaction: {text}"
        );
        assert!(
            render_verbose(&stalled).contains("mini-swe-mcp kill"),
            "verbose keeps kill"
        );
    }

    /// The default body is the three lines an orchestrator has to read: the
    /// headline, the worker's progress, and one command. The long guidance and
    /// the remaining commands wait for `--verbose`.
    #[test]
    fn compact_events_carry_a_headline_a_one_line_summary_and_one_command() {
        let done = select_event(
            &state("completed", 3, WorkerMetrics::default(), 100, 10),
            None,
            110,
        )
        .unwrap();
        let compact = render(&done);
        assert_eq!(compact.lines().count(), 3, "{compact}");
        assert!(compact.contains("w completed"), "{compact}");
        assert!(compact.contains("Verified: true"), "{compact}");
        assert!(compact.contains("Diff: 0 files, +0 -0"), "{compact}");
        assert!(compact.contains("Done."), "{compact}");
        assert!(compact.contains("mini-swe-mcp steer"), "{compact}");
        assert!(!compact.contains("Review the diff"), "{compact}");

        let verbose = render_verbose(&done);
        assert!(verbose.contains("Review the diff"), "{verbose}");
        assert!(verbose.contains("git diff HEAD...'worker-w'"), "{verbose}");
        assert!(verbose.contains("Diff: 0 files, +0 -0"), "{verbose}");
    }

    /// A step whose command is still executing is not a stall: a long build
    /// or test gate past the idle threshold produces no event at all.
    #[test]
    fn a_command_still_running_is_not_a_stall() {
        let mut running = state("running", 2, WorkerMetrics::default(), 100, 10);
        running["command_started_at"] = json!(1150);
        assert!(
            select_event(&running, None, 1200).is_none(),
            "a command in flight must not be reported as stalled"
        );
        // Once the command returns, the idle clock decides again.
        assert_eq!(
            select_event(
                &state("running", 2, WorkerMetrics::default(), 100, 10),
                None,
                1200
            )
            .unwrap()["event"],
            "stalled"
        );
    }

    /// A command that outlived its budget becomes a background job and keeps
    /// running: the worker is waiting on it, not idle. The conversion lands on
    /// the idle threshold, so reading it as inactivity would fire a stall at
    /// the exact moment the worker did the right thing. Idle after the job is
    /// gone still stalls.
    #[test]
    fn a_job_conversion_is_activity_not_a_stall() {
        let mut converted = state("running", 2, WorkerMetrics::default(), 100, 10);
        // The 600 s step timeout put the command into job 1 at t=700; the
        // executor's mark cleared with it, but the job is still running.
        converted["jobs"] = json!(["job 1: cargo test (running, 3s)"]);
        let now = 701; // 601 s since the last step, past the stall threshold.
        assert!(
            select_event(&converted, None, now).is_none(),
            "a step that ended in a job conversion must not be reported as stalled"
        );
        // The job is reaped and the worker never steps again: the same view
        // without a live job is a genuine stall.
        converted["jobs"] = json!(["job 1: cargo test (finished, 480s)"]);
        assert_eq!(
            select_event(&converted, None, now).unwrap()["event"],
            "stalled",
            "an idle worker whose job has ended is stalled"
        );
        // A worker idle for the whole threshold after the conversion stalls too.
        assert_eq!(
            select_event(&converted, None, now + 600).unwrap()["event"],
            "stalled"
        );
    }

    /// `--json` is the complete machine-readable contract: it must not shrink
    /// when the human-facing body does.
    #[test]
    fn json_output_stays_complete_and_ignores_verbose() {
        let event = select_event(
            &state("completed", 3, WorkerMetrics::default(), 100, 10),
            None,
            110,
        )
        .unwrap();
        let mut plain = Vec::new();
        print_events_to(&mut plain, std::slice::from_ref(&event), true, false, false).unwrap();
        let mut verbose = Vec::new();
        print_events_to(&mut verbose, &[event], true, false, true).unwrap();
        assert_eq!(plain, verbose, "the JSON body is verbose-independent");
        let printed: Value = serde_json::from_slice(&plain).unwrap();
        assert!(printed["commands"].is_array(), "{printed}");
        assert!(
            printed["next_step"]
                .as_str()
                .unwrap_or("")
                .contains("Review"),
            "{printed}"
        );
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
        print_events_to(&mut out, &batch, false, false, false).expect("print");
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
        print_events_to(
            &mut out,
            &[missed("w-old", "Late."), live],
            false,
            false,
            false,
        )
        .expect("print");
        let text = String::from_utf8(out).expect("utf8");
        assert_eq!(text.matches(MISSED_HEADING).count(), 1, "{text}");
        assert!(text.contains("Late.") && text.contains("go on?"), "{text}");
    }
}
