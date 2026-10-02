use crate::pool::clamp_string;

use super::*;

/// Byte budget a ready worker's task title is clamped to, so one round cannot
/// flood the summary with a wall of prose.
const READY_TITLE_BUDGET: usize = 120;

/// `consolidate`: the round's shape in one line, then the workers it will
/// integrate, then the watch hint.
///
/// A consolidator's dispatch payload carries the round manifest as rendered
/// text (the same block embedded in the consolidator's task) plus the group and
/// the worker id, so the plain-text view parses that block rather than dumping
/// the JSON an orchestrator would have to read by eye.
pub fn format_consolidate(val: &serde_json::Value) -> String {
    let wid = val.get("worker_id").and_then(|v| v.as_str()).unwrap_or("");
    let round = val.get("round").and_then(|v| v.as_str()).unwrap_or("");
    let group = val
        .get("group")
        .and_then(|v| v.as_str())
        .filter(|g| !g.is_empty())
        .or_else(|| round_group(round))
        .unwrap_or("");
    if val.get("amended").and_then(|v| v.as_bool()) == Some(true) {
        return format_amend(val);
    }
    let manifest = RoundText::parse(round);
    let mut out = format!(
        "Consolidator {wid} dispatched for group {group}: {} ready, {} not ready, {} interaction points",
        manifest.ready.len(),
        manifest.not_ready,
        manifest.interactions,
    );
    for worker in &manifest.ready {
        out.push_str(&format!(
            "\n  {}: {}",
            worker.id,
            clamp_string(&worker.title, READY_TITLE_BUDGET)
        ));
    }
    out.push_str(&watch_command_line(val));
    out
}

/// The `consolidate --set` answer: what the round will now run, so the caller
/// can see the setting it just changed instead of having to remember it.
///
/// The settings are quoted verbatim (including a cleared one, which is the
/// empty string) because the whole point of the verb is that the value stored
/// is the value that matters: an "unset" that used to look like the auto-detected
/// gate is indistinguishable from a typo unless it is shown.
fn format_amend(val: &serde_json::Value) -> String {
    let group = val.get("group").and_then(|v| v.as_str()).unwrap_or("");
    let gate = match val.get("verify").and_then(|v| v.as_str()) {
        Some("") => "none (auto-detect)".to_string(),
        Some(cmd) => cmd.to_string(),
        None => "unchanged".to_string(),
    };
    let model = match val.get("model").and_then(|v| v.as_str()) {
        Some(alias) => alias.to_string(),
        None => "unchanged".to_string(),
    };
    format!("Round {group} amended: consolidator model {model}, gate {gate}")
}

/// The group named by the manifest's header line, for a payload that carries
/// the round text without a sibling `group` key.
fn round_group(round: &str) -> Option<&str> {
    let rest = round
        .lines()
        .next()?
        .strip_prefix("ROUND MANIFEST group=")?;
    Some(rest.split(' ').next().unwrap_or(rest))
}

/// The part of the rendered manifest the summary needs: the ready workers to
/// list, and the counts the headline quotes.
#[derive(Default)]
struct RoundText {
    ready: Vec<ReadyWorker>,
    not_ready: usize,
    interactions: usize,
}

struct ReadyWorker {
    id: String,
    title: String,
}

impl RoundText {
    /// Read the manifest's three sections, each introduced by its own heading
    /// and holding two-space items (worker lines, or one interaction point per
    /// file). `files:` lines are indented four spaces and belong to the worker
    /// above them, so they are skipped.
    fn parse(round: &str) -> Self {
        let mut out = Self::default();
        let mut section = Section::Other;
        for line in round.lines() {
            if line.starts_with("ready (") {
                section = Section::Ready;
            } else if line.starts_with("not ready:") {
                section = Section::NotReady;
            } else if line.starts_with("interaction points") {
                section = Section::Interactions;
            } else if let Some((id, title)) = worker_line(line) {
                match section {
                    Section::Ready => out.ready.push(ReadyWorker { id, title }),
                    Section::NotReady => out.not_ready += 1,
                    Section::Other | Section::Interactions => {}
                }
            } else if section == Section::Interactions && interaction_line(line) {
                out.interactions += 1;
            }
        }
        out
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Section {
    Other,
    Ready,
    NotReady,
    Interactions,
}

/// The id and task title from one `  <id> <state> verified=<v> task="<task>"`
/// line, or `None` for a heading, a `files:` line or the `(none)` placeholder.
fn worker_line(line: &str) -> Option<(String, String)> {
    if !line.starts_with("  ") || line.starts_with("    ") {
        return None;
    }
    let (id, rest) = line.trim_start().split_once(' ')?;
    let title = rest.split_once(" task=\"")?.1.strip_suffix('"')?;
    Some((id.to_string(), title.to_string()))
}

/// A listed interaction point: a two-space line that is not the `(none)`
/// placeholder (the `files:` lines were already filtered out as worker lines).
fn interaction_line(line: &str) -> bool {
    line.starts_with("  ") && !line.starts_with("    ") && line.trim() != "(none)"
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const ROUND: &str = r#"ROUND MANIFEST group=round-1 base=master
ready (completed, branch not yet merged):
  w1 Completed verified=yes task="fix the parser"
    files: src/a.rs
  w2 Completed verified=no task="add the renderer"
    files: src/b.rs
not ready:
  w3 Running verified=unknown task="still going"
    files: src/c.rs
interaction points (touched by more than one worker):
  src/shared.rs: w1, w2
"#;

    fn payload(round: &str) -> serde_json::Value {
        json!({
            "worker_id": "w0",
            "group": "round-1",
            "status": "dispatched",
            "watch_command": "MINI_SWE_WATCH_TOKEN=t mini-swe-mcp watch",
            "round": round,
        })
    }

    /// The headline counts the round, and each ready worker gets one line with
    /// its task title.
    #[test]
    fn test_format_consolidate_summarises_the_round() {
        let out = format_consolidate(&payload(ROUND));
        assert!(
            out.starts_with(
                "Consolidator w0 dispatched for group round-1: 2 ready, 1 not ready, 1 interaction points\n"
            ),
            "{out}"
        );
        assert!(out.contains("\n  w1: fix the parser"), "{out}");
        assert!(out.contains("\n  w2: add the renderer"), "{out}");
        // Only ready workers are listed: a not-ready one is a count, not work.
        assert!(!out.contains("w3"), "{out}");
        assert!(
            out.contains("To wait for it: MINI_SWE_WATCH_TOKEN=t mini-swe-mcp watch"),
            "{out}"
        );
    }

    /// The group comes from the payload, and from the manifest header when the
    /// payload does not carry one.
    #[test]
    fn test_format_consolidate_falls_back_to_the_manifest_group() {
        let mut val = payload(ROUND);
        val.as_object_mut().unwrap().remove("group");
        assert!(
            format_consolidate(&val).contains("dispatched for group round-1:"),
            "{}",
            format_consolidate(&val)
        );
    }

    /// An empty round renders zeroed counts and no worker lines, so a payload
    /// without a manifest is still readable.
    #[test]
    fn test_format_consolidate_handles_an_empty_round() {
        let empty = format_consolidate(&json!({"worker_id": "w0", "group": "g"}));
        assert_eq!(
            empty,
            "Consolidator w0 dispatched for group g: 0 ready, 0 not ready, 0 interaction points"
        );
        // A manifest whose sections are all placeholders is not a crash either.
        let none = format_consolidate(&payload(
            "ROUND MANIFEST group=g base=master\nready (completed, branch not yet merged):\n  (none)\nnot ready:\n  (none)\ninteraction points (touched by more than one worker):\n  (none)\n",
        ));
        assert!(
            none.contains(": 0 ready, 0 not ready, 0 interaction points"),
            "{none}"
        );
    }

    /// A long task title is clamped so one worker cannot flood the summary.
    #[test]
    fn test_format_consolidate_clamps_a_long_task_title() {
        let long = "x".repeat(400);
        let round = format!(
            "ROUND MANIFEST group=g base=master\nready (completed, branch not yet merged):\n  w1 Completed verified=yes task=\"{long}\"\n"
        );
        let out = format_consolidate(&payload(&round));
        assert!(
            !out.contains(&long),
            "the whole title must not survive: {out}"
        );
        assert!(
            out.contains("truncated"),
            "the clamp must be visible: {out}"
        );
    }

    /// The parser reads what [`RoundManifest::render`] actually writes, so a
    /// change to either side of the format is caught here.
    #[test]
    fn test_format_consolidate_parses_the_rendered_manifest() {
        use crate::pool::{RoundManifest, RoundWorker};

        let worker = |id: &str, title: &str| RoundWorker {
            id: id.to_string(),
            state: "Completed".to_string(),
            verified: Some(true),
            task: title.to_string(),
            // A body the compact render must keep out of the summary.
            full_task: format!("{title}\nfull body"),
            files: vec!["src/a.rs".to_string()],
        };
        let manifest = RoundManifest {
            group: "round-9".to_string(),
            base_branch: Some("master".to_string()),
            ready: vec![worker("w1", "fix the parser"), worker("w2", "add the view")],
            not_ready: vec![worker("w3", "still running")],
            interaction_points: vec![("src/shared.rs".to_string(), vec!["w1".to_string()])],
        };
        let out = format_consolidate(&json!({
            "worker_id": "w0",
            "group": "round-9",
            "round": manifest.render(),
        }));
        assert!(
            out.starts_with(
                "Consolidator w0 dispatched for group round-9: 2 ready, 1 not ready, 1 interaction points\n"
            ),
            "{out}"
        );
        assert!(
            out.contains("\n  w1: fix the parser") && out.contains("\n  w2: add the view"),
            "{out}"
        );
    }
}
