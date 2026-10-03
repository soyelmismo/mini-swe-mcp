//! Plain-text renderers for the system-catalog verbs.
//!
//! Formatters behind `manifest`, `list` and `prune` — the actions that describe
//! the *installation* rather than one worker: the model catalog and its
//! defaults, the worker table, and the housekeeping confirmation.
//!
//! Like every formatter in this package they are pure functions over
//! [`serde_json::Value`] with no I/O, which is what makes them unit-testable
//! independently of the dispatch layer.

pub fn format_manifest(val: &serde_json::Value) -> String {
    let mut out = String::new();
    if let Some(default_model) = val.get("default_model").and_then(|v| v.as_str()) {
        out.push_str(&format!("Default model: {default_model}\n\n"));
    }
    out.push_str("Models:\n");
    if let Some(models) = val.get("models").and_then(|v| v.as_object()) {
        let mut entries: Vec<(&String, &serde_json::Value)> = models.iter().collect();
        entries.sort_by_key(|(k, _)| (*k).clone());

        for (name, def) in entries {
            let id = def.get("id").and_then(|v| v.as_str()).unwrap_or(name);
            let mut meta = Vec::new();
            meta.push(format!("id: {id}"));
            if let Some(temp) = def.get("temperature").and_then(|v| v.as_f64()) {
                let temp_str = format!("{temp:.2}");
                let temp_clean = temp_str.trim_end_matches('0').trim_end_matches('.');
                meta.push(format!("temp: {temp_clean}"));
            }
            if let Some(turns) = def.get("max_turns").and_then(|v| v.as_u64()) {
                meta.push(format!("max turns: {turns}"));
            }
            out.push_str(&format!("  - {} ({})\n", name, meta.join(", ")));
            if let Some(role) = def.get("role").and_then(|v| v.as_str()) {
                out.push_str(&format!("    Role: {role}\n"));
            }
        }
    }
    push_review_modes(&mut out, val);
    out.trim_end().to_string()
}

/// The available review modes: the built-ins plus the manifest-declared ones.
///
/// A declared mode names its default reviewer when one is set; the built-ins
/// `quality` and `security` are always available even when the manifest
/// declares none, and a declared entry of the same name overrides the
/// built-in prompt.
fn push_review_modes(out: &mut String, val: &serde_json::Value) {
    let declared = val
        .get("review_modes")
        .and_then(|v| v.as_object())
        .cloned()
        .unwrap_or_default();
    out.push_str("\n\nReview modes:\n");
    for builtin in ["quality", "security"] {
        if let Some(def) = declared.get(builtin) {
            out.push_str(&format!("  - {builtin}{}\n", review_mode_suffix(def)));
        } else {
            out.push_str(&format!("  - {builtin} (built-in)\n"));
        }
    }
    let mut extra: Vec<&String> = declared
        .keys()
        .filter(|name| *name != "quality" && *name != "security")
        .collect();
    extra.sort();
    for name in extra {
        let def = &declared[name];
        out.push_str(&format!("  - {name}{}\n", review_mode_suffix(def)));
    }
}

/// The parenthetical for one declared review mode: its default reviewer when
/// one is set, and its trigger count.
fn review_mode_suffix(def: &serde_json::Value) -> String {
    let model = def
        .get("model")
        .and_then(|v| v.as_str())
        .filter(|m| !m.trim().is_empty());
    let triggers = def
        .get("triggers")
        .and_then(|v| v.as_array())
        .map(|a| a.len())
        .unwrap_or(0);
    match (model, triggers) {
        (Some(model), 0) => format!(" (model: {model})"),
        (Some(model), n) => format!(" (model: {model}, triggers: {n})"),
        (None, 0) => String::new(),
        (None, n) => format!(" (triggers: {n})"),
    }
}

/// The retired workers' final reports, oldest first.
///
/// One block per line: who ran, what they were asked, how they left and what
/// they reported. An empty archive says so rather than printing an empty table,
/// because "nothing retired yet" and "the archive is empty for another reason"
/// are different things and the caller can only tell them apart here.
pub fn format_archive(val: &serde_json::Value) -> String {
    let empty = Vec::new();
    let entries = val
        .get("entries")
        .and_then(|v| v.as_array())
        .unwrap_or(&empty);
    if entries.is_empty() {
        return "No retired worker reports yet (archive is empty).".to_string();
    }
    let mut out = format!("Retired worker reports ({}):\n", entries.len());
    for entry in entries {
        let field = |key: &str| entry.get(key).and_then(|v| v.as_str()).unwrap_or("");
        let id = field("worker_id");
        out.push_str(&format!("\n{id} ({})", field("reason")));
        let group = field("group");
        if !group.is_empty() {
            out.push_str(&format!(" [group {group}]"));
        }
        if let Some(at) = entry.get("retired_at").and_then(|v| v.as_u64()) {
            out.push_str(&format!(" at {at}"));
        }
        out.push('\n');
        let task = field("task");
        if !task.is_empty() {
            out.push_str(&format!("  task: {task}\n"));
        }
        let mut meta = vec![format!("status: {}", field("status"))];
        match entry.get("verified") {
            Some(serde_json::Value::Bool(true)) => meta.push("verified: yes".to_string()),
            Some(serde_json::Value::Bool(false)) => meta.push("verified: no".to_string()),
            _ => {}
        }
        if let Some(commit) = entry.get("commit").and_then(|v| v.as_str())
            && !commit.is_empty()
        {
            meta.push(format!("commit: {commit}"));
        }
        out.push_str(&format!("  {}\n", meta.join(" | ")));
        // The four REPORT fields, in the order the worker wrote them, with the
        // label the worker used so a reader can match it to the prompt.
        if let Some(report) = entry.get("report").filter(|r| r.is_object()) {
            for key in ["done", "files", "tests", "risks"] {
                let value = report.get(key).and_then(|v| v.as_str()).unwrap_or("");
                if !value.is_empty() {
                    out.push_str(&format!("  {key}: {value}\n"));
                }
            }
        } else {
            out.push_str("  report: none recorded\n");
        }
    }
    out.trim_end().to_string()
}

pub fn format_list(val: &serde_json::Value) -> String {
    let empty_vec = Vec::new();
    let workers = val
        .get("workers")
        .and_then(|v| v.as_array())
        .unwrap_or(&empty_vec);

    if workers.is_empty() {
        return "No active or recent workers found.".to_string();
    }

    let mut out = format!("Workers ({}):\n", workers.len());
    for w in workers {
        let id = w.get("id").and_then(|v| v.as_str()).unwrap_or("unknown");
        let model = w.get("model").and_then(|v| v.as_str()).unwrap_or("");
        let group = w.get("group").and_then(|v| v.as_str()).unwrap_or("default");
        let state_obj = w.get("state");
        let status = state_obj
            .and_then(|s| s.get("status"))
            .and_then(|v| v.as_str())
            .unwrap_or("Unknown");

        let mut details = Vec::new();
        if group != "default" {
            details.push(format!("group: {group}"));
        }
        if let Some(pid) = state_obj
            .and_then(|s| s.get("pid"))
            .and_then(|v| v.as_u64())
        {
            details.push(format!("pid: {pid}"));
        }
        if !model.is_empty() {
            details.push(format!("model: {model}"));
        }
        if let Some(turns) = state_obj
            .and_then(|s| s.get("turns"))
            .and_then(|v| v.as_u64())
        {
            details.push(format!("turns: {turns}"));
        } else if let Some(step) = state_obj
            .and_then(|s| s.get("step"))
            .and_then(|v| v.as_u64())
        {
            details.push(format!("step: {step}"));
        }
        if let Some(op) = state_obj
            .and_then(|s| s.get("last_command"))
            .and_then(|v| v.as_str())
            && !op.is_empty()
            && op != "initializing"
        {
            details.push(format!("op: {op}"));
        }
        if let Some(err) = state_obj
            .and_then(|s| s.get("error"))
            .and_then(|v| v.as_str())
        {
            details.push(format!("error: {err}"));
        }
        if let Some(approved) = w.get("approved").filter(|v| !v.is_null()) {
            match approved.get("note").and_then(|v| v.as_str()) {
                Some(note) if !note.is_empty() => details.push(format!("approved: {note}")),
                _ => details.push("approved".to_string()),
            }
        }

        let detail_str = if details.is_empty() {
            String::new()
        } else {
            format!(" ({})", details.join(", "))
        };

        out.push_str(&format!("  - {id} [{status}]{detail_str}\n"));
        if let Some(task) = w.get("task").and_then(|v| v.as_str()) {
            let task_preview = if task.len() > 60 {
                let cut = task.floor_char_boundary(57);
                format!("{}...", &task[..cut])
            } else {
                task.to_string()
            };
            out.push_str(&format!("    Task: {task_preview}\n"));
        }
    }
    out.trim_end().to_string()
}

pub fn format_prune(val: &serde_json::Value) -> String {
    let msg = val
        .get("message")
        .and_then(|v| v.as_str())
        .unwrap_or("Stale worktrees and orphaned worker branches pruned");
    format!("✓ {msg}.")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(s: &str) -> serde_json::Value {
        serde_json::from_str(s).expect("fixture must be valid JSON")
    }

    #[test]
    fn test_format_manifest_lists_models_sorted_with_metadata() {
        let out = format_manifest(&v(r#"{"default_model":"z","models":{
                 "zeta":{"id":"z-model","temperature":0.0,"max_turns":7,"role":"coder"},
                 "alpha":{"id":"a-model","temperature":0.75}}}"#));
        assert!(out.starts_with("Default model: z\n\nModels:\n"));
        // Sorted by alias, not by insertion order.
        let alpha = out.find("- alpha").expect("alpha row");
        let zeta = out.find("- zeta").expect("zeta row");
        assert!(alpha < zeta, "models must be sorted by name: {out}");
        assert!(out.contains("  - alpha (id: a-model, temp: 0.75)"));
        assert!(out.contains("  - zeta (id: z-model, temp: 0, max turns: 7)"));
        assert!(out.contains("    Role: coder"));
    }

    #[test]
    fn test_format_manifest_lists_review_modes_with_defaults_and_triggers() {
        let out = format_manifest(&v(r#"{"default_model":"z","models":{"a":{"id":"a-model"}},
                 "review_modes":{"perf":{"checklist":"Check.","model":"nerd","triggers":["src/hot/**"]},"style":{"checklist":"Names."}}}"#));
        assert!(out.contains("Review modes:"), "modes are listed: {out}");
        assert!(
            out.contains("- quality (built-in)"),
            "built-ins are listed: {out}"
        );
        assert!(
            out.contains("- security (built-in)"),
            "built-ins are listed: {out}"
        );
        assert!(
            out.contains("- perf (model: nerd, triggers: 1)"),
            "default model and trigger count are shown: {out}"
        );
        assert!(
            out.contains("- style"),
            "a mode without a model is listed: {out}"
        );
        // Sorted by name after the built-ins.
        let perf = out.find("- perf").expect("perf row");
        let style = out.find("- style").expect("style row");
        assert!(perf < style, "modes must be sorted by name: {out}");
    }

    #[test]
    fn test_format_manifest_lists_only_builtins_without_declared_modes() {
        let out = format_manifest(&v(r#"{"models":{"a":{"id":"a-model"}}}"#));
        assert!(
            out.contains("Review modes:"),
            "the section is always present: {out}"
        );
        assert!(out.contains("- quality (built-in)"));
        assert!(out.contains("- security (built-in)"));
    }

    #[test]
    fn test_format_list_renders_worker_rows_and_previews() {
        let out = format_list(&v(r#"{"workers":[{
                 "id":"w1","model":"m","group":"g","task":"a long task description that goes well past the sixty character preview limit",
                 "state":{"status":"Running","pid":42,"turns":3,"last_command":"cargo test","error":"boom"}}]}"#));
        assert!(out.starts_with("Workers (1):\n"));
        assert!(out.contains(
            "- w1 [Running] (group: g, pid: 42, model: m, turns: 3, op: cargo test, error: boom)"
        ));
        assert!(out.contains("Task: a long task description that goes well past the "));
        assert!(out.contains("..."), "long tasks are elided: {out}");
    }

    #[test]
    fn test_format_list_omits_defaults_and_empty_state() {
        let out = format_list(&v(
            r#"{"workers":[{"id":"w1","state":{"status":"Queued","last_command":"initializing"}}]}"#,
        ));
        assert_eq!(out, "Workers (1):\n  - w1 [Queued]");
    }
}
