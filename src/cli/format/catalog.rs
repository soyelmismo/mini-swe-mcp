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
        if let Some(pid) = state_obj.and_then(|s| s.get("pid")).and_then(|v| v.as_u64()) {
            details.push(format!("pid: {pid}"));
        }
        if !model.is_empty() {
            details.push(format!("model: {model}"));
        }
        if let Some(turns) = state_obj.and_then(|s| s.get("turns")).and_then(|v| v.as_u64()) {
            details.push(format!("turns: {turns}"));
        } else if let Some(step) = state_obj.and_then(|s| s.get("step")).and_then(|v| v.as_u64()) {
            details.push(format!("step: {step}"));
        }
        if let Some(op) = state_obj.and_then(|s| s.get("last_command")).and_then(|v| v.as_str())
            && !op.is_empty() && op != "initializing"
        {
            details.push(format!("op: {op}"));
        }
        if let Some(err) = state_obj.and_then(|s| s.get("error")).and_then(|v| v.as_str()) {
            details.push(format!("error: {err}"));
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
        let out = format_manifest(&v(
            r#"{"default_model":"z","models":{
                 "zeta":{"id":"z-model","temperature":0.0,"max_turns":7,"role":"coder"},
                 "alpha":{"id":"a-model","temperature":0.75}}}"#,
        ));
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
    fn test_format_list_renders_worker_rows_and_previews() {
        let out = format_list(&v(
            r#"{"workers":[{
                 "id":"w1","model":"m","group":"g","task":"a long task description that goes well past the sixty character preview limit",
                 "state":{"status":"Running","pid":42,"turns":3,"last_command":"cargo test","error":"boom"}}]}"#,
        ));
        assert!(out.starts_with("Workers (1):\n"));
        assert!(out.contains("- w1 [Running] (group: g, pid: 42, model: m, turns: 3, op: cargo test, error: boom)"));
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
