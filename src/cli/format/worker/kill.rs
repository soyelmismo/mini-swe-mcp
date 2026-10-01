pub fn format_kill(val: &serde_json::Value) -> String {
    let wid = val.get("worker_id").and_then(|v| v.as_str()).unwrap_or("");
    let killed = val.get("killed").and_then(|v| v.as_bool()).unwrap_or(false);
    if killed {
        format!("✓ Worker {wid} terminated.")
    } else {
        format!("Worker {wid} was not running.")
    }
}
