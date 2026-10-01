use super::*;

/// Render the `logs` action: the bounded window plus the counters that make any
/// truncation visible instead of silent (audit 07, R7).
pub fn format_logs(val: &serde_json::Value) -> String {
    let wid = val.get("worker_id").and_then(|v| v.as_str()).unwrap_or("");
    let mut out = format!("Worker {wid} step logs\n");
    if let Some(entries) = val.get("logs").and_then(|v| v.as_array()) {
        for entry in entries {
            let step = entry.get("step").and_then(|v| v.as_u64()).unwrap_or(0);
            let command = entry.get("command").and_then(|v| v.as_str()).unwrap_or("");
            out.push_str(&format!("  [{step}] {command}\n"));
        }
    }
    out.push_str(&format!(
        "{}
",
        log_counters_line(val)
    ));
    out.trim_end().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::format::worker::tests::v;
    #[test]
    fn test_format_logs_numbers_steps_and_surfaces_truncation() {
        let out = format_logs(&v(
            r#"{"worker_id":"w","logs":[{"step":1,"command":"ls"},{"step":2,"command":"pwd"}],
                 "total_steps":9,"logs_omitted":4,"logs_dropped":1}"#,
        ));
        assert_eq!(
            out,
            "Worker w step logs\n  [1] ls\n  [2] pwd\ntotal_steps: 9 | retained: 6 | omitted: 4 | dropped: 1"
        );
    }
}
