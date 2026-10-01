use super::review::approval_line;
use super::*;

pub fn format_status(val: &serde_json::Value) -> String {
    let wid = val.get("worker_id").and_then(|v| v.as_str()).unwrap_or("");
    let mut out = format!("Worker: {wid}\n");
    if let Some(state) = val.get("state") {
        if let Some(status_str) = state.as_str() {
            out.push_str(&format!("State: {status_str}\n"));
        } else {
            let tag = state.get("state").and_then(|v| v.as_str());
            let details = state.get("details").and_then(|v| v.as_object());

            if let Some(t) = tag {
                out.push_str(&format!("State: {t}\n"));
                if let Some(d) = details {
                    if let Some(turns) = d.get("turns").and_then(|v| v.as_u64()) {
                        out.push_str(&format!("Turns: {turns}\n"));
                    }
                    if let Some(step) = d.get("step").and_then(|v| v.as_u64()) {
                        out.push_str(&format!("Step: {step}\n"));
                    }
                    if let Some(summary) = d.get("summary").and_then(|v| v.as_str()) {
                        out.push_str(&format!("Summary: {summary}\n"));
                    }
                    if let Some(err) = d.get("error").and_then(|v| v.as_str()) {
                        out.push_str(&format!("Error: {err}\n"));
                    }
                    if let Some(q) = d.get("question").and_then(|v| v.as_str()) {
                        out.push_str(&format!("Question: {q}\n"));
                    }
                    push_verified_line(&mut out, d.get("verified"));
                }
            } else if let Some(obj) = state.as_object() {
                for (state_name, d) in obj {
                    out.push_str(&format!("State: {state_name}\n"));
                    if let Some(turns) = d.get("turns").and_then(|v| v.as_u64()) {
                        out.push_str(&format!("Turns: {turns}\n"));
                    }
                    if let Some(step) = d.get("step").and_then(|v| v.as_u64()) {
                        out.push_str(&format!("Step: {step}\n"));
                    }
                    if let Some(summary) = d.get("summary").and_then(|v| v.as_str()) {
                        out.push_str(&format!("Summary: {summary}\n"));
                    }
                    if let Some(err) = d.get("error").and_then(|v| v.as_str()) {
                        out.push_str(&format!("Error: {err}\n"));
                    }
                    if let Some(q) = d.get("question").and_then(|v| v.as_str()) {
                        out.push_str(&format!("Question: {q}\n"));
                    }
                    push_verified_line(&mut out, d.get("verified"));
                }
            }
        }
    }
    if let Some(line) = approval_line(val) {
        out.push_str(&line);
        out.push('\n');
    }
    if let Some(health) = health_line(val) {
        out.push_str(&health);
        out.push('\n');
    }
    // A finished worker's status tells the orchestrator the loop exists: the
    // branch is still there, and steering this worker resumes it in place.
    if let Some(next) = val.get("next_step").and_then(|v| v.as_str())
        && !next.trim().is_empty()
    {
        out.push_str(&format!("Next step: {next}\n"));
    }
    out.trim_end().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::format::worker::tests::v;
    #[test]
    fn test_format_status_covers_string_tagged_and_object_states() {
        let bare = format_status(&v(r#"{"worker_id":"w","state":"Running"}"#));
        assert_eq!(bare, "Worker: w\nState: Running");

        let tagged = format_status(&v(
            r#"{"worker_id":"w","state":{"state":"Paused","details":{"turns":4,"step":2,"summary":"s","question":"q"}}}"#,
        ));
        assert_eq!(
            tagged,
            "Worker: w\nState: Paused\nTurns: 4\nStep: 2\nSummary: s\nQuestion: q"
        );

        let keyed = format_status(&v(
            r#"{"worker_id":"w","state":{"Failed":{"turns":1,"error":"e"}}}"#,
        ));
        assert_eq!(keyed, "Worker: w\nState: Failed\nTurns: 1\nError: e");
    }
}
