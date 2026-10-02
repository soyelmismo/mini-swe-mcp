use super::review::approval_line;
use super::*;

pub fn format_status(val: &serde_json::Value) -> String {
    let wid = val.get("worker_id").and_then(|v| v.as_str()).unwrap_or("");
    let mut out = format!("Worker: {wid}\n");
    let Some(state) = val.get("state") else {
        return out.trim_end().to_string();
    };
    let Some((tag, details)) = state_and_details(state) else {
        return out.trim_end().to_string();
    };
    out.push_str(&format!("State: {tag}\n"));
    let field = |key: &str| details.and_then(|d| d.get(key));

    // One line for the progress clock: the step against its budget and how
    // long the worker has run.
    let mut progress = Vec::new();
    if let Some(step) = field("step")
        .or_else(|| field("turns"))
        .and_then(|v| v.as_u64())
    {
        progress.push(match field("max_turns").and_then(|v| v.as_u64()) {
            Some(max) if max > 0 => format!("Step {step}/{max}"),
            _ => format!("Step {step}"),
        });
    }
    if let Some(elapsed) = field("elapsed").and_then(|v| v.as_u64()) {
        progress.push(format!("elapsed {elapsed}s"));
    }
    if !progress.is_empty() {
        out.push_str(&progress.join(" | "));
        out.push('\n');
    }

    // What the worker is doing right now: the command in flight, or the build
    // slot it is queued for. Either way it is work, not a stall.
    if let Some(command) = field("last_command")
        .and_then(|v| v.as_str())
        .filter(|c| !c.is_empty())
    {
        match field("command_elapsed").and_then(|v| v.as_u64()) {
            Some(secs) => out.push_str(&format!("Command: {command} (running for {secs}s)\n")),
            None => out.push_str(&format!("Command: {command}\n")),
        }
    }
    if let Some(waiting) = field("waiting_for_slot").and_then(|v| v.as_u64()) {
        out.push_str(&format!("Waiting for a build slot ({waiting} ahead)\n"));
    }

    let line = |label: &str, key: &str| {
        field(key)
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(|s| format!("{label}: {s}\n"))
    };
    if let Some(line) = line("Summary", "summary") {
        out.push_str(&line);
    }
    if let Some(line) = line("Question", "question") {
        out.push_str(&line);
    }
    if let Some(line) = line("Error", "error") {
        out.push_str(&line);
    }
    if let Some(line) = line("Reason", "reason") {
        out.push_str(&line);
    }
    push_verified_line(&mut out, field("verified"));
    if let Some(security) = field("security_review") {
        let count = security
            .get("findings")
            .and_then(|v| v.as_u64())
            .map(|n| n.to_string())
            .unwrap_or_else(|| "not reported".to_string());
        out.push_str(&format!("Security review: {count} findings\n"));
    }
    if let Some(revision) = field("revision").and_then(|v| v.as_u64()) {
        out.push_str(&format!("Revision: {revision}\n"));
    }
    if let Some(approval) = approval_line(val) {
        out.push_str(&approval);
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

/// The state name and its details object, from any of the three shapes a
/// payload uses: a bare string, a `{"state":..,"details":..}` tag, or a keyed
/// `{"Completed": {..}}`.
fn state_and_details(
    state: &serde_json::Value,
) -> Option<(&str, Option<&serde_json::Map<String, serde_json::Value>>)> {
    if let Some(name) = state.as_str() {
        return Some((name, None));
    }
    let object = state.as_object()?;
    if let Some(name) = object.get("state").and_then(|v| v.as_str()) {
        return Some((name, object.get("details").and_then(|v| v.as_object())));
    }
    object
        .iter()
        .next()
        .map(|(name, body)| (name.as_str(), body.as_object()))
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
            "Worker: w\nState: Paused\nStep 2\nSummary: s\nQuestion: q"
        );

        let keyed = format_status(&v(
            r#"{"worker_id":"w","state":{"Failed":{"turns":1,"error":"e"}}}"#,
        ));
        assert_eq!(keyed, "Worker: w\nState: Failed\nStep 1\nError: e");
    }
}
