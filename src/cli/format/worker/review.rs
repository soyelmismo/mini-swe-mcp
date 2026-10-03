use super::*;

/// Render the `review` action: the compact view of one worker's branch.
///
/// Everything the orchestrator needs to decide what to do next, in the order it
/// decides it: what the task was, whether it verified, how big the change is
/// per file, the summary and revision, whether the branch still merges into the
/// base tip, and the command that acts on that answer.
pub fn format_review(val: &serde_json::Value) -> String {
    let wid = val.get("worker_id").and_then(|v| v.as_str()).unwrap_or("");
    let state = val
        .get("state")
        .and_then(|v| v.as_str())
        .unwrap_or("Unknown");
    let revision = val.get("revision").and_then(|v| v.as_u64()).unwrap_or(0);
    let mut out = format!("Worker {wid} ({state}) revision {revision}\n");
    if let Some(line) = approval_line(val) {
        out.push_str(&line);
        out.push('\n');
    }
    if let Some(task) = val
        .get("task")
        .and_then(|v| v.as_str())
        .filter(|t| !t.is_empty())
    {
        out.push_str(&format!("Task: {task}\n"));
    }
    push_verified_line(&mut out, val.get("verified"));
    if let Some(tail) = val
        .get("verify_tail")
        .and_then(|v| v.as_str())
        .filter(|t| !t.is_empty())
    {
        out.push_str(&format!("Verify tail:\n{tail}\n"));
    }
    out.push_str(&format!("Diff: {}\n", diff_stat_line(val)));
    if let Some(scope) = val
        .get("diff_scope")
        .and_then(|v| v.as_str())
        .filter(|_| val.get("diff").is_some_and(|diff| !diff.is_null()))
        && let Some(diff) = val
            .get("diff")
            .and_then(|v| v.as_str())
            .filter(|diff| !diff.is_empty())
    {
        out.push_str(&format!("\nDiff ({scope}):\n{diff}\n"));
    }
    push_test_files(&mut out, val);
    push_docs(&mut out, val);
    if let Some(summary) = val
        .get("summary")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
    {
        out.push_str(&format!("Summary: {summary}\n"));
    }
    // A consolidator's per-worker verdicts, one line each: the round's
    // headline says the round happened, these say what happened to each worker
    // in it. They come after the summary, so the compact view still reads
    // first.
    if let Some(verdicts) = val
        .get("verdicts")
        .and_then(|v| v.as_array())
        .filter(|verdicts| !verdicts.is_empty())
    {
        for line in verdicts.iter().filter_map(|v| v.as_str()) {
            out.push_str(&format!("{line}\n"));
        }
    }
    if let Some(branch) = val.get("branch").and_then(|v| v.as_str()) {
        out.push_str(&format!("Branch: {branch}\n"));
    }
    if let Some(merge) = val.get("merge") {
        out.push_str(&format!("Merge: {}\n", merge_line(merge)));
    }
    if let Some(next) = val
        .get("next_command")
        .and_then(|v| v.as_str())
        .filter(|next| !next.is_empty())
    {
        out.push_str(&format!("Next: {next}\n"));
    }
    out
}

/// `3 files, +40 -12` from a `diff_stat` object, with each file's own counts
/// indented under it.
pub(super) fn diff_stat_line(val: &serde_json::Value) -> String {
    let Some(stat) = val.get("diff_stat") else {
        return "no diff measured".to_string();
    };
    let count = |key: &str| stat.get(key).and_then(|v| v.as_u64()).unwrap_or(0);
    let (files, insertions, deletions) = (count("files"), count("insertions"), count("deletions"));
    let mut line = format!(
        "{files} file{}, +{insertions} -{deletions}",
        if files == 1 { "" } else { "s" }
    );
    if let Some(per_file) = stat.get("per_file").and_then(|v| v.as_array()) {
        for file in per_file {
            let path = file.get("path").and_then(|v| v.as_str()).unwrap_or("");
            let added = file.get("insertions").and_then(|v| v.as_u64()).unwrap_or(0);
            let deleted = file.get("deletions").and_then(|v| v.as_u64()).unwrap_or(0);
            line.push_str(&format!("\n  {path}  +{added} -{deleted}"));
        }
    }
    line
}

/// The approval line for a payload carrying an `approved` object, if any.
///
/// The timestamp is the raw registry value; the note is appended when present.
pub(super) fn approval_line(val: &serde_json::Value) -> Option<String> {
    let approved = val.get("approved").filter(|v| !v.is_null())?;
    let at = approved
        .get("at")
        .and_then(|v| v.as_u64())
        .unwrap_or_default();
    match approved.get("note").and_then(|v| v.as_str()) {
        Some(note) if !note.is_empty() => Some(format!("Approved: {at} ({note})")),
        _ => Some(format!("Approved: {at}")),
    }
}

/// One line per test file: the test cases its added and removed lines declare.
pub(super) fn push_test_files(out: &mut String, val: &serde_json::Value) {
    let Some(tests) = val
        .get("test_files")
        .and_then(|v| v.as_array())
        .filter(|tests| !tests.is_empty())
    else {
        return;
    };
    out.push_str("Tests:\n");
    for test in tests {
        let path = test.get("path").and_then(|v| v.as_str()).unwrap_or("");
        let added = test
            .get("added_cases")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let removed = test
            .get("removed_cases")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        out.push_str(&format!("  {path}  +{added} -{removed} cases\n"));
    }
}

/// One line per documentation file: its +/- counts only.
pub(super) fn push_docs(out: &mut String, val: &serde_json::Value) {
    let Some(docs) = val
        .get("docs")
        .and_then(|v| v.as_array())
        .filter(|docs| !docs.is_empty())
    else {
        return;
    };
    out.push_str("Docs:\n");
    for doc in docs {
        let path = doc.get("path").and_then(|v| v.as_str()).unwrap_or("");
        let added = doc.get("insertions").and_then(|v| v.as_u64()).unwrap_or(0);
        let deleted = doc.get("deletions").and_then(|v| v.as_u64()).unwrap_or(0);
        out.push_str(&format!("  {path}  +{added} -{deleted}\n"));
    }
}

/// The merge answer in one line: clean, conflicting (naming the files), or the
/// reason no answer was possible.
pub(super) fn merge_line(merge: &serde_json::Value) -> String {
    let base = merge
        .get("base_branch")
        .and_then(|v| v.as_str())
        .unwrap_or("the base branch");
    match merge.get("clean").and_then(|v| v.as_bool()) {
        Some(true) => format!("clean into {base}"),
        Some(false) => {
            let conflicts = merge
                .get("conflicts")
                .and_then(|v| v.as_array())
                .map(|files| {
                    files
                        .iter()
                        .filter_map(|f| f.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                })
                .unwrap_or_default();
            format!("conflicts with {base}: {conflicts}")
        }
        None => match merge.get("error").and_then(|v| v.as_str()) {
            Some(error) => format!("unknown ({error})"),
            None => "unknown".to_string(),
        },
    }
}
