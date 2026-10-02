/// `discard`: one line naming the worker and everything the retirement took
/// with it, so the operator sees the branch and the scratch are gone.
pub fn format_discard(val: &serde_json::Value) -> String {
    let wid = val.get("worker_id").and_then(|v| v.as_str()).unwrap_or("");
    let branch = val
        .get("branch_deleted")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let worktree = val
        .get("worktree_reclaimed")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let mut line = format!("✓ Worker {wid} discarded.");
    if branch {
        line.push_str(" Its branch and scratch state are gone;");
    }
    if worktree {
        line.push_str(" its worktree was reclaimed;");
    }
    line.push_str(" nothing of it is left to merge, steer or watch.");
    line
}
