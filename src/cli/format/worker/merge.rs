/// `merge`: one line naming the commit and what the cleanup reclaimed, or the
/// compact report of a batch that landed a whole round with one gate.
///
/// The gate is the part an operator wants to know about without reading a
/// paragraph: whether it ran, or why it was skipped.
pub fn format_merge(val: &serde_json::Value) -> String {
    // A batch answer carries `merged` as a list; a single merge answers with
    // one commit.
    if val.get("merged").and_then(|v| v.as_array()).is_some() {
        return format_merge_approved(val);
    }
    let commit = val.get("commit").and_then(|v| v.as_str()).unwrap_or("");
    let base = val
        .get("base_branch")
        .and_then(|v| v.as_str())
        .unwrap_or("the base branch");
    let branch = val.get("branch").and_then(|v| v.as_str()).unwrap_or("");
    let kept = !val
        .get("branch_deleted")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let gate = match val.get("gate").and_then(|v| v.as_str()) {
        Some("skipped") => "gate skipped (branch already verified)",
        _ => "gate passed",
    };
    let cleaned = val
        .get("cleaned")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        })
        .unwrap_or_default();
    let mut out = format!("✓ Merged {branch} into {base} as {commit} ({gate}).");
    if kept {
        out.push_str(&format!(" Branch {branch} kept."));
    } else if !cleaned.is_empty() {
        out.push_str(&format!(" Cleaned: {cleaned}."));
    }
    out
}

/// `merge --approved`: the merged ids, the skipped ones, the one gate and what
/// the cleanup reclaimed, on as few lines as that fits.
pub(super) fn format_merge_approved(val: &serde_json::Value) -> String {
    let base = val
        .get("base_branch")
        .and_then(|v| v.as_str())
        .unwrap_or("the base branch");
    let merged: Vec<String> = val
        .get("merged")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|m| {
                    let id = m.get("worker_id").and_then(|v| v.as_str())?;
                    let commit = m.get("commit").and_then(|v| v.as_str()).unwrap_or("?");
                    Some(format!("{id} as {commit}"))
                })
                .collect()
        })
        .unwrap_or_default();
    let skipped: Vec<String> = val
        .get("skipped")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|s| {
                    let id = s.get("worker_id").and_then(|v| v.as_str())?;
                    let files = s
                        .get("files")
                        .and_then(|v| v.as_array())
                        .map(|f| {
                            f.iter()
                                .filter_map(|v| v.as_str())
                                .collect::<Vec<_>>()
                                .join(", ")
                        })
                        .unwrap_or_default();
                    Some(if files.is_empty() {
                        id.to_string()
                    } else {
                        let steer = s.get("steer").and_then(|v| v.as_str()).unwrap_or_default();
                        if steer.is_empty() {
                            format!("{id} (conflicts in {files})")
                        } else {
                            format!("{id} (conflicts in {files}; {steer})")
                        }
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    let gate = match val.get("gate_command").and_then(|v| v.as_str()) {
        Some(command) => format!(
            "gate {command} passed in {}ms",
            val.get("gate_duration_ms")
                .and_then(|v| v.as_u64())
                .unwrap_or(0)
        ),
        None => "no gate ran".to_string(),
    };
    let cleaned = val
        .get("cleaned")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        })
        .unwrap_or_default();
    let mut out = format!(
        "✓ Merged {} into {base} ({gate}).",
        if merged.is_empty() {
            "nothing".to_string()
        } else {
            merged.join(", ")
        }
    );
    if !skipped.is_empty() {
        out.push_str(&format!(" Skipped: {}.", skipped.join("; ")));
    }
    if !cleaned.is_empty() {
        out.push_str(&format!(" Cleaned: {cleaned}."));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::format::worker::tests::v;
    /// `merge --approved` renders the merged ids, the skipped ones with their
    /// steer, the one gate and its duration.
    #[test]
    fn test_format_merge_renders_a_batch() {
        let out = format_merge(&v(
            r#"{"approved":true,"base_branch":"main","merged":[{"worker_id":"w1","commit":"abc1234"},{"worker_id":"w3","commit":"def5678"}],"skipped":[{"worker_id":"w2","files":["shared.txt"],"steer":"steer w2 \"merge conflicts in shared.txt\""}],"gate_command":"cargo test","gate_duration_ms":420,"cleaned":["branch worker-w1 deleted"]}"#,
        ));
        assert!(
            out.contains("Merged w1 as abc1234, w3 as def5678 into main"),
            "{out}"
        );
        assert!(out.contains("gate cargo test passed in 420ms"), "{out}");
        assert!(
            out.contains("Skipped: w2 (conflicts in shared.txt; steer w2"),
            "{out}"
        );
        assert!(out.contains("Cleaned: branch worker-w1 deleted"), "{out}");
    }
}
