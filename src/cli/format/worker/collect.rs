use super::review::diff_stat_line;
use super::*;

pub fn format_collect(val: &serde_json::Value) -> String {
    let wid = val.get("worker_id").and_then(|v| v.as_str()).unwrap_or("");
    let diff = val
        .get("state")
        .and_then(|s| s.get("details").or_else(|| s.get("Completed")))
        .and_then(|c| c.get("diff"))
        .or_else(|| val.get("diff"))
        .and_then(|v| v.as_str())
        .unwrap_or("");

    let counters = log_counters_line(val);
    let health = health_line(val)
        .map(|h| format!("\n{h}"))
        .unwrap_or_default();
    let next = val
        .get("next_step")
        .and_then(|v| v.as_str())
        .filter(|next| !next.trim().is_empty())
        .map(|next| format!("\nNext step: {next}"))
        .unwrap_or_default();
    if diff.trim().is_empty() {
        // No diff was asked for (or none was produced), so the per-file stat is
        // what stands in for it.
        format!(
            "Worker {wid}: {}\n{counters}{health}{next}",
            diff_stat_line(val)
        )
    } else {
        format!("{diff}\n{counters}{health}{next}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::format::worker::tests::v;
    #[test]
    fn test_format_collect_prefers_the_diff_and_keeps_counters() {
        let with_diff = format_collect(&v(
            r#"{"worker_id":"w","state":{"details":{"diff":"--- a\n+++ b"}},"total_steps":2}"#,
        ));
        assert!(with_diff.starts_with("--- a\n+++ b\n"));
        assert!(with_diff.ends_with("total_steps: 2"));

        // The default collect withholds the diff, so the per-file stat is what
        // the view reports in its place.
        let stat_only = format_collect(&v(
            r#"{"worker_id":"w","diff_stat":{"files":2,"insertions":3,"deletions":1,
                 "per_file":[{"path":"a.rs","insertions":3,"deletions":0},
                             {"path":"b.rs","insertions":0,"deletions":1}]}}"#,
        ));
        assert_eq!(
            stat_only,
            "Worker w: 2 files, +3 -1\n  a.rs  +3 -0\n  b.rs  +0 -1\nno step logs"
        );

        let without = format_collect(&v(r#"{"worker_id":"w","diff":"  "}"#));
        assert_eq!(without, "Worker w: no diff measured\nno step logs");
    }
}
