//! Structured completion event regression coverage.
use mini_swe_mcp::cli::watch;
use serde_json::json;

#[test]
fn compact_completion_uses_done_files_and_risks() {
    let event = json!({
        "worker_id":"w-report", "event":"completed", "summary":"## Summary",
        "report":{"done":"Fix completion reporting", "files":["src/a.rs"],
                  "tests":"cargo test: passed", "risks":"Changes completion feedback"},
        "diff_stat":{"files":1,"insertions":3,"deletions":2,
                     "per_file":[{"path":"src/a.rs","insertions":3,"deletions":2}]},
        "step":2,"max_turns":5,"elapsed":1,"task":"report probe"
    });
    let text = watch::render(&event);
    assert!(text.contains("Fix completion reporting"), "{text}");
    assert!(text.contains("files: src/a.rs (+3 -2)"), "{text}");
    assert!(text.contains("risks: Changes completion feedback"), "{text}");
    assert!(text.lines().count() <= 5, "{text}");
}
