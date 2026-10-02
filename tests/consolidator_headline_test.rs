//! A consolidator's completion headline is its round summary, not the bare
//! word `REPORT`.
//!
//! The consolidator procedure asks for a per-worker line per integrated branch
//! (`REPORT <id> approved|returned|fixed: ...`). Those lines sit next to the
//! standard block every worker writes, and the block parser opens on the first
//! line that is exactly `REPORT` — so a consolidator whose per-worker lines
//! come first used to leave the parser reading them as the round's own report,
//! and a message that opened with the block marker and then wrote the block's
//! own keys under it read as a summary of nothing. The block must win, the
//! per-worker lines must stay in the message, and a bare `REPORT` must never
//! become a headline.

use mini_swe_mcp::agent::CONSOLIDATOR_INSTRUCTIONS;
use mini_swe_mcp::pool::parse_report;

/// The message a consolidator writes after the standard block and the
/// per-worker lines: the block first (its `done:` is the round summary), then
/// one line per integrated worker.
fn consolidator_message() -> String {
    [
        "REPORT",
        "done: integrated 2 branches, gate green",
        "files: src/a.rs, src/b.rs",
        "tests: cargo fmt --check && cargo clippy --all-targets -- -D warnings && cargo test: passed",
        "risks: none",
        "",
        "REPORT 2a9aaca3 approved: fixed the parser",
        "REPORT 41b0fde1 fixed: resolved the interaction in src/b.rs",
    ]
    .join("\n")
}

#[test]
fn a_consolidator_block_first_message_yields_its_done_as_the_report() {
    let report = parse_report(&consolidator_message())
        .expect("a consolidator that writes the standard block must parse");
    assert_eq!(report.done, "integrated 2 branches, gate green");
    assert_eq!(report.files, "src/a.rs, src/b.rs");
    assert!(report.tests.contains("clippy"), "{}", report.tests);
    assert_eq!(report.risks, "none");
}

#[test]
fn the_per_worker_lines_stay_in_the_message_for_the_orchestrator_to_read() {
    let message = consolidator_message();
    for line in [
        "REPORT 2a9aaca3 approved: fixed the parser",
        "REPORT 41b0fde1 fixed: resolved the interaction in src/b.rs",
    ] {
        assert!(
            message.contains(line),
            "the per-worker line must survive: {message}"
        );
    }
}

/// A message that carries only the bare marker is no report at all: a `None`
/// result is what makes the harness ask once, so it must never be able to
/// answer with the word `REPORT`.
#[test]
fn a_bare_report_marker_is_never_a_summary() {
    assert_eq!(parse_report("REPORT"), None);
    assert_eq!(parse_report("REPORT\nREPORT 2a9aaca3 approved: fixed the parser"), None);
}

/// A per-worker line is not a block opener, so it cannot swallow the keys that
/// follow it into a report nobody wrote.
#[test]
fn a_per_worker_line_does_not_open_the_block() {
    let message = [
        "REPORT 2a9aaca3 approved: fixed the parser",
        "REPORT 41b0fde1 fixed: resolved the interaction",
    ]
    .join("\n");
    assert_eq!(parse_report(&message), None, "{message}");
}

#[test]
fn a_worker_block_still_parses_as_before() {
    let report = parse_report(
        "All done.\n\nREPORT\ndone: Fixed the parser\nfiles: src/a.rs\ntests: cargo test: passed\nrisks: none",
    )
    .expect("the ordinary worker block must still parse");
    assert_eq!(report.done, "Fixed the parser");
    assert_eq!(report.files, "src/a.rs");
    assert_eq!(report.tests, "cargo test: passed");
    assert_eq!(report.risks, "none");
}

#[test]
fn the_procedure_asks_for_the_standard_block_before_the_per_worker_lines() {
    let block = CONSOLIDATOR_INSTRUCTIONS
        .find("done: <one line")
        .expect("the procedure must spell out the standard block's `done:` line");
    let per_worker = CONSOLIDATOR_INSTRUCTIONS
        .find("REPORT <id>")
        .expect("the procedure must keep the per-worker line");
    assert!(
        block < per_worker,
        "the standard block must be asked for first: {CONSOLIDATOR_INSTRUCTIONS}"
    );
    assert!(CONSOLIDATOR_INSTRUCTIONS.contains("tests: <the full gate result>"));
    assert!(CONSOLIDATOR_INSTRUCTIONS.contains("files: <paths changed"));
    assert!(CONSOLIDATOR_INSTRUCTIONS.contains("risks: <"));
}
