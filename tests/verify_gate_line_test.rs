//! The worker's first message names the exact completion gate.
//!
//! The harness reuses an identical passing verify run on an unchanged tree,
//! but only when the worker ran exactly the configured verify string. The
//! opening message therefore has to spell that string out, and has to leave it
//! out entirely when no verify is configured.

use mini_swe_mcp::pool::opening_task_message;

const VERIFY: &str = "cargo test --test verify_gate_line_test";

#[test]
fn the_first_message_names_the_exact_completion_gate() {
    let message = opening_task_message("do the thing", Some(VERIFY));
    assert!(
        message.contains("TASK:\ndo the thing"),
        "the task must still lead the message, got:\n{message}"
    );
    let gate = format!(
        "Completion gate: `{VERIFY}`. Run exactly this command as your last check; an identical passing run on the same tree is reused."
    );
    assert!(
        message.contains(&gate),
        "the gate line must carry the exact verify string, got:\n{message}"
    );
}

#[test]
fn the_first_message_omits_the_gate_without_a_verify() {
    let message = opening_task_message("do the thing", None);
    assert!(
        !message.contains("Completion gate:"),
        "no configured verify means no gate line, got:\n{message}"
    );
}
