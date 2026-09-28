//! Integration tests for the command-parsing helpers in
//! [`mini_swe_mcp::pool`].
//!
//! The pool intercepts bash commands issued by agents and recognises two
//! "control" commands that let a worker communicate with the orchestrator:
//!
//! * `echo REQUEST_TURNS: <n>`  -> ask for `n` more turns
//! * `echo "ASK_ORCHESTRATOR: <question>"` -> pause and escalate a question
//!
//! It also builds short, log-friendly summaries of every command it runs.
//! These tests exercise those three pure functions through the public library
//! surface (i.e. the way an external integration test would).

use mini_swe_mcp::pool::{parse_ask_orchestrator, parse_request_turns, summarize_command};

// ---------------------------------------------------------------------------
// parse_request_turns
// ---------------------------------------------------------------------------

#[test]
fn test_parse_request_turns_echo_returns_count() {
    assert_eq!(parse_request_turns("echo REQUEST_TURNS: 15", ""), Some(15));
}

#[test]
fn test_parse_request_turns_variants() {
    // `printf` is accepted just like `echo`.
    assert_eq!(parse_request_turns("printf 'REQUEST_TURNS: 15'", ""), Some(15));
    // Different counts, and surrounding whitespace is irrelevant.
    assert_eq!(parse_request_turns("echo REQUEST_TURNS: 20", ""), Some(20));
    assert_eq!(parse_request_turns("echo REQUEST_TURNS: 1", ""), Some(1));
    assert_eq!(parse_request_turns("   echo REQUEST_TURNS: 42   ", ""), Some(42));
    // Leading whitespace between the token and the number is skipped.
    assert_eq!(parse_request_turns("echo REQUEST_TURNS:   7", ""), Some(7));
    // Trailing characters after the number are ignored.
    assert_eq!(parse_request_turns("echo REQUEST_TURNS: 15 # more", ""), Some(15));
}

#[test]
fn test_parse_request_turns_non_matching_commands_return_none() {
    // Right token, wrong command.
    assert_eq!(parse_request_turns("cat file.rs", "REQUEST_TURNS: 15"), None);
    // Right command, no token.
    assert_eq!(parse_request_turns("echo nothing", "normal output"), None);
    // Unrelated commands.
    assert_eq!(parse_request_turns("ls -la", "total 12"), None);
    assert_eq!(parse_request_turns("git status", "On branch main"), None);
    // The output stream is never consulted for these control commands.
    assert_eq!(parse_request_turns("echo done", "REQUEST_TURNS: 15"), None);
    // Not an echo/printf invocation at all.
    assert_eq!(parse_request_turns("REQUEST_TURNS: 15", ""), None);
    assert_eq!(parse_request_turns("myecho REQUEST_TURNS: 15", ""), None);
}

#[test]
fn test_parse_request_turns_rejects_degenerate_numbers() {
    // Zero turns is not a meaningful request.
    assert_eq!(parse_request_turns("echo REQUEST_TURNS: 0", ""), None);
    // Missing / non-numeric value.
    assert_eq!(parse_request_turns("echo REQUEST_TURNS:", ""), None);
    assert_eq!(parse_request_turns("echo REQUEST_TURNS: abc", ""), None);
    assert_eq!(parse_request_turns("echo REQUEST_TURNS: -3", ""), None);
    // Overflows `usize`.
    assert_eq!(
        parse_request_turns("echo REQUEST_TURNS: 99999999999999999999999", ""),
        None
    );
}

#[test]
fn test_parse_request_turns_empty_input() {
    assert_eq!(parse_request_turns("", ""), None);
    assert_eq!(parse_request_turns("   \n\t ", ""), None);
}

// ---------------------------------------------------------------------------
// parse_ask_orchestrator
// ---------------------------------------------------------------------------

#[test]
fn test_parse_ask_orchestrator_double_quoted_echo() {
    assert_eq!(
        parse_ask_orchestrator("echo ASK_ORCHESTRATOR: \"Proceed?\"", ""),
        Some("Proceed?".to_string())
    );
    assert_eq!(
        parse_ask_orchestrator("echo \"ASK_ORCHESTRATOR: is this ok?\"", ""),
        Some("is this ok?".to_string())
    );
}

#[test]
fn test_parse_ask_orchestrator_single_quoted_echo() {
    assert_eq!(
        parse_ask_orchestrator("echo 'ASK_ORCHESTRATOR: should I delete old code?'", ""),
        Some("should I delete old code?".to_string())
    );
    assert_eq!(
        parse_ask_orchestrator("printf 'ASK_ORCHESTRATOR: continue?'", ""),
        Some("continue?".to_string())
    );
    // Surrounding whitespace of the command is trimmed first.
    assert_eq!(
        parse_ask_orchestrator("   echo \"ASK_ORCHESTRATOR:  hello  \"   ", ""),
        Some("hello".to_string())
    );
}

#[test]
fn test_parse_ask_orchestrator_non_matching_commands_return_none() {
    // Right token, wrong command (and the output is ignored).
    assert_eq!(
        parse_ask_orchestrator(
            "cat src/agent.rs",
            "echo 'ASK_ORCHESTRATOR: <your specific question>'"
        ),
        None
    );
    // Unrelated command and output.
    assert_eq!(parse_ask_orchestrator("ls -la", "total 12"), None);
    assert_eq!(parse_ask_orchestrator("cargo build", "Finished release"), None);
    // Right command, no token.
    assert_eq!(parse_ask_orchestrator("echo done", "all good"), None);
    // A bare command that merely mentions the token.
    assert_eq!(parse_ask_orchestrator("ASK_ORCHESTRATOR: why?", ""), None);
    assert_eq!(parse_ask_orchestrator("grepecho ASK_ORCHESTRATOR: why?", ""), None);
}

#[test]
fn test_parse_ask_orchestrator_placeholders_and_blanks_are_rejected() {
    // The literal templates documented for agents must not escalate.
    assert_eq!(
        parse_ask_orchestrator("echo 'ASK_ORCHESTRATOR: <your specific question>'", ""),
        None
    );
    assert_eq!(parse_ask_orchestrator("echo \"ASK_ORCHESTRATOR: <question>\"", ""), None);
    // Empty question.
    assert_eq!(parse_ask_orchestrator("echo ASK_ORCHESTRATOR:", ""), None);
    assert_eq!(parse_ask_orchestrator("echo \"ASK_ORCHESTRATOR:   \"", ""), None);
    assert_eq!(parse_ask_orchestrator("", ""), None);
}

// ---------------------------------------------------------------------------
// summarize_command
// ---------------------------------------------------------------------------

#[test]
fn test_summarize_command_uses_only_the_first_line() {
    // Only the first line is considered...
    let cmd = "cargo build\nrm -rf /tmp/whatever\ncurl https://example.com";
    assert_eq!(summarize_command(cmd), "cargo build");

    // ...and at most the first four whitespace separated words of it.
    let long_first_line = "git commit -m 'a message that is quite long' --amend";
    assert_eq!(summarize_command(long_first_line), "git commit -m 'a");
}

#[test]
fn test_summarize_command_collapses_whitespace() {
    assert_eq!(summarize_command("  ls   -la    /tmp  "), "ls -la /tmp");
    // Only the first line of a CRLF command is summarised.
    assert_eq!(summarize_command("echo\t\thello\r\nworld"), "echo hello");
    // A leading blank line means the first line is empty -> generic fallback.
    assert_eq!(summarize_command("\n\nls -la"), "bash");
}

#[test]
fn test_summarize_command_falls_back_to_bash() {
    assert_eq!(summarize_command(""), "bash");
    assert_eq!(summarize_command("   \n\t  \n"), "bash");
}

#[test]
fn test_summarize_command_truncates_long_summaries() {
    // Four long words -> well over the 40 byte cap, so the summary is cut to
    // at most 37 bytes plus an ellipsis.
    let cmd = "echo alpha-bravo-charlie-delta-echo-foxtrot-golf-hotel-india-juliett";
    let summary = summarize_command(cmd);
    assert!(summary.ends_with("..."), "expected truncation, got {summary:?}");
    assert_eq!(summary.len(), 40, "summary should be 37 bytes + '...'");
    assert!(summary.is_char_boundary(summary.len()));
}

#[test]
fn test_summarize_command_multibyte_utf8() {
    // A long word containing non-ASCII characters is still truncated safely.
    let cmd = "echo 'esta_es_una_palabra_extremadamente_larga_con_ñ_y_acentos_para_superar_limite'";
    let summary = summarize_command(cmd);
    assert!(summary.ends_with("..."), "expected truncation, got {summary:?}");
    assert_eq!(summary.len(), 40);
    // The visible prefix must be valid UTF-8 and made of whole characters.
    let visible = summary.trim_end_matches("...");
    assert!(std::str::from_utf8(visible.as_bytes()).is_ok());
    assert!(visible.is_char_boundary(visible.len()));
}

#[test]
fn test_summarize_command_multibyte_boundary_is_not_split() {
    // '€' is three bytes and would straddle the byte-37 cut point, so the
    // implementation must back off to a char boundary instead of slicing a
    // partial code point.
    let mut special = "a".repeat(36);
    special.push('€');
    special.push_str(" rest of command");
    let summary = summarize_command(&special);
    assert!(summary.ends_with("..."), "expected truncation, got {summary:?}");
    assert_eq!(summary, format!("{}...", "a".repeat(36)));
    assert_eq!(summary.len(), 39);
    assert!(std::str::from_utf8(summary.as_bytes()).is_ok());
}

#[test]
fn test_summarize_command_other_multibyte_boundaries() {
    // A 2-byte char straddling the cut point: 'Ж' occupies bytes 36..38, so
    // the cut backs off to byte 36.
    let mut cyrillic = "a".repeat(36);
    cyrillic.push('Ж');
    cyrillic.push_str(" tail");
    let summary = summarize_command(&cyrillic);
    assert_eq!(summary, format!("{}...", "a".repeat(36)));
    assert_eq!(summary.len(), 39);

    // The 4-byte 🦀 occupies bytes 34..38, so the cut backs off to byte 34.
    let mut crab = "b".repeat(34);
    crab.push('🦀');
    crab.push_str(" tail");
    let summary = summarize_command(&crab);
    assert_eq!(summary, format!("{}...", "b".repeat(34)));
    assert_eq!(summary.len(), 37);
    assert!(std::str::from_utf8(summary.as_bytes()).is_ok());
}

#[test]
fn test_summarize_command_short_multibyte_is_untouched() {
    // No truncation: the summary is returned verbatim.
    assert_eq!(summarize_command("echo 'ñandú café ☕'"), "echo 'ñandú café ☕'");
    assert_eq!(summarize_command("ls 日本語 ファイル"), "ls 日本語 ファイル");
    // Byte length above 40 is not reached by this one.
    let s = summarize_command("echo 'ñandú café ☕'");
    assert!(s.len() < 40);
    assert!(!s.ends_with("..."));
}
