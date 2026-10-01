//! The orchestrator control sentinels recognised inside a subagent bash command.
//!
//! The execution loop is deliberately dumb about orchestrator protocol: a
//! subagent only ever emits an ordinary `echo`/`printf` command, and the
//! parsers here are the single place that decides whether that command carries
//! a control signal ([`parse_request_turns`] for a turn-budget extension,
//! [`parse_ask_orchestrator`] for a blocking question,
//! [`parse_consolidate_merge`] for a consolidator's branch integration) or is
//! just work.
//!
//! [`summarize_command`] lives here too because it is the same "read a bash
//! command" concern: it renders the bounded one-line label the registry, the
//! step log and the orchestrator's poll views all display.

/// One-line label for a command, truncated to a fixed byte budget.
///
/// Only the first four whitespace-separated words of the first line survive, so
/// the label can never blow up the registry entry, the step log, or a progress
/// view. Truncation happens on a char boundary so multi-byte UTF-8 input is
/// never sliced in the middle of a code point.
pub fn summarize_command(cmd: &str) -> String {
    let first_line = cmd.lines().next().unwrap_or("").trim();
    if first_line.is_empty() {
        return "bash".to_string();
    }

    let mut out = String::with_capacity(40);
    let mut words = first_line.split_whitespace().take(4);

    if let Some(first) = words.next() {
        out.push_str(first);
        for word in words {
            out.push(' ');
            out.push_str(word);
        }
    }

    if out.len() > 40 {
        let cut = out.floor_char_boundary(37);
        out.truncate(cut);
        out.push_str("...");
    }

    out
}

/// The completion sentinel a subagent echoes to finish its task.
pub const COMPLETION_SENTINEL: &str = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT";

/// Whether `cmd` is the completion request: the *last* shell segment of the
/// command is `echo`/`printf` of the sentinel (optionally quoted).
///
/// A substring match is not enough: `grep -rn COMPLETE_TASK...`, a heredoc
/// writing a test fixture, or a code block quoting the system prompt all
/// contain the sentinel without asking to finish, and used to end the worker
/// with nothing done. `cargo test && echo COMPLETE_...` still counts.
pub fn is_completion_request(cmd: &str) -> bool {
    let Some(last_line) = cmd.lines().map(str::trim).rfind(|l| !l.is_empty()) else {
        return false;
    };
    let segment = last_line
        .rsplit(['&', ';', '|'])
        .next()
        .unwrap_or(last_line)
        .trim();
    let Some(arg) = segment
        .strip_prefix("echo ")
        .or_else(|| segment.strip_prefix("printf "))
    else {
        return false;
    };
    arg.trim()
        .trim_matches(|c| c == '"' || c == '\'')
        .trim_end_matches("\\n")
        == COMPLETION_SENTINEL
}

/// `echo "REQUEST_TURNS: N"` → the extra turns the subagent is asking for.
///
/// Returns `None` for a zero request (a request that grants nothing would let
/// a worker stall forever) and for any command that is not the sentinel.
pub fn parse_request_turns(cmd: &str) -> Option<usize> {
    let trimmed = cmd.trim();
    if (trimmed.starts_with("echo") || trimmed.starts_with("printf"))
        && let Some(pos) = trimmed.find("REQUEST_TURNS:")
    {
        let rest = &trimmed[pos + "REQUEST_TURNS:".len()..];
        let num_str: String = rest
            .chars()
            .skip_while(|c| c.is_whitespace())
            .take_while(|c| c.is_ascii_digit())
            .collect();
        if let Ok(n) = num_str.parse::<usize>()
            && n > 0
        {
            return Some(n);
        }
    }
    None
}

/// `echo "ASK_ORCHESTRATOR: <question>"` → the question blocking the worker.
///
/// Placeholder text (the angle-bracket templates printed in the system prompt)
/// is treated as "no question asked" so a subagent that merely echoes the
/// template cannot deadlock the pool waiting for an orchestrator reply.
pub fn parse_ask_orchestrator(cmd: &str) -> Option<String> {
    let trimmed = cmd.trim();
    if (trimmed.starts_with("echo") || trimmed.starts_with("printf"))
        && let Some(pos) = trimmed.find("ASK_ORCHESTRATOR:")
    {
        let rest = &trimmed[pos + "ASK_ORCHESTRATOR:".len()..];
        let line = rest
            .lines()
            .next()
            .unwrap_or("")
            .trim()
            .trim_matches('"')
            .trim_matches('\'')
            .trim();
        if !line.is_empty() && line != "<your specific question>" && line != "<question>" {
            return Some(line.to_string());
        }
    }
    None
}

/// `echo/printf "CONSOLIDATE_MERGE <id> ..."` → the workers to integrate.
///
/// Only a consolidator interprets this sentinel: for an ordinary worker the
/// identical command stays plain bash. Ids are whatever the pool's id resolver
/// accepts (a full id or a unique prefix); they are validated here only as
/// tokens, so a grep of the sentinel or a sentence that merely contains it is
/// never a request.
pub fn parse_consolidate_merge(cmd: &str) -> Option<Vec<String>> {
    let trimmed = cmd.trim();
    let arg = trimmed
        .strip_prefix("echo ")
        .or_else(|| trimmed.strip_prefix("printf "))?
        .trim()
        .trim_matches(['"', '\''])
        .trim_end_matches("\\n");
    let mut words = arg.split_whitespace();
    if words.next()? != "CONSOLIDATE_MERGE" {
        return None;
    }
    let ids: Vec<String> = words.map(str::to_string).collect();
    let tokens = !ids.is_empty()
        && ids.iter().all(|id| {
            id.bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
        });
    tokens.then_some(ids)
}

/// The four keys of a completion report, in the order the system prompt asks
/// for them. A line whose key is not one of these is ignored, so a worker that
/// adds a fifth line still parses.
const REPORT_KEYS: [&str; 4] = ["done", "files", "tests", "risks"];

/// The one follow-up the harness sends when a completion turn carries no
/// report: the block, then the sentinel again.
pub const REPORT_FOLLOWUP: &str = "Reply with the REPORT block only, then the completion sentinel";

/// Parse the `REPORT` block out of a completion message.
///
/// The block is the worker's structured answer, so the parser is deliberately
/// forgiving about everything that is not the content: a markdown fence around
/// the block, `-`/`*`/`1.` bullets, a trailing colon after the key, and any
/// capitalisation of the key. The first line that is exactly `REPORT` (modulo
/// fences and bullets) opens the block; the keys that follow fill it, and a
/// `None` result means "no block", never "an empty block".
///
/// Keys are matched case-insensitively and only the first occurrence of each
/// wins, so a worker that repeats a line cannot rewrite an earlier answer.
pub fn parse_report(message: &str) -> Option<super::super::WorkerReport> {
    let mut report = super::super::WorkerReport::default();
    let mut in_block = false;
    for line in message.lines() {
        let line = strip_markup(line);
        if !in_block {
            if line
                .trim_matches(['*', '_', '`', '#', '>', ' '])
                .eq_ignore_ascii_case("REPORT")
            {
                in_block = true;
            }
            continue;
        }
        // A blank line ends the block only once a key has been read: a worker
        // that spaces its lines out is not a worker that stopped reporting.
        if line.is_empty() {
            if report.is_empty() {
                continue;
            }
            break;
        }
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let key = key
            .trim()
            .trim_matches(['*', '_', '`', '#'])
            .to_ascii_lowercase();
        let value = value.trim();
        if !REPORT_KEYS.contains(&key.as_str()) || value.is_empty() {
            continue;
        }
        let slot = match key.as_str() {
            "done" => &mut report.done,
            "files" => &mut report.files,
            "tests" => &mut report.tests,
            _ => &mut report.risks,
        };
        if slot.is_empty() {
            slot.push_str(value);
        }
    }
    (!report.is_empty()).then_some(report)
}

/// Peel the markdown a model wraps a block in: code fences, list bullets and
/// the heading or emphasis markers around the key.
///
/// Only the *markers* are removed, never the content: a value that happens to
/// contain a digit (`cargo test: 4 passed`) must survive untouched, which is
/// why a numbered bullet is recognised by its leading digits rather than by
/// the first digit anywhere on the line.
fn strip_markup(line: &str) -> String {
    let mut line = line.trim().trim_matches('`').trim();
    for marker in ["- ", "* ", "+ "] {
        if let Some(rest) = line.strip_prefix(marker) {
            line = rest.trim();
            break;
        }
    }
    // A numbered bullet: `1. done: ...` or `1) done: ...`.
    if let Some(rest) = line.strip_prefix(|c: char| c.is_ascii_digit())
        && let Some(sep) = rest.strip_prefix('.').or_else(|| rest.strip_prefix(')'))
    {
        line = sep.trim();
    }
    line.trim_start_matches(['#', '>', '*', '_'])
        .trim()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::{
        REPORT_FOLLOWUP, is_completion_request, parse_ask_orchestrator, parse_consolidate_merge,
        parse_report, parse_request_turns, summarize_command,
    };
    use crate::pool::WorkerReport;

    #[test]
    fn a_plain_report_block_yields_all_four_lines() {
        let message = "All done.\n\nREPORT\ndone: Fixed the parser\nfiles: src/a.rs, src/b.rs\ntests: cargo test: passed\nrisks: none\n\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT";
        assert_eq!(
            parse_report(message),
            Some(WorkerReport {
                done: "Fixed the parser".to_string(),
                files: "src/a.rs, src/b.rs".to_string(),
                tests: "cargo test: passed".to_string(),
                risks: "none".to_string(),
            })
        );
    }

    #[test]
    fn a_fenced_and_bulleted_report_block_still_parses() {
        let message = "```\nREPORT\n- DONE: Fixed the parser\n* Files: src/a.rs\n1. tests: cargo test: passed\n- risks: none\n```";
        let report = parse_report(message).expect("a fenced block must parse");
        assert_eq!(report.done, "Fixed the parser");
        assert_eq!(report.files, "src/a.rs");
        assert_eq!(report.tests, "cargo test: passed");
        assert_eq!(report.risks, "none");
    }

    #[test]
    fn a_report_block_with_missing_keys_keeps_what_it_has() {
        let message = "REPORT\ndone: Fixed the parser\nrisks: changes the wire format";
        let report = parse_report(message).expect("a partial block must parse");
        assert_eq!(report.done, "Fixed the parser");
        assert_eq!(report.risks, "changes the wire format");
        assert!(report.files.is_empty(), "{report:?}");
        assert!(report.tests.is_empty(), "{report:?}");
    }

    #[test]
    fn a_message_without_a_report_block_has_no_report() {
        for absent in [
            "Now I'll make the edits.",
            "## Summary",
            "REPORT",
            "done: Fixed the parser\nfiles: src/a.rs",
            "",
        ] {
            assert_eq!(parse_report(absent), None, "{absent:?} carries no block");
        }
    }

    #[test]
    fn the_follow_up_asks_for_the_block_and_nothing_else() {
        assert!(REPORT_FOLLOWUP.contains("REPORT"));
        assert!(REPORT_FOLLOWUP.contains("completion sentinel"));
    }

    #[test]
    fn consolidate_merge_requires_an_echo_of_the_sentinel() {
        for yes in [
            "echo CONSOLIDATE_MERGE abc abcdef12",
            "printf 'CONSOLIDATE_MERGE abc abcdef12\n'",
            "  echo \"CONSOLIDATE_MERGE abc\"  ",
        ] {
            let ids = parse_consolidate_merge(yes);
            assert!(ids.is_some(), "{yes:?} must request a merge");
            assert!(
                ids.as_deref().unwrap_or(&[]).contains(&"abc".to_string()),
                "{yes:?} must name the worker: {ids:?}"
            );
        }
        for no in [
            "echo ordinary text",
            "grep -rn CONSOLIDATE_MERGE src/",
            "echo CONSOLIDATE_MERGE",
            "echo other CONSOLIDATE_MERGE abc",
            "",
        ] {
            assert_eq!(parse_consolidate_merge(no), None, "{no:?} is not a request");
        }
    }

    #[test]
    fn test_summarize_command_utf8() {
        let cmd =
            "echo 'esta_es_una_palabra_extremadamente_larga_con_ñ_y_acentos_para_superar_limite'";
        let summary = summarize_command(cmd);
        assert!(summary.ends_with("..."));

        // Multi-byte character exactly crossing byte 37
        let mut special = "a".repeat(36);
        special.push('€');
        special.push_str(" rest of command");
        let summary_special = summarize_command(&special);
        assert!(summary_special.ends_with("..."));
    }

    #[test]
    fn test_completion_request_requires_the_sentinel_as_the_final_echo() {
        for yes in [
            "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
            "  echo \"COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\"  ",
            "cargo test && echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
            "printf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\n'",
            "cargo test\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n",
        ] {
            assert!(is_completion_request(yes), "{yes:?} must complete");
        }
        for no in [
            "grep -rn COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT src/",
            "cat > t.rs <<'EOF'\nlet cmd = \"echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\";\nEOF",
            "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cargo test",
            "echo not done COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
            "",
        ] {
            assert!(!is_completion_request(no), "{no:?} must not complete");
        }
    }

    #[test]
    fn test_parse_request_turns() {
        assert_eq!(parse_request_turns("echo REQUEST_TURNS: 20"), Some(20));
        assert_eq!(parse_request_turns("printf 'REQUEST_TURNS: 15'"), Some(15));
        assert_eq!(parse_request_turns("cat file.rs"), None);
        assert_eq!(parse_request_turns("echo nothing"), None);
        assert_eq!(parse_request_turns("echo REQUEST_TURNS: 0"), None);
    }

    #[test]
    fn test_parse_ask_orchestrator() {
        assert_eq!(
            parse_ask_orchestrator("echo 'ASK_ORCHESTRATOR: should I delete old code?'"),
            Some("should I delete old code?".to_string())
        );
        assert_eq!(
            parse_ask_orchestrator("echo \"ASK_ORCHESTRATOR: is this ok?\""),
            Some("is this ok?".to_string())
        );
        assert_eq!(parse_ask_orchestrator("cat src/agent.rs"), None);
        assert_eq!(
            parse_ask_orchestrator("echo 'ASK_ORCHESTRATOR: <your specific question>'"),
            None
        );
        assert_eq!(parse_ask_orchestrator("ls -la"), None);
    }
}
