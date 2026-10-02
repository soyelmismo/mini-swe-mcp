//! The orchestrator control sentinels recognised inside a subagent bash command.
//!
//! The execution loop is deliberately dumb about orchestrator protocol: a
//! subagent only ever emits an ordinary `echo`/`printf` command, and the
//! parsers here are the single place that decides whether that command carries
//! a control signal ([`parse_request_turns`] for a turn-budget extension,
//! [`parse_ask_orchestrator`] for a blocking question, [`parse_wait_job`] and
//! [`parse_kill_job`] for a background job,
//! [`parse_consolidate_merge`] for a consolidator's branch integration,
//! [`parse_consolidate_steer`] for routing a failure back to the worker that
//! owns it, [`parse_consolidate_wait`] for blocking until that group stops) or
//! is just work.
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

/// `echo "WAIT_JOB: <n>"` → the background job to block on.
///
/// Echo/`printf` form only, like every other sentinel: the job number is the
/// first integer after the keyword, so `printf 'WAIT_JOB %d' 1` parses too.
/// A missing or zero number yields `None`, which the loop answers with "no
/// such job" rather than waiting on nothing.
pub fn parse_wait_job(cmd: &str) -> Option<u64> {
    parse_job_id(cmd, "WAIT_JOB")
}

/// `echo "KILL_JOB: <n>"` → the background job to stop.
pub fn parse_kill_job(cmd: &str) -> Option<u64> {
    parse_job_id(cmd, "KILL_JOB")
}

/// The job number an `echo`/`printf` of `keyword <n>` names.
///
/// Shared by both job sentinels so they cannot disagree about what a job
/// number looks like. The keyword must appear in an `echo`/`printf` command,
/// which keeps a `grep -rn WAIT_JOB` or a heredoc fixture from being read as a
/// request to wait.
fn parse_job_id(cmd: &str, keyword: &str) -> Option<u64> {
    let trimmed = cmd.trim();
    if (trimmed.starts_with("echo") || trimmed.starts_with("printf"))
        && let Some(pos) = trimmed.find(keyword)
    {
        let num_str: String = trimmed[pos + keyword.len()..]
            .chars()
            .skip_while(|c| c.is_whitespace() || *c == ':')
            .take_while(|c| c.is_ascii_digit())
            .collect();
        if let Ok(n) = num_str.parse::<u64>()
            && n > 0
        {
            return Some(n);
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

/// The byte budget of one parsed report field.
///
/// A completion is read at every `status`/`collect`/`watch`, so a single
/// rogue field must not bloat the in-memory record, the registry row, the
/// channel payload or the rendered notification. The same
/// [`crate::pool::clamp_string`] helper the step log already uses caps the
/// value: the truncation marker is charged against the budget, and the cut
/// never splits a UTF-8 code point.
pub const REPORT_FIELD_BYTES: usize = 4096;

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
            if opens_report_block(&line) {
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
            slot.push_str(&crate::pool::clamp_string(value, REPORT_FIELD_BYTES));
        }
    }
    (!report.is_empty()).then_some(report)
}

/// The first line of a completion message worth using as a summary, or `None`.
///
/// This is the fallback for a worker that never wrote a `done:` line, and it is
/// where a bare `REPORT` used to leak into every consumer: the harness derived
/// the summary from the whole last message, so a consolidator that opened its
/// answer with the block marker (or listed its per-worker verdicts) had the
/// marker itself become the round's headline. The marker and a consolidator's
/// per-worker verdict are both protocol, not prose, so they are skipped and the
/// first line that says something is what names the round.
///
/// A message with nothing but protocol on it yields `None`, so the caller falls
/// back to the task headline rather than printing a bare marker.
pub fn summary_line(message: &str) -> Option<&str> {
    message.lines().map(str::trim).find(|line| {
        if line.is_empty() {
            return false;
        }
        let peeled = strip_markup(line);
        !is_per_worker_line(&peeled) && !opens_report_block(&peeled)
    })
}

/// Whether `line` opens the block: the word `REPORT` alone, optionally wrapped
/// in markdown emphasis or carried as the quoted argument of the bash command
/// that echoes it, as in `printf 'REPORT\ndone: ...'`.
///
/// A consolidator's per-worker line (`REPORT <id> approved: <...>`) is not an
/// opener: it is a routing note about someone else's branch, and the round's
/// own block is what follows it. The exact-equality match below already rejects
/// it, but it is called out here so a future loosening of that match cannot
/// start reading one worker's verdict as the round's report.
fn opens_report_block(line: &str) -> bool {
    let line = line.trim();
    if is_per_worker_line(line) {
        return false;
    }
    if line
        .trim_matches(['*', '_', '`', '#', '>', ' '])
        .eq_ignore_ascii_case("REPORT")
    {
        return true;
    }
    line.rsplit(['\'', '"'])
        .next()
        .is_some_and(|tail| tail.trim().eq_ignore_ascii_case("REPORT"))
}

/// Whether `line` is a consolidator's per-worker verdict, `REPORT <id> ...`.
///
/// The id is what separates it from the block marker: the marker stands alone,
/// a verdict always names the branch it is about.
fn is_per_worker_line(line: &str) -> bool {
    let Some(rest) = line.trim().strip_prefix("REPORT") else {
        return false;
    };
    // A verdict separates the marker from the id with whitespace; the marker on
    // its own, or glued to its id, is not one.
    rest.starts_with(char::is_whitespace) && !rest.trim().is_empty()
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

/// Deadline of a `CONSOLIDATE_WAIT` that names none, and the ceiling on one
/// that does.
///
/// A consolidator that waits without `timeout=` gives its group fifteen
/// minutes; one that asks for longer is capped at an hour, so a wait can never
/// outlive the turn it is spent inside.
pub const CONSOLIDATE_WAIT_DEFAULT_SECS: u64 = 900;
pub const CONSOLIDATE_WAIT_MAX_SECS: u64 = 3600;

/// `echo/printf "CONSOLIDATE_STEER <id> <message...>"` → the worker to steer
/// and the message, verbatim.
///
/// Only a consolidator interprets this sentinel: for an ordinary worker the
/// identical command stays plain bash. Everything after the id is the message,
/// spaces and quotes included, so a correction reaches the worker the way the
/// consolidator wrote it; a request with no message is not a request.
pub fn parse_consolidate_steer(cmd: &str) -> Option<(String, String)> {
    let trimmed = cmd.trim();
    let arg = trimmed
        .strip_prefix("echo ")
        .or_else(|| trimmed.strip_prefix("printf "))?
        .trim()
        .trim_matches(['"', '\''])
        .trim_end_matches("\\n");
    // The sentinel must be a whole word: `CONSOLIDATE_STEERED` is not a request.
    let rest = arg.strip_prefix("CONSOLIDATE_STEER")?;
    if !rest.starts_with(char::is_whitespace) {
        return None;
    }
    let (id, message) = rest.trim_start().split_once(char::is_whitespace)?;
    let message = message.trim();
    let id_is_token = !id.is_empty()
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_');
    (!message.is_empty() && id_is_token).then(|| (id.to_string(), message.to_string()))
}

/// `echo/printf "CONSOLIDATE_WAIT <id> [<id> ...] [timeout=<secs>]"` → the
/// workers to wait on and the deadline in seconds, when one was given.
///
/// Only a consolidator interprets this sentinel. The optional `timeout=` is a
/// trailing word; the caller clamps it to [`CONSOLIDATE_WAIT_MAX_SECS`]. Ids
/// are validated as tokens, exactly as [`parse_consolidate_merge`] validates
/// them, so a grep of the sentinel is never a request.
pub fn parse_consolidate_wait(cmd: &str) -> Option<(Vec<String>, Option<u64>)> {
    let trimmed = cmd.trim();
    let arg = trimmed
        .strip_prefix("echo ")
        .or_else(|| trimmed.strip_prefix("printf "))?
        .trim()
        .trim_matches(['"', '\''])
        .trim_end_matches("\\n");
    let mut words = arg.split_whitespace();
    if words.next()? != "CONSOLIDATE_WAIT" {
        return None;
    }
    let mut ids = Vec::new();
    let mut timeout = None;
    for word in words {
        if let Some(secs) = word.strip_prefix("timeout=") {
            timeout = Some(secs.parse::<u64>().ok()?);
        } else if word
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
        {
            ids.push(word.to_string());
        } else {
            return None;
        }
    }
    (!ids.is_empty()).then_some((ids, timeout))
}

#[cfg(test)]
mod tests {
    use super::{
        REPORT_FIELD_BYTES, REPORT_FOLLOWUP, is_completion_request, parse_ask_orchestrator,
        parse_consolidate_merge, parse_consolidate_steer, parse_consolidate_wait, parse_kill_job,
        is_per_worker_line, opens_report_block, parse_report, parse_request_turns,
        parse_wait_job,
        summarize_command, summary_line,
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
    fn oversized_utf8_report_fields_are_bounded_without_splitting_characters() {
        // 4 KiB * 3 bytes/code point, then doubled: even a 16 KiB test stays
        // well above the budget, so the test exercises truncation, not a
        // coincidence of the input.
        let value = "界🦀é".repeat(2000);
        let message =
            format!("REPORT\ndone: {value}\nfiles: {value}\ntests: {value}\nrisks: {value}");
        let report = parse_report(&message).expect("oversized fields still parse");
        for field in [&report.done, &report.files, &report.tests, &report.risks] {
            assert!(
                field.len() <= REPORT_FIELD_BYTES,
                "field retained {} bytes, budget {}",
                field.len(),
                REPORT_FIELD_BYTES
            );
            let (head, marker) = field.split_once("... [").expect("a truncation marker");
            assert!(!head.is_empty(), "a byte always remains before the marker");
            assert!(
                value.starts_with(head),
                "only complete UTF-8 characters survive"
            );
            assert!(marker.ends_with("bytes truncated]"));
        }
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
    fn a_report_carried_inside_a_printf_command_parses() {
        // The block reaches the parser with its line breaks unfolded, the way
        // `append_report_text` hands it over.
        let command = "printf 'REPORT\\ndone: Fixed the parser\\nfiles: src/a.rs\\n' && echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT";
        assert_eq!(
            parse_report(&command.replace("\\n", "\n")),
            Some(WorkerReport {
                done: "Fixed the parser".to_string(),
                files: "src/a.rs".to_string(),
                ..Default::default()
            })
        );
    }

    /// The block marker and a consolidator's per-worker verdict are protocol,
    /// not prose: skipping them is what keeps a round's headline from being
    /// the bare word `REPORT`.
    #[test]
    fn the_summary_skips_the_block_marker_and_the_per_worker_verdicts() {
        // A protocol-only message has nothing to say, so there is no summary.
        assert_eq!(
            summary_line("REPORT\nREPORT 2a9aaca3 approved: parser"),
            None
        );
        // The marker and the verdicts are skipped; the round's own words stand.
        let with_prose = "REPORT\nREPORT 2a9aaca3 approved: parser\nRound integrated and green";
        assert_eq!(summary_line(with_prose), Some("Round integrated and green"));
    }

    #[test]
    fn a_bare_marker_is_never_the_summary() {
        assert_eq!(summary_line("REPORT"), None);
        assert_eq!(
            summary_line("REPORT 2a9aaca3 approved: fixed the parser"),
            None
        );
        assert_eq!(summary_line("   \n\n"), None);
    }

    #[test]
    fn a_per_worker_verdict_does_not_open_the_block() {
        for line in [
            "REPORT 2a9aaca3 approved: fixed the parser",
            "  REPORT 2a9aaca3 fixed: resolved the interaction",
        ] {
            assert!(!opens_report_block(line), "{line:?} is not a block opener");
        }
        // The marker alone still opens it, markdown and all.
        for line in ["REPORT", "**REPORT**", "**REPORT"] {
            assert!(opens_report_block(line), "{line:?} is a block opener");
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
    fn consolidate_steer_takes_the_id_and_the_verbatim_message() {
        let (id, message) =
            parse_consolidate_steer("echo CONSOLIDATE_STEER w1 fix the \"quoted\" name").unwrap();
        assert_eq!(id, "w1");
        assert_eq!(message, "fix the \"quoted\" name");

        let (id, message) =
            parse_consolidate_steer("printf 'CONSOLIDATE_STEER abc-1 revert it\\n'").unwrap();
        assert_eq!(id, "abc-1");
        assert_eq!(message, "revert it");

        for no in [
            "echo ordinary text",
            "grep -rn CONSOLIDATE_STEER src/",
            "echo CONSOLIDATE_STEER w1",
            "echo CONSOLIDATE_STEERED w1 fix it",
            "echo other CONSOLIDATE_STEER w1 fix it",
            "",
        ] {
            assert_eq!(parse_consolidate_steer(no), None, "{no:?} is not a request");
        }
    }

    #[test]
    fn consolidate_wait_parses_ids_and_the_optional_timeout() {
        let (ids, timeout) = parse_consolidate_wait("echo CONSOLIDATE_WAIT w1 w2 w3").unwrap();
        assert_eq!(ids, vec!["w1", "w2", "w3"]);
        assert_eq!(timeout, None);

        let (ids, timeout) =
            parse_consolidate_wait("echo CONSOLIDATE_WAIT w1 w2 timeout=120").unwrap();
        assert_eq!(ids, vec!["w1", "w2"]);
        assert_eq!(timeout, Some(120));

        let (ids, timeout) =
            parse_consolidate_wait("printf 'CONSOLIDATE_WAIT w1 timeout=0\\n'").unwrap();
        assert_eq!(ids, vec!["w1"]);
        assert_eq!(timeout, Some(0));

        for no in [
            "echo ordinary text",
            "grep -rn CONSOLIDATE_WAIT src/",
            "echo CONSOLIDATE_WAIT",
            "echo CONSOLIDATE_WAIT timeout=30",
            "echo CONSOLIDATE_WAIT w1 timeout=soon",
            "echo CONSOLIDATE_WAITED w1",
            "",
        ] {
            assert_eq!(parse_consolidate_wait(no), None, "{no:?} is not a request");
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
    fn test_parse_job_sentinels() {
        for (cmd, id) in [
            ("echo WAIT_JOB 1", Some(1)),
            ("echo WAIT_JOB: 7", Some(7)),
            ("printf 'WAIT_JOB 12\n'", Some(12)),
            ("echo WAIT_JOB", None),
            ("echo WAIT_JOB 0", None),
            ("cat job.log", None),
            ("grep -rn WAIT_JOB src/", None),
        ] {
            assert_eq!(parse_wait_job(cmd), id, "{cmd:?}");
        }
        for (cmd, id) in [
            ("echo KILL_JOB 2", Some(2)),
            ("echo KILL_JOB: 9", Some(9)),
            ("echo KILL_JOB", None),
            ("echo WAIT_JOB 4", None),
        ] {
            assert_eq!(parse_kill_job(cmd), id, "{cmd:?}");
        }
        // A job number is not a turn request, and the other way round.
        assert_eq!(parse_request_turns("echo WAIT_JOB 5"), None);
        assert_eq!(parse_wait_job("echo REQUEST_TURNS: 5"), None);
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
