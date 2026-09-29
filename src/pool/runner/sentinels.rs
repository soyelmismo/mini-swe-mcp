//! The orchestrator control sentinels recognised inside a subagent bash command.
//!
//! The execution loop is deliberately dumb about orchestrator protocol: a
//! subagent only ever emits an ordinary `echo`/`printf` command, and the two
//! parsers here are the single place that decides whether that command carries
//! a control signal ([`parse_request_turns`] for a turn-budget extension,
//! [`parse_ask_orchestrator`] for a blocking question) or is just work.
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
    arg.trim().trim_matches(|c| c == '"' || c == '\'' ).trim_end_matches("\\n") == COMPLETION_SENTINEL
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

#[cfg(test)]
mod tests {
    use super::{
        is_completion_request, parse_ask_orchestrator, parse_request_turns, summarize_command,
    };

    #[test]
    fn test_summarize_command_utf8() {
        let cmd = "echo 'esta_es_una_palabra_extremadamente_larga_con_ñ_y_acentos_para_superar_limite'";
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
