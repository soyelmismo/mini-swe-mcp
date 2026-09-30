//! In-place conversation compaction, shared by implementer and reviewer turns.
//!
//! Keep the prompt, task and recent exchanges intact. Older calls and their ids
//! remain in order, but their text is replaced, not merely sliced, so the live
//! history and saved revisions no longer own the original output allocations.

use crate::agent::{ChatMessage, Role};
use crate::config::env_parse;

use super::turn::{COMMAND_OUTPUT_PREFIX, NO_COMMAND_NUDGE, VERIFICATION_OUTPUT_PREFIX};

const DEFAULT_FULL_TURNS: usize = 12;
const OUTPUT_PREVIEW_BYTES: usize = 300;
const PROSE_BYTES: usize = 500;
const REASONING_BYTES: usize = 200;
const PROSE_SUFFIX: &str = " [prose elided]";
const REASONING_SUFFIX: &str = " [reasoning elided]";
const OUTPUT_PREFIX: &str = "[output elided: exit ";

pub(super) fn compact_history(messages: &mut [ChatMessage]) {
    compact_with_policy(
        messages,
        env_parse::<usize>("HISTORY_FULL_TURNS").unwrap_or(DEFAULT_FULL_TURNS),
        env_parse::<u8>("HISTORY_KEEP_ALL_REASONING") == Some(1),
    );
}

fn compact_with_policy(messages: &mut [ChatMessage], full_turns: usize, keep_reasoning: bool) {
    // An exchange starts at its assistant message and includes all its results.
    let cutoff = messages
        .iter()
        .enumerate()
        .rev()
        .filter(|(_, m)| m.role() == Role::Assistant)
        .nth(full_turns.saturating_sub(1))
        .map_or(0, |(i, _)| i);
    let cutoff = if full_turns == 0 {
        messages.len()
    } else {
        cutoff
    };
    let task = messages.iter().position(|m| m.role() == Role::User);
    for (i, message) in messages[..cutoff].iter_mut().enumerate() {
        if message.role() == Role::System || Some(i) == task {
            continue;
        }
        if let Some(text) = message.content() {
            let compacted = match message.role() {
                Role::Assistant => shorten(text, PROSE_BYTES, PROSE_SUFFIX),
                Role::Tool => output_stub(text),
                Role::User
                    if text.starts_with(COMMAND_OUTPUT_PREFIX) || text == NO_COMMAND_NUDGE =>
                {
                    output_stub(text)
                }
                Role::System | Role::User => None,
            };
            if let Some(text) = compacted {
                message.replace_content(text);
            }
        }
        if !keep_reasoning
            && message.role() == Role::Assistant
            && let Some(reasoning) = message.reasoning_content()
            && let Some(text) = shorten(reasoning, REASONING_BYTES, REASONING_SUFFIX)
        {
            message.replace_reasoning_content(text);
        }
    }
}

fn shorten(text: &str, bytes: usize, suffix: &str) -> Option<String> {
    if text.len() <= bytes || (text.len() <= bytes + suffix.len() && text.ends_with(suffix)) {
        return None;
    }
    Some(format!(
        "{}{suffix}",
        &text[..text.floor_char_boundary(bytes)]
    ))
}

fn output_stub(text: &str) -> Option<String> {
    // Recognize our bounded stub so reapplying compaction preserves byte counts.
    if text.starts_with(OUTPUT_PREFIX) && text.ends_with(']') && text.len() <= 400 {
        return None;
    }
    let exit = text
        .strip_prefix(COMMAND_OUTPUT_PREFIX)
        .or_else(|| text.strip_prefix(VERIFICATION_OUTPUT_PREFIX))
        .and_then(|rest| rest.split_once(')'))
        .and_then(|(code, _)| code.parse::<i32>().ok());
    let exit = exit.map_or_else(|| "unknown".to_string(), |code| code.to_string());
    let preview = &text[..text.floor_char_boundary(OUTPUT_PREVIEW_BYTES)];
    Some(format!(
        "{OUTPUT_PREFIX}{exit}, {} bytes; first lines: {preview}]",
        text.len()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::{ToolCall, ToolCallFn};

    fn exchange(messages: &mut Vec<ChatMessage>, turn: usize, calls: usize) {
        let tool_calls = (0..calls)
            .map(|call| ToolCall {
                id: format!("call_{turn}_{call}"),
                r#type: "function".into(),
                function: ToolCallFn {
                    name: "bash".into(),
                    arguments: format!(r#"{{"command":"echo {turn} {call}"}}"#),
                },
            })
            .collect();
        messages.push(
            ChatMessage::assistant_with_tool_calls(Some("文".repeat(1000)), tool_calls)
                .with_reasoning_content(Some("考".repeat(1000))),
        );
        for call in 0..calls {
            messages.push(ChatMessage::tool_result(
                format!("call_{turn}_{call}"),
                format!(
                    "COMMAND OUTPUT (exit code: -7):\n```\n{}\n```",
                    "出".repeat(1000)
                ),
            ));
        }
    }

    #[test]
    fn compaction_keeps_recent_exchanges_and_every_call_answered() {
        let mut messages = vec![
            ChatMessage::text(Role::System, "system".repeat(1000)),
            ChatMessage::text(Role::User, "task".repeat(1000)),
        ];
        exchange(&mut messages, 0, 2);
        messages.push(ChatMessage::text(Role::User, "guidance".repeat(1000)));
        exchange(&mut messages, 1, 1);
        exchange(&mut messages, 2, 1);
        let original = serde_json::to_value(&messages).unwrap();
        compact_with_policy(&mut messages, 2, false);
        let compacted = serde_json::to_value(&messages).unwrap();
        assert_eq!(compacted[0], original[0]);
        assert_eq!(compacted[1], original[1]);
        assert_eq!(
            compacted.as_array().unwrap()[6..],
            original.as_array().unwrap()[6..]
        );
        assert_eq!(compacted[2]["tool_calls"], original[2]["tool_calls"]);
        assert!(messages[2].content().unwrap().len() <= PROSE_BYTES + PROSE_SUFFIX.len());
        assert!(messages[2].content().unwrap().ends_with(PROSE_SUFFIX));
        assert!(
            messages[2]
                .reasoning_content()
                .unwrap()
                .ends_with(REASONING_SUFFIX)
        );
        assert!(
            messages[2].reasoning_content().unwrap().len()
                <= REASONING_BYTES + REASONING_SUFFIX.len()
        );
        for message in &messages[3..=4] {
            assert!(message.content().unwrap().starts_with(OUTPUT_PREFIX));
        }
        assert_eq!(compacted[5], original[5], "guidance is not command output");
        assert!(
            messages[3]
                .content()
                .unwrap()
                .starts_with("[output elided: exit -7, ")
        );
        assert!(
            messages[3]
                .content()
                .unwrap()
                .contains(&format!("{} bytes;", messages_bytes(&original[3])))
        );
        for (before, after) in original
            .as_array()
            .unwrap()
            .iter()
            .zip(compacted.as_array().unwrap())
        {
            assert_eq!(before.get("tool_call_id"), after.get("tool_call_id"));
            assert_eq!(before.get("tool_calls"), after.get("tool_calls"));
        }
        compact_with_policy(&mut messages, 2, false);
        assert_eq!(serde_json::to_value(&messages).unwrap(), compacted);
    }

    fn messages_bytes(message: &serde_json::Value) -> usize {
        message["content"].as_str().unwrap().len()
    }

    #[test]
    fn zero_window_compacts_fallback_and_preserves_reasoning_when_requested() {
        let mut messages = vec![
            ChatMessage::text(Role::System, "system"),
            ChatMessage::text(Role::User, "task"),
            ChatMessage::text(Role::Assistant, "prose".repeat(1000))
                .with_reasoning_content(Some("thinking".repeat(1000))),
            ChatMessage::text(
                Role::User,
                "COMMAND OUTPUT (exit code: 0):\n```\nhello\n```",
            ),
            ChatMessage::text(Role::Assistant, "no command"),
            ChatMessage::text(Role::User, NO_COMMAND_NUDGE),
        ];
        compact_with_policy(&mut messages, 0, true);
        assert_eq!(
            messages[2].reasoning_content(),
            Some("thinking".repeat(1000).as_str())
        );
        assert_eq!(messages[4].reasoning_content(), None);
        assert!(
            messages[3]
                .content()
                .unwrap()
                .starts_with("[output elided: exit 0, ")
        );
        assert!(
            messages[5]
                .content()
                .unwrap()
                .starts_with("[output elided: exit unknown, ")
        );
        let compacted = serde_json::to_value(&messages).unwrap();
        compact_with_policy(&mut messages, 0, true);
        assert_eq!(serde_json::to_value(&messages).unwrap(), compacted);
    }

    #[test]
    fn old_orchestrator_instructions_survive_beside_compacted_outputs() {
        let instructions = [
            format!(
                "{}\n{}",
                crate::pool::revision::REVISION_PREFIX,
                "fix empty inputs".repeat(100)
            ),
            format!("ORCHESTRATOR GUIDANCE:\n{}", "keep validation".repeat(100)),
            "STEER / ORCHESTRATOR GUIDANCE:\nrun the full suite".to_string(),
            "ORCHESTRATOR RESPONSE / GUIDANCE:\nuse the existing API".to_string(),
            "TURN EXTENSION REFUSED: wrap up now".to_string(),
            "VERIFICATION FAILED (exit 1) - fix these problems before completing:\nfailed test"
                .to_string(),
        ];
        let mut messages = vec![
            ChatMessage::text(Role::System, "system"),
            ChatMessage::text(Role::User, "task"),
            ChatMessage::text(Role::Assistant, "```bash\necho hello\n```"),
            ChatMessage::text(
                Role::User,
                "COMMAND OUTPUT (exit code: 0):\n```\nhello\n```",
            ),
        ];
        messages.extend(
            instructions
                .iter()
                .map(|text| ChatMessage::text(Role::User, text)),
        );
        exchange(&mut messages, 1, 1);
        compact_with_policy(&mut messages, 1, false);
        assert!(messages[3].content().unwrap().starts_with(OUTPUT_PREFIX));
        for (message, expected) in messages[4..].iter().zip(&instructions) {
            assert_eq!(message.content(), Some(expected.as_str()));
        }
        let compacted = serde_json::to_value(&messages).unwrap();
        compact_with_policy(&mut messages, 1, false);
        assert_eq!(serde_json::to_value(&messages).unwrap(), compacted);
    }

    #[test]
    fn a_window_larger_than_the_history_changes_nothing() {
        let mut messages = vec![ChatMessage::text(Role::User, "task")];
        exchange(&mut messages, 0, 1);
        let original = serde_json::to_value(&messages).unwrap();
        compact_with_policy(&mut messages, usize::MAX, false);
        assert_eq!(serde_json::to_value(&messages).unwrap(), original);
    }
}
