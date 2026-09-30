//! In-place conversation compaction, shared by implementer and reviewer turns.
//!
//! Keep the prompt, task and recent exchanges intact. Older calls and their ids
//! remain in order, but their text is replaced, not merely sliced, so the live
//! history and saved revisions no longer own the original output allocations.
//! The newest outputs share a byte budget, with at least four exchanges kept
//! verbatim. An explicit HISTORY_FULL_TURNS restores the fixed-turn window.

use crate::agent::{ChatMessage, Role};
use crate::config::env_parse;

use super::turn::{COMMAND_OUTPUT_PREFIX, NO_COMMAND_NUDGE, VERIFICATION_OUTPUT_PREFIX};

const DEFAULT_BUDGET_BYTES: usize = 160_000;
const MIN_VERBATIM_EXCHANGES: usize = 4;
const OUTPUT_KEEP_BYTES: usize = 4_000;

#[derive(Clone, Copy)]
enum KeepPolicy {
    Budget(usize),
    Turns(usize),
}

impl KeepPolicy {
    fn from_limits(full_turns: Option<usize>, budget_bytes: usize) -> Self {
        full_turns.map_or(Self::Budget(budget_bytes), Self::Turns)
    }
}
const OUTPUT_PREVIEW_BYTES: usize = 300;
const PROSE_BYTES: usize = 500;
const REASONING_BYTES: usize = 200;
const PROSE_SUFFIX: &str = " [prose elided]";
const REASONING_SUFFIX: &str = " [reasoning elided]";
const OUTPUT_PREFIX: &str = "[output elided: exit ";

pub(crate) fn compact_history(messages: &mut [ChatMessage]) {
    compact_with_policy(
        messages,
        KeepPolicy::from_limits(
            env_parse("HISTORY_FULL_TURNS"),
            env_parse("HISTORY_BUDGET_BYTES").unwrap_or(DEFAULT_BUDGET_BYTES),
        ),
        env_parse::<u8>("HISTORY_KEEP_ALL_REASONING") == Some(1),
    );
}

// Count original output sizes in stubs so repeated passes choose the same window.
fn output_bytes(message: &ChatMessage) -> usize {
    let Some(text) = message.content() else {
        return 0;
    };
    if message.role() != Role::Tool && !(message.role() == Role::User && is_command_output(text)) {
        return 0;
    }
    if let Some(stub) = text.strip_prefix(OUTPUT_PREFIX)
        && let Some((_, rest)) = stub.split_once(", ")
        && let Some((bytes, _)) = rest.split_once(" bytes; first lines: ")
        && let Ok(bytes) = bytes.parse::<usize>()
    {
        return bytes;
    }
    text.len()
}

fn is_command_output(text: &str) -> bool {
    text.starts_with(COMMAND_OUTPUT_PREFIX)
        || text.starts_with(OUTPUT_PREFIX)
        || text == NO_COMMAND_NUDGE
}

fn verbatim_cutoff(messages: &[ChatMessage], policy: KeepPolicy) -> usize {
    if let KeepPolicy::Turns(full_turns) = policy {
        return if full_turns == 0 {
            messages.len()
        } else {
            messages
                .iter()
                .enumerate()
                .rev()
                .filter(|(_, m)| m.role() == Role::Assistant)
                .nth(full_turns - 1)
                .map_or(0, |(i, _)| i)
        };
    }
    let KeepPolicy::Budget(budget) = policy else {
        unreachable!();
    };
    let mut bytes = 0usize;
    let mut kept = 0;
    let mut cutoff = messages.len();
    // All outputs between assistant messages belong to one exchange.
    for (i, message) in messages.iter().enumerate().rev() {
        bytes = bytes.saturating_add(output_bytes(message));
        if message.role() == Role::Assistant {
            if kept >= MIN_VERBATIM_EXCHANGES && bytes > budget {
                return cutoff;
            }
            kept += 1;
            cutoff = i;
        }
    }
    0
}

fn compact_with_policy(messages: &mut [ChatMessage], policy: KeepPolicy, keep_reasoning: bool) {
    let cutoff = verbatim_cutoff(messages, policy);
    let task = messages.iter().position(|m| m.role() == Role::User);
    for (i, message) in messages[..cutoff].iter_mut().enumerate() {
        if message.role() == Role::System || Some(i) == task {
            continue;
        }
        if let Some(text) = message.content() {
            let compacted = match message.role() {
                Role::Assistant => shorten(text, PROSE_BYTES, PROSE_SUFFIX),
                Role::Tool => output_stub(text),
                Role::User if is_command_output(text) => output_stub(text),
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
    // Small read-like outputs and already-bounded stubs stay whole.
    if text.len() <= OUTPUT_KEEP_BYTES {
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
                    "出".repeat(2000)
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
        compact_with_policy(&mut messages, KeepPolicy::Turns(2), false);
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
        compact_with_policy(&mut messages, KeepPolicy::Turns(2), false);
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
                format!(
                    "COMMAND OUTPUT (exit code: 0):\n```\n{}\n```",
                    "hello".repeat(1000)
                ),
            ),
            ChatMessage::text(Role::Assistant, "no command"),
            ChatMessage::text(Role::User, NO_COMMAND_NUDGE),
        ];
        compact_with_policy(&mut messages, KeepPolicy::Turns(0), true);
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
        assert_eq!(messages[5].content(), Some(NO_COMMAND_NUDGE));
        let compacted = serde_json::to_value(&messages).unwrap();
        compact_with_policy(&mut messages, KeepPolicy::Turns(0), true);
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
                format!(
                    "COMMAND OUTPUT (exit code: 0):\n```\n{}\n```",
                    "hello".repeat(1000)
                ),
            ),
        ];
        messages.extend(
            instructions
                .iter()
                .map(|text| ChatMessage::text(Role::User, text)),
        );
        exchange(&mut messages, 1, 1);
        compact_with_policy(&mut messages, KeepPolicy::Turns(1), false);
        assert!(messages[3].content().unwrap().starts_with(OUTPUT_PREFIX));
        for (message, expected) in messages[4..].iter().zip(&instructions) {
            assert_eq!(message.content(), Some(expected.as_str()));
        }
        let compacted = serde_json::to_value(&messages).unwrap();
        compact_with_policy(&mut messages, KeepPolicy::Turns(1), false);
        assert_eq!(serde_json::to_value(&messages).unwrap(), compacted);
    }

    #[test]
    fn a_window_larger_than_the_history_changes_nothing() {
        let mut messages = vec![ChatMessage::text(Role::User, "task")];
        exchange(&mut messages, 0, 1);
        let original = serde_json::to_value(&messages).unwrap();
        compact_with_policy(&mut messages, KeepPolicy::Turns(usize::MAX), false);
        assert_eq!(serde_json::to_value(&messages).unwrap(), original);
    }
}

#[cfg(test)]
mod budget_tests {
    use super::*;
    use crate::agent::{ToolCall, ToolCallFn};

    fn history(sizes: &[usize]) -> Vec<ChatMessage> {
        let mut messages = vec![
            ChatMessage::text(Role::System, "system"),
            ChatMessage::text(Role::User, "task"),
        ];
        for (turn, &size) in sizes.iter().enumerate() {
            let id = format!("budget_call_{turn}");
            messages.push(
                ChatMessage::assistant_with_tool_calls(
                    Some("prose".repeat(200)),
                    vec![ToolCall {
                        id: id.clone(),
                        r#type: "function".into(),
                        function: ToolCallFn {
                            name: "bash".into(),
                            arguments: format!(r#"{{"command":"echo {turn}"}}"#),
                        },
                    }],
                )
                .with_reasoning_content(Some("thinking".repeat(200))),
            );
            let prefix = "COMMAND OUTPUT (exit code: 0):\n```\n";
            assert!(size >= prefix.len() + 4);
            messages.push(ChatMessage::tool_result(
                id,
                format!("{prefix}{}\n```", "x".repeat(size - prefix.len() - 4)),
            ));
        }
        messages
    }

    #[test]
    fn many_small_outputs_keep_whole_exchanges() {
        let mut messages = history(&[1_000; 60]);
        let original = serde_json::to_value(&messages).unwrap();
        compact_with_policy(
            &mut messages,
            KeepPolicy::Budget(DEFAULT_BUDGET_BYTES),
            false,
        );
        assert_eq!(serde_json::to_value(&messages).unwrap(), original);
    }

    fn assert_idempotent(messages: &mut [ChatMessage], policy: KeepPolicy) {
        let once = serde_json::to_value(&*messages).unwrap();
        compact_with_policy(messages, policy, false);
        assert_eq!(serde_json::to_value(messages).unwrap(), once);
    }

    #[test]
    fn older_large_outputs_are_stubbed_but_newer_small_exchanges_survive() {
        let mut sizes = vec![20_000; 9];
        sizes.extend([1_000; 12]);
        let mut messages = history(&sizes);
        let original = serde_json::to_value(&messages).unwrap();
        let policy = KeepPolicy::Budget(DEFAULT_BUDGET_BYTES);
        compact_with_policy(&mut messages, policy, false);
        // 12 KB of recent outputs plus seven 20 KB outputs fit; eight do not.
        assert!(messages[3].content().unwrap().starts_with(OUTPUT_PREFIX));
        assert!(messages[5].content().unwrap().starts_with(OUTPUT_PREFIX));
        let after = serde_json::to_value(&messages).unwrap();
        assert_eq!(
            after.as_array().unwrap()[6..],
            original.as_array().unwrap()[6..]
        );
        assert_idempotent(&mut messages, policy);
    }

    #[test]
    fn mandatory_four_exchanges_count_against_the_budget() {
        let mut messages = history(&[20_000; 7]);
        let original = serde_json::to_value(&messages).unwrap();
        let policy = KeepPolicy::Budget(1_000);
        compact_with_policy(&mut messages, policy, false);
        for turn in 0..3 {
            assert!(
                messages[3 + turn * 2]
                    .content()
                    .unwrap()
                    .starts_with(OUTPUT_PREFIX)
            );
        }
        let after = serde_json::to_value(&messages).unwrap();
        assert_eq!(
            after.as_array().unwrap()[8..],
            original.as_array().unwrap()[8..]
        );
        assert_idempotent(&mut messages, policy);

        // The mandatory window costs 80 KB, so another 20 KB cannot fit in 90 KB.
        let mut messages = history(&[20_000; 7]);
        compact_with_policy(&mut messages, KeepPolicy::Budget(90_000), false);
        assert!(messages[7].content().unwrap().starts_with(OUTPUT_PREFIX));
    }

    #[test]
    fn fixed_turn_override_wins_and_can_keep_fewer_than_four() {
        let mut messages = history(&[5_000; 6]);
        let original = serde_json::to_value(&messages).unwrap();
        let policy = KeepPolicy::from_limits(Some(2), usize::MAX);
        compact_with_policy(&mut messages, policy, false);
        assert!(messages[9].content().unwrap().starts_with(OUTPUT_PREFIX));
        let after = serde_json::to_value(&messages).unwrap();
        assert_eq!(
            after.as_array().unwrap()[10..],
            original.as_array().unwrap()[10..]
        );
        assert_idempotent(&mut messages, policy);
    }

    #[test]
    fn small_outputs_survive_outside_window_and_large_preview_is_utf8_safe() {
        let mut messages = history(&[4_000, 4_001]);
        let small = messages[3].content().unwrap().to_string();
        let policy = KeepPolicy::Turns(0);
        compact_with_policy(&mut messages, policy, false);
        assert_eq!(messages[3].content(), Some(small.as_str()));
        assert!(messages[2].content().unwrap().ends_with(PROSE_SUFFIX));
        assert!(messages[5].content().unwrap().starts_with(OUTPUT_PREFIX));
        assert_idempotent(&mut messages, policy);
        let unicode = format!("COMMAND OUTPUT (exit code: -7):\n{}", "出".repeat(2_000));
        let stub = output_stub(&unicode).unwrap();
        assert!(stub.starts_with("[output elided: exit -7, "));
        assert!(stub.len() <= OUTPUT_PREVIEW_BYTES + 100);
        assert_eq!(output_stub(&stub), None);
    }

    #[test]
    fn budget_counts_all_tool_results_and_fallback_outputs_not_guidance() {
        let mut messages = history(&[10_000; 5]);
        // A second result in the fifth-oldest exchange tips it over 55 KB.
        messages.insert(
            4,
            ChatMessage::tool_result("extra".into(), "x".repeat(10_000)),
        );
        let original = serde_json::to_value(&messages).unwrap();
        let policy = KeepPolicy::Budget(55_000);
        compact_with_policy(&mut messages, policy, false);
        assert!(messages[3].content().unwrap().starts_with(OUTPUT_PREFIX));
        let after = serde_json::to_value(&messages).unwrap();
        assert_eq!(
            after.as_array().unwrap()[5..],
            original.as_array().unwrap()[5..]
        );
        assert_idempotent(&mut messages, policy);

        // User-role command outputs charge the same budget; instructions do not.
        let mut messages = history(&[20_000; 7]);
        for message in &mut messages {
            if message.role() == Role::Tool {
                *message = ChatMessage::text(Role::User, message.content().unwrap());
            }
        }
        messages.push(ChatMessage::text(Role::User, "guidance".repeat(20_000)));
        compact_with_policy(&mut messages, KeepPolicy::Budget(100_000), false);
        assert!(messages[5].content().unwrap().starts_with(OUTPUT_PREFIX));
        assert_eq!(messages[7].content().unwrap().len(), 20_000);
        assert_eq!(messages.last().unwrap().content().unwrap().len(), 160_000);
        assert_idempotent(&mut messages, KeepPolicy::Budget(100_000));
    }
}
