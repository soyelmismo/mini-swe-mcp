//! In-place conversation compaction, shared by implementer and reviewer turns.
//!
//! Keep the prompt, task and recent exchanges intact. Older calls and their ids
//! remain in order, but their text is replaced, not merely sliced, so the live
//! history and saved revisions no longer own the original output allocations.

use crate::agent::{ChatMessage, Role};
use crate::config::env_parse;

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
    let cutoff = if full_turns == 0 { messages.len() } else { cutoff };
    let task = messages.iter().position(|m| m.role() == Role::User);
    for (i, message) in messages[..cutoff].iter_mut().enumerate() {
        if message.role() == Role::System || Some(i) == task {
            continue;
        }
        if let Some(text) = message.content() {
            let compacted = match message.role() {
                Role::Assistant => shorten(text, PROSE_BYTES, PROSE_SUFFIX),
                Role::User | Role::Tool => output_stub(text),
                Role::System => None,
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
    Some(format!("{}{suffix}", &text[..text.floor_char_boundary(bytes)]))
}

fn output_stub(text: &str) -> Option<String> {
    // Recognize our bounded stub so reapplying compaction preserves byte counts.
    if text.starts_with(OUTPUT_PREFIX) && text.ends_with(']') && text.len() <= 400 {
        return None;
    }
    let exit = text
        .strip_prefix("COMMAND OUTPUT (exit code: ")
        .or_else(|| text.strip_prefix("VERIFICATION FAILED (exit "))
        .and_then(|rest| rest.split_once(')'))
        .and_then(|(code, _)| code.parse::<i32>().ok());
    let exit = exit.map_or_else(|| "unknown".to_string(), |code| code.to_string());
    let preview = &text[..text.floor_char_boundary(OUTPUT_PREVIEW_BYTES)];
    Some(format!("{OUTPUT_PREFIX}{exit}, {} bytes; first lines: {preview}]", text.len()))
}
