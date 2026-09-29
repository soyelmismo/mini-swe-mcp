use regex::Regex;
use std::collections::{BTreeMap, HashSet};
use std::sync::LazyLock;

use super::types::{
    BashArgs, ChatCompletionResponse, LlmResponse, MAX_STREAMED_CONTENT_BYTES,
    MAX_TOOL_ARGUMENT_BYTES, SSE_BUFFER_HINT_BYTES, StreamChunk, StreamToolCall, ToolCall,
    ToolCallFn, generate_call_id,
};

/// Recover a bash command from the first ```bash / ```sh fenced block.
///
/// Uses a single `find()` on a capture-free pattern and slices the body out
/// of the match by hand, which is materially cheaper than `captures()`.
/// A literally empty body (```` ```bash\n``` ````) yields `None`, because
/// the pattern requires a newline before the closing fence; a
/// whitespace-only body yields `Some("")`.
pub(crate) fn extract_command(text: &str) -> Option<String> {
    let full = BASH_BLOCK_RE.find(text)?.as_str();
    let open_line_end = full.find('\n')?;
    let close_start = full.len() - "\n```".len();
    if close_start <= open_line_end {
        return None;
    }
    Some(full[open_line_end + 1..close_start].trim().to_string())
}

/// Pattern used to recover a shell command from a ```bash/```sh fenced block
/// when the model did not use the structured `bash` tool call.
///
/// The pattern is a compile-time constant (no interpolation), so the compiled
/// program is memoized process-wide: it is built at most once, no matter how
/// many `AgentRunner`s (one per worker) exist, and the compiled automaton's
/// lazy DFA cache is shared instead of duplicated per worker.
static BASH_BLOCK_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)```(?:bash|sh)[ \t\r\n]*\n.*?\n```").expect("bash block regex must compile")
});

/// A tool call being assembled from streaming deltas.
#[derive(Debug, Default)]
pub(crate) struct StreamedToolCall {
    pub(crate) id: String,
    pub(crate) name: String,
    pub(crate) arguments: String,
    /// Set when the call blew past [`MAX_TOOL_ARGUMENT_BYTES`] or the provider
    /// used the same `index` for two different ids. Such calls are dropped
    /// rather than replayed into the conversation history.
    pub(crate) malformed: bool,
}

impl StreamedToolCall {
    /// `true` for a padding / never-populated slot: no id, no name, no args.
    pub(crate) fn is_placeholder(&self) -> bool {
        self.id.trim().is_empty() && self.name.trim().is_empty() && self.arguments.trim().is_empty()
    }
}

/// Streaming state machine shared by the SSE reader and its tests.
#[derive(Debug, Default)]
pub(crate) struct SseAccumulator {
    pub(crate) content: String,
    /// Keyed by the provider's `index`, *not* positional. A sparse index
    /// (`index: 3` on the first frame) used to resize a `Vec` and fabricate
    /// empty placeholder tool calls that were later fed back to the model with
    /// duplicated ids — and whose empty `arguments` shadowed the real command.
    ///
    /// `BTreeMap` keeps deterministic (index-ordered) iteration and collapses
    /// repeated `index` values from re-indexing proxies / retries.
    pub(crate) tools: BTreeMap<usize, StreamedToolCall>,
    /// Set when `content` hit [`MAX_STREAMED_CONTENT_BYTES`].
    pub(crate) content_capped: bool,
    /// Count of frames that were not valid UTF-8 (decoded lossily).
    pub(crate) invalid_utf8_lines: usize,
}

/// Outcome of feeding one complete SSE frame to the accumulator.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum FrameOutcome {
    /// Ordinary frame: consumed, keep reading.
    Consumed,
    /// `data: [DONE]` sentinel: stop reading the body.
    Done,
}

impl SseAccumulator {
    /// Append raw chunk bytes and drain every *complete* frame they finish.
    ///
    /// The buffer only ever holds bytes that have not yet been framed, so the
    /// newline search is bounded by the tail of the buffer instead of the whole
    /// accumulated stream: each byte is scanned at most once, even when a frame
    /// is split across thousands of one-byte TCP segments.
    pub(crate) fn push(&mut self, bytes: &[u8], buffer: &mut Vec<u8>) -> Option<FrameOutcome> {
        if bytes.is_empty() {
            return None;
        }
        if buffer.is_empty() {
            buffer.reserve(SSE_BUFFER_HINT_BYTES);
        }
        buffer.extend_from_slice(bytes);

        let mut outcome = None;
        let mut start = 0usize;
        // Search for the next newline with a single forward pass. `start` only
        // ever moves right, so a byte is never re-scanned; and we stop at the
        // first newline-free remainder instead of rescanning it per chunk.
        while let Some(rel_pos) = buffer[start..].iter().position(|&b| b == b'\n') {
            let pos = start + rel_pos;
            let raw_line = &buffer[start..pos];
            start = pos + 1;

            match self.handle_line(raw_line) {
                FrameOutcome::Consumed => {}
                FrameOutcome::Done => {
                    outcome = Some(FrameOutcome::Done);
                    break;
                }
            }
        }

        // Discard the consumed prefix. `Vec::drain(..start)` memmoves the whole
        // *remainder* left, so skip it entirely when nothing is left and only
        // pay for it when an unterminated tail survives. After `[DONE]` nothing
        // in the buffer matters any more, so drop it outright.
        if outcome == Some(FrameOutcome::Done) || start >= buffer.len() {
            buffer.clear();
        } else if start > 0 {
            buffer.drain(..start);
        }

        outcome
    }

    /// Frame one newline-delimited line: decode, filter, parse, accumulate.
    pub(crate) fn handle_line(&mut self, raw_line: &[u8]) -> FrameOutcome {
        let trimmed: &str = match std::str::from_utf8(raw_line) {
            Ok(s) => s.trim(),
            Err(e) => {
                self.invalid_utf8_lines += 1;
                tracing::warn!(
                    valid_up_to = e.valid_up_to(),
                    error_len = e.error_len(),
                    line_len = raw_line.len(),
                    "SSE line is not valid UTF-8; decoding lossily"
                );
                return self.handle_text_line(String::from_utf8_lossy(raw_line).trim());
            }
        };

        self.handle_text_line(trimmed)
    }

    /// Apply the shared SSE filter/parse/accumulate path to one trimmed line.
    pub(crate) fn handle_text_line(&mut self, trimmed: &str) -> FrameOutcome {
        if trimmed.is_empty() || trimmed.starts_with(':') {
            return FrameOutcome::Consumed;
        }
        let Some(data) = trimmed.strip_prefix("data:") else {
            return FrameOutcome::Consumed;
        };
        let data = data.trim();
        if data == "[DONE]" {
            return FrameOutcome::Done;
        }

        if let Ok(chunk) = serde_json::from_str::<StreamChunk>(data)
            && let Some(choice) = chunk.choices.first()
        {
            if let Some(c) = &choice.delta.content {
                self.push_content(c);
            }
            for tc in &choice.delta.tool_calls {
                self.accumulate_tool_call(tc);
            }
        }
        FrameOutcome::Consumed
    }

    /// Append streamed content, respecting [`MAX_STREAMED_CONTENT_BYTES`].
    pub(crate) fn push_content(&mut self, text: &str) {
        if self.content.len() >= MAX_STREAMED_CONTENT_BYTES {
            if !self.content_capped {
                self.content_capped = true;
                tracing::warn!(
                    limit = MAX_STREAMED_CONTENT_BYTES,
                    "Streamed assistant content exceeded the retention budget; truncating"
                );
            }
            return;
        }
        let room = MAX_STREAMED_CONTENT_BYTES - self.content.len();
        if text.len() <= room {
            self.content.push_str(text);
            return;
        }
        // Snap to a char boundary so we never store a partial code point.
        let cut = self.content.len() + text.floor_char_boundary(room);
        self.content.push_str(&text[..cut]);
        self.content_capped = true;
        tracing::warn!(
            limit = MAX_STREAMED_CONTENT_BYTES,
            "Streamed assistant content exceeded the retention budget; truncating"
        );
    }

    /// Fold one streamed `tool_calls` delta into the index-keyed map.
    pub(crate) fn accumulate_tool_call(&mut self, tc: &StreamToolCall) {
        let entry = self.tools.entry(tc.index).or_default();
        if entry.malformed {
            return;
        }

        if let Some(id) = &tc.id {
            if !entry.id.is_empty() && &entry.id != id {
                tracing::warn!(
                    index = tc.index,
                    "Conflicting tool_call ids for the same index; dropping the call"
                );
                entry.malformed = true;
                return;
            }
            entry.id = id.clone();
        }

        if let Some(fn_info) = &tc.function {
            if let Some(name) = &fn_info.name
                && !name.is_empty()
            {
                entry.name.push_str(name);
            }
            if let Some(args) = &fn_info.arguments
                && !args.is_empty()
            {
                if entry.arguments.len() + args.len() > MAX_TOOL_ARGUMENT_BYTES {
                    tracing::warn!(
                        index = tc.index,
                        limit = MAX_TOOL_ARGUMENT_BYTES,
                        "Streamed tool_call arguments exceeded the retention budget; dropping the call"
                    );
                    entry.malformed = true;
                    return;
                }
                entry.arguments.push_str(args);
            }
        }
    }

    /// Compact the index-keyed map into provider- and history-compatible
    /// `ToolCall`s, dropping placeholders and malformed entries and guaranteeing
    /// unique, non-empty ids. Returns the calls plus the set of ids actually
    /// emitted (so the caller never hands back an id absent from history).
    pub(crate) fn finalize_from(
        tools: &BTreeMap<usize, StreamedToolCall>,
    ) -> (Vec<ToolCall>, HashSet<String>) {
        let mut tcs = Vec::with_capacity(tools.len());
        let mut seen_ids: HashSet<String> = HashSet::new();
        for entry in tools.values() {
            if entry.malformed || entry.is_placeholder() {
                continue;
            }
            let mut id = entry.id.trim().to_string();
            if id.is_empty() || !seen_ids.insert(id.clone()) {
                if !id.is_empty() {
                    tracing::warn!(
                        original_id = %id,
                        "Duplicate tool_call id in stream; generated a unique replacement"
                    );
                }
                id = generate_call_id();
                seen_ids.insert(id.clone());
            }
            let name = if entry.name.trim().is_empty() {
                "bash".to_string()
            } else {
                entry.name.clone()
            };
            tcs.push(ToolCall {
                id,
                r#type: "function".to_string(),
                function: ToolCallFn {
                    name,
                    arguments: entry.arguments.clone(),
                },
            });
        }
        (tcs, seen_ids)
    }

    /// Turn the accumulated stream into the [`LlmResponse`] the agent loop consumes.
    ///
    /// The command comes from the structured `bash` tool call when the model
    /// emitted one, and falls back to a fenced code block in the content
    /// otherwise. The reported tool-call id is only kept when it survived
    /// streaming, so the next request never carries a dangling id.
    pub(crate) fn finish(self) -> LlmResponse {
        let SseAccumulator {
            content,
            tools,
            invalid_utf8_lines,
            ..
        } = self;

        if invalid_utf8_lines > 0 {
            tracing::warn!(
                invalid_utf8_lines,
                "Streamed response contained frames that were not valid UTF-8; decoded lossily"
            );
        }

        let bash_tc = tools.values().find(|tc| {
            !tc.malformed
                && (tc.name == "bash" || tc.name.trim().is_empty())
                && !tc.arguments.trim().is_empty()
        });

        let command = bash_tc
            .and_then(|tc| {
                serde_json::from_str::<BashArgs>(&tc.arguments)
                    .map(|a| a.command)
                    .ok()
                    .or_else(|| {
                        serde_json::from_str::<serde_json::Value>(&tc.arguments)
                            .ok()
                            .and_then(|v| {
                                v.get("command")
                                    .or_else(|| v.get("cmd"))
                                    .and_then(|c| c.as_str())
                                    .map(|s| s.to_string())
                            })
                    })
            })
            .or_else(|| extract_command(&content));

        let (finalized, known_ids) = SseAccumulator::finalize_from(&tools);
        let tool_calls = (!finalized.is_empty()).then_some(finalized);
        let tool_call_id = tool_calls.as_ref().and_then(|tcs| {
            bash_tc.and_then(|tc| {
                let candidate = if tc.id.trim().is_empty() {
                    tcs.iter()
                        .find(|t| t.function.arguments == tc.arguments)
                        .map(|t| t.id.clone())
                } else {
                    Some(tc.id.trim().to_string())
                };
                candidate.filter(|id| known_ids.contains(id))
            })
        });

        LlmResponse {
            content,
            command,
            tool_calls,
            tool_call_id,
            invalid_utf8_lines,
        }
    }

    /// Handle non-streaming JSON response fallback.
    pub(crate) fn handle_non_stream_fallback(&mut self, buffer: &[u8]) {
        if self.content.is_empty()
            && self.tools.is_empty()
            && !buffer.is_empty()
            && let Ok(result) = serde_json::from_slice::<ChatCompletionResponse>(buffer)
            && let Some(choice) = result.choices.first()
        {
            self.push_content(choice.message.content.as_deref().unwrap_or(""));
            for (n, tc) in choice.message.tool_calls.iter().enumerate() {
                let entry = self.tools.entry(n).or_default();
                entry.id = tc.id.clone();
                entry.name = tc.function.name.clone();
                entry.arguments = tc.function.arguments.clone();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::types::StreamFunction;

    fn sse_body(frames: &[&str]) -> Vec<u8> {
        let mut v = Vec::new();
        for f in frames {
            v.extend_from_slice(format!("data: {f}\n\n").as_bytes());
        }
        v.extend_from_slice(b"data: [DONE]\n\n");
        v
    }

    fn tc(
        index: usize,
        id: Option<&str>,
        name: Option<&str>,
        args: Option<&str>,
    ) -> StreamToolCall {
        let function = if name.is_some() || args.is_some() {
            Some(StreamFunction {
                name: name.map(str::to_string),
                arguments: args.map(str::to_string),
            })
        } else {
            None
        };
        StreamToolCall {
            index,
            id: id.map(str::to_string),
            function,
        }
    }

    #[test]
    fn push_clears_buffer_when_fully_consumed() {
        let mut acc = SseAccumulator::default();
        let mut buffer = Vec::new();
        let body = sse_body(&[r#"{"choices":[{"delta":{"content":"x"}}]}"#]);
        assert_eq!(acc.push(&body, &mut buffer), Some(FrameOutcome::Done));
        assert!(buffer.is_empty(), "fully consumed buffer must be cleared");
    }

    #[test]
    fn push_keeps_only_the_unterminated_remainder() {
        let mut acc = SseAccumulator::default();
        let mut buffer = Vec::new();
        let head = br#"data: {"choices":[{"delta":{"content":"a"}}]}"#;
        let mut chunk = head.to_vec();
        chunk.extend_from_slice(b"\n\npar");
        acc.push(&chunk, &mut buffer);
        assert_eq!(acc.content, "a");
        assert_eq!(buffer, b"par", "only the unframed tail is retained");
    }

    #[test]
    fn byte_at_a_time_large_body_is_exact() {
        let text = "x".repeat(5000);
        let frame = serde_json::json!({"choices":[{"delta":{"content":text}}]}).to_string();
        let body = sse_body(&[&frame]);
        let mut acc = SseAccumulator::default();
        let mut buffer = Vec::new();
        let mut done = false;
        for b in &body {
            if acc.push(&[*b], &mut buffer) == Some(FrameOutcome::Done) {
                done = true;
                break;
            }
        }
        assert!(done, "[DONE] must terminate the stream");
        assert_eq!(acc.content.len(), 5000, "content must reassemble exactly");
        assert!(buffer.is_empty(), "no tail may survive [DONE]");
    }

    #[test]
    fn unparsable_frame_is_skipped_and_stream_continues() {
        let body = sse_body(&["{not json", r#"{"choices":[{"delta":{"content":"ok"}}]}"#]);
        let mut acc = SseAccumulator::default();
        let mut buffer = Vec::new();
        let mut done = false;
        for b in &body {
            if acc.push(&[*b], &mut buffer) == Some(FrameOutcome::Done) {
                done = true;
                break;
            }
        }
        assert!(done);
        assert_eq!(acc.content, "ok");
        assert_eq!(acc.invalid_utf8_lines, 0);
    }

    #[test]
    fn invalid_utf8_frame_is_counted_not_dropped() {
        let mut line = br#"data: {"choices":[{"delta":{"content":""#.to_vec();
        line.extend_from_slice(&[0xFF, 0xFE]);
        line.extend_from_slice(br#""}}]}"#);
        line.extend_from_slice(b"\n\ndata: [DONE]\n\n");
        let mut acc = SseAccumulator::default();
        let mut buffer = Vec::new();
        acc.push(&line, &mut buffer);
        assert_eq!(acc.invalid_utf8_lines, 1, "corruption must be counted");
        assert!(
            acc.content.contains('\u{FFFD}'),
            "corrupted bytes must surface as U+FFFD: {:?}",
            acc.content
        );
    }

    #[test]
    fn push_content_respects_cap_and_char_boundaries() {
        let mut acc = SseAccumulator::default();
        acc.push_content(&"€".repeat(MAX_STREAMED_CONTENT_BYTES));
        assert!(acc.content.len() <= MAX_STREAMED_CONTENT_BYTES);
        assert!(acc.content_capped, "overflow must be flagged");
        assert!(!acc.content.contains('\u{FFFD}'), "no partial code point");
        assert!(acc.content.is_char_boundary(acc.content.len()));
    }

    #[test]
    fn push_content_ignores_deltas_after_the_cap() {
        let mut acc = SseAccumulator::default();
        acc.push_content(&"a".repeat(MAX_STREAMED_CONTENT_BYTES));
        let before = acc.content.len();
        acc.push_content("more text");
        assert_eq!(acc.content.len(), before, "no growth past the cap");
    }

    #[test]
    fn sparse_index_does_not_fabricate_placeholders() {
        let mut acc = SseAccumulator::default();
        acc.accumulate_tool_call(&tc(3, Some("r"), Some("bash"), Some(r#"{"command":"ls"}"#)));
        let (tcs, ids) = SseAccumulator::finalize_from(&acc.tools);
        assert_eq!(tcs.len(), 1, "no phantom calls may be emitted: {tcs:?}");
        assert_eq!(tcs[0].id, "r");
        assert_eq!(tcs[0].function.arguments, r#"{"command":"ls"}"#);
        assert!(ids.contains("r"));
    }

    #[test]
    fn empty_entry_is_filtered_as_placeholder() {
        let mut acc = SseAccumulator::default();
        acc.tools.insert(0, Default::default());
        let (tcs, _) = SseAccumulator::finalize_from(&acc.tools);
        assert!(tcs.is_empty(), "placeholder must not reach history");
    }

    #[test]
    fn duplicate_ids_are_made_unique() {
        let mut acc = SseAccumulator::default();
        acc.accumulate_tool_call(&tc(
            0,
            Some("dup"),
            Some("bash"),
            Some(r#"{"command":"a"}"#),
        ));
        acc.accumulate_tool_call(&tc(
            1,
            Some("dup"),
            Some("bash"),
            Some(r#"{"command":"b"}"#),
        ));
        let (tcs, ids) = SseAccumulator::finalize_from(&acc.tools);
        assert_eq!(tcs.len(), 2);
        assert_ne!(tcs[0].id, tcs[1].id, "ids must be unique");
        assert_eq!(ids.len(), 2);
        assert!(tcs.iter().all(|c| !c.id.is_empty()));
    }

    #[test]
    fn missing_id_is_generated() {
        let mut acc = SseAccumulator::default();
        acc.accumulate_tool_call(&tc(0, None, Some("bash"), Some(r#"{"command":"a"}"#)));
        let (tcs, _) = SseAccumulator::finalize_from(&acc.tools);
        assert_eq!(tcs.len(), 1);
        assert!(tcs[0].id.starts_with("call_") && tcs[0].id.len() == 13);
    }

    #[test]
    fn missing_name_defaults_to_bash() {
        let mut acc = SseAccumulator::default();
        acc.accumulate_tool_call(&StreamToolCall {
            index: 0,
            id: Some("i".into()),
            function: Some(StreamFunction {
                name: None,
                arguments: Some(r#"{"command":"a"}"#.into()),
            }),
        });
        let (tcs, _) = SseAccumulator::finalize_from(&acc.tools);
        assert_eq!(tcs[0].function.name, "bash");
    }

    #[test]
    fn repeated_index_accumulates_arguments() {
        let mut acc = SseAccumulator::default();
        acc.accumulate_tool_call(&tc(0, Some("c"), Some("bash"), Some(r#"{"comm"#)));
        acc.accumulate_tool_call(&tc(0, None, None, Some(r#"and":"pwd"}"#)));
        let (tcs, _) = SseAccumulator::finalize_from(&acc.tools);
        assert_eq!(tcs.len(), 1);
        assert_eq!(tcs[0].function.arguments, r#"{"command":"pwd"}"#);
    }

    #[test]
    fn conflicting_ids_for_one_index_mark_call_malformed() {
        let mut acc = SseAccumulator::default();
        acc.accumulate_tool_call(&tc(0, Some("a"), Some("bash"), Some("{}")));
        acc.accumulate_tool_call(&tc(0, Some("b"), None, None));
        let (tcs, _) = SseAccumulator::finalize_from(&acc.tools);
        assert!(tcs.is_empty(), "ambiguous call must be dropped: {tcs:?}");
    }

    #[test]
    fn oversized_arguments_mark_call_malformed() {
        let mut acc = SseAccumulator::default();
        let chunk = "a".repeat(1024);
        let rounds = MAX_TOOL_ARGUMENT_BYTES / chunk.len() + 2;
        for _ in 0..rounds {
            acc.accumulate_tool_call(&tc(0, Some("x"), Some("bash"), Some(&chunk)));
        }
        let entry = acc.tools.get(&0).expect("entry exists");
        assert!(entry.malformed, "call must be flagged malformed");
        assert!(entry.arguments.len() <= MAX_TOOL_ARGUMENT_BYTES);
        let (tcs, _) = SseAccumulator::finalize_from(&acc.tools);
        assert!(tcs.is_empty(), "malformed call must not reach history");
    }

    #[test]
    fn malformed_call_ignores_later_deltas() {
        let mut acc = SseAccumulator::default();
        let chunk = "a".repeat(MAX_TOOL_ARGUMENT_BYTES + 1);
        acc.accumulate_tool_call(&tc(0, Some("x"), Some("bash"), Some(&chunk)));
        assert!(acc.tools.get(&0).expect("entry").malformed);
        acc.accumulate_tool_call(&tc(0, Some("x"), Some("bash"), Some("{}")));
        let entry = acc.tools.get(&0).expect("entry");
        assert!(entry.malformed, "must stay malformed");
        assert!(entry.arguments.len() <= MAX_TOOL_ARGUMENT_BYTES);
    }

    #[test]
    fn finalize_orders_by_provider_index() {
        let mut acc = SseAccumulator::default();
        acc.accumulate_tool_call(&tc(5, Some("five"), Some("bash"), Some("{}")));
        acc.accumulate_tool_call(&tc(1, Some("one"), Some("bash"), Some("{}")));
        let (tcs, _) = SseAccumulator::finalize_from(&acc.tools);
        assert_eq!(
            tcs.iter().map(|c| c.id.as_str()).collect::<Vec<_>>(),
            vec!["one", "five"]
        );
    }
}
