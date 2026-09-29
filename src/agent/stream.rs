use regex::Regex;
use std::collections::{BTreeMap, HashSet};
use std::sync::LazyLock;

use super::types::{
    BashArgs, ChatCompletionResponse, LlmResponse, MAX_SSE_FRAME_BYTES, MAX_STREAMED_CONTENT_BYTES,
    MAX_TOOL_ARGUMENT_BYTES, SSE_BUFFER_HINT_BYTES, StreamChunk, StreamToolCall, ToolCall,
    ToolCallFn, generate_call_id,
};

/// Recover a bash command from the first ```bash / ```sh fenced block.
///
/// Uses a single `find()` on a capture-free pattern and slices the body out by
/// hand, materially cheaper than `captures()`. A literally empty body
/// (```` ```bash\n``` ````) yields `None` (the pattern requires a newline before
/// the closing fence); a whitespace-only body yields `Some("")`.
pub(crate) fn extract_command(text: &str) -> Option<String> {
    let full = BASH_BLOCK_RE.find(text)?.as_str();
    let open_line_end = full.find('\n')?;
    let close_start = full.len() - "\n```".len();
    if close_start <= open_line_end {
        return None;
    }
    Some(full[open_line_end + 1..close_start].trim().to_string())
}

/// Pattern recovering a shell command from a ```bash/```sh fenced block when
/// the model did not use the structured `bash` tool call.
///
/// A compile-time constant (no interpolation), so the compiled program is
/// memoized process-wide: built once regardless of how many `AgentRunner`s
/// exist, and its lazy DFA cache is shared instead of duplicated per worker.
static BASH_BLOCK_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)```(?:bash|sh)[ \t\r\n]*\n.*?\n```").expect("bash block regex must compile")
});

/// The `data:` field name of the Server-Sent Events wire format, as bytes.
const DATA_FIELD: &[u8] = b"data:";

/// The end-of-stream sentinel carried in a `data:` field.
const DONE_SENTINEL: &[u8] = b"[DONE]";

/// String view of [`DONE_SENTINEL`] for the post-decode parity check.
const DONE_SENTINEL_STR: &str = "[DONE]";

/// Classify one raw (undecoded) SSE line and return the bytes of its payload.
///
/// Returns `None` for every line carrying no `data` payload — blank lines, SSE
/// comments (leading `:`), and fields such as `event:`, `id:` or `retry:` — so
/// the caller skips them without decoding or allocating. For a `data:` line it
/// returns the payload with leading/trailing ASCII whitespace removed, matching
/// the spec's single optional space after the colon.
///
/// Operating on `&[u8]` keeps the filter allocation-free and lets the caller
/// run it *before* UTF-8 validation, so a keep-alive comment costs one
/// comparison instead of a decode.
fn data_field(raw_line: &[u8]) -> Option<&[u8]> {
    let line = trim_ascii(raw_line);
    if line.is_empty() || line[0] == b':' {
        return None;
    }
    // The field name is `data` followed by a colon; anything else is ignored.
    let payload = line.strip_prefix(DATA_FIELD)?;
    Some(trim_ascii(payload))
}

/// Trim ASCII whitespace from both ends of a byte slice.
///
/// SSE framing only introduces ASCII spaces/tabs/CR, and a `&[u8]` cannot carry
/// the full Unicode whitespace set, so this is exact for the lines the spec
/// allows. A no-op for the common already-trimmed case, which the fast path
/// detects in four comparisons.
#[inline]
fn trim_ascii(mut bytes: &[u8]) -> &[u8] {
    // Leading whitespace: skip while the front is ASCII whitespace.
    while let [first, rest @ ..] = bytes {
        if first.is_ascii_whitespace() {
            bytes = rest;
        } else {
            break;
        }
    }
    // Trailing whitespace: find the last non-whitespace byte, then slice.
    let mut end = bytes.len();
    while end > 0 && bytes[end - 1].is_ascii_whitespace() {
        end -= 1;
    }
    &bytes[..end]
}

/// Append the UTF-8-lossy decoding of `bytes` to `out` without allocating a
/// temporary `String`.
///
/// Equivalent to `String::from_utf8_lossy(bytes)` but writes byte-for-byte:
/// each maximal valid subsequence is copied verbatim and each maximal invalid
/// one collapses to a single U+FFFD. `out` must be empty; it is cleared and
/// grown in place.
fn lossy_decode_into(bytes: &[u8], out: &mut Vec<u8>) {
    out.clear();
    out.reserve(bytes.len() + REPLACEMENT_CHAR.len());
    let mut rest = bytes;
    loop {
        match std::str::from_utf8(rest) {
            Ok(valid) => {
                out.extend_from_slice(valid.as_bytes());
                return;
            }
            Err(err) => {
                let (valid, after_valid) = rest.split_at(err.valid_up_to());
                out.extend_from_slice(valid);
                // One replacement char for the invalid run. A truncated trailing
                // sequence (`error_len() == None`) leaves nothing to append.
                out.extend_from_slice(REPLACEMENT_CHAR);
                match err.error_len() {
                    Some(len) => rest = &after_valid[len..],
                    None => return,
                }
            }
        }
    }
}

/// UTF-8 encoding of U+FFFD, the substitution character the lossy decoder emits.
const REPLACEMENT_CHAR: &[u8] = "\u{FFFD}".as_bytes();

/// A tool call being assembled from streaming deltas.
#[derive(Debug, Default)]
pub(crate) struct StreamedToolCall {
    pub(crate) id: String,
    pub(crate) name: String,
    pub(crate) arguments: String,
    /// Set when the call blew past [`MAX_TOOL_ARGUMENT_BYTES`] and is therefore
    /// unusable; such calls are dropped rather than replayed into history.
    ///
    /// A *conflicting id* on a re-used `index` no longer lands here: that is a
    /// provider numbering quirk, not corruption, so the delta is redirected to
    /// a fresh slot. See [`SseAccumulator::accumulate_tool_call`].
    pub(crate) malformed: bool,
}

impl StreamedToolCall {
    /// `true` for a never-populated slot: no id, no name, no args.
    pub(crate) fn is_placeholder(&self) -> bool {
        self.id.trim().is_empty() && self.name.trim().is_empty() && self.arguments.trim().is_empty()
    }
}

/// Streaming state machine shared by the SSE reader and its tests.
#[derive(Debug, Default)]
pub(crate) struct SseAccumulator {
    pub(crate) content: String,
    pub(crate) reasoning_content: String,
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
    /// Set when `reasoning_content` hit [`MAX_STREAMED_CONTENT_BYTES`].
    pub(crate) reasoning_capped: bool,
    /// Frames that were not valid UTF-8 (decoded lossily).
    pub(crate) invalid_utf8_lines: usize,
    /// Reusable decode buffer for the lossy UTF-8 path, so a corrupting stream
    /// does not allocate one `String` per frame. Never escapes the accumulator.
    lossy_scratch: Vec<u8>,
    /// Latch so the unframed-tail cap is reported once per stream, not once per
    /// offending chunk (a runaway stream would otherwise log in a hot loop).
    frame_cap_logged: bool,
    /// Ignore bytes until the next newline after an oversized line.
    discarding_line: bool,
    /// Latch so an unparsable `data:` frame is warned about once per stream
    /// rather than once per frame.
    parse_error_logged: bool,
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
    /// Frame chunk bytes line by line. Incomplete lines are retained up to the
    /// cap; oversized lines are ignored through their terminating newline.
    /// Complete lines in a single chunk are borrowed directly, without copying
    /// the whole chunk into the framing buffer.
    pub(crate) fn push(&mut self, mut bytes: &[u8], buffer: &mut Vec<u8>) -> Option<FrameOutcome> {
        while !bytes.is_empty() {
            if self.discarding_line {
                let pos = bytes.iter().position(|&b| b == b'\n')?;
                bytes = &bytes[pos + 1..];
                self.discarding_line = false;
                continue;
            }

            let newline = bytes.iter().position(|&b| b == b'\n');
            let (line, rest) = match newline {
                Some(pos) => (&bytes[..pos], &bytes[pos + 1..]),
                None => (bytes, &bytes[bytes.len()..]),
            };
            if buffer.len().saturating_add(line.len()) > MAX_SSE_FRAME_BYTES {
                buffer.clear();
                if !self.frame_cap_logged {
                    self.frame_cap_logged = true;
                    tracing::warn!(
                        limit = MAX_SSE_FRAME_BYTES,
                        "SSE frame exceeded the unframed-tail budget; dropping the partial frame and resyncing"
                    );
                }
                // If this chunk does not end the oversized line, skip future
                // chunks too: their suffix could otherwise look like `data:`.
                self.discarding_line = newline.is_none();
            } else if newline.is_some() {
                let outcome = if buffer.is_empty() {
                    self.handle_line(line)
                } else {
                    buffer.extend_from_slice(line);
                    let outcome = self.handle_line(buffer);
                    buffer.clear();
                    outcome
                };
                if outcome == FrameOutcome::Done {
                    return Some(FrameOutcome::Done);
                }
            } else {
                if buffer.is_empty() {
                    buffer.reserve(SSE_BUFFER_HINT_BYTES);
                }
                buffer.extend_from_slice(line);
            }
            bytes = rest;
        }
        None
    }

    /// Frame one newline-delimited line: decode, filter, parse, accumulate.
    ///
    /// Framing is done purely on `&[u8]`. The field filter only recognises
    /// three cheap byte patterns (empty line, `:` comment, the ASCII `data:`
    /// field name), so a line is classified *before* any UTF-8 work:
    /// keep-alive comments and unknown fields are rejected without ever being
    /// validated or copied. Only a `data:` line — the payload — reaches the
    /// decoder, borrowed in place, so no intermediate `String` is materialised
    /// on the happy path.
    pub(crate) fn handle_line(&mut self, raw_line: &[u8]) -> FrameOutcome {
        // Cheap byte classification first: skip blanks, SSE comments (`:`) and
        // any field other than `data` without decoding.
        let Some(payload) = data_field(raw_line) else {
            return FrameOutcome::Consumed;
        };

        // `[DONE]` is a fixed ASCII token, matched on bytes and never decoded.
        if payload == DONE_SENTINEL {
            return FrameOutcome::Done;
        }

        // Validate once and branch: a valid payload is borrowed straight out of
        // the caller's buffer, and the error is already in hand on the slow
        // path. Validating twice (to recover the `Utf8Error` via `unwrap_err`)
        // re-scanned the whole line for nothing.
        let err = match std::str::from_utf8(payload) {
            Ok(text) => return self.handle_payload(text),
            Err(err) => err,
        };

        // Slow path: count, log, then decode lossily. The decode reuses
        // `self.lossy_scratch`'s capacity across frames, so a stream that keeps
        // sending corrupted lines allocates once and only regrows when a
        // *larger* corrupted line arrives — instead of one fresh `String` per
        // line, as `String::from_utf8_lossy` would. Log against `payload`, the
        // region actually being decoded, so the reported offsets line up with
        // the lossy substitution that follows.
        self.invalid_utf8_lines += 1;
        tracing::warn!(
            valid_up_to = err.valid_up_to(),
            error_len = err.error_len(),
            line_len = payload.len(),
            "SSE payload is not valid UTF-8; decoding lossily"
        );

        // `handle_payload` takes `&mut self`, so the decoded `&str` must not be
        // borrowed from `self`. Decode into a local starting with the capacity
        // stashed in `lossy_scratch` (moved out, so no allocation after the
        // first corrupted line), handle the line, then hand the capacity back.
        let mut decoded = std::mem::take(&mut self.lossy_scratch);
        lossy_decode_into(payload, &mut decoded);
        // `lossy_decode_into` only ever emits valid UTF-8, so this cannot fail.
        let text = match String::from_utf8(decoded) {
            Ok(text) => text,
            Err(e) => String::from_utf8_lossy(e.as_bytes()).into_owned(),
        };
        let outcome = self.handle_payload(&text);
        // Recycle the allocation for the next corrupted line.
        self.lossy_scratch = text.into_bytes();
        outcome
    }

    /// Parse and accumulate one already-classified `data:` payload.
    ///
    /// Filtering (blank lines, comments, field name, `[DONE]`) was already done
    /// on bytes by [`data_field`] / the `DONE_SENTINEL` check, so this stage
    /// only turns JSON into deltas.
    fn handle_payload(&mut self, data: &str) -> FrameOutcome {
        // `data_field` already ASCII-trimmed the payload on bytes (the only
        // whitespace SSE framing can introduce). This zero-alloc Unicode trim
        // restores exact parity with the old `str::trim` path for exotic
        // whitespace (e.g. NBSP) at payload edges, and re-checks the sentinel
        // for a `[DONE]` wrapped in such whitespace.
        let data = data.trim();
        if data.is_empty() {
            return FrameOutcome::Consumed;
        }
        if data == DONE_SENTINEL_STR {
            return FrameOutcome::Done;
        }
        match serde_json::from_str::<StreamChunk>(data) {
            Ok(chunk) => {
                if let Some(choice) = chunk.choices.first() {
                    if let Some(r) = choice.delta.reasoning() {
                        self.push_reasoning_content(r);
                    }
                    if let Some(c) = &choice.delta.content {
                        self.push_content(c);
                    }
                    for tc in &choice.delta.tool_calls {
                        self.accumulate_tool_call(tc);
                    }
                }
            }
            // A dropped frame loses content, tool calls and reasoning at once
            // and is otherwise invisible, so surface it — but only once per
            // stream, or a persistently misbehaving provider floods the log.
            Err(err) => {
                if !self.parse_error_logged {
                    self.parse_error_logged = true;
                    tracing::warn!(
                        error = %err,
                        "Failed to parse an SSE data frame; dropping it and the rest of this stream's delta"
                    );
                }
            }
        }
        FrameOutcome::Consumed
    }

    /// Append streamed reasoning content, respecting [`MAX_STREAMED_CONTENT_BYTES`].
    pub(crate) fn push_reasoning_content(&mut self, text: &str) {
        if self.reasoning_content.len() >= MAX_STREAMED_CONTENT_BYTES {
            if !self.reasoning_capped {
                self.reasoning_capped = true;
                tracing::warn!(
                    limit = MAX_STREAMED_CONTENT_BYTES,
                    "Streamed assistant reasoning exceeded the retention budget; truncating"
                );
            }
            return;
        }
        let room = MAX_STREAMED_CONTENT_BYTES - self.reasoning_content.len();
        if text.len() <= room {
            self.reasoning_content.push_str(text);
            return;
        }
        self.reasoning_content
            .push_str(&text[..text.floor_char_boundary(room)]);
        self.reasoning_capped = true;
        tracing::warn!(
            limit = MAX_STREAMED_CONTENT_BYTES,
            "Streamed assistant reasoning exceeded the retention budget; truncating"
        );
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
        // Longest prefix of `text` that fits in `room` bytes, snapped to a char
        // boundary so we never store a partial code point.
        //
        // `floor_char_boundary` is *relative* to `text`, so the slice length is
        // exactly that value. Adding `self.content.len()` here (an absolute
        // offset) made the index run past the end of `text` and panic as soon
        // as the buffer was non-empty — the normal case for a long stream
        // crossing the cap.
        self.content
            .push_str(&text[..text.floor_char_boundary(room)]);
        self.content_capped = true;
        tracing::warn!(
            limit = MAX_STREAMED_CONTENT_BYTES,
            "Streamed assistant content exceeded the retention budget; truncating"
        );
    }

    /// Fold one streamed `tool_calls` delta into the index-keyed map.
    ///
    /// Providers that omit `index` (serde defaults it to `0`) or re-send
    /// `index: 0` for every call in a turn deliver *several distinct calls* on
    /// the same slot. Treating the second id as a malicious collision marked
    /// the entry `malformed` and dropped **both** calls, leaving an empty turn
    /// with no command to run — an infinite retry loop. Instead, a differing id
    /// landing on a slot that already has an id or command *opens a new slot*
    /// just past the highest index in use, so both calls survive and stay
    /// deterministically ordered.
    pub(crate) fn accumulate_tool_call(&mut self, tc: &StreamToolCall) {
        // Pick the slot this delta belongs to. A fresh id landing on an
        // already-populated slot means the provider restarted its call
        // numbering, so redirect it to a brand-new index instead of clobbering
        // the call we already accumulated.
        // A continuation chunk re-sends the *slot* fields with empty strings
        // (`"id":"", "type":"", "name":""`), so an empty id means "no id in this
        // delta" — not a new call. Only a genuinely different, non-empty id
        // redirects the delta to a new slot.
        let new_id = tc.id.as_deref().map(str::trim).filter(|id| !id.is_empty());
        let target_index = match self.tools.get(&tc.index) {
            Some(entry)
                if !entry.malformed
                    && new_id.is_some_and(|id| id != entry.id)
                    && (!entry.id.is_empty() || !entry.arguments.trim().is_empty()) =>
            {
                let next = self.tools.keys().max().copied().unwrap_or(0) + 1;
                tracing::warn!(
                    index = tc.index,
                    new_index = next,
                    "New tool_call id on an already-populated index; opening a new slot"
                );
                next
            }
            _ => tc.index,
        };

        let entry = self.tools.entry(target_index).or_default();
        if entry.malformed {
            return;
        }

        if let Some(id) = new_id {
            entry.id = id.to_string();
        }

        if let Some(fn_info) = &tc.function {
                if let Some(name) = fn_info.name.as_deref().map(str::trim)
                && !name.is_empty()
            {
                entry.name.push_str(name);
            }
            if let Some(args) = &fn_info.arguments
                && !args.is_empty()
            {
                        if entry.arguments.len() + args.len() > MAX_TOOL_ARGUMENT_BYTES {
                    tracing::warn!(
                        index = target_index,
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
        let mut seen_ids: HashSet<String> = HashSet::with_capacity(tools.len());
        for entry in tools.values() {
            if entry.malformed || entry.is_placeholder() {
                continue;
            }
            // Reserve up front: counts are known from the map, avoiding
            // rehashing on the insert-heavy duplicate path. The `ToolCall` and
            // the set each need an owned `String`, so the clone into `seen_ids`
            // stays; removed is the extra round of cloning the *generated* id,
            // which the old shape did twice.
            let id = entry.id.trim().to_string();
            let id = if !id.is_empty() && seen_ids.insert(id.clone()) {
                id
            } else {
                if !id.is_empty() {
                    tracing::warn!(
                        original_id = %id,
                        "Duplicate tool_call id in stream; generated a unique replacement"
                    );
                }
                let generated = generate_call_id();
                seen_ids.insert(generated.clone());
                generated
            };
            tcs.push(ToolCall {
                id,
                r#type: "function".to_string(),
                function: ToolCallFn {
                    name: if entry.name.trim().is_empty() {
                        "bash".to_string()
                    } else {
                        entry.name.clone()
                    },
                    arguments: entry.arguments.clone(),
                },
            });
        }
        (tcs, seen_ids)
    }

    /// Turn the accumulated stream into the [`LlmResponse`] the agent loop consumes.
    ///
    /// The command comes from the structured `bash` tool call when the model
    /// emitted one, falling back to a fenced code block in the content
    /// otherwise. The reported tool-call id is only kept when it survived
    /// streaming, so the next request never carries a dangling id.
    pub(crate) fn finish(self) -> LlmResponse {
        let SseAccumulator {
            content,
            reasoning_content,
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
        let tool_call_id = bash_tc.and_then(|tc| {
            let candidate = if tc.id.trim().is_empty() {
                finalized
                    .iter()
                    .find(|t| t.function.arguments == tc.arguments)
                    .map(|t| t.id.clone())
            } else {
                Some(tc.id.trim().to_string())
            };
            candidate.filter(|id| known_ids.contains(id))
        });
        // One step runs one command and answers with one `tool` message, so only
        // the executed call may go back into history: sibling calls replayed
        // without their results would leave unanswered `tool_calls` on the
        // assistant turn, which the tool-call protocol rejects on the next
        // request.
        let emitted = finalized.len();
        let mut tool_calls = finalized;
        if let Some(id) = &tool_call_id {
            tool_calls.retain(|tc| &tc.id == id);
        }
        if tool_calls.len() < emitted {
            tracing::debug!(
                dropped = emitted - tool_calls.len(),
                kept = tool_calls.len(),
                "Model returned several tool calls; replaying only the executed one"
            );
        }
        let tool_calls = (!tool_calls.is_empty()).then_some(tool_calls);

        let reasoning = (!reasoning_content.trim().is_empty()).then_some(reasoning_content);

        LlmResponse {
            content,
            reasoning_content: reasoning,
            command,
            tool_calls,
            tool_call_id,
            invalid_utf8_lines,
        }
    }

    /// Handle a non-streaming JSON response fallback.
    pub(crate) fn handle_non_stream_fallback(&mut self, buffer: &[u8]) {
        if self.content.is_empty()
            && self.reasoning_content.is_empty()
            && self.tools.is_empty()
            && !buffer.is_empty()
            && let Ok(result) = serde_json::from_slice::<ChatCompletionResponse>(buffer)
            && let Some(choice) = result.choices.first()
        {
            if let Some(r) = choice.message.reasoning() {
                self.push_reasoning_content(r);
            }
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

    /// The exact chunk shapes captured from the OpenAI-compatible proxy: a first
    /// tool-call chunk with a real id, continuation chunks that re-send the slot
    /// with *empty* id/name/type, and reasoning chunks carrying BOTH
    /// `reasoning` and `reasoning_content`. The empty id must not open a new
    /// slot (one call, not one per fragment), reasoning must be captured exactly
    /// once, and the full arguments must parse to `command: "ls"`.
    #[test]
    fn real_provider_chunk_shapes_accumulate_one_call_and_reasoning_once() {
        let mut acc = SseAccumulator::default();
        let mut buffer = Vec::new();
        let body = sse_body(&[
            r#"{"choices":[{"delta":{"role":"assistant","content":"","tool_calls":[{"index":0,"id":"chatcmpl-tool-9d6a7c55214c5666","type":"function","function":{"name":"bash","arguments":""}}]}}]}"#,
            r##"{"choices":[{"delta":{"role":"assistant","content":"","tool_calls":[{"index":0,"id":"","type":"","function":{"name":"","arguments":"{\"command\": "}}]}}]}"##,
            r#"{"choices":[{"delta":{"role":"assistant","content":"","reasoning":"thought A","reasoning_content":"thought A"}}]}"#,
            r##"{"choices":[{"delta":{"role":"assistant","content":"","tool_calls":[{"index":0,"id":"","type":"","function":{"name":"","arguments":"\"ls\"}"}}]}}]}"##,
        ]);
        for b in &body {
            if acc.push(&[*b], &mut buffer) == Some(FrameOutcome::Done) {
                break;
            }
        }
        assert_eq!(acc.tools.len(), 1, "one call, not one per fragment: {:?}", acc.tools);
        let (tcs, _) = SseAccumulator::finalize_from(&acc.tools);
        assert_eq!(tcs.len(), 1, "exactly one tool call: {tcs:?}");
        assert_eq!(tcs[0].id, "chatcmpl-tool-9d6a7c55214c5666");
        assert_eq!(tcs[0].function.arguments, r#"{"command": "ls"}"#);
        assert_eq!(acc.reasoning_content, "thought A", "reasoning captured once");
        let resp = acc.finish();
        assert_eq!(resp.command.as_deref(), Some("ls"));
        assert_eq!(resp.reasoning_content.as_deref(), Some("thought A"));
    }

    /// A genuinely different, non-empty id on an already-populated index must
    /// still open a new slot (the redirect behaviour is preserved).
    #[test]
    fn non_empty_different_id_still_opens_a_new_slot() {
        let mut acc = SseAccumulator::default();
        acc.accumulate_tool_call(&tc(0, Some("a"), Some("bash"), Some("{}")));
        acc.accumulate_tool_call(&tc(0, Some("b"), None, None));
        assert_eq!(acc.tools.len(), 2, "a real new id must open a slot: {:?}", acc.tools);
    }

    /// Explicit `null` fields must not drop the chunk: `tool_calls: null`,
    /// `choices: null`, `function: null`, `content: null`, `id: null`.
    #[test]
    fn explicit_nulls_do_not_drop_a_chunk() {
        let mut acc = SseAccumulator::default();
        let mut buffer = Vec::new();
        let body = sse_body(&[
            r#"{"choices":null}"#,
            r#"{"choices":[{"delta":{"content":null,"reasoning_content":null,"tool_calls":null}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":null,"function":null}]}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"x","function":{"name":"bash","arguments":"{\"command\":\"ls\"}"}}]}}]}"#,
        ]);
        for b in &body {
            if acc.push(&[*b], &mut buffer) == Some(FrameOutcome::Done) {
                break;
            }
        }
        let resp = acc.finish();
        assert_eq!(resp.command.as_deref(), Some("ls"), "nulls must not drop the chunk");
    }

    /// A data frame that fails to parse is warned about (once per stream) rather
    /// than silently dropped.
    #[test]
    fn unparsable_frame_is_warned_once() {
        let mut acc = SseAccumulator::default();
        let mut buffer = Vec::new();
        let body = sse_body(&["{not json", "{also bad", r#"{"choices":[{"delta":{"content":"ok"}}]}"#]);
        for b in &body {
            if acc.push(&[*b], &mut buffer) == Some(FrameOutcome::Done) {
                break;
            }
        }
        assert!(acc.parse_error_logged, "the parse failure must be latched");
        assert_eq!(acc.content, "ok", "the valid frame must still be consumed");
    }

    /// When the model returns more than one tool call, only the executed one is
    /// replayed into history (matching `tool_call_id`); the rest are dropped.
    #[test]
    fn finish_keeps_only_the_executed_tool_call_in_history() {
        let mut acc = SseAccumulator::default();
        acc.accumulate_tool_call(&tc(0, Some("a"), Some("bash"), Some(r#"{"command":"ls"}"#)));
        acc.accumulate_tool_call(&tc(1, Some("b"), Some("bash"), Some(r#"{"command":"pwd"}"#)));
        let resp = acc.finish();
        assert_eq!(resp.command.as_deref(), Some("ls"));
        let calls = resp.tool_calls.expect("tool_calls present");
        assert_eq!(calls.len(), 1, "only the executed call may be replayed: {calls:?}");
        assert_eq!(calls[0].id, "a");
        assert_eq!(resp.tool_call_id.as_deref(), Some("a"));
    }

    /// A provider that sends several tool calls with the same `index` (or omits
    /// `index`, which serde defaults to `0`) used to have the *second* id treated
    /// as a malicious collision: the entry was marked `malformed` and both calls
    /// were dropped, so the turn carried no command and the agent loop spun
    /// forever. Both calls must survive in separate slots instead.
    #[test]
    fn conflicting_ids_for_one_index_open_a_new_slot_instead_of_dropping() {
        let mut acc = SseAccumulator::default();
        acc.accumulate_tool_call(&tc(0, Some("a"), Some("bash"), Some("{}")));
        acc.accumulate_tool_call(&tc(0, Some("b"), None, None));
        let (tcs, _) = SseAccumulator::finalize_from(&acc.tools);
        assert_eq!(tcs.len(), 2, "both calls must survive: {tcs:?}");
        assert_eq!(tcs[0].id, "a");
        assert_eq!(tcs[1].id, "b");
        assert!(
            acc.tools.values().all(|e| !e.malformed),
            "a numbering quirk is not corruption: {:?}",
            acc.tools
        );
    }

    /// The realistic shape: a full multi-call turn where *every* call arrives on
    /// `index: 0`, each with its own id and complete arguments. All of them must
    /// be emitted, in arrival order, with unique ids.
    #[test]
    fn sequential_calls_all_reusing_index_zero_are_all_retained() {
        let mut acc = SseAccumulator::default();
        for (i, cmd) in ["ls", "pwd", "whoami"].iter().enumerate() {
            acc.accumulate_tool_call(&tc(
                0,
                Some(&format!("call_{i}")),
                Some("bash"),
                Some(&format!(r#"{{"command":"{cmd}"}}"#)),
            ));
        }
        let (tcs, ids) = SseAccumulator::finalize_from(&acc.tools);
        assert_eq!(tcs.len(), 3, "no call may be dropped: {tcs:?}");
        let commands: Vec<&str> = tcs
            .iter()
            .map(|c| c.function.arguments.as_str())
            .collect();
        assert_eq!(
            commands,
            vec![
                r#"{"command":"ls"}"#,
                r#"{"command":"pwd"}"#,
                r#"{"command":"whoami"}"#
            ],
            "arrival order must be preserved"
        );
        assert_eq!(ids.len(), 3, "ids must be unique");
    }

    /// A delta that merely *continues* a call (no id in the delta, matching the
    /// one already on the slot) must keep accumulating into the same entry — the
    /// new-slot path must not fire for a legitimate continuation.
    #[test]
    fn continuation_deltas_still_accumulate_into_the_same_slot() {
        let mut acc = SseAccumulator::default();
        acc.accumulate_tool_call(&tc(0, Some("c0"), Some("bash"), Some(r#"{"comm"#)));
        acc.accumulate_tool_call(&tc(0, Some("c0"), None, Some(r#"and":"pwd"}"#)));
        let (tcs, _) = SseAccumulator::finalize_from(&acc.tools);
        assert_eq!(tcs.len(), 1, "same id must stay one call: {tcs:?}");
        assert_eq!(tcs[0].function.arguments, r#"{"command":"pwd"}"#);
    }

    /// A differing id on a slot that is *still empty* (no id, no arguments) is
    /// not a conflict: it is the first frame of a call, and it just fills the
    /// slot rather than opening another one.
    #[test]
    fn id_on_an_empty_slot_fills_it_without_opening_another() {
        let mut acc = SseAccumulator::default();
        acc.accumulate_tool_call(&tc(0, Some("first"), Some("bash"), Some("{}")));
        assert_eq!(acc.tools.len(), 1);
        assert_eq!(acc.tools.get(&0).expect("slot 0").id, "first");
    }

    /// The new slot is opened at `max_index + 1`, so it never collides with an
    /// index a later delta legitimately claims.
    #[test]
    fn new_slot_is_placed_past_the_highest_index_in_use() {
        let mut acc = SseAccumulator::default();
        acc.accumulate_tool_call(&tc(7, Some("seven"), Some("bash"), Some("{}")));
        acc.accumulate_tool_call(&tc(0, Some("zero"), Some("bash"), Some("{}")));
        // Conflicting id on index 0, which already holds a call: must land on 8.
        acc.accumulate_tool_call(&tc(0, Some("other"), Some("bash"), Some("{}")));
        assert!(acc.tools.contains_key(&8), "slots: {:?}", acc.tools.keys());
        let (tcs, _) = SseAccumulator::finalize_from(&acc.tools);
        assert_eq!(tcs.len(), 3, "all three calls must survive: {tcs:?}");
        // Ordering stays index-ordered, so 7 comes after 0 and 8.
        assert_eq!(tcs[0].id, "zero");
        assert_eq!(tcs[1].id, "seven");
        assert_eq!(tcs[2].id, "other");
    }

    /// Long chain-of-thought replies used to be cut at 16 KiB, losing the tail of
    /// the model's reasoning. The budget now matches the tool-argument budget.
    #[test]
    fn content_budget_is_64_kib() {
        assert_eq!(MAX_STREAMED_CONTENT_BYTES, 64 * 1024);
        let mut acc = SseAccumulator::default();
        // A CoT stream of 48 KiB must be retained in full, untruncated.
        acc.push_content(&"r".repeat(48 * 1024));
        assert_eq!(acc.content.len(), 48 * 1024, "no CoT truncation");
        assert!(!acc.content_capped, "48 KiB must fit in the budget");
        // The cap must still exist past the new budget.
        acc.push_content(&"s".repeat(MAX_STREAMED_CONTENT_BYTES));
        assert_eq!(acc.content.len(), MAX_STREAMED_CONTENT_BYTES);
        assert!(acc.content_capped, "the cap must still fire past 64 KiB");
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

    // ---- UTF-8 decode / lossy regression tests ----

    /// `lossy_decode_into` must match `String::from_utf8_lossy` byte-for-byte,
    /// including truncated tails and multi-byte sequences split by bad bytes.
    #[test]
    fn lossy_decode_into_matches_std() {
        let cases: Vec<Vec<u8>> = vec![
            b"plain ascii".to_vec(),
            b"caf\xc3\xa9 \xe2\x82\xac".to_vec(),  // valid
            b"caf\xc3\xa9".to_vec(),               // truncated 2-byte tail
            b"emoji \xf0\x9f\x98\x80 ok".to_vec(), // valid 4-byte
            b"\xf0\x9f\x98".to_vec(),              // truncated 4-byte tail
            b"\xff\xfe".to_vec(),                  // two invalid bytes
            b"a\xffb\xffc".to_vec(),
            b"pre\xffmid\x80post".to_vec(),
            b"\xed\xa0\x80".to_vec(), // surrogate half
            b"\xc0\xaf".to_vec(),     // overlong
            b"".to_vec(),
        ];
        for case in cases {
            let mut got = Vec::new();
            lossy_decode_into(&case, &mut got);
            let want = String::from_utf8_lossy(&case);
            assert_eq!(
                std::str::from_utf8(&got).expect("decoder must emit valid UTF-8"),
                want,
                "lossy mismatch for {case:?}"
            );
            // And the output must itself be valid UTF-8, byte-identical in kind.
            assert_eq!(got, want.as_bytes());
        }
    }

    /// The lossy path must preserve the *decoded content* of a corrupted frame,
    /// not just count it. A refactor that empties the buffer before handing it
    /// to the parser would silently drop the model output here.
    #[test]
    fn corrupted_frame_keeps_its_decodable_content() {
        let mut line = br#"data: {"choices":[{"delta":{"content":"keep"#.to_vec();
        line.extend_from_slice(&[0xFF]);
        line.extend_from_slice(br#"me"}}]}"#);
        line.extend_from_slice(b"\n\ndata: [DONE]\n\n");
        let mut acc = SseAccumulator::default();
        let mut buffer = Vec::new();
        assert_eq!(acc.push(&line, &mut buffer), Some(FrameOutcome::Done));
        assert_eq!(acc.invalid_utf8_lines, 1);
        assert_eq!(
            acc.content, "keep\u{FFFD}me",
            "corrupted frame must still contribute its content"
        );
    }

    /// Capacity is recycled across corrupted lines, so a long corrupting stream
    /// must not allocate one buffer per frame.
    #[test]
    fn lossy_scratch_is_reused_across_frames() {
        let mut acc = SseAccumulator::default();
        let mut buffer = Vec::new();
        let mut body = Vec::new();
        for _ in 0..64 {
            let mut f = br#"data: {"choices":[{"delta":{"content":"x"}}"#.to_vec();
            f.extend_from_slice(&[0xFF]);
            f.extend_from_slice(br#""}}]}"#);
            f.extend_from_slice(b"\n\n");
            body.extend_from_slice(&f);
        }
        body.extend_from_slice(b"data: [DONE]\n\n");
        acc.push(&body, &mut buffer);
        assert_eq!(acc.invalid_utf8_lines, 64);
        // Capacity is retained (not dropped) after the stream is processed.
        assert!(
            acc.lossy_scratch.capacity() > 0,
            "decode buffer must keep its capacity for reuse"
        );
    }

    /// Regression: truncating a *partially filled* buffer used to index `text`
    /// with an absolute offset, panicking with "end byte index out of bounds".
    #[test]
    fn content_truncation_from_a_partially_filled_buffer_does_not_panic() {
        let mut acc = SseAccumulator {
            content: "x".repeat(MAX_STREAMED_CONTENT_BYTES - 1),
            ..Default::default()
        };
        acc.push_content("abcdefghij");
        assert_eq!(acc.content.len(), MAX_STREAMED_CONTENT_BYTES);
        assert!(acc.content_capped);
    }

    /// Truncation must respect multi-byte boundaries in the delta itself: with
    /// only 2 bytes of room a 3-byte `€` cannot fit, so it is dropped whole
    /// rather than split into an invalid partial code point.
    #[test]
    fn content_truncation_snaps_to_char_boundary_in_delta() {
        let mut acc = SseAccumulator {
            content: "x".repeat(MAX_STREAMED_CONTENT_BYTES - 2),
            ..Default::default()
        };
        acc.push_content("\u{20AC}\u{20AC}\u{20AC}"); // 3 x 3 bytes
        assert_eq!(
            acc.content.len(),
            MAX_STREAMED_CONTENT_BYTES - 2,
            "a 3-byte char must not be split across a 2-byte remainder"
        );
        assert!(acc.content.is_char_boundary(acc.content.len()));
        assert!(!acc.content.contains('\u{FFFD}'), "no partial code point");
    }

    // ---- byte-slice field classification ----

    #[test]
    fn data_field_extracts_and_trims_payload() {
        assert_eq!(data_field(b"data: {}"), Some(&b"{}"[..]));
        assert_eq!(data_field(b"data:{}"), Some(&b"{}"[..]));
        assert_eq!(
            data_field(b"  data:   {\"a\":1}  \r"),
            Some(&b"{\"a\":1}"[..])
        );
        assert_eq!(data_field(b"\t data: x \t"), Some(&b"x"[..]));
        // A payload that is itself blank is still a data field (empty payload).
        assert_eq!(data_field(b"data:   "), Some(&b""[..]));
    }

    #[test]
    fn data_field_rejects_non_data_lines() {
        // Blank lines and every kind of whitespace-only line.
        assert_eq!(data_field(b""), None);
        assert_eq!(data_field(b"   "), None);
        assert_eq!(data_field(b"\r\n"), None);
        assert_eq!(data_field(b"\t"), None);
        // SSE comments / keep-alives.
        assert_eq!(data_field(b": ping"), None);
        assert_eq!(data_field(b":"), None);
        assert_eq!(data_field(b"   : keep-alive"), None);
        // Other SSE fields must not be mistaken for a payload.
        assert_eq!(data_field(b"event: message"), None);
        assert_eq!(data_field(b"id: 42"), None);
        assert_eq!(data_field(b"retry: 100"), None);
        // Near-misses on the field name.
        assert_eq!(data_field(b"datax: {}"), None);
        assert_eq!(data_field(b"data : {}"), None);
        assert_eq!(data_field(b"dat: {}"), None);
    }

    #[test]
    fn data_field_handles_invalid_utf8_without_panicking() {
        // Classification is byte-wise, so a corrupt line must still be routed
        // (and counted) rather than crashing on a char-boundary assumption.
        let bad = b"data: \xff\xfe".to_vec();
        assert_eq!(data_field(&bad), Some(&b"\xff\xfe"[..]));
        let bad_comment = b": \xff\xfe".to_vec();
        assert_eq!(data_field(&bad_comment), None);
    }

    #[test]
    fn data_field_is_exhaustively_agreeable_with_trim() {
        // `data_field` replaces the old `str::trim` + `strip_prefix("data:")`
        // logic; prove the byte version accepts exactly the same set of lines.
        for raw in [
            &b"data: x"[..],
            &b" data: x "[..],
            &b"\r\ndata: x\r\n"[..],
            &b"data:x"[..],
            &b"data:   "[..],
        ] {
            let as_str = std::str::from_utf8(raw).expect("ascii");
            let via_str = as_str.trim().strip_prefix("data:").map(str::trim);
            let via_bytes = data_field(raw).map(|b| std::str::from_utf8(b).unwrap());
            assert_eq!(via_bytes, via_str, "mismatch for {raw:?}");
        }
    }

    #[test]
    fn unknown_fields_and_comments_do_not_count_as_invalid_utf8() {
        // A corrupt comment line carries no payload, so it is filtered out on
        // bytes and never decoded: it must not inflate the corruption counter.
        let mut acc = SseAccumulator::default();
        let mut buffer = Vec::new();
        let mut body = b": \xff\xfe keep-alive\n\n".to_vec();
        body.extend_from_slice(b"event: \xff\xfe\n\n");
        body.extend_from_slice(b"data: [DONE]\n\n");
        assert_eq!(acc.push(&body, &mut buffer), Some(FrameOutcome::Done));
        assert_eq!(acc.invalid_utf8_lines, 0, "comments are not payload");
    }

    // ---- unbounded-growth regression (ill-formed streams) ----

    #[test]
    fn newline_free_stream_does_not_grow_the_buffer_without_bound() {
        // A provider that never emits a `\n` would otherwise make the framing
        // buffer absorb the whole (unbounded) body chunk by chunk. The reader
        // must drop the over-long unterminated tail instead of retaining it.
        let mut acc = SseAccumulator::default();
        let mut buffer = Vec::new();
        let chunk = vec![b'x'; 64 * 1024];
        for _ in 0..(MAX_SSE_FRAME_BYTES / chunk.len() + 16) {
            acc.push(&chunk, &mut buffer);
        }
        assert!(
            buffer.len() <= MAX_SSE_FRAME_BYTES,
            "unterminated frame must be capped, got {} bytes",
            buffer.len()
        );
    }

    #[test]
    fn oversized_single_line_is_discarded_and_framing_resyncs() {
        // One absurd line (no newline inside) must be dropped whole, and the
        // *next* well-formed frame must still be parsed: the reader resyncs on
        // the next `\n` rather than wedging.
        let mut acc = SseAccumulator::default();
        let mut buffer = Vec::new();
        let mut body = vec![b'y'; MAX_SSE_FRAME_BYTES * 2];
        body.push(b'\n');
        body.extend_from_slice(br#"data: {"choices":[{"delta":{"content":"ok"}}]}"#);
        body.extend_from_slice(b"\n\n");
        acc.push(&body, &mut buffer);
        assert_eq!(
            acc.content, "ok",
            "reader must resync after the garbage line"
        );
        assert!(
            buffer.is_empty(),
            "no tail may survive the well-formed frame"
        );
    }

    #[test]
    fn oversized_line_suffix_cannot_be_parsed_as_a_new_frame() {
        let mut acc = SseAccumulator::default();
        let mut buffer = Vec::new();
        acc.push(&vec![b'x'; MAX_SSE_FRAME_BYTES + 1], &mut buffer);
        // The next chunk still belongs to the dropped line. Its data-looking
        // suffix must not be interpreted until the first newline is seen.
        acc.push(
            b"data: {\"choices\":[{\"delta\":{\"content\":\"forged\"}}]}\n",
            &mut buffer,
        );
        assert!(acc.content.is_empty());
        acc.push(
            b"data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n",
            &mut buffer,
        );
        assert_eq!(acc.content, "ok");
        assert!(buffer.is_empty());
    }

    #[test]
    fn oversized_complete_data_line_is_skipped() {
        let mut acc = SseAccumulator::default();
        let mut buffer = Vec::new();
        let mut chunk = b"data: ".to_vec();
        chunk.extend(vec![b' '; MAX_SSE_FRAME_BYTES]);
        chunk.extend_from_slice(b"[DONE]\n");
        assert_eq!(acc.push(&chunk, &mut buffer), None);
        assert!(acc.frame_cap_logged);
        assert!(buffer.is_empty());
        assert!(!acc.discarding_line);
    }

    #[test]
    fn a_large_but_legal_frame_split_across_chunks_is_never_truncated() {
        // Guard against the cap misfiring: a single frame well under the budget
        // but delivered in tiny pieces must still reassemble exactly, and the
        // cap must not have dropped the tail.
        // The content stays under `MAX_STREAMED_CONTENT_BYTES` so the assertion
        // below tests the *framing* cap, not the separate content budget.
        let text = "z".repeat(MAX_STREAMED_CONTENT_BYTES - 1024);
        let frame = serde_json::json!({"choices":[{"delta":{"content":text}}]}).to_string();
        assert!(
            frame.len() < MAX_SSE_FRAME_BYTES,
            "test frame must stay within the cap"
        );
        let body = sse_body(&[&frame]);

        let mut acc = SseAccumulator::default();
        let mut buffer = Vec::new();
        let mut done = false;
        for chunk in body.chunks(1024) {
            if acc.push(chunk, &mut buffer) == Some(FrameOutcome::Done) {
                done = true;
                break;
            }
        }
        assert!(done, "[DONE] must terminate the stream");
        assert_eq!(acc.content.len(), text.len(), "content must be exact");
        assert!(!acc.content_capped, "a legal frame must not be truncated");
        assert!(!acc.frame_cap_logged, "the cap must not have fired");
    }
}
