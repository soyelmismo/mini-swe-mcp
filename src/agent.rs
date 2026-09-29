use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{LazyLock, OnceLock};
use std::time::Duration;
use regex::Regex;
use tokio::process::Command;

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

/// Default idle (per-chunk) deadline for the SSE body read.
///
/// A *whole-request* deadline is the wrong tool for a token stream: it kills
/// healthy-but-slow generations regardless of progress. `read_timeout` /
/// `connect_timeout` plus a per-chunk `tokio::time::timeout` only abort genuine
/// stalls.
pub const DEFAULT_STREAM_IDLE_TIMEOUT: Duration = Duration::from_secs(120);

/// Hard cap on the assistant text retained from a stream. The bash-output
/// budget is 16 KiB; the assistant's own reasoning is bounded by the same
/// order of magnitude so a runaway stream cannot inflate memory (nor be
/// re-sent verbatim on the next request).
pub const MAX_STREAMED_CONTENT_BYTES: usize = 16 * 1024;

/// Hard cap on the serialized `arguments` accumulated for a single tool call.
/// A model that streams megabytes of arguments is treated as malformed and the
/// call is dropped rather than buffered.
pub const MAX_TOOL_ARGUMENT_BYTES: usize = 64 * 1024;

/// Byte size of a typical SSE frame; used to pre-reserve the read buffer so a
/// long stream does not repeatedly reallocate as chunks arrive.
const SSE_BUFFER_HINT_BYTES: usize = 8 * 1024;

pub const SYSTEM_PROMPT: &str = r#"You are an autonomous software engineering subagent running in a Linux bash environment.
You are given a task to complete within a git repository.

LOCATION & SCOPE:
- You are ALREADY located at the root of the repository worktree ($PWD).
- Never execute `cd` to parent directories (like /home/rot, /repo, or /). All repository files are right here in the current directory.

WORKFLOW:
1. Explore: Use tools like `git status`, `find`, `grep -rn`, or `ls` to locate relevant files in the current repository.
2. Edit & Test: Make minimal, clean edits (using sed, python, cat << 'EOF', etc.) and run existing test suites to verify.
3. Every response MUST execute EXACTLY ONE command using the `bash` tool. If the bash tool is unavailable, use a ```bash ... ``` code block instead.
4. When finished:
   - For code tasks: verify with tests and execute:
     echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
   - For audit/analysis tasks: print your concise findings report to stdout and in the same or next turn execute:
     echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT

REPORTS & ARTIFACTS:
- If you generate reports, audits, benchmarks, or handoffs, save them under `audits/`, `reports/`, or `.agents/` (e.g. `audits/audit_01_feature.md` or `.agents/handoff.md`).
- Files created in these directories are automatically preserved and synchronized back to the main repository when your task completes.

COMMUNICATION WITH ORCHESTRATOR:
- Need more turns: If you are close to finishing verification/refactoring and need more steps, execute:
  echo "REQUEST_TURNS: <number>"
- Ask orchestrator / Critical ambiguity: If you face critical blockers, breaking decisions, or require orchestrator confirmation, execute:
  echo "ASK_ORCHESTRATOR: <your specific question>"
  This will immediately pause execution until the orchestrator replies with guidance."#;

/// Chat roles accepted by the OpenAI chat-completions API.
///
/// Modelling the role as an enum instead of a free-form `String` turns an
/// invalid role from a provider-side `400` into a compile error: the wire
/// strings are pinned by `rename_all = "lowercase"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

impl Role {
    /// The exact string the chat API expects for this role.
    ///
    /// Kept in lockstep with the `rename_all = "lowercase"` derive and asserted
    /// against it in the tests, so there is a single source of truth for the
    /// wire spelling.
    pub const fn as_wire_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
        }
    }
}

/// A single outbound conversation message.
///
/// All fields are private so that the three constructors below are the *only*
/// way to build one. That closes the invalid states an all-`pub` struct allows
/// (a `tool` message with no `tool_call_id`, an `assistant` message with
/// `tool_calls: Some(vec![])`), while keeping the per-field
/// `skip_serializing_if` needed for each role's wire shape.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    role: Role,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

/// Outbound tool_call representation for assistant messages in the conversation history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub r#type: String,
    pub function: ToolCallFn,
}

/// The nested `function` object of a [`ToolCall`].
///
/// The OpenAI wire format nests the call one level deep
/// (`{"id":..,"type":"function","function":{"name":..,"arguments":..}}`), so
/// this struct is load-bearing rather than a premature abstraction. It is also
/// reused by the non-streaming inbound path ([`ToolCallOutput`]) to avoid
/// duplicating a structurally identical type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallFn {
    pub name: String,
    pub arguments: String,
}

impl ChatMessage {
    /// A plain single-content message: the shape used by `system`, `user` and
    /// `assistant` turns. `tool` content must go through [`Self::tool_result`],
    /// which also records the `tool_call_id` the API requires.
    pub fn text(role: Role, content: impl Into<String>) -> Self {
        debug_assert_ne!(
            role,
            Role::Tool,
            "tool messages require a tool_call_id; use ChatMessage::tool_result"
        );
        Self {
            role,
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// An assistant turn that requests tool execution.
    ///
    /// `content` is `None` when the model emitted tool calls with no prose, in
    /// which case the field is omitted on the wire. An empty `tool_calls` vec is
    /// normalised to `None` so the API never sees a call-less assistant turn
    /// carrying an empty array.
    pub fn assistant_with_tool_calls(content: Option<String>, tool_calls: Vec<ToolCall>) -> Self {
        Self {
            role: Role::Assistant,
            content,
            tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
            tool_call_id: None,
        }
    }

    /// The `tool` turn that answers a previous tool call, keyed by its id.
    pub fn tool_result(tool_call_id: String, content: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: Some(tool_call_id),
        }
    }

    /// Read-only view of the (already validated) role.
    ///
    /// The field itself is private; this accessor exists so callers and tests can
    /// observe the role without being able to set an arbitrary one.
    pub fn role(&self) -> Role {
        self.role
    }

    /// The message content, if any.
    pub fn content(&self) -> Option<&str> {
        self.content.as_deref()
    }
}

/// The single tool this agent advertises to the provider.
///
/// Per §2 of `audits/overeng_01_agent_structs.md`, `agent.rs` deliberately holds
/// no tool *schema* types — the dedicated `ToolDefinition`/`ToolFunction` pair
/// that used to wrap this single constant is gone, and the request DTO now
/// carries a ready-made `Vec<serde_json::Value>`.
///
/// If a second tool is ever needed (write file, search, MCP passthrough) the fix
/// is **not** to reintroduce a schema struct here, but to promote the tool
/// catalog in `manifest.rs` (today only a rendered `String` from
/// `build_tool_description()`) into a real descriptor type and render it into
/// `Vec<Value>` at this boundary.
fn bash_tool() -> Vec<serde_json::Value> {
    vec![serde_json::json!({
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a bash command in the repository working directory",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute"
                    }
                },
                "required": ["command"]
            }
        }
    })]
}

#[derive(Debug, Serialize)]
struct ChatCompletionRequest<'a> {
    model: &'a str,
    messages: &'a [ChatMessage],
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    // `<[_]>::is_empty` rather than `Vec::is_empty`: serde hands the predicate a
    // `&&[Value]`, and slices are what this borrowed request DTO stores.
    #[serde(skip_serializing_if = "<[_]>::is_empty")]
    tools: &'a [serde_json::Value],
    stream: bool,
}

#[derive(Debug, Deserialize)]
struct StreamChunk {
    #[serde(default)]
    choices: Vec<StreamChoice>,
}

#[derive(Debug, Deserialize)]
struct StreamChoice {
    #[serde(default)]
    delta: StreamDelta,
}

#[derive(Debug, Default, Deserialize)]
struct StreamDelta {
    content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<StreamToolCall>,
}

#[derive(Debug, Deserialize)]
struct StreamToolCall {
    #[serde(default)]
    index: usize,
    id: Option<String>,
    #[serde(default)]
    function: Option<StreamFunction>,
}

#[derive(Debug, Deserialize)]
struct StreamFunction {
    name: Option<String>,
    arguments: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Debug, Clone, Deserialize)]
struct ChatChoice {
    message: ChatMessageOutput,
}

#[derive(Debug, Clone, Deserialize)]
struct ChatMessageOutput {
    content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<ToolCallOutput>,
}

#[derive(Debug, Clone, Deserialize)]
struct ToolCallOutput {
    id: String,
    // Same object as the outbound `ToolCall::function`; reusing `ToolCallFn`
    // removes a structurally identical struct without any `Option` juggling.
    function: ToolCallFn,
}

#[derive(Debug, Deserialize)]
struct BashArgs {
    #[serde(alias = "cmd")]
    command: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentStepLog {
    pub step: usize,
    pub command: String,
    pub output: String,
    pub exit_code: Option<i32>,
}

#[derive(Debug, Clone)]
pub struct LlmResponse {
    /// Full text content from the assistant (may be empty if model only used tool_calls)
    pub content: String,
    /// Extracted bash command — from tool_calls first, regex fallback second
    pub command: Option<String>,
    /// Raw tool_calls from the response, for re-insertion into conversation history
    pub tool_calls: Option<Vec<ToolCall>>,
    /// The tool_call id that produced the command (for tool response messages)
    pub tool_call_id: Option<String>,
    /// Number of SSE frames whose bytes were not valid UTF-8 and therefore had
    /// to be decoded lossily. Surfaced so a fleet-wide corruption rate is
    /// observable instead of being silently absorbed.
    pub invalid_utf8_lines: usize,
}



/// Cheap `call_xxxxxxxx` identifier derived from the low 32 bits of a UUID.
///
/// `Uuid::new_v4().to_string()[..8]` formatted the full hyphenated UUID into a
/// throw-away `String` only to slice eight characters off the front.
pub(crate) fn generate_call_id() -> String {
    format!("call_{:08x}", uuid::Uuid::new_v4().as_u128() as u32)
}

/// A tool call being assembled from streaming deltas.
#[derive(Debug, Default)]
struct StreamedToolCall {
    id: String,
    name: String,
    arguments: String,
    /// Set when the call blew past [`MAX_TOOL_ARGUMENT_BYTES`] or the provider
    /// used the same `index` for two different ids. Such calls are dropped
    /// rather than replayed into the conversation history.
    malformed: bool,
}

impl StreamedToolCall {
    /// `true` for a padding / never-populated slot: no id, no name, no args.
    fn is_placeholder(&self) -> bool {
        self.id.trim().is_empty() && self.name.trim().is_empty() && self.arguments.trim().is_empty()
    }
}

/// Streaming state machine shared by the SSE reader and its tests.
#[derive(Debug, Default)]
struct SseAccumulator {
    content: String,
    /// Keyed by the provider's `index`, *not* positional. A sparse index
    /// (`index: 3` on the first frame) used to resize a `Vec` and fabricate
    /// empty placeholder tool calls that were later fed back to the model with
    /// duplicated ids — and whose empty `arguments` shadowed the real command.
    ///
    /// `BTreeMap` keeps deterministic (index-ordered) iteration and collapses
    /// repeated `index` values from re-indexing proxies / retries.
    tools: BTreeMap<usize, StreamedToolCall>,
    /// Set when `content` hit [`MAX_STREAMED_CONTENT_BYTES`].
    content_capped: bool,
    /// Count of frames that were not valid UTF-8 (decoded lossily).
    invalid_utf8_lines: usize,
}

/// Outcome of feeding one complete SSE frame to the accumulator.
#[derive(Debug, PartialEq, Eq)]
enum FrameOutcome {
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
    fn push(&mut self, bytes: &[u8], buffer: &mut Vec<u8>) -> Option<FrameOutcome> {
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
    fn handle_line(&mut self, raw_line: &[u8]) -> FrameOutcome {
        // A line is always complete in the buffer before decoding (framing is
        // newline-delimited), so a multi-byte character split across TCP chunks
        // is safe.
        // Only genuinely malformed bytes can fail here — never drop them
        // silently: log and decode lossily so a corrupted frame degrades
        // visibly instead of vanishing from the model's reply.
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
                // Lossy decode so a corrupted frame degrades visibly (with a
                // replacement char) instead of vanishing from the reply.
                return self.handle_text_line(String::from_utf8_lossy(raw_line).trim());
            }
        };

        self.handle_text_line(trimmed)
    }

    /// Apply the shared SSE filter/parse/accumulate path to one trimmed line.
    fn handle_text_line(&mut self, trimmed: &str) -> FrameOutcome {
        // SSE comments (`: keep-alive`) and empty keep-alive lines.
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
    fn push_content(&mut self, text: &str) {
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
    fn accumulate_tool_call(&mut self, tc: &StreamToolCall) {
        let entry = self.tools.entry(tc.index).or_default();
        if entry.malformed {
            return;
        }

        if let Some(id) = &tc.id {
            if !entry.id.is_empty() && &entry.id != id {
                // Same index, two different ids: the stream is inconsistent and
                // we can no longer tell which id a reply belongs to.
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
    fn finalize_from(tools: &BTreeMap<usize, StreamedToolCall>) -> (Vec<ToolCall>, HashSet<String>) {
        let mut tcs = Vec::with_capacity(tools.len());
        let mut seen_ids: HashSet<String> = HashSet::new();
        for entry in tools.values() {
            if entry.malformed || entry.is_placeholder() {
                continue;
            }
            let mut id = entry.id.trim().to_string();
            if id.is_empty() || !seen_ids.insert(id.clone()) {
                // Providers reject a duplicated `tool_call_id`, so mint a fresh
                // one rather than replaying a colliding id into history.
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
}

pub struct AgentRunner {
    pub http_client: reqwest::Client,
    pub api_base: String,
    pub api_key: String,
    pub model: String,
    pub temperature: Option<f32>,
    /// Idle deadline applied to each SSE body read.
    pub stream_idle_timeout: Duration,
}

impl AgentRunner {
    pub fn new(
        api_base: String,
        api_key: String,
        model: String,
        temperature: Option<f32>,
    ) -> Self {
        let http_client = reqwest::Client::builder()
            .user_agent(format!("mini-swe-mcp/{}", env!("CARGO_PKG_VERSION")))
            .connect_timeout(Duration::from_secs(30))
            .read_timeout(DEFAULT_STREAM_IDLE_TIMEOUT)
            .build()
            .expect("Failed to build HTTP client");

        Self {
            http_client,
            api_base,
            api_key,
            model,
            temperature,
            stream_idle_timeout: DEFAULT_STREAM_IDLE_TIMEOUT,
        }
    }

    /// Override the per-chunk idle deadline (used by tests to keep them fast).
    pub fn with_stream_idle_timeout(mut self, timeout: Duration) -> Self {
        self.stream_idle_timeout = timeout;
        self
    }

    /// Recover a bash command from the first ```bash / ```sh fenced block.
    ///
    /// Uses a single `find()` on a capture-free pattern and slices the body out
    /// of the match by hand, which is materially cheaper than `captures()`.
    /// A literally empty body (```` ```bash\n``` ````) yields `None`, because
    /// the pattern requires a newline before the closing fence; a
    /// whitespace-only body yields `Some("")`, as before.
    pub fn extract_command(&self, text: &str) -> Option<String> {
        let full = BASH_BLOCK_RE.find(text)?.as_str();
        // The match always starts with "```", so the opening line ends at its
        // first newline; the closing fence is the final "\n```" of the match.
        let open_line_end = full.find('\n')?;
        let close_start = full.len() - "\n```".len();
        if close_start <= open_line_end {
            return None;
        }
        Some(full[open_line_end + 1..close_start].trim().to_string())
    }

    pub async fn run_step_llm(&self, messages: &[ChatMessage]) -> Result<LlmResponse> {
        let url = format!("{}/chat/completions", self.api_base.trim_end_matches('/'));

        // Handle models that reject temperature parameter (e.g. kimi-k3)
        let temperature = if self.model.contains("kimi-k3") {
            None
        } else {
            self.temperature.or(Some(0.2))
        };

        let tools = bash_tool();
        let payload = ChatCompletionRequest {
            model: &self.model,
            messages,
            temperature,
            tools: &tools,
            stream: true,
        };

        let mut attempts = 0;
        let accumulator = loop {
            attempts += 1;
            let mut resp = match self
                .http_client
                .post(&url)
                .header("Authorization", format!("Bearer {}", self.api_key))
                .header("Content-Type", "application/json")
                .json(&payload)
                .send()
                .await
            {
                Ok(r) => {
                    let status = r.status();
                    let is_transient = status == reqwest::StatusCode::TOO_MANY_REQUESTS
                        || status == reqwest::StatusCode::SERVICE_UNAVAILABLE
                        || status == reqwest::StatusCode::BAD_GATEWAY
                        || status == reqwest::StatusCode::GATEWAY_TIMEOUT;

                    if is_transient && attempts < 4 {
                        let delay = r
                            .headers()
                            .get(reqwest::header::RETRY_AFTER)
                            .and_then(|v| v.to_str().ok())
                            .and_then(|v| v.parse::<u64>().ok())
                            .map(Duration::from_secs)
                            .unwrap_or_else(|| Duration::from_millis(500 * (1 << (attempts - 1))))
                            .min(Duration::from_secs(10));

                        tracing::warn!(
                            status = %status,
                            attempt = attempts,
                            delay_ms = delay.as_millis(),
                            "LLM API rate-limited or unavailable; retrying with backoff"
                        );
                        tokio::time::sleep(delay).await;
                        continue;
                    }
                    r
                }
                Err(_e) if attempts < 4 => {
                    let delay = Duration::from_millis(500 * (1 << (attempts - 1)));
                    tracing::warn!(
                        attempt = attempts,
                        delay_ms = delay.as_millis(),
                        "LLM API network error; retrying with backoff"
                    );
                    tokio::time::sleep(delay).await;
                    continue;
                }
                Err(e) => return Err(e).context("Failed to send request to LLM API after retries"),
            };

            if !resp.status().is_success() {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                anyhow::bail!("LLM API returned HTTP {}: {}", status, body);
            }

            let mut acc = SseAccumulator::default();
            let mut buffer: Vec<u8> = Vec::new();
            let mut stream_err = false;

            'stream: loop {
                // F7: idle (per-chunk) deadline rather than a whole-request one.
                // Progress resets the clock, so a long but healthy generation is
                // allowed to run indefinitely while a stalled stream is killed
                // and retried.
                let next_chunk =
                    match tokio::time::timeout(self.stream_idle_timeout, resp.chunk()).await {
                        Ok(result) => result,
                        Err(_) => {
                            if attempts < 4 {
                                let delay = Duration::from_millis(500 * (1 << (attempts - 1)));
                                tracing::warn!(
                                    attempt = attempts,
                                    delay_ms = delay.as_millis(),
                                    idle_timeout_secs = self.stream_idle_timeout.as_secs(),
                                    "LLM SSE stream stalled (no chunk within the idle timeout); retrying request with backoff"
                                );
                                tokio::time::sleep(delay).await;
                                stream_err = true;
                                break 'stream;
                            } else {
                                anyhow::bail!(
                                    "LLM SSE stream stalled for {}s and did not resume after {} attempts",
                                    self.stream_idle_timeout.as_secs(),
                                    attempts
                                );
                            }
                        }
                    };

                match next_chunk {
                    Ok(Some(bytes)) => {
                        if acc.push(&bytes, &mut buffer) == Some(FrameOutcome::Done) {
                            break 'stream;
                        }
                    }
                    Ok(None) => break 'stream,
                    Err(e) => {
                        if attempts < 4 {
                            let delay = Duration::from_millis(500 * (1 << (attempts - 1)));
                            tracing::warn!(
                                attempt = attempts,
                                delay_ms = delay.as_millis(),
                                error = %e,
                                "LLM SSE stream chunk read failed; retrying request with backoff"
                            );
                            tokio::time::sleep(delay).await;
                            stream_err = true;
                            break 'stream;
                        } else {
                            return Err(e).context("Failed reading stream chunk after retries");
                        }
                    }
                }
            }

            if stream_err {
                continue;
            }

            // Fallback for a non-streaming response: a proxy may ignore
            // `stream=true` and answer with a single raw JSON body, which lands
            // in the unframed tail of the buffer. `SseAccumulator` reads bytes
            // offset-independently, so the shared line handler applies here too.
            if acc.content.is_empty()
                && acc.tools.is_empty()
                && !buffer.is_empty()
                && let Ok(result) = serde_json::from_slice::<ChatCompletionResponse>(&buffer)
                && let Some(choice) = result.choices.first()
            {
                acc.push_content(choice.message.content.as_deref().unwrap_or(""));
                for (n, tc) in choice.message.tool_calls.iter().enumerate() {
                    let entry = acc.tools.entry(n).or_default();
                    entry.id = tc.id.clone();
                    entry.name = tc.function.name.clone();
                    entry.arguments = tc.function.arguments.clone();
                }
            }

            break acc;
        };

        let SseAccumulator {
            content,
            tools,
            invalid_utf8_lines,
            ..
        } = accumulator;

        if invalid_utf8_lines > 0 {
            tracing::warn!(
                invalid_utf8_lines,
                model = %self.model,
                "Streamed response contained frames that were not valid UTF-8; decoded lossily"
            );
        }

        // Priority 1: extract the command from tool_calls (OpenAI function
        // calling). Only entries with *usable* arguments qualify — a
        // never-populated slot must not shadow the real command (F1).
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
            // Priority 2: fallback to regex extraction from content (code block models)
            .or_else(|| self.extract_command(&content));

        // Convert API tool_calls to history-compatible format, dropping
        // placeholders/malformed calls and guaranteeing unique ids (F1).
        let (finalized, known_ids) = SseAccumulator::finalize_from(&tools);
        let tool_calls = (!finalized.is_empty()).then_some(finalized);
        let tool_call_id = tool_calls.as_ref().and_then(|tcs| {
            bash_tc.and_then(|tc| {
                let candidate = if tc.id.trim().is_empty() {
                    tcs.iter().find(|t| t.function.arguments == tc.arguments).map(|t| t.id.clone())
                } else {
                    Some(tc.id.trim().to_string())
                };
                // Never hand back an id that is not in the history we will send.
                candidate.filter(|id| known_ids.contains(id))
            })
        });

        Ok(LlmResponse {
            content,
            command,
            tool_calls,
            tool_call_id,
            invalid_utf8_lines,
        })
    }

    pub async fn execute_bash(&self, dir: &Path, command: &str) -> Result<(String, Option<i32>)> {
        if let Err(reason) = validate_bash_command(command) {
            return Ok((
                format!(
                    "COMMAND BLOCKED BY WORKTREE GUARDRAIL:\n{}\nPlease run your command within the current repository directory ($PWD).",
                    reason
                ),
                Some(1),
            ));
        }

        let default_parallelism = std::thread::available_parallelism()
            .map(|n| (n.get() / 2).max(1))
            .unwrap_or(2);
        let parallelism = std::env::var("BUILD_PARALLELISM")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default_parallelism)
            .to_string();

        let target_dir = if let Some(custom) = std::env::var_os("CARGO_TARGET_DIR") {
            PathBuf::from(custom)
        } else {
            let dir_name = dir
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("default");
            crate::worktree::swe_base_dir().join(format!("swe-target-{dir_name}"))
        };
        let _ = std::fs::create_dir_all(&target_dir);

        let mut cmd = Command::new("nice");
        cmd.kill_on_drop(true);
        #[cfg(unix)]
        cmd.process_group(0);
        cmd.stdin(std::process::Stdio::null());
        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());

        let use_sandbox = has_bwrap() && std::env::var("SWE_DISABLE_SANDBOX").as_deref() != Ok("1");

        if use_sandbox {
            let dir_str = dir.to_string_lossy();
            let target_str = target_dir.to_string_lossy();

            cmd.args([
                "-n",
                "10",
                "bwrap",
                "--die-with-parent",
                "--new-session",
                "--unshare-pid",
                "--unshare-ipc",
                "--ro-bind",
                "/usr",
                "/usr",
                "--symlink",
                "usr/bin",
                "/bin",
                "--symlink",
                "usr/bin",
                "/sbin",
                "--symlink",
                "usr/lib",
                "/lib",
                "--symlink",
                "usr/lib",
                "/lib64",
                "--ro-bind-try",
                "/etc",
                "/etc",
                "--proc",
                "/proc",
                "--dev",
                "/dev",
                "--tmpfs",
                "/tmp",
            ]);

            // Isolate user home: mount empty tmpfs, expose only toolchain caches read-only
            if let Some(home) = std::env::var_os("HOME").map(PathBuf::from) {
                let home_str = home.to_string_lossy();
                cmd.args(["--tmpfs", &home_str]);
                for cache_dir in [".cargo", ".rustup", ".local/bin"] {
                    let full = home.join(cache_dir);
                    if full.exists() {
                        let full_str = full.to_string_lossy();
                        cmd.args(["--ro-bind-try", &full_str, &full_str]);
                    }
                }
                let cache_tmp = home.join(".cache");
                let cache_tmp_str = cache_tmp.to_string_lossy();
                cmd.args(["--tmpfs", &cache_tmp_str]);
            }

            if let Some(cargo_home) = std::env::var_os("CARGO_HOME").map(PathBuf::from)
                && cargo_home.exists()
            {
                let p = cargo_home.to_string_lossy();
                cmd.args(["--ro-bind-try", &p, &p]);
            }
            if let Some(rustup_home) = std::env::var_os("RUSTUP_HOME").map(PathBuf::from)
                && rustup_home.exists()
            {
                let p = rustup_home.to_string_lossy();
                cmd.args(["--ro-bind-try", &p, &p]);
            }

            // Expose the worktree directory read-write
            cmd.args(["--bind", &dir_str, &dir_str]);

            // Expose the common .git directory READ-ONLY so git can resolve refs/objects
            // without permitting the sandbox to prune or delete repository branches!
            if let Some((common_git, worktree_gitdir)) = find_git_dirs(dir) {
                let common_str = common_git.to_string_lossy();
                cmd.args(["--ro-bind", &common_str, &common_str]);

                // Expose ONLY this worker's worktree gitdir read-write so it can update its local index
                if let Some(wt_gitdir) = worktree_gitdir
                    && wt_gitdir.is_dir()
                {
                    let wt_str = wt_gitdir.to_string_lossy();
                    cmd.args(["--bind", &wt_str, &wt_str]);
                }
            }

            // Bind isolated build target directory read-write
            cmd.args(["--bind", &target_str, &target_str]);

            // Modular shared package/compiler caches
            let home_path = std::env::var_os("HOME").map(PathBuf::from);
            crate::cache::append_bwrap_cache_args(&mut cmd, home_path.as_deref());

            // Working directory
            cmd.args(["--chdir", &dir_str]);

            // Bash command
            cmd.args(["/usr/bin/bash", "-c", command]);
        } else {
            cmd.current_dir(dir)
                .args(["-n", "10", "bash", "-c", command]);
        }

        // Universal build and test parallelism caps
        cmd.env("CARGO_TARGET_DIR", &target_dir)
            .env("CARGO_BUILD_JOBS", &parallelism)
            .env("RUST_TEST_THREADS", &parallelism)
            .env("NEXTEST_TEST_THREADS", &parallelism)
            .env("MAKEFLAGS", format!("-j{parallelism}"))
            .env("CMAKE_BUILD_PARALLEL_LEVEL", &parallelism)
            .env("RAYON_NUM_THREADS", &parallelism)
            .env("OMP_NUM_THREADS", &parallelism)
            .env("OPENBLAS_NUM_THREADS", &parallelism)
            .env("MKL_NUM_THREADS", &parallelism)
            .env("GOMAXPROCS", &parallelism);

        // Modular shared compiler & package manager cache environment
        crate::cache::apply_shared_cache_env(&mut cmd);

        let is_heavy = is_heavy_command(command);
        let default_timeout = if is_heavy {
            std::env::var("COMMAND_HEAVY_TIMEOUT_SECS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(600)
        } else {
            std::env::var("COMMAND_LIGHT_TIMEOUT_SECS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(120)
        };
        let timeout_secs = std::env::var("COMMAND_TIMEOUT_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default_timeout);
        let timeout_duration = Duration::from_secs(timeout_secs);
        let child = cmd.spawn().context("Failed to spawn bash process")?;
        let child_pid = child.id();

        let output_res = tokio::time::timeout(timeout_duration, child.wait_with_output()).await;

        let output = match output_res {
            Ok(Ok(out)) => out,
            Ok(Err(e)) => return Err(e).context("Failed waiting for bash process"),
            Err(_) => {
                #[cfg(unix)]
                if let Some(pid) = child_pid {
                    let _ = std::process::Command::new("kill")
                        .args(["-KILL", &format!("-{pid}")])
                        .status();
                }
                return Ok((
                    format!(
                        "Command timed out after {}s and was terminated.",
                        timeout_secs
                    ),
                    Some(124),
                ));
            }
        };

        // NOTE: this still materialises stdout+stderr in full before truncating,
        // so peak memory here remains O(output size). Streaming truncation (a
        // head buffer plus a rolling tail, never holding the middle) is the
        // remaining follow-up; see audit 05 section 6. In the meantime,
        // `from_utf8_lossy` already returns a `Cow`, so push it directly rather
        // than forcing an extra deep copy of every stream byte via `.to_string()`.
        let mut combined = String::new();
        if !output.stdout.is_empty() {
            combined.push_str(&String::from_utf8_lossy(&output.stdout));
        }
        if !output.stderr.is_empty() {
            if !combined.is_empty() {
                combined.push('\n');
            }
            combined.push_str(&String::from_utf8_lossy(&output.stderr));
        }

        let combined = truncate_output(&combined);

        Ok((combined, output.status.code()))
    }
}

/// Byte budget above which captured command output is truncated.
pub const TRUNCATE_LIMIT: usize = 16_384;
/// Bytes retained from the *start* of over-budget output (floored to a
/// character boundary).
pub const TRUNCATE_HEAD: usize = 12_288;
/// Bytes retained from the *end* of over-budget output (ceiled to a character
/// boundary).
pub const TRUNCATE_TAIL: usize = 4_096;

/// Literal text of the marker inserted in place of the discarded middle,
/// excluding the decimal byte count that is rendered between the two halves.
const TRUNCATE_MARKER: &str = "\n... [Truncated ";
/// Literal text of the second half of the marker, after the byte count.
const TRUNCATE_MARKER_SUFFIX: &str = " bytes] ...\n";
/// Upper bound on the decimal digits of a `usize` (2^64 - 1 has 20 digits).
/// Used to size the result buffer up front so the marker needs no allocation.
const USIZE_MAX_DIGITS: usize = 20;

/// Bound command output to [`TRUNCATE_LIMIT`] bytes, keeping the head and the
/// tail of the text and reporting how many bytes were discarded.
///
/// Both cut points are snapped to UTF-8 character boundaries (`floor` for the
/// head, `ceil` for the tail), so no character is ever split and
/// `head + dropped + tail == input.len()` holds exactly.
///
/// The result is assembled **once** into an exactly-sized `String`: the marker
/// is pushed directly (the byte count is rendered into a stack buffer) and the
/// 4 KiB tail is never copied through an intermediate allocation.
pub fn truncate_output(combined: &str) -> String {
    let total = combined.len();
    if total <= TRUNCATE_LIMIT {
        // Fast path: the output fits the budget, so hand back one copy of it.
        return combined.to_string();
    }

    // `total > TRUNCATE_LIMIT > TRUNCATE_TAIL`, so this cannot underflow.
    let head_end = combined.floor_char_boundary(TRUNCATE_HEAD);
    let tail_start = combined.ceil_char_boundary(total - TRUNCATE_TAIL);
    let dropped = total - (head_end + (total - tail_start));

    // Exact capacity: head + marker + digits + suffix + tail. Sizing it
    // correctly means the result needs a single allocation and no reallocation.
    let mut out = String::with_capacity(
        head_end + TRUNCATE_MARKER.len() + USIZE_MAX_DIGITS + TRUNCATE_MARKER_SUFFIX.len()
            + (total - tail_start),
    );
    // Render the byte count into a stack buffer so the marker adds no allocation.
    let mut digits = [0u8; USIZE_MAX_DIGITS];
    let count = render_decimal(&mut digits, dropped);

    out.push_str(&combined[..head_end]);
    out.push_str(TRUNCATE_MARKER);
    out.push_str(count);
    out.push_str(TRUNCATE_MARKER_SUFFIX);
    out.push_str(&combined[tail_start..]);
    debug_assert!(
        out.capacity() >= out.len(),
        "result buffer must be sized up front"
    );
    out
}

/// Render `value` as decimal ASCII digits into `buf` and return the used
/// prefix as a `&str`.
///
/// Digits are written right-to-left into the end of `buf`; the returned slice
/// is valid UTF-8 because every byte written is an ASCII digit. This lets the
/// truncation marker embed its byte count without any heap allocation.
fn render_decimal(buf: &mut [u8; USIZE_MAX_DIGITS], mut value: usize) -> &str {
    debug_assert!(value > 0, "the marker is only emitted with a dropped region");
    let mut idx = buf.len();
    while value > 0 {
        idx -= 1;
        buf[idx] = b'0' + u8::try_from(value % 10).expect("remainder is a single digit");
        value /= 10;
    }
    // Safe: the written prefix is ASCII.
    std::str::from_utf8(&buf[idx..]).expect("ASCII digits are valid UTF-8")
}

/// Validate that a subagent command does not attempt to escape the worktree
/// or trigger runaway recursive scans of root or home filesystems.
pub fn validate_bash_command(command: &str) -> Result<(), &'static str> {
    let trimmed = command.trim();

    // 1. Block recursive searches starting at root, home, or system directories
    const FORBIDDEN_SEARCHES: &[&str] = &[
        "find / ",
        "find / -",
        "find /\"",
        "find /'",
        "find ~",
        "find /home",
        "find /root",
        "find /etc",
        "find /var",
        "find /usr",
        "grep -rn / ",
        "grep -r / ",
    ];

    for token in FORBIDDEN_SEARCHES {
        if trimmed.contains(token) {
            return Err(
                "Scanning root '/' or system directories is forbidden. Confine searches to the current repository ($PWD).",
            );
        }
    }

    // 2. Block escaping to parent or root directory via cd
    const FORBIDDEN_CDS: &[&str] = &[
        "cd / ", "cd /;", "cd /&&", "cd /||", "cd /home", "cd ~", "cd $HOME", "cd /root",
    ];

    for token in FORBIDDEN_CDS {
        if trimmed.contains(token) || trimmed.ends_with("cd /") {
            return Err(
                "Navigating outside the repository with 'cd' is forbidden. All files are in $PWD.",
            );
        }
    }

    Ok(())
}

/// Distinguish CPU-heavy commands (compilations, test runners) from
/// lightweight exploration commands (git status, cat, ls, grep, etc.).
pub fn is_heavy_command(command: &str) -> bool {
    let lower = command.to_lowercase();
    if lower.starts_with("cargo") || lower.contains("cargo ") || lower.contains("cargo\t") {
        return true;
    }
    if lower == "make"
        || lower.starts_with("make ")
        || lower.contains(" make ")
        || lower.contains(" make\t")
    {
        return true;
    }
    const HEAVY_PATTERNS: &[&str] = &[
        "rustc", "pytest", "unittest", "cmake", "ninja", "gcc", "g++", "clang", "npm ", "yarn ",
        "pnpm ", "mvn ", "gradle", "go test", "go build",
    ];
    HEAVY_PATTERNS.iter().any(|pattern| lower.contains(pattern))
}

/// Check if the bubblewrap (`bwrap`) sandbox utility is available on this system.
pub fn has_bwrap() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        std::process::Command::new("bwrap")
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    })
}

/// If a worktree's `.git` is a gitdir reference pointing to a parent git directory,
/// locate both the common `.git` directory and the specific worktree gitdir.
pub fn find_git_dirs(worktree_dir: &Path) -> Option<(PathBuf, Option<PathBuf>)> {
    let dot_git = worktree_dir.join(".git");
    if dot_git.is_file()
        && let Ok(content) = std::fs::read_to_string(&dot_git)
        && let Some(gitdir_line) = content.lines().find(|l| l.starts_with("gitdir: "))
    {
        let raw_path = gitdir_line.trim_start_matches("gitdir: ").trim();
        let gitdir_path = PathBuf::from(raw_path);
        for ancestor in gitdir_path.ancestors() {
            if ancestor.file_name().and_then(|n| n.to_str()) == Some(".git") {
                return Some((ancestor.to_path_buf(), Some(gitdir_path)));
            }
        }
    }
    None
}

/// Backwards-compatible helper returning only the common `.git` root.
pub fn find_git_common_dir(worktree_dir: &Path) -> Option<PathBuf> {
    find_git_dirs(worktree_dir).map(|(common, _)| common)
}

#[cfg(test)]
mod tests {
    use super::{
        AgentRunner, ChatCompletionRequest, ChatMessage, FrameOutcome, Role, SseAccumulator,
        BASH_BLOCK_RE, MAX_STREAMED_CONTENT_BYTES,
    };

    fn runner() -> AgentRunner {
        AgentRunner::new(
            "http://localhost".to_string(),
            "test-key".to_string(),
            "test-model".to_string(),
            None,
        )
    }

    /// The `tools` array must keep the exact OpenAI function-calling shape after
    /// `ToolDefinition`/`ToolFunction` were inlined, and must be omitted when
    /// the slice is empty (a behaviour the old always-one-element slice could not
    /// express).
    #[test]
    fn test_chat_completion_request_tools_wire_shape() {
        let messages = [ChatMessage::text(Role::User, "hi")];
        let tools = super::bash_tool();

        let with_tools = ChatCompletionRequest {
            model: "test-model",
            messages: &messages,
            temperature: Some(0.2),
            tools: &tools,
            stream: true,
        };
        let body = serde_json::to_string(&with_tools).unwrap();
        let value: serde_json::Value = serde_json::from_str(&body).unwrap();
        // `serde_json::Value` objects are sorted maps (no `preserve_order`
        // feature), so the emitted key order is deterministic even though the
        // literal in `bash_tool()` is written in OpenAI spec order.
        assert_eq!(
            body,
            concat!(
                r#"{"model":"test-model","messages":[{"role":"user","content":"hi"}],"#,
                r#""temperature":0.2,"tools":[{"function":{"description":"Execute a "#,
                r#"bash command in the repository working directory","name":"bash","#,
                r#""parameters":{"properties":{"command":{"description":"The bash "#,
                r#"command to execute","type":"string"}},"required":["command"],"#,
                r#""type":"object"}},"type":"function"}],"stream":true}"#
            ),
            "request body drifted from the pinned wire shape"
        );
        let tool = &value["tools"][0];
        assert_eq!(tool["type"], "function");
        assert_eq!(tool["function"]["name"], "bash");
        assert_eq!(
            tool["function"]["description"],
            "Execute a bash command in the repository working directory"
        );
        assert_eq!(tool["function"]["parameters"]["type"], "object");
        assert_eq!(
            tool["function"]["parameters"]["properties"]["command"]["type"],
            "string"
        );
        assert_eq!(
            tool["function"]["parameters"]["required"],
            serde_json::json!(["command"])
        );

        let no_tools: Vec<serde_json::Value> = Vec::new();
        let without_tools = ChatCompletionRequest {
            model: "test-model",
            messages: &messages,
            temperature: None,
            tools: &no_tools,
            stream: true,
        };
        let value: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(&without_tools).unwrap()).unwrap();
        assert!(
            value.get("tools").is_none(),
            "empty tool slice must be omitted, got {value}"
        );
        assert!(value.get("temperature").is_none());
    }

    /// `Role` serializes to the lowercase strings the API expects; the enum is
    /// the only source of role values now, so this pins the wire contract.
    #[test]
    fn test_role_wire_strings() {
        for role in [Role::System, Role::User, Role::Assistant, Role::Tool] {
            let expected = role.as_wire_str();
            assert_eq!(serde_json::to_string(&role).unwrap(), format!("\"{expected}\""));
            let back: Role = serde_json::from_str(&format!("\"{expected}\"")).unwrap();
            assert_eq!(back, role);
        }
    }

    /// The inbound non-streaming fallback reuses `ToolCallFn` for
    /// `ToolCallOutput.function`; it must still parse the OpenAI shape.
    #[test]
    fn test_non_stream_tool_call_output_parses() {
        let raw = r#"{
            "choices": [{
                "message": {
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "bash", "arguments": "{\"command\":\"ls\"}"}
                    }]
                }
            }]
        }"#;
        let parsed: super::ChatCompletionResponse = serde_json::from_str(raw).unwrap();
        let choice = parsed.choices.first().unwrap();
        assert!(choice.message.content.is_none());
        let tc = &choice.message.tool_calls[0];
        assert_eq!(tc.id, "call_1");
        assert_eq!(tc.function.name, "bash");
        assert_eq!(tc.function.arguments, r#"{"command":"ls"}"#);
    }

    /// `text(Role::Tool, ..)` is the one misuse the type system alone cannot
    /// catch (the role is legal, but a `tool` message also needs a
    /// `tool_call_id`), so the constructor asserts against it. Debug builds
    /// panic; release builds keep the previous behaviour of emitting a
    /// `tool_call_id`-less tool message rather than failing the whole turn.
    #[test]
    #[should_panic(expected = "tool messages require a tool_call_id")]
    fn test_text_rejects_tool_role() {
        let _ = ChatMessage::text(Role::Tool, "raw output");
    }

    /// An assistant turn must never carry `tool_calls: []`, which the API
    /// rejects; the constructor normalises it to an omitted field.
    #[test]
    fn test_assistant_with_empty_tool_calls_is_normalised() {
        let msg = ChatMessage::assistant_with_tool_calls(Some("thinking".into()), Vec::new());
        let value: serde_json::Value = serde_json::from_str(&serde_json::to_string(&msg).unwrap()).unwrap();
        assert_eq!(msg.role(), Role::Assistant);
        assert!(value.get("tool_calls").is_none(), "got {value}");
        assert_eq!(value["content"], "thinking");
    }

    #[test]
    fn extract_command_single_line_bash() {
        let reply = "Run this:\n```bash\necho hello\n```";

        assert_eq!(
            runner().extract_command(reply),
            Some("echo hello".to_string())
        );
    }

    #[test]
    fn extract_command_multiline_bash() {
        let reply = "```bash\ncd /tmp\nls -la\n```";

        assert_eq!(
            runner().extract_command(reply),
            Some("cd /tmp\nls -la".to_string())
        );
    }

    #[test]
    fn extract_command_missing_bash_block() {
        let reply = "There is no command block in this response.";

        assert_eq!(runner().extract_command(reply), None);
    }

    #[test]
    fn test_truncate_output_utf8_boundary() {
        let mut s = "a".repeat(super::TRUNCATE_HEAD - 1);
        s.push('€'); // bytes 12287..12290
        s.push_str(&"b".repeat(9000));
        let truncated = super::truncate_output(&s);
        assert!(truncated.contains("... [Truncated"));
    }

    #[test]
    fn test_truncate_output_constants_match_the_budget() {
        use super::{TRUNCATE_HEAD, TRUNCATE_LIMIT, TRUNCATE_TAIL};
        // The head and tail budgets make up the truncation limit.
        const { assert!(TRUNCATE_HEAD + TRUNCATE_TAIL == TRUNCATE_LIMIT) };
        // Both slices are strictly smaller than the limit, so the
        // `total - TRUNCATE_TAIL` subtraction inside the truncation branch
        // cannot underflow.
        const { assert!(TRUNCATE_TAIL < TRUNCATE_LIMIT && TRUNCATE_HEAD < TRUNCATE_LIMIT) };
    }

    #[test]
    fn test_validate_bash_command_blocks_escapes() {
        use super::validate_bash_command;

        // Blocked: root find
        assert!(validate_bash_command("find / -name 'foo'").is_err());
        assert!(validate_bash_command("find ~ -name 'foo'").is_err());
        assert!(validate_bash_command("find /home -name 'foo'").is_err());

        // Blocked: cd to root or home
        assert!(validate_bash_command("cd / && ls").is_err());
        assert!(validate_bash_command("cd /home && ls").is_err());
        assert!(validate_bash_command("cd ~").is_err());
        assert!(validate_bash_command("cd /").is_err());

        // Allowed: within worktree
        assert!(validate_bash_command("find . -name 'foo'").is_ok());
        assert!(validate_bash_command("find src -type f").is_ok());
        assert!(validate_bash_command("cd src && cargo test").is_ok());
        assert!(validate_bash_command("grep -rn 'WorkerState' src/").is_ok());
    }

    #[test]
    fn test_is_heavy_command() {
        use super::is_heavy_command;

        assert!(is_heavy_command("cargo build"));
        assert!(is_heavy_command("cargo test --all"));
        assert!(is_heavy_command("cargo"));
        assert!(is_heavy_command("pytest tests/"));
        assert!(is_heavy_command("make -j4"));
        assert!(is_heavy_command("make"));
        assert!(is_heavy_command("gcc -O3 main.c"));

        assert!(!is_heavy_command("git status"));
        assert!(!is_heavy_command("git diff HEAD"));
        assert!(!is_heavy_command("ls -la"));
        assert!(!is_heavy_command("cat src/agent.rs"));
        assert!(!is_heavy_command("find . -name '*.rs'"));
        assert!(!is_heavy_command(
            "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
        ));
    }

    #[test]
    fn test_has_bwrap_returns_boolean() {
        let _ = super::has_bwrap();
    }

    #[test]
    fn test_find_git_common_dir_on_regular_dir() {
        let tmp = std::env::temp_dir();
        assert_eq!(super::find_git_common_dir(&tmp), None);
    }

    // ---- SSE accumulator internals -------------------------------------

    fn sse_body(frames: &[&str]) -> Vec<u8> {
        let mut v = Vec::new();
        for f in frames {
            v.extend_from_slice(format!("data: {f}\n\n").as_bytes());
        }
        v.extend_from_slice(b"data: [DONE]\n\n");
        v
    }

    fn tc(index: usize, id: Option<&str>, name: Option<&str>, args: Option<&str>) -> super::StreamToolCall {
        let function = if name.is_some() || args.is_some() {
            Some(super::StreamFunction {
                name: name.map(str::to_string),
                arguments: args.map(str::to_string),
            })
        } else {
            None
        };
        super::StreamToolCall { index, id: id.map(str::to_string), function }
    }

    /// F3: when the consumed prefix reaches the end of the buffer, it is
    /// cleared outright instead of memmoving an empty remainder.
    #[test]
    fn push_clears_buffer_when_fully_consumed() {
        let mut acc = SseAccumulator::default();
        let mut buffer = Vec::new();
        let body = sse_body(&[r#"{"choices":[{"delta":{"content":"x"}}]}"#]);
        assert_eq!(acc.push(&body, &mut buffer), Some(FrameOutcome::Done));
        assert!(buffer.is_empty(), "fully consumed buffer must be cleared");
    }

    /// F3/F4: an unterminated tail survives; the consumed prefix is dropped
    /// without disturbing it.
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

    /// F4: byte-at-a-time delivery of a large frame is reassembled exactly, and
    /// the framing pass never needs a re-scan of consumed bytes.
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

    /// A `data:` line that is not the sentinel and not valid JSON is skipped
    /// without poisoning the stream.
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

    /// F2: a frame with invalid UTF-8 is counted and its bytes survive a lossy
    /// decode rather than vanishing.
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

    /// F5: content retention is capped and the cut never splits a character.
    #[test]
    fn push_content_respects_cap_and_char_boundaries() {
        let mut acc = SseAccumulator::default();
        acc.push_content(&"€".repeat(MAX_STREAMED_CONTENT_BYTES));
        assert!(acc.content.len() <= MAX_STREAMED_CONTENT_BYTES);
        assert!(acc.content_capped, "overflow must be flagged");
        assert!(!acc.content.contains('\u{FFFD}'), "no partial code point");
        assert!(acc.content.is_char_boundary(acc.content.len()));
    }

    /// F5: `content.push_str` stops appending once the cap is reached.
    #[test]
    fn push_content_ignores_deltas_after_the_cap() {
        let mut acc = SseAccumulator::default();
        acc.push_content(&"a".repeat(MAX_STREAMED_CONTENT_BYTES));
        let before = acc.content.len();
        acc.push_content("more text");
        assert_eq!(acc.content.len(), before, "no growth past the cap");
    }

    /// F1: a sparse index must not fabricate placeholder calls at finalization.
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

    /// F1: an entirely empty entry is a placeholder and is filtered out.
    #[test]
    fn empty_entry_is_filtered_as_placeholder() {
        let mut acc = SseAccumulator::default();
        acc.tools.insert(0, Default::default());
        let (tcs, _) = SseAccumulator::finalize_from(&acc.tools);
        assert!(tcs.is_empty(), "placeholder must not reach history");
    }

    /// F1: two entries sharing a provider id get unique replacements, because
    /// providers reject a duplicated `tool_call_id` on the next turn.
    #[test]
    fn duplicate_ids_are_made_unique() {
        let mut acc = SseAccumulator::default();
        acc.accumulate_tool_call(&tc(0, Some("dup"), Some("bash"), Some(r#"{"command":"a"}"#)));
        acc.accumulate_tool_call(&tc(1, Some("dup"), Some("bash"), Some(r#"{"command":"b"}"#)));
        let (tcs, ids) = SseAccumulator::finalize_from(&acc.tools);
        assert_eq!(tcs.len(), 2);
        assert_ne!(tcs[0].id, tcs[1].id, "ids must be unique");
        assert_eq!(ids.len(), 2);
        assert!(tcs.iter().all(|c| !c.id.is_empty()));
    }

    /// F1: an entry with no id still receives a generated one.
    #[test]
    fn missing_id_is_generated() {
        let mut acc = SseAccumulator::default();
        acc.accumulate_tool_call(&tc(0, None, Some("bash"), Some(r#"{"command":"a"}"#)));
        let (tcs, _) = SseAccumulator::finalize_from(&acc.tools);
        assert_eq!(tcs.len(), 1);
        assert!(tcs[0].id.starts_with("call_") && tcs[0].id.len() == 13);
    }

    /// A nameless entry defaults to the `bash` tool.
    #[test]
    fn missing_name_defaults_to_bash() {
        let mut acc = SseAccumulator::default();
        acc.accumulate_tool_call(&super::StreamToolCall {
            index: 0,
            id: Some("i".into()),
            function: Some(super::StreamFunction {
                name: None,
                arguments: Some(r#"{"command":"a"}"#.into()),
            }),
        });
        let (tcs, _) = SseAccumulator::finalize_from(&acc.tools);
        assert_eq!(tcs[0].function.name, "bash");
    }

    /// A repeated index accumulates by concatenation rather than creating a
    /// second call.
    #[test]
    fn repeated_index_accumulates_arguments() {
        let mut acc = SseAccumulator::default();
        acc.accumulate_tool_call(&tc(0, Some("c"), Some("bash"), Some(r#"{"comm"#)));
        acc.accumulate_tool_call(&tc(0, None, None, Some(r#"and":"pwd"}"#)));
        let (tcs, _) = SseAccumulator::finalize_from(&acc.tools);
        assert_eq!(tcs.len(), 1);
        assert_eq!(tcs[0].function.arguments, r#"{"command":"pwd"}"#);
    }

    /// F1: two different ids for the same index make the call ambiguous; it is
    /// dropped rather than replayed with an unverifiable identity.
    #[test]
    fn conflicting_ids_for_one_index_mark_call_malformed() {
        let mut acc = SseAccumulator::default();
        acc.accumulate_tool_call(&tc(0, Some("a"), Some("bash"), Some("{}")));
        acc.accumulate_tool_call(&tc(0, Some("b"), None, None));
        let (tcs, _) = SseAccumulator::finalize_from(&acc.tools);
        assert!(tcs.is_empty(), "ambiguous call must be dropped: {tcs:?}");
    }

    /// F5: an over-long `arguments` stream is flagged malformed and stops
    /// buffering, so memory stays bounded.
    #[test]
    fn oversized_arguments_mark_call_malformed() {
        let mut acc = SseAccumulator::default();
        let chunk = "a".repeat(1024);
        let rounds = super::MAX_TOOL_ARGUMENT_BYTES / chunk.len() + 2;
        for _ in 0..rounds {
            acc.accumulate_tool_call(&tc(0, Some("x"), Some("bash"), Some(&chunk)));
        }
        let entry = acc.tools.get(&0).expect("entry exists");
        assert!(entry.malformed, "call must be flagged malformed");
        assert!(entry.arguments.len() <= super::MAX_TOOL_ARGUMENT_BYTES);
        let (tcs, _) = SseAccumulator::finalize_from(&acc.tools);
        assert!(tcs.is_empty(), "malformed call must not reach history");
    }

    /// F5: a malformed call is abandoned — later deltas do not revive it.
    #[test]
    fn malformed_call_ignores_later_deltas() {
        let mut acc = SseAccumulator::default();
        let chunk = "a".repeat(super::MAX_TOOL_ARGUMENT_BYTES + 1);
        acc.accumulate_tool_call(&tc(0, Some("x"), Some("bash"), Some(&chunk)));
        assert!(acc.tools.get(&0).expect("entry").malformed);
        acc.accumulate_tool_call(&tc(0, Some("x"), Some("bash"), Some("{}")));
        let entry = acc.tools.get(&0).expect("entry");
        assert!(entry.malformed, "must stay malformed");
        assert!(entry.arguments.len() <= super::MAX_TOOL_ARGUMENT_BYTES);
    }

    /// Entries are emitted in provider index order, so history is stable.
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

    /// F6: generated ids are unique and well-formed without a throw-away
    /// hyphenated `Uuid` string in the middle.
    #[test]
    fn generate_call_id_is_unique_and_prefixed() {
        let a = super::generate_call_id();
        let b = super::generate_call_id();
        assert!(a.starts_with("call_"), "{a}");
        assert_eq!(a.len(), 13, "call_ + 8 hex chars: {a}");
        assert!(a[5..].chars().all(|c| c.is_ascii_hexdigit()), "{a}");
        assert_ne!(a, b, "ids must not collide");
    }



    #[tokio::test]
    async fn test_execute_bash_sandbox_runs_and_blocks_write() {
        let tmp = std::env::temp_dir().join(format!("swe-test-bwrap-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&tmp);
        let r = runner();

        // 1. Basic command within worktree succeeds
        let (out, code) = r
            .execute_bash(&tmp, "echo 'hello from sandbox'")
            .await
            .unwrap();
        assert_eq!(code, Some(0));
        assert!(out.contains("hello from sandbox"));

        // 2. Writing to read-only host root /usr fails when bwrap is active
        if super::has_bwrap() {
            let (out, code) = r
                .execute_bash(&tmp, "touch /usr/forbidden_write_test 2>&1")
                .await
                .unwrap();
            assert_ne!(code, Some(0));
            assert!(
                out.contains("Read-only")
                    || out.contains("sólo lectura")
                    || out.contains("Permission denied")
            );
        }

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn bash_block_regex_is_a_process_wide_static() {
        // The extractor pattern is a compile-time literal, so it is memoized in
        // a `static LazyLock`: one compiled program for the whole process, shared
        // by every `AgentRunner` (one per worker) instead of one per instance.
        let a: &regex::Regex = &BASH_BLOCK_RE;
        let b: &regex::Regex = &BASH_BLOCK_RE;
        assert!(
            std::ptr::eq(a, b),
            "the memoized regex must be a single shared instance"
        );
        assert_eq!(a.as_str(), r"(?s)```(?:bash|sh)[ \t\r\n]*\n.*?\n```");
    }

    #[test]
    fn extract_command_is_instance_independent() {
        // Separate `AgentRunner`s (as created per worker) must extract
        // identically, since they all read the shared static.
        let one = runner();
        let two = AgentRunner::new(
            "http://example.invalid".to_string(),
            "other-key".to_string(),
            "other-model".to_string(),
            Some(0.7),
        );

        for reply in [
            "```bash\necho one\n```",
            "```sh\nmake test\n```",
            "```bash\ncd /tmp\nls -la\n```",
            "no block here",
            "```rust\nfn main() {}\n```",
        ] {
            assert_eq!(
                one.extract_command(reply),
                two.extract_command(reply),
                "extract_command must not depend on the runner instance: {reply}"
            );
        }
    }
}
