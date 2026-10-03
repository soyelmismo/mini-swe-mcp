use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Deserialize a field, mapping an explicit `null` to `Default::default()`.
///
/// `#[serde(default)]` only fills in an *absent* key: `null` still has to match
/// the field type, so `"tool_calls": null` failed the whole chunk and its
/// siblings (content, reasoning) with it. This deserializer accepts both, which
/// is what providers actually send.
fn deserialize_null_tolerant<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
    D: serde::Deserializer<'de>,
    T: serde::Deserialize<'de> + Default,
{
    Ok(Option::<T>::deserialize(deserializer)?.unwrap_or_default())
}

/// Idle (per-chunk) deadline for the SSE body read.
///
/// A *whole-request* deadline is the wrong tool for a token stream: it kills
/// healthy-but-slow generations regardless of progress. Only a per-chunk
/// `tokio::time::timeout` aborts genuine stalls, so no `read_timeout` is set
/// on the HTTP client.
pub const DEFAULT_STREAM_IDLE_TIMEOUT: Duration = Duration::from_secs(120);

/// Hard cap on the assistant text retained from a stream.
///
/// Matched to [`MAX_TOOL_ARGUMENT_BYTES`] so a turn's reasoning and its tool
/// arguments share one budget: at 16 KiB long chain-of-thought replies were
/// truncated mid-sentence, corrupting the context fed back to the model.
/// Enforced in [`crate::agent::stream`], so a runaway stream can neither inflate
/// memory nor be re-sent verbatim on the next request.
pub const MAX_STREAMED_CONTENT_BYTES: usize = 64 * 1024;

/// Hard cap on the serialized `arguments` accumulated for one tool call.
/// A model streaming megabytes of arguments is treated as malformed: the call is
/// dropped rather than buffered.
pub const MAX_TOOL_ARGUMENT_BYTES: usize = 64 * 1024;

/// Byte size of a typical SSE frame; pre-reserves the read buffer so a long
/// stream stops reallocating as chunks arrive.
pub(crate) const SSE_BUFFER_HINT_BYTES: usize = 8 * 1024;

/// Hard cap on a single SSE line retained or parsed by the framing buffer.
///
/// `content` and tool-call `arguments` are budgeted, the raw framing buffer is
/// not: a provider (or proxy) streaming bytes with no newline would otherwise
/// make the buffer absorb the whole body with no upper bound. Longer tails are
/// dropped and the reader resyncs on the next newline, so an ill-formed stream
/// degrades instead of exhausting memory. Sized well above
/// [`SSE_BUFFER_HINT_BYTES`] and [`MAX_TOOL_ARGUMENT_BYTES`].
pub(crate) const MAX_SSE_FRAME_BYTES: usize = 1024 * 1024;

pub const SYSTEM_PROMPT: &str = r#"You are an autonomous software engineering subagent running in a Linux bash environment.
You are given a task to complete within a git repository.

LOCATION & SCOPE:
- You are ALREADY located at the root of the repository worktree ($PWD).
- Never execute `cd` to parent directories (like /home/rot, /repo, or /). All repository files are right here in the current directory.

WORKFLOW:
1. Explore: Use tools like `git status`, `find`, `grep -rn`, or `ls` to locate relevant files in the current repository.
2. Edit & Test: Make minimal, clean edits (using sed, python, cat << 'EOF', etc.) and run existing test suites to verify.
3. Every response MUST execute EXACTLY ONE command using the `bash` tool. If the bash tool is unavailable, use a ```bash ... ``` code block instead.
4. When finished, your last message must contain this REPORT block, then the sentinel:
   REPORT
   done: <one line: what changed>
   files: <paths changed, comma-separated>
   tests: <the commands run and their result, one line>
   risks: <security/contract/behaviour risks, or none>
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
  This will immediately pause execution until the orchestrator replies with guidance.

DISCIPLINE:
1. Before echoing COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT on a code task, run the project's tests and linters in the current state and see them pass.
2. Edit files in place; never create parallel copies, backups or encoded dumps in scratch dirs and iterate on them.
3. Check `git diff --stat` regularly; if several turns pass without measurable progress, change approach or ASK_ORCHESTRATOR instead of repeating.
4. Never re-run a command whose output you already have.
5. Use REQUEST_TURNS only with demonstrable progress and a short concrete plan.
6. Before deleting or renaming a symbol, grep for every use (tests, docs, re-exports) and update them in the same change.
7. When editing comments, leave complete sentences; no orphaned fragments.
8. Do not add code that nothing uses.
9. Exactly one bash tool call per response; chain steps with && inside it; do not probe the harness.
10. You are already at the repository root ($PWD); do not cd elsewhere.
11. Scripted edits (python/sed) can silently match nothing: assert the old text is present before replacing and confirm with `git diff --stat` that the change landed.
12. Keep each command small: never paste a whole large file into one command (tool arguments over 64 KiB are dropped); edit in targeted chunks.
13. Do not run git commit/stash/checkout/reset: the repository metadata is read-only in the sandbox and the harness commits your work; use git only to inspect.
14. Reuse before writing: search for an existing function that already does the job and call it, or extract a shared core that both callers use. Never copy a block of logic into a second place.
15. Concurrency: never hold a lock, guard or permit across a wait that can be long (network, child process, another worker). Anything that joins a queue or takes a slot must give it back when the operation fails or is cancelled (release it in a guard/Drop/finally, not only on the success path).
16. Tests must be hermetic and deterministic: give every file, directory, daemon or registry they touch a temporary location passed to the code under test; do not mutate process-global state (environment variables) in tests that run in parallel; do not depend on the order of concurrent replies; poll for a condition instead of sleeping. Never write an assertion that cannot fail: a new test must fail without your change.
17. Do not call a failure "environmental" or "pre-existing" without proof: show the same failure on the unmodified code (e.g. `git show HEAD:<path>`) and explain the mechanism; otherwise it is yours to fix.
18. Other workers change the same repository in parallel: keep the diff to what the task needs, and put new tests in a module or file dedicated to your change rather than appending to the end of a large shared test file.
19. A command that reaches its timeout is not killed: it keeps running as job <n>, and the tool result tells you the number. Wait for it with `echo WAIT_JOB <n>` (blocks until it exits or 600s, then reports its exit code and the tail of its output) and stop it with `echo KILL_JOB <n>`. Never background a build yourself (`nohup cargo test > log &`) and never poll it with `sleep`; that burns a turn per poll and looks like a stall."#;

/// Chat roles accepted by the OpenAI chat-completions API.
///
/// An enum instead of a free-form `String` turns an invalid role from a
/// provider-side `400` into a compile error: `rename_all = "lowercase"` pins
/// the wire strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// A single outbound conversation message.
///
/// All fields are private, so the three constructors below are the *only* way to
/// build one. That closes the invalid states an all-`pub` struct allows (a `tool`
/// message with no `tool_call_id`, an `assistant` message with
/// `tool_calls: Some(vec![])`) while keeping the per-field `skip_serializing_if`
/// each role's wire shape needs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    role: Role,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ToolCall>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

/// The history as it must go on the wire for a thinking-mode conversation.
///
/// Thinking-mode providers (DeepSeek) reject a request whose assistant turns
/// lack `reasoning_content` ("must be passed back to the API"), yet the same
/// model sometimes answers a turn with no reasoning at all, which is stored as
/// `None`. Once any assistant turn carried reasoning the conversation is in
/// thinking mode, and every assistant turn is sent with the field, empty where
/// the model produced none. A conversation that never showed reasoning is sent
/// unchanged, so providers that reject unknown message fields never see it.
pub fn with_replayed_reasoning(messages: &[ChatMessage]) -> std::borrow::Cow<'_, [ChatMessage]> {
    let is_bare_assistant =
        |m: &ChatMessage| m.role == Role::Assistant && m.reasoning_content.is_none();
    let thinking = messages
        .iter()
        .any(|m| m.role == Role::Assistant && m.reasoning_content.is_some());
    if !thinking || !messages.iter().any(is_bare_assistant) {
        return std::borrow::Cow::Borrowed(messages);
    }
    let mut owned = messages.to_vec();
    for message in owned.iter_mut().filter(|m| is_bare_assistant(m)) {
        message.reasoning_content = Some(String::new());
    }
    std::borrow::Cow::Owned(owned)
}

/// Outbound `tool_calls` entry of an assistant message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub r#type: String,
    pub function: ToolCallFn,
}

/// The nested `function` object of a [`ToolCall`], one level deep as the wire
/// format requires (`{"id":..,"type":"function","function":{..}}`). Reused by the
/// non-streaming inbound path (`ToolCallOutput`) to avoid a duplicate type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallFn {
    pub name: String,
    pub arguments: String,
}

impl ChatMessage {
    /// Plain single-content message for `system`, `user` and `assistant` turns.
    /// `tool` content must go through [`Self::tool_result`], which records the
    /// required `tool_call_id`.
    pub fn text(role: Role, content: impl Into<String>) -> Self {
        debug_assert_ne!(
            role,
            Role::Tool,
            "tool messages require a tool_call_id; use ChatMessage::tool_result"
        );
        Self {
            role,
            content: Some(content.into()),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// An assistant turn requesting tool execution.
    ///
    /// `content` is `None` when the model emitted calls with no prose, and the
    /// field is then omitted on the wire. An empty `tool_calls` vec normalises to
    /// `None`: the API must never see a call-less assistant turn carrying `[]`.
    pub fn assistant_with_tool_calls(content: Option<String>, tool_calls: Vec<ToolCall>) -> Self {
        Self {
            role: Role::Assistant,
            content,
            reasoning_content: None,
            tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
            tool_call_id: None,
        }
    }

    /// The `tool` turn that answers a previous tool call, keyed by its id.
    pub fn tool_result(tool_call_id: String, content: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content: Some(content.into()),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: Some(tool_call_id),
        }
    }

    /// Attach reasoning content (chain of thought) to this message.
    ///
    /// Thinking-mode models (DeepSeek, etc.) require previous assistant reasoning
    /// to be replayed back to the API in multi-turn conversation history.
    pub fn with_reasoning_content(mut self, reasoning: Option<String>) -> Self {
        self.reasoning_content = reasoning.filter(|r| !r.trim().is_empty());
        self
    }

    /// Read-only view of the (already validated) role; the field is private so
    /// no caller can set an arbitrary one.
    pub fn role(&self) -> Role {
        self.role
    }

    /// Whether this message is a shape the provider accepts.
    ///
    /// The constructors above cannot build an invalid one, but a conversation
    /// reloaded from disk ([`crate::pool::revision`]) is deserialized straight
    /// into the fields, so the loader re-checks the two invariants that would
    /// otherwise make the whole request fail: a `tool` turn must name the call
    /// it answers, and an assistant turn must not advertise an empty call list.
    pub(crate) fn is_wire_valid(&self) -> bool {
        match self.role {
            Role::Tool => self
                .tool_call_id
                .as_deref()
                .is_some_and(|id| !id.is_empty()),
            Role::Assistant => match &self.tool_calls {
                Some(calls) => !calls.is_empty(),
                None => true,
            },
            Role::System | Role::User => true,
        }
    }

    /// The message content, if any.
    pub fn content(&self) -> Option<&str> {
        self.content.as_deref()
    }

    /// Replace text without changing the message's role or tool-call linkage.
    pub(crate) fn replace_content(&mut self, content: String) {
        self.content = Some(content);
    }

    /// Replace existing reasoning while preserving thinking-mode replay.
    pub(crate) fn replace_reasoning_content(&mut self, reasoning: String) {
        self.reasoning_content = Some(reasoning);
    }

    /// The reasoning content, if any.
    pub fn reasoning_content(&self) -> Option<&str> {
        self.reasoning_content.as_deref()
    }
}

/// The single tool this agent advertises to the provider.
pub(crate) fn bash_tool() -> Vec<serde_json::Value> {
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
pub(crate) struct ChatCompletionRequest<'a> {
    pub(crate) model: &'a str,
    pub(crate) messages: &'a [ChatMessage],
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) temperature: Option<f32>,
    #[serde(skip_serializing_if = "<[_]>::is_empty")]
    pub(crate) tools: &'a [serde_json::Value],
    pub(crate) stream: bool,
}

#[derive(Debug, Deserialize)]
pub(crate) struct StreamChunk {
    #[serde(default, deserialize_with = "deserialize_null_tolerant")]
    pub(crate) choices: Vec<StreamChoice>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct StreamChoice {
    #[serde(default)]
    pub(crate) delta: StreamDelta,
}

#[derive(Debug, Default, Deserialize)]
pub(crate) struct StreamDelta {
    /// `deserialize_null_tolerant` everywhere: providers send explicit `null`
    /// for absent fields, and a plain `Option`/`Vec` would fail the *whole*
    /// chunk on a null the field tolerates.
    #[serde(default, deserialize_with = "deserialize_null_tolerant")]
    pub(crate) content: Option<String>,
    /// Chain of thought, under the canonical key some providers use.
    #[serde(default, deserialize_with = "deserialize_null_tolerant")]
    pub(crate) reasoning_content: Option<String>,
    /// Chain of thought, under the alias others use. Kept as a *separate* field
    /// instead of a serde `alias`: proxies that send both keys made an aliased
    /// field fail with a duplicate-field error, dropping the whole chunk
    /// (reasoning, content and tool calls alike). Merge with
    /// [`Self::reasoning`], which prefers `reasoning_content` and never yields
    /// the same text twice.
    #[serde(default, deserialize_with = "deserialize_null_tolerant")]
    pub(crate) reasoning: Option<String>,
    #[serde(default, deserialize_with = "deserialize_null_tolerant")]
    pub(crate) tool_calls: Vec<StreamToolCall>,
}

impl StreamDelta {
    /// The reasoning text of this delta: `reasoning_content` when present,
    /// otherwise the `reasoning` alias. A proxy sending *both* keys repeats the
    /// same text, so only the preferred field is taken — appending both would
    /// double every reasoning token.
    pub(crate) fn reasoning(&self) -> Option<&str> {
        self.reasoning_content
            .as_deref()
            .or(self.reasoning.as_deref())
    }
}

#[derive(Debug, Deserialize)]
pub(crate) struct StreamToolCall {
    #[serde(default)]
    pub(crate) index: usize,
    #[serde(default, deserialize_with = "deserialize_null_tolerant")]
    pub(crate) id: Option<String>,
    #[serde(default, deserialize_with = "deserialize_null_tolerant")]
    pub(crate) function: Option<StreamFunction>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct StreamFunction {
    #[serde(default, deserialize_with = "deserialize_null_tolerant")]
    pub(crate) name: Option<String>,
    #[serde(default, deserialize_with = "deserialize_null_tolerant")]
    pub(crate) arguments: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ChatCompletionResponse {
    #[serde(default, deserialize_with = "deserialize_null_tolerant")]
    pub(crate) choices: Vec<ChatChoice>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ChatChoice {
    pub(crate) message: ChatMessageOutput,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ChatMessageOutput {
    #[serde(default, deserialize_with = "deserialize_null_tolerant")]
    pub(crate) content: Option<String>,
    #[serde(default, deserialize_with = "deserialize_null_tolerant")]
    pub(crate) reasoning_content: Option<String>,
    /// `reasoning` alias; see [`StreamDelta::reasoning`].
    #[serde(default, deserialize_with = "deserialize_null_tolerant")]
    pub(crate) reasoning: Option<String>,
    #[serde(default, deserialize_with = "deserialize_null_tolerant")]
    pub(crate) tool_calls: Vec<ToolCallOutput>,
}

impl ChatMessageOutput {
    /// The reasoning text of this message: `reasoning_content` when present,
    /// otherwise the `reasoning` alias.
    pub(crate) fn reasoning(&self) -> Option<&str> {
        self.reasoning_content
            .as_deref()
            .or(self.reasoning.as_deref())
    }
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ToolCallOutput {
    pub(crate) id: String,
    pub(crate) function: ToolCallFn,
}

#[derive(Debug, Deserialize)]
pub(crate) struct BashArgs {
    #[serde(alias = "cmd")]
    pub(crate) command: String,
}

/// One executed step of a subagent run.
///
/// `Deserialize` is intentionally absent: step logs are built in
/// `pool::run_worker` and serialized outward, never parsed back.
#[derive(Debug, Clone, Serialize)]
pub struct AgentStepLog {
    pub step: usize,
    pub command: String,
    pub output: String,
    pub exit_code: Option<i32>,
}

#[derive(Debug, Clone)]
pub struct LlmResponse {
    /// Assistant text; empty when the model only used `tool_calls`.
    pub content: String,
    /// Captured chain-of-thought reasoning, required for multi-turn history with reasoning models
    pub reasoning_content: Option<String>,
    /// Extracted bash command: `tool_calls` first, fenced-block fallback second.
    pub command: Option<String>,
    /// Raw `tool_calls`, for re-insertion into conversation history.
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Id of the call that produced the command, for the tool response message.
    pub tool_call_id: Option<String>,
    /// SSE frames whose bytes were not valid UTF-8 and had to be decoded
    /// lossily. Surfaced so fleet-wide corruption is observable rather than
    /// silently absorbed.
    pub invalid_utf8_lines: usize,
}

/// Cheap `call_xxxxxxxx` identifier derived from the low 32 bits of a UUID.
pub(crate) fn generate_call_id() -> String {
    format!("call_{:08x}", uuid::Uuid::new_v4().as_u128() as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Thinking mode: a turn the model answered without reasoning is replayed
    /// with an empty `reasoning_content`; a conversation that never reasoned
    /// is sent untouched.
    #[test]
    fn reasoning_is_replayed_on_every_assistant_turn_in_thinking_mode() {
        let thought =
            ChatMessage::text(Role::Assistant, "a").with_reasoning_content(Some("because".into()));
        let bare = ChatMessage::assistant_with_tool_calls(None, Vec::new());
        let user = ChatMessage::text(Role::User, "u");

        let history = vec![user.clone(), thought.clone(), user.clone(), bare.clone()];
        let wire = super::with_replayed_reasoning(&history);
        let json = serde_json::to_value(wire.as_ref()).expect("serialize");
        assert_eq!(json[1]["reasoning_content"], "because");
        assert_eq!(
            json[3]["reasoning_content"], "",
            "the bare turn gets an empty field"
        );
        assert!(
            json[0].get("reasoning_content").is_none(),
            "user turns are untouched"
        );

        let plain = vec![user, bare];
        let wire = super::with_replayed_reasoning(&plain);
        assert!(matches!(wire, std::borrow::Cow::Borrowed(_)));
        let json = serde_json::to_value(wire.as_ref()).expect("serialize");
        assert!(
            json[1].get("reasoning_content").is_none(),
            "no thinking mode, no field"
        );
    }

    #[test]
    fn test_chat_completion_request_tools_wire_shape() {
        let messages = [ChatMessage::text(Role::User, "hi")];
        let tools = bash_tool();

        let with_tools = ChatCompletionRequest {
            model: "test-model",
            messages: &messages,
            temperature: Some(0.2),
            tools: &tools,
            stream: true,
        };
        let body = serde_json::to_string(&with_tools).unwrap();
        let value: serde_json::Value = serde_json::from_str(&body).unwrap();
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

    #[test]
    fn test_role_wire_strings() {
        for (role, expected) in [
            (Role::System, "system"),
            (Role::User, "user"),
            (Role::Assistant, "assistant"),
            (Role::Tool, "tool"),
        ] {
            assert_eq!(
                serde_json::to_string(&role).unwrap(),
                format!("\"{expected}\"")
            );
            let back: Role = serde_json::from_str(&format!("\"{expected}\"")).unwrap();
            assert_eq!(back, role);
        }
    }

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
        let parsed: ChatCompletionResponse = serde_json::from_str(raw).unwrap();
        let choice = parsed.choices.first().unwrap();
        assert!(choice.message.content.is_none());
        let tc = &choice.message.tool_calls[0];
        assert_eq!(tc.id, "call_1");
        assert_eq!(tc.function.name, "bash");
        assert_eq!(tc.function.arguments, r#"{"command":"ls"}"#);
    }

    #[test]
    #[should_panic(expected = "tool messages require a tool_call_id")]
    fn test_text_rejects_tool_role() {
        let _ = ChatMessage::text(Role::Tool, "raw output");
    }

    #[test]
    fn test_assistant_with_empty_tool_calls_is_normalised() {
        let msg = ChatMessage::assistant_with_tool_calls(Some("thinking".into()), Vec::new());
        let value: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(&msg).unwrap()).unwrap();
        assert_eq!(msg.role(), Role::Assistant);
        assert!(value.get("tool_calls").is_none(), "got {value}");
        assert_eq!(value["content"], "thinking");
    }

    #[test]
    fn generate_call_id_is_unique_and_prefixed() {
        let a = generate_call_id();
        let b = generate_call_id();
        assert!(a.starts_with("call_"), "{a}");
        assert_eq!(a.len(), 13, "call_ + 8 hex chars: {a}");
        assert!(a[5..].chars().all(|c| c.is_ascii_hexdigit()), "{a}");
        assert_ne!(a, b, "ids must not collide");
    }
}
