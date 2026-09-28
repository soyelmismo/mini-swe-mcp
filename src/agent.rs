use anyhow::{Context, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::Duration;
use tokio::process::Command;

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

COMMUNICATION WITH ORCHESTRATOR:
- Need more turns: If you are close to finishing verification/refactoring and need more steps, execute:
  echo "REQUEST_TURNS: <number>"
- Ask orchestrator / Critical ambiguity: If you face critical blockers, breaking decisions, or require orchestrator confirmation, execute:
  echo "ASK_ORCHESTRATOR: <your specific question>"
  This will immediately pause execution until the orchestrator replies with guidance."#;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// Outbound tool_call representation for assistant messages in the conversation history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub r#type: String,
    pub function: ToolCallFn,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallFn {
    pub name: String,
    pub arguments: String,
}

impl ChatMessage {
    pub fn text(role: &str, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    pub fn assistant_with_tool_calls(content: Option<String>, tool_calls: Vec<ToolCall>) -> Self {
        Self {
            role: "assistant".into(),
            content,
            tool_calls: Some(tool_calls),
            tool_call_id: None,
        }
    }

    pub fn tool_result(tool_call_id: String, content: impl Into<String>) -> Self {
        Self {
            role: "tool".into(),
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: Some(tool_call_id),
        }
    }
}

#[derive(Debug, Serialize)]
struct ToolFunction {
    name: &'static str,
    description: &'static str,
    parameters: serde_json::Value,
}

#[derive(Debug, Serialize)]
struct ToolDefinition {
    r#type: &'static str,
    function: ToolFunction,
}

#[derive(Debug, Serialize)]
struct ChatCompletionRequest<'a> {
    model: &'a str,
    messages: &'a [ChatMessage],
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    tools: &'a [ToolDefinition],
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
    function: ToolCallFunction,
}

#[derive(Debug, Clone, Deserialize)]
struct ToolCallFunction {
    name: String,
    arguments: String,
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

pub struct LlmResponse {
    /// Full text content from the assistant (may be empty if model only used tool_calls)
    pub content: String,
    /// Extracted bash command — from tool_calls first, regex fallback second
    pub command: Option<String>,
    /// Raw tool_calls from the response, for re-insertion into conversation history
    pub tool_calls: Option<Vec<ToolCall>>,
    /// The tool_call id that produced the command (for tool response messages)
    pub tool_call_id: Option<String>,
}

pub struct AgentRunner {
    pub http_client: reqwest::Client,
    pub api_base: String,
    pub api_key: String,
    pub model: String,
    pub temperature: Option<f32>,
    pub command_regex: Regex,
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
            .timeout(Duration::from_secs(120))
            .build()
            .expect("Failed to build HTTP client");

        let command_regex = Regex::new(r"```(?:bash|sh)\s*\n([\s\S]*?)\n```").unwrap();

        Self {
            http_client,
            api_base,
            api_key,
            model,
            temperature,
            command_regex,
        }
    }

    pub fn extract_command(&self, text: &str) -> Option<String> {
        self.command_regex
            .captures(text)
            .and_then(|cap| cap.get(1))
            .map(|m| m.as_str().trim().to_string())
    }

    pub async fn run_step_llm(&self, messages: &[ChatMessage]) -> Result<LlmResponse> {
        let url = format!("{}/chat/completions", self.api_base.trim_end_matches('/'));

        // Handle models that reject temperature parameter (e.g. kimi-k3)
        let temperature = if self.model.contains("kimi-k3") {
            None
        } else {
            self.temperature.or(Some(0.2))
        };

        let bash_tool = ToolDefinition {
            r#type: "function",
            function: ToolFunction {
                name: "bash",
                description: "Execute a bash command in the repository working directory",
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The bash command to execute"
                        }
                    },
                    "required": ["command"]
                }),
            },
        };

        let payload = ChatCompletionRequest {
            model: &self.model,
            messages,
            temperature,
            tools: &[bash_tool],
            stream: true,
        };

        let mut attempts = 0;
        let mut resp = loop {
            attempts += 1;
            match self
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
                    break r;
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
            }
        };

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("LLM API returned HTTP {}: {}", status, body);
        }

        let mut content = String::new();
        let mut accumulated_tools: Vec<(String, String, String)> = Vec::new();
        let mut buffer: Vec<u8> = Vec::new();

        'stream: while let Some(bytes) = resp.chunk().await.context("Failed reading stream chunk")? {
            buffer.extend_from_slice(&bytes);
            while let Some(pos) = buffer.iter().position(|&b| b == b'\n') {
                let line_bytes: Vec<u8> = buffer.drain(..=pos).collect();
                let line = String::from_utf8_lossy(&line_bytes);
                let trimmed = line.trim();
                if trimmed.is_empty() || trimmed.starts_with(':') {
                    continue;
                }
                let data = if let Some(d) = trimmed.strip_prefix("data:") {
                    d.trim()
                } else {
                    continue;
                };
                if data == "[DONE]" {
                    break 'stream;
                }
                if let Ok(chunk) = serde_json::from_str::<StreamChunk>(data)
                    && let Some(choice) = chunk.choices.first()
                {
                    if let Some(c) = &choice.delta.content {
                        content.push_str(c);
                    }
                    for tc in &choice.delta.tool_calls {
                        if tc.index >= accumulated_tools.len() {
                            accumulated_tools.resize(
                                tc.index + 1,
                                (String::new(), String::new(), String::new()),
                            );
                        }
                        if let Some(id) = &tc.id {
                            accumulated_tools[tc.index].0 = id.clone();
                        }
                        if let Some(fn_info) = &tc.function {
                            if let Some(name) = &fn_info.name {
                                accumulated_tools[tc.index].1.push_str(name);
                            }
                            if let Some(args) = &fn_info.arguments {
                                accumulated_tools[tc.index].2.push_str(args);
                            }
                        }
                    }
                }
            }
        }

        // Fallback for non-streaming response if proxy ignored stream=true and sent raw JSON in remaining buffer
        if content.is_empty() && accumulated_tools.is_empty() && !buffer.is_empty() {
            let raw = String::from_utf8_lossy(&buffer);
            if let Ok(result) = serde_json::from_str::<ChatCompletionResponse>(&raw)
                && let Some(choice) = result.choices.first()
            {
                content = choice.message.content.clone().unwrap_or_default();
                for tc in &choice.message.tool_calls {
                    accumulated_tools.push((
                        tc.id.clone(),
                        tc.function.name.clone(),
                        tc.function.arguments.clone(),
                    ));
                }
            }
        }

        // Priority 1: extract command from tool_calls (OpenAI function calling)
        let bash_tc = accumulated_tools
            .iter()
            .find(|(_, name, _)| name == "bash" || name.is_empty());

        let command = bash_tc
            .and_then(|(_, _, args)| {
                serde_json::from_str::<BashArgs>(args)
                    .map(|a| a.command)
                    .ok()
                    .or_else(|| {
                        serde_json::from_str::<serde_json::Value>(args)
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

        // Convert API tool_calls to history-compatible format
        let (tool_calls, tool_call_id) = if accumulated_tools.is_empty() {
            (None, None)
        } else {
            let generated_id = format!("call_{}", &uuid::Uuid::new_v4().to_string()[..8]);
            let tc_id = bash_tc.map(|(id, _, _)| {
                if id.is_empty() {
                    generated_id.clone()
                } else {
                    id.clone()
                }
            });
            let tcs: Vec<ToolCall> = accumulated_tools
                .into_iter()
                .map(|(id, name, arguments)| {
                    let id = if id.is_empty() {
                        generated_id.clone()
                    } else {
                        id
                    };
                    let name = if name.is_empty() {
                        "bash".to_string()
                    } else {
                        name
                    };
                    ToolCall {
                        id,
                        r#type: "function".to_string(),
                        function: ToolCallFn { name, arguments },
                    }
                })
                .collect();
            (Some(tcs), tc_id)
        };

        Ok(LlmResponse { content, command, tool_calls, tool_call_id })
    }

    pub async fn execute_bash(&self, dir: &Path, command: &str) -> Result<(String, Option<i32>)> {
        let default_parallelism = std::thread::available_parallelism()
            .map(|n| (n.get() / 2).max(1))
            .unwrap_or(2);
        let parallelism = std::env::var("BUILD_PARALLELISM")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default_parallelism)
            .to_string();

        let mut cmd = Command::new("nice");
        cmd.kill_on_drop(true);
        cmd.current_dir(dir)
            .args(["-n", "10", "bash", "-c", command])
            // Universal build and test parallelism caps
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

        if std::env::var_os("CARGO_TARGET_DIR").is_none() {
            cmd.env("CARGO_TARGET_DIR", "/tmp/swe-cargo-target");
        }

        let timeout_secs = std::env::var("COMMAND_TIMEOUT_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(600); // 10 minutes default for builds/tests
        let timeout_duration = Duration::from_secs(timeout_secs);
        let output = tokio::time::timeout(timeout_duration, cmd.output())
            .await
            .context(format!("Command timed out after {}s", timeout_secs))?
            .context("Failed to spawn bash process")?;

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

pub fn truncate_output(combined: &str) -> String {
    if combined.len() > 16384 {
        let head_end = combined.floor_char_boundary(12288);
        let tail_start = combined.ceil_char_boundary(combined.len().saturating_sub(4096));
        let truncated = format!(
            "\n... [Truncated {} bytes] ...\n{}",
            combined.len() - (head_end + (combined.len() - tail_start)),
            &combined[tail_start..]
        );
        format!("{}{}", &combined[..head_end], truncated)
    } else {
        combined.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::AgentRunner;

    fn runner() -> AgentRunner {
        AgentRunner::new(
            "http://localhost".to_string(),
            "test-key".to_string(),
            "test-model".to_string(),
            None,
        )
    }

    #[test]
    fn extract_command_single_line_bash() {
        let reply = "Run this:\n```bash\necho hello\n```";

        assert_eq!(runner().extract_command(reply), Some("echo hello".to_string()));
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
        let mut s = "a".repeat(12287);
        s.push('€'); // bytes 12287..12290
        s.push_str(&"b".repeat(9000));
        let truncated = super::truncate_output(&s);
        assert!(truncated.contains("... [Truncated"));
    }
}
