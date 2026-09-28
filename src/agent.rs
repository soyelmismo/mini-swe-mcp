use anyhow::{Context, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::Duration;
use tokio::process::Command;

pub const SYSTEM_PROMPT: &str = r#"You are an autonomous software engineering subagent running in a Linux bash environment.
You are given a task to complete within a git repository.

WORKFLOW:
1. Explore: Use tools like `git status`, `find`, `grep -rn`, or `ls` to locate relevant files.
2. Edit & Test: Make minimal, clean edits (using sed, python, cat << 'EOF', etc.) and run existing test suites to verify.
3. Every response MUST execute EXACTLY ONE command using the `bash` tool. If the bash tool is unavailable, use a ```bash ... ``` code block instead.
4. When finished and verified, complete your work by executing:
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
        };

        let mut attempts = 0;
        let resp = loop {
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
                Ok(r) => break r,
                Err(_e) if attempts < 3 => {
                    tokio::time::sleep(Duration::from_secs(2 * attempts)).await;
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

        let result: ChatCompletionResponse = resp
            .json()
            .await
            .context("Failed to parse LLM JSON response")?;

        let choice = result.choices.first().context("Empty choices in LLM response")?;
        let content = choice.message.content.clone().unwrap_or_default();

        // Priority 1: extract command from tool_calls (OpenAI function calling)
        let bash_tc = choice
            .message
            .tool_calls
            .iter()
            .find(|tc| tc.function.name == "bash" || tc.function.name.is_empty());

        let command = bash_tc
            .and_then(|tc| {
                serde_json::from_str::<BashArgs>(&tc.function.arguments)
                    .map(|args| args.command)
                    .ok()
                    .or_else(|| {
                        serde_json::from_str::<serde_json::Value>(&tc.function.arguments)
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
        let (tool_calls, tool_call_id) = if choice.message.tool_calls.is_empty() {
            (None, None)
        } else {
            let generated_id = format!("call_{}", &uuid::Uuid::new_v4().to_string()[..8]);
            let tc_id = bash_tc.map(|tc| {
                if tc.id.is_empty() {
                    generated_id.clone()
                } else {
                    tc.id.clone()
                }
            });
            let tcs: Vec<ToolCall> = choice
                .message
                .tool_calls
                .iter()
                .map(|tc| {
                    let id = if tc.id.is_empty() {
                        generated_id.clone()
                    } else {
                        tc.id.clone()
                    };
                    let name = if tc.function.name.is_empty() {
                        "bash".to_string()
                    } else {
                        tc.function.name.clone()
                    };
                    ToolCall {
                        id,
                        r#type: "function".to_string(),
                        function: ToolCallFn {
                            name,
                            arguments: tc.function.arguments.clone(),
                        },
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

        // Truncate output to 4096 chars if too long to prevent context explosion
        if combined.len() > 4096 {
            let truncated = format!(
                "\n... [Truncated {} bytes] ...\n{}",
                combined.len() - 4096,
                &combined[combined.len() - 2048..]
            );
            combined = format!("{}{}", &combined[..2048], truncated);
        }

        Ok((combined, output.status.code()))
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
}
