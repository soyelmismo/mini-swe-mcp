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
3. Every response MUST contain your short reasoning followed by EXACTLY ONE bash command in a ```bash ... ``` block.
4. When finished and verified, complete your work by executing:
```bash
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
```

COMMUNICATION WITH ORCHESTRATOR:
- Need more turns: If you are close to finishing verification/refactoring and need more steps, execute:
  echo "REQUEST_TURNS: <number>"
- Ask orchestrator / Critical ambiguity: If you face critical blockers, breaking decisions, or require orchestrator confirmation, execute:
  echo "ASK_ORCHESTRATOR: <your specific question>"
  This will immediately pause execution until the orchestrator replies with guidance."#;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize)]
struct ChatCompletionRequest<'a> {
    model: &'a str,
    messages: &'a [ChatMessage],
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentStepLog {
    pub step: usize,
    pub command: String,
    pub output: String,
    pub exit_code: Option<i32>,
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

    pub async fn run_step_llm(&self, messages: &[ChatMessage]) -> Result<String> {
        let url = format!("{}/chat/completions", self.api_base.trim_end_matches('/'));

        // Handle models that reject temperature parameter (e.g. kimi-k3)
        let temperature = if self.model.contains("kimi-k3") {
            None
        } else {
            self.temperature.or(Some(0.2))
        };

        let payload = ChatCompletionRequest {
            model: &self.model,
            messages,
            temperature,
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

        let content = result
            .choices
            .first()
            .and_then(|c| c.message.content.clone())
            .unwrap_or_default();

        Ok(content)
    }

    pub async fn execute_bash(&self, dir: &Path, command: &str) -> Result<(String, Option<i32>)> {
        let mut cmd = Command::new("bash");
        cmd.current_dir(dir).args(["-c", command]);

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
