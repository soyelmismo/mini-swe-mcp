use anyhow::{Context, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
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

REPORTS & ARTIFACTS:
- If you generate reports, audits, benchmarks, or handoffs, save them under `audits/`, `reports/`, or `.agents/` (e.g. `audits/audit_01_feature.md` or `.agents/handoff.md`).
- Files created in these directories are automatically preserved and synchronized back to the main repository when your task completes.

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

        // Capture-free pattern: the body is sliced out of the full match by hand
        // (see `extract_command`). `captures()` pays for a `Captures`
        // allocation plus per-group offset tracking on every match, which
        // dominates the cost of this regex and scales badly with block size.
        //
        // `(?s)` lets `.` match newlines (cheaper than the `[\s\S]` class), and
        // the info-string gap is restricted to ASCII whitespace
        // (`[ \t\r\n]*`, a strict subset of `\s*`).
        let command_regex = Regex::new(r"(?s)```(?:bash|sh)[ \t\r\n]*\n.*?\n```").unwrap();

        Self {
            http_client,
            api_base,
            api_key,
            model,
            temperature,
            command_regex,
        }
    }

    /// Recover a bash command from the first ```bash / ```sh fenced block.
    ///
    /// Uses a single `find()` on a capture-free pattern and slices the body out
    /// of the match by hand, which is materially cheaper than `captures()`.
    /// A literally empty body (```` ```bash\n``` ````) yields `None`, because
    /// the pattern requires a newline before the closing fence; a
    /// whitespace-only body yields `Some("")`, as before.
    pub fn extract_command(&self, text: &str) -> Option<String> {
        let full = self.command_regex.find(text)?.as_str();
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
        let (content, accumulated_tools) = loop {
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

            let mut content = String::new();
            let mut accumulated_tools: Vec<(String, String, String)> = Vec::new();
            let mut buffer: Vec<u8> = Vec::new();
            let mut stream_err = false;

            'stream: loop {
                match resp.chunk().await {
                    Ok(Some(bytes)) => {
                        buffer.extend_from_slice(&bytes);
                        let mut start = 0;
                        while let Some(rel_pos) = buffer[start..].iter().position(|&b| b == b'\n') {
                            let pos = start + rel_pos;
                            let raw_line = &buffer[start..pos];
                            start = pos + 1;

                            let trimmed = match std::str::from_utf8(raw_line) {
                                Ok(s) => s.trim(),
                                Err(_) => continue,
                            };

                            if trimmed.is_empty() || trimmed.starts_with(':') {
                                continue;
                            }
                            let data = if let Some(d) = trimmed.strip_prefix("data:") {
                                d.trim()
                            } else {
                                continue;
                            };
                            if data == "[DONE]" {
                                buffer.drain(..start);
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
                        if start > 0 {
                            buffer.drain(..start);
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

            break (content, accumulated_tools);
        };

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
            let dir_name = dir.file_name().and_then(|n| n.to_str()).unwrap_or("default");
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
                "-n", "10",
                "bwrap",
                "--die-with-parent",
                "--new-session",
                "--unshare-pid",
                "--unshare-ipc",
                "--ro-bind", "/usr", "/usr",
                "--symlink", "usr/bin", "/bin",
                "--symlink", "usr/bin", "/sbin",
                "--symlink", "usr/lib", "/lib",
                "--symlink", "usr/lib", "/lib64",
                "--ro-bind-try", "/etc", "/etc",
                "--proc", "/proc",
                "--dev", "/dev",
                "--tmpfs", "/tmp",
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
                    format!("Command timed out after {}s and was terminated.", timeout_secs),
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
            return Err("Scanning root '/' or system directories is forbidden. Confine searches to the current repository ($PWD).");
        }
    }

    // 2. Block escaping to parent or root directory via cd
    const FORBIDDEN_CDS: &[&str] = &[
        "cd / ",
        "cd /;",
        "cd /&&",
        "cd /||",
        "cd /home",
        "cd ~",
        "cd $HOME",
        "cd /root",
    ];

    for token in FORBIDDEN_CDS {
        if trimmed.contains(token) || trimmed.ends_with("cd /") {
            return Err("Navigating outside the repository with 'cd' is forbidden. All files are in $PWD.");
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
        assert!(!is_heavy_command("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"));
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

    #[tokio::test]
    async fn test_execute_bash_sandbox_runs_and_blocks_write() {
        let tmp = std::env::temp_dir().join(format!("swe-test-bwrap-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&tmp);
        let r = runner();

        // 1. Basic command within worktree succeeds
        let (out, code) = r.execute_bash(&tmp, "echo 'hello from sandbox'").await.unwrap();
        assert_eq!(code, Some(0));
        assert!(out.contains("hello from sandbox"));

        // 2. Writing to read-only host root /usr fails when bwrap is active
        if super::has_bwrap() {
            let (out, code) = r.execute_bash(&tmp, "touch /usr/forbidden_write_test 2>&1").await.unwrap();
            assert_ne!(code, Some(0));
            assert!(out.contains("Read-only") || out.contains("sólo lectura") || out.contains("Permission denied"));
        }

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
