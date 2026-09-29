use anyhow::{Context, Result};
use regex::Regex;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;
use std::time::Duration;
use tokio::process::Command;

use super::sandbox::{
    find_git_dirs, has_bwrap, is_heavy_command, truncate_output, validate_bash_command,
};
use super::stream::{FrameOutcome, SseAccumulator};
use super::types::{
    bash_tool, BashArgs, ChatCompletionRequest, ChatMessage, LlmResponse,
    DEFAULT_STREAM_IDLE_TIMEOUT,
};

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

pub struct AgentRunner {
    pub http_client: reqwest::Client,
    pub api_base: String,
    pub api_key: String,
    pub model: String,
    pub temperature: Option<f32>,
    /// Idle deadline applied to each SSE body read.
    pub stream_idle_timeout: Duration,
    pub max_retries: usize,
    pub initial_retry_delay: Duration,
}

pub const DEFAULT_MAX_RETRIES: usize = 6;
pub const INITIAL_RETRY_DELAY_MS: u64 = 500;
pub const MAX_RETRY_DELAY: Duration = Duration::from_secs(30);

pub fn max_llm_retries() -> usize {
    std::env::var("LLM_MAX_RETRIES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_MAX_RETRIES)
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
            max_retries: max_llm_retries(),
            initial_retry_delay: Duration::from_millis(INITIAL_RETRY_DELAY_MS),
        }
    }

    /// Override the per-chunk idle deadline (used by tests to keep them fast).
    pub fn with_stream_idle_timeout(mut self, timeout: Duration) -> Self {
        self.stream_idle_timeout = timeout;
        self
    }

    /// Override the maximum retry attempts for transient errors.
    pub fn with_max_retries(mut self, retries: usize) -> Self {
        self.max_retries = retries;
        self
    }

    /// Override the initial retry delay.
    pub fn with_initial_retry_delay(mut self, delay: Duration) -> Self {
        self.initial_retry_delay = delay;
        self
    }

    pub fn calculate_retry_delay(&self, attempt: usize, retry_after: Option<Duration>) -> Duration {
        if let Some(ra) = retry_after {
            return ra.min(MAX_RETRY_DELAY);
        }
        let shift = (attempt.saturating_sub(1)).min(6);
        let base_ms = self.initial_retry_delay.as_millis() as u64;
        let ms = base_ms.saturating_mul(1 << shift);
        Duration::from_millis(ms).min(MAX_RETRY_DELAY)
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

        let max_retries = self.max_retries;
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

                    if is_transient && attempts < max_retries {
                        let retry_after = r
                            .headers()
                            .get(reqwest::header::RETRY_AFTER)
                            .and_then(|v| v.to_str().ok())
                            .and_then(|v| v.parse::<u64>().ok())
                            .map(Duration::from_secs);
                        let delay = self.calculate_retry_delay(attempts, retry_after);

                        tracing::warn!(
                            status = %status,
                            attempt = attempts,
                            max_retries,
                            delay_ms = delay.as_millis(),
                            "LLM API rate-limited or unavailable; retrying with backoff"
                        );
                        tokio::time::sleep(delay).await;
                        continue;
                    }
                    r
                }
                Err(e) if attempts < max_retries => {
                    let delay = self.calculate_retry_delay(attempts, None);
                    tracing::warn!(
                        attempt = attempts,
                        max_retries,
                        delay_ms = delay.as_millis(),
                        error = %e,
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
                let next_chunk =
                    match tokio::time::timeout(self.stream_idle_timeout, resp.chunk()).await {
                        Ok(result) => result,
                        Err(_) => {
                            if attempts < max_retries {
                                let delay = self.calculate_retry_delay(attempts, None);
                                tracing::warn!(
                                    attempt = attempts,
                                    max_retries,
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
                        if attempts < max_retries {
                            let delay = self.calculate_retry_delay(attempts, None);
                            tracing::warn!(
                                attempt = attempts,
                                max_retries,
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

            acc.handle_non_stream_fallback(&buffer);

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
            .or_else(|| self.extract_command(&content));

        let (finalized, known_ids) = SseAccumulator::finalize_from(&tools);
        let tool_calls = (!finalized.is_empty()).then_some(finalized);
        let tool_call_id = tool_calls.as_ref().and_then(|tcs| {
            bash_tc.and_then(|tc| {
                let candidate = if tc.id.trim().is_empty() {
                    tcs.iter().find(|t| t.function.arguments == tc.arguments).map(|t| t.id.clone())
                } else {
                    Some(tc.id.trim().to_string())
                };
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
                        .stdin(std::process::Stdio::null())
                        .stdout(std::process::Stdio::null())
                        .stderr(std::process::Stdio::null())
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

#[cfg(test)]
mod tests {
    use super::*;

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
    fn bash_block_regex_is_a_process_wide_static() {
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

    #[tokio::test]
    async fn test_execute_bash_sandbox_runs_and_blocks_write() {
        let unique_id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let tmp = crate::worktree::swe_base_dir().join(format!("bwrap-test-{}-{unique_id}", std::process::id()));
        let _ = std::fs::create_dir_all(&tmp);
        let r = runner();

        // 1. Basic command within worktree succeeds
        let (out, code) = r
            .execute_bash(&tmp, "echo 'hello from sandbox'")
            .await
            .unwrap();
        assert_eq!(code, Some(0), "command failed with output: {out:?}");
        assert!(out.contains("hello from sandbox"));

        // 2. Writing to read-only host root /usr fails when bwrap is active
        if has_bwrap() {
            let (out, code) = r
                .execute_bash(&tmp, "touch /usr/forbidden_write_test 2>&1")
                .await
                .unwrap();
            assert_ne!(code, Some(0));
            assert!(
                out.contains("Read-only")
                    || out.contains("sólo lectura")
                    || out.contains("solo lectura")
                    || out.contains("Permission denied")
                    || out.contains("Permiso denegado"),
                "unexpected touch output: {out:?}, code: {code:?}"
            );
        }

        let target_dir = crate::worktree::swe_base_dir()
            .join(format!("swe-target-bwrap-test-{}-{unique_id}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let _ = std::fs::remove_dir_all(&target_dir);
    }
}
