//! LLM transport for the agent: HTTP chat completions and SSE streaming.
//!
//! Owns one worker step's conversation with the model: building the request,
//! POSTing it with retry/backoff, pumping the streamed response into an
//! [`SseAccumulator`], and assembling the [`LlmResponse`] the agent loop
//! consumes. The command-execution half lives in [`super::exec`].

use anyhow::{Context, Result};
use std::time::Duration;

use super::retry;
use super::stream::{FrameOutcome, SseAccumulator};
use super::types::{
    ChatCompletionRequest, ChatMessage, DEFAULT_STREAM_IDLE_TIMEOUT, LlmResponse, bash_tool,
};

/// Temperature when the caller configured none. Low but non-zero: the model
/// must break ties differently across steps, yet not wander.
const DEFAULT_TEMPERATURE: f32 = 0.2;

/// Outcome of one pass over an SSE body.
#[derive(Debug, PartialEq, Eq)]
enum StreamRun {
    /// Stream ended (or emitted `[DONE]`); accumulator is complete.
    Completed,
    /// Stream stalled or failed mid-flight; caller should retry.
    Retry,
}

pub struct AgentRunner {
    pub http_client: reqwest::Client,
    pub api_base: String,
    pub api_key: String,
    pub model: String,
    pub temperature: Option<f32>,
    /// `true` (from `network: "offline"`) confines each bash step to an
    /// isolated network namespace; `false` keeps normal connectivity.
    pub network_offline: bool,
    /// Idle deadline applied to each SSE body read.
    pub stream_idle_timeout: Duration,
    /// Fixed per-step command budget in seconds, replacing the light/heavy
    /// classification (tests use it to avoid mutating the process env).
    pub command_timeout_override: Option<u64>,
    pub max_retries: usize,
    pub initial_retry_delay: Duration,
}

impl AgentRunner {
    pub fn new(api_base: String, api_key: String, model: String, temperature: Option<f32>) -> Self {
        let http_client = reqwest::Client::builder()
            .user_agent(format!("mini-swe-mcp/{}", env!("CARGO_PKG_VERSION")))
            .connect_timeout(Duration::from_secs(30))
            .build()
            .expect("Failed to build HTTP client");

        Self {
            http_client,
            api_base,
            api_key,
            model,
            temperature,
            network_offline: false,
            stream_idle_timeout: DEFAULT_STREAM_IDLE_TIMEOUT,
            command_timeout_override: None,
            max_retries: retry::max_llm_retries(),
            initial_retry_delay: Duration::from_millis(retry::INITIAL_RETRY_DELAY_MS),
        }
    }

    /// Confine every bash step to an isolated network namespace (`unshare -n`).
    pub fn with_network_offline(mut self, offline: bool) -> Self {
        self.network_offline = offline;
        self
    }

    /// Give every bash step the same `secs` budget, whatever the command.
    pub fn with_command_timeout(mut self, secs: u64) -> Self {
        self.command_timeout_override = Some(secs);
        self
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

    /// Run one agent step: post `messages` and turn the streamed response into
    /// an [`LlmResponse`].
    ///
    /// Transient failures (network errors, 429/502/503/504, stalled SSE reads)
    /// retry with exponential backoff up to `max_retries`; anything else is a
    /// hard error.
    pub async fn run_step_llm(&self, messages: &[ChatMessage]) -> Result<LlmResponse> {
        let tools = bash_tool();
        let payload = self.chat_request(messages, &tools);

        let mut attempts = 0;
        let accumulator = loop {
            attempts += 1;

            let Some(mut resp) = self.send_with_retry(&payload, attempts).await? else {
                continue;
            };

            if !resp.status().is_success() {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                anyhow::bail!("LLM API returned HTTP {}: {}", status, body);
            }

            let mut acc = SseAccumulator::default();
            let mut buffer: Vec<u8> = Vec::new();

            match self
                .pump_sse(&mut resp, &mut acc, &mut buffer, attempts)
                .await?
            {
                StreamRun::Completed => {
                    acc.handle_non_stream_fallback(&buffer);
                    break acc;
                }
                StreamRun::Retry => continue,
            }
        };

        Ok(accumulator.finish())
    }

    /// Build the request payload, dropping `temperature` for models known to
    /// reject it (e.g. kimi-k3) and defaulting low otherwise.
    fn chat_request<'a>(
        &'a self,
        messages: &'a [ChatMessage],
        tools: &'a [serde_json::Value],
    ) -> ChatCompletionRequest<'a> {
        let temperature = if self.model.contains("kimi-k3") {
            None
        } else {
            self.temperature.or(Some(DEFAULT_TEMPERATURE))
        };

        ChatCompletionRequest {
            model: &self.model,
            messages,
            temperature,
            tools,
            stream: true,
        }
    }

    /// POST `payload`, retrying transient conditions with backoff.
    ///
    /// Returns `Ok(None)` when the caller should start another attempt (the
    /// backoff sleep is already awaited), and `Ok(Some(response))` for a
    /// response to inspect — including a non-2xx one, so the caller can report
    /// the status and body verbatim.
    async fn send_with_retry(
        &self,
        payload: &ChatCompletionRequest<'_>,
        attempt: usize,
    ) -> Result<Option<reqwest::Response>> {
        let url = format!("{}/chat/completions", self.api_base.trim_end_matches('/'));

        let result = self
            .http_client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(payload)
            .send()
            .await;

        match result {
            Ok(resp) => {
                let status = resp.status();
                if retry::is_transient_status(status) && attempt < self.max_retries {
                    let retry_after = retry::parse_retry_after(resp.headers());
                    let delay = retry::retry_delay(self.initial_retry_delay, attempt, retry_after);

                    tracing::warn!(
                        status = %status,
                        attempt,
                        max_retries = self.max_retries,
                        delay_ms = delay.as_millis(),
                        "LLM API rate-limited or unavailable; retrying with backoff"
                    );
                    tokio::time::sleep(delay).await;
                    return Ok(None);
                }
                Ok(Some(resp))
            }
            Err(e) if attempt < self.max_retries => {
                let delay = retry::retry_delay(self.initial_retry_delay, attempt, None);
                tracing::warn!(
                    attempt,
                    max_retries = self.max_retries,
                    delay_ms = delay.as_millis(),
                    error = %e,
                    "LLM API network error; retrying with backoff"
                );
                tokio::time::sleep(delay).await;
                Ok(None)
            }
            Err(e) => Err(e).context("Failed to send request to LLM API after retries"),
        }
    }

    /// Feed SSE chunks from `resp` into `acc` until the stream ends.
    ///
    /// Each read is bounded by `stream_idle_timeout` so a half-open connection
    /// cannot wedge a worker forever. A stall or read error retries while
    /// attempts remain, and is a hard error once they are exhausted.
    async fn pump_sse(
        &self,
        resp: &mut reqwest::Response,
        acc: &mut SseAccumulator,
        buffer: &mut Vec<u8>,
        attempt: usize,
    ) -> Result<StreamRun> {
        loop {
            let next_chunk = match tokio::time::timeout(self.stream_idle_timeout, resp.chunk())
                .await
            {
                Ok(result) => result,
                Err(_) => {
                    if attempt < self.max_retries {
                        self.backoff_for_stream_failure(
                                attempt,
                                "LLM SSE stream stalled (no chunk within the idle timeout); retrying request with backoff",
                            )
                            .await;
                        return Ok(StreamRun::Retry);
                    }
                    anyhow::bail!(
                        "LLM SSE stream stalled for {}s and did not resume after {} attempts",
                        self.stream_idle_timeout.as_secs(),
                        attempt
                    );
                }
            };

            match next_chunk {
                Ok(Some(bytes)) => {
                    if acc.push(&bytes, buffer) == Some(FrameOutcome::Done) {
                        return Ok(StreamRun::Completed);
                    }
                }
                Ok(None) => return Ok(StreamRun::Completed),
                Err(e) => {
                    if attempt < self.max_retries {
                        self.backoff_for_stream_failure(
                            attempt,
                            "LLM SSE stream chunk read failed; retrying request with backoff",
                        )
                        .await;
                        return Ok(StreamRun::Retry);
                    }
                    return Err(e).context("Failed reading stream chunk after retries");
                }
            }
        }
    }

    /// Log a stream failure and sleep out the backoff before the next attempt.
    async fn backoff_for_stream_failure(&self, attempt: usize, message: &'static str) {
        let delay = retry::retry_delay(self.initial_retry_delay, attempt, None);
        tracing::warn!(
            attempt,
            max_retries = self.max_retries,
            delay_ms = delay.as_millis(),
            idle_timeout_secs = self.stream_idle_timeout.as_secs(),
            message
        );
        tokio::time::sleep(delay).await;
    }
}
