//! LLM transport for the agent: HTTP chat completions and SSE streaming.
//!
//! Owns one worker step's conversation with the model: building the request,
//! POSTing it with retry/backoff, pumping the streamed response into an
//! [`SseAccumulator`], and assembling the [`LlmResponse`] the agent loop
//! consumes. The command-execution half lives in [`super::exec`].

use anyhow::{Context, Result};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use super::jobs::{JobHandle, JobWait};
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

/// How many process-wide HTTP clients have been built.
///
/// `reqwest::Client::clone` shares the inner connection pool, so every runner
/// constructed from [`shared_http_client`] reuses the same connections and TLS
/// sessions; the counter pins that down to one build per process.
static SHARED_CLIENT_BUILDS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

fn shared_http_client() -> &'static reqwest::Client {
    static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
    CLIENT.get_or_init(|| {
        SHARED_CLIENT_BUILDS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        reqwest::Client::builder()
            .user_agent(format!("mini-swe-mcp/{}", env!("CARGO_PKG_VERSION")))
            .connect_timeout(Duration::from_secs(30))
            .build()
            .expect("Failed to build HTTP client")
    })
}

fn llm_gate(limit: usize) -> Option<Arc<Semaphore>> {
    (limit > 0).then(|| Arc::new(Semaphore::new(limit)))
}

/// Initialized once for the process, just like the shared HTTP client.
fn shared_llm_gate() -> &'static Option<Arc<Semaphore>> {
    static GATE: OnceLock<Option<Arc<Semaphore>>> = OnceLock::new();
    GATE.get_or_init(|| llm_gate(crate::config::llm_concurrency()))
}

async fn acquire_llm(gate: &Option<Arc<Semaphore>>) -> Result<Option<OwnedSemaphorePermit>> {
    match gate {
        Some(gate) => Ok(Some(
            gate.clone()
                .acquire_owned()
                .await
                .context("LLM gate closed")?,
        )),
        None => Ok(None),
    }
}

impl AgentRunner {
    /// How many process-wide HTTP clients have been built (test support).
    ///
    /// Building more runners must not move this counter: they all clone the
    /// one shared client, so connections and TLS sessions are reused.
    #[doc(hidden)]
    pub fn __test_shared_client_builds() -> usize {
        SHARED_CLIENT_BUILDS.load(std::sync::atomic::Ordering::Relaxed)
    }
}

/// Lock the runner's last-job slot, recovering from a poisoned lock: the id
/// behind it is still the truth about the last command.
fn lock_last_job(slot: &Mutex<Option<u64>>) -> std::sync::MutexGuard<'_, Option<u64>> {
    slot.lock().unwrap_or_else(|poison| poison.into_inner())
}

#[derive(Clone)]
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
    /// Jobs granted by the heavy-command admission controller for the next
    /// `execute_bash` call. `None` keeps the default parallelism; `BUILD_PARALLELISM`
    /// still overrides either way.
    pub build_jobs: Option<usize>,
    /// Shared target selected by the pool; absent until the first heavy step.
    pub(crate) build_target_dir: Option<std::path::PathBuf>,
    /// Extra variables layered on top of the sanitized environment for one
    /// command. The differential verify gate uses it to replay the same
    /// command in the dispatcher's ambient environment; ordinary steps leave
    /// it empty.
    pub extra_env: Vec<(String, String)>,
    /// The worker's background jobs. `None` for a runner no worker owns, in
    /// which case a command that outlives its budget is simply killed.
    pub(crate) jobs: Option<JobHandle>,
    /// Job number the last `execute_bash` call backgrounded, taken by the
    /// caller so it is reported exactly once.
    pub(crate) last_job: Arc<Mutex<Option<u64>>>,
    /// Set for a run the harness owns rather than the model: a command that
    /// outlives its budget is stopped instead of becoming a background job.
    pub(crate) jobs_disabled: bool,
}

impl AgentRunner {
    pub fn new(api_base: String, api_key: String, model: String, temperature: Option<f32>) -> Self {
        let http_client = shared_http_client().clone();

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
            build_jobs: None,
            build_target_dir: None,
            extra_env: Vec::new(),
            jobs: None,
            last_job: Arc::new(Mutex::new(None)),
            jobs_disabled: false,
        }
    }

    /// Confine every bash step to an isolated network namespace (`unshare -n`).
    pub fn with_network_offline(mut self, offline: bool) -> Self {
        self.network_offline = offline;
        self
    }

    /// Layer `vars` on top of the sanitized environment of the next
    /// `execute_bash` call. The caller clones per command, so the overlay
    /// never leaks into an unrelated step.
    pub fn with_extra_env(mut self, vars: Vec<(String, String)>) -> Self {
        self.extra_env = vars;
        self
    }

    /// Refuse to turn a command that outlives its budget into a job.
    ///
    /// For a run the harness owns -- the completion verify, its divergent
    /// variant, the merge gate -- and not the model: a job nobody waits on
    /// would hold a build slot for its whole ceiling, and a gate must be
    /// decided by the command's real exit code. Such a command is stopped and
    /// reported as the timeout it is.
    pub fn without_job_conversion(mut self) -> Self {
        self.jobs_disabled = true;
        self
    }

    /// The job table a command may background into, if this run allows one.
    pub(crate) fn job_handle(&self) -> Option<&JobHandle> {
        if self.jobs_disabled {
            None
        } else {
            self.jobs.as_ref()
        }
    }

    /// Attach the worker's background jobs to this runner.
    ///
    /// A runner without them has nowhere to register a command that outlived
    /// its budget, so such a command is killed instead of continued.
    pub fn with_jobs(mut self, jobs: JobHandle) -> Self {
        self.jobs = Some(jobs);
        self
    }

    /// Wait for background job `id`, up to `limit`.
    ///
    /// `None` when this runner has no job `id`, which is how the caller tells
    /// "no such job" from "the job is still running".
    pub async fn wait_job(&self, id: u64, limit: Duration) -> Option<JobWait> {
        self.jobs.as_ref()?.wait(id, limit).await
    }

    /// Stop background job `id`; `false` when this runner has no such job.
    pub fn kill_job(&self, id: u64) -> bool {
        self.jobs.as_ref().is_some_and(|handle| handle.kill(id))
    }

    /// Keep `guard` alive exactly as long as background job `id`.
    ///
    /// The admission permit a heavy command was granted under is moved into the
    /// job it became, so a job never outlives the build slot it was admitted
    /// with. A job that has already been reaped drops it immediately.
    pub fn attach_job_guard(&self, id: u64, guard: Box<dyn std::any::Any + Send>) -> bool {
        let Some(job) = self.jobs.as_ref().and_then(|handle| handle.job(id)) else {
            return false;
        };
        job.retain(guard);
        true
    }

    /// The job the last `execute_bash` call backgrounded, taken so it is
    /// reported once.
    ///
    /// The turn loop runs one command at a time, so the id cannot belong to any
    /// other command's job.
    pub fn take_last_job_id(&self) -> Option<u64> {
        lock_last_job(&self.last_job).take()
    }

    /// Record that `execute_bash` backgrounded job `id`.
    pub(crate) fn set_last_job_id(&self, id: u64) {
        *lock_last_job(&self.last_job) = Some(id);
    }

    /// Carry the admission controller's granted job count into the next
    /// `execute_bash` call (one command; the caller clones per command).
    pub fn with_build_jobs(mut self, jobs: usize) -> Self {
        self.build_jobs = Some(jobs.max(1));
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
        let messages = super::types::with_replayed_reasoning(messages);
        let payload = self.chat_request(&messages, &tools);

        let mut attempts = 0;
        let accumulator = loop {
            attempts += 1;

            let mut permit = acquire_llm(shared_llm_gate()).await?;
            let Some(mut resp) = self
                .send_with_retry(&payload, attempts, &mut permit)
                .await?
            else {
                continue;
            };

            if !resp.status().is_success() {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                let message = format!("LLM API returned HTTP {status}: {body}");
                if retry::is_transient_status(status) {
                    return Err(retry::LlmUnavailable(message).into());
                }
                anyhow::bail!(message);
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
                StreamRun::Retry => {
                    // Neither the response nor its slot survives retry backoff.
                    drop(resp);
                    drop(permit);
                    self.backoff_for_stream_failure(
                        attempts,
                        "LLM SSE stream failed; retrying request with backoff",
                    )
                    .await;
                    continue;
                }
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
        permit: &mut Option<OwnedSemaphorePermit>,
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
                    drop(resp);
                    drop(permit.take());
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
                drop(permit.take());
                tokio::time::sleep(delay).await;
                Ok(None)
            }
            Err(e) => Err(retry::LlmUnavailable(format!(
                "Failed to send request to LLM API after retries: {e}"
            ))
            .into()),
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
            let next_chunk =
                match tokio::time::timeout(self.stream_idle_timeout, resp.chunk()).await {
                    Ok(result) => result,
                    Err(_) => {
                        if attempt < self.max_retries {
                            tracing::warn!(attempt, "LLM SSE stream stalled (idle timeout)");
                            return Ok(StreamRun::Retry);
                        }
                        return Err(retry::LlmUnavailable(format!(
                            "LLM SSE stream stalled for {}s and did not resume after {} attempts",
                            self.stream_idle_timeout.as_secs(),
                            attempt
                        ))
                        .into());
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
                        tracing::warn!(attempt, error = %e, "LLM SSE stream chunk read failed");
                        return Ok(StreamRun::Retry);
                    }
                    return Err(retry::LlmUnavailable(format!(
                        "Failed reading stream chunk after retries: {e}"
                    ))
                    .into());
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
