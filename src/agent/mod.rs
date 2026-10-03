//! Agent subsystem: LLM transport, command interception, and execution.

pub mod consolidate;
pub mod env;
pub mod exec;
pub mod intercept;
pub mod jobs;
pub mod reap;
pub mod retry;
pub mod runner;
pub mod sandbox;
pub mod stream;
pub mod types;

pub use consolidate::CONSOLIDATOR_INSTRUCTIONS;
pub use env::{
    ALLOWED_VARS, AMBIENT_ENV_MAX_BYTES, CARGO_HOME_VAR, RUSTUP_HOME_VAR, TOOLCHAIN_VARS,
    ambient_environment_snapshot, apply_clean_environment_cmd, build_clean_environment,
    host_cargo_home, is_secret_name, resolve_cargo_home, sanitize_ambient_value,
};
pub use exec::{has_unshare, wrap_network_command};
pub use jobs::{
    DEFAULT_JOB_MAX_SECS, DEFAULT_WAIT_JOB_SECS, JobEnd, JobHandle, JobOutcome, JobState,
    JobStatus, JobTable, JobWait, job_max_secs, wait_job_secs,
};
pub use retry::{DEFAULT_MAX_RETRIES, INITIAL_RETRY_DELAY_MS, MAX_RETRY_DELAY, max_llm_retries};
pub use runner::AgentRunner;
pub use sandbox::{
    DISABLE_LANDLOCK_ENV, KernelConfinement, LandlockPlan, SeccompFilter, TRUNCATE_HEAD,
    TRUNCATE_LIMIT, TRUNCATE_TAIL, build_landlock_plan, find_git_common_dir, find_git_dirs,
    has_bwrap, is_heavy_command, truncate_output, validate_bash_command,
};
pub use stream::extract_command;
pub use types::{
    AgentStepLog, ChatMessage, DEFAULT_STREAM_IDLE_TIMEOUT, LlmResponse,
    MAX_STREAMED_CONTENT_BYTES, MAX_TOOL_ARGUMENT_BYTES, Role, SYSTEM_PROMPT, ToolCall, ToolCallFn,
    strip_replayed_reasoning,
};
