//! Agent subsystem: LLM transport, command interception, and execution.

pub mod env;
pub mod exec;
pub mod intercept;
pub mod retry;
pub mod runner;
pub mod sandbox;
pub mod stream;
pub mod types;

pub use env::{
    ALLOWED_VARS, CARGO_HOME_VAR, CommandEnv, RUSTUP_HOME_VAR, TOOLCHAIN_VARS,
    apply_clean_environment, apply_clean_environment_cmd, build_clean_environment, host_cargo_home,
    is_sensitive_var, resolve_cargo_home,
};
pub use exec::{has_unshare, wrap_network_command};
pub use runner::{
    AgentRunner, DEFAULT_MAX_RETRIES, INITIAL_RETRY_DELAY_MS, MAX_RETRY_DELAY, max_llm_retries,
};
pub use sandbox::{
    DISABLE_LANDLOCK_ENV, LandlockPlan, TRUNCATE_HEAD, TRUNCATE_LIMIT, TRUNCATE_TAIL,
    apply_landlock_sandbox, build_landlock_plan, find_git_common_dir, find_git_dirs, has_bwrap,
    is_heavy_command, truncate_output, validate_bash_command,
};
pub use types::{
    AgentStepLog, ChatMessage, DEFAULT_STREAM_IDLE_TIMEOUT, LlmResponse,
    MAX_STREAMED_CONTENT_BYTES, MAX_TOOL_ARGUMENT_BYTES, Role, SYSTEM_PROMPT, ToolCall, ToolCallFn,
};
