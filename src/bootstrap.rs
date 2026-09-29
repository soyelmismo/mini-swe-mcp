//! Process bootstrap: runtime sizing and `.env` discovery.
//!
//! One-shot, side-effecting steps that must run before any CLI action is
//! dispatched; none is interesting to the dispatch logic itself.

use crate::config::xdg_config_dir;
use anyhow::Result;
use std::env;

/// Tokio worker threads for the MCP server runtime.
///
/// Pinned explicitly so the server never degrades to a single worker on a
/// one-core deployment target: with one worker a long `execute_bash` wait would
/// block every progress notification and every other in-flight request (see
/// audit F8). Overridable at runtime with `MINI_SWE_WORKER_THREADS`.
pub const DEFAULT_WORKER_THREADS: usize = 4;

/// Build the multi-threaded runtime the server and CLI run on.
pub fn runtime() -> Result<tokio::runtime::Runtime> {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(resolve_worker_threads())
        .enable_all()
        .build()
        .map_err(Into::into)
}

/// Resolve the Tokio worker thread count from the environment.
pub fn resolve_worker_threads() -> usize {
    env::var("MINI_SWE_WORKER_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|n| *n > 0)
        .unwrap_or(DEFAULT_WORKER_THREADS)
}

/// Populate the process environment from every `.env` location we trust, in
/// precedence order. `dotenvy` never overrides an already-set variable, so the
/// first source to define a key wins.
pub fn load_dotenv_files() {
    // 1. Current working directory or ancestor directories
    dotenvy::dotenv().ok();

    // 2. XDG standard config directory ($XDG_CONFIG_HOME/mini-swe/.env or ~/.config/mini-swe/.env)
    if env::var("OPENAI_API_KEY").is_err()
        && let Some(dir) = xdg_config_dir()
    {
        dotenvy::from_path(dir.join("mini-swe").join(".env")).ok();
    }

    // 3. Alongside the executable or from ancestor folders
    if env::var("OPENAI_API_KEY").is_err()
        && let Ok(exe) = env::current_exe()
        && let Some(parent) = exe.parent()
    {
        dotenvy::from_path(parent.join(".env")).ok();
        if let Some(grandparent) = parent.parent().and_then(|p| p.parent()) {
            dotenvy::from_path(grandparent.join(".env")).ok();
        }
    }

    // 4. Explicitly specified ENV_FILE
    if env::var("OPENAI_API_KEY").is_err()
        && let Ok(custom_env) = env::var("ENV_FILE")
    {
        dotenvy::from_path(custom_env).ok();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_worker_threads_defaults_to_the_pinned_count() {
        const { assert!(DEFAULT_WORKER_THREADS > 1) };
        if env::var("MINI_SWE_WORKER_THREADS").is_err() {
            assert_eq!(resolve_worker_threads(), DEFAULT_WORKER_THREADS);
        }
    }

    #[test]
    fn test_resolve_worker_threads_ignores_nonsense_values() {
        let saved = env::var("MINI_SWE_WORKER_THREADS").ok();
        unsafe { env::set_var("MINI_SWE_WORKER_THREADS", "0") };
        assert_eq!(resolve_worker_threads(), DEFAULT_WORKER_THREADS);
        unsafe { env::set_var("MINI_SWE_WORKER_THREADS", "-3") };
        assert_eq!(resolve_worker_threads(), DEFAULT_WORKER_THREADS);
        unsafe { env::set_var("MINI_SWE_WORKER_THREADS", "not-a-number") };
        assert_eq!(resolve_worker_threads(), DEFAULT_WORKER_THREADS);

        match saved {
            Some(v) => unsafe { env::set_var("MINI_SWE_WORKER_THREADS", v) },
            None => unsafe { env::remove_var("MINI_SWE_WORKER_THREADS") },
        }
    }
}
