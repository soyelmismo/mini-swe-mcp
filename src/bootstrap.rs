//! Process bootstrap: runtime sizing and `.env` discovery.
//!
//! One-shot, side-effecting steps that must run before any CLI action is
//! dispatched; none is interesting to the dispatch logic itself.

use crate::config::{env_parse, xdg_config_dir};
use anyhow::Result;
use std::env;

/// Tokio worker threads for the daemon runtime.
///
/// Pinned explicitly so the server never degrades to a single worker on a
/// one-core deployment target: with one worker a long `execute_bash` wait would
/// block every progress notification and every other in-flight request (see
/// audit F8). Overridable at runtime with `MINI_SWE_WORKER_THREADS`.
///
/// Only the hub daemon sizes a runtime from this: the pool it owns runs real
/// worker turns, and blocking work is offloaded to the blocking pool. The thin
/// transports — the `--stdio` proxy and every CLI verb — relay JSON-RPC frames
/// and own no worker, so they run on [`runtime_current_thread`] instead.
pub const DEFAULT_WORKER_THREADS: usize = 4;

/// Build the multi-threaded runtime the hub daemon runs on.
///
/// Only [`crate::hub::run_daemon`] needs worker threads: it owns the single
/// [`crate::pool::WorkerPool`], whose turns and `spawn_blocking` calls are the
/// process's real concurrency. See [`runtime_current_thread`] for every other
/// entry point.
pub fn runtime() -> Result<tokio::runtime::Runtime> {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(resolve_worker_threads())
        .enable_all()
        .build()
        .map_err(Into::into)
}

/// Build the single-threaded runtime the thin transports run on.
///
/// The `--stdio` proxy and every CLI verb only relay newline-delimited
/// JSON-RPC: one socket or pipe in, one out, no worker turns of their own. A
/// current-thread runtime therefore serves them fully while costing one thread
/// instead of a multi-thread scheduler (a relay process that paid for worker
/// threads it never used resident-swapped a megabyte of stack per session).
///
/// Blocking work still leaves this thread: the pool's `spawn_blocking` calls
/// run on the runtime's blocking pool, which exists on a current-thread runtime
/// too, so a proxy that starts a daemon inherits nothing that needs workers.
pub fn runtime_current_thread() -> Result<tokio::runtime::Runtime> {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(Into::into)
}

/// The scheduler a process needs.
///
/// [`Flavor::MultiThread`] belongs to the processes that own a worker
/// pool in this process — the hub daemon and the `MINI_SWE_NO_DAEMON=1`
/// escape hatch — whose worker turns and `spawn_blocking` calls are the
/// process's real concurrency. Every other invocation is a thin
/// transport that relays newline-delimited JSON-RPC to the daemon, and
/// a relay is served completely by [`Flavor::CurrentThread`]: one
/// thread, no worker pool to pay for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Flavor {
    /// One thread: the caller's own, which `block_on` drives directly.
    CurrentThread,
    /// A worker pool sized by [`DEFAULT_WORKER_THREADS`].
    MultiThread,
}

/// The scheduler the invocation `action` needs.
///
/// `action` is the verb the binary was invoked with, `None` for the
/// `--stdio` proxy, which has no verb of its own; `no_daemon` is
/// `MINI_SWE_NO_DAEMON=1`, the escape hatch that serves MCP in this
/// process instead of through the hub.
///
/// The `daemon` verb always owns the pool. Without the escape hatch
/// nothing else does: the proxy and every CLI verb answer through the
/// hub daemon, so they relay. With it, the process runs the pool
/// itself and needs workers — except for the verbs that only read the
/// registry or the environment, which never build a pool and stay thin
/// either way.
pub fn flavor_for(action: Option<&str>, no_daemon: bool) -> Flavor {
    if action == Some("daemon") {
        return Flavor::MultiThread;
    }
    if !no_daemon {
        return Flavor::CurrentThread;
    }
    match action {
        Some("monitor" | "supervisor" | "whoami" | "help" | "watch") => {
            Flavor::CurrentThread
        }
        _ => Flavor::MultiThread,
    }
}

/// Resolve the Tokio worker thread count from the environment.
pub fn resolve_worker_threads() -> usize {
    env_parse("MINI_SWE_WORKER_THREADS")
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


#[cfg(test)]
mod flavor_tests {
    use super::*;

    /// The verbs that go through the `worker` tool, plus the verbs that
    /// answer from the registry, so both sides of the table are covered.
    const VERBS: &[&str] = &[
        "dispatch", "collect", "review", "merge", "kill", "discard",
        "logs", "status", "consolidate", "steer", "approve", "unapprove",
        "list", "manifest", "reap", "prune",
    ];

    /// The hub daemon owns the only pool, so it is the one process that
    /// always needs the multi-thread runtime.
    #[test]
    fn the_daemon_always_keeps_the_worker_pool() {
        assert_eq!(flavor_for(Some("daemon"), false), Flavor::MultiThread);
        assert_eq!(flavor_for(Some("daemon"), true), Flavor::MultiThread);
    }

    /// The `--stdio` proxy and every CLI verb answer through the hub
    /// daemon, so they relay frames and run on one thread.
    #[test]
    fn the_proxy_and_every_cli_verb_relay_on_one_thread() {
        assert_eq!(flavor_for(None, false), Flavor::CurrentThread);
        for verb in VERBS {
            assert_eq!(
                flavor_for(Some(verb), false),
                Flavor::CurrentThread,
                "{verb} must relay through the hub"
            );
        }
    }

    /// The escape hatch serves MCP in this process, so it owns the pool
    /// and needs the workers the daemon would otherwise run.
    #[test]
    fn the_escape_hatch_owns_the_pool_it_serves() {
        assert_eq!(flavor_for(None, true), Flavor::MultiThread);
        for verb in VERBS {
            assert_eq!(
                flavor_for(Some(verb), true),
                Flavor::MultiThread,
                "{verb} must run its worker in this process"
            );
        }
    }

    /// The verbs that only read the registry or the environment never
    /// build a pool, so they stay thin even in the escape hatch.
    #[test]
    fn registry_verbs_stay_thin_in_the_escape_hatch() {
        for verb in ["monitor", "supervisor", "whoami", "help", "watch"] {
            assert_eq!(
                flavor_for(Some(verb), true),
                Flavor::CurrentThread,
                "{verb} reads the registry, not the pool"
            );
        }
    }
}
