//! Hub daemon: the single long-lived process owning the only
//! [`WorkerPool`](crate::pool::WorkerPool).
//!
//! Thin MCP clients dial [`HubPaths::socket`] and speak the same
//! newline-delimited JSON-RPC the stdio server speaks; every connection is
//! served by [`McpServer::serve_connection`](crate::mcp::McpServer::serve_connection)
//! against the one shared pool. [`hub_dir`] owns the socket, the lock and the
//! log, and refuses a directory it does not exclusively own. The log is bounded
//! rather than archived: past [`LOG_ROTATE_BYTES`] it is renamed to
//! `hub.log.1` and a fresh one takes over, so the hub directory holds at most
//! two generations (see [`rotated_log_path`]).
//!
//! Because the pool is shared, each connection carries an agent identity (see
//! [`crate::mcp::ConnectionContext::agent`]): the `MINI_SWE_AGENT_ID` of its
//! `hub/hello`, else the identity a `MINI_SWE_WATCH_TOKEN` names, else the host
//! process [`identity`] walks up to — the `claude` or `opencode` process that
//! spawned both the connection and the agent's shell commands — qualified by
//! the session running inside it, and only then its `initialize` `clientInfo`.
//! Workers belong to the agent that dispatched them, and the mutating verbs
//! refuse anyone else. [`client`]'s `admin` hello is the operator's override.
//!
//! [`WatchTokens`] is what ties the two halves together: a shell cannot know
//! its session, so the daemon mints it one token per identity and hands it out
//! as the `watch_command` of every dispatch and steer answer.
//!
//! A daemon that goes away under a long-lived client is not that client's end:
//! [`client::reconnect_following`] re-dials within one budget
//! ([`client::reconnect_deadline`]) while [`client::daemon_went_away`] decides
//! what counts as gone, so the CLI watch, the stdio proxy and every other thin
//! transport resume on the replacement daemon — with the same identity and, for
//! a watch, the events it missed replayed.
//!
//! A daemon is always this executable, spawned through [`exe_path`]: a rebuilt
//! binary makes `current_exe()` read ` (deleted)`, so the shared helper there
//! is what lets both the client's auto-start and the daemon's handover respawn
//! the replacement instead of failing on a path that no longer exists.

pub(crate) mod auto_consolidate;
pub(crate) mod auto_handover;
pub mod client;
mod daemon;
pub mod exe_path;
pub mod identity;

pub use client::{
    DEFAULT_RECONNECT_SECS, HubClient, RECONNECT_DEADLINE_ENV, connect_or_spawn, daemon_went_away,
    decode_ambient_env, proxy_stdio, reconnect_deadline, reconnect_following,
};
pub use daemon::{
    HubConfig, HubEndpoint, HubPaths, HubServer, LOG_ROTATE_BYTES, LOG_ROTATE_INTERVAL,
    WatchTokens, connect_endpoint, hub_dir, hub_dir_in, run_daemon, rotated_log_path,
    watch_token_identity,
};
