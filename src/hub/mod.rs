//! Hub daemon: the single long-lived process owning the only [`WorkerPool`].
//!
//! Thin MCP clients dial [`HubPaths::socket`] and speak the same
//! newline-delimited JSON-RPC the stdio server speaks; every connection is
//! served by [`McpServer::serve_connection`](crate::mcp::McpServer::serve_connection)
//! against the one shared pool. [`hub_dir`] owns the socket, the lock and the
//! log, and refuses a directory it does not exclusively own.
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

pub mod client;
mod daemon;
pub mod identity;

pub use client::{HubClient, connect_or_spawn, decode_ambient_env, proxy_stdio};
pub use daemon::{
    HubConfig, HubPaths, HubServer, WatchTokens, hub_dir, run_daemon, watch_token_identity,
};
