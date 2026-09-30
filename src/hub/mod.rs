//! Hub daemon: the single long-lived process owning the only [`WorkerPool`].
//!
//! Thin MCP clients (a later task) dial [`HubPaths::socket`] and speak the same
//! newline-delimited JSON-RPC the stdio server speaks; every connection is
//! served by [`McpServer::serve_connection`](crate::mcp::McpServer::serve_connection)
//! against the one shared pool. [`hub_dir`] owns the socket, the lock and the
//! log, and refuses a directory it does not exclusively own.

mod daemon;

pub use daemon::{HubConfig, HubPaths, HubServer, hub_dir, run_daemon};
