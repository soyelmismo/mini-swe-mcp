//! MCP server: JSON-RPC stdio transport, tool schema and worker verbs.
//!
//! Split by responsibility while keeping the historical
//! `mini_swe_mcp::mcp::*` surface byte-for-byte identical through the
//! re-exports below:
//!
//! * `protocol` — JSON-RPC 2.0 envelopes and the `tools/call` result
//!   serializer.
//! * `events` — the `claude/channel` worker events pushed into the
//!   orchestrator session while it runs.
//! * `schema` — the advertised `worker` tool contract: verb list, property
//!   table and precomputed `tools/list` payload.
//! * `handlers` — argument extraction, progress notifications and the handler
//!   behind every verb in [`WORKER_ACTIONS`].
//! * `server` — [`McpServer`] itself: construction, the `initialize`
//!   handshake, the stdio run loop and the shared wait loop.

mod events;
mod handlers;
mod protocol;
mod schema;
mod server;

pub use events::{
    ChannelEvent, EventKind, Outcome, WorkerSnapshot, WorkerView, channel_frame, diff_events,
    render_for_test,
};
pub use schema::{NETWORK_DEFAULT, NETWORK_MODES, WORKER_ACTIONS};
pub use server::{ConnectionContext, McpServer};
