//! Everything the command line needs that is not the server itself.
//!
//! * [`args`] — argv → `worker` tool arguments, plus the `--json` / `--stdio`
//!   selectors and the dispatch usage line.
//! * [`format`] — the plain-text renderer for every tool payload, behind the
//!   single [`format_output`](format::format_output) entry point.
//! * [`suggest`] — the "did you mean …?" machinery behind the unknown-action
//!   error, and the single source of truth for the accepted verb list.
//!
//! All three are pure and synchronous, so they are unit-tested here rather than
//! only through the integration tests that spawn the binary.

pub mod args;
pub mod format;
pub mod suggest;

pub use self::args::{DISPATCH_USAGE, action_of, json_requested, stdio_requested, strip_json_flag, tool_args};
pub use self::format::format_output;
pub use self::suggest::{available_actions, suggest_action};
