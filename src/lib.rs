//! Library surface for `mini-swe-mcp`.
//!
//! The binary in `main.rs` is a thin wrapper around these modules so they can
//! also be exercised from integration tests in `tests/`.

pub mod agent;
pub mod bootstrap;
pub mod cache;
pub mod cli;
pub mod config;
pub mod hub;
pub mod manifest;
pub mod mcp;
pub mod monitor;
pub mod pool;
pub mod telemetry;
pub mod worktree;
