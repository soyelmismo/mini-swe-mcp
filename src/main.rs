mod agent;
mod mcp;
mod pool;
mod worktree;

use anyhow::{Context, Result};
use pool::WorkerPool;
use mcp::McpServer;
use std::env;
use std::path::Path;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

#[tokio::main]
async fn main() -> Result<()> {
    // Crucial: log to STDERR, because STDOUT is dedicated to MCP JSON-RPC protocol
    tracing_subscriber::registry()
        .with(fmt::layer().with_writer(std::io::stderr))
        .with(EnvFilter::from_default_env().add_directive(tracing::Level::INFO.into()))
        .init();

    // 1. Try loading from current working directory or ancestor directories
    dotenvy::dotenv().ok();

    // 2. Try loading from XDG standard config directory ($XDG_CONFIG_HOME/mini-swe/.env or ~/.config/mini-swe/.env)
    if env::var("OPENAI_API_KEY").is_err() {
        let config_dir = env::var("XDG_CONFIG_HOME")
            .map(std::path::PathBuf::from)
            .or_else(|_| env::var("HOME").map(|h| Path::new(&h).join(".config")))
            .ok();

        if let Some(dir) = config_dir {
            dotenvy::from_path(dir.join("mini-swe").join(".env")).ok();
        }
    }

    // 3. Try loading alongside the executable or from ancestor folders
    if env::var("OPENAI_API_KEY").is_err() {
        if let Ok(exe) = env::current_exe() {
            if let Some(parent) = exe.parent() {
                dotenvy::from_path(parent.join(".env")).ok();
                if let Some(grandparent) = parent.parent().and_then(|p| p.parent()) {
                    dotenvy::from_path(grandparent.join(".env")).ok();
                }
            }
        }
    }

    // 4. Try loading from explicitly specified ENV_FILE
    if env::var("OPENAI_API_KEY").is_err() {
        if let Ok(custom_env) = env::var("ENV_FILE") {
            dotenvy::from_path(custom_env).ok();
        }
    }

    let api_base = env::var("OPENAI_API_BASE")
        .unwrap_or_else(|_| "https://api.openai.com/v1".to_string());

    let api_key = env::var("OPENAI_API_KEY")
        .context("Missing OPENAI_API_KEY. Please provide it via environment variable or .env file.")?;

    let default_model = env::var("DEFAULT_MODEL")
        .unwrap_or_else(|_| "combo:ninja".to_string());

    let max_workers = env::var("MAX_CONCURRENT_WORKERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(64); // Supports up to 64 concurrent subagents out of the box

    let pool = WorkerPool::new(max_workers, api_base, api_key);
    let server = McpServer::new(pool, default_model);

    server.run_stdio().await
}
