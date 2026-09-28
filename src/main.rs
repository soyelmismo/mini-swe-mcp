mod agent;
mod config;
mod manifest;
mod mcp;
mod pool;
mod worktree;

use anyhow::{Context, Result};
use config::xdg_config_dir;
use manifest::ModelManifest;
use mcp::McpServer;
use pool::WorkerPool;
use std::env;
use tracing_subscriber::{EnvFilter, fmt, prelude::*};

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
    if env::var("OPENAI_API_KEY").is_err()
        && let Some(dir) = xdg_config_dir()
    {
        dotenvy::from_path(dir.join("mini-swe").join(".env")).ok();
    }

    // 3. Try loading alongside the executable or from ancestor folders
    if env::var("OPENAI_API_KEY").is_err()
        && let Ok(exe) = env::current_exe()
        && let Some(parent) = exe.parent()
    {
        dotenvy::from_path(parent.join(".env")).ok();
        if let Some(grandparent) = parent.parent().and_then(|p| p.parent()) {
            dotenvy::from_path(grandparent.join(".env")).ok();
        }
    }

    // 4. Try loading from explicitly specified ENV_FILE
    if env::var("OPENAI_API_KEY").is_err()
        && let Ok(custom_env) = env::var("ENV_FILE")
    {
        dotenvy::from_path(custom_env).ok();
    }

    let api_base =
        env::var("OPENAI_API_BASE").unwrap_or_else(|_| "https://api.openai.com/v1".to_string());

    let api_key = env::var("OPENAI_API_KEY").context(
        "Missing OPENAI_API_KEY. Please provide it via environment variable or .env file.",
    )?;

    let manifest = ModelManifest::load();

    let default_model = env::var("DEFAULT_MODEL").unwrap_or_else(|_| {
        manifest
            .default
            .clone()
            .unwrap_or_else(|| "ninja".to_string())
    });

    let max_workers = env::var("MAX_CONCURRENT_WORKERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(64); // Supports up to 64 concurrent subagents out of the box

    let pool = WorkerPool::new(max_workers, api_base, api_key);
    let server = McpServer::new(pool, default_model, manifest);

    let cli_args: Vec<String> = env::args().collect();
    if cli_args.len() > 1 && cli_args[1] != "--stdio" {
        let action = &cli_args[1];
        let mut tool_args = serde_json::Map::new();
        tool_args.insert("action".into(), serde_json::Value::String(action.clone()));

        match action.as_str() {
            "dispatch" => {
                if cli_args.len() < 3 {
                    eprintln!(
                        "Usage: mini-swe-mcp dispatch <task> [--model <model>] [--repo <repo>] [--wait]"
                    );
                    return Ok(());
                }
                tool_args.insert(
                    "task".into(),
                    serde_json::Value::String(cli_args[2].clone()),
                );
                let mut i = 3;
                while i < cli_args.len() {
                    match cli_args[i].as_str() {
                        "--model" | "-m" => {
                            if i + 1 < cli_args.len() {
                                tool_args.insert(
                                    "model".into(),
                                    serde_json::Value::String(cli_args[i + 1].clone()),
                                );
                                i += 1;
                            }
                        }
                        "--repo" | "-r" => {
                            if i + 1 < cli_args.len() {
                                tool_args.insert(
                                    "repo_path".into(),
                                    serde_json::Value::String(cli_args[i + 1].clone()),
                                );
                                i += 1;
                            }
                        }
                        "--wait" | "-w" => {
                            tool_args.insert("wait".into(), serde_json::Value::Bool(true));
                        }
                        _ => {}
                    }
                    i += 1;
                }
            }
            "status" | "collect" | "kill" => {
                if cli_args.len() > 2 {
                    tool_args.insert(
                        "worker_id".into(),
                        serde_json::Value::String(cli_args[2].clone()),
                    );
                }
            }
            "steer" => {
                if cli_args.len() > 3 {
                    tool_args.insert(
                        "worker_id".into(),
                        serde_json::Value::String(cli_args[2].clone()),
                    );
                    tool_args.insert(
                        "message".into(),
                        serde_json::Value::String(cli_args[3].clone()),
                    );
                }
            }
            "manifest" | "list" => {}
            _ => {
                eprintln!(
                    "Unknown action: {}. Available: dispatch, status, steer, collect, list, kill, manifest",
                    action
                );
                return Ok(());
            }
        }

        let result = server
            .execute_tool("worker", serde_json::Value::Object(tool_args))
            .await?;
        println!("{}", serde_json::to_string_pretty(&result)?);
        return Ok(());
    }

    server.run_stdio().await
}
