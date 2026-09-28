use anyhow::{Context, Result};
use mini_swe_mcp::config::xdg_config_dir;
use mini_swe_mcp::manifest::ModelManifest;
use mini_swe_mcp::mcp::McpServer;
use mini_swe_mcp::pool::{WorkerPool, WorkerState};
use mini_swe_mcp::worktree;
use std::env;
use tracing_subscriber::{EnvFilter, fmt, prelude::*};

#[tokio::main]
async fn main() -> Result<()> {
    let cli_args: Vec<String> = env::args().collect();

    // Early CLI flag handling without requiring API keys
    if cli_args.len() > 1 {
        match cli_args[1].as_str() {
            "--version" | "-V" => {
                println!("mini-swe-mcp {}", env!("CARGO_PKG_VERSION"));
                return Ok(());
            }
            "--help" | "-h" => {
                println!("mini-swe-mcp {}", env!("CARGO_PKG_VERSION"));
                println!("Usage: mini-swe-mcp [--stdio | <action> [args...]]");
                println!("\nActions:");
                println!("  dispatch <task> [--model <model>] [--repo <repo>] [--wait]");
                println!("  status <worker_id>");
                println!("  collect <worker_id>");
                println!("  steer <worker_id> <message>");
                println!("  list");
                println!("  kill <worker_id>");
                println!("  manifest");
                println!("  prune");
                println!("\nFlags:");
                println!("  -h, --help     Print help");
                println!("  -V, --version  Print version");
                return Ok(());
            }
            _ => {}
        }
    }

    // Crucial: log to STDERR, because STDOUT is dedicated to MCP JSON-RPC protocol
    tracing_subscriber::registry()
        .with(fmt::layer().with_writer(std::io::stderr))
        .with(EnvFilter::from_default_env().add_directive(tracing::Level::INFO.into()))
        .init();

    // Safe startup cleanup: prune leftover zombie worktrees/branches from dead processes
    worktree::prune_stale_worktrees(&std::path::PathBuf::from("."));

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
    let server = McpServer::new(pool.clone(), default_model, manifest);

    if cli_args.len() > 1 && cli_args[1] != "--stdio" {
        let action = &cli_args[1];

        if action == "prune" {
            worktree::prune_stale_worktrees(&std::path::PathBuf::from("."));
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "status": "ok",
                    "message": "Stale worktrees and orphaned worker branches pruned"
                }))?
            );
            return Ok(());
        }

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
                    "Unknown action: {}. Available: dispatch, status, steer, collect, list, kill, manifest, prune",
                    action
                );
                std::process::exit(1);
            }
        }

        let mut result = server
            .execute_tool("worker", serde_json::Value::Object(tool_args))
            .await?;

        while result.get("status").and_then(|v| v.as_str()) == Some("needs_input") {
            let wid = result["worker_id"].as_str().unwrap_or("").to_string();
            let q = result["question"].as_str().unwrap_or("");
            eprintln!("\n[mini-swe] Worker {} is PAUSED: {}", wid, q);
            eprint!("Reply with guidance (or press Enter to abort): ");
            let mut input = String::new();
            std::io::stdin().read_line(&mut input)?;
            let input = input.trim().to_string();
            if input.is_empty() {
                eprintln!("[mini-swe] No input provided; terminating worker.");
                pool.kill(&wid).await;
                break;
            }

            pool.steer(&wid, input).await?;
            eprintln!("[mini-swe] Guidance sent. Resuming execution...");

            loop {
                tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                if let Some(state) = pool.get_worker_state(&wid).await {
                    match state {
                        WorkerState::Completed { .. }
                        | WorkerState::Failed { .. } => {
                            let logs = pool.get_worker_logs(&wid).await.unwrap_or_default();
                            result = serde_json::json!({
                                "worker_id": wid,
                                "state": state,
                                "logs": logs
                            });
                            break;
                        }
                        WorkerState::Paused {
                            ref question,
                            step,
                            ..
                        } => {
                            result = serde_json::json!({
                                "worker_id": wid,
                                "status": "needs_input",
                                "question": question,
                                "step": step,
                                "message": "Worker is paused waiting for orchestrator steering."
                            });
                            break;
                        }
                        _ => {}
                    }
                }
            }
        }

        println!("{}", serde_json::to_string_pretty(&result)?);
        return Ok(());
    }

    tokio::select! {
        res = server.run_stdio() => res,
        _ = tokio::signal::ctrl_c() => {
            tracing::info!("Received SIGINT, shutting down stdio server");
            Ok(())
        }
    }
}
