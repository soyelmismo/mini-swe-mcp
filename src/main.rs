//! `mini-swe-mcp` entry point: hand argv to the CLI, or serve MCP over stdio.
//!
//! Deliberately thin: dispatch and orchestration only. Everything else lives in
//! the library so it can be unit-tested directly.

use anyhow::Result;
use mini_swe_mcp::cli::args::{action_of, json_requested, stdio_requested, strip_json_flag, tool_args};
use mini_swe_mcp::cli::format::format_output;
use mini_swe_mcp::manifest::{BUILTIN_DEFAULT_MODEL, ModelManifest};
use mini_swe_mcp::mcp::McpServer;
use mini_swe_mcp::pool::WorkerPool;
use mini_swe_mcp::{bootstrap, config, telemetry, worktree};
use std::env;
use std::sync::Arc;

fn main() -> Result<()> {
    bootstrap::runtime()?.block_on(async_main())
}

async fn async_main() -> Result<()> {
    let raw_args: Vec<String> = env::args().collect();

    // `--version` and `--help` must work without an API key, so they are
    // answered before any configuration is resolved.
    if let Some(first) = raw_args.get(1).map(String::as_str) {
        match first {
            "--version" | "-V" => {
                println!("mini-swe-mcp {}", env!("CARGO_PKG_VERSION"));
                return Ok(());
            }
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            _ => {}
        }
    }

    let json_output = json_requested(&raw_args);
    let cli_args = strip_json_flag(raw_args);

    // The dashboards read only the on-disk registry, so they run before any
    // configuration is resolved and work without an API key in any flag order.
    if let Some("monitor" | "supervisor") = action_of(&cli_args) {
        let once = cli_args.iter().any(|arg| arg == "--once");
        return mini_swe_mcp::monitor::run_monitor(once).await;
    }

    telemetry::init(stdio_requested(&cli_args));
    bootstrap::load_dotenv_files();

    if stdio_requested(&cli_args) || action_of(&cli_args).is_none() {
        if env::var("MINI_SWE_NO_DAEMON").ok().as_deref() == Some("1") {
            return run_local_stdio().await;
        }
        return mini_swe_mcp::hub::proxy_stdio().await;
    }

    if let Some(action) = action_of(&cli_args)
        && action != "daemon"
        && env::var("MINI_SWE_NO_DAEMON").ok().as_deref() != Some("1")
    {
        return run_remote_action(action, &cli_args, json_output).await;
    }

    let api_key = env::var("OPENAI_API_KEY").unwrap_or_default();
    let manifest = ModelManifest::load();
    let default_model = default_model(&manifest);
    let pool = WorkerPool::new(max_concurrent_workers(), api_base(), api_key.clone())
        .with_manifest(Arc::new(manifest));
    let server = McpServer::new(pool.clone(), default_model);

    if let Some(action) = action_of(&cli_args) {
        if action == "daemon" {
            return run_daemon_cmd(&server).await;
        }
        return run_action(&server, &pool, action, &cli_args, json_output, !api_key.is_empty()).await;
    }

    run_local_stdio().await
}

/// Serve MCP over stdio in this process (escape hatch for `MINI_SWE_NO_DAEMON=1`).
async fn run_local_stdio() -> Result<()> {
    let api_key = env::var("OPENAI_API_KEY").unwrap_or_default();
    if api_key.is_empty() {
        anyhow::bail!("Missing OPENAI_API_KEY. Please provide it via environment variable or .env file.");
    }
    let manifest = ModelManifest::load();
    let default_model = default_model(&manifest);
    let pool = WorkerPool::new(max_concurrent_workers(), api_base(), api_key)
        .with_manifest(Arc::new(manifest));
    let server = McpServer::new(pool.clone(), default_model);
    tokio::select! {
        res = server.run_stdio() => res,
        _ = tokio::signal::ctrl_c() => {
            tracing::info!("Received SIGINT, shutting down stdio server");
            let killed = pool.kill_all().await;
            if killed > 0 {
                tracing::info!(workers = killed, "Terminated active workers on shutdown");
            }
            Ok(())
        }
    }
}

/// Route one CLI action through the hub daemon: initialize, hello and one
/// `tools/call` whose payload and plain-text rendering match the local path.
async fn run_remote_action(action: &str, cli_args: &[String], json_output: bool) -> Result<()> {
    let api_key_present = !env::var("OPENAI_API_KEY").unwrap_or_default().is_empty();
    let Some(tool_args) = tool_args(action, cli_args, api_key_present)? else {
        return Ok(());
    };
    let mut client = mini_swe_mcp::hub::HubClient::connect().await?;
    let mut result = client
        .worker(serde_json::Value::Object(tool_args))
        .await?;

    // Interactive steering: the same operator dialogue the local pool drives,
    // spoken as `steer` + `wait` tool calls so the daemon owns the worker.
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
            client
                .worker(serde_json::json!({"action": "kill", "worker_id": wid}))
                .await?;
            break;
        }

        client
            .worker(serde_json::json!({"action": "steer", "worker_id": wid, "message": input}))
            .await?;
        eprintln!("[mini-swe] Guidance sent. Resuming execution...");
        // `wait` blocks in the daemon until the worker finishes, fails or
        // pauses again, mirroring the shared wait loop of the local path.
        result = client
            .worker(serde_json::json!({"action": "wait", "worker_id": wid}))
            .await?;
    }

    print_result(action, &result, json_output)
}

/// Run the hub daemon: the single process owning the only worker pool.
///
/// Foreground only; clients dial `hub.sock` and speak the same JSON-RPC the
/// stdio server speaks. No API key is required to start: the pool is built
/// exactly like the stdio path, and keyless dispatches fail lazily per call.
async fn run_daemon_cmd(server: &McpServer) -> Result<()> {
    let dir = mini_swe_mcp::hub::hub_dir()?;
    let running = mini_swe_mcp::hub::run_daemon(std::sync::Arc::new(server.clone()), dir, None).await?;
    if !running {
        println!("hub already running");
    }
    Ok(())
}

/// Execute one CLI-selected action and print the result.
async fn run_action(
    server: &McpServer,
    pool: &WorkerPool,
    action: &str,
    cli_args: &[String],
    json_output: bool,
    api_key_present: bool,
) -> Result<()> {
    // `prune` is a direct worktree call, not a `worker` tool verb.
    if action == "prune" {
        worktree::prune_stale_worktrees(&std::path::PathBuf::from("."));
        let res = serde_json::json!({
            "status": "ok",
            "message": "Stale worktrees and orphaned worker branches pruned"
        });
        return print_result(action, &res, json_output);
    }

    // `None` means the verb was already answered (or exited) by `tool_args`.
    let Some(tool_args) = tool_args(action, cli_args, api_key_present)? else {
        return Ok(());
    };

    let mut result = server
        .execute_tool("worker", serde_json::Value::Object(tool_args.clone()))
        .await?;

    // Interactive steering: whenever the shared wait loop reports the worker
    // paused for input, prompt the operator and resume. Re-waiting goes through
    // the same helper the MCP stdio dispatch path uses, so both callers share
    // one polling/termination algorithm.
    let wait_max_turns = tool_args
        .get("max_turns")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
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

        result = server
            .await_worker_result(&wid, wait_max_turns, None, None)
            .await?;
    }

    print_result(action, &result, json_output)
}

/// Print `result` as pretty JSON or through the action's plain-text renderer.
fn print_result(action: &str, result: &serde_json::Value, json_output: bool) -> Result<()> {
    if json_output {
        println!("{}", serde_json::to_string_pretty(result)?);
    } else {
        println!("{}", format_output(action, result));
    }
    Ok(())
}

fn print_help() {
    use mini_swe_mcp::cli::args::DISPATCH_USAGE;

    println!("mini-swe-mcp {}", env!("CARGO_PKG_VERSION"));
    println!("Usage: mini-swe-mcp [--stdio | [--json] <action> [args...]]");
    println!("\nActions:");
    println!("  {DISPATCH_USAGE}");
    println!("  status <worker_id>");
    println!("  collect <worker_id>");
    println!("  logs <worker_id>");
    println!("  reap");
    println!("  steer <worker_id> <message> [--wait] [--timeout <secs>]");
    println!("  wait <worker_id> [--timeout <secs>]");
    println!("  list");
    println!("  monitor [--once]");
    println!("  supervisor [--once]");
    println!("  kill <worker_id>");
    println!("  manifest");
    println!("  prune");
    println!("  daemon");
    println!("\nFlags:");
    println!("      --json     Output in JSON format (default is formatted plain text)");
    println!("  -h, --help     Print help");
    println!("  -V, --version  Print version");
}

/// Concurrent subagents supported out of the box.
fn max_concurrent_workers() -> usize {
    config::env_parse("MAX_CONCURRENT_WORKERS").unwrap_or(64)
}

/// OpenAI-compatible base URL, overridable through `OPENAI_API_BASE`.
fn api_base() -> String {
    env::var("OPENAI_API_BASE").unwrap_or_else(|_| "https://api.openai.com/v1".to_string())
}

/// Model handed to a dispatch that does not name one.
///
/// A `default` that names no known alias is already dropped by
/// `ModelManifest::normalize`, so reaching the built-in fallback here is
/// deliberate.
fn default_model(manifest: &ModelManifest) -> String {
    env::var("DEFAULT_MODEL").unwrap_or_else(|_| {
        manifest
            .default
            .clone()
            .unwrap_or_else(|| BUILTIN_DEFAULT_MODEL.to_string())
    })
}
