//! `mini-swe-mcp` entry point: hand argv to the CLI, or serve MCP over stdio.
//!
//! Deliberately thin: dispatch and orchestration only. Everything else lives in
//! the library so it can be unit-tested directly.

use anyhow::Result;
use mini_swe_mcp::cli::args::{
    action_of, admin_requested, json_requested, stdio_requested, strip_admin_flag,
    strip_json_flag, tool_args,
};
use mini_swe_mcp::cli::format::format_output;
use mini_swe_mcp::manifest::{BUILTIN_DEFAULT_MODEL, ModelManifest};
use mini_swe_mcp::mcp::McpServer;
use mini_swe_mcp::pool::WorkerPool;
use mini_swe_mcp::{bootstrap, config, telemetry};
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
    // `--admin` is a connection flag like `--json`: stripped before the
    // positional parse, remembered as the operator's ownership override.
    let admin = admin_requested(&raw_args);
    let cli_args = strip_admin_flag(strip_json_flag(raw_args));

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
        return run_remote_action(action, &cli_args, json_output, admin).await;
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
        return run_action(&server, action, &cli_args, json_output, !api_key.is_empty(), admin).await;
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
///
/// `admin` is the operator's `--admin`: the handshake then lifts the per-agent
/// ownership check, so the CLI can steer, kill, collect and wait on a worker
/// another agent dispatched.
async fn run_remote_action(
    action: &str,
    cli_args: &[String],
    json_output: bool,
    admin: bool,
) -> Result<()> {
    let api_key_present = !env::var("OPENAI_API_KEY").unwrap_or_default().is_empty();
    let Some(tool_args) = tool_args(action, cli_args, api_key_present)? else {
        return Ok(());
    };
    let mut client = mini_swe_mcp::hub::HubClient::connect_as_admin(admin).await?;
    let result = drive_worker_call(tool_args, async |args| client.worker(args).await).await?;
    print_result(action, &result, json_output)
}

/// Issue one `worker` tool call and, while the worker is paused for input,
/// run the operator dialogue as `steer` + `wait` calls.
///
/// `call` is the only transport difference between the in-process pool and
/// the hub daemon, so both CLI paths share this one algorithm.
async fn drive_worker_call(
    tool_args: serde_json::Map<String, serde_json::Value>,
    mut call: impl AsyncFnMut(serde_json::Value) -> Result<serde_json::Value>,
) -> Result<serde_json::Value> {
    let mut result = call(serde_json::Value::Object(tool_args)).await?;
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
            call(serde_json::json!({"action": "kill", "worker_id": wid})).await?;
            break;
        }

        call(serde_json::json!({"action": "steer", "worker_id": wid, "message": input})).await?;
        eprintln!("[mini-swe] Guidance sent. Resuming execution...");
        // `wait` blocks until the worker finishes, fails or pauses again.
        result = call(serde_json::json!({"action": "wait", "worker_id": wid})).await?;
    }
    Ok(result)
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

/// Execute one CLI-selected action in this process and print the result.
///
/// `admin` is the operator's `--admin`: the in-process connection owns every
/// worker it dispatches, but the override is what lets it act on the rows a
/// hub-mode agent left in the shared registry.
async fn run_action(
    server: &McpServer,
    action: &str,
    cli_args: &[String],
    json_output: bool,
    api_key_present: bool,
    admin: bool,
) -> Result<()> {
    // `None` means the verb was already answered (or exited) by `tool_args`.
    let Some(tool_args) = tool_args(action, cli_args, api_key_present)? else {
        return Ok(());
    };
    let ctx = mini_swe_mcp::mcp::ConnectionContext {
        admin,
        ..mini_swe_mcp::mcp::ConnectionContext::stdio()
    };
    let result = drive_worker_call(tool_args, async |args| {
        server.execute_tool_for("worker", args, &ctx).await
    })
    .await?;
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
    println!("  list [--all]");
    println!("  monitor [--once]");
    println!("  supervisor [--once]");
    println!("  kill <worker_id>");
    println!("  manifest");
    println!("  prune");
    println!("  daemon");
    println!("\nFlags:");
    println!("{}", mini_swe_mcp::cli::HELP_FLAGS);
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
