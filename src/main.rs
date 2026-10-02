//! `mini-swe-mcp` entry point: hand argv to the CLI, or serve MCP over stdio.
//!
//! Deliberately thin: dispatch and orchestration only. Everything else lives in
//! the library so it can be unit-tested directly.

use anyhow::Result;
use mini_swe_mcp::cli::args::{
    action_of, admin_requested, json_requested, quiet_requested, stdio_requested, strip_admin_flag,
    strip_json_flag, strip_quiet_flag, tool_args,
};
use mini_swe_mcp::cli::format::{format_dispatch_quiet, format_output};
use mini_swe_mcp::manifest::{BUILTIN_DEFAULT_MODEL, ModelManifest};
use mini_swe_mcp::mcp::McpServer;
use mini_swe_mcp::pool::WorkerPool;
use mini_swe_mcp::{bootstrap, config, telemetry};
use std::env;
use std::sync::Arc;

fn main() -> Result<()> {
    let args = strip_admin_flag(strip_json_flag(env::args().collect()));
    if action_of(&args) == Some("status") && args.iter().any(|arg| arg == "--line") {
        mini_swe_mcp::monitor::print_status_line();
        return Ok(());
    }
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
    // `--quiet` is a rendering selector like `--json`: stripped before the
    // positional parse and remembered for the dispatch view.
    let quiet = quiet_requested(&raw_args);
    let cli_args = strip_quiet_flag(strip_admin_flag(strip_json_flag(raw_args)));

    // The dashboards read only the on-disk registry, so they run before any
    // configuration is resolved and work without an API key in any flag order.
    if let Some("monitor" | "supervisor") = action_of(&cli_args) {
        let once = cli_args.iter().any(|arg| arg == "--once");
        return mini_swe_mcp::monitor::run_monitor(once).await;
    }

    telemetry::init(stdio_requested(&cli_args));
    bootstrap::load_dotenv_files();

    // `whoami` answers from `/proc` and the environment alone: no hub, no API
    // key, no manifest, so it works wherever the agent itself can run. It runs
    // after the dotenv load so it reports the identity a dispatch would use.
    if action_of(&cli_args) == Some("whoami") {
        let identity = mini_swe_mcp::hub::identity::identity(mini_swe_mcp::mcp::CLI_AGENT);
        println!("agent {}", identity.id);
        println!("derived from {}", identity.explain());
        return Ok(());
    }

    // `help <topic>` prints the long-form guidance the MCP tool description
    // points at, so an agent fetches one concern without paying for all of
    // them in every session's context. It needs no key and no daemon.
    if action_of(&cli_args) == Some("help") {
        match cli_args.get(2).map(String::as_str) {
            None => print_help(),
            Some(topic) => match mini_swe_mcp::cli::help::topic_text(topic) {
                Some(text) => println!("{text}"),
                None => {
                    eprintln!("Unknown help topic: {topic}");
                    eprintln!("Topics: {}", mini_swe_mcp::cli::help::TOPICS.join(", "));
                    std::process::exit(2);
                }
            },
        }
        return Ok(());
    }

    if action_of(&cli_args) == Some("watch") {
        let code = mini_swe_mcp::cli::watch::run(&cli_args, json_output, admin).await?;
        std::process::exit(code);
    }

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
        return run_remote_action(action, &cli_args, json_output, admin, quiet).await;
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
        return run_action(
            &server,
            action,
            &cli_args,
            json_output,
            !api_key.is_empty(),
            admin,
            quiet,
        )
        .await;
    }

    run_local_stdio().await
}

/// Serve MCP over stdio in this process (escape hatch for `MINI_SWE_NO_DAEMON=1`).
async fn run_local_stdio() -> Result<()> {
    let api_key = env::var("OPENAI_API_KEY").unwrap_or_default();
    if api_key.is_empty() {
        anyhow::bail!(
            "Missing OPENAI_API_KEY. Please provide it via environment variable or .env file."
        );
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
/// ownership check, so the CLI can steer, kill, collect and watch a worker
/// another agent dispatched.
///
/// `quiet` is the operator's `--quiet`: the dispatch answer then prints the
/// worker ids alone instead of the human-facing view.
async fn run_remote_action(
    action: &str,
    cli_args: &[String],
    json_output: bool,
    admin: bool,
    quiet: bool,
) -> Result<()> {
    let api_key_present = !env::var("OPENAI_API_KEY").unwrap_or_default().is_empty();
    let Some(tool_args) = tool_args(action, cli_args, api_key_present)? else {
        return Ok(());
    };
    let mut client = mini_swe_mcp::hub::HubClient::connect_as_admin(admin).await?;
    let result = drive_worker_call(tool_args, async |args| client.worker(args).await).await?;
    print_result(action, &result, json_output, quiet)
}

/// Issue one worker call through either transport. Dispatch and steer detach.
async fn drive_worker_call(
    tool_args: serde_json::Map<String, serde_json::Value>,
    mut call: impl AsyncFnMut(serde_json::Value) -> Result<serde_json::Value>,
) -> Result<serde_json::Value> {
    call(serde_json::Value::Object(tool_args)).await
}

/// Run the hub daemon: the single process owning the only worker pool.
///
/// Foreground only; clients dial `hub.sock` and speak the same JSON-RPC the
/// stdio server speaks. No API key is required to start: the pool is built
/// exactly like the stdio path, and keyless dispatches fail lazily per call.
async fn run_daemon_cmd(server: &McpServer) -> Result<()> {
    let dir = mini_swe_mcp::hub::hub_dir()?;
    let running =
        mini_swe_mcp::hub::run_daemon(std::sync::Arc::new(server.clone()), dir, None).await?;
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
///
/// `quiet` is the operator's `--quiet`: the dispatch answer then prints the
/// worker ids alone instead of the human-facing view.
async fn run_action(
    server: &McpServer,
    action: &str,
    cli_args: &[String],
    json_output: bool,
    api_key_present: bool,
    admin: bool,
    quiet: bool,
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
    print_result(action, &result, json_output, quiet)
}

/// Print `result` as pretty JSON or through the action's plain-text renderer.
///
/// `quiet` changes only `dispatch`: the dumped worker id(s) replace the
/// human-facing view, so a script reads them without a JSON parser.
fn print_result(
    action: &str,
    result: &serde_json::Value,
    json_output: bool,
    quiet: bool,
) -> Result<()> {
    if json_output {
        println!("{}", serde_json::to_string_pretty(result)?);
    } else if quiet && action == "dispatch" {
        print_quiet_dispatch(result)?;
    } else {
        println!("{}", format_output(action, result));
    }
    Ok(())
}

/// `dispatch --quiet`: the started worker ids on stdout, one per line, and each
/// entry error on stderr. A batch that lost an entry fails after printing the
/// ids it did start, so a pipe never reads a partial batch as a clean success.
fn print_quiet_dispatch(result: &serde_json::Value) -> Result<()> {
    let view = format_dispatch_quiet(result);
    for id in &view.worker_ids {
        println!("{id}");
    }
    if !view.errors.is_empty() {
        anyhow::bail!("{}", view.errors.join("\n"));
    }
    Ok(())
}

fn print_help() {
    use mini_swe_mcp::cli::args::{CONSOLIDATE_USAGE, DISPATCH_USAGE};

    println!("mini-swe-mcp {}", env!("CARGO_PKG_VERSION"));
    println!("Usage: mini-swe-mcp [--stdio | [--json] [--admin] <action> [args...]]");
    println!("\nActions:");
    println!("  {DISPATCH_USAGE}");
    println!("           Start a worker on its own branch; always detaches.");
    println!(
        "  watch [<worker_id>...] [--group <g>] [--all] [--follow] [--json] [--timeout <secs>]"
    );
    println!(
        "           Block until the next worker event (replaying missed ones), print it and exit;"
    );
    println!("           run it in the background and the host CLI wakes you when it ends.");
    println!("  status <worker_id> | status --line");
    println!("           Final status/diff, or a one-line pool summary for statusLine.");
    println!("  collect <worker_id> [--full] [--file <path>]");
    println!("           Final message with a per-file diff stat; --full adds the whole diff,");
    println!("           --file narrows it to one path (repeatable).");
    println!("  review <worker_id> [--diff code|all|none]");
    println!("           One compact view of a finished worker: task, verification, the code");
    println!("           diff, per-file stat, tests summarised, and whether it still merges.");
    println!("  approve <worker_id> [\"note\"]");
    println!("           Record your verdict on a completed worker (owner-only).");
    println!("  unapprove <worker_id>");
    println!("           Withdraw that approval.");
    println!("  logs <worker_id>");
    println!("           Recent commands and their output.");
    println!("  steer <worker_id> <message> [--max-turns <n>]");
    println!("           Correct a completed worker or continue a stopped one.");
    println!("  consolidate {CONSOLIDATE_USAGE}");
    println!(
        "           Integrate one group's round: merge the finished branches, run the full gate"
    );
    println!("           once, route each failure to its owner, review every diff, and report.");
    println!("  list [--all]");
    println!("           Workers you own; --all (with --admin) lists every agent's.");
    println!("  kill <worker_id>");
    println!("           Terminate a worker.");
    println!("  discard <worker_id>");
    println!("           Drop a stopped worker for good: branch, row, history, steer files and");
    println!("           worktree leftovers, with no merge. A running worker is refused; kill it");
    println!("           first.");
    println!("  merge <worker_id> [--no-delete]");
    println!("           Merge a finished worker's branch into its base branch: trial merge,");
    println!("           verify gate on the merge result, then merge --no-ff and clean up.");
    println!("  merge --approved [--group <group>]");
    println!("           Land every approved worker of a group with ONE gate on the combined");
    println!("           result: a conflicting worker is skipped, the rest merge with --no-ff.");
    println!("  reap");
    println!("           Evict expired terminal worker records.");
    println!("  prune");
    println!("           Clean stale worktrees and caches.");
    println!("  manifest");
    println!("           Print the models catalog.");
    println!("  monitor [--once]");
    println!("           Full-screen view of the pool.");
    println!("  supervisor [--once]");
    println!("           Health view of workers and the hub.");
    println!("  daemon");
    println!("           Run the shared hub in the foreground.");
    println!("  whoami");
    println!("           Print this session's agent identity and how it was derived.");
    println!("  help <topic>");
    println!("           Long-form guidance on one concern (see Topics below).");
    println!("\nTopics:");
    println!(
        "  mini-swe-mcp help <topic>   {}",
        mini_swe_mcp::cli::help::TOPICS.join(", ")
    );
    println!("\nFlags:");
    println!("{}", mini_swe_mcp::cli::HELP_FLAGS);
}

/// Concurrent subagents supported out of the box.
fn max_concurrent_workers() -> usize {
    config::max_concurrent_workers()
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
