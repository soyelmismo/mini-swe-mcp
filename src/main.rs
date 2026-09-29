use anyhow::Result;
use mini_swe_mcp::config::xdg_config_dir;
use mini_swe_mcp::manifest::{BUILTIN_DEFAULT_MODEL, ModelManifest};
use mini_swe_mcp::mcp::{McpServer, WORKER_ACTIONS};
use mini_swe_mcp::pool::WorkerPool;
use mini_swe_mcp::worktree;
use std::env;
use tracing_subscriber::{EnvFilter, fmt, prelude::*};

/// Tokio worker threads for the MCP server runtime.
///
/// Pinned explicitly so the server never degrades to a single worker on a
/// one-core deployment target: with one worker a long `execute_bash` wait would
/// block every progress notification and every other in-flight request (see
/// audit F8). Overridable at runtime with `MINI_SWE_WORKER_THREADS`.
const DEFAULT_WORKER_THREADS: usize = 4;

fn main() -> Result<()> {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(resolve_worker_threads())
        .enable_all()
        .build()?
        .block_on(async_main())
}

/// Resolve the Tokio worker thread count from the environment.
pub fn resolve_worker_threads() -> usize {
    env::var("MINI_SWE_WORKER_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|n| *n > 0)
        .unwrap_or(DEFAULT_WORKER_THREADS)
}

async fn async_main() -> Result<()> {
    let raw_args: Vec<String> = env::args().collect();

    // Early CLI flag handling without requiring API keys
    if raw_args.len() > 1 {
        match raw_args[1].as_str() {
            "--version" | "-V" => {
                println!("mini-swe-mcp {}", env!("CARGO_PKG_VERSION"));
                return Ok(());
            }
            "--help" | "-h" => {
                println!("mini-swe-mcp {}", env!("CARGO_PKG_VERSION"));
                println!("Usage: mini-swe-mcp [--stdio | [--json] <action> [args...]]");
                println!("\nActions:");
                println!("  dispatch <task> [--model <model>] [--review-after <model>] [--repo <repo>] [--wait] [--max-turns <n>] [--group <group>]");
                println!("  status <worker_id>");
                println!("  collect <worker_id>");
                println!("  logs <worker_id>");
                println!("  reap");
                println!("  steer <worker_id> <message>");
                println!("  list");
                println!("  monitor [--once]");
                println!("  supervisor [--once]");
                println!("  kill <worker_id>");
                println!("  manifest");
                println!("  prune");
                println!("\nFlags:");
                println!("      --json     Output in JSON format (default is formatted plain text)");
                println!("  -h, --help     Print help");
                println!("  -V, --version  Print version");
                return Ok(());
            }
            "monitor" | "supervisor" => {
                let once = raw_args.iter().any(|arg| arg == "--once");
                return mini_swe_mcp::monitor::run_monitor(once).await;
            }
            _ => {}
        }
    }

    let json_output = raw_args.iter().any(|arg| arg == "--json");
    let cli_args: Vec<String> = raw_args.into_iter().filter(|arg| arg != "--json").collect();

struct ShortFormatter;

impl<S, N> tracing_subscriber::fmt::FormatEvent<S, N> for ShortFormatter
where
    S: tracing::Subscriber + for<'a> tracing_subscriber::registry::LookupSpan<'a>,
    N: for<'a> tracing_subscriber::fmt::FormatFields<'a> + 'static,
{
    fn format_event(
        &self,
        ctx: &tracing_subscriber::fmt::FmtContext<'_, S, N>,
        mut writer: tracing_subscriber::fmt::format::Writer<'_>,
        event: &tracing::Event<'_>,
    ) -> std::fmt::Result {
        let meta = event.metadata();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let secs = now % 60;
        let mins = (now / 60) % 60;
        let hours = (now / 3600) % 24;

        let target = meta.target();
        let short_target = target.strip_prefix("mini_swe_mcp::").unwrap_or(target);

        let lvl = match *meta.level() {
            tracing::Level::ERROR => "\x1b[31mERRO\x1b[0m",
            tracing::Level::WARN => "\x1b[33mWARN\x1b[0m",
            tracing::Level::INFO => "\x1b[32mINFO\x1b[0m",
            tracing::Level::DEBUG => "\x1b[34mDEBG\x1b[0m",
            tracing::Level::TRACE => "\x1b[35mTRCE\x1b[0m",
        };

        write!(writer, "{:02}:{:02}:{:02} {} [{}] ", hours, mins, secs, lvl, short_target)?;
        ctx.field_format().format_fields(writer.by_ref(), event)?;
        writeln!(writer)
    }
}

    let stdio_mode = cli_args.iter().any(|arg| arg == "--stdio");

    /// Non-blocking stderr writer for tracing.
    struct AsyncStderrWriter {
        tx: std::sync::mpsc::Sender<String>,
    }

    impl std::io::Write for AsyncStderrWriter {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            let text = String::from_utf8_lossy(buf);
            self.tx
                .send(text.into_owned())
                .map_err(|_| std::io::Error::new(std::io::ErrorKind::BrokenPipe, "log sink gone"))?;
            Ok(buf.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for AsyncStderrWriter {
        type Writer = AsyncStderrWriter;

        fn make_writer(&'a self) -> Self::Writer {
            AsyncStderrWriter {
                tx: self.tx.clone(),
            }
        }
    }

    fn non_blocking_stderr() -> AsyncStderrWriter {
        let (tx, rx) = std::sync::mpsc::channel::<String>();
        std::thread::Builder::new()
            .name("telemetry-stderr".to_string())
            .spawn(move || {
                use std::io::Write as _;
                let mut stderr = std::io::stderr();
                while let Ok(line) = rx.recv() {
                    if let Err(e) = stderr.write_all(line.as_bytes()) {
                        let _ = e;
                        break;
                    }
                    let _ = stderr.flush();
                }
            })
            .ok();
        AsyncStderrWriter { tx }
    }

    let default_level = if stdio_mode {
        tracing::Level::WARN
    } else {
        tracing::Level::INFO
    };

    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::default().add_directive(default_level.into()));

    // Crucial: log to STDERR, because STDOUT is dedicated to MCP JSON-RPC protocol.
    // The writer is non-blocking so telemetry can never stall a runtime thread.
    let stderr_writer = non_blocking_stderr();
    tracing_subscriber::registry()
        .with(
            fmt::layer()
                .event_format(ShortFormatter)
                .with_writer(stderr_writer),
        )
        .with(env_filter)
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

    let api_key = env::var("OPENAI_API_KEY").unwrap_or_default();

    let manifest = ModelManifest::load();

    // A `default` that names no known alias is already dropped by
    // `ModelManifest::normalize`, so reaching the fallback here is deliberate.
    let default_model = env::var("DEFAULT_MODEL").unwrap_or_else(|_| {
        manifest
            .default
            .clone()
            .unwrap_or_else(|| BUILTIN_DEFAULT_MODEL.to_string())
    });

    let max_workers = env::var("MAX_CONCURRENT_WORKERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(64); // Supports up to 64 concurrent subagents out of the box

    let pool = WorkerPool::new(max_workers, api_base, api_key.clone());
    let server = McpServer::new(pool.clone(), default_model, manifest);

    if cli_args.len() > 1 && cli_args[1] != "--stdio" {
        let action = &cli_args[1];

        if action == "prune" {
            worktree::prune_stale_worktrees(&std::path::PathBuf::from("."));
            let res = serde_json::json!({
                "status": "ok",
                "message": "Stale worktrees and orphaned worker branches pruned"
            });
            if json_output {
                println!("{}", serde_json::to_string_pretty(&res)?);
            } else {
                println!("{}", format_prune(&res));
            }
            return Ok(());
        }

        let mut tool_args = serde_json::Map::new();
        tool_args.insert("action".into(), serde_json::Value::String(action.clone()));

        match action.as_str() {
            "dispatch" => {
                if api_key.is_empty() {
                    anyhow::bail!("Missing OPENAI_API_KEY. Please provide it via environment variable or .env file.");
                }
                if cli_args.len() < 3 {
                    eprintln!(
                        "Usage: mini-swe-mcp dispatch <task> [--model <model>] [--review-after <model>] [--repo <repo>] [--wait] [--max-turns <n>] [--group <group>]"
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
                        "--review-after" => {
                            if i + 1 < cli_args.len() {
                                tool_args.insert(
                                    "review_after".into(),
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
                        "--max-turns" | "-t" => {
                            if i + 1 < cli_args.len() {
                                if let Ok(turns) = cli_args[i + 1].parse::<u64>() {
                                    tool_args.insert(
                                        "max_turns".into(),
                                        serde_json::Value::Number(turns.into()),
                                    );
                                }
                                i += 1;
                            }
                        }
                        "--group" | "-g" if i + 1 < cli_args.len() => {
                            tool_args.insert(
                                "group".into(),
                                serde_json::Value::String(cli_args[i + 1].clone()),
                            );
                            i += 1;
                        }
                        _ => {}
                    }
                    i += 1;
                }
            }
            "status" | "collect" | "logs" | "kill" => {
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
            "monitor" | "supervisor" => {
                let once = cli_args.iter().any(|arg| arg == "--once");
                return mini_swe_mcp::monitor::run_monitor(once).await;
            }
            "manifest" | "list" | "reap" => {}
            _ => {
                let actions = available_actions();
                if let Some(suggestion) = suggest_action(action, &actions) {
                    eprintln!(
                        "Unknown action: {action}. Did you mean '{suggestion}'?\nAvailable: {}",
                        actions.join(", ")
                    );
                } else {
                    eprintln!(
                        "Unknown action: {action}. Available: {}",
                        actions.join(", ")
                    );
                }
                std::process::exit(1);
            }
        }

        let mut result = server
            .execute_tool("worker", serde_json::Value::Object(tool_args.clone()))
            .await?;

        // Interactive steering: whenever the shared wait loop reports the worker
        // paused for input, prompt the operator and resume. Re-waiting goes through
        // the exact same helper the MCP stdio dispatch path uses, so both callers
        // share one polling/termination algorithm.
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

        if json_output {
            println!("{}", serde_json::to_string_pretty(&result)?);
        } else {
            println!("{}", format_output(action, &result));
        }
        return Ok(());
    }

    if api_key.is_empty() {
        anyhow::bail!("Missing OPENAI_API_KEY. Please provide it via environment variable or .env file.");
    }

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

/// CLI-only verbs, i.e. actions the binary handles directly instead of
/// dispatching through the `worker` tool.
const CLI_ONLY_ACTIONS: &[&str] = &["monitor", "supervisor"];

/// Everything the CLI accepts: the tool's own actions (single-sourced from the
/// MCP server) plus the CLI-only verbs.
fn available_actions() -> Vec<&'static str> {
    WORKER_ACTIONS
        .iter()
        .copied()
        .chain(CLI_ONLY_ACTIONS.iter().copied())
        .collect()
}

fn levenshtein(a: &str, b: &str) -> usize {
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut curr = vec![0; b.len() + 1];

    for (i, ca) in a.chars().enumerate() {
        curr[0] = i + 1;
        for (j, cb) in b.chars().enumerate() {
            let cost = if ca == cb { 0 } else { 1 };
            curr[j + 1] = (prev[j + 1] + 1)
                .min(curr[j] + 1)
                .min(prev[j] + cost);
        }
        prev.clone_from_slice(&curr);
    }
    prev[b.len()]
}

fn suggest_action<'a>(unknown: &str, candidates: &[&'a str]) -> Option<&'a str> {
    let unknown_lower = unknown.to_lowercase();
    // 1. Prefix match (min len 3 to avoid false positives)
    if unknown_lower.len() >= 3
        && let Some(&m) = candidates.iter().find(|&&c| c.starts_with(&unknown_lower))
    {
        return Some(m);
    }
    // 2. Levenshtein edit distance <= 2
    candidates
        .iter()
        .map(|&c| (c, levenshtein(&unknown_lower, c)))
        .filter(|&(_, dist)| dist <= 2)
        .min_by_key(|&(_, dist)| dist)
        .map(|(c, _)| c)
}

fn format_manifest(val: &serde_json::Value) -> String {
    let mut out = String::new();
    if let Some(default_model) = val.get("default_model").and_then(|v| v.as_str()) {
        out.push_str(&format!("Default model: {default_model}\n\n"));
    }
    out.push_str("Models:\n");
    if let Some(models) = val.get("models").and_then(|v| v.as_object()) {
        let mut entries: Vec<(&String, &serde_json::Value)> = models.iter().collect();
        entries.sort_by_key(|(k, _)| (*k).clone());

        for (name, def) in entries {
            let id = def.get("id").and_then(|v| v.as_str()).unwrap_or(name);
            let mut meta = Vec::new();
            meta.push(format!("id: {id}"));
            if let Some(temp) = def.get("temperature").and_then(|v| v.as_f64()) {
                let temp_str = format!("{temp:.2}");
                let temp_clean = temp_str.trim_end_matches('0').trim_end_matches('.');
                meta.push(format!("temp: {temp_clean}"));
            }
            if let Some(turns) = def.get("max_turns").and_then(|v| v.as_u64()) {
                meta.push(format!("max turns: {turns}"));
            }
            out.push_str(&format!("  - {} ({})\n", name, meta.join(", ")));
            if let Some(role) = def.get("role").and_then(|v| v.as_str()) {
                out.push_str(&format!("    Role: {role}\n"));
            }
        }
    }
    out.trim_end().to_string()
}

fn format_list(val: &serde_json::Value) -> String {
    let empty_vec = Vec::new();
    let workers = val
        .get("workers")
        .and_then(|v| v.as_array())
        .unwrap_or(&empty_vec);

    if workers.is_empty() {
        return "No active or recent workers found.".to_string();
    }

    let mut out = format!("Workers ({}):\n", workers.len());
    for w in workers {
        let id = w.get("id").and_then(|v| v.as_str()).unwrap_or("unknown");
        let model = w.get("model").and_then(|v| v.as_str()).unwrap_or("");
        let group = w.get("group").and_then(|v| v.as_str()).unwrap_or("default");
        let state_obj = w.get("state");
        let status = state_obj
            .and_then(|s| s.get("status"))
            .and_then(|v| v.as_str())
            .unwrap_or("Unknown");

        let mut details = Vec::new();
        if group != "default" {
            details.push(format!("group: {group}"));
        }
        if let Some(pid) = state_obj.and_then(|s| s.get("pid")).and_then(|v| v.as_u64()) {
            details.push(format!("pid: {pid}"));
        }
        if !model.is_empty() {
            details.push(format!("model: {model}"));
        }
        if let Some(turns) = state_obj.and_then(|s| s.get("turns")).and_then(|v| v.as_u64()) {
            details.push(format!("turns: {turns}"));
        } else if let Some(step) = state_obj.and_then(|s| s.get("step")).and_then(|v| v.as_u64()) {
            details.push(format!("step: {step}"));
        }
        if let Some(op) = state_obj.and_then(|s| s.get("last_command")).and_then(|v| v.as_str())
            && !op.is_empty() && op != "initializing"
        {
            details.push(format!("op: {op}"));
        }
        if let Some(err) = state_obj.and_then(|s| s.get("error")).and_then(|v| v.as_str()) {
            details.push(format!("error: {err}"));
        }

        let detail_str = if details.is_empty() {
            String::new()
        } else {
            format!(" ({})", details.join(", "))
        };

        out.push_str(&format!("  - {id} [{status}]{detail_str}\n"));
        if let Some(task) = w.get("task").and_then(|v| v.as_str()) {
            let task_preview = if task.len() > 60 {
                let cut = task.floor_char_boundary(57);
                format!("{}...", &task[..cut])
            } else {
                task.to_string()
            };
            out.push_str(&format!("    Task: {task_preview}\n"));
        }
    }
    out.trim_end().to_string()
}

fn format_prune(val: &serde_json::Value) -> String {
    let msg = val
        .get("message")
        .and_then(|v| v.as_str())
        .unwrap_or("Stale worktrees and orphaned worker branches pruned");
    format!("✓ {msg}.")
}

fn format_status(val: &serde_json::Value) -> String {
    let wid = val.get("worker_id").and_then(|v| v.as_str()).unwrap_or("");
    let mut out = format!("Worker: {wid}\n");
    if let Some(state) = val.get("state") {
        if let Some(status_str) = state.as_str() {
            out.push_str(&format!("State: {status_str}\n"));
        } else {
            let tag = state.get("state").and_then(|v| v.as_str());
            let details = state.get("details").and_then(|v| v.as_object());

            if let Some(t) = tag {
                out.push_str(&format!("State: {t}\n"));
                if let Some(d) = details {
                    if let Some(turns) = d.get("turns").and_then(|v| v.as_u64()) {
                        out.push_str(&format!("Turns: {turns}\n"));
                    }
                    if let Some(step) = d.get("step").and_then(|v| v.as_u64()) {
                        out.push_str(&format!("Step: {step}\n"));
                    }
                    if let Some(summary) = d.get("summary").and_then(|v| v.as_str()) {
                        out.push_str(&format!("Summary: {summary}\n"));
                    }
                    if let Some(err) = d.get("error").and_then(|v| v.as_str()) {
                        out.push_str(&format!("Error: {err}\n"));
                    }
                    if let Some(q) = d.get("question").and_then(|v| v.as_str()) {
                        out.push_str(&format!("Question: {q}\n"));
                    }
                }
            } else if let Some(obj) = state.as_object() {
                for (state_name, d) in obj {
                    out.push_str(&format!("State: {state_name}\n"));
                    if let Some(turns) = d.get("turns").and_then(|v| v.as_u64()) {
                        out.push_str(&format!("Turns: {turns}\n"));
                    }
                    if let Some(step) = d.get("step").and_then(|v| v.as_u64()) {
                        out.push_str(&format!("Step: {step}\n"));
                    }
                    if let Some(summary) = d.get("summary").and_then(|v| v.as_str()) {
                        out.push_str(&format!("Summary: {summary}\n"));
                    }
                    if let Some(err) = d.get("error").and_then(|v| v.as_str()) {
                        out.push_str(&format!("Error: {err}\n"));
                    }
                    if let Some(q) = d.get("question").and_then(|v| v.as_str()) {
                        out.push_str(&format!("Question: {q}\n"));
                    }
                }
            }
        }
    }
    out.trim_end().to_string()
}

fn format_collect(val: &serde_json::Value) -> String {
    let wid = val.get("worker_id").and_then(|v| v.as_str()).unwrap_or("");
    let diff = val
        .get("state")
        .and_then(|s| s.get("details").or_else(|| s.get("Completed")))
        .and_then(|c| c.get("diff"))
        .or_else(|| val.get("diff"))
        .and_then(|v| v.as_str())
        .unwrap_or("");

    let counters = log_counters_line(val);
    if diff.trim().is_empty() {
        format!("Worker {wid}: No git diff produced.\n{counters}")
    } else {
        format!("{diff}\n{counters}")
    }
}

/// Render the `logs` action: the bounded window plus the counters that make any
/// truncation visible instead of silent (audit 07, R7).
fn format_logs(val: &serde_json::Value) -> String {
    let wid = val.get("worker_id").and_then(|v| v.as_str()).unwrap_or("");
    let mut out = format!("Worker {wid} step logs\n");
    if let Some(entries) = val.get("logs").and_then(|v| v.as_array()) {
        for entry in entries {
            let step = entry.get("step").and_then(|v| v.as_u64()).unwrap_or(0);
            let command = entry.get("command").and_then(|v| v.as_str()).unwrap_or("");
            out.push_str(&format!("  [{step}] {command}\n"));
        }
    }
    out.push_str(&format!(
        "{}
",
        log_counters_line(val)
    ));
    out.trim_end().to_string()
}

fn format_reap(val: &serde_json::Value) -> String {
    let reaped = val.get("reaped").and_then(|v| v.as_u64()).unwrap_or(0);
    let ids = val
        .get("worker_ids")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        })
        .unwrap_or_default();
    if reaped == 0 {
        "✓ No expired terminal worker records to reap.".to_string()
    } else {
        format!("✓ Reaped {reaped} expired worker record(s): {ids}")
    }
}

/// One-line summary of the step-log counters present in `val`.
fn log_counters_line(val: &serde_json::Value) -> String {
    let read = |key: &str| val.get(key).and_then(|v| v.as_u64());
    let total = read("total_steps");
    let emitted = val
        .get("logs")
        .and_then(|v| v.as_array())
        .map(|a| a.len() as u64);
    let retained = read("logs_retained")
        .or_else(|| emitted.map(|n| n.saturating_add(read("logs_omitted").unwrap_or(0))));
    let omitted = read("logs_omitted");
    let dropped = read("logs_dropped");
    let mut parts: Vec<String> = Vec::new();
    if let Some(total) = total {
        parts.push(format!("total_steps: {total}"));
    }
    if let Some(retained) = retained {
        parts.push(format!("retained: {retained}"));
    }
    if let Some(omitted) = omitted
        && omitted > 0
    {
        parts.push(format!("omitted: {omitted}"));
    }
    if let Some(dropped) = dropped
        && dropped > 0
    {
        parts.push(format!("dropped: {dropped}"));
    }
    if let Some(notice) = val.get("logs_truncation_notice").and_then(|v| v.as_str()) {
        parts.push(notice.to_string());
    }
    if parts.is_empty() {
        return "no step logs".to_string();
    }
    parts.join(" | ")
}
fn format_dispatch(val: &serde_json::Value) -> String {
    let wid = val.get("worker_id").and_then(|v| v.as_str()).unwrap_or("");
    if val.get("status").and_then(|v| v.as_str()) == Some("dispatched") {
        format!("✓ Worker {wid} dispatched in background.\nUse 'mini-swe-mcp status {wid}' to check progress.")
    } else {
        let mut out = format!("✓ Worker {wid} finished.\n");
        if let Some(state) = val.get("state") {
            let state_name = state.get("state").and_then(|v| v.as_str()).unwrap_or("");
            let details = state
                .get("details")
                .or_else(|| state.get("Completed"))
                .or_else(|| state.get("Failed"));

            if state_name == "Completed" || state.get("Completed").is_some() {
                if let Some(turns) = details.and_then(|d| d.get("turns")).and_then(|v| v.as_u64()) {
                    out.push_str(&format!("Turns: {turns}\n"));
                }
                if let Some(summary) = details.and_then(|d| d.get("summary")).and_then(|v| v.as_str()) {
                    out.push_str(&format!("Summary: {summary}\n"));
                }
                if let Some(branch) = details.and_then(|d| d.get("branch")).and_then(|v| v.as_str()) {
                    out.push_str(&format!("Branch: {branch}\n"));
                }
                if let Some(artifacts) = details.and_then(|d| d.get("artifacts")).and_then(|v| v.as_array())
                    && !artifacts.is_empty()
                {
                    let list: Vec<&str> = artifacts.iter().filter_map(|a| a.as_str()).collect();
                    out.push_str(&format!("Preserved Artifacts: {}\n", list.join(", ")));
                }
                if let Some(diff) = details.and_then(|d| d.get("diff")).and_then(|v| v.as_str())
                    && !diff.trim().is_empty()
                {
                    out.push_str(&format!("\nDiff:\n{diff}\n"));
                }
            } else if state_name == "Failed" || state.get("Failed").is_some() {
                out.push_str("State: Failed\n");
                if let Some(err) = details.and_then(|d| d.get("error")).and_then(|v| v.as_str()) {
                    out.push_str(&format!("Error: {err}\n"));
                }
            }
        }
        out.trim_end().to_string()
    }
}

fn format_steer(val: &serde_json::Value) -> String {
    let wid = val.get("worker_id").and_then(|v| v.as_str()).unwrap_or("");
    let msg = val
        .get("message")
        .and_then(|v| v.as_str())
        .unwrap_or("Steering instruction queued");
    format!("✓ Worker {wid}: {msg}")
}

fn format_kill(val: &serde_json::Value) -> String {
    let wid = val.get("worker_id").and_then(|v| v.as_str()).unwrap_or("");
    let killed = val.get("killed").and_then(|v| v.as_bool()).unwrap_or(false);
    if killed {
        format!("✓ Worker {wid} terminated.")
    } else {
        format!("Worker {wid} was not running.")
    }
}

fn format_output(action: &str, val: &serde_json::Value) -> String {
    match action {
        "manifest" => format_manifest(val),
        "list" => format_list(val),
        "prune" => format_prune(val),
        "status" => format_status(val),
        "collect" => format_collect(val),
        "logs" => format_logs(val),
        "reap" => format_reap(val),
        "dispatch" => format_dispatch(val),
        "steer" => format_steer(val),
        "kill" => format_kill(val),
        _ => serde_json::to_string_pretty(val).unwrap_or_default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_worker_threads_defaults_to_the_pinned_count() {
        const { assert!(DEFAULT_WORKER_THREADS > 1) };
        if std::env::var("MINI_SWE_WORKER_THREADS").is_err() {
            assert_eq!(resolve_worker_threads(), DEFAULT_WORKER_THREADS);
        }
    }

    #[test]
    fn test_resolve_worker_threads_ignores_nonsense_values() {
        let saved = std::env::var("MINI_SWE_WORKER_THREADS").ok();
        unsafe { std::env::set_var("MINI_SWE_WORKER_THREADS", "0") };
        assert_eq!(resolve_worker_threads(), DEFAULT_WORKER_THREADS);
        unsafe { std::env::set_var("MINI_SWE_WORKER_THREADS", "-3") };
        assert_eq!(resolve_worker_threads(), DEFAULT_WORKER_THREADS);
        unsafe { std::env::set_var("MINI_SWE_WORKER_THREADS", "not-a-number") };
        assert_eq!(resolve_worker_threads(), DEFAULT_WORKER_THREADS);

        match saved {
            Some(v) => unsafe { std::env::set_var("MINI_SWE_WORKER_THREADS", v) },
            None => unsafe { std::env::remove_var("MINI_SWE_WORKER_THREADS") },
        }
    }
}
