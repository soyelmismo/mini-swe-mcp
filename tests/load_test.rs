//! Load test: many agents of many workers through one hub daemon.
//!
//! `#[ignore]` by default so `cargo test` stays fast; run it with
//! `cargo test --test load_test -- --ignored --nocapture`. The shape is
//! configurable through the environment (`LOAD_AGENTS`,
//! `LOAD_WORKERS_PER_AGENT`, `LOAD_MAX_WORKERS`, `LOAD_MAX_HEAVY`,
//! `LOAD_HEAVY_SECS`, `LOAD_TIMEOUT_SECS`) and defaults to a small smoke run;
//! `LOAD_AGENTS=5 LOAD_WORKERS_PER_AGENT=20` is the full 5x20 load the hub is
//! sized for.
//!
//! What it measures, against one real daemon process serving one real pool:
//!
//! * every worker completes, dispatched by five distinct agent identities
//!   (`MINI_SWE_AGENT_ID` in the client's `hub/hello`),
//! * the daemon's resident set stays bounded (`VmHWM` sampled every 250 ms),
//! * heavy commands are dosed by the admission controller: the number in
//!   flight never exceeds `BASH_BUILD_LIMIT`,
//! * no agent finishes all of its workers before another agent has completed
//!   any of its own (the fair scheduler rotates the slots across owners).
//!
//! The LLM is [`common::fake_llm::FakeLlm`]: turn 1 runs a light command, turn
//! 2 runs a heavy one, turn 3 asks to finish.

mod common;

use common::TempDir;
use common::fake_llm::FakeLlm;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;

use serde_json::json;

/// How often the daemon's RSS and the heavy-command count are sampled.
const SAMPLE_INTERVAL: Duration = Duration::from_millis(250);

/// Generous ceiling on the daemon's peak resident set, in KiB.
///
/// The pool is designed to hold ~100 workers with bounded log buffers, so a
/// daemon that reaches 300 MB has leaked something; the bound is deliberately
/// loose so the test measures a regression rather than host noise.
const PEAK_RSS_LIMIT_KB: u64 = 300 * 1024;

/// The shape of one run, read from the environment.
struct LoadShape {
    /// Distinct agent identities dispatching workers.
    agents: usize,
    /// Workers each of those agents dispatches.
    per_agent: usize,
    /// `MAX_CONCURRENT_WORKERS`: worker slots the fair scheduler rotates.
    worker_slots: usize,
    /// `BASH_BUILD_LIMIT`: heavy commands admitted at once.
    max_heavy: usize,
    /// Seconds the heavy command burns CPU for.
    heavy_secs: u64,
}

impl LoadShape {
    fn from_env() -> Self {
        let cores = cores();
        Self {
            agents: env_count("LOAD_AGENTS", 2),
            per_agent: env_count("LOAD_WORKERS_PER_AGENT", 3),
            worker_slots: env_count("LOAD_MAX_WORKERS", cores * 2),
            max_heavy: env_count("LOAD_MAX_HEAVY", cores),
            heavy_secs: env_count("LOAD_HEAVY_SECS", 2) as u64,
        }
    }

    fn total(&self) -> usize {
        self.agents * self.per_agent
    }
}

/// A positive environment count, or `default` when unset, blank or unusable.
fn env_count(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|raw| raw.trim().parse().ok())
        .filter(|count| *count > 0)
        .unwrap_or(default)
}

/// Available CPU cores, for the report and the default limits.
fn cores() -> usize {
    std::thread::available_parallelism()
        .map(|cores| cores.get())
        .unwrap_or(2)
}

/// The daemon under test, with the pid the sampler watches.
struct Daemon {
    child: tokio::process::Child,
    pid: u32,
}

impl Daemon {
    async fn stop(&mut self) {
        let _ = self.child.kill().await;
        let _ = self.child.wait().await;
    }
}

/// One MCP connection to the hub, speaking newline-delimited JSON-RPC.
struct HubLink {
    reader: BufReader<tokio::net::unix::OwnedReadHalf>,
    writer: tokio::net::unix::OwnedWriteHalf,
    next_id: u64,
}

impl HubLink {
    /// Dial the hub socket and announce `agent` as this connection's identity.
    ///
    /// The agent id travels in `hub/hello` exactly as `MINI_SWE_AGENT_ID` does
    /// for the CLI, so the workers this link dispatches belong to `agent`.
    async fn connect(socket: &Path, agent: &str, admin: bool) -> Self {
        let stream = UnixStream::connect(socket)
            .await
            .unwrap_or_else(|e| panic!("could not connect to {}: {e}", socket.display()));
        let (reader, writer) = stream.into_split();
        let mut link = Self {
            reader: BufReader::new(reader),
            writer,
            next_id: 1,
        };
        let initialize = link
            .request(
                "initialize",
                json!({
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "load-test", "version": "1"}
                }),
            )
            .await;
        assert_eq!(initialize["protocolVersion"], "2024-11-05");
        link.request(
            "hub/hello",
            json!({
                "agent_id": agent,
                "pid": std::process::id(),
                "version": env!("CARGO_PKG_VERSION"),
                "admin": admin
            }),
        )
        .await;
        link
    }

    /// Send one request and return its decoded `result`.
    async fn request(&mut self, method: &str, params: serde_json::Value) -> serde_json::Value {
        let id = self.next_id;
        self.next_id += 1;
        let frame = json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params});
        self.writer
            .write_all(format!("{frame}\n").as_bytes())
            .await
            .expect("write frame");
        self.writer.flush().await.expect("flush frame");
        loop {
            let Some(line) = self.line_until(Instant::now() + Duration::from_secs(30)).await else {
                panic!("the hub closed the connection while answering {method}");
            };
            let reply: serde_json::Value =
                serde_json::from_str(line.trim()).expect("response is JSON");
            // Event notifications carry no `id`; the answer carries ours.
            if reply.get("id") == Some(&serde_json::json!(id)) {
                if let Some(error) = reply.get("error") {
                    panic!("{method} failed: {error}");
                }
                return reply["result"].clone();
            }
        }
    }

    /// Read one newline-terminated frame, or `None` at the deadline or on EOF.
    async fn line_until(&mut self, deadline: Instant) -> Option<String> {
        let mut line = String::new();
        match tokio::time::timeout_at(deadline.into(), self.reader.read_line(&mut line)).await {
            Ok(Ok(read)) if read > 0 => Some(line),
            _ => None,
        }
    }

    /// The next notification about a worker that finished.
    ///
    /// The hub pushes one `claude/channel` notification per lifecycle
    /// transition, so the order these arrive in is the order the workers
    /// finished in — which a poll of the (unordered) `list` payload cannot
    /// show.
    async fn next_finished(&mut self, deadline: Instant) -> Option<(String, String)> {
        while let Some(line) = self.line_until(deadline).await {
            let Ok(frame) = serde_json::from_str::<serde_json::Value>(line.trim()) else {
                continue;
            };
            if frame["method"].as_str() != Some("notifications/claude/channel") {
                continue;
            }
            let meta = &frame["params"]["meta"];
            let event = meta["event"].as_str().unwrap_or_default();
            if !matches!(event, "completed" | "failed") {
                continue;
            }
            let worker = meta["worker_id"].as_str()?;
            return Some((worker.to_string(), event.to_string()));
        }
        None
    }

    /// Dispatch one worker and return its id.
    async fn dispatch(&mut self, task: &str, repo: &Path, max_turns: usize) -> String {
        let result = self
            .request(
                "tools/call",
                json!({
                    "name": "worker",
                    "arguments": {
                        "action": "dispatch",
                        "task": task,
                        "repo_path": repo.to_string_lossy(),
                        "model": "ninja",
                        "max_turns": max_turns,
                        // The verify gate would re-run the build after the
                        // sentinel; the load test measures the worker loop.
                        "verify": ""
                    }
                }),
            )
            .await;
        let text = result["content"][0]["text"]
            .as_str()
            .expect("dispatch payload text");
        let payload: serde_json::Value = serde_json::from_str(text).expect("payload is JSON");
        payload["worker_id"]
            .as_str()
            .expect("dispatch returns a worker id")
            .to_string()
    }
}

/// What the sampler observed while the load ran.
#[derive(Default)]
struct Sample {
    /// Highest `VmRSS` seen, in KiB.
    rss_kb: u64,
    /// Highest `VmHWM` seen, in KiB: the daemon's peak resident set.
    peak_rss_kb: u64,
    /// Highest number of heavy commands in flight.
    peak_heavy: usize,
    /// Samples taken.
    ticks: usize,
}

/// Watch the daemon's resident set and the heavy commands in flight.
async fn sample_daemon(pid: u32, token: String, running: Arc<AtomicUsize>) -> Sample {
    let mut sample = Sample::default();
    while running.load(Ordering::Relaxed) > 0 {
        if let Some((rss, peak)) = rss_of(pid) {
            sample.rss_kb = sample.rss_kb.max(rss);
            sample.peak_rss_kb = sample.peak_rss_kb.max(peak);
        }
        sample.peak_heavy = sample.peak_heavy.max(heavy_in_flight(pid, &token));
        sample.ticks += 1;
        tokio::time::sleep(SAMPLE_INTERVAL).await;
    }
    sample
}

/// `VmRSS` and `VmHWM` of `pid`, in KiB.
fn rss_of(pid: u32) -> Option<(u64, u64)> {
    let status = std::fs::read_to_string(format!("/proc/{pid}/status")).ok()?;
    let field = |name: &str| {
        status
            .lines()
            .find(|line| line.starts_with(name))
            .and_then(|line| line.split_whitespace().nth(1))
            .and_then(|kb| kb.parse().ok())
    };
    Some((field("VmRSS:")?, field("VmHWM:")?))
}

/// Heavy commands the daemon is running right now.
///
/// Every bash step is a direct child of the daemon, and the heavy command is
/// the only one whose command line carries `token` (a per-run marker set as a
/// shell variable assignment in front of its `cargo build`). Counting the
/// daemon's own children is what keeps the number exact: the sandbox wrapper
/// and the shell it execs are nested below that child, and the `timeout` and
/// compiler processes the command spawns carry no marker at all.
fn heavy_in_flight(daemon_pid: u32, token: &str) -> usize {
    let Ok(entries) = std::fs::read_dir("/proc") else {
        return 0;
    };
    entries
        .flatten()
        .filter(|entry| {
            let name = entry.file_name();
            let name = name.to_str().unwrap_or_default();
            !name.is_empty()
                && name.bytes().all(|b| b.is_ascii_digit())
                && parent_of(&entry.path()).is_some_and(|parent| parent == daemon_pid)
        })
        .filter(|entry| {
            std::fs::read(entry.path().join("cmdline"))
                .map(|cmdline| String::from_utf8_lossy(&cmdline).contains(token))
                .unwrap_or(false)
        })
        .count()
}

/// The parent pid of a `/proc/<pid>` entry, from its `status` file.
fn parent_of(dir: &Path) -> Option<u32> {
    let status = std::fs::read_to_string(dir.join("status")).ok()?;
    status
        .lines()
        .find(|line| line.starts_with("PPid:"))
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|ppid| ppid.parse().ok())
}

/// A git repository holding one tiny crate, so every worker's worktree has a
/// real (if small) `cargo build` to run.
fn seed_repo(repo: &Path) {
    common::git(repo, &["init", "-b", "master"]);
    common::git(repo, &["config", "user.name", "load-test"]);
    common::git(repo, &["config", "user.email", "load-test@localhost"]);
    std::fs::create_dir_all(repo.join("src")).expect("create src");
    std::fs::write(
        repo.join("Cargo.toml"),
        "[package]\nname = \"load-probe\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    )
    .expect("write Cargo.toml");
    std::fs::write(
        repo.join("src/main.rs"),
        "fn main() {\n    println!(\"load probe\");\n}\n",
    )
    .expect("write main.rs");
    common::git(repo, &["add", "."]);
    common::git(repo, &["commit", "-m", "seed"]);
}

/// Start the real daemon on a scratch hub directory.
fn spawn_daemon(hub: &Path, swe: &Path, api_base: &str, shape: &LoadShape) -> Daemon {
    let stderr = hub.join("daemon.err");
    let file = std::fs::File::create(&stderr).expect("open the daemon log");
    let mut command = tokio::process::Command::new(common::binary_path());
    command
        .arg("daemon")
        .env("SWE_HUB_DIR", hub)
        .env("SWE_TEMP_DIR", swe)
        .env("TMPDIR", swe)
        .env("OPENAI_API_BASE", api_base)
        .env("OPENAI_API_KEY", "test-key-not-used-by-the-fake-llm")
        .env("ENV_FILE", hub.join("absent.env"))
        .env("MODELS_FILE", format!("{}/models.yaml", env!("CARGO_MANIFEST_DIR")))
        .env("HUB_IDLE_SECS", "600")
        .env("MAX_CONCURRENT_WORKERS", shape.worker_slots.to_string())
        .env("BASH_BUILD_LIMIT", shape.max_heavy.to_string())
        // Only the admission controller's decisions: the evidence that heavy
        // commands were queued rather than all started at once.
        .env("RUST_LOG", "mini_swe_mcp::pool::admission=debug")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::from(file))
        .kill_on_drop(true);
    let child = command.spawn().expect("spawn the hub daemon");
    let pid = child.id().expect("daemon pid");
    Daemon { child, pid }
}

/// Wait until the hub socket accepts a connection.
async fn wait_for_socket(socket: &Path) {
    for _ in 0..200 {
        if UnixStream::connect(socket).await.is_ok() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("hub socket {} never came up", socket.display());
}

/// Restrict a scratch directory to 0700, as the hub requires.
fn private_dir(dir: &Path) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700))
            .unwrap_or_else(|e| panic!("could not restrict {}: {e}", dir.display()));
    }
}

/// The heavy command: a real (if small) build, then a bounded CPU burn.
///
/// `token` rides in a shell variable assignment in front of the `cargo`, which
/// is what makes the command both heavy (`is_heavy_command` matches the `cargo`
/// word) and uniquely identifiable in `/proc`. The burn keeps the heavy window
/// long enough to observe whatever the admission controller does with it, even
/// where the sandbox denies cargo a writable cache.
fn heavy_command(token: &str, secs: u64) -> String {
    format!("LOAD_PROBE={token} cargo build --offline -q; timeout {secs} sh -c 'while :; do :; done'")
}

/// The load test itself: 5 agents x 20 workers through one hub daemon.
#[tokio::test]
#[ignore]
async fn hub_handles_five_agents_of_twenty_workers() {
    let shape = LoadShape::from_env();
    let total = shape.total();
    let root = TempDir::new_in_tmp("load");
    let hub = root.subdir("hub");
    let swe = root.subdir("swe");
    let repo = root.subdir("repo");
    private_dir(&hub);
    private_dir(&swe);
    seed_repo(&repo);

    let token = common::unique_suffix("heavy");
    let llm = FakeLlm::spawn("ls -la", &heavy_command(&token, shape.heavy_secs)).await;
    let mut daemon = spawn_daemon(&hub, &swe, llm.base_url(), &shape);
    let socket = hub.join("hub.sock");
    wait_for_socket(&socket).await;

    // The observer is connected before anything is dispatched, so no terminal
    // notification can arrive before somebody is listening for it.
    let mut observer = HubLink::connect(&socket, "load-test-observer", true).await;
    let sampling = Arc::new(AtomicUsize::new(1));
    let sampler = tokio::spawn(sample_daemon(daemon.pid, token.clone(), sampling.clone()));

    // One connection per agent identity, dispatching all of that agent's
    // workers: the shape a real orchestrator's clients produce.
    let started = Instant::now();
    let mut links = Vec::new();
    for agent in 1..=shape.agents {
        links.push((
            format!("agent-{agent}"),
            HubLink::connect(&socket, &format!("agent-{agent}"), false).await,
        ));
    }
    let mut dispatched: Vec<(String, String)> = Vec::with_capacity(total);
    for (agent, link) in links.iter_mut() {
        for index in 0..shape.per_agent {
            let wid = link
                .dispatch(&format!("load probe {index}"), &repo, 3)
                .await;
            dispatched.push((agent.clone(), wid));
        }
    }
    let dispatch_secs = started.elapsed().as_secs_f64();

    // Collect the workers in the order the daemon reports them finished.
    let mut order: Vec<(String, String)> = Vec::with_capacity(total);
    let mut statuses: HashMap<String, String> = HashMap::new();
    let deadline = started + Duration::from_secs(env_count("LOAD_TIMEOUT_SECS", 600) as u64);
    while order.len() < total {
        let Some((worker, event)) = observer.next_finished(deadline).await else {
            break;
        };
        if statuses.insert(worker.clone(), event.clone()).is_some() {
            continue;
        }
        let owner = dispatched
            .iter()
            .find(|(_, wid)| *wid == worker)
            .map(|(agent, _)| agent.clone())
            .unwrap_or_else(|| panic!("{worker} was never dispatched"));
        order.push((owner, worker));
    }
    let wall_secs = started.elapsed().as_secs_f64();

    sampling.store(0, Ordering::Relaxed);
    let sample = sampler.await.expect("sampler task joins");
    let queued = std::fs::read_to_string(hub.join("daemon.err"))
        .map(|log| log.matches("Heavy command waiting for admission").count())
        .unwrap_or(0);
    daemon.stop().await;

    report(
        &shape,
        &sample,
        wall_secs,
        dispatch_secs,
        &order,
        &statuses,
        &dispatched,
        llm.requests(),
        llm.heavy_commands(),
        queued,
    );

    // Every dispatched worker must have finished, and finished well.
    assert_eq!(
        order.len(),
        total,
        "only {} of {total} workers finished; unfinished: {}",
        order.len(),
        dispatched
            .iter()
            .filter(|(_, wid)| !statuses.contains_key(wid))
            .map(|(_, wid)| wid.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    );
    let failed: Vec<&String> = statuses
        .iter()
        .filter(|(_, status)| *status != "completed")
        .map(|(id, _)| id)
        .collect();
    assert!(
        failed.is_empty(),
        "every worker must complete; failed: {failed:?}"
    );

    // The daemon's resident set stays bounded.
    assert!(
        sample.peak_rss_kb < PEAK_RSS_LIMIT_KB,
        "daemon peaked at {} KiB RSS (VmHWM), above the {} KiB bound",
        sample.peak_rss_kb,
        PEAK_RSS_LIMIT_KB
    );

    // Heavy commands are dosed: never more in flight than the controller
    // allows, and at least one was actually observed running.
    assert_eq!(
        llm.heavy_commands(),
        total,
        "every worker must have been scripted its heavy command"
    );
    assert!(
        sample.peak_heavy >= 1,
        "the sampler never saw a heavy command in flight; the probe is broken"
    );
    assert!(
        sample.peak_heavy <= shape.max_heavy,
        "{} heavy commands ran at once, above BASH_BUILD_LIMIT={}",
        sample.peak_heavy,
        shape.max_heavy
    );

    // Fairness: no agent's last worker finishes before another agent's first.
    if shape.agents > 1 {
        for (agent, _) in &dispatched {
            let last = order
                .iter()
                .rposition(|(owner, _)| owner == agent)
                .unwrap_or_else(|| panic!("{agent} never finished a worker"));
            for (other, _) in &dispatched {
                if other == agent {
                    continue;
                }
                let first = order
                    .iter()
                    .position(|(owner, _)| owner == other)
                    .unwrap_or_else(|| panic!("{other} never finished a worker"));
                assert!(
                    first < last,
                    "{agent} finished all its workers before {other} completed any"
                );
            }
        }
    }
}

/// Print the one-screen report the run is judged on.
#[allow(clippy::too_many_arguments)]
fn report(
    shape: &LoadShape,
    sample: &Sample,
    wall_secs: f64,
    dispatch_secs: f64,
    order: &[(String, String)],
    statuses: &HashMap<String, String>,
    dispatched: &[(String, String)],
    requests: usize,
    heavy: usize,
    queued: usize,
) {
    let mut agents: Vec<&str> = dispatched.iter().map(|(agent, _)| agent.as_str()).collect();
    agents.dedup();
    println!(
        "\n=== hub load test: {} agents x {} workers ===",
        shape.agents, shape.per_agent
    );
    println!(
        "workers {total} | slots {slots} | max heavy {heavy_max} | cores {cores}",
        total = shape.total(),
        slots = shape.worker_slots,
        heavy_max = shape.max_heavy,
        cores = cores(),
    );
    println!(
        "wall {wall:.1}s (dispatch {dispatch:.1}s) | samples {ticks} @ {interval}ms",
        wall = wall_secs,
        dispatch = dispatch_secs,
        ticks = sample.ticks,
        interval = SAMPLE_INTERVAL.as_millis()
    );
    println!(
        "peak daemon RSS {rss:.1} MB (VmHWM {peak:.1} MB, bound {bound} MB)",
        rss = sample.rss_kb as f64 / 1024.0,
        peak = sample.peak_rss_kb as f64 / 1024.0,
        bound = PEAK_RSS_LIMIT_KB / 1024
    );
    println!(
        "peak heavy in flight {peak} (max {max}) | heavy served {served} | queued for admission {queued}",
        peak = sample.peak_heavy,
        max = shape.max_heavy,
        served = heavy
    );
    println!(
        "fake LLM requests {requests} | completed {done}/{total} | failed {failed}",
        done = statuses.values().filter(|status| *status == "completed").count(),
        total = shape.total(),
        failed = statuses
            .values()
            .filter(|status| *status != "completed")
            .count()
    );
    for agent in &agents {
        let done = order.iter().filter(|(owner, _)| owner == agent).count();
        println!("  {agent}: {done}/{} completed", shape.per_agent);
    }
    println!(
        "completion order: {}",
        order
            .iter()
            .map(|(agent, _)| agent.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    );
}
