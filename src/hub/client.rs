//! Thin transports to the shared hub, with bounded startup retries.
//!
//! Reconnects live here, not in the callers: a daemon that goes away under a
//! long-lived client is one event, and [`reconnect_following`] is the single
//! place that turns it into "dial again, say `hub/hello` with the same identity,
//! resume" while [`daemon_went_away`] is the single test for "the daemon is
//! gone" that every transport error is read through.

use anyhow::{Context, Result};
use serde_json::{Value, json};
use std::future::Future;
use std::os::unix::fs::OpenOptionsExt;
use std::os::unix::process::CommandExt;
use std::process::Stdio;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWrite, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;

use super::daemon::hub_lock_held;
use super::exe_path;
use super::identity;
use super::{HubPaths, hub_dir};

/// Dial the hub, starting a detached daemon if none is listening.
/// Racing starters are serialized by the daemon's exclusive flock.
///
/// The daemon is this executable, through [`exe_path::executable`]: a `cargo
/// build` under the client's feet leaves `current_exe()` marked ` (deleted)`,
/// and starting that string is the `No such file or directory` the client
/// would otherwise report.
pub async fn connect_or_spawn() -> Result<UnixStream> {
    let paths = HubPaths::new(hub_dir()?);
    if let Ok(stream) = super::daemon::connect_endpoint(&paths.probe_endpoint()).await {
        return Ok(stream);
    }
    let exe = exe_path::executable()?;
    spawn_daemon(&paths, &exe)?;

    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    let mut delay = Duration::from_millis(20);
    loop {
        match super::daemon::connect_endpoint(&paths.probe_endpoint()).await {
            Ok(stream) => return Ok(stream),
            Err(error) if tokio::time::Instant::now() >= deadline => {
                // The daemon is not there: reported with the refusal that proved
                // it, so a client following it reads this as a daemon that has
                // yet to come up rather than as its own failure.
                return Err(error).context("Hub did not start; inspect hub.log");
            }
            Err(_) => {}
        }
        tokio::time::sleep_until((tokio::time::Instant::now() + delay).min(deadline)).await;
        delay = (delay * 2).min(Duration::from_millis(250));
    }
}

/// Start a detached daemon; both auto-start and handover use this path.
///
/// `exe` is resolved by the caller through [`exe_path::executable`] (or the
/// same helper on a path the daemon is already known to run), so no caller has
/// to remember that a replaced binary is reported as ` (deleted)`.
pub(crate) fn spawn_daemon(paths: &HubPaths, exe: &std::path::Path) -> Result<()> {
    let log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW)
        .open(paths.log())
        .context("Could not open hub log")?;
    let mut command = std::process::Command::new(exe);
    command
        .arg("daemon")
        .env("SWE_HUB_DIR", paths.dir())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(log);
    // SAFETY: setsid is async-signal-safe and touches no Rust state after fork.
    unsafe {
        command.pre_exec(|| {
            if libc::setsid() == -1 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    let mut child = tokio::process::Command::from(command)
        .spawn()
        .context("Could not start hub daemon")?;
    tokio::spawn(async move {
        let _ = child.wait().await;
    });

    Ok(())
}

/// Whether `error` means the daemon went away rather than refused the request.
///
/// Every way a hub socket can die reads as one answer, because the client's
/// answer is the same for all of them: follow the daemon. A handover — or any
/// daemon restart — closes the connection underneath a long-lived watch, and
/// the abrupt version of that cut surfaces as a reset or a broken pipe instead
/// of an orderly EOF; while the replacement is still starting, the dial fails
/// with a refused connection or a socket that is not there yet. Matching the
/// `ErrorKind`s covers all of them; the messages below cover the two the kinds
/// cannot — the clean EOF the client reports as its own line, and the auto-start
/// that never came up.
pub fn daemon_went_away(error: &anyhow::Error) -> bool {
    // `chain` starts at the error itself (a bare `io::Error` has no source, so
    // walking `source` alone would never reach it) and ends at the root cause.
    for cause in error.chain() {
        if is_gone_away_kind(cause) {
            return true;
        }
    }
    // `anyhow::Error` reports only the outermost context, which need not be
    // the transport error itself, so read the whole rendered chain too.
    let chain = error_chain(error);
    chain.contains("Hub closed the connection")
        || chain.contains("Connection reset by peer")
        || chain.contains("Broken pipe")
        || chain.contains("BrokenPipe")
        || chain.contains("Connection refused")
        || chain.contains("ConnectionRefused")
        || chain.contains("No such file or directory")
        || chain.contains("NotFound")
        || chain.contains("Did not start")
}

/// The I/O failure kinds that mean "the daemon is no longer on this socket".
fn is_gone_away_kind(cause: &(dyn std::error::Error + 'static)) -> bool {
    let io = cause.downcast_ref::<std::io::Error>();
    matches!(
        io.map(std::io::Error::kind),
        Some(
            std::io::ErrorKind::ConnectionReset
                | std::io::ErrorKind::BrokenPipe
                | std::io::ErrorKind::ConnectionRefused
                | std::io::ErrorKind::NotFound
        )
    ) || cause.to_string().contains("Hub closed the connection")
}

/// Env var overriding how long a client follows a daemon that keeps going away.
pub const RECONNECT_DEADLINE_ENV: &str = "MINI_SWE_RECONNECT_SECS";

/// Default reconnect budget: long enough to outlast a handover and the
/// recovery the replacement runs, short enough that a hub that never comes back
/// ends the watch with an explanation instead of hanging on it.
pub const DEFAULT_RECONNECT_SECS: u64 = 60;

/// Bounds on the reconnect budget, so the environment cannot make a client
/// give up instantly or wait forever.
const MIN_RECONNECT_SECS: u64 = 1;
const MAX_RECONNECT_SECS: u64 = 24 * 60 * 60;

/// First pause between dials while the replacement starts.
const RECONNECT_BACKOFF_START: Duration = Duration::from_millis(50);

/// Ceiling for the exponential backoff between dials.
const RECONNECT_BACKOFF_MAX: Duration = Duration::from_millis(500);

/// How long a client follows a daemon that goes away, from the environment or
/// [`DEFAULT_RECONNECT_SECS`].
///
/// It bounds the whole chase, not the number of attempts: a handover costs a
/// handful of fast failures (no socket yet, lock still held, recovery running),
/// and a busy daemon whose replacement keeps failing costs seconds. The budget
/// starts on the first failure and is never extended, so a watch cannot hold a
/// decision open forever.
pub fn reconnect_deadline() -> Duration {
    Duration::from_secs(reconnect_secs(None))
}

/// The budget in seconds: the environment override, else the default, clamped.
fn reconnect_secs(requested: Option<u64>) -> u64 {
    requested
        .or_else(|| crate::config::env_parse(RECONNECT_DEADLINE_ENV))
        .unwrap_or(DEFAULT_RECONNECT_SECS)
        .clamp(MIN_RECONNECT_SECS, MAX_RECONNECT_SECS)
}

/// The failure that ends the chase for good, once the budget is spent.
fn give_up(waited: Duration, last: &anyhow::Error) -> anyhow::Error {
    anyhow::anyhow!(
        "The hub went away and did not come back within {}s ({}); \
         start the hub again and re-run the command",
        waited.as_secs(),
        error_chain(last)
    )
}

/// Every frame of an error, so a reported cause is never the outermost context
/// alone.
fn error_chain(error: &anyhow::Error) -> String {
    error
        .chain()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(": ")
}

/// Re-announce this identity on a new connection and return the replacement.
///
/// The caller marks its next snapshot `initial`, so whatever the old daemon
/// never delivered is asked for again. `with` re-sends the hello through the
/// caller's own connection path, so the stdio proxy keeps the identity its MCP
/// client established while the CLI hands back a fresh [`HubClient`].
pub async fn reconnect_following<R>(deadline: Duration, with: R) -> Result<HubClient>
where
    R: Reconnect<Output = HubClient>,
{
    follow_until(deadline, with).await
}

/// One dial-and-greet attempt, in the shape every hub transport reconnects with:
/// a closure returning a future, so the chase never knows what it is dialling.
///
/// Implemented by [`reconnect_following`]'s callers and, in tests, by any
/// stand-in for a dial.
pub trait Reconnect {
    type Output;
    /// One attempt: the connection it made, or why there is none.
    fn attempt(&mut self) -> impl Future<Output = Result<Self::Output>>;
}

/// The chase itself: dial, and if the daemon is simply gone, dial again after a
/// backoff until `deadline`.
#[cfg_attr(not(test), allow(dead_code))]
async fn follow_until<R: Reconnect>(deadline: Duration, mut with: R) -> Result<R::Output> {
    let started = tokio::time::Instant::now();
    let mut delay = RECONNECT_BACKOFF_START;
    loop {
        match with.attempt().await {
            Ok(client) => return Ok(client),
            // A fault that is not "the daemon is gone" is the caller's own
            // error (a refused identity, a malformed reply): report it now
            // rather than dialling again until the deadline.
            Err(error) if !daemon_went_away(&error) => return Err(error),
            Err(error) => {
                let waited = started.elapsed();
                anyhow::ensure!(waited < deadline, "{}", give_up(waited, &error));
            }
        }
        tokio::time::sleep(delay.min(deadline.saturating_sub(started.elapsed()))).await;
        delay = (delay * 2).min(RECONNECT_BACKOFF_MAX);
    }
}

/// Any closure that produces one connection attempt is a [`Reconnect`].
impl<F, Fut, T> Reconnect for F
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T>>,
{
    type Output = T;

    fn attempt(&mut self) -> impl Future<Output = Result<Self::Output>> {
        self()
    }
}

/// Announce this process to the daemon.
///
/// `agent_id` is the operator's `MINI_SWE_AGENT_ID`, the one way a client can
/// name the agent its workers belong to; without it the daemon uses the
/// `host_id` this client walked up to — the agent's host process, shared by its
/// MCP connection and its shell commands (see [`crate::hub::identity`]) — and
/// only then the `initialize` `clientInfo` (see
/// [`crate::mcp::ConnectionContext::agent`]). `admin` is the operator override
/// that lifts the per-agent ownership check.
fn client_version() -> String {
    #[cfg(debug_assertions)]
    if let Ok(version) = std::env::var("MINI_SWE_FAKE_VERSION") {
        return version;
    }
    env!("CARGO_PKG_VERSION").to_string()
}

/// This build's identity: `id` matches only the exact binary, `ts` is the
/// comparable build clock, both stamped by `build.rs`.
///
/// `MINI_SWE_FAKE_BUILD_TS` is the test seam, as `MINI_SWE_FAKE_VERSION` is for
/// the release, so a test can present a client built before or after the hub.
fn client_build() -> Value {
    #[cfg(debug_assertions)]
    if let Some(ts) = std::env::var("MINI_SWE_FAKE_BUILD_TS")
        .ok()
        .and_then(|ts| ts.parse::<u64>().ok())
    {
        return json!({"id": format!("test-{ts:016x}"), "ts": ts});
    }
    build()
}

/// The build identity of this binary, in the shape the handshake carries it and
/// the daemon answers in its own `hub/hello` reply.
fn build() -> Value {
    json!({"id": env!("MINI_SWE_BUILD_ID"), "ts": build_ts()})
}

/// The build's clock in unix nanoseconds; it only fails to parse if `build.rs`
/// did not run, which cannot leave a compiled binary behind.
fn build_ts() -> u64 {
    env!("MINI_SWE_BUILD_TS").parse().unwrap_or_default()
}

/// The `hub/hello` parameters: who this client is, and which agent it speaks
/// for.
///
/// `host_id` is computed here, on the client, because only the client can see
/// its own ancestry: the daemon would otherwise have to trust a pid it cannot
/// walk. It is a coordination identity, never an authenticated one.
///
/// `session_id` and `watch_token` are read from the environment for the same
/// reason: only the client can see them. The session is the one the host named
/// ([`identity::SESSION_ENV_VARS`]); the token is what a shell that cannot know
/// its session presents to act as the session that dispatched a worker. The
/// daemon combines the host with the session, so one shared connection can
/// still carry several sessions — each MCP call overrides the session through
/// its own `_meta.sessionID`.
fn hello_params(admin: bool, version: &str, build: &Value) -> Value {
    json!({"agent_id": std::env::var("MINI_SWE_AGENT_ID").ok(),
           "host_id": identity::host_identity().map(|host| host.to_string()),
           "session_id": identity::session_from_env(),
           "watch_token": std::env::var(identity::WATCH_TOKEN_ENV).ok(),
           "pid": std::process::id(), "version": version, "build": build,
           "cwd": std::env::current_dir().ok(), "admin": admin,
           "ambient_env": ambient_env_frame()})
}

/// The caller's filtered environment for the `hub/hello` handshake.
///
/// Built with the same secret filter the sandbox applies to children
/// ([`crate::agent::env::is_secret_name`]), so a key or token never crosses
/// the wire; bounded to [`crate::agent::env::AMBIENT_ENV_MAX_BYTES`] whole
/// variables, so a pathological shell cannot grow the handshake frame.
fn ambient_env_frame() -> Value {
    let pairs = crate::agent::env::ambient_environment_snapshot();
    serde_json::to_value(
        pairs
            .iter()
            .map(|(k, v)| json!({"name": k, "value": v}))
            .collect::<Vec<_>>(),
    )
    .unwrap_or(Value::Null)
}

/// Decode an `ambient_env` handshake frame into whole variables.
///
/// The client and the daemon both apply the secret filter, so a frame that was
/// tampered with (or built by an older client) is still safe to store: any
/// credential-bearing name is dropped here as well.
pub fn decode_ambient_env(frame: &Value) -> Vec<(String, String)> {
    let Some(items) = frame.as_array() else {
        return Vec::new();
    };
    let mut out = Vec::new();
    let mut total = 0usize;
    for item in items {
        let (Some(name), Some(value)) = (
            item.get("name").and_then(Value::as_str),
            item.get("value").and_then(Value::as_str),
        ) else {
            continue;
        };
        if name.is_empty() || value.is_empty() {
            continue;
        }
        // The same value filter the client applied, so a frame that was
        // tampered with (or built by an older client) still cannot carry a
        // credential under an innocent name.
        let Some(value) = crate::agent::env::sanitize_ambient_value(name, value) else {
            continue;
        };
        let cost = name.len() + value.len() + 2;
        if total + cost > crate::agent::env::AMBIENT_ENV_MAX_BYTES {
            break;
        }
        total += cost;
        out.push((name.to_string(), value));
    }
    out
}

/// Compare release versions numerically; prereleases precede the same release.
fn newer(client: &str, daemon: &str) -> bool {
    fn parts(version: &str) -> Option<([u64; 3], Option<&str>)> {
        let version = version.split('+').next()?;
        let (core, pre) = version
            .split_once('-')
            .map_or((version, None), |(v, p)| (v, Some(p)));
        let mut numbers = core.split('.');
        let tuple = [
            numbers.next()?.parse().ok()?,
            numbers.next()?.parse().ok()?,
            numbers.next()?.parse().ok()?,
        ];
        numbers.next().is_none().then_some((tuple, pre))
    }
    match (parts(client), parts(daemon)) {
        (Some((c, cp)), Some((d, dp))) => c > d || (c == d && cp.is_none() && dp.is_some()),
        _ => false,
    }
}

/// Whether this client should step over the daemon it just greeted.
///
/// A newer release always wins. At the same release — what every `cargo build`
/// leaves behind, since `CARGO_PKG_VERSION` does not move — the build clocks
/// decide, so a rebuilt binary replaces an idle hub built before it. A daemon
/// that reports no `build` predates the build handshake and is judged on the
/// release alone.
pub(crate) fn supersedes(version: &str, build: &Value, daemon: &str, daemon_build: &Value) -> bool {
    if version != daemon && newer(version, daemon) {
        return true;
    }
    match (
        build["id"].as_str(),
        build["ts"].as_u64(),
        daemon_build["id"].as_str(),
        daemon_build["ts"].as_u64(),
    ) {
        (Some(id), Some(ts), Some(daemon_id), Some(daemon_ts)) => id != daemon_id && ts > daemon_ts,
        _ => false,
    }
}

/// Env var overriding how long a busy daemon waits for a quiet moment before
/// it hands over anyway.
pub const HANDOVER_DEADLINE_ENV: &str = "HUB_HANDOVER_SECS";

/// Default handover deadline: long enough for a build to finish, short enough
/// that a daemon busy all day still picks up the newer build.
pub const DEFAULT_HANDOVER_SECS: u64 = 15 * 60;

/// Bounds on the handover deadline, so a request can neither cut a running
/// command off nor park the daemon forever.
const MIN_HANDOVER_SECS: u64 = 1;
const MAX_HANDOVER_SECS: u64 = 24 * 60 * 60;

/// How long a handover waits for a quiet moment before it happens anyway.
///
/// The env var is the operator's default; a `hub/handover` request may name its
/// own, clamped to [`MIN_HANDOVER_SECS`]..=[`MAX_HANDOVER_SECS`].
pub(crate) fn handover_deadline(requested: Option<u64>) -> Duration {
    let secs = requested
        .or_else(|| crate::config::env_parse(HANDOVER_DEADLINE_ENV))
        .unwrap_or(DEFAULT_HANDOVER_SECS)
        .clamp(MIN_HANDOVER_SECS, MAX_HANDOVER_SECS);
    Duration::from_secs(secs)
}

/// File in the hub directory remembering which (client build, hub build)
/// pairs have already warned.
const NEWER_WARNING_FILE: &str = "newer-warnings";

/// Print the "client newer than hub" warning unless this exact build pair has
/// already warned from `dir`.
///
/// A long-lived daemon that cannot step aside would otherwise repeat the line
/// on every CLI command for hours. The record is best-effort: an unreadable or
/// unwritable file makes the warning repeat rather than disappear.
fn warn_newer_once(dir: &std::path::Path, client_id: &str, daemon_id: &str, message: &str) {
    let pair = format!("{client_id} {daemon_id}");
    let path = dir.join(NEWER_WARNING_FILE);
    let known = std::fs::read_to_string(&path).unwrap_or_default();
    if known.lines().any(|line| line == pair) {
        return;
    }
    eprintln!("{message}");
    // Bound the file: only recent mismatches are worth remembering.
    let mut lines: Vec<&str> = known.lines().collect();
    lines.push(&pair);
    if lines.len() > 32 {
        lines.drain(..lines.len() - 32);
    }
    let mut body = lines.join("\n");
    body.push('\n');
    let _ = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW)
        .open(&path)
        .and_then(|mut file| std::io::Write::write_all(&mut file, body.as_bytes()));
}

/// How one side of the handshake names itself when they have to be told apart.
fn label(version: &str, build: &Value) -> String {
    match build["id"].as_str() {
        Some(id) => format!("{version} build {id}"),
        None => version.to_string(),
    }
}

/// Negotiate once, replacing only an older idle daemon. The retry is bounded.
async fn negotiated(admin: bool, cli: bool) -> Result<HubClient> {
    negotiated_identity(cli, hello_params(admin, &client_version(), &client_build())).await
}

async fn negotiated_identity(cli: bool, params: Value) -> Result<HubClient> {
    let version = client_version();
    let build = client_build();
    for attempt in 0..2 {
        let mut client = HubClient {
            stream: BufReader::new(connect_or_spawn().await?),
            next_id: 1,
            notifications: Vec::new(),
            watch_line: Vec::new(),
        };
        // Announce the identity before any replay. Both transports send the
        // host process they walked up to; the CLI additionally answers the
        // `initialize` the daemon expects from it below, and a client that
        // names no host at all keeps the `cli`/`clientInfo` fallback.
        let reply = match client.request("hub/hello", params.clone()).await {
            Ok(reply) => reply,
            // A daemon from before the version handshake only knows hello as
            // a notification: announce the identity that way and keep going,
            // since it cannot be asked to step aside either.
            Err(error) if error.to_string().starts_with("Method not found") => {
                client.notify("hub/hello", params.clone()).await?;
                eprintln!(
                    "[mini-swe] The running hub predates the version handshake; restart it when idle to pick up {version}."
                );
                Value::Null
            }
            Err(error) => return Err(error),
        };
        let daemon = reply["version"].as_str().unwrap_or("");
        let daemon_build = &reply["build"];
        if supersedes(&version, &build, daemon, daemon_build) {
            // An idle daemon is replaced on the spot: this client stops it and
            // starts its own replacement. One that cannot be replaced now is
            // offered a planned handover instead, so it picks the newer build
            // up at its first quiet moment. An idle daemon is never offered
            // one: the client replaces it directly, and a handover there
            // would only stop and respawn the fresh replacement for no gain —
            // a daemon no reaper has a pid for.
            let mut offer_handover = reply["busy"].as_bool().unwrap_or(true);
            if !offer_handover && attempt == 0 {
                match client.request("hub/shutdown", json!({})).await {
                    Ok(_) => {
                        // The reply precedes teardown. Wait for EOF, not merely
                        // the reply, so connect_or_spawn cannot dial the old hub.
                        let mut byte = [0u8; 1];
                        tokio::time::timeout(Duration::from_secs(5), client.stream.read(&mut byte))
                            .await??;
                        drop(client);
                        let paths = HubPaths::new(hub_dir()?);
                        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
                        // Teardown removes the socket before it drops the lock;
                        // wait for both, or the replacement would find the lock
                        // still held and exit as "hub already running".
                        loop {
                            if !paths.socket().exists() && !hub_lock_held(&paths.lock())? {
                                break;
                            }
                            anyhow::ensure!(
                                tokio::time::Instant::now() < deadline,
                                "Old hub did not stop"
                            );
                            tokio::time::sleep(Duration::from_millis(20)).await;
                        }
                        continue;
                    }
                    Err(_) => {
                        // A dispatch may have made the daemon busy since hello.
                        offer_handover = true;
                    }
                }
            }
            // Older hubs may not implement planned handover; keep their warning.
            if offer_handover {
                let _ = client.request("hub/handover", json!({})).await;
            }
            warn_newer_once(
                &hub_dir()?,
                build["id"].as_str().unwrap_or(&version),
                daemon_build["id"].as_str().unwrap_or(daemon),
                &format!(
                    "[mini-swe] Client {} is newer than hub {}; continuing with the existing daemon (busy or replacement unavailable).",
                    label(&version, &build),
                    label(daemon, daemon_build)
                ),
            );
        }
        if cli {
            client
                .request(
                    "initialize",
                    json!({
                        "protocolVersion": "2024-11-05", "capabilities": {},
                        "clientInfo": {"name": crate::mcp::CLI_CLIENT_NAME, "version": version}
                    }),
                )
                .await?;
        }
        return Ok(client);
    }
    unreachable!("the second negotiation always returns")
}

/// Read complete input frames into a bounded queue across daemon reconnects.
async fn pump_stdin(tx: tokio::sync::mpsc::Sender<Vec<u8>>) -> Result<()> {
    let mut stdin = BufReader::new(tokio::io::stdin());
    loop {
        let mut frame = Vec::new();
        let count = (&mut stdin)
            .take(1024 * 1024 + 1)
            .read_until(b'\n', &mut frame)
            .await?;
        if count == 0 {
            return Ok(());
        }
        anyhow::ensure!(frame.len() <= 1024 * 1024, "MCP request exceeds 1 MiB");
        if tx.send(frame).await.is_err() {
            return Ok(());
        }
    }
}

/// Answer each request sent to a lost daemon once, without replaying side effects.
async fn answer_cut_requests<W: AsyncWrite + Unpin>(
    stdout: &mut W,
    pending: &mut std::collections::BTreeMap<String, Value>,
) -> Result<()> {
    for (_, id) in std::mem::take(pending) {
        let frame = json!({"jsonrpc":"2.0", "id":id, "error": {
            "code":-32000, "message":"Hub restarted; retry the request",
            "data":{"retryable":true}}});
        stdout.write_all(format!("{frame}\n").as_bytes()).await?;
    }
    stdout.flush().await?;
    Ok(())
}

/// Follow daemon restarts, preserving identity and failing only requests in flight.
pub async fn proxy_stdio() -> Result<()> {
    let mut identity = hello_params(false, &client_version(), &client_build());
    if identity["agent_id"].is_null()
        && identity["host_id"].is_null()
        && identity["session_id"].is_null()
        && identity["watch_token"].is_null()
    {
        identity["agent_id"] = json!(format!("proxy:{}", uuid::Uuid::new_v4()));
    }
    let (tx, mut rx) = tokio::sync::mpsc::channel(16);
    let input = tokio::spawn(pump_stdin(tx));
    let mut pending = std::collections::BTreeMap::new();
    let mut initialize: Option<Value> = None;
    let mut stdout = tokio::io::stdout();
    let result = async {
        loop {
            // A daemon that goes away is not the end of the session: follow it,
            // so the MCP client sees one connection to the hub for its whole
            // life. The budget is spent across the whole session rather than per
            // reconnect, because a hub that never comes back must not leave a
            // proxy that can still serve one tool call running forever.
            let mut client = match reconnect_following(reconnect_deadline(), || {
                negotiated_identity(false, identity.clone())
            })
            .await
            {
                Ok(client) => client,
                Err(error) => {
                    answer_cut_requests(&mut stdout, &mut pending).await?;
                    return Err(error);
                }
            };
            if let Some(params) = &initialize {
                client.request("initialize", params.clone()).await?;
            }
            for frame in client.notifications.drain(..) {
                stdout.write_all(&frame).await?;
            }
            stdout.flush().await?;
            // Retain partial reply bytes across cancelled reads in select.
            let mut reply = Vec::new();
            loop {
                tokio::select! {
                    frame = rx.recv(), if pending.len() < 128 => {
                        let Some(frame) = frame else { return Ok::<(), anyhow::Error>(()); };
                        if let Ok(value) = serde_json::from_slice::<Value>(&frame) {
                            if value["method"] == "initialize" {
                                initialize = Some(value["params"].clone());
                            }
                            if let Some(id) = value.get("id") {
                                pending.insert(id.to_string(), id.clone());
                            }
                        }
                        if client.stream.get_mut().write_all(&frame).await.is_err() { break; }
                    }
                    count = async {
                        (&mut client.stream).take(32 * 1024 * 1024 + 1 - reply.len() as u64)
                            .read_until(b'\n', &mut reply).await
                    } => {
                        match count {
                            Ok(0) | Err(_) => break,
                            Ok(_) => {}
                        }
                        anyhow::ensure!(reply.len() <= 32 * 1024 * 1024, "Hub frame exceeds 32 MiB");
                        if let Ok(value) = serde_json::from_slice::<Value>(&reply)
                            && let Some(id) = value.get("id") {
                            pending.remove(&id.to_string());
                        }
                        stdout.write_all(&reply).await?;
                        stdout.flush().await?;
                        reply.clear();
                    }
                }
            }
            answer_cut_requests(&mut stdout, &mut pending).await?;
        }
    }.await;
    input.abort();
    result
}

/// Sequential CLI requests, ignoring asynchronous MCP event notifications.
pub struct HubClient {
    stream: BufReader<UnixStream>,
    next_id: u64,
    notifications: Vec<Vec<u8>>,
    watch_line: Vec<u8>,
}

impl HubClient {
    /// Connect as the CLI: the agent its host process names, or `cli` when
    /// that host cannot be walked.
    pub async fn connect() -> Result<Self> {
        Self::connect_as_admin(false).await
    }

    /// Connect as the CLI, optionally with the operator's admin override.
    ///
    /// The identity is the host process this CLI runs under (H-3), so a worker
    /// dispatched by one invocation stays steerable from the next one and from
    /// the agent's MCP connection; `admin` is the human operator's
    /// `mini-swe-mcp --admin` bypass of the per-agent ownership check.
    pub async fn connect_as_admin(admin: bool) -> Result<Self> {
        negotiated(admin, true).await
    }

    /// Send a JSON-RPC notification (no id, no reply).
    async fn notify(&mut self, method: &str, params: Value) -> Result<()> {
        let frame = json!({"jsonrpc": "2.0", "method": method, "params": params});
        self.stream
            .get_mut()
            .write_all(format!("{frame}\n").as_bytes())
            .await?;
        Ok(())
    }

    async fn request(&mut self, method: &str, params: Value) -> Result<Value> {
        let id = self.next_id;
        self.next_id += 1;
        let frame = json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params});
        self.stream
            .get_mut()
            .write_all(format!("{frame}\n").as_bytes())
            .await?;
        loop {
            // Tool responses include bounded diffs and logs, larger than requests.
            const MAX_REPLY_BYTES: u64 = 32 * 1024 * 1024;
            let mut line = Vec::new();
            (&mut self.stream)
                .take(MAX_REPLY_BYTES + 1)
                .read_until(b'\n', &mut line)
                .await?;
            anyhow::ensure!(!line.is_empty(), "Hub closed the connection");
            anyhow::ensure!(
                line.len() as u64 <= MAX_REPLY_BYTES,
                "Hub response exceeds 32 MiB"
            );
            let reply: Value = serde_json::from_slice(&line)?;
            if reply.get("id") != Some(&json!(id)) {
                // Negotiation can race owner replay. Preserve it for the proxy
                // with both frame count and byte retention bounded.
                if self.notifications.len() < 100
                    && self.notifications.iter().map(Vec::len).sum::<usize>() + line.len()
                        <= 1024 * 1024
                {
                    self.notifications.push(line);
                }
                continue;
            }
            if let Some(error) = reply.get("error") {
                anyhow::bail!(
                    "{}",
                    error["message"].as_str().unwrap_or("Hub request failed")
                );
            }
            return Ok(reply["result"].clone());
        }
    }

    /// Owner-scoped actionable replay and the current watch set.
    pub async fn watch_snapshot(
        &mut self,
        ids: &std::collections::BTreeSet<String>,
        groups: &std::collections::BTreeSet<String>,
        initial: bool,
        all: bool,
    ) -> Result<Value> {
        if !self.watch_line.is_empty() {
            self.next_watch_notification().await?;
        }
        // One call carries the whole selection, so repeated `--group` flags
        // stay one watch over several rounds.
        self.request(
            "hub/watch",
            json!({"worker_ids":ids,"group":groups,"initial":initial,"all":all}),
        )
        .await
    }

    /// Acknowledge only after the caller successfully printed an event.
    pub async fn watch_ack(&mut self, sequence: u64) -> Result<()> {
        self.request("hub/watch/ack", json!({"sequence":sequence}))
            .await?;
        Ok(())
    }

    /// Consume the existing channel stream; snapshot replay repairs dropped frames.
    pub async fn next_watch_notification(&mut self) -> Result<()> {
        if !self.notifications.is_empty() {
            self.notifications.clear();
            return Ok(());
        }
        loop {
            let bytes = self.stream.fill_buf().await?;
            anyhow::ensure!(!bytes.is_empty(), "Hub closed the connection");
            let count = bytes
                .iter()
                .position(|byte| *byte == b'\n')
                .map_or(bytes.len(), |end| end + 1);
            anyhow::ensure!(
                self.watch_line.len() + count <= 1024 * 1024,
                "Hub notification exceeds 1 MiB"
            );
            self.watch_line.extend_from_slice(&bytes[..count]);
            self.stream.consume(count);
            if self.watch_line.last() == Some(&b'\n') {
                self.watch_line.clear();
                return Ok(());
            }
        }
    }

    pub async fn worker(&mut self, arguments: Value) -> Result<Value> {
        let result = self
            .request(
                "tools/call",
                json!({"name": "worker", "arguments": arguments}),
            )
            .await?;
        let text = result["content"][0]["text"]
            .as_str()
            .context("Missing worker tool result")?;
        Ok(serde_json::from_str(text)?)
    }
}

#[cfg(test)]
#[path = "client_reconnect_tests.rs"]
mod reconnect_tests;
