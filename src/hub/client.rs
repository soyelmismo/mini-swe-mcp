//! Thin transports to the shared hub, with bounded startup retries.

use anyhow::{Context, Result};
use serde_json::{Value, json};
use std::os::unix::fs::OpenOptionsExt;
use std::os::unix::process::CommandExt;
use std::process::Stdio;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;

use super::daemon::hub_lock_held;
use super::identity;
use super::{HubPaths, hub_dir};

/// Dial the hub, starting a detached daemon if none is listening.
/// Racing starters are serialized by the daemon's exclusive flock.
pub async fn connect_or_spawn() -> Result<UnixStream> {
    let paths = HubPaths::new(hub_dir()?);
    if let Ok(stream) = UnixStream::connect(paths.socket()).await {
        return Ok(stream);
    }
    let log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW)
        .open(paths.log())
        .context("Could not open hub log")?;
    let mut command = std::process::Command::new(std::env::current_exe()?);
    command
        .arg("daemon")
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

    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    let mut delay = Duration::from_millis(20);
    loop {
        match UnixStream::connect(paths.socket()).await {
            Ok(stream) => return Ok(stream),
            Err(error) if tokio::time::Instant::now() >= deadline => {
                return Err(error).context("Hub did not start within 5 seconds; inspect hub.log");
            }
            Err(_) => {}
        }
        tokio::time::sleep_until((tokio::time::Instant::now() + delay).min(deadline)).await;
        delay = (delay * 2).min(Duration::from_millis(250));
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
fn supersedes(version: &str, build: &Value, daemon: &str, daemon_build: &Value) -> bool {
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
        let params = hello_params(admin, &version, &build);
        let reply = match client.request("hub/hello", params.clone()).await {
            Ok(reply) => reply,
            // A daemon from before the version handshake only knows hello as
            // a notification: announce the identity that way and keep going,
            // since it cannot be asked to step aside either.
            Err(error) if error.to_string().starts_with("Method not found") => {
                client.notify("hub/hello", params).await?;
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
            if !reply["busy"].as_bool().unwrap_or(true) && attempt == 0 {
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
                    }
                }
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

/// Forward bytes unchanged with fixed-size buffers and immediate output flushes.
async fn forward<R: AsyncRead + Unpin, W: AsyncWrite + Unpin>(
    mut reader: R,
    mut writer: W,
) -> Result<()> {
    let mut buffer = [0; 8192];
    loop {
        let count = reader.read(&mut buffer).await?;
        if count == 0 {
            return Ok(());
        }
        writer.write_all(&buffer[..count]).await?;
        writer.flush().await?;
    }
}

/// Proxy stdio until either input closes. The daemon owns all MCP semantics.
pub async fn proxy_stdio() -> Result<()> {
    let client = negotiated(false, false).await?;
    let buffered = client.stream.buffer().to_vec();
    let (reader, writer) = client.stream.into_inner().into_split();
    let output = async move {
        let mut stdout = tokio::io::stdout();
        for frame in client.notifications {
            stdout.write_all(&frame).await?;
        }
        stdout.write_all(&buffered).await?;
        stdout.flush().await?;
        forward(reader, stdout).await
    };
    tokio::select! {
        result = forward(tokio::io::stdin(), writer) => result,
        result = output => result,
    }
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
        group: Option<&str>,
        initial: bool,
    ) -> Result<Value> {
        if !self.watch_line.is_empty() {
            self.next_watch_notification().await?;
        }
        self.request(
            "hub/watch",
            json!({"worker_ids":ids,"group":group,"initial":initial}),
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
