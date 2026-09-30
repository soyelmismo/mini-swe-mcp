//! Thin transports to the shared hub, with bounded startup retries.

use anyhow::{Context, Result};
use serde_json::{Value, json};
use std::os::unix::fs::OpenOptionsExt;
use std::os::unix::process::CommandExt;
use std::process::Stdio;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;

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
/// name the agent its workers belong to; without it the daemon derives the
/// identity from the `initialize` `clientInfo` (see [`crate::mcp::ConnectionContext::agent`]).
/// `admin` is the operator override that lifts the per-agent ownership check.
fn client_version() -> String {
    #[cfg(debug_assertions)]
    if let Ok(version) = std::env::var("MINI_SWE_FAKE_VERSION") {
        return version;
    }
    env!("CARGO_PKG_VERSION").to_string()
}

fn hello_params(admin: bool, version: &str) -> Value {
    json!({"agent_id": std::env::var("MINI_SWE_AGENT_ID").ok(),
           "pid": std::process::id(), "version": version,
           "cwd": std::env::current_dir().ok(), "admin": admin})
}

/// Compare release versions numerically; prereleases precede the same release.
fn newer(client: &str, daemon: &str) -> bool {
    fn parts(version: &str) -> Option<([u64; 3], Option<&str>)> {
        let version = version.split('+').next()?;
        let (core, pre) = version.split_once('-').map_or((version, None), |(v, p)| (v, Some(p)));
        let mut numbers = core.split('.');
        let tuple = [numbers.next()?.parse().ok()?, numbers.next()?.parse().ok()?, numbers.next()?.parse().ok()?];
        numbers.next().is_none().then_some((tuple, pre))
    }
    match (parts(client), parts(daemon)) {
        (Some((c, cp)), Some((d, dp))) => c > d || (c == d && cp.is_none() && dp.is_some()),
        _ => false,
    }
}

/// Negotiate once, replacing only an older idle daemon. The retry is bounded.
async fn negotiated(admin: bool, cli: bool) -> Result<HubClient> {
    let version = client_version();
    for attempt in 0..2 {
        let mut client = HubClient {
            stream: BufReader::new(connect_or_spawn().await?),
            next_id: 1,
            notifications: Vec::new(),
        };
        // CLI identity is stable before hello's replay; the proxy's identity
        // comes from its hello or the host's later initialize.
        if cli {
            client.request("initialize", json!({
                "protocolVersion": "2024-11-05", "capabilities": {},
                "clientInfo": {"name": crate::mcp::CLI_CLIENT_NAME, "version": version}
            })).await?;
        }
        let reply = client.request("hub/hello", hello_params(admin, &version)).await?;
        let daemon = reply["version"].as_str().unwrap_or("");
        if daemon != version && newer(&version, daemon) {
            if !reply["busy"].as_bool().unwrap_or(true) && attempt == 0 {
                match client.request("hub/shutdown", json!({})).await {
                    Ok(_) => {
                        // The reply precedes teardown. Wait for EOF, not merely
                        // the reply, so connect_or_spawn cannot dial the old hub.
                        let mut byte = [0u8; 1];
                        tokio::time::timeout(Duration::from_secs(5), client.stream.read(&mut byte)).await??;
                        drop(client);
                        let paths = HubPaths::new(hub_dir()?);
                        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
                        while paths.socket().exists() {
                            anyhow::ensure!(tokio::time::Instant::now() < deadline, "Old hub did not stop");
                            tokio::time::sleep(Duration::from_millis(20)).await;
                        }
                        continue;
                    }
                    Err(_) => {
                        // A dispatch may have made the daemon busy since hello.
                    }
                }
            }
            {
                eprintln!("[mini-swe] Client {version} is newer than hub {daemon}; continuing with the existing daemon (busy or replacement unavailable).");
            }
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
}

impl HubClient {
    /// Connect as the CLI: identity `cli`, shared by every invocation.
    pub async fn connect() -> Result<Self> {
        Self::connect_as_admin(false).await
    }

    /// Connect as the CLI, optionally with the operator's admin override.
    ///
    /// The `clientInfo.name` sent in `initialize` is what gives the CLI its
    /// stable `cli` identity (H-3), and `admin` is the human operator's
    /// `mini-swe-mcp --admin` bypass of the per-agent ownership check.
    pub async fn connect_as_admin(admin: bool) -> Result<Self> {
        negotiated(admin, true).await
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
                    && self.notifications.iter().map(Vec::len).sum::<usize>() + line.len() <= 1024 * 1024
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
