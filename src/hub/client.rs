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
    command.arg("daemon").stdin(Stdio::null()).stdout(Stdio::null()).stderr(log);
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
    tokio::spawn(async move { let _ = child.wait().await; });

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

async fn hello<W: AsyncWrite + Unpin>(writer: &mut W) -> Result<()> {
    let frame = json!({
        "jsonrpc": "2.0", "method": "hub/hello",
        "params": {"agent_id": std::env::var("MINI_SWE_AGENT_ID").ok(),
                   "pid": std::process::id(), "version": env!("CARGO_PKG_VERSION"),
                   "cwd": std::env::current_dir()?}
    });
    writer.write_all(format!("{frame}\n").as_bytes()).await?;
    writer.flush().await?;
    Ok(())
}

/// Forward bytes unchanged with fixed-size buffers and immediate output flushes.
async fn forward<R: AsyncRead + Unpin, W: AsyncWrite + Unpin>(mut reader: R, mut writer: W) -> Result<()> {
    let mut buffer = [0; 8192];
    loop {
        let count = reader.read(&mut buffer).await?;
        if count == 0 { return Ok(()); }
        writer.write_all(&buffer[..count]).await?;
        writer.flush().await?;
    }
}

/// Proxy stdio until either input closes. The daemon owns all MCP semantics.
pub async fn proxy_stdio() -> Result<()> {
    let mut stream = connect_or_spawn().await?;
    hello(&mut stream).await?;
    let (reader, writer) = stream.into_split();
    tokio::select! {
        result = forward(tokio::io::stdin(), writer) => result,
        result = forward(reader, tokio::io::stdout()) => result,
    }
}

/// Sequential CLI requests, ignoring asynchronous MCP event notifications.
pub struct HubClient {
    stream: BufReader<UnixStream>,
    next_id: u64,
}

impl HubClient {
    pub async fn connect() -> Result<Self> {
        let mut client = Self { stream: BufReader::new(connect_or_spawn().await?), next_id: 1 };
        client.request("initialize", json!({
            "protocolVersion": "2024-11-05", "capabilities": {},
            "clientInfo": {"name": "mini-swe-cli", "version": env!("CARGO_PKG_VERSION")}
        })).await?;
        hello(client.stream.get_mut()).await?;
        Ok(client)
    }

    async fn request(&mut self, method: &str, params: Value) -> Result<Value> {
        let id = self.next_id;
        self.next_id += 1;
        let frame = json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params});
        self.stream.get_mut().write_all(format!("{frame}\n").as_bytes()).await?;
        loop {
            // Tool responses include bounded diffs and logs, larger than requests.
            const MAX_REPLY_BYTES: u64 = 32 * 1024 * 1024;
            let mut line = Vec::new();
            (&mut self.stream).take(MAX_REPLY_BYTES + 1).read_until(b'\n', &mut line).await?;
            anyhow::ensure!(!line.is_empty(), "Hub closed the connection");
            anyhow::ensure!(line.len() as u64 <= MAX_REPLY_BYTES, "Hub response exceeds 32 MiB");
            let reply: Value = serde_json::from_slice(&line)?;
            if reply.get("id") != Some(&json!(id)) { continue; }
            if let Some(error) = reply.get("error") {
                anyhow::bail!("{}", error["message"].as_str().unwrap_or("Hub request failed"));
            }
            return Ok(reply["result"].clone());
        }
    }

    pub async fn worker(&mut self, arguments: Value) -> Result<Value> {
        let result = self.request("tools/call", json!({"name": "worker", "arguments": arguments})).await?;
        let text = result["content"][0]["text"].as_str().context("Missing worker tool result")?;
        Ok(serde_json::from_str(text)?)
    }
}
