//! A fake OpenAI-compatible streaming server, scripted by conversation turn.
//!
//! One `TcpListener`, one task per connection, and a script keyed on the
//! *turn count* of the conversation rather than on a connection counter: the
//! agent replays its whole history on every request, so the number of `tool`
//! results already in the body is exactly the number of commands that have
//! run. A retried request therefore re-asks the same turn instead of silently
//! shifting the script.
//!
//! The script is the three-step shape the load test measures:
//!
//! 1. a light command (`ls`), which takes a bash slot and nothing else,
//! 2. one heavy command recognised by
//!    [`is_heavy_command`](mini_swe_mcp::agent::is_heavy_command), which the
//!    pool's admission controller has to dose,
//! 3. the completion sentinel, which ends the worker.
//!
//! Every delta carries `reasoning_content` beside its payload, so the
//! accumulator sees the thinking-mode shape a real provider sends.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

/// A running fake LLM plus the counters a test reports on.
pub struct FakeLlm {
    /// `http://127.0.0.1:<port>`, ready for `OPENAI_API_BASE`.
    base_url: String,
    /// Chat-completion requests answered.
    requests: Arc<AtomicUsize>,
    /// Of those, the ones that scripted the heavy command.
    heavy: Arc<AtomicUsize>,
}

impl FakeLlm {
    /// Bind loopback and serve `light` on turn 1, `heavy` on turn 2 and the
    /// completion sentinel from turn 3 on, for as long as the handle lives.
    pub async fn spawn(light: &str, heavy: &str) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind the fake LLM on loopback");
        let base_url = format!("http://{}", listener.local_addr().expect("local addr"));
        let requests = Arc::new(AtomicUsize::new(0));
        let heavy_served = Arc::new(AtomicUsize::new(0));
        let light = light.to_string();
        let heavy = heavy.to_string();
        let counted = requests.clone();
        let heavy_counted = heavy_served.clone();

        tokio::spawn(async move {
            loop {
                let Ok((socket, _)) = listener.accept().await else {
                    return;
                };
                let light = light.clone();
                let heavy = heavy.clone();
                let requests = counted.clone();
                let heavy_served = heavy_counted.clone();
                // One task per connection: a slow or stalled client must never
                // hold up the conversations behind it.
                tokio::spawn(async move {
                    serve_turn(socket, &light, &heavy, requests, heavy_served).await;
                });
            }
        });

        Self {
            base_url,
            requests,
            heavy: heavy_served,
        }
    }

    /// The base URL to hand the pool as `OPENAI_API_BASE`.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Chat-completion requests answered so far.
    pub fn requests(&self) -> usize {
        self.requests.load(Ordering::Relaxed)
    }

    /// Requests that scripted the heavy command.
    pub fn heavy_commands(&self) -> usize {
        self.heavy.load(Ordering::Relaxed)
    }
}

/// Answer one request with the SSE body its turn calls for.
async fn serve_turn(
    mut socket: TcpStream,
    light: &str,
    heavy: &str,
    requests: Arc<AtomicUsize>,
    heavy_served: Arc<AtomicUsize>,
) {
    let Some(body) = read_request(&mut socket).await else {
        return;
    };
    requests.fetch_add(1, Ordering::Relaxed);
    let turn = turn_of(&body);
    let command = match turn {
        0 => light.to_string(),
        1 => {
            heavy_served.fetch_add(1, Ordering::Relaxed);
            heavy.to_string()
        }
        // Past the script the worker is done: keep answering the sentinel so a
        // stray extra turn ends the run instead of hanging it.
        _ => format!("echo {}", mini_swe_mcp::pool::COMPLETION_SENTINEL),
    };
    let response = sse_response(&command, turn);
    let _ = socket.write_all(response.as_bytes()).await;
    let _ = socket.flush().await;
    let _ = socket.shutdown().await;
}

/// Read one HTTP request head plus its `Content-Length` body.
///
/// Returns `None` when the peer closes the connection before the head is
/// complete, so a client killed mid-request cannot take the server down.
pub async fn read_request(socket: &mut TcpStream) -> Option<String> {
    let mut head: Vec<u8> = Vec::new();
    let mut chunk = [0u8; 4096];
    let header_end = loop {
        let read = socket.read(&mut chunk).await.ok()?;
        if read == 0 {
            return None;
        }
        head.extend_from_slice(&chunk[..read]);
        if head.windows(4).any(|window| window == b"\r\n\r\n") {
            break head
                .windows(4)
                .position(|window| window == b"\r\n\r\n")
                .map(|at| at + 4)?;
        }
    };
    let headers = String::from_utf8_lossy(&head[..header_end]).to_lowercase();
    let length: usize = headers
        .lines()
        .find_map(|line| line.strip_prefix("content-length:")?.trim().parse().ok())
        .unwrap_or(0);
    let mut body = head[header_end..].to_vec();
    while body.len() < length {
        let read = socket.read(&mut chunk).await.ok()?;
        if read == 0 {
            break;
        }
        body.extend_from_slice(&chunk[..read]);
    }
    body.truncate(length);
    Some(String::from_utf8_lossy(&body).into_owned())
}

/// The turn a request body is asking for: the commands already answered.
///
/// The agent replays its whole conversation, so counting the `tool` results it
/// carries is both the turn number and a retry-safe script index.
fn turn_of(body: &str) -> usize {
    let Ok(request) = serde_json::from_str::<Value>(body) else {
        return 0;
    };
    request["messages"]
        .as_array()
        .map(|messages| {
            messages
                .iter()
                .filter(|message| message["role"].as_str() == Some("tool"))
                .count()
        })
        .unwrap_or(0)
}

/// The full HTTP reply for one turn: headers, the streamed deltas and `[DONE]`.
fn sse_response(command: &str, turn: usize) -> String {
    let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n";
    let call_id = format!("call-{turn}");
    let arguments = json!({ "command": command }).to_string();
    let reasoning = json!({
        "choices": [{
            "delta": {
                "reasoning_content": format!("turn {turn}: planning the next command"),
                "content": format!("```bash\n{command}\n```")
            }
        }]
    });
    let tool_call = json!({
        "choices": [{
            "delta": {
                "reasoning_content": format!("turn {turn}: running it"),
                "tool_calls": [{
                    "index": 0,
                    "id": call_id,
                    "function": { "name": "bash", "arguments": arguments }
                }]
            }
        }]
    });
    format!(
        "{head}data: {reasoning}\n\ndata: {tool_call}\n\ndata: [DONE]\n\n",
        reasoning = reasoning,
        tool_call = tool_call
    )
}
