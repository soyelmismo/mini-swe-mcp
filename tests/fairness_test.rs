//! Exercise the global LLM cap in isolated processes without mutating env.
use mini_swe_mcp::agent::{AgentRunner, ChatMessage, Role};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// `HUB_LLM_CONCURRENCY` caps the chat-completion requests in flight across the
/// whole process: the gate is taken around the HTTP request *and* its stream,
/// so the fake server never sees more requests at once than the cap allows.
///
/// The cap is read once per process, so each case runs in a child process
/// rather than mutating this one's environment.
#[test]
fn llm_concurrency_cap_limits_in_flight_requests() {
    for cap in [0, 1, 2] {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "llm_cap_child", "--nocapture"])
            .env("HUB_LLM_CONCURRENCY", cap.to_string())
            .env("MINI_SWE_LLM_CAP_CHILD", "1")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "cap {cap}: {}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[tokio::test]
async fn llm_cap_child() {
    if std::env::var("MINI_SWE_LLM_CAP_CHILD").is_err() {
        return;
    }
    let cap: usize = std::env::var("HUB_LLM_CONCURRENCY")
        .unwrap()
        .parse()
        .unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let active = Arc::new(AtomicUsize::new(0));
    let peak = Arc::new(AtomicUsize::new(0));
    let server = tokio::spawn({
        let active = active.clone();
        let peak = peak.clone();
        async move {
            let mut handlers = Vec::new();
            for _ in 0..6 {
                let (mut socket, _) = listener.accept().await.unwrap();
                let active = active.clone();
                let peak = peak.clone();
                handlers.push(tokio::spawn(async move {
                    let mut head = Vec::new();
                    let mut buffer = [0u8; 4096];
                    while !head.windows(4).any(|w| w == b"\r\n\r\n") {
                        let n = socket.read(&mut buffer).await.unwrap();
                        assert!(n > 0);
                        head.extend_from_slice(&buffer[..n]);
                    }
                    let now = active.fetch_add(1, Ordering::SeqCst) + 1;
                    peak.fetch_max(now, Ordering::SeqCst);
                    // Answer the headers, then hold the SSE body open: the cap
                    // must cover the stream, not just receipt of the headers.
                    socket
                        .write_all(
                            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n",
                        )
                        .await
                        .unwrap();
                    tokio::time::sleep(Duration::from_millis(100)).await;
                    active.fetch_sub(1, Ordering::SeqCst);
                    socket.write_all(b"data: [DONE]\n\n").await.unwrap();
                    socket.shutdown().await.unwrap();
                }));
            }
            for handler in handlers {
                handler.await.unwrap();
            }
        }
    });
    // Independently constructed runners must all draw from the same cap.
    let mut requests = Vec::new();
    for _ in 0..6 {
        let runner = AgentRunner::new(base.clone(), "k".into(), "m".into(), None);
        requests.push(tokio::spawn(async move {
            runner
                .run_step_llm(&[ChatMessage::text(Role::User, "hello")])
                .await
                .unwrap();
        }));
    }
    tokio::time::timeout(Duration::from_secs(10), async {
        for request in requests {
            request.await.unwrap();
        }
        server.await.unwrap();
    })
    .await
    .expect("all requests must finish and release their slots");
    assert_eq!(peak.load(Ordering::SeqCst), if cap == 0 { 6 } else { cap });
    assert_eq!(active.load(Ordering::SeqCst), 0);
}
