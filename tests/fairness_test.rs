//! Fairness of the pool's worker slots, and the optional LLM concurrency cap.
//!
//! Tier 1 of the pool used to be one FIFO semaphore: whichever agent queued
//! first owned every slot and the other agents waited behind its whole backlog.
//! These tests drive the real `WorkerPool` (and the real `AgentRunner`) against
//! a loopback fake LLM server, so they observe the *grant order* the hub
//! actually produces rather than the scheduler's internal decision.

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use mini_swe_mcp::agent::{AgentRunner, ChatMessage, Role};
use mini_swe_mcp::pool::WorkerPool;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// Shared counters handed to each accepted connection.
#[derive(Clone)]
struct FakeState {
    arrivals: Arc<Mutex<Vec<String>>>,
    in_flight: Arc<AtomicUsize>,
    peak: Arc<AtomicUsize>,
}

/// A loopback server that answers every request with one SSE frame, after a
/// delay, and records the order in which the requests arrive.
struct FakeLlm {
    base_url: String,
    /// Arrival order, one entry per request.
    arrivals: Arc<Mutex<Vec<String>>>,
    /// Requests being served right now.
    in_flight: Arc<AtomicUsize>,
    /// Highest `in_flight` observed.
    peak: Arc<AtomicUsize>,
}

impl FakeLlm {
    async fn spawn(delay: Duration) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind loopback");
        let addr = listener.local_addr().expect("local addr");
        let arrivals = Arc::new(Mutex::new(Vec::new()));
        let in_flight = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));

        let state = FakeState {
            arrivals: arrivals.clone(),
            in_flight: in_flight.clone(),
            peak: peak.clone(),
        };
        tokio::spawn(async move {
            while let Ok((mut socket, _)) = listener.accept().await {
                let fake = state.clone();
                tokio::spawn(async move {
                    // Read the request head, then the JSON body it announces.
                    let mut head = Vec::new();
                    let mut probe = [0u8; 4096];
                    let mut body = Vec::new();
                    loop {
                        match socket.read(&mut probe).await {
                            Ok(0) | Err(_) => return,
                            Ok(n) => {
                                head.extend_from_slice(&probe[..n]);
                                if let Some(split) = head.windows(4).position(|w| w == b"\r\n\r\n") {
                                    body = head.split_off(split + 4);
                                    break;
                                }
                            }
                        }
                    }
                    let content_length = String::from_utf8_lossy(&head)
                        .lines()
                        .find_map(|line| {
                            let (name, value) = line.split_once(':')?;
                            name.eq_ignore_ascii_case("content-length")
                                .then(|| value.trim().parse::<usize>().ok())
                                .flatten()
                        });
                    if let Some(content_length) = content_length {
                        while body.len() < content_length {
                            let mut chunk = vec![0u8; content_length - body.len()];
                            match socket.read(&mut chunk).await {
                                Ok(0) | Err(_) => return,
                                Ok(n) => body.extend_from_slice(&chunk[..n]),
                            }
                        }
                    }
                    let text = String::from_utf8_lossy(&body).into_owned();
                    // The task text carries the dispatch marker.
                    let worker = ["a0", "a1", "a2", "a3", "b0", "b1"]
                        .iter()
                        .find(|marker| text.contains(&format!("fairness-marker-{marker}")))
                        .map(|marker| format!("marker-{marker}"))
                        .unwrap_or_else(|| "unknown".to_string());
                    fake.arrivals.lock().expect("arrivals lock").push(worker);

                    let now = fake.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
                    fake.peak.fetch_max(now, Ordering::SeqCst);
                    tokio::time::sleep(delay).await;
                    let payload = b"data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\ndata: [DONE]\n\n";
                    let head = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        payload.len()
                    );
                    if socket.write_all(head.as_bytes()).await.is_err() {
                        return;
                    }
                    let _ = socket.write_all(payload).await;
                    let _ = socket.flush().await;
                    fake.in_flight.fetch_sub(1, Ordering::SeqCst);
                });
            }
        });

        Self {
            base_url: format!("http://{addr}"),
            arrivals,
            in_flight,
            peak,
        }
    }

    /// Arrival order, one worker id per request.
    async fn arrivals(&self) -> Vec<String> {
        tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                let arrivals = self.arrivals.lock().expect("arrivals lock").clone();
                if arrivals.len() >= 6 {
                    return arrivals;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .map_err(|_| "every dispatched worker must reach the LLM")
        .unwrap()
    }
}

/// A git repository a worker can check out inside.
fn scratch_repo(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("fairness-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create scratch repo");
    let git = |args: &[&str]| {
        std::process::Command::new("git")
            .args(args)
            .current_dir(&dir)
            .output()
            .unwrap_or_else(|e| panic!("git {args:?} failed: {e}"))
    };
    for args in [
        &["init", "--quiet"][..],
        &["config", "user.email", "test@example.com"][..],
        &["config", "user.name", "test"][..],
    ] {
        assert!(git(args).status.success(), "git {args:?}");
    }
    std::fs::write(dir.join("README.md"), "# scratch\n").expect("write readme");
    assert!(
        git(&["add", "-A"]).status.success() && git(&["commit", "--quiet", "-m", "init"]).status.success(),
        "seed commit"
    );
    dir
}

/// The hub shape: agent A queues four workers, agent B queues two, and the pool
/// grants two slots. The first two slots go to A (it queued first with nothing
/// else waiting); after that the grants must alternate between the owners, so
/// B's workers are never stuck behind A's whole backlog.
#[tokio::test]
async fn worker_slots_alternate_between_owners_once_the_pool_is_full() {
    let scheduler = mini_swe_mcp::pool::FairScheduler::new(2);
    let held = [
        scheduler.acquire("agent-a").await,
        scheduler.acquire("agent-a").await,
    ];
    let mut queued = Vec::new();
    for owner in ["agent-a", "agent-a", "agent-b", "agent-b"] {
        queued.push(tokio::spawn({
            let scheduler = scheduler.clone();
            async move { scheduler.acquire(owner).await }
        }));
    }
    // A's first two workers hold both slots; the other four are queued.
    while scheduler.waiting() < 4 {
        tokio::time::sleep(Duration::from_millis(5)).await;
    }

    // Freeing one slot at a time must hand them to the owners in turn, and
    // never grant more than the pool allows.
    let mut owners = Vec::new();
    for slot in held {
        drop(slot);
        let next = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if let Some(handle) = queued.iter().position(|task| task.is_finished()) {
                    break queued.remove(handle).await.expect("queued worker");
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("a freed slot must be granted");
        owners.push(next);
    }
    assert_eq!(
        owners,
        ["agent-b", "agent-a", "agent-b", "agent-a"],
        "after the first slots the grants must interleave the owners"
    );
    for task in queued {
        task.abort();
    }
}

/// `HUB_LLM_CONCURRENCY` caps the chat-completion requests in flight across the
/// whole process: the gate is taken around the HTTP request and its stream, so
/// the fake server never sees more requests at once than the cap allows.
#[tokio::test]
async fn llm_concurrency_cap_limits_in_flight_requests() {
    let server = FakeLlm::spawn(Duration::from_millis(250)).await;
    let messages = vec![ChatMessage::text(Role::User, "hello")];

    for cap in [1usize, 2, 3] {
        let server = FakeLlm::spawn(Duration::from_millis(250)).await;
        let gate = std::sync::Arc::new(tokio::sync::Semaphore::new(cap));
        let runners: Vec<AgentRunner> = (0..cap + 3)
            .map(|_| {
                AgentRunner::new(
                    server.base_url.clone(),
                    "test-key".to_string(),
                    "test-model".to_string(),
                    None,
                )
                .with_stream_idle_timeout(Duration::from_secs(10))
                .with_llm_gate(Some(gate.clone()))
            })
            .collect();
        let requests: Vec<_> = runners
            .iter()
            .map(|runner| {
                let runner = runner.clone();
                let messages = messages.clone();
                tokio::spawn(async move { runner.run_step_llm(&messages).await })
            })
            .collect();
        for request in requests {
            request
                .await
                .expect("request task")
                .expect("fake SSE step");
        }
        assert!(
            server.peak.load(Ordering::SeqCst) <= cap,
            "cap {cap}: the gate must bound the requests in flight (peak {})",
            server.peak.load(Ordering::SeqCst)
        );
        assert_eq!(
            server.in_flight.load(Ordering::SeqCst),
            0,
            "cap {cap}: every permit is released"
        );
    }
}

