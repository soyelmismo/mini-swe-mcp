//! End-to-end coverage for the SSE streaming path of [`AgentRunner::run_step_llm`].
//!
//! The streaming core had **zero** test coverage, which is why two critical
//! bugs survived review: a sparse `tool_call` index fabricated phantom tool
//! calls and silently dropped the real command, and a frame that was not valid
//! UTF-8 was discarded without a trace. Both are reachable with a handful of
//! bytes on a loopback socket, so the harness below speaks real HTTP + SSE over
//! a `TcpListener` with *attacker-chosen chunk boundaries* — the framing edge
//! cases cannot be produced through a normal HTTP client.

use std::net::SocketAddr;
use std::time::Duration;

use mini_swe_mcp::agent::{AgentRunner, ChatMessage, LlmResponse, Role};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// How the fake server should answer the request.
struct MockSse {
    /// Raw body segments, written in order (each becomes one TCP write, so the
    /// client can observe them as separate chunks).
    segments: Vec<Vec<u8>>,
    /// Delay inserted before writing the segment at this index.
    delays: Vec<(usize, Duration)>,
    /// Serve the body as a non-streaming JSON payload instead of SSE frames.
    status_line: &'static str,
}

impl MockSse {
    /// A 200 SSE response made of the given raw segments.
    fn sse(segments: Vec<Vec<u8>>) -> Self {
        Self {
            segments,
            delays: Vec::new(),
            status_line: "HTTP/1.1 200 OK",
        }
    }

    /// Convenience: a well-formed SSE body of `data:` frames, terminated by
    /// `data: [DONE]`.
    fn frames(frames: &[&str]) -> Self {
        let mut segments: Vec<Vec<u8>> =
            frames.iter().map(|f| format!("data: {f}\n\n").into_bytes()).collect();
        segments.push(b"data: [DONE]\n\n".to_vec());
        Self::sse(segments)
    }

    fn with_delay(mut self, index: usize, delay: Duration) -> Self {
        self.delays.push((index, delay));
        self
    }
}

/// A one-shot loopback HTTP server that speaks a canned SSE body, then closes.
///
/// Returns the base URL to point an [`AgentRunner`] at.
async fn spawn_sse_server(body: MockSse) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind loopback");
    let addr: SocketAddr = listener.local_addr().expect("local addr");

    tokio::spawn(async move {
        let (mut socket, _) = match listener.accept().await {
            Ok(pair) => pair,
            Err(_) => return,
        };

        // Read just the request head; we never act on it beyond replying.
        let mut req = Vec::new();
        let mut probe = [0u8; 1024];
        loop {
            match socket.read(&mut probe).await {
                Ok(0) | Err(_) => break,
                Ok(n) => {
                    req.extend_from_slice(&probe[..n]);
                    if req.windows(4).any(|w| w == b"\r\n\r\n") {
                        break;
                    }
                }
            }
        }

        let mut head = format!(
            "{}\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n",
            body.status_line
        )
        .into_bytes();
        if let Err(e) = socket.write_all(&head).await {
            let _ = e;
            return;
        }
        head.clear();

        for (i, segment) in body.segments.iter().enumerate() {
            if let Some((_, delay)) = body.delays.iter().find(|(idx, _)| *idx == i) {
                tokio::time::sleep(*delay).await;
            }
            // A short sleep between writes pushes a real TCP segment boundary so
            // reqwest hands the client a partial frame.
            tokio::time::sleep(Duration::from_millis(1)).await;
            if socket.write_all(segment).await.is_err() {
                return;
            }
            let _ = socket.flush().await;
        }
        let _ = socket.shutdown().await;
    });

    format!("http://{addr}")
}

fn runner(base: &str) -> AgentRunner {
    AgentRunner::new(
        base.to_string(),
        "test-key".to_string(),
        "test-model".to_string(),
        None,
    )
    .with_stream_idle_timeout(Duration::from_secs(10))
}

fn user_turn() -> Vec<ChatMessage> {
    vec![ChatMessage::text(Role::User, "hello")]
}

// ---------------------------------------------------------------------------
// F1 — sparse tool_call index must not fabricate phantoms or lose the command
// ---------------------------------------------------------------------------

/// A model may legally open with `index: 3` (gateways re-index or skip slots).
/// The old code resized a `Vec` to length 4, inventing three empty tool calls
/// with duplicated ids; the *first* of those placeholders (empty arguments) then
/// shadowed the real command, so `command` came back `None` and the agent loop
/// stalled forever. Only the real call must survive, and it must carry the
/// model's command.
#[tokio::test]
async fn sparse_tool_call_index_yields_single_real_call_with_command() {
    let body = MockSse::frames(&[
        r#"{"choices":[{"delta":{"tool_calls":[{"index":3,"id":"real","function":{"name":"bash","arguments":"{\"command\":\"ls\"}"}}]}}]}"#,
    ]);
    let base = spawn_sse_server(body).await;
    let resp: LlmResponse = runner(&base).run_step_llm(&user_turn()).await.expect("step");

    assert_eq!(
        resp.command.as_deref(),
        Some("ls"),
        "the real command must survive a sparse index"
    );
    let calls = resp.tool_calls.expect("tool_calls present");
    assert_eq!(calls.len(), 1, "placeholders must not be fabricated: {calls:?}");
    assert_eq!(calls[0].id, "real");
    assert_eq!(calls[0].function.name, "bash");
    assert_eq!(resp.tool_call_id.as_deref(), Some("real"));
}

/// Every emitted `tool_call` needs a distinct, non-empty id: providers reject a
/// duplicated `tool_call_id` on the next turn, making the failure sticky.
#[tokio::test]
async fn duplicate_tool_call_ids_are_replaced_with_unique_ids() {
    // Two calls that (wrongly) share the same provider id.
    let frame = r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"dup","function":{"name":"bash","arguments":"{\"command\":\"a\"}"}},{"index":1,"id":"dup","function":{"name":"bash","arguments":"{\"command\":\"b\"}"}}]}}]}"#;
    let body = MockSse::frames(&[frame]);
    let base = spawn_sse_server(body).await;
    let resp = runner(&base).run_step_llm(&user_turn()).await.expect("step");

    let calls = resp.tool_calls.expect("tool_calls present");
    assert_eq!(calls.len(), 2);
    assert_ne!(calls[0].id, calls[1].id, "ids must be unique");
    assert!(calls.iter().all(|c| !c.id.is_empty()));
}

/// A repeated `index` across deltas keeps accumulating into one call rather than
/// creating a second one.
#[tokio::test]
async fn repeated_index_deltas_accumulate_into_one_call() {
    let body = MockSse::frames(&[
        r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c0","function":{"name":"bash","arguments":"{\"comm"}}]}}]}"#,
        r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"and\":\"pwd\"}"}}]}}]}"#,
    ]);
    let base = spawn_sse_server(body).await;
    let resp = runner(&base).run_step_llm(&user_turn()).await.expect("step");

    let calls = resp.tool_calls.expect("tool_calls present");
    assert_eq!(calls.len(), 1, "same index must stay one call: {calls:?}");
    assert_eq!(calls[0].function.arguments, r#"{"command":"pwd"}"#);
    assert_eq!(resp.command.as_deref(), Some("pwd"));
}

// ----------
// F1b — a provider that re-uses `index: 0` must not lose every tool call
// ----------

/// Regression (PENDING_ROADMAP #2): providers that omit `index` (serde defaults
/// it to `0`) or that send every call in a turn as `index: 0` used to hit
/// `accumulate_tool_call`'s collision branch, which marked the entry
/// `malformed` and dropped **both** calls. The turn came back with no command
/// at all, so the agent loop re-issued the same prompt forever.
///
/// Two sequential calls, both on `index: 0`, must both survive the round trip.
#[tokio::test]
async fn sequential_calls_reusing_index_zero_both_survive() {
    let body = MockSse::frames(&[
        r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"a","function":{"name":"bash","arguments":"{\"command\":\"ls\"}"}}]}}]}"#,
        r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"b","function":{"name":"bash","arguments":"{\"command\":\"pwd\"}"}}]}}]}"#,
    ]);
    let base = spawn_sse_server(body).await;
    let resp = runner(&base).run_step_llm(&user_turn()).await.expect("step");

    let calls = resp.tool_calls.expect("tool_calls present");
    assert_eq!(
        calls.len(),
        2,
        "a numbering quirk must not drop calls: {calls:?}"
    );
    assert_eq!(calls[0].id, "a");
    assert_eq!(calls[1].id, "b");
    assert_eq!(calls[0].function.arguments, r#"{"command":"ls"}"#);
    assert_eq!(calls[1].function.arguments, r#"{"command":"pwd"}"#);
    // The whole point of the fix: the turn is no longer empty.
    assert_eq!(resp.command.as_deref(), Some("ls"));
    assert!(resp.tool_call_id.is_some(), "tool_call_id must be reported");
}

/// A delta with **no** `index` field at all deserializes to `0`; two such calls
/// in one turn are the exact shape that deadlocked the agent.
#[tokio::test]
async fn calls_without_an_index_field_are_not_dropped() {
    let body = MockSse::frames(&[
        r#"{"choices":[{"delta":{"tool_calls":[{"id":"x","function":{"name":"bash","arguments":"{\"command\":\"whoami\"}"}}]}}]}"#,
        r#"{"choices":[{"delta":{"tool_calls":[{"id":"y","function":{"name":"bash","arguments":"{\"command\":\"id\"}"}}]}}]}"#,
    ]);
    let base = spawn_sse_server(body).await;
    let resp = runner(&base).run_step_llm(&user_turn()).await.expect("step");

    let calls = resp.tool_calls.expect("tool_calls present");
    assert_eq!(
        calls.len(),
        2,
        "index-less calls must both survive: {calls:?}"
    );
    assert_eq!(calls[0].function.arguments, r#"{"command":"whoami"}"#);
    assert_eq!(calls[1].function.arguments, r#"{"command":"id"}"#);
}

// ---------------------------------------------------------------------------
// F2 — invalid UTF-8 must degrade visibly, not vanish
// ---------------------------------------------------------------------------

/// A frame containing raw `0xFF 0xFE` used to be `continue`d away with no log
/// and no counter, so the surrounding reply silently lost a segment. It must
/// now be decoded lossily (visible as U+FFFD) and counted on the response.
#[tokio::test]
async fn invalid_utf8_frame_is_lossy_decoded_and_counted() {
    let mut bad = b"data: {\"choices\":[{\"delta\":{\"content\":\"".to_vec();
    bad.extend_from_slice(&[0xFF, 0xFE]); // invalid bytes inside the content
    bad.extend_from_slice(b"\"}}]}\n\n");
    let body = MockSse::sse(vec![
        bad,
        b"data: {\"choices\":[{\"delta\":{\"content\":\"after\"}}]}\n\n".to_vec(),
        b"data: [DONE]\n\n".to_vec(),
    ]);
    let base = spawn_sse_server(body).await;
    let resp = runner(&base).run_step_llm(&user_turn()).await.expect("step");

    assert_eq!(
        resp.invalid_utf8_lines, 1,
        "the corrupted frame must be counted"
    );
    // The `after` token lives in a later, valid frame: previously the *bad*
    // frame's segment vanished; now the content is present and marked lossy.
    assert!(
        resp.content.contains("after"),
        "the valid segment must still be present: {:?}",
        resp.content
    );
    assert!(
        resp.content.contains('\u{FFFD}'),
        "the invalid bytes must surface as U+FFFD, not vanish: {:?}",
        resp.content
    );
}

/// A healthy stream reports zero corruption.
#[tokio::test]
async fn valid_stream_reports_no_invalid_utf8() {
    let body = MockSse::frames(&[
        r#"{"choices":[{"delta":{"content":"clean"}}]}"#,
    ]);
    let base = spawn_sse_server(body).await;
    let resp = runner(&base).run_step_llm(&user_turn()).await.expect("step");
    assert_eq!(resp.invalid_utf8_lines, 0);
    assert_eq!(resp.content, "clean");
}

// ---------------------------------------------------------------------------
// §4 — framing correctness that must be preserved
// ---------------------------------------------------------------------------

/// Newline framing guarantees a line is complete before it is decoded, so a
/// multi-byte character split across arbitrary TCP segment boundaries is
/// byte-exact. Probed at several segment sizes.
#[tokio::test]
async fn multibyte_utf8_split_across_tiny_tcp_segments_is_exact() {
    let text = "€😀 mixed multibyte ✅";
    let payload = format!(
        "data: {}\n\n",
        serde_json::json!({"choices":[{"delta":{"content": text}}]})
    );

    // Split the whole payload into N-byte segments.
    for seg in [1usize, 2, 3, 5, 7] {
        let bytes = payload.clone().into_bytes();
        let segments: Vec<Vec<u8>> = bytes.chunks(seg).map(|c| c.to_vec()).collect();
        let body = MockSse::sse(segments);
        let base = spawn_sse_server(body).await;
        let resp = runner(&base).run_step_llm(&user_turn()).await.expect("step");
        assert_eq!(
            resp.content, text,
            "content corrupted at {seg}-byte TCP segments"
        );
        assert_eq!(resp.invalid_utf8_lines, 0);
    }
}

/// A frame delivered one byte at a time (the worst case for a re-scanning
/// buffer) still reassembles exactly — the scan is a single forward pass.
#[tokio::test]
async fn byte_at_a_time_frames_reassemble() {
    let body = MockSse::frames(&[
        r#"{"choices":[{"delta":{"content":"a"}}]}"#,
        r#"{"choices":[{"delta":{"content":"b"}}]}"#,
        r#"{"choices":[{"delta":{"content":"c"}}]}"#,
    ]);
    // The frames() helper already splits into segments; re-chunk to 1 byte.
    let mut all = Vec::new();
    for seg in &body.segments {
        all.extend_from_slice(seg);
    }
    let segments: Vec<Vec<u8>> = all.chunks(1).map(|c| c.to_vec()).collect();
    let base = spawn_sse_server(MockSse::sse(segments)).await;
    let resp = runner(&base).run_step_llm(&user_turn()).await.expect("step");
    assert_eq!(resp.content, "abc");
}

/// A JSON object split mid-token across frames reassembles by concatenation.
#[tokio::test]
async fn arguments_split_mid_json_reassemble() {
    let body = MockSse::frames(&[
        r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c","function":{"name":"bash","arguments":"{\"comm"}}]}}]}"#,
        r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"and\":\"ls -la\"}"}}]}}]}"#,
    ]);
    let base = spawn_sse_server(body).await;
    let resp = runner(&base).run_step_llm(&user_turn()).await.expect("step");
    assert_eq!(resp.command.as_deref(), Some("ls -la"));
}

/// SSE comments (`:` keep-alives), blank lines, and the `[DONE]` sentinel are
/// skipped / honoured; only the real frames contribute.
#[tokio::test]
async fn comments_keepalives_and_done_sentinel_are_handled() {
    let mut body = Vec::new();
    body.extend_from_slice(b": keep-alive\n\n");
    body.extend_from_slice(b"\n");
    body.extend_from_slice(
        b"data: {\"choices\":[{\"delta\":{\"content\":\"kept\"}}]}\n\n",
    );
    // [DONE] then a trailing frame: everything after [DONE] must be ignored.
    body.extend_from_slice(b"data: [DONE]\n\n");
    body.extend_from_slice(
        b"data: {\"choices\":[{\"delta\":{\"content\":\"dropped\"}}]}\n\n",
    );
    let base = spawn_sse_server(MockSse::sse(vec![body])).await;
    let resp = runner(&base).run_step_llm(&user_turn()).await.expect("step");
    assert_eq!(resp.content, "kept", "frames after [DONE] must be ignored");
}

/// A stream that ends without a `[DONE]` sentinel is still finalized from what
/// arrived (some proxies just close the socket).
#[tokio::test]
async fn stream_without_done_sentinel_is_finalized() {
    let body = MockSse::sse(vec![
        b"data: {\"choices\":[{\"delta\":{\"content\":\"no-done\"}}]}\n\n".to_vec(),
    ]);
    let base = spawn_sse_server(body).await;
    let resp = runner(&base).run_step_llm(&user_turn()).await.expect("step");
    assert_eq!(resp.content, "no-done");
}

/// A proxy that ignores `stream=true` and returns a plain (non-streaming) JSON
/// completion is handled by the buffer fallback path.
#[tokio::test]
async fn non_streaming_json_body_is_parsed_via_fallback() {
    let body = r#"{"choices":[{"message":{"content":"plain","tool_calls":[{"id":"nc1","function":{"name":"bash","arguments":"{\"command\":\"echo hi\"}"}}]}}]}"#;
    let base = spawn_sse_server(MockSse::sse(vec![body.as_bytes().to_vec()])).await;
    let resp = runner(&base).run_step_llm(&user_turn()).await.expect("step");
    assert_eq!(resp.content, "plain");
    assert_eq!(resp.command.as_deref(), Some("echo hi"));
    let calls = resp.tool_calls.expect("tool_calls present");
    assert_eq!(calls[0].id, "nc1");
}

// ---------------------------------------------------------------------------
// F5 — accumulated content is capped
// ---------------------------------------------------------------------------

/// A model that streams a very long response must not inflate memory without
/// bound: the retained content is capped (a marker is appended).
///
/// The input deliberately overshoots the budget by 2x. Before the budget was
/// raised this test streamed exactly 64 KiB and asserted `<=`, which no longer
/// proves anything once the cap *is* 64 KiB — a pass/fail boundary case is
/// indistinguishable from a broken cap. Overflowing the cap is the only way to
/// show the guard still fires after the constant moved.
#[tokio::test]
async fn streamed_content_is_capped() {
    let limit = mini_swe_mcp::agent::MAX_STREAMED_CONTENT_BYTES;
    let big = "x".repeat(limit * 2);
    let payload = format!(
        "data: {}\n\n",
        serde_json::json!({"choices":[{"delta":{"content": big}}]})
    );
    let body = MockSse::sse(vec![payload.into_bytes(), b"data: [DONE]\n\n".to_vec()]);
    let base = spawn_sse_server(body).await;
    let resp = runner(&base).run_step_llm(&user_turn()).await.expect("step");
    assert!(
        resp.content.len() <= limit,
        "content must be capped, got {} bytes",
        resp.content.len()
    );
    assert_eq!(
        resp.content.len(),
        limit,
        "a {}-byte stream must retain exactly the {} byte budget",
        limit * 2,
        limit
    );
}

/// A long chain-of-thought reply used to be cut at 16 KiB, silently losing the
/// tail of the model's reasoning before it was fed back as context. With the
/// budget raised to 64 KiB, a 48 KiB CoT stream must round-trip intact.
#[tokio::test]
async fn long_reasoning_stream_is_not_truncated() {
    let cot = "r".repeat(48 * 1024);
    let body = MockSse::frames(&[&serde_json::json!({
        "choices": [{"delta": {"content": cot}}]
    })
    .to_string()]);
    let base = spawn_sse_server(body).await;
    let resp = runner(&base).run_step_llm(&user_turn()).await.expect("step");

    assert_eq!(
        resp.content.len(),
        cot.len(),
        "a 48 KiB CoT stream must survive intact"
    );
    assert_eq!(resp.content, cot, "content must be byte-exact");
}

// ---------------------------------------------------------------------------
// F7 — idle vs whole-request timeout
// ---------------------------------------------------------------------------

/// A stalled stream is aborted by the *idle* deadline (not a whole-request
/// deadline) and retried, rather than hanging forever.
#[tokio::test]
async fn stalled_stream_hits_idle_timeout_and_errors_after_retries() {
    // Server sends one frame, then stalls far beyond the (short) idle timeout.
    let body = MockSse::frames(&[r#"{"choices":[{"delta":{"content":"start"}}]}"#])
        .with_delay(1, Duration::from_secs(30));
    let base = spawn_sse_server(body).await;
    let runner = AgentRunner::new(
        base,
        "test-key".to_string(),
        "test-model".to_string(),
        None,
    )
    .with_stream_idle_timeout(Duration::from_millis(200))
    .with_max_retries(3)
    .with_initial_retry_delay(Duration::from_millis(50));

    let err = runner
        .run_step_llm(&user_turn())
        .await
        .expect_err("a permanently stalled stream must not hang forever");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("stalled") || msg.contains("after retries"),
        "unexpected error: {msg}"
    );
}

/// A slow-but-progressing stream that exceeds any single-read budget is *not*
/// aborted, because the deadline resets on every chunk. This is the exact
/// behaviour the old 120 s whole-request timeout got wrong.
#[tokio::test]
async fn slow_but_progressing_stream_is_not_aborted() {
    // Three frames, each delayed by less than the idle timeout but whose total
    // exceeds it. With a per-chunk timeout this succeeds; with a whole-request
    // deadline it would fail.
    let body = MockSse::frames(&[
        r#"{"choices":[{"delta":{"content":"one-"}}]}"#,
        r#"{"choices":[{"delta":{"content":"two-"}}]}"#,
        r#"{"choices":[{"delta":{"content":"three"}}]}"#,
    ])
    .with_delay(0, Duration::from_millis(120))
    .with_delay(1, Duration::from_millis(120))
    .with_delay(2, Duration::from_millis(120));
    let base = spawn_sse_server(body).await;
    let runner = AgentRunner::new(
        base,
        "test-key".to_string(),
        "test-model".to_string(),
        None,
    )
    .with_stream_idle_timeout(Duration::from_millis(400));

    let resp = runner.run_step_llm(&user_turn()).await.expect("must not be aborted");
    assert_eq!(resp.content, "one-two-three");
}

/// A body that never terminates a line (no `\n` at all) must not be buffered
/// without bound, and the reader must still resync on the newline that follows.
///
/// This is the end-to-end counterpart of the `push` unframed-tail cap: before
/// it, `SseAccumulator::push` retained every byte a malformed stream sent, so a
/// hostile or broken provider could grow the framing buffer for the lifetime of
/// the request. The oversized run below is deliberately larger than the cap, and
/// a *well-formed* frame is appended after it to prove the connection recovers
/// rather than wedging.
#[tokio::test]
async fn unterminated_oversized_run_is_capped_and_the_stream_recovers() {
    // ~2 MiB of newline-free garbage: far past any real SSE frame, and past the
    // reader's unframed-tail budget.
    let mut segments: Vec<Vec<u8>> = Vec::new();
    let garbage = vec![b'x'; 64 * 1024];
    for _ in 0..32 {
        segments.push(garbage.clone());
    }
    // A real frame right after the garbage, then the terminator.
    segments.push(b"\ndata: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n".to_vec());
    segments.push(b"data: [DONE]\n\n".to_vec());

    let base = spawn_sse_server(MockSse::sse(segments)).await;
    let resp = runner(&base)
        .run_step_llm(&user_turn())
        .await
        .expect("step");
    // The garbage line is dropped, but the reader resynced and delivered the
    // frame that followed it.
    assert_eq!(
        resp.content, "ok",
        "reader must resync past the garbage run"
    );
    assert_eq!(resp.invalid_utf8_lines, 0);
}
