# Architecture & Systems Design

`mini-swe-mcp` is an autonomous, high-throughput software engineering subagent runner packaged as a Model Context Protocol (MCP) stdio server.

---

## 1. Concurrency Architecture: Two-Tier Semaphore

To prevent host resource saturation when fanning out dozens of parallel subagents on constrained hardware, execution is governed by a two-tier semaphore model:

```
[ Inbound Dispatches (MCP Client) ]
               │
               ▼
┌───────────────────────────────────────────────┐
│ Tier 1: Outer Subagent Pool                   │
│ Arc<Semaphore> (default: 64 permits)          │
│ Governs active subagent tasks & LLM dialogs   │
└──────────────────────┬────────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────────┐
│ Tier 2: Inner Bash Execution Gate             │
│ Arc<Semaphore> (default: cores / 2 permits)   │
│ Strictly throttles heavy OS subprocesses      │
└──────────────────────┬────────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────────┐
│ Subprocess Environment Governance             │
│ - nice -n 10 priority demotion                │
│ - kill_on_drop(true) process tree containment │
│ - Injected thread caps:                       │
│   CARGO_BUILD_JOBS, RUST_TEST_THREADS,        │
│   MAKEFLAGS (-j), CMAKE_BUILD_PARALLEL_LEVEL, │
│   RAYON_NUM_THREADS, OMP_NUM_THREADS,         │
│   GOMAXPROCS                                  │
└───────────────────────────────────────────────┘
```

- **Tier 1 (Outer Pool)**: Permits long-running agent loops (I/O bound waiting for LLM tokens) up to `MAX_CONCURRENT_WORKERS` (default 64).
- **Tier 2 (Inner Gate)**: Gates CPU- and memory-intensive commands (compilations, full test suites, ripgrep) to `BASH_CONCURRENT_LIMIT` (default: host CPU cores / 2). Even with 64 subagents active, only 2-4 compilation or test jobs execute simultaneously on the host.

---

## 2. LLM Protocol & Streaming Abort

The agent runner interacts with OpenAI-compatible endpoints using native Server-Sent Events (SSE) streaming and the official `tool_calls` format:

1. **Protocol State**:
   - `ChatMessage::text(role, content)`: a `system` / `user` / `assistant` turn.
   - `ChatMessage::assistant_with_tool_calls(content, tool_calls)`: records model intent.
   - `ChatMessage::tool_result(tool_call_id, content)`: records bash execution result matching the unique tool call ID.
   - The role is a validated `Role` enum, not a free-form `String`, and every `ChatMessage` field is
     private: the three constructors above are the only way to build a message, so an invalid role or
     a malformed tool pairing is a compile error rather than a provider-side `400`.
   - Fallback parser: transparently handles models that output markdown code blocks instead of structured tool calls.
2. **SSE Streaming & Immediate Abort**:
   - `reqwest` stream reader processes chunks via `resp.chunk().await`.
   - Streaming buffers delimiter frames (`\n`) and terminates immediately upon receiving `data: [DONE]`.
   - When a worker is killed or dropped, the stream future is aborted immediately, closing the underlying TCP socket and preventing orphaned token consumption or API stalls.
3. **Framing (`SseAccumulator`)**:
   - The read buffer only ever holds bytes that have not yet been framed, so the newline
     search is a single forward pass: no byte is scanned twice, even when a frame is split
     across thousands of one-byte TCP segments.
   - Multi-byte UTF-8 split across chunk boundaries is safe (a line is always complete
     before decoding). A genuinely malformed frame is logged at `warn` and decoded
     lossily, and counted on `LlmResponse::invalid_utf8_lines` — never silently dropped.
   - `tool_calls` are accumulated in a `BTreeMap` keyed by the provider's `index`, so a
     sparse index (e.g. `index: 3` on the first frame) cannot fabricate placeholder calls.
     Placeholders and malformed calls are filtered at finalization, and every emitted id is
     unique and non-empty.
   - Retention is bounded: `MAX_STREAMED_CONTENT_BYTES` (16 KiB) for assistant text and
     `MAX_TOOL_ARGUMENT_BYTES` (64 KiB) per tool call; overflow is logged and the call dropped.
4. **Idle (not whole-request) Timeout**:
   - The client uses `connect_timeout` for the handshake and `read_timeout` for a single
     stalled read; `run_step_llm` additionally wraps each `resp.chunk()` in
     `tokio::time::timeout(DEFAULT_STREAM_IDLE_TIMEOUT)`.
   - The deadline resets on every chunk, so a healthy-but-slow long generation runs to
     completion while a genuinely stalled stream is aborted and retried with backoff.
   - The process-wide `COMMAND_REGEX` (`LazyLock`) and `bash_tool_schema()` (`OnceLock`) are
     built once instead of once per worker / per request.

---

## 3. Git Worktree Isolation & RAII Hygiene

Subagents execute exclusively inside isolated git worktrees rather than modifying the working tree directly:

- **Worktree Initialization**:
  - Creates a temporary branch `worker-<id>`.
  - Attaches an isolated working directory at `$TMPDIR/swe-wt-<id>`.
- **Diff Tracking**:
  - `git add -N .` registers intent-to-add so untracked new files appear in `git diff HEAD`.
- **RAII Cleanup (`Drop`)**:
  - When the subagent completes, fails, or is aborted, the `WorktreeGuard` destructor triggers:
    1. `git worktree remove --force <path>`
    2. `git branch -D worker-<id>`
    3. Filesystem cleanup of any lingering build artifacts or untracked directories.

---

## 4. Orchestrator Steering & Pause Protocol

Workers communicate status and request human/orchestrator intervention via shell sentinels:

- **Turn Expansion (`REQUEST_TURNS: <n>`)**:
  - Worker prints `REQUEST_TURNS: 20` when nearing turn exhaustion. The pool detects the sentinel and expands the remaining allowance dynamically up to a hard cap of 150 turns.
- **Orchestrator Guidance (`ASK_ORCHESTRATOR: <question>`)**:
  - Worker prints `ASK_ORCHESTRATOR: "..."` when facing ambiguous requirements or breaking decisions.
  - The worker transitions into `WorkerState::Paused { question, step, paused_at }`.
  - A tokio one-shot/MPSC channel pauses the task until the orchestrator calls the MCP `steer` action, passing guidance that resumes the worker seamlessly.
