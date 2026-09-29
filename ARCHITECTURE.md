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
   - Retention is bounded: `MAX_STREAMED_CONTENT_BYTES` (16 KiB) for assistant text,
     `MAX_TOOL_ARGUMENT_BYTES` (64 KiB) per tool call, and `MAX_SSE_FRAME_BYTES`
     (1 MiB) per SSE line. Oversized lines are discarded through their newline
     before framing resumes; complete lines within a chunk need no buffer copy.
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

- **Cross-Process Steering (`steer` from another terminal)**:
  - The in-memory `pending_steer` queue and the resume channel only exist in the
    process that owns the worker, so a `steer` issued from a different shell used
    to fail with `Worker not found`.
  - `steer` therefore has a second delivery path: the message is appended
    atomically to the worker's mailbox, `<base>/swe-wt-<id>.steer` (`<base>` is
    `swe_base_dir()`, i.e. `/var/tmp` or `$SWE_TEMP_DIR`).
  - The mailbox is JSON lines (`{message, sent_at, pid}` per line), so a
    multi-line message cannot be split into bogus ones. Appends are single
    `O_APPEND` writes of a `\n`-terminated payload; drains claim the file with
    an atomic `rename` before reading, so the implementation loop and the review
    loop can never both deliver the same message.
  - Both loops drain the mailbox once per turn and append the messages as
    `ORCHESTRATOR GUIDANCE`. The worker deletes the mailbox on exit, so a
    finished worker leaves nothing behind for a later worker reusing the id.

---

## 5. Persistent Role Memory

Every dispatch used to start from a byte-identical, static `SYSTEM_PROMPT`:
nothing a previous run learned survived its worktree. Role memory gives each
*role* a durable notes file that is spliced into the system prompt at the exact
moment the prompt is built.

```
repo root
  │
  ▼
[1] ModelManifest::alias_for_model        resolved id ("combo:ninja") -> alias
  ▼
[2] build_system_prompt                  src/manifest/catalog.rs
  │     SYSTEM_PROMPT  +  memory section (only if a memory file exists)
  ▼
[3] load_agent_memory                    src/manifest/memory.rs
  │     <repo>/.agents/memory/<alias>.md, read fresh every dispatch
  ▼
ChatMessage::text(Role::System, prompt)   implementer loop AND review phase
```

Both the implementer loop and the review phase build their prompt through the
same `build_system_prompt`, so the reviewer inherits the *reviewer's* memory
(`nerd.md`) rather than the implementer's — the two roles never contaminate each
other.

Three invariants are load-bearing:

- **Optional.** `load_agent_memory` returns `None` — never an empty section — when
  the file is missing, blank, non-UTF-8, or is not even a file. A repository
  without `.agents/memory/` produces the exact static prompt it produced before
  this feature existed.
- **Bounded.** Injected memory is capped at `MAX_MEMORY_PROMPT_BYTES` (8 KiB),
  keeping the *newest* entries and cutting on a line boundary so the system
  instructions are never squeezed out by an ever-growing memory file. Memory is
  deliberately *not* memoized process-wide (unlike the catalog in `cache.rs`):
  it changes on disk between renders, so it must be re-read on every build.
- **Atomic.** `append_agent_memory` stages the new file next to the target and
  publishes it with `rename` (atomic within a filesystem), and serializes the
  read-modify-write under a process-wide lock, so a concurrent reader — or a
  `worktree` artifact sync — sees either the whole old file or the whole new one,
  and two agents of the same role finishing at once cannot discard each other's
  note.

The alias is reduced to a conservative `[a-z0-9_-]` slug before it touches the
filesystem, so a hostile `model` argument (`../../etc/passwd`) collapses to
`etc-passwd.md` inside the memory directory and can never escape it.

---

## 6. Step-Log Retention & Worker Residency

Each bash step appends an `AgentStepLog` to `WorkerRecord.logs`. Left
append-only, that buffer is the one structure in the pool that grows without
bound and is only released by an explicit `collect`, so a fleet of N workers
running M turns costs `N x M x ~2.1 KiB` resident bytes forever.

`logs` is therefore a `LogBuffer`: a `VecDeque` sliding window pre-allocated to
its retention size, bounded by *both* an entry count and a byte budget, evicting
strictly oldest-first and counting every eviction.

```
bash real
  │
  ▼
[1] agent::truncate_output            head + tail, hard ceiling 16384 B
  ▼
[2] pool::build_step_log              command <= 64 B, output <= 2048 B
  │                                    (marker charged against the budget)
  ▼
[3] LogBuffer::push                   evict oldest until
  │                                    len <= max_retained AND bytes <= max_bytes
  ▼
[4] emit_view(max_emitted)            tail only, for one response
  ▼
[5] expired_terminal_ids              Completed/Failed older than the TTL
```

- **Bounded per worker**: `WORKER_MAX_RETAINED_LOGS` (default 200, ceiling 1000).
  Growth in turns is O(1): a 600-turn worker retains the same ~430 KiB as a
  50-turn one, and across 64 workers the resident step-log payload is capped at
  ~27 MiB instead of 156 MiB.
- **Bounded per response**: `WORKER_MAX_EMITTED_LOGS` (default 40, ceiling 500)
  caps the logs inlined into one `collect` / `dispatch --wait` / `logs` reply,
  removing the transient 2-3x serialization spike a full 150-turn history used
  to cost per request.
- **Bounded lifetime**: `WORKER_TERMINAL_TTL_SECS` (default 300) evicts
  `Completed` / `Failed` records — with their registry rows — via a background
  reaper and lazily on the next `dispatch`. Fresh terminal records are kept, so
  a `collect` immediately after `wait: true` still resolves. `Running` and
  `Paused` records are never expired.
- **No silent degradation**: `total_steps`, `logs_retained`, `logs_omitted` and
  `logs_dropped` are reported on every log-bearing response, with a
  `logs_truncation_notice` whenever history is missing.

Invariants are declared on the types themselves: `LogBuffer` documents
`len <= max_retained` and `bytes <= max_bytes` after every push, and
`expired_terminal_ids` absorbs clock skew via `saturating_sub` so a
future-dated record is never evicted.
