# Architecture & Systems Design

`mini-swe-mcp` is an autonomous, high-throughput software engineering subagent runner packaged as a Model Context Protocol (MCP) stdio server.

---

## 1. Concurrency: Pool, Bash and Build Semaphores

To prevent host resource saturation when fanning out dozens of parallel subagents on constrained hardware, execution is governed by three semaphores:

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
│ Tier 2: Bash Command Gate                     │
│ Arc<Semaphore> (default: one slot per worker) │
│ Bounds concurrent bash steps across the fleet │
└──────────────────────┬────────────────────────┘
                       │  (heavy commands only)
                       ▼
┌───────────────────────────────────────────────┐
│ Tier 3: Build Gate                            │
│ Arc<Semaphore> (default: cores / 2 permits)   │
│ Throttles cargo, make, pytest, compilers ...  │
└──────────────────────┬────────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────────┐
│ Subprocess Environment Governance             │
│ - nice -n 10 priority demotion                │
│ - kill_on_drop(true) process tree containment │
│ - per-worker CARGO_TARGET_DIR                 │
│ - Injected thread caps:                       │
│   CARGO_BUILD_JOBS, RUST_TEST_THREADS,        │
│   NEXTEST_TEST_THREADS, MAKEFLAGS (-j),       │
│   CMAKE_BUILD_PARALLEL_LEVEL,                 │
│   RAYON_NUM_THREADS, OMP_NUM_THREADS,         │
│   OPENBLAS_NUM_THREADS, MKL_NUM_THREADS,      │
│   GOMAXPROCS                                  │
└───────────────────────────────────────────────┘
```

- **Tier 1 (Outer Pool)**: permits long-running agent loops (I/O bound waiting for LLM tokens) up to `MAX_CONCURRENT_WORKERS` (default 64).
- **Tier 2 (Bash Gate)**: every bash step takes one permit. The default is **one slot per worker**, because a worker only ever runs one command at a time, so the gate is inert until an operator sets `BASH_CONCURRENT_LIMIT` to opt into a tighter fleet-wide cap.
- **Tier 3 (Build Gate)**: a command classified heavy by `is_heavy_command` (`cargo`, `make`, `rustc`, `pytest`, `cmake`, `ninja`, C/C++ compilers, `npm`/`yarn`/`pnpm`, `mvn`, `gradle`, `go test|build`) takes an *extra* permit, so independent builds cannot all run at once even when every worker is busy. `BASH_BUILD_LIMIT` sets the width (default: host CPU cores / 2, min 1, fallback 2).
- **Thread caps**: the number injected above is `BUILD_PARALLELISM`, itself defaulting to cores / 2, so a single heavy command cannot spawn one thread per core either.

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
   - Reasoning is replayed with the assistant turn through `with_reasoning_content`, because
     thinking-mode providers reject a follow-up request whose previous assistant message lost it.
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
     before decoding). A frame that is not valid UTF-8 is decoded lossily and counted on
     `LlmResponse::invalid_utf8_lines`, so corruption is observable instead of absorbed; a
     frame that does not parse as JSON is skipped with a warning emitted once per stream
     rather than once per frame.
   - The accumulator is deliberately tolerant of the shapes real providers emit: a
     continuation delta that re-sends empty `id` / `name` fields continues the call it
     belongs to instead of opening a new one; a frame carrying explicit `null`s
     (`"tool_calls": null`, `"content": null`) is consumed like any other; and a chunk
     with both `reasoning` and `reasoning_content` is recorded **once**, so a proxy
     echoing the same text under both keys cannot double the reasoning in history.
   - `tool_calls` are accumulated in a `BTreeMap` keyed by the provider's `index`, so a
     sparse index (e.g. `index: 3` on the first frame) cannot fabricate placeholder calls.
     Placeholders and malformed calls are filtered at finalization, and every emitted id is
     unique and non-empty. A *conflicting id* on an already-populated `index` is treated as a
     provider numbering quirk rather than corruption: the delta is redirected to a fresh slot
     just past the highest index in use, so a provider that sends every call in a turn as
     `index: 0` (or omits `index`, which serde defaults to `0`) still yields every call instead
     of an empty turn.
   - Retention is bounded: `MAX_STREAMED_CONTENT_BYTES` (64 KiB) for assistant text and for
     reasoning, `MAX_TOOL_ARGUMENT_BYTES` (64 KiB) per tool call, and `MAX_SSE_FRAME_BYTES`
     (1 MiB) per SSE line. Oversized lines are discarded through their newline before framing
     resumes; complete lines within a chunk need no buffer copy.
   - One step runs one command and answers with one `tool` message, so `finish` replays
     **only the executed call** into history. Sibling calls the model emitted but that were
     never run are dropped, since replaying them would leave unanswered `tool_calls` on the
     assistant turn — which the tool-call protocol rejects on the next request.
4. **Idle (not whole-request) Timeout**:
   - The HTTP client sets only `connect_timeout` (30 s). A whole-request or
     `read_timeout` deadline would kill a healthy-but-slow generation, so the stream
     body is instead guarded per chunk: `run_step_llm` wraps every `resp.chunk()` in
     `tokio::time::timeout(DEFAULT_STREAM_IDLE_TIMEOUT)`.
   - The deadline resets on every chunk, so a healthy-but-slow long generation runs to
     completion while a genuinely stalled stream is aborted and retried with backoff
     (`LLM_MAX_RETRIES` attempts, `agent::retry`).
   - The process-wide `BASH_BLOCK_RE` (`LazyLock`) — the fenced bash/sh code-block pattern of the
     fallback parser — is compiled once instead of once per worker, so its DFA cache is
     shared rather than duplicated. The `bash` tool schema itself is a plain value built per
     request; it carries no state worth memoizing.

---

## 3. Command Guard & Filesystem Confinement

Two independent layers stand between a model-authored command and the host.

- **Text guard (`agent::intercept::check_command`)** — a single function, not a pipeline of
  interceptors, run before any child is spawned. It canonicalises the command (unquoting
  tokens, neutralising expansions that start a path) and then judges only what would
  actually *run*: the command words of every segment (`rm -rf /`, `mkfs*`, `dd if=…`, the
  classic fork bomb), plus the body of a heredoc when a shell word appears on its opening
  line (`bash <<EOF`, `cat <<EOF | sh`). A heredoc that only writes data — a source file or a
  fixture that happens to mention `rm -rf /` — is stripped before scanning, so editing the
  guard itself is never blocked by it. Plain developer verbs (`cargo`, `git`, `ls`, `cat`)
  with no shell composition take a fast path. The same stripped command then faces
  `sandbox::validate_bash_command`, which refuses whole-filesystem searches (`find /`,
  `grep -r /`) that would turn a worktree into a disk-wide scan. Neither check is an `Err`:
  both are reported to the model as output with a non-zero exit code, so the worker can
  recover on its next turn.
- **Filesystem confinement** — bubblewrap builds the mount namespace when it is available
  (`SWE_DISABLE_SANDBOX=1` opts out). Otherwise there is exactly one Landlock path:
  `sandbox::build_landlock_plan` builds the plan in the parent (ABI probe,
  canonicalisation, syscall-backed existence checks) and a `Command::pre_exec` hook installs
  it, because `landlock_restrict_self` restricts the *calling* process and `pre_exec` runs
  after `fork` but before `exec`, in the child that is about to become the command. The two
  are never stacked — bwrap already confines the process, so a second restriction could only
  conflict. A kernel without Landlock — or `SWE_DISABLE_LANDLOCK=1` — yields `Ok(None)`,
  registers no hook, and the command runs unconfined rather than failing; a malformed plan
  (a missing worktree or target dir) is a caller bug and stays an `Err`.
- **Output truncation (`truncate_with_dropped`)** — a step's combined output is bounded by
  `TRUNCATE_LIMIT` (16 KiB), keeping `TRUNCATE_HEAD` (12 KiB) at the start and
  `TRUNCATE_TAIL` (4 KiB) at the end. The pipes are read through a fixed head/tail buffer, so
  a chatty command costs 16 KiB of memory whether it printed 16 KiB or 16 GiB, and the
  elision count is folded into the truncation marker instead of being forgotten.

---

## 4. Git Worktree Isolation & RAII Hygiene

Subagents execute exclusively inside isolated git worktrees rather than modifying the working tree directly:

- **Worktree Initialization** (`worktree::WorktreeGuard::new`):
  - Creates a branch `worker-<id>` and attaches an isolated working directory at
    `<swe_base_dir()>/swe-wt-<id>` (`/var/tmp`, or `$SWE_TEMP_DIR`).
  - Records the checkout's `HEAD` as the worker's **base commit** and fingerprints the
    artifact directories (`audits`, `reports`, `.agents`, `artifacts`) it seeded in, so the
    way out can tell worker output from files that were merely there.
- **Diff Tracking** (`get_diff`):
  - `git add -N .` registers intent-to-add so untracked new files appear in the diff.
  - The diff is taken against the **base commit**, not `HEAD`, so everything since the
    worker started — including the auto-checkpoint commits — is reported at once.
- **Artifact Sync** (`sync_artifacts`):
  - Only what the worker produced travels back: a file it created, or one whose content
    still differs from the seeded copy. A seeded file the worker never touched is not
    written, so a repo root that moved on while the worker ran is not reverted to a stale
    copy. Caches and build output are skipped by name, and each file is published
    atomically, so a concurrent reader never sees a half-written artifact.
- **RAII Cleanup (`Drop`)**:
  1. `sync_artifacts` (best-effort safety net for teardown paths that skipped it),
  2. `git worktree remove --force <path>`,
  3. `git branch -D worker-<id>` — **only** when the branch carries no commits beyond the
     base and was not explicitly preserved; a worker that committed work keeps its branch,
  4. removal of the pid lease, the directory if `worktree remove` could not, and the
     per-worktree target dir.
- **Prune** (`worktree::prune`): a worktree whose owning pid is gone is *salvaged first* —
  its uncommitted changes are committed onto the `worker-<id>` branch, so a crashed worker
  loses nothing — and only then removed. Branches with commits missing from `HEAD` are
  preserved rather than deleted.
- **Merge** (`pool::merge`): the one-command landing of a finished worker's branch on the
  base branch recorded for it. Every refusal is taken before anything is written — the
  worker must be terminal, the repository must already have the base branch checked out
  (`git merge-tree --write-tree` decides cleanliness, and only a file the merge would write
  is a reason to refuse a dirty tree), and the verify gate runs on the merge result in a
  throwaway worktree under the scratch root. Only then does `git merge --no-ff` run, after
  which the branch, the history file and the worktree leftovers are reclaimed with the same
  helpers the prune sweep uses. It never moves `HEAD` and never pushes.
- **Batch merge** (`pool::merge`, `merge --approved`): the same operation for a whole round. It
  selects the caller's completed workers that carry an approval (optionally one group), composes
  their branches in approval order with `git merge-tree` on top of the previous result, skips a
  conflicting branch and reports it with the `steer` that sends it back, and runs the shared verify
  gate **once** on the combined tree. Only a passing gate merges: one `--no-ff` merge commit per
  worker, then the same per-worker cleanup. A failing gate merges nothing and attributes every
  `path:line` the failure names to the worker whose branch touched it, flagging a file several
  workers touched as an interaction point.

---

## 5. Turn Engine, Completion & Verification

`pool/runner/turn.rs` is the single turn engine. The implementer loop and the reviewer
loop both drive it, differing only in four knobs carried by `TurnConfig`: the command label
prefix, the steering message prefix, whether the orchestrator sentinels apply
(`apply_sentinels`), and how an LLM API error is handled (checkpoint + pause for the
orchestrator, versus end the review phase quietly). Every other step of a turn is shared, so
the two loops cannot drift:

```
steer drain ──► turn-limit warning (5 / 2 left) ──► auto-checkpoint (every 20 turns)
   │
   ▼
LLM call ──► command extraction ──► repetition detector ──► semaphores ──► sentinels
   │
   ▼
history push + in-memory state + registry row
```

- **History stays protocol-correct.** A tool-call turn is an assistant message with the
  call answered by a matching `tool` message; a prose turn is an assistant message answered
  by a user message. A response whose tool arguments carry no parseable command is *not*
  silently dropped: the assistant turn keeps its `tool_calls` and every id gets a
  `tool_result` saying so, which is the only shape the provider accepts.
- **Completion is explicit.** A run ends only when the *last* shell segment of a command
  `echo`s (or `printf`s) `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`. A substring match is not
  enough — `grep -rn COMPLETE_…`, a heredoc writing a fixture, or the prompt quoting its own
  sentinel all contain it without asking to finish. `cargo test && echo COMPLETE_…` counts.
- **Verify gate** (`pool::detect_verify_command` at dispatch, `TurnEngine::handle_completion`
  on the sentinel):
  - The `verify` dispatch argument is the gate. An explicit command is used verbatim, an
    explicit **empty string disables** it, and an absent argument auto-detects from the
    repository layout: `Cargo.toml` → `cargo build --all-targets && cargo test`,
    a `package.json` with a `test` script → `npm test`, `pyproject.toml` / `pytest.ini` →
    `pytest -q`, otherwise no gate.
  - The command runs through the same sandboxed, semaphore-gated bash path as any step.
    Exit 0 completes the run as verified; a non-zero exit pushes the output back to the
    model as `VERIFICATION FAILED` for another turn. After three failures the worker
    completes anyway, flagged unverified — the gate is a floor, not a dead end.
- **A command that outlives its budget becomes a background job, not a corpse.** The step
  timeout no longer kills a running command (`agent/jobs.rs`): the process group keeps
  running, its output keeps streaming to a capped log in the worker's private scratch, and the
  turn is answered with the job number. `echo WAIT_JOB <n>` blocks for up to 600 s without
  spending a turn and then reports the exit code plus the bounded tail; `echo KILL_JOB <n>`
  stops it. A job is confined exactly like the command that started it — same sandbox, same
  build-dir lease, same process group, so the reap sweep still reaches it — and the
  heavy-command admission permit moves into the job, so a job never outlives the build slot it
  was admitted with. An absolute ceiling (45 min, `JOB_MAX_SECS`) and the end of the worker
  both kill it. This replaces the `nohup … &` plus `sleep`-polling loop that cost one turn per
  poll.
- **Health metrics.** `WorkerMeta.metrics` carries the per-worker counters (turns used,
  extensions granted/refused, repeat blocks, stagnation nudges, loop pauses, verify runs and
  failures, final diff size). They move at the point each guard fires and are written with
  the registry row (`RegistryStatus` is an enum: `Running`, `Paused`, `Reviewing`,
  `Completed`, `Failed`, `Stopped`), so `status` and the monitor can report health without
  re-deriving it.
- **Kill preserves work**: `WorkerPool::kill` commits the worker's uncommitted changes onto
  its branch *before* aborting the task, so a manual kill costs at most the work since the
  last checkpoint.

---

## 6. Orchestrator Steering & Pause Protocol

Workers communicate status and request human/orchestrator intervention via shell sentinels:

- **Turn Expansion (`REQUEST_TURNS: <n>`)**:
  - The pool detects the sentinel and expands the remaining allowance, bounded by
    `extension_budget`: a worker may self-grant at most **half the budget its dispatch was
    given** (and never past the manifest ceiling). A request past that budget is refused
    with an explicit message and counted in `extensions_refused`; without the bound a
    confused model walks itself from 150 to 500 turns with nobody watching.
  - The implementer is warned proactively when 5 or 2 turns remain, so the request lands
    before the loop stops rather than after.
- **Orchestrator Guidance (`ASK_ORCHESTRATOR: <question>`)**:
  - Worker prints `ASK_ORCHESTRATOR: "..."` when facing ambiguous requirements or breaking decisions.
  - The worker transitions into `WorkerState::Paused { question, step, paused_at }`.
  - A tokio one-shot/MPSC channel pauses the task until the orchestrator calls the MCP `steer` action, passing guidance that resumes the worker seamlessly.
- **Repetition & stagnation**:
  - A command byte-identical to the previous turn's is answered with its own output rather
    than re-run, and three such turns in a row park the worker on the orchestrator.
  - The worktree is sampled every 10 turns; 30 turns with no change (fingerprint: `HEAD` id
    plus `git diff --stat HEAD`, so a commit counts as progress) injects a "stop exploring:
    make the edit, or ask" nudge.
  - A dispatch that already names the files to edit is sampled every turn instead: after
    `POOL_READ_ONLY_NUDGE_TURNS` (default 15) read-only turns the worker is told to write
    the first edit or ask what is missing, and once more at `POOL_READ_ONLY_ESCALATE_TURNS`
    (default: twice the first). Any change resets the streak, and it never fails the worker.
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

## 7. Persistent Role Memory

Every dispatch starts from the same static `SYSTEM_PROMPT`: nothing a previous run learned
survived its worktree. Role memory gives each *role* a durable notes file that is spliced
into the system prompt at the exact moment the prompt is built.

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

Both the implementer loop and the review phase build their prompt through the same
`build_system_prompt`, so the reviewer inherits the *reviewer's* memory (`nerd.md`) rather
than the implementer's — the two roles never contaminate each other.

Memory is **read-only** from the runtime's side: the server loads a role's notes and never
writes them, so a memory file is a checked-in or hand-edited artefact rather than something
a run can silently rewrite. Three invariants are load-bearing:

- **Optional.** `load_agent_memory` returns `None` — never an empty section — when
  the file is missing, blank, non-UTF-8, or is not even a file. A repository
  without `.agents/memory/` produces the exact static prompt it produced before
  this feature existed.
- **Bounded.** Injected memory is capped at `MAX_MEMORY_PROMPT_BYTES` (8 KiB),
  keeping the *newest* entries and cutting on a line boundary so the system
  instructions are never squeezed out by an ever-growing memory file. Memory is
  deliberately *not* memoized process-wide — nothing in the crate caches the
  manifest catalog or a rendered prompt — because it changes on disk between
  renders and must be re-read on every build.
- **Labelled.** The injected block is prefixed with `PERSISTENT ROLE MEMORY
  (from .agents/memory/):` so a model can tell its own notes from the static
  instructions.

The alias is reduced to a conservative `[a-z0-9_-]` slug (64 bytes, capped, `-` for every
other character) before it touches the filesystem, so a hostile `model` argument
(`../../etc/passwd`) collapses to `-etc-passwd.md` inside the memory directory and can never
escape it.

### Per-model instructions

`models.yaml` may attach an `instructions:` block to any model entry: rules appended to the
system prompt of every worker that runs on that model, under a
`MANDATORY DIRECTIVES FOR YOUR MODEL` header that frames them as directives rather than as
optional background. It is the operator's lever for correcting one model's habits (a small
model that reads files in many small ranges is told to read whole files) without touching
the repository's own instruction files, which every model would share.

Two properties are load-bearing:

- **One spelling, normalized once.** The block accepts a multi-line string or a list of
  strings (`serde` untagged) and deserializes into one ordered `Vec<String>`, one entry per
  line/bullet, with blank entries dropped. Nothing downstream branches on which form a
  catalog used.
- **Optional and bounded.** An entry without the block deserializes to `None`, so a catalog
  written before this field existed renders byte-identical prompts. A block above
  `MAX_MODEL_INSTRUCTIONS_BYTES` (4 KiB) is warned about by `validate` and cut by
  `normalize`, keeping the *head* entry by entry and marking the block truncated so the
  prompt never presents a cut rulebook as a whole one.

`build_system_prompt` appends it last — after the repository files and the role memory — and
the review phase resolves the *reviewer's* alias first, so a reviewer's habits never bleed
into the implementer's prompt.

---

## 8. Step-Log Retention, Residency & Health

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
  `Completed` / `Failed` records via a background reaper and lazily on the next
  `dispatch`. Fresh terminal records are kept, so a `collect` immediately after
  `wait: true` still resolves. `Running` and `Paused` records are never
  expired. The TTL bounds *memory* only: the evicted record leaves its registry
  row and its saved conversation in place, because a finished worker stays
  steerable for as long as its branch does. Those are retired by
  `WORKER_RETENTION_SECS` (default 7 days), or by `prune` once the branch is
  gone — never by the in-memory eviction.
- **No silent degradation**: `total_steps`, `logs_retained`, `logs_omitted` and
  `logs_dropped` are reported on every log-bearing response, with a
  `logs_truncation_notice` whenever history is missing.

Invariants are declared on the types themselves: `LogBuffer` documents
`len <= max_retained` and `bytes <= max_bytes` after every push, and
`expired_terminal_ids` absorbs clock skew via `saturating_sub` so a
future-dated record is never evicted.

---

## 9. Transport & Configuration

- **MCP stdio** (`mcp::protocol`): every frame is serde-derived JSON-RPC 2.0 — a
  request struct whose `id` stays a `RawValue` so it is echoed byte for byte,
  and one newline-terminated frame buffer per response. Inbound frames are
  refused above `MAX_FRAME_BYTES` (1 MiB) before the parse runs. The stdio wait
  loop polls a worker's lightweight progress snapshot every 500 ms and emits a
  `notifications/progress` frame whenever its step advances (plus one terminal
  frame); the CLI path passes no token and gets the same loop without them.
- **Configuration** (`config::env_parse`): every optional numeric setting is read
  through one helper that treats unset, blank and unparsable values alike — all
  return `None` and the caller falls back to its documented default (or clamps to
  a ceiling). Numeric settings, therefore, never fail startup on a typo; the
  `.env` discovery order and the manifest search order are documented in
  `.env.example`.
