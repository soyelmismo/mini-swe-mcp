# mini-swe-mcp

High-throughput autonomous software engineering subagent orchestrator speaking the Model Context Protocol (MCP) over `stdio`. Designed to execute concurrent coding subagents in fully isolated Git worktrees with fine-grained concurrency governance.

---

## Key Features

- **MCP Stdio Protocol**: Seamlessly interfaces with Antigravity, Claude Desktop, Cursor, and any JSON-RPC 2.0 MCP client.
- **Three-Tier Semaphore Concurrency Governance**:
  - **Worker Pool Semaphore** (default 64): High-concurrency async dispatch for LLM inference.
  - **Bash Command Semaphore** (default one slot per worker, tunable via `BASH_CONCURRENT_LIMIT`): caps simultaneous bash steps; heavy commands (`cargo`, `make`, `pytest`, compilers, …) also take a slot from the **Build Semaphore** (default `cores / 2`, `BASH_BUILD_LIMIT`).
  - Child shells run with `nice -n 10` and a per-command thread cap (`BUILD_PARALLELISM`, default `cores / 2`).
- **Strict Git Worktree Isolation**:
  - Each subagent operates on its own dedicated worktree branch (`worker-<id>`).
  - No lock collisions or git state corruption across concurrent workers.
  - Sibling PID tracking prevents accidental pruning while subagents are running; a stale worktree has its uncommitted work salvaged onto the worker branch before it is removed.
  - Automated RAII worktree cleanup on worker completion or failure, keeping the branch whenever it carries commits.
- **Native Tool Calls & SSE Streaming**:
  - Native OpenAI `tool_calls` with a fenced-code-block fallback for models that do not use them.
  - Server-Sent Events (`stream: true`) with instant TCP socket termination on cancellation.
- **Verification Gate**:
  - A dispatch can name a `verify` command (`--verify "cargo test"`); when it is absent one is auto-detected from the repository layout.
  - The gate runs through the same sandboxed bash path before a completion sentinel is honoured, and its failure output is fed back to the model.
- **Interactive Orchestrator Steering**:
  - Workers can pause execution and ask the orchestrator questions (`ASK_ORCHESTRATOR: <question>`).
  - Request dynamic turn extensions (`REQUEST_TURNS: <n>`), bounded to half the dispatch's own budget.
  - Repetition and stagnation detectors nudge or park a worker that stops making progress.
- **Persistent Role Memory**:
  - `.agents/memory/<alias>.md` is read into the system prompt on every dispatch, per role; the runtime never writes it.

---

## Installation & Setup

### Prerequisites

- Rust 1.85+ (Rust 2024 edition)
- Git 2.30+
- Linux or POSIX environment

### Building and Installing

```bash
# Clone the repository
git clone https://github.com/soyelmismo/mini-swe-mcp.git
cd mini-swe-mcp

# Build optimized release binary
cargo build --release

# Install to ~/.local/bin
install -m 755 target/release/mini-swe-mcp ~/.local/bin/mini-swe-mcp
```

Ensure `~/.local/bin` is in your `PATH`.

The release profile is tuned for this I/O-bound stdio daemon
(see `audits/opt_10_cargo_codegen.md`):

```toml
[profile.release]
opt-level = 2        # smaller and faster to build than 3 for a stdio/JSON daemon
lto = "thin"         # 34% faster release link than fat LTO for a ~5% size cost
codegen-units = 1    # deterministic single-core thin-LTO link
panic = "abort"      # no unwinding tables (~473 KB smaller)
strip = true         # drops ~9.9 MB of debug symbols
```

Measured result: ~34% faster release compilation and a smaller binary than the
previous `opt-level = 3` / fat-LTO configuration.

### Configuration (`.env`)

Copy `.env.example` to `.env` and set your credentials:

```bash
cp .env.example .env
```

Configuration resolution follows a 4-tier cascading precedence:
1. `./.env` (current working directory or parent directory)
2. `$XDG_CONFIG_HOME/mini-swe/.env` (or `~/.config/mini-swe/.env`)
3. `.env` alongside the executable
4. Custom path specified via `ENV_FILE=/path/to/.env`

Required environment variables:
```bash
OPENAI_API_KEY=sk-...
OPENAI_API_BASE=https://api.openai.com/v1   # Optional, default: https://api.openai.com/v1
DEFAULT_MODEL=ninja                         # Optional, default: ninja
OPENAI_API_KEY=sk-...
MAX_CONCURRENT_WORKERS=64                   # Optional, default: 64
BASH_CONCURRENT_LIMIT=4                     # Optional, default: one slot per worker
BASH_BUILD_LIMIT=2                          # Optional, default: cores / 2
BUILD_PARALLELISM=2                         # Optional, default: cores / 2
COMMAND_TIMEOUT_SECS=600                    # Optional, default: 600 heavy / 120 light
WORKER_MAX_RETAINED_LOGS=200                # Optional, default: 200 (ceiling 1000)
WORKER_MAX_EMITTED_LOGS=40                  # Optional, default: 40 (ceiling 500)
WORKER_TERMINAL_TTL_SECS=300                # Optional, default: 300
```

The three `WORKER_*` variables bound the per-worker step-log memory and the
lifetime of finished worker records; see `.env.example` for the full contract
and [Step-Log Retention](#step-log-retention) below. Every optional number is
parsed by one helper: unset, blank and non-numeric values all fall back to the
documented default, and a value above a ceiling is clamped rather than rejected.

---

## MCP Server Integration

### Antigravity / Claude Desktop / Cursor

Add `mini-swe-mcp` to your MCP configuration file (e.g. `~/.config/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "mini-swe": {
      "command": "mini-swe-mcp",
      "args": ["--stdio"],
      "env": {
        "OPENAI_API_KEY": "sk-your-openai-api-key",
        "OPENAI_API_BASE": "https://api.openai.com/v1",
        "DEFAULT_MODEL": "ninja"
      }
    }
  }
}
```

---

## CLI Direct Usage

`mini-swe-mcp` can be driven directly from the terminal without an MCP client:

### Flags
- `mini-swe-mcp --version` / `-V`: Print binary version.
- `mini-swe-mcp --help` / `-h`: Print command line help.

### Actions

#### 1. Dispatch a Subagent
```bash
# Async dispatch (returns worker_id immediately)
mini-swe-mcp dispatch "Implement unit tests for src/config.rs" --model ninja --repo .

# Synchronous dispatch with interactive steering (--wait)
mini-swe-mcp dispatch "Refactor auth middleware" --model nerd --repo . --wait

# Offline dispatch: every bash step runs with no network egress
mini-swe-mcp dispatch "Rename the internal helper" --model ninja --repo . --offline

# Pin the verify gate; --verify "" disables it for this dispatch
mini-swe-mcp dispatch "Fix the flaky parser test" --model ninja --repo . \
  --verify "cargo test --all-targets" --max-turns 80
```

Full usage:

```
dispatch <task> [--model <model>] [--review-after <model>] [--repo <repo>]
          [--wait] [--max-turns <n>] [--group <group>] [--offline]
          [--verify <cmd>]
```

The `--offline` flag is the CLI spelling of the tool's optional `network`
property — see [Network Policy](#network-policy). `--verify` is the spelling of
the `verify` property; see [Verification Gate](#verification-gate).

#### 2. Monitor and List Workers
```bash
# List all active and past workers
mini-swe-mcp list

# Check status of a specific worker
mini-swe-mcp status <worker_id>
```

#### 3. Steer a Running or Paused Subagent
```bash
mini-swe-mcp steer <worker_id> "Focus on unit tests first, skip integration tests for now."
```

`steer` works from **any terminal, including one that did not dispatch the
worker**. A worker running in another `mini-swe-mcp` process is steered through
a per-worker mailbox file at `<base>/swe-wt-<worker_id>.steer`, where `<base>` is
`/var/tmp` by default or `$SWE_TEMP_DIR` when set. The message is appended
atomically and picked up by the worker on its next step, so guidance sent to a
worker dispatched with `--wait` in a different shell is never lost.

The mailbox is a JSON-lines file (one `{message, sent_at, pid}` record per line),
so multi-line messages — a pasted stack trace, a diff hunk — survive intact. The
worker drains it once per turn in both the implementation loop and the review
loop, and deletes it on exit.

```bash
# Terminal A: dispatch and block
mini-swe-mcp dispatch "Refactor auth middleware" --model nerd --repo . --wait

# Terminal B: steer that worker mid-flight
mini-swe-mcp steer <worker_id> "Skip the integration tests for now."
```

#### 4. Collect Diff & Logs
```bash
# Collect diff and execution summary (automatically removes the worker record)
mini-swe-mcp collect <worker_id>
```

#### 5. Inspect a Worker's Step Logs
```bash
# Read a live worker's retained (bounded) step history without collecting it
mini-swe-mcp logs <worker_id>
```

The response always carries `total_steps`, `logs_retained`, `logs_omitted` and
`logs_dropped`, plus a `logs_truncation_notice` whenever part of the history is
missing — see [Step-Log Retention](#step-log-retention).

#### 6. Kill a Worker
```bash
mini-swe-mcp kill <worker_id>
```

#### 7. Reap Expired Worker Records
```bash
# Evict terminal worker records whose TTL expired (also runs in the background)
mini-swe-mcp reap
```

#### 8. Prune Stale Worktrees
```bash
# Clean up orphaned branches and stale temporary worktrees whose processes died
mini-swe-mcp prune
```

#### 9. Inspect Model Manifest
```bash
mini-swe-mcp manifest
```

---

## Network Policy

Every `dispatch` may declare a network policy for its worker:

```json
{ "action": "dispatch", "task": "Refactor the parser", "network": "offline" }
```

| Value | Effect |
|---|---|
| `allow` (default) | Steps run with the host's normal connectivity |
| `offline` | Each bash step runs inside its own network namespace with no egress |

`network` is optional. When it is absent the dispatch default for the resolved
model applies: its `policy.network` from `models.yaml` if the model declares one,
otherwise `allow`. An explicit argument always wins. Any other value — or a
non-string — is rejected as a tool error instead of being silently downgraded: a
request that asked for isolation must never quietly get connectivity back.

`offline` is enforced at the kernel level with `unshare -n`, with no containers
and no external firewall. Inside the namespace there is no interface and no
route, so `curl`, `git fetch` or `cargo add` fail immediately with
`Network is unreachable` rather than blocking out a connect timeout — the step
returns fast and the model can adapt on its next turn. The worktree, the build
environment, the command timeout and the output plumbing are all unchanged; the
policy also covers the reviewer phase of a `review_after` dispatch.

Use it for pure refactors, analysis, renames or formatting passes, where an
outbound request would be a defect rather than a feature. Note that `cargo
build` / `cargo test` still work offline as long as their dependencies are
already vendored or present in the local registry cache.

If the host forbids creating network namespaces (an unprivileged container
without `CAP_SYS_ADMIN`), the wrapper is still applied and the step fails
loudly — a policy that quietly did not apply would be worse than one that is
visibly unavailable.

---

## Verification Gate

A dispatch may declare the command that decides whether a finished worker is
actually finished:

```bash
mini-swe-mcp dispatch "Fix the parser" --model ninja --repo . --verify "cargo clippy --all-targets -- -D warnings && cargo test"
```

The gate is resolved once, at dispatch:

| `verify` argument | Gate |
|---|---|
| a command | used verbatim |
| an empty string (`--verify ""`) | disabled for this dispatch |
| absent | auto-detected: `Cargo.toml` -> `cargo build --all-targets && cargo test`; a `package.json` with a `test` script -> `npm test`; `pyproject.toml` / `pytest.ini` -> `pytest -q`; otherwise none |

The gate runs through the same sandboxed, semaphore-gated bash path as any other
step, and only runs when the worker tries to finish (`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`).
A non-zero exit is pushed back to the model as `VERIFICATION FAILED` with the
output, so the worker gets another turn to fix it; after three failed
verifications the run completes anyway and is reported as *not verified* rather
than looping forever. `verify_runs` and `verify_failures` are reported in the
worker's health metrics.

---

## Step-Log Retention

Every bash step a worker runs appends an `AgentStepLog` to its in-memory history.
That history is bounded on three independent axes so a long-running server's
residency tracks *concurrent* workers, not historical ones:

| Axis | Variable | Default | What it bounds |
|---|---|---|---|
| Entries per worker | `WORKER_MAX_RETAINED_LOGS` | 200 (max 1000) | Sliding window; the oldest entries are evicted |
| Bytes per worker | derived from the window | ~430 KiB | Each entry stores a `<= 64 B` command summary and a `<= 2048 B` output excerpt |
| Entries per response | `WORKER_MAX_EMITTED_LOGS` | 40 (max 500) | A single `collect` / `dispatch --wait` / `logs` reply |
| Terminal record TTL | `WORKER_TERMINAL_TTL_SECS` | 300 | How long a finished worker is kept before eviction |

The truncation marker (`... [N bytes truncated]`) is charged *against* the
2048-byte budget, so the stored `output` is at most 2048 bytes rather than
2048-plus-marker.

Nothing degrades silently: every log-bearing response reports
`total_steps`, `logs_retained`, `logs_omitted` and `logs_dropped`, and adds a
`logs_truncation_notice` when part of the history is not shown. Use
`logs <worker_id>` to page through the retained window of a live worker, and
`reap` (or the background reaper) to reclaim finished workers.

---

## Model Manifest (`models.yaml`)

Define custom subagent roles and aliases in `models.yaml`:

```yaml
default: ninja
models:
  ninja:
    id: combo:ninja
    role: "Fast subagent. Best for repo exploration, running tests, syntax bugfixes, and focused edits."
    temperature: 0.2
    max_turns: 150
    policy:
      network: "allow"
  nerd:
    id: combo:nerd
    role: "Deep reasoning subagent. Best for root-cause debugging, complex multi-file logic, and architecture changes."
    temperature: 0.6
    max_turns: 200
    policy:
      network: "allow"
```

### Execution policy

A model entry may declare the execution policy its workers run under. The
`policy:` block and its field are both optional, and an entry without one keeps
the runtime defaults, so a manifest written before this block existed is still
valid:

| Field | Accepted values | Meaning |
|-------|-----------------|---------|
| `network` | `"offline"` \| `"allow"` | The dispatch default for the worker: `offline` runs every bash step in its own network namespace (no egress); `allow` keeps the host's normal connectivity. Same spelling as the `network` argument of the `worker` tool, and an explicit argument on the dispatch wins over the manifest. |

Values are trimmed and matched case-insensitively, so `Offline` and
`" offLINE "` are accepted. An **unknown value never fails the load**: it is
reported as a warning naming the exact text you wrote and is repaired to the
*restrictive* default (`offline`), so a typo can never quietly widen a sandbox.
Warnings are emitted on startup, alongside the other manifest checks (temperature
bounds, turn budgets, duplicate ids).

`fs:` is **not supported**: it is ignored with a startup warning, because
filesystem confinement is not negotiated per model. A worker always writes inside
its own worktree and build directory, confined by the sandbox rather than by the
manifest.

---

## Persistent Role Memory

A subagent conversation is volatile: every dispatch starts from the same static
system prompt. Role memory gives each *role* a durable notes file at the repo
root that is loaded into the system prompt when the worker starts:

```
.agents/
└── memory/
    ├── ninja.md    # fast-execution lessons (compile/test loop, minimal diffs)
    └── nerd.md     # architecture & review lessons (root cause, invariants)
```

The file is keyed by the **alias** from `models.yaml`, so `ninja` and `nerd` have
separate memories and neither leaks into the other's prompt. Memory is read fresh
on every dispatch (never memoized), so a note written by one run is visible to the
next:

- **Absent, blank or unreadable** -> no memory section at all; the system prompt
  is byte-identical to the pre-memory behaviour. Nothing to configure, nothing to
  break.
- **Bounded** -> at most 8 KiB of memory is injected (the newest entries), so a
  memory file that grows forever can never squeeze out the instructions.
- **Traversal-safe** -> the alias is reduced to an `[a-z0-9_-]` slug before it
  touches the filesystem, so a `model` argument can never read outside
  `.agents/memory/`.

Memory is **read-only** from the server's side: it loads a role's notes and never
writes them, so the files stay a reviewable artefact of the repository rather
than something a run can silently rewrite. Edit `.agents/memory/<alias>.md` (or
ask a subagent to) and the next dispatch picks it up.

---

## Verification & Testing

Run the full automated test suite:

```bash
# Unit tests + CLI tests + MCP JSON-RPC integration + WorktreeGuard tests
cargo test

# Strict clippy linting over every target
cargo clippy --all-targets -- -D warnings
```

---

## License

GNU General Public License v3.0 (GPL-3.0-or-later)
