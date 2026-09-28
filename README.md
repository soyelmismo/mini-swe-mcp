# mini-swe-mcp

High-throughput autonomous software engineering subagent orchestrator speaking the Model Context Protocol (MCP) over `stdio`. Designed to execute concurrent coding subagents in fully isolated Git worktrees with fine-grained concurrency governance.

---

## Key Features

- **MCP Stdio Protocol**: Seamlessly interfaces with Antigravity, Claude Desktop, Cursor, and any JSON-RPC 2.0 MCP client.
- **Two-Tier Semaphore Concurrency Governance**:
  - **Outer LLM Semaphore** (default 64): High-concurrency async dispatch for LLM inference.
  - **Inner Bash Semaphore** (default `cores / 2`, configurable via `BASH_CONCURRENT_LIMIT`): Guards machine resources by capping simultaneous build and test subprocesses, executing child shells with `nice -n 10` and capping Cargo/Make build threads.
- **Strict Git Worktree Isolation**:
  - Each subagent operates on its own dedicated worktree branch (`worker-<id>`).
  - No lock collisions or git state corruption across concurrent workers.
  - Sibling PID tracking prevents accidental pruning while subagents are running.
  - Automated RAII worktree cleanup on worker completion or failure.
- **Dual Tool Call Support & SSE Streaming**:
  - Full native OpenAI `tool_calls` priority with regex extraction fallback.
  - Server-Sent Events (`stream: true`) with instant TCP socket termination on cancellation.
- **Interactive Orchestrator Steering**:
  - Workers can pause execution and ask the orchestrator questions (`ASK_ORCHESTRATOR: <question>`).
  - Request dynamic turn extensions (`REQUEST_TURNS: <n>`).

---

## Installation & Setup

### Prerequisites

- Rust 1.85+ (Rust 2024 edition)
- Git 2.30+
- Linux or POSIX environment

### Building and Installing

```bash
# Clone the repository
git clone https://github.com/oracle/mini-swe-mcp.git
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
MAX_CONCURRENT_WORKERS=64                   # Optional, default: 64
BASH_CONCURRENT_LIMIT=2                     # Optional, default: cores / 2
```

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
```

#### 2. Monitor and List Workers
```bash
# List all active and past workers
mini-swe-mcp list

# Check status of a specific worker
mini-swe-mcp status <worker_id>
```

#### 3. Steer a Paused Subagent
```bash
mini-swe-mcp steer <worker_id> "Focus on unit tests first, skip integration tests for now."
```

#### 4. Collect Diff & Logs
```bash
# Collect diff and execution summary (automatically removes the worker record)
mini-swe-mcp collect <worker_id>
```

#### 5. Kill a Worker
```bash
mini-swe-mcp kill <worker_id>
```

#### 6. Prune Stale Worktrees
```bash
# Clean up orphaned branches and stale temporary worktrees whose processes died
mini-swe-mcp prune
```

#### 7. Inspect Model Manifest
```bash
mini-swe-mcp manifest
```

---

## Model Manifest (`models.yaml`)

Define custom subagent roles and aliases in `models.yaml`:

```yaml
default: ninja
models:
  ninja:
    id: combo:ninja
    role: "Fast, precise, low-token autonomous execution for targeted fixes and audits."
    temperature: 0.2
    max_turns: 50
  nerd:
    id: combo:nerd
    role: "Deep architectural reasoning, extensive documentation, and heavy refactors."
    temperature: 0.6
    max_turns: 100
```

---

## Verification & Testing

Run the full automated test suite:

```bash
# Unit tests + CLI tests + MCP JSON-RPC integration + WorktreeGuard tests
cargo test

# Strict clippy linting
cargo clippy -- -D warnings
```

---

## License

MIT OR Apache-2.0
