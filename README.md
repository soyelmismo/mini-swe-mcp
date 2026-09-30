# mini-swe-mcp

Autonomous software-engineering subagent orchestrator speaking the Model Context Protocol (MCP). One long-lived **hub daemon** owns a single worker pool and serves every orchestrator agent that connects to it; each subagent runs in its own Git worktree on a `worker-<id>` branch. The CLI and `--stdio` are both thin front ends to that daemon, and both start it automatically on first use.

## Install

```bash
cargo build --release      # target/release/mini-swe-mcp
cargo install --path .     # or put it on PATH
```

Requirements: `git`, an OpenAI-compatible endpoint, and `OPENAI_API_KEY`. A `.env` in the working directory is loaded automatically (`ENV_FILE` points elsewhere); `OPENAI_API_BASE` defaults to `https://api.openai.com/v1`.

Model aliases resolve from the first `models.yaml` found: `MODELS_FILE`, then `./models.yaml`, then `$XDG_CONFIG_HOME/mini-swe/models.yaml`, then next to the executable, then the built-in catalog. `mini-swe-mcp manifest` prints the resolved catalog; `DEFAULT_MODEL` names the alias used when a dispatch omits `--model`.

## Connect an agent

The server speaks MCP over stdio. `mini-swe-mcp --stdio` starts (or joins) the shared hub and proxies the connection to it, so many orchestrators share one pool.

Claude Code (`claude mcp add` writes the same block):

```json
{
  "mcpServers": {
    "mini-swe": {
      "command": "mini-swe-mcp",
      "args": ["--stdio"]
    }
  }
}
```

Any other stdio MCP client is configured the same way (`command` plus `args`). The CLI verbs below are the same calls the `worker` MCP tool exposes, for shells and scripts.

## Workflow

Dispatch, watch, review, steer, merge — one worker per focused concern.

### 1. dispatch

`dispatch` **always detaches**: it returns a `worker_id` immediately and never blocks. The worker starts on its own `worker-<id>` branch in an isolated worktree.

```bash
mini-swe-mcp dispatch "fix the flaky retry test in src/retry.rs; gate: cargo test retry"
```

`dispatch <task>` takes:

- `--model <alias>` — otherwise `DEFAULT_MODEL` or the manifest default.
- `--review-after <alias>` — run a reviewer phase over the produced diff before completing.
- `--repo <path>` — operate on a different repository.
- `--max-turns <n>` — turn budget.
- `--group <g>` — tag workers for `watch --group` and `list`.
- `--offline` — no outbound network during the run.
- `--verify <cmd>` — command the worker must pass before completing (auto-detected otherwise).

Write the task as ONE focused concern with the files in scope and an acceptance gate. Avoid parallel workers whose scopes share files.

### 2. watch

`watch` is the only way to wait. It reports an event when a watched worker **completes, fails, needs input or stalls**, and **replays events a late watcher missed**, so there is no `wait` action and no `--wait` flag.

```bash
mini-swe-mcp watch --follow             # every worker this agent owns
mini-swe-mcp watch <id> [...] --follow  # named workers
mini-swe-mcp watch --group build        # first event in the group, then return
```

Without `--follow` it prints the next event and returns; with `--follow` it streams until every watched worker is terminal. `--timeout <secs>` bounds the wait; `--json` emits the raw event stream.

### 3. review

```bash
mini-swe-mcp status <id> --json   # status, branch and diff stat
mini-swe-mcp collect <id>         # full transcript and final message
mini-swe-mcp logs <id>            # last commands and their output
```

Review the diff on `worker-<id>` before merging. `dispatch --review-after` adds an independent reviewer turn over the same worktree.

### 4. steer

`steer` corrects a **completed** worker and continues any **stopped** one (`failed`, `killed`, or `interrupted` by a crash). Either way it resumes on the preserved `worker-<id>` branch with the full conversation plus your message, on a fresh turn budget (`--max-turns <n>`, default 60).

```bash
mini-swe-mcp steer <id> "the retry logic still drops the last page"
```

Send corrections here instead of dispatching a second worker on the same files, and hand a worker a merge conflict the same way. A worker continued without a saved conversation (legacy) restarts cold on the same branch; only a missing branch is an error.

### 5. merge

Merge `worker-<id>` yourself once the diff is reviewed and the base branch is green. Before reporting completion the worker syncs the base branch into its worktree (`WORKER_SYNC_BASE=0` disables that).

## The hub

One daemon, many orchestrators.

- **One daemon.** The CLI and `--stdio` auto-start the hub when none is running. Its socket lives in `SWE_HUB_DIR` (default `<SWE_TEMP_DIR>/mini-swe-hub-<uid>`, private to your uid). `mini-swe-mcp daemon` runs it in the foreground; it exits after `HUB_IDLE_SECS` without clients.
- **Ownership & privacy.** A client may only read, steer, kill, collect and watch the workers it dispatched. Identity is `MINI_SWE_AGENT_ID`, or the MCP `initialize` client info when that is unset.
- **`--admin`.** The human operator's override on the CLI: act on workers owned by any agent. `list --all` requires it.
- **Crash recovery.** If the hub dies, its workers become `interrupted`; on restart it auto-resumes them from their durable conversation (`HUB_AUTO_RESUME=0` disables this).
- **`MINI_SWE_NO_DAEMON=1`.** No daemon: each process serves MCP and owns its own pool. Useful for tests and single-shot use, but its state is invisible to other clients.

## Sandbox

Every tool command is confined by the kernel by default:

- **Landlock** (filesystem) plus **seccomp** (syscalls). `MINI_SWE_LANDLOCK_ENFORCE=1` makes an unavailable Landlock fatal instead of best-effort; `SWE_DISABLE_LANDLOCK=1` turns Landlock off.
- **bubblewrap is opt-in** with `SWE_SANDBOX=bwrap`; it is no longer the default. On a kernel with neither Landlock nor seccomp the sandbox falls back to bwrap when installed and otherwise runs unconfined.
- `SWE_DISABLE_SANDBOX=1` disables all confinement.

## Resource management

- **Admission.** Heavy commands are classified and dosed: at most `BASH_BUILD_LIMIT` (default the core count) heavy builds at once, gated by free memory (`HUB_MEM_RESERVE_MB`, `HUB_BUILD_MEM_MB`). Light commands use `BASH_CONCURRENT_LIMIT` slots (default one per worker).
- **Fair scheduling.** `MAX_CONCURRENT_WORKERS` bounds the pool and `MAX_WORKERS_PER_AGENT` caps each agent so one orchestrator cannot starve the others; runnable workers are scheduled across agents.
- **Shared warm build dirs.** Workers share compiler/package caches under `SWE_CACHE_DIR`, so the second build is warm. `SWE_SHARED_CACHES` adds custom cache binds; `SWE_DISABLE_KACHE=1` (or `KACHE_DISABLED=1`) turns the kache layer off. Shared build slots are pruned by `HUB_TARGET_TTL_HOURS` / `HUB_TARGET_MAX_GB`.
- **History compaction.** Long conversations are compacted to stay inside `HISTORY_BUDGET_BYTES` while keeping the prompt, task and recent turns intact; `HISTORY_FULL_TURNS` restores a fixed-turn window and `HISTORY_KEEP_ALL_REASONING=1` retains all reasoning.
- **Log retention.** `WORKER_MAX_RETAINED_LOGS` and `WORKER_MAX_EMITTED_LOGS` bound per-worker logs.

## statusLine

`mini-swe-mcp status --line` prints one line summarising the pool, e.g. `⚙ 3 running · 1 needs input · 2 done`, for a Claude Code `statusLine` command; it prints nothing when the pool is idle. `MONITOR_WIDTH` overrides the width used by the full `monitor`.

## Environment variables

Defaults are what the code uses when the variable is unset.

| Variable | Default | Meaning |
| --- | --- | --- |
| `OPENAI_API_KEY` | — | API key; required by `dispatch`. |
| `OPENAI_API_BASE` | `https://api.openai.com/v1` | OpenAI-compatible base URL. |
| `DEFAULT_MODEL` | manifest default | Alias when a dispatch omits `--model`. |
| `MODELS_FILE` | — | Path to `models.yaml` (else `./models.yaml`, XDG, exe). |
| `ENV_FILE` | `.env` | Dotenv file loaded at startup. |
| `MAX_CONCURRENT_WORKERS` | `128` | Workers the pool can run at once. |
| `MAX_WORKERS_PER_AGENT` | `0` (unlimited) | Running workers one agent may hold. |
| `BASH_CONCURRENT_LIMIT` | `MAX_CONCURRENT_WORKERS` | Slots for light commands. |
| `BASH_BUILD_LIMIT` | core count | Concurrent heavy builds. |
| `HUB_MEM_RESERVE_MB` | `2048` | Free memory held back from admission. |
| `HUB_BUILD_MEM_MB` | `1536` | Memory one heavy build is assumed to need. |
| `BUILD_PARALLELISM` | granted jobs, else half the cores | Parallelism exported to a build. |
| `HUB_LLM_CONCURRENCY` | `0` (unlimited) | In-flight LLM requests hub-wide. |
| `LLM_MAX_RETRIES` | `6` | Attempts per completion. |
| `LLM_OUTAGE_PATIENCE_SECS` | `600` | How long a turn waits out an outage. |
| `COMMAND_HEAVY_TIMEOUT_SECS` | `600` | Timeout for a heavy command. |
| `COMMAND_LIGHT_TIMEOUT_SECS` | `120` | Timeout for a light command. |
| `COMMAND_TIMEOUT_SECS` | — | Overrides both timeout tiers. |
| `HISTORY_BUDGET_BYTES` | `160000` | Compaction budget for a conversation. |
| `HISTORY_FULL_TURNS` | — | Fixed-turn history window instead of a budget. |
| `HISTORY_KEEP_ALL_REASONING` | unset | `1` keeps every reasoning block. |
| `WORKER_SYNC_BASE` | `1` | Sync the base branch before completing; `0` disables. |
| `WORKER_TERMINAL_TTL_SECS` | `300` | How long a terminal worker stays listed. |
| `WORKER_MAX_RETAINED_LOGS` | `200` | Command logs retained (ceiling `1000`). |
| `WORKER_MAX_EMITTED_LOGS` | `40` | Log lines shown per worker (ceiling `500`). |
| `SWE_TEMP_DIR` | `/var/tmp`, else `$TMPDIR` | Root for worktrees, hub and caches. |
| `SWE_HUB_DIR` | `<SWE_TEMP_DIR>/mini-swe-hub-<uid>` | Hub socket and lock directory. |
| `HUB_IDLE_SECS` | `600` | Idle seconds before the hub exits. |
| `HUB_AUTO_RESUME` | `1` | Auto-resume interrupted workers; `0` disables. |
| `MINI_SWE_AGENT_ID` | — | Agent identity used for ownership. |
| `MINI_SWE_NO_DAEMON` | `0` | `1` runs the in-process server with no hub. |
| `MINI_SWE_WORKER_THREADS` | `4` | Tokio runtime worker threads. |
| `SWE_SANDBOX` | Landlock + seccomp | `bwrap` selects the bubblewrap backend. |
| `SWE_DISABLE_LANDLOCK` | `0` | `1` disables Landlock. |
| `MINI_SWE_LANDLOCK_ENFORCE` | `0` | `1` makes a missing Landlock fatal. |
| `SWE_DISABLE_SANDBOX` | `0` | `1` disables all confinement. |
| `SWE_CACHE_DIR` | `<SWE_TEMP_DIR>/swe-cache` | Shared compiler/package cache root. |
| `SWE_SHARED_CACHES` | — | Extra cache binds for the sandbox. |
| `SWE_DISABLE_KACHE` / `KACHE_DISABLED` | unset | `1` disables the kache layer. |
| `HUB_TARGET_TTL_HOURS` | `24` | Prune shared build slots older than this. |
| `HUB_TARGET_MAX_GB` | `40` | Size cap on shared build slots. |
| `MONITOR_WIDTH` | terminal size | Width used by `monitor` / `status`. |
| `COLUMNS` | terminal size | Fallback width when `MONITOR_WIDTH` is unset. |
| `XDG_CONFIG_HOME` | `$HOME/.config` | Base for `mini-swe/models.yaml`. |

## Load testing

`tests/load_test.rs` is an opt-in (`#[ignore]`) load test: it fans one real hub daemon out to many agents of many workers and asserts the hub keeps the machine busy without memory pressure. It is excluded from `cargo test` so the normal suite stays fast — run it explicitly:

```bash
# Quick smoke: 2 agents x 3 workers (the defaults), ~5 s.
cargo test --test load_test -- --ignored --nocapture

# Full load: 5 agents x 20 workers (100 total), ~75 s on 4 cores.
LOAD_AGENTS=5 LOAD_WORKERS_PER_AGENT=20 \
    cargo test --test load_test -- --ignored --nocapture
```

Every worker is driven by a fake OpenAI-compatible SSE server (`tests/common/fake_llm.rs`) that scripts three turns: a light command (`ls`), a heavy command (`cargo build` plus a bounded CPU burn, classified heavy by `is_heavy_command` so the pool's admission controller has to dose it), then the completion sentinel. The test dispatches each worker over the hub socket with a distinct agent identity (`agent-1`..`agent-5`) in `hub/hello`, samples the daemon's `/proc/<pid>/status` `VmHWM`/`VmRSS` every 250 ms, counts the heavy commands in flight by watching the daemon's children, and records the order workers finish in. It then asserts — and prints a one-screen report of:

- **all workers complete** — every dispatched worker reaches `completed`;
- **bounded daemon RSS** — peak `VmHWM` stays under a generous 300 MB;
- **dosed heavy commands** — the number in flight never exceeds `BASH_BUILD_LIMIT`;
- **fairness** — no agent finishes all of its workers before another agent has completed any of its own.

The shape is configurable through the environment: `LOAD_AGENTS`, `LOAD_WORKERS_PER_AGENT`, `LOAD_MAX_WORKERS` (`MAX_CONCURRENT_WORKERS`), `LOAD_MAX_HEAVY` (`BASH_BUILD_LIMIT`), `LOAD_HEAVY_SECS`, and `LOAD_TIMEOUT_SECS`.

Measured on a 4-core host (all other settings at their defaults):

| run         | wall  | peak daemon RSS (VmHWM) | peak heavy in flight | completed |
| ----------- | ----- | ---------- | ---------- | --------- |
| smoke 2 x 3 | 4.7 s | 20.1 MB                 | 4 / 4                | 6 / 6     |
| full 5 x 20 | 72 s  | 24.6 MB                 | 4 / 4                | 100 / 100 |

In the full run the admission controller logged 108 heavy-command admission waits for the four build slots and the completion order interleaved all five agents, so the daemon stayed busy and bounded under 100 concurrent workers.

---

## License

GNU General Public License v3.0 (GPL-3.0-or-later)
