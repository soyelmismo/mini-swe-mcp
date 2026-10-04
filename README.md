# mini-swe-mcp

Autonomous software-engineering subagent orchestrator speaking the Model Context Protocol (MCP). One long-lived **hub daemon** owns a single worker pool and serves every orchestrator agent that connects to it; each subagent runs in its own Git worktree on a `worker-<id>` branch. The CLI and `--stdio` are both thin front ends to that daemon, and both start it automatically on first use.

## Install

```bash
cargo build --release      # target/release/mini-swe-mcp
cargo install --path .     # or put it on PATH
```

Requirements: `git`, an OpenAI-compatible endpoint, and `OPENAI_API_KEY`. A `.env` in the working directory is loaded automatically (`ENV_FILE` points elsewhere); `OPENAI_API_BASE` defaults to `https://api.openai.com/v1`.

Model aliases resolve from the first `models.yaml` found: `MODELS_FILE`, then `./models.yaml`, then `$XDG_CONFIG_HOME/mini-swe/models.yaml`, then next to the executable, then the built-in catalog. `mini-swe-mcp manifest` prints the resolved catalog; `DEFAULT_MODEL` names the alias used when a dispatch omits `--model`.

### Per-model instructions

Any model entry may carry an optional `instructions:` block: extra rules appended to the system prompt of every worker that runs on that model, after the repository's own instruction files and the role memory. It is how you correct a model's habits from the catalog -- a small model that reads files in many small ranges is told to read whole files. Both spellings mean one instruction per line:

```yaml
models:
  small:
    id: combo:small
    instructions: |
      Read whole files instead of many small ranges.
      Run the cheap gate before the full suite.
  # equivalently:
  small:
    id: combo:small
    instructions:
      - Read whole files instead of many small ranges.
      - Run the cheap gate before the full suite.
```

The block is optional: an entry without one behaves exactly as before. The review phase uses the block of the *reviewer's* model (`--review-after <alias>`), so review habits never leak into the implementer's prompt. It is bounded at 4 KB per model: beyond that the tail is dropped, the cut is marked in the prompt and the manifest load warns. `mini-swe-mcp manifest` prints the instruction count per model.

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

## Repository instructions

A dispatched worker reads the target repository's own agent rules from its root
and appends them to its system prompt: `AGENTS.md`, `CLAUDE.md`, `GEMINI.md`,
`.github/copilot-instructions.md` and `.cursorrules`. Each existing file is
included once, deduplicated by content. The injected block is bounded (16 KB,
marked when truncated) and only regular files inside the repository are read, so
a symlink pointing outside the repo is ignored.

## Workflow

Dispatch, watch, review, steer, merge — one worker per focused concern.

### 1. dispatch

`dispatch` **always detaches**: it returns a `worker_id` immediately and never blocks. The worker starts on its own `worker-<id>` branch in an isolated worktree.

```bash
mini-swe-mcp dispatch "fix the flaky retry test in src/retry.rs; gate: cargo test retry"
```

`dispatch <task>` takes:

- `--model <alias>` — otherwise `DEFAULT_MODEL` or the manifest default.
- `--review-after <model>[:<mode>]` — run a reviewer phase over the produced diff before completing (`quality`, `security`, or a `review_modes:` entry from `models.yaml`; e.g. `review_modes: {perf: {checklist: "Check for N+1 queries.", model: nerd}}` selected via `--review-after nerd:perf`). A declared mode carries a `checklist` and an optional `model`, and runs only when you select it explicitly. The one automatic review is the adversarial security pass: a diff touching a path listed in the `## Sensitive paths` section of `AGENTS.md` gets `--review-after <model>:security` on its own.
- `--repo <path>` — operate on a different repository.
- `--max-turns <n>` — turn budget.
- `--group <g>` — tag workers for `watch --group` and `list`.
- `--offline` — no outbound network during the run.
- `--verify <cmd>` — command the worker must pass before completing (auto-detected otherwise).

Write the task as ONE focused concern with the files in scope and an acceptance gate. Dispatch independent tasks in parallel -- many workers at once is the intended use; each worker integrates the latest base branch and resolves conflicts before completing. Split work so two workers do not rewrite the same function at the same time.

### 2. watch

`watch` is the only way to wait, and you should run it in the **background**: it blocks until a watched worker **completes, fails, needs input or stalls**, prints it and exits -- so the host CLI wakes you when the command ends -- and **replays events a late watcher missed** first.

```bash
mini-swe-mcp watch --follow             # every worker this agent owns
mini-swe-mcp watch <id> [...] --follow  # named workers
mini-swe-mcp watch --group build        # first event in the group, then return
```

Without `--follow` it prints the next event and returns; with `--follow` it streams until every watched worker is terminal. `--timeout <secs>` bounds the wait; `--json` emits the raw event stream.

An agent with no shell can call the `watch` action instead, passing `timeout_secs` below its host's tool deadline and calling it again on `no_event`. A Claude Code session started with channels enabled also receives the same events as push notifications.

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

Or let `merge` do the whole sequence in one command:

```bash
mini-swe-mcp merge <id>            # trial merge, gate, merge --no-ff, cleanup
mini-swe-mcp merge <id> --no-delete  # keep the branch afterwards
```

It merges into the base branch recorded for the worker, and refuses -- changing nothing -- while the worker still runs, while the repository has another branch checked out, while a file the merge would touch has uncommitted changes (untracked and unrelated files are left alone), or while the branch conflicts, in which case it prints the conflicting files and the `steer` that sends them back. The verify gate (the worker's own command, or the auto-detected one) runs on the merge result in a throwaway worktree and is skipped when the branch already contains the base tip and the worker's last verify passed. On success it prints one line naming the merge commit and what it cleaned up. It never pushes.

### 6. discard

Not every stopped worker is worth landing. A failed consolidator whose gate can never run, a duplicate dispatch, a branch whose work was abandoned: `discard` removes such a worker in one command instead of by hand.

```bash
mini-swe-mcp discard <id>   # branch, row, history, steer files, round base, worktree
```

It retires the worker through the same path a merge uses, so nothing is left to clean up: the `worker-<id>` branch, the registry row, the history JSONL, the steering mailbox and steer-source, the pinned round base, the watch acknowledgements and the worktree leftovers all go. It merges nothing and runs no gate, so the base branch does not move. Like `kill` it is owner-only, and a **running or paused** worker is refused with a hint to `kill` it first — a discard deletes unmerged work with no gate, so it must never be a quiet way to stop a worker that is still producing. Nothing it removes can be recovered: read `collect <id>` first if the result is still wanted.

## The hub

One daemon, many orchestrators.

- **One daemon.** The CLI and `--stdio` auto-start the hub when none is running. Its socket lives in `SWE_HUB_DIR` (default `<SWE_TEMP_DIR>/mini-swe-hub-<uid>`, private to your uid). `mini-swe-mcp daemon` runs it in the foreground; it exits after `HUB_IDLE_SECS` without clients.
- **Ownership & privacy.** A client may only read, steer, kill, discard, collect and watch the workers it dispatched. Identity is `MINI_SWE_AGENT_ID`, else a `MINI_SWE_WATCH_TOKEN`, else the agent's **host process plus the session inside it**, else the MCP `initialize` client info.
- **One identity per session.** The identity is the agent's host process — the first ancestor of the client that is not a shell, a wrapper or a service manager (`mini-swe-mcp <- bash <- claude` resolves to `claude`), named `host:<comm>:<pid>:<starttime>`. The MCP connection and the agent's shell commands share it, so a `mini-swe-mcp watch` in the shell sees the workers its own MCP connection dispatched, and two hosts never share workers. It survives MCP reconnects and hub restarts while the host lives; the start time keeps a recycled pid from colliding. A process daemonized with `setsid -f` is reparented to the user's `systemd`, which is never named as a host.
- **Sessions inside one host.** One host process is not always one session: opencode v2 runs a tab per session inside one process, over one shared MCP connection. When the session is known it qualifies the host — `host:<comm>:<pid>:<starttime>/session:<id>` — so two tabs are two agents. opencode v2 sends it on every call as `CallToolRequest.params._meta.sessionID`, which is read per call and never cached on the connection; the handshakes read it from the environment instead (`CLAUDE_CODE_SESSION_ID`, then `OPENCODE_SESSION_ID`, then `MINI_SWE_SESSION_ID`). A host with no session information keeps the plain host identity. `mini-swe-mcp whoami` prints the identity and how it was derived.
- **Watch tokens.** A shell cannot know its session — none of those variables reaches the agent's `bash` tool — so every `dispatch` and `steer` answer carries a `watch_command`: the exact command that waits on *your* workers, e.g. `MINI_SWE_WATCH_TOKEN=<32 hex> mini-swe-mcp watch`. The token is bound to the caller's identity, one per identity, created on first need and stored `0600` in the hub directory so it survives a daemon restart; a CLI presenting it acts as exactly that identity and never as `admin`. `mini-swe-mcp whoami` reports it as the derivation.
- **`--admin`.** The human operator's override on the CLI: act on workers owned by any agent. `list --all` requires it.
- **Crash recovery.** If the hub dies, its workers become `interrupted`; on restart it auto-resumes them from their durable conversation (`HUB_AUTO_RESUME=0` disables this).
- **`MINI_SWE_NO_DAEMON=1`.** No daemon: each process serves MCP and owns its own pool. Useful for tests and single-shot use, but its state is invisible to other clients.

## Sandbox

Every tool command is confined by the kernel by default:

- **Landlock** (filesystem) plus **seccomp** (syscalls). `MINI_SWE_LANDLOCK_ENFORCE=1` makes an unavailable Landlock fatal instead of best-effort; `SWE_DISABLE_LANDLOCK=1` turns Landlock off.
- **bubblewrap is opt-in** with `SWE_SANDBOX=bwrap`; it is no longer the default. On a kernel with neither Landlock nor seccomp the sandbox falls back to bwrap when installed and otherwise runs unconfined.
- `SWE_DISABLE_SANDBOX=1` disables all confinement.

## Differential verification

A verify gate that only runs in the sandbox proves the suite is hermetic *against the sandbox*, not against the shell the code is later verified in. So when the gate passes in the canonical environment the same command runs once more in a **divergent** one: the dispatcher's ambient variables (filtered by the sandbox's secret filter, so no key or token ever crosses the wire), a fresh `HOME` and `TMPDIR`, and a `TZ` shifted far from the host's. Same sandbox, same worktree, same command.

If the second run fails the completion is refused and the model is told which variables differ, that `HOME`/`TMPDIR`/`TZ` differ, and what failed. Around both runs the harness audits what the suite left behind — refs and worktrees in the shared repository, processes the reap sweep had to kill, new files in the main checkout — and refuses with the exact list, cleaning up what it can. `WORKER_DIVERGENT_VERIFY=0` turns the second run off. Nothing here assumes a language: the gate is whatever verify command the dispatch or the auto-detection chose.

## Resource management

- **Admission.** Heavy commands are classified and dosed: at most `BASH_BUILD_LIMIT` (default the core count) heavy builds at once, gated by free memory (`HUB_MEM_RESERVE_MB`, `HUB_BUILD_MEM_MB`) and by Linux pressure-stall information — CPU `some avg10` (`HUB_CPU_PRESSURE_MAX`), memory and IO `full avg10` (`HUB_MEM_PRESSURE_MAX`, `HUB_IO_PRESSURE_MAX`). PSI measures the time tasks actually stalled on a resource, so it is not fooled by I/O wait or unrelated processes the way the 1-minute load average is; when `/proc/pressure` is unavailable the controller falls back to that load average. Light commands use `BASH_CONCURRENT_LIMIT` slots (default one per worker).
- **Fair scheduling.** `MAX_CONCURRENT_WORKERS` bounds the pool and `MAX_WORKERS_PER_AGENT` caps each agent so one orchestrator cannot starve the others; runnable workers are scheduled across agents.
- **Shared warm build dirs.** Workers share compiler/package caches under `SWE_CACHE_DIR`, so the second build is warm. `SWE_SHARED_CACHES` adds custom cache binds; `SWE_DISABLE_KACHE=1` (or `KACHE_DISABLED=1`) turns the kache layer off. Shared build slots are pruned by `HUB_TARGET_TTL_HOURS` / `HUB_TARGET_MAX_GB`, and a slot that grew past `MINI_SWE_TARGET_SLOT_MAX_GIB` is emptied when the next worker leases it.
- **Any ecosystem.** The caches, the sandbox grants and the parallelism caps are not Rust-specific: each tool is pointed at a shared cache under `SWE_CACHE_DIR` through its own variable, and the sandbox grants that directory (never the operator's home) with the rights the tool needs.

| Ecosystem | Shared cache (under `SWE_CACHE_DIR`) | Variable the child sees | Credentials kept out |
| --- | --- | --- | --- |
| Rust | `kache`, `~/.cargo`, `~/.rustup` (read-only) | `CARGO_HOME`, `RUSTUP_HOME` | `~/.cargo/credentials.toml` |
| Node | `node/npm`, `node/yarn`, `node/pnpm`, `node/pnpm-store` | `npm_config_cache`, `YARN_CACHE_FOLDER`, `PNPM_HOME`, `PNPM_STORE_DIR` | `~/.npmrc`, `~/.yarnrc` |
| Python | `python/pip`, `python/uv` | `PIP_CACHE_DIR`, `UV_CACHE_DIR` | `~/.pypirc`, `~/.netrc` |
| Go | `go/build`, `go/mod` | `GOCACHE`, `GOMODCACHE` | — |
| JVM (Maven) | `java/m2` | `MAVEN_OPTS=-Dmaven.repo.local=...` | `~/.m2/settings.xml` |
| JVM (Gradle) | `java/gradle` | `GRADLE_USER_HOME` | `~/.gradle/gradle.properties` |

The caches are writable in the sandbox (Landlock and bubblewrap alike) because builds must be able to populate them; the credential files beside them are never granted, so an npm token or a Maven `settings.xml` password stays unreadable. `SWE_ALLOW_TOOLCHAIN_CREDENTIALS=1` is the operator's opt-in to expose them. Parallelism uses the same granted job count as Cargo for every tool that has a variable for it: `GOMAXPROCS` (Go), `MAVEN_OPTS=-T` (Maven), `GRADLE_OPTS=-Dorg.gradle.workers.max` (Gradle), `PYTEST_XDIST_AUTO_NUM_WORKERS` (pytest-xdist), plus the existing `CARGO_BUILD_JOBS`, `MAKEFLAGS`, `CMAKE_BUILD_PARALLEL_LEVEL` and the BLAS/OpenMP thread caps.
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
| `HUB_CPU_PRESSURE_MAX` | `60` | CPU `some avg10` ceiling (PSI, percent). |
| `HUB_MEM_PRESSURE_MAX` | `10` | Memory `full avg10` ceiling (PSI, percent). |
| `HUB_IO_PRESSURE_MAX` | `40` | IO `full avg10` ceiling (PSI, percent). |
| `BUILD_PARALLELISM` | granted jobs, else half the cores | Parallelism exported to a build. |
| `WORKER_BUILD_DEBUG` | `0` | `1` keeps Cargo's debug info and incremental state in worker builds. |
| `CARGO_PROFILE_DEV_DEBUG`, `CARGO_PROFILE_TEST_DEBUG`, `CARGO_INCREMENTAL` | `0` | Forced on worker builds to cut target-dir I/O; an operator-exported value is respected. |
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
| `MINI_SWE_AGENT_ID` | — | Agent identity used for ownership; overrides the host, the session and any watch token. |
| `CLAUDE_CODE_SESSION_ID` | — | Session inside the host process; first of the session variables checked. |
| `OPENCODE_SESSION_ID` | — | opencode v2's session variable; checked second. |
| `MINI_SWE_SESSION_ID` | — | Generic session variable for any other host; checked last. |
| `MINI_SWE_WATCH_TOKEN` | — | Acts as the identity this token was minted for (see `watch_command`); never `admin`. |
| `MINI_SWE_NO_DAEMON` | `0` | `1` runs the in-process server with no hub. |
| `MINI_SWE_WORKER_THREADS` | `4` | Tokio runtime worker threads. |
| `SWE_SANDBOX` | Landlock + seccomp | `bwrap` selects the bubblewrap backend. |
| `SWE_DISABLE_LANDLOCK` | `0` | `1` disables Landlock. |
| `MINI_SWE_LANDLOCK_ENFORCE` | `0` | `1` makes a missing Landlock fatal. |
| `SWE_DISABLE_SANDBOX` | `0` | `1` disables all confinement. |
| `WORKER_DIVERGENT_VERIFY` | `1` | `0` skips the divergent second verify run. |
| `SWE_CACHE_DIR` | `<SWE_TEMP_DIR>/swe-cache` | Shared compiler/package cache root. |
| `SWE_SHARED_CACHES` | — | Extra cache binds for the sandbox. |
| `SWE_ALLOW_TOOLCHAIN_CREDENTIALS` | `0` | `1` exposes the credential files beside the shared caches (`~/.npmrc`, `~/.m2/settings.xml`, `~/.gradle/gradle.properties`, `~/.cargo/credentials.toml`). |
| `SWE_DISABLE_KACHE` / `KACHE_DISABLED` | unset | `1` disables the kache layer. |
| `KACHE_CACHE_EXECUTABLES` | `0` | kache re-caches a worker's own test executables; worker builds disable it unless the operator sets the variable. |
| `MINI_SWE_TARGET_SLOT_MAX_GIB` | `4` | Empty a leased build slot over this size; `0` disables the cap. |
| `HUB_TARGET_TTL_HOURS` | `24` | Prune shared build slots older than this. |
| `HUB_TARGET_MAX_GB` | `40` | Size cap on shared build slots. |
| `MONITOR_WIDTH` | terminal size | Width used by `monitor` / `status`. |
| `COLUMNS` | terminal size | Fallback width when `MONITOR_WIDTH` is unset. |
| `XDG_CONFIG_HOME` | `$HOME/.config` | Base for `mini-swe/models.yaml`. |

## Load testing

`tests/it/load_test.rs` is an opt-in (`#[ignore]`) load test: it fans one real hub daemon out to many agents of many workers and asserts the hub keeps the machine busy without memory pressure. It is excluded from `cargo test` so the normal suite stays fast — run it explicitly:

```bash
# Quick smoke: 2 agents x 3 workers (the defaults), ~5 s.
cargo test --test it load_test:: -- --ignored --nocapture

# Full load: 5 agents x 20 workers (100 total), ~75 s on 4 cores.
LOAD_AGENTS=5 LOAD_WORKERS_PER_AGENT=20 \
    cargo test --test it load_test:: -- --ignored --nocapture
```

Every worker is driven by a fake OpenAI-compatible SSE server (`tests/it/common/fake_llm.rs`) that scripts three turns: a light command (`ls`), a heavy command (`cargo build` plus a bounded CPU burn, classified heavy by `is_heavy_command` so the pool's admission controller has to dose it), then the completion sentinel. The test dispatches each worker over the hub socket with a distinct agent identity (`agent-1`..`agent-5`) in `hub/hello`, samples the daemon's `/proc/<pid>/status` `VmHWM`/`VmRSS` every 250 ms, counts the heavy commands in flight by watching the daemon's children, and records the order workers finish in. It then asserts — and prints a one-screen report of:

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
