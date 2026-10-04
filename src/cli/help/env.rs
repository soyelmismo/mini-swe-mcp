/// `env`: the startup environment variables.
///
/// The startup variables come first, then one line per knob that tunes a running
/// server, so an operator reading this top to bottom learns which variable to
/// set and what happens when they do. `.env.example` carries the same list with
/// the reasoning behind each default; a unit test fails if a variable the code
/// reads is missing from one of the two.
pub(super) const TEXT: &str = concat!(
    "Read at startup: OPENAI_API_KEY (required to dispatch), OPENAI_API_BASE (default https://api.openai.com/v1), DEFAULT_MODEL (the default model alias), MODELS_FILE (the models catalog), MINI_SWE_NO_DAEMON=1 (serve MCP in-process instead of through the hub), MINI_SWE_AGENT_ID (pin the session's agent identity), and MINI_SWE_WATCH_TOKEN (set by a dispatch so the watch its shell runs is attributed to your session). A .env file is loaded first.",
    "\n",
    "MAX_CONCURRENT_WORKERS (max subagents in flight, default 128) and MAX_WORKERS_PER_AGENT (max per agent identity, 0 = unlimited) size the worker pool; BASH_CONCURRENT_LIMIT and BASH_BUILD_LIMIT (default cores/2) cap concurrent commands and heavy ones, and BUILD_PARALLELISM is the job/thread cap exported into each command.",
    "\n",
    "COMMAND_TIMEOUT_SECS (one budget for every command) overrides COMMAND_HEAVY_TIMEOUT_SECS (600) and COMMAND_LIGHT_TIMEOUT_SECS (120).",
    "\n",
    "JOB_MAX_SECS (ceiling on one background job's lifetime, default 2700) and WAIT_JOB_SECS (how long one wait on a job blocks before reporting it still running, default 600) bound the builds a worker leaves running in the background.",
    "\n",
    "HUB_AUTO_HANDOVER=0 stops the hub from handing over to a rebuilt executable by itself; HUB_HANDOVER_SECS (default 900, clamped to 1..=86400) is how long that handover waits for a quiet moment; MINI_SWE_HUB_EXE_POLL_MS (2000) and MINI_SWE_HUB_EXE_STABLE_MS (3000) are the watch's stat interval and its settle window, which tests shorten and production leaves alone.",
    "\n",
    "MINI_SWE_RECONNECT_SECS (default 60, clamped to 1..=86400) bounds how long a client follows a hub that goes away; MINI_SWE_TEARDOWN_WAIT_SECS (default 60) bounds the hub's shutdown wait for live workers' teardowns; HUB_IDLE_SECS makes an idle hub exit; HUB_AUTO_RESUME=0 leaves interrupted workers interrupted after a restart; SWE_HUB_DIR overrides the hub's socket directory.",
    "\n",
    "HUB_HEAVY_IONICE picks the I/O class of a heavy command: unset (or anything else) demotes it to the bottom of the best-effort class, 0 disables the demotion, and idle (or 3) moves it to the idle class.",
    "\n",
    "POOL_BUDGET_EXTENSION_PCT (25) and POOL_BUDGET_EXTENSION_MAX (30) size the one automatic extension a worker that is still making progress gets past its turn budget; 0 disables it. POOL_READ_ONLY_NUDGE_TURNS (15), POOL_READ_ONLY_ESCALATE_TURNS (30) and POOL_READ_ONLY_PAUSE_TURNS (45) are the three read-only streak thresholds for a dispatch that names files, so a worker told to edit is nudged, then handed its plan, then parked.",
    "\n",
    "LLM_MAX_RETRIES (6) and LLM_OUTAGE_PATIENCE_SECS (600) bound one model step; HUB_LLM_CONCURRENCY caps in-flight requests process-wide (0 = unlimited).",
    "\n",
    "HISTORY_FULL_TURNS and HISTORY_BUDGET_BYTES bound the turn history replayed into a worker's context, HISTORY_KEEP_ALL_REASONING=1 keeps the reasoning as well as the answer, WORKER_MAX_RETAINED_LOGS (200) and WORKER_MAX_EMITTED_LOGS (40) bound the step log, WORKER_TERMINAL_TTL_SECS (300) how long a finished record stays in memory, WORKER_RETENTION_SECS (604800) how long its registry row and conversation live while the branch exists, and WORKER_RETIRED_GRACE_SECS (86400) how long they outlive a branch that is gone.",
    "\n",
    "Scratch and sandbox: SWE_TEMP_DIR (/var/tmp), SWE_CACHE_DIR (<SWE_TEMP_DIR>/swe-cache), CARGO_TARGET_DIR (per-worker targets by default; WORKER_BUILD_DEBUG=1 keeps full debug info in them), SWE_SHARED_CACHES, SWE_DISABLE_SANDBOX=1, SWE_DISABLE_LANDLOCK=1, SWE_DISABLE_KACHE=1, KACHE_DISABLED=1, SWE_SANDBOX (force a backend) and SWE_ALLOW_TOOLCHAIN_CREDENTIALS=1 (expose the credential files the cache directories sit beside) are described in the sandbox topic.",
    "\n",
    "Build target slots are bounded twice over: MINI_SWE_TARGET_SLOT_MAX_GIB (default 4, 0 disables) empties a slot that grew past it and is idle, which costs one rebuild of your crate because the compiler cache restores the dependencies, while HUB_TARGET_TTL_HOURS (24) and HUB_TARGET_MAX_GB (40) bound the whole swept set.",
    "\n",
    "MONITOR_WIDTH (120) and COLUMNS override the width of the monitor view; RUST_LOG sets the tracing filter; ENV_FILE and XDG_CONFIG_HOME move the .env and models.yaml lookups. Test-only hooks exist but are deliberately not listed here; they are not operator settings.",
);
