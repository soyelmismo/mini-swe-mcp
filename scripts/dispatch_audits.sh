#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/rot/Proyectos/oracle/mini-swe-mcp"
LOG_DIR="/tmp/swe-audit-workers"
mkdir -p "$LOG_DIR"

REPORTS=(
  "opt_01_sse_buffer.md"
  "opt_02_pool_locks.md"
  "opt_04_mcp_throughput.md"
  "opt_05_agent_truncate.md"
  "opt_06_manifest_cache.md"
  "opt_07_step_log_memory.md"
  "opt_09_test_harness_perf.md"
  "opt_10_cargo_codegen.md"
  "overeng_01_agent_structs.md"
  "overeng_02_cli_parsing.md"
  "overeng_03_worker_state.md"
  "overeng_04_mcp_schemas.md"
  "overeng_05_worktree_pid.md"
  "overeng_06_manifest_rules.md"
  "overeng_08_error_handling.md"
  "overeng_09_test_boilerplate.md"
  "overeng_10_dead_code.md"
)

echo "=== Dispatching 17 audit implementation workers (200 turns each) ==="
for REPORT in "${REPORTS[@]}"; do
  NAME="${REPORT%.md}"
  LOG_FILE="$LOG_DIR/${NAME}.log"
  
  TASK="Implement all fixes and recommendations specified in audits/${REPORT}.
1. Read audits/${REPORT} carefully and extract all specific actionable tasks.
2. Edit the relevant codebase files in the worktree.
3. Verify your changes thoroughly with:
   cargo test
   cargo clippy -- -D warnings
4. Confirm that all items from audits/${REPORT} are 100% completed and zero warnings/errors remain.
5. Finish by printing your completion summary and running:
   echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"

  echo "Spawning worker for $REPORT -> $LOG_FILE"
  nohup mini-swe-mcp dispatch "[$NAME] $TASK" --model ninja --repo "$REPO_DIR" --group "audits" --max-turns 200 --wait > "$LOG_FILE" 2>&1 &
  sleep 1
done

disown -a 2>/dev/null || true
echo "All 17 workers spawned in background and disowned."
