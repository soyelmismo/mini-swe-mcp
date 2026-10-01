set -e
BIN=${CARGO_TARGET_DIR}/debug/mini-swe-mcp
ROOT=$(mktemp -d $TMPDIR/smoke-root.XXXXXX)
mkdir -p "$ROOT/swe-registry"
cat > "$ROOT/swe-registry/smoke.json" << 'JSON'
{"id":"smoke","pid":1,"task":"t","model":"m","status":"Completed","step":1,"max_turns":5,
 "last_command":"","question":null,"started_at":0,"updated_at":0,"group":null,
 "repo_path":"/tmp","owner":null,"metrics":{},"base_branch":"main","base_commit":null,
 "revision":0,"auto_continues":0}
JSON
echo "SWE_TEMP_DIR seen by child: $(SWE_TEMP_DIR="$ROOT" printenv SWE_TEMP_DIR)"
echo "--- status smoke ---"
SWE_TEMP_DIR="$ROOT" MINI_SWE_NO_DAEMON=1 HOME="$ROOT" OPENAI_API_KEY=x "$BIN" status smoke 2>&1 | tail -5
