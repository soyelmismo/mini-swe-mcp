set -e
BIN=${CARGO_TARGET_DIR:-/var/tmp/swe-wt-b0e08f68/target/debug}/debug/mini-swe-mcp
ROOT=$(mktemp -d $TMPDIR/smoke-root.XXXXXX)
REPO=$(mktemp -d $TMPDIR/smoke-repo.XXXXXX)
cd "$REPO"
git init -q --initial-branch=main .
git config user.email s@s.s; git config user.name s
echo base > README.md; git add .; git commit -qm base
git checkout -q -b worker-smoke
echo work > work.txt; git add .; git commit -qm "worker work"
git checkout -q main
echo later > later.txt; git add .; git commit -qm "base moved"
python3 - "$ROOT" "$REPO" << 'PY'
import json, sys, os
root, repo = sys.argv[1], sys.argv[2]
meta = {"task":"smoke the merge path","group":None,"model":"m","temperature":None,
        "repo_path":repo,"base_commit":"0"*40,"base_branch":"main","branch":"worker-smoke",
        "network_offline":False,"verify":"test -f work.txt && test -f later.txt",
        "client_env":[],"max_turns":5,"review_after":None,"revision":0,"auto_continues":0,
        "owner":None,"messages":[]}
with open(os.path.join(root,"swe-wt-smoke.history.jsonl"),"w") as f:
    f.write(json.dumps(meta)+"\n")
    f.write(json.dumps({"role":"system","content":"you are a worker"})+"\n")
PY
echo "=== CLI merge (plain) ==="
SWE_TEMP_DIR="$ROOT" MINI_SWE_NO_DAEMON=1 HOME="$REPO" OPENAI_API_KEY=x "$BIN" --admin merge smoke
echo "=== state after ==="
git log --oneline --graph --all | head
echo "--- branches ---"; git branch -a
echo "--- scratch root ---"; ls "$ROOT"
echo "=== CLI merge again (branch gone) ==="
SWE_TEMP_DIR="$ROOT" MINI_SWE_NO_DAEMON=1 HOME="$REPO" OPENAI_API_KEY=x "$BIN" --admin merge smoke || echo "exit=$?"
echo "=== --json form ==="
SWE_TEMP_DIR="$ROOT" MINI_SWE_NO_DAEMON=1 HOME="$REPO" OPENAI_API_KEY=x "$BIN" --admin merge smoke --json || echo "exit=$?"
