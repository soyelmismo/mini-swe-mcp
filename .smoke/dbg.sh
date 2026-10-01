set -e
BIN=${CARGO_TARGET_DIR}/debug/mini-swe-mcp
ROOT=$(mktemp -d $TMPDIR/smoke-root.XXXXXX)
REPO=$(mktemp -d $TMPDIR/smoke-repo.XXXXXX)
cd "$REPO"
git init -q --initial-branch=main .
git config user.email s@s.s; git config user.name s
echo base > README.md; git add .; git commit -qm base
git checkout -q -b worker-smoke
echo work > work.txt; git add .; git commit -qm "worker work"
git checkout -q main
python3 - "$ROOT" "$REPO" << 'PY'
import json, sys, os
root, repo = sys.argv[1], sys.argv[2]
meta = {"task":"smoke the merge path","group":None,"model":"m","temperature":None,
        "repo_path":repo,"base_commit":"0"*40,"base_branch":"main","branch":"worker-smoke",
        "network_offline":False,"verify":None,"client_env":[],"max_turns":5,
        "review_after":None,"revision":0,"auto_continues":0,"owner":None}
p=os.path.join(root,"swe-wt-smoke.history.jsonl")
with open(p,"w") as f:
    f.write(json.dumps(meta)+"\n")
    f.write(json.dumps({"role":"system","content":"you are a worker"})+"\n")
print("wrote", p, os.path.getsize(p))
PY
echo "--- root contents ---"; ls -la "$ROOT"
echo "--- merge ---"
SWE_TEMP_DIR="$ROOT" MINI_SWE_NO_DAEMON=1 HOME="$REPO" OPENAI_API_KEY=x "$BIN" merge smoke || echo "exit=$?"
