import json, os, subprocess, time, pathlib, shutil, sys
base = pathlib.Path(os.environ["REPRO_DIR"])
shutil.rmtree(base, ignore_errors=True)
repo, swe, hub = base/"repo", base/"swe", base/"hub"
for d in (repo, swe, hub): d.mkdir(parents=True)
os.chmod(hub, 0o700); os.chmod(swe, 0o700); os.chmod(repo, 0o755)
def git(*a, cwd=repo):
    subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True)
git("init", "-b", "master"); git("config","user.name","t"); git("config","user.email","t@l")
git("commit","--allow-empty","-m","seed"); git("branch","worker-orphan1")
wid, agent, pid = "orphan1", "listen-first", 0
now = int(time.time())
row = {"id": wid, "pid": pid, "task":"t","model":"ninja","status":"running","step":1,
       "max_turns":10,"last_command":"cargo test","started_at":now-60,"updated_at":now-60,
       "owner":agent,"repo_path":str(repo)}
(swe/"swe-registry").mkdir()
(swe/"swe-registry"/f"{wid}.json").write_text(json.dumps(row))
env = dict(os.environ)
env.update({"SWE_HUB_DIR":str(hub), "SWE_TEMP_DIR":str(swe), "TMPDIR":str(swe),
            "HUB_IDLE_SECS":"60","HUB_AUTO_RESUME":"0",
            "MINI_SWE_HUB_RECOVERY_DELAY_MS":"400",
            "OPENAI_API_KEY":"test-key","ENV_FILE":str(base/"absent.env"),
            "MODELS_FILE": str(pathlib.Path(os.environ["REPO"])/"models.yaml")})
log = open(base/"daemon.log","w")
p = subprocess.Popen([sys.argv[1],"daemon"], env=env, stdout=log, stderr=subprocess.STDOUT)
time.sleep(4)
rp = swe/"swe-registry"/f"{wid}.json"
print("== row after startup:", rp.read_text() if rp.exists() else "GONE")
print("== registry dir:", sorted(q.name for q in (swe/"swe-registry").iterdir()))
p.terminate(); p.wait(timeout=10); log.close()
print((base/"daemon.log").read_text()[-2500:])
