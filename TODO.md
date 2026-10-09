# mini-swe-mcp — pending work (session 2026-09-29/30)

Legend: `[x]` done and merged on master · `[~]` in progress (worker id) · `[ ]` pending

## Hub (single daemon, many agents)
- [x] H1 daemon core (flock, socket 0600, SO_PEERCRED, idle shutdown)
- [x] H2 thin clients: stdio proxy + CLI through the hub, auto-start, caller cwd
- [x] H3 per-agent ownership, `--admin`, per-agent cap
- [x] H4 owner-routed events with replay, version handshake, idle `hub/shutdown`
- [x] H4b build identity so a rebuilt binary replaces an idle hub
- [x] Legacy-hub compatibility (newer client vs pre-handshake daemon)
- [x] H5a event-driven waits, shared HTTP client, coalesced registry writes
- [x] H5b `status --line`, crash recovery with salvage, history at checkpoints
- [x] H6a resource-aware admission (memory/load), cancellation-safe queue
- [x] H6b slot-affine shared build dirs, private per-worker scratch
- [x] H6c bounded conversation memory (compaction keeps orchestrator messages)
- [x] H6d fair round-robin slots across agents, optional LLM concurrency cap
- [x] H7 kernel-only sandbox (Landlock + seccomp), bwrap opt-in
- [x] H8 revise a finished worker via `steer`; `next_step` guidance
- [x] H9 harness merges the moving base branch before completion
- [x] H10 `mini-swe-mcp watch` (self-contained events, missed-event replay, governance) + orchestrator guidelines in tool/CLI descriptions
- [x] H11 dispatch/steer always detach (remove `--wait`, MCP `watch` action), workers private to their owner (status/logs/collect/list), recovery cleans target dir + .pid
- [x] H12 load test 5 agents x 20 workers with a fake LLM (100/100 in 72 s, daemon peak RSS 24.6 MB, heavy capped 4/4)
- [x] H13 continue any stopped worker with `steer` (incremental JSONL history, cold continuation, `interrupted` + auto-resume, honest steer replies)

## Harness robustness
- [x] DeepSeek thinking mode: replay `reasoning_content` on every assistant turn
- [x] Wait out LLM provider outages; a resumed worker never dies on the next error
- [x] Nested runs inside the sandbox use private scratch (`SWE_TEMP_DIR`)
- [x] System prompt rules 14-18 (reuse, cancellation safety, hermetic tests, proof before "environmental", test placement)
- [x] Hermetic test suite (no real hub, no env races, order-independent)

- [x] H14 (confirmed: 16/116 reads were repeats, all >12 steps after the original) — byte-budget compaction. Was: check whether history compaction (HISTORY_FULL_TURNS=12) makes workers re-read files they already read (H11 re-read handlers.rs every ~30 steps); tune the window or keep read outputs longer if the evidence holds

- [x] H15 harness: shared slot target dirs let another live worker's uplifted binary (target/debug/<bin>, CARGO_BIN_EXE_*) replace yours; lease a target dir to ONE worker for its lifetime, reuse it only by later workers
- [x] H16 watch polish: all missed events in one call, short verify tail, diff stat from the branch for torn-down workers
- [x] T1 test leaks worker branches into the real repo (dispatch without repo_path)
- [x] T2 flaky hub_test a_checkpointed_worker_survives_hub_sigkill_and_revision: fake LLM panics when the SIGKILL drops an in-flight request
- [x] T2b three tests still mutate process env (SWE_HUB_DIR hub_test, MAX_WORKERS_PER_AGENT mcp_test, SWE_TEMP_DIR watch_test, never restored) because the code reads them in-process; ENV_MUTEX does not cover children spawned meanwhile: give that code explicit config and drop the mutations — round43 (merged)

- [x] T5 identity per agent session shared by MCP and shell (host process ancestor); today MCP=conn-scoped and every shell="cli" (shared!)

- [x] T6 sessions sharing one host process (verified live: two tabs on one MCP connection isolated; watch token works, wrong token refused) (opencode v2 tabs): host+session identity (_meta.sessionID per MCP call, CLAUDE_CODE_SESSION_ID/OPENCODE_SESSION_ID/MINI_SWE_SESSION_ID env), watch tokens for shells without session info

- [x] Killed 7 orphans: 2 from my old launch.sh (dispatch --wait blocked on tail -f stdin), 5 left by worker H11 (`cargo test &` in its deleted worktree)
- [x] T9 harness: kill each step's process group when it returns; sweep processes left in the worker's dirs at worker end and in recovery

## Agent UX (token savings)
- [x] U1 load the repo's AGENTS.md/CLAUDE.md/... into the worker system prompt; AGENTS.md for this repo
- [x] U2 short ids: unique prefixes and `last`
- [x] U3 batch dispatch (MCP tasks[], CLI -f tasks.yaml) — 577ccd24 (steered: restore "never dispatch a replacement" in the schema)
- [x] U4 compact watch events, seen-on-interaction, version warning once — b4b40250
- [x] U5 review action + compact collect (merge-tree conflict check) — 9145367d
- [x] U6 shorter MCP tool description + help topics
- [x] U7 merge <id>: safe one-command merge — b0e08f68
- (implicit watch token dropped: a shell without session info cannot tell its tabs apart; the token stays explicit)

## Finish
- [x] Merge H10 ✓, H12 ✓, H13 ✓, H11 ✓ (review; corrections and conflicts go back to the worker via `steer`)
- [x] Delete branch `worker-b576da08` once H12 is merged
- [x] Run the load test (full 5x20) and record the numbers in README
- [x] Docs (D2): README/--help for the hub workflow (dispatch → watch → review → steer → merge), statusLine, sandbox, env vars
- [x] Crate-wide `cargo fmt` once no branch is open; add `cargo fmt --check` to the verify gate
- [x] Verify end to end with a real worker (dispatch → watch → steer revision → watch; another agent cannot see/steer/kill it)
- [x] T3 race when a newer client replaces an idle hub (new daemon loses the lock the old one still holds)
- [x] T4 tool/CLI descriptions: MCP agents wait with the `watch` action (not the shell command); steer continues any stopped worker
- [x] Restart the hub daemon on the final binary (automatic replacement verified)
- [x] Update `~/.claude` MCP config if needed (`mini-swe-mcp --stdio`, connected, proxies to the hub); my orchestration now waits with `mini-swe-mcp watch`
- [x] Push master (be47ed4)
- [x] Push again after T5
- [x] Notify the user again when T5 is merged
- [x] T7 hermetic identity tests (one read the real CLAUDE_CODE_SESSION_ID)
- [x] T8 tests spawning the binary inherit the caller's session env (2 hub tests fail from an agent shell)
- [x] Push, notify the user
- [x] Merge T9 (737 tests; 0 processes left in worker dirs after the suite)
- [x] T10 differential verification (clean vs orchestrator env, HOME/TMPDIR/TZ) + side-effect audit, language-agnostic
- [x] T11 registry/scratch root injectable per pool (in-process tests share the real registry; intermittent list test)
- [x] T12 polyglot: heavy-command classification, shared caches + sandbox grants, parallelism caps for npm/pip/go/maven/gradle/dotnet/make
- [x] T13 guidance: dispatch independent tasks in parallel (base sync handles overlap)
- [x] T14 watch without ids follows workers dispatched after it started (missed T13's completion) + at most one active watch per agent identity
- [x] T16 Revision counter does not increment on a second continuation (steer reply said "Revision 1" twice for 1b584d43)
- [x] T15 watch never exits silently; falls back to polling when the hub reply is older
- [x] T17 Build-dir sweep must also evict legacy `swe-target-<repo>-slotN` dirs (6.7 GB left behind by the H6b naming after H15)
- [x] Push (a03c8cc), notify the user
- [x] Audit false positive (exiting children/zombies counted as left processes) fixed in ed14b7c; daemon crash-restarted: 7 workers auto-continued
- [x] Daemon must listen before slow recovery (7 workers took >5 s; the first client failed with 'Hub did not start within 5 seconds') — H11 a036e08b
- [x] Audit false positives fixed: leaked sleeper from the repo's own test (ef4ef92), ignored build output counted as new files (2d03679)
- [x] Flaky under heavy load: tests/worker_reap_test.rs await_process_in waits only 4 s for a process to appear (failed once at load 9) — T20 3383be84
- [x] T18 admission by PSI (cpu/memory/io pressure) instead of load average (only 1 build ran: load 6.4 from I/O waits while CPU pressure was 9%)
- [x] watch reports 'stalled' for a worker whose COMPLETION gate is queued for admission (T14 only covered step commands) (fixed by P3: completion gates publish their admission wait)
- [x] DESTRUCTIVE audit bug fixed (331cebf): it deleted refs/worktrees of concurrently dispatched workers (T18 lost its checkout); now hub-managed refs/worktrees are ignored and nothing is ever deleted. Confinement now fails closed (T18's commands had run unconfined). T18 restored by cold continuation.
- [x] Tests leak temp hub dirs (also /tmp/mswe-* socket fallbacks of SIGKILLed test daemons): 159 /tmp/swe-test-hub-* (cwd-dir/thin-dir/...) left behind; make them clean up (and add to the side-effect thinking: suites must not leave temp dirs) Also ~120 empty /var/tmp/swe-target-env-{sanitize,toolchain}-test-* and swe-target-exec-*-test dirs. — T21 e1b27029
- [x] Unix socket path too long (SUN_LEN) with deep hub dirs: short fallback (exposed by divergent verify, blocked every completion)
## Disk I/O and worker flow
- [x] T19 builds without debug info / incremental; + KACHE_CACHE_EXECUTABLES=0 for workers (store was 22 GiB of own-crate test binaries, ~0 hits)
- [x] P2 verification cache: reuse an identical passing run on the same tree
- [x] P3 heavy commands at idle I/O priority; completion gates first in the admission queue
- [x] P4 AGENTS.md: iterate with targeted tests, full gates once
- [x] P5 tmpfs build dirs: DROPPED by decision. After T19 (0.87 GB/build dir) + slice_idle=0 the IO PSI is 0.3-3% idle; 9 workers on tmpfs would take ~8 of 16 GB RAM (OOM risk > gain)
- [x] Workers burn turns polling background builds with `sleep N; cat log` (T19 spent ~15 of 60 turns) because one command may run at most 600 s: give the harness a wait primitive (block until a background job exits or a file changes, not a turn, not a stall) — H12 bf056838 (timeout -> background job + WAIT_JOB)
- [x] Dedupe the verify-tail helpers: src/mcp/handlers.rs (U5 review) re-implements src/mcp/events.rs verify_tail — M1 9f377140
- [x] A graceful daemon shutdown (SIGTERM/SIGINT) marks live workers Failed ("Server shutting down"); it must mark them interrupted so the next daemon auto-continues them, like crash recovery — H10
- [x] Finished workers become un-steerable ~5 min after completion: the terminal reaper deletes registry row + history (all of this round lost theirs) — H13
- [x] P2 fingerprint tests leak /tmp/exec-fp-* — T22
- [x] Test leftovers per full run (61 -> 2 after T23; remaining: /tmp/mswe-* sockets, see E10) (gate.sh diff, 61 new): 26 /var/tmp/swe-loop-poo-*, ~18 /var/tmp/swe-target-<test-repo>-K, 10 /var/tmp/swe-tmp-swe-cmdjob-* (H12 tests), 2 swe-tmp-swe-reap-it-step-*, 2 swe-watch-re-*, /tmp/mswe-* sockets, /tmp/swe-env-test-cargo-home-fallback-* — T23
- [x] V1 completed events must not show a stale failing verify tail (Verified: true + "timed out") — c87391a2
- [x] Round batching (manual trial) — superseded by consolidator + merge --approved: workers verify with fmt+clippy+targeted tests; one full gate per merged batch; failures routed to the owner by file. Measure, then decide whether to automate as a hub merge queue


## Consolidator round workflow (approved 2026-10-01)
- [x] C1 role=consolidate, delegation check (same owner+group), harness-mediated CONSOLIDATE_MERGE — cc31f638
- [x] C2 CONSOLIDATE_STEER (route a failure to its owner) + CONSOLIDATE_WAIT (block until owners stop, no stall)
- [x] C3 glue: `consolidate` action/CLI verb with defaults, built-in consolidator prompt (merge, one full gate, attribute by touched files, review criteria, compact report), `help consolidate`, cheap-verify guidance in dispatch descriptions
- [x] Measure: disk I/O and Claude tokens/turns per round, old vs new; reported (calls/worker 15.1 -> 11.3 -> 8.0; result KB/worker 24.6 -> 14.4 -> 6.3)

## Orchestrator flow without direct commands (approved 2026-10-01: "flujo seamless con el mcp")
- [x] D1 busy daemon hands over to a newer build at the first quiet moment; stdio proxy and watch reconnect — 5286747d
- [x] D2 structured REPORT from workers; events carry done/files(+/-)/risks — 4023e6bb
- [x] D3 review shows code diff + summarized tests; approve/unapprove — 3300aab4
- [x] D4 merge --approved: all approved workers of a group with ONE sandboxed gate, failure attributed by file — e958af42
- [x] W1 watch guidance: run as-is in background (no >, &, wrappers), re-run after each event — b3d3c00d

## First consolidated round (D1-D4, T23, W1 in group selfimprove)
- [x] When D1, D2, D4 finish: daemon on the new build (idle auto-replace), `consolidate --group selfimprove --model nerd` — b39f741c (manifest only saw D1+D2, see E11)
- [x] Read its REPORT; hand-review only sandbox/governance/identity parts (D1: daemon handover + stdio proxy reconnect); merge the consolidator branch once — it found+fixed a daemon leak in D1; 975/975
- [x] Measure and report to the user: disk I/O and Claude tokens/turns per worker for (a) per-worker review+gate (start of session), (b) manual batches (rounds.log), (c) consolidated round

## Problems and inefficiencies seen while orchestrating through the MCP
- [x] kache: purged this project's own crates from the store (25.8 -> 2.1 GiB; 35 crates, 36 hits in total) and set `cache_executables = false` in ~/.config/kache/config.toml
- [x] E1 a worker that runs out of turns is reported `completed` (D2 at 150/150, C1 at 60/60, no verification): it must read as stopped/exhausted with a continue hint, never as done — fe0a72f2 (round2) — merged via consolidator 72047156
- [x] E2 every dispatch/steer reply repeats "To wait for it: MINI_SWE_WATCH_TOKEN=... mini-swe-mcp watch" even while the caller already has a watch running: print it only when no watch of that caller is active — aa9bf916 (merged via consolidator 38ee57f3)
- [x] E3 `status <id>` text shows only State/Step; the useful health (last op, elapsed, health counters, verified, approval) needs --json: render it in the text view — e5b98c65 (merged via consolidator 38ee57f3)
- [x] E4 the verify-reuse cache (P2) only hits when the worker ran exactly the verify string, but the worker is never told that string: state the completion gate in the task/system prompt ("run exactly `<verify>` last") — d3dc0212 (merged via consolidator 38ee57f3)
- [x] E5 scripting a dispatch needs --json + a JSON parser to get the id: `dispatch --quiet` prints only the id(s) — 6f4c6cb6 (merged via consolidator 38ee57f3)
- [x] E6 hot shared files cause base-merge conflicts that cost long workers many turns (src/cli/args.rs, src/mcp/schema.rs, src/cli/help.rs: U5, U7, C1, D3): split them into per-feature registration (one file per CLI verb / schema property group / help topic) so parallel features stop editing the same lines — 8b6f4846 (round2) — merged via consolidator 72047156
- [x] E7 compact events built from a registry row (after a daemon restart) drop "Verified" — 1f1388de (round2) — merged via consolidator 72047156
- [x] E8 workers spend 60+ read-only steps before the first edit (C1, D2) despite stagnation nudges: measure the nudge thresholds and make the first nudge earlier / more concrete — 6c37e52a (round2) — merged via consolidator 72047156
- [x] E9 a worker that fails (turn budget) in the middle of resolving a base merge loses the partial resolution: the worktree is removed and the branch stays at "checkpoint before base integration" (D2 lost a 6-file resolution). Keep the in-progress merge (or its resolved files) for the continuation — a96f4f28 (round2) — merged via consolidator 72047156
- [x] E10 a full test run still leaves 2 /tmp/mswe-<uid>-<hash> socket fallback dirs (test daemons killed without cleanup) and /var/tmp/swe-tmp-swe-merge-approved (D4 test) — adfb4c63 (round2) — merged via consolidator 72047156
- [x] E11 the round manifest builds only from registry rows: rows reaped by a pre-H13 daemon made D3/D4/T23/W1 invisible to the consolidator; also it shows verified=unknown because the row has no `verified` (pairs with E7) — covered by E7 (round2) — merged via consolidator 72047156
- [x] E12 `mini-swe-mcp consolidate` prints raw JSON instead of a rendered summary — c5322ba2 (merged via consolidator 38ee57f3)
- [x] E13 a worker told to resolve conflicts with master cannot start the base merge itself (sandbox: .git read-only) and has to request completion first to trigger it: steer of a stopped worker should integrate the base up front (conflicts left in place), or offer a SYNC_BASE sentinel — covered by E9 (round2) — merged via consolidator 72047156
- [x] E14 consolidator attribution: a failure in a file owned by worker A caused by a type/field worker B changed (D1 fixture vs D2 WorkerMeta.report) was routed to A; the built-in procedure must treat "symbol changed by another worker of the round" as an interaction it fixes itself — 44554860 (round2) — merged via consolidator 72047156
- [x] E15 a worker paused after a CONSOLIDATE_STEER asks the orchestrator, while the consolidator loops on CONSOLIDATE_WAIT: WAIT must return the question and the procedure must say "answer a paused worker with CONSOLIDATE_STEER (or fix it yourself); never re-WAIT a paused worker"; consider routing such questions to the consolidator — covered by E14 (round2) — merged via consolidator 72047156
- [x] E16 the same completion is delivered twice (D1 rev: once live with Verified/summary, then again from the registry row as "done" without Verified): dedupe by (worker, revision, event) — 253b5ec3 (round2) — merged via consolidator 72047156
- [x] E17 a merged worker branch is pruned (and its history retired) on the next dispatch; if the orchestrator then reverts that merge, the worker can no longer be continued (D4): keep the history for a grace period after the branch disappears — cb0ca546 (round2) — merged via consolidator 72047156
- [x] E19 the REPORT given in the follow-up answer is not captured (report null, headline falls back) — merged via consolidator 72047156
- [x] E20 `status` of a completed worker is 13 KB: full diff + 111 "artifacts" (pre-existing .agents/ and audits/ files synced back): status must be compact (no diff, artifacts capped/omitted unless asked) — round 3 — in E3 e5b98c65 (merged via consolidator 38ee57f3)
- [x] E21 every new registry/meta field forces edits in ~15 test files that build WorkerRegistryEntry/WorkerMeta literals (round2 manifest: 27 interaction points, mostly tests): one test fixture builder (Default + for_test helpers in tests/common and test_support) so a new field touches one place — round 3 — 72fd7d96 (merged via consolidator 38ee57f3)
- [x] E22 while a consolidator runs, completion/failed events of the workers it steered still reach the orchestrator (noise: the consolidator waits on them itself); deliver them only to the consolidator (E14 did it for questions) — round 3 — 1c6c7f9c (merged via consolidator 38ee57f3)
- [x] E23 after a daemon restart/handover, every terminal event is replayed as "missed" (10 already-merged round2 workers): persist the per-owner acknowledged state across restarts, and never replay events of workers whose branch is merged or gone — merged via consolidator c7380e32
- [x] E24 `watch --group <g> --all`: one event when the whole group has stopped (or one needs an answer), instead of one per worker (orchestrator cost per round ~2N calls -> 2) — merged via consolidator c7380e32
- [x] E26 the round manifest gives the consolidator only the FIRST line of each task, so it judges scope blindly (it made E4 drop an explicitly requested AGENTS.md line): pass the full task text (bounded) or let it read it via review — merged via consolidator c7380e32
- [x] E27 ninja workers ignore the read-only nudges (E23/E24 hit 3 nudges at ~55 steps before editing): after the 2nd nudge, inject the task's named files/functions as a short edit plan prompt, or escalate to ASK_ORCHESTRATOR automatically — merged via consolidator 75fd53d7
- [x] E28 integrated workers are garbage, not something to hide: once a worker's work is in the base branch (merge, merge --approved, or integrated by a consolidator whose branch was merged) retire it completely and immediately: branch, registry row, history JSONL, steer mailbox/source, ack entries, build-dir lease (measured now: 21 merged branches + 21 rows + 21 histories + 7 steer-source files; 826 history files / 126 MB in /var/tmp overall). The 24 h grace (E17) applies only to branches that vanish WITHOUT being merged. Plus a sweep at daemon start and after merges for orphans (history without row and branch). — merged via consolidator 75fd53d7
- [x] E29 an auto-consolidation setting (model/verify) cannot be corrected after dispatch: the daemon keeps it in memory and rewrites the file (a typo in --consolidate-verify needed a daemon restart). Add `consolidate --group <g> --set [--model m] [--verify cmd]` (MCP: consolidate with update) to amend a pending round, and validate the verify command parses as shell (sh -n) at dispatch — 68a4ac30 (round7) — merged via consolidator 15c9cb9c
- [x] E30 `watch --all` requires one --group, and only one watch runs per session, so an orchestrator with two concurrent rounds cannot get round-level events for both: accept several --group values, and with no group cover all of the caller's live rounds (one event per finished round) — 1a98e641 (round7) — merged via consolidator 15c9cb9c
- [x] F2 E28's sweep retired 0 of 23 integrated workers: their registry rows have base_branch null — d757fd67 (round7) — merged via consolidator 15c9cb9c
- [x] F3 a client respawning the daemon after a rebuild uses current_exe() = '<path> (deleted)' and fails ('Could not start hub daemon') — dbf1d222 (round7) — merged via consolidator 15c9cb9c
- [x] F4 an interrupted worker with no commit yet is lost on a hub restart (its empty branch is deleted, auto-continue fails, the sweep removes its history; G1/G2 of round6 were lost this way) — 3580ab0a (round8) — merged via consolidator c4f708d1
- [x] F5 `watch` dies on a daemon handover with "Connection reset by peer" (reconnect only handles a clean EOF) — round8 — merged via consolidator c4f708d1
- [x] F6 a command that converts into a background job at its 600 s timeout produces a false "stalled | no step for 601s" event at the conversion (the stall detector should count the conversion step as activity) — merged via consolidator 5a4f6fef
- [x] F7 merge/retirement drops the watch acks while the router still holds the event, so a delivered completion is replayed after `merge` (observed with c4f708d1) — merged via consolidator df397293
- [x] E31 E23 in practice: after a daemon restart the first watch still replayed 17 events of workers whose branches are merged into master (acks predating E23 were in memory only, but the merged-branch suppression should have caught them). Re-check after E28 retires merged workers; if replays persist, fix the merged-branch check in the replay path — resolved at the root by E28+F2: integrated workers are retired, so nothing old is left to replay (verified: first watch on the new daemon replayed nothing)
- [x] E34 workers raise the tools/list budget instead of shortening text (E25 raised it 350 B, then restored after review; E29 moved the baseline 6990->7440): state "never raise the budget constant" in the budget test failure message itself, so any worker sees it when the test fails — 4a8a6f2e (round8) — merged via consolidator c4f708d1
- [x] E32 no verb discards a stopped worker on purpose (a failed consolidator with an unrunnable gate had to be removed by hand: branch + row + history): add `discard <id>` (owner-only; refuses a running worker; retires everything like E28); retirement must also remove the `.round-base` file (E14), which E28's list may miss — 1257f3c5 (round7) — merged via consolidator 15c9cb9c
- [x] G1 help workflow describes the round flow; MCP `help` action (shell-less agents could not read any topic); repo rules into AGENTS.md — round6 workers lost (see F4); redispatched d6eaf62a (round8) — merged via consolidator c4f708d1
- [x] G2 a --consolidate round gives workers the cheap gate by default (language-agnostic cheap-gate detector) — redispatched c5dd0f01 (round8) — merged via consolidator c4f708d1
- [x] E27, E28 — round5, consolidator 75fd53d7 (first auto-dispatched consolidator had an unrunnable gate from my launch.sh quoting; redone by hand)
- [x] E25 `dispatch --group <g> --consolidate` (and for batch -f): the hub dispatches the group's consolidator automatically when every worker of the group has stopped; the orchestrator only receives the consolidator's report — merged via consolidator c7380e32
- [x] E18 during a consolidated round, a worker steered by the consolidator re-integrates the CURRENT master on completion; if master moved meanwhile, it faces fresh conflicts (D2: 11 files, failed twice). Workers steered by a consolidator should integrate the round base (or the consolidator branch), not a moving master; the orchestrator rule meanwhile: do not move master while a round runs — covered by E14 (round2) — merged via consolidator 72047156

## Closing
- [x] Delete the old test leftovers on the host (/tmp/swe-test-hub-*, /tmp/mswe-*, /var/tmp/swe-target-env-*, swe-loop-poo-*, ...) once no test run is active (removed 1243 entries older than 1 h + empty test build dirs)
- [x] Update Claude memory: cheap worker verify + consolidator rounds, plain background `watch` re-armed per event
- [x] Drop rules from launch.sh's _common.txt that AGENTS.md / the system prompt already carry (12 rules -> 4; the old 'final message' rule conflicted with the REPORT block)
- [x] F8 a full test run still leaves 7 temporary entries (swe-batch-mc-*, swe-tmp-swe-merge-oa-main/prcons, one /tmp/mswe-* socket dir) — merged via consolidator c3cc0a72
- [x] F9 one /tmp/mswe-* socket fallback dir survives every full test run (7 -> 1 after F8) — merged via consolidator c47d925a
- [x] F10 under a deep TMPDIR one empty /tmp/mswe-* dir still survives a full run (F9 fixed the normal-TMPDIR case) — merged via consolidator 3bcd9c9f
- [x] F10 security follow-up: client-side fallback socket dir accepted without owner check (squattable /tmp path) + no client-side peer-uid check of the daemon (also affects abstract sockets) — steered to consolidator 3bcd9c9f; reviewed by Claude and merged (owner+mode check, peer-uid check on every connect, tests for symlink/open dir and foreign uid)
- [x] E35 sandbox/guardrail denials are not audited: the hub log records only step summaries, and the tool outputs that carry "BLOCKED BY WORKTREE GUARDRAIL" / Landlock denials live only in worker histories, which are deleted on retirement; record every guardrail block and confinement denial in hub.log (worker, owner, command summary, rule) so isolation can be audited after the fact — merged via consolidator e055a37c (audit line: worker, owner, rule, reason, bounded command summary)
- [x] E36 a worker that exhausted its budget while a consolidator was correcting it stays `Exhausted` in list/monitor after the consolidator finished the fix itself (F10 586b588a in round13), which reads as a dangling problem: when the consolidator absorbs a worker's pending correction, mark that worker `absorbed` (by consolidator <id>) so it is not shown as needing action (586b588a discarded by hand with the new `discard`) — round17; review: absorption must exclude completed non-integrated workers (data-loss risk), routed back via consolidator 8a7ec34c — merged via consolidator 8a7ec34c (absorbed only if stopped-not-completed; non-integrated work kept)
- [x] W2 a second, broader watch widens the running one (union of selections) instead of exit 5; a covered request exits 0 "already covered" — merged via consolidator b6e6e2d9
- [x] M1 model-based test of the event router (exactly-once per owner, ownership, acks across restart, retirement, consolidator routing, round events, no loss) + fixes; Claude reviews the invariants and the model itself — round14 — reviewed: 4 mutations caught for (b),(c),(e) and its own fix; blind spot in (d) (routing decision bypassed, dead-consolidator question hidden) routed back via consolidator c3587631; fixed: real routing decision covered, dead-consolidator questions now reach the owner (router bug the model had mirrored) — merged via consolidator c3587631
- [x] F11 a consolidator blocked in CONSOLIDATE_WAIT is reported "stalled | no step for 1801s" and `status` shows no command in flight (c3587631 waiting on 00f760e1): the wait must hold the command_running mark visible to status and the stall detector (C2 intended it) — merged via consolidator 2a9aaca3
- [x] R1 `--stdio` proxy/CLI on a single-thread runtime (each proxy runs 7 threads; real private memory ~0.3 MB) — round19 worker lost to H16 at handover; redispatched (round22) — completed; consolidating manually (e3ddb51c) since the auto consolidator was lost to H16 — merged via consolidator e3ddb51c (measured on the host after deploy: idle --stdio proxy 7 -> 3 threads: main, telemetry, one blocking stdin reader)
- [x] E37 a consolidator's completion event headline reads just "REPORT": its `REPORT <id> verdict: ...` lines clash with D2's REPORT block parser (done:/files:/tests:/risks:), so the summary falls back to the bare word; the consolidator should emit a D2 block (done: = round summary) plus its per-worker lines, or the parser should recognise the consolidator format — merged via consolidator 502310cd
- [x] H15 a worker's worktree cannot be recreated after a stale git registration ('missing but already registered'; round17 consolidator failed, fixed by hand with git worktree prune) — merged via consolidator 502310cd
- [x] H14 (withdrawn: misdiagnosis) the daemon did hand over at 10:13:41 as soon as a newer client called; it looked stale because Claude merged rounds 9-15 without rebuilding the release binary. Lesson for the orchestrator: rebuild the release after merges
- [x] H16 the start sweep retires just-dispatched/interrupted workers with zero commits as 'integrated' (branch tip == base); R1 and H14 were silently retired at a handover — round22; struck 3 times (also killed its own first fix worker 6bfdd332, H17 af4e917d and the round22 consolidator) — redispatched b8dc3f4f (round24); NO release rebuild until it is merged — merged via consolidator fd273d19 (only Completed + tip beyond recorded base + merged)
- [x] H17 OTA: the daemon notices its rebuilt executable and arms the planned handover by itself (today only a newer client call triggers it) — round23 — first worker lost to H16; redispatched 9677de6b (round24) — merged via consolidator fd273d19 (stable+executable+--build-id probe+supersedes)
- [x] R2 `--review-after <model>:security` adversarial review mode, language-agnostic gate (no hardcoded cargo), auto-trigger on sensitive paths declared in AGENTS.md — round25 — merged via consolidator b5009e7b (sensitive paths read from the main checkout, not the worker tree)
- [x] F12 the read-only escalation (E8/E27) also pauses consolidators ("No edit after 45 read-only turns"), whose job is reading, merging and gating without editing: exempt role=consolidate (and the review phase) from it; plus AGENTS.md itself as a sensitive path — merged via consolidator 2618479f
- [x] H18 handover race: the new daemon auto-continues and recreates a worktree while the old daemon is still cleaning it up ("not a git repository"; 0934b549 failed at step 0) — review: unbounded wait on teardown (hung git would hang the handover forever) routed back via consolidator b193150f; fixed (60 s bound + per-worktree .teardown marker the successor waits on) and security-reviewed automatically by R2 — merged via consolidator b193150f
- [x] X1 context pack in the first message (named paths outlined, backticked symbols located) to cut the 45-turn read-only phase — 43c738d2 — merged via consolidator cf94aeb0
- [x] X2 one bounded automatic budget extension when a worker is progressing at its limit — 8686c841 — merged via consolidator cf94aeb0
- [x] X3 consolidator per-worker REPORT/RISK lines stored and shown in its event and review — 3e4532aa — merged via consolidator cf94aeb0
- [x] X4 archive.jsonl of retired workers' final REPORTs + `archive` verb — 490bcb31 — merged via consolidator cf94aeb0
- [x] F13 F11 incomplete: a consolidator in CONSOLIDATE_WAIT shows "running for 0s" after 30 min (the start time is refreshed each poll) and the --all round watch still reports it "stalled | no step for 1801s" — merged via consolidator b99e8371
- [x] F14 round/compact watch lines show mojibake for non-ASCII text (an em dash rendered as "â"): UTF-8 is decoded or clamped incorrectly somewhere in the event/headline path — merged via consolidator afc26c1e (Captured head/tail cuts on char boundaries)
- [x] R3 security review audits only new commits since its last approval (a550eabd was reviewed 7x; consolidators re-review already-reviewed branches) — round31 (merged 63610a2)
- [x] R4 security review defaults to the strongest model (auto reviews of ninja workers used ninja) — round31 (merged 63610a2)
- [x] R5 review modes declared in models.yaml (review_modes: checklist, model; no triggers: the orchestrator chooses the mode per dispatch); built-in quality/security overridable; `<model>:<mode>` selects any declared mode — round31 — consolidator reintroduced triggers (P3); routed back to remove them (merged 63610a2, no triggers)
- [x] R3b round31 review gaps found by orchestrator: consolidator excludes merged branch TIPS (not approved commits) from its audit; approval recorded before the harness commit of the reviewed tree (re-review) — steered e32dcc08 rev4 — fix1 (approved SHA exclusion) merged; fix2 ineffective (guard runs after the harness commit) -> R3c round38 (minis)
- [x] L1 loops of equivalent commands without progress are not detected (consolidator e32dcc08: ~60 steps re-running the suite; consolidators exempt from escalation) — round32 (merged 365dece)
- [x] L2 workers retry writes to /tmp after a sandbox denial instead of using $TMPDIR: add a hint — round32 (merged 365dece)
- [x] L3 degenerate reasoning: 55 consecutive turns of reasoning_content == 32 x "!" in consolidator e32dcc08, unnoticed by the harness (root cause + guard) — round32; evidence: each degenerate reply had no tool call and the harness answered "No bash command found" ~55 times without escalating (merged 365dece)
- [x] L4 `steer <id> sal del loop!` (unquoted) delivered only "sal": CLI keeps the first positional word and drops the rest silently — round32 (merged 365dece)
- [x] minis model added to models.yaml (combo:minis) and trialled on M2 (help usage line repeated its verb): correct fix + meaningful test, but 57/60 steps and 17 min for a one-line bug, one blocked `cd /home/rot` — 3b34785c merged
- [x] M3-M6 minis round: rustdoc link warnings (43) in two halves, 21 undocumented env vars + drift test, sensitive paths missing intercept/env/steer/handlers — merged via consolidator 1b41841f (minis: 4/4 completed; M5 needed one steer)
- [x] M7 4 rustdoc warnings came back from code merged meanwhile; fix + add `RUSTDOCFLAGS=-D warnings cargo doc` to the AGENTS.md full gate (minis) — merged via consolidator e1d01220
- [x] P1 per-model `instructions` (string or list) in models.yaml appended to that model's system prompt; then give minis "read whole files" — round34 (merged c3168ff)
- [x] P2 workers are not told that the harness checkpoints mid-run: `git diff` goes empty and they think edits are lost (affab410); give base sha + `git diff <base>` hint (opening message, rule 13, checkpoint notice) — merged via consolidator 8ff64c72
- [x] P3 the consolidator sees only the original task (E26), not the orchestrator's later scope steers, so it reverted the user's "no triggers" decision on R5; give it the steer amendments — round36 (merged, nonce-authenticated steer log)
- [~] minis instructions after P1: measured 25/26, 33/37, 39/100 range reads (`sed -n X,Yp`) vs ~0 whole-file reads -> "read whole files" — added to models.yaml (cat whole files, batch reads, stop at gate); measure on next minis task; R3c (old header): 12 range / 4 cat in 57 turns (21% vs 39-96% before); round44 still got the OLD header (daemon on a stale build: release not rebuilt since round36 — rebuilt 19:09, handover armed) -> re-measure on the next minis task; W3b with RD1 active: 45 turns, 10 range reads, 3 auto-expanded; UI3: 74 turns, 23 ranges on monitor.rs (2126 lines, legitimately large)
- [x] F6 tests/watch_reconnect_test flaked in the divergent gate env under host load (release build running concurrently); round34 consolidator raised its two polling timeouts 30s->60s — check it does not recur — did not recur in final gate 3 (2 envs green)
- [x] P4 per-model instructions header reads as optional background ("Model-specific instructions (declared ...)"): make it MANDATORY DIRECTIVES — round39 (merged)
- [x] G1 merge of a consolidator succeeds silently while an integrated worker has commits it never integrated (44e2eb23 4ca2e32 after round31) -> refuse unless --force — round40 (steered: integrated = ancestor OR merge-tree no-op, round38 squash case); steered: also check round members left out (not only integrated set) (merged 4d0d1fe)
- [x] G2 retirement sweep keeps squash-integrated workers forever (ce1813b4 after round38: content in master, tip not an ancestor); use the G1 integrated predicate (ancestor OR merge-tree no-op) for retirement too — after round40 (2nd case: caca0617 after round42) — round45 (merged)
- [x] G3 auto-consolidation started round41 while 9c8264b1 was in its review phase (status Reviewing counted as stopped); the consolidator listed it "not ready" and completed without CONSOLIDATE_WAIT. Fix: Reviewing is live for the round trigger; consolidator prompt: wait for not-ready workers instead of finishing without them — round42; ROOT CAUSE: registry rows written non-atomically (fs::write), tick read a torn row and dropped 9c8264b1 (hub.log 20:59:15) -> steered 62cc1c52: atomic shared helper (merged, atomic registry rows)
- [x] Q1 multi-line ASK_ORCHESTRATOR is truncated to its first line and the rest is lost (49e3e4dd step 52: "Two decisions on merge-G1, both forced by your revision request." only; hub.log 23:52:15): keep the whole question (bounded) in the pause state, events and status — round44 (minis) (merged)
- [x] G3b registry rows: atomic_write_registry_row fsyncs every write (sync_all) though only rename atomicity is needed; one fsync per worker step is avoidable I/O — round44 (minis) (merged)
- [x] R5r triggers residue on master: README documents review-mode triggers, dead validate_glob — round40; then discard 44e2eb23 (merged 4d0d1fe; 44e2eb23 discarded)
- [x] T1 test policy in AGENTS.md (tests only for breakable behaviour; never prompt/help wording; extend the module's existing test file) — round41 (merged f757d30)
- [x] T2 unify 88 integration test binaries (12.8 GiB, 89 links per gate; rebuild 4m25s after touching lib) into tests/it/ — fix 3 env/cwd-mutating files; baseline link time first — round41 (merged f757d30: rebuild 4m25s->47s, test exes 12.8 GiB->262 MiB)
- [x] T2b three tests still mutate process env (SWE_HUB_DIR hub_test, MAX_WORKERS_PER_AGENT mcp_test, SWE_TEMP_DIR watch_test, never restored) because the code reads them in-process; ENV_MUTEX does not cover children spawned meanwhile: give that code explicit config and drop the mutations — round43 (merged)
- [x] T3 drop/relax tests that only pin prompt/nudge prose (keep CLI/wire contract tests) — round41 (merged f757d30)
- [x] W3 dispatch --quiet drops the watch reminder (I forgot to arm watch after round45; user asked for a "remember to use the watch" tip) — round46 (minis) (merged)
- [x] RD1 range reads of small files: minis still did 17 sed ranges on 221-409-line files in 61 turns under the MANDATORY header; user chose a harness fix for all models (expand small range reads to the whole file, worktree-confined, repeat guard) — round47; review: TOCTOU (intermediate dir swapped to symlink between canonicalize and open) -> steered 7b8c9804: verify fd via /proc/self/fd + dev/ino, add whole_file.rs to sensitive paths (merged)
- [x] W3b dispatch --quiet --consolidate reminder suggests the token watch (`MINI_SWE_WATCH_TOKEN=… mini-swe-mcp watch`) instead of the round form; should be `MINI_SWE_WATCH_TOKEN=… mini-swe-mcp watch --group <g> --all` (round48 dispatch) — round49 (minis) (merged)
- [x] H1 tests leave swe-tmp-swe-test-rep-* dirs and swe-wt-h4-steer-gate.steer-log.jsonl in the real /var/tmp (final gate 3) — round48 (merged)
- [x] RV1 (proposal, ask user) auto security review on small sensitive diffs costs ~2x the work (W3b: 45 minis turns + ~79 review turns for a reminder line); option: scale the review budget with diff size, reviewer can still REQUEST_TURNS — user approved; round66 (merged + pushed)
- [x] LG1 hub.log 6.6 MB and unbounded (no rotation); 1.7 MB of it is raw cargo test output logged by "Divergent verify variant B finished" — round50 (merged)
- [x] TS1 build-target slots grew to 39 GB (6 slots, never pruned; one-time cleanup freed ~33 GB incl. 2590 stale test artifacts and 774 old test leftover dirs) -> per-slot size cap on lease + 7-day unused-slot reclaim — round51 (merged)
- [x] R6 review config: built-in quality/security prompts hard-coded, mode field `model`, `--review-after security` parsed as a model, separate `strongest:` (automatic security review ran on ninja); user decision: review_modes only, built-ins as catalog entries, `default_model`, `--review-after <mode>` / `<model>:<mode>`, drop strongest; `help models` documents all top-level keys — round52 (merged; user to add review_modes.security.default_model in models.yaml)
- [x] ENV1 ~50 set_var/remove_var in lib unit tests (config, bootstrap, cache SWE_TEMP_DIR, divergent, agent/env secrets, agent/exec incl. DISABLE_LANDLOCK) run in parallel in one process; recurring worker habit (T2b, TS1): replace with explicit seams + a guard test that fails on any set_var/remove_var in src/ or tests/ — round53 (merged 26e8b20, guard test enforces it)
- [x] UI1 user request: compact interactive monitor (mini-TUI, no new crates: list + per-worker turn detail, keys, terminal restore, incremental history reads) — round54 (merged)
- [x] UI2 monitor ignores terminal width (rows 123 cols at 60), no borders/colour, PID and combo: prefix waste space; operator: "se ve peor en pantallas con poco ancho" — bordered compact layout with width clip and column dropping — round55 (merged; verified in a 50/70-col pty)
- [x] OTA1 hub handover auto-continue restarts interrupted workers with DEFAULT_REVISION_TURNS (60) instead of their own budget (250): 6df0481f exhausted at step 90 mid-gate (hub.log 08:22:46) — round56; until merged: do not rebuild release while workers run (each rebuild = handover = budget cut); continue any worker exhausted by it; review: clamp resumed max_turns to MAX_TURNS_LIMIT + 10-turn floor past resumed step -> steered 2a76a307 (merged)
- [x] UI3 monitor polish at 50 cols: redundant [group]/[review] tags eat the op (~6 chars), columns not dropped to protect op, COLUMNS ignored without TTY — round57 (minis) (merged; verified in a 50-col pty)
- [x] RV2 consolidator security scope includes base-branch commits merged in for conflict resolution (round57 re-audited ENV1 files for 60+ turns) -> exclude commits reachable from the base tip — round58 (merged; uses --remerge-diff so conflict resolutions stay in scope)
- [x] UI4 monitor drops optional columns per row (ragged table at 60 cols) -> decide once per frame, aligned op column — round59 (minis) (merged)
- [x] F7 tests/it hub_handover test flakes intermittently (1/25 under load) — seen by round59 consolidator; unrelated to monitor; details in its REPORT — the_cli_watch_survives_a_daemon_restart, watch exit 2 at :630 — round60 (merged: watch_snapshot skeleton seeded from list_workers state; 610/610 runs under load)
- [x] RO1 the read-only stagnation guard paused F7 (e755438f) after 45 turns although it was running a reproduction loop and narrowing a race: running tests/scripts is experimentation, not reading — count only pure reads, or exempt turns whose output is new — round61 (merged)
- [x] RV2b own_history fallback for git < 2.36 omits merge-commit files (plain `git log --name-only` prints no diff for merges): add `--cc` (or `-m`) so the fallback is really a superset, as its comment claims. Host has git >= 2.36, low priority — round61 (minis) (merged, -m)
- [x] H2 lib unit tests leave /tmp/swe-slot-cap-base-* (cache.rs) and /var/tmp/swe-tmp-divergent-* (divergent.rs); hygiene check covers only tests/it — round62 (minis) (merged)
- [x] H3 harness checkpoint committed a 164+173 MB kache cache (.envcheck/home/.cache) from worker c47b0d24, blocking the push -> size cap + cache-dir exclusion on every harness commit — round63 (merged + pushed)
- [x] UI5 interactive monitor: OPOST cleared in raw mode -> bare \n staircases lines; full-screen clear per refresh flickers; detail drifts and turns are hard to tell apart (operator report) -> \r\n, flicker-free redraw, separated turn blocks — round64; pty test: CRLF ok, no full clears; detail defects (reversed order, opens at oldest, fence lines, clipped separator, clipped footer) -> steered 0e30acde (merged + pushed; pty re-test 0 bare LF)
- [x] Final full gate (1524/1524 x2, 0 leftovers), release build, daemon handed over, pushed a03c8cc..67b56d8 (history rewritten to drop .envcheck kache blobs; backup/pre-filter kept locally)

## Round 2 (group round2) then round 3
- [x] When round2 workers finish: `consolidate --group round2 --model nerd`, read REPORT, merge with `mini-swe-mcp merge <consolidator>` — 72047156: 10 workers, 7 approved / 3 fixed, merge took 0.08 s (gate skipped: verified + up to date)
- [x] Round 3 (group round3) — consolidator 38ee57f3: 7 workers, 6 interaction points (27 in round 2), merged with `merge` (gate skipped): E2 watch hint only without active watch, E3 status text health, E4 tell the worker the exact verify string, E5 dispatch --quiet, E12 consolidate renderer

- [x] L5 loop detector missed 25 turns of `echo "ORCHESTRATOR_CHECK_<n>"` with an incrementing n (11a7900d, steps 205-229, after its REPORT): check why L1 (digit-folded output, K=4/12) did not fire for echo commands — likely echo is treated as a sentinel/exempt — round65 (merged + pushed)
## Pending for the next session (2026-10-04, quota closing)
- round63 H3 (commit size cap + cache-dir exclusion, worker 11a7900d): was in its security review. Verify `merge-tree` integration of its consolidator, then `mini-swe-mcp merge <consolidator>`.
- round64 UI5 (interactive monitor): consolidator 0e30acde steered for the last bug (multi-line command text breaks the frame; no-command turns shown as ✗). Re-test in a 60x22 pty (script + stty, check 0 bare LF), then merge.
- After both: cargo build --release (OTA is safe), full gate (`bash /dev/shm/mini-swe-selfimprove/gate.sh`; /dev/shm is lost on reboot — recreate from rounds.log notes if needed), check blobs > 50 MB in origin/master..master, `git push origin master`.
