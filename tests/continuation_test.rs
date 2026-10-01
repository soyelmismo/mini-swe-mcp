//! Continuing a stopped worker: the durable log, the CONTINUE prefix, cold
//! continuation, the steer reply and the transient-failure guarantee.
//!
//! Every test here pins one property of the principle that a worker which
//! stopped for *any* reason is continued with `steer` on the same id and
//! branch, and that `failed` only means "this cannot continue by itself".

mod common;

use std::path::Path;
use std::time::Duration;

use mini_swe_mcp::agent::{ChatMessage, Role};
use mini_swe_mcp::mcp::{LOCAL_AGENT, McpServer};
use mini_swe_mcp::pool::{
    LogBuffer, RegistryStatus, WorkerHistory, WorkerMetrics, WorkerPool, WorkerRecord, WorkerState,
    append_history_message_in, history_log_path_in, load_registry_entry_in, load_worker_history_in,
    remove_registry_entry_in, save_registry_entry_in,
};

/// A per-test scratch root, owning its directory.
///
/// Every pool and every registry/history/steer call in this file resolves
/// under it, so no test writes to the real registry under `swe_base_dir()`.
struct Scratch {
    _dir: common::TempDir,
    dir: std::path::PathBuf,
}

impl Scratch {
    fn new(tag: &str) -> Self {
        let dir = common::TempDir::new_in_tmp(tag);
        let path = dir.path().to_path_buf();
        Self {
            _dir: dir,
            dir: path,
        }
    }

    fn path(&self) -> &Path {
        &self.dir
    }

    /// The root, as the `*_in` helpers and `WorkerPool::with_scratch` want it.
    fn root(&self) -> mini_swe_mcp::worktree::ScratchRoot {
        mini_swe_mcp::worktree::ScratchRoot::new(&self.dir)
    }
}

/// A minimal replayable conversation for `worker_id`.
fn history(worker_id: &str, repo: &Path) -> WorkerHistory {
    WorkerHistory {
        task: "fix the parser".to_string(),
        group: None,
        role: mini_swe_mcp::pool::WorkerRole::Worker,
        model: "test-model".to_string(),
        temperature: None,
        repo_path: repo.to_string_lossy().to_string(),
        base_commit: "base".to_string(),
        base_branch: Some("master".to_string()),
        branch: format!("worker-{worker_id}"),
        network_offline: false,
        verify: None,
        client_env: Vec::new(),
        max_turns: 10,
        review_after: None,
        revision: 1,
        auto_continues: 0,
        owner: None,
        messages: vec![
            ChatMessage::text(Role::System, "system prompt"),
            ChatMessage::text(Role::User, "TASK:\nfix the parser"),
            ChatMessage::text(Role::Assistant, "I will look at the parser."),
            ChatMessage::text(Role::User, "ok"),
        ],
    }
}

/// A git repo with a `master` branch and a `worker-<id>` branch, as dispatch
/// leaves behind.
fn repo_with_branch(tag: &str, id: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "swe-cont-repo-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let git = |args: &[&str]| {
        let out = std::process::Command::new("git")
            .current_dir(&dir)
            .args(args)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "git {args:?}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    };
    git(&["init", "--initial-branch=master"]);
    git(&["config", "user.email", "t@t"]);
    git(&["config", "user.name", "t"]);
    std::fs::write(dir.join("a.txt"), "base\n").unwrap();
    git(&["add", "."]);
    git(&["commit", "-m", "base"]);
    git(&["checkout", "-b", &format!("worker-{id}")]);
    std::fs::write(dir.join("a.txt"), "work\n").unwrap();
    git(&["add", "."]);
    git(&["commit", "-m", "work"]);
    // Back to master, so the repo looks like the `repo_path` dispatch records:
    // the base branch is what is checked out there.
    git(&["checkout", "master"]);
    dir
}

/// A registry row for `id` in `status`, as a stopped run leaves behind.
fn row(id: &str, repo: &Path, status: RegistryStatus) -> mini_swe_mcp::pool::WorkerRegistryEntry {
    mini_swe_mcp::pool::WorkerRegistryEntry {
        id: id.to_string(),
        pid: std::process::id(),
        task: "fix the parser".to_string(),
        group: None,
        role: mini_swe_mcp::pool::WorkerRole::Worker,
        model: "test-model".to_string(),
        status,
        step: 4,
        max_turns: 10,
        last_command: "cargo test".into(),
        question: None,
        repo_path: Some(repo.to_string_lossy().to_string()),
        started_at: 0,
        updated_at: 0,
        owner: None,
        metrics: Default::default(),
        base_branch: Some("master".into()),
        base_commit: Some("base".into()),
        head_commit: None,
        revision: 1,
        auto_continues: 0,
        report: None,
        approved: None,
    }
}

#[test]
fn the_append_only_log_survives_a_torn_last_line() {
    let scratch = Scratch::new("torn");
    let root = scratch.root();
    let repo = repo_with_branch("torn", "torn1");
    let meta = history("torn1", &repo);

    // One line per message, as the turn loop pushes them.
    for msg in &meta.messages {
        append_history_message_in(&root, "torn1", &meta, msg).expect("append");
    }

    // Simulate a crash mid-append: the last line is torn.
    let path = history_log_path_in(&root, "torn1");
    let raw = std::fs::read_to_string(&path).unwrap();
    let lines: Vec<&str> = raw.lines().collect();
    assert_eq!(lines.len(), 5, "metadata plus four messages");
    eprintln!("LINES={} PATH={}", lines.len(), path.display());
    let torn = format!(
        "{}\n{}",
        lines[..4].join("\n"),
        &lines[4][..lines[4].len() / 2]
    );
    std::fs::write(&path, torn).unwrap();

    // The reload keeps everything before the torn line and never fails.
    let reloaded =
        load_worker_history_in(&root, "torn1").expect("a torn last line must not fail the reload");
    assert_eq!(
        reloaded.messages.len(),
        3,
        "the torn line is dropped, the messages before it are kept"
    );
    assert_eq!(reloaded.branch, "worker-torn1");
    assert_eq!(reloaded.task, "fix the parser");

    // A continuation can still be built from what survived.
    assert!(mini_swe_mcp::pool::is_replayable(&reloaded.messages));
    let _ = std::fs::remove_dir_all(&repo);
    drop(scratch);
}

#[test]
fn the_log_is_append_only_and_never_rewritten() {
    let scratch = Scratch::new("append");
    let root = scratch.root();
    let repo = repo_with_branch("append", "app1");
    let meta = history("app1", &repo);
    let path = history_log_path_in(&root, "app1");

    append_history_message_in(&root, "app1", &meta, &meta.messages[0]).unwrap();
    let first = std::fs::read_to_string(&path).unwrap();
    append_history_message_in(&root, "app1", &meta, &meta.messages[1]).unwrap();
    let second = std::fs::read_to_string(&path).unwrap();

    assert!(
        second.starts_with(&first),
        "appending must not rewrite what is already there"
    );
    assert_eq!(second.lines().count(), first.lines().count() + 1);
    let _ = std::fs::remove_dir_all(&repo);
    drop(scratch);
}

#[test]
fn a_legacy_whole_file_history_is_still_read() {
    let scratch = Scratch::new("legacy");
    let root = scratch.root();
    let repo = repo_with_branch("legacy", "leg1");
    let meta = history("leg1", &repo);
    let raw = serde_json::to_string(&meta).unwrap();
    std::fs::write(
        scratch.path().join("swe-wt-leg1.history.json"),
        raw.as_bytes(),
    )
    .unwrap();

    let loaded =
        load_worker_history_in(&root, "leg1").expect("the legacy whole-file form is still read");
    assert_eq!(loaded.messages.len(), 4);
    assert!(
        !history_log_path_in(&root, "leg1").exists(),
        "no log is created by a read"
    );
    let _ = std::fs::remove_dir_all(&repo);
    drop(scratch);
}

#[tokio::test]
async fn steer_on_a_failed_worker_with_history_continues_with_the_continue_prefix() {
    let scratch = Scratch::new("warm");
    let root = scratch.root();
    let repo = repo_with_branch("warm", "warm1");
    let meta = history("warm1", &repo);
    for msg in &meta.messages {
        append_history_message_in(&root, "warm1", &meta, msg).unwrap();
    }
    save_registry_entry_in(&root, &row("warm1", &repo, RegistryStatus::Failed));

    let pool = WorkerPool::with_scratch(1, "http://x".into(), "k".into(), root.clone());
    let outcome = pool
        .steer("warm1", "the retry logic still drops the last page".into())
        .await
        .expect("a failed worker with a history is continued");

    assert!(
        matches!(
            outcome,
            mini_swe_mcp::pool::SteerOutcome::Continuing { cold: false, .. }
        ),
        "a surviving conversation is revised, not rebuilt: {outcome:?}"
    );
    assert_eq!(outcome.verb(), "revising");

    // The continuation message names the reason, so the model knows what it is
    // picking up from.
    let reloaded = load_worker_history_in(&root, "warm1").unwrap();
    let last = reloaded.messages.last().unwrap();
    let text = serde_json::to_value(last).unwrap();
    let content = text["content"].as_str().unwrap();
    assert!(
        content.starts_with("CONTINUE: your previous run stopped (Failed). Orchestrator:"),
        "the prefix must fit the reason, got: {content}"
    );
    assert!(content.contains("the retry logic still drops the last page"));

    // The branch is untouched: the continuation re-attaches to it.
    assert_eq!(reloaded.branch, "worker-warm1");
    let _ = std::fs::remove_dir_all(&repo);
    drop(scratch);
}

#[tokio::test]
async fn steer_on_a_worker_without_history_continues_cold_on_the_same_branch() {
    let scratch = Scratch::new("cold");
    let root = scratch.root();
    let repo = repo_with_branch("cold", "cold1");
    // No history at all: a legacy worker whose conversation was never saved.
    save_registry_entry_in(&root, &row("cold1", &repo, RegistryStatus::Failed));

    let pool = WorkerPool::with_scratch(1, "http://x".into(), "k".into(), root.clone());
    let outcome = pool
        .steer("cold1", "keep going".into())
        .await
        .expect("a worker with no history is continued cold");

    assert!(
        matches!(
            outcome,
            mini_swe_mcp::pool::SteerOutcome::Continuing { cold: true, .. }
        ),
        "no history means a cold continuation: {outcome:?}"
    );
    assert_eq!(outcome.verb(), "continuing");

    // The fresh conversation names the original task, the branch that already
    // holds the previous attempt's work, and the steer message.
    let reloaded = load_worker_history_in(&root, "cold1").unwrap();
    assert_eq!(reloaded.branch, "worker-cold1");
    let content = serde_json::to_value(&reloaded.messages[1]).unwrap();
    let content = content["content"].as_str().unwrap();
    assert!(
        content.contains("fix the parser"),
        "the original task is replayed"
    );
    assert!(content.contains("git log --oneline <base>..HEAD"));
    assert!(content.contains("git diff <base>...HEAD --stat"));
    assert!(content.contains("keep going"));
    // The base is resolved from the branch, so the diff is measured honestly.
    assert!(
        !reloaded.base_commit.is_empty(),
        "a base commit is resolved"
    );
    let _ = std::fs::remove_dir_all(&repo);
    drop(scratch);
}

#[tokio::test]
async fn a_missing_branch_is_the_only_cold_continuation_error() {
    let scratch = Scratch::new("nobranch");
    let root = scratch.root();
    let repo = repo_with_branch("nobranch", "gone1");
    save_registry_entry_in(&root, &row("gone1", &repo, RegistryStatus::Failed));
    // The branch is gone: nothing a continuation can work around.
    let out = std::process::Command::new("git")
        .current_dir(&repo)
        .args(["branch", "-D", "worker-gone1"])
        .output()
        .unwrap();
    assert!(out.status.success());

    let pool = WorkerPool::with_scratch(1, "http://x".into(), "k".into(), root.clone());
    let err = pool.steer("gone1", "keep going".into()).await.unwrap_err();
    assert!(
        err.to_string()
            .contains("branch worker-gone1 no longer exists"),
        "the error must name the missing branch, got: {err}"
    );
    let _ = std::fs::remove_dir_all(&repo);
    drop(scratch);
}

#[test]
fn a_history_without_a_base_branch_gets_one_detected_on_continuation() {
    let scratch = Scratch::new("basebranch");
    let root = scratch.root();
    let repo = repo_with_branch("basebranch", "bb1");
    let mut meta = history("bb1", &repo);
    // A history saved before base-branch tracking existed.
    meta.base_branch = None;
    for msg in &meta.messages {
        append_history_message_in(&root, "bb1", &meta, msg).unwrap();
    }

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let detected = runtime.block_on(mini_swe_mcp::pool::ensure_base_branch(&mut meta, &repo));
    assert_eq!(
        detected.as_deref(),
        Some("master"),
        "the base branch is detected the way dispatch does"
    );
    assert_eq!(meta.base_branch.as_deref(), Some("master"));
    let _ = std::fs::remove_dir_all(&repo);
    drop(scratch);
}

#[test]
fn the_auto_continue_cap_is_three() {
    // The daemon continues an interrupted worker at most this many times before
    // it leaves it to the orchestrator.
    assert_eq!(mini_swe_mcp::pool::MAX_AUTO_CONTINUES, 3);
}

#[test]
fn the_auto_continue_budget_counts_down_from_the_cap() {
    let scratch = Scratch::new("budget");
    let root = scratch.root();
    let repo = repo_with_branch("budget", "bud1");
    let mut meta = history("bud1", &repo);
    for msg in &meta.messages {
        append_history_message_in(&root, "bud1", &meta, msg).unwrap();
    }
    save_registry_entry_in(&root, &row("bud1", &repo, RegistryStatus::Interrupted));

    let pool = WorkerPool::with_scratch(1, "http://x".into(), "k".into(), root.clone());
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    // Three automatic continuations, then the worker is left to the
    // orchestrator: a worker the hub keeps losing is not making progress.
    let mut spent = 0;
    for expected in (1..=mini_swe_mcp::pool::MAX_AUTO_CONTINUES).rev() {
        assert_eq!(
            runtime.block_on(pool.auto_continue_budget("bud1")),
            expected,
            "the budget counts down as continuations are spent"
        );
        meta.auto_continues += 1;
        spent += 1;
        // The counter lives in the history metadata, so it survives a restart.
        std::fs::remove_file(history_log_path_in(&root, "bud1")).unwrap();
        for msg in &meta.messages {
            append_history_message_in(&root, "bud1", &meta, msg).unwrap();
        }
    }
    assert_eq!(spent, mini_swe_mcp::pool::MAX_AUTO_CONTINUES);
    assert_eq!(
        runtime.block_on(pool.auto_continue_budget("bud1")),
        0,
        "past the cap the worker stays interrupted for the orchestrator"
    );

    // An interrupted worker with a history is still continuable by hand.
    let outcome = runtime
        .block_on(pool.steer("bud1", "resume".into()))
        .unwrap();
    assert!(matches!(
        outcome,
        mini_swe_mcp::pool::SteerOutcome::Continuing { cold: false, .. }
    ));
    let reloaded = load_worker_history_in(&root, "bud1").unwrap();
    let last = serde_json::to_value(reloaded.messages.last().unwrap()).unwrap();
    assert!(
        last["content"]
            .as_str()
            .unwrap()
            .starts_with("CONTINUE: your previous run stopped (Interrupted). Orchestrator:"),
        "an interrupted worker is continued, not failed"
    );
    let _ = std::fs::remove_dir_all(&repo);
    drop(scratch);
}

#[test]
fn a_transient_llm_error_never_marks_the_worker_failed() {
    // The turn engine pauses with a question instead of failing, so a steer
    // later resumes it: `failed` only means "cannot continue by itself".
    let source = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/pool/runner/turn.rs"
    ))
    .expect("read the turn engine");
    assert!(
        source.contains("Send steer/resume to retry."),
        "a transient LLM error must ask the orchestrator, not fail the worker"
    );
    assert!(
        source.contains("outage_waited"),
        "a provider outage is waited out before the orchestrator is asked"
    );
    // And no transient path reaches the failure transition from an LLM error.
    let failures = source.matches("w.fail(").count();
    assert!(
        failures <= 1,
        "the turn engine must not fail a worker on a transient LLM error"
    );
}

/// A worker that reaches a terminal state, or a panic after the deadline.
async fn wait_until_terminal(pool: &WorkerPool, id: &str) -> WorkerState {
    for _ in 0..600 {
        if let Some(state) = pool.get_worker_state(id).await {
            match state {
                WorkerState::Completed { .. } | WorkerState::Failed { .. } => return state,
                WorkerState::Running { .. } | WorkerState::Paused { .. } => {}
            }
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("worker {id} never reached a terminal state");
}

/// The revision a payload carries, from whichever terminal state it is in.
fn payload_revision(state: &WorkerState) -> Option<usize> {
    match state {
        WorkerState::Completed { revision, .. } | WorkerState::Failed { revision, .. } => {
            Some(*revision)
        }
        WorkerState::Running { .. } | WorkerState::Paused { .. } => None,
    }
}

/// The revision counter advances on every continuation, not just the first.
///
/// The counter is read from three stores (the log metadata line, the registry
/// row and the in-process record) and any of them can lag; a stale read made
/// the second and third steer of one worker report revision 1 again. Steer one
/// stopped worker three times and watch the reply, the payload and the durable
/// history count 1, 2, 3.
#[test]
fn three_continuations_number_one_two_three() {
    let scratch = Scratch::new("three-revisions");
    let root = scratch.root();
    let id = "rev3x";
    let repo = repo_with_branch("three-revisions", id);

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        // A fake LLM that finishes every turn with the completion sentinel, so
        // each revision reaches a terminal state and the next steer continues
        // it instead of queueing behind a running one.
        let llm = common::fake_llm::FakeLlm::spawn("ls -la", "ls -la").await;
        let pool =
            WorkerPool::with_scratch(1, llm.base_url().to_string(), "k".to_string(), root.clone());
        let server = McpServer::new(pool.clone(), "ninja".to_string());

        // What a completed dispatch leaves behind: a log whose metadata line
        // names revision 0 and a terminal record owned by this connection.
        let mut meta = history(id, &repo);
        meta.revision = 0;
        meta.owner = Some(LOCAL_AGENT.to_string());
        // The real base commit, so the preserved branch keeps its commits when
        // the revision's worktree is torn down.
        meta.base_commit = std::process::Command::new("git")
            .current_dir(&repo)
            .args(["rev-parse", "master"])
            .output()
            .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string())
            .expect("rev-parse master");
        for message in &meta.messages {
            append_history_message_in(&root, id, &meta, message).expect("seed the history log");
        }
        pool.__test_insert_worker(WorkerRecord {
            id: id.to_string(),
            task: meta.task.clone(),
            model: meta.model.clone(),
            owner: LOCAL_AGENT.to_string(),
            state: WorkerState::Completed {
                turns: 1,
                diff: String::new(),
                summary: "did what it could".to_string(),
                completed_at: 0,
                artifacts: Vec::new(),
                branch: Some(format!("worker-{id}")),
                verified: None,
                metrics: WorkerMetrics::default(),
                revision: 0,
                report: None,
            },
            metrics: WorkerMetrics::default(),
            logs: LogBuffer::new(),
            pending_steer: Vec::new(),
            resume_tx: None,
            handle: None,
            revision: 0,
        })
        .await;

        // Steer the same stopped worker three times, letting each revision
        // finish before the next steer, and record what every surface reported.
        let mut replies = Vec::new();
        let mut payloads = Vec::new();
        let mut histories = Vec::new();
        for n in 1..=3 {
            let reply = server
                .execute_tool(
                    "worker",
                    serde_json::json!({
                        "action": "steer",
                        "worker_id": id,
                        "message": format!("carry on ({n})"),
                    }),
                )
                .await
                .expect("steering a stopped worker must start a continuation");
            assert_eq!(reply["status"], "revising", "reply: {reply}");
            replies.push(reply["message"].as_str().unwrap_or_default().to_string());
            payloads.push(
                load_registry_entry_in(&root, id)
                    .expect("registry row")
                    .revision,
            );
            histories.push(load_worker_history_in(&root, id).expect("history").revision);
            wait_until_terminal(&pool, id).await;
        }

        for (n, reply) in replies.iter().enumerate() {
            let revision = n + 1;
            assert!(
                reply.starts_with(&format!("Revision {revision} started")),
                "the reply must name revision {revision}, got: {reply}"
            );
        }
        assert_eq!(
            payloads,
            vec![1, 2, 3],
            "the payload of each continuation must be its revision"
        );
        assert_eq!(
            histories,
            vec![1, 2, 3],
            "the durable history must advance with each continuation"
        );
        assert_eq!(
            load_registry_entry_in(&root, id)
                .expect("registry row")
                .revision,
            3,
            "the final registry row must carry the last revision"
        );
        assert_eq!(
            payload_revision(&pool.get_worker_state(id).await.expect("record")),
            Some(3),
            "the final completion payload must carry the last revision"
        );

        // `collect` evicts the record and a `prune` retires the registry row,
        // so a reaped worker can be its durable log and nothing else: the
        // counter must still advance from there instead of restarting at one.
        assert!(
            pool.collect(id).await.is_some(),
            "collecting the terminal worker evicts it from the pool"
        );
        // With the row gone the log is the only copy of the counter.
        remove_registry_entry_in(&root, id);
        assert!(
            load_registry_entry_in(&root, id).is_none(),
            "the row is gone, so only the saved conversation can name the revision"
        );
        let reply = server
            .execute_tool(
                "worker",
                serde_json::json!({
                    "action": "steer",
                    "worker_id": id,
                    "message": "one more",
                }),
            )
            .await
            .expect("a reaped worker with a saved conversation is still continuable");
        assert!(
            reply["message"]
                .as_str()
                .unwrap_or_default()
                .starts_with("Revision 4 started"),
            "a continuation after a reap must advance the counter: {reply}"
        );
        assert_eq!(
            load_worker_history_in(&root, id).expect("history").revision,
            4,
            "the log alone must carry the counter after a reap"
        );
        assert_eq!(
            load_registry_entry_in(&root, id)
                .expect("fresh registry row")
                .revision,
            4,
            "the continuation re-registers the worker at its revision"
        );
        wait_until_terminal(&pool, id).await;
    });

    let _ = std::fs::remove_dir_all(&repo);
    drop(scratch);
}
