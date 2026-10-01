//! A finished worker stays steerable after its in-memory record is reaped.
//!
//! The in-memory TTL bounds *memory*: a `Completed`/`Failed` record leaves the
//! pool after `WORKER_TERMINAL_TTL_SECS`. What must not leave with it is the
//! worker's durable state — its registry row and its saved conversation —
//! because `steer` continues a completed or failed worker on its own id and
//! branch, and an orchestrator does that hours later (review, batch gates,
//! merge conflicts).
//!
//! These tests pin the rule from both ends: evicting a record keeps the row and
//! the log, and only a gone branch or an expired retention retires them.

mod common;

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use mini_swe_mcp::agent::{ChatMessage, Role};
use mini_swe_mcp::pool::{
    CollectedWorker, DEFAULT_TERMINAL_RETENTION_SECS, DEFAULT_WORKER_RETIRED_GRACE_SECS, LogBuffer,
    RegistryStatus, SteerOutcome, WorkerHistory, WorkerMetrics, WorkerPool, WorkerRecord,
    WorkerState, append_history_message_in, history_log_path_in, load_registry_entry_in,
    load_worker_history_in, prune_orphan_histories_with_retention_and_grace_in,
    prune_orphan_histories_with_retention_in, save_registry_entry_in,
};

/// A per-test scratch root, owning its directory.
struct Scratch {
    _dir: common::TempDir,
    dir: PathBuf,
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

    fn root(&self) -> mini_swe_mcp::worktree::ScratchRoot {
        mini_swe_mcp::worktree::ScratchRoot::new(&self.dir)
    }
}

/// A git repo on `master` with a `worker-<id>` branch, as dispatch leaves it.
fn repo_with_branch(tag: &str, id: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "swe-retain-repo-{tag}-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
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
    git(&["checkout", "master"]);
    dir
}

/// The trimmed stdout of one git command in `repo`, asserting it succeeded.
fn git_stdout(repo: &Path, args: &[&str]) -> String {
    let out = std::process::Command::new("git")
        .current_dir(repo)
        .args(args)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "git {args:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}

/// The replayable conversation a finished worker leaves behind.
fn history(worker_id: &str, repo: &Path) -> WorkerHistory {
    WorkerHistory {
        role: mini_swe_mcp::pool::WorkerRole::Worker,
        task: "fix the parser".to_string(),
        group: None,
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

/// A registry row for `id` in `status`, as a stopped run leaves one behind.
fn row(id: &str, repo: &Path, status: RegistryStatus) -> mini_swe_mcp::pool::WorkerRegistryEntry {
    mini_swe_mcp::pool::WorkerRegistryEntry {
        task: "fix the parser".to_string(),
        model: "test-model".to_string(),
        status,
        step: 4,
        last_command: "cargo test".into(),
        repo_path: Some(repo.to_string_lossy().to_string()),
        // Fresh: the retention clock starts when the row was last written.
        updated_at: mini_swe_mcp::pool::unix_timestamp(),
        owner: None,
        base_branch: Some("master".into()),
        base_commit: Some("base".into()),
        revision: 1,
        ..mini_swe_mcp::pool::WorkerRegistryEntry::test_row(id, "")
    }
}

/// A terminal record whose age is already past the pool's TTL, so one `reap`
/// evicts it without the test having to wait or to touch the environment.
fn terminal_record(id: &str, state: WorkerState) -> WorkerRecord {
    WorkerRecord {
        id: id.to_string(),
        task: "fix the parser".into(),
        model: "test-model".into(),
        owner: "test-owner".into(),
        state,
        metrics: WorkerMetrics::default(),
        logs: LogBuffer::new(),
        pending_steer: Vec::new(),
        resume_tx: None,
        handle: None,
        revision: 1,
    }
}

/// Write the durable state a finished worker leaves: row plus conversation.
fn durable_state(scratch: &Scratch, repo: &Path, id: &str, status: RegistryStatus) {
    let meta = history(id, repo);
    for msg in &meta.messages {
        append_history_message_in(&scratch.root(), id, &meta, msg).expect("append history");
    }
    save_registry_entry_in(&scratch.root(), &row(id, repo, status));
}

/// A completed worker whose record the reaper dropped is still continuable.
#[tokio::test]
async fn a_reaped_completed_worker_keeps_its_row_and_history_and_stays_continuable() {
    let scratch = Scratch::new("reapcomp");
    let root = scratch.root();
    let repo = repo_with_branch("reapcomp", "rc1");
    durable_state(&scratch, &repo, "rc1", RegistryStatus::Completed);

    let pool = WorkerPool::with_scratch(1, "http://x".into(), "k".into(), root.clone());
    pool.__test_insert_worker(terminal_record(
        "rc1",
        WorkerState::Completed {
            report: None,
            turns: 4,
            diff: String::new(),
            summary: "done".into(),
            completed_at: 0,
            artifacts: Vec::new(),
            branch: Some("worker-rc1".into()),
            verified: Some(true),
            metrics: WorkerMetrics::default(),
            revision: 1,
        },
    ))
    .await;

    let reaped = pool.reap().await;
    assert_eq!(reaped, vec!["rc1".to_string()], "the record is evicted");
    assert!(
        pool.get_worker_state("rc1").await.is_none(),
        "the in-memory record is gone"
    );

    // The durable state survives the eviction: that is what `steer` reads.
    assert!(
        load_registry_entry_in(&root, "rc1").is_some(),
        "evicting a terminal record must not delete its registry row"
    );
    assert!(
        history_log_path_in(&root, "rc1").exists(),
        "evicting a terminal record must not delete its history log"
    );

    let outcome = pool
        .steer("rc1", "the retry logic still drops the last page".into())
        .await
        .expect("a reaped completed worker must still be continuable");
    assert!(
        matches!(outcome, SteerOutcome::Continuing { cold: false, .. }),
        "the saved conversation is replayed, not rebuilt: {outcome:?}"
    );
    let _ = std::fs::remove_dir_all(&repo);
    drop(scratch);
}

/// A failed worker is continuable for exactly the same reason.
#[tokio::test]
async fn a_reaped_failed_worker_keeps_its_row_and_history_and_stays_continuable() {
    let scratch = Scratch::new("reapfail");
    let root = scratch.root();
    let repo = repo_with_branch("reapfail", "rf1");
    durable_state(&scratch, &repo, "rf1", RegistryStatus::Failed);

    let pool = WorkerPool::with_scratch(1, "http://x".into(), "k".into(), root.clone());
    pool.__test_insert_worker(terminal_record(
        "rf1",
        WorkerState::Failed {
            error: "boom".into(),
            step: 4,
            failed_at: 0,
            metrics: WorkerMetrics::default(),
            revision: 1,
        },
    ))
    .await;

    assert_eq!(pool.reap().await, vec!["rf1".to_string()]);
    assert!(
        load_registry_entry_in(&root, "rf1").is_some(),
        "a failed worker's row must survive its eviction"
    );
    assert!(
        history_log_path_in(&root, "rf1").exists(),
        "a failed worker's history must survive its eviction"
    );

    let outcome = pool
        .steer("rf1", "keep going".into())
        .await
        .expect("a reaped failed worker must still be continuable");
    assert!(
        matches!(outcome, SteerOutcome::Continuing { .. }),
        "a failed worker is continued, not refused: {outcome:?}"
    );
    let _ = std::fs::remove_dir_all(&repo);
    drop(scratch);
}

/// `collect` reads the registry row once the record is gone, so a reaped
/// worker's branch is still reported.
#[tokio::test]
async fn collect_answers_from_the_registry_row_after_the_record_is_reaped() {
    let scratch = Scratch::new("reapcoll");
    let root = scratch.root();
    let repo = repo_with_branch("reapcoll", "rco1");
    durable_state(&scratch, &repo, "rco1", RegistryStatus::Completed);

    let pool = WorkerPool::with_scratch(1, "http://x".into(), "k".into(), root.clone());
    pool.__test_insert_worker(terminal_record(
        "rco1",
        WorkerState::Completed {
            report: None,
            turns: 4,
            diff: String::new(),
            summary: "done".into(),
            completed_at: 0,
            artifacts: Vec::new(),
            branch: Some("worker-rco1".into()),
            verified: Some(true),
            metrics: WorkerMetrics::default(),
            revision: 1,
        },
    ))
    .await;
    pool.reap().await;

    let collected: CollectedWorker = pool
        .collect("rco1")
        .await
        .expect("a reaped worker is still collectible from its registry row");
    match &collected.state {
        WorkerState::Completed { branch, .. } => {
            assert_eq!(branch.as_deref(), Some("worker-rco1"));
        }
        other => panic!("the row describes a completed worker, got {other:?}"),
    }
    let _ = std::fs::remove_dir_all(&repo);
    drop(scratch);
}

/// The branch is what keeps a worker alive, but its disappearance no longer
/// retires the worker at once: the row and conversation survive the retired
/// grace period so a reverted merge can still be continued, and go past it.
#[tokio::test]
async fn a_worker_whose_branch_is_gone_is_kept_through_its_grace_then_retired() {
    let scratch = Scratch::new("branchgone");
    let root = scratch.root();
    let repo = repo_with_branch("branchgone", "bg1");
    durable_state(&scratch, &repo, "bg1", RegistryStatus::Completed);

    let prune = || {
        prune_orphan_histories_with_retention_and_grace_in(
            &root,
            &repo,
            DEFAULT_TERMINAL_RETENTION_SECS,
            DEFAULT_WORKER_RETIRED_GRACE_SECS,
        )
    };

    // The branch is still there, so a prune keeps everything: an eviction is
    // not a reason to delete.
    assert_eq!(prune(), 0, "a worker whose branch exists is not pruned");
    assert!(load_registry_entry_in(&root, "bg1").is_some());
    assert!(history_log_path_in(&root, "bg1").exists());

    // The branch is merged away. Within the grace the row and conversation
    // survive, so the worker can still be continued after a reverted merge.
    let out = std::process::Command::new("git")
        .current_dir(&repo)
        .args(["branch", "-D", "worker-bg1"])
        .output()
        .unwrap();
    assert!(out.status.success());

    assert_eq!(prune(), 0, "the grace keeps the row and the conversation");
    assert!(
        load_registry_entry_in(&root, "bg1").is_some(),
        "the row survives the branch within the grace"
    );
    assert!(
        history_log_path_in(&root, "bg1").exists(),
        "the conversation survives the branch within the grace"
    );

    // Past the grace (still inside the retention) the trace goes.
    let mut aged = row("bg1", &repo, RegistryStatus::Completed);
    aged.updated_at = mini_swe_mcp::pool::unix_timestamp() - DEFAULT_WORKER_RETIRED_GRACE_SECS - 1;
    save_registry_entry_in(&root, &aged);

    assert_eq!(prune(), 1, "a worker past its retired grace is retired");
    assert!(
        load_registry_entry_in(&root, "bg1").is_none(),
        "the row goes with the grace"
    );
    assert!(
        !history_log_path_in(&root, "bg1").exists(),
        "the conversation goes with the grace"
    );
    let _ = std::fs::remove_dir_all(&repo);
    drop(scratch);
}

/// Wait for `id` to reach a terminal state, so its worker task (and the
/// worktree it owns) is finished before the test tears down the repo.
async fn wait_until_terminal(pool: &WorkerPool, id: &str) -> WorkerState {
    for _ in 0..600 {
        if let Some(state) = pool.get_worker_state(id).await
            && matches!(
                state,
                WorkerState::Completed { .. }
                    | WorkerState::Failed { .. }
                    | WorkerState::Exhausted { .. }
            )
            && !pool.scratch_root().join(format!("swe-wt-{id}")).exists()
        {
            return state;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("worker {id} never reached a terminal state");
}

/// A merge pruned the branch, but the row and conversation survive the grace,
/// so a continuation recreates `worker-<id>` from the head commit the
/// completion recorded instead of failing "branch no longer exists".
#[tokio::test]
async fn a_continuation_recreates_a_pruned_branch_from_the_recorded_head() {
    let scratch = Scratch::new("recreate");
    let root = scratch.root();
    let repo = repo_with_branch("recreate", "rc1");
    let head = git_stdout(&repo, &["rev-parse", "refs/heads/worker-rc1"]);
    let base = git_stdout(&repo, &["rev-parse", "master"]);

    // A saved conversation plus the row a completion leaves, naming the head
    // of the branch it committed to.
    let mut meta = history("rc1", &repo);
    meta.base_commit = base.clone();
    for message in &meta.messages {
        append_history_message_in(&root, "rc1", &meta, message).expect("seed the history log");
    }
    let mut recorded = row("rc1", &repo, RegistryStatus::Completed);
    recorded.base_commit = Some(base);
    recorded.head_commit = Some(head.clone());
    save_registry_entry_in(&root, &recorded);

    let out = std::process::Command::new("git")
        .current_dir(&repo)
        .args(["branch", "-D", "worker-rc1"])
        .output()
        .unwrap();
    assert!(out.status.success());
    assert_eq!(
        prune_orphan_histories_with_retention_and_grace_in(
            &root,
            &repo,
            DEFAULT_TERMINAL_RETENTION_SECS,
            DEFAULT_WORKER_RETIRED_GRACE_SECS,
        ),
        0,
        "the grace keeps the pruned worker continuable"
    );

    let llm = common::fake_llm::FakeLlm::spawn("ls -la", "ls -la").await;
    let pool =
        WorkerPool::with_scratch(1, llm.base_url().to_string(), "k".to_string(), root.clone());
    let outcome = pool
        .steer("rc1", "reapply the parser fix".into())
        .await
        .expect("a pruned branch within the grace must be continuable");
    assert!(
        matches!(outcome, SteerOutcome::Continuing { cold: false, .. }),
        "the saved conversation is replayed: {outcome:?}"
    );
    assert_eq!(
        git_stdout(&repo, &["rev-parse", "refs/heads/worker-rc1"]),
        head,
        "the continuation recreated the branch at the recorded head"
    );
    wait_until_terminal(&pool, "rc1").await;
    let _ = std::fs::remove_dir_all(&repo);
    drop(scratch);
}

/// The observed failure: a cold worker (no saved conversation) whose branch
/// was pruned. The row survives the grace, so the continuation rebuilds the
/// branch from its recorded head and the conversation from the row.
#[tokio::test]
async fn a_cold_continuation_recreates_a_pruned_branch_from_the_recorded_head() {
    let scratch = Scratch::new("recreatecold");
    let root = scratch.root();
    let repo = repo_with_branch("recreatecold", "rc2");
    let head = git_stdout(&repo, &["rev-parse", "refs/heads/worker-rc2"]);
    let base = git_stdout(&repo, &["rev-parse", "master"]);
    let mut recorded = row("rc2", &repo, RegistryStatus::Completed);
    recorded.base_commit = Some(base);
    recorded.head_commit = Some(head.clone());
    save_registry_entry_in(&root, &recorded);

    let out = std::process::Command::new("git")
        .current_dir(&repo)
        .args(["branch", "-D", "worker-rc2"])
        .output()
        .unwrap();
    assert!(out.status.success());
    assert_eq!(
        prune_orphan_histories_with_retention_and_grace_in(
            &root,
            &repo,
            DEFAULT_TERMINAL_RETENTION_SECS,
            DEFAULT_WORKER_RETIRED_GRACE_SECS,
        ),
        0,
        "the grace keeps the cold worker's row"
    );

    let llm = common::fake_llm::FakeLlm::spawn("ls -la", "ls -la").await;
    let pool =
        WorkerPool::with_scratch(1, llm.base_url().to_string(), "k".to_string(), root.clone());
    let outcome = pool
        .steer("rc2", "reapply the parser fix".into())
        .await
        .expect("a cold worker whose row survived the grace must be continuable");
    assert!(
        matches!(outcome, SteerOutcome::Continuing { cold: true, .. }),
        "the conversation is rebuilt from the row: {outcome:?}"
    );
    assert_eq!(
        git_stdout(&repo, &["rev-parse", "refs/heads/worker-rc2"]),
        head,
        "the cold continuation recreated the branch at the recorded head"
    );
    wait_until_terminal(&pool, "rc2").await;
    let _ = std::fs::remove_dir_all(&repo);
    drop(scratch);
}

/// The retention is the other end of the rule: a worker whose branch remains
/// is retired once it is older than the retention, and not before.
#[tokio::test]
async fn a_worker_past_its_retention_is_retired_even_though_its_branch_remains() {
    let scratch = Scratch::new("retention");
    let root = scratch.root();
    let repo = repo_with_branch("retention", "rt1");
    durable_state(&scratch, &repo, "rt1", RegistryStatus::Completed);
    // Age the row past the retention without touching the clock.
    let mut aged = row("rt1", &repo, RegistryStatus::Completed);
    aged.updated_at = mini_swe_mcp::pool::unix_timestamp() - DEFAULT_TERMINAL_RETENTION_SECS - 1;
    save_registry_entry_in(&root, &aged);

    assert_eq!(
        prune_orphan_histories_with_retention_in(&root, &repo, DEFAULT_TERMINAL_RETENTION_SECS),
        1,
        "a terminal worker past its retention is retired"
    );
    assert!(load_registry_entry_in(&root, "rt1").is_none());
    assert!(!history_log_path_in(&root, "rt1").exists());

    // A fresh row is kept: the retention is a week, not a TTL. Its own branch
    // exists, so only the age could retire it.
    let fresh_repo = repo_with_branch("retention2", "rt2");
    durable_state(&scratch, &fresh_repo, "rt2", RegistryStatus::Completed);
    assert_eq!(
        prune_orphan_histories_with_retention_in(
            &root,
            &fresh_repo,
            DEFAULT_TERMINAL_RETENTION_SECS
        ),
        0,
        "a worker inside its retention is kept"
    );
    assert!(load_registry_entry_in(&root, "rt2").is_some());
    assert!(history_log_path_in(&root, "rt2").exists());
    let _ = std::fs::remove_dir_all(&repo);
    let _ = std::fs::remove_dir_all(&fresh_repo);
    drop(scratch);
}

/// A worker with no row at all is still continuable from its log alone, and a
/// prune that cannot see a branch keeps it.
#[tokio::test]
async fn a_history_without_a_row_survives_a_prune_while_its_branch_lives() {
    let scratch = Scratch::new("norow");
    let root = scratch.root();
    let repo = repo_with_branch("norow", "nr1");
    let meta = history("nr1", &repo);
    for msg in &meta.messages {
        append_history_message_in(&root, "nr1", &meta, msg).expect("append history");
    }

    assert_eq!(
        prune_orphan_histories_with_retention_in(&root, &repo, DEFAULT_TERMINAL_RETENTION_SECS),
        0,
        "a log whose branch lives has nothing to be pruned for"
    );
    assert!(history_log_path_in(&root, "nr1").exists());
    let reloaded = load_worker_history_in(&root, "nr1").expect("the log is still readable");
    assert_eq!(reloaded.branch, "worker-nr1");
    let _ = std::fs::remove_dir_all(&repo);
    drop(scratch);
}

/// The two clocks are independent: the in-memory TTL is seconds, the durable
/// retention is a week. The ordering is the invariant the whole rule rests on,
/// so it is checked at compile time rather than at run time.
const _: () =
    assert!(DEFAULT_TERMINAL_RETENTION_SECS > mini_swe_mcp::pool::DEFAULT_TERMINAL_TTL_SECS);
const _: () = assert!(DEFAULT_TERMINAL_RETENTION_SECS == 7 * 24 * 60 * 60);
