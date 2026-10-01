//! The `verified` verdict is persisted on the registry row, so a compact event
//! or a round manifest built from rows alone still reports it after the
//! in-memory record is gone (a restart, a reaped worker, another process).
//!
//! `WorkerMeta::entry` writes the verdict with the terminal row, `collect`
//! reads it back for a record this process no longer holds, `round_manifest`
//! falls back to it, and a new revision starts from a row with no verdict.

mod common;

use std::path::{Path, PathBuf};
use std::time::Duration;

use mini_swe_mcp::agent::{ChatMessage, Role};
use mini_swe_mcp::mcp::{LOCAL_AGENT, McpServer};
use mini_swe_mcp::pool::{
    RegistryStatus, WorkerHistory, WorkerMetrics, WorkerPool, WorkerRegistryEntry, WorkerRole,
    WorkerState, append_history_message_in, load_registry_entry_in, save_registry_entry_in,
    unix_timestamp,
};
use mini_swe_mcp::worktree::ScratchRoot;
use serde_json::json;

const OWNER: &str = LOCAL_AGENT;
const GROUP: &str = "registry-verified";

fn git(dir: &Path, args: &[&str]) {
    let out = std::process::Command::new("git")
        .current_dir(dir)
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("git {args:?}: {e}"));
    assert!(
        out.status.success(),
        "git {args:?} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// A throwaway git repository a dispatch runs against.
struct TestRepo {
    dir: PathBuf,
}

impl TestRepo {
    fn new(tag: &str) -> Self {
        let dir = std::env::temp_dir().join(common::unique_suffix(&format!("regver-{tag}")));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create scratch repo");
        let dir = dir.canonicalize().expect("canonicalize scratch repo");
        git(&dir, &["init", "--initial-branch=master"]);
        git(&dir, &["config", "user.name", "mini-swe-test"]);
        git(&dir, &["config", "user.email", "test@localhost"]);
        std::fs::write(dir.join("README.md"), "# scratch\n").expect("seed file");
        git(&dir, &["add", "README.md"]);
        git(&dir, &["commit", "-m", "baseline"]);
        Self { dir }
    }

    fn path(&self) -> &Path {
        &self.dir
    }

    /// Add the `worker-<id>` branch a finished run leaves behind.
    fn with_worker_branch(&self, id: &str) {
        let branch = format!("worker-{id}");
        git(&self.dir, &["checkout", "-q", "-b", &branch]);
        std::fs::write(self.dir.join("README.md"), "# scratch\nwork\n").expect("edit file");
        git(&self.dir, &["add", "README.md"]);
        git(&self.dir, &["commit", "-m", "work"]);
        git(&self.dir, &["checkout", "-q", "master"]);
    }
}

impl Drop for TestRepo {
    fn drop(&mut self) {
        // A revision leases a build directory keyed by this repo's hash, filed
        // next to the scratch base, so removing the repo takes the lease too.
        mini_swe_mcp::cache::remove_build_dir_leases(&self.dir);
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// A registry row for a worker this process no longer holds.
fn row(
    id: &str,
    repo: &Path,
    status: RegistryStatus,
    verified: Option<bool>,
) -> WorkerRegistryEntry {
    WorkerRegistryEntry {
        id: id.to_string(),
        pid: std::process::id(),
        task: format!("task for {id}"),
        model: "test-model".to_string(),
        status,
        step: 4,
        max_turns: 10,
        last_command: "cargo test".to_string(),
        question: None,
        repo_path: Some(repo.to_string_lossy().into_owned()),
        started_at: 0,
        updated_at: unix_timestamp(),
        group: Some(GROUP.to_string()),
        role: WorkerRole::Worker,
        owner: Some(OWNER.to_string()),
        metrics: WorkerMetrics::default(),
        base_branch: Some("master".to_string()),
        base_commit: Some("base".to_string()),
        revision: 0,
        auto_continues: 0,
        report: None,
        approved: None,
        verified,
    }
}

/// A replayable conversation for `id`, so a revision resumes rather than rebuilds.
fn history(id: &str, repo: &Path) -> WorkerHistory {
    WorkerHistory {
        task: "fix the parser".to_string(),
        group: Some(GROUP.to_string()),
        role: WorkerRole::Worker,
        model: "test-model".to_string(),
        temperature: None,
        repo_path: repo.to_string_lossy().to_string(),
        base_commit: "base".to_string(),
        base_branch: Some("master".to_string()),
        branch: format!("worker-{id}"),
        network_offline: false,
        verify: None,
        client_env: Vec::new(),
        max_turns: 10,
        review_after: None,
        revision: 0,
        auto_continues: 0,
        owner: Some(OWNER.to_string()),
        messages: vec![
            ChatMessage::text(Role::System, "system prompt"),
            ChatMessage::text(Role::User, "TASK:\nfix the parser"),
            ChatMessage::text(Role::Assistant, "I will look at the parser."),
            ChatMessage::text(Role::User, "ok"),
        ],
    }
}

/// Poll until `id` reaches a terminal state.
async fn wait_for_terminal(pool: &WorkerPool, id: &str) -> WorkerState {
    for _ in 0..600 {
        if let Some(state) = pool.get_worker_state(id).await {
            match state {
                WorkerState::Completed { .. } | WorkerState::Failed { .. } => return state,
                WorkerState::Running { .. } | WorkerState::Paused { .. } => {}
            }
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("worker {id} did not reach a terminal state");
}

/// A worker that completes with a passed verify gate leaves the verdict on its
/// registry row, where a later view reads it.
#[tokio::test]
async fn a_completed_workers_row_persists_its_verdict() {
    let repo = TestRepo::new("persist");
    let llm = common::fake_llm::FakeLlm::spawn("rustc --version", "echo done").await;
    let scratch = common::TempDir::new_in_tmp("regver-persist");
    let root = ScratchRoot::new(scratch.path());
    let pool = WorkerPool::with_scratch(
        1,
        llm.base_url().to_string(),
        "test-key".to_string(),
        root.clone(),
    );
    let id = pool
        .dispatch(
            OWNER.to_string(),
            "persist the verdict".to_string(),
            "test-model".to_string(),
            None,
            repo.path().to_path_buf(),
            6,
            Some(GROUP.to_string()),
            None,
            false,
            Some("rustc --version".to_string()),
            Vec::new(),
        )
        .await
        .expect("dispatch the worker");

    let state = wait_for_terminal(&pool, &id).await;
    assert!(
        matches!(
            state,
            WorkerState::Completed {
                verified: Some(true),
                ..
            }
        ),
        "expected a verified completion: {state:?}"
    );

    let saved = load_registry_entry_in(&root, &id).expect("the completion wrote a row");
    assert_eq!(saved.status, RegistryStatus::Completed);
    assert_eq!(
        saved.verified,
        Some(true),
        "the terminal row must carry the verdict: {saved:?}"
    );
}

/// A pool holding no record answers `collect` from the row, verdict included.
#[tokio::test]
async fn a_collected_row_still_reports_its_verdict() {
    let repo = TestRepo::new("collect");
    let scratch = common::TempDir::new_in_tmp("regver-collect");
    let root = ScratchRoot::new(scratch.path());
    let id = "regver-collect";
    save_registry_entry_in(
        &root,
        &row(id, repo.path(), RegistryStatus::Completed, Some(true)),
    );

    let pool = WorkerPool::with_scratch(1, "http://localhost:1".to_string(), "k".to_string(), root);
    let collected = pool
        .collect(id)
        .await
        .expect("a completed row is collectible");
    assert!(
        matches!(
            &collected.state,
            WorkerState::Completed {
                verified: Some(true),
                ..
            }
        ),
        "collect_from_registry must read the verdict: {:?}",
        collected.state
    );
}

/// The round manifest is built from rows, so it must read the row's verdict
/// instead of printing `verified=unknown`.
#[tokio::test]
async fn the_round_manifest_reads_the_verdict_from_the_row() {
    let repo = TestRepo::new("manifest");
    let scratch = common::TempDir::new_in_tmp("regver-manifest");
    let root = ScratchRoot::new(scratch.path());
    let id = "regver-manifest";
    repo.with_worker_branch(id);
    save_registry_entry_in(
        &root,
        &row(id, repo.path(), RegistryStatus::Completed, Some(true)),
    );

    let pool = WorkerPool::with_scratch(1, "http://localhost:1".to_string(), "k".to_string(), root);
    let manifest = pool.round_manifest(OWNER, GROUP, repo.path()).await;
    let rendered = manifest.render();
    assert!(
        rendered.contains("verified=yes"),
        "the manifest must show the row's verdict:\n{rendered}"
    );
}

/// A watch view built from a registry-only row carries the verdict.
#[tokio::test]
async fn the_watch_view_reads_the_verdict_from_the_row() {
    let repo = TestRepo::new("watch");
    let scratch = common::TempDir::new_in_tmp("regver-watch");
    let root = ScratchRoot::new(scratch.path());
    let id = "regver-watch";
    // A terminal row is pruned from the whole-registry scan without a branch
    // or worktree to point at, so leave the branch a finished run would.
    repo.with_worker_branch(id);
    save_registry_entry_in(
        &root,
        &row(id, repo.path(), RegistryStatus::Completed, Some(true)),
    );

    let pool = WorkerPool::with_scratch(1, "http://localhost:1".to_string(), "k".to_string(), root);
    let server = McpServer::new(pool, "test-model".to_string());
    let result = server
        .execute_tool("worker", json!({"action":"watch","worker_id":id}))
        .await
        .expect("a completed row is reported on the first poll");

    let events = result["events"].as_array().expect("events array");
    assert_eq!(events.len(), 1, "one event per terminal worker: {result}");
    assert_eq!(events[0]["event"], json!("completed"));
    assert_eq!(events[0]["verified"], json!(true), "{result}");
    let compact = mini_swe_mcp::cli::watch::render(&events[0]);
    assert!(
        compact.contains("Verified: true"),
        "the compact event must show the verdict: {compact}"
    );
}

/// A new revision starts a fresh review, so the row it writes must have no
/// verdict until the relaunched worker completes.
#[tokio::test]
async fn a_new_revision_clears_the_persisted_verdict() {
    let repo = TestRepo::new("revision");
    let scratch = common::TempDir::new_in_tmp("regver-revision");
    let root = ScratchRoot::new(scratch.path());
    let id = "regver-revision";
    repo.with_worker_branch(id);
    save_registry_entry_in(
        &root,
        &row(id, repo.path(), RegistryStatus::Completed, Some(true)),
    );
    let meta = history(id, repo.path());
    for message in &meta.messages {
        append_history_message_in(&root, id, &meta, message).expect("append history");
    }

    // An unreachable provider keeps the relaunched worker from completing, so
    // the row the revision started from is the one read back.
    let pool = WorkerPool::with_scratch(
        1,
        "http://127.0.0.1:1".to_string(),
        "k".to_string(),
        root.clone(),
    );
    let outcome = pool
        .steer(id, "the retry logic still drops the last page".to_string())
        .await
        .expect("a completed worker is revised");
    assert!(
        matches!(outcome, mini_swe_mcp::pool::SteerOutcome::Continuing { .. }),
        "a completed worker must be revised: {outcome:?}"
    );

    let started = load_registry_entry_in(&root, id).expect("the revision wrote a row");
    assert_eq!(
        started.verified, None,
        "a fresh revision must clear the verdict: {started:?}"
    );
}
