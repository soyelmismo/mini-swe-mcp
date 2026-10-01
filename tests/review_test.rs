//! The compact review path: what `collect` withholds by default, and what
//! `review` answers in one payload.
//!
//! Both verbs exist because reviewing a worker used to cost a `status`, a
//! `collect` (the whole diff, however large), the `logs` and a hand-run
//! `git merge-tree`. These tests pin the two properties that make the compact
//! form safe to rely on: the diff leaves a `collect` only when it was asked
//! for, and a `review` reports the merge against the base branch tip without
//! touching any worktree.
//!
//! Every repository here is a throwaway one under a per-test scratch root, so
//! the suite never reads or writes the crate's own git state.

mod common;

use common::{IsolatedPool, TempDir};

use mini_swe_mcp::agent::AgentStepLog;
use mini_swe_mcp::mcp::{ConnectionContext, McpServer};
use mini_swe_mcp::pool::{
    LogBuffer, RegistryStatus, WorkerMetrics, WorkerRecord, WorkerRegistryEntry, WorkerState,
    save_registry_entry_in,
};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::process::Command;

/// The agent that owns every worker these tests dispatch.
const OWNER: &str = "review-agent";

/// A connection for `OWNER`, so the ownership check (H-3) passes.
fn owner_context() -> ConnectionContext {
    ConnectionContext {
        agent_id: Some(OWNER.to_string()),
        ..ConnectionContext::hub_connection(7)
    }
}

/// A connection for somebody else's agent.
fn stranger_context() -> ConnectionContext {
    ConnectionContext {
        agent_id: Some("somebody-else".to_string()),
        ..ConnectionContext::hub_connection(8)
    }
}

/// Run `git` in `dir`, panicking with git's own stderr on failure.
fn git(dir: &Path, args: &[&str]) {
    let out = Command::new("git")
        .current_dir(dir)
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("failed to run git {args:?} in {}: {e}", dir.display()));
    assert!(
        out.status.success(),
        "git {args:?} in {} failed: {}",
        dir.display(),
        String::from_utf8_lossy(&out.stderr)
    );
}

/// A two-file unified diff, as the worker loop stores it.
fn two_file_diff() -> String {
    [
        "diff --git a/src/parser.rs b/src/parser.rs",
        "index 1111111..2222222 100644",
        "--- a/src/parser.rs",
        "+++ b/src/parser.rs",
        "@@ -1,2 +1,4 @@",
        " fn parse() {",
        "-    old();",
        "+    new();",
        "+    extra();",
        " }",
        "diff --git a/README.md b/README.md",
        "index 3333333..4444444 100644",
        "--- a/README.md",
        "+++ b/README.md",
        "@@ -1 +1 @@",
        "-old docs",
        "+new docs",
    ]
    .join("\n")
}

/// A completed worker owned by [`OWNER`], carrying `diff`.
fn completed_worker(id: &str, diff: &str) -> WorkerRecord {
    WorkerRecord {
        id: id.to_string(),
        task: "Fix the parser\nand its docs".to_string(),
        model: "test-model".to_string(),
        owner: OWNER.to_string(),
        state: WorkerState::Completed {
            turns: 3,
            diff: diff.to_string(),
            summary: "parser now handles empty input".to_string(),
            completed_at: 0,
            artifacts: Vec::new(),
            branch: Some(format!("worker-{id}")),
            verified: Some(true),
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
    }
}

/// A registry row for a worker whose branch lives in `repo`, as the pool's
/// coalescing writer would have left it.
fn registry_row(id: &str, repo: &Path, revision: usize) -> WorkerRegistryEntry {
    WorkerRegistryEntry {
        task: "Fix the parser\nand its docs".to_string(),
        model: "test-model".to_string(),
        status: RegistryStatus::Completed,
        step: 3,
        max_turns: 60,
        last_command: "cargo test".to_string(),
        repo_path: Some(repo.to_string_lossy().into_owned()),
        base_branch: Some("master".to_string()),
        revision,
        ..WorkerRegistryEntry::test_row(id, OWNER)
    }
}

/// A repository whose `master` moved on after two worker branches forked from
/// it: `worker-rev-clean` only adds a file, `worker-rev-conflict` rewrites the
/// same line `master` rewrote.
fn branch_repo(root: &TempDir) -> PathBuf {
    let repo = root.subdir("repo");
    git(&repo, &["init", "-b", "master", "."]);
    git(&repo, &["config", "user.name", "mini-swe-test"]);
    git(&repo, &["config", "user.email", "test@localhost"]);
    std::fs::write(repo.join("shared.txt"), "base\n").expect("seed the repository");
    git(&repo, &["add", "shared.txt"]);
    git(&repo, &["commit", "-m", "baseline"]);

    git(&repo, &["checkout", "-b", "worker-rev-clean"]);
    std::fs::write(repo.join("added.txt"), "clean work\n").expect("write the clean branch");
    git(&repo, &["add", "added.txt"]);
    git(&repo, &["commit", "-m", "clean work"]);

    git(&repo, &["checkout", "master"]);
    git(&repo, &["checkout", "-b", "worker-rev-conflict"]);
    std::fs::write(repo.join("shared.txt"), "worker side\n").expect("write the conflicting branch");
    git(&repo, &["add", "shared.txt"]);
    git(&repo, &["commit", "-m", "conflicting work"]);

    git(&repo, &["checkout", "master"]);
    std::fs::write(repo.join("shared.txt"), "base tip\n").expect("move the base tip");
    git(&repo, &["add", "shared.txt"]);
    git(&repo, &["commit", "-m", "base moves on"]);
    repo
}

/// `collect` answers with the summary, the verification outcome, the branch and
/// a per-file diff stat — and with no diff at all.
#[tokio::test]
async fn collect_defaults_to_the_stat_and_withholds_the_diff() {
    let owned = IsolatedPool::new(4, "review-collect");
    let server = McpServer::new(owned.pool.clone(), "ninja".to_string());
    owned
        .pool
        .__test_insert_worker(completed_worker("col-default", &two_file_diff()))
        .await;

    let payload = server
        .execute_tool_for(
            "worker",
            json!({ "action": "collect", "worker_id": "col-default" }),
            &owner_context(),
        )
        .await
        .expect("the owner may collect its own worker");

    assert_eq!(payload["summary"], "parser now handles empty input");
    assert_eq!(payload["verified"], true);
    assert_eq!(payload["branch"], "worker-col-default");
    assert_eq!(
        payload["state"]["details"]["diff"],
        Value::Null,
        "the default collect must not carry the diff: {payload}"
    );
    let stat = &payload["diff_stat"];
    assert_eq!(stat["files"], 2);
    assert_eq!(stat["insertions"], 3);
    assert_eq!(stat["deletions"], 2);
    let per_file = stat["per_file"].as_array().expect("per-file counts");
    assert_eq!(per_file.len(), 2);
    assert_eq!(per_file[0]["path"], "src/parser.rs");
    assert_eq!(per_file[0]["insertions"], 2);
    assert_eq!(per_file[0]["deletions"], 1);
    assert_eq!(per_file[1]["path"], "README.md");
    assert_eq!(per_file[1]["insertions"], 1);
    assert_eq!(per_file[1]["deletions"], 1);
}

/// `full: true` is the opt-in that restores the whole diff.
#[tokio::test]
async fn collect_full_returns_the_whole_diff() {
    let owned = IsolatedPool::new(4, "review-full");
    let server = McpServer::new(owned.pool.clone(), "ninja".to_string());
    owned
        .pool
        .__test_insert_worker(completed_worker("col-full", &two_file_diff()))
        .await;

    let payload = server
        .execute_tool_for(
            "worker",
            json!({ "action": "collect", "worker_id": "col-full", "full": true }),
            &owner_context(),
        )
        .await
        .expect("the owner may collect its own worker");

    assert_eq!(
        payload["state"]["details"]["diff"],
        json!(two_file_diff()),
        "full must return the diff verbatim"
    );
    assert_eq!(payload["diff_stat"]["files"], 2);
}

/// `files: [...]` narrows the diff to the named paths and nothing else.
#[tokio::test]
async fn collect_files_returns_only_those_paths() {
    let owned = IsolatedPool::new(4, "review-files");
    let server = McpServer::new(owned.pool.clone(), "ninja".to_string());
    owned
        .pool
        .__test_insert_worker(completed_worker("col-file", &two_file_diff()))
        .await;

    let payload = server
        .execute_tool_for(
            "worker",
            json!({ "action": "collect", "worker_id": "col-file", "files": ["README.md"] }),
            &owner_context(),
        )
        .await
        .expect("the owner may collect its own worker");

    let diff = payload["state"]["details"]["diff"]
        .as_str()
        .expect("a named file must still return its diff");
    assert!(
        diff.contains("diff --git a/README.md b/README.md"),
        "{diff}"
    );
    assert!(
        !diff.contains("src/parser.rs"),
        "a file that was not named must not be returned: {diff}"
    );
    // The stat still describes the whole change, so the caller can see what it
    // is not being shown.
    assert_eq!(payload["diff_stat"]["files"], 2);
}

/// `collect` is owner-only, exactly like the verb it replaces.
#[tokio::test]
async fn collect_refuses_a_stranger() {
    let owned = IsolatedPool::new(4, "review-collect-owner");
    let server = McpServer::new(owned.pool.clone(), "ninja".to_string());
    owned
        .pool
        .__test_insert_worker(completed_worker("col-owner", &two_file_diff()))
        .await;

    let error = server
        .execute_tool_for(
            "worker",
            json!({ "action": "collect", "worker_id": "col-owner" }),
            &stranger_context(),
        )
        .await
        .expect_err("another agent's worker must be refused");
    assert_eq!(
        error.to_string(),
        "worker col-owner belongs to agent review-agent"
    );
}

/// A `review` of a branch that still merges says so, and names the merge.
#[tokio::test]
async fn review_reports_a_clean_merge() {
    let scratch = TempDir::new_in_tmp("review-clean");
    let repo = branch_repo(&scratch);
    let owned = IsolatedPool::new(4, "review-clean-pool");
    let server = McpServer::new(owned.pool.clone(), "ninja".to_string());
    save_registry_entry_in(&owned.root(), &registry_row("rev-clean", &repo, 2));

    let payload = server
        .execute_tool_for(
            "worker",
            json!({ "action": "review", "worker_id": "rev-clean" }),
            &owner_context(),
        )
        .await
        .expect("the owner may review its own worker");

    assert_eq!(payload["task"], "Fix the parser");
    assert_eq!(payload["revision"], 2);
    assert_eq!(payload["branch"], "worker-rev-clean");
    assert_eq!(payload["merge"]["base_branch"], "master");
    assert_eq!(payload["merge"]["clean"], true);
    assert_eq!(payload["merge"]["conflicts"], json!([]));
    assert_eq!(payload["next_command"], "git merge worker-rev-clean");
    // The stat is measured from the branch the orchestrator would merge.
    let per_file = payload["diff_stat"]["per_file"]
        .as_array()
        .expect("per-file");
    assert_eq!(per_file.len(), 1);
    assert_eq!(per_file[0]["path"], "added.txt");
    assert_eq!(per_file[0]["insertions"], 1);
}

/// A `review` of a branch that no longer merges lists the conflicting files and
/// sends them back to the worker that owns them.
#[tokio::test]
async fn review_reports_a_conflicting_merge() {
    let scratch = TempDir::new_in_tmp("review-conflict");
    let repo = branch_repo(&scratch);
    let owned = IsolatedPool::new(4, "review-conflict-pool");
    let server = McpServer::new(owned.pool.clone(), "ninja".to_string());
    save_registry_entry_in(&owned.root(), &registry_row("rev-conflict", &repo, 0));

    let payload = server
        .execute_tool_for(
            "worker",
            json!({ "action": "review", "worker_id": "rev-conflict" }),
            &owner_context(),
        )
        .await
        .expect("the owner may review its own worker");

    assert_eq!(payload["merge"]["clean"], false);
    let conflicts = payload["merge"]["conflicts"].as_array().expect("conflicts");
    assert!(
        conflicts.iter().any(|file| file == "shared.txt"),
        "the conflicting file must be named: {payload}"
    );
    assert_eq!(
        payload["next_command"],
        "mini-swe-mcp steer rev-conflict \"resolve the merge conflicts with master: shared.txt\""
    );
}

/// The merge probe is read-only: it answers without touching the worktree, and
/// the worker it reviewed is still there afterwards.
#[tokio::test]
async fn review_never_evicts_the_worker_it_reviewed() {
    let scratch = TempDir::new_in_tmp("review-live");
    let repo = branch_repo(&scratch);
    let owned = IsolatedPool::new(4, "review-live-pool");
    let server = McpServer::new(owned.pool.clone(), "ninja".to_string());
    let mut record = completed_worker("rev-live", &two_file_diff());
    record.state = WorkerState::Completed {
        turns: 3,
        diff: two_file_diff(),
        summary: "parser now handles empty input".to_string(),
        completed_at: 0,
        artifacts: Vec::new(),
        branch: Some("worker-rev-clean".to_string()),
        verified: Some(false),
        metrics: WorkerMetrics::default(),
        revision: 1,
        report: None,
    };
    record.logs.push(AgentStepLog {
        step: 3,
        command: "[verify] cargo test".to_string(),
        output: "test parser::empty ... FAILED\nassertion failed: parse(\"\")".to_string(),
        exit_code: Some(101),
    });
    owned.pool.__test_insert_worker(record).await;
    save_registry_entry_in(&owned.root(), &registry_row("rev-live", &repo, 1));

    let payload = server
        .execute_tool_for(
            "worker",
            json!({ "action": "review", "worker_id": "rev-live" }),
            &owner_context(),
        )
        .await
        .expect("the owner may review its own worker");

    // A live worker's own diff is the stat, and its failed verify is the tail.
    assert_eq!(payload["diff_stat"]["files"], 2);
    assert_eq!(payload["verified"], false);
    assert!(
        payload["verify_tail"]
            .as_str()
            .expect("a failed verify must be shown")
            .contains("assertion failed"),
        "{payload}"
    );
    assert_eq!(payload["revision"], 1);
    assert_eq!(payload["merge"]["clean"], true);
    assert!(
        owned.pool.worker_progress("rev-live").await.is_some(),
        "review is a read: it must not collect the worker"
    );
}

/// `review` is owner-only, like `collect`.
#[tokio::test]
async fn review_refuses_a_stranger() {
    let scratch = TempDir::new_in_tmp("review-owner");
    let repo = branch_repo(&scratch);
    let owned = IsolatedPool::new(4, "review-owner-pool");
    let server = McpServer::new(owned.pool.clone(), "ninja".to_string());
    save_registry_entry_in(&owned.root(), &registry_row("rev-owner", &repo, 0));

    let error = server
        .execute_tool_for(
            "worker",
            json!({ "action": "review", "worker_id": "rev-owner" }),
            &stranger_context(),
        )
        .await
        .expect_err("another agent's worker must be refused");
    assert_eq!(
        error.to_string(),
        "worker rev-owner belongs to agent review-agent"
    );
}

/// An unknown worker is a clear error, never an empty view.
#[tokio::test]
async fn review_of_an_unknown_worker_is_a_clear_error() {
    let owned = IsolatedPool::new(4, "review-unknown");
    let server = McpServer::new(owned.pool.clone(), "ninja".to_string());

    let error = server
        .execute_tool_for(
            "worker",
            json!({ "action": "review", "worker_id": "rev-nobody" }),
            &owner_context(),
        )
        .await
        .expect_err("an unknown worker must be refused");
    assert_eq!(error.to_string(), "Worker not found: rev-nobody");
}
