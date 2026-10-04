//! Batch `dispatch`: several tasks in one `tools/call` or one CLI file.
//!
//! A batch must start every entry (a bad one only answers with its own error),
//! share the top-level dispatch values as defaults, and stay hermetic: every
//! worker here builds its worktree in a throwaway repository under a temporary
//! scratch root, never in the crate's own checkout.

use crate::common;
use crate::common::{IsolatedPool, TempDir};
use mini_swe_mcp::mcp::McpServer;
use mini_swe_mcp::pool::remove_registry_entry_in;
use serde_json::{Value, json};
use std::path::{Path, PathBuf};

/// A throwaway git repository a dispatched worker can build its worktree from.
fn scratch_repo(dir: &Path) -> PathBuf {
    let repo = dir.join("repo");
    std::fs::create_dir_all(&repo).expect("create the scratch repository");
    common::git(&repo, &["init", "-b", "master"]);
    common::git(&repo, &["config", "user.name", "mini-swe-test"]);
    common::git(&repo, &["config", "user.email", "test@localhost"]);
    std::fs::write(repo.join("README.md"), "# scratch\n").expect("seed the repository");
    common::git(&repo, &["add", "README.md"]);
    common::git(&repo, &["commit", "-m", "baseline"]);
    repo
}

/// Run one batch dispatch through the in-process tool path.
async fn dispatch_batch(server: &McpServer, tasks: Value, shared: Value) -> Value {
    let mut args = shared;
    args["action"] = json!("dispatch");
    args["tasks"] = tasks;
    server
        .execute_tool("worker", args)
        .await
        .expect("a batch dispatch must answer")
}

/// The worker ids of the entries that started, in order.
fn worker_ids(payload: &Value) -> Vec<String> {
    payload["workers"]
        .as_array()
        .expect("a batch result carries a workers array")
        .iter()
        .filter_map(|entry| entry["worker_id"].as_str().map(str::to_string))
        .collect()
}

/// Drop the records and worktrees a test's batch created.
async fn reap(owned: &IsolatedPool, ids: &[String]) {
    for wid in ids {
        owned.pool.kill(wid).await;
        remove_registry_entry_in(&owned.root(), wid);
    }
}

/// Three tasks in one call start three workers, all reported in the reply.
#[tokio::test]
async fn three_tasks_in_one_call_give_three_workers() {
    let dir = TempDir::new_in_tmp("batch-mcp-three");
    let repo = scratch_repo(dir.path());
    let owned = IsolatedPool::new(8, "batch-mcp-three-pool");
    let server = McpServer::new(owned.pool.clone(), "ninja".to_string());

    let repo_path = repo.to_string_lossy().into_owned();
    let tasks = json!([
        { "task": "one", "repo_path": repo_path },
        { "task": "two", "repo_path": repo_path },
        { "task": "three", "repo_path": repo_path },
    ]);
    let payload = dispatch_batch(&server, tasks, json!({})).await;

    assert_eq!(payload["dispatched"], json!(3), "{payload}");
    assert_eq!(payload["failed"], json!(0), "{payload}");
    let ids = worker_ids(&payload);
    assert_eq!(ids.len(), 3, "{payload}");
    let mut unique = ids.clone();
    unique.sort();
    unique.dedup();
    assert_eq!(
        unique.len(),
        3,
        "each entry needs its own worker: {payload}"
    );

    reap(&owned, &ids).await;
}

/// One bad entry reports its own error without holding back the others.
#[tokio::test]
async fn a_bad_entry_reports_its_own_error_while_the_others_start() {
    let dir = TempDir::new_in_tmp("batch-mcp-bad");
    let repo = scratch_repo(dir.path());
    let owned = IsolatedPool::new(8, "batch-mcp-bad-pool");
    let server = McpServer::new(owned.pool.clone(), "ninja".to_string());

    let repo_path = repo.to_string_lossy().into_owned();
    let tasks = json!([
        { "task": "before", "repo_path": repo_path },
        { "repo_path": repo_path },
        { "task": "after", "repo_path": repo_path },
    ]);
    let payload = dispatch_batch(&server, tasks, json!({})).await;

    assert_eq!(payload["dispatched"], json!(2), "{payload}");
    assert_eq!(payload["failed"], json!(1), "{payload}");
    let workers = payload["workers"].as_array().expect("workers array");
    assert!(
        workers[0]["worker_id"]
            .as_str()
            .is_some_and(|s| !s.is_empty()),
        "{payload}"
    );
    assert_eq!(workers[1]["index"], json!(1), "{payload}");
    assert!(workers[1].get("worker_id").is_none(), "{payload}");
    assert!(
        workers[1]["error"]
            .as_str()
            .is_some_and(|error| error.contains("task")),
        "the bad entry must name its own problem: {payload}"
    );
    assert!(
        workers[2]["worker_id"]
            .as_str()
            .is_some_and(|s| !s.is_empty()),
        "{payload}"
    );

    reap(&owned, &worker_ids(&payload)).await;
}

/// Shared top-level dispatch values are the defaults every entry inherits;
/// an entry's own key wins.
#[tokio::test]
async fn shared_top_level_values_apply_as_defaults() {
    let dir = TempDir::new_in_tmp("batch-mcp-defaults");
    let repo = scratch_repo(dir.path());
    let owned = IsolatedPool::new(8, "batch-mcp-defaults-pool");
    let server = McpServer::new(owned.pool.clone(), "ninja".to_string());

    let repo_path = repo.to_string_lossy().into_owned();
    let tasks = json!([
        { "task": "inherits" },
        { "task": "overrides", "network": "allow" },
    ]);
    let shared = json!({ "repo_path": repo_path, "network": "offline" });
    let payload = dispatch_batch(&server, tasks, shared).await;

    assert_eq!(payload["dispatched"], json!(2), "{payload}");
    assert_eq!(payload["failed"], json!(0), "{payload}");
    let workers = payload["workers"].as_array().expect("workers array");
    assert_eq!(workers[0]["network"], json!("offline"), "{payload}");
    assert_eq!(workers[1]["network"], json!("allow"), "{payload}");

    reap(&owned, &worker_ids(&payload)).await;
}
/// The batch answer names the round it dispatched into when every entry agrees
/// on one group, because that is what a `--quiet` caller needs to wait with
/// `watch --group <g> --all`; a batch split across two groups names none, so no
/// group is silently left unwatched.
#[tokio::test]
async fn a_batch_answer_names_its_round_only_when_the_entries_agree() {
    let dir = TempDir::new_in_tmp("batch-mcp-round");
    let repo = scratch_repo(dir.path());
    let owned = IsolatedPool::new(8, "batch-mcp-round-pool");
    let server = McpServer::new(owned.pool.clone(), "ninja".to_string());

    let repo_path = repo.to_string_lossy().into_owned();
    let one_round = dispatch_batch(
        &server,
        json!([{ "task": "a" }, { "task": "b" }]),
        json!({ "repo_path": repo_path.clone(), "group": "round-1" }),
    )
    .await;
    assert_eq!(one_round["group"], json!("round-1"), "{one_round}");
    reap(&owned, &worker_ids(&one_round)).await;

    let two_rounds = dispatch_batch(
        &server,
        json!([{ "task": "c" }, { "task": "d", "group": "round-2" }]),
        json!({ "repo_path": repo_path, "group": "round-1" }),
    )
    .await;
    assert!(
        two_rounds.get("group").is_none(),
        "two groups have no single round to wait on: {two_rounds}"
    );
    reap(&owned, &worker_ids(&two_rounds)).await;
}

/// A batch that did not ask to consolidate must not say that it did. Automatic
/// consolidation is the hub daemon's to start, so a `consolidate` key on the
/// answer is the only thing the `--quiet` reminder has to go on, and one that
/// rode along on a dispatch which declined it would rewrite the reminder into a
/// round wait no consolidator will ever satisfy. The dispatch that does ask is
/// covered where the hub's auto-consolidation store can be installed without
/// running its scheduler (`a_consolidated_dispatch_answer_carries_the_round_and_its_token`).
#[tokio::test]
async fn a_batch_answer_claims_a_round_only_when_consolidation_was_requested() {
    let dir = TempDir::new_in_tmp("batch-mcp-no-consolidate");
    let repo = scratch_repo(dir.path());
    let owned = IsolatedPool::new(8, "batch-mcp-no-consolidate-pool");
    let server = McpServer::new(owned.pool.clone(), "ninja".to_string());

    let payload = dispatch_batch(
        &server,
        json!([{ "task": "a" }, { "task": "b" }]),
        json!({
            "repo_path": repo.to_string_lossy(),
            "group": "round-48",
            "consolidate": false,
        }),
    )
    .await;
    assert_eq!(payload["dispatched"], json!(2), "{payload}");
    assert_eq!(payload["group"], json!("round-48"), "{payload}");
    assert!(
        payload.get("consolidate").is_none(),
        "a dispatch that declined the round must not answer as one: {payload}"
    );

    reap(&owned, &worker_ids(&payload)).await;
}
