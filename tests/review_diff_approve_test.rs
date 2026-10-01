//! `review`'s diff scope and `approve`/`unapprove`: what a review shows and
//! what the orchestrator's verdict records.
//!
//! The review default is the code diff, because test and doc churn is what
//! makes a full diff expensive to read; a completed worker's approval is
//! written to its registry row so it outlives the in-memory record `collect`
//! evicts, and a new revision clears it.
//!
//! Every repository and registry row here lives under a per-test scratch root,
//! so the suite never touches the crate's own git state.

mod common;

use common::{IsolatedPool, TempDir};

use mini_swe_mcp::agent::{ChatMessage, Role};
use mini_swe_mcp::mcp::{ConnectionContext, McpServer};
use mini_swe_mcp::pool::{
    LogBuffer, RegistryStatus, WorkerApproval, WorkerHistory, WorkerMetrics, WorkerPool,
    WorkerRecord, WorkerRole, WorkerState, append_history_message_in, load_registry_entry_in,
    save_registry_entry_in,
};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::process::Command;

/// The agent that owns every worker these tests dispatch.
const OWNER: &str = "review-diff-agent";

/// A connection for `OWNER`, so the ownership check (H-3) passes.
fn owner_context() -> ConnectionContext {
    ConnectionContext {
        agent_id: Some(OWNER.to_string()),
        ..ConnectionContext::hub_connection(11)
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

/// A diff over four files: Rust code, a Rust test file, a Python test file and
/// a doc. The Rust test file adds three cases and removes two; the Python one
/// adds two and removes one.
fn mixed_diff() -> String {
    [
        "diff --git a/src/parser.rs b/src/parser.rs",
        "--- a/src/parser.rs",
        "+++ b/src/parser.rs",
        "@@ -1,2 +1,3 @@",
        " fn parse() {",
        "-    old();",
        "+    new();",
        "+    extra();",
        "diff --git a/tests/parser_test.rs b/tests/parser_test.rs",
        "--- a/tests/parser_test.rs",
        "+++ b/tests/parser_test.rs",
        "@@ -1,2 +1,3 @@",
        "-#[test]",
        "-fn test_old() {}",
        "+#[test]",
        "+fn test_new() {}",
        "+#[test]",
        "diff --git a/test_parser.py b/test_parser.py",
        "--- a/test_parser.py",
        "+++ b/test_parser.py",
        "@@ -1,2 +1,4 @@",
        "-def test_old():",
        "-    pass",
        "+def test_new():",
        "+    pass",
        "+def test_extra():",
        "+    pass",
        "diff --git a/README.md b/README.md",
        "--- a/README.md",
        "+++ b/README.md",
        "@@ -1 +1,2 @@",
        "-old docs",
        "+new docs",
        "+more docs",
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

/// A registry row for `id` in `status` and `owner`, under `repo`.
fn registry_row(
    id: &str,
    status: RegistryStatus,
    owner: &str,
    repo: Option<&Path>,
) -> mini_swe_mcp::pool::WorkerRegistryEntry {
    mini_swe_mcp::pool::WorkerRegistryEntry {
        task: "Fix the parser\nand its docs".to_string(),
        model: "test-model".to_string(),
        status,
        step: 3,
        max_turns: 60,
        last_command: "cargo test".to_string(),
        // Fresh, so the terminal-TTL filter in `list` still shows the row.
        updated_at: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or_default(),
        repo_path: repo.map(|repo| repo.to_string_lossy().into_owned()),
        base_branch: Some("master".to_string()),
        ..mini_swe_mcp::pool::WorkerRegistryEntry::test_row(id.to_string(), owner.to_string())
    }
}

/// The `review` payload for `id`, optionally asking for one `diff` scope.
async fn review_with(server: &McpServer, id: &str, scope: Option<&str>) -> Value {
    let mut args = json!({ "action": "review", "worker_id": id });
    if let Some(scope) = scope {
        args["diff"] = Value::String(scope.to_string());
    }
    server
        .execute_tool_for("worker", args, &owner_context())
        .await
        .unwrap_or_else(|e| panic!("review must answer: {e}"))
}

/// The `(added, removed)` test-case counts the payload reports for `path`.
fn test_case_counts(payload: &Value, path: &str) -> (u64, u64) {
    let entry = payload["test_files"]
        .as_array()
        .expect("test_files is an array")
        .iter()
        .find(|entry| entry["path"] == path)
        .unwrap_or_else(|| panic!("no test file summary for {path}: {payload}"));
    (
        entry["added_cases"].as_u64().expect("added_cases"),
        entry["removed_cases"].as_u64().expect("removed_cases"),
    )
}

/// The default review shows the code diff and summarises the test files, with
/// the right case counts for a Rust and a Python test.
#[tokio::test]
async fn review_defaults_to_the_code_diff_and_summarises_tests() {
    let owned = IsolatedPool::new(4, "review-diff");
    let server = McpServer::new(owned.pool.clone(), "ninja".to_string());
    owned
        .pool
        .__test_insert_worker(completed_worker("diff-default", &mixed_diff()))
        .await;

    let payload = review_with(&server, "diff-default", None).await;
    assert_eq!(payload["diff_scope"], "code");

    let shown = payload["diff"].as_str().expect("the code diff is shown");
    assert!(shown.contains("src/parser.rs"), "{shown}");
    assert!(!shown.contains("tests/parser_test.rs"), "{shown}");
    assert!(!shown.contains("test_parser.py"), "{shown}");
    assert!(!shown.contains("README.md"), "{shown}");

    assert_eq!(test_case_counts(&payload, "tests/parser_test.rs"), (3, 2));
    assert_eq!(test_case_counts(&payload, "test_parser.py"), (2, 1));

    let docs = payload["docs"].as_array().expect("docs is an array");
    assert_eq!(docs.len(), 1, "{payload}");
    assert_eq!(docs[0]["path"], "README.md");
    assert_eq!(docs[0]["insertions"], 2);
    assert_eq!(docs[0]["deletions"], 1);

    // The stat still covers every file, code or not.
    assert_eq!(payload["diff_stat"]["files"], 4);
}

/// `diff: "all"` shows the whole diff; `diff: "none"` withholds it. The test
/// summary survives both.
#[tokio::test]
async fn review_diff_all_shows_everything_and_none_withholds_it() {
    let owned = IsolatedPool::new(4, "review-scope");
    let server = McpServer::new(owned.pool.clone(), "ninja".to_string());
    owned
        .pool
        .__test_insert_worker(completed_worker("diff-scope", &mixed_diff()))
        .await;

    let all = review_with(&server, "diff-scope", Some("all")).await;
    assert_eq!(all["diff_scope"], "all");
    let shown = all["diff"].as_str().expect("--diff all shows the diff");
    for needle in [
        "src/parser.rs",
        "tests/parser_test.rs",
        "test_parser.py",
        "README.md",
    ] {
        assert!(shown.contains(needle), "missing {needle}: {shown}");
    }
    assert_eq!(test_case_counts(&all, "tests/parser_test.rs"), (3, 2));

    let none = review_with(&server, "diff-scope", Some("none")).await;
    assert_eq!(none["diff_scope"], "none");
    assert!(none["diff"].is_null(), "diff none must withhold it: {none}");
    assert_eq!(test_case_counts(&none, "test_parser.py"), (2, 1));

    let typo = server
        .execute_tool_for(
            "worker",
            json!({ "action": "review", "worker_id": "diff-scope", "diff": "everything" }),
            &owner_context(),
        )
        .await;
    assert!(typo.is_err(), "an unknown scope must be refused");
}

/// `approve` records the verdict on the completed worker's registry row, so it
/// is still there when the in-memory record is gone, and `review` reports it.
#[tokio::test]
async fn approve_records_the_verdict_on_a_completed_worker() {
    let owned = IsolatedPool::new(4, "approve-ok");
    let server = McpServer::new(owned.pool.clone(), "ninja".to_string());
    // A branch, so `list`'s terminal-row pruning keeps the row: the worker was
    // never in this process's memory, which is what a collected worker looks
    // like.
    let repo = repo_with_branch("ok", "approve-ok");
    save_registry_entry_in(
        &owned.root(),
        &registry_row("approve-ok", RegistryStatus::Completed, OWNER, Some(&repo)),
    );

    let payload = server
        .execute_tool_for(
            "worker",
            json!({ "action": "approve", "worker_id": "approve-ok", "message": "looks right" }),
            &owner_context(),
        )
        .await
        .expect("the owner may approve its completed worker");
    assert_eq!(payload["status"], "approved");
    assert_eq!(payload["approved"]["note"], "looks right");

    let entry = load_registry_entry_in(&owned.root(), "approve-ok").expect("row survives");
    let approved = entry.approved.expect("the row records the approval");
    assert_eq!(approved.note.as_deref(), Some("looks right"));

    // The verdict survives the in-memory eviction: review reads the row.
    let reviewed = review_with(&server, "approve-ok", Some("none")).await;
    assert_eq!(reviewed["approved"]["note"], "looks right");

    // `status` and `list` report the same verdict.
    let status = server
        .execute_tool_for(
            "worker",
            json!({ "action": "status", "worker_id": "approve-ok" }),
            &owner_context(),
        )
        .await
        .expect("status answers for the owner");
    assert_eq!(status["approved"]["note"], "looks right");
    let listed = server
        .execute_tool_for("worker", json!({ "action": "list" }), &owner_context())
        .await
        .expect("list answers for the owner");
    let row = listed["workers"]
        .as_array()
        .expect("workers is an array")
        .iter()
        .find(|row| row["id"] == "approve-ok")
        .expect("the approved worker is listed");
    assert_eq!(row["approved"]["note"], "looks right");

    // `unapprove` removes it again.
    server
        .execute_tool_for(
            "worker",
            json!({ "action": "unapprove", "worker_id": "approve-ok" }),
            &owner_context(),
        )
        .await
        .expect("the owner may withdraw its approval");
    assert!(
        load_registry_entry_in(&owned.root(), "approve-ok")
            .expect("row survives")
            .approved
            .is_none(),
        "unapprove must clear the verdict"
    );
    let _ = std::fs::remove_dir_all(&repo);
}

/// `approve` refuses a worker that is still running.
#[tokio::test]
async fn approve_refuses_a_running_worker() {
    let owned = IsolatedPool::new(4, "approve-running");
    let server = McpServer::new(owned.pool.clone(), "ninja".to_string());
    save_registry_entry_in(
        &owned.root(),
        &registry_row("approve-running", RegistryStatus::Running, OWNER, None),
    );

    let refusal = server
        .execute_tool_for(
            "worker",
            json!({ "action": "approve", "worker_id": "approve-running" }),
            &owner_context(),
        )
        .await
        .expect_err("a running worker cannot be approved");
    assert!(
        refusal.to_string().contains("not completed"),
        "the refusal must name the reason: {refusal}"
    );
}

/// `approve` refuses another owner's completed worker, like `collect` (H-3).
#[tokio::test]
async fn approve_refuses_another_owners_worker() {
    let owned = IsolatedPool::new(4, "approve-foreign");
    let server = McpServer::new(owned.pool.clone(), "ninja".to_string());
    save_registry_entry_in(
        &owned.root(),
        &registry_row(
            "approve-foreign",
            RegistryStatus::Completed,
            "someone-else",
            None,
        ),
    );

    let refusal = server
        .execute_tool_for(
            "worker",
            json!({ "action": "approve", "worker_id": "approve-foreign" }),
            &owner_context(),
        )
        .await
        .expect_err("only the owner may approve");
    assert!(
        refusal
            .to_string()
            .contains("belongs to agent someone-else"),
        "the refusal must name the owner: {refusal}"
    );

    assert!(
        load_registry_entry_in(&owned.root(), "approve-foreign")
            .expect("row survives")
            .approved
            .is_none(),
        "a refused approve must not write anything"
    );
}

/// A git repository with a `master` branch and `worker-<id>`, as dispatch
/// leaves behind.
fn repo_with_branch(tag: &str, id: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "swe-review-approve-repo-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    git(&dir, &["init", "--initial-branch=master"]);
    git(&dir, &["config", "user.email", "t@t"]);
    git(&dir, &["config", "user.name", "t"]);
    std::fs::write(dir.join("a.txt"), "base\n").unwrap();
    git(&dir, &["add", "."]);
    git(&dir, &["commit", "-m", "base"]);
    git(&dir, &["checkout", "-b", &format!("worker-{id}")]);
    std::fs::write(dir.join("a.txt"), "work\n").unwrap();
    git(&dir, &["add", "."]);
    git(&dir, &["commit", "-m", "work"]);
    git(&dir, &["checkout", "master"]);
    dir
}

/// A replayable conversation for `id`.
fn history(id: &str, repo: &Path) -> WorkerHistory {
    WorkerHistory {
        task: "fix the parser".to_string(),
        group: None,
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

/// Steering a completed worker into a new revision clears the approval: the
/// branch changed, so the old review no longer applies.
#[tokio::test]
async fn a_new_revision_clears_the_approval() {
    let scratch = TempDir::new_in_tmp("approve-rev");
    let root = mini_swe_mcp::worktree::ScratchRoot::new(scratch.path());
    let repo = repo_with_branch("rev", "rev1");
    let meta = history("rev1", &repo);
    for msg in &meta.messages {
        append_history_message_in(&root, "rev1", &meta, msg).expect("append");
    }
    let mut row = registry_row("rev1", RegistryStatus::Completed, OWNER, Some(&repo));
    row.approved = Some(WorkerApproval {
        at: 1_700_000_000,
        note: Some("approved once".to_string()),
    });
    save_registry_entry_in(&root, &row);

    let pool = WorkerPool::with_scratch(1, "http://x".into(), "k".into(), root.clone());
    pool.steer("rev1", "one more pass".into())
        .await
        .expect("a completed worker with a surviving conversation is revised");

    let cleared = load_registry_entry_in(&root, "rev1").expect("row survives");
    assert!(
        cleared.approved.is_none(),
        "a new revision needs a new review: {:?}",
        cleared.approved
    );
    let _ = std::fs::remove_dir_all(&repo);
}
