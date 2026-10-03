//! `discard <id>`: dropping a stopped worker on purpose (pool-E32).
//!
//! A failed consolidator whose gate can never run used to have to be removed by
//! hand -- `git branch -D`, the registry row, the history JSONL, the steer
//! mailbox, the steer-source and the `.round-base` file, one command at a time.
//! `discard` is that removal, and it goes through the one shared retirement a
//! merge uses, so a discarded worker leaves as little behind as a merged one.
//!
//! Every test builds its own repository and its own scratch root and passes
//! that root to the code under test, so no test can see or remove another's
//! registry, history or branch.

mod common;

use common::{IsolatedPool, TempDir, git, git_ref_exists};
use mini_swe_mcp::agent::{ChatMessage, Role};
use mini_swe_mcp::mcp::{ConnectionContext, McpServer};
use mini_swe_mcp::pool::{
    RegistryStatus, WorkerHistory, WorkerRecord, WorkerRegistryEntry, WorkerRole, WorkerState,
    append_history_message_in, load_registry_entry_in, save_registry_entry_in,
};
use mini_swe_mcp::worktree::ScratchRoot;
use serde_json::json;
use std::path::{Path, PathBuf};

const OWNER: &str = "discard-agent";

/// A repository on `main` plus an isolated scratch root and pool.
struct Fixture {
    repo: TempDir,
    pool: IsolatedPool,
}

impl Fixture {
    fn new(tag: &str) -> Self {
        let repo = TempDir::new_in_tmp(tag);
        git(repo.path(), &["init", "-b", "main"]);
        git(repo.path(), &["config", "user.name", "discard test"]);
        git(repo.path(), &["config", "user.email", "discard@test"]);
        write(repo.path(), "README.md", "base\n");
        git(repo.path(), &["add", "."]);
        git(repo.path(), &["commit", "-m", "base"]);
        Self {
            repo,
            pool: IsolatedPool::new(4, tag),
        }
    }

    fn root(&self) -> ScratchRoot {
        self.pool.root()
    }

    fn repo(&self) -> &Path {
        self.repo.path()
    }

    /// Commit on `worker-<id>` and leave `main` checked out, the branch shape a
    /// stopped worker leaves behind.
    fn worker_branch(&self, id: &str) {
        let branch = format!("worker-{id}");
        git(self.repo(), &["checkout", "-q", "-b", &branch]);
        write(
            self.repo(),
            &format!("{id}.txt"),
            &format!("work by {id}\n"),
        );
        git(self.repo(), &["add", "."]);
        git(self.repo(), &["commit", "-m", &format!("worker {id}")]);
        git(self.repo(), &["checkout", "-q", "main"]);
    }

    /// Record the failed consolidator `id`: its history, its row and every
    /// scratch companion a live one holds, including the pinned round base.
    fn record_failed_consolidator(&self, id: &str) {
        let history = WorkerHistory {
            task: "consolidate the round".to_string(),
            role: WorkerRole::Consolidate,
            group: Some("round5".to_string()),
            model: "test".to_string(),
            temperature: None,
            repo_path: self.repo().to_string_lossy().into_owned(),
            base_commit: git(self.repo(), &["rev-parse", "HEAD"]).trim().to_string(),
            base_branch: Some("main".to_string()),
            branch: format!("worker-{id}"),
            network_offline: false,
            // A gate that can never run: the whole reason this worker is
            // discarded rather than merged.
            verify: Some("cargo test --a-gate-that-cannot-run".to_string()),
            client_env: Vec::new(),
            max_turns: 10,
            review_after: None,
            revision: 0,
            auto_continues: 0,
            owner: Some(OWNER.to_string()),
            messages: vec![ChatMessage::text(Role::System, "you are a consolidator")],
        };
        append_history_message_in(
            &self.root(),
            id,
            &history,
            &ChatMessage::text(Role::System, "you are a consolidator"),
        )
        .expect("history log must be writable");
        let row = WorkerRegistryEntry {
            task: "consolidate the round".to_string(),
            status: RegistryStatus::Failed,
            step: 3,
            repo_path: Some(self.repo().to_string_lossy().into_owned()),
            base_branch: Some("main".to_string()),
            role: WorkerRole::Consolidate,
            group: Some("round5".to_string()),
            ..WorkerRegistryEntry::test_row(id, OWNER)
        };
        save_registry_entry_in(&self.root(), &row);
        // The leftovers a hand-written `pool.kill` never cleaned: the pinned
        // round base, the steer-source, the mailbox, the worktree and the
        // private target dir beside it.
        write(
            self.pool.scratch.path(),
            &format!("swe-wt-{id}.round-base"),
            "abc123\n",
        );
        write(
            self.pool.scratch.path(),
            &format!("swe-wt-{id}.steer-source"),
            "{\"consolidator\":\"c1\",\"round_base\":\"abc123\"}\n",
        );
        write(
            self.pool.scratch.path(),
            &format!("swe-wt-{id}.steer"),
            "guidance\n",
        );
        let worktree: PathBuf = self.pool.scratch.path().join(format!("swe-wt-{id}"));
        std::fs::create_dir_all(worktree.join("nested")).unwrap();
        write(&worktree, "nested/junk.txt", "junk\n");
        std::fs::create_dir_all(
            self.pool
                .scratch
                .path()
                .join(format!("swe-target-swe-wt-{id}")),
        )
        .unwrap();
    }

    fn scratch_file(&self, name: &str) -> PathBuf {
        self.pool.scratch.path().join(name)
    }

    fn history_exists(&self, id: &str) -> bool {
        self.scratch_file(&format!("swe-wt-{id}.history.jsonl"))
            .exists()
    }

    fn row_exists(&self, id: &str) -> bool {
        load_registry_entry_in(&self.root(), id).is_some()
    }

    /// A server over this fixture's pool, driven by `OWNER`.
    fn server(&self) -> McpServer {
        McpServer::new(self.pool.pool.clone(), "ninja".to_string())
    }
}

fn owner_context() -> ConnectionContext {
    ConnectionContext {
        agent_id: Some(OWNER.to_string()),
        ..ConnectionContext::hub_connection(3)
    }
}

fn write(dir: &Path, name: &str, contents: &str) {
    std::fs::write(dir.join(name), contents).unwrap_or_else(|e| panic!("write {name}: {e}"));
}

/// One `discard` call through the same entry point every other verb uses.
async fn discard(
    server: &McpServer,
    worker_id: &str,
    ctx: &ConnectionContext,
) -> anyhow::Result<serde_json::Value> {
    server
        .execute_tool_for(
            "worker",
            json!({ "action": "discard", "worker_id": worker_id }),
            ctx,
        )
        .await
}

/// `discard <id>` retires everything of a stopped worker in one call: the
/// branch, the registry row, the history, the steer mailbox, the steer-source,
/// the pinned `.round-base` and the worktree leftovers. Nothing is left to
/// remove by hand, and the base branch is untouched -- a discard merges nothing.
#[tokio::test]
async fn discard_removes_every_leftover_of_a_stopped_worker() {
    let f = Fixture::new("discard-leftovers");
    f.worker_branch("cons1");
    f.record_failed_consolidator("cons1");
    let worktree = f.scratch_file("swe-wt-cons1");
    let base_tip = git(f.repo(), &["rev-parse", "main"]).trim().to_string();

    // Proof the fixture is the problem the task describes: every leftover is
    // there before the call.
    assert!(git_ref_exists(f.repo(), "worker-cons1"));
    assert!(f.row_exists("cons1"));
    assert!(f.history_exists("cons1"));
    assert!(worktree.exists());
    for name in [
        "swe-wt-cons1.round-base",
        "swe-wt-cons1.steer-source",
        "swe-wt-cons1.steer",
    ] {
        assert!(f.scratch_file(name).exists(), "{name} must exist first");
    }

    let result = discard(&f.server(), "cons1", &owner_context())
        .await
        .expect("a stopped worker must be discardable");

    assert_eq!(result["worker_id"], "cons1", "{result}");
    assert_eq!(result["discarded"], true, "{result}");
    assert_eq!(
        result["branch_deleted"], true,
        "the unrunnable gate is exactly why the branch must still go: {result}"
    );
    assert_eq!(result["row_removed"], true, "{result}");
    assert_eq!(result["worktree_reclaimed"], true, "{result}");

    assert!(
        !git_ref_exists(f.repo(), "worker-cons1"),
        "the branch must be deleted"
    );
    assert!(!f.row_exists("cons1"), "the registry row must be gone");
    assert!(!f.history_exists("cons1"), "the history file must be gone");
    assert!(!worktree.exists(), "the worktree must be reclaimed");
    for name in [
        "swe-wt-cons1.round-base",
        "swe-wt-cons1.steer-source",
        "swe-wt-cons1.steer",
        "swe-target-swe-wt-cons1",
    ] {
        assert!(
            !f.scratch_file(name).exists(),
            "{name} must be removed, not left by hand"
        );
    }
    assert_eq!(
        git(f.repo(), &["rev-parse", "main"]).trim(),
        base_tip,
        "a discard merges nothing: the base branch must not move"
    );
    // The live record goes too, so `list` stops showing a worker nothing is
    // left of.
    assert!(
        f.pool.pool.get_worker_state("cons1").await.is_none(),
        "the live record must go with the files"
    );
}

/// The round base is removed from *every* directory the scratch root sweeps,
/// not just the one the consolidator happened to be dispatched under: that is
/// what makes the removal complete wherever the file landed.
#[tokio::test]
async fn discard_removes_the_round_base_from_every_swept_directory() {
    let f = Fixture::new("discard-round-base");
    f.worker_branch("cons2");
    f.record_failed_consolidator("cons2");
    // A second base directory, the one `ScratchRoot::from_env` sweeps
    // alongside its own: a `.round-base` written there must still go.
    let extra = TempDir::new_in_tmp("discard-round-base-extra");
    let root = ScratchRoot::from_dirs(&[
        f.pool.scratch.path().to_path_buf(),
        extra.path().to_path_buf(),
    ]);
    write(extra.path(), "swe-wt-cons2.round-base", "def456\n");
    assert!(extra.path().join("swe-wt-cons2.round-base").exists());

    // The shared retirement, with the same context `discard` builds.
    mini_swe_mcp::pool::retire_worker_reporting(
        &root,
        "cons2",
        &mini_swe_mcp::pool::RetireContext {
            repo: Some(f.repo()),
            ack_dir: None,
            reason: None,
            merge_commit: None,
            keep_branch: false,
        },
    );

    assert!(
        !extra.path().join("swe-wt-cons2.round-base").exists(),
        "a .round-base in a swept companion directory must be removed too"
    );
    assert!(
        !f.scratch_file("swe-wt-cons2.round-base").exists(),
        "and the one in the root directory"
    );
}

/// A live worker is never discarded behind the caller's back: the refusal
/// names the `kill` that must come first, and nothing is removed.
#[tokio::test]
async fn discard_refuses_a_running_worker_and_names_kill() {
    let f = Fixture::new("discard-running");
    f.worker_branch("busy1");
    f.record_failed_consolidator("busy1");
    f.pool
        .pool
        .__test_insert_worker(WorkerRecord {
            id: "busy1".to_string(),
            task: "still working".to_string(),
            model: "test".to_string(),
            owner: OWNER.to_string(),
            state: WorkerState::Running {
                step: 2,
                last_command: "cargo test".to_string(),
                started_at: 0,
            },
            ..synthetic_record("busy1")
        })
        .await;

    let error = discard(&f.server(), "busy1", &owner_context())
        .await
        .expect_err("a running worker must be refused");
    let message = error.to_string();
    assert!(
        message.contains("still running") && message.contains("kill busy1"),
        "the refusal must name the state and the kill to run first: {message}"
    );

    // The refusal changed nothing.
    assert!(git_ref_exists(f.repo(), "worker-busy1"));
    assert!(f.row_exists("busy1"));
    assert!(f.history_exists("busy1"));
    assert!(f.scratch_file("swe-wt-busy1.round-base").exists());
    assert!(f.scratch_file("swe-wt-busy1").exists());
}

/// A paused worker is stopped but not finished, so it is refused for the same
/// reason and with the same hint: a discard is for a worker that will never
/// come back on its own.
#[tokio::test]
async fn discard_refuses_a_paused_worker_too() {
    let f = Fixture::new("discard-paused");
    f.worker_branch("ask1");
    f.record_failed_consolidator("ask1");
    f.pool
        .pool
        .__test_insert_worker(WorkerRecord {
            id: "ask1".to_string(),
            task: "waiting for an answer".to_string(),
            model: "test".to_string(),
            owner: OWNER.to_string(),
            state: WorkerState::Paused {
                question: "which database?".to_string(),
                step: 1,
                paused_at: 0,
            },
            ..synthetic_record("ask1")
        })
        .await;

    let error = discard(&f.server(), "ask1", &owner_context())
        .await
        .expect_err("a paused worker must be refused");
    let message = error.to_string();
    assert!(
        message.contains("paused") && message.contains("kill ask1"),
        "the refusal must name the state and the kill to run first: {message}"
    );
    assert!(f.row_exists("ask1"), "a refused discard removes nothing");
}

/// Ownership is the boundary, exactly as for `kill`: another agent cannot
/// delete a worker's branch, and the refusal changes nothing.
#[tokio::test]
async fn discard_refuses_another_owners_worker() {
    let f = Fixture::new("discard-foreign");
    f.worker_branch("theirs1");
    f.record_failed_consolidator("theirs1");
    let foreign = ConnectionContext {
        agent_id: Some("somebody-else".to_string()),
        ..ConnectionContext::hub_connection(3)
    };

    let error = discard(&f.server(), "theirs1", &foreign)
        .await
        .expect_err("another agent's worker must be refused");
    assert_eq!(
        error.to_string(),
        format!("worker theirs1 belongs to agent {OWNER}"),
        "the refusal must name the owning agent: {error}"
    );
    assert!(git_ref_exists(f.repo(), "worker-theirs1"));
    assert!(f.row_exists("theirs1"));
    assert!(f.history_exists("theirs1"));
    assert!(f.scratch_file("swe-wt-theirs1.round-base").exists());
}

/// An id nothing knows is refused rather than spliced into a scratch path or a
/// git argument: a discard deletes, so it must only ever act on a worker this
/// pool or its registry actually knows -- a traversal payload included.
#[tokio::test]
async fn discard_refuses_an_id_nothing_knows() {
    let f = Fixture::new("discard-unknown");
    let victim = f.scratch_file("swe-wt-../../escaped");
    // A payload that would escape the scratch base if it were ever joined.
    let hostile = "../../escaped";

    let error = discard(&f.server(), hostile, &owner_context())
        .await
        .expect_err("an id nothing knows must be refused");
    assert_eq!(
        error.to_string(),
        format!("Worker not found: {hostile}"),
        "the refusal must be the plain not-found answer: {error}"
    );
    assert!(
        !victim.exists() && !victim.with_extension("").exists(),
        "an unknown id must never reach a path: {}",
        victim.display()
    );
}

/// The renderer the CLI prints, so `discard` has a human-facing answer instead
/// of raw JSON.
#[test]
fn the_cli_renders_a_discard_as_one_line() {
    let out = mini_swe_mcp::cli::format::format_output(
        "discard",
        &json!({
            "worker_id": "cons1",
            "discarded": true,
            "branch_deleted": true,
            "worktree_reclaimed": true,
            "row_removed": true,
        }),
    );
    assert_eq!(
        out,
        "✓ Worker cons1 discarded. Its branch and scratch state are gone; \
         its worktree was reclaimed; nothing of it is left to merge, steer or watch."
    );
}

/// A record with every field the struct requires, defaulted around the two the
/// test sets, so the ownership and state under test are the only ones named.
fn synthetic_record(id: &str) -> WorkerRecord {
    WorkerRecord {
        id: id.to_string(),
        task: "task".to_string(),
        model: "test".to_string(),
        owner: OWNER.to_string(),
        state: WorkerState::Failed {
            error: "gate cannot run".to_string(),
            step: 3,
            failed_at: 0,
            metrics: Default::default(),
            revision: 0,
        },
        metrics: Default::default(),
        logs: Default::default(),
        pending_steer: Vec::new(),
        resume_tx: None,
        handle: None,
        revision: 0,
    }
}
