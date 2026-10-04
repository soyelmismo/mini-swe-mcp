//! Hermetic tests for the `watch_command` a dispatch or steer answer carries.
//!
//! Each test builds a private scratch root: the steer path writes worker
//! mailboxes and history under the pool's scratch, so an ambient root would let
//! a parallel test (or the real hub) collide with it.

use super::*;
use crate::hub::WatchTokens;
use crate::mcp::server::ConnectionContext;
use crate::pool::{LogBuffer, WorkerMetrics, WorkerPool, WorkerRecord, WorkerState};
use crate::worktree::ScratchRoot;
use std::sync::Arc;

/// A private scratch directory, unique per test run, so parallel tests never
/// share registry rows, mailboxes or token files.
fn scratch(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "mcp-watch-line-{tag}-{}-{}",
        std::process::id(),
        uuid::Uuid::new_v4().simple()
    ));
    std::fs::create_dir_all(&dir).expect("create the scratch root");
    dir
}

/// The `watch_command` is only useful to a caller with no watch running: once
/// this identity holds the hub's one watch slot the field is dropped, while a
/// watch held by another identity does not suppress it.
#[tokio::test]
async fn watch_command_is_omitted_only_while_the_callers_own_watch_runs() {
    let base = scratch("own-watch");
    let server = McpServer::new(
        WorkerPool::with_scratch(
            1,
            "http://localhost:1".to_string(),
            "test-key".to_string(),
            ScratchRoot::new(&base),
        ),
        "ninja".to_string(),
    );
    server
        .pool
        .__test_insert_worker(WorkerRecord {
            id: "watch-line-1".to_string(),
            task: "probe".to_string(),
            model: "test".to_string(),
            owner: "orchestrator".to_string(),
            state: WorkerState::Running {
                step: 1,
                last_command: "probe".to_string(),
                started_at: 0,
            },
            metrics: WorkerMetrics::default(),
            logs: LogBuffer::new(),
            pending_steer: Vec::new(),
            resume_tx: None,
            handle: None,
            revision: 0,
        })
        .await;

    // The token store lives inside the private scratch root too, so nothing is
    // written to the real hub directory.
    let tokens_dir = base.join("watch-tokens");
    std::fs::create_dir_all(&tokens_dir).expect("create the token directory");
    let mut ctx = ConnectionContext::hub_connection(1)
        .with_watch_tokens(Arc::new(WatchTokens::new(tokens_dir)));
    ctx.agent_id = Some("orchestrator".to_string());

    async fn steer(server: &McpServer, ctx: &ConnectionContext) -> Value {
        server
            .execute_tool_for(
                "worker",
                json!({"action": "steer", "worker_id": "watch-line-1", "message": "go"}),
                ctx,
            )
            .await
            .expect("steer a seeded worker")
    }

    // No watch running: the answer hands the caller the command.
    let no_watch = steer(&server, &ctx).await;
    assert!(
        no_watch["watch_command"].is_string(),
        "a caller with no watch must be told how to start one: {no_watch}"
    );

    // Another identity's watch is not this caller's: the command stays.
    let other = server
        .hub_events
        .lock()
        .await
        .begin_watch(
            "someone-else",
            2,
            None,
            &crate::mcp::events::WatchSelection::default(),
        )
        .started()
        .expect("another identity claims its own slot");
    let foreign = steer(&server, &ctx).await;
    assert!(
        foreign["watch_command"].is_string(),
        "another identity's watch must not silence this caller: {foreign}"
    );
    drop(other);

    // This caller's own watch: the hub enforces one watch per session and the
    // running one delivers the next event, so the command is redundant.
    let own = server
        .hub_events
        .lock()
        .await
        .begin_watch(
            "orchestrator",
            1,
            None,
            &crate::mcp::events::WatchSelection::default(),
        )
        .started()
        .expect("this identity claims the slot");
    let watching = steer(&server, &ctx).await;
    assert!(
        watching.get("watch_command").is_none(),
        "a running watch makes the command redundant: {watching}"
    );
    drop(own);

    let _ = std::fs::remove_dir_all(&base);
}

/// A second `watch` call of the same session never competes for the running
/// one's events: an identical request is covered, a broader one widens the
/// stored selection to the union, and the running watch keeps its slot.
#[tokio::test]
async fn a_second_watch_call_is_covered_or_widens_the_running_one() {
    use crate::mcp::events::WatchSelection;
    let base = scratch("second-watch");
    let server = McpServer::new(
        WorkerPool::with_scratch(
            1,
            "http://localhost:1".to_string(),
            "test-key".to_string(),
            ScratchRoot::new(&base),
        ),
        "ninja".to_string(),
    );
    let tokens_dir = base.join("watch-tokens");
    std::fs::create_dir_all(&tokens_dir).expect("create the token directory");
    let mut ctx = ConnectionContext::hub_connection(1)
        .with_watch_tokens(Arc::new(WatchTokens::new(tokens_dir)));
    ctx.agent_id = Some("orchestrator".to_string());

    let narrow = WatchSelection::new(Vec::<String>::new(), ["round-a".to_string()], true);
    let _held = server
        .hub_events
        .lock()
        .await
        .begin_watch("orchestrator", 1, None, &narrow)
        .started()
        .expect("the first watch holds the slot");

    // The same request on another connection is covered, not an error.
    let covered = server.hub_events.lock().await.begin_watch(
        "orchestrator",
        2,
        None,
        &WatchSelection::new(Vec::<String>::new(), ["round-a".to_string()], true),
    );
    assert!(
        matches!(covered, crate::mcp::events::WatchStart::Covered { .. }),
        "an identical request must be covered"
    );

    // A broader request widens the stored selection to the union.
    let widened = server.hub_events.lock().await.begin_watch(
        "orchestrator",
        2,
        None,
        &WatchSelection::new(Vec::<String>::new(), Vec::<String>::new(), true),
    );
    match widened {
        crate::mcp::events::WatchStart::Widened { selection, .. } => {
            assert!(
                selection.contains("--all"),
                "the union keeps round mode: {selection}"
            );
        }
        _ => panic!("a broader request must widen the running watch"),
    }
    let stored = server
        .hub_events
        .lock()
        .await
        .selection_of("orchestrator")
        .expect("the slot is still held");
    assert!(
        stored.covers(&WatchSelection::new(
            Vec::<String>::new(),
            Vec::<String>::new(),
            true
        )),
        "the running watch now follows the union"
    );

    let _ = std::fs::remove_dir_all(&base);
}

/// `consolidate: false` is the absence of a round, not a round: a dispatch that
/// spells it out still answers with its group and its token-bound watch, so the
/// `--quiet` reminder keeps following this caller's own workers instead of being
/// rewritten into a round wait that will never be satisfied.
#[tokio::test]
async fn a_declined_consolidate_round_does_not_rewrite_the_watch_command() {
    let base = scratch("declined-consolidate");
    let repo = base.join("repo");
    std::fs::create_dir_all(&repo).expect("create the scratch repository");
    crate::worktree::git(&repo, "init", &["-b", "master"]).expect("init the scratch repository");
    crate::worktree::git(&repo, "config", &["user.name", "mini-swe-test"]).expect("config");
    crate::worktree::git(&repo, "config", &["user.email", "test@localhost"]).expect("config");
    std::fs::write(repo.join("README.md"), "# scratch\n").expect("seed the repository");
    crate::worktree::git(&repo, "add", &["README.md"]).expect("stage the seed");
    crate::worktree::git(&repo, "commit", &["-m", "baseline"]).expect("commit the seed");

    let server = McpServer::new(
        WorkerPool::with_scratch(
            4,
            "http://localhost:1".to_string(),
            "test-key".to_string(),
            ScratchRoot::new(&base),
        ),
        "ninja".to_string(),
    );
    let tokens_dir = base.join("watch-tokens");
    std::fs::create_dir_all(&tokens_dir).expect("create the token directory");
    let mut ctx = ConnectionContext::hub_connection(1)
        .with_watch_tokens(Arc::new(WatchTokens::new(tokens_dir)));
    ctx.agent_id = Some("orchestrator".to_string());

    let answer = server
        .execute_tool_for(
            "worker",
            json!({
                "action": "dispatch",
                "task": "probe",
                "repo_path": repo.to_string_lossy(),
                "group": "round-48",
                "consolidate": false,
            }),
            &ctx,
        )
        .await
        .expect("a dispatch with no round requested must answer");
    let wid = answer["worker_id"]
        .as_str()
        .expect("a dispatched worker")
        .to_string();

    assert_eq!(
        answer["group"], json!("round-48"),
        "the answer still names the group: {answer}"
    );
    assert!(
        answer.get("consolidate").is_none(),
        "no round was requested, so no answer claims one: {answer}"
    );
    assert_eq!(
        crate::cli::format::format_dispatch_quiet(&answer).watch_command,
        answer["watch_command"].as_str().expect("the hub minted a token-bound command"),
        "a declined round keeps the command that follows this caller's workers: {answer}"
    );

    server.pool.kill(&wid).await;
    crate::pool::remove_registry_entry_in(&ScratchRoot::new(&base), &wid);
    let _ = std::fs::remove_dir_all(&base);
}
