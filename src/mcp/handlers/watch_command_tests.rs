//! Hermetic tests for the `watch_command` a dispatch or steer answer carries.
//!
//! Each test builds a private scratch root: the steer path writes worker
//! mailboxes and history under the pool's scratch, so an ambient root would let
//! a parallel test (or the real hub) collide with it. The dispatch tests below
//! go one step further: they install the watch-token store and the round store
//! where the hub daemon installs them, but under that scratch root and without
//! the scheduler loop, so nothing behind a test dispatches a real worker run.

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

/// A throwaway git repository a dispatched worker can build its worktree from.
fn scratch_repo(base: &std::path::Path) -> std::path::PathBuf {
    let repo = base.join("repo");
    std::fs::create_dir_all(&repo).expect("create the scratch repository");
    crate::worktree::git(&repo, "init", &["-b", "master"]).expect("init the repository");
    crate::worktree::git(&repo, "config", &["user.name", "mini-swe-test"]).expect("config");
    crate::worktree::git(&repo, "config", &["user.email", "test@localhost"]).expect("config");
    std::fs::write(repo.join("README.md"), "# scratch\n").expect("seed the repository");
    crate::worktree::git(&repo, "add", &["README.md"]).expect("stage the seed");
    crate::worktree::git(&repo, "commit", &["-m", "baseline"]).expect("commit the seed");
    repo
}

/// A pool on `base`'s own scratch root and a hub-style connection whose watch
/// token store lives inside it, so a dispatched worker, its registry row and
/// the token file all stay in the directory the test removes.
fn dispatch_server(base: &std::path::Path, tag: &str) -> (McpServer, ConnectionContext) {
    let server = McpServer::new(
        WorkerPool::with_scratch(
            4,
            "http://localhost:1".to_string(),
            "test-key".to_string(),
            ScratchRoot::new(base),
        ),
        "ninja".to_string(),
    );
    let tokens_dir = base.join("watch-tokens");
    std::fs::create_dir_all(&tokens_dir).expect("create the token directory");
    let mut ctx = ConnectionContext::hub_connection(1)
        .with_watch_tokens(Arc::new(WatchTokens::new(tokens_dir)));
    ctx.agent_id = Some(format!("orchestrator-{tag}"));
    (server, ctx)
}

/// Drop the row, the worktree and the directory one dispatched worker left.
async fn reap_worker(server: &McpServer, base: &std::path::Path, wid: &str) {
    server.pool.kill(wid).await;
    crate::pool::remove_registry_entry_in(&ScratchRoot::new(base), wid);
    let _ = std::fs::remove_dir_all(base);
}

/// Claim `identity`'s one watch slot the way the shell watch behind a
/// dispatch answer does: a `hub/watch` poll, which claims the slot for the
/// connection it arrives on and holds it for that connection's lifetime.
async fn hold_watch_slot(server: &McpServer, identity: &str, connection: u64) {
    let mut ctx = ConnectionContext::hub_connection(connection);
    ctx.agent_id = Some(identity.to_string());
    crate::mcp::events::watch_request(
        &server.pool,
        &server.hub_events,
        &ctx,
        json!({"worker_ids": [], "group": [], "initial": false, "all": false}),
        false,
    )
    .await
    .expect("a watch poll claims the identity's slot");
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
    hold_watch_slot(&server, "someone-else", 2).await;
    let foreign = steer(&server, &ctx).await;
    assert!(
        foreign["watch_command"].is_string(),
        "another identity's watch must not silence this caller: {foreign}"
    );

    // This caller's own watch: the hub enforces one watch per session and the
    // running one delivers the next event, so the command is redundant.
    hold_watch_slot(&server, "orchestrator", 1).await;
    let watching = steer(&server, &ctx).await;
    assert!(
        watching.get("watch_command").is_none(),
        "a running watch makes the command redundant: {watching}"
    );

    let _ = std::fs::remove_dir_all(&base);
}

/// The `watch` action's answer carries the command bound to *this* caller: the
/// hub's watch token is what makes a shell watch follow this identity's
/// workers, and a plain `mini-swe-mcp watch` would resolve to whichever
/// identity the operator's shell happens to be.
#[tokio::test]
async fn the_watch_action_binds_its_command_to_the_caller() {
    let base = scratch("watch-action-token");
    let (server, ctx) = dispatch_server(&base, "action");

    // No worker is dispatched and none of this caller's ids are resolved: the
    // answer is the command to run, so a name it does not know rides along
    // verbatim rather than being refused.
    let answer = server
        .execute_tool_for(
            "worker",
            json!({"action": "watch", "worker_id": "abc9", "all": true}),
            &ctx,
        )
        .await
        .expect("the watch action answers");

    assert_eq!(answer["status"], "use_shell", "{answer}");
    let command = answer["watch_command"]
        .as_str()
        .expect("the action names the command to run");
    assert!(
        command.starts_with("MINI_SWE_WATCH_TOKEN="),
        "the command must be bound to this caller's identity: {command}"
    );
    assert!(
        command.ends_with(" mini-swe-mcp watch abc9 --all"),
        "the command must name the selection the call asked for: {command}"
    );

    let _ = std::fs::remove_dir_all(&base);
}

/// A dispatch that asks to consolidate answers with what a `--quiet` caller has
/// to wait on: the round's group and the `consolidate` key beside the token-bound
/// `watch_command`. The flag matters past the answer -- only the hub's scheduler
/// starts the round's consolidator -- and the token matters past the reminder,
/// because the round command it names (`watch --group <g> --all`) run without it
/// would resolve to whichever identity the operator's shell is, and follow that
/// agent's workers instead of this round.
#[tokio::test]
async fn a_consolidated_dispatch_answer_carries_the_round_and_its_token() {
    let base = scratch("consolidated-round");
    let repo = scratch_repo(&base);
    let (server, ctx) = dispatch_server(&base, "round");
    // The auto-consolidation store, installed where the daemon installs it. The
    // scheduler loop is deliberately not started: what is under test is the
    // flag's validation and the answer it rides out on, and nothing behind this
    // test may dispatch a consolidator.
    let hub_dir = base.join("hub");
    std::fs::create_dir_all(&hub_dir).expect("create the hub directory");
    *server.auto_consolidate.lock().expect("the store mutex") = Some(
        crate::hub::auto_consolidate::AutoConsolidate::open(hub_dir).expect("open the round store"),
    );

    let answer = server
        .execute_tool_for(
            "worker",
            json!({
                "action": "dispatch",
                "task": "probe",
                "repo_path": repo.to_string_lossy(),
                "group": "round-48",
                "consolidate": "nerd",
            }),
            &ctx,
        )
        .await
        .expect("a consolidated dispatch must answer");
    let wid = answer["worker_id"]
        .as_str()
        .expect("a dispatched worker")
        .to_string();
    assert_eq!(answer["group"], json!("round-48"), "{answer}");
    assert_eq!(answer["consolidate"], json!("nerd"), "{answer}");
    let token = answer["watch_command"]
        .as_str()
        .expect("the hub minted a token-bound command")
        .strip_prefix("MINI_SWE_WATCH_TOKEN=")
        .and_then(|rest| rest.strip_suffix(" mini-swe-mcp watch"))
        .expect("token between the assignment and the command")
        .to_string();
    assert_eq!(
        crate::cli::format::format_dispatch_quiet(&answer).watch_command,
        format!("MINI_SWE_WATCH_TOKEN={token} mini-swe-mcp watch --group round-48 --all"),
        "the reminder must wait for the round, as this caller: {answer}"
    );

    reap_worker(&server, &base, &wid).await;
}

/// `consolidate: false` is the absence of a round, not a round: a dispatch that
/// spells it out still answers with its group and its token-bound watch, so the
/// `--quiet` reminder keeps following this caller's own workers instead of being
/// rewritten into a round wait that will never be satisfied.
#[tokio::test]
async fn a_declined_consolidate_round_does_not_rewrite_the_watch_command() {
    let base = scratch("declined-consolidate");
    let repo = scratch_repo(&base);
    let (server, ctx) = dispatch_server(&base, "declined");

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
        answer["group"],
        json!("round-48"),
        "the answer still names the group: {answer}"
    );
    assert!(
        answer.get("consolidate").is_none(),
        "no round was requested, so no answer claims one: {answer}"
    );
    assert_eq!(
        crate::cli::format::format_dispatch_quiet(&answer).watch_command,
        answer["watch_command"]
            .as_str()
            .expect("the hub minted a token-bound command"),
        "a declined round keeps the command that follows this caller's workers: {answer}"
    );

    reap_worker(&server, &base, &wid).await;
}
