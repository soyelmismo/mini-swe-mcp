//! Amending a round's auto-consolidation settings: a gate that cannot be parsed
//! is refused where it enters (the dispatch), and a gate that was accepted can
//! still be replaced on a round nobody has consumed yet.
//
// Both failures this covers were found the hard way -- once from a
//! `--consolidate-verify` whose shell quoting split it and once from editing
//! `<hub_dir>/auto-consolidate.json` by hand, which the daemon overwrote because
//! it keeps the rounds in memory. So the tests drive the real handler through
//! the real hub store: a refused dispatch must leave no round behind, and
//! `--set` must land on disk rather than in the daemon's memory only.

use crate::common;
use mini_swe_mcp::cli::args::tool_args;
use mini_swe_mcp::mcp::{ConnectionContext, McpServer};
use mini_swe_mcp::pool::WorkerPool;
use mini_swe_mcp::worktree::ScratchRoot;
use serde_json::json;
const OWNER: &str = "round-owner";

/// A repository with one baseline commit plus the server and scheduler a round
/// needs. The LLM endpoint is unreachable: every test here is refused or
/// answered from the store, so no worker ever reaches it.
struct Harness {
    _scratch: common::TempDir,
    repo: std::path::PathBuf,
    server: McpServer,
    hub: std::path::PathBuf,
}

impl Harness {
    fn new(tag: &str) -> Self {
        let scratch = common::TempDir::new_in_tmp(tag);
        let repo = scratch.subdir("repo");
        common::git(&repo, &["init", "-b", "main"]);
        common::git(&repo, &["config", "user.email", "test@example.test"]);
        common::git(&repo, &["config", "user.name", "Test"]);
        std::fs::write(repo.join("README"), "base\n").unwrap();
        common::git(&repo, &["add", "."]);
        common::git(&repo, &["commit", "-m", "base"]);
        let root = ScratchRoot::new(scratch.subdir("workers"));
        let hub = scratch.subdir("hub");
        let pool =
            WorkerPool::with_scratch(4, "http://localhost:1".into(), "test-key".into(), root);
        let server = McpServer::new(pool, "nerd".into());
        Self {
            _scratch: scratch,
            repo,
            server,
            hub,
        }
    }

    fn ctx(agent: &str) -> ConnectionContext {
        let mut ctx = ConnectionContext::stdio();
        ctx.agent_id = Some(agent.to_string());
        ctx
    }

    /// The rounds the daemon has on disk, as the JSON the orchestrator would
    /// read if editing the file worked (it does not: the daemon rewrites it).
    fn stored_rounds(&self) -> Vec<serde_json::Value> {
        serde_json::from_slice(&std::fs::read(self.hub.join("auto-consolidate.json")).unwrap())
            .unwrap()
    }

    async fn start(&self) -> tokio::task::JoinHandle<()> {
        self.server
            .start_auto_consolidate(self.hub.clone())
            .await
            .unwrap()
    }
}

/// The `--consolidate-verify` value the observed failures carried: a quote the
/// caller's shell split, leaving a command `sh` cannot even parse.
const MANGLED: &str = "'cargo";

/// An unparsable gate is refused at the dispatch that carries it, with `sh`'s
/// own parse error -- and the refusal names the field, so the caller knows which
/// argument to fix. Both spellings are covered: the round gate
/// (`consolidate_verify`) and the worker's own (`verify`), which is stored on
/// the worker for the same reason.
#[tokio::test]
async fn an_unparsable_gate_is_refused_at_dispatch() {
    let harness = Harness::new("amend-refuse");
    let scheduler = harness.start().await;
    for (field, extra) in [
        ("consolidate_verify", json!({"consolidate": true})),
        ("verify", json!({})),
    ] {
        let error = harness
            .server
            .execute_tool_for(
                "worker",
                json!({
                    "action": "dispatch", "task": "a contribution",
                    "repo_path": harness.repo.as_path(), "group": "round",
                    "verify": "",
                })
                .as_object()
                .unwrap()
                .clone()
                .into_iter()
                .chain(extra.as_object().unwrap().clone())
                .chain([(field.to_string(), json!(MANGLED))])
                .collect(),
                &Harness::ctx(OWNER),
            )
            .await
            .expect_err("a gate `sh` cannot parse must not reach a worker");
        let text = error.to_string();
        assert!(
            text.contains(field),
            "the refusal must name {field}: {text}"
        );
        assert!(
            text.contains("valid shell command") && text.contains("sh:"),
            "the refusal must carry sh's own parse error: {text}"
        );
    }
    // Nothing was recorded: a refused dispatch opens no round at all.
    assert!(
        !harness.hub.join("auto-consolidate.json").exists(),
        "a refused dispatch must not persist a round"
    );
    scheduler.abort();
    let _ = scheduler.await;
}

/// A gate that only fails when it *runs* is still accepted: the parse check is
/// about the shell syntax, not about whether the command fits this checkout.
/// Without this the refusal would be a surprise on every gate whose tools land
/// later.
#[test]
fn a_gate_that_only_fails_at_run_time_is_accepted() {
    assert!(
        mini_swe_mcp::pool::validate_verify_command("cargo test --all", "verify").is_ok(),
        "a run-time failure is not a parse error"
    );
    // The empty gate is how a dispatch disables it, so it must not be parsed.
    assert!(
        mini_swe_mcp::pool::validate_verify_command("   ", "verify").is_ok(),
        "an empty gate disables rather than fails"
    );
}

/// `--set` amends the pending round of the caller, and the amendment is on disk
/// (not only in the daemon's memory), so it survives the restart that a hand
/// edit would not.
#[tokio::test]
async fn set_amends_a_pending_round_and_persists_it() {
    let harness = Harness::new("amend-pending");
    let scheduler = harness.start().await;
    let opened = harness
        .server
        .execute_tool_for(
            "worker",
            json!({
                "action": "dispatch", "task": "a contribution",
                "repo_path": harness.repo.as_path(), "group": "round",
                "consolidate": true, "consolidate_verify": "cargo build",
            }),
            &Harness::ctx(OWNER),
        )
        .await
        .unwrap();
    assert!(
        opened["worker_id"].is_string(),
        "the round must have opened a worker to consolidate: {opened}"
    );

    let amended = harness
        .server
        .execute_tool_for(
            "worker",
            json!({
                "action": "consolidate", "group": "round", "set": true,
                "verify": "cargo test --all-features",
            }),
            &Harness::ctx(OWNER),
        )
        .await
        .unwrap();
    assert_eq!(amended["amended"], true, "{amended}");
    assert_eq!(amended["group"], "round", "{amended}");
    let rounds = harness.stored_rounds();
    assert_eq!(rounds.len(), 1, "{rounds:?}");
    assert_eq!(
        rounds[0]["verify"], "cargo test --all-features",
        "the amendment must be on disk, not only in memory: {rounds:?}"
    );
    // The model half was not named, so it must be untouched.
    assert_eq!(rounds[0]["model"], json!(null), "{rounds:?}");
    scheduler.abort();
    let _ = scheduler.await;
}

/// `--set` without a gate to install would silently do nothing, so it is
/// refused: the caller either meant to change something or meant to dispatch.
#[tokio::test]
async fn set_without_a_setting_is_refused() {
    let harness = Harness::new("amend-empty");
    let scheduler = harness.start().await;
    harness
        .server
        .execute_tool_for(
            "worker",
            json!({
                "action": "dispatch", "task": "a contribution",
                "repo_path": harness.repo.as_path(), "group": "round", "consolidate": true,
            }),
            &Harness::ctx(OWNER),
        )
        .await
        .unwrap();
    let error = harness
        .server
        .execute_tool_for(
            "worker",
            json!({"action": "consolidate", "group": "round", "set": true}),
            &Harness::ctx(OWNER),
        )
        .await
        .expect_err("an amend that changes nothing must be refused");
    assert!(
        error.to_string().contains("'model'"),
        "the refusal must say what to pass: {error}"
    );
    scheduler.abort();
    let _ = scheduler.await;
}

/// The CLI spelling reaches the same tool arguments as the MCP path, so the two
/// cannot drift: `--set` and the flags it amends are folded into `set`, `group`,
/// `model` and `verify` exactly as the handler reads them.
#[test]
fn the_cli_parses_consolidate_set() {
    let argv: Vec<String> = [
        "mini-swe-mcp",
        "consolidate",
        "--group",
        "round-1",
        "--set",
        "--model",
        "ninja",
        "--verify",
        "cargo test",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let args = tool_args("consolidate", &argv, true)
        .expect("consolidate is a known action")
        .expect("consolidate goes through the worker tool");
    assert_eq!(
        args,
        json!({
            "action": "consolidate",
            "group": "round-1",
            "set": true,
            "model": "ninja",
            "verify": "cargo test",
        })
        .as_object()
        .unwrap()
        .clone(),
        "{args:?}"
    );
}

/// The CLI prints the amended settings, because the whole point of the verb is
/// that the caller can see the gate it just installed instead of having to
/// remember it.
#[test]
fn the_amend_answer_renders_the_settings_it_installed() {
    let out = mini_swe_mcp::cli::format::format_output(
        "consolidate",
        &json!({"amended": true, "group": "round-1", "model": "ninja", "verify": "cargo test"}),
    );
    assert_eq!(
        out, "Round round-1 amended: consolidator model ninja, gate cargo test",
        "{out}"
    );
    // A cleared gate says so, rather than looking like the one just installed.
    let cleared = mini_swe_mcp::cli::format::format_output(
        "consolidate",
        &json!({"amended": true, "group": "round-1", "model": null, "verify": ""}),
    );
    assert!(
        cleared.contains("gate none (auto-detect)"),
        "a cleared gate must be visible: {cleared}"
    );
    // The dispatch meaning of the same action is unchanged by the flag.
    let dispatched = mini_swe_mcp::cli::format::format_output(
        "consolidate",
        &json!({"worker_id": "w0", "group": "round-1"}),
    );
    assert!(
        dispatched.starts_with("Consolidator w0 dispatched"),
        "{dispatched}"
    );
}
