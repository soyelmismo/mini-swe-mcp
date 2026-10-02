//! A consolidated round gives its workers the cheap gate and its one
//! consolidator the full gate.
//!
//! Without that split every worker of a round runs the whole suite, and the
//! consolidator runs it again over the integrated result. So a dispatch that
//! asks to consolidate the round (`consolidate` set, no explicit `verify`)
//! records the cheap static gate on its workers and the full auto-detected
//! gate on the consolidator, while an explicit `verify` still wins for both.
//!
//! The gate a worker runs is spelled out in its opening message and persisted
//! in its conversation log, so that durable record is what these tests read --
//! no model answer has to be asserted on. Each repository is a throwaway tree
//! whose manifest alone decides which gate is detected, and every pool, log
//! and worktree lives under a temporary scratch root.

mod common;

use common::{IsolatedPool, TempDir, git};
use mini_swe_mcp::mcp::{ConnectionContext, McpServer};
use mini_swe_mcp::pool::history_log_path_in;
use mini_swe_mcp::worktree::ScratchRoot;
use serde_json::{Value, json};
use std::path::Path;
use std::time::Duration;

/// The cheap gate a Rust manifest selects, spelled out so the test pins the
/// exact string rather than re-deriving it from the code under test.
const RUST_CHEAP: &str = "cargo fmt --check && cargo clippy --all-targets -- -D warnings";

/// The full gate the same manifest selects: the consolidator runs this once
/// over the integrated round, so it is the half a worker must never be given.
const RUST_FULL: &str = "cargo build --all-targets && cargo test";

/// A throwaway repository with a baseline commit, optionally carrying the
/// manifest whose presence selects the ecosystem.
fn repo(tag: &str, manifest: Option<(&str, &str)>) -> TempDir {
    let repo = TempDir::new_in_tmp(tag);
    git(repo.path(), &["init", "-b", "master"]);
    git(repo.path(), &["config", "user.name", "mini-swe-test"]);
    git(repo.path(), &["config", "user.email", "test@localhost"]);
    if let Some((name, body)) = manifest {
        std::fs::write(repo.path().join(name), body).unwrap();
    }
    std::fs::write(repo.path().join("README.md"), "# baseline\n").unwrap();
    git(repo.path(), &["add", "."]);
    git(repo.path(), &["commit", "-m", "baseline"]);
    repo
}

/// The gate `worker_id` was told to run, read back from its durable
/// conversation log, or `None` when it was told none.
///
/// Polls rather than sleeping a fixed span: the opening message is written at
/// the dispatch, so the deadline bounds how long the file may take to appear,
/// not how long a gate is expected to exist. Returns as soon as the opening
/// message is in the log, which is also the answer when that message carries
/// no gate -- so the "no cheap gate for this ecosystem" arms assert on `None`
/// instead of waiting for a line that must never arrive.
fn recorded_gate(root: &ScratchRoot, worker_id: &str) -> Option<String> {
    let path = history_log_path_in(root, worker_id);
    let deadline = std::time::Instant::now() + Duration::from_secs(20);
    loop {
        let raw = std::fs::read_to_string(&path).unwrap_or_default();
        for line in raw.lines() {
            let Ok(value) = serde_json::from_str::<Value>(line) else {
                continue;
            };
            let content = value["content"].as_str().unwrap_or_default();
            if content.contains("Begin by exploring the repository.") {
                // The opening message is the last thing written before the
                // first turn, so its presence settles the question either way.
                return content
                    .split("Completion gate: `")
                    .nth(1)
                    .and_then(|rest| rest.split('`').next())
                    .map(str::to_string);
            }
        }
        assert!(
            std::time::Instant::now() < deadline,
            "worker {worker_id} never wrote its opening message to {}:\n{raw}",
            path.display()
        );
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// A pool, its scratch root, and the auto-consolidation scheduler `consolidate`
/// requires. The fake LLM only has to answer: these tests read the gate a
/// worker was *given*, so what the model replies is irrelevant.
struct Round {
    harness: IsolatedPool,
    server: McpServer,
    scheduler: tokio::task::JoinHandle<()>,
}

impl Round {
    async fn new(tag: &str) -> Self {
        let harness = IsolatedPool::new(4, tag);
        let hub = harness.scratch.subdir("hub");
        let server = McpServer::new(harness.pool.clone(), "test-model".into());
        // `consolidate` is refused without the hub's scheduler, so every
        // consolidated-round dispatch goes through a real one. It owns a
        // temporary hub directory and is aborted in `Drop`.
        let scheduler = server
            .start_auto_consolidate(hub)
            .await
            .expect("the auto-consolidation scheduler starts");
        Self {
            harness,
            server,
            scheduler,
        }
    }

    fn pool(&self) -> mini_swe_mcp::pool::WorkerPool {
        self.harness.pool.clone()
    }

    fn root(&self) -> ScratchRoot {
        self.harness.root()
    }

    /// Commit `files` on a `worker-<id>` branch off `master` and record the
    /// matching completed registry row: the shape of a finished worker the
    /// round manifest lists as ready to integrate.
    fn finished_worker(&self, repo: &Path, worker_id: &str, files: &[(&str, &str)]) {
        let branch = format!("worker-{worker_id}");
        git(repo, &["checkout", "-q", "master"]);
        git(repo, &["checkout", "-q", "-b", &branch]);
        for (file, content) in files {
            std::fs::write(repo.join(file), content).unwrap();
            git(repo, &["add", file]);
            git(
                repo,
                &[
                    "-c",
                    "user.name=w",
                    "-c",
                    "user.email=w@x",
                    "commit",
                    "-m",
                    file,
                ],
            );
        }
        git(repo, &["checkout", "-q", "master"]);
        mini_swe_mcp::pool::save_registry_entry_in(
            &self.root(),
            &mini_swe_mcp::pool::WorkerRegistryEntry {
                task: "finished work".to_string(),
                status: mini_swe_mcp::pool::RegistryStatus::Completed,
                group: Some("round".to_string()),
                role: mini_swe_mcp::pool::WorkerRole::Worker,
                repo_path: Some(repo.to_string_lossy().to_string()),
                base_branch: Some("master".to_string()),
                ..mini_swe_mcp::pool::WorkerRegistryEntry::test_row(worker_id, "cheap-gate-owner")
            },
        );
    }

    /// Dispatch the consolidator for `group` and return its worker id.
    async fn consolidate(&self, repo: &Path, group: &str, args: Value) -> String {
        let mut ctx = ConnectionContext::stdio();
        ctx.agent_id = Some("cheap-gate-owner".into());
        let mut args = args;
        args["action"] = json!("consolidate");
        args["repo_path"] = json!(repo);
        args["group"] = json!(group);
        let result = self
            .server
            .execute_tool_for("worker", args, &ctx)
            .await
            .expect("the consolidate is accepted");
        result["worker_id"]
            .as_str()
            .unwrap_or_else(|| panic!("the consolidate answered without a worker id: {result}"))
            .to_string()
    }

    /// Dispatch `args` as a worker action and return the first worker id.
    async fn dispatch(&self, repo: &Path, args: Value) -> String {
        let mut ctx = ConnectionContext::stdio();
        ctx.agent_id = Some("cheap-gate-owner".into());
        let mut args = args;
        args["action"] = json!("dispatch");
        args["repo_path"] = json!(repo);
        if let Some(object) = args.as_object_mut() {
            object.entry("group").or_insert_with(|| json!("round"));
        }
        let result = self
            .server
            .execute_tool_for("worker", args, &ctx)
            .await
            .expect("the dispatch is accepted");
        // A single dispatch answers with its own id; a batch answers with a
        // `workers` array. These tests dispatch one task, so take the id from
        // whichever shape came back rather than assuming one.
        let id = result
            .get("worker_id")
            .or_else(|| {
                result
                    .get("workers")
                    .and_then(|w| w.get(0))
                    .and_then(|w| w.get("worker_id"))
            })
            .and_then(Value::as_str)
            .unwrap_or_else(|| panic!("the dispatch answered without a worker id: {result}"));
        id.to_string()
    }
}

impl Drop for Round {
    fn drop(&mut self) {
        // The scheduler is a detached background task: leaving it running
        // would keep the scratch root alive past the assertions.
        self.scheduler.abort();
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_consolidated_round_gives_its_workers_the_cheap_gate() {
    let repo = repo("cheap-gate-round", Some(("Cargo.toml", "[package]\n")));
    let round = Round::new("cheap-gate-round-pool").await;
    let worker = round
        .dispatch(
            repo.path(),
            json!({"task": "add a parser", "consolidate": true, "max_turns": 2}),
        )
        .await;

    assert_eq!(
        recorded_gate(&round.root(), &worker).as_deref(),
        Some(RUST_CHEAP),
        "a consolidated round's worker must be gated on the cheap static subset, \
         not the full suite the consolidator runs again"
    );
    let _ = round.pool().kill_all().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn a_plain_dispatch_still_gets_the_full_gate() {
    let repo = repo("cheap-gate-plain", Some(("Cargo.toml", "[package]\n")));
    let round = Round::new("cheap-gate-plain-pool").await;
    let worker = round
        .dispatch(repo.path(), json!({"task": "add a parser", "max_turns": 2}))
        .await;

    assert_eq!(
        recorded_gate(&round.root(), &worker).as_deref(),
        Some(RUST_FULL),
        "without 'consolidate' the worker keeps the full auto-detected gate"
    );
    let _ = round.pool().kill_all().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn an_explicit_verify_wins_over_the_cheap_gate() {
    let repo = repo("cheap-gate-explicit", Some(("Cargo.toml", "[package]\n")));
    let round = Round::new("cheap-gate-explicit-pool").await;
    let worker = round
        .dispatch(
            repo.path(),
            json!({
                "task": "add a parser",
                "consolidate": true,
                "verify": "cargo test --test explicit_gate",
                "max_turns": 2,
            }),
        )
        .await;

    assert_eq!(
        recorded_gate(&round.root(), &worker).as_deref(),
        Some("cargo test --test explicit_gate"),
        "an explicit verify is the dispatcher's decision and is never overridden"
    );
    let _ = round.pool().kill_all().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn an_empty_verify_still_disables_the_gate_on_a_consolidated_round() {
    let repo = repo("cheap-gate-disabled", Some(("Cargo.toml", "[package]\n")));
    let round = Round::new("cheap-gate-disabled-pool").await;
    let worker = round
        .dispatch(
            repo.path(),
            json!({"task": "add a parser", "consolidate": true, "verify": "", "max_turns": 2}),
        )
        .await;

    assert_eq!(
        recorded_gate(&round.root(), &worker),
        None,
        "an explicit empty verify means no gate, not the cheap default"
    );
    let _ = round.pool().kill_all().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn a_repository_without_a_cheap_gate_is_dispatched_without_one() {
    let repo = repo("cheap-gate-none", None);
    let round = Round::new("cheap-gate-none-pool").await;
    let worker = round
        .dispatch(
            repo.path(),
            json!({"task": "add a parser", "consolidate": true, "max_turns": 2}),
        )
        .await;

    assert_eq!(
        recorded_gate(&round.root(), &worker),
        None,
        "the detector's 'otherwise none' arm must not be replaced by the full suite: \
         a worker the project cannot gate cheaply runs no gate"
    );
    let _ = round.pool().kill_all().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn a_node_round_gates_its_workers_on_the_package_scripts() {
    let repo = repo(
        "cheap-gate-node",
        Some((
            "package.json",
            r#"{"name":"x","scripts":{"lint":"eslint ."}}"#,
        )),
    );
    let round = Round::new("cheap-gate-node-pool").await;
    let worker = round
        .dispatch(
            repo.path(),
            json!({"task": "add a parser", "consolidate": true, "max_turns": 2}),
        )
        .await;

    assert_eq!(
        recorded_gate(&round.root(), &worker).as_deref(),
        Some("npm run lint"),
        "the package's own lint script is its cheap gate"
    );
    let _ = round.pool().kill_all().await;
}

/// The other half of the split: the round's one consolidator runs the full
/// gate, even though every worker of that round ran the cheap one. This is
/// the property that makes the cheap default pay off -- the suite runs once,
/// over the integrated result, instead of once per worker plus once at the
/// end.
#[tokio::test(flavor = "multi_thread")]
async fn the_round_consolidator_still_gets_the_full_gate() {
    let repo = repo(
        "cheap-gate-consolidator",
        Some(("Cargo.toml", "[package]\n")),
    );
    let round = Round::new("cheap-gate-consolidator-pool").await;
    round.finished_worker(repo.path(), "aaaa1111", &[("worker.rs", "fn main() {}\n")]);

    let worker = round
        .dispatch(
            repo.path(),
            json!({"task": "add a parser", "consolidate": true, "max_turns": 2}),
        )
        .await;
    let consolidator = round
        .consolidate(repo.path(), "round", json!({"max_turns": 2}))
        .await;

    assert_eq!(
        recorded_gate(&round.root(), &worker).as_deref(),
        Some(RUST_CHEAP),
        "the worker ran the cheap gate"
    );
    assert_eq!(
        recorded_gate(&round.root(), &consolidator).as_deref(),
        Some(RUST_FULL),
        "the consolidator alone runs the full gate over the integrated round"
    );
    let _ = round.pool().kill_all().await;
}
