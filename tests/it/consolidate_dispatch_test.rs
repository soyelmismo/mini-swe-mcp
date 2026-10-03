//! Integration tests for the consolidator's dispatch: the round manifest the
//! hub computes, the refusal that keeps a consolidator from being sent after
//! branches nobody finished, and the CLI/help spellings of the verb.
//!
//! The manifest is the whole contract between the hub and the consolidator, so
//! it is driven through [`mini_swe_mcp::pool::WorkerPool::round_manifest`] on a
//! temporary repository -- no LLM, no sandbox, and never the host repository.

use crate::common::{IsolatedPool, TempDir, git, unique_suffix};
use mini_swe_mcp::cli::args::tool_args;
use mini_swe_mcp::cli::help::topic_text;
use mini_swe_mcp::pool::{
    RegistryStatus, WorkerMetrics, WorkerPool, WorkerRecord, WorkerRegistryEntry, WorkerRole,
    WorkerState, save_registry_entry_in,
};
use mini_swe_mcp::worktree::ScratchRoot;
use serde_json::{Map, Value, json};
use std::path::Path;

const OWNER: &str = "agent-a";
const GROUP: &str = "round-1";

/// A temporary repository with one baseline commit, plus the pool and scratch
/// root the registry rows are filed under.
struct Harness {
    repo: TempDir,
    pool: IsolatedPool,
}

impl Harness {
    fn new(tag: &str) -> Self {
        let repo = TempDir::new_in_tmp(tag);
        git(repo.path(), &["init", "-b", "master"]);
        git(repo.path(), &["config", "user.name", "mini-swe-test"]);
        git(repo.path(), &["config", "user.email", "test@localhost"]);
        std::fs::write(repo.path().join("README.md"), "# baseline\n").unwrap();
        git(repo.path(), &["add", "README.md"]);
        git(repo.path(), &["commit", "-m", "baseline"]);
        Self {
            repo,
            pool: IsolatedPool::new(4, tag),
        }
    }

    fn path(&self) -> &Path {
        self.repo.path()
    }

    fn root(&self) -> ScratchRoot {
        self.pool.root()
    }

    /// Commit `files` on a `worker-<id>` branch off `master`: the shape of a
    /// finished worker's preserved branch.
    fn worker_branch(&self, worker_id: &str, files: &[(&str, &str)]) {
        let branch = format!("worker-{worker_id}");
        git(self.path(), &["checkout", "-q", "master"]);
        git(self.path(), &["checkout", "-q", "-b", &branch]);
        for (file, content) in files {
            std::fs::write(self.path().join(file), content).unwrap();
            git(self.path(), &["add", file]);
            git(
                self.path(),
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
        git(self.path(), &["checkout", "-q", "master"]);
    }

    /// The registry row a worker leaves behind, in `group` (or in none).
    fn row(
        &self,
        worker_id: &str,
        task: &str,
        status: RegistryStatus,
        role: WorkerRole,
        group: Option<&str>,
    ) {
        let entry = WorkerRegistryEntry {
            task: task.to_string(),
            status,
            step: 3,
            last_command: "completed".to_string(),
            group: group.map(str::to_string),
            role,
            repo_path: Some(self.path().to_string_lossy().to_string()),
            base_branch: Some("master".to_string()),
            ..WorkerRegistryEntry::test_row(worker_id, OWNER)
        };
        save_registry_entry_in(&self.root(), &entry);
    }

    /// The in-process record of a worker this process still holds, so the
    /// manifest can read the verification outcome no registry row carries.
    async fn live_completed(&self, worker_id: &str, verified: Option<bool>) {
        self.pool
            .pool
            .__test_insert_worker(WorkerRecord {
                id: worker_id.to_string(),
                task: "live worker".to_string(),
                model: "test".to_string(),
                owner: OWNER.to_string(),
                state: WorkerState::Completed {
                    report: None,
                    turns: 3,
                    diff: String::new(),
                    summary: "done".to_string(),
                    completed_at: 0,
                    artifacts: Vec::new(),
                    branch: Some(format!("worker-{worker_id}")),
                    verified,
                    metrics: WorkerMetrics::default(),
                    revision: 0,
                    verdicts: None,
                },
                metrics: WorkerMetrics::default(),
                logs: mini_swe_mcp::pool::LogBuffer::default(),
                pending_steer: Vec::new(),
                resume_tx: None,
                handle: None,
                revision: 0,
            })
            .await;
    }
}

/// The round manifest `owner` would consolidate in `group`.
fn manifest(
    pool: &WorkerPool,
    owner: &str,
    group: &str,
    repo: &Path,
) -> mini_swe_mcp::pool::RoundManifest {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime")
        .block_on(pool.round_manifest(owner, group, repo))
}

/// The one worker of `workers` named `id`.
fn worker<'a>(
    workers: &'a [mini_swe_mcp::pool::RoundWorker],
    id: &str,
) -> &'a mini_swe_mcp::pool::RoundWorker {
    workers
        .iter()
        .find(|worker| worker.id == id)
        .unwrap_or_else(|| panic!("no worker {id} in {workers:?}"))
}

/// Two finished branches that share a file, one still-running worker, and a
/// branch already merged into the base: the manifest separates all three.
#[test]
fn the_round_manifest_separates_ready_workers_from_the_rest() {
    let h = Harness::new("round-manifest");
    let first = format!("w1-{}", unique_suffix("w"));
    let second = format!("w2-{}", unique_suffix("w"));
    let running = format!("w3-{}", unique_suffix("w"));
    let merged = format!("w4-{}", unique_suffix("w"));

    // The interaction point: the same file rewritten by two workers.
    h.worker_branch(
        &first,
        &[("first.txt", "first worker\n"), ("shared.rs", "first\n")],
    );
    h.worker_branch(
        &second,
        &[("second.txt", "second worker\n"), ("shared.rs", "second\n")],
    );
    h.worker_branch(&running, &[("third.txt", "still running\n")]);
    h.worker_branch(&merged, &[("merged.txt", "already integrated\n")]);
    // Integrate one branch into the base, so it is no longer this round's work.
    git(
        h.path(),
        &[
            "merge",
            "--no-ff",
            "-m",
            "integrate",
            &format!("worker-{merged}"),
        ],
    );

    h.row(
        &first,
        "first task\nbody",
        RegistryStatus::Completed,
        WorkerRole::Worker,
        Some(GROUP),
    );
    h.row(
        &second,
        "second task",
        RegistryStatus::Completed,
        WorkerRole::Worker,
        Some(GROUP),
    );
    h.row(
        &running,
        "third task",
        RegistryStatus::Running,
        WorkerRole::Worker,
        Some(GROUP),
    );
    h.row(
        &merged,
        "merged task",
        RegistryStatus::Completed,
        WorkerRole::Worker,
        Some(GROUP),
    );
    // A worker of another group is not this round, whatever its state.
    let foreign = format!("w5-{}", unique_suffix("w"));
    h.worker_branch(&foreign, &[("foreign.txt", "another round\n")]);
    h.row(
        &foreign,
        "another round's task",
        RegistryStatus::Completed,
        WorkerRole::Worker,
        Some("other-round"),
    );

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");
    runtime.block_on(h.live_completed(&first, Some(true)));

    let manifest = manifest(&h.pool.pool, OWNER, GROUP, h.path());

    let ready: Vec<&str> = manifest.ready.iter().map(|w| w.id.as_str()).collect();
    assert_eq!(ready, vec![first.as_str(), second.as_str()], "{manifest:?}");
    let not_ready: Vec<&str> = manifest.not_ready.iter().map(|w| w.id.as_str()).collect();
    assert_eq!(
        not_ready,
        vec![running.as_str()],
        "a running worker is listed separately, never as ready: {manifest:?}"
    );
    assert!(
        !manifest.ready.iter().any(|w| w.id == foreign)
            && !manifest.not_ready.iter().any(|w| w.id == foreign),
        "a worker of another group is not part of this round: {manifest:?}"
    );

    let first_worker = worker(&manifest.ready, &first);
    assert_eq!(first_worker.state, "Completed");
    assert_eq!(
        first_worker.verified,
        Some(true),
        "the live record carries the verification outcome"
    );
    assert_eq!(
        first_worker.task, "first task",
        "only the task's first line"
    );
    assert_eq!(first_worker.files, vec!["first.txt", "shared.rs"]);

    let second_worker = worker(&manifest.ready, &second);
    assert_eq!(
        second_worker.verified, None,
        "a registry-only row cannot claim a verification outcome"
    );
    assert_eq!(second_worker.files, vec!["second.txt", "shared.rs"]);

    assert_eq!(
        worker(&manifest.not_ready, &running).state,
        "Running",
        "a running worker is listed separately, never as ready"
    );
    assert_eq!(
        manifest.interaction_points,
        vec![("shared.rs".to_string(), vec![first.clone(), second.clone()])],
        "a file two workers touched is the round's interaction point: {manifest:?}"
    );

    // The merged branch is gone from both lists, and the text the consolidator
    // reads names every section.
    let text = manifest.render();
    for needle in [
        "ROUND MANIFEST group=round-1 base=master",
        "ready (completed, branch not yet merged):",
        "not ready:",
        "interaction points (touched by more than one worker):",
        &format!("{first} Completed verified=yes task=\"first task\""),
        "shared.rs: ",
    ] {
        assert!(
            text.contains(needle),
            "the manifest must show {needle}:\n{text}"
        );
    }
    assert!(
        !text.contains(&merged),
        "an already-merged branch is not part of the round:\n{text}"
    );
}

/// A group whose workers are all still running, or all already merged, has
/// nothing to integrate: the dispatch refuses instead of sending a
/// consolidator after branches nobody finished.
#[test]
fn a_group_with_no_ready_worker_is_refused() {
    let h = Harness::new("round-refuse");
    let running = format!("w1-{}", unique_suffix("w"));
    h.worker_branch(&running, &[("third.txt", "still running\n")]);
    h.row(
        &running,
        "third task",
        RegistryStatus::Running,
        WorkerRole::Worker,
        Some(GROUP),
    );

    let manifest = manifest(&h.pool.pool, OWNER, GROUP, h.path());
    assert!(!manifest.has_ready(), "{manifest:?}");
    assert_eq!(manifest.ready.len(), 0);

    // The same refusal the MCP verb answers with, driven through the server so
    // the message an orchestrator reads is the one under test.
    let server = mini_swe_mcp::mcp::McpServer::new(h.pool.pool.clone(), "ninja".to_string());
    let ctx = mini_swe_mcp::mcp::ConnectionContext {
        agent_id: Some(OWNER.to_string()),
        ..mini_swe_mcp::mcp::ConnectionContext::hub_connection(3)
    };
    let error = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime")
        .block_on(server.execute_tool_for(
            "worker",
            json!({
                "action": "consolidate",
                "group": GROUP,
                "repo_path": h.path().to_string_lossy(),
            }),
            &ctx,
        ))
        .expect_err("a group with nothing ready must be refused");
    let message = error.to_string();
    assert!(
        message.contains("no completed, unmerged worker"),
        "the refusal must say what is missing: {message}"
    );
    assert!(message.contains(GROUP), "it must name the group: {message}");
}

/// A completed worker whose branch was already merged is not a round
/// contribution either.
#[test]
fn an_already_merged_group_is_refused_too() {
    let h = Harness::new("round-merged");
    let done = format!("w1-{}", unique_suffix("w"));
    h.worker_branch(&done, &[("done.txt", "already integrated\n")]);
    git(
        h.path(),
        &[
            "merge",
            "--no-ff",
            "-m",
            "integrate",
            &format!("worker-{done}"),
        ],
    );
    h.row(
        &done,
        "done task",
        RegistryStatus::Completed,
        WorkerRole::Worker,
        Some(GROUP),
    );

    let manifest = manifest(&h.pool.pool, OWNER, GROUP, h.path());
    assert!(!manifest.has_ready(), "{manifest:?}");
    assert!(manifest.not_ready.is_empty(), "{manifest:?}");
}

/// The consolidator's built-in instructions name every verb of the round
/// workflow, the review criteria and the report the orchestrator reads.
#[test]
fn the_consolidator_instructions_cover_the_whole_round_workflow() {
    let text = mini_swe_mcp::agent::CONSOLIDATOR_INSTRUCTIONS;
    for needle in [
        "CONSOLIDATE_MERGE",
        "CONSOLIDATE_STEER",
        "CONSOLIDATE_WAIT",
        "FULL gate once",
        "REPORT <id> approved|returned|fixed:",
        "RISK:",
        "never weaken",
        "sandbox, governance or identity",
        "hermetic and meaningful",
    ] {
        assert!(
            text.contains(needle),
            "the instructions must mention {needle}: {text}"
        );
    }
}

/// The task the consolidator is dispatched with carries the round it inherits,
/// the one gate it must run, and the procedure it follows.
#[test]
fn the_consolidator_task_embeds_the_round_the_gate_and_the_procedure() {
    let h = Harness::new("round-task");
    let done = format!("w1-{}", unique_suffix("w"));
    h.worker_branch(&done, &[("done.txt", "done\n")]);
    h.row(
        &done,
        "done task",
        RegistryStatus::Completed,
        WorkerRole::Worker,
        Some(GROUP),
    );

    let manifest = manifest(&h.pool.pool, OWNER, GROUP, h.path());
    let task = manifest.task_text(Some("cargo test --all-targets"));
    assert!(task.contains(&done), "the round's workers: {task}");
    assert!(task.contains("done.txt"), "the files they touched: {task}");
    assert!(
        task.contains("Full gate for this round: `cargo test --all-targets`"),
        "the gate it must run once: {task}"
    );
    assert!(
        task.contains(mini_swe_mcp::agent::CONSOLIDATOR_INSTRUCTIONS),
        "the procedure it follows: {task}"
    );
}

/// The consolidator judges scope from the whole task, so its prompt carries a
/// clearly delimited section with every worker's *full* task, bounded per
/// worker -- while the compact `round` payload the orchestrator reads keeps
/// only each task's first line.
#[test]
fn the_consolidator_task_carries_full_worker_tasks_bounded() {
    let h = Harness::new("round-full-task");
    let scoped = format!("w1-{}", unique_suffix("w"));
    let verbose = format!("w2-{}", unique_suffix("w"));
    h.worker_branch(&scoped, &[("scoped.txt", "scoped\n")]);
    h.worker_branch(&verbose, &[("verbose.txt", "verbose\n")]);

    let scoped_task =
        "Keep the header line.\nRemove nothing the task asked for.\nScope: src/lib.rs only.";
    h.row(
        &scoped,
        scoped_task,
        RegistryStatus::Completed,
        WorkerRole::Worker,
        Some(GROUP),
    );
    // One short heading and a body well past the 4 KiB per-worker budget.
    let verbose_task = format!(
        "verbose heading\n{}\nTAIL-OF-VERBOSE-TASK",
        "v".repeat(8 * 1024)
    );
    h.row(
        &verbose,
        &verbose_task,
        RegistryStatus::Completed,
        WorkerRole::Worker,
        Some(GROUP),
    );

    let manifest = manifest(&h.pool.pool, OWNER, GROUP, h.path());
    let task = manifest.task_text(Some("cargo test --all-targets"));

    assert!(
        task.contains("FULL TASKS OF THE ROUND'S WORKERS"),
        "the full-task section must be labelled and separate: {task}"
    );
    for line in scoped_task.lines() {
        assert!(
            task.contains(line),
            "the worker's whole multi-line task must reach the consolidator: {line}"
        );
    }
    assert!(
        task.contains("[truncated]"),
        "a task past the per-worker budget must be marked: {task}"
    );
    assert!(
        !task.contains("TAIL-OF-VERBOSE-TASK"),
        "the tail of an oversized task must be cut, not embedded whole"
    );
    assert!(
        !task.contains(&"v".repeat(4 * 1024)),
        "a bounded task must stay under the per-worker budget"
    );

    // The compact manifest the `round` payload and the CLI read keeps its
    // shape: first line only, no bodies.
    let compact = manifest.render();
    assert!(
        compact.contains(&format!(
            "{scoped} Completed verified=unknown task=\"Keep the header line.\""
        )),
        "the compact manifest keeps each task's first line: {compact}"
    );
    for body in [
        "Remove nothing the task asked for.",
        "Scope: src/lib.rs only.",
        "TAIL-OF-VERBOSE-TASK",
    ] {
        assert!(
            !compact.contains(body),
            "the compact manifest must not gain task bodies: {body}\n{compact}"
        );
    }
    assert!(
        !compact.contains(&"v".repeat(64))
            && !compact.contains("FULL TASKS OF THE ROUND'S WORKERS"),
        "the full-task section lives only in the task, never in the compact round text: {compact}"
    );
}

/// The full-task section is bounded overall: a round of many verbose workers
/// reports what the section budget left out instead of embedding everything.
#[test]
fn the_full_task_section_is_bounded_overall() {
    use mini_swe_mcp::pool::{RoundManifest, RoundWorker};

    let worker = |id: usize| RoundWorker {
        id: format!("w{id}"),
        state: "Completed".to_string(),
        verified: Some(true),
        task: "heading".to_string(),
        full_task: format!("heading {id}\n{}", "z".repeat(8 * 1024)),
        steers: Vec::new(),
        files: Vec::new(),
    };
    let manifest = RoundManifest {
        group: GROUP.to_string(),
        base_branch: Some("master".to_string()),
        ready: (1..=8).map(worker).collect(),
        not_ready: Vec::new(),
        interaction_points: Vec::new(),
    };
    let section = manifest.render_full_tasks();
    assert!(
        section.len() <= 16 * 1024,
        "the section budget is a hard cap, footer included: {} bytes",
        section.len()
    );
    assert!(
        section.contains("omitted"),
        "workers the overall budget left out must be counted: {section}"
    );
    assert_eq!(
        manifest.render_full_tasks(),
        section,
        "the section is stable across calls"
    );
}

/// The CLI turns `consolidate` argv into the same tool arguments the MCP path
/// sends, and refuses a missing group rather than guessing one.
#[test]
fn the_cli_parses_the_consolidate_flags() {
    let argv: Vec<String> = ["mini-swe-mcp", "consolidate", "--group", "round-1"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let args = tool_args("consolidate", &argv, true)
        .expect("consolidate is a known action")
        .expect("consolidate goes through the worker tool");
    assert_eq!(args["action"], "consolidate");
    assert_eq!(args["group"], "round-1");
    assert!(!args.contains_key("model"), "{args:?}");
    assert!(!args.contains_key("verify"), "{args:?}");
    assert!(!args.contains_key("max_turns"), "{args:?}");

    let full: Vec<String> = [
        "mini-swe-mcp",
        "consolidate",
        "--group",
        "round-1",
        "--model",
        "nerd",
        "--verify",
        "cargo test",
        "--max-turns",
        "7",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let args = tool_args("consolidate", &full, true)
        .expect("valid flags")
        .expect("consolidate goes through the worker tool");
    let expected: Map<String, Value> = json!({
        "action": "consolidate",
        "group": "round-1",
        "model": "nerd",
        "verify": "cargo test",
        "max_turns": 7,
    })
    .as_object()
    .expect("an object")
    .clone();
    assert_eq!(args, expected);

    // `-g`/`-m`/`-t` are the short spellings dispatch already accepts.
    let short: Vec<String> = [
        "mini-swe-mcp",
        "consolidate",
        "-g",
        "round-2",
        "-m",
        "ninja",
        "-t",
        "3",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let args = tool_args("consolidate", &short, true)
        .expect("valid flags")
        .expect("consolidate goes through the worker tool");
    assert_eq!(args["group"], "round-2");
    assert_eq!(args["model"], "ninja");
    assert_eq!(args["max_turns"], 3);

    let missing: Vec<String> = ["mini-swe-mcp", "consolidate"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let error = tool_args("consolidate", &missing, true)
        .expect_err("a consolidate without a group must be refused");
    assert!(
        error.to_string().contains("--group"),
        "the refusal must name the missing flag: {error}"
    );
}

/// `mini-swe-mcp help consolidate` describes the round workflow, and the tool
/// description points at it.
#[test]
fn the_consolidate_help_topic_describes_the_round_workflow() {
    let text = topic_text("consolidate").expect("consolidate is a topic");
    for needle in [
        "CHEAP gate",
        "ONE consolidator",
        "mini-swe-mcp consolidate --group",
        "FULL gate",
        "CONSOLIDATE_STEER",
        "CONSOLIDATE_WAIT",
        "merge ONLY the consolidator's branch",
    ] {
        assert!(
            text.contains(needle),
            "the topic must mention {needle}: {text}"
        );
    }

    let server = mini_swe_mcp::mcp::McpServer::new(
        mini_swe_mcp::pool::WorkerPool::new(1, "http://localhost:1".to_string(), "k".to_string()),
        "ninja".to_string(),
    );
    let tools = server.tools_list();
    let description = tools["tools"][0]["description"]
        .as_str()
        .expect("the worker tool needs a description");
    assert!(
        description.contains("consolidate"),
        "the tool description must point at the consolidate topic: {description}"
    );
    let actions = tools["tools"][0]["inputSchema"]["properties"]["action"]["enum"]
        .as_array()
        .expect("the action enum");
    assert!(
        actions.iter().any(|action| action == "consolidate"),
        "the action enum must advertise consolidate: {actions:?}"
    );
    let verify = tools["tools"][0]["inputSchema"]["properties"]["verify"]["description"]
        .as_str()
        .expect("the verify description");
    assert!(
        verify.contains("Cheap for workers"),
        "the verify description must point at the cheap worker gate: {verify}"
    );
}
