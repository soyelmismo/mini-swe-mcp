//! The consolidator judges each worker's scope against the task it was
//! dispatched with *as the orchestrator later amended it*.
//!
//! The regression these guard: the orchestrator steered worker `44e2eb23` with a
//! user-approved scope change ("do NOT add `triggers`"), but the round
//! consolidator only ever saw the FULL ORIGINAL task, which asked for
//! `triggers` -- so it judged the worker's omission an out-of-scope revert and
//! silently undid the user's decision. The amendment must reach the
//! consolidator, in order, after the task it amends, bounded, and the
//! consolidator's own steers must stay out of it.

mod common;

use common::{IsolatedPool, TempDir, git, unique_suffix};
use mini_swe_mcp::pool::{
    LogBuffer, RegistryStatus, WorkerMeta, WorkerRecord, WorkerRegistryEntry, WorkerRole,
    WorkerState, save_registry_entry_in,
};
use std::path::Path;

const OWNER: &str = "agent-a";
const GROUP: &str = "round-amend";

/// A temporary repository with one baseline commit, plus an isolated pool.
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

    /// File a completed registry row for `worker_id` in the group, with a
    /// `worker-<id>` branch carrying one commit: the shape of a finished worker
    /// the round has to integrate.
    fn row(&self, worker_id: &str, task: &str) {
        git(self.repo.path(), &["checkout", "-q", "master"]);
        git(
            self.repo.path(),
            &["checkout", "-q", "-b", &format!("worker-{worker_id}")],
        );
        std::fs::write(self.repo.path().join(format!("{worker_id}.txt")), "done\n").unwrap();
        git(self.repo.path(), &["add", "."]);
        git(
            self.repo.path(),
            &[
                "-c",
                "user.name=w",
                "-c",
                "user.email=w@x",
                "commit",
                "-m",
                "work",
            ],
        );
        git(self.repo.path(), &["checkout", "-q", "master"]);

        let entry = WorkerRegistryEntry {
            task: task.to_string(),
            status: RegistryStatus::Completed,
            step: 3,
            last_command: "completed".to_string(),
            group: Some(GROUP.to_string()),
            role: WorkerRole::Worker,
            repo_path: Some(self.repo.path().to_string_lossy().to_string()),
            base_branch: Some("master".to_string()),
            ..WorkerRegistryEntry::test_row(worker_id, OWNER)
        };
        save_registry_entry_in(&self.pool.root(), &entry);
    }

    /// A running worker this process holds, so `steer` takes the in-memory
    /// delivery path and succeeds without relaunching a loop.
    async fn insert_running(&self, worker_id: &str) {
        self.pool
            .pool
            .__test_insert_worker(WorkerRecord {
                id: worker_id.to_string(),
                task: "t".into(),
                model: "m".into(),
                owner: OWNER.to_string(),
                state: WorkerState::Running {
                    step: 1,
                    last_command: "ls".into(),
                    started_at: 0,
                },
                metrics: Default::default(),
                logs: LogBuffer::new(),
                pending_steer: Vec::new(),
                resume_tx: None,
                handle: None,
                revision: 0,
            })
            .await;
    }

    /// The round manifest this group consolidates into.
    fn manifest(&self) -> mini_swe_mcp::pool::RoundManifest {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime")
            .block_on(
                self.pool
                    .pool
                    .round_manifest(OWNER, GROUP, self.repo.path()),
            )
    }

    fn path(&self) -> &Path {
        self.repo.path()
    }
}

/// The user-approved scope change reaches the consolidator, after the original
/// task it amends and labelled as superseding it. This is the regression: with
/// only the original task, the consolidator read "add triggers" and reverted
/// the user's "do NOT add `triggers`".
#[tokio::test]
async fn a_orchestrator_steer_reaches_the_consolidator_after_the_task_it_amends() {
    let h = Harness::new("consolidate-scope-amend");
    let worker = format!("44e2eb23-{}", unique_suffix("w"));
    let task = "Add `triggers` to the sync path and wire them into the schema.";
    h.row(&worker, task);
    h.insert_running(&worker).await;

    h.pool
        .pool
        .steer(&worker, "user approved a scope change: do NOT add `triggers`".into())
        .await
        .unwrap();

    let text = h.manifest().task_text(Some("cargo test"));

    let amendment = "ORCHESTRATOR STEERS AFTER DISPATCH";
    assert!(
        text.contains(amendment),
        "the amendment must be labelled as a steer, not shown as task text: {text}"
    );
    assert!(
        text.contains("do NOT add `triggers`"),
        "the orchestrator's own words must reach the consolidator verbatim: {text}"
    );
    // After the task: an amendment rendered before the text it amends reads as
    // part of the original task.
    let (task_at, steer_at) = (
        text.find(task).expect("original task embedded"),
        text.find(amendment).expect("amendment embedded"),
    );
    assert!(
        task_at < steer_at,
        "the amendment must follow the task it supersedes: {text}"
    );
    assert!(
        text.contains("SUPERSEDE"),
        "the section must say the steer wins where it conflicts with the task: {text}"
    );
}

/// Steers keep the order the worker received them in: a later one can retract
/// an earlier one, so the consolidated order is the whole content.
#[tokio::test]
async fn the_steers_of_one_worker_keep_their_arrival_order() {
    let h = Harness::new("consolidate-scope-order");
    let worker = format!("w1-{}", unique_suffix("w"));
    h.row(&worker, "heading\nbody");
    h.insert_running(&worker).await;

    for message in [
        "FIRST-STEER: drop the triggers field",
        "SECOND-STEER: keep the triggers field after all",
    ] {
        h.pool
            .pool
            .steer(&worker, message.to_string())
            .await
            .unwrap();
    }

    let text = h.manifest().task_text(Some("cargo test"));
    let first = text.find("FIRST-STEER").expect("first steer embedded");
    let second = text.find("SECOND-STEER").expect("second steer embedded");
    assert!(first < second, "steers must be in arrival order: {text}");
    // The two live inside one amendment block, not as two separate sections.
    assert_eq!(
        text.matches("ORCHESTRATOR STEERS AFTER DISPATCH").count(),
        1,
        "one block per steered worker: {text}"
    );
}

/// A consolidator's own CONSOLIDATE_STEER is not an orchestrator amendment: it
/// is reviewing its own work, and showing it its past steers would let it treat
/// its own routing decisions as the scope it must judge against.
#[tokio::test]
async fn a_consolidators_own_steer_is_not_listed_as_a_scope_amendment() {
    let h = Harness::new("consolidate-own-steer");
    let worker = format!("w1-{}", unique_suffix("w"));
    h.row(&worker, "heading\nbody");
    h.insert_running(&worker).await;

    let mut actor = WorkerMeta {
        task: "integrate".into(),
        group: Some(GROUP.to_string()),
        role: WorkerRole::Consolidate,
        ..WorkerMeta::test_meta(&format!("consol-{}", unique_suffix("c")), OWNER)
    };
    actor.id = format!("consol-{}", unique_suffix("c"));
    let refusal = h
        .pool
        .pool
        .consolidate_steer(&actor, &worker, "CONSOLIDATOR-OWN-STEER: revert the field".into())
        .await;
    assert!(
        refusal.contains("refused") || refusal.contains("revision"),
        "the harness must have attempted the steer, got: {refusal}"
    );

    let text = h.manifest().task_text(Some("cargo test"));
    assert!(
        !text.contains("CONSOLIDATOR-OWN-STEER"),
        "a consolidator's own steer must not appear as a scope amendment: {text}"
    );
}

/// The amendment block is bounded: a chatty orchestrator cannot push the real
/// amendment (or the task) out of the section's 16 KiB budget.
#[tokio::test]
async fn the_amendment_block_is_bounded_per_worker() {
    use mini_swe_mcp::pool::{RoundManifest, RoundWorker};

    let steers: Vec<String> = (0..40)
        .map(|i| format!("steer {i}: {}", "s".repeat(200)))
        .collect();
    let manifest = RoundManifest {
        group: GROUP.to_string(),
        base_branch: None,
        ready: vec![RoundWorker {
            id: "w1".to_string(),
            state: "Completed".to_string(),
            verified: None,
            task: "heading".to_string(),
            full_task: "heading\nbody".to_string(),
            steers,
            files: Vec::new(),
        }],
        not_ready: Vec::new(),
        interaction_points: Vec::new(),
    };
    let section = manifest.render_full_tasks();
    let (_, block) = section
        .split_once("ORCHESTRATOR STEERS AFTER DISPATCH")
        .expect("the block is rendered");
    let block = format!("ORCHESTRATOR STEERS AFTER DISPATCH{block}");
    assert!(
        block.len() <= 2 * 1024,
        "the per-worker steer budget is 2 KiB, got {} bytes",
        block.len()
    );
    assert!(
        block.contains("[truncated]"),
        "a cut amendment block must be marked: {block}"
    );
    assert!(
        section.contains("body"),
        "the task must survive a chatty orchestrator: {section}"
    );
}
