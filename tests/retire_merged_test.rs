//! Retirement of integrated workers: `merge`, a merged consolidator and the
//! sweep (pool-E28).
//!
//! Nothing of an integrated worker is needed any more -- its commits are in the
//! base branch -- so a merge retires it completely and the sweep reclaims
//! whatever any other path left behind. Every test builds a real throwaway
//! repository and its own scratch root, and passes that root to the code under
//! test, so no test can see or remove another's registry, history or branch.

mod common;

use common::{TempDir, git, git_ref_exists};
use mini_swe_mcp::agent::{ChatMessage, Role};
use mini_swe_mcp::pool::{
    MergeRequest, WorkerHistory, WorkerMeta, WorkerPool, WorkerRecord, WorkerRegistryEntry,
    WorkerRole, WorkerState, append_history_message_in, load_registry_entry_in, merge_worker_in,
    save_registry_entry_in, sweep_retired_workers_in,
};
use mini_swe_mcp::worktree::ScratchRoot;
use std::path::{Path, PathBuf};

/// A repository on `main` plus a scratch root no other test can see.
struct Fixture {
    repo: TempDir,
    scratch: TempDir,
}

impl Fixture {
    fn new(tag: &str) -> Self {
        let repo = TempDir::new_in_tmp(tag);
        git(repo.path(), &["init", "--initial-branch=main"]);
        git(repo.path(), &["config", "user.email", "retire@test"]);
        git(repo.path(), &["config", "user.name", "retire test"]);
        write(repo.path(), "README.md", "base\n");
        git(repo.path(), &["add", "."]);
        git(repo.path(), &["commit", "-m", "base"]);
        Self {
            repo,
            scratch: TempDir::new_in_tmp(&format!("{tag}-scratch")),
        }
    }

    fn root(&self) -> ScratchRoot {
        ScratchRoot::new(self.scratch.path())
    }

    fn repo(&self) -> &Path {
        self.repo.path()
    }

    /// Commit `path` on `worker-<id>` and leave `main` checked out again.
    fn commit_on_worker_branch(&self, id: &str, path: &str, contents: &str) {
        let branch = format!("worker-{id}");
        git(self.repo(), &["checkout", "-q", "-b", &branch]);
        write(self.repo(), path, contents);
        git(self.repo(), &["add", "."]);
        git(self.repo(), &["commit", "-m", &format!("worker {id}")]);
        git(self.repo(), &["checkout", "-q", "main"]);
    }

    /// Record the worker's conversation and registry row.
    ///
    /// `verify` is the gate command the merge would otherwise refuse to run
    /// without; `None` means the caller passes a known-verified merge.
    fn record(&self, id: &str) {
        self.record_with_verify(id, None);
    }

    fn record_with_verify(&self, id: &str, verify: Option<&str>) {
        let history = WorkerHistory {
            task: format!("do the {id} work"),
            role: Default::default(),
            group: None,
            model: "test".to_string(),
            temperature: None,
            repo_path: self.repo().to_string_lossy().into_owned(),
            base_commit: git(self.repo(), &["rev-parse", "HEAD"]).trim().to_string(),
            base_branch: Some("main".to_string()),
            branch: format!("worker-{id}"),
            network_offline: false,
            verify: verify.map(str::to_string),
            client_env: Vec::new(),
            max_turns: 10,
            review_after: None,
            revision: 0,
            auto_continues: 0,
            owner: None,
            messages: vec![ChatMessage::text(Role::System, "you are a worker")],
        };
        append_history_message_in(
            &self.root(),
            id,
            &history,
            &ChatMessage::text(Role::System, "you are a worker"),
        )
        .expect("history log must be writable");
        let row = WorkerRegistryEntry {
            task: format!("do the {id} work"),
            status: mini_swe_mcp::pool::RegistryStatus::Completed,
            step: 1,
            repo_path: Some(self.repo().to_string_lossy().into_owned()),
            base_branch: Some("main".to_string()),
            ..WorkerRegistryEntry::test_row(id, "")
        };
        save_registry_entry_in(&self.root(), &row);
    }

    /// Record a registry row that names no base branch, the way a build from
    /// before base-branch tracking wrote one. The saved conversation is then the
    /// row's only remaining evidence of which branch it was based on.
    fn record_row_without_base_branch(&self, id: &str) {
        self.record(id);
        let mut row = load_registry_entry_in(&self.root(), id).expect("the row was just written");
        row.base_branch = None;
        row.base_commit = None;
        save_registry_entry_in(&self.root(), &row);
    }

    /// Write a `.steer-source` marker, as a consolidator that steered this
    /// worker leaves behind.
    fn write_steer_source(&self, id: &str) {
        write(
            self.scratch.path(),
            &format!("swe-wt-{id}.steer-source"),
            "{\"consolidator\":\"c1\",\"base\":\"abc123\"}\n",
        );
    }

    fn history_exists(&self, id: &str) -> bool {
        self.scratch
            .path()
            .join(format!("swe-wt-{id}.history.jsonl"))
            .exists()
    }

    /// Path of a worker's saved conversation log, the one that carries its
    /// base branch on its metadata line.
    fn history_path(&self, id: &str) -> PathBuf {
        self.scratch
            .path()
            .join(format!("swe-wt-{id}.history.jsonl"))
    }

    fn steer_source_exists(&self, id: &str) -> bool {
        self.scratch
            .path()
            .join(format!("swe-wt-{id}.steer-source"))
            .exists()
    }

    fn row_exists(&self, id: &str) -> bool {
        mini_swe_mcp::pool::load_registry_entry_in(&self.root(), id).is_some()
    }

    /// `merge <id> --no-delete`, then the same post-merge sweep the MCP handler
    /// runs, so the operator contract is tested end to end.
    fn merge_keeping_branch_then_sweep(
        &self,
        id: &str,
    ) -> anyhow::Result<mini_swe_mcp::pool::RetireSweep> {
        merge_worker_in(
            &self.root(),
            &MergeRequest {
                worker_id: id,
                verified: Some(true),
                keep_branch: true,
                admission: None,
            },
        )?;
        // No exemption list: the retirement marked the row, and the sweep
        // skips a `keep_branch` row while its branch lives.
        Ok(self.sweep())
    }

    /// Merge `id` through the same entry point the MCP handler uses.
    fn merge(&self, id: &str) -> anyhow::Result<()> {
        merge_worker_in(
            &self.root(),
            &MergeRequest {
                worker_id: id,
                // The consolidator merges work whose branches it integrated, so
                // its own gate runs exactly like any other merge's.
                verified: Some(true),
                keep_branch: false,
                admission: None,
            },
        )
        .map(|_| ())
    }

    /// `merge <id>` returning the full report, for tests that inspect what the
    /// merge retired.
    fn merge_report(&self, id: &str) -> anyhow::Result<mini_swe_mcp::pool::MergeReport> {
        merge_worker_in(
            &self.root(),
            &MergeRequest {
                worker_id: id,
                verified: Some(true),
                keep_branch: false,
                admission: None,
            },
        )
    }

    /// One sweep pass over this fixture's scratch root.
    fn sweep(&self) -> mini_swe_mcp::pool::RetireSweep {
        sweep_retired_workers_in(&self.root(), None)
    }
}

/// A row written before base-branch tracking names no base, so the sweep has to
/// take it from the worker's own saved conversation -- otherwise a worker whose
/// branch *is* merged stays on the books forever, which is exactly what a
/// daemon-start sweep once reported as `workers=0` over a pool of merged
/// workers. The fallback is still a positive proof: nothing is retired until
/// `worker-<id>` is an ancestor of the base the worker recorded.
#[test]
fn the_sweep_retires_a_merged_worker_whose_row_names_no_base_branch() {
    let f = Fixture::new("retire-sweep-historical-base");

    // 1. The row that needs the fallback: completed, no base branch recorded,
    //    and its branch already merged into `main`.
    f.commit_on_worker_branch("nb1", "nb1.txt", "nb1\n");
    f.record_row_without_base_branch("nb1");
    git(
        f.repo(),
        &["merge", "--no-ff", "-m", "integrate nb1", "worker-nb1"],
    );
    assert!(
        load_registry_entry_in(&f.root(), "nb1").is_some_and(|row| row.base_branch.is_none()),
        "the fixture must present a row without a base branch"
    );

    // 2. A second such row whose branch is NOT merged: the fallback must not
    //    retire it.
    f.commit_on_worker_branch("nb2", "nb2.txt", "nb2\n");
    f.record_row_without_base_branch("nb2");

    // 3. A third such row with no saved conversation left to name its base:
    //    nothing positive can be proved, so it must survive even though the
    //    repository's checked-out branch is exactly the branch it merged into.
    f.commit_on_worker_branch("nb3", "nb3.txt", "nb3\n");
    f.record_row_without_base_branch("nb3");
    git(
        f.repo(),
        &["merge", "--no-ff", "-m", "integrate nb3", "worker-nb3"],
    );
    std::fs::remove_file(f.history_path("nb3")).expect("drop the conversation");

    let sweep = f.sweep();

    assert_eq!(
        sweep.workers,
        vec!["nb1".to_string()],
        "only the branch with a recorded base and a positive ancestry proof may be retired"
    );
    assert!(
        !f.row_exists("nb1"),
        "the merged worker's row must be retired"
    );
    assert!(!f.history_exists("nb1"), "its conversation must be retired");
    assert!(
        !git_ref_exists(f.repo(), "worker-nb1"),
        "its branch must be retired"
    );
    assert!(f.row_exists("nb2"), "an unmerged worker must survive");
    assert!(
        git_ref_exists(f.repo(), "worker-nb2"),
        "an unmerged branch must survive"
    );
    assert!(
        f.row_exists("nb3"),
        "a worker with no provable base must survive even though its branch merged"
    );
}

/// The base branch belongs on the row from the first write, because the sweep
/// proves integration from the row alone: a worker whose branch is already
/// merged must be retirable the moment its row exists, whichever path wrote it.
/// So `WorkerMeta::entry` -- the row every registry write of a worker rebuilds
/// from -- carries it.
#[test]
fn every_row_a_meta_writes_carries_its_base_branch() {
    let f = Fixture::new("retire-meta-base");
    let base_commit = git(f.repo(), &["rev-parse", "HEAD"]).trim().to_string();
    let meta = WorkerMeta {
        id: "mb1".to_string(),
        task: "carry the base".to_string(),
        owner: "agent-a".to_string(),
        group: Some("round-1".to_string()),
        role: WorkerRole::Worker,
        repo_path: Some(f.repo().to_string_lossy().into_owned()),
        base_branch: Some("main".to_string()),
        base_commit: Some(base_commit.clone()),
        started_at: 0,
        pid: std::process::id(),
        revision: 0,
        auto_continues: 0,
        metrics: Default::default(),
        report: None,
        verified: None,
        security_review: None,
    };
    let pool = WorkerPool::with_scratch(
        1,
        "http://localhost:1".to_string(),
        "test-key".to_string(),
        f.root(),
    );
    // No throttling: both rows are the ones a worker really writes -- the
    // dispatch's `running` row and its terminal `completed` row -- and the
    // second one is the row the sweep later reads.
    pool.__test_reset_registry_throttle("mb1");
    pool.__test_save_status(
        &meta,
        "test-model",
        mini_swe_mcp::pool::RegistryStatus::Running,
        0,
        10,
        "initializing",
        None,
    );
    pool.__test_reset_registry_throttle("mb1");
    pool.__test_save_status(
        &meta,
        "test-model",
        mini_swe_mcp::pool::RegistryStatus::Completed,
        3,
        10,
        "completed",
        None,
    );

    let row = load_registry_entry_in(&f.root(), "mb1").expect("the terminal row must be written");
    assert_eq!(
        row.base_branch.as_deref(),
        Some("main"),
        "a worker's row must name the base its diff is measured against"
    );
    assert_eq!(
        row.base_commit.as_deref(),
        Some(base_commit.as_str()),
        "and the commit it branched off"
    );
}

/// The dispatch detects the base itself, for every role: the registry row a
/// consolidator is dispatched with is exactly as unable to prove integration as
/// an ordinary worker's when it names no base branch.
#[tokio::test]
async fn a_dispatch_records_the_base_branch_on_every_role() {
    for (label, role) in [
        ("worker", WorkerRole::Worker),
        ("consolidator", WorkerRole::Consolidate),
    ] {
        let f = Fixture::new(&format!("retire-dispatch-{label}"));
        let llm = common::fake_llm::FakeLlm::spawn("echo staged", "echo staged").await;
        let pool = WorkerPool::with_scratch(
            1,
            llm.base_url().to_string(),
            "test-key".to_string(),
            f.root(),
        );
        let group = format!("retire-{label}");
        let id = pool
            .dispatch_with_role(
                "agent-a".to_string(),
                format!("dispatch the {label}"),
                "test-model".to_string(),
                None,
                f.repo().to_path_buf(),
                6,
                Some(group),
                None,
                false,
                None,
                Vec::new(),
                role,
            )
            .await
            .expect("the dispatch must succeed");

        // The row the dispatch writes before any turn runs.
        let row = load_registry_entry_in(&f.root(), &id).expect("the dispatch wrote its row");
        assert_eq!(
            row.base_branch.as_deref(),
            Some("main"),
            "a dispatched {label}'s row must name the base branch"
        );
        assert_eq!(
            row.base_commit.as_deref(),
            Some(git(f.repo(), &["rev-parse", "HEAD"]).trim()),
            "a dispatched {label}'s row must name the base commit"
        );

        // The row a completion writes, which is the one the sweep reads.
        wait_for_terminal(&pool, &id).await;
        let done = load_registry_entry_in(&f.root(), &id).expect("the completion wrote its row");
        assert!(
            matches!(done.status, mini_swe_mcp::pool::RegistryStatus::Completed),
            "the scripted worker must complete: {:?}",
            done.status
        );
        assert_eq!(
            done.base_branch.as_deref(),
            Some("main"),
            "the terminal row must still name the base branch"
        );
    }
}

/// Poll until the worker reaches a terminal state.
async fn wait_for_terminal(pool: &WorkerPool, id: &str) -> WorkerState {
    for _ in 0..600 {
        if let Some(state) = pool.get_worker_state(id).await {
            match state {
                WorkerState::Running { .. } | WorkerState::Paused { .. } => {}
                other => return other,
            }
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    panic!("worker {id} never reached a terminal state");
}

fn write(dir: &Path, name: &str, contents: &str) {
    std::fs::write(dir.join(name), contents).unwrap_or_else(|e| panic!("write {name}: {e}"));
}

/// `merge <id>` retires everything the worker owned: branch, registry row,
/// history, steer mailbox, steer-source and the worktree and target
/// directories left behind. Its commits are in the base branch, so nothing it
/// leaves can still be needed.
#[test]
fn merge_retires_everything_of_the_worker() {
    let f = Fixture::new("retire-merge");
    f.commit_on_worker_branch("w1", "worker.txt", "from the worker\n");
    f.record("w1");
    f.write_steer_source("w1");

    // The scratch companions a live worker holds: its worktree, the private
    // target dir beside it and its steering mailbox.
    let worktree: PathBuf = f.scratch.path().join("swe-wt-w1");
    std::fs::create_dir_all(worktree.join("nested")).unwrap();
    write(&worktree, "nested/junk.txt", "junk\n");
    std::fs::create_dir_all(f.scratch.path().join("swe-target-swe-wt-w1")).unwrap();
    write(f.scratch.path(), "swe-target-swe-wt-w1/build.o", "junk\n");
    write(f.scratch.path(), "swe-wt-w1.steer", "guidance\n");

    f.merge("w1").expect("a clean merge must succeed");

    assert!(
        !git_ref_exists(f.repo(), "worker-w1"),
        "the merged branch must be deleted"
    );
    assert!(!f.row_exists("w1"), "the registry row must be retired");
    assert!(!f.history_exists("w1"), "the history file must be removed");
    assert!(
        !f.steer_source_exists("w1"),
        "the steer-source marker must be removed"
    );
    assert!(
        !f.scratch.path().join("swe-wt-w1.steer").exists(),
        "the steering mailbox must be removed"
    );
    assert!(!worktree.exists(), "the worktree must be reclaimed");
    assert!(
        !f.scratch.path().join("swe-target-swe-wt-w1").exists(),
        "the target dir must be reclaimed"
    );
    assert!(
        git(f.repo(), &["log", "--oneline", "main"]).contains("worker w1"),
        "the worker's commits must be in the base branch"
    );
}

/// Merging a consolidator retires the workers it integrated: their branches are
/// in the base branch through the consolidator, so nothing of them is needed
/// either. The list comes from the consolidator's own row.
#[test]
fn merging_a_consolidator_retires_the_workers_it_integrated() {
    let f = Fixture::new("retire-consolidator");
    // Three finished workers the consolidator absorbed.
    for id in ["c-a", "c-b"] {
        f.commit_on_worker_branch(id, &format!("{id}.txt"), &format!("{id}\n"));
        f.record(id);
    }
    // The consolidator integrates both branches into ITS OWN branch: checkout
    // `worker-cons` first, or the merges would land on `main` and never exercise
    // the consolidator path this test exists for.
    f.commit_on_worker_branch("cons", "cons.txt", "consolidated\n");
    git(f.repo(), &["checkout", "-q", "worker-cons"]);
    git(
        f.repo(),
        &["merge", "--no-ff", "-m", "integrate a", "worker-c-a"],
    );
    git(
        f.repo(),
        &["merge", "--no-ff", "-m", "integrate b", "worker-c-b"],
    );
    git(f.repo(), &["checkout", "-q", "main"]);
    // Proof the fixture is realistic: the consolidator's branch exists and
    // carries the round, while `main` does not have it yet -- so the members'
    // retirement below can only come from the consolidator's round.
    assert!(
        git_ref_exists(f.repo(), "worker-cons"),
        "the consolidator's branch must carry the round"
    );
    assert!(
        !f.repo().join("c-a.txt").exists(),
        "the round must not already be in main"
    );

    // Its row records the round it integrated, and its history names a gate so
    // the merge has one to run.
    f.record_with_verify("cons", Some("true"));
    let row = WorkerRegistryEntry {
        task: "consolidate the round".to_string(),
        status: mini_swe_mcp::pool::RegistryStatus::Completed,
        step: 1,
        repo_path: Some(f.repo().to_string_lossy().into_owned()),
        base_branch: Some("main".to_string()),
        integrated: vec!["c-a".to_string(), "c-b".to_string()],
        ..WorkerRegistryEntry::test_row("cons", "")
    };
    save_registry_entry_in(&f.root(), &row);
    f.write_steer_source("cons");

    f.merge("cons")
        .expect("a clean consolidator merge must succeed");

    for id in ["cons", "c-a", "c-b"] {
        assert!(
            !f.row_exists(id),
            "{id} must be retired with the consolidator"
        );
        assert!(!f.history_exists(id), "{id}'s history must be removed");
        assert!(
            !git_ref_exists(f.repo(), &format!("worker-{id}")),
            "worker-{id}'s branch must be deleted"
        );
    }
    let log = git(f.repo(), &["log", "--oneline", "main"]);
    assert!(
        log.contains("integrate a") && log.contains("integrate b"),
        "the round's work must still be in the base branch: {log}"
    );
}

/// The sweep retires what a merge could not reach and deletes orphan files,
/// while leaving an unmerged completed worker and a vanished-unmerged branch's
/// history exactly where they were.
#[test]
fn the_sweep_retires_merged_workers_and_orphan_histories_only() {
    let f = Fixture::new("retire-sweep");

    // 1. A completed worker whose branch is merged into `main`: the sweep's
    //    main job. Its branch is deleted, which the sweep proves for itself.
    f.commit_on_worker_branch("m1", "m1.txt", "m1\n");
    f.record("m1");
    git(
        f.repo(),
        &["merge", "--no-ff", "-m", "integrate m1", "worker-m1"],
    );

    // 2. A completed worker still awaiting integration: nothing may touch it.
    f.commit_on_worker_branch("u1", "u1.txt", "u1\n");
    f.record("u1");

    // 3. A worker whose branch vanished WITHOUT being merged: its history is
    //    the only record of the work, so the retired grace period keeps it.
    f.commit_on_worker_branch("v1", "v1.txt", "v1\n");
    f.record("v1");
    git(f.repo(), &["branch", "-D", "worker-v1"]);

    // 4. Two histories with neither a row nor a branch: pure orphans. They name
    //    a real repository whose branch is genuinely gone, which is the only
    //    shape the sweep may delete (a rowless *live* branch is kept, below).
    for id in ["o1", "o2"] {
        f.commit_on_worker_branch(id, &format!("{id}.txt"), &format!("{id}\n"));
        git(f.repo(), &["branch", "-D", &format!("worker-{id}")]);
        write(
            f.scratch.path(),
            &format!("swe-wt-{id}.history.jsonl"),
            &format!(
                "{{\"repo_path\":\"{}\",\"branch\":\"worker-{id}\"}}\n",
                f.repo().display()
            ),
        );
    }

    // 5. A history whose row is gone but whose BRANCH still lives: the
    //    conversation is the only record of work that is still in git, so the
    //    sweep must keep it even though there is no row to match it.
    f.commit_on_worker_branch("k1", "k1.txt", "k1\n");
    write(
        f.scratch.path(),
        "swe-wt-k1.history.jsonl",
        &format!(
            "{{\"repo_path\":\"{}\",\"branch\":\"worker-k1\"}}\n",
            f.repo().display()
        ),
    );

    // 6. A standalone ownerless companion: no history, no row. Its branch is
    //    probed against every repository the registry names, and none has it, so
    //    it is an orphan and goes -- the whole point of the sweep.
    write(f.scratch.path(), "swe-wt-s1.steer", "guidance for nobody\n");
    write(
        f.scratch.path(),
        "swe-wt-s2.steer-source",
        "{\"consolidator\":\"gone\",\"base\":\"abc\"}\n",
    );

    // 7. A rowless companion whose branch STILL LIVES: its file must survive,
    //    because the branch makes it reachable even with no row.
    f.commit_on_worker_branch("s3", "s3.txt", "s3\n");
    write(f.scratch.path(), "swe-wt-s3.steer", "live guidance\n");
    write(
        f.scratch.path(),
        "swe-wt-s3.history.jsonl",
        &format!(
            "{{\"repo_path\":\"{}\",\"branch\":\"worker-s3\"}}\n",
            f.repo().display()
        ),
    );

    let sweep = f.sweep();

    assert_eq!(
        sweep.workers,
        vec!["m1".to_string()],
        "only the merged worker may be retired: {:?}",
        sweep.workers
    );
    assert!(
        !f.row_exists("m1") && !f.history_exists("m1"),
        "the merged worker must be retired completely"
    );
    assert!(
        !git_ref_exists(f.repo(), "worker-m1"),
        "the merged branch must be deleted"
    );
    assert!(
        !f.scratch.path().join("swe-wt-o1.history.jsonl").exists()
            && !f.scratch.path().join("swe-wt-o2.history.jsonl").exists(),
        "orphan histories with no row and no branch must be deleted"
    );
    assert!(
        f.history_exists("k1"),
        "a rowless history whose branch still exists must be kept: it is the only record of work still in git"
    );
    assert!(
        !f.scratch.path().join("swe-wt-s1.steer").exists()
            && !f.scratch.path().join("swe-wt-s2.steer-source").exists(),
        "standalone ownerless companions whose branch no known repository has must be deleted"
    );
    assert!(
        f.scratch.path().join("swe-wt-s3.steer").exists(),
        "a rowless companion whose branch still lives must be kept"
    );

    // The unmerged completed worker is untouched: still listed, still
    // steerable, branch intact.
    assert!(f.row_exists("u1"), "an unmerged worker must keep its row");
    assert!(
        f.history_exists("u1"),
        "an unmerged worker keeps its history"
    );
    assert!(
        git_ref_exists(f.repo(), "worker-u1"),
        "an unmerged worker must keep its branch"
    );

    // The vanished-unmerged branch keeps its history inside the grace period.
    assert!(
        f.history_exists("v1"),
        "a branch that vanished unmerged must keep its history within the grace period"
    );
    assert!(
        f.row_exists("v1"),
        "a branch that vanished unmerged must keep its row within the grace period"
    );
}

/// A worker revised *after* the consolidator integrated its earlier tip keeps
/// its branch: its history file records the old tip as integrated, but the
/// branch now carries commits the base does not have, and deleting it would
/// destroy that work. The other member of the same round is still provably
/// integrated and is retired.
#[test]
fn a_worker_revised_after_its_tip_was_integrated_is_not_retired() {
    let f = Fixture::new("retire-revised");
    for id in ["r-ok", "r-new"] {
        f.commit_on_worker_branch(id, &format!("{id}.txt"), &format!("{id}\n"));
        f.record(id);
    }
    // The consolidator integrates both tips.
    f.commit_on_worker_branch("rcons", "rcons.txt", "consolidated\n");
    // Integrate both members into the consolidator's own branch, so this
    // exercises the consolidator path rather than a direct merge into main.
    git(f.repo(), &["checkout", "-q", "worker-rcons"]);
    git(
        f.repo(),
        &["merge", "--no-ff", "-m", "integrate", "worker-r-ok"],
    );
    git(
        f.repo(),
        &["merge", "--no-ff", "-m", "integrate", "worker-r-new"],
    );
    git(f.repo(), &["checkout", "-q", "main"]);
    f.record_with_verify("rcons", Some("true"));
    save_registry_entry_in(
        &f.root(),
        &WorkerRegistryEntry {
            task: "consolidate".to_string(),
            status: mini_swe_mcp::pool::RegistryStatus::Completed,
            step: 1,
            repo_path: Some(f.repo().to_string_lossy().into_owned()),
            base_branch: Some("main".to_string()),
            integrated: vec!["r-ok".to_string(), "r-new".to_string()],
            ..WorkerRegistryEntry::test_row("rcons", "")
        },
    );

    // `r-new` is revised *after* the round: a new commit lands on its own
    // branch that nobody put into the consolidator's branch, so it is NOT in
    // main and must survive.
    git(f.repo(), &["checkout", "-q", "worker-r-new"]);
    write(f.repo(), "r-new.txt", "revised work\n");
    git(f.repo(), &["add", "."]);
    git(f.repo(), &["commit", "-m", "revision after integration"]);
    git(f.repo(), &["checkout", "-q", "main"]);

    f.merge("rcons")
        .expect("the consolidator merge must succeed");

    assert!(
        !f.row_exists("r-ok"),
        "a member whose whole branch is in main is retired"
    );
    assert!(
        f.row_exists("r-new") && git_ref_exists(f.repo(), "worker-r-new"),
        "a member revised after integration must keep its branch and row: its new work is not in main"
    );
}

/// A consolidator's recorded round survives the status writes that follow it.
///
/// Every registry write passes through the pool's coalescing writer, and each
/// status update rebuilds the row from the worker's metadata, which knows
/// nothing about the round. Without the merge inside the writer, the very next
/// step erases the round and the workers it names could never be retired.
///
/// Driven through the pool's own status write -- not the raw row helper -- so it
/// covers the real path a consolidator's completion takes.
#[test]
fn a_consolidators_round_survives_its_later_status_writes() {
    let f = Fixture::new("retire-round-survives");
    f.record_with_verify("wcons", Some("true"));
    save_registry_entry_in(
        &f.root(),
        &WorkerRegistryEntry {
            status: mini_swe_mcp::pool::RegistryStatus::Completed,
            step: 3,
            integrated: vec!["x1".to_string()],
            ..WorkerRegistryEntry::test_row("wcons", "agent-a")
        },
    );

    let pool = mini_swe_mcp::pool::WorkerPool::with_scratch(
        1,
        "http://localhost:1".to_string(),
        "test-key".to_string(),
        f.root(),
    );
    let meta = mini_swe_mcp::pool::WorkerMeta {
        id: "wcons".to_string(),
        task: "consolidate".to_string(),
        owner: "agent-a".to_string(),
        group: None,
        role: Default::default(),
        repo_path: Some(f.repo().to_string_lossy().into_owned()),
        base_branch: Some("main".to_string()),
        base_commit: None,
        started_at: 0,
        pid: std::process::id(),
        revision: 0,
        auto_continues: 0,
        metrics: Default::default(),
        report: None,
        verified: None,
        security_review: None,
    };
    pool.__test_reset_registry_throttle("wcons");
    pool.__test_save_status(
        &meta,
        "test",
        mini_swe_mcp::pool::RegistryStatus::Completed,
        9,
        10,
        "done",
        None,
    );

    let row = mini_swe_mcp::pool::load_registry_entry_in(&f.root(), "wcons")
        .expect("the consolidator's row must survive its completion");
    assert_eq!(
        row.integrated,
        vec!["x1".to_string()],
        "the status write must not erase the round the consolidator integrated"
    );
}

/// `merge --no-delete` is an explicit operator decision, and the sweep that runs
/// immediately after must not undo it: the branch, its row and its history stay,
/// so the worker is still a known, steerable, re-mergeable worker.
#[test]
fn a_no_delete_merge_survives_the_post_merge_sweep() {
    let f = Fixture::new("retire-no-delete");
    f.commit_on_worker_branch("nd1", "nd1.txt", "nd1\n");
    f.record_with_verify("nd1", Some("true"));
    f.write_steer_source("nd1");

    f.merge_keeping_branch_then_sweep("nd1")
        .expect("a --no-delete merge must succeed");
    // Durable, not a one-off exemption: a *later* sweep must also leave it be.
    let later = f.sweep();
    assert!(
        !later.workers.iter().any(|id| id == "nd1"),
        "a keep_branch worker must survive every later sweep too: {:?}",
        later.workers
    );

    assert!(
        git_ref_exists(f.repo(), "worker-nd1"),
        "--no-delete must keep the branch through the post-merge sweep"
    );
    assert!(
        f.row_exists("nd1"),
        "--no-delete must keep the row: the branch is what keeps the worker known"
    );
}

/// A merged worker leaves this process's live records too, not just its row.
///
/// `list_workers` reads both the registry and this process's live records, so
/// deleting only the row would keep an integrated worker visible as "Completed"
/// next to the ones still awaiting integration -- exactly the hiding problem the
/// retirement exists to remove.
#[tokio::test]
async fn a_retired_worker_leaves_the_live_list() {
    let f = Fixture::new("retire-live-list");
    f.commit_on_worker_branch("lv1", "lv1.txt", "lv1\n");
    f.record_with_verify("lv1", Some("true"));

    let pool = mini_swe_mcp::pool::WorkerPool::with_scratch(
        1,
        "http://localhost:1".to_string(),
        "test-key".to_string(),
        f.root(),
    );
    pool.__test_insert_worker(WorkerRecord {
        id: "lv1".to_string(),
        task: "do the work".to_string(),
        model: "test".to_string(),
        owner: "agent-a".to_string(),
        state: mini_swe_mcp::pool::WorkerState::Completed {
            turns: 3,
            diff: "1 file changed".to_string(),
            summary: "done".to_string(),
            completed_at: 1,
            artifacts: Vec::new(),
            branch: Some("worker-lv1".to_string()),
            verified: Some(true),
            metrics: Default::default(),
            report: None,
            revision: 0,
        },
        metrics: Default::default(),
        logs: mini_swe_mcp::pool::LogBuffer::new(),
        pending_steer: Vec::new(),
        resume_tx: None,
        handle: None,
        revision: 0,
    })
    .await;

    let listed: Vec<String> = pool
        .list_workers()
        .await
        .iter()
        .filter_map(|row| row["id"].as_str().map(str::to_string))
        .collect();
    assert!(
        listed.iter().any(|id| id == "lv1"),
        "a completed worker in this process is listed before it is retired: {listed:?}"
    );

    f.merge("lv1").expect("the merge must succeed");
    // Exactly what the merge handler and the sweep do.
    pool.forget_retired_workers(&["lv1".to_string()]).await;

    let listed: Vec<String> = pool
        .list_workers()
        .await
        .iter()
        .filter_map(|row| row["id"].as_str().map(str::to_string))
        .collect();
    assert!(
        !listed.iter().any(|id| id == "lv1"),
        "a retired worker must leave the live list: {listed:?}"
    );
    assert!(
        !f.row_exists("lv1"),
        "and must leave no registry row either"
    );
}

/// Two `CONSOLIDATE_MERGE` recordings separated by status writes both survive.
///
/// A consolidator integrates its round in several calls, and each call appends
/// to the row's list while the pool keeps writing status updates. The writer
/// must *union* what the caller supplies with what is on disk -- preferring its
/// own cache would silently drop a round another process recorded, and replacing
/// outright would drop the second call's additions.
#[test]
fn successive_round_recordings_survive_status_writes_between_them() {
    let f = Fixture::new("retire-round-union");
    let pool = mini_swe_mcp::pool::WorkerPool::with_scratch(
        1,
        "http://localhost:1".to_string(),
        "test-key".to_string(),
        f.root(),
    );
    f.record_with_verify("wunion", Some("true"));
    let meta = mini_swe_mcp::pool::WorkerMeta {
        id: "wunion".to_string(),
        task: "consolidate".to_string(),
        owner: "agent-a".to_string(),
        group: None,
        role: Default::default(),
        repo_path: Some(f.repo().to_string_lossy().into_owned()),
        base_branch: Some("main".to_string()),
        base_commit: None,
        started_at: 0,
        pid: std::process::id(),
        revision: 0,
        auto_continues: 0,
        metrics: Default::default(),
        report: None,
        verified: None,
        security_review: None,
    };
    let status = |pool: &mini_swe_mcp::pool::WorkerPool| {
        pool.__test_reset_registry_throttle("wunion");
        pool.__test_save_status(
            &meta,
            "test",
            mini_swe_mcp::pool::RegistryStatus::Completed,
            5,
            10,
            "integrating",
            None,
        );
    };

    // First merge call records x1, then a status write follows.
    save_registry_entry_in(
        &f.root(),
        &WorkerRegistryEntry {
            status: mini_swe_mcp::pool::RegistryStatus::Completed,
            step: 1,
            integrated: vec!["x1".to_string()],
            ..WorkerRegistryEntry::test_row("wunion", "agent-a")
        },
    );
    status(&pool);
    // Second merge call records x2; the status write must not have erased x1.
    save_registry_entry_in(
        &f.root(),
        &WorkerRegistryEntry {
            status: mini_swe_mcp::pool::RegistryStatus::Completed,
            step: 2,
            integrated: vec!["x1".to_string(), "x2".to_string()],
            ..WorkerRegistryEntry::test_row("wunion", "agent-a")
        },
    );
    status(&pool);

    let row = mini_swe_mcp::pool::load_registry_entry_in(&f.root(), "wunion")
        .expect("the consolidator's row must survive");
    assert_eq!(
        row.integrated,
        vec!["x1".to_string(), "x2".to_string()],
        "both recorded merges must survive every intervening status write"
    );
}

/// The real MCP `merge` path drops a retired worker's acknowledgements, in
/// memory and on disk, and a later write cannot resurrect them.
///
/// The event router may already hold the ack store in memory when a merge runs,
/// so editing the file alone would be undone by the router's next `persist`.
/// This drives the tool exactly as an orchestrator does, with the ack store
/// already loaded, and checks the entry is gone afterwards.
#[tokio::test]
async fn the_mcp_merge_path_forgets_the_retired_workers_acknowledgements() {
    use mini_swe_mcp::pool::{WorkerMetrics, WorkerState};

    let f = Fixture::new("retire-mcp-ack");
    f.commit_on_worker_branch("ac1", "ac1.txt", "ac1\n");
    f.record_with_verify("ac1", Some("true"));

    // The hub directory the event router loads its ack store from.
    let hub = f.scratch.path().join("hub");
    std::fs::create_dir_all(&hub).unwrap();
    std::fs::write(
        hub.join("watch_acks.json"),
        r#"{"agent-a":{"ac1":{"revision":1,"event":"completed"}}}"#,
    )
    .unwrap();

    let pool = mini_swe_mcp::pool::WorkerPool::with_scratch(
        1,
        "http://localhost:1".to_string(),
        "test-key".to_string(),
        f.root(),
    );
    // The worker is a live, verified completion, so the merge skips its gate.
    pool.__test_insert_worker(WorkerRecord {
        id: "ac1".to_string(),
        task: "do the work".to_string(),
        model: "test".to_string(),
        owner: "agent-a".to_string(),
        state: WorkerState::Completed {
            turns: 3,
            diff: "1 file changed".to_string(),
            summary: "done".to_string(),
            completed_at: 1,
            artifacts: Vec::new(),
            branch: Some("worker-ac1".to_string()),
            verified: Some(true),
            metrics: WorkerMetrics::default(),
            report: None,
            revision: 0,
        },
        metrics: WorkerMetrics::default(),
        logs: mini_swe_mcp::pool::LogBuffer::new(),
        pending_steer: Vec::new(),
        resume_tx: None,
        handle: None,
        revision: 0,
    })
    .await;

    let server = mini_swe_mcp::mcp::McpServer::new(pool.clone(), "agent-a".to_string());
    let events = server.start_hub_events(Some(&hub)).await;

    let mut ctx = mini_swe_mcp::mcp::ConnectionContext::stdio();
    ctx.agent_id = Some("agent-a".to_string());
    let result = server
        .execute_tool_for(
            "worker",
            serde_json::json!({"action": "merge", "worker_id": "ac1"}),
            &ctx,
        )
        .await
        .expect("the merge tool must succeed");
    assert!(
        result.get("worker_id").is_some(),
        "the merge must report the worker it landed: {result}"
    );

    let acked: serde_json::Value =
        serde_json::from_slice(&std::fs::read(hub.join("watch_acks.json")).unwrap()).unwrap();
    assert!(
        acked.get("agent-a").and_then(|a| a.get("ac1")).is_none(),
        "the retired worker's acknowledgement must be gone from the store: {acked}"
    );

    events.abort();
}

/// A failed branch deletion still retires the row, history and acknowledgements.
///
/// `git branch -D` legitimately fails when another worktree still has the branch
/// checked out, but the row and history are removed regardless. The merge
/// report must therefore name the worker as retired on the *row* result, or
/// those removals leak a live record and stale acknowledgements.
#[test]
fn a_failed_branch_deletion_still_reports_the_worker_as_retired() {
    let f = Fixture::new("retire-external-worktree");
    f.commit_on_worker_branch("ew1", "ew1.txt", "ew1\n");
    f.record_with_verify("ew1", Some("true"));

    // An external worktree pins `worker-ew1`, so its deletion must fail.
    let external = f.scratch.path().join("external-wt");
    let created = git(
        f.repo(),
        &["worktree", "add", external.to_str().unwrap(), "worker-ew1"],
    );
    assert!(
        git_ref_exists(f.repo(), "worker-ew1"),
        "the external worktree must pin the branch: {created}"
    );

    let report = f
        .merge_report("ew1")
        .expect("the merge itself must succeed");
    assert!(
        !report.branch_deleted,
        "the branch cannot be deleted while a worktree holds it: {report:?}"
    );
    assert!(
        report.retired.contains(&"ew1".to_string()),
        "the worker is still retired (row, history, acks) and must be reported: {report:?}"
    );
    assert!(
        !f.row_exists("ew1"),
        "the row is removed even when the branch deletion fails"
    );
    assert!(
        !f.history_exists("ew1"),
        "the history is removed even when the branch deletion fails"
    );
}

/// The real MCP consolidator path retires the whole round from every view: no
/// live records in `list`, and no acknowledgements in memory or on disk.
#[tokio::test]
async fn the_mcp_consolidator_merge_retires_its_round_from_every_view() {
    use mini_swe_mcp::pool::{WorkerMetrics, WorkerState};

    let f = Fixture::new("retire-mcp-round");
    for id in ["cr-a", "cr-b"] {
        f.commit_on_worker_branch(id, &format!("{id}.txt"), &format!("{id}\n"));
        f.record(id);
    }
    f.commit_on_worker_branch("crcons", "crcons.txt", "consolidated\n");
    git(f.repo(), &["checkout", "-q", "worker-crcons"]);
    git(
        f.repo(),
        &["merge", "--no-ff", "-m", "integrate a", "worker-cr-a"],
    );
    git(
        f.repo(),
        &["merge", "--no-ff", "-m", "integrate b", "worker-cr-b"],
    );
    git(f.repo(), &["checkout", "-q", "main"]);
    f.record_with_verify("crcons", Some("true"));
    save_registry_entry_in(
        &f.root(),
        &WorkerRegistryEntry {
            task: "consolidate".to_string(),
            status: mini_swe_mcp::pool::RegistryStatus::Completed,
            step: 1,
            owner: Some("agent-a".to_string()),
            repo_path: Some(f.repo().to_string_lossy().into_owned()),
            base_branch: Some("main".to_string()),
            integrated: vec!["cr-a".to_string(), "cr-b".to_string()],
            ..WorkerRegistryEntry::test_row("crcons", "agent-a")
        },
    );

    let hub = f.scratch.path().join("hub");
    std::fs::create_dir_all(&hub).unwrap();
    std::fs::write(
        hub.join("watch_acks.json"),
        r#"{"agent-a":{"crcons":{"revision":1,"event":"completed"},"cr-a":{"revision":1,"event":"completed"},"cr-b":{"revision":1,"event":"completed"}}}"#,
    )
    .unwrap();

    let pool = mini_swe_mcp::pool::WorkerPool::with_scratch(
        1,
        "http://localhost:1".to_string(),
        "test-key".to_string(),
        f.root(),
    );
    // Every worker is a live, verified completion owned by the merge caller.
    for id in ["crcons", "cr-a", "cr-b"] {
        pool.__test_insert_worker(WorkerRecord {
            id: id.to_string(),
            task: "do the work".to_string(),
            model: "test".to_string(),
            owner: "agent-a".to_string(),
            state: WorkerState::Completed {
                turns: 3,
                diff: "1 file changed".to_string(),
                summary: "done".to_string(),
                completed_at: 1,
                artifacts: Vec::new(),
                branch: Some(format!("worker-{id}")),
                verified: Some(true),
                metrics: WorkerMetrics::default(),
                report: None,
                revision: 0,
            },
            metrics: WorkerMetrics::default(),
            logs: mini_swe_mcp::pool::LogBuffer::new(),
            pending_steer: Vec::new(),
            resume_tx: None,
            handle: None,
            revision: 0,
        })
        .await;
    }

    let server = mini_swe_mcp::mcp::McpServer::new(pool.clone(), "agent-a".to_string());
    let events = server.start_hub_events(Some(&hub)).await;
    let mut ctx = mini_swe_mcp::mcp::ConnectionContext::stdio();
    ctx.agent_id = Some("agent-a".to_string());
    let result = server
        .execute_tool_for(
            "worker",
            serde_json::json!({"action": "merge", "worker_id": "crcons"}),
            &ctx,
        )
        .await
        .expect("the consolidator merge must succeed");
    assert!(result.get("worker_id").is_some(), "{result}");

    // Persisted acknowledgements: none of the round survives.
    let acked: serde_json::Value =
        serde_json::from_slice(&std::fs::read(hub.join("watch_acks.json")).unwrap()).unwrap();
    for id in ["crcons", "cr-a", "cr-b"] {
        assert!(
            acked.get("agent-a").and_then(|a| a.get(id)).is_none(),
            "{id} must not keep an acknowledgement: {acked}"
        );
    }
    // Live records: `list` no longer shows any of them.
    let listed: Vec<String> = pool
        .list_workers()
        .await
        .iter()
        .filter_map(|row| row["id"].as_str().map(str::to_string))
        .collect();
    for id in ["crcons", "cr-a", "cr-b"] {
        assert!(
            !listed.iter().any(|seen| seen == id),
            "{id} must leave the live list: {listed:?}"
        );
    }
    events.abort();
}

/// A worker whose registry row was already pruned is still reported retired, and
/// its acknowledgement is still cleared.
///
/// The race this pins: the retirement deletes the branch and only then the row,
/// and a concurrent reader's `branch_exists` probe prunes a terminal row whose
/// branch is gone. If retirement reported "nothing removed" because it found no
/// row, that worker would drop out of the merge report -- keeping its watch
/// acknowledgement and replay state forever. Retirement is idempotent, so what it
/// must report is that no row *remains*.
#[tokio::test]
async fn an_already_pruned_row_is_still_reported_retired_with_its_ack_cleared() {
    let f = Fixture::new("retire-pruned-row");
    for id in ["pr-cons", "pr-a"] {
        f.commit_on_worker_branch(id, &format!("{id}.txt"), &format!("{id}\n"));
        f.record(id);
    }
    f.commit_on_worker_branch("prcons", "prcons.txt", "consolidated\n");
    git(f.repo(), &["checkout", "-q", "worker-prcons"]);
    git(
        f.repo(),
        &["merge", "--no-ff", "-m", "integrate a", "worker-pr-a"],
    );
    git(f.repo(), &["checkout", "-q", "main"]);
    f.record_with_verify("prcons", Some("true"));
    save_registry_entry_in(
        &f.root(),
        &WorkerRegistryEntry {
            task: "consolidate".to_string(),
            status: mini_swe_mcp::pool::RegistryStatus::Completed,
            step: 1,
            owner: Some("agent-a".to_string()),
            repo_path: Some(f.repo().to_string_lossy().into_owned()),
            base_branch: Some("main".to_string()),
            integrated: vec!["pr-a".to_string()],
            ..WorkerRegistryEntry::test_row("prcons", "agent-a")
        },
    );

    let hub = f.scratch.path().join("hub");
    std::fs::create_dir_all(&hub).unwrap();
    std::fs::write(
        hub.join("watch_acks.json"),
        r#"{"agent-a":{"pr-a":{"revision":1,"event":"completed"}}}"#,
    )
    .unwrap();

    let pool = mini_swe_mcp::pool::WorkerPool::with_scratch(
        1,
        "http://localhost:1".to_string(),
        "test-key".to_string(),
        f.root(),
    );
    let server = mini_swe_mcp::mcp::McpServer::new(pool.clone(), "agent-a".to_string());
    let events = server.start_hub_events(Some(&hub)).await;

    // Simulate the concurrent prune: the round member's row disappears before
    // the merge reaches it, exactly as the registry's own branch probe would.
    assert!(f.row_exists("pr-a"));
    save_registry_entry_in(
        &f.root(),
        &WorkerRegistryEntry {
            status: mini_swe_mcp::pool::RegistryStatus::Completed,
            step: 9,
            ..WorkerRegistryEntry::test_row("pr-a", "agent-a")
        },
    );
    let row_path = f.root().join("swe-registry").join("pr-a.json");
    std::fs::remove_file(row_path).expect("simulate the concurrent row prune");
    assert!(!f.row_exists("pr-a"), "the row is gone before the merge");

    let mut ctx = mini_swe_mcp::mcp::ConnectionContext::stdio();
    ctx.agent_id = Some("agent-a".to_string());
    server
        .execute_tool_for(
            "worker",
            serde_json::json!({"action": "merge", "worker_id": "prcons"}),
            &ctx,
        )
        .await
        .expect("the consolidator merge must succeed");

    // The already-pruned worker must still have been retired, so its
    // acknowledgement goes with it.
    let acked: serde_json::Value =
        serde_json::from_slice(&std::fs::read(hub.join("watch_acks.json")).unwrap()).unwrap();
    assert!(
        acked.get("agent-a").and_then(|a| a.get("pr-a")).is_none(),
        "an already-pruned member must still lose its acknowledgement: {acked}"
    );
    events.abort();
}

/// The post-merge sweep clears an orphan worker's acknowledgement.
///
/// An orphan has no row and no branch, so the merge report can never name it and
/// the registry cannot be asked about it later: the only place its id exists is
/// the sweep's reclaimed-orphan list. If that list is not chained into the
/// retirement, the worker keeps its acknowledged position and replay state after
/// its last file has been deleted.
#[tokio::test]
async fn the_post_merge_sweep_clears_a_reclaimed_orphans_acknowledgement() {
    let f = Fixture::new("retire-orphan-ack");
    // A merged worker, so the merge has something to do and runs a real sweep.
    f.commit_on_worker_branch("oa-main", "oa.txt", "main work\n");
    f.record_with_verify("oa-main", Some("true"));
    // The merge is owner-checked, so the merged worker needs its owner.
    let mut row = mini_swe_mcp::pool::load_registry_entry_in(&f.root(), "oa-main").unwrap();
    row.owner = Some("agent-a".to_string());
    save_registry_entry_in(&f.root(), &row);

    // A second, still-listed worker in the same repository. The orphan scan
    // probes the repositories the registry names, so the repository must still
    // be known after the merge deletes its own row -- which is the normal case
    // in a pool with more than one worker.
    f.commit_on_worker_branch("oa-live", "oa-live.txt", "live work\n");
    f.record_with_verify("oa-live", Some("true"));
    save_registry_entry_in(
        &f.root(),
        &WorkerRegistryEntry {
            status: mini_swe_mcp::pool::RegistryStatus::Completed,
            step: 1,
            owner: Some("agent-a".to_string()),
            repo_path: Some(f.repo().to_string_lossy().into_owned()),
            base_branch: Some("main".to_string()),
            ..WorkerRegistryEntry::test_row("oa-live", "agent-a")
        },
    );

    // The orphan: a companion file only, no row and no branch anywhere.
    write(
        f.scratch.path(),
        "swe-wt-oa-gone.steer",
        "guidance for nobody\n",
    );

    let hub = f.scratch.path().join("hub");
    std::fs::create_dir_all(&hub).unwrap();
    std::fs::write(
        hub.join("watch_acks.json"),
        r#"{"agent-a":{"oa-gone":{"revision":1,"event":"completed"}}}"#,
    )
    .unwrap();

    let pool = mini_swe_mcp::pool::WorkerPool::with_scratch(
        1,
        "http://localhost:1".to_string(),
        "test-key".to_string(),
        f.root(),
    );
    let server = mini_swe_mcp::mcp::McpServer::new(pool.clone(), "agent-a".to_string());
    let events = server.start_hub_events(Some(&hub)).await;
    let mut ctx = mini_swe_mcp::mcp::ConnectionContext::stdio();
    ctx.agent_id = Some("agent-a".to_string());
    server
        .execute_tool_for(
            "worker",
            serde_json::json!({"action": "merge", "worker_id": "oa-main"}),
            &ctx,
        )
        .await
        .expect("the merge must succeed");

    assert!(
        !f.scratch.path().join("swe-wt-oa-gone.steer").exists(),
        "the orphan's file must be reclaimed by the post-merge sweep"
    );
    let acked: serde_json::Value =
        serde_json::from_slice(&std::fs::read(hub.join("watch_acks.json")).unwrap()).unwrap();
    assert!(
        acked
            .get("agent-a")
            .and_then(|a| a.get("oa-gone"))
            .is_none(),
        "a reclaimed orphan must lose its acknowledgement too: {acked}"
    );
    events.abort();
}
