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
    MergeRequest, WorkerHistory, WorkerRegistryEntry, append_history_message_in, merge_worker_in,
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

    fn steer_source_exists(&self, id: &str) -> bool {
        self.scratch
            .path()
            .join(format!("swe-wt-{id}.steer-source"))
            .exists()
    }

    fn row_exists(&self, id: &str) -> bool {
        mini_swe_mcp::pool::load_registry_entry_in(&self.root(), id).is_some()
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

    /// One sweep pass, with the workers a `--no-delete` merge protected.
    fn sweep(&self) -> mini_swe_mcp::pool::RetireSweep {
        self.sweep_exempting(&[])
    }

    fn sweep_exempting(&self, exempt: &[String]) -> mini_swe_mcp::pool::RetireSweep {
        sweep_retired_workers_in(&self.root(), None, exempt)
    }
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
    // The consolidator integrates both, so its branch carries their commits.
    f.commit_on_worker_branch("cons", "cons.txt", "consolidated\n");
    git(
        f.repo(),
        &["merge", "--no-ff", "-m", "integrate a", "worker-c-a"],
    );
    git(
        f.repo(),
        &["merge", "--no-ff", "-m", "integrate b", "worker-c-b"],
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

    // 6. A history the sweep cannot place at all (unreadable first line): its
    //    unreachability is unproven, so it must survive.
    write(f.scratch.path(), "swe-wt-k2.history.jsonl", "not json at all\n");

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
        f.history_exists("k2"),
        "a history whose repository cannot be read must be kept: unreachability is unproven"
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
    git(f.repo(), &["merge", "--no-ff", "-m", "integrate", "worker-r-ok"]);
    git(f.repo(), &["merge", "--no-ff", "-m", "integrate", "worker-r-new"]);
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

    f.merge("rcons").expect("the consolidator merge must succeed");

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
        started_at: 0,
        pid: std::process::id(),
        revision: 0,
        auto_continues: 0,
        metrics: Default::default(),
        report: None,
        verified: None,
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
