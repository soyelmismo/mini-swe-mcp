//! Integration tests for the round workers a consolidator absorbs.
//!
//! A consolidator that steers a worker and then finishes without integrating it
//! has taken the correction over: the worker is *absorbed*, its row is recorded
//! on the consolidator's own row, `list` shows it as absorbed, and
//! `merge <consolidator>` retires it together with the consolidator -- WIP
//! branch and all. A worker outside the consolidator's round is never touched.

mod common;

use common::{IsolatedPool, TempDir, git, git_ref_exists, unique_suffix};
use mini_swe_mcp::pool::{
    MergeRequest, RegistryStatus, WorkerMeta, WorkerPool, WorkerRegistryEntry, WorkerRole,
    load_registry_entry_in, merge_worker_in, save_registry_entry_in,
};
use mini_swe_mcp::worktree::ScratchRoot;
use std::path::Path;

const OWNER: &str = "agent-a";
const GROUP: &str = "round-1";
const OTHER_GROUP: &str = "round-2";

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

    /// Commit `file` on a new branch off `master` and return its name: the
    /// shape of a stopped worker's checkpointed branch, merged or not.
    fn worker_branch(&self, worker_id: &str, file: &str, content: &str) {
        let branch = format!("worker-{worker_id}");
        git(self.path(), &["checkout", "-q", "master"]);
        git(self.path(), &["checkout", "-q", "-b", &branch]);
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
        git(self.path(), &["checkout", "-q", "master"]);
    }

    /// A stopped, not-completed worker row: the state a worker whose budget ran
    /// out mid-correction is left in.
    fn exhausted_row(&self, worker_id: &str, group: &str) {
        let entry = WorkerRegistryEntry {
            task: "fix the gate".to_string(),
            status: RegistryStatus::Exhausted,
            step: 4,
            last_command: "stopped".to_string(),
            group: Some(group.to_string()),
            role: WorkerRole::Worker,
            repo_path: Some(self.path().to_string_lossy().to_string()),
            base_branch: Some("master".to_string()),
            ..WorkerRegistryEntry::test_row(worker_id, OWNER)
        };
        save_registry_entry_in(&self.root(), &entry);
    }

    /// The `.steer-source` marker a consolidator leaves when it steers a worker.
    fn steered_by(&self, worker_id: &str, consolidator: &str) {
        std::fs::write(
            self.pool
                .scratch
                .path()
                .join(format!("swe-wt-{worker_id}.steer-source")),
            format!("{{\"consolidator\":\"{consolidator}\",\"round_base\":\"abc123\"}}\n"),
        )
        .unwrap();
    }

    /// The consolidator's own terminal row, as its completion wrote it.
    fn consolidator_row(&self, id: &str) {
        let entry = WorkerRegistryEntry {
            task: "integrate the round".to_string(),
            status: RegistryStatus::Completed,
            step: 6,
            last_command: "completed".to_string(),
            group: Some(GROUP.to_string()),
            role: WorkerRole::Consolidate,
            repo_path: Some(self.path().to_string_lossy().to_string()),
            base_branch: Some("master".to_string()),
            ..WorkerRegistryEntry::test_row(id, OWNER)
        };
        save_registry_entry_in(&self.root(), &entry);
    }

    fn consolidator_meta(&self, id: &str) -> WorkerMeta {
        WorkerMeta {
            task: "integrate the round".to_string(),
            group: Some(GROUP.to_string()),
            role: WorkerRole::Consolidate,
            repo_path: Some(self.path().to_string_lossy().to_string()),
            ..WorkerMeta::test_meta(id, OWNER)
        }
    }

    /// `merge <consolidator>` through the entry point the MCP handler uses.
    fn merge(&self, id: &str) -> mini_swe_mcp::pool::MergeReport {
        merge_worker_in(
            &self.root(),
            &MergeRequest {
                worker_id: id,
                verified: Some(true),
                keep_branch: false,
                admission: None,
            },
        )
        .expect("the consolidator's merge lands")
    }

    fn row_exists(&self, id: &str) -> bool {
        load_registry_entry_in(&self.root(), id).is_some()
    }
}

/// Drive the pool's async API from a synchronous test.
fn block_on<F: std::future::Future>(future: F) -> F::Output {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime")
        .block_on(future)
}

/// The status `list` reports for `id`.
fn listed_status(pool: &WorkerPool, id: &str) -> String {
    let workers = block_on(pool.list_workers());
    workers
        .iter()
        .find(|w| w.get("id").and_then(|v| v.as_str()) == Some(id))
        .and_then(|w| w.pointer("/state/status").and_then(|v| v.as_str()))
        .unwrap_or("<absent>")
        .to_string()
}

#[test]
fn an_exhausted_steered_worker_is_absorbed_and_retired_with_the_consolidator() {
    let h = Harness::new("absorb");
    let consolidator = format!("consol-{}", unique_suffix("c"));
    let steered = format!("w1-{}", unique_suffix("w"));

    // The worker ran out of budget mid-correction: two WIP commits on its own
    // branch, none of them an ancestor of master.
    h.worker_branch(&steered, "wip.txt", "half a correction\n");
    h.exhausted_row(&steered, GROUP);
    h.steered_by(&steered, &consolidator);

    h.worker_branch(&consolidator, "round.txt", "the consolidator's fix\n");
    h.consolidator_row(&consolidator);

    let meta = h.consolidator_meta(&consolidator);
    block_on(h.pool.pool.record_consolidator_absorbed(&meta, &[]));

    // The consolidator's row records the worker it absorbed...
    let row = load_registry_entry_in(&h.root(), &consolidator).expect("consolidator row");
    assert_eq!(row.absorbed, vec![steered.clone()], "{row:?}");
    // ...and `list` says so, instead of leaving it looking unresolved.
    assert_eq!(
        listed_status(&h.pool.pool, &steered),
        format!("absorbed by {consolidator}")
    );

    let report = h.merge(&consolidator);
    assert!(
        report.retired.contains(&steered),
        "the absorbed worker retires with the round: {:?}",
        report.retired
    );
    assert!(
        report.retired.contains(&consolidator),
        "{:?}",
        report.retired
    );
    assert!(
        !git_ref_exists(h.path(), &format!("worker-{steered}")),
        "the absorbed worker's WIP branch is discarded"
    );
    assert!(!h.row_exists(&steered), "its row goes too");
    assert!(
        !git_ref_exists(h.path(), &format!("worker-{consolidator}")),
        "the consolidator's branch is deleted by its own merge"
    );
    // The absorbed work is not in master: it was discarded, not landed.
    assert!(!std::fs::read_to_string(h.path().join("wip.txt")).is_ok());
}

#[test]
fn a_worker_the_consolidator_reported_fixed_is_absorbed_too() {
    let h = Harness::new("absorbfix");
    let consolidator = format!("consol-{}", unique_suffix("c"));
    let fixed = format!("w2-{}", unique_suffix("w"));

    h.worker_branch(&fixed, "fixed.txt", "superseded work\n");
    h.exhausted_row(&fixed, GROUP);
    h.worker_branch(&consolidator, "round.txt", "the consolidator's fix\n");
    h.consolidator_row(&consolidator);

    let meta = h.consolidator_meta(&consolidator);
    block_on(
        h.pool
            .pool
            .record_consolidator_absorbed(&meta, std::slice::from_ref(&fixed)),
    );

    let row = load_registry_entry_in(&h.root(), &consolidator).expect("consolidator row");
    assert_eq!(row.absorbed, vec![fixed.clone()], "{row:?}");

    let report = h.merge(&consolidator);
    assert!(report.retired.contains(&fixed), "{:?}", report.retired);
    assert!(!git_ref_exists(h.path(), &format!("worker-{fixed}")));
    assert!(!h.row_exists(&fixed));
}

#[test]
fn an_unrelated_exhausted_worker_in_another_group_is_not_touched() {
    let h = Harness::new("absorbother");
    let consolidator = format!("consol-{}", unique_suffix("c"));
    let unrelated = format!("u1-{}", unique_suffix("w"));

    // Another group's worker, steered by nobody in this round.
    h.worker_branch(&unrelated, "other.txt", "another round's work\n");
    h.exhausted_row(&unrelated, OTHER_GROUP);
    h.worker_branch(&consolidator, "round.txt", "the consolidator's fix\n");
    h.consolidator_row(&consolidator);

    let meta = h.consolidator_meta(&consolidator);
    // Even a report line naming it must not reach into another round.
    block_on(
        h.pool
            .pool
            .record_consolidator_absorbed(&meta, std::slice::from_ref(&unrelated)),
    );

    let row = load_registry_entry_in(&h.root(), &consolidator).expect("consolidator row");
    assert!(row.absorbed.is_empty(), "{row:?}");
    assert_eq!(listed_status(&h.pool.pool, &unrelated), "Exhausted");

    let report = h.merge(&consolidator);
    assert!(
        !report.retired.contains(&unrelated),
        "a worker outside the round is never discarded: {:?}",
        report.retired
    );
    assert!(
        git_ref_exists(h.path(), &format!("worker-{unrelated}")),
        "the unrelated worker's branch survives"
    );
    assert!(h.row_exists(&unrelated), "its row survives");
}

#[test]
fn a_steered_worker_that_is_running_again_is_not_discarded() {
    let h = Harness::new("absorblive");
    let consolidator = format!("consol-{}", unique_suffix("c"));
    let resumed = format!("w3-{}", unique_suffix("w"));

    h.worker_branch(&resumed, "live.txt", "work in flight\n");
    h.exhausted_row(&resumed, GROUP);
    h.steered_by(&resumed, &consolidator);
    // The orchestrator resumed it after the round: its branch is live work.
    let mut row = load_registry_entry_in(&h.root(), &resumed).expect("row");
    row.status = RegistryStatus::Running;
    save_registry_entry_in(&h.root(), &row);

    h.worker_branch(&consolidator, "round.txt", "the consolidator's fix\n");
    h.consolidator_row(&consolidator);

    let meta = h.consolidator_meta(&consolidator);
    block_on(h.pool.pool.record_consolidator_absorbed(&meta, &[]));

    let row = load_registry_entry_in(&h.root(), &consolidator).expect("consolidator row");
    assert!(row.absorbed.is_empty(), "a running worker is not absorbed");

    let report = h.merge(&consolidator);
    assert!(!report.retired.contains(&resumed), "{:?}", report.retired);
    assert!(git_ref_exists(h.path(), &format!("worker-{resumed}")));
    assert!(h.row_exists(&resumed));
}

#[test]
fn a_completed_worker_reported_fixed_is_never_discarded() {
    let h = Harness::new("absorbdone");
    let consolidator = format!("consol-{}", unique_suffix("c"));
    let finished = format!("w4-{}", unique_suffix("w"));

    // A completed worker whose branch was never integrated: its commits are
    // real work awaiting an approval, not superseded WIP.
    h.worker_branch(&finished, "done.txt", "finished, unmerged work\n");
    let entry = WorkerRegistryEntry {
        status: RegistryStatus::Completed,
        group: Some(GROUP.to_string()),
        role: WorkerRole::Worker,
        repo_path: Some(h.path().to_string_lossy().to_string()),
        base_branch: Some("master".to_string()),
        ..WorkerRegistryEntry::test_row(&finished, OWNER)
    };
    save_registry_entry_in(&h.root(), &entry);
    h.worker_branch(&consolidator, "round.txt", "the consolidator's fix\n");
    h.consolidator_row(&consolidator);

    let meta = h.consolidator_meta(&consolidator);
    // The report names it `fixed`, but a completed worker is not stopped: the
    // owner and group match alone must not absorb it.
    block_on(
        h.pool
            .pool
            .record_consolidator_absorbed(&meta, std::slice::from_ref(&finished)),
    );
    let row = load_registry_entry_in(&h.root(), &consolidator).expect("consolidator row");
    assert!(row.absorbed.is_empty(), "{row:?}");

    // A stale record (a row written before the guard, or another process's
    // write) must still not cost the worker its branch: the merge re-checks.
    let mut stale = load_registry_entry_in(&h.root(), &consolidator).expect("row");
    stale.absorbed.push(finished.clone());
    save_registry_entry_in(&h.root(), &stale);

    let report = h.merge(&consolidator);
    assert!(
        !report.retired.contains(&finished),
        "a completed, unintegrated worker is never retired: {:?}",
        report.retired
    );
    assert!(
        report
            .cleaned
            .iter()
            .any(|line| *line == format!("kept {finished}: not integrated")),
        "the kept worker is reported: {:?}",
        report.cleaned
    );
    assert!(
        git_ref_exists(h.path(), &format!("worker-{finished}")),
        "its branch survives"
    );
    assert!(h.row_exists(&finished), "its row survives");
    assert!(
        std::fs::read_to_string(h.path().join("done.txt")).is_err(),
        "its work is not in master"
    );
}

#[test]
fn a_completed_absorbed_worker_whose_branch_is_integrated_is_retired() {
    let h = Harness::new("absorbint");
    let consolidator = format!("consol-{}", unique_suffix("c"));
    let landed = format!("w5-{}", unique_suffix("w"));
    let unmerged = format!("w6-{}", unique_suffix("w"));

    // A completed worker whose branch is already an ancestor of master.
    // Its row records the base commit it forked from: the integration proof
    // needs it to tell a branch carrying work from one still sitting on its
    // base.
    let landed_base = git(h.path(), &["rev-parse", "HEAD"]).trim().to_string();
    h.worker_branch(&landed, "landed.txt", "already integrated\n");
    git(h.path(), &["checkout", "-q", "master"]);
    git(
        h.path(),
        &["merge", "-q", "--no-edit", &format!("worker-{landed}")],
    );
    let entry = WorkerRegistryEntry {
        status: RegistryStatus::Completed,
        group: Some(GROUP.to_string()),
        role: WorkerRole::Worker,
        repo_path: Some(h.path().to_string_lossy().to_string()),
        base_branch: Some("master".to_string()),
        base_commit: Some(landed_base),
        ..WorkerRegistryEntry::test_row(&landed, OWNER)
    };
    save_registry_entry_in(&h.root(), &entry);

    // A second completed worker whose branch is not integrated anywhere.
    h.worker_branch(&unmerged, "unmerged.txt", "still waiting\n");
    let entry = WorkerRegistryEntry {
        status: RegistryStatus::Completed,
        group: Some(GROUP.to_string()),
        role: WorkerRole::Worker,
        repo_path: Some(h.path().to_string_lossy().to_string()),
        base_branch: Some("master".to_string()),
        ..WorkerRegistryEntry::test_row(&unmerged, OWNER)
    };
    save_registry_entry_in(&h.root(), &entry);

    h.worker_branch(&consolidator, "round.txt", "the consolidator's fix\n");
    h.consolidator_row(&consolidator);

    // A stale record lists both as absorbed; the merge must tell them apart.
    let mut stale = load_registry_entry_in(&h.root(), &consolidator).expect("row");
    stale.absorbed = vec![landed.clone(), unmerged.clone()];
    save_registry_entry_in(&h.root(), &stale);

    let report = h.merge(&consolidator);
    assert!(
        report.retired.contains(&landed),
        "an already-integrated worker retires with the round: {:?}",
        report.retired
    );
    assert!(
        !report.retired.contains(&unmerged),
        "an unintegrated worker is never retired: {:?}",
        report.retired
    );
    assert!(
        report
            .cleaned
            .iter()
            .any(|line| *line == format!("kept {unmerged}: not integrated")),
        "{:?}",
        report.cleaned
    );
    assert!(
        !git_ref_exists(h.path(), &format!("worker-{landed}")),
        "the integrated worker's branch is gone with the round"
    );
    assert!(!h.row_exists(&landed));
    assert!(
        git_ref_exists(h.path(), &format!("worker-{unmerged}")),
        "the unintegrated worker's branch survives"
    );
    assert!(h.row_exists(&unmerged));
}
