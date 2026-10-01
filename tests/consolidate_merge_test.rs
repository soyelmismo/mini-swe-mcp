//! Integration tests for the consolidator's harness-mediated merges.
//!
//! A consolidator integrates its group's finished workers by echoing
//! `CONSOLIDATE_MERGE <id> ...`; the merges themselves run on the harness, on a
//! temporary repository, through the same machinery the base sync uses. These
//! tests drive [`mini_swe_mcp::pool::WorkerPool::consolidate_merge`] directly --
//! no LLM, no sandbox -- and never touch the host repository.

mod common;

use common::{IsolatedPool, TempDir, git, git_ref_exists, unique_suffix};
use mini_swe_mcp::pool::{
    RegistryStatus, WorkerMeta, WorkerMetrics, WorkerRegistryEntry, WorkerRole, WorkerPool,
    load_registry_entry_in, save_registry_entry_in,
};
use mini_swe_mcp::worktree::{ScratchRoot, WorktreeGuard};
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

    /// Commit `file` on a new branch off `master` and return its name: the
    /// shape of a finished worker's preserved branch.
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

    /// The registry row a finished worker leaves behind.
    fn completed_row(&self, worker_id: &str, owner: &str, group: Option<&str>, role: WorkerRole) {
        let entry = WorkerRegistryEntry {
            id: worker_id.to_string(),
            pid: std::process::id(),
            task: "do the work".to_string(),
            model: "test".to_string(),
            status: RegistryStatus::Completed,
            step: 3,
            max_turns: 10,
            last_command: "completed".to_string(),
            question: None,
            started_at: 0,
            updated_at: 0,
            group: group.map(str::to_string),
            role,
            repo_path: Some(self.path().to_string_lossy().to_string()),
            owner: Some(owner.to_string()),
            metrics: WorkerMetrics::default(),
            base_branch: Some("master".to_string()),
            base_commit: None,
            revision: 0,
            auto_continues: 0,
        };
        save_registry_entry_in(&self.root(), &entry);
    }

    /// The consolidator's own registry row, as its dispatch wrote it.
    fn consolidator_meta(&self, id: &str) -> WorkerMeta {
        WorkerMeta {
            id: id.to_string(),
            task: "integrate the round".to_string(),
            group: Some(GROUP.to_string()),
            role: WorkerRole::Consolidate,
            repo_path: Some(self.path().to_string_lossy().to_string()),
            owner: OWNER.to_string(),
            started_at: 0,
            pid: std::process::id(),
            revision: 0,
            auto_continues: 0,
            metrics: WorkerMetrics::default(),
        }
    }

    /// The consolidator's worktree, with one commit of its own so its branch
    /// has diverged from the workers' branches.
    fn consolidator_worktree(&self, id: &str) -> WorktreeGuard {
        let mut guard = WorktreeGuard::new_in(&self.root(), self.path(), id).expect("worktree");
        std::fs::write(guard.path.join("consolidator.txt"), "round summary\n").unwrap();
        guard.commit_changes("consolidator: round summary").ok();
        guard
    }
}

/// The one line of the observation that answers `id`.
fn line<'a>(observation: &'a str, id: &str) -> &'a str {
    observation
        .lines()
        .find(|line| line.starts_with(&format!("{id} ")))
        .unwrap_or_else(|| panic!("no line for {id} in:\n{observation}"))
}

/// Drive the pool's async merge loop from a synchronous test.
fn merge(
    pool: &WorkerPool,
    meta: &WorkerMeta,
    guard: &WorktreeGuard,
    ids: &[String],
) -> mini_swe_mcp::pool::ConsolidateMerge {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime")
        .block_on(pool.consolidate_merge(meta, guard, ids))
}

#[test]
fn two_clean_branches_merge_and_a_conflict_stops_the_run() {
    let h = Harness::new("consol");
    let consolidator = format!("consol-{}", unique_suffix("c"));
    let first = format!("w1-{}", unique_suffix("w"));
    let second = format!("w2-{}", unique_suffix("w"));
    let conflicting = format!("w3-{}", unique_suffix("w"));
    let last = format!("w4-{}", unique_suffix("w"));

    h.worker_branch(&first, "first.txt", "first worker\n");
    h.worker_branch(&second, "second.txt", "second worker\n");
    // A branch that rewrites the same line the consolidator rewrote.
    h.worker_branch(&conflicting, "README.md", "# conflicting worker\n");
    h.worker_branch(&last, "last.txt", "never reached\n");
    for id in [&first, &second, &conflicting, &last] {
        h.completed_row(id, OWNER, Some(GROUP), WorkerRole::Worker);
    }

    let meta = h.consolidator_meta(&consolidator);
    let mut guard = h.consolidator_worktree(&consolidator);
    std::fs::write(guard.path.join("README.md"), "# consolidator\n").unwrap();
    guard.commit_changes("consolidator: rewrite readme").ok();

    let result = merge(
        &h.pool.pool,
        &meta,
        &guard,
        &[first.clone(), second.clone(), conflicting.clone(), last.clone()],
    );

    assert!(
        line(&result.observation, &first).contains("merged (1 files)"),
        "{}",
        result.observation
    );
    assert!(
        line(&result.observation, &second).contains("merged (1 files)"),
        "{}",
        result.observation
    );
    let conflict = line(&result.observation, &conflicting);
    assert!(conflict.contains("conflict: README.md"), "{conflict}");
    assert!(
        conflict.contains("resolve the markers, then continue"),
        "{conflict}"
    );
    assert_eq!(
        line(&result.observation, &last),
        format!("{last} skipped (an earlier merge conflicted)")
    );
    assert!(result.integrated, "two branches were merged");

    // The conflict is left in the worktree for the model, with the markers.
    let readme = std::fs::read_to_string(guard.path.join("README.md")).unwrap();
    assert!(readme.contains("<<<<<<<"), "markers must stay: {readme}");
    assert!(
        std::fs::read_to_string(guard.path.join("first.txt")).is_ok(),
        "the clean merges landed before the conflict"
    );
    drop(guard);

    // The consolidator's branch survives: it now carries the merged work.
    assert!(git_ref_exists(h.path(), &format!("worker-{consolidator}")));
}

#[test]
fn a_branch_of_another_owner_is_refused() {
    let h = Harness::new("consolown");
    let consolidator = format!("consol-{}", unique_suffix("c"));
    let mine = format!("w1-{}", unique_suffix("w"));
    let theirs = format!("w2-{}", unique_suffix("w"));

    h.worker_branch(&mine, "mine.txt", "mine\n");
    h.worker_branch(&theirs, "theirs.txt", "theirs\n");
    h.completed_row(&mine, OWNER, Some(GROUP), WorkerRole::Worker);
    h.completed_row(&theirs, "agent-b", Some(GROUP), WorkerRole::Worker);

    let meta = h.consolidator_meta(&consolidator);
    let guard = h.consolidator_worktree(&consolidator);
    let result = merge(&h.pool.pool, &meta, &guard, &[mine.clone(), theirs.clone()]);

    assert!(
        line(&result.observation, &mine).contains("merged (1 files)"),
        "{}",
        result.observation
    );
    let refused = line(&result.observation, &theirs);
    assert!(
        refused.starts_with(&format!("{theirs} refused:")),
        "{refused}"
    );
    assert!(refused.contains("another owner"), "{refused}");
    assert!(
        !std::fs::read_to_string(guard.path.join("theirs.txt")).is_ok(),
        "a refused branch must not be merged"
    );
}

#[test]
fn a_worker_that_is_not_completed_is_refused() {
    let h = Harness::new("consolrun");
    let consolidator = format!("consol-{}", unique_suffix("c"));
    let running = format!("w1-{}", unique_suffix("w"));
    h.worker_branch(&running, "running.txt", "still working\n");
    h.completed_row(&running, OWNER, Some(GROUP), WorkerRole::Worker);
    let mut entry = load_registry_entry_in(&h.root(), &running).expect("row");
    entry.status = RegistryStatus::Running;
    save_registry_entry_in(&h.root(), &entry);

    let meta = h.consolidator_meta(&consolidator);
    let guard = h.consolidator_worktree(&consolidator);
    let result = merge(&h.pool.pool, &meta, &guard, &[running.clone()]);

    assert!(
        line(&result.observation, &running).contains("not completed"),
        "{}",
        result.observation
    );
    assert!(!result.integrated);
    assert!(
        git_ref_exists(h.path(), &format!("worker-{running}")),
        "the refused worker's branch must survive untouched"
    );
}

#[test]
fn another_consolidator_of_the_same_group_is_refused() {
    let h = Harness::new("consolc");
    let consolidator = format!("consol-{}", unique_suffix("c"));
    let other = format!("c2-{}", unique_suffix("c"));
    h.worker_branch(&other, "other.txt", "another integration\n");
    h.completed_row(&other, OWNER, Some(GROUP), WorkerRole::Consolidate);

    let meta = h.consolidator_meta(&consolidator);
    let guard = h.consolidator_worktree(&consolidator);
    let result = merge(&h.pool.pool, &meta, &guard, &[other.clone()]);

    let refused = line(&result.observation, &other);
    assert!(refused.contains("consolidator"), "{refused}");
    assert!(!result.integrated);
}
