//! Integration tests for `mini-swe-mcp merge <id>`.
//!
//! Every test builds a real throwaway repository, a real `worker-<id>` branch
//! and a real saved conversation, then merges through
//! [`mini_swe_mcp::pool::merge_worker_in`] exactly as the MCP handler does.
//! Nothing here touches the developer's repository, scratch root or registry:
//! each test owns its own [`TempDir`] and passes it as the scratch root.

mod common;

use common::{TempDir, git, git_ref_exists};
use mini_swe_mcp::agent::{ChatMessage, Role};
use mini_swe_mcp::pool::RegistryStatus;
use mini_swe_mcp::pool::{
    MergeRequest, WorkerHistory, WorkerRegistryEntry, append_history_message_in, merge_worker_in,
    save_registry_entry_in,
};
use mini_swe_mcp::worktree::ScratchRoot;
use std::path::{Path, PathBuf};

/// A repository with a base branch and, optionally, a worker branch off it.
struct Fixture {
    repo: TempDir,
    scratch: TempDir,
}

impl Fixture {
    /// A fresh repository on `main` with one committed file, plus a scratch
    /// root no other test can see.
    fn new(tag: &str) -> Self {
        let repo = TempDir::new_in_tmp(tag);
        git(repo.path(), &["init", "--initial-branch=main"]);
        git(repo.path(), &["config", "user.email", "merge@test"]);
        git(repo.path(), &["config", "user.name", "merge test"]);
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

    /// Record the worker's conversation: the merge reads the base branch, the
    /// branch and the gate from it, exactly as a revision does.
    fn record_worker(&self, id: &str, verify: Option<&str>) {
        let history = WorkerHistory {
            task: format!("do the {id} work"),
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
            // The loader requires a replayable conversation, which starts with
            // the system prompt.
            messages: vec![ChatMessage::text(Role::System, "you are a worker")],
        };
        append_history_message_in(
            &self.root(),
            id,
            &history,
            &ChatMessage::text(Role::System, "you are a worker"),
        )
        .expect("history log must be writable");
    }

    /// Write the registry row the merge's "still running" check reads.
    fn record_status(&self, id: &str, status: RegistryStatus) {
        let entry = WorkerRegistryEntry {
            id: id.to_string(),
            pid: std::process::id(),
            task: format!("do the {id} work"),
            model: "test".to_string(),
            status,
            step: 1,
            max_turns: 10,
            last_command: String::new(),
            question: None,
            started_at: 0,
            updated_at: 0,
            group: None,
            repo_path: Some(self.repo().to_string_lossy().into_owned()),
            owner: None,
            metrics: Default::default(),
            base_branch: Some("main".to_string()),
            base_commit: None,
            revision: 0,
            auto_continues: 0,
        };
        save_registry_entry_in(&self.root(), &entry);
    }

    /// Commit `path` on `worker-<id>`, leaving `main` checked out again.
    fn commit_on_worker_branch(&self, id: &str, path: &str, contents: &str) {
        let branch = format!("worker-{id}");
        git(self.repo(), &["checkout", "-q", "-b", &branch]);
        write(self.repo(), path, contents);
        git(self.repo(), &["add", "."]);
        git(self.repo(), &["commit", "-m", &format!("worker {id}")]);
        git(self.repo(), &["checkout", "-q", "main"]);
    }

    /// Commit `path` on `main`, i.e. move the base past the worker's branch.
    fn commit_on_base(&self, path: &str, contents: &str) {
        write(self.repo(), path, contents);
        git(self.repo(), &["add", "."]);
        git(self.repo(), &["commit", "-m", "base moves on"]);
    }

    fn merge(&self, id: &str, verified: Option<bool>, keep_branch: bool) -> anyhow::Result<()> {
        merge_worker_in(
            &self.root(),
            &MergeRequest {
                worker_id: id,
                verified,
                keep_branch,
            },
        )
        .map(|_| ())
    }

    fn history_exists(&self, id: &str) -> bool {
        self.scratch
            .path()
            .join(format!("swe-wt-{id}.history.jsonl"))
            .exists()
    }
}

fn write(dir: &Path, name: &str, contents: &str) {
    std::fs::write(dir.join(name), contents).expect("fixture file must be writable");
}

/// A clean merge whose branch already contains the base tip skips the gate.
///
/// The gate here is `exit 1`: if it ran at all the merge would be refused, so a
/// successful merge *is* the proof that it was skipped.
#[test]
fn clean_merge_skips_the_gate_when_the_branch_is_already_verified() {
    let f = Fixture::new("merge-skip");
    f.commit_on_worker_branch("w1", "worker.txt", "from the worker\n");
    f.record_worker("w1", Some("exit 1"));

    f.merge("w1", Some(true), false)
        .expect("clean merge must succeed");

    // The merge commit is on main and carries the worker's credit.
    let subjects = git(f.repo(), &["log", "--format=%s", "-n", "1"]);
    assert_eq!(subjects.trim(), "do the w1 work (worker w1)");
    // The worker's file landed on the base branch.
    assert_eq!(
        std::fs::read_to_string(f.repo().join("worker.txt")).unwrap(),
        "from the worker\n"
    );
    // Three commits reachable from main: the base, the worker's own commit and
    // the merge (--no-ff, never a fast-forward).
    let count = git(f.repo(), &["rev-list", "--count", "main"])
        .trim()
        .to_string();
    assert_eq!(count, "3");
    // The merge is a real merge commit, so the worker's history stays visible.
    let parents = git(f.repo(), &["log", "--format=%P", "-n", "1"])
        .split_whitespace()
        .count();
    assert_eq!(parents, 2, "merge commit must have two parents");
}

/// A branch the base has moved past re-runs the gate, on the merge result.
///
/// The gate asserts that *both* sides' files are present, which is only true of
/// the merge result -- never of the branch tip alone.
#[test]
fn stale_branch_runs_the_gate_on_the_merge_result() {
    let f = Fixture::new("merge-stale");
    f.commit_on_worker_branch("w2", "worker.txt", "from the worker\n");
    f.commit_on_base("base.txt", "from the base\n");
    f.record_worker("w2", Some("test -f worker.txt && test -f base.txt"));

    f.merge("w2", Some(true), false)
        .expect("gate must pass on the merge result");

    assert_eq!(
        std::fs::read_to_string(f.repo().join("base.txt")).unwrap(),
        "from the base\n"
    );
    assert_eq!(
        std::fs::read_to_string(f.repo().join("worker.txt")).unwrap(),
        "from the worker\n"
    );
    // The gate's throwaway worktree is gone, and it never lived in the
    // operator's checkout.
    let leftovers: Vec<String> = std::fs::read_dir(f.scratch.path())
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|name| name.starts_with("swe-merge-"))
        .collect();
    assert!(
        leftovers.is_empty(),
        "the gate worktree must be reclaimed: {leftovers:?}"
    );
    let registered = git(f.repo(), &["worktree", "list", "--porcelain"]);
    assert!(
        !registered.contains("swe-merge-"),
        "the gate worktree must be unregistered: {registered}"
    );
}

/// A failing gate refuses the merge and leaves the repository untouched.
#[test]
fn failing_gate_refuses_and_reports_the_tail() {
    let f = Fixture::new("merge-gate-fail");
    f.commit_on_worker_branch("w3", "worker.txt", "from the worker\n");
    f.commit_on_base("base.txt", "from the base\n");
    f.record_worker(
        "w3",
        Some("echo 'first line of noise'; echo 'the assertion that failed' >&2; exit 3"),
    );

    let err = f
        .merge("w3", None, false)
        .expect_err("a failing gate must refuse");
    let msg = err.to_string();
    assert!(
        msg.contains("the assertion that failed"),
        "the refusal must carry the failing tail: {msg}"
    );
    assert!(
        msg.contains("exit 3"),
        "the refusal must name the exit: {msg}"
    );
    // Nothing was merged and the branch is still there.
    assert!(git_ref_exists(f.repo(), "worker-w3"));
    assert_eq!(
        git(f.repo(), &["rev-list", "--count", "main"]).trim(),
        "2",
        "a refused merge must not add a commit"
    );
    assert!(f.history_exists("w3"), "a refused merge keeps the history");
}

/// A conflicting branch is refused with the steer hint that fixes it.
#[test]
fn conflict_is_refused_with_the_steer_hint() {
    let f = Fixture::new("merge-conflict");
    f.commit_on_worker_branch("w4", "shared.txt", "worker side\n");
    f.commit_on_base("shared.txt", "base side\n");
    f.record_worker("w4", None);

    let err = f
        .merge("w4", None, false)
        .expect_err("a conflict must refuse");
    let msg = err.to_string();
    assert!(
        msg.contains("shared.txt"),
        "the conflict must be named: {msg}"
    );
    assert!(
        msg.contains("steer w4 \"merge conflicts in shared.txt\""),
        "the refusal must suggest steering the worker: {msg}"
    );
    // The trial merge touched nothing.
    assert_eq!(
        std::fs::read_to_string(f.repo().join("shared.txt")).unwrap(),
        "base side\n"
    );
    assert_eq!(
        git(f.repo(), &["status", "--porcelain"]).trim(),
        "",
        "a refused merge leaves a clean working tree"
    );
    assert!(git_ref_exists(f.repo(), "worker-w4"));
}

/// An uncommitted change in a file the merge would touch refuses the merge,
/// while an unrelated untracked file is left alone.
#[test]
fn dirty_touched_file_refuses_but_unrelated_files_are_ignored() {
    let f = Fixture::new("merge-dirty");
    f.commit_on_worker_branch("w5", "shared.txt", "worker side\n");
    f.commit_on_base("base.txt", "from the base\n");
    f.record_worker("w5", None);
    // Unrelated local state the merge must never touch.
    write(f.repo(), "notes.txt", "the operator's own scratch\n");
    write(f.repo(), "shared.txt", "half-written local edit\n");

    let err = f
        .merge("w5", None, false)
        .expect_err("a dirty touched file must refuse");
    assert!(
        msg_names(&err, "shared.txt"),
        "the refusal must name the touched file: {}",
        err
    );
    assert!(
        !msg_names(&err, "notes.txt"),
        "unrelated files are not a reason to refuse: {}",
        err
    );
    // Neither the operator's edit nor their untracked file was disturbed.
    assert_eq!(
        std::fs::read_to_string(f.repo().join("shared.txt")).unwrap(),
        "half-written local edit\n"
    );
    assert_eq!(
        std::fs::read_to_string(f.repo().join("notes.txt")).unwrap(),
        "the operator's own scratch\n"
    );
    assert!(git_ref_exists(f.repo(), "worker-w5"));
}

/// A repository with another branch checked out is refused, and HEAD is not
/// moved for the operator.
#[test]
fn wrong_checked_out_branch_refuses() {
    let f = Fixture::new("merge-wrong-branch");
    f.commit_on_worker_branch("w6", "worker.txt", "from the worker\n");
    git(f.repo(), &["checkout", "-q", "-b", "somewhere-else"]);
    f.record_worker("w6", None);

    let err = f
        .merge("w6", None, false)
        .expect_err("merging into an unchecked-out base must refuse");
    assert!(
        msg_names(&err, "main"),
        "the refusal must name the base branch: {}",
        err
    );
    assert!(
        msg_names(&err, "somewhere-else"),
        "the refusal must name what is checked out: {}",
        err
    );
    let head = git(f.repo(), &["symbolic-ref", "--short", "HEAD"])
        .trim()
        .to_string();
    assert_eq!(head, "somewhere-else", "a refused merge must not move HEAD");
    assert!(git_ref_exists(f.repo(), "worker-w6"));
}

/// A live worker is refused: its branch is still being written to.
#[test]
fn running_worker_refuses() {
    let f = Fixture::new("merge-running");
    f.commit_on_worker_branch("w7", "worker.txt", "from the worker\n");
    f.record_worker("w7", None);
    f.record_status("w7", RegistryStatus::Running);

    let err = f
        .merge("w7", None, false)
        .expect_err("a running worker must refuse");
    assert!(msg_names(&err, "still Running"), "{}", err);
    assert!(git_ref_exists(f.repo(), "worker-w7"));
}

/// After a merge the branch, the history file and the worktree leftovers are
/// gone; `--no-delete` keeps the branch.
#[test]
fn merge_cleans_up_and_no_delete_keeps_the_branch() {
    let f = Fixture::new("merge-cleanup");
    f.commit_on_worker_branch("w8", "worker.txt", "from the worker\n");
    f.record_worker("w8", None);
    f.record_status("w8", RegistryStatus::Completed);
    // A leftover worktree directory, as a crashed run would leave behind.
    let leftover: PathBuf = f.scratch.path().join("swe-wt-w8");
    std::fs::create_dir_all(leftover.join("nested")).unwrap();
    std::fs::write(leftover.join("nested").join("junk.txt"), "junk\n").unwrap();
    std::fs::create_dir_all(f.scratch.path().join("swe-target-swe-wt-w8")).unwrap();
    std::fs::write(
        f.scratch
            .path()
            .join("swe-target-swe-wt-w8")
            .join("build.o"),
        "junk\n",
    )
    .unwrap();

    f.merge("w8", None, false)
        .expect("clean merge must succeed");

    assert!(
        !git_ref_exists(f.repo(), "worker-w8"),
        "the merged branch must be deleted"
    );
    assert!(!f.history_exists("w8"), "the history file must be removed");
    assert!(
        !leftover.exists(),
        "the leftover worktree must be reclaimed"
    );
    assert!(
        !f.scratch.path().join("swe-target-swe-wt-w8").exists(),
        "the leftover target dir must be reclaimed"
    );
    // The registry row survives so `status` and `collect` still answer.
    assert!(
        mini_swe_mcp::pool::load_registry_entry_in(&f.root(), "w8").is_some(),
        "the registry row outlives the merge"
    );

    // `--no-delete` keeps the branch and everything else still merges.
    let g = Fixture::new("merge-no-delete");
    g.commit_on_worker_branch("w9", "worker.txt", "from the worker\n");
    g.record_worker("w9", None);
    g.merge("w9", Some(true), true)
        .expect("a --no-delete merge must succeed");
    assert!(
        git_ref_exists(g.repo(), "worker-w9"),
        "--no-delete must keep the branch"
    );
    assert!(!g.history_exists("w9"), "the history file is still removed");
}

/// A merge never pushes: the only refs it writes are the local merge commit and
/// the branch deletion.
#[test]
fn merge_never_pushes() {
    let f = Fixture::new("merge-no-push");
    f.commit_on_worker_branch("w10", "worker.txt", "from the worker\n");
    f.record_worker("w10", None);

    f.merge("w10", Some(true), false)
        .expect("clean merge must succeed");

    // No remote was ever configured, so a push would have failed the merge;
    // the successful merge plus the absence of a remote proves neither was
    // attempted.
    let remotes = git(f.repo(), &["remote"]).trim().to_string();
    assert!(remotes.is_empty(), "the merge must not configure a remote");
    let reflog = git(f.repo(), &["reflog", "--format=%gs", "-n", "20"]);
    assert!(
        !reflog.contains("push"),
        "the merge must not push: {reflog}"
    );
}

/// A leftover worktree that is still *registered* on the worker branch must not
/// block the branch deletion: the sweep runs before `git branch -D`.
#[test]
fn a_registered_leftover_worktree_does_not_block_the_branch_deletion() {
    let f = Fixture::new("merge-registered-leftover");
    f.commit_on_worker_branch("w11", "worker.txt", "from the worker\n");
    f.record_worker("w11", None);
    f.record_status("w11", RegistryStatus::Completed);
    let registered = f.scratch.path().join("swe-wt-w11");
    git(
        f.repo(),
        &[
            "worktree",
            "add",
            &registered.to_string_lossy(),
            "worker-w11",
        ],
    );
    assert!(
        common::worktree_is_registered(f.repo(), &registered),
        "the fixture must leave a registered worktree behind"
    );

    f.merge("w11", Some(true), false)
        .expect("a registered leftover must not block the merge");

    assert!(
        !git_ref_exists(f.repo(), "worker-w11"),
        "the branch must be deleted even with a leftover worktree"
    );
    assert!(
        !registered.exists(),
        "the leftover worktree must be reclaimed"
    );
    assert!(
        !common::worktree_is_registered(f.repo(), &registered),
        "the leftover worktree must be unregistered"
    );
}

fn msg_names(err: &anyhow::Error, needle: &str) -> bool {
    err.to_string().contains(needle)
}
