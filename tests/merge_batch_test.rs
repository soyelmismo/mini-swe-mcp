//! Integration tests for `mini-swe-mcp merge --approved`.
//!
//! Every test builds a real throwaway repository, real `worker-<id>` branches
//! and real registry rows, then lands the round through
//! [`mini_swe_mcp::pool::merge_approved_in`] exactly as the MCP handler does.
//! Nothing here touches the developer's repository, scratch root or registry:
//! each test owns its own [`TempDir`] and passes it as the scratch root.
//!
//! The gate is a trivial shell command so a batch runs in seconds. It counts its
//! own invocations by appending to a file inside the build directory the gate
//! leases for the repository, which is the one writable place that outlives the
//! throwaway gate worktree; the file name carries the test's unique tag, so two
//! tests running in parallel can never read each other's count.

mod common;

use common::{TempDir, git, git_ref_exists};
use mini_swe_mcp::agent::{ChatMessage, Role};
use mini_swe_mcp::pool::RegistryStatus;
use mini_swe_mcp::pool::{
    MergeApprovedRequest, WorkerHistory, WorkerRegistryEntry, append_history_message_in,
    merge_approved_in, save_registry_entry_in,
};
use mini_swe_mcp::worktree::ScratchRoot;
use std::path::{Path, PathBuf};

/// A repository with a base branch and worker branches off it.
struct Fixture {
    repo: TempDir,
    scratch: TempDir,
    /// Unique tag stamped into the gate's counter file name.
    tag: String,
}

impl Fixture {
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
            tag: common::unique_suffix(tag),
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

    /// Write the registry row the batch's selection reads: completed, owned,
    /// grouped and -- when `approved` is `Some` -- carrying an approval.
    fn record_status(&self, id: &str, owner: &str, group: Option<&str>, approved: Option<u64>) {
        let entry = WorkerRegistryEntry {
            approved: approved.map(|at| mini_swe_mcp::pool::WorkerApproval { at, note: None }),
            id: id.to_string(),
            pid: std::process::id(),
            task: format!("do the {id} work"),
            role: Default::default(),
            model: "test".to_string(),
            status: RegistryStatus::Completed,
            step: 1,
            max_turns: 10,
            last_command: String::new(),
            question: None,
            started_at: 0,
            updated_at: 0,
            group: group.map(str::to_string),
            repo_path: Some(self.repo().to_string_lossy().into_owned()),
            owner: Some(owner.to_string()),
            metrics: Default::default(),
            base_branch: Some("main".to_string()),
            base_commit: None,
            revision: 0,
            auto_continues: 0,
        };
        save_registry_entry_in(&self.root(), &entry);
    }

    /// A completed, approved worker of `owner` with `verify` as its gate.
    fn approved_worker(&self, id: &str, owner: &str, verify: Option<&str>) {
        self.record_worker(id, verify);
        self.record_status(id, owner, None, Some(1));
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

    /// The gate command: one counted line per invocation, then the files the
    /// composed tree must contain. A per-worker gate would see only its own
    /// file, so a passing gate is also proof it ran on the combined result.
    fn counting_gate(&self, files: &[&str]) -> String {
        let present = files
            .iter()
            .map(|f| format!("test -f {f}"))
            .collect::<Vec<_>>()
            .join(" && ");
        format!(
            "echo run >> \"$CARGO_TARGET_DIR/gate-runs-{}\" && {present}",
            self.tag
        )
    }

    /// How many times the gate ran for this test's repository, reclaiming the
    /// counter file so no run leaves state behind.
    fn gate_runs(&self) -> usize {
        let name = format!("gate-runs-{}", self.tag);
        let mut runs = 0;
        if let Ok(entries) = std::fs::read_dir(mini_swe_mcp::worktree::swe_base_dir()) {
            for entry in entries.flatten() {
                let counter = entry.path().join(&name);
                if let Ok(text) = std::fs::read_to_string(&counter) {
                    runs += text.lines().count();
                    let _ = std::fs::remove_file(&counter);
                }
            }
        }
        runs
    }

    fn merge_approved(
        &self,
        owner: Option<&str>,
        group: Option<&str>,
    ) -> anyhow::Result<mini_swe_mcp::pool::MergeApprovedReport> {
        merge_approved_in(
            &self.root(),
            &MergeApprovedRequest {
                owner,
                group,
                admission: None,
            },
        )
    }

    fn history_exists(&self, id: &str) -> bool {
        self.scratch
            .path()
            .join(format!("swe-wt-{id}.history.jsonl"))
            .exists()
    }

    /// The subjects of the merge commits on `main`, newest first.
    fn merge_subjects(&self) -> Vec<String> {
        git(self.repo(), &["log", "--merges", "--format=%s"])
            .lines()
            .map(str::to_string)
            .collect()
    }
}

fn write(dir: &Path, name: &str, contents: &str) {
    std::fs::write(dir.join(name), contents).expect("fixture file must be writable");
}

/// Three approved workers land with ONE gate run on the combined tree.
#[test]
fn three_approved_workers_merge_with_one_gate_run() {
    let f = Fixture::new("batch-three");
    f.commit_on_worker_branch("w1", "w1.txt", "one\n");
    f.commit_on_worker_branch("w2", "w2.txt", "two\n");
    f.commit_on_worker_branch("w3", "w3.txt", "three\n");
    let gate = f.counting_gate(&["w1.txt", "w2.txt", "w3.txt"]);
    f.approved_worker("w1", "agent-a", Some(&gate));
    f.approved_worker("w2", "agent-a", Some(&gate));
    f.approved_worker("w3", "agent-a", Some(&gate));

    let report = f
        .merge_approved(Some("agent-a"), None)
        .expect("the batch must land");

    let merged: Vec<&str> = report.merged.iter().map(|m| m.worker_id.as_str()).collect();
    assert_eq!(merged, vec!["w1", "w2", "w3"], "merged: {report:?}");
    assert!(report.skipped.is_empty(), "skipped: {report:?}");
    assert_eq!(report.gate_command.as_deref(), Some(gate.as_str()));
    // The whole point of the batch: the round's gate ran exactly once.
    assert_eq!(
        f.gate_runs(),
        1,
        "the gate must run once for the whole batch"
    );

    // One merge commit per worker, in approval order, with the same subject
    // format `merge <id>` uses.
    assert_eq!(
        f.merge_subjects(),
        vec![
            "do the w3 work (worker w3)",
            "do the w2 work (worker w2)",
            "do the w1 work (worker w1)",
        ]
    );
    for id in ["w1", "w2", "w3"] {
        assert!(
            !git_ref_exists(f.repo(), &format!("worker-{id}")),
            "worker {id}'s branch must be cleaned up"
        );
        assert!(
            !f.history_exists(id),
            "worker {id}'s history file must be removed"
        );
    }
    assert!(
        !report.cleaned.is_empty(),
        "the report names what the cleanup reclaimed"
    );
}

/// A conflicting worker is skipped and reported; the rest still land.
#[test]
fn a_conflicting_worker_is_skipped_and_the_rest_merge() {
    let f = Fixture::new("batch-conflict");
    f.commit_on_worker_branch("w1", "shared.txt", "from w1\n");
    f.commit_on_worker_branch("w2", "shared.txt", "from w2\n");
    f.commit_on_worker_branch("w3", "w3.txt", "three\n");
    let gate = f.counting_gate(&["shared.txt", "w3.txt"]);
    f.approved_worker("w1", "agent-a", Some(&gate));
    f.approved_worker("w2", "agent-a", Some(&gate));
    f.approved_worker("w3", "agent-a", Some(&gate));

    let report = f
        .merge_approved(Some("agent-a"), None)
        .expect("the round must land without the conflicting worker");

    let merged: Vec<&str> = report.merged.iter().map(|m| m.worker_id.as_str()).collect();
    assert_eq!(merged, vec!["w1", "w3"], "merged: {report:?}");
    assert_eq!(report.skipped.len(), 1, "skipped: {report:?}");
    let skipped = &report.skipped[0];
    assert_eq!(skipped.worker_id, "w2");
    assert!(
        skipped.files.iter().any(|f| f == "shared.txt"),
        "the conflicting file must be named: {skipped:?}"
    );
    assert!(
        skipped.steer.contains("steer w2") && skipped.steer.contains("shared.txt"),
        "the skip carries the steer that sends the conflicts back: {skipped:?}"
    );
    // The gate still ran once, on the tree without the conflicting worker.
    assert_eq!(f.gate_runs(), 1);
    // The skipped worker keeps its branch and its conversation: it is not lost.
    assert!(git_ref_exists(f.repo(), "worker-w2"));
    assert!(f.history_exists("w2"));
}

/// A failing gate merges nothing and attributes the failing file to its worker.
#[test]
fn a_failing_gate_merges_nothing_and_attributes_the_file() {
    let f = Fixture::new("batch-gate-fail");
    f.commit_on_worker_branch("w1", "w1.txt", "one\n");
    f.commit_on_worker_branch("w2", "w2.txt", "two\n");
    // The gate names the file it failed on, the way a test runner does.
    let gate = "echo 'error: w1.txt:3: broken' >&2; exit 1".to_string();
    f.approved_worker("w1", "agent-a", Some(&gate));
    f.approved_worker("w2", "agent-a", Some(&gate));

    let err = f
        .merge_approved(Some("agent-a"), None)
        .expect_err("a failing gate must refuse the batch");

    let text = err.to_string();
    assert!(text.contains("verify gate failed"), "{text}");
    assert!(
        text.contains("w1.txt:3"),
        "the failing tail is reported: {text}"
    );
    assert!(
        text.contains("w1.txt: w1"),
        "the failing file is attributed to its worker: {text}"
    );
    assert!(
        !text.contains("w2.txt: w2"),
        "a file the failure never named is not attributed: {text}"
    );
    // Nothing landed: no merge commit, both branches and both histories intact.
    assert!(
        f.merge_subjects().is_empty(),
        "a failing gate merges nothing: {:?}",
        f.merge_subjects()
    );
    for id in ["w1", "w2"] {
        assert!(git_ref_exists(f.repo(), &format!("worker-{id}")));
        assert!(f.history_exists(id));
    }
}

/// A file several workers touched is flagged as an interaction point.
#[test]
fn a_file_several_workers_touched_is_an_interaction_point() {
    let f = Fixture::new("batch-interaction");
    // Both workers write the same bytes, so the merge is clean and both land in
    // the batch -- but both of them touched the file the gate then fails on.
    f.commit_on_worker_branch("w1", "shared.txt", "same\n");
    f.commit_on_worker_branch("w2", "shared.txt", "same\n");
    let gate = "echo 'error: shared.txt:7: broken' >&2; exit 1".to_string();
    f.approved_worker("w1", "agent-a", Some(&gate));
    f.approved_worker("w2", "agent-a", Some(&gate));

    let err = f
        .merge_approved(Some("agent-a"), None)
        .expect_err("a failing gate must refuse the batch");

    let text = err.to_string();
    assert!(
        text.contains("shared.txt: w1, w2") && text.contains("interaction point"),
        "a file several workers touched is flagged: {text}"
    );
    assert!(git_ref_exists(f.repo(), "worker-w1"), "nothing was merged");
    assert!(git_ref_exists(f.repo(), "worker-w2"), "nothing was merged");
}

/// An unapproved worker is never selected, even when it is completed.
#[test]
fn an_unapproved_worker_is_not_selected() {
    let f = Fixture::new("batch-unapproved");
    f.commit_on_worker_branch("w1", "w1.txt", "one\n");
    f.commit_on_worker_branch("w2", "w2.txt", "two\n");
    let gate = f.counting_gate(&["w1.txt"]);
    f.approved_worker("w1", "agent-a", Some(&gate));
    f.record_worker("w2", Some(&gate));
    f.record_status("w2", "agent-a", None, None);

    let report = f
        .merge_approved(Some("agent-a"), None)
        .expect("the approved worker must land");

    let merged: Vec<&str> = report.merged.iter().map(|m| m.worker_id.as_str()).collect();
    assert_eq!(merged, vec!["w1"], "merged: {report:?}");
    assert!(git_ref_exists(f.repo(), "worker-w2"), "w2 keeps its branch");
    assert!(f.history_exists("w2"), "w2 keeps its conversation");
}

/// Another owner's approved worker is never selected (H-3).
#[test]
fn another_owners_approved_worker_is_never_selected() {
    let f = Fixture::new("batch-owner");
    f.commit_on_worker_branch("w1", "w1.txt", "one\n");
    f.commit_on_worker_branch("w2", "w2.txt", "two\n");
    let gate = f.counting_gate(&["w1.txt"]);
    f.approved_worker("w1", "agent-a", Some(&gate));
    f.approved_worker("w2", "agent-b", Some(&gate));

    let report = f
        .merge_approved(Some("agent-a"), None)
        .expect("the caller's own worker must land");

    let merged: Vec<&str> = report.merged.iter().map(|m| m.worker_id.as_str()).collect();
    assert_eq!(merged, vec!["w1"], "merged: {report:?}");
    assert!(
        git_ref_exists(f.repo(), "worker-w2"),
        "another owner's branch is untouched"
    );
}

/// `--group` narrows the batch to that group's approved workers.
#[test]
fn a_group_narrows_the_batch() {
    let f = Fixture::new("batch-group");
    f.commit_on_worker_branch("w1", "w1.txt", "one\n");
    f.commit_on_worker_branch("w2", "w2.txt", "two\n");
    let gate = f.counting_gate(&["w1.txt"]);
    f.record_worker("w1", Some(&gate));
    f.record_status("w1", "agent-a", Some("round-1"), Some(1));
    f.record_worker("w2", Some(&gate));
    f.record_status("w2", "agent-a", Some("round-2"), Some(2));

    let report = f
        .merge_approved(Some("agent-a"), Some("round-1"))
        .expect("the group's worker must land");

    let merged: Vec<&str> = report.merged.iter().map(|m| m.worker_id.as_str()).collect();
    assert_eq!(merged, vec!["w1"], "merged: {report:?}");
    assert!(
        git_ref_exists(f.repo(), "worker-w2"),
        "w2 is in another group"
    );
}

/// The batch takes the same checkout refusal `merge <id>` takes: a repository
/// with another branch checked out is left alone.
#[test]
fn a_wrong_checked_out_branch_refuses_the_batch() {
    let f = Fixture::new("batch-checkout");
    f.commit_on_worker_branch("w1", "w1.txt", "one\n");
    f.approved_worker("w1", "agent-a", Some(&f.counting_gate(&["w1.txt"])));
    git(f.repo(), &["checkout", "-q", "-b", "elsewhere"]);

    let err = f
        .merge_approved(Some("agent-a"), None)
        .expect_err("a wrong checkout must refuse the batch");

    assert!(
        err.to_string().contains("has elsewhere checked out"),
        "{err}"
    );
    assert!(git_ref_exists(f.repo(), "worker-w1"), "nothing was merged");
}

/// A batch with no approved worker at all is a refusal, not an empty merge.
#[test]
fn no_approved_worker_is_a_refusal() {
    let f = Fixture::new("batch-empty");
    f.commit_on_worker_branch("w1", "w1.txt", "one\n");
    f.record_worker("w1", Some("exit 1"));
    f.record_status("w1", "agent-a", None, None);

    let err = f
        .merge_approved(Some("agent-a"), None)
        .expect_err("nothing approved means nothing to merge");

    assert!(
        err.to_string()
            .contains("no completed worker of yours carries an approval"),
        "{err}"
    );
}

/// The workers' shared verify command is the batch's gate; when they disagree
/// the project's auto-detected gate is used instead.
#[test]
fn the_shared_verify_command_is_the_batch_gate() {
    let f = Fixture::new("batch-shared-gate");
    f.commit_on_worker_branch("w1", "w1.txt", "one\n");
    f.commit_on_worker_branch("w2", "w2.txt", "two\n");
    let gate = f.counting_gate(&["w1.txt", "w2.txt"]);
    f.approved_worker("w1", "agent-a", Some(&gate));
    f.approved_worker("w2", "agent-a", Some(&gate));

    let report = f
        .merge_approved(Some("agent-a"), None)
        .expect("the shared gate must pass");

    assert_eq!(report.gate_command.as_deref(), Some(gate.as_str()));
    assert_eq!(f.gate_runs(), 1);
}

/// A batch whose workers record no gate and whose project has none is refused
/// rather than landed ungated.
#[test]
fn no_gate_anywhere_refuses_the_batch() {
    let f = Fixture::new("batch-no-gate");
    f.commit_on_worker_branch("w1", "w1.txt", "one\n");
    f.approved_worker("w1", "agent-a", None);

    let err = f
        .merge_approved(Some("agent-a"), None)
        .expect_err("a batch with no gate must be refused");

    assert!(
        err.to_string()
            .contains("record no verify command they agree on"),
        "{err}"
    );
    assert!(git_ref_exists(f.repo(), "worker-w1"), "nothing was merged");
}

/// The approval stamp orders the batch: the earliest approved worker merges
/// first, whatever the registry's own row order is.
#[test]
fn approval_order_decides_the_merge_order() {
    let f = Fixture::new("batch-order");
    f.commit_on_worker_branch("w1", "w1.txt", "one\n");
    f.commit_on_worker_branch("w2", "w2.txt", "two\n");
    let gate = f.counting_gate(&["w1.txt", "w2.txt"]);
    f.record_worker("w1", Some(&gate));
    f.record_status("w1", "agent-a", None, Some(200));
    f.record_worker("w2", Some(&gate));
    f.record_status("w2", "agent-a", None, Some(100));

    let report = f
        .merge_approved(Some("agent-a"), None)
        .expect("both workers must land");

    let merged: Vec<&str> = report.merged.iter().map(|m| m.worker_id.as_str()).collect();
    assert_eq!(merged, vec!["w2", "w1"], "merged: {report:?}");
    assert_eq!(
        f.merge_subjects(),
        vec!["do the w1 work (worker w1)", "do the w2 work (worker w2)",],
        "the earliest approved worker's merge commit is the older one"
    );
}

/// A dirty file the batch would touch refuses it, exactly as `merge <id>` does,
/// and leaves every branch where it was.
#[test]
fn a_dirty_touched_file_refuses_the_batch() {
    let f = Fixture::new("batch-dirty");
    f.commit_on_worker_branch("w1", "w1.txt", "one\n");
    f.approved_worker("w1", "agent-a", Some(&f.counting_gate(&["w1.txt"])));
    write(f.repo(), "w1.txt", "uncommitted\n");

    let err = f
        .merge_approved(Some("agent-a"), None)
        .expect_err("a dirty touched file must refuse the batch");

    assert!(
        err.to_string().contains("uncommitted change(s) in file(s)"),
        "{err}"
    );
    assert!(git_ref_exists(f.repo(), "worker-w1"), "nothing was merged");
}

/// A batch whose approved workers span two repositories is refused with a
/// message naming both, rather than silently merging one of them.
#[test]
fn workers_of_two_repositories_are_refused() {
    let f = Fixture::new("batch-two-repos");
    f.commit_on_worker_branch("w1", "w1.txt", "one\n");
    f.approved_worker("w1", "agent-a", Some(&f.counting_gate(&["w1.txt"])));
    // A second repository, with its own completed and approved worker.
    let other = TempDir::new_in_tmp("batch-two-repos-other");
    git(other.path(), &["init", "--initial-branch=main"]);
    git(other.path(), &["config", "user.email", "merge@test"]);
    git(other.path(), &["config", "user.name", "merge test"]);
    write(other.path(), "README.md", "base\n");
    git(other.path(), &["add", "."]);
    git(other.path(), &["commit", "-m", "base"]);
    git(other.path(), &["checkout", "-q", "-b", "worker-w2"]);
    write(other.path(), "w2.txt", "two\n");
    git(other.path(), &["add", "."]);
    git(other.path(), &["commit", "-m", "worker w2"]);
    git(other.path(), &["checkout", "-q", "main"]);
    let mut history = WorkerHistory {
        task: "do the w2 work".to_string(),
        role: Default::default(),
        group: None,
        model: "test".to_string(),
        temperature: None,
        repo_path: other.path().to_string_lossy().into_owned(),
        base_commit: String::new(),
        base_branch: Some("main".to_string()),
        branch: "worker-w2".to_string(),
        network_offline: false,
        verify: None,
        client_env: Vec::new(),
        max_turns: 10,
        review_after: None,
        revision: 0,
        auto_continues: 0,
        owner: None,
        messages: vec![ChatMessage::text(Role::System, "you are a worker")],
    };
    append_history_message_in(
        &f.root(),
        "w2",
        &history,
        &ChatMessage::text(Role::System, "you are a worker"),
    )
    .expect("history log must be writable");
    history.repo_path = other.path().to_string_lossy().into_owned();
    let entry = WorkerRegistryEntry {
        approved: Some(mini_swe_mcp::pool::WorkerApproval { at: 1, note: None }),
        id: "w2".to_string(),
        pid: std::process::id(),
        task: "do the w2 work".to_string(),
        role: Default::default(),
        model: "test".to_string(),
        status: RegistryStatus::Completed,
        step: 1,
        max_turns: 10,
        last_command: String::new(),
        question: None,
        started_at: 0,
        updated_at: 0,
        group: None,
        repo_path: Some(other.path().to_string_lossy().into_owned()),
        owner: Some("agent-a".to_string()),
        metrics: Default::default(),
        base_branch: Some("main".to_string()),
        base_commit: None,
        revision: 0,
        auto_continues: 0,
    };
    save_registry_entry_in(&f.root(), &entry);

    let err = f
        .merge_approved(Some("agent-a"), None)
        .expect_err("one batch is one repository");

    assert!(
        err.to_string()
            .contains("merge each repository's approved workers"),
        "{err}"
    );
    assert!(git_ref_exists(f.repo(), "worker-w1"), "nothing was merged");
    let _ = PathBuf::from(other.path());
}

/// A missing recorded command is not agreement with another worker's command.
#[test]
fn one_missing_verify_requires_a_project_gate() {
    let f = Fixture::new("batch-missing-verify");
    f.commit_on_worker_branch("w1", "w1.txt", "one\n");
    f.commit_on_worker_branch("w2", "w2.txt", "two\n");
    f.approved_worker("w1", "agent-a", Some("true"));
    f.approved_worker("w2", "agent-a", None);
    let err = f
        .merge_approved(Some("agent-a"), None)
        .expect_err("missing command is not agreement");
    assert!(
        err.to_string()
            .contains("record no verify command they agree on"),
        "{err}"
    );
    assert!(f.merge_subjects().is_empty());
}
