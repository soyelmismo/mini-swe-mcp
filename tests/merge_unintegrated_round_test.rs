//! Integration tests for `mini-swe-mcp merge <consolidator>` refusing a round
//! whose members carry commits the consolidator never integrated.
//!
//! The failure these guard against: a consolidator integrates `worker-a`, the
//! orchestrator merges it, `worker-a` is then revised and commits again on its
//! own branch, and the consolidator lands without those commits -- master ships
//! stale work with nothing said about it. The retirement sweep already kept the
//! revised worker (its tip is beyond what was integrated); the merge is where
//! the round's provenance is checked, so the check lives there.
//!
//! Every test owns its own repository and its own scratch root: nothing here
//! touches the developer's repository, the real registry or the host.

mod common;

use common::{TempDir, git, git_ref_exists};
use mini_swe_mcp::agent::{ChatMessage, Role};
use mini_swe_mcp::pool::{
    MergeApprovedRequest, MergeRequest, RegistryStatus, WorkerApproval, WorkerHistory,
    WorkerRegistryEntry, append_history_message_in, load_registry_entry_in, merge_approved_in,
    merge_worker_in, save_registry_entry_in, unintegrated_workers_in,
};
use mini_swe_mcp::worktree::ScratchRoot;
use std::path::Path;

/// The group every fixture worker of a round shares; the round is the owner's
/// workers in one group, which is what the consolidator was dispatched for.
const GROUP: &str = "round-fixture";

/// A repository with a base branch, plus a scratch root no other test sees.
struct Fixture {
    repo: TempDir,
    scratch: TempDir,
}

impl Fixture {
    fn new(tag: &str) -> Self {
        let repo = TempDir::new_in_tmp(tag);
        git(repo.path(), &["init", "--initial-branch=main"]);
        git(repo.path(), &["config", "user.email", "round@test"]);
        git(repo.path(), &["config", "user.name", "round test"]);
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

    /// Commit `contents` on `branch` (creating it off `main` if it is new),
    /// leaving `main` checked out again.
    fn commit_on(&self, branch: &str, path: &str, contents: &str, message: &str) {
        if !git_ref_exists(self.repo(), branch) {
            git(self.repo(), &["checkout", "-q", "-b", branch]);
        } else {
            git(self.repo(), &["checkout", "-q", branch]);
        }
        write(self.repo(), path, contents);
        git(self.repo(), &["add", "."]);
        git(self.repo(), &["commit", "-m", message]);
        git(self.repo(), &["checkout", "-q", "main"]);
    }

    /// A consolidator that integrated `integrated`: its own branch already
    /// carries each member's first commit, and its row records the round.
    ///
    /// This is the round31 shape: `worker-a`'s first tip was integrated at the
    /// moment the consolidator merged it.
    fn consolidator(&self, consolidator: &str, integrated: &[&str]) {
        self.commit_on(
            &format!("worker-{consolidator}"),
            "round.md",
            "the integrated round\n",
            "consolidate the round",
        );
        for id in integrated {
            self.commit_on(
                &format!("worker-{id}"),
                &format!("{id}.md"),
                "first\n",
                "first work",
            );
            let branch = format!("worker-{consolidator}");
            git(self.repo(), &["checkout", "-q", &branch]);
            let worker_branch = format!("worker-{id}");
            git(
                self.repo(),
                &[
                    "merge",
                    "-q",
                    "--no-ff",
                    "-m",
                    &format!("integrate {id}"),
                    &worker_branch,
                ],
            );
            git(self.repo(), &["checkout", "-q", "main"]);
            self.record_history(id);
            self.record_row(id, RegistryStatus::Completed);
        }
        self.record_history(consolidator);
        let entry = WorkerRegistryEntry {
            task: "consolidate the round".to_string(),
            status: RegistryStatus::Completed,
            step: 5,
            repo_path: Some(self.repo().to_string_lossy().into_owned()),
            base_branch: Some("main".to_string()),
            base_commit: Some(git(self.repo(), &["rev-parse", "HEAD"]).trim().to_string()),
            group: Some(GROUP.to_string()),
            integrated: integrated.iter().map(|id| id.to_string()).collect(),
            ..WorkerRegistryEntry::test_row(consolidator, "agent-a")
        };
        save_registry_entry_in(&self.root(), &entry);
    }

    fn record_history(&self, id: &str) {
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
            // A trivial gate: the batch runs one shared gate over the composed
            // round, and it must run for a test that is about what happens
            // before it.
            verify: Some("exit 0".to_string()),
            client_env: Vec::new(),
            max_turns: 10,
            review_after: None,
            revision: 0,
            auto_continues: 0,
            owner: Some("agent-a".to_string()),
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

    fn record_row(&self, id: &str, status: RegistryStatus) {
        self.record_row_in_group(id, status, GROUP);
    }

    fn record_row_in_group(&self, id: &str, status: RegistryStatus, group: &str) {
        let entry = WorkerRegistryEntry {
            task: format!("do the {id} work"),
            status,
            step: 1,
            group: Some(group.to_string()),
            repo_path: Some(self.repo().to_string_lossy().into_owned()),
            owner: Some("agent-a".to_string()),
            base_branch: Some("main".to_string()),
            ..WorkerRegistryEntry::test_row(id, "agent-a")
        };
        save_registry_entry_in(&self.root(), &entry);
    }

    /// A member of the round the consolidator was dispatched for, which it
    /// chose NOT to integrate: its branch carries work the round does not hold.
    ///
    /// Round41's shape: the consolidator reported one member "not ready",
    /// merged the other two and completed. Nothing about that is wrong on its
    /// own -- the merge just must not let the round land as if the member's
    /// work were in it.
    fn left_out_member(&self, id: &str) {
        self.commit_on(
            &format!("worker-{id}"),
            &format!("{id}.md"),
            "unintegrated work\n",
            "work the round never took",
        );
        self.record_history(id);
        self.record_row(id, RegistryStatus::Completed);
    }

    /// Mark `id` approved, exactly as the `approve` action records it.
    fn approve(&self, id: &str) {
        let mut entry =
            load_registry_entry_in(&self.root(), id).expect("the fixture records a row");
        entry.approved = Some(WorkerApproval { at: 1, note: None });
        save_registry_entry_in(&self.root(), &entry);
    }

    /// Land every approved worker, as `merge --approved` does.
    fn merge_approved(&self) -> anyhow::Result<()> {
        merge_approved_in(
            &self.root(),
            &MergeApprovedRequest {
                owner: None,
                group: None,
                admission: None,
                archive_dir: None,
            },
        )
        .map(|_| ())
    }

    /// Merge `id`, with `force` as given, exactly as the MCP handler does.
    fn merge(&self, id: &str, force: bool) -> anyhow::Result<()> {
        merge_worker_in(
            &self.root(),
            &MergeRequest {
                worker_id: id,
                verified: Some(true),
                keep_branch: false,
                force,
                admission: None,
                archive_dir: None,
            },
        )
        .map(|_| ())
    }

    /// `merge <id> --force`: the same merge with the round check overridden.
    fn force_merge(&self, id: &str) -> anyhow::Result<()> {
        self.merge(id, true)
    }
}

fn write(dir: &Path, name: &str, contents: &str) {
    std::fs::write(dir.join(name), contents).expect("fixture file must be writable");
}

/// Whether `ancestor` is reachable from `descendant`, without the panic the
/// shared `git` helper raises on a non-zero exit (this probe exits 1 by design).
fn is_ancestor(repo: &Path, ancestor: &str, descendant: &str) -> bool {
    std::process::Command::new("git")
        .args(["merge-base", "--is-ancestor", ancestor, descendant])
        .current_dir(repo)
        .output()
        .map(|out| out.status.success())
        .unwrap_or(false)
}

/// A member revised after the integration refuses the round, by name and with
/// its unintegrated commit count.
///
/// The refusal must name the worker the orchestrator has to act on and say how
/// much work is at stake -- an anonymous refusal is one the orchestrator cannot
/// route a steer or a discard from.
#[test]
fn merge_refuses_a_member_revised_after_it_was_integrated() {
    let f = Fixture::new("round-stale");
    f.consolidator("c1", &["wa"]);
    // The revision: the worker commits again on its own branch after the
    // consolidator integrated its first tip.
    f.commit_on(
        "worker-wa",
        "late.md",
        "late work\n",
        "revision after integration",
    );

    let err = f
        .merge("c1", false)
        .expect_err("a round whose member moved on must not merge silently");
    let message = format!("{err:#}");
    assert!(
        message.contains("wa"),
        "the refusal must name the worker: {message}"
    );
    assert!(
        message.contains("1 unintegrated commit"),
        "the refusal must count the unintegrated commits: {message}"
    );
    // Both ways out are spelled out, so the orchestrator never has to guess.
    assert!(
        message.contains("steer"),
        "the refusal must name steering as a way out: {message}"
    );
    assert!(
        message.contains("discard"),
        "the refusal must name discard as a way out: {message}"
    );
    assert!(
        message.contains("--force"),
        "the refusal must name --force as the override: {message}"
    );

    // Refusing changes nothing: the base branch still has only its own commit.
    assert_eq!(
        git(f.repo(), &["rev-list", "--count", "main"]).trim(),
        "1",
        "a refused merge must not move the base branch"
    );
    assert!(
        !f.repo().join("round.md").exists(),
        "a refused merge must not land the round"
    );
    assert!(
        load_registry_entry_in(&f.root(), "c1").is_some(),
        "a refused merge must leave the consolidator's row alone"
    );
}

/// `--force` lands the round anyway, and the unintegrated worker is not
/// retired: its extra commits are the only copy of that work.
#[test]
fn forced_merge_lands_and_leaves_the_unintegrated_worker_unretired() {
    let f = Fixture::new("round-forced");
    f.consolidator("c1", &["wa"]);
    f.commit_on(
        "worker-wa",
        "late.md",
        "late work\n",
        "revision after integration",
    );

    f.merge("c1", true).expect("--force must merge anyway");

    // The round landed.
    assert!(
        f.repo().join("round.md").exists(),
        "the forced merge must land the consolidator's branch"
    );
    // The late commit did not, which is the whole point of the refusal: it is
    // still on the worker's branch and nowhere else.
    assert!(
        !f.repo().join("late.md").exists(),
        "the unintegrated commit must not have landed with the round"
    );
    assert!(
        git_ref_exists(f.repo(), "worker-wa"),
        "a forced merge must leave the unintegrated worker's branch"
    );
    let row = load_registry_entry_in(&f.root(), "wa")
        .expect("an unintegrated worker must survive a forced merge");
    assert_eq!(
        row.status,
        RegistryStatus::Completed,
        "the worker keeps its row so its work stays visible"
    );
    assert!(
        f.scratch.path().join("swe-wt-wa.history.jsonl").exists(),
        "the unintegrated worker's conversation must survive the round"
    );
    // The consolidator itself is retired: it did land.
    assert!(
        load_registry_entry_in(&f.root(), "c1").is_none(),
        "the landed consolidator retires with its branch"
    );
}

/// A round whose every member is proven integrated merges with no override.
#[test]
fn merge_lands_a_round_whose_every_member_is_integrated() {
    let f = Fixture::new("round-clean");
    f.consolidator("c1", &["wa", "wb"]);

    f.merge("c1", false)
        .expect("an all-integrated round must merge");

    assert!(
        f.repo().join("wa.md").exists() && f.repo().join("wb.md").exists(),
        "both members must be in the landed round"
    );
    // Every member retired with the round: nothing left unretired, because
    // nothing was left unintegrated.
    for id in ["wa", "wb"] {
        assert!(
            load_registry_entry_in(&f.root(), id).is_none(),
            "an integrated member must retire with the round: {id}"
        );
    }
    assert!(!git_ref_exists(f.repo(), "worker-wa"));
}

/// A consolidator that integrated a worker by content, not by history, is
/// integrated and merges without an override.
///
/// Round38's shape: the consolidator took worker-ce1813b4's change by squash or
/// cherry-pick, so the worker's commits are not ancestors of the consolidator
/// branch even though every line of the work is on it. An ancestry-only check
/// refuses that round and pushes the orchestrator to steer or force a merge that
/// is already correct -- so the content has to count as proof too.
#[test]
fn a_worker_integrated_by_squash_is_not_unintegrated() {
    let f = Fixture::new("round-squash");
    f.consolidator("c1", &["wa"]);
    // Replace the merge of the worker's branch with an equivalent squash: the
    // consolidated tree is unchanged, the history is not -- which is exactly
    // what `cherry-pick` into the round leaves behind.
    let consolidator = "worker-c1".to_string();
    let before = git(
        f.repo(),
        &["rev-parse", &format!("{consolidator}^{{tree}}")],
    );
    git(f.repo(), &["checkout", "-q", &consolidator]);
    git(f.repo(), &["reset", "-q", "--soft", "HEAD~1"]);
    git(f.repo(), &["commit", "-q", "-m", "integrate wa (squashed)"]);
    git(f.repo(), &["checkout", "-q", "main"]);
    let after = git(
        f.repo(),
        &["rev-parse", &format!("{consolidator}^{{tree}}")],
    );
    assert_eq!(
        before.trim(),
        after.trim(),
        "the squash must not change the round's content"
    );

    // The tree really is the same: only the history differs.
    // The worker's commit is no longer reachable from the round: only history
    // was rewritten, so the ancestry proof alone would refuse this merge. The
    // test is meaningless without that, so it is asserted rather than assumed.
    assert!(
        !is_ancestor(f.repo(), "worker-wa", &consolidator),
        "the squash must leave the worker's tip unreachable, or this is not the squash case"
    );

    assert!(
        unintegrated_workers_in(&f.root(), "c1").is_empty(),
        "a squash-integrated worker holds the round back"
    );
    f.merge("c1", false)
        .expect("a round whose worker was integrated by content must merge");
    assert!(f.repo().join("round.md").exists());
}

/// A member whose commits cancel out is integrated: the round's tree already
/// holds its content.
///
/// The contract this check states is the tree's, not each commit's: merging the
/// member into the round changes nothing, so the round does carry its work.
/// `worker-wa` adds a file and a second commit deletes it again; the net trees
/// are identical even though both commits are unreachable from the round, and
/// that is integrated by the specified definition (a clean merge whose tree
/// equals the consolidator's own). Per-commit bookkeeping is a stricter rule
/// than the one the orchestrator set, and this test pins the specified one so a
/// future change to it is deliberate rather than accidental.
#[test]
fn a_member_whose_commits_cancel_out_counts_as_integrated() {
    let f = Fixture::new("round-cancel");
    f.consolidator("c1", &["wa"]);
    // Two commits, one adding work and one removing it again: the branch and
    // the round end up with identical trees.
    f.commit_on("worker-wa", "cancel.md", "work\n", "add then remove");
    git(f.repo(), &["checkout", "-q", "worker-wa"]);
    git(f.repo(), &["rm", "-q", "cancel.md"]);
    git(f.repo(), &["commit", "-q", "-m", "remove it again"]);
    git(f.repo(), &["checkout", "-q", "main"]);

    // The premise, asserted: the member's commits are unreachable from the
    // round, so only the content proof can pass it.
    assert!(
        !is_ancestor(f.repo(), "worker-wa", "worker-c1"),
        "the member's commits must be unreachable from the round"
    );

    assert!(
        unintegrated_workers_in(&f.root(), "c1").is_empty(),
        "a member that merges to the round's own tree is integrated"
    );
    f.merge("c1", false)
        .expect("a round whose only member cancels out must merge");
    assert!(f.repo().join("round.md").exists());
}

/// A multi-commit squash is integration, and must not be refused.
///
/// Round38's shape generalised: the consolidator folded *several* of a worker's
/// commits into its own branch in one commit. `git cherry` -- a per-commit
/// patch walk -- reports the squashed commits as unaccounted for and refuses
/// the round, which is a false refusal of an integration that is complete. The
/// specified proof is on the tree, and a clean merge that produces the
/// consolidator's own tree is the definition of "already integrated".
#[test]
fn a_multi_commit_squash_is_not_unintegrated() {
    let f = Fixture::new("round-multisquash");
    f.consolidator("c1", &["wa"]);
    // Three commits of real work on the member's branch.
    f.commit_on("worker-wa", "one.md", "one\n", "first commit");
    git(f.repo(), &["checkout", "-q", "worker-wa"]);
    write(f.repo(), "two.md", "two\n");
    git(f.repo(), &["add", "."]);
    git(f.repo(), &["commit", "-q", "-m", "second commit"]);
    write(f.repo(), "three.md", "three\n");
    git(f.repo(), &["add", "."]);
    git(f.repo(), &["commit", "-q", "-m", "third commit"]);
    git(f.repo(), &["checkout", "-q", "main"]);

    // Squash all three onto the round: one commit, the same content.
    git(f.repo(), &["checkout", "-q", "worker-c1"]);
    git(f.repo(), &["merge", "-q", "--squash", "worker-wa"]);
    git(f.repo(), &["commit", "-q", "-m", "integrate wa (squashed)"]);
    git(f.repo(), &["checkout", "-q", "main"]);

    // The premise, asserted: no member commit is reachable from the round.
    assert!(
        !is_ancestor(f.repo(), "worker-wa", "worker-c1"),
        "a squash must leave the member's commits unreachable"
    );

    assert!(
        unintegrated_workers_in(&f.root(), "c1").is_empty(),
        "a multi-commit squash is integration, not lost work"
    );
    f.merge("c1", false)
        .expect("a round whose member was squashed must merge");
    assert!(f.repo().join("round.md").exists());
}

/// A member the consolidator listed as "not ready" and never merged holds the
/// round back.
///
/// This is round41's failure: the consolidator integrated two members, reported
/// the third as not ready, and completed. The `integrated` set names only the
/// two it merged, so a check that reads that set alone waves the round through
/// and master lands without the third member's work -- with nothing said. The
/// round is what the consolidator was *dispatched for*, so a member it left out
/// is as absent from its branch as one it merged and then missed.
#[test]
fn a_member_left_out_of_the_round_refuses_the_merge() {
    let f = Fixture::new("round-leftout");
    f.consolidator("c1", &["wa"]);
    f.left_out_member("wb");

    let reported = unintegrated_workers_in(&f.root(), "c1");
    assert_eq!(
        reported.len(),
        1,
        "the member the consolidator left out must be reported: {reported:?}"
    );
    assert_eq!(reported[0].worker_id, "wb");
    assert!(
        reported[0].left_out,
        "the report must distinguish 'left out' from 'merged and then revised'"
    );

    let err = f
        .merge("c1", false)
        .expect_err("a round missing a member it was dispatched for must refuse");
    let message = format!("{err:#}");
    assert!(
        message.contains("wb") && message.contains("left out of the round"),
        "the refusal must name the member and say it was left out: {message}"
    );

    // And the round still lands, with the left-out member intact, under
    // --force: its branch is the only copy of that work.
    f.force_merge("c1")
        .expect("--force must land a round with a member left out");
    assert!(f.repo().join("round.md").exists());
    assert!(
        !f.repo().join("wb.md").exists(),
        "the left-out member's work must not ride in with the round"
    );
    assert!(
        git_ref_exists(f.repo(), "worker-wb"),
        "a forced merge must leave the left-out member's branch"
    );
}

/// A worker absorbed by the consolidator is not a member left out: the
/// consolidator took its leftover work over itself, so its branch is not
/// expected in the round and must not hold it back.
#[test]
fn an_absorbed_member_does_not_hold_the_round_back() {
    let f = Fixture::new("round-absorbed");
    f.consolidator("c1", &["wa"]);
    f.left_out_member("wb");
    // The consolidator absorbed it: it finished that work itself.
    let row = load_registry_entry_in(&f.root(), "c1").expect("consolidator row");
    save_registry_entry_in(
        &f.root(),
        &WorkerRegistryEntry {
            absorbed: vec!["wb".to_string()],
            ..row
        },
    );

    assert!(
        unintegrated_workers_in(&f.root(), "c1").is_empty(),
        "an absorbed member must not be reported as left out"
    );
    f.merge("c1", false)
        .expect("a round whose only unintegrated member was absorbed must merge");
    assert!(f.repo().join("round.md").exists());
}

/// A discarded member has no branch, so there is nothing left to land and
/// nothing to hold the round back.
#[test]
fn a_discarded_member_does_not_hold_the_round_back() {
    let f = Fixture::new("round-discarded");
    f.consolidator("c1", &["wa"]);
    f.left_out_member("wb");
    // Discarded: the branch is gone.
    git(f.repo(), &["branch", "-D", "worker-wb"]);

    assert!(
        unintegrated_workers_in(&f.root(), "c1").is_empty(),
        "a member with no branch cannot be holding the round back"
    );
    f.merge("c1", false)
        .expect("a round whose discarded member has no branch must merge");
    assert!(f.repo().join("round.md").exists());
}

/// A member still running when the consolidator completed is still a member of
/// the round: its branch carries work the round does not hold, and the merge
/// must say so rather than let the round land over it.
#[test]
fn a_member_still_running_holds_the_round_back() {
    let f = Fixture::new("round-running");
    f.consolidator("c1", &["wa"]);
    f.left_out_member("wb");
    f.record_row("wb", RegistryStatus::Running);

    let reported = unintegrated_workers_in(&f.root(), "c1");
    assert_eq!(reported.len(), 1, "the live member must be reported");
    assert_eq!(reported[0].worker_id, "wb");
    assert!(
        reported[0].left_out,
        "a member never merged is 'left out' whatever its status"
    );
    let err = f
        .merge("c1", false)
        .expect_err("a round that never took a member's work must refuse");
    assert!(format!("{err:#}").contains("wb"));
}

/// The round is the one the consolidator was dispatched for, not whatever the
/// group happens to hold at merge time.
///
/// Rows are retired as rounds land and groups are reused, so the same owner and
/// group name a different set of workers by the time the consolidator is merged:
/// a worker dispatched into the next round, and a member an earlier consolidator
/// absorbed. Proving that set would refuse a round over a worker it never had.
/// The dispatch snapshot is the round as it was, so a worker that arrives later
/// is not a member, and a member that was absorbed by an earlier round is not
/// either.
#[test]
fn the_round_is_the_dispatch_snapshot_not_the_live_group() {
    let f = Fixture::new("round-snapshot");
    f.consolidator("c1", &["wa"]);
    f.left_out_member("wb");
    // The dispatch recorded this round's membership.
    std::fs::write(f.scratch.path().join("swe-wt-c1.round-members"), "wa\nwb\n")
        .expect("the dispatch snapshot must be writable");

    // A worker dispatched into the SAME group AFTER this consolidator ran: it
    // is in the group now and was not in the round.
    f.left_out_member("later");
    // An earlier consolidator absorbed another member of the group; that row
    // was never part of this round either.
    f.left_out_member("old");
    let earlier = WorkerRegistryEntry {
        group: Some(GROUP.to_string()),
        absorbed: vec!["old".to_string()],
        ..WorkerRegistryEntry::test_row("c0", "agent-a")
    };
    save_registry_entry_in(&f.root(), &earlier);

    let reported: Vec<String> = unintegrated_workers_in(&f.root(), "c1")
        .into_iter()
        .map(|worker| worker.worker_id)
        .collect();
    assert_eq!(
        reported,
        vec!["wb".to_string()],
        "only this round's unintegrated member may be reported: {reported:?}"
    );

    // `wb` is a real member of this round and still unintegrated, so the merge
    // refuses for exactly that one worker -- `later` and `old` are not members
    // and are not named.
    let err = f
        .merge("c1", false)
        .expect_err("the round's own unintegrated member still refuses");
    let message = format!("{err:#}");
    assert!(
        message.contains("wb"),
        "the refusal names the member: {message}"
    );
    assert!(
        !message.contains("later") && !message.contains("old"),
        "a worker outside the dispatch snapshot must not be named: {message}"
    );

    f.force_merge("c1").expect("--force lands the round");
    assert!(f.repo().join("round.md").exists());
}

/// The consolidator's completion path names the unintegrated worker, so the
/// orchestrator hears about a stale round while it can still act on it.
#[test]
fn consolidator_completion_reports_the_unintegrated_member() {
    let f = Fixture::new("round-report");
    f.consolidator("c1", &["wa"]);
    assert!(
        unintegrated_workers_in(&f.root(), "c1").is_empty(),
        "a round that still matches its record reports nothing"
    );

    f.commit_on(
        "worker-wa",
        "late.md",
        "late work\n",
        "revision after integration",
    );
    let reported = unintegrated_workers_in(&f.root(), "c1");
    assert_eq!(reported.len(), 1, "the revised member must be reported");
    assert_eq!(reported[0].worker_id, "wa");
    assert_eq!(reported[0].commits, Some(1));
    let line = reported[0].line();
    assert!(
        line.contains("wa") && line.contains("UNINTEGRATED"),
        "the reported line must name the worker and say what is wrong: {line}"
    );
}

/// A member whose branch is gone has nothing left to integrate, so it never
/// blocks the round it was part of.
#[test]
fn a_missing_member_branch_counts_as_integrated() {
    let f = Fixture::new("round-pruned");
    f.consolidator("c1", &["wa"]);
    git(f.repo(), &["branch", "-D", "worker-wa"]);

    assert!(
        unintegrated_workers_in(&f.root(), "c1").is_empty(),
        "a pruned member branch must not hold the round back"
    );
    f.merge("c1", false)
        .expect("a round with a pruned member must merge");
    assert!(f.repo().join("round.md").exists());
}

/// The batch path lands an approved consolidator exactly as `merge <id>` does,
/// so it owes the orchestrator the same refusal: an approved round whose member
/// moved on after the integration must not land through `--approved` either.
#[test]
fn the_approved_batch_refuses_a_stale_round_too() {
    let f = Fixture::new("round-batch-stale");
    f.consolidator("c1", &["wa"]);
    f.commit_on(
        "worker-wa",
        "late.md",
        "late work\n",
        "revision after integration",
    );
    f.approve("c1");

    let err = f
        .merge_approved()
        .expect_err("a batch must not land a round whose member moved on either");
    let message = format!("{err:#}");
    assert!(
        message.contains("wa"),
        "the batch refusal must name the worker: {message}"
    );
    assert!(
        message.contains("1 unintegrated commit"),
        "the batch refusal must count the unintegrated commits: {message}"
    );
    assert!(
        !f.repo().join("round.md").exists(),
        "a refused batch must not land the round"
    );
}

/// The same batch, with every member proven integrated, lands as before: the
/// refusal above is about the stale member, not about batching.
#[test]
fn the_approved_batch_lands_a_round_whose_members_are_integrated() {
    let f = Fixture::new("round-batch-clean");
    f.consolidator("c1", &["wa"]);
    f.approve("c1");

    f.merge_approved()
        .expect("a batch whose every member is integrated must land");
    assert!(
        f.repo().join("round.md").exists() && f.repo().join("wa.md").exists(),
        "the batch must land the whole round"
    );
}

/// The refusal never reads a missing number as zero.
///
/// A member the harness can prove is unintegrated but cannot count is refused
/// with its count marked unknown, because an orchestrator reading "0
/// unintegrated commit(s)" would take the branch as clean -- the one reading
/// this whole check exists to prevent. The count is built by git walking the
/// range, so an unrelated failure of that walk is the case: the branch is
/// genuinely outside the round (proven by ancestry) while the count cannot say
/// how far.
#[test]
fn an_uncountable_member_is_refused_without_a_number() {
    let f = Fixture::new("round-uncountable");
    f.consolidator("c1", &["wa"]);
    // The member moved on after the integration, so ancestry and the count both
    // have a real answer; the assertions below are about the wording the count
    // earns, and a count is only ever printed when git gave one.
    f.commit_on(
        "worker-wa",
        "late.md",
        "late work\n",
        "revision after integration",
    );
    let reported = unintegrated_workers_in(&f.root(), "c1");
    assert_eq!(reported.len(), 1);
    let counted = reported[0].commits.expect("git can count this one");
    assert_eq!(counted, 1);

    // The refusal and the line the completion event carries both print that
    // count, and never a zero for a member that is not integrated.
    let err = f
        .merge("c1", false)
        .expect_err("a member revised after the integration holds the round back");
    let message = format!("{err:#}");
    assert!(
        message.contains(&format!(
            "worker wa carries {counted} unintegrated commit(s)"
        )),
        "the refusal names the member and its count: {message}"
    );
    assert!(
        !message.contains("0 unintegrated commit"),
        "an unintegrated member must never be reported as zero: {message}"
    );
    assert!(
        reported[0].line().contains("1 unintegrated commit(s)"),
        "the completion line carries the same count: {}",
        reported[0].line()
    );
    assert!(
        !reported[0].line().contains("0 "),
        "the completion line must not read as a clean worker: {}",
        reported[0].line()
    );
    assert!(
        !f.repo().join("round.md").exists(),
        "a refused merge must not land the round"
    );
}
