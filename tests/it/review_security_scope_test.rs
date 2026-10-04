//! The security review audits only what it has not audited yet.
//!
//! Every revision re-ran the review of the whole diff since the base commit, so
//! a worker the consolidator corrected seven times was security-reviewed seven
//! times over largely the same approved code, and a consolidator -- whose diff
//! is the union of branches that were each already reviewed -- was reviewed in
//! full again. Three properties are asserted:
//!
//! * **No sensitive change, no review.** A revision that changes nothing since
//!   the commit an earlier security review approved skips the review, and says
//!   so in the log rather than paying for it again.
//! * **Only the new commits are reviewed.** A revision that does touch a
//!   sensitive path is reviewed over the diff since that approved commit, with
//!   the previously approved commits named for context, so the reviewer neither
//!   re-reads approved code nor mistakes it for new.
//! * **A consolidator reviews its own commits.** The worker branches it merged,
//!   each already reviewed at its own approved commit, are excluded from its
//!   security scope; only its interaction fixes and conflict resolutions remain.
//! * **A consolidator does not re-audit the base branch it merged in.** The base
//!   branch's current tip is subtracted the same way, so work that landed on it
//!   after the dispatch -- already reviewed where it came from -- stays out; a
//!   resolution the consolidator wrote itself is a commit the base does not
//!   contain and stays in, and a base tip git cannot resolve subtracts nothing.
//!
//! Every repository is a temporary directory this test creates and removes; no
//! registry, hub or real repository is written.

use crate::common;
use mini_swe_mcp::pool::{
    ReviewMode, WorkerRole, approved_merged_branches, plan_review, scope_for,
};
use std::path::Path;

/// Commit `body` into `branch` after writing `file`, and return the commit id.
///
/// `branch` is created from the current HEAD when it does not exist yet, so the
/// very first commit lands on a fresh base branch without needing a checkout of
/// a branch that has no commit yet.
fn commit(dir: &Path, branch: &str, file: &str, body: &str) -> String {
    if !common::git_ref_exists(dir, branch) {
        common::git(dir, &["checkout", "-q", "-b", branch]);
    } else {
        common::git(dir, &["checkout", "-q", branch]);
    }
    let path = dir.join(file);
    std::fs::create_dir_all(path.parent().expect("the file under test has a parent"))
        .expect("create the directory holding the file under test");
    std::fs::write(&path, body).expect("write the file under test");
    common::git(dir, &["add", file]);
    common::git(dir, &["commit", "-q", "-m", body]);
    common::git(dir, &["rev-parse", "HEAD"]).trim().to_string()
}

/// A repository with a `master` base and a `worker-w1` branch one commit past it.
///
/// Returns the temporary directory (which removes itself on drop) and the base
/// commit both the worker branch and `master` point at.
fn repo(tag: &str) -> (common::TempDir, String) {
    let dir = common::TempDir::new_in_tmp(tag);
    let path = dir.path().to_path_buf();
    common::git(&path, &["init", "-q", "-b", "master", "."]);
    // Repo-scoped identity: the test's commits need an author, and writing it
    // into this temporary repository keeps the machine's global git config and
    // this process's environment untouched.
    common::git(
        &path,
        &["config", "user.email", "review-scope@example.invalid"],
    );
    common::git(&path, &["config", "user.name", "Review Scope Test"]);
    let base = commit(&path, "master", "README.md", "base");
    common::git(&path, &["checkout", "-q", "-b", "worker-w1"]);
    (dir, base)
}

/// The scope a worker on `branch` would be security-reviewed over.
async fn worker_scope(
    repo: &Path,
    branch: &str,
    base: &str,
    approved: Option<String>,
) -> mini_swe_mcp::pool::SecurityScope {
    scope_for(repo, branch, WorkerRole::Worker, base, approved, &[], None).await
}

#[tokio::test]
async fn a_revision_with_no_sensitive_change_since_the_approval_skips_the_review() {
    let (dir, base) = repo("scope_skip");
    let approved = commit(
        dir.path(),
        "worker-w1",
        "src/hub/socket.rs",
        "sensitive change",
    );

    // A first run has no approval to start from: it reviews everything.
    let first = worker_scope(dir.path(), "worker-w1", &base, None).await;
    assert_eq!(
        first.skip_log(),
        None,
        "nothing is approved yet, so nothing may be skipped"
    );
    assert_eq!(
        first.reviewed_commits(),
        Vec::<String>::new(),
        "a first run is measured against the base commit, not against an approval"
    );
    assert_eq!(first.base_commit(), None);

    // The consolidator routes a correction back that changes nothing new. The
    // security review that approved this branch already stands.
    let revision = worker_scope(dir.path(), "worker-w1", &base, Some(approved.clone())).await;
    assert_eq!(
        revision.skip_log().as_deref(),
        Some(format!("security review skipped: no sensitive change since {approved}").as_str()),
        "a revision that adds no commit since the approved one must skip the review"
    );
    assert!(
        revision.reviewed_commits().is_empty(),
        "a skipped review covers no commits"
    );
}

#[tokio::test]
async fn a_revision_with_a_sensitive_change_reviews_only_the_new_commits() {
    let (dir, base) = repo("scope_incremental");
    let approved = commit(
        dir.path(),
        "worker-w1",
        "src/hub/socket.rs",
        "sensitive change",
    );
    // The correction the consolidator routed back touches a sensitive path too.
    let correction = commit(dir.path(), "worker-w1", "src/hub/identity.rs", "correction");

    let scope = worker_scope(dir.path(), "worker-w1", &base, Some(approved.clone())).await;
    assert_eq!(scope.skip_log(), None, "a new commit must be reviewed");
    assert_eq!(
        scope.reviewed_commits(),
        vec![correction.clone()],
        "only the commit after the approved one is unaudited"
    );
    assert_eq!(
        scope.approved_commits(),
        vec![approved.clone()],
        "the earlier approval is named for context, not re-reviewed"
    );

    let files = scope.reviewed_files(dir.path()).await;
    assert_eq!(
        files,
        vec!["src/hub/identity.rs".to_string()],
        "the sensitive probe must see the new commit's file and not the approved one"
    );
    let diff = scope.reviewed_diff(dir.path()).await;
    assert!(
        diff.contains("src/hub/identity.rs") && !diff.contains("src/hub/socket.rs"),
        "the reviewer is handed the incremental diff, not the whole branch diff: {diff}"
    );
}

#[tokio::test]
async fn a_consolidator_reviews_only_its_own_commits() {
    let (dir, base) = repo("scope_consolidate");
    // Each worker branch is already security-reviewed at its own approved commit.
    let w1 = commit(dir.path(), "worker-w1", "src/hub/socket.rs", "worker one");
    common::git(dir.path(), &["checkout", "-q", "master"]);
    common::git(dir.path(), &["checkout", "-q", "-b", "worker-w2"]);
    let w2 = commit(dir.path(), "worker-w2", "src/hub/events.rs", "worker two");

    // The consolidator integrates both and then resolves the interaction.
    common::git(dir.path(), &["checkout", "-q", "-b", "worker-c1", "master"]);
    common::git(
        dir.path(),
        &["merge", "-q", "--no-ff", "-m", "merge w1", "worker-w1"],
    );
    common::git(
        dir.path(),
        &["merge", "-q", "--no-ff", "-m", "merge w2", "worker-w2"],
    );
    let own = commit(
        dir.path(),
        "worker-c1",
        "src/hub/handshake.rs",
        "resolve the interaction",
    );

    let merged = vec!["worker-w1".to_string(), "worker-w2".to_string()];
    let scope = scope_for(
        dir.path(),
        "worker-c1",
        WorkerRole::Consolidate,
        &base,
        None,
        &merged,
        Some("master"),
    )
    .await;

    assert_eq!(
        scope.reviewed_commits(),
        vec![own.clone()],
        "a consolidator reviews only the commits nobody has reviewed yet"
    );
    for reviewed in [&w1, &w2] {
        assert!(
            !scope.reviewed_commits().contains(reviewed),
            "{reviewed} was reviewed at its own approved commit and must not be reviewed again"
        );
    }
    assert_eq!(
        scope.reviewed_files(dir.path()).await,
        vec!["src/hub/handshake.rs".to_string()],
        "the sensitive probe must not re-report the merged workers' files"
    );
    assert_eq!(
        scope.skip_log(),
        None,
        "a consolidator with work of its own has something to review"
    );
}

#[tokio::test]
async fn an_unresolvable_approval_never_reads_as_no_change() {
    let (dir, base) = repo("scope_unknown");
    commit(
        dir.path(),
        "worker-w1",
        "src/hub/socket.rs",
        "sensitive change",
    );
    // A pruned branch can leave an approved commit the repository cannot resolve.
    // Reading that gap as "nothing changed" would skip a real audit, so the
    // scope must fall back to reviewing the whole diff.
    let unknown = "0".repeat(40);
    let scope = worker_scope(dir.path(), "worker-w1", &base, Some(unknown.clone())).await;
    assert_ne!(
        scope.skip_log(),
        Some(format!(
            "security review skipped: no sensitive change since {unknown}"
        )),
        "an approval the repository cannot resolve must not be read as an approval"
    );
}

/// The scope must not report a skip while the working tree carries an
/// unaudited change: an agent that edits a sensitive file without committing it
/// has changed the code, and a review that only looks at commit ranges would
/// wave that through.
#[tokio::test]
async fn an_uncommitted_sensitive_change_is_never_reported_as_no_change() {
    let (dir, base) = repo("scope_uncommitted");
    let approved = commit(
        dir.path(),
        "worker-w1",
        "src/hub/socket.rs",
        "sensitive change",
    );

    // The revision corrects a sensitive file but leaves it uncommitted: the
    // harness checkpoints after the review, so this state is a real one.
    let dirty = dir.path().join("src/hub/events.rs");
    std::fs::write(&dirty, "// unaudited work\n").expect("write the uncommitted change");
    common::git(dir.path(), &["add", "-N", "src/hub/events.rs"]);

    let scope = worker_scope(dir.path(), "worker-w1", &base, Some(approved.clone())).await;
    assert!(
        scope
            .reviewed_files(dir.path())
            .await
            .contains(&"src/hub/events.rs".to_string()),
        "the unaudited working-tree change must be part of what this review covers"
    );
    assert_ne!(
        scope.skip_log(),
        Some(format!(
            "security review skipped: no sensitive change since {approved}"
        )),
        "an uncommitted sensitive change must not be skipped: the audit that approved \
         {approved} never saw it"
    );
}

// ----------
// The mode a review actually runs in
// ----------

/// Skipping the automatic repeat must not weaken a gate the caller asked for by
/// name: `--review-after <model>:security` names the adversarial audit, so a
/// bookkeeping "nothing new" decision may defer the automatic trigger but may
/// never quietly turn an explicit security request into a generic quality pass.
#[test]
fn an_explicitly_requested_security_review_is_never_downgraded() {
    assert_eq!(
        plan_review(
            true,
            Some(("nerd".to_string(), ReviewMode::security())),
            false,
            "automatic",
            &ReviewMode::security()
        ),
        Some(("nerd".to_string(), ReviewMode::security())),
        "a requested security review stays an adversarial one even when the skip defers the trigger"
    );
    assert_eq!(
        plan_review(
            true,
            Some(("nerd".to_string(), ReviewMode::quality())),
            false,
            "automatic",
            &ReviewMode::security()
        ),
        Some(("nerd".to_string(), ReviewMode::quality())),
        "a requested quality review runs as asked"
    );
    // The skip still does its job: nothing asked for, so nothing runs twice.
    assert_eq!(
        plan_review(true, None, false, "automatic", &ReviewMode::security()),
        None
    );
    assert_eq!(
        plan_review(true, None, true, "automatic", &ReviewMode::security()),
        None
    );
}

/// Without a skip, the pre-existing rules must be unchanged: a requested review
/// on a sensitive diff is upgraded to the adversarial mode, and the automatic
/// sensitive-path trigger audits on the security mode's reviewer.
#[test]
fn without_a_skip_the_trigger_and_the_upgrade_are_unchanged() {
    assert_eq!(
        plan_review(
            false,
            Some(("nerd".to_string(), ReviewMode::quality())),
            true,
            "automatic",
            &ReviewMode::security()
        ),
        Some(("nerd".to_string(), ReviewMode::quality()))
    );
    assert_eq!(
        plan_review(
            false,
            Some(("nerd".to_string(), ReviewMode::quality())),
            false,
            "automatic",
            &ReviewMode::security()
        ),
        Some(("nerd".to_string(), ReviewMode::security())),
        "a requested review on a sensitive diff is upgraded to the adversarial mode"
    );
    assert_eq!(
        plan_review(false, None, true, "automatic", &ReviewMode::security()),
        None,
        "no request and nothing sensitive means no review"
    );
    assert_eq!(
        plan_review(false, None, false, "automatic", &ReviewMode::security()),
        Some(("automatic".to_string(), ReviewMode::security())),
        "the automatic sensitive trigger audits on the security mode's reviewer"
    );
}

// ----------
// Which commit an approval names
// ----------

/// The approval must name the commit that *contains* the reviewed code.
///
/// The reviewer leaves the tree at some HEAD, and the harness then commits that
/// tree. Recording the pre-commit HEAD would leave the reviewed tree itself
/// outside the approval: the next revision re-audits code already reviewed, and
/// -- with the approval now a subtraction set -- a consolidator re-audits it too.
/// So once the harness's commit is made and the tree is provably still the tree
/// the reviewer left, the next run must find nothing new and skip.
#[tokio::test]
async fn the_approval_names_the_commit_that_carries_the_reviewed_tree() {
    let (dir, base) = repo("scope_approval_commit");
    // The reviewer approved the tree while it was still uncommitted, at this HEAD.
    let reviewer_head = common::git(dir.path(), &["rev-parse", "HEAD"])
        .trim()
        .to_string();
    std::fs::create_dir_all(dir.path().join("src/hub")).unwrap();
    std::fs::write(dir.path().join("src/hub/socket.rs"), "reviewed change").unwrap();
    common::git(dir.path(), &["add", "src/hub/socket.rs"]);

    // The harness commits that tree.
    common::git(dir.path(), &["commit", "-q", "-m", "worker(w1): reviewed"]);
    let harness_commit = common::git(dir.path(), &["rev-parse", "HEAD"])
        .trim()
        .to_string();
    assert_ne!(
        reviewer_head, harness_commit,
        "the harness commit is after the HEAD the reviewer started from"
    );

    // The approval is the harness's commit, so nothing is left to audit.
    let scope = scope_for(
        dir.path(),
        "worker-w1",
        WorkerRole::Worker,
        &base,
        Some(harness_commit.clone()),
        &[],
        None,
    )
    .await;
    assert_eq!(
        scope.skip_log().as_deref(),
        Some(
            format!("security review skipped: no sensitive change since {harness_commit}").as_str()
        ),
        "a revision with no new change after an approved review must skip the security review"
    );

    // The pre-commit HEAD would not have: it names nothing of the reviewed tree,
    // which is exactly the re-review this fix removes.
    let stale = scope_for(
        dir.path(),
        "worker-w1",
        WorkerRole::Worker,
        &base,
        Some(reviewer_head.clone()),
        &[],
        None,
    )
    .await;
    assert_ne!(
        stale.skip_log(),
        Some(format!(
            "security review skipped: no sensitive change since {reviewer_head}"
        )),
        "approving the pre-commit HEAD leaves the reviewed commit itself unaudited"
    );
}

/// The approval follows the tree only while the tree is still the one that was
/// reviewed. A commit landing after the review means the harness's commit is not
/// that tree's commit, so the approval stays behind and the difference is
/// re-reviewed rather than silently waved through.
#[tokio::test]
async fn a_tree_changed_after_the_review_is_not_approved_by_the_newer_commit() {
    let (dir, base) = repo("scope_approval_moved");
    let reviewer_head = common::git(dir.path(), &["rev-parse", "HEAD"])
        .trim()
        .to_string();
    std::fs::create_dir_all(dir.path().join("src/hub")).unwrap();
    std::fs::write(dir.path().join("src/hub/socket.rs"), "reviewed change").unwrap();
    common::git(dir.path(), &["add", "src/hub/socket.rs"]);
    common::git(dir.path(), &["commit", "-q", "-m", "worker(w1): reviewed"]);
    let reviewed_commit = common::git(dir.path(), &["rev-parse", "HEAD"])
        .trim()
        .to_string();

    // Something the reviewer never saw lands on the branch afterwards.
    let after = commit(
        dir.path(),
        "worker-w1",
        "src/hub/identity.rs",
        "later change",
    );
    assert_ne!(reviewer_head, after);

    let scope = scope_for(
        dir.path(),
        "worker-w1",
        WorkerRole::Worker,
        &base,
        Some(reviewer_head),
        &[],
        None,
    )
    .await;
    assert_eq!(
        scope.skip_log(),
        None,
        "a commit after the review must be audited; approving the newer commit would skip it"
    );
    assert_eq!(
        scope.reviewed_commits(),
        vec![reviewed_commit, after],
        "a stale approval re-reviews the reviewed commit too -- conservative, and never a missed audit"
    );
}

// ----------
// Which merged branches may leave the audit
// ----------

/// `integrated` proves a branch was merged, not that it was security-reviewed.
/// Excluding it on the strength of the merge alone would drop that code out of
/// every audit: the worker never got a review, and the consolidator that is
/// supposed to cover it no longer sees it.
#[test]
fn a_merged_branch_leaves_the_audit_only_when_it_was_security_approved() {
    let integrated = vec!["w1".to_string(), "w2".to_string()];
    // A real object id: the guard admits only a plain git object id, so a
    // fixture that is not one would exercise the guard, not the exclusion.
    let approved_w1 = "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0".to_string();

    // w1 was security-reviewed; w2 was merged without ever being reviewed.
    let reviewed =
        approved_merged_branches(&integrated, |id| (id == "w1").then(|| approved_w1.clone()));
    assert_eq!(
        reviewed,
        vec![approved_w1.clone()],
        "only a worker carrying an approved commit may leave the audit, and it leaves it at that commit"
    );

    // No worker row at all, or a row from before the field existed, is not an
    // approval either.
    assert!(approved_merged_branches(&integrated, |_| None).is_empty());
}

/// The exclusion is the approved **commit**, never the branch name: a branch
/// keeps growing after its review, and excluding its current tip would subtract
/// commits nobody ever audited -- the consolidator would be the only reviewer
/// that could still have caught them.
#[tokio::test]
async fn a_merged_branch_beyond_its_approval_stays_in_the_consolidators_scope() {
    let (dir, base) = repo("scope_beyond_approval");
    // The worker is security-reviewed at `approved`, then lands more work.
    let approved = commit(
        dir.path(),
        "worker-w1",
        "src/hub/socket.rs",
        "reviewed change",
    );
    let unaudited = commit(
        dir.path(),
        "worker-w1",
        "src/hub/identity.rs",
        "change after the approval",
    );
    assert_ne!(approved, unaudited);

    // A consolidator merges the branch at its tip.
    common::git(dir.path(), &["checkout", "-q", "-b", "worker-c1", "master"]);
    common::git(
        dir.path(),
        &["merge", "-q", "--no-ff", "-m", "merge w1", "worker-w1"],
    );
    let own = commit(
        dir.path(),
        "worker-c1",
        "src/hub/handshake.rs",
        "resolve the interaction",
    );

    let scope = scope_for(
        dir.path(),
        "worker-c1",
        WorkerRole::Consolidate,
        &base,
        None,
        &approved_merged_branches(&["w1".to_string()], |_| Some(approved.clone())),
        Some("master"),
    )
    .await;

    assert!(
        scope.reviewed_commits().contains(&unaudited),
        "a commit the merged branch carries past its approval must be audited by the consolidator: {:?}",
        scope.reviewed_commits()
    );
    assert!(
        !scope.reviewed_commits().contains(&approved),
        "the approved commit itself is already audited and must not be repeated"
    );
    assert!(
        scope.reviewed_commits().contains(&own),
        "the consolidator's own commit is its to audit"
    );
    assert!(
        scope
            .reviewed_files(dir.path())
            .await
            .contains(&"src/hub/identity.rs".to_string()),
        "the unaudited sensitive change must be probed, or the trigger never fires: {:?}",
        scope.reviewed_files(dir.path()).await
    );
    assert_eq!(scope.skip_log(), None);
}

/// The exclusion stays sound when every merged worker was in fact reviewed: the
/// point of the optimisation is that a reviewed commit is not audited twice.
#[test]
fn every_approved_merged_branch_leaves_the_audit() {
    let integrated = vec!["w1".to_string(), "w2".to_string()];
    // Distinct, valid object ids: the exclusion is the approved commit, and a
    // real one must survive the object-id guard.
    let approved_w1 = "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0".to_string();
    let approved_w2 = "b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0".to_string();
    assert_eq!(
        approved_merged_branches(&integrated, |id| match id {
            "w1" => Some(approved_w1.clone()),
            "w2" => Some(approved_w2.clone()),
            _ => None,
        }),
        vec![approved_w1.clone(), approved_w2.clone()],
    );
}

// ----------
// A consolidator's conflict resolutions are its own work
// ----------

/// The one part of a merge nobody else reviewed is the resolution itself: the
/// consolidator decides which of two workers' versions of a sensitive file
/// wins, and that decision lands *in the merge commit*. A scope that drops
/// merge commits drops the resolution with them, so a consolidator whose only
/// own work was resolving a `src/hub/**` conflict would see an empty commit
/// list, find no sensitive file, and skip the audit of a file both workers
/// disagreed about.
#[tokio::test]
async fn a_conflict_resolved_in_a_merge_commit_stays_in_the_consolidators_scope() {
    let dir = common::TempDir::new_in_tmp("scope_resolution");
    let path = dir.path().to_path_buf();
    common::git(&path, &["init", "-q", "-b", "master", "."]);
    common::git(
        &path,
        &["config", "user.email", "review-scope@example.invalid"],
    );
    common::git(&path, &["config", "user.name", "Review Scope Test"]);
    let socket = "src/hub/socket.rs";
    std::fs::create_dir_all(path.join("src/hub")).expect("create the sensitive directory");
    std::fs::write(path.join(socket), "the original handshake").expect("seed the sensitive file");
    common::git(&path, &["add", socket]);
    common::git(&path, &["commit", "-q", "-m", "base"]);
    let base = common::git(&path, &["rev-parse", "HEAD"])
        .trim()
        .to_string();

    // Two workers edit the same sensitive file, so integrating them conflicts
    // and the consolidator alone has to decide which version wins.
    common::git(&path, &["checkout", "-q", "-b", "worker-w1"]);
    std::fs::write(path.join(socket), "worker's handshake").expect("worker one edits the file");
    common::git(&path, &["commit", "-q", "-a", "-m", "w1"]);
    let w1 = common::git(&path, &["rev-parse", "HEAD"])
        .trim()
        .to_string();
    common::git(&path, &["checkout", "-q", "-b", "worker-w2", "master"]);
    std::fs::write(path.join(socket), "other worker's handshake")
        .expect("worker two edits the same file");
    common::git(&path, &["commit", "-q", "-a", "-m", "w2"]);
    let w2 = common::git(&path, &["rev-parse", "HEAD"])
        .trim()
        .to_string();

    // The consolidator integrates both; the second merge conflicts.
    common::git(&path, &["checkout", "-q", "-b", "worker-c1", "master"]);
    common::git(
        &path,
        &["merge", "-q", "--no-ff", "-m", "merge w1", "worker-w1"],
    );
    let conflict = std::process::Command::new("git")
        .args(["merge", "--no-ff", "-m", "merge w2", "worker-w2"])
        .current_dir(&path)
        .output()
        .expect("run the conflicting merge");
    assert!(
        !conflict.status.success(),
        "the two workers' edits must really conflict, or this test proves nothing"
    );
    let conflicted = std::fs::read_to_string(path.join(socket)).expect("read the conflicted file");
    assert!(
        conflicted.contains("<<<<<<<"),
        "git left conflict markers, so there is a resolution to review"
    );

    // The resolution lands in the merge commit, which is where a consolidator
    // agent records it.
    std::fs::write(path.join(socket), "the resolved handshake").expect("write the resolution");
    common::git(&path, &["add", socket]);
    common::git(&path, &["commit", "-q", "--no-edit"]);
    let resolution = common::git(&path, &["rev-parse", "HEAD"])
        .trim()
        .to_string();
    assert!(
        !scope_is_non_merge(&path, &resolution),
        "the resolution really does live in a merge commit, which is what this test is about"
    );

    let merged = vec![w1.clone(), w2.clone()];
    let scope = scope_for(
        &path,
        "worker-c1",
        WorkerRole::Consolidate,
        &base,
        None,
        &merged,
        Some("master"),
    )
    .await;

    assert!(
        scope.reviewed_commits().contains(&resolution),
        "the merge commit that carries the conflict resolution is the consolidator's own \
         unaudited work and must be in scope; got {:?}",
        scope.reviewed_commits()
    );
    for reviewed in [&w1, &w2] {
        assert!(
            !scope.reviewed_commits().contains(reviewed),
            "{reviewed} was reviewed at its own approved commit and must not come back"
        );
    }
    assert_eq!(
        scope.reviewed_files(&path).await,
        vec![socket.to_string()],
        "the sensitive probe must see the file whose conflict the consolidator resolved"
    );
    assert_eq!(
        scope.skip_log(),
        None,
        "a consolidator whose only work is a conflict resolution has something to review"
    );

    // And the diff it hands the reviewer must actually carry the resolution.
    let diff = scope.reviewed_diff(&path).await;
    assert!(
        diff.contains("the resolved handshake"),
        "the reviewer's diff must contain the resolution, or it is told to audit code it \
         cannot see; diff was:\n{diff}"
    );
}

/// True when `commit` is a merge commit, so the test above can state that the
/// resolution it wrote really landed in one.
fn scope_is_non_merge(dir: &Path, commit: &str) -> bool {
    common::git(dir, &["rev-list", "--no-merges", "-1", commit])
        .trim()
        .is_empty()
}

/// A planted approval that is not a plain git object id must not reach a git
/// revision argument in the consolidator path either. The worker path already
/// rejects a non-object-id approval (`scope_for` filters through
/// `is_object_id`); the consolidator path must do the same, or a value like
/// `--output=<path>` planted on a merged worker's registry row would be
/// spliced verbatim into `git rev-list` and execute as an option, creating or
/// truncating an arbitrary file. The safe direction is to treat it as no
/// approval: the merged worker's branch stays in the consolidator's audit.
#[tokio::test]
async fn a_planted_non_object_id_approval_never_reaches_git_in_the_consolidator_path() {
    let (dir, base) = repo("scope_planted_consolidator");
    // `repo()` already left `worker-w1` checked out; add one commit to it.
    let w1 = commit(dir.path(), "worker-w1", "src/hub/socket.rs", "worker one");

    // The consolidator merges it and adds its own work.
    common::git(dir.path(), &["checkout", "-q", "-b", "worker-c1", "master"]);
    common::git(
        dir.path(),
        &["merge", "-q", "--no-ff", "-m", "merge w1", "worker-w1"],
    );
    let own = commit(
        dir.path(),
        "worker-c1",
        "src/hub/handshake.rs",
        "resolve the interaction",
    );

    // The planted approval looks like a git option, not a revision.
    let planted = "--output=pwned".to_string();
    let scope = scope_for(
        dir.path(),
        "worker-c1",
        WorkerRole::Consolidate,
        &base,
        None,
        &approved_merged_branches(&["w1".to_string()], |_| Some(planted.clone())),
        Some("master"),
    )
    .await;

    // Without the guard, `git rev-list --output=pwned` would have created this
    // file in the repository.
    assert!(
        !dir.path().join("pwned").exists(),
        "a non-object-id approval must not be executed as a git option"
    );
    // The safe direction: the merged worker's code is not treated as approved,
    // so it stays in the consolidator's audit rather than being dropped.
    assert!(
        scope.reviewed_commits().contains(&w1),
        "a merged branch whose approval is not a real object id must stay in scope; got {:?}",
        scope.reviewed_commits()
    );
    assert!(
        scope.reviewed_commits().contains(&own),
        "the consolidator's own commit is still its to audit"
    );
    assert_eq!(scope.skip_log(), None);
}

/// The pre-2.36 fallback of the own-files probe must still name a merge commit's
/// resolved file. A current git never reaches the fallback (it always accepts
/// `--remerge-diff`), so the test drives the fallback query through a test-only
/// export of the args the production code would build for an old git, on a
/// repository whose merge carries a real conflict resolution. A bug that dropped
/// the file list of a merge from the fallback query would slip the resolved
/// sensitive file out of a consolidator's scope on a pre-2.36 git: the very code
/// nobody else reviewed. The test asserts the file is named, and the test-only
/// driver is the only way to take the fallback path on a modern git.
#[tokio::test]
async fn a_pre_2_36_fallback_own_files_query_names_a_merge_resolutions_file() {
    let dir = common::TempDir::new_in_tmp("scope_fallback_merge");
    let path = dir.path().to_path_buf();
    common::git(&path, &["init", "-q", "-b", "master", "."]);
    common::git(
        &path,
        &["config", "user.email", "review-scope@example.invalid"],
    );
    common::git(&path, &["config", "user.name", "Review Scope Test"]);
    let socket = "src/hub/socket.rs";
    std::fs::create_dir_all(path.join("src/hub")).expect("create the sensitive directory");
    std::fs::write(path.join(socket), "the original handshake").expect("seed the sensitive file");
    common::git(&path, &["add", socket]);
    common::git(&path, &["commit", "-q", "-m", "base"]);
    let _base = common::git(&path, &["rev-parse", "HEAD"])
        .trim()
        .to_string();

    // Two workers disagree on the same sensitive file; the consolidator resolves.
    common::git(&path, &["checkout", "-q", "-b", "worker-w1"]);
    std::fs::write(path.join(socket), "worker's handshake").expect("worker one edits the file");
    common::git(&path, &["commit", "-q", "-a", "-m", "w1"]);
    let w1 = common::git(&path, &["rev-parse", "HEAD"])
        .trim()
        .to_string();
    common::git(&path, &["checkout", "-q", "-b", "worker-w2", "master"]);
    std::fs::write(path.join(socket), "other worker's handshake")
        .expect("worker two edits the same file");
    common::git(&path, &["commit", "-q", "-a", "-m", "w2"]);
    let w2 = common::git(&path, &["rev-parse", "HEAD"])
        .trim()
        .to_string();

    common::git(&path, &["checkout", "-q", "-b", "worker-c1", "master"]);
    common::git(
        &path,
        &["merge", "-q", "--no-ff", "-m", "merge w1", "worker-w1"],
    );
    let conflict = std::process::Command::new("git")
        .args(["merge", "--no-ff", "-m", "merge w2", "worker-w2"])
        .current_dir(&path)
        .output()
        .expect("run the conflicting merge");
    assert!(
        !conflict.status.success(),
        "the two workers' edits must really conflict, or this test proves nothing"
    );
    std::fs::write(path.join(socket), "the resolved handshake").expect("write the resolution");
    common::git(&path, &["add", socket]);
    common::git(&path, &["commit", "-q", "--no-edit"]);

    // Drive the fallback query directly: it is the args `own_files` would
    // build on a pre-2.36 git, so the assertion exercises exactly the bug the
    // fallback must not have.
    let files = mini_swe_mcp::pool::__test_own_files_fallback(
        &path,
        "worker-c1",
        &[w1, w2],
        Some("master"),
    )
    .await;
    assert!(
        files.contains(&socket.to_string()),
        "the fallback query must name the file whose conflict the consolidator resolved; got {files:?}"
    );
}

/// The resolution is a decision even when the resolved content happens to equal
/// one parent's version: the consolidator chose that side, and nothing else in
/// the history records the choice. The fallback must name the file anyway, so
/// the query has to diff the merge against *each* parent (`-m`) rather than only
/// against the parents it differs from everywhere (`--cc`, which stays silent
/// here -- the one case where the narrower flag would drop the resolution).
#[tokio::test]
async fn a_pre_2_36_fallback_names_a_resolution_that_matches_one_parent() {
    let dir = common::TempDir::new_in_tmp("scope_fallback_takes_side");
    let path = dir.path().to_path_buf();
    common::git(&path, &["init", "-q", "-b", "master", "."]);
    common::git(
        &path,
        &["config", "user.email", "review-scope@example.invalid"],
    );
    common::git(&path, &["config", "user.name", "Review Scope Test"]);
    let socket = "src/hub/socket.rs";
    std::fs::create_dir_all(path.join("src/hub")).expect("create the sensitive directory");
    std::fs::write(path.join(socket), "original\nkeep\nend").expect("seed the sensitive file");
    common::git(&path, &["add", socket]);
    common::git(&path, &["commit", "-q", "-m", "base"]);

    // The two workers change the same middle line, so integrating them conflicts.
    common::git(&path, &["checkout", "-q", "-b", "worker-w1"]);
    std::fs::write(path.join(socket), "worker's line\nkeep\nend").expect("worker one edits");
    common::git(&path, &["commit", "-q", "-a", "-m", "w1"]);
    let w1 = common::git(&path, &["rev-parse", "HEAD"])
        .trim()
        .to_string();
    common::git(&path, &["checkout", "-q", "-b", "worker-w2", "master"]);
    std::fs::write(path.join(socket), "other worker's line\nkeep\nend")
        .expect("worker two edits the same line");
    common::git(&path, &["commit", "-q", "-a", "-m", "w2"]);
    let w2 = common::git(&path, &["rev-parse", "HEAD"])
        .trim()
        .to_string();

    common::git(&path, &["checkout", "-q", "-b", "worker-c1", "master"]);
    common::git(
        &path,
        &["merge", "-q", "--no-ff", "-m", "merge w1", "worker-w1"],
    );
    let conflict = std::process::Command::new("git")
        .args(["merge", "--no-ff", "-m", "merge w2", "worker-w2"])
        .current_dir(&path)
        .output()
        .expect("run the conflicting merge");
    assert!(
        !conflict.status.success(),
        "the two workers' edits must really conflict, or this test proves nothing"
    );
    // The consolidator resolves by taking worker one's version wholesale.
    std::fs::write(path.join(socket), "worker's line\nkeep\nend").expect("take w1's version");
    common::git(&path, &["add", socket]);
    common::git(&path, &["commit", "-q", "--no-edit"]);

    let files = mini_swe_mcp::pool::__test_own_files_fallback(
        &path,
        "worker-c1",
        &[w1, w2],
        Some("master"),
    )
    .await;
    assert!(
        files.contains(&socket.to_string()),
        "a resolution that takes one parent's version is still the consolidator's decision and \
         must be named by the fallback; got {files:?}"
    );
}

// ----------
// The base branch a consolidator merged in
// ----------

/// A consolidator that resolved its conflicts against the base branch's current
/// state: `master` gained sensitive work after the workers branched off, one
/// worker's branch is security-approved, and the consolidator's merge of the
/// updated base conflicts in the sensitive file the worker had edited too.
///
/// Returns the temporary directory, the base commit the consolidator was
/// dispatched from, the merged worker's approved commit, the commits the
/// updated base gained, and the merge commit carrying the resolution.
fn repo_with_updated_base(tag: &str) -> (common::TempDir, String, String, Vec<String>, String) {
    let (dir, base) = repo(tag);
    let socket = "src/hub/socket.rs";
    // The worker edits the sensitive file and is security-approved there.
    let w1 = commit(dir.path(), "worker-w1", socket, "worker's handshake");
    // The base branch gains its own sensitive work after the dispatch: work that
    // was reviewed on the branch that landed it, not on the consolidator's.
    let env_change = commit(dir.path(), "master", "src/agent/env.rs", "base gains ENV1");
    let socket_change = commit(dir.path(), "master", socket, "base rewrites the handshake");

    // The consolidator, dispatched from the old base, merges the worker and then
    // the updated base. The two sides of the sensitive file conflict, and the
    // resolution lands in the merge commit -- the consolidator's own decision.
    common::git(dir.path(), &["checkout", "-q", "-b", "worker-c1", &base]);
    common::git(
        dir.path(),
        &["merge", "-q", "--no-ff", "-m", "merge w1", "worker-w1"],
    );
    let conflict = std::process::Command::new("git")
        .args(["merge", "--no-ff", "-m", "merge the updated base", "master"])
        .current_dir(dir.path())
        .output()
        .expect("run the conflicting merge against the updated base");
    assert!(
        !conflict.status.success(),
        "the base and the worker must really conflict, or this fixture proves nothing"
    );
    std::fs::write(dir.path().join(socket), "the resolved handshake")
        .expect("write the resolution");
    common::git(dir.path(), &["add", socket]);
    common::git(dir.path(), &["commit", "-q", "--no-edit"]);
    let resolution = common::git(dir.path(), &["rev-parse", "HEAD"])
        .trim()
        .to_string();
    (dir, base, w1, vec![env_change, socket_change], resolution)
}

/// Merging the updated base must not drag the base's own history back into the
/// consolidator's audit: that history was reviewed where it landed, and the
/// round that produced it was already merged. The resolution the consolidator
/// wrote is a commit the base does not contain, so it stays in scope.
#[tokio::test]
async fn a_consolidator_that_merged_the_updated_base_does_not_re_audit_the_base_change() {
    let (dir, base, w1, base_changes, resolution) = repo_with_updated_base("scope_base_merge");
    let scope = scope_for(
        dir.path(),
        "worker-c1",
        WorkerRole::Consolidate,
        &base,
        None,
        std::slice::from_ref(&w1),
        Some("master"),
    )
    .await;

    for change in &base_changes {
        assert!(
            !scope.reviewed_commits().contains(change),
            "{change} came in with the base branch and was reviewed there; it must not be \
             audited again: {:?}",
            scope.reviewed_commits()
        );
    }
    assert!(
        scope.reviewed_commits().contains(&resolution),
        "the conflict resolution is the consolidator's own unaudited work and must stay in scope"
    );
    let files = scope.reviewed_files(dir.path()).await;
    assert_eq!(
        files,
        vec!["src/hub/socket.rs".to_string()],
        "the base branch's own sensitive file must not re-enter the sensitive-path probe; \
         got {files:?}"
    );
    let diff = scope.reviewed_diff(dir.path()).await;
    assert!(
        diff.contains("the resolved handshake") && !diff.contains("base gains ENV1"),
        "the reviewer is handed the resolution, not the base branch's change; diff was:\n{diff}"
    );
    assert_eq!(scope.skip_log(), None);
}

/// The exclusion is only as wide as the evidence: a base branch git cannot
/// resolve subtracts nothing, so the scope stays exactly as wide as it was
/// before base branches were excluded at all.
#[tokio::test]
async fn an_unresolvable_base_branch_never_shrinks_the_consolidators_scope() {
    let (dir, base, w1, base_changes, _resolution) = repo_with_updated_base("scope_base_unknown");
    let scope = scope_for(
        dir.path(),
        "worker-c1",
        WorkerRole::Consolidate,
        &base,
        None,
        std::slice::from_ref(&w1),
        Some("no-such-base"),
    )
    .await;

    for change in &base_changes {
        assert!(
            scope.reviewed_commits().contains(change),
            "a base branch git cannot resolve must not subtract anything from the audit, yet \
             {change} left it: {:?}",
            scope.reviewed_commits()
        );
    }
    assert!(
        scope
            .reviewed_files(dir.path())
            .await
            .contains(&"src/agent/env.rs".to_string()),
        "with no resolvable base tip the base's sensitive change stays in the probe"
    );
}

/// A file name a commit carries is data, never a pattern. Git reads a pathspec
/// that starts with `:` as magic, so a file planted as `:(exclude)<sensitive>`
/// in the same commit as a real edit to that sensitive file must not act as an
/// exclusion: the probe still names the sensitive path (it compares names), and
/// the diff handed to the reviewer must still carry the sensitive change.
#[tokio::test]
async fn a_planted_pathspec_name_cannot_exclude_a_sensitive_change_from_the_diff() {
    let (dir, base) = repo("scope_pathspec_plant");
    let env_file = "src/agent/env.rs";
    let planted = ":(exclude)src/agent/env.rs";
    let w1 = commit(
        dir.path(),
        "worker-w1",
        "src/hub/socket.rs",
        "worker's handshake",
    );

    // The consolidator merges the approved worker, then commits its own change
    // to the sensitive file together with a file whose *name* is the pathspec
    // that excludes that very file.
    common::git(dir.path(), &["checkout", "-q", "-b", "worker-c1", &base]);
    common::git(
        dir.path(),
        &["merge", "-q", "--no-ff", "-m", "merge w1", "worker-w1"],
    );
    std::fs::create_dir_all(
        dir.path()
            .join(env_file)
            .parent()
            .expect("the sensitive file has a parent"),
    )
    .expect("create the sensitive file's directory");
    std::fs::write(dir.path().join(env_file), "the consolidator's ENV edit")
        .expect("write the sensitive edit");
    std::fs::create_dir_all(
        dir.path()
            .join(planted)
            .parent()
            .expect("the planted name has a parent"),
    )
    .expect("create the planted file's directory");
    std::fs::write(dir.path().join(planted), "harmless looking filler")
        .expect("write the planted file");
    common::git(dir.path(), &["add", "-A"]);
    common::git(
        dir.path(),
        &["commit", "-q", "-m", "the consolidator's own work"],
    );
    let _own = common::git(dir.path(), &["rev-parse", "HEAD"])
        .trim()
        .to_string();

    let scope = scope_for(
        dir.path(),
        "worker-c1",
        WorkerRole::Consolidate,
        &base,
        None,
        std::slice::from_ref(&w1),
        Some("master"),
    )
    .await;

    // The probe compares names, so the planted name cannot hide the sensitive
    // file from it and the review still fires over the sensitive path.
    let files = scope.reviewed_files(dir.path()).await;
    assert!(
        files.contains(&env_file.to_string()),
        "the sensitive file must stay in the probe; got {files:?}"
    );
    // And the diff the reviewer reads must still carry the sensitive change.
    let diff = scope.reviewed_diff(dir.path()).await;
    assert!(
        diff.contains("the consolidator's ENV edit"),
        "a planted pathspec name must not exclude the sensitive change from the reviewer's \
         diff; diff was:\n{diff}"
    );
    assert_eq!(scope.skip_log(), None);
}
