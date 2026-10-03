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
//!
//! Every repository is a temporary directory this test creates and removes; no
//! registry, hub or real repository is written.

mod common;

use std::path::Path;

use mini_swe_mcp::pool::{
    ReviewMode, WorkerRole, approved_merged_branches, plan_review, scope_for,
};

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
    scope_for(repo, branch, WorkerRole::Worker, base, approved, &[]).await
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
            Some(("nerd".to_string(), ReviewMode::Security)),
            false,
            "strongest",
        ),
        Some(("nerd".to_string(), ReviewMode::Security)),
        "a requested security review stays an adversarial one even when the skip defers the trigger"
    );
    assert_eq!(
        plan_review(
            true,
            Some(("nerd".to_string(), ReviewMode::Quality)),
            false,
            "strongest",
        ),
        Some(("nerd".to_string(), ReviewMode::Quality)),
        "a requested quality review runs as asked"
    );
    // The skip still does its job: nothing asked for, so nothing runs twice.
    assert_eq!(plan_review(true, None, false, "strongest"), None);
    assert_eq!(plan_review(true, None, true, "strongest"), None);
}

/// Without a skip, the pre-existing rules must be unchanged: a requested review
/// on a sensitive diff is upgraded to the adversarial mode, and the automatic
/// sensitive-path trigger audits on the manifest's strongest tier.
#[test]
fn without_a_skip_the_trigger_and_the_upgrade_are_unchanged() {
    assert_eq!(
        plan_review(
            false,
            Some(("nerd".to_string(), ReviewMode::Quality)),
            true,
            "strongest",
        ),
        Some(("nerd".to_string(), ReviewMode::Quality))
    );
    assert_eq!(
        plan_review(
            false,
            Some(("nerd".to_string(), ReviewMode::Quality)),
            false,
            "strongest",
        ),
        Some(("nerd".to_string(), ReviewMode::Security)),
        "a requested review on a sensitive diff is upgraded to the adversarial mode"
    );
    assert_eq!(
        plan_review(false, None, true, "strongest"),
        None,
        "no request and nothing sensitive means no review"
    );
    assert_eq!(
        plan_review(false, None, false, "strongest"),
        Some(("strongest".to_string(), ReviewMode::Security)),
        "the automatic sensitive trigger audits on the manifest's strongest tier"
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

    // w1 was security-reviewed; w2 was merged without ever being reviewed.
    let reviewed =
        approved_merged_branches(&integrated, |id| (id == "w1").then(|| "abc123".to_string()));
    assert_eq!(
        reviewed,
        vec!["abc123".to_string()],
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
    assert_eq!(
        approved_merged_branches(&integrated, |id| Some(format!("approved-{id}"))),
        vec!["approved-w1".to_string(), "approved-w2".to_string()],
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
    let base = commit(&path, "master", "README.md", "base");
    let socket = "src/hub/socket.rs";

    // Two workers branch off the same base and edit the same sensitive file, so
    // integrating them conflicts and the consolidator has to decide the result.
    common::git(&path, &["checkout", "-q", "-b", "worker-w1"]);
    let w1 = commit(&path, "worker-w1", socket, "worker's handshake");
    common::git(&path, &["checkout", "-q", "-b", "worker-w2", "master"]);
    let w2 = commit(&path, "worker-w2", socket, "other worker's handshake");
    common::git(&path, &["checkout", "-q", "-b", "worker-c1", "master"]);

    // Merge, resolve the conflict, and record the resolution in the merge
    // commit -- which is what git does and what the consolidator agent does.
    common::git(
        &path,
        &["merge", "-q", "--no-ff", "-m", "merge w1", "worker-w1"],
    );
    let conflicted = std::fs::read_to_string(path.join(socket)).expect("the conflicted file");
    assert!(
        conflicted.contains("<<<<<<<"),
        "the two workers' edits must really conflict, or this test proves nothing"
    );
    std::fs::write(path.join(socket), "the resolved handshake").expect("write the resolution");
    common::git(&path, &["add", socket]);
    common::git(&path, &["commit", "-q", "--no-edit"]);
    let resolution = common::git(&path, &["rev-parse", "HEAD"])
        .trim()
        .to_string();

    let merged = vec![w1.clone(), w2.clone()];
    let scope = scope_for(
        &path,
        "worker-c1",
        WorkerRole::Consolidate,
        &base,
        None,
        &merged,
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
        diff.contains("+the resolved handshake"),
        "the reviewer's diff must contain the resolution, or it is told to audit code it \
         cannot see; diff was:\n{diff}"
    );
}
