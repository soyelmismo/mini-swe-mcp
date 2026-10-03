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

use mini_swe_mcp::pool::{WorkerRole, scope_for};

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
    common::git(&path, &["config", "user.email", "review-scope@example.invalid"]);
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
    let approved = commit(dir.path(), "worker-w1", "src/hub/socket.rs", "sensitive change");

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
    let approved = commit(dir.path(), "worker-w1", "src/hub/socket.rs", "sensitive change");
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
    common::git(dir.path(), &["merge", "-q", "--no-ff", "-m", "merge w1", "worker-w1"]);
    common::git(dir.path(), &["merge", "-q", "--no-ff", "-m", "merge w2", "worker-w2"]);
    let own = commit(dir.path(), "worker-c1", "src/hub/handshake.rs", "resolve the interaction");

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
    commit(dir.path(), "worker-w1", "src/hub/socket.rs", "sensitive change");
    // A pruned branch can leave an approved commit the repository cannot resolve.
    // Reading that gap as "nothing changed" would skip a real audit, so the
    // scope must fall back to reviewing the whole diff.
    let unknown = "0".repeat(40);
    let scope = worker_scope(dir.path(), "worker-w1", &base, Some(unknown.clone())).await;
    assert_ne!(
        scope.skip_log(),
        Some(format!("security review skipped: no sensitive change since {unknown}")),
        "an approval the repository cannot resolve must not be read as an approval"
    );
}
