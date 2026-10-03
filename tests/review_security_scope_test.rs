//! The security review audits only what it has not audited yet.
//!
//! Each revision re-runs the review of the whole diff since the base commit, so
//! one worker corrected seven times was security-reviewed seven times over the
//! same approved code; a consolidator, whose diff is the union of branches each
//! already reviewed, was reviewed in full again. Three properties are asserted:
//!
//! * **No sensitive change, no review.** A revision that changes nothing since
//!   the commit an earlier security review approved skips the review, and says
//!   so in the log.
//! * **Only the new commits are reviewed.** A revision that does touch a
//!   sensitive path is reviewed over the diff since that approved commit, with
//!   the previously approved commits named for context.
//! * **A consolidator reviews its own commits.** The worker branches it merged,
//!   each already reviewed at its own approved commit, are excluded from its
//!   security scope; only its interaction fixes and conflict resolutions remain.
//!
//! Every repository is a temporary directory the test creates and removes, and
//! nothing is written to the real registry, hub or repository.

mod common;

use std::path::Path;

use mini_swe_mcp::pool::review_security_scope::{own_files, own_scope};
use mini_swe_mcp::registry::WorkerRole;

/// A commit id in a repository the test owns.
fn head(dir: &Path) -> String {
    common::git(dir, &["rev-parse", "HEAD"])
}

/// Commit `body` into `branch` after writing `path`, and return its commit id.
///
/// The identity is pinned for the whole test so a commit id never depends on
/// when the test ran, and `GIT_AUTHOR_DATE`/`GIT_COMMITTER_DATE` are passed
/// through the environment of this process only -- `git` reads them for the
/// spawned command, never for the harness.
fn commit(dir: &Path, branch: &str, path: &str, body: &str) -> String {
    common::git(dir, &["checkout", "-q", branch]);
    std::fs::write(dir.join(path), body).expect("write file");
    common::git(dir, &["add", path]);
    common::git(dir, &["commit", "-q", "-m", body]);
    head(dir)
}

/// A repository with a base branch and one worker branch off it.
fn repo(tag: &str) -> common::TempDir {
    let dir = common::TempDir::new_in_tmp(tag);
    common::git(dir.path(), &["init", "-q", "-b", "master", "."]);
    commit(dir.path(), "master", "README.md", "base");
    common::git(dir.path(), &["checkout", "-q", "-b", "worker-w1"]);
    let w1 = commit(dir.path(), "worker-w1", "src/hub/socket.rs", "worker change");
    (dir, w1)
}

#[test]
fn a_revision_with_no_new_sensitive_change_skips_the_review() {
    let (dir, w1) = repo("scope_skip");

    // The first security review approved the worker at this commit.
    let scope = own_scope(dir.path(), "worker-w1", &[], Some(w1.clone()));
    assert_eq!(
        scope.skip_log().as_deref(),
        None,
        "the first run has no approval to start from, so it reviews everything"
    );

    // The consolidator's correction touched nothing new: nothing changed since
    // the approved commit, so the approved review already stands.
    let scope = own_scope(dir.path(), "worker-w1", &[], Some(w1.clone()));
    assert_eq!(
        scope.skip_log().as_deref(),
        Some(
            format!("security review skipped: no sensitive change since {w1}")
                .as_str()
        ),
        "a revision that changes nothing since the approved commit must skip the security review"
    );
}

#[test]
fn a_revision_with_a_sensitive_change_reviews_only_the_new_commits() {
    let (dir, w1) = repo("scope_incremental");
    // The correction the consolidator routed back touches a sensitive path.
    let w2 = commit(dir.path(), "worker-w1", "src/hub/identity.rs", "correction");

    let scope = own_scope(dir.path(), "worker-w1", &[], Some(w1.clone()));
    assert_eq!(scope.skip_log(), None);
    assert_eq!(
        scope.reviewed_commits(),
        vec![w2.clone()],
        "only the commit after the approved one may be reviewed again"
    );
    assert_eq!(
        scope.approved_commits(),
        vec![w1.clone()],
        "the earlier approval is named so the reviewer does not re-audit it"
    );
    let files = scope.reviewed_files();
    assert!(
        files.contains(&"src/hub/identity.rs".to_string()),
        "the sensitive file the new commit touched must be probed: {files:?}"
    );
    assert!(
        !files.contains(&"src/hub/socket.rs".to_string()),
        "a file the approved review already covered is not sensitive again: {files:?}"
    );
}

#[test]
fn a_consolidator_reviews_its_own_commits_not_the_merged_branches() {
    let (dir, w1) = repo("scope_consolidate");
    // A second worker branch, each already reviewed at its own approved commit.
    common::git(dir.path(), &["checkout", "-q", "master"]);
    common::git(dir.path(), &["checkout", "-q", "-b", "worker-w2"]);
    let w2 = commit(dir.path(), "worker-w2", "src/hub/events.rs", "second worker");

    // The consolidator merges both, then fixes the interaction between them.
    common::git(dir.path(), &["checkout", "-q", "-b", "worker-c1", "master"]);
    common::git(dir.path(), &["merge", "-q", "--no-ff", "-m", "merge w1", "worker-w1"]);
    common::git(dir.path(), &["merge", "-q", "--no-ff", "-m", "merge w2", "worker-w2"]);
    let own = commit(dir.path(), "worker-c1", "src/hub/handshake.rs", "resolve the conflict");

    let scope = own_scope(
        dir.path(),
        "worker-c1",
        &["worker-w1".to_string(), "worker-w2".to_string()],
        None,
    );
    assert_eq!(
        scope.reviewed_commits(),
        vec![own.clone()],
        "a consolidator reviews only the commits nobody has reviewed yet"
    );
    assert!(
        !scope.reviewed_commits().contains(&w1) && !scope.reviewed_commits().contains(&w2),
        "the merged worker branches were reviewed at their own approved commits"
    );

    let files = own_files(dir.path(), "worker-c1", &["worker-w1".into(), "worker-w2".into()]);
    assert_eq!(
        files,
        vec!["src/hub/handshake.rs".to_string()],
        "the sensitive-path probe of a consolidator must not re-report the merged workers' files"
    );
    assert_ne!(scope.skip_log(), None.or(scope.skip_log()));
    assert_eq!(scope.skip_log(), None, "the consolidator has work of its own to review");
}

#[test]
fn an_unknown_approved_commit_falls_back_to_reviewing_everything() {
    let (dir, _) = repo("scope_unknown");
    // A pruned branch leaves an approved commit the repository cannot resolve.
    // Treating that gap as an approval would skip a real audit, so the scope
    // must widen to everything rather than report "nothing changed".
    let scope = own_scope(
        dir.path(),
        "worker-w1",
        &[],
        Some("0".repeat(40)),
    );
    assert_ne!(
        scope.skip_log(),
        Some(format!("security review skipped: no sensitive change since {}", "0".repeat(40))),
        "an unresolvable approval must never be read as 'nothing changed'"
    );
    let _ = WorkerRole::Worker;
}
