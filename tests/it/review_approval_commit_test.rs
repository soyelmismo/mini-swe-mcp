//! A security approval is the harness's own commit of the reviewed tree.
//!
//! The tree a security reviewer leaves behind is normally *uncommitted*: the
//! harness's `finalize_worktree` commits it only after the review phase
//! returns. So the commit that carries the audited code does not exist yet when
//! the review finishes, and any approval measured against the pre-commit HEAD
//! can never name it. When it fell back to the pre-commit HEAD, the next
//! revision -- and the consolidator, whose diff is the union of the branches --
//! re-audited code the review had already approved, which is the whole point of
//! recording an approval.
//!
//! Two properties:
//!
//! 1. Reviewed uncommitted edits: the approval is the harness's commit.
//! 2. The tree changed after the review snapshot: the approval stays at the
//!    pre-commit HEAD, which names less code than was reviewed, so the
//!    difference is re-reviewed rather than missed.

use crate::common;
use crate::common::fake_llm::FakeLlm;

use mini_swe_mcp::pool::{
    __test_commit_matches_snapshot, __test_snapshot_worktree_tree, WorkerPool, WorkerRegistryEntry,
};
use mini_swe_mcp::worktree::ScratchRoot;

const TEST_OWNER: &str = "review-approval-commit";

/// Dispatch a worker whose implementer writes an uncommitted sensitive file,
/// whose security reviewer edits more and then completes, and return the
/// registry row plus the branch head the harness's commit produced.
async fn review_uncommitted_changes(tag: &str) -> (WorkerRegistryEntry, String, String) {
    let repo = common::TestRepo::new(tag);
    repo.declare_sensitive(&["src/hub/**"]);
    let llm = FakeLlm::spawn_sse(vec![
        common::write_turn("call_write", "src/hub/mod.rs"),
        common::completion_turn("call_impl", "REPORT\ndone: impl\nrisks: none"),
        // The reviewer's own edit, left uncommitted in the worktree.
        common::write_turn("call_review", "src/hub/reviewed.rs"),
        common::security_completion_turn("call_review_done", 0),
    ])
    .await;

    let scratch = common::TempDir::new_in_tmp(tag);
    let pool = WorkerPool::with_scratch(
        1,
        llm.base_url().to_string(),
        "test-key".to_string(),
        ScratchRoot::new(scratch.path()),
    );
    let worker_id = pool
        .dispatch(
            TEST_OWNER.to_string(),
            "review the approved tree".to_string(),
            "test-model".to_string(),
            None,
            repo.path().to_path_buf(),
            8,
            Some("approval-commit".to_string()),
            None,
            false,
            None,
            Vec::new(),
        )
        .await
        .expect("dispatch the worker");
    let _state = common::wait_for_terminal(&pool, &worker_id).await;
    let entry = mini_swe_mcp::pool::load_registry_entry_in(pool.scratch_root(), &worker_id)
        .expect("the worker's registry row");
    // The branch head the harness's commit produced, as the repository has it.
    let harness_commit = common::git(
        repo.path(),
        &["rev-parse", &format!("refs/heads/worker-{worker_id}")],
    )
    .trim()
    .to_string();
    // The branch's own start: the HEAD the reviewer began from, which is the
    // value the approval must *not* fall back to.
    let pre_commit_head = common::git(repo.path(), &["rev-parse", &format!("worker-{worker_id}^")])
        .trim()
        .to_string();
    let _ = pool.kill(&worker_id).await;
    (entry, harness_commit, pre_commit_head)
}

/// The approval names the commit that carries the reviewed code, not the
/// pre-commit HEAD the reviewer started from.
#[tokio::test]
async fn reviewed_uncommitted_edits_approve_the_harness_commit() {
    let (entry, harness_commit, pre_commit_head) =
        review_uncommitted_changes("approval-commit").await;
    let approved = entry
        .security_approved_commit
        .expect("a completing review records an approval");
    assert_ne!(
        pre_commit_head, harness_commit,
        "the reviewed tree was uncommitted, so the harness's commit is not the \
         HEAD the reviewer started from"
    );
    assert_eq!(
        approved, harness_commit,
        "the approval is the harness's commit of the reviewed tree, not the \
         pre-commit HEAD (which would re-audit the reviewed code)"
    );
}

/// A tree that changed after the review snapshot is not approved.
///
/// The snapshot is taken through a temporary index, so the harness's own commit
/// of a *later* tree is a different tree object and must not be approved -- the
/// pre-commit HEAD stays, naming less code than was reviewed.
#[tokio::test]
async fn a_tree_changed_after_the_snapshot_keeps_the_pre_commit_head() {
    let repo = common::TestRepo::new("approval-commit-moved");
    let branch = "worker-moved";
    common::git(repo.path(), &["checkout", "-q", "-b", branch]);
    std::fs::write(repo.path().join("first.rs"), "// first\n").unwrap();
    common::git(repo.path(), &["add", "-A"]);
    common::git(repo.path(), &["commit", "-q", "-m", "first"]);
    let pre_commit_head = common::git(repo.path(), &["rev-parse", "HEAD"])
        .trim()
        .to_string();

    // The tree as a reviewer would leave it.
    std::fs::write(repo.path().join("reviewed.rs"), "// reviewed\n").unwrap();
    let snapshot = __test_snapshot_worktree_tree(repo.path())
        .await
        .expect("the reviewer's tree is snapshot-able");

    // Something touches the tree afterwards: the artifact sync, a steer landing
    // mid-flight, a second checkpoint.
    std::fs::write(repo.path().join("injected.rs"), "// not reviewed\n").unwrap();
    common::git(repo.path(), &["add", "-A"]);
    common::git(repo.path(), &["commit", "-q", "-m", "harness commit"]);

    let harness_commit = common::git(repo.path(), &["rev-parse", "HEAD"])
        .trim()
        .to_string();
    let matches = __test_commit_matches_snapshot(
        repo.path(),
        &Some(snapshot.clone()),
        &Some(harness_commit.clone()),
    )
    .await;
    assert!(
        !matches,
        "the harness's commit carries an edit the reviewer never saw"
    );

    // The approval that must be recorded instead: the pre-commit HEAD.
    assert_ne!(
        harness_commit, pre_commit_head,
        "the harness's commit moved HEAD past the review"
    );
    // An unknown snapshot or head is equally unapprovable.
    assert!(
        !__test_commit_matches_snapshot(repo.path(), &None, &Some(harness_commit.clone())).await
    );
    assert!(!__test_commit_matches_snapshot(repo.path(), &Some(snapshot.clone()), &None).await);
    // A planted value that is not a git object id never reaches git as a
    // revision argument.
    assert!(
        !__test_commit_matches_snapshot(
            repo.path(),
            &Some("--output=pwned".to_string()),
            &Some(harness_commit),
        )
        .await
    );
}

/// The harness's own commit is only the reviewed tree while the tree is still
/// uncommitted: the reviewer's edit sits in the worktree, the security review
/// completes, and nothing touches it afterwards. The approval must then name
/// that commit, so the audited code is never re-audited on the next revision.
#[tokio::test]
async fn a_reviewed_uncommitted_tree_approves_the_harness_commit() {
    let repo = common::TestRepo::new("approval-dirty-unchanged");
    common::git(repo.path(), &["checkout", "-q", "-b", "worker-dirty"]);
    std::fs::write(repo.path().join("before.rs"), "// before\n").unwrap();
    common::git(repo.path(), &["add", "-A"]);
    common::git(repo.path(), &["commit", "-q", "-m", "before"]);
    let pre_commit_head = common::git(repo.path(), &["rev-parse", "HEAD"])
        .trim()
        .to_string();

    // What a security reviewer leaves behind: an edit in the working tree, with
    // no commit of its own -- exactly what the implementer's uncommitted work
    // looks like too.
    std::fs::write(repo.path().join("reviewed.rs"), "// reviewed\n").unwrap();
    let snapshot = __test_snapshot_worktree_tree(repo.path())
        .await
        .expect("the reviewer's tree is snapshot-able");
    let status = common::git(repo.path(), &["status", "--porcelain"]);
    assert!(
        status.contains("reviewed.rs"),
        "the reviewed edit must still be uncommitted for this test to mean anything"
    );

    // The harness commits the tree it reviewed; nothing moved it in between.
    common::git(repo.path(), &["add", "-A"]);
    common::git(repo.path(), &["commit", "-q", "-m", "harness commit"]);
    let harness_commit = common::git(repo.path(), &["rev-parse", "HEAD"])
        .trim()
        .to_string();

    assert!(
        __test_commit_matches_snapshot(
            repo.path(),
            &Some(snapshot),
            &Some(harness_commit.clone()),
        )
        .await,
        "the harness's commit is the reviewed tree when nothing moved it"
    );
    assert_ne!(
        harness_commit, pre_commit_head,
        "the harness's commit is not the HEAD the reviewer started from"
    );
}
