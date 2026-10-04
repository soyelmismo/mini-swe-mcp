//! The suite files no `swe-*` entry in the real scratch base.
//!
//! `WorktreeGuard::new` and `WorkerPool::new` resolve
//! [`swe_base_dir`](mini_swe_mcp::worktree::swe_base_dir) — on a host without
//! `SWE_TEMP_DIR` that is `/var/tmp` itself. A fixture that uses either seam
//! without cleaning up leaves a checkout, a `swe-tmp-<leaf>` private scratch or
//! a `swe-wt-<id>.history.jsonl` row in the operator's real scratch directory,
//! once per run, for as long as nobody prunes it.
//!
//! The property is pinned per fixture rather than by scanning the base: the
//! suite runs in parallel and other tests legitimately create entries there, so
//! a whole-base scan would be both racy and unfalsifiable. Each test below names
//! the exact companions its fixture derives and asserts they are gone.
//!
//! A pool's own scratch root is a different property, covered where the pool is
//! built: `consolidate_steer_log_hardening_test` and the `IsolatedPool` seam.

use crate::common::TestRepo;
use mini_swe_mcp::worktree::swe_base_dir;

/// Whether `dir` exists, without following a symlink.
fn exists(dir: &std::path::Path) -> bool {
    dir.symlink_metadata().is_ok()
}

/// The private scratch and legacy target dirs the runner derives from a
/// worktree whose leaf name is `leaf`, in the real base.
fn companions_of(leaf: &str) -> Vec<std::path::PathBuf> {
    let base = swe_base_dir();
    vec![
        base.join(format!("swe-tmp-{leaf}")),
        base.join(format!("swe-target-{leaf}")),
    ]
}

/// A fixture repository drops the private scratch the runner derives from it.
///
/// `scratch_dir` is keyed on the *leaf* of whatever path a step is handed, so a
/// repository a worker ran against owns a `swe-tmp-<leaf>` entry in the base
/// whether or not a worktree was ever branched from it. Removing the repository
/// directory alone does not remove it.
#[test]
fn a_dropped_test_repository_takes_its_private_scratch_with_it() {
    let repo = TestRepo::new("hygiene-repo");
    let leaf = repo
        .path()
        .file_name()
        .and_then(|n| n.to_str())
        .expect("the fixture repository has a leaf name")
        .to_string();
    let companions = companions_of(&leaf);

    // A step against the repository derives the companion the assertion is
    // about; creating it directly is the same derivation, minus the model.
    std::fs::create_dir_all(&companions[0]).expect("create the derived scratch");
    assert!(exists(&companions[0]), "precondition: the companion exists");

    drop(repo);

    for path in &companions {
        assert!(
            !exists(path),
            "{} outlived the fixture that derived it",
            path.display()
        );
    }
}

/// A checkout made through [`TestRepo::guard`] leaves nothing behind.
///
/// The seam builds the checkout under a scratch root the fixture owns, rather
/// than the real base `WorktreeGuard::new` resolves. Dropping the fixture must
/// take that root, the checkout in it, and the companions derived from both --
/// otherwise a suite that guards a worktree per test accumulates one checkout
/// and one `swe-tmp-<leaf>` in the operator's scratch directory per run.
#[test]
fn a_dropped_checkout_fixture_leaves_no_checkout_and_no_companion() {
    let repo = TestRepo::new("hygiene-guard");
    let guard_root = repo.path().with_extension("worktrees");

    let guard = repo.guard("hygiene-checkout");
    let leaf = guard
        .path
        .file_name()
        .and_then(|n| n.to_str())
        .expect("the checkout has a leaf name")
        .to_string();
    let companions = companions_of(&leaf);
    let checkout = guard.path.clone();

    // The sandbox derives the private scratch from the checkout path itself.
    std::fs::create_dir_all(&companions[0]).expect("create the derived scratch");
    assert!(exists(&companions[0]), "precondition: the companion exists");
    assert!(exists(&checkout), "precondition: the checkout exists");
    drop(guard);

    for path in &companions {
        assert!(
            !exists(path),
            "{} outlived the fixture that derived it",
            path.display()
        );
    }

    drop(repo);
    assert!(
        !exists(&checkout),
        "{} outlived its fixture",
        checkout.display()
    );
    assert!(
        !exists(&guard_root),
        "{} outlived its fixture",
        guard_root.display()
    );
}
