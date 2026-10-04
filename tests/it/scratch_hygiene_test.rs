//! The suite files no `swe-*` entry in the real scratch base.
//!
//! [`WorktreeGuard::new`] and [`WorkerPool::new`] resolve
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

use crate::common::{self, TestRepo};
use mini_swe_mcp::worktree::{ScratchRoot, WorktreeGuard, swe_base_dir};

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

/// A worker checkout a fixture created is removed together with the private
/// scratch and legacy target the sandbox derives from its leaf name.
#[test]
fn a_dropped_worker_checkout_takes_its_private_scratch_with_it() {
    let repo = TestRepo::new("hygiene-guard");
    let leaf = format!("swe-wt-hygiene-{}-{}", std::process::id(), unique_nanos());
    let guard_root = repo.path().with_extension("worktrees");
    let companions = companions_of(&leaf);

    let guard = {
        let id = leaf.trim_start_matches("swe-wt-").to_string();
        std::fs::create_dir_all(&guard_root).expect("create the scratch root");
        let guard = WorktreeGuard::new_in(
            &ScratchRoot::new(&guard_root),
            repo.path(),
            &id,
        )
        .expect("worktree creation failed");
        assert_eq!(
            guard.path.file_name().and_then(|n| n.to_str()),
            Some(leaf.as_str()),
            "precondition: the checkout carries the leaf the companion is keyed on"
        );
        // The sandbox derives the companion from the checkout path itself.
        std::fs::create_dir_all(&companions[0]).expect("create the derived scratch");
        guard
    };
    assert!(exists(&companions[0]), "precondition: the companion exists");
    drop(guard);
    mini_swe_mcp::worktree::remove_scratch_root_worktrees(&guard_root);
    let _ = std::fs::remove_dir_all(&guard_root);

    for path in &companions {
        assert!(
            !exists(path),
            "{} outlived the checkout that derived it",
            path.display()
        );
    }
}

/// A pool handed an explicit scratch root files nothing in the real base.
///
/// The steer log is the durable artifact that exposed this: the log's own
/// nonce makes it one file per test run, so a pool on the default root wrote
/// `swe-wt-<id>.steer-log.jsonl` straight into the operator's `/var/tmp`.
#[test]
fn a_pool_on_an_explicit_root_files_no_row_in_the_real_base() {
    let leaf = format!("hygiene-steer-log-{}-{}", std::process::id(), unique_nanos());
    let steer_log = swe_base_dir().join(format!("swe-wt-{leaf}.steer-log.jsonl"));

    let root = common::TempDir::new_in_tmp("hygiene-pool");
    let pool = mini_swe_mcp::pool::WorkerPool::with_scratch(
        1,
        "http://localhost:1".to_string(),
        "test-key".to_string(),
        ScratchRoot::new(root.path()),
    );
    assert_ne!(
        pool.scratch_root().path(),
        swe_base_dir(),
        "precondition: the pool runs on its own root, not the real base"
    );
    drop(pool);

    assert!(
        !exists(&steer_log),
        "{} was filed in the real base by a pool that had its own root",
        steer_log.display()
    );
}

fn unique_nanos() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock before the epoch")
        .as_nanos()
}
