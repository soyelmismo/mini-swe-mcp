//! Scratch directories for unit tests.
//!
//! A test that runs the sandbox needs a worktree the runner can derive its
//! private `swe-tmp-<name>` scratch from, and that companion is filed next to
//! the scratch base rather than inside the worktree. A bare `create_dir_all`
//! therefore leaves it behind even when the test removes the worktree itself.
//! [`TestScratch`] owns the worktree and drops both the tree and every
//! `swe-tmp-*` / `swe-target-*` companion derived from its leaf name, so a test
//! leaves nothing behind even when it panics.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Keeps two calls in the same nanosecond distinct.
static COUNTER: AtomicU64 = AtomicU64::new(0);

/// A unique scratch worktree under the scratch base, removed on drop together
/// with the private scratch the runner derives from its leaf name.
pub(crate) struct TestScratch {
    path: PathBuf,
}

impl TestScratch {
    /// Create a fresh, unique worktree directory.
    pub(crate) fn new(tag: &str) -> Self {
        let path = unique_path(tag);
        std::fs::create_dir_all(&path)
            .unwrap_or_else(|e| panic!("create test scratch {}: {e}", path.display()));
        Self { path }
    }

    /// Own an existing path: nothing is created, but the path and its
    /// companions are removed on drop.
    ///
    /// Used for unique names a test needs to build itself and for the
    /// fail-closed test whose worktree must stay absent.
    pub(crate) fn own(path: PathBuf) -> Self {
        Self { path }
    }

    /// A path that is deliberately absent, for tests pinning the fail-closed
    /// path (a command against a worktree that no longer exists).
    pub(crate) fn missing(tag: &str) -> Self {
        let path = unique_path(tag);
        let _ = std::fs::remove_dir_all(&path);
        Self { path }
    }

    /// The worktree path itself. Create it first for [`own`](Self::own) and
    /// [`missing`](Self::missing).
    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    /// Create and return a fresh subdirectory, removed with the worktree.
    pub(crate) fn subdir(&self, name: &str) -> PathBuf {
        let path = self.path.join(name);
        std::fs::create_dir_all(&path)
            .unwrap_or_else(|e| panic!("create test scratch {}: {e}", path.display()));
        path
    }
}

fn unique_path(tag: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock is before the unix epoch")
        .as_nanos();
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    crate::worktree::swe_base_dir().join(format!("{tag}-{}-{nanos}-{n}", std::process::id()))
}

impl Drop for TestScratch {
    fn drop(&mut self) {
        // The runner files private scratch next to the scratch base, keyed by
        // the worktree's leaf name, so removing the worktree alone is not
        // enough. `remove_target_dirs` also tolerates a path that is absent.
        crate::worktree::remove_target_dirs(&self.path);
        let _ = std::fs::remove_dir_all(&self.path);
    }
}
