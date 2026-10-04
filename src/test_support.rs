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
    /// The root this scratch was created under: the directory the derived
    /// `swe-tmp-<leaf>` companions are filed in, and the only directory this
    /// scratch may reclaim them from.
    root: crate::worktree::ScratchRoot,
    /// The paths handed out by [`Self::subdir`], each with the scratch root
    /// that owns it. The runner derives a `swe-tmp-<leaf>` companion from
    /// whichever path it is handed, so each one is dropped with the scratch
    /// that created it rather than left behind.
    subdirs: std::sync::Mutex<Vec<(PathBuf, crate::worktree::ScratchRoot)>>,
}

impl TestScratch {
    /// Create a fresh, unique worktree directory.
    pub(crate) fn new(tag: &str) -> Self {
        let path = unique_path(tag);
        std::fs::create_dir_all(&path)
            .unwrap_or_else(|e| panic!("create test scratch {}: {e}", path.display()));
        let root =
            crate::worktree::ScratchRoot::new(path.parent().unwrap_or_else(|| Path::new(".")));
        Self {
            path,
            subdirs: std::sync::Mutex::new(Vec::new()),
            root,
        }
    }

    /// Own an existing path: nothing is created, but the path and its
    /// companions are removed on drop.
    ///
    /// Used for unique names a test needs to build itself and for the
    /// fail-closed test whose worktree must stay absent.
    pub(crate) fn own(path: PathBuf) -> Self {
        let root =
            crate::worktree::ScratchRoot::new(path.parent().unwrap_or_else(|| Path::new(".")));
        Self {
            path,
            subdirs: std::sync::Mutex::new(Vec::new()),
            root,
        }
    }

    /// A path that is deliberately absent, for tests pinning the fail-closed
    /// path (a command against a worktree that no longer exists).
    pub(crate) fn missing(tag: &str) -> Self {
        let path = unique_path(tag);
        let _ = std::fs::remove_dir_all(&path);
        let root =
            crate::worktree::ScratchRoot::new(path.parent().unwrap_or_else(|| Path::new(".")));
        Self {
            path,
            subdirs: std::sync::Mutex::new(Vec::new()),
            root,
        }
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
        let mut subdirs = self.subdirs.lock().unwrap_or_else(|e| e.into_inner());
        if !subdirs.iter().any(|(p, _)| *p == path) {
            subdirs.push((path.clone(), self.root.clone()));
        }
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
        // enough. `remove_target_dirs_in` also tolerates a path that is absent.
        //
        // Every subdirectory this scratch handed out is a path the runner can be
        // given too, so each one's own `swe-tmp-<leaf>` companion goes with it.
        // Without this a test that calls [`TestScratch::subdir`] with a fixed
        // name ("worktree", "target") would leave one `swe-tmp-<name>` entry
        // behind per run.
        //
        // Reclamation is scoped to the root this scratch was created under. The
        // leaf of a subdirectory is whatever the *test* named it, so resolving
        // against the default root instead would delete a fixed, predictable
        // `swe-tmp-worktree` / `swe-tmp-target` in the shared scratch base --
        // which is where a sibling agent's private scratch lives.
        let subdirs = std::mem::take(&mut *self.subdirs.lock().unwrap_or_else(|e| e.into_inner()));
        for (subdir, root) in subdirs {
            crate::worktree::remove_target_dirs_in(&root, &subdir);
        }
        crate::worktree::remove_target_dirs_in(&self.root, &self.path);
        crate::cache::remove_build_dir_leases(&self.path);
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

/// A `swe-*` entry in `dir` whose name mentions `needle`.
///
/// Scoped to the exact name the guard under test derives, never the whole base:
/// a test binary runs its tests in parallel and sibling tests legitimately
/// create entries beside it, so a base-wide scan would be both racy and
/// unfalsifiable. Both callers assert a *named* leak is gone, which is what
/// makes the assertion specific enough to fail.
#[cfg(test)]
pub(crate) fn swe_entry_for(dir: &Path, needle: &str) -> Option<PathBuf> {
    std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .map(|entry| entry.path())
        .find(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("swe-") && name.contains(needle))
        })
}

/// Run `body` and report whether it unwound, with the default hook silenced so a
/// deliberate panic does not print a backtrace that reads like a real failure.
///
/// Scratch hygiene is only observable on the failure path -- a `Drop` guard is
/// indistinguishable from a sequential cleanup that happens to run -- so a test
/// that pins it has to make the fixture actually unwind.
#[cfg(test)]
pub(crate) fn panics<F: FnOnce()>(body: F) -> bool {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(body)).is_err();
    std::panic::set_hook(previous);
    unwound
}
