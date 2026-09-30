//! Build-directory leases: one live worker per directory.
//!
//! A build dir used to be picked by admission slot, so two live workers of one
//! repository alternated in it and cargo's uplifted binaries overwrote each
//! other. These tests pin the replacement: a directory is leased exclusively
//! for the leasing worker's lifetime, later workers reuse freed directories,
//! and a leased directory is invisible to the sweep.

use mini_swe_mcp::cache::BuildDirLease;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// One repository's pool of build dirs, with the dirs this test leased.
///
/// The repository root is unique per test, so the pool names derived from it
/// are unique too and the cleanup below only ever removes this test's dirs.
struct Pool {
    repo: PathBuf,
    leased: Vec<PathBuf>,
}

impl Pool {
    fn new(tag: &str) -> Self {
        let repo = std::env::temp_dir()
            .join(format!("swe-lease-repo-{tag}-{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&repo).expect("repository root must be creatable");
        Self {
            repo,
            leased: Vec::new(),
        }
    }

    /// Lease a dir the way a worker's first heavy command does. The returned
    /// guard is the worker's claim on it: dropping it ends the worker.
    fn lease(&mut self) -> BuildDirLease {
        let lease = BuildDirLease::acquire(&self.repo).expect("a worker must lease a dir");
        self.leased.push(lease.dir().to_path_buf());
        lease
    }

    /// Whether `dir` is free for another worker: the lock file a lease holds is
    /// exactly what the sweep probes to decide a dir is busy.
    fn is_free(&self, dir: &Path) -> bool {
        let file = match std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(dir.join(".swe-target.lease"))
        {
            Ok(file) => file,
            Err(_) => return false,
        };
        // SAFETY: `file` owns a live descriptor for the duration of the call.
        let locked =
            unsafe { libc::flock(std::os::fd::AsRawFd::as_raw_fd(&file), libc::LOCK_EX | libc::LOCK_NB) };
        if locked == 0 {
            // SAFETY: releasing a lock this call just took.
            unsafe { libc::flock(std::os::fd::AsRawFd::as_raw_fd(&file), libc::LOCK_UN) };
            true
        } else {
            false
        }
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        for dir in &self.leased {
            let _ = std::fs::remove_dir_all(dir);
        }
        let _ = std::fs::remove_dir_all(&self.repo);
    }
}

/// Two workers of one repository that are live at the same time must never
/// share a build directory: that is the overwrite the bug report describes.
#[test]
fn two_live_workers_of_one_repo_never_share_a_build_dir() {
    let mut pool = Pool::new("shared");
    let first = pool.lease();
    let second = pool.lease();

    assert_ne!(
        first.dir(),
        second.dir(),
        "two live workers of one repository shared {}",
        first.dir().display()
    );
    assert!(!pool.is_free(first.dir()), "the first worker's dir must stay locked");
    assert!(!pool.is_free(second.dir()), "the second worker's dir must stay locked");
}

/// The warm-cache benefit survives: a directory freed by a finished worker is
/// the one the next worker takes, dependencies and all.
#[test]
fn a_released_dir_is_reused_by_the_next_worker() {
    let mut pool = Pool::new("reuse");
    let first = pool.lease();
    let dir = first.dir().to_path_buf();
    std::fs::create_dir_all(dir.join("debug")).unwrap();
    std::fs::write(dir.join("debug").join("deps"), b"warm").unwrap();
    // The worker ends: its claim on the dir goes away, the dir stays on disk.
    drop(first);

    let second = pool.lease();
    assert_eq!(
        second.dir(),
        dir,
        "a freed dir must be reused, not orphaned next to a fresh one"
    );
    assert_eq!(
        std::fs::read(second.dir().join("debug").join("deps")).unwrap(),
        b"warm",
        "the reused dir must keep its warm contents"
    );
}

/// A lease is held for the whole worker lifetime, so alternating heavy and
/// light commands cannot hand the directory to another worker in between.
#[test]
fn a_lease_survives_heavy_and_light_alternation() {
    let mut pool = Pool::new("alternating");
    let mut held = Vec::new();
    for _ in 0..4 {
        // A heavy command leases; a light command only reads what is leased.
        let lease = pool.lease();
        assert!(!pool.is_free(lease.dir()), "a live lease must be held");
        held.push(lease);
    }
    let unique: BTreeSet<&Path> = held.iter().map(|lease| lease.dir()).collect();
    assert_eq!(
        unique.len(),
        held.len(),
        "every live worker of the repository needs its own dir"
    );

    // Ending one worker releases exactly its directory, and nothing else.
    let ended = held.remove(1).dir().to_path_buf();
    assert!(pool.is_free(&ended), "the ended worker's dir must be free again");
    for lease in &held {
        assert!(!pool.is_free(lease.dir()), "another live worker lost its dir");
    }
}

/// Dropping the future that owns the guard releases the directory: a killed or
/// aborted worker must not pin a dir for the rest of the hub's life.
#[test]
fn dropping_the_worker_releases_its_dir() {
    let mut pool = Pool::new("abort");
    let lease = pool.lease();
    let dir = lease.dir().to_path_buf();
    assert!(!pool.is_free(&dir), "a live worker must hold its dir");

    // The lease is dropped here, exactly as an abort drops the worker future.
    drop(lease);
    assert!(pool.is_free(&dir), "a dropped worker must release its dir");

    let next = pool.lease();
    assert_eq!(next.dir(), dir, "the released dir must go back into the pool");
}

/// The sweep must never take a directory a live worker is building in.
#[test]
fn a_leased_dir_is_never_swept() {
    let mut pool = Pool::new("swept");
    let lease = pool.lease();
    let dir = lease.dir().to_path_buf();
    std::fs::create_dir_all(dir.join("target")).unwrap();
    std::fs::write(dir.join("target").join("keep"), b"artifact").unwrap();

    assert!(
        !pool.is_free(&dir),
        "the sweep must see a leased dir as busy, or it will evict a live build"
    );
    assert!(dir.join("target").join("keep").exists());
    drop(lease);
}
