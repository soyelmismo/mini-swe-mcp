//! Reaping the processes a worker's commands leave behind.
//!
//! A process-group kill only reaches processes that stayed in the group. A
//! shell can defeat it two ways: `cmd &` keeps the job in the group (so the
//! step's own teardown in [`super::exec`] ends it), while `setsid cmd &` or a
//! double fork moves the job into a session of its own and reparents it to
//! init, where nothing but its working directory still ties it to the worker
//! that spawned it. That is the leak this module closes: at worker end -- and
//! again when a crashed hub's worktrees are recovered -- every process of this
//! uid whose working directory is inside one of the worker's directories is
//! taken down.
//!
//! The sweep is deliberately narrow:
//!
//! * it signals individual pids, never process groups, because a detached
//!   job's group may contain processes belonging to no worker at all;
//! * it never signals this process or any of its ancestors, so the hub and the
//!   shell that started it survive however their working directory is spelled;
//! * it acts only on a pid whose uid it could read and whose working directory
//!   it could resolve -- an unreadable `/proc` entry is left alone rather than
//!   guessed at.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tracing::warn;

/// Grace a swept process gets to exit on `SIGTERM` before the `SIGKILL`.
const TERM_GRACE: Duration = Duration::from_millis(300);

/// How long the sweep waits for the `SIGKILL` to take effect.
const KILL_GRACE: Duration = Duration::from_millis(200);

/// Poll interval while waiting for a signalled process to disappear.
const POLL: Duration = Duration::from_millis(20);

/// Bound on the ancestor walk, so a corrupt `/proc` cannot spin here.
const MAX_ANCESTORS: usize = 64;

/// Take down every process of this uid whose working directory is inside one of
/// `dirs`, and report how many were signalled.
///
/// `dirs` is the whole definition of "belongs to this worker": the worktree, its
/// private scratch dir, its target dirs and its leased build dir. Nothing
/// outside them is ever signalled, so a live sibling worker is never touched.
///
/// One log line is emitted per sweep that killed something, so a leak stays
/// visible in the worker's log instead of silently burning CPU somewhere else.
pub(crate) fn sweep_worker_processes(worker_id: &str, dirs: &[PathBuf]) -> usize {
    let targets = processes_in_dirs(dirs);
    if targets.is_empty() {
        return 0;
    }
    for pid in &targets {
        signal_pid(*pid, libc::SIGTERM);
    }
    wait_until_gone(&targets, TERM_GRACE);
    let survivors: Vec<u32> = targets
        .iter()
        .copied()
        .filter(|pid| pid_is_alive(*pid))
        .collect();
    for pid in &survivors {
        signal_pid(*pid, libc::SIGKILL);
    }
    wait_until_gone(&survivors, KILL_GRACE);
    warn!(
        worker = %worker_id,
        count = targets.len(),
        "Killed processes left running in the worker's directories"
    );
    targets.len()
}

/// The directories a worker's commands may run in, derived from its worktree
/// path: the worktree itself, its private scratch dir and its target dirs.
///
/// Shared by the worker-end sweep and crash recovery so both agree on which
/// processes belong to a worker, and so neither has to know the naming scheme.
pub(crate) fn worker_dirs(worktree: &Path) -> Vec<PathBuf> {
    let mut dirs = vec![worktree.to_path_buf()];
    if let Some(name) = worktree.file_name().and_then(|n| n.to_str()) {
        for base in crate::worktree::swe_base_dirs() {
            dirs.push(base.join(format!("swe-tmp-{name}")));
            dirs.push(base.join(format!("swe-target-{name}")));
        }
    }
    dirs
}

/// Pids of this uid's processes whose working directory is inside one of
/// `dirs`, excluding this process and every ancestor of it.
///
/// The sweep's targeting rule, exposed so tests can assert on it directly
/// instead of on the side effect of a kill.
#[doc(hidden)]
pub fn processes_in_dirs(dirs: &[PathBuf]) -> Vec<u32> {
    let protected = protected_pids();
    let dirs: Vec<PathBuf> = dirs.iter().map(|dir| resolve_dir(dir)).collect();
    process_pids()
        .into_iter()
        .filter(|pid| !protected.contains(pid))
        .filter(|pid| cwd_is_inside(*pid, &dirs))
        .collect()
}

/// Pids of the live processes in the process group `pgid`.
///
/// A snapshot rather than a `kill(-pgid, 0)` probe: on the clean-exit path the
/// group leader has already been reaped, so nothing pins the group id any more
/// and a recycled id must never be mistaken for this one.
pub(crate) fn process_group_members(pgid: u32) -> Vec<u32> {
    process_pids()
        .into_iter()
        .filter(|pid| process_group_of(*pid) == Some(pgid))
        .collect()
}

/// Whether `pid` still exists.
pub(crate) fn pid_is_alive(pid: u32) -> bool {
    #[cfg(unix)]
    {
        // SAFETY: `kill` takes no pointers; signal 0 only probes for existence.
        unsafe { libc::kill(pid as libc::pid_t, 0) == 0 }
    }
    #[cfg(not(unix))]
    {
        let _ = pid;
        false
    }
}

/// This process and every ancestor of it: the hub, the shell that started it
/// and the service manager above that. A sweep must never signal any of them,
/// however their working directory happens to be spelled.
fn protected_pids() -> Vec<u32> {
    let mut protected = vec![std::process::id()];
    let mut cursor = std::process::id();
    while protected.len() < MAX_ANCESTORS {
        match parent_pid(cursor) {
            Some(parent) if parent > 0 && !protected.contains(&parent) => {
                protected.push(parent);
                cursor = parent;
            }
            _ => break,
        }
    }
    protected
}

/// The parent pid `/proc/<pid>/status` reports, when it can be read.
fn parent_pid(pid: u32) -> Option<u32> {
    let status = std::fs::read_to_string(format!("/proc/{pid}/status")).ok()?;
    status
        .lines()
        .find_map(|line| line.strip_prefix("PPid:")?.trim().parse().ok())
}

/// Every numeric entry of `/proc`, i.e. every process visible to this uid.
fn process_pids() -> Vec<u32> {
    let Ok(entries) = std::fs::read_dir("/proc") else {
        return Vec::new();
    };
    entries
        .flatten()
        .filter_map(|entry| entry.file_name().to_str()?.parse().ok())
        .collect()
}

/// The process group `/proc/<pid>/stat` reports for `pid` (field 5).
///
/// The executable name in field 2 is parenthesised and may itself contain
/// spaces or parentheses, so the fields after it are addressed relative to the
/// *last* `)` in the line.
fn process_group_of(pid: u32) -> Option<u32> {
    let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    let (_, tail) = stat.rsplit_once(')')?;
    // Fields after the name: state, ppid, pgrp.
    tail.split_whitespace().nth(2)?.parse().ok()
}

/// The real uid `/proc/<pid>/status` reports, when it can be read.
fn uid_of(pid: u32) -> Option<u32> {
    let status = std::fs::read_to_string(format!("/proc/{pid}/status")).ok()?;
    status.lines().find_map(|line| {
        line.strip_prefix("Uid:")?
            .split_whitespace()
            .next()?
            .parse()
            .ok()
    })
}

/// Whether `pid`'s working directory is inside one of `dirs`.
fn cwd_is_inside(pid: u32, dirs: &[PathBuf]) -> bool {
    if uid_of(pid) != Some(current_uid()) {
        return false;
    }
    match cwd_of(pid) {
        Some(cwd) => dirs.iter().any(|dir| cwd.starts_with(dir)),
        None => false,
    }
}

/// The working directory `/proc/<pid>/cwd` reports for `pid`.
///
/// A directory that has since been deleted is reported with a ` (deleted)`
/// suffix, which is stripped so the path still matches the directory it came
/// from: a sweep that runs after its worktree was removed must still find the
/// processes that were running in it.
fn cwd_of(pid: u32) -> Option<PathBuf> {
    let link = std::fs::read_link(format!("/proc/{pid}/cwd")).ok()?;
    let text = link.to_string_lossy();
    let trimmed = text.strip_suffix(" (deleted)").unwrap_or(&text);
    Some(PathBuf::from(trimmed))
}

/// Resolve `dir` the way `/proc/<pid>/cwd` spells paths: through every symlink
/// in the prefix, so a base directory that is itself a link still matches.
///
/// A directory that does not exist yet cannot hold a process, but its *name*
/// still has to match a deleted-directory cwd, so the unresolved path is kept
/// as the fallback rather than dropped.
fn resolve_dir(dir: &Path) -> PathBuf {
    if let Ok(resolved) = std::fs::canonicalize(dir) {
        return resolved;
    }
    match (dir.parent(), dir.file_name()) {
        (Some(parent), Some(name)) => std::fs::canonicalize(parent)
            .map(|resolved| resolved.join(name))
            .unwrap_or_else(|_| dir.to_path_buf()),
        _ => dir.to_path_buf(),
    }
}

/// This process's real uid, or `u32::MAX` when it cannot be read (in which case
/// no process ever matches and the sweep stays inert rather than guessing).
fn current_uid() -> u32 {
    #[cfg(unix)]
    {
        // SAFETY: `getuid` takes no arguments and cannot fail.
        unsafe { libc::getuid() as u32 }
    }
    #[cfg(not(unix))]
    {
        u32::MAX
    }
}

/// Wait until every pid in `pids` is gone or `grace` elapses.
fn wait_until_gone(pids: &[u32], grace: Duration) {
    let deadline = Instant::now() + grace;
    while pids.iter().any(|pid| pid_is_alive(*pid)) {
        if Instant::now() >= deadline {
            break;
        }
        std::thread::sleep(POLL);
    }
}

/// Send `sig` to one pid, ignoring every error: a process that exited between
/// the snapshot and the signal is exactly the outcome being waited for.
#[cfg(unix)]
fn signal_pid(pid: u32, sig: libc::c_int) {
    // SAFETY: `kill` takes no pointers and signals only this pid.
    unsafe { libc::kill(pid as libc::pid_t, sig) };
}

/// Non-unix stub: there is no `/proc` to sweep and no signal to send.
#[cfg(not(unix))]
fn signal_pid(_pid: u32, _sig: i32) {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::{Command, Stdio};

    /// A scratch directory under the same base the crate's worktrees use, so a
    /// test's paths are shaped like a real worker's.
    fn worker_dir(tag: &str) -> PathBuf {
        let dir =
            crate::worktree::swe_base_dir().join(format!("swe-reap-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("scratch dir must be creatable");
        dir
    }

    /// Spawn `setsid sleep 300` with `dir` as its working directory: detached
    /// from every process group this test owns, exactly like the job a model
    /// backgrounds with `(setsid sleep 300 &)`.
    fn detached_sleeper(dir: &Path) -> std::process::Child {
        Command::new("setsid")
            .args(["sleep", "300"])
            .current_dir(dir)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("setsid must spawn")
    }

    /// Wait until `dir` holds at least one process, so a test never asserts on
    /// a sweep that ran before the sleeper existed.
    fn await_process_in(dir: &Path) {
        for _ in 0..200 {
            if !processes_in_dirs(&[dir.to_path_buf()]).is_empty() {
                return;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        panic!("no process ever appeared in {}", dir.display());
    }

    /// A detached job inside the worker's directories is taken down.
    #[test]
    fn a_detached_process_inside_the_directories_is_killed() {
        let dir = worker_dir("inside");
        let mut sleeper = detached_sleeper(&dir);
        await_process_in(&dir);

        let dirs = [dir.clone()];
        let killed = sweep_worker_processes("worker-test", &dirs);

        assert_eq!(killed, 1, "the detached sleeper must be signalled");
        // Reaped through `try_wait`, because a killed child of this test stays
        // a zombie until it is waited on and a zombie still answers `kill -0`.
        let mut gone = false;
        for _ in 0..100 {
            if sleeper
                .try_wait()
                .expect("try_wait must not fail")
                .is_some()
            {
                gone = true;
                break;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        assert!(
            gone,
            "the sweep must leave no process behind in the worker's directories"
        );
        assert!(processes_in_dirs(&dirs).is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A process outside the worker's directories is never signalled, and the
    /// sweep reports nothing.
    #[test]
    fn a_process_outside_the_directories_is_never_signalled() {
        let inside = worker_dir("scope-inside");
        let outside = worker_dir("scope-outside");
        let mut sleeper = detached_sleeper(&outside);
        let pid = sleeper.id();
        await_process_in(&outside);

        let dirs = [inside.clone()];
        let killed = sweep_worker_processes("worker-test", &dirs);

        assert_eq!(
            killed, 0,
            "nothing outside the worker's directories may be killed"
        );
        assert!(
            pid_is_alive(pid),
            "a process outside the worker's directories must survive the sweep"
        );
        signal_pid(pid, libc::SIGKILL);
        let _ = sleeper.wait();
        let _ = std::fs::remove_dir_all(&inside);
        let _ = std::fs::remove_dir_all(&outside);
    }

    /// The sweep's own process and its ancestors are never targets, whatever
    /// their working directory happens to be.
    #[test]
    fn the_sweep_never_signals_itself_or_its_ancestors() {
        let protected = protected_pids();

        assert!(
            protected.contains(&std::process::id()),
            "the sweeping process must never be a target"
        );
        assert!(
            protected.len() > 1,
            "the ancestors that started the hub must be protected too: {protected:?}"
        );
        assert!(
            protected.iter().all(|pid| *pid != 0),
            "pid 0 addresses a group, never a process: {protected:?}"
        );
    }

    /// A group member is found by its process group, which is what the step-end
    /// termination waits on before escalating to `SIGKILL`.
    #[test]
    fn process_group_members_reports_the_group() {
        let mut child = Command::new("sleep")
            .arg("300")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("sleep must spawn");
        let pid = child.id();
        // A direct child shares this test's process group.
        assert!(
            process_group_members(process_group_of(pid).expect("a process group")).contains(&pid),
            "the child must be a member of its own process group"
        );
        signal_pid(pid, libc::SIGKILL);
        let _ = child.wait();
    }
}
