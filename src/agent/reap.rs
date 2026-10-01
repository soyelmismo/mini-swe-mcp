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
//! taken down if its ancestry leads to the hub or an orphan adopter, not a
//! terminal/login session.
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

use std::collections::BTreeMap;
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
/// outside them is ever signalled. Processes from unrelated terminal/login
/// sessions are excluded even when their cwd is inside these directories.
///
/// One log line is emitted per sweep that killed something, so a leak stays
/// visible in the worker's log instead of silently burning CPU somewhere else.
///
/// The whole call is bounded by [`TERM_GRACE`] plus [`KILL_GRACE`], which is
/// what makes it safe to run from a `Drop`: a worker's teardown already blocks
/// on `git`, and half a second more is the price of not leaving a build behind.
pub(crate) fn sweep_worker_processes(worker_id: &str, dirs: &[PathBuf]) -> usize {
    sweep_owned_processes(worker_id, dirs, std::process::id()).len()
}

/// One process the sweep had to kill, named for the report that names it.
#[derive(Debug, Clone)]
pub(crate) struct ReapedProcess {
    pub pid: u32,
    pub command: String,
}

/// [`sweep_worker_processes`] with every killed process named, so the
/// side-effect audit can tell the model exactly what it left running.
pub(crate) fn sweep_worker_processes_named(
    worker_id: &str,
    dirs: &[PathBuf],
) -> Vec<ReapedProcess> {
    let hub_pid = std::process::id();
    // A command that just returned can still have children on their way out
    // (test binaries, build helpers, zombies awaiting their parent): only a
    // process that is still there after a short settle, and not a zombie, was
    // left behind. Its name is read before any signal, so it is never empty.
    let first = owned_processes_in_dirs(dirs, hub_pid);
    if first.is_empty() {
        return Vec::new();
    }
    std::thread::sleep(SETTLE);
    let lingering: Vec<ReapedProcess> = owned_processes_in_dirs(dirs, hub_pid)
        .into_iter()
        .filter(|pid| first.contains(pid) && !is_zombie(*pid))
        .map(|pid| ReapedProcess {
            pid,
            command: comm_of(pid),
        })
        .filter(|process| !process.command.is_empty())
        .collect();
    if lingering.is_empty() {
        return lingering;
    }
    sweep_owned_processes(worker_id, dirs, hub_pid);
    lingering
}

/// How long the audit lets a just-finished command's children exit on their
/// own before it counts them as left behind.
const SETTLE: Duration = Duration::from_millis(500);

/// Whether `pid` is a zombie (state `Z` in `/proc/<pid>/stat`): it already
/// exited and only waits to be reaped by its parent.
fn is_zombie(pid: u32) -> bool {
    std::fs::read_to_string(format!("/proc/{pid}/stat"))
        .ok()
        .and_then(|stat| {
            let after_comm = stat.rsplit_once(')')?.1;
            after_comm
                .split_whitespace()
                .next()
                .map(|state| state == "Z")
        })
        .unwrap_or(true)
}

/// `/proc/<pid>/comm`, or the empty string when it is no longer readable.
fn comm_of(pid: u32) -> String {
    std::fs::read_to_string(format!("/proc/{pid}/comm"))
        .map(|comm| comm.trim().to_string())
        .unwrap_or_default()
}

fn sweep_owned_processes(worker_id: &str, dirs: &[PathBuf], hub_pid: u32) -> Vec<u32> {
    let targets = owned_processes_in_dirs(dirs, hub_pid);
    if targets.is_empty() {
        return targets;
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
    targets
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
/// `dirs`, with worker/orphan ancestry and excluding this process and its
/// ancestors. Unrelated terminal/login sessions are never targets.
///
/// The sweep's targeting rule, exposed so tests can assert on it directly
/// instead of on the side effect of a kill.
#[doc(hidden)]
pub fn processes_in_dirs(dirs: &[PathBuf]) -> Vec<u32> {
    owned_processes_in_dirs(dirs, std::process::id())
}

fn owned_processes_in_dirs(dirs: &[PathBuf], hub_pid: u32) -> Vec<u32> {
    let protected = protected_pids();
    let dirs: Vec<PathBuf> = dirs.iter().map(|dir| resolve_dir(dir)).collect();
    let ancestry: BTreeMap<u32, (u32, String)> = process_pids()
        .into_iter()
        .filter_map(|pid| {
            let parent = parent_pid(pid)?;
            let comm = std::fs::read_to_string(format!("/proc/{pid}/comm")).ok()?;
            Some((pid, (parent, comm.trim().to_string())))
        })
        .collect();
    ancestry
        .keys()
        .copied()
        .filter(|pid| !protected.contains(pid))
        .filter(|pid| cwd_is_inside(*pid, &dirs))
        .filter(|pid| worker_ancestry(*pid, hub_pid, &ancestry, &protected))
        .collect()
}

/// A live hub descendant or an orphan adopted by init/systemd belongs to the
/// worker only if the ancestry has no terminal/login boundary. Merely reaching
/// init through a user's terminal does not make that terminal's children orphans.
/// Linux does not expose another process's PR_SET_CHILD_SUBREAPER flag in /proc;
/// unknown adopters are left alone rather than treated as service managers.
fn worker_ancestry(
    mut pid: u32,
    hub_pid: u32,
    ancestry: &BTreeMap<u32, (u32, String)>,
    protected: &[u32],
) -> bool {
    for _ in 0..MAX_ANCESTORS {
        if pid == hub_pid {
            return true;
        }
        if protected.contains(&pid) {
            return false;
        }
        let Some((parent, comm)) = ancestry.get(&pid) else {
            return false;
        };
        if login_boundary(comm) {
            return false;
        }
        if *parent == 1 {
            return true;
        }
        if let Some((_, parent_comm)) = ancestry.get(parent)
            && matches!(parent_comm.as_str(), "systemd" | "init")
        {
            return true;
        }
        if *parent == pid || *parent == 0 {
            return false;
        }
        pid = *parent;
    }
    false
}

fn login_boundary(comm: &str) -> bool {
    matches!(
        comm,
        "sshd"
            | "sshd-session"
            | "login"
            | "agetty"
            | "getty"
            | "su"
            | "sudo"
            | "xterm"
            | "uxterm"
            | "konsole"
            | "gnome-terminal-"
            | "gnome-terminal"
            | "kgx"
            | "xfce4-terminal"
            | "mate-terminal"
            | "alacritty"
            | "kitty"
            | "wezterm-gui"
            | "foot"
            | "urxvt"
            | "rxvt"
            | "tmux: server"
            | "screen"
    )
}

/// Pids of the live processes in the process group `pgid`.
///
/// A snapshot rather than a `kill(-pgid, 0)` probe: on the clean-exit path the
/// group leader has already been reaped, so nothing pins the group id any more
/// and a recycled id must never be mistaken for this one.
pub(crate) fn process_group_members(pgid: u32) -> Vec<u32> {
    // A pgid of 0 is "no group known", never a group to look for.
    if pgid == 0 {
        return Vec::new();
    }
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

    /// The audit only names processes that really stay: a child that exits on
    /// its own during the settle and a zombie are not "left behind", and a
    /// lingering one is reported with its name (read before it is killed).
    #[test]
    fn the_named_sweep_ignores_exiting_children_and_zombies() {
        let dir = std::env::temp_dir().join(format!("reap-settle-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("create the scratch dir");
        let dirs = vec![dir.clone()];

        let mut quick = std::process::Command::new("sleep")
            .arg("0.1")
            .current_dir(&dir)
            .spawn()
            .expect("spawn a short-lived child");
        let zombie = std::process::Command::new("true")
            .current_dir(&dir)
            .spawn()
            .expect("spawn a child left unreaped");
        std::thread::sleep(Duration::from_millis(50));
        assert!(
            sweep_worker_processes_named("settle-test", &dirs).is_empty(),
            "exiting children and zombies are not leftovers"
        );
        let _ = quick.wait();
        drop(zombie);

        let mut lingering = std::process::Command::new("sleep")
            .arg("30")
            .current_dir(&dir)
            .spawn()
            .expect("spawn a lingering child");
        let reaped = sweep_worker_processes_named("settle-test", &dirs);
        let _ = lingering.wait();
        assert_eq!(reaped.len(), 1, "{reaped:?}");
        assert_eq!(reaped[0].command, "sleep");
        let _ = std::fs::remove_dir_all(&dir);
    }
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

    /// A `setsid` session leader forked by a helper that exits at once, so the
    /// leader's parent is init rather than this test: the shape of a terminal
    /// the operator opened by hand.
    fn spawn_unrelated_shell(dir: &Path, pid_file: &Path) -> std::process::Child {
        // `setsid` detaches the leader into its own session, which is what a
        // terminal the operator opened by hand looks like from the outside.
        Command::new("setsid")
            .args(["bash", "-c", "sleep 300 & echo $! > $1; wait", "bash"])
            .arg(pid_file)
            .current_dir(dir)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("setsid must spawn")
    }

    /// Wait until `file` holds a pid, so a test never races the fork that
    /// writes it.
    fn wait_for_pid(file: &Path) -> u32 {
        for _ in 0..200 {
            if let Ok(text) = std::fs::read_to_string(file)
                && let Ok(pid) = text.trim().parse()
            {
                return pid;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        panic!("no pid ever appeared in {}", file.display());
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
        let killed = sweep_owned_processes("worker-test", &dirs, std::process::id());

        assert_eq!(killed.len(), 1, "the detached sleeper must be signalled");
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

    /// A double-forked orphan in the worker's directories is killed: once its
    /// parents are gone it is adopted by init or the user's service manager,
    /// which is exactly the leak the sweep exists for.
    #[test]
    fn a_reparented_orphan_inside_the_directories_is_killed() {
        let dir = worker_dir("orphan");
        let pid_file = dir.join("orphan.pid");
        // The helper double-forks and exits at once, so the sleeper is
        // reparented to init or the service manager before the sweep runs.
        let mut helper = Command::new("bash")
            .args([
                "-c",
                &format!("setsid sleep 300 & echo $! > {}", pid_file.display()),
            ])
            .current_dir(&dir)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("the helper must spawn");
        let _ = helper.wait();
        let orphan = wait_for_pid(&pid_file);
        // The helper is gone, so the sleeper is reparented before the sweep.
        for _ in 0..200 {
            if parent_pid(orphan).is_some_and(|p| p != helper.id()) {
                break;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        await_process_in(&dir);

        let dirs = [dir.clone()];
        let killed = sweep_owned_processes("worker-test", &dirs, std::process::id());

        assert_eq!(killed.len(), 1, "the reparented orphan must be signalled");
        for _ in 0..100 {
            if !pid_is_alive(orphan) {
                break;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        assert!(
            !pid_is_alive(orphan),
            "the sweep must leave no orphan behind in the worker's directories"
        );
        assert!(owned_processes_in_dirs(&dirs, std::process::id()).is_empty());
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
        let killed = sweep_owned_processes("worker-test", &dirs, std::process::id());

        assert_eq!(
            killed.len(),
            0,
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

    /// A process whose parent is an unrelated long-lived process survives the
    /// sweep: it belongs to a terminal/login session, not to this worker.
    ///
    /// The "user shell" is a `setsid` session leader, which is exactly how a
    /// terminal the operator opened by hand appears from the outside: its child
    /// is adopted by the user's service manager, never by this worker's hub.
    ///
    /// The hub here is a pid that is not on the shell's ancestry at all, which
    /// is what makes the shell unrelated: a real hub would be the daemon the
    /// operator started, and this terminal would sit beside it, not below it.
    #[test]
    fn a_process_from_an_unrelated_parent_survives_the_sweep() {
        let dir = worker_dir("unrelated");
        let pid_file = dir.join("child.pid");
        let mut shell = spawn_unrelated_shell(&dir, &pid_file);
        let shell_pid = shell.id();
        let child_pid = wait_for_pid(&pid_file);
        await_process_in(&dir);

        // The hub is a process that is *not* an ancestor of the shell: the
        // shell's own ancestry never passes through it, so the sweep must find
        // nothing to kill.
        let dirs = [dir.clone()];
        let killed = sweep_owned_processes("worker-test", &dirs, child_pid + 1);

        assert_eq!(
            killed.len(),
            0,
            "a process whose ancestry is an unrelated session must survive \
             (shell {shell_pid}, child {child_pid})"
        );
        assert!(
            pid_is_alive(shell_pid) && pid_is_alive(child_pid),
            "neither the unrelated shell nor its child may be signalled"
        );
        signal_pid(shell_pid, libc::SIGKILL);
        signal_pid(child_pid, libc::SIGKILL);
        let _ = shell.wait();
        let _ = std::fs::remove_dir_all(&dir);
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
