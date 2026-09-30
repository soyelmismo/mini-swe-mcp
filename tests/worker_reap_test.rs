//! A worker's commands must not outlive the worker.
//!
//! Two escapes defeat a process-group kill. A job backgrounded with `&` stays
//! in the step's process group, so the step's own teardown ends it; `setsid`
//! (or a double fork) moves the job into a session of its own and reparents it
//! to init, so nothing but its working directory still ties it to the worker
//! that spawned it. These tests pin both ends of that: the first dies with the
//! step, the second survives the step and is taken down by the worker-end
//! sweep, and nothing outside the worker's directories is ever signalled.

use mini_swe_mcp::agent::reap::processes_in_dirs;
use mini_swe_mcp::agent::AgentRunner;
use mini_swe_mcp::worktree::{swe_base_dir, WorktreeGuard};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// A scratch directory under the same base the crate's worktrees use, so a
/// test's paths are shaped like a real worker's.
fn unique_dir(tag: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = swe_base_dir().join(format!("swe-reap-it-{tag}-{}-{nanos}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch dir must be creatable");
    dir
}

fn runner() -> AgentRunner {
    AgentRunner::new(
        "http://localhost".to_string(),
        "test-key".to_string(),
        "test-model".to_string(),
        None,
    )
}

/// A temporary git repository, so a `WorktreeGuard` can be built for the test.
struct TestRepo {
    dir: PathBuf,
}

impl TestRepo {
    fn new(tag: &str) -> Self {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("swe-reap-repo-{tag}-{}-{nanos}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        for args in [
            vec!["init", "-b", "master"],
            vec!["config", "user.name", "mini-swe-test"],
            vec!["config", "user.email", "test@localhost"],
        ] {
            let output = Command::new("git")
                .current_dir(&dir)
                .args(&args)
                .output()
                .unwrap_or_else(|e| panic!("git {args:?} must run: {e}"));
            assert!(output.status.success(), "git {args:?} failed");
        }
        std::fs::write(dir.join("README.md"), "# reap test\n").unwrap();
        for args in [vec!["add", "README.md"], vec!["commit", "-m", "baseline"]] {
            let output = Command::new("git")
                .current_dir(&dir)
                .args(&args)
                .output()
                .unwrap_or_else(|e| panic!("git {args:?} must run: {e}"));
            assert!(output.status.success(), "git {args:?} failed");
        }
        Self { dir }
    }

    fn path(&self) -> &Path {
        &self.dir
    }
}

impl Drop for TestRepo {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// Wait until at least one process runs with `dir` as its working directory.
fn await_process_in(dir: &Path) {
    for _ in 0..200 {
        if !processes_in_dirs(&[dir.to_path_buf()]).is_empty() {
            return;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    panic!("no process ever appeared in {}", dir.display());
}

/// Wait until no process is left with `dir` as its working directory.
async fn await_empty(dir: &Path) {
    for _ in 0..200 {
        if processes_in_dirs(&[dir.to_path_buf()]).is_empty() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

/// A job the shell backgrounded with `&` is still a member of the step's
/// process group, so the step's teardown takes it down.
#[tokio::test]
async fn a_backgrounded_job_dies_with_its_step() {
    let dir = unique_dir("step");

    let (out, code) = runner()
        .execute_bash(&dir, "sleep 300 & echo started")
        .await
        .expect("the step must run");

    assert_eq!(code, Some(0), "{out:?}");
    assert!(out.contains("started"), "{out:?}");
    await_empty(&dir).await;
    assert!(
        processes_in_dirs(&[dir.clone()]).is_empty(),
        "a job backgrounded with `&` must not outlive the step that started it"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// A job the shell detached with `setsid` is in no process group of ours, so it
/// survives its step -- and the worker-end sweep takes it down.
#[tokio::test]
async fn a_detached_job_survives_the_step_and_dies_at_worker_end() {
    let repo = TestRepo::new("detached");
    let id = format!("reap-detached-{}", std::process::id());
    let worktree = {
        let guard = WorktreeGuard::new(repo.path(), &id).expect("worktree must be created");
        let (out, code) = runner()
            .execute_bash(&guard.path, "(setsid sleep 300 &); echo started")
            .await
            .expect("the step must run");
        assert_eq!(code, Some(0), "{out:?}");
        assert!(out.contains("started"), "{out:?}");
        await_process_in(&guard.path);
        assert!(
            !processes_in_dirs(&[guard.path.clone()]).is_empty(),
            "a detached job must survive the step that started it"
        );
        guard.path.clone()
        // The guard drops here: worker end, on the completion path.
    };

    await_empty(&worktree).await;
    assert!(
        processes_in_dirs(&[worktree.clone()]).is_empty(),
        "the worker-end sweep must take a detached job down"
    );
}

/// The sweep matches on the worker's own directories, so a process running
/// somewhere else is never signalled -- not even by a worker that is ending.
#[tokio::test]
async fn the_worker_end_sweep_never_signals_a_process_outside_the_workers_directories() {
    let repo = TestRepo::new("scope");
    let id = format!("reap-scope-{}", std::process::id());
    let outside = unique_dir("outside");
    let mut sleeper = Command::new("setsid")
        .args(["sleep", "300"])
        .current_dir(&outside)
        .spawn()
        .expect("setsid must spawn");
    await_process_in(&outside);

    let worktree = {
        let guard = WorktreeGuard::new(repo.path(), &id).expect("worktree must be created");
        guard.path.clone()
    };

    assert!(
        !processes_in_dirs(&[outside.clone()]).is_empty(),
        "a process outside the worker's directories must survive the worker-end sweep"
    );
    let _ = Command::new("kill")
        .args(["-9", &sleeper.id().to_string()])
        .status();
    let _ = sleeper.wait();
    let _ = std::fs::remove_dir_all(&outside);
    let _ = std::fs::remove_dir_all(&worktree);
}
