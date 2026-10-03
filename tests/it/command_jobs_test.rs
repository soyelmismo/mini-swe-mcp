//! Background jobs: a command that outlives its wall-clock budget keeps running
//! as a job the worker can wait on or stop, instead of being killed.
//!
//! Every test drives the real executor through `AgentRunner::execute_bash` with
//! a one-second command budget, so a job is created in milliseconds rather than
//! after the 600 s a heavy command would take. The scratch each test touches is
//! a temporary directory handed to the code under test.

use crate::common;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::common::TempDir;
use mini_swe_mcp::agent::AgentRunner;
use mini_swe_mcp::agent::jobs::{JobHandle, JobTable, JobWait};
use mini_swe_mcp::pool::{parse_kill_job, parse_wait_job};

/// A runner whose commands outlive a one-second budget, filing its jobs under
/// `worker` in `table`.
fn runner(worker: &str, table: &Arc<JobTable>) -> (AgentRunner, JobHandle) {
    let handle = JobHandle::new(Arc::clone(table), worker);
    let runner = AgentRunner::new(
        "http://localhost:1".to_string(),
        "test-key".to_string(),
        "test-model".to_string(),
        None,
    )
    .with_command_timeout(1)
    .with_jobs(handle.clone());
    (runner, handle)
}

/// Wait until the job has written its pid to `path`, and return it.
async fn job_pid(path: &std::path::Path) -> u32 {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if let Ok(text) = std::fs::read_to_string(path)
            && let Ok(pid) = text.trim().parse::<u32>()
        {
            return pid;
        }
        assert!(Instant::now() < deadline, "the job never started: {path:?}");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

/// Whether `pid` still has a live process behind it.
///
/// A reaped-but-unwaited child stays in `/proc` as a zombie, so the state field
/// is read rather than the mere existence of the directory.
fn process_alive(pid: u32) -> bool {
    let Ok(stat) = std::fs::read_to_string(format!("/proc/{pid}/stat")) else {
        return false;
    };
    match stat.rsplit_once(')') {
        Some((_, rest)) => !rest.trim_start().starts_with('Z'),
        None => false,
    }
}

/// Poll until `pid` is gone, and report whether it went away in time.
async fn wait_for_exit(pid: u32) -> bool {
    let deadline = Instant::now() + Duration::from_secs(10);
    while process_alive(pid) {
        if Instant::now() >= deadline {
            return false;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    true
}

/// A command that outlives its budget becomes job 1 and keeps running, with its
/// output streaming to a log in the worker's private scratch.
#[tokio::test]
async fn a_command_that_outlives_its_budget_becomes_a_job_and_keeps_running() {
    let work = TempDir::new_in_tmp("cmdjob");
    let table = JobTable::new();
    let (runner, handle) = runner("w-live", &table);
    let pid_file = work.path().join("job.pid");

    let (output, code) = runner
        .execute_bash(
            work.path(),
            &format!("echo starting; echo $$ > {}; sleep 30", pid_file.display()),
        )
        .await
        .expect("a command that outlives its budget must still be spawned");

    assert!(output.contains("job 1"), "{output}");
    assert!(output.contains("WAIT_JOB 1"), "{output}");
    assert!(output.contains("KILL_JOB 1"), "{output}");
    assert!(
        output.contains("still running"),
        "the command must be reported as still running, not as finished: {output}"
    );
    // The timeout's exit code, not 0: the command has not finished, so a
    // completion gate that outlived its budget has not passed.
    assert_eq!(code, Some(124), "{output}");

    let jobs = handle.summaries();
    assert_eq!(jobs.len(), 1, "{jobs:?}");
    assert_eq!(jobs[0].id, 1);
    assert!(!jobs[0].finished, "{jobs:?}");

    // The command was not killed: its process is still there.
    let pid = job_pid(&pid_file).await;
    assert!(
        process_alive(pid),
        "job 1 must keep running past its budget"
    );

    // Its output keeps streaming to a bounded log in the worker's private
    // scratch, which is the `swe-tmp-<worktree>` directory beside the worktree.
    let log = handle.job(1).expect("job 1 is live").log();
    let scratch = log.parent().expect("a job log lives in a directory");
    assert!(
        scratch
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with("swe-tmp-")),
        "a job's output must stream to the worker's private scratch: {}",
        log.display()
    );
    let deadline = Instant::now() + Duration::from_secs(10);
    while std::fs::read_to_string(&log).is_ok_and(|text| !text.contains("starting")) {
        assert!(
            Instant::now() < deadline,
            "the job log never filled: {log:?}"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    // A wait that runs out of budget reports the job as still running rather
    // than blocking the turn forever.
    let wait = runner
        .wait_job(1, Duration::from_millis(200))
        .await
        .expect("job 1 is live");
    let (report, code) = wait.report(1);
    assert!(report.contains("still running"), "{report}");
    assert_eq!(code, Some(0), "{report}");
    assert!(
        matches!(wait, JobWait::Running { .. }),
        "the per-call budget must not wait for the job: {wait:?}"
    );

    // Leave nothing running behind the test.
    assert!(runner.kill_job(1));
    assert!(wait_for_exit(pid).await, "the job must stop when killed");
}

/// Waiting on a job reports its exit code and the tail of its output, and the
/// job leaves the worker's list once its outcome has been collected.
#[tokio::test]
async fn waiting_on_a_job_reports_its_exit_code_and_the_tail_of_its_output() {
    let work = TempDir::new_in_tmp("cmdjob");
    let table = JobTable::new();
    let (runner, handle) = runner("w-wait", &table);

    let (output, _) = runner
        .execute_bash(work.path(), "echo building; sleep 2; echo built; exit 3")
        .await
        .expect("a command that outlives its budget must still be spawned");
    assert!(output.contains("job 1"), "{output}");

    let wait = runner
        .wait_job(1, Duration::from_secs(30))
        .await
        .expect("job 1 is live");
    let (report, code) = wait.report(1);
    assert_eq!(code, Some(3), "{report}");
    assert!(report.contains("exited with code 3"), "{report}");
    match &wait {
        JobWait::Finished { outcome, output } => {
            assert_eq!(outcome.code, Some(3));
            assert!(output.contains("built"), "{output}");
        }
        JobWait::Running { .. } => panic!("the job must have finished: {wait:?}"),
    }

    // A collected job leaves the list, so a finished job never accumulates.
    assert!(handle.summaries().is_empty(), "{:?}", handle.summaries());
    assert!(
        runner.wait_job(1, Duration::from_secs(1)).await.is_none(),
        "a collected job is gone"
    );
}

/// `KILL_JOB` stops the job's whole process group.
#[tokio::test]
async fn killing_a_job_stops_it() {
    let work = TempDir::new_in_tmp("cmdjob");
    let table = JobTable::new();
    let (runner, handle) = runner("w-kill", &table);
    let pid_file = work.path().join("job.pid");

    let (output, _) = runner
        .execute_bash(
            work.path(),
            &format!("echo $$ > {}; sleep 30", pid_file.display()),
        )
        .await
        .expect("a command that outlives its budget must still be spawned");
    assert!(output.contains("job 1"), "{output}");
    let pid = job_pid(&pid_file).await;
    assert!(
        process_alive(pid),
        "job 1 must be running before it is killed"
    );

    assert!(runner.kill_job(1), "job 1 must be killable");
    assert!(
        wait_for_exit(pid).await,
        "KILL_JOB must stop the job's process group"
    );
    assert!(handle.summaries().is_empty(), "{:?}", handle.summaries());
    assert!(
        !runner.kill_job(1),
        "a stopped job is no longer waiting to be killed"
    );
}

/// A job dies with its worker: ending the worker stops every job it left.
#[tokio::test]
async fn a_job_dies_with_its_worker() {
    let work = TempDir::new_in_tmp("cmdjob");
    let table = JobTable::new();
    let (runner, _handle) = runner("w-ended", &table);
    let pid_file = work.path().join("job.pid");

    let (output, _) = runner
        .execute_bash(
            work.path(),
            &format!("echo $$ > {}; sleep 30", pid_file.display()),
        )
        .await
        .expect("a command that outlives its budget must still be spawned");
    assert!(output.contains("job 1"), "{output}");
    let pid = job_pid(&pid_file).await;
    assert!(
        process_alive(pid),
        "job 1 must be running while its worker is"
    );

    // What the worker's teardown does when the worker ends, however it ends.
    assert_eq!(table.kill_all("w-ended"), 1);
    assert!(
        wait_for_exit(pid).await,
        "a job must not outlive the worker that started it"
    );
    assert!(
        table.summaries("w-ended").is_empty(),
        "an ended worker keeps no jobs"
    );
}

/// A harness gate run past the step timeout is not converted into a job: the
/// command is stopped and its real exit code decides, so no job is left holding
/// a build slot nobody waits on.
#[tokio::test]
async fn a_gate_run_past_its_budget_is_not_converted_into_a_job() {
    let work = TempDir::new_in_tmp("cmdjob");
    let table = JobTable::new();
    let (runner, handle) = runner("w-gate", &table);
    // What the completion verify, its divergent variant and the merge gate all
    // do to the runner they execute with.
    let runner = runner.without_job_conversion();
    let pid_file = work.path().join("gate.pid");

    let (output, code) = runner
        .execute_bash(
            work.path(),
            &format!("echo $$ > {}; sleep 30", pid_file.display()),
        )
        .await
        .expect("a gate that outlives its budget must still be spawned");

    assert!(
        output.contains("timed out after 1s"),
        "a gate run must report the timeout it hit: {output}"
    );
    assert!(!output.contains("job 1"), "{output}");
    assert_eq!(code, Some(124), "{output}");
    assert!(
        handle.summaries().is_empty(),
        "a gate run must not create a job: {:?}",
        handle.summaries()
    );

    // The command really was stopped rather than left running behind the gate.
    let pid = job_pid(&pid_file).await;
    assert!(
        wait_for_exit(pid).await,
        "a gate run must be stopped, not backgrounded"
    );
}

/// The two job sentinels are recognised in `echo`/`printf` form only, and never
/// confuse each other or the turn-budget sentinel.
#[test]
fn the_job_sentinels_are_parsed_from_an_echo_command() {
    assert_eq!(parse_wait_job("echo WAIT_JOB 1"), Some(1));
    assert_eq!(parse_wait_job("echo WAIT_JOB: 12"), Some(12));
    assert_eq!(parse_wait_job("printf 'WAIT_JOB 4\\n'"), Some(4));
    assert_eq!(parse_kill_job("echo KILL_JOB 2"), Some(2));
    assert_eq!(parse_kill_job("echo KILL_JOB: 9"), Some(9));
    for cmd in [
        "echo WAIT_JOB",
        "echo WAIT_JOB 0",
        "echo KILL_JOB",
        "cat job.log",
        "grep -rn WAIT_JOB src/",
        "echo REQUEST_TURNS: 5",
    ] {
        assert_eq!(parse_wait_job(cmd), None, "{cmd:?}");
        assert_eq!(parse_kill_job(cmd), None, "{cmd:?}");
    }
}
