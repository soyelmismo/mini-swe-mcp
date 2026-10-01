//! Background jobs: a command that outlived its wall-clock budget.
//!
//! A worker command may run at most its timeout (600 s for a heavy build), and
//! on a loaded host a full build plus test run routinely takes longer. Killing
//! it throws the build away and teaches the model to background it by hand
//! (`nohup cargo test > log &`) and then burn a turn per poll. This module is
//! the sanctioned alternative: when a command reaches its budget it is *not*
//! killed. It keeps running in its own process group, its output keeps
//! streaming into a bounded log in the worker's private scratch, and the worker
//! is handed a job number it can wait on (`WAIT_JOB`) or stop (`KILL_JOB`).
//!
//! # Confinement
//!
//! A job is exactly the process the command already was: the same sandbox, the
//! same working directory, the same process group, so the worker's existing
//! process-group sweep still reaches it. What is added is *lifetime*: the
//! admission permit the command was granted under is moved into the job, so a
//! job never outlives the build slot it was admitted with, and the table is
//! keyed by worker id, so ending a worker ends its jobs.
//!
//! # Bounds
//!
//! Every job is capped twice: an absolute ceiling ([`job_max_secs`],
//! configurable) after which the group is killed, and the per-call budget of a
//! `WAIT_JOB` ([`wait_job_secs`]). Output is bounded by the executor's own
//! head/tail truncation plus a capped log file, so a job that prints gigabytes
//! costs the same as one that prints nothing.

use std::any::Any;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use tokio::sync::Notify;

use super::exec::{
    Captured, PipeBuffer, TERM_GRACE, TIMEOUT_EXIT_CODE, combine_streams, signal_process_group,
    terminate_process_group,
};

/// Absolute ceiling on a background job's lifetime (seconds).
///
/// A job is a build the worker is still waiting on, not a daemon: past this
/// ceiling the group is killed whatever it is doing.
pub const DEFAULT_JOB_MAX_SECS: u64 = 45 * 60;

/// Env var overriding [`DEFAULT_JOB_MAX_SECS`].
pub const JOB_MAX_SECS_ENV: &str = "JOB_MAX_SECS";

/// How long one `WAIT_JOB` call blocks before it reports "still running".
pub const DEFAULT_WAIT_JOB_SECS: u64 = 600;

/// Env var overriding [`DEFAULT_WAIT_JOB_SECS`].
pub const WAIT_JOB_SECS_ENV: &str = "WAIT_JOB_SECS";

/// [`DEFAULT_JOB_MAX_SECS`] as configured by the operator.
pub fn job_max_secs() -> u64 {
    crate::config::env_parse(JOB_MAX_SECS_ENV).unwrap_or(DEFAULT_JOB_MAX_SECS)
}

/// [`DEFAULT_WAIT_JOB_SECS`] as configured by the operator.
pub fn wait_job_secs() -> u64 {
    crate::config::env_parse(WAIT_JOB_SECS_ENV).unwrap_or(DEFAULT_WAIT_JOB_SECS)
}

/// Every worker's background jobs, keyed by worker id.
///
/// The pool owns one table for its whole lifetime: a job is registered by the
/// runner that started it, listed by the worker's status and killed when the
/// worker ends, so this is the single place that knows a job outlives its step.
/// A worker's entry exists only while it has a live job, so the table cannot
/// grow with the number of workers that ever ran.
#[derive(Default)]
pub struct JobTable {
    workers: Mutex<BTreeMap<String, WorkerJobs>>,
}

/// One worker's jobs, numbered from 1 in the order they were backgrounded.
#[derive(Default)]
struct WorkerJobs {
    next_id: u64,
    jobs: BTreeMap<u64, Arc<JobState>>,
}

impl JobTable {
    /// An empty table, for a caller that owns no pool.
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Add `job` as the worker's next job number.
    fn register(&self, worker: &str, job: Arc<JobState>) -> u64 {
        let mut workers = lock(&self.workers);
        let entry = workers.entry(worker.to_string()).or_default();
        entry.next_id += 1;
        let id = entry.next_id;
        entry.jobs.insert(id, job);
        id
    }

    /// The worker's job `id`, if it is still live.
    fn get(&self, worker: &str, id: u64) -> Option<Arc<JobState>> {
        lock(&self.workers)
            .get(worker)
            .and_then(|entry| entry.jobs.get(&id))
            .cloned()
    }

    /// Forget the worker's job `id`, dropping its entry once the last one goes.
    fn forget(&self, worker: &str, id: u64) {
        let mut workers = lock(&self.workers);
        let empty = workers
            .get_mut(worker)
            .is_some_and(|entry| entry.jobs.remove(&id).is_some() && entry.jobs.is_empty());
        if empty {
            workers.remove(worker);
        }
    }

    /// Stop every job of `worker` and forget them all.
    ///
    /// Called from a `Drop`, so it only signals: each supervisor reaps its own
    /// child and records the outcome for anyone still waiting.
    pub fn kill_all(&self, worker: &str) -> usize {
        let jobs: Vec<Arc<JobState>> = lock(&self.workers)
            .remove(worker)
            .map(|entry| entry.jobs.into_values().collect())
            .unwrap_or_default();
        for job in &jobs {
            job.kill();
        }
        jobs.len()
    }

    /// The worker's live jobs, oldest first, as its status lists them.
    pub fn summaries(&self, worker: &str) -> Vec<JobStatus> {
        lock(&self.workers)
            .get(worker)
            .map(|entry| {
                entry
                    .jobs
                    .iter()
                    .map(|(id, job)| job.status(*id))
                    .collect()
            })
            .unwrap_or_default()
    }
}

/// The runner's view of one worker's jobs.
///
/// Cloned per command like the rest of the runner's per-command state. A runner
/// that was never given one has no jobs at all: a command that outlives its
/// budget is then killed, exactly as it was before jobs existed.
#[derive(Clone)]
pub struct JobHandle {
    table: Arc<JobTable>,
    worker: String,
}

impl JobHandle {
    /// A handle on `worker`'s half of `table`.
    pub fn new(table: Arc<JobTable>, worker: impl Into<String>) -> Self {
        Self {
            table,
            worker: worker.into(),
        }
    }

    /// Register `job` as this worker's next job and start supervising it.
    pub fn spawn(&self, job: Arc<JobState>) -> u64 {
        let id = self.table.register(&self.worker, Arc::clone(&job));
        job.supervise(Duration::from_secs(job_max_secs()));
        id
    }

    /// This worker's job `id`, if it is still live.
    pub fn job(&self, id: u64) -> Option<Arc<JobState>> {
        self.table.get(&self.worker, id)
    }

    /// Stop this worker's job `id`; `false` when no such job is live.
    pub fn kill(&self, id: u64) -> bool {
        let Some(job) = self.table.get(&self.worker, id) else {
            return false;
        };
        let killed = job.kill();
        self.table.forget(&self.worker, id);
        killed
    }

    /// This worker's live jobs, oldest first.
    pub fn summaries(&self) -> Vec<JobStatus> {
        self.table.summaries(&self.worker)
    }
}

/// One background job: the process group, its bounded output and its lifetime.
pub struct JobState {
    inner: Mutex<JobInner>,
    /// Signalled once, when the supervisor has reaped the job and recorded the
    /// outcome, so a waiter never has to poll.
    done: Notify,
}

/// The mutable half of a job, behind one lock.
struct JobInner {
    /// Process group leader, until the job is reaped.
    pid: Option<u32>,
    /// Taken by the supervisor for the whole wait, so no lock is ever held
    /// across an await.
    child: Option<tokio::process::Child>,
    /// Likewise taken by the supervisor, which finishes the drain at the end.
    out: Option<PipeBuffer>,
    err: Option<PipeBuffer>,
    /// Shared with the readers above, so the tail is readable at any moment.
    out_bytes: Arc<Mutex<Captured>>,
    err_bytes: Arc<Mutex<Captured>>,
    /// Recorded once, by the supervisor, when the job is reaped.
    outcome: Option<JobOutcome>,
    /// Bounded one-line label of the command that started the job.
    summary: String,
    /// Unix time the job was backgrounded.
    started_at: u64,
    /// Where the job's output streams to, inside the worker's private scratch.
    log: PathBuf,
    /// Guards whose lifetime must match the job's -- the admission permit the
    /// command was granted under -- dropped when the job is reaped.
    guards: Vec<Box<dyn Any + Send>>,
}

impl JobState {
    /// A job for a command that just outlived its budget.
    ///
    /// Crate-private: a job is built by the executor that ran the command, and
    /// the pipes it hands over are the executor's own bounded drain.
    pub(super) fn new(
        pid: Option<u32>,
        child: tokio::process::Child,
        out: PipeBuffer,
        err: PipeBuffer,
        summary: String,
        log: PathBuf,
    ) -> Arc<Self> {
        let out_bytes = out.shared();
        let err_bytes = err.shared();
        Arc::new(Self {
            inner: Mutex::new(JobInner {
                pid,
                child: Some(child),
                out: Some(out),
                err: Some(err),
                out_bytes,
                err_bytes,
                outcome: None,
                summary,
                started_at: unix_now(),
                log,
                guards: Vec::new(),
            }),
            done: Notify::new(),
        })
    }

    /// Keep `guard` alive exactly as long as this job.
    ///
    /// The admission permit a heavy command was granted under is moved here, so
    /// a job that outlives its step never outlives the build slot it was
    /// admitted with.
    pub fn retain(&self, guard: Box<dyn Any + Send>) {
        lock(&self.inner).guards.push(guard);
    }

    /// Stop the job's whole process group.
    ///
    /// `SIGKILL` rather than a graceful `SIGTERM`: the worker asked for the job
    /// to stop, and a build that ignores `SIGTERM` would keep holding the build
    /// slot the job still carries. The supervisor reaps the group and records
    /// the outcome afterwards.
    pub fn kill(&self) -> bool {
        let pid = {
            let inner = lock(&self.inner);
            inner.outcome.is_none().then(|| inner.pid).flatten()
        };
        let Some(pid) = pid else {
            return false;
        };
        signal_process_group(Some(pid), libc::SIGKILL);
        true
    }

    /// Watch the job until it exits or its absolute ceiling kills it.
    pub fn supervise(self: &Arc<Self>, cap: Duration) {
        let state = Arc::clone(self);
        tokio::spawn(async move { state.run(cap).await });
    }

    /// The supervised wait itself; see [`supervise`](Self::supervise).
    async fn run(self: Arc<Self>, cap: Duration) {
        let taken = {
            let mut inner = lock(&self.inner);
            (
                inner.pid,
                inner.child.take(),
                inner.out.take(),
                inner.err.take(),
            )
        };
        let (Some(pid), Some(mut child), Some(out), Some(err)) = taken else {
            // Already supervised: a second task must not wait on the same child.
            return;
        };

        let waited = tokio::time::timeout(cap, child.wait()).await;
        let (code, end) = match waited {
            Ok(Ok(status)) => (status.code(), JobEnd::Exited),
            // The child's fate is unknown, so the group goes down here too.
            Ok(Err(e)) => {
                tracing::warn!(error = %e, "waiting on a background job failed; killing its group");
                terminate_process_group(Some(pid), &mut child, TERM_GRACE).await;
                (None, JobEnd::Exited)
            }
            Err(_elapsed) => {
                // The absolute ceiling: the group goes down, gracefully first.
                terminate_process_group(Some(pid), &mut child, TERM_GRACE).await;
                (Some(TIMEOUT_EXIT_CODE), JobEnd::Capped)
            }
        };
        // The group is gone, so the readers reach EOF and the drain converges;
        // a grandchild that escaped the group can still hold a pipe open, which
        // `finish` bounds.
        let _ = tokio::join!(out.finish(), err.finish());

        {
            let mut inner = lock(&self.inner);
            inner.outcome = Some(JobOutcome { code, end });
            inner.pid = None;
            // Dropped here, and only here: the build slot the job was admitted
            // under is held until the job is reaped.
            inner.guards.clear();
        }
        self.done.notify_waiters();
    }

    /// Block until the job is reaped or `limit` elapses.
    pub async fn wait(&self, limit: Duration) -> JobWait {
        // The notification is registered *before* the outcome is read, so a job
        // reaped in between cannot leave this waiter asleep until the limit.
        let mut notified = std::pin::pin!(self.done.notified());
        notified.as_mut().enable();
        if self.outcome().is_none() {
            let _ = tokio::time::timeout(limit, notified).await;
        }
        match self.outcome() {
            Some(outcome) => JobWait::Finished {
                outcome,
                output: self.tail(),
            },
            None => JobWait::Running { output: self.tail() },
        }
    }

    /// The recorded outcome, once the job has been reaped.
    pub fn outcome(&self) -> Option<JobOutcome> {
        lock(&self.inner).outcome.clone()
    }

    /// The bounded head and tail of the job's output so far.
    ///
    /// The executor's own truncation is reused verbatim, so a job's tail is
    /// exactly what the command's output would have been.
    pub fn tail(&self) -> String {
        let inner = lock(&self.inner);
        let (out, out_dropped) = captured(&inner.out_bytes);
        let (err, err_dropped) = captured(&inner.err_bytes);
        combine_streams(&out, &err, out_dropped + err_dropped)
    }

    /// Where the job's output streams to.
    pub fn log(&self) -> PathBuf {
        lock(&self.inner).log.clone()
    }

    /// How this job appears in the worker's status.
    pub fn status(&self, id: u64) -> JobStatus {
        let inner = lock(&self.inner);
        JobStatus {
            id,
            command: inner.summary.clone(),
            elapsed_secs: unix_now().saturating_sub(inner.started_at),
            finished: inner.outcome.is_some(),
        }
    }
}

/// How a background job ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JobEnd {
    /// The command exited on its own.
    Exited,
    /// Killed at its absolute ceiling.
    Capped,
    /// Stopped by `KILL_JOB`.
    Killed,
}

/// What a finished job left behind.
#[derive(Clone, Debug)]
pub struct JobOutcome {
    /// Exit code; `None` when the job died on a signal or could not be reaped.
    pub code: Option<i32>,
    /// How it ended.
    pub end: JobEnd,
}

/// What one `WAIT_JOB` call learned.
#[derive(Clone, Debug)]
pub enum JobWait {
    /// The job is gone; this is its exit code and the tail of its output.
    Finished { outcome: JobOutcome, output: String },
    /// The per-call budget elapsed first; this is the tail so far.
    Running { output: String },
}

impl JobWait {
    /// The tool result a `WAIT_JOB` produces, with the exit code the loop
    /// reports beside it.
    pub fn report(&self, id: u64) -> (String, Option<i32>) {
        match self {
            JobWait::Finished { outcome, output } => {
                let headline = match (outcome.end, outcome.code) {
                    (JobEnd::Capped, _) => format!("job {id} was killed at its ceiling"),
                    (JobEnd::Killed, _) => format!("job {id} was stopped"),
                    (JobEnd::Exited, Some(code)) => format!("job {id} exited with code {code}"),
                    (JobEnd::Exited, None) => format!("job {id} exited on a signal"),
                };
                (format!("{headline}:\n{output}"), outcome.code)
            }
            JobWait::Running { output } => (
                format!(
                    "job {id} is still running; `echo WAIT_JOB {id}` waits for it again:\n{output}"
                ),
                Some(0),
            ),
        }
    }
}

/// One background job as the worker's status lists it.
#[derive(Clone, Debug)]
pub struct JobStatus {
    /// Job number the worker waits on.
    pub id: u64,
    /// Bounded one-line label of the command that started it.
    pub command: String,
    /// Seconds the job has been running.
    pub elapsed_secs: u64,
    /// Whether the job has been reaped.
    pub finished: bool,
}

impl JobStatus {
    /// The one-line form a status view shows.
    pub fn label(&self) -> String {
        let state = if self.finished {
            "finished"
        } else {
            "running"
        };
        format!(
            "job {}: {} ({state}, {}s)",
            self.id, self.command, self.elapsed_secs
        )
    }
}

/// The bytes worth showing from one drained pipe, plus how many were elided.
fn captured(bytes: &Arc<Mutex<Captured>>) -> (Vec<u8>, usize) {
    let guard = lock(bytes);
    (guard.captured(), guard.dropped())
}

/// Lock a mutex, recovering from a poisoned lock rather than propagating the
/// panic: the data behind it is still the truth about a job's output.
fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|poison| poison.into_inner())
}

/// Unix time now, for the elapsed figure a status reports.
fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs())
        .unwrap_or(0)
}
