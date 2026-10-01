//! Resource-aware admission control for heavy commands.
//!
//! One hub daemon fans out to ~100 workers on a 4-core / 15 GB machine. The
//! workers themselves are cheap — most of a worker's life is spent waiting on
//! the LLM — but a *heavy* command (`cargo`, `rustc`, a test suite: see
//! [`is_heavy_command`](crate::agent::is_heavy_command)) is a real load on the
//! machine, and a fixed-width semaphore cannot tell a host with 12 GB free
//! from one that is already swapping.
//!
//! This module replaces that semaphore with a controller that grants a heavy
//! slot only while the host can actually take another build, and sizes the
//! build's own thread count from how many builds already run:
//!
//! * `running_heavy < max_heavy` (`BASH_BUILD_LIMIT`, default the core count),
//! * `MemAvailable - HUB_MEM_RESERVE_MB >= HUB_BUILD_MEM_MB`,
//! * CPU `some avg10 < HUB_CPU_PRESSURE_MAX` (default 60),
//! * memory `full avg10 < HUB_MEM_PRESSURE_MAX` (default 10),
//! * IO `full avg10 < HUB_IO_PRESSURE_MAX` (default 40).
//!
//! The pressure criteria come from Linux PSI (`/proc/pressure/*`), which
//! measures the time tasks actually stalled on a resource instead of the
//! 1-minute load average — that average counts I/O wait and unrelated desktop
//! processes, and lags by a minute. When `/proc/pressure` is unavailable
//! (older kernels, containers without PSI) the controller falls back to the
//! 1-minute load average `< cores * 1.25`.
//!
//! The one exception is the progress guarantee: with nothing heavy running a
//! slot is *always* granted, so a saturated host can never deadlock a worker
//! that has no other build to wait behind.
//!
//! [`admit`] is the pure decision function and carries the unit tests for
//! every branch; the `/proc` readers beside it return `None` when unreadable,
//! and an unreadable criterion is skipped rather than read as a failure. The
//! controller adds only what a pure function cannot express: the running
//! counter, priority queueing and the re-evaluation wakeups.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use tokio::sync::Notify;
use tracing::debug;

/// Env var overriding how many heavy commands may run at once.
pub const BUILD_LIMIT_ENV: &str = "BASH_BUILD_LIMIT";
/// Env var overriding the memory held back from the admission estimate (MB).
pub const MEM_RESERVE_ENV: &str = "HUB_MEM_RESERVE_MB";
/// Env var overriding the estimated memory of one heavy command (MB).
pub const BUILD_MEM_ENV: &str = "HUB_BUILD_MEM_MB";

/// Memory held back from the admission estimate, in MB.
pub const DEFAULT_MEM_RESERVE_MB: u64 = 2048;
/// Estimated memory of one heavy command, in MB.
pub const DEFAULT_BUILD_MEM_MB: u64 = 1536;

/// How far the 1-minute load average may exceed the core count.
const LOAD_HEADROOM: f64 = 1.25;

/// Env var overriding the CPU `some avg10` ceiling (percent).
pub const CPU_PRESSURE_MAX_ENV: &str = "HUB_CPU_PRESSURE_MAX";
/// Env var overriding the memory `full avg10` ceiling (percent).
pub const MEM_PRESSURE_MAX_ENV: &str = "HUB_MEM_PRESSURE_MAX";
/// Env var overriding the IO `full avg10` ceiling (percent).
pub const IO_PRESSURE_MAX_ENV: &str = "HUB_IO_PRESSURE_MAX";

/// Default CPU `some avg10` ceiling, in percent.
pub const DEFAULT_CPU_PRESSURE_MAX: f64 = 60.0;
/// Default memory `full avg10` ceiling, in percent.
pub const DEFAULT_MEM_PRESSURE_MAX: f64 = 10.0;
/// Default IO `full avg10` ceiling, in percent.
pub const DEFAULT_IO_PRESSURE_MAX: f64 = 40.0;

/// How often a queued heavy command is re-evaluated when nothing wakes it.
pub const RE_EVALUATE_INTERVAL: Duration = Duration::from_secs(2);

/// How long one `/proc` reading stays usable.
///
/// Every waiter re-evaluates on the same wakeup, so the readings are shared
/// for a moment instead of re-read per waiter.
const SAMPLE_TTL: Duration = Duration::from_millis(500);

/// The host readings one admission decision is made from.
///
/// `None` means "could not read it", and that criterion is then skipped: a
/// container without `/proc/meminfo` must not stop every build.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct HostSample {
    /// `MemAvailable` from `/proc/meminfo`, in MB.
    pub mem_available_mb: Option<u64>,
    /// The 1-minute load average from `/proc/loadavg`.
    pub load1: Option<f64>,
    /// CPU `some avg10` from `/proc/pressure/cpu`, in percent.
    pub cpu_some_avg10: Option<f64>,
    /// Memory `full avg10` from `/proc/pressure/memory`, in percent.
    pub mem_full_avg10: Option<f64>,
    /// IO `full avg10` from `/proc/pressure/io`, in percent.
    pub io_full_avg10: Option<f64>,
}

/// Everything [`admit`] decides on. Plain data so the decision stays a pure
/// function of its inputs, with no `/proc` read and no clock inside it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdmissionInputs {
    /// Heavy commands running right now.
    pub running: usize,
    /// Heavy commands allowed at once.
    pub max: usize,
    /// `MemAvailable` in MB, or `None` when unreadable.
    pub mem_available_mb: Option<u64>,
    /// Memory held back from the estimate, in MB.
    pub reserve_mb: u64,
    /// Estimated memory of one heavy command, in MB.
    pub estimate_mb: u64,
    /// 1-minute load average, or `None` when unreadable.
    pub load1: Option<f64>,
    /// Cores the job count is divided over.
    pub cores: usize,
    /// CPU `some avg10` ceiling, in percent.
    pub cpu_pressure_max: f64,
    /// Memory `full avg10` ceiling, in percent.
    pub mem_pressure_max: f64,
    /// IO `full avg10` ceiling, in percent.
    pub io_pressure_max: f64,
    /// CPU `some avg10`, or `None` when unreadable.
    pub cpu_some_avg10: Option<f64>,
    /// Memory `full avg10`, or `None` when unreadable.
    pub mem_full_avg10: Option<f64>,
    /// IO `full avg10`, or `None` when unreadable.
    pub io_full_avg10: Option<f64>,
}

/// Why a heavy command was not admitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Blocked {
    /// `max` heavy commands already run.
    SlotLimit,
    /// `MemAvailable` minus the reserve cannot cover one more build.
    Memory,
    /// The 1-minute load average reached the ceiling.
    Load,
    /// CPU `some avg10` reached its ceiling.
    CpuPressure,
    /// Memory `full avg10` reached its ceiling.
    MemPressure,
    /// IO `full avg10` reached its ceiling.
    IoPressure,
}

impl Blocked {
    /// Human-readable reason, for the debug line logged while waiting.
    pub fn reason(self) -> &'static str {
        match self {
            Blocked::SlotLimit => "heavy slot limit reached",
            Blocked::Memory => "available memory below the build reserve",
            Blocked::Load => "load average at the ceiling",
            Blocked::CpuPressure => "CPU pressure at the ceiling",
            Blocked::MemPressure => "memory pressure at the ceiling",
            Blocked::IoPressure => "IO pressure at the ceiling",
        }
    }
}

/// The outcome of one admission request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    /// The command may run, with this many jobs.
    Granted {
        /// Jobs the command may use: `max(1, cores / running_after_grant)`.
        jobs: usize,
    },
    /// The command must wait; re-evaluated when a slot frees or every 2 s.
    Waiting(Blocked),
}

/// Jobs a heavy command may use once `running_after_grant` of them run.
///
/// Dividing the cores over the builds already admitted is what keeps the CPU
/// busy instead of swapping: the first build owns the machine, and each
/// further one takes a smaller share, never below one.
pub fn jobs_for(cores: usize, running_after_grant: usize) -> usize {
    (cores / running_after_grant.max(1)).max(1)
}

/// Decide whether one more heavy command may start.
///
/// The slot limit is checked first and the progress guarantee second: with
/// nothing heavy running the host is by definition not saturated *by us*, so
/// the first build always starts even when memory is tight or the pressure is
/// already high. Only a build that would stack on top of another one is dosed.
///
/// The memory estimate is always checked when readable. The PSI criteria are
/// checked when `/proc/pressure` is available, each skipped when its own file
/// is unreadable; the load average is the fallback for hosts without PSI.
pub fn admit(inputs: &AdmissionInputs) -> Decision {
    if inputs.running >= inputs.max {
        return Decision::Waiting(Blocked::SlotLimit);
    }
    if inputs.running == 0 {
        return Decision::Granted {
            jobs: jobs_for(inputs.cores, inputs.running + 1),
        };
    }
    if let Some(available) = inputs.mem_available_mb
        && available < inputs.reserve_mb + inputs.estimate_mb
    {
        return Decision::Waiting(Blocked::Memory);
    }
    // PSI is available when at least one `/proc/pressure` file was readable;
    // then the load average is ignored, because it counts I/O wait and
    // unrelated processes and lags by a minute.
    let psi_available = inputs.cpu_some_avg10.is_some()
        || inputs.mem_full_avg10.is_some()
        || inputs.io_full_avg10.is_some();
    if psi_available {
        if let Some(cpu) = inputs.cpu_some_avg10
            && cpu >= inputs.cpu_pressure_max
        {
            return Decision::Waiting(Blocked::CpuPressure);
        }
        if let Some(mem) = inputs.mem_full_avg10
            && mem >= inputs.mem_pressure_max
        {
            return Decision::Waiting(Blocked::MemPressure);
        }
        if let Some(io) = inputs.io_full_avg10
            && io >= inputs.io_pressure_max
        {
            return Decision::Waiting(Blocked::IoPressure);
        }
    } else if let Some(load) = inputs.load1
        && load >= inputs.cores as f64 * LOAD_HEADROOM
    {
        return Decision::Waiting(Blocked::Load);
    }
    Decision::Granted {
        jobs: jobs_for(inputs.cores, inputs.running + 1),
    }
}

/// `MemAvailable` from `/proc/meminfo`, in MB; `None` when unreadable.
pub fn mem_available_mb() -> Option<u64> {
    mem_available_mb_from(&std::fs::read_to_string("/proc/meminfo").ok()?)
}

/// Pure core of [`mem_available_mb`]: read the `MemAvailable:` line of a
/// `/proc/meminfo` body and convert its kB value to MB.
fn mem_available_mb_from(meminfo: &str) -> Option<u64> {
    let line = meminfo.lines().find(|l| l.starts_with("MemAvailable:"))?;
    let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb / 1024)
}

/// The 1-minute load average from `/proc/loadavg`; `None` when unreadable.
pub fn loadavg_1m() -> Option<f64> {
    loadavg_1m_from(&std::fs::read_to_string("/proc/loadavg").ok()?)
}

/// Pure core of [`loadavg_1m`]: the first field of a `/proc/loadavg` body.
fn loadavg_1m_from(loadavg: &str) -> Option<f64> {
    loadavg.split_whitespace().next()?.parse().ok()
}

/// CPU `some avg10` from `/proc/pressure/cpu`, in percent; `None` when
/// unreadable.
pub fn cpu_pressure_some_avg10() -> Option<f64> {
    psi_some_avg10_from(&std::fs::read_to_string("/proc/pressure/cpu").ok()?)
}

/// Memory `full avg10` from `/proc/pressure/memory`, in percent; `None` when
/// unreadable.
pub fn mem_pressure_full_avg10() -> Option<f64> {
    psi_full_avg10_from(&std::fs::read_to_string("/proc/pressure/memory").ok()?)
}

/// IO `full avg10` from `/proc/pressure/io`, in percent; `None` when
/// unreadable.
pub fn io_pressure_full_avg10() -> Option<f64> {
    psi_full_avg10_from(&std::fs::read_to_string("/proc/pressure/io").ok()?)
}

/// Pure core of [`cpu_pressure_some_avg10`]: the `some` line's `avg10` field.
fn psi_some_avg10_from(body: &str) -> Option<f64> {
    psi_avg10_from(body, "some")
}

/// Pure core of [`mem_pressure_full_avg10`] and [`io_pressure_full_avg10`]:
/// the `full` line's `avg10` field.
fn psi_full_avg10_from(body: &str) -> Option<f64> {
    psi_avg10_from(body, "full")
}

/// Parse the `avg10=` field of the `kind` (`some` or `full`) line of a PSI
/// file body. A PSI file looks like:
///
/// ```text
/// some avg10=0.00 avg60=0.00 avg300=0.00 total=123456
/// full avg10=0.00 avg60=0.00 avg300=0.00 total=987654
/// ```
fn psi_avg10_from(body: &str, kind: &str) -> Option<f64> {
    let line = body
        .lines()
        .find(|line| line.split_whitespace().next() == Some(kind))?;
    let field = line.split_whitespace().find(|f| f.starts_with("avg10="))?;
    field.strip_prefix("avg10=")?.parse().ok()
}

/// Completion checks take precedence over exploratory builds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdmissionClass {
    /// A worker still exploring or iterating.
    Exploratory,
    /// The canonical or divergent completion verification.
    Completion,
}

/// Select the next request, preserving arrival order within each class.
fn next_request(queue: &VecDeque<(u64, AdmissionClass)>) -> Option<usize> {
    queue
        .iter()
        .position(|(_, class)| *class == AdmissionClass::Completion)
        .or_else(|| (!queue.is_empty()).then_some(0))
}

/// The mutable half of the controller: the running count and priority queue.
struct Gate {
    /// Heavy commands holding a slot right now.
    running: usize,
    /// Occupied slot indices; allocation always chooses the lowest free one.
    slots: Vec<bool>,
    /// Requests in arrival order, selected by class and then FIFO.
    queue: VecDeque<(u64, AdmissionClass)>,
}

/// The `/proc` readings, cached for [`SAMPLE_TTL`].
struct HostCache {
    /// Test seam: when set, returned instead of reading `/proc`.
    fixed: Option<HostSample>,
    /// The last real reading and when it was taken.
    sampled: Option<(Instant, HostSample)>,
}

struct Inner {
    cores: usize,
    max_heavy: usize,
    reserve_mb: u64,
    estimate_mb: u64,
    cpu_pressure_max: f64,
    mem_pressure_max: f64,
    io_pressure_max: f64,
    /// Serializes decide-and-reserve so two waiters waking on the same
    /// notification cannot both be granted the last slot. Never held across an
    /// await point.
    gate: std::sync::Mutex<Gate>,
    /// Signalled whenever a slot frees or a request is queued.
    wake: Notify,
    host: std::sync::Mutex<HostCache>,
    next_id: AtomicU64,
}

/// Priority admission controller for heavy commands.
///
/// Cloneable and cheap: every clone shares one counter, one queue and one
/// wakeup channel, so the pool hands the same controller to every worker.
#[derive(Clone)]
pub struct AdmissionController {
    inner: std::sync::Arc<Inner>,
}

impl AdmissionController {
    /// A controller with explicit limits.
    pub fn new(
        max_heavy: usize,
        cores: usize,
        reserve_mb: u64,
        estimate_mb: u64,
        cpu_pressure_max: f64,
        mem_pressure_max: f64,
        io_pressure_max: f64,
    ) -> Self {
        Self {
            inner: std::sync::Arc::new(Inner {
                cores,
                max_heavy: max_heavy.max(1),
                reserve_mb,
                estimate_mb,
                cpu_pressure_max,
                mem_pressure_max,
                io_pressure_max,
                gate: std::sync::Mutex::new(Gate {
                    running: 0,
                    slots: vec![false; max_heavy.max(1)],
                    queue: VecDeque::new(),
                }),
                wake: Notify::new(),
                host: std::sync::Mutex::new(HostCache {
                    fixed: None,
                    sampled: None,
                }),
                next_id: AtomicU64::new(1),
            }),
        }
    }

    /// Limits from the environment: `BASH_BUILD_LIMIT` (default the core
    /// count), `HUB_MEM_RESERVE_MB` and `HUB_BUILD_MEM_MB`.
    pub fn from_env() -> Self {
        crate::cache::start_target_sweep();
        let cores = crate::config::cores();
        let max_heavy = crate::config::env_parse(BUILD_LIMIT_ENV).unwrap_or(cores);
        Self::new(
            max_heavy,
            cores,
            crate::config::env_parse(MEM_RESERVE_ENV).unwrap_or(DEFAULT_MEM_RESERVE_MB),
            crate::config::env_parse(BUILD_MEM_ENV).unwrap_or(DEFAULT_BUILD_MEM_MB),
            crate::config::env_parse(CPU_PRESSURE_MAX_ENV).unwrap_or(DEFAULT_CPU_PRESSURE_MAX),
            crate::config::env_parse(MEM_PRESSURE_MAX_ENV).unwrap_or(DEFAULT_MEM_PRESSURE_MAX),
            crate::config::env_parse(IO_PRESSURE_MAX_ENV).unwrap_or(DEFAULT_IO_PRESSURE_MAX),
        )
    }

    /// Heavy commands allowed at once.
    pub fn max_heavy(&self) -> usize {
        self.inner.max_heavy
    }

    /// Cores the job count is divided over.
    pub fn cores(&self) -> usize {
        self.inner.cores
    }

    /// Memory held back from the admission estimate, in MB.
    pub fn reserve_mb(&self) -> u64 {
        self.inner.reserve_mb
    }

    /// Estimated memory of one heavy command, in MB.
    pub fn estimate_mb(&self) -> u64 {
        self.inner.estimate_mb
    }

    /// Heavy commands holding a slot right now.
    pub fn running_heavy(&self) -> usize {
        self.lock_gate().running
    }

    /// Number of requests queued for a heavy slot.
    pub fn waiting(&self) -> usize {
        self.lock_gate().queue.len()
    }

    /// Take a heavy slot, prioritizing completion checks over exploration and
    /// waiting FIFO within each class. The permit holds the slot until dropped.
    ///
    /// Cancellation-safe: a request dropped while queued (its worker killed,
    /// its step timed out) leaves the queue, so it can never hold the head of
    /// the line for every later request.
    pub async fn acquire(&self, class: AdmissionClass) -> HeavyPermit {
        let id = self.inner.next_id.fetch_add(1, Ordering::Relaxed);
        let ticket = QueueTicket {
            controller: self,
            id,
        };
        loop {
            // Register before deciding so a release between the decision and
            // the await cannot lose its wakeup.
            let notified = self.inner.wake.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            let sample = self.sample();
            // One critical section decides *and* reserves, so two waiters
            // woken by the same notification cannot both take the last slot.
            let outcome = {
                let mut gate = self.lock_gate();
                if !gate.queue.iter().any(|(queued, _)| *queued == id) {
                    gate.queue.push_back((id, class));
                }
                let next = next_request(&gate.queue).expect("queued request");
                if gate.queue[next].0 != id {
                    None
                } else {
                    match admit(&AdmissionInputs {
                        running: gate.running,
                        max: self.inner.max_heavy,
                        mem_available_mb: sample.mem_available_mb,
                        reserve_mb: self.inner.reserve_mb,
                        estimate_mb: self.inner.estimate_mb,
                        load1: sample.load1,
                        cores: self.inner.cores,
                        cpu_pressure_max: self.inner.cpu_pressure_max,
                        mem_pressure_max: self.inner.mem_pressure_max,
                        io_pressure_max: self.inner.io_pressure_max,
                        cpu_some_avg10: sample.cpu_some_avg10,
                        mem_full_avg10: sample.mem_full_avg10,
                        io_full_avg10: sample.io_full_avg10,
                    }) {
                        Decision::Granted { jobs } => {
                            gate.queue.remove(next);
                            let slot = gate.slots.iter().position(|used| !used).expect("free slot");
                            gate.slots[slot] = true;
                            gate.running += 1;
                            Some(Ok((jobs, slot)))
                        }
                        Decision::Waiting(blocked) => Some(Err(blocked)),
                    }
                }
            };
            match outcome {
                Some(Ok((jobs, slot))) => {
                    // Granted: the ticket already left the queue.
                    std::mem::forget(ticket);
                    // The slot this request took may be the one the next
                    // queued request was waiting for.
                    self.inner.wake.notify_waiters();
                    return HeavyPermit {
                        controller: self.clone(),
                        jobs,
                        slot,
                    };
                }
                Some(Err(blocked)) => debug!(
                    running = self.running_heavy(),
                    max = self.inner.max_heavy,
                    cores = self.inner.cores,
                    mem_available_mb = sample.mem_available_mb,
                    reserve_mb = self.inner.reserve_mb,
                    estimate_mb = self.inner.estimate_mb,
                    load1 = sample.load1,
                    cpu_some_avg10 = sample.cpu_some_avg10,
                    cpu_pressure_max = self.inner.cpu_pressure_max,
                    mem_full_avg10 = sample.mem_full_avg10,
                    mem_pressure_max = self.inner.mem_pressure_max,
                    io_full_avg10 = sample.io_full_avg10,
                    io_pressure_max = self.inner.io_pressure_max,
                    reason = blocked.reason(),
                    "Heavy command waiting for admission"
                ),
                // A higher-priority or older peer is ahead; re-evaluate
                // when that one is granted or gives up.
                None => {}
            }
            tokio::select! {
                _ = &mut notified => {}
                _ = tokio::time::sleep(RE_EVALUATE_INTERVAL) => {}
            }
        }
    }

    /// Pin the host readings, bypassing `/proc` (test seam).
    #[doc(hidden)]
    pub fn __test_set_host_sample(&self, sample: Option<HostSample>) {
        self.lock_host().fixed = sample;
    }

    fn lock_gate(&self) -> std::sync::MutexGuard<'_, Gate> {
        self.inner
            .gate
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn lock_host(&self) -> std::sync::MutexGuard<'_, HostCache> {
        self.inner
            .host
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// The host readings, from the test seam, the cache or `/proc`.
    fn sample(&self) -> HostSample {
        let mut host = self.lock_host();
        if let Some(fixed) = host.fixed {
            return fixed;
        }
        if let Some((at, sample)) = host.sampled
            && at.elapsed() < SAMPLE_TTL
        {
            return sample;
        }
        let fresh = HostSample {
            mem_available_mb: mem_available_mb(),
            load1: loadavg_1m(),
            cpu_some_avg10: cpu_pressure_some_avg10(),
            mem_full_avg10: mem_pressure_full_avg10(),
            io_full_avg10: io_pressure_full_avg10(),
        };
        host.sampled = Some((Instant::now(), fresh));
        fresh
    }
}

/// A queued request's place in line, released if the request is dropped.
struct QueueTicket<'a> {
    controller: &'a AdmissionController,
    id: u64,
}

impl Drop for QueueTicket<'_> {
    fn drop(&mut self) {
        let mut gate = self.controller.lock_gate();
        let was_head =
            next_request(&gate.queue).is_some_and(|index| gate.queue[index].0 == self.id);
        gate.queue.retain(|(queued, _)| *queued != self.id);
        drop(gate);
        if was_head {
            // The next request in line may be admissible right now.
            self.controller.inner.wake.notify_waiters();
        }
    }
}

/// A granted heavy slot, held for the whole command.
///
/// Dropping it frees the slot and wakes the oldest queued request, so the
/// count and the queue stay consistent on every exit path.
pub struct HeavyPermit {
    controller: AdmissionController,
    jobs: usize,
    slot: usize,
}

impl HeavyPermit {
    /// Exclusive build slot index, held until this permit is dropped.
    pub fn slot(&self) -> usize {
        self.slot
    }

    /// Jobs the command may use.
    pub fn jobs(&self) -> usize {
        self.jobs
    }

    /// Heavy commands running, this one included.
    pub fn running_heavy(&self) -> usize {
        self.controller.running_heavy()
    }
}

impl Drop for HeavyPermit {
    fn drop(&mut self) {
        {
            let mut gate = self.controller.lock_gate();
            gate.slots[self.slot] = false;
            gate.running = gate.running.saturating_sub(1);
        }
        self.controller.inner.wake.notify_waiters();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn open() -> AdmissionInputs {
        AdmissionInputs {
            running: 0,
            max: 4,
            mem_available_mb: Some(12_000),
            reserve_mb: 2048,
            estimate_mb: 1536,
            load1: Some(1.0),
            cores: 4,
            cpu_pressure_max: DEFAULT_CPU_PRESSURE_MAX,
            mem_pressure_max: DEFAULT_MEM_PRESSURE_MAX,
            io_pressure_max: DEFAULT_IO_PRESSURE_MAX,
            cpu_some_avg10: None,
            mem_full_avg10: None,
            io_full_avg10: None,
        }
    }

    #[tokio::test]
    async fn slots_are_exclusive_and_reuse_the_lowest_free_index() {
        let controller = AdmissionController::new(3, 4, 0, 0, 60.0, 10.0, 40.0);
        controller.__test_set_host_sample(Some(HostSample::default()));
        let first = controller.acquire(AdmissionClass::Exploratory).await;
        let second = controller.acquire(AdmissionClass::Exploratory).await;
        let third = controller.acquire(AdmissionClass::Exploratory).await;
        assert_eq!((first.slot(), second.slot(), third.slot()), (0, 1, 2));
        drop(second);
        let reused = controller.acquire(AdmissionClass::Exploratory).await;
        assert_eq!(reused.slot(), 1);
        assert_eq!(controller.running_heavy(), 3);
        drop((first, third, reused));
        assert_eq!(controller.running_heavy(), 0);
        assert_eq!(
            controller.acquire(AdmissionClass::Exploratory).await.slot(),
            0
        );
    }

    #[test]
    fn first_build_is_always_admitted() {
        // The progress guarantee: with nothing heavy running the slot is
        // granted even when memory is tight and the load is high.
        let inputs = AdmissionInputs {
            running: 0,
            mem_available_mb: Some(100),
            load1: Some(99.0),
            ..open()
        };
        assert!(matches!(admit(&inputs), Decision::Granted { .. }));
    }

    #[test]
    fn slot_limit_blocks() {
        let inputs = AdmissionInputs {
            running: 4,
            ..open()
        };
        assert_eq!(admit(&inputs), Decision::Waiting(Blocked::SlotLimit));
    }

    #[test]
    fn short_memory_blocks() {
        let inputs = AdmissionInputs {
            running: 1,
            mem_available_mb: Some(2048 + 1535),
            ..open()
        };
        assert_eq!(admit(&inputs), Decision::Waiting(Blocked::Memory));
    }

    #[test]
    fn enough_memory_admits() {
        let inputs = AdmissionInputs {
            running: 1,
            mem_available_mb: Some(2048 + 1536),
            ..open()
        };
        assert!(matches!(admit(&inputs), Decision::Granted { .. }));
    }

    #[test]
    fn high_load_blocks() {
        let inputs = AdmissionInputs {
            running: 1,
            load1: Some(5.0),
            ..open()
        };
        assert_eq!(admit(&inputs), Decision::Waiting(Blocked::Load));
    }

    #[test]
    fn load_just_below_ceiling_admits() {
        let inputs = AdmissionInputs {
            running: 1,
            load1: Some(4.99),
            ..open()
        };
        assert!(matches!(admit(&inputs), Decision::Granted { .. }));
    }

    #[test]
    fn unreadable_readings_are_skipped() {
        let inputs = AdmissionInputs {
            running: 1,
            mem_available_mb: None,
            load1: None,
            ..open()
        };
        assert!(matches!(admit(&inputs), Decision::Granted { .. }));
    }

    #[test]
    fn slot_limit_beats_progress_guarantee() {
        // `max == 0` can never admit, whatever the progress guarantee says.
        let inputs = AdmissionInputs {
            running: 0,
            max: 0,
            ..open()
        };
        assert_eq!(admit(&inputs), Decision::Waiting(Blocked::SlotLimit));
    }

    #[test]
    fn job_count_divides_cores_over_running() {
        assert_eq!(jobs_for(4, 1), 4);
        assert_eq!(jobs_for(4, 2), 2);
        assert_eq!(jobs_for(4, 3), 1);
        assert_eq!(jobs_for(4, 4), 1);
        assert_eq!(jobs_for(4, 99), 1);
        assert_eq!(jobs_for(4, 0), 4);
        assert_eq!(jobs_for(0, 1), 1);
    }

    #[test]
    fn granted_job_count_matches_running_after_grant() {
        let first = admit(&open());
        assert!(matches!(first, Decision::Granted { jobs: 4 }));
        let second = admit(&AdmissionInputs {
            running: 1,
            ..open()
        });
        assert!(matches!(second, Decision::Granted { jobs: 2 }));
    }

    #[test]
    fn meminfo_reader_parses_available() {
        let body = "MemTotal:       16384000 kB\nMemAvailable:    8192000 kB\n";
        assert_eq!(mem_available_mb_from(body), Some(8000));
        assert_eq!(mem_available_mb_from("MemTotal: 1 kB\n"), None);
    }

    #[test]
    fn loadavg_reader_parses_first_field() {
        assert_eq!(loadavg_1m_from("2.50 1.75 1.25 4/512 12345\n"), Some(2.5));
        assert_eq!(loadavg_1m_from(""), None);
    }

    #[test]
    fn cpu_pressure_blocks() {
        let inputs = AdmissionInputs {
            running: 1,
            cpu_some_avg10: Some(65.0),
            ..open()
        };
        assert_eq!(admit(&inputs), Decision::Waiting(Blocked::CpuPressure));
    }

    #[test]
    fn cpu_pressure_below_ceiling_admits() {
        let inputs = AdmissionInputs {
            running: 1,
            cpu_some_avg10: Some(59.9),
            ..open()
        };
        assert!(matches!(admit(&inputs), Decision::Granted { .. }));
    }

    #[test]
    fn memory_pressure_blocks() {
        let inputs = AdmissionInputs {
            running: 1,
            mem_full_avg10: Some(12.0),
            ..open()
        };
        assert_eq!(admit(&inputs), Decision::Waiting(Blocked::MemPressure));
    }

    #[test]
    fn io_pressure_blocks() {
        let inputs = AdmissionInputs {
            running: 1,
            io_full_avg10: Some(45.0),
            ..open()
        };
        assert_eq!(admit(&inputs), Decision::Waiting(Blocked::IoPressure));
    }

    #[test]
    fn unreadable_psi_criterion_is_skipped() {
        // CPU pressure is unreadable (None) but the other two are fine, so the
        // missing file is skipped rather than read as a failure.
        let inputs = AdmissionInputs {
            running: 1,
            cpu_some_avg10: None,
            mem_full_avg10: Some(5.0),
            io_full_avg10: Some(5.0),
            ..open()
        };
        assert!(matches!(admit(&inputs), Decision::Granted { .. }));
    }

    #[test]
    fn psi_present_ignores_the_load_average() {
        // PSI is available and clear, so a high load average must not block:
        // the load average is only the PSI-unavailable fallback.
        let inputs = AdmissionInputs {
            running: 1,
            load1: Some(99.0),
            cpu_some_avg10: Some(9.0),
            mem_full_avg10: Some(1.0),
            io_full_avg10: Some(2.0),
            ..open()
        };
        assert!(matches!(admit(&inputs), Decision::Granted { .. }));
    }

    #[test]
    fn psi_missing_falls_back_to_the_load_average() {
        // No PSI file was readable, so the load-average criterion applies.
        let inputs = AdmissionInputs {
            running: 1,
            load1: Some(5.0),
            cpu_some_avg10: None,
            mem_full_avg10: None,
            io_full_avg10: None,
            ..open()
        };
        assert_eq!(admit(&inputs), Decision::Waiting(Blocked::Load));
    }

    #[test]
    fn progress_guarantee_beats_pressure() {
        // With nothing heavy running the slot is granted even under maximum
        // pressure, so a saturated host can never deadlock the first build.
        let inputs = AdmissionInputs {
            running: 0,
            cpu_some_avg10: Some(99.0),
            mem_full_avg10: Some(99.0),
            io_full_avg10: Some(99.0),
            ..open()
        };
        assert!(matches!(admit(&inputs), Decision::Granted { .. }));
    }

    #[test]
    fn psi_reader_parses_avg10_by_field_name() {
        let body = "some avg10=0.00 avg60=0.00 avg300=0.00 total=123456\n                    full avg10=1.50 avg60=0.00 avg300=0.00 total=789012\n";
        assert_eq!(psi_avg10_from(body, "some"), Some(0.0));
        assert_eq!(psi_avg10_from(body, "full"), Some(1.5));
    }

    /// The queue order is a pure function of the requests' classes and
    /// arrival order: completion first, FIFO inside each class.
    #[test]
    fn next_request_serves_completion_before_exploration() {
        use AdmissionClass::{Completion, Exploratory};
        type Case = (&'static [(u64, AdmissionClass)], Option<usize>);
        let cases: &[Case] = &[
            (&[], None),
            (&[(1, Exploratory)], Some(0)),
            (&[(1, Completion)], Some(0)),
            // A late completion jumps ahead of an older exploration.
            (&[(1, Exploratory), (2, Completion)], Some(1)),
            // ... but not ahead of an older completion: FIFO inside the class.
            (
                &[(1, Completion), (2, Exploratory), (3, Completion)],
                Some(0),
            ),
            (
                &[(1, Exploratory), (2, Exploratory), (3, Completion)],
                Some(2),
            ),
            // Two completions keep their arrival order.
            (&[(1, Completion), (2, Completion)], Some(0)),
            (
                &[(1, Exploratory), (2, Completion), (3, Completion)],
                Some(1),
            ),
        ];
        for (queue, expected) in cases {
            let queue: VecDeque<(u64, AdmissionClass)> = queue.iter().copied().collect();
            assert_eq!(
                next_request(&queue),
                *expected,
                "wrong next request for {queue:?}"
            );
        }
    }

    async fn wait_for_queue(controller: &AdmissionController, expected: usize) {
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            while controller.waiting() != expected {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("requests must reach the expected queue length");
    }

    /// A completion request queued behind an exploratory one is served first
    /// when the slot frees, and the exploratory one follows.
    #[tokio::test]
    async fn a_completion_request_is_served_before_an_exploratory_one() {
        let controller = AdmissionController::new(1, 4, 2048, 1536, 60.0, 10.0, 40.0);
        controller.__test_set_host_sample(Some(HostSample {
            mem_available_mb: Some(12_000),
            load1: Some(0.5),
            cpu_some_avg10: None,
            mem_full_avg10: None,
            io_full_avg10: None,
        }));
        let held = controller.acquire(AdmissionClass::Exploratory).await;

        let order = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let spawn = |class: AdmissionClass, tag: &'static str| {
            let controller = controller.clone();
            let order = order.clone();
            tokio::spawn(async move {
                let _permit = controller.acquire(class).await;
                order.lock().expect("order lock").push(tag);
            })
        };
        let exploring = spawn(AdmissionClass::Exploratory, "exploring");
        // The exploratory request must be queued before the completion one.
        wait_for_queue(&controller, 1).await;
        let completing = spawn(AdmissionClass::Completion, "completing");
        wait_for_queue(&controller, 2).await;
        assert_eq!(controller.waiting(), 2, "both requests must be queued");

        drop(held);
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            completing.await.expect("completion task finishes");
            exploring.await.expect("exploration task finishes");
        })
        .await
        .expect("both queued requests must make progress");
        assert_eq!(
            *order.lock().expect("order lock"),
            vec!["completing", "exploring"],
            "the completion gate must not wait behind an exploratory build"
        );
    }

    /// A cancelled exploratory request cannot hold a completion one back: the
    /// ticket leaves its queue, so the completion request is served next.
    #[tokio::test]
    async fn a_cancelled_exploration_does_not_block_a_completion() {
        let controller = AdmissionController::new(1, 4, 2048, 1536, 60.0, 10.0, 40.0);
        controller.__test_set_host_sample(Some(HostSample {
            mem_available_mb: Some(12_000),
            load1: Some(0.5),
            cpu_some_avg10: None,
            mem_full_avg10: None,
            io_full_avg10: None,
        }));
        let held = controller.acquire(AdmissionClass::Exploratory).await;
        let doomed = tokio::spawn({
            let controller = controller.clone();
            async move { controller.acquire(AdmissionClass::Exploratory).await }
        });
        wait_for_queue(&controller, 1).await;
        let completing = tokio::spawn({
            let controller = controller.clone();
            async move { controller.acquire(AdmissionClass::Completion).await }
        });
        wait_for_queue(&controller, 2).await;
        assert_eq!(controller.waiting(), 2);

        doomed.abort();
        let _ = doomed.await;
        assert_eq!(controller.waiting(), 1, "the aborted request must leave");
        drop(held);
        let permit = tokio::time::timeout(std::time::Duration::from_secs(5), completing)
            .await
            .expect("the completion request must not wait behind the ghost")
            .expect("completion task completes");
        drop(permit);
    }

    #[test]
    fn psi_reader_is_robust_to_shape() {
        // The requested line is absent.
        assert_eq!(
            psi_avg10_from("full avg10=2.00 avg60=0.00 avg300=0.00 total=1\n", "some"),
            None
        );
        // avg10 is matched by name, so a reordered body still parses.
        assert_eq!(
            psi_avg10_from("some avg60=0.00 avg10=3.25 avg300=0.00 total=1\n", "some"),
            Some(3.25)
        );
        // A non-numeric value yields None rather than a panic.
        assert_eq!(
            psi_avg10_from("some avg10=nope avg60=0.00 avg300=0.00 total=1\n", "some"),
            None
        );
        // An empty body yields None.
        assert_eq!(psi_avg10_from("", "some"), None);
    }
}
