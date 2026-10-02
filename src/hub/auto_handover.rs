//! The daemon notices its own rebuilt executable and hands over by itself.
//!
//! A planned handover (D1: wait for a quiet moment, at most the handover
//! deadline, then graceful stop that checkpoints and interrupts live workers,
//! respawn from the executable path, the new daemon auto-continues them) was
//! armed only when a *newer client* connected and sent `hub/handover`. After
//! `cargo build --release` the old daemon therefore kept serving until some
//! CLI call happened to arrive: a rebuild alone did nothing.
//!
//! This module makes the handover OTA-like. The daemon polls its own
//! executable path — the stripped path from [`crate::hub::exe_path`], because
//! a `cargo build` replaces the binary in place — and when a *different,
//! complete* build has been sitting there stably, it arms the very same
//! planned handover a newer client would.
//!
//! "Complete" is three checks, in order, and a build that fails any of them
//! is never handed over to:
//!
//! 1. The path is a regular, executable file.
//! 2. Its fingerprint (inode, size, mtime) has been unchanged for
//!    [`stable_for`], so a `cargo` write in progress is not picked up.
//! 3. The binary actually runs: `<exe> --build-id` exits 0 and reports a
//!    build identity, because a daemon must never hand over to a build that
//!    fails to start.
//!
//! Only a build the existing identity check
//! ([`crate::hub::client::supersedes`]) calls newer arms the handover.
//! `HUB_AUTO_HANDOVER=0` disables the watch, and only a daemon that can
//! respawn itself ([`crate::hub::HubConfig`] built by `run_daemon`) watches at
//! all. `MINI_SWE_HUB_EXE_POLL_MS` and `MINI_SWE_HUB_EXE_STABLE_MS` shorten
//! the two windows for tests; production never sets them.

use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::Value;
use tracing::info;

use crate::mcp::McpServer;

/// How often the daemon re-stats its own executable.
const DEFAULT_POLL: Duration = Duration::from_secs(2);

/// How long a changed executable must look identical before it counts as a
/// complete build rather than a `cargo` write in progress.
const DEFAULT_STABLE: Duration = Duration::from_secs(3);

/// How long the `<exe> --build-id` probe may take before it is treated as a
/// build that fails to start.
const PROBE_TIMEOUT: Duration = Duration::from_secs(10);

/// Environment variable that disables the watch.
pub const AUTO_HANDOVER_ENV: &str = "HUB_AUTO_HANDOVER";

/// Test seam: poll interval in milliseconds.
const POLL_ENV: &str = "MINI_SWE_HUB_EXE_POLL_MS";

/// Test seam: stability window in milliseconds.
const STABLE_ENV: &str = "MINI_SWE_HUB_EXE_STABLE_MS";

/// Whether the daemon watches its own executable for a newer build.
pub fn enabled() -> bool {
    std::env::var(AUTO_HANDOVER_ENV)
        .map(|v| v != "0")
        .unwrap_or(true)
}

/// How often the daemon re-stats its own executable, overridable for tests.
fn poll_interval() -> Duration {
    crate::config::env_parse::<u64>(POLL_ENV)
        .map(Duration::from_millis)
        .unwrap_or(DEFAULT_POLL)
}

/// How long a changed executable must look identical, overridable for tests.
fn stable_for() -> Duration {
    crate::config::env_parse::<u64>(STABLE_ENV)
        .map(Duration::from_millis)
        .unwrap_or(DEFAULT_STABLE)
}

/// What a stat of the executable says: the identity of the file currently at
/// the path, not of the build running from it. Cargo replaces the binary in
/// place, so a rebuild always moves at least one of these fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fingerprint {
    ino: u64,
    size: u64,
    mtime: i64,
    mtime_nsec: i64,
}

impl Fingerprint {
    /// The fingerprint of the regular file at `path`, or `None` when the
    /// path is missing, not a file, or cannot be stat'ed.
    pub fn of(path: &Path) -> Option<Self> {
        let meta = std::fs::metadata(path).ok()?;
        if !meta.is_file() {
            return None;
        }
        Some(Self {
            ino: meta.ino(),
            size: meta.size(),
            mtime: meta.mtime(),
            mtime_nsec: meta.mtime_nsec(),
        })
    }

    /// Whether the file at `path` is executable at all.
    fn is_executable(path: &Path) -> bool {
        std::fs::metadata(path)
            .map(|meta| meta.is_file() && meta.permissions().mode() & 0o111 != 0)
            .unwrap_or(false)
    }
}

/// What one observation of the executable path decided.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// Keep watching.
    Wait,
    /// The path has held a different fingerprint, unchanged, for the whole
    /// stability window: probe the build that is there now.
    Probe,
}

/// The watch state for one executable path.
///
/// The state machine is separate from the async loop so it can be exercised
/// without a real executable or a clock.
#[derive(Debug)]
pub struct ExeWatch {
    /// The fingerprint of the build this daemon is running, or of the last
    /// replacement that was probed and did not arm a handover.
    baseline: Option<Fingerprint>,
    /// A fingerprint that differs from `baseline`, and when it was first
    /// seen. Reset whenever the file moves or disappears.
    changed: Option<(Fingerprint, Instant)>,
}

impl ExeWatch {
    /// A watch whose baseline is the build currently at `path` (or `None`
    /// when there is no file there yet).
    pub fn new(baseline: Option<Fingerprint>) -> Self {
        Self {
            baseline,
            changed: None,
        }
    }

    /// Feed the fingerprint the path shows now.
    ///
    /// `Verdict::Probe` is returned once a changed fingerprint has been
    /// stable for `stable_for`; the caller then probes the build and either
    /// arms the handover or [`ExeWatch::adopt`]s the fingerprint as the new
    /// baseline.
    pub fn observe(
        &mut self,
        fp: Option<Fingerprint>,
        at: Instant,
        stable_for: Duration,
    ) -> Verdict {
        let Some(fp) = fp else {
            // The path is gone: a build is being written over it. A change
            // only counts once the new file has been there, stably, so
            // forget any candidate and keep the baseline.
            self.changed = None;
            return Verdict::Wait;
        };
        match self.changed {
            Some((changed, since)) if changed == fp => {
                if at.saturating_duration_since(since) >= stable_for {
                    Verdict::Probe
                } else {
                    Verdict::Wait
                }
            }
            _ if self.baseline == Some(fp) => {
                self.changed = None;
                Verdict::Wait
            }
            _ => {
                self.changed = Some((fp, at));
                Verdict::Wait
            }
        }
    }

    /// Adopt the fingerprint the path now holds as the baseline: the build
    /// there was probed and did not arm a handover (it is the same build,
    /// an older one, or one that does not run).
    pub fn adopt(&mut self, fp: Option<Fingerprint>) {
        self.baseline = fp;
        self.changed = None;
    }
}

/// The build identity `<exe> --build-id` reports, or `None` when the binary
/// does not run — a half-written file, a non-executable one, or one that
/// fails to start are all `None`, and a daemon never hands over to any of
/// them.
async fn probe_build(exe: &Path) -> Option<Value> {
    let probe = tokio::process::Command::new(exe)
        .arg("--build-id")
        .stdin(std::process::Stdio::null())
        .kill_on_drop(true)
        .output();
    let output = tokio::time::timeout(PROBE_TIMEOUT, probe)
        .await
        .ok()?
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8(output.stdout).ok()?;
    serde_json::from_str(text.trim()).ok()
}

/// Watch `path` until a newer build has replaced it, then arm the planned
/// handover on `server`.
///
/// Returns when the handover is armed (the daemon's shutdown watcher then
/// stops it) or immediately when the watch is disabled.
pub async fn watch(
    path: PathBuf,
    running: Value,
    server: Arc<McpServer>,
    deadline: Duration,
    log: PathBuf,
) {
    // The baseline is the build this daemon actually runs, not whatever is at
    // the path now: `/proc/self/exe` still resolves to the replaced inode, so
    // a rebuild that landed between `exec` and this watcher is not mistaken
    // for the running build. The path is the fallback without procfs.
    let running_fp =
        Fingerprint::of(Path::new("/proc/self/exe")).or_else(|| Fingerprint::of(&path));
    let mut watch = ExeWatch::new(running_fp);
    let poll = poll_interval();
    let stable = stable_for();
    loop {
        tokio::time::sleep(poll).await;
        let fp = Fingerprint::of(&path);
        if watch.observe(fp, Instant::now(), stable) != Verdict::Probe {
            continue;
        }
        // A build has been sitting there, unchanged, for the whole window.
        // It still has to prove it is a complete, runnable build before the
        // daemon hands over to it.
        if !Fingerprint::is_executable(&path) {
            watch.adopt(fp);
            continue;
        }
        let Some(build) = probe_build(&path).await else {
            watch.adopt(fp);
            continue;
        };
        if !crate::hub::client::supersedes(
            env!("CARGO_PKG_VERSION"),
            &build,
            env!("CARGO_PKG_VERSION"),
            &running,
        ) {
            watch.adopt(fp);
            continue;
        }
        let id = build["id"].as_str().unwrap_or_default();
        info!("executable changed: arming handover to build {id}");
        super::daemon::append_log(
            &log,
            &format!("executable changed: arming handover to build {id}"),
        );
        server.request_handover(deadline);
        return;
    }
}

#[cfg(test)]
#[path = "auto_handover_tests.rs"]
mod tests;
