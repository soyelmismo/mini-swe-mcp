//! A watch outlives the daemon underneath it, however the daemon goes away.
//!
//! A handover is not an orderly event: the daemon closes its listeners, and
//! whatever the connection's unread buffer still holds is discarded with them,
//! so the client can see an EOF, a reset or a broken pipe for the same stop.
//! Every one of those must answer with the same chase — dial again, say hello
//! with the same identity, replay what was missed — and the chase must end
//! with an explanation when no daemon comes back.

mod common;

use mini_swe_mcp::pool::{RegistryStatus, WorkerMeta};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};

/// The env var the CLI and the hub client both read for the reconnect budget.
const RECONNECT_ENV: &str = mini_swe_mcp::hub::RECONNECT_DEADLINE_ENV;

/// Seconds this test gives the reconnect budget. Long enough for the
/// replacement daemon to start and its recovery to run before the watch
/// reconnects, and bounded so a watch that never reconnects ends on its own.
const RECONNECT_SECS: &str = "60";

/// Wait until the hub log holds at least `count` lines mentioning `event`, so
/// the test polls for the daemon instead of sleeping for it.
async fn wait_for_log(hub_dir: &Path, event: &str, count: usize) {
    let deadline = std::time::Instant::now() + Duration::from_secs(30);
    loop {
        let seen = std::fs::read_to_string(hub_dir.join("hub.log"))
            .map(|log| log.lines().filter(|line| mentions(line, event)).count())
            .unwrap_or(0);
        if seen >= count {
            return;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "hub.log never showed {count} x {event}:\n{}",
            std::fs::read_to_string(hub_dir.join("hub.log")).unwrap_or_default()
        );
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}

/// Whether a hub-log line is the event this test waits for.
fn mentions(line: &str, event: &str) -> bool {
    match event {
        "listening" => line.ends_with(" listening"),
        "recovered" => line.contains("Recovered orphaned hub workers"),
        _ => line.contains(event),
    }
}

/// The pids the hub log announced, newest last.
fn daemon_pids(hub_dir: &Path) -> Vec<i32> {
    let log = std::fs::read_to_string(hub_dir.join("hub.log")).unwrap_or_default();
    log.lines()
        .filter(|line| line.ends_with(" listening"))
        .filter_map(|line| {
            line.split_whitespace()
                .find_map(|word| word.strip_prefix("pid=")?.parse().ok())
        })
        .collect()
}

/// Signal `pid`, turning a pid that is already gone into `None`.
fn signal(pid: i32, sig: libc::c_int) -> Option<()> {
    // SAFETY: `kill` takes plain integers; a stale pid only yields ESRCH.
    if unsafe { libc::kill(pid, sig) } == 0 {
        Some(())
    } else {
        None
    }
}

/// The daemon a CLI `watch` process starts for itself, so a `SIGTERM` stops
/// that one and not any other daemon on the host.
struct WatchDaemon {
    hub_dir: PathBuf,
    sleeper: Child,
    pids: BTreeSet<i32>,
    armed: bool,
}

impl WatchDaemon {
    fn new(hub_dir: &Path) -> Self {
        Self {
            hub_dir: hub_dir.to_path_buf(),
            sleeper: Command::new("sleep")
                .arg("60")
                .kill_on_drop(true)
                .spawn()
                .expect("stand-in worker process"),
            pids: BTreeSet::new(),
            armed: false,
        }
    }

    /// The pid of the daemon the watch is talking to.
    fn current(&self) -> i32 {
        *daemon_pids(&self.hub_dir)
            .last()
            .expect("a daemon announced itself")
    }

    /// Stop that daemon the way a handover does: abruptly, so the connection is
    /// cut rather than closed in an orderly way.
    fn cut(&mut self) {
        assert!(
            signal(self.current(), libc::SIGKILL).is_some(),
            "kill the daemon the watch is talking to"
        );
        self.pids.insert(self.current());
    }

    /// Stop every daemon that answered this test's socket, and take the stand-in
    /// worker with them, so a failing test leaves nothing behind.
    fn close(&mut self) {
        for pid in std::mem::take(&mut self.pids) {
            signal(pid, libc::SIGTERM);
        }
        self.pids.clear();
        for pid in daemon_pids(&self.hub_dir) {
            signal(pid, libc::SIGTERM);
        }
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        while std::time::Instant::now() < deadline
            && self.pids.iter().any(|pid| matches!(signal(*pid, 0), Some(())))
        {
            std::thread::sleep(Duration::from_millis(20));
        }
        self.armed = false;
        let _ = self.sleeper.start_kill();
        drop(self.sleeper.wait());
    }
}

impl Drop for WatchDaemon {
    fn drop(&mut self) {
        if self.armed {
            self.close();
        }
    }
}

/// A CLI `watch` following a daemon that is cut from under it: the watch
/// reconnects to the replacement and still delivers the next event.
///
/// The cut is abrupt on purpose — the process is killed, nothing is flushed —
/// so the watch sees a transport failure rather than a clean EOF, and the
/// reconnect has to re-announce the identity and replay what it missed.
#[tokio::test]
async fn a_watch_follows_a_daemon_cut_with_a_reset() {
    let isolated = common::IsolatedPool::new(2, "watch-cut");
    let swe = isolated.root().path().to_path_buf();
    let pool = isolated.pool.clone();
    // Its own directory, not the pool's base: the watch spawns the replacement
    // daemon itself, so everything it starts has to stay inside this test.
    let hub = common::TempDir::new(&swe.join("hub"), "watch-cut");
    let hub_dir = hub.path().to_path_buf();

    // The watch owns the hub directory from here on, so the replacement daemon
    // is the one it starts and the only one this test ever signals.
    let mut owned = WatchDaemon::new(&hub_dir);
    owned.armed = true;

    // A live worker owned by the watching agent, standing in for one the
    // orchestrator dispatched. Its pid is a real process, so neither the cut
    // nor the replacement's recovery treats it as dead.
    let mut meta = WorkerMeta {
        task: "reconnect probe".to_string(),
        ..WorkerMeta::test_meta("watch-cut", "reconnect-test")
    };
    meta.pid = owned.sleeper.id().expect("the sleeper has a pid");
    pool.__test_save_status(
        &meta,
        "test",
        RegistryStatus::Running,
        1,
        10,
        "probe",
        None,
    );

    let mut watch = Command::from(watch_command(&hub_dir, &swe))
        .kill_on_drop(true)
        .spawn()
        .expect("start the watch");
    let mut output = BufReader::new(watch.stdout.take().expect("watch stdout"));
    let mut errors = BufReader::new(watch.stderr.take().expect("watch stderr"));
    // The watch starts the hub itself, so its own output is the only place a
    // refusal would show: wait for it, and say what it said instead.
    tokio::select! {
        _ = wait_for_log(&hub_dir, "Serving MCP connection", 1) => {}
        _ = tokio::time::timeout(Duration::from_secs(20), read_to_end(&mut errors)) => {
            let status = watch.wait().await.expect("the watch process joins");
            panic!(
                "the watch never reached the daemon: {status:?}, stderr: {:?}, hub.dir={}, swe={}, hub.log:\n{}",
                read_to_end(&mut errors).await,
                hub_dir.display(),
                swe.display(),
                std::fs::read_to_string(hub_dir.join("hub.log")).unwrap_or_default(),
            );
        }
    }
    owned.cut();

    // The watch follows the daemon it lost: the replacement hub starts, answers
    // it again, and the CLI process is still the one watching.
    wait_for_log(&hub_dir, "listening", 2).await;
    wait_for_log(&hub_dir, "recovered", 2).await;
    assert!(
        matches!(watch.try_wait(), Ok(None)),
        "the watch must survive the reset, not exit with it; stderr: {:?}",
        read_to_end(&mut errors).await
    );

    // And the delivery continues: the worker's terminal status is the next
    // event, and it reaches the same CLI process over the new connection.
    std::fs::create_dir_all(format!("{}/swe-wt-watch-cut", swe.display()))
        .expect("the worker's worktree");
    pool.__test_reset_registry_throttle("watch-cut");
    pool.__test_save_status(
        &meta,
        "test",
        RegistryStatus::Failed,
        2,
        10,
        "probe ended",
        None,
    );

    match read_until(&mut output, "watch-cut", Duration::from_secs(30)).await {
        Some(printed) => {
            assert!(
                printed.contains("watch-cut"),
                "the event belongs to the watched worker: {printed}"
            );
        }
        None => panic!(
            "the reconnected watch must deliver the next event; saw {:?}",
            read_to_end(&mut output).await
        ),
    }
    let finished = tokio::time::timeout(Duration::from_secs(30), watch.wait())
        .await
        .expect("the watch ends once its worker is terminal")
        .expect("the watch exits");
    assert!(
        finished.success(),
        "the watch reported its last event: {:?}",
        read_to_end(&mut errors).await
    );
    owned.close();
}

/// The `watch` CLI this test drives: a child whose identity is pinned to
/// `reconnect-test`, so the replacement's recovery has rows to hand back to the
/// watch it follows.
fn watch_command(hub_dir: &Path, swe: &Path) -> std::process::Command {
    let mut watch = common::binary_command(&common::binary_path());
    watch
        .args(["watch", "--json"])
        .env_remove("MINI_SWE_NO_DAEMON")
        .env("SWE_HUB_DIR", hub_dir)
        .env("SWE_TEMP_DIR", swe)
        .env(RECONNECT_ENV, RECONNECT_SECS)
        .env("ENV_FILE", "/nonexistent-mini-swe-reconnect")
        .env("OPENAI_API_KEY", "test-key-not-used-by-the-reconnect-test")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    // Set after `binary_command`'s scrub, which clears exactly this override:
    // the replacement's recovery has rows to hand back only to this identity.
    watch.env("MINI_SWE_AGENT_ID", "reconnect-test");
    watch
}


/// Drain `stream` to its end, so a child that is never waited on cannot fill a
/// pipe and block on its next write.
async fn read_to_end<R: tokio::io::AsyncRead + Unpin>(stream: &mut R) -> Vec<u8> {
    let mut seen = Vec::new();
    let _ = tokio::io::AsyncReadExt::read_to_end(stream, &mut seen).await;
    seen
}

/// Read from `stream` until `needle` appears, so the test polls for the event
/// instead of sleeping for it.
async fn read_until<R>(
    stream: &mut R,
    needle: &str,
    within: Duration,
) -> Option<String>
where
    R: tokio::io::AsyncBufRead + Unpin,
{
    let mut seen = String::new();
    let deadline = tokio::time::Instant::now() + within;
    loop {
        let left = deadline.saturating_duration_since(tokio::time::Instant::now());
        if left.is_zero() {
            return if seen.contains(needle) { Some(seen) } else { None };
        }
        let mut line = String::new();
        match tokio::time::timeout(left, stream.read_line(&mut line)).await {
            Ok(Ok(0)) | Err(_) => {
                return if seen.contains(needle) { Some(seen) } else { None };
            }
            Ok(Ok(_)) => seen.push_str(&line),
            Ok(Err(_)) => return None,
        }
    }
}
