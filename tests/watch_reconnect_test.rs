//! A watch outlives the daemon underneath it, however the daemon goes away.
//!
//! A handover is not an orderly event: the daemon closes its listeners, and
//! whatever the connection's unread buffer still holds is discarded with them,
//! so the client can see an EOF, a reset or a broken pipe for the same stop.
//! Every one of those must answer with the same chase — dial again, say hello
//! with the same identity, replay what was missed — and the chase must end
//! with an explanation when no daemon comes back.

mod common;

use mini_swe_mcp::hub::HubPaths;
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

/// Wait until the hub directory's socket accepts a connection.
///
/// The socket is not necessarily inside the hub directory: a path too long for
/// `sun_path` moves it to a short fallback directory, so the test asks
/// [`HubPaths`] where this one is instead of guessing.
async fn wait_for_socket(hub_dir: &Path) {
    let socket = HubPaths::new(hub_dir.to_path_buf()).socket();
    let deadline = std::time::Instant::now() + Duration::from_secs(30);
    loop {
        if std::os::unix::net::UnixStream::connect(&socket).is_ok() {
            return;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "nothing is listening on {}:\n{}",
            socket.display(),
            std::fs::read_to_string(hub_dir.join("hub.log")).unwrap_or_default()
        );
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}

/// The daemon under test: the same executable the watch would start for
/// itself, spawned here so the test can cut it.
fn daemon_command(hub_dir: &Path, swe: &Path) -> std::process::Command {
    let mut daemon = common::binary_command(&common::binary_path());
    daemon
        .arg("daemon")
        .env("SWE_HUB_DIR", hub_dir)
        .env("SWE_TEMP_DIR", swe)
        .env_remove("MINI_SWE_NO_DAEMON")
        .env("HUB_IDLE_SECS", "60")
        // Auto-resume off, so the replacement's recovery keeps the status the
        // row already had instead of starting a worker this test never wanted.
        .env("HUB_AUTO_RESUME", "0")
        .env("ENV_FILE", "/nonexistent-mini-swe-reconnect")
        .env("OPENAI_API_KEY", "test-key-not-used-by-the-reconnect-test")
        .stdout(Stdio::null())
        .stderr(
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(hub_dir.join("hub.log"))
                .expect("open the hub log"),
        );
    daemon
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

/// The daemons this test spawns in its own hub directory, and the stand-in
/// worker they keep alive.
///
/// Every daemon that answers this directory is in `pids` — including the
/// replacement the watch starts for itself — so a failing test signals only
/// those, and never a daemon belonging to anything else on the host.
struct WatchDaemon {
    hub_dir: PathBuf,
    sleeper: Child,
    pids: BTreeSet<i32>,
    /// Whether this test still has to stop its daemons. The struct that owns
    /// the directory arms this as soon as it starts the first one.
    armed: bool,
}

impl WatchDaemon {
    /// A stand-in worker process: a real pid, so neither the cut nor the
    /// replacement's recovery can decide the row it backs is dead.
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

    /// The pid of the newest daemon on this directory.
    fn current(&self) -> i32 {
        *daemon_pids(&self.hub_dir)
            .last()
            .expect("a daemon announced itself")
    }

    /// Stop the daemon the watch is talking to the way a busy handover does:
    /// abruptly, with nothing flushed, so the connection is cut rather than
    /// closed in an orderly way.
    fn cut(&mut self) {
        let pid = self.current();
        assert!(signal(pid, libc::SIGKILL).is_some(), "kill the daemon");
        self.pids.insert(pid);
    }

    /// Stop every daemon that answered this directory, and take the stand-in
    /// worker with them, so a failing test leaves nothing running.
    fn close(&mut self) {
        for pid in std::mem::take(&mut self.pids) {
            signal(pid, libc::SIGTERM);
        }
        for pid in daemon_pids(&self.hub_dir) {
            signal(pid, libc::SIGTERM);
        }
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        while std::time::Instant::now() < deadline
            && self
                .pids
                .iter()
                .chain(daemon_pids(&self.hub_dir).iter())
                .any(|pid| matches!(signal(*pid, 0), Some(())))
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
/// so the reconnect has to re-announce the identity and replay what it missed
/// rather than inherit a live connection. Which errno a given cut arrives as is
/// the kernel's business; `client_reconnect_tests` covers the transport errors
/// themselves, up to the reset an unread connection buffer produces.
#[tokio::test]
async fn a_watch_follows_a_daemon_that_is_cut_under_it() {
    let isolated = common::IsolatedPool::new(2, "watch-cut");
    let swe = isolated.root().path().to_path_buf();
    let pool = isolated.pool.clone();
    let hub = common::TempDir::new_in_tmp("watch-cut");
    let hub_dir = hub.path().to_path_buf();
    // The daemon refuses a hub directory anyone but its owner can read, and a
    // short fallback socket directory derived from it is owned the same way.
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(&hub_dir, std::fs::Permissions::from_mode(0o700))
        .expect("restrict the hub directory");
    let _fallback = common::fallback_socket_dir(&hub_dir);

    // From here on this test owns every daemon on this directory, including the
    // replacement the watch starts for itself.
    let mut owned = WatchDaemon::new(&hub_dir);
    owned.armed = true;

    // A live worker owned by the watching agent, standing in for one the
    // orchestrator dispatched. Its pid is a real process, so neither the cut
    // nor the replacement's recovery can decide the row it backs is dead.
    let mut meta = WorkerMeta {
        task: "reconnect probe".to_string(),
        ..WorkerMeta::test_meta("watch-cut", "reconnect-test")
    };
    meta.pid = owned.sleeper.id().expect("the sleeper has a pid");
    pool.__test_save_status(&meta, "test", RegistryStatus::Running, 1, 10, "probe", None);

    let mut daemon = Command::from(daemon_command(&hub_dir, &swe))
        .kill_on_drop(true)
        .spawn()
        .expect("start the daemon the watch follows");
    owned
        .pids
        .insert(daemon.id().expect("the daemon has a pid") as i32);
    wait_for_socket(&hub_dir).await;

    let mut watch = Command::from(watch_command(&hub_dir, &swe))
        .kill_on_drop(true)
        .spawn()
        .expect("start the watch");
    let mut output = BufReader::new(watch.stdout.take().expect("watch stdout"));
    let mut errors = BufReader::new(watch.stderr.take().expect("watch stderr"));
    wait_for_log(&hub_dir, "Serving MCP connection", 1).await;

    owned.cut();
    assert!(
        !daemon
            .wait()
            .await
            .expect("the daemon was killed")
            .success(),
        "the daemon exits on the kill, not on its own"
    );

    // The watch follows the daemon it lost: the replacement hub starts, answers
    // it again, and the CLI process is still the one watching.
    wait_for_socket(&hub_dir).await;
    wait_for_log(&hub_dir, "Serving MCP connection", 2).await;
    assert!(
        matches!(watch.try_wait(), Ok(None)),
        "the watch must survive the cut, not exit with it; stderr: {:?}",
        read_to_end(&mut errors).await
    );
    // The replacement's recovery handed the row back, so the watch has a
    // worker to follow on the new connection rather than nothing to watch.
    wait_for_log(&hub_dir, "recovered", 2).await;

    // And the delivery continues: the worker's terminal status is the next
    // event, and it reaches the same CLI process over the new connection.
    std::fs::create_dir_all(format!("{}/swe-wt-watch-cut", swe.display()))
        .expect("the worker\'s worktree");
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
        Some(printed) => assert!(
            printed.contains("watch-cut"),
            "the event belongs to the watched worker: {printed}"
        ),
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
///
/// The CLI prints its own failures to stdout (the JSON document is the
/// interface), so `main`'s `Debug` rendering of them never reaches stderr.
async fn read_to_end<R: tokio::io::AsyncRead + Unpin>(stream: &mut R) -> Vec<u8> {
    let mut seen = Vec::new();
    let _ = tokio::io::AsyncReadExt::read_to_end(stream, &mut seen).await;
    seen
}

/// Read from `stream` until `needle` appears, so the test polls for the event
/// instead of sleeping for it.
async fn read_until<R>(stream: &mut R, needle: &str, within: Duration) -> Option<String>
where
    R: tokio::io::AsyncBufRead + Unpin,
{
    let mut seen = String::new();
    let deadline = tokio::time::Instant::now() + within;
    loop {
        let left = deadline.saturating_duration_since(tokio::time::Instant::now());
        if left.is_zero() {
            return if seen.contains(needle) {
                Some(seen)
            } else {
                None
            };
        }
        let mut line = String::new();
        match tokio::time::timeout(left, stream.read_line(&mut line)).await {
            Ok(Ok(0)) | Err(_) => {
                return if seen.contains(needle) {
                    Some(seen)
                } else {
                    None
                };
            }
            Ok(Ok(_)) => seen.push_str(&line),
            Ok(Err(_)) => return None,
        }
    }
}
