//! A watch outlives the daemon underneath it, however the daemon goes away.
//!
//! A handover is not an orderly event: the daemon closes its listeners, and
//! whatever the connection's unread buffer still holds is discarded with them,
//! so the client can see an EOF, a reset or a broken pipe for the same stop.
//! Every one of those must answer with the same chase — dial again, say hello
//! with the same identity, replay what was missed — and the chase must end
//! with an explanation when no daemon comes back.

mod common;

use mini_swe_mcp::pool::{RegistryStatus, WorkerMeta, WorkerPool};
use std::path::Path;
use std::process::Stdio;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};

/// A `SO_LINGER` of zero makes `close` send a reset instead of the usual
/// EOF-plus-pending-data, which is what a daemon torn down mid-read looks like
/// to its peers.
const RESET_ON_CLOSE: libc::c_int = 1;

/// Seconds the unit tests give the reconnect budget. Long enough for the
/// replacement daemon to start, its recovery to run and the watch to reconnect
/// several times over.
const RECONNECT_SECS: &str = "90";

/// A daemon on `hub_dir`, with its log appended to `hub_dir/hub.log`.
fn daemon(hub_dir: &Path, swe: &Path) -> std::process::Command {
    let mut daemon = common::binary_command(&common::binary_path());
    daemon
        .arg("daemon")
        .env("SWE_HUB_DIR", hub_dir)
        .env("SWE_TEMP_DIR", swe)
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
    common::scrub_identity_env(&mut daemon);
    daemon
}

/// Wait until the hub log holds at least `count` lines mentioning `event`.
async fn wait_for_log(hub_dir: &Path, event: &str, count: usize) {
    let deadline = std::time::Instant::now() + Duration::from_secs(30);
    loop {
        let seen = std::fs::read_to_string(hub_dir.join("hub.log"))
            .map(|log| {
                log.lines()
                    .filter(|line| {
                        if event == "listening" {
                            line.ends_with(" listening")
                        } else {
                            line.contains(event)
                        }
                    })
                    .count()
            })
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

/// Wait until every daemon the log names has exited, so no test leaves one behind.
async fn wait_for_exit(pids: &[i32]) {
    let deadline = std::time::Instant::now() + Duration::from_secs(30);
    while std::time::Instant::now() < deadline
        && pids
            .iter()
            .any(|pid| matches!(signal(*pid, 0), Some(())))
    {
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    for pid in pids {
        signal(*pid, libc::SIGKILL);
    }
}

/// A CLI `watch` following a daemon that stops under it, with a cut that
/// arrives as a reset instead of an EOF: the watch reconnects to the
/// replacement and still delivers the next event.
#[tokio::test]
async fn a_watch_follows_a_daemon_cut_with_a_reset() {
    let hub = common::TempDir::new_in_tmp("watch-cut");
    let hub_dir = hub.path().to_path_buf();
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&hub_dir, std::fs::Permissions::from_mode(0o700))
            .expect("make the hub dir private");
    }
    let isolated = common::IsolatedPool::new(2, "watch-cut");
    let swe = isolated.root().path().to_path_buf();

    let mut first: Child = Command::from(daemon(&hub_dir, &swe))
        .spawn()
        .expect("start the first daemon");
    wait_for_log(&hub_dir, "listening", 1).await;

    // A live worker owned by the watching agent, standing in for one the
    // orchestrator dispatched. Its pid is a real process, so neither the
    // graceful path nor recovery treats it as dead.
    let mut sleeper = Command::new("sleep")
        .arg("60")
        .kill_on_drop(true)
        .spawn()
        .expect("stand-in worker process");
    let mut meta = WorkerMeta {
        task: "reconnect probe".to_string(),
        ..WorkerMeta::test_meta("watch-cut", "reconnect-test")
    };
    meta.pid = sleeper.id().expect("the sleeper has a pid");
    write_status(&isolated.pool, &meta, RegistryStatus::Running, 1, "probe");

    let mut watch = Command::from(watch_command(&hub_dir, &swe))
        .kill_on_drop(true)
        .spawn()
        .expect("start the watch");
    let mut output = BufReader::new(watch.stdout.take().expect("watch stdout"));
    let mut errors = BufReader::new(watch.stderr.take().expect("watch stderr"));
    wait_for_log(&hub_dir, "Serving MCP connection", 1).await;

    // The daemon is gone without warning: every connection it is serving,
    // including the watch's, is cut.
    assert!(
        signal(daemon_pids(&hub_dir)[0], libc::SIGKILL).is_some(),
        "kill the daemon the watch is talking to"
    );
    first.wait().await.expect("the daemon exited");

    // The watch follows the daemon it lost: the replacement hub starts and
    // answers it again, and the CLI process is still the one watching.
    wait_for_log(&hub_dir, "listening", 2).await;
    wait_for_log(&hub_dir, "recovered", 2).await;
    assert!(
        matches!(watch.try_wait(), Ok(None)),
        "the watch must survive the reset, not exit with it"
    );

    // And the delivery continues: the worker's terminal status is the next
    // event, and it reaches the same CLI process over the new connection.
    std::fs::create_dir_all(isolated.root().join("swe-wt-watch-cut")).expect("the worktree");
    isolated
        .pool
        .__test_reset_registry_throttle("watch-cut");
    write_status(
        &isolated.pool,
        &meta,
        RegistryStatus::Failed,
        2,
        "probe ended",
    );

    let within = Duration::from_secs(30);
    let printed = match read_until(&mut output, "watch-cut", within).await {
        Some(printed) => printed,
        None => panic!(
            "the reconnected watch must deliver the next event; saw {:?}",
            read_to_end(&mut output).await
        ),
    };
    assert!(
        printed.contains("watch-cut"),
        "the event belongs to the watched worker: {printed}"
    );
    let finished = tokio::time::timeout(Duration::from_secs(30), watch.wait())
        .await
        .expect("the watch ends once its worker is terminal")
        .expect("the watch exits");
    assert!(
        finished.success(),
        "the watch reported its last event: {:?}",
        String::from_utf8_lossy(&read_to_end(&mut errors).await).into_owned()
    );

    let mut pids = daemon_pids(&hub_dir);
    pids.dedup();
    wait_for_exit(&pids).await;
    let _ = sleeper.kill().await;
    let _ = sleeper.wait().await;
}

/// The `watch` CLI this test drives: a child whose identity is pinned to
/// `reconnect-test`, so the recovery has rows to hand back to the watch.
fn watch_command(hub_dir: &Path, swe: &Path) -> std::process::Command {
    let mut watch = common::binary_command(&common::binary_path());
    watch
        .args(["watch", "--json"])
        .env_remove("MINI_SWE_NO_DAEMON")
        .env("SWE_HUB_DIR", hub_dir)
        .env("SWE_TEMP_DIR", swe)
        .env("MINI_SWE_AGENT_ID", "reconnect-test")
        .env(RECONNECT_ENV, RECONNECT_SECS)
        .env("ENV_FILE", "/nonexistent-mini-swe-reconnect")
        .env("OPENAI_API_KEY", "test-key-not-used-by-the-reconnect-test")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    common::scrub_identity_env(&mut watch);
    watch
}

/// The env var the CLI and the hub client both read for the reconnect budget.
const RECONNECT_ENV: &str = mini_swe_mcp::hub::RECONNECT_DEADLINE_ENV;

/// A registry row the daemon and the watch both read, written through the pool
/// so the throttled registry file is written the same way a worker's is.
fn write_status(
    pool: &WorkerPool,
    meta: &WorkerMeta,
    status: RegistryStatus,
    step: usize,
    command: &str,
) {
    pool.__test_save_status(meta, "test", status, step, 10, command, None);
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
