//! Unit tests for the executable watcher's state machine and probe.

use super::{ExeWatch, Fingerprint, Verdict, probe_build};
use std::os::unix::fs::PermissionsExt;
use std::time::{Duration, Instant};

fn fp(ino: u64, size: u64) -> Fingerprint {
    Fingerprint {
        ino,
        size,
        mtime: 1,
        mtime_nsec: 0,
    }
}

/// A change is only a handover candidate once it has been stable for the
/// whole window: a fingerprint that keeps moving (a `cargo` write in
/// progress) never reaches `Probe`.
#[test]
fn a_change_must_be_stable_before_it_is_probed() {
    let t0 = Instant::now();
    let stable = Duration::from_secs(3);
    let mut watch = ExeWatch::new(Some(fp(1, 100)));

    // Same file: nothing to do.
    assert_eq!(watch.observe(Some(fp(1, 100)), t0, stable), Verdict::Wait);

    // A new file appears: wait, it may still be being written.
    assert_eq!(watch.observe(Some(fp(2, 50)), t0, stable), Verdict::Wait);
    // Still the same partial write, inside the window: wait.
    assert_eq!(
        watch.observe(Some(fp(2, 50)), t0 + Duration::from_secs(1), stable),
        Verdict::Wait
    );
    // The write moved on: the window restarts.
    assert_eq!(
        watch.observe(Some(fp(2, 90)), t0 + Duration::from_secs(2), stable),
        Verdict::Wait
    );
    // The file is gone: a cargo replace unlinks before it writes.
    assert_eq!(watch.observe(None, t0 + Duration::from_secs(2), stable), Verdict::Wait);
    // The new file has been there, unchanged, for the whole window.
    assert_eq!(
        watch.observe(Some(fp(3, 120)), t0 + Duration::from_secs(3), stable),
        Verdict::Wait
    );
    assert_eq!(
        watch.observe(Some(fp(3, 120)), t0 + Duration::from_secs(6), stable),
        Verdict::Probe
    );
}

/// Once the caller adopts the probed fingerprint, the watcher does not
/// re-probe the same file.
#[test]
fn adopting_a_fingerprint_stops_reprobing_it() {
    let t0 = Instant::now();
    let stable = Duration::from_secs(1);
    let mut watch = ExeWatch::new(Some(fp(1, 100)));
    watch.observe(Some(fp(2, 120)), t0, stable);
    assert_eq!(
        watch.observe(Some(fp(2, 120)), t0 + Duration::from_secs(2), stable),
        Verdict::Probe
    );
    watch.adopt(Some(fp(2, 120)));
    assert_eq!(
        watch.observe(Some(fp(2, 120)), t0 + Duration::from_secs(4), stable),
        Verdict::Wait
    );
}

/// A file that comes back to the running build's fingerprint is not a
/// change at all.
#[test]
fn a_return_to_the_baseline_is_not_a_change() {
    let t0 = Instant::now();
    let stable = Duration::from_secs(1);
    let mut watch = ExeWatch::new(Some(fp(1, 100)));
    watch.observe(Some(fp(2, 120)), t0, stable);
    assert_eq!(
        watch.observe(Some(fp(1, 100)), t0 + Duration::from_secs(1), stable),
        Verdict::Wait
    );
    assert_eq!(
        watch.observe(Some(fp(1, 100)), t0 + Duration::from_secs(3), stable),
        Verdict::Wait
    );
}

/// The probe reads the build identity a runnable binary prints, and
/// refuses anything that is not one: a non-executable file, a script
/// that does not print the identity, and a failing command.
#[tokio::test]
async fn the_probe_only_accepts_a_runnable_build_identity() {
    let scratch = crate::test_support::TestScratch::new("auto-handover-probe");

    // A binary that prints the identity: accepted.
    let good = scratch.path().join("good");
    std::fs::write(
        &good,
        "#!/bin/sh\necho '{\"id\":\"abc123\",\"ts\":42}'\n",
    )
    .unwrap();
    std::fs::set_permissions(&good, std::fs::Permissions::from_mode(0o755)).unwrap();
    let build = probe_build(&good).await.expect("a runnable build prints its id");
    assert_eq!(build["id"], "abc123");
    assert_eq!(build["ts"], 42);

    // The same file without its executable bit: refused.
    let not_exec = scratch.path().join("not-exec");
    std::fs::write(
        &not_exec,
        "#!/bin/sh\necho '{\"id\":\"abc123\",\"ts\":42}'\n",
    )
    .unwrap();
    std::fs::set_permissions(&not_exec, std::fs::Permissions::from_mode(0o644)).unwrap();
    assert!(probe_build(&not_exec).await.is_none());

    // A command that fails: refused, so a build that fails to start is
    // never handed over to.
    let fails = scratch.path().join("fails");
    std::fs::write(&fails, "#!/bin/sh\nexit 3\n").unwrap();
    std::fs::set_permissions(&fails, std::fs::Permissions::from_mode(0o755)).unwrap();
    assert!(probe_build(&fails).await.is_none());

    // A command that does not print the identity: refused.
    let garbage = scratch.path().join("garbage");
    std::fs::write(&garbage, "#!/bin/sh\necho not-json\n").unwrap();
    std::fs::set_permissions(&garbage, std::fs::Permissions::from_mode(0o755)).unwrap();
    assert!(probe_build(&garbage).await.is_none());
}
