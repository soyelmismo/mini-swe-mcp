//! The archive must never read through a planted link, and never hang on one.
//!
//! The write path refuses a symlinked `archive.jsonl` with `O_NOFOLLOW` and
//! never rotates one, so a link planted in a hub directory cannot redirect a
//! write. The read path has to hold the same line: a `read_to_string` that
//! follows links would undo that defence on the only path a caller reaches the
//! file through, harvesting whatever the link's target holds -- another agent's
//! reports, or an unrelated file -- as retired workers of this pool.
//!
//! The FIFO case is the sharper one. A FIFO planted as `archive.jsonl` makes
//! `open(O_RDONLY)` block until a writer arrives, so the type check has to be
//! reachable *without* blocking in `open(2)` itself: `O_NONBLOCK` at the open,
//! then the handle's own type.
//!
//! Every hub directory, link and target here is created under the temporary
//! base dir, so nothing touches a real hub.

mod common;

use common::TempDir;
use mini_swe_mcp::pool::archive::{self, ArchiveRecord};

/// A line that belongs to `owner`, so a scoped read that returns it is proof it
/// came from the plant rather than from the archive.
fn planted_record(owner: &str) -> ArchiveRecord {
    ArchiveRecord {
        worker_id: format!("w-{owner}"),
        owner: owner.to_string(),
        group: None,
        task: "a line the hub never wrote".to_string(),
        status: "Completed".to_string(),
        verified: Some(true),
        report: None,
        commit: None,
        retired_at: 1,
        reason: "merged".to_string(),
    }
}

/// Write `planted_record` into a fresh directory and return its path, so the
/// link has something with real, parseable content to aim at.
fn plant_target(tag: &str) -> (TempDir, std::path::PathBuf) {
    let dir = TempDir::new_in_tmp(tag);
    let victim = dir.path().join("planted.jsonl");
    std::fs::write(
        &victim,
        format!("{}\n", serde_json::to_string(&planted_record("victim-owner")).unwrap()),
    )
    .expect("the plant target must be writable");
    (dir, victim)
}

/// `archive.jsonl` planted as a symlink is never harvested, by an admin read or
/// by an owner-scoped one that names the owner inside the target.
#[test]
fn a_symlinked_live_archive_is_never_read_through() {
    let hub = TempDir::new_in_tmp("archive-read-live-link");
    let (_target, victim) = plant_target("archive-read-live-target");
    std::os::unix::fs::symlink(&victim, hub.path().join(archive::ARCHIVE_FILE))
        .expect("the test plants the link");

    let all = archive::read_records(hub.path(), None, None, None).expect("the read must not fail");
    assert!(
        all.is_empty(),
        "no line may come from behind a planted link: {all:?}"
    );

    // Scoped to the owner inside the target: the strongest form of the read, and
    // the one that would return the plant if the link were followed.
    let scoped =
        archive::read_records(hub.path(), Some("victim-owner"), None, None).expect("readable");
    assert!(
        scoped.is_empty(),
        "a scoped read must not answer with a planted line: {scoped:?}"
    );
}

/// The rotated generation is read first and must be held to the same rule: a
/// link planted as `archive.jsonl.1` is exactly the file `read_records` opens
/// before it ever looks at the live one.
#[test]
fn a_symlinked_rotated_archive_is_never_read_through() {
    let hub = TempDir::new_in_tmp("archive-read-rotated-link");
    let (_target, victim) = plant_target("archive-read-rotated-target");
    std::os::unix::fs::symlink(&victim, hub.path().join(archive::ARCHIVE_ROTATED_FILE))
        .expect("the test plants the link");

    let scoped =
        archive::read_records(hub.path(), Some("victim-owner"), None, None).expect("readable");
    assert!(
        scoped.is_empty(),
        "a scoped read must not answer with a planted rotated line: {scoped:?}"
    );
}

/// A FIFO planted as the archive must be refused, not waited on: an `archive`
/// call that blocks forever on a planted FIFO is a denial of service on the
/// only way to read the archive. The test therefore *is* the timeout -- if the
/// open blocked, this test would hang and the suite would fail.
#[test]
fn a_fifo_archive_is_refused_instead_of_blocking() {
    let hub = TempDir::new_in_tmp("archive-read-fifo");
    let fifo = hub.path().join(archive::ARCHIVE_FILE);
    let raw = std::ffi::CString::new(fifo.to_str().expect("a utf-8 temp path")).expect("no NUL");
    // SAFETY: `raw` is a valid NUL-terminated path to a path this test owns,
    // and `mkfifo` only creates the node.
    let rc = unsafe { libc::mkfifo(raw.as_ptr(), 0o600) };
    assert_eq!(
        rc,
        0,
        "could not plant the FIFO: {}",
        std::io::Error::last_os_error()
    );

    let records = archive::read_records(hub.path(), None, None, None).expect("the read must not fail");
    assert!(
        records.is_empty(),
        "a FIFO is not the archive and must yield no lines: {records:?}"
    );
}

/// The hardening must not cost a real archive its lines: a genuine regular file
/// is still read, in order, across both generations.
#[test]
fn a_real_archive_is_still_read_in_order() {
    let hub = TempDir::new_in_tmp("archive-read-regular");
    for id in ["w1", "w2", "w3"] {
        archive::append_record(hub.path(), &planted_record(id)).expect("the append must succeed");
    }
    let all = archive::read_records(hub.path(), None, None, None).expect("readable");
    assert_eq!(
        all.iter().map(|r| r.worker_id.as_str()).collect::<Vec<_>>(),
        ["w-w1", "w-w2", "w-w3"],
        "a regular file must still be read, oldest first: {all:?}"
    );
}
