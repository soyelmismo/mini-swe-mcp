//! Rotation must never cost a report.
//!
//! The archive's whole promise is that the *only* copy of a retired worker's
//! REPORT survives the retirement that deletes the row, the conversation and the
//! branch. Capping the file is a bound, not a reason to lose that copy, so a
//! rotation that cannot be performed must not take the append down with it.
//!
//! The rotated generation's name is a fixed, predictable `<hub dir>/archive.jsonl.1`
//! that nothing in the archive ever opens for writing, so it is the one name in
//! the hub directory that can be occupied by something a rename cannot replace
//! (a directory, for one). That is the shape these tests exercise: the live
//! generation is already past the cap, the rotated name cannot be replaced, and
//! the report must still land.
//!
//! Every directory here is test-owned under the temporary base dir.

mod common;

use common::TempDir;
use mini_swe_mcp::pool::archive::{self, ARCHIVE_FILE, ARCHIVE_MAX_BYTES, ArchiveRecord};
use std::path::Path;

fn record(id: &str) -> ArchiveRecord {
    ArchiveRecord {
        worker_id: id.to_string(),
        owner: "agent-one".to_string(),
        group: None,
        task: format!("task for {id}"),
        status: "Completed".to_string(),
        verified: Some(true),
        report: None,
        commit: None,
        retired_at: 1,
        reason: "merged".to_string(),
    }
}

/// Drive `dir`'s live generation past the cap so the next append must rotate.
fn fill_past_the_cap(hub: &Path) {
    let live = hub.join(ARCHIVE_FILE);
    let mut filler = String::new();
    while filler.len() as u64 <= ARCHIVE_MAX_BYTES {
        filler.push_str("0123456789abcdef0123456789abcdef\n");
    }
    std::fs::write(&live, filler).expect("the live generation is writable");
}

/// The rotated name occupied by a directory cannot be renamed over, so the
/// rotation fails. The append must still write: the report is the only copy.
#[test]
fn a_rotation_that_cannot_complete_still_writes_the_report() {
    let hub = TempDir::new_in_tmp("archive-rotate-blocked-hub");
    fill_past_the_cap(hub.path());
    // A non-empty directory: neither the unlink the code attempts nor the
    // rename can replace it, and `rmdir` on it is never the archive's to do.
    let blocker = hub.path().join(archive::ARCHIVE_ROTATED_FILE);
    std::fs::create_dir(&blocker).expect("the test plants the blocker");
    std::fs::write(blocker.join("occupied"), "not ours\n").expect("the blocker has content");

    archive::append_record(hub.path(), &record("w1"))
        .expect("a report must survive a rotation that cannot be performed");

    assert!(
        blocker.is_dir() && blocker.join("occupied").exists(),
        "the occupying directory must be left exactly as it was"
    );
    let live = std::fs::read_to_string(hub.path().join(ARCHIVE_FILE)).expect("the live file");
    assert!(
        live.contains("\"w1\""),
        "the report must be in the archive: {live}"
    );
    let records = archive::read_records(hub.path(), None, None, None).expect("the read must not fail");
    assert_eq!(
        records.iter().map(|r| r.worker_id.as_str()).collect::<Vec<_>>(),
        ["w1"],
        "the only report written must be readable back: {records:?}"
    );
}

/// The same shape, seen through a retirement rather than through the append:
/// the archive write is best effort for the *merge*, but it must not be best
/// effort by dropping the line.
#[test]
fn a_blocked_rotation_keeps_every_report_of_a_long_run() {
    let hub = TempDir::new_in_tmp("archive-rotate-blocked-run");
    let blocker = hub.path().join(archive::ARCHIVE_ROTATED_FILE);
    std::fs::create_dir(&blocker).expect("the test plants the blocker");
    fill_past_the_cap(hub.path());

    let mut written = Vec::new();
    for i in 0..50 {
        let id = format!("w{i}");
        archive::append_record(hub.path(), &record(&id))
            .unwrap_or_else(|e| panic!("append {id} must succeed: {e}"));
        written.push(id);
    }

    let records = archive::read_records(hub.path(), None, None, None).expect("the read must not fail");
    let read: Vec<&str> = records.iter().map(|r| r.worker_id.as_str()).collect();
    for id in &written {
        assert!(read.contains(&id.as_str()), "{id} must survive: {read:?}");
    }
    assert!(
        blocker.is_dir(),
        "the occupying directory must still be there: {}",
        blocker.display()
    );
}
