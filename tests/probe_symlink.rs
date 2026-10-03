//! A planted `archive.jsonl` symlink must never be rotated, renamed or read.
//!
//! The cap check that decides whether to rotate has to look at the *link*, not
//! at what it points at: a hub directory somebody else can write to may hold an
//! `archive.jsonl` symlink aimed at any file on the host. Measuring the target
//! and then renaming the link into `archive.jsonl.1` would leave the archive
//! reading through a symlink of the plant's choosing, and would delete or
//! expose a file the pool never wrote.
//!
//! This owns its hub directory, its link and its "victim", all under the
//! temporary base dir, so nothing here touches a real hub.

mod common;

use common::TempDir;
use mini_swe_mcp::pool::archive::{self, ARCHIVE_FILE, ARCHIVE_MAX_BYTES, ArchiveRecord};
use std::path::{Path, PathBuf};

/// A record whose `task` is `pad` bytes, so the caller controls the line length.
fn record(id: &str, pad: usize) -> ArchiveRecord {
    let mut value = serde_json::to_value(ArchiveRecord {
        worker_id: id.to_string(),
        owner: "agent-one".to_string(),
        group: None,
        task: "t".to_string(),
        status: "Completed".to_string(),
        verified: None,
        report: None,
        commit: None,
        retired_at: 1,
        reason: "merged".to_string(),
    })
    .expect("a record serializes");
    value["task"] = serde_json::Value::String("x".repeat(pad));
    serde_json::from_value(value).expect("the padded record parses")
}

fn is_symlink(path: &Path) -> bool {
    std::fs::symlink_metadata(path)
        .map(|meta| meta.file_type().is_symlink())
        .unwrap_or(false)
}

/// The append is refused and the planted link is left exactly as it was.
#[test]
fn a_symlinked_archive_is_neither_rotated_nor_written_through() {
    let hub = TempDir::new_in_tmp("probe-archive-hub");
    let elsewhere = TempDir::new_in_tmp("probe-archive-target");
    let victim: PathBuf = elsewhere.path().join("victim.txt");
    let contents = "important\n".repeat(ARCHIVE_MAX_BYTES as usize);
    std::fs::write(&victim, &contents).expect("the victim file must be writable");

    let archive_path = hub.path().join(ARCHIVE_FILE);
    std::os::unix::fs::symlink(&victim, &archive_path).expect("the test plants the link");

    // The victim is over the cap, so a `metadata`-based check would rotate now.
    let result = archive::append_record(hub.path(), &record("w1", 0));

    assert!(
        result.is_err(),
        "writing through a planted symlink must be refused, got {result:?}"
    );
    assert!(
        !hub.path().join(archive::ARCHIVE_ROTATED_FILE).exists(),
        "the planted link must never be renamed into the rotated generation"
    );
    assert!(
        is_symlink(&archive_path),
        "the planted link must be left where it was, not replaced"
    );
    assert_eq!(
        std::fs::read_to_string(&victim).expect("the victim must still be readable"),
        contents,
        "the file behind the link must be untouched"
    );
    // Nothing the archive reads may resolve through the link either.
    let read = archive::read_records(hub.path(), None, None, None).expect("the read must not fail");
    assert!(
        read.is_empty(),
        "a line must never be harvested from behind a planted symlink: {read:?}"
    );
}
