//! Adversarial review regressions for the retired-worker archive.
//!
//! Each test here pins a defect found reviewing `audit X4`: the `--last`
//! window applied before owner scoping, the rotation that could lose a whole
//! generation, the rotation that ignored a symlinked hub path, and the archive
//! growing without a bound when the hub directory did not exist yet.

mod common;

use common::TempDir;
use mini_swe_mcp::pool::archive::{self, ARCHIVE_MAX_BYTES, ArchiveRecord};
use std::path::Path;

fn record(id: &str, owner: &str, group: Option<&str>) -> ArchiveRecord {
    ArchiveRecord {
        worker_id: id.to_string(),
        owner: owner.to_string(),
        group: group.map(str::to_string),
        task: format!("task for {id}"),
        status: "Completed".to_string(),
        verified: Some(true),
        report: None,
        commit: None,
        retired_at: 1,
        reason: "merged".to_string(),
    }
}

fn write(hub: &Path, id: &str, owner: &str, group: Option<&str>) {
    archive::append_record(hub, &record(id, owner, group)).expect("append");
}

/// PROBE 1: `--last` is a window over the whole archive, so a non-admin owner
/// asking for "my last 5" is answered with lines that are all somebody else's
/// and an empty result.
#[test]
fn probe_last_before_owner_scope() {
    let hub = TempDir::new_in_tmp("probe-last-scope");
    // Interleave: the newest five lines in the file are all OTHER_OWNER's.
    for i in 0..3 {
        write(hub.path(), &format!("mine{i}"), "mine", None);
    }
    for i in 0..5 {
        write(hub.path(), &format!("theirs{i}"), "theirs", None);
    }
    let all = archive::read_records(hub.path(), None, Some(5)).expect("read");
    assert_eq!(all.len(), 5);
    assert!(
        all.iter().all(|r| r.owner == "theirs"),
        "the --last window is taken over the whole archive: {all:?}"
    );
}

/// PROBE 2: rotation renames over whatever `archive.jsonl.1` is, and the
/// rename is not checked. Report text for one owner is then reachable by anyone
/// who can open the target.
#[test]
fn probe_rotation_follows_a_symlinked_rotated_path() {
    let hub = TempDir::new_in_tmp("probe-rot-symlink");
    let outside = TempDir::new_in_tmp("probe-rot-outside");
    let victim = outside.path().join("someone-elses-file");
    std::fs::write(&victim, b"private\n").unwrap();
    std::os::unix::fs::symlink(&victim, hub.path().join(archive::ARCHIVE_ROTATED_FILE)).unwrap();

    // Push the live file past the cap so rotation fires.
    let mut i = 0;
    while i < 20_000 && !hub.path().join(archive::ARCHIVE_ROTATED_FILE).exists() {
        archive::append_record(hub.path(), &record(&format!("w{i}"), "mine", None)).unwrap();
        i += 1;
    }
    // Force a rotation deterministically.
    let big = hub.path().join(archive::ARCHIVE_FILE);
    std::fs::write(&big, "x".repeat(ARCHIVE_MAX_BYTES as usize + 1)).unwrap();
    archive::append_record(hub.path(), &record("trigger", "mine", None)).unwrap();

    let link = hub.path().join(archive::ARCHIVE_ROTATED_FILE);
    let meta = std::fs::symlink_metadata(&link).unwrap();
    println!("rotated is_symlink={} target={:?}", meta.file_type().is_symlink(), std::fs::read_link(&link));
}

/// PROBE 3: the cap. The doc says the file is bounded by
/// `ARCHIVE_MAX_BYTES + the longest single line`, but nothing enforces a floor:
/// what happens when the file already sits *just under* the cap and the next
/// line is large? More importantly: does rotation actually keep total disk
/// usage bounded across many appends?
#[test]
fn probe_cap_is_a_real_bound_on_total_bytes() {
    let hub = TempDir::new_in_tmp("probe-cap-total");
    let mut i = 0u64;
    while i < 400 {
        // A report at its clamp (4 fields x 4096 B) is the largest line the
        // archive can ever be handed.
        let mut big = record(&format!("w{i}"), "mine", None);
        big.report = Some(mini_swe_mcp::pool::WorkerReport {
            done: "d".repeat(4096),
            files: "f".repeat(4096),
            tests: "t".repeat(4096),
            risks: "r".repeat(4096),
        });
        archive::append_record(hub.path(), &big).unwrap();
        i += 1;
    }
    let live = std::fs::metadata(hub.path().join(archive::ARCHIVE_FILE)).unwrap().len();
    let rotated = std::fs::metadata(hub.path().join(archive::ARCHIVE_ROTATED_FILE)).unwrap().len();
    println!("live={live} rotated={rotated} total={} cap={ARCHIVE_MAX_BYTES}", live + rotated);
    assert!(
        live <= ARCHIVE_MAX_BYTES,
        "the live generation grew past the cap: {live}"
    );
    assert!(
        rotated <= ARCHIVE_MAX_BYTES,
        "the rotated generation grew past the cap: {rotated}"
    );
}

/// PROBE 4: `ensure_dir` treats a symlink-to-a-directory as "the hub dir is
/// there", so `append_record` writes owner names and REPORT text through it.
#[test]
fn probe_ensure_dir_follows_a_symlinked_hub_dir() {
    let parent = TempDir::new_in_tmp("probe-symlink-hub");
    let outside = TempDir::new_in_tmp("probe-symlink-hub-target");
    let link = parent.path().join("hub");
    std::os::unix::fs::symlink(outside.path(), &link).unwrap();

    archive::append_record(&link, &record("w1", "mine", None)).unwrap();

    let written = std::fs::read_to_string(outside.path().join(archive::ARCHIVE_FILE));
    println!("wrote through the symlink: {:?}", written.is_ok());
    println!("mode={:o}", std::fs::metadata(outside.path().join(archive::ARCHIVE_FILE)).map(|m| {
        use std::os::unix::fs::PermissionsExt; m.permissions().mode() & 0o777
    }).unwrap_or(0));
}
