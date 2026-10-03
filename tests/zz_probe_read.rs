mod common;
use common::TempDir;
use mini_swe_mcp::pool::archive::{self, ArchiveRecord};

#[test]
fn read_through_symlink_probe() {
    let hub = TempDir::new_in_tmp("probe-read-hub");
    let elsewhere = TempDir::new_in_tmp("probe-read-target");
    let victim = elsewhere.path().join("planted.jsonl");
    let rec = ArchiveRecord {
        worker_id: "attacker".into(), owner: "admin".into(), group: None,
        task: "planted line".into(), status: "Completed".into(), verified: Some(true),
        report: None, commit: None, retired_at: 1, reason: "merged".into(),
    };
    std::fs::write(&victim, format!("{}\n", serde_json::to_string(&rec).unwrap())).unwrap();
    std::os::unix::fs::symlink(&victim, hub.path().join(archive::ARCHIVE_FILE)).unwrap();
    let read = archive::read_records(hub.path(), None, None, None).unwrap();
    println!("READ RECORDS = {read:?}");
    assert!(read.is_empty(), "read followed the planted symlink: {read:?}");
}
