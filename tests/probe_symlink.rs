mod common;
use common::TempDir;
use mini_swe_mcp::pool::archive::{self, ArchiveRecord, ARCHIVE_MAX_BYTES};
use std::path::PathBuf;

fn rec(id: &str, pad: usize) -> ArchiveRecord {
    ArchiveRecord {
        worker_id: id.into(), owner: "o".into(), group: None, task: "t".into(),
        status: "Completed".into(), verified: None, report: None, commit: None,
        retired_at: 1, reason: "merged".into(),
    }
    .tap_pad(pad)
}

trait Pad { fn tap_pad(self, n: usize) -> Self; }
impl Pad for ArchiveRecord {
    fn tap_pad(self, n: usize) -> Self {
        let mut m = serde_json::to_value(&self).unwrap();
        m["task"] = serde_json::Value::String("x".repeat(n));
        serde_json::from_value(m).unwrap()
    }
}

#[test]
fn probe_rotation_with_symlinked_archive() {
    let hub = TempDir::new_in_tmp("probe-rot-hub");
    let target = TempDir::new_in_tmp("probe-rot-target");
    let tpath: PathBuf = target.path().join("victim.txt");
    // A victim file larger than the cap, so metadata() sees size > cap.
    std::fs::write(&tpath, "important\n".repeat(ARCHIVE_MAX_BYTES as usize)).unwrap();
    std::os::unix::fs::symlink(&tpath, hub.path().join(archive::ARCHIVE_FILE)).unwrap();
    let r = archive::append_record(hub.path(), &rec("w1", 0));
    println!("append result: {r:?}");
    println!("victim still there: {}", tpath.exists());
    let rotated = hub.path().join(archive::ARCHIVE_ROTATED_FILE);
    println!("rotated exists: {}", rotated.exists());
    if rotated.exists() {
        let md = std::fs::symlink_metadata(&rotated).unwrap();
        println!("rotated is symlink: {}", md.file_type().is_symlink());
        if let Ok(link) = std::fs::read_link(&rotated) { println!("rotated -> {link:?}"); }
        println!("victim still there after: {}", tpath.exists());
    }
}
