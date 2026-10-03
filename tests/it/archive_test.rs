//! Integration tests for the retired-worker archive (`audit X4`).
//!
//! Every retirement deletes the row, the conversation and the branch -- and
//! with them the REPORT block that run ended with. These tests pin the three
//! properties the archive exists for: exactly one line per retirement, carrying
//! the report; owner scoping on the read; and the size cap that keeps a
//! long-lived hub bounded. They build their own repository, scratch root and
//! hub directory, so nothing here touches the developer's.

use crate::common;
use crate::common::{TempDir, git};
use mini_swe_mcp::pool::archive::{self, ARCHIVE_MAX_BYTES, ArchiveRecord, RetireReason};
use mini_swe_mcp::pool::{
    MergeRequest, RegistryStatus, RetireContext, WorkerRegistryEntry, WorkerReport,
    load_registry_entry_in, merge_worker_in, retire_expired_terminal_workers_in,
    retire_worker_reporting, save_registry_entry_in,
};
use mini_swe_mcp::worktree::ScratchRoot;
use std::path::{Path, PathBuf};

const OWNER: &str = "agent-one";
const OTHER_OWNER: &str = "agent-two";

struct Fixture {
    repo: TempDir,
    scratch: TempDir,
    /// Stands in for `<hub dir>`: a directory the code under test writes into,
    /// created by this test and removed with it.
    hub: TempDir,
}

impl Fixture {
    fn new(tag: &str) -> Self {
        let repo = TempDir::new_in_tmp(tag);
        git(repo.path(), &["init", "--initial-branch=main"]);
        git(repo.path(), &["config", "user.email", "archive@test"]);
        git(repo.path(), &["config", "user.name", "archive test"]);
        write(repo.path(), "README.md", "base\n");
        git(repo.path(), &["add", "."]);
        git(repo.path(), &["commit", "-m", "base"]);
        Self {
            repo,
            scratch: TempDir::new_in_tmp(&format!("{tag}-scratch")),
            hub: TempDir::new_in_tmp(&format!("{tag}-hub")),
        }
    }

    fn root(&self) -> ScratchRoot {
        ScratchRoot::new(self.scratch.path())
    }

    fn hub_dir(&self) -> PathBuf {
        self.hub.path().to_path_buf()
    }

    fn repo(&self) -> &Path {
        self.repo.path()
    }

    /// A terminal worker with a REPORT, the gate verdict and a group.
    fn record(&self, id: &str, owner: &str, group: Option<&str>) {
        let entry = WorkerRegistryEntry {
            task: format!("fix the {id} regression\nand its follow-up"),
            status: RegistryStatus::Completed,
            step: 4,
            repo_path: Some(self.repo().to_string_lossy().into_owned()),
            owner: Some(owner.to_string()),
            group: group.map(str::to_string),
            base_branch: Some("main".to_string()),
            verified: Some(true),
            report: Some(WorkerReport {
                done: "fixed the parser".to_string(),
                files: "src/a.rs".to_string(),
                tests: "cargo test: 12 before, 12 after".to_string(),
                risks: "none".to_string(),
            }),
            ..WorkerRegistryEntry::test_row(id, owner)
        };
        save_registry_entry_in(&self.root(), &entry);
    }

    /// Give `id` a branch off `main`, so a merge has something to land.
    fn commit_on_worker_branch(&self, id: &str) {
        let branch = format!("worker-{id}");
        git(self.repo(), &["checkout", "-q", "-b", &branch]);
        write(self.repo(), &format!("{id}.txt"), "from the worker\n");
        git(self.repo(), &["add", "."]);
        git(self.repo(), &["commit", "-m", &format!("worker {id}")]);
        git(self.repo(), &["checkout", "-q", "main"]);
    }

    fn merge(&self, id: &str) -> anyhow::Result<mini_swe_mcp::pool::MergeReport> {
        merge_worker_in(
            &self.root(),
            &MergeRequest {
                worker_id: id,
                verified: Some(true),
                keep_branch: false,
                admission: None,
                archive_dir: Some(self.hub_dir()),
            },
        )
    }

    /// The archive's own lines, oldest first, as the reader parses them.
    fn records(&self) -> Vec<ArchiveRecord> {
        archive::read_records(self.hub.path(), None, None, None)
            .expect("the archive must be readable")
    }

    fn archive_path(&self) -> PathBuf {
        self.hub.path().join(archive::ARCHIVE_FILE)
    }
}

fn write(dir: &Path, name: &str, contents: &str) {
    std::fs::write(dir.join(name), contents).expect("fixture file must be writable");
}

/// A merged worker's report outlives the merge: exactly one line, carrying the
/// REPORT fields, the owner, the group, the gate verdict and the landing commit.
#[test]
fn a_merge_writes_exactly_one_line_with_the_report_fields() {
    let f = Fixture::new("archive-merge");
    f.commit_on_worker_branch("w1");
    f.record("w1", OWNER, Some("round-1"));

    let report = f.merge("w1").expect("the merge lands");

    // The row is gone; that is what the archive exists for.
    assert!(
        load_registry_entry_in(&f.root(), "w1").is_none(),
        "the merge must have retired the row the archive line was built from"
    );
    let records = f.records();
    assert_eq!(
        records.len(),
        1,
        "one retirement writes one line: {records:?}"
    );
    let line = &records[0];
    assert_eq!(line.worker_id, "w1");
    assert_eq!(line.owner, OWNER);
    assert_eq!(line.group.as_deref(), Some("round-1"));
    // The task is stored as its first line only.
    assert_eq!(line.task, "fix the w1 regression");
    assert_eq!(line.status, "Completed");
    assert_eq!(line.verified, Some(true));
    assert_eq!(line.reason, "merged");
    assert_eq!(line.commit.as_deref(), Some(report.commit.as_str()));
    assert!(line.retired_at > 0, "the line must say when it retired");
    let archived = line.report.as_ref().expect("the REPORT must be archived");
    assert_eq!(archived.done, "fixed the parser");
    assert_eq!(archived.files, "src/a.rs");
    assert_eq!(archived.tests, "cargo test: 12 before, 12 after");
    assert_eq!(archived.risks, "none");
}

/// The file is owner-only and carries no diff: the report and the identity, and
/// nothing the scratch tree already held.
#[test]
fn the_archive_line_is_private_and_holds_no_diff() {
    use std::os::unix::fs::PermissionsExt;

    let f = Fixture::new("archive-mode");
    f.commit_on_worker_branch("w1");
    f.record("w1", OWNER, None);
    f.merge("w1").expect("the merge lands");

    let mode = std::fs::metadata(f.archive_path())
        .expect("the archive exists")
        .permissions()
        .mode()
        & 0o777;
    assert_eq!(mode, 0o600, "the archive names owners: {mode:o}");
    let text = std::fs::read_to_string(f.archive_path()).expect("the archive is readable");
    assert!(
        !text.contains("from the worker"),
        "the archive keeps reports, not the worker's files: {text}"
    );
    assert!(!text.contains("diff"), "no diff is kept: {text}");
}

/// A discard is a retirement too, and it says so rather than claiming a merge.
#[test]
fn a_discard_archives_the_report_as_discarded() {
    let f = Fixture::new("archive-discard");
    f.record("w1", OWNER, Some("round-2"));

    retire_worker_reporting(
        &f.root(),
        "w1",
        &RetireContext {
            ack_dir: Some(&f.hub_dir()),
            reason: Some(RetireReason::Discarded),
            ..RetireContext::default()
        },
    );

    let records = f.records();
    assert_eq!(records.len(), 1, "{records:?}");
    assert_eq!(records[0].reason, "discarded");
    assert!(
        records[0].commit.is_none(),
        "a discard has no merge commit: {:?}",
        records[0].commit
    );
}

/// An expired retention is the fourth door in, and it archives too.
#[test]
fn an_expired_retention_archives_the_report_as_expired() {
    let f = Fixture::new("archive-expired");
    f.record("w1", OWNER, None);

    // The row must look as old as the retention allows: `updated_at == 0` is
    // how the library itself spells "never stamped", and the retention test
    // never accepts it.
    let mut entry = load_registry_entry_in(&f.root(), "w1").expect("the row is on disk");
    entry.updated_at = 1;
    save_registry_entry_in(&f.root(), &entry);

    let retired = retire_expired_terminal_workers_in(&f.root(), 0, Some(&f.hub_dir()));

    assert_eq!(retired, 1, "the terminal row must have been retired");
    let records = f.records();
    assert_eq!(records.len(), 1, "{records:?}");
    assert_eq!(records[0].reason, "expired");
    assert_eq!(records[0].worker_id, "w1");
}

/// The read is owner-scoped: one agent cannot read another agent's retired
/// worker's report.
#[test]
fn the_read_is_owner_scoped() {
    let f = Fixture::new("archive-owner");
    for (id, owner) in [("w1", OWNER), ("w2", OTHER_OWNER)] {
        f.record(id, owner, None);
        retire_worker_reporting(
            &f.root(),
            id,
            &RetireContext {
                ack_dir: Some(&f.hub_dir()),
                reason: Some(RetireReason::Merged),
                ..RetireContext::default()
            },
        );
    }

    // Both lines are on disk; scoping is what decides who reads them.
    assert_eq!(f.records().len(), 2, "both retirements are archived");

    let owned = archive::read_records(f.hub.path(), Some(OWNER), None, None).expect("readable");
    assert_eq!(owned.len(), 1, "{owned:?}");
    assert_eq!(owned[0].worker_id, "w1");

    // The admin override is the only way to see every owner's.
    let all = archive::read_records(f.hub.path(), None, None, None).expect("readable");
    assert_eq!(all.len(), 2, "{all:?}");
}

/// `--group` keeps one round, `--last` keeps the newest N of what survives.
#[test]
fn group_and_last_narrow_the_read() {
    let f = Fixture::new("archive-filter");
    for (id, group) in [("w1", "round-a"), ("w2", "round-a"), ("w3", "round-b")] {
        f.record(id, OWNER, Some(group));
        retire_worker_reporting(
            &f.root(),
            id,
            &RetireContext {
                ack_dir: Some(&f.hub_dir()),
                reason: Some(RetireReason::Merged),
                ..RetireContext::default()
            },
        );
    }

    let round_a =
        archive::read_records(f.hub.path(), None, Some("round-a"), None).expect("readable");
    assert_eq!(
        round_a
            .iter()
            .map(|r| r.worker_id.as_str())
            .collect::<Vec<_>>(),
        ["w1", "w2"],
    );
    let newest = archive::read_records(f.hub.path(), None, None, Some(1)).expect("readable");
    assert_eq!(
        newest
            .iter()
            .map(|r| r.worker_id.as_str())
            .collect::<Vec<_>>(),
        ["w3"],
        "--last keeps the newest of the whole archive"
    );
    let capped =
        archive::read_records(f.hub.path(), None, Some("round-a"), Some(1)).expect("readable");
    assert_eq!(
        capped
            .iter()
            .map(|r| r.worker_id.as_str())
            .collect::<Vec<_>>(),
        ["w2"],
        "--last applies after --group"
    );
}

/// The cap holds: past it the file rotates, so a hub keeps two bounded
/// generations instead of growing forever.
#[test]
fn the_archive_rotates_past_its_size_cap() {
    let f = Fixture::new("archive-cap");
    let record = |id: &str| ArchiveRecord {
        worker_id: id.to_string(),
        owner: OWNER.to_string(),
        group: None,
        task: format!("task for {id}"),
        status: "Completed".to_string(),
        verified: Some(true),
        report: None,
        commit: None,
        retired_at: 1,
        reason: "merged".to_string(),
    };

    // Append until the live file crosses the cap. The bound is a backstop, not
    // the trigger: how many lines fit depends on their length, so the loop is
    // what actually reaches the cap.
    let mut writes = 0usize;
    while writes < 10_000 && !f.hub.path().join(archive::ARCHIVE_ROTATED_FILE).exists() {
        let id = format!("w{writes:06}");
        archive::append_record(f.hub.path(), &record(&id)).expect("the append must succeed");
        writes += 1;
    }
    let size = std::fs::metadata(f.archive_path())
        .expect("the archive exists")
        .len();
    assert!(
        size <= ARCHIVE_MAX_BYTES,
        "the live file must stay capped, grew to {size}"
    );
    let rotated = f.hub.path().join(archive::ARCHIVE_ROTATED_FILE);
    assert!(
        rotated.exists(),
        "the previous generation must be kept as archive.jsonl.1"
    );

    // The reader spans both generations, and both hold real lines: rotation
    // moves reports into `archive.jsonl.1`, it does not drop them.
    let all = f.records();
    let rotated_lines = std::fs::read_to_string(&rotated).expect("the rotated file is readable");
    assert!(
        !rotated_lines.is_empty() && !all.is_empty(),
        "both generations must be readable"
    );
    assert!(
        all.len() > writes / 2,
        "rotation must not lose most of the reports: {} of {writes}",
        all.len()
    );
    assert!(
        all.len() <= writes + 1,
        "rotation must not duplicate them either: {}",
        all.len()
    );
}

/// An owner who never retired anything is told the archive is empty rather than
/// handed another agent's work.
#[test]
fn an_owner_with_nothing_retired_reads_an_empty_archive() {
    let f = Fixture::new("archive-empty");
    assert!(
        archive::read_records(f.hub.path(), Some(OWNER), None, None)
            .expect("readable")
            .is_empty()
    );
    assert!(
        archive::read_records(f.hub.path(), None, None, None)
            .expect("readable")
            .is_empty()
    );
}
