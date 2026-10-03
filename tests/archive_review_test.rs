//! Regression tests for defects found reviewing the retired-worker archive.
//!
//! Four properties the archive must hold that a reviewer's probes turned up:
//! `--last` is a window over the *caller's own* lines rather than over every
//! owner's; `merge --no-delete`, which retires nobody, archives nobody; the
//! archive never writes through a symlinked hub directory and never leaves a
//! world-readable `archive.jsonl` behind it; and both generations stay inside
//! the size cap.
//!
//! Every test owns its repository, scratch root and hub directory under the
//! temporary base dir, so nothing here touches a developer's own hub.

mod common;

use common::TempDir;
use mini_swe_mcp::pool::archive::{self, ARCHIVE_MAX_BYTES, ArchiveRecord};
use mini_swe_mcp::pool::{
    MergeRequest, RegistryStatus, WorkerRegistryEntry, WorkerReport, load_registry_entry_in,
    merge_worker_in, save_registry_entry_in,
};
use mini_swe_mcp::worktree::ScratchRoot;
use std::path::{Path, PathBuf};

const OWNER: &str = "agent-one";
const OTHER_OWNER: &str = "agent-two";

/// A repo, a scratch root and a stand-in hub directory, all owned by the test.
struct Fixture {
    repo: TempDir,
    scratch: TempDir,
    hub: TempDir,
}

impl Fixture {
    fn new(tag: &str) -> Self {
        let repo = TempDir::new_in_tmp(tag);
        common::git(repo.path(), &["init", "--initial-branch=main"]);
        common::git(repo.path(), &["config", "user.email", "review@test"]);
        common::git(repo.path(), &["config", "user.name", "review test"]);
        std::fs::write(repo.path().join("README.md"), "base\n").expect("writable");
        common::git(repo.path(), &["add", "."]);
        common::git(repo.path(), &["commit", "-m", "base"]);
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

    /// A terminal worker with a REPORT, the gate verdict and a group.
    fn record(&self, id: &str, owner: &str, group: Option<&str>) {
        let entry = WorkerRegistryEntry {
            task: format!("fix the {id} regression\nand its follow-up"),
            status: RegistryStatus::Completed,
            step: 4,
            repo_path: Some(self.repo.path().to_string_lossy().into_owned()),
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
        common::git(self.repo.path(), &["checkout", "-q", "-b", &branch]);
        std::fs::write(self.repo.path().join(format!("{id}.txt")), "from the worker\n")
            .expect("writable");
        common::git(self.repo.path(), &["add", "."]);
        common::git(self.repo.path(), &["commit", "-m", &format!("worker {id}")]);
        common::git(self.repo.path(), &["checkout", "-q", "main"]);
    }
}

/// A line with no report, as the reader parses it back.
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
    archive::append_record(hub, &record(id, owner, group)).expect("the append must succeed");
}

/// `--last` is a window over the caller's own lines. Scoping it after the
/// window means a busy hub's newest five lines -- all another agent's -- leave
/// the caller with an empty answer to "my last five".
#[test]
fn last_counts_the_callers_own_lines() {
    let hub = TempDir::new_in_tmp("archive-last-owner");
    for i in 0..3 {
        write(hub.path(), &format!("mine{i}"), OWNER, None);
    }
    for i in 0..5 {
        write(hub.path(), &format!("theirs{i}"), OTHER_OWNER, None);
    }

    let mine = archive::read_records(hub.path(), Some(OWNER), None, Some(2)).expect("readable");
    assert_eq!(
        mine.iter()
            .map(|r| r.worker_id.as_str())
            .collect::<Vec<_>>(),
        ["mine1", "mine2"],
        "--last 2 is this agent's two most recent, not the file's two newest: {mine:?}"
    );
    assert!(
        mine.iter().all(|r| r.owner == OWNER),
        "another owner's report must never reach a scoped read: {mine:?}"
    );

    // The admin read still sees everything, and its own `--last` is the newest
    // of the whole file.
    let all = archive::read_records(hub.path(), None, None, Some(2)).expect("readable");
    assert_eq!(
        all.iter()
            .map(|r| r.worker_id.as_str())
            .collect::<Vec<_>>(),
        ["theirs3", "theirs4"],
        "{all:?}"
    );

    // Owner and group compose, and `--last` applies after both.
    for i in 0..2 {
        write(hub.path(), &format!("r{i}"), OWNER, Some("round-7"));
    }
    for i in 2..5 {
        write(hub.path(), &format!("r{i}"), OTHER_OWNER, Some("round-7"));
    }
    let scoped =
        archive::read_records(hub.path(), Some(OWNER), Some("round-7"), Some(1)).expect("readable");
    assert_eq!(
        scoped
            .iter()
            .map(|r| r.worker_id.as_str())
            .collect::<Vec<_>>(),
        ["r1"],
        "--last applies after the owner and group filters: {scoped:?}"
    );
}

/// `merge --no-delete` keeps the branch, the row and the history, so the worker
/// is still there to be merged again. Archiving it would put a "retired" line
/// in the archive for a worker that has not retired.
#[test]
fn no_delete_archives_nobody() {
    let f = Fixture::new("archive-no-delete");
    f.commit_on_worker_branch("w1");
    f.record("w1", OWNER, Some("round-1"));

    merge_worker_in(
        &f.root(),
        &MergeRequest {
            worker_id: "w1",
            verified: Some(true),
            keep_branch: true,
            admission: None,
            archive_dir: Some(f.hub_dir()),
        },
    )
    .expect("the merge lands");

    assert!(
        load_registry_entry_in(&f.root(), "w1").is_some(),
        "--no-delete must leave the row in place"
    );
    let archived = archive::read_records(f.hub.path(), None, None, None).expect("readable");
    assert!(
        archived.is_empty(),
        "--no-delete retires nobody, so it archives nobody: {archived:?}"
    );

    // The same worker's real retirement does archive it, so the fix is not
    // "the merge stopped archiving": a merge that really deletes the branch
    // writes the line the `--no-delete` merge did not.
    common::git(f.repo.path(), &["checkout", "-q", "worker-w1"]);
    std::fs::write(f.repo.path().join("w1.txt"), "more from the worker\n").expect("writable");
    common::git(f.repo.path(), &["add", "."]);
    common::git(f.repo.path(), &["commit", "-m", "worker w1 again"]);
    common::git(f.repo.path(), &["checkout", "-q", "main"]);
    let mut row = load_registry_entry_in(&f.root(), "w1").expect("the row survived");
    row.keep_branch = false;
    save_registry_entry_in(&f.root(), &row);
    // The gate needs something to run: an empty manifest makes
    // `detect_verify_command` fall through, and "unknown" never skips it.
    // The gate needs something to run: a repository with no recognised manifest
    // has no detectable verify command, and "unknown" never skips the gate.
    std::fs::write(f.repo.path().join("Makefile"), "test:\n\t@true\n").expect("writable");
    merge_worker_in(
        &f.root(),
        &MergeRequest {
            worker_id: "w1",
            verified: Some(true),
            keep_branch: false,
            admission: None,
            archive_dir: Some(f.hub_dir()),
        },
    )
    .expect("the deleting merge lands");
    let archived = archive::read_records(f.hub.path(), None, None, None).expect("readable");
    assert_eq!(
        archived.len(),
        1,
        "the retirement that really removes the worker archives it: {archived:?}"
    );
    assert_eq!(archived[0].worker_id, "w1");
    assert_eq!(archived[0].reason, "merged");
}

/// The archive names owners and carries REPORT text, so it goes in the
/// directory the caller named and nowhere else: a hub directory that is a
/// symlink is refused rather than written through.
#[test]
fn a_symlinked_hub_dir_is_refused() {
    let parent = TempDir::new_in_tmp("archive-symlink-parent");
    let elsewhere = TempDir::new_in_tmp("archive-symlink-target");
    let link = parent.path().join("hub");
    std::os::unix::fs::symlink(elsewhere.path(), &link).expect("the test creates the link");

    let outcome = archive::append_record(&link, &record("w1", OWNER, None));

    assert!(
        outcome.is_err(),
        "a symlinked hub dir must be refused, not written through"
    );
    assert!(
        !elsewhere.path().join(archive::ARCHIVE_FILE).exists(),
        "no archive may appear in the directory the symlink pointed at"
    );
}

/// `mode(0o600)` only applies to a file this call creates, so an
/// `archive.jsonl` that already existed world-readable would keep that mode
/// while the append added one more agent's owner and REPORT to it.
#[test]
fn an_existing_loose_archive_is_tightened() {
    use std::os::unix::fs::PermissionsExt;

    let hub = TempDir::new_in_tmp("archive-loose-mode");
    let path = hub.path().join(archive::ARCHIVE_FILE);
    std::fs::write(&path, b"").expect("writable");
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).expect("chmod");

    archive::append_record(hub.path(), &record("w1", OWNER, None)).expect("the append must succeed");

    let mode = std::fs::metadata(&path).expect("the archive exists").permissions().mode() & 0o777;
    assert_eq!(mode, 0o600, "an existing archive is tightened to owner-only: {mode:o}");
}

/// The cap is a bound on what is kept, so it holds for both generations and for
/// the largest line the archive can ever be handed (four clamped REPORT
/// fields).
#[test]
fn both_generations_stay_inside_the_cap() {
    let hub = TempDir::new_in_tmp("archive-cap-both");
    let big = |id: &str| ArchiveRecord {
        report: Some(WorkerReport {
            done: "d".repeat(4096),
            files: "f".repeat(4096),
            tests: "t".repeat(4096),
            risks: "r".repeat(4096),
        }),
        ..record(id, OWNER, None)
    };

    for i in 0..200 {
        archive::append_record(hub.path(), &big(&format!("w{i}"))).expect("the append must succeed");
    }

    let live = std::fs::metadata(hub.path().join(archive::ARCHIVE_FILE))
        .expect("the live generation exists")
        .len();
    let rotated = std::fs::metadata(hub.path().join(archive::ARCHIVE_ROTATED_FILE))
        .expect("200 maximal lines must have crossed the cap and rotated")
        .len();
    assert!(live <= ARCHIVE_MAX_BYTES, "the live generation grew past the cap: {live}");
    assert!(
        rotated <= ARCHIVE_MAX_BYTES,
        "the rotated generation grew past the cap: {rotated}"
    );

    // Two generations are the whole promise: at most two maximal lines' worth
    // of reports is kept, and both generations are readable.
    let all = archive::read_records(hub.path(), None, None, None).expect("readable");
    assert!(
        !all.is_empty() && all.len() <= 32,
        "the archive keeps two bounded generations, not every line: {}",
        all.len()
    );
}

/// The reader spans both generations oldest first, so `--last` answers "the N
/// most recent" across the rotation boundary rather than only within the live
/// file.
#[test]
fn last_spans_the_rotation_boundary() {
    let hub = TempDir::new_in_tmp("archive-last-across-rotation");
    // Maximal lines: a generation holds ~15 of them, so the cap is crossed
    // inside the first two passes.
    for i in 0..20 {
        archive::append_record(
            hub.path(),
            &ArchiveRecord {
                report: Some(WorkerReport {
                    done: "d".repeat(4096),
                    files: "f".repeat(4096),
                    tests: "t".repeat(4096),
                    risks: "r".repeat(4096),
                }),
                ..record(&format!("w{i:04}"), OWNER, None)
            },
        )
        .expect("the append must succeed");
    }
    assert!(
        hub.path().join(archive::ARCHIVE_ROTATED_FILE).exists(),
        "20 maximal lines must have crossed the cap"
    );
    let newest = archive::read_records(hub.path(), None, None, Some(3)).expect("readable");
    assert_eq!(
        newest
            .iter()
            .map(|r| r.worker_id.as_str())
            .collect::<Vec<_>>(),
        ["w0017", "w0018", "w0019"],
        "{newest:?}"
    );
}
