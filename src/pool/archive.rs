//! The hub's compact archive of retired workers' final REPORTs.
//!
//! Retirement deletes everything a worker leaves behind -- its branch, its
//! registry row, its saved conversation -- so the REPORT block it ended with
//! (`done` / `files` / `tests` / `risks`, often the only before/after
//! measurement the run ever produced) would be gone the instant a merge lands.
//! Every retirement therefore appends *one* JSON line to
//! `<hub dir>/archive.jsonl` before it deletes anything, so the report outlives
//! the worker that wrote it.
//!
//! The line is deliberately narrow: who ran, what they were asked, how the run
//! ended, whether the gate passed and the four REPORT fields. No diffs, no tool
//! output, no artifacts and nothing that is not already in the row -- so the
//! archive cannot become a second copy of the scratch tree, and reading it
//! costs what one `status` answer costs.
//!
//! Size is bounded by rotation rather than by a trim: past
//! [`ARCHIVE_MAX_BYTES`] the file becomes `archive.jsonl.1` and a fresh one
//! starts, so a long-lived hub keeps at most two generations and can never grow
//! without bound. Both files are owner-only (`0600`), like the hub's own state.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};

use super::WorkerRegistryEntry;

/// The archive itself: one JSON line per retired worker.
pub const ARCHIVE_FILE: &str = "archive.jsonl";

/// The previous generation, written when [`ARCHIVE_FILE`] crosses the cap.
pub const ARCHIVE_ROTATED_FILE: &str = "archive.jsonl.1";

/// Size past which [`ARCHIVE_FILE`] rotates. Two generations bound the archive
/// to twice this: enough to answer "what did the last round of workers report",
/// small enough that an unremarkable hub leaves nothing worth pruning.
pub const ARCHIVE_MAX_BYTES: u64 = 256 * 1024;

/// Longest `task` first line kept in a line. The task is the only free-form
/// field here; the REPORT fields are already clamped where they are parsed
/// (`crate::pool::runner::REPORT_FIELD_BYTES`).
pub const ARCHIVE_TASK_BYTES: usize = 512;

/// Why a worker was retired, which is the one thing the deleted row alone would
/// not have said: the same worker can leave through five different doors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetireReason {
    /// Its own branch was merged into the base branch.
    Merged,
    /// It was a member of the round a consolidator integrated and landed.
    Integrated,
    /// The consolidator that took its round over absorbed its leftover work.
    Absorbed,
    /// It was dropped on purpose, with no merge at all.
    Discarded,
    /// Its terminal retention ran out and nobody continued it.
    Expired,
}

impl RetireReason {
    /// The word the archive line and the `archive` view both spell, so the file
    /// and the CLI can never disagree about it.
    pub fn word(self) -> &'static str {
        match self {
            RetireReason::Merged => "merged",
            RetireReason::Integrated => "integrated",
            RetireReason::Absorbed => "absorbed",
            RetireReason::Discarded => "discarded",
            RetireReason::Expired => "expired",
        }
    }
}

/// What one retirement leaves behind: the run's identity and verdict, plus the
/// REPORT fields the worker ended with.
///
/// Every field is already bounded (the task to [`ARCHIVE_TASK_BYTES`], the four
/// REPORT fields where they were parsed), so one line cannot carry a diff.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArchiveRecord {
    /// The worker this line retires.
    pub worker_id: String,
    /// Agent that dispatched it, or the label for a row that names none.
    pub owner: String,
    /// Round it belonged to, when it was dispatched as part of one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub group: Option<String>,
    /// First line of the task: enough to recognise the run, never the whole
    /// prompt.
    pub task: String,
    /// Status its row carried when it was retired.
    pub status: String,
    /// Whether its completion passed the verify gate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verified: Option<bool>,
    /// The REPORT block it finished with, if it finished with one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub report: Option<super::WorkerReport>,
    /// The merge commit that landed it, where the retirement knows one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub commit: Option<String>,
    /// Unix time the line was written.
    pub retired_at: u64,
    /// Which door the worker left through.
    pub reason: String,
}

impl ArchiveRecord {
    /// The line `entry` earns as it is retired for `reason`.
    ///
    /// `commit` is the merge commit when the retirement knows one (a merge, or
    /// the round a landed consolidator carried); a sweep or a retention expiry
    /// never knows it and says so instead of guessing.
    pub fn from_entry(
        entry: &WorkerRegistryEntry,
        reason: RetireReason,
        commit: Option<&str>,
        retired_at: u64,
    ) -> Self {
        Self {
            worker_id: entry.id.clone(),
            owner: super::registry_owner_label(entry).to_string(),
            group: entry.group.clone(),
            task: super::clamp_string(&super::round::first_line(&entry.task), ARCHIVE_TASK_BYTES),
            status: entry.status.display_name().to_string(),
            verified: entry.verified,
            report: entry.report.clone(),
            commit: commit.map(str::to_string),
            retired_at,
            reason: reason.word().to_string(),
        }
    }
}

/// `<dir>/archive.jsonl`.
pub fn archive_path(dir: &Path) -> PathBuf {
    dir.join(ARCHIVE_FILE)
}

/// `<dir>/archive.jsonl.1`, the rotated generation.
pub fn rotated_path(dir: &Path) -> PathBuf {
    dir.join(ARCHIVE_ROTATED_FILE)
}

/// Append one record to `<dir>/archive.jsonl`, rotating it first if the line
/// would carry the file past [`ARCHIVE_MAX_BYTES`].
///
/// `O_APPEND` keeps concurrent retirements from overwriting each other's line:
/// the append is atomic in the kernel, and every writer rotates against the
/// same cap. A line that does not fit at all is still written -- capping the
/// file is not a reason to lose the only copy of a report -- so the file is
/// bounded by [`ARCHIVE_MAX_BYTES`] plus the longest single line.
pub fn append_record(dir: &Path, record: &ArchiveRecord) -> Result<()> {
    let mut line = serde_json::to_string(record).context("archive record did not serialize")?;
    line.push('\n');
    let path = archive_path(dir);
    // The directory first: rotation renames a file out of it, so a directory
    // that does not exist yet would make the rename fail and lose the report on
    // the very call meant to keep it.
    ensure_dir(dir)?;
    // Rotation before the open: the rename and the append cannot interleave
    // into the same file, so the rotated generation is always a whole file.
    if let Ok(size) = std::fs::metadata(&path).map(|meta| meta.len())
        && size.saturating_add(line.len() as u64) > ARCHIVE_MAX_BYTES
    {
        let rotated = rotated_path(dir);
        // Replace the previous generation outright rather than appending to it:
        // the cap is a bound, not a queue. The previous generation is unlinked
        // before the rename, so the rename cannot clobber a file this process
        // did not write.
        let _ = std::fs::remove_file(&rotated);
        std::fs::rename(&path, &rotated)
            .with_context(|| format!("could not rotate {}", path.display()))?;
    }
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW)
        .open(&path)
        .with_context(|| format!("could not open {}", path.display()))?;
    // `mode` only applies to a file this call creates, so an `archive.jsonl`
    // that already exists keeps whatever mode it had -- a world-readable one
    // written before this hardening, or planted in a `SWE_HUB_DIR` somebody
    // else made. The archive names owners and carries REPORT text, so tighten
    // an existing file to owner-only before the first byte goes in.
    if let Ok(meta) = file.metadata()
        && meta.permissions().mode() & 0o077 != 0
    {
        let _ = file.set_permissions(std::fs::Permissions::from_mode(0o600));
    }
    file.write_all(line.as_bytes())
        .with_context(|| format!("could not append to {}", path.display()))
}

/// The archive of `dir`, oldest line first, across both generations.
///
/// `owner` keeps one agent's records and is applied *before* `last`, so
/// `--last 5` is "the five most recent of this agent's" and never a window over
/// the whole archive that another agent's lines happen to fill. `group` keeps
/// one round's records; `last` keeps the newest `n` of whatever survived both
/// filters. A line that does not parse is skipped rather than failing the whole
/// read: a retirement interrupted mid-append leaves a torn tail, and one torn
/// line must not cost the caller every report before it.
pub fn read_records(
    dir: &Path,
    owner: Option<&str>,
    group: Option<&str>,
    last: Option<usize>,
) -> Result<Vec<ArchiveRecord>> {
    let mut records: Vec<ArchiveRecord> = Vec::new();
    // Oldest generation first, so the ordering stays chronological.
    for path in [rotated_path(dir), archive_path(dir)] {
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        for line in text.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let Ok(record) = serde_json::from_str::<ArchiveRecord>(line) else {
                continue;
            };
            if let Some(owner) = owner
                && record.owner != owner
            {
                continue;
            }
            if let Some(group) = group
                && record.group.as_deref() != Some(group)
            {
                continue;
            }
            records.push(record);
        }
    }
    if let Some(last) = last
        && records.len() > last
    {
        records.drain(..records.len() - last);
    }
    Ok(records)
}

/// Create the hub directory the archive lives in, owner-only.
///
/// The hub daemon creates and hardens it long before anything retires, so this
/// is only the path a caller that archives without a daemon takes; it is
/// created `0700` because the archive names owners.
///
/// The existing-directory check uses [`symlink_metadata`] and not
/// [`Path::is_dir`], which follows a link: a `hub` entry that is a symlink to
/// some other directory reads as "the hub dir is there", and the append below
/// would write one agent's owners and REPORT text through it into a directory
/// the hub never claimed. Refusing the link keeps the archive in the directory
/// the caller named, or nowhere.
fn ensure_dir(dir: &Path) -> Result<()> {
    match std::fs::symlink_metadata(dir) {
        Ok(meta) if meta.file_type().is_symlink() => {
            anyhow::bail!(
                "refusing to archive through the symlink {}",
                dir.display()
            );
        }
        Ok(meta) if meta.is_dir() => return Ok(()),
        Ok(_) => {
            anyhow::bail!("archive path {} is not a directory", dir.display());
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => {
            return Err(e).with_context(|| format!("could not inspect {}", dir.display()));
        }
    }
    std::fs::DirBuilder::new()
        .mode(0o700)
        .recursive(true)
        .create(dir)
        .with_context(|| format!("could not create {}", dir.display()))
}
