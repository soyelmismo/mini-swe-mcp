//! Cross-process steering mailbox: the disk-backed counterpart of
//! [`WorkerPool::steer`](super::WorkerPool::steer)'s in-memory `pending_steer`.
//!
//! `pending_steer` only lives in the address space of the process that owns the
//! worker, so a `mini-swe-mcp steer <id> "…"` typed in *another* terminal used
//! to fail with `Worker not found`. This module gives the orchestrator a second
//! delivery path that survives that boundary: a per-worker mailbox file beside
//! the worktree, drained by the worker loop on every turn.
//!
//! ## File format and path
//!
//! The mailbox is `<base>/swe-wt-<worker_id>.steer`, where `<base>` is
//! [`swe_base_dir`](crate::worktree::swe_base_dir) — i.e. `/var/tmp` by
//! default, or `$SWE_TEMP_DIR` when set. It sits next to the worktree
//! directory so one `ls` shows the full set of live workers.
//!
//! Note that `prune_stale_worktrees` does *not* reclaim these files: its sweep
//! matches `swe-wt-*` directories and `swe-wt-*.pid` leases, and a
//! `swe-wt-<id>.steer` file matches neither. Cleanup is therefore the worker
//! loop's own responsibility ([`remove_steer_file`], invoked by a `Drop` guard
//! held for the worker's whole lifetime), not the pruner's — a mailbox only
//! exists while some `steer` call created it, and every worker removes its own
//! on exit.
//!
//! Messages are **JSON lines**, one object per line:
//! `{"message": "…", "sent_at": 1700000000, "pid": 4242}`.
//! JSON lines (rather than raw text) make appends self-delimiting, so a message
//! containing newlines — a multi-line diff hunk, a pasted stack trace — can
//! never be split into two bogus messages or glued onto its neighbour.
//!
//! ## Why appends and drains are both safe
//!
//! * **Append** is a single `O_APPEND` `write(2)` of a payload that already ends
//!   in `\n`. POSIX guarantees such a write is atomic with respect to other
//!   appends, so N concurrent orchestrators never interleave or tear a line.
//! * **Drain** renames the mailbox to a claim path and *then* reads it. `rename`
//!   is atomic, so exactly one drainer ever claims a given generation of the
//!   file; a writer that appends after the rename recreates the original path
//!   and its bytes are picked up by the next drain. Without the claim step two
//!   drains (e.g. the implementer loop and the review loop) would both read the
//!   same file and deliver every message twice.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};

use tracing::{debug, warn};

use crate::worktree::ScratchRoot;

/// One orchestrator message as it is stored in a mailbox.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct SteerRecord {
    message: String,
    #[serde(default)]
    sent_at: u64,
    #[serde(default)]
    pid: u32,
}

/// Path of the steering mailbox for `worker_id`.
///
/// Public so the CLI can report exactly where a cross-process steer was
/// deposited.
pub fn steer_path(worker_id: &str) -> PathBuf {
    steer_path_in(&ScratchRoot::from_env(), worker_id)
}

/// [`steer_path`] under an explicit scratch root.
pub fn steer_path_in(root: &ScratchRoot, worker_id: &str) -> PathBuf {
    root.join(format!("swe-wt-{worker_id}.steer"))
}

fn claim_path_in(root: &ScratchRoot, worker_id: &str) -> PathBuf {
    let path = steer_path_in(root, worker_id);
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(format!(".{}.claim", std::process::id()));
    path.with_file_name(name)
}

/// Append `message` to `worker_id`'s mailbox, atomically with respect to other
/// appenders.
///
/// Returns the path written, or an error the caller can surface: a steer that
/// cannot be queued must be reported, never silently dropped.
pub fn write_steer_message(worker_id: &str, message: &str) -> std::io::Result<PathBuf> {
    write_steer_message_in(&ScratchRoot::from_env(), worker_id, message)
}

/// [`write_steer_message`] under an explicit scratch root.
pub fn write_steer_message_in(
    root: &ScratchRoot,
    worker_id: &str,
    message: &str,
) -> std::io::Result<PathBuf> {
    let path = steer_path_in(root, worker_id);
    let record = SteerRecord {
        message: message.to_string(),
        sent_at: super::unix_timestamp(),
        pid: std::process::id(),
    };
    // `to_string` on a `Value` cannot fail.
    let mut payload = serde_json::to_string(&record).unwrap_or_default();
    payload.push('\n');

    let mut file = OpenOptions::new().create(true).append(true).open(&path)?;
    // Single write of a `\n`-terminated payload: atomic under O_APPEND.
    file.write_all(payload.as_bytes())?;
    file.flush()?;
    debug!(worker = %worker_id, path = %path.display(), "Queued cross-process steering message");
    Ok(path)
}

/// Read and remove every message currently queued for `worker_id`.
///
/// Returns them in arrival order. A missing mailbox is the common case (nobody
/// has steered this worker) and yields an empty vector, not an error. Messages
/// that fail to parse are skipped with a warning rather than aborting the drain,
/// so one corrupt line cannot strand the worker and the rest of the guidance
/// still arrives.
pub fn drain_steer_messages(worker_id: &str) -> Vec<String> {
    drain_steer_messages_in(&ScratchRoot::from_env(), worker_id)
}

/// [`drain_steer_messages`] under an explicit scratch root.
pub fn drain_steer_messages_in(root: &ScratchRoot, worker_id: &str) -> Vec<String> {
    let path = steer_path_in(root, worker_id);
    if !path.exists() {
        return Vec::new();
    }
    let claim = claim_path_in(root, worker_id);
    // Claim the current generation atomically. A losing racer observes
    // `NotFound` here and simply finds nothing to drain.
    if let Err(e) = std::fs::rename(&path, &claim) {
        if e.kind() == std::io::ErrorKind::NotFound {
            return Vec::new();
        }
        warn!(worker = %worker_id, error = %e, "Failed to claim steering mailbox");
        return Vec::new();
    }

    let drained = read_records(&claim);
    // The claim file is ours alone; removing it is the drain's commit.
    if let Err(e) = std::fs::remove_file(&claim)
        && e.kind() != std::io::ErrorKind::NotFound
    {
        warn!(worker = %worker_id, error = %e, "Failed to remove claimed steering mailbox");
    }
    drained
}

/// Parse a claimed mailbox into messages, tolerating malformed lines.
fn read_records(claim: &Path) -> Vec<String> {
    let Ok(content) = std::fs::read_to_string(claim) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for line in content.lines() {
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<SteerRecord>(line) {
            Ok(record) => out.push(record.message),
            Err(e) => warn!(error = %e, "Skipping unparsable steering mailbox line"),
        }
    }
    out
}

/// Delete `worker_id`'s mailbox and any stale claim file.
///
/// Called on worker exit so a finished worker leaves no mailbox behind that a
/// *new* worker reusing the id would later pick up as phantom guidance. Both
/// paths are removed regardless of which one exists.
pub fn remove_steer_file(worker_id: &str) {
    remove_steer_file_in(&ScratchRoot::from_env(), worker_id)
}

/// [`remove_steer_file`] under an explicit scratch root.
pub fn remove_steer_file_in(root: &ScratchRoot, worker_id: &str) {
    for path in [
        steer_path_in(root, worker_id),
        claim_path_in(root, worker_id),
    ] {
        match std::fs::remove_file(&path) {
            Ok(()) => {
                debug!(worker = %worker_id, path = %path.display(), "Removed steering mailbox")
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => {
                warn!(worker = %worker_id, path = %path.display(), error = %e, "Failed to remove steering mailbox")
            }
        }
    }
}

/// The last consolidator to steer a worker, and its immutable round base.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub(super) struct SteerSource {
    pub consolidator: String,
    pub round_base: Option<String>,
}

/// Every worker whose steer-source names `consolidator`, in id order.
///
/// The steer-source file is the durable record of "this consolidator steered
/// this worker": it is written when the consolidator routes a correction to the
/// worker and survives until the worker is retired. Scanning it is how a
/// consolidator's completion finds the round members it took responsibility
/// for, so none of them is left dangling once the consolidator's own branch
/// lands.
pub(super) fn steered_workers_of(root: &ScratchRoot, consolidator: &str) -> Vec<String> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(root.path()) else {
        return out;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        let Some(id) = name
            .strip_prefix("swe-wt-")
            .and_then(|rest| rest.strip_suffix(".steer-source"))
        else {
            continue;
        };
        if read_source(root, id).is_some_and(|source| source.consolidator == consolidator)
            && !out.iter().any(|known| known == id)
        {
            out.push(id.to_string());
        }
    }
    out.sort();
    out
}

pub(super) fn read_source(root: &ScratchRoot, id: &str) -> Option<SteerSource> {
    serde_json::from_slice(&std::fs::read(root.join(format!("swe-wt-{id}.steer-source"))).ok()?)
        .ok()
}

pub(super) fn write_source(
    root: &ScratchRoot,
    id: &str,
    source: Option<&SteerSource>,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        !id.is_empty()
            && id
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-')),
        "Invalid worker id: {id}"
    );
    let path = root.join(format!("swe-wt-{id}.steer-source"));
    if let Some(source) = source {
        let temporary = path.with_extension(format!("{}.tmp", uuid::Uuid::new_v4()));
        std::fs::write(&temporary, serde_json::to_vec(source)?)?;
        let result = std::fs::rename(&temporary, &path);
        if result.is_err() {
            let _ = std::fs::remove_file(&temporary);
        }
        result?;
    } else if let Err(error) = std::fs::remove_file(path)
        && error.kind() != std::io::ErrorKind::NotFound
    {
        return Err(error.into());
    }
    Ok(())
}
