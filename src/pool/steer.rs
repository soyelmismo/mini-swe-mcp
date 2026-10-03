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
//! `swe-wt-<id>.steer` file matches neither. The same is true of the
//! orchestrator steer log ([`record_orchestrator_steer_in`]), which is swept as
//! a worker companion by retirement instead.
//!
//! ## What the steer log is, and what may not reach it
//!
//! The orchestrator steer log (`swe-wt-<id>.steer-log.jsonl`) is never drained:
//! it is the round's record of the orchestrator's *own* scope amendments, read
//! back when a consolidator is dispatched much later. Two properties follow from
//! that, and both hold at the `open(2)` itself rather than in a check before it:
//!
//! * **It is written and read only as a regular file, never through a link.**
//!   The name sits in the shared scratch base, so a local user can pre-create
//!   it. Following a planted symlink would turn an orchestrator steer into an
//!   append into an arbitrary file, and would let a planted file be read back as
//!   the orchestrator's voice -- text the consolidator is told *supersedes* the
//!   worker's task. `O_NOFOLLOW` plus a check of the opened handle's own type
//!   refuses a link, a FIFO, or a device, in both directions.
//! * **Neither direction may block.** `O_NONBLOCK` at the open is what makes
//!   the type check reachable: `O_APPEND` on a FIFO blocks inside `open(2)`
//!   until a reader arrives, and a read-only `open` on a FIFO blocks until a
//!   writer does. `round_manifest` is async and reads the log on the reactor, so
//!   either would stall every request on the pool. Cleanup is therefore the
//!   worker loop's own responsibility ([`remove_steer_file`], invoked by a
//!   `Drop` guard held for the worker's whole lifetime), not the pruner's — a
//!   mailbox only exists while some `steer` call created it, and every worker
//!   removes its own on exit.
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
use std::io::{Read, Write};
use std::os::unix::fs::OpenOptionsExt;
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
    parse_records(&content, claim)
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

/// Path of the durable orchestrator-steer log for `worker_id`.
///
/// Unlike the mailbox this file is never drained: it is the round's record of
/// what the orchestrator told each worker *after* dispatch, so a consolidator
/// dispatched later can be shown the scope amendments that superseded the
/// original task. Written by [`record_orchestrator_steer_in`] alone, which
/// only the orchestrator's own steer path calls.
pub(super) fn steer_log_path_in(root: &ScratchRoot, worker_id: &str) -> PathBuf {
    root.join(format!("swe-wt-{worker_id}.steer-log.jsonl"))
}

/// Append one orchestrator-authored steer to `worker_id`'s durable log.
///
/// Only messages that reached the worker are recorded, so the log never names
/// an amendment the worker was never told. The consolidator routes its own
/// corrections through the same delivery path but under a recorded
/// [`SteerSource`], and it never calls this: the log is the *orchestrator's*
/// voice, and a consolidator must not be shown its own past steers as scope
/// amendments to a task it is judging.
///
/// A log that cannot be written is a warning, never a failed steer: the
/// message has already been delivered and the round loses an amendment, which
/// is strictly less bad than refusing guidance the worker needs.
pub(super) fn record_orchestrator_steer_in(root: &ScratchRoot, worker_id: &str, message: &str) {
    let path = steer_log_path_in(root, worker_id);
    let record = SteerRecord {
        message: message.to_string(),
        sent_at: super::unix_timestamp(),
        pid: std::process::id(),
    };
    // `to_string` on a `Value` cannot fail.
    let mut payload = serde_json::to_string(&record).unwrap_or_default();
    payload.push('\n');
    match open_steer_log_for_append(&path) {
        Ok(mut file) => {
            if let Err(e) = file.write_all(payload.as_bytes()) {
                warn!(worker = %worker_id, path = %path.display(), error = %e, "Failed to record orchestrator steer");
            }
        }
        Err(e) => {
            warn!(worker = %worker_id, path = %path.display(), error = %e, "Failed to record orchestrator steer");
        }
    }
}

/// Open the steer log to append one record, refusing anything but our own file.
///
/// The log lives in the shared scratch base, which any local user can write, so
/// the name alone proves nothing: a symlink planted at the path would turn
/// this append into an append into whatever the link names -- a file the
/// attacker cannot otherwise write -- with orchestrator-authored bytes in it.
/// `O_NOFOLLOW` refuses a link at the open itself, rather than a `symlink_metadata`
/// check that precedes it and leaves a window to swap a regular file for one.
///
/// `O_NONBLOCK` is here for the type that is not a link: `O_APPEND` on a FIFO
/// blocks inside `open(2)` until a reader shows up, which would park the whole
/// pool on a steer. The handle's own `fstat` then proves the type on the object
/// actually opened, so a FIFO or a device cannot be substituted for the log
/// either. A regular file ignores `O_NONBLOCK`, so the real log pays nothing.
fn open_steer_log_for_append(path: &Path) -> std::io::Result<std::fs::File> {
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)?;
    let meta = file.metadata()?;
    if !meta.file_type().is_file() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "steer log is not a regular file",
        ));
    }
    Ok(file)
}

/// Open the steer log to read it, refusing a link and any non-regular file.
///
/// The companion of [`open_steer_log_for_append`] on the read side: this text is
/// rendered to the consolidator as the orchestrator's scope amendments, so what
/// is read here is treated as the orchestrator's own voice. A planted file must
/// never be able to speak in that voice, and a FIFO must never be able to stall
/// the async caller that builds the manifest.
fn open_steer_log_for_read(path: &Path) -> std::io::Result<std::fs::File> {
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)?;
    let meta = file.metadata()?;
    if !meta.file_type().is_file() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "steer log is not a regular file",
        ));
    }
    Ok(file)
}

/// Ceiling on the bytes [`orchestrator_steers_in`] reads from one log.
///
/// The caller renders each worker's block through [`STEERS_BUDGET`], so a
/// record past that point cannot reach the prompt; reading the rest only costs
/// memory on a path that has to answer promptly. The log is append-only and
/// read from the start, where the amendments that matter are: the oldest
/// steering is the one a later steer retracts. The cap is deliberately
/// generous relative to [`STEERS_BUDGET`] so ordinary rounds are never cut.
const STEER_LOG_READ_CAP: u64 = 256 * 1024;

/// Every orchestrator steer `worker_id` received after dispatch, in order.
///
/// A missing log is the common case (nobody steered this worker) and yields an
/// empty vector. Lines that fail to parse are skipped with a warning, so one
/// torn or corrupt append cannot hide the amendments around it.
///
/// Bounded by [`STEER_LOG_READ_CAP`]: the log sits in the shared scratch base,
/// which any local user can write, so its size is not this process's to
/// assume. Only whole leading lines are parsed, so a cap never yields a
/// half-decoded record.
pub(super) fn orchestrator_steers_in(root: &ScratchRoot, worker_id: &str) -> Vec<String> {
    let path = steer_log_path_in(root, worker_id);
    // Opened the way the archive reader opens its own shared file: `O_NOFOLLOW`
    // so a planted link is never harvested as the orchestrator's own words, and
    // `O_NONBLOCK` so the type can be established without blocking in
    // `open(2)` -- a FIFO read-only blocks until a writer arrives, which would
    // stall every request on the pool. The handle's own type is then checked, so
    // a FIFO or a device cannot stand in for the log either. A missing log is
    // the common case and yields nothing.
    let file = match open_steer_log_for_read(&path) {
        Ok(file) => file,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Vec::new(),
        Err(e) => {
            warn!(worker = %worker_id, path = %path.display(), error = %e, "Refusing to read orchestrator steer log");
            return Vec::new();
        }
    };
    let mut content = String::new();
    if let Err(e) = file.take(STEER_LOG_READ_CAP).read_to_string(&mut content)
        && e.kind() != std::io::ErrorKind::UnexpectedEof
    {
        warn!(worker = %worker_id, path = %path.display(), error = %e, "Failed to read orchestrator steer log");
        return Vec::new();
    }
    // A cut at the cap can leave a partial trailing record; dropping it keeps a
    // half-written line from being read as guidance.
    if !content.is_empty() && !content.ends_with('\n') {
        content.truncate(content.rfind('\n').map_or(0, |i| i + 1));
    }
    parse_records(&content, &path)
}

/// The parsed records of an already-read mailbox body.
fn parse_records(content: &str, source: &Path) -> Vec<String> {
    let mut out = Vec::new();
    for line in content.lines() {
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<SteerRecord>(line) {
            Ok(record) => out.push(record.message),
            Err(e) => {
                warn!(path = %source.display(), error = %e, "Skipping unparsable steering mailbox line")
            }
        }
    }
    out
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
