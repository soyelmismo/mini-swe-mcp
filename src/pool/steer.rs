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
//!   either would stall every request on the pool.
//!
//!   Cleanup is therefore the worker loop's own responsibility
//!   ([`remove_steer_file`], invoked by a `Drop` guard held for the worker's
//!   whole lifetime), not the pruner's — a mailbox only exists while some
//!   `steer` call created it, and every worker removes its own on exit.
//!
//! Messages are **JSON lines**, one object per line:
//! `{"message": "…", "sent_at": 1700000000, "pid": 4242}`.
//! A record in the durable orchestrator steer log also carries the scratch
//! root's `nonce`, which is what proves the log is this pool's own voice rather
//! than a file planted in the shared base (see [`log_nonce_in`]).
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
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
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
    /// Proof that this scratch root wrote the record (see [`log_nonce_in`]).
    ///
    /// `None` for a record written before this field existed, and `None` for
    /// one an attacker composed: the field is exactly what separates them, so
    /// the reader rejects it rather than guessing from the other fields.
    #[serde(default)]
    nonce: Option<String>,
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
        // A mailbox message is claimed by rename and handed to the worker as
        // guidance, never rendered as the orchestrator's authority, so it needs
        // no nonce. Only the durable log does (see [`log_nonce_in`]).
        nonce: None,
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
    parse_mailbox(&content, claim)
}

/// Every well-formed message line of a mailbox, in order.
///
/// No authentication here, and deliberately so: a mailbox is claimed by
/// [`drain_steer_messages_in`] with a `rename`, so only one reader ever sees a
/// generation, and its messages are handed to the worker as guidance to
/// consider rather than rendered as the orchestrator's authority. The durable
/// log ([`orchestrator_steers_in`]) is the one whose text reaches the
/// consolidator as scope amendments, and that is where [`parse_records`] and
/// its nonce check live.
fn parse_mailbox(content: &str, source: &Path) -> Vec<String> {
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

/// Path of this scratch root's steer-log nonce.
fn log_nonce_path_in(root: &ScratchRoot) -> PathBuf {
    root.join(".steer-log-nonce")
}

/// The secret that marks a steer-log record as this scratch root's own.
///
/// The log sits in the shared scratch base, which any local user can write, so
/// a planted file at the log path is indistinguishable from a real one by
/// content: a `message`, a `sent_at` and a `pid` are all forgeable, and the
/// writer cannot be "the process that wrote it" because a later consolidator
/// legitimately reads a log an earlier process wrote. Ownership and permissions
/// do not separate them either, because the attacker plants the file as the
/// same user the pool runs as.
///
/// What does separate them is a secret the pool holds and an attacker does not
/// have. Every record carries it, and a record without a matching one is not
/// this pool's voice: it is dropped, whatever it claims to say. This is the
/// difference between the two trust models -- "the file exists" (plantable) and
/// "only this pool could have written this file" (not, without the secret).
///
/// That only holds if the pool *chooses* the value, so the nonce is created
/// with `O_EXCL` and an existing file is never adopted: the name is fixed and
/// predictable in the shared base, and a value found there is one the local
/// user chose, so trusting it would hand them the very trust this exists to
/// establish. A file already at the name is read back only when it is a regular
/// file with owner-only permissions -- which is what this pool's own `0600`
/// create leaves -- and refused otherwise, so a planted, linked or
/// world-readable file authenticates nothing.
///
/// The nonce is per *scratch root*, not per process, so a consolidator started
/// later verifies the records the orchestrator wrote before it. It is created
/// on first use, owner-only, and never leaves the root; if it cannot be created
/// or read, the log is refused rather than trusted unauthenticated -- an
/// unreadable secret means an unauthenticated log, which is the case the whole
/// mechanism exists to prevent.
fn log_nonce_in(root: &ScratchRoot) -> Option<String> {
    log_nonce_of(root)
}

/// This scratch root's steer-log nonce (test support).
///
/// Exposed so a test that exercises the read *cap* can plant records the reader
/// will accept: authentication and truncation are independent properties, and a
/// test of one must not have to defeat the other to see the thing it is testing.
#[doc(hidden)]
pub fn __test_log_nonce_in(root: &ScratchRoot) -> Option<String> {
    log_nonce_of(root)
}

fn log_nonce_of(root: &ScratchRoot) -> Option<String> {
    let path = log_nonce_path_in(root);
    let fresh = uuid::Uuid::new_v4().simple().to_string();
    // Established with `O_EXCL`, never adopted from whatever is already at the
    // name. The nonce is only a secret if the pool chose it, and the name is
    // fixed and predictable in the shared scratch base: an unprivileged local
    // user who creates it first, or who writes it before this pool ever runs,
    // would otherwise hold the secret and could sign records that the reader
    // then renders to the consolidator as the orchestrator's own scope
    // amendments -- the one text that SUPERSEDES a worker's task. So the first
    // `open` either creates the file (this pool chose the value) or finds it
    // already there, and only the *create* branch is trusted; a file that was
    // there before us is refused outright rather than read, so a planted
    // nonce cannot authenticate anything.
    match create_nonce_exclusive(&path, &fresh) {
        // This call created the file, so the value it wrote is the secret.
        Ok(()) => return Some(fresh),
        // The name was already taken (`O_EXCL` reports it as `AlreadyExists`):
        // the existing content is not trusted for merely having been found, so
        // it is resolved below.
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(e) => {
            warn!(path = %path.display(), error = %e, "Cannot establish the steer-log nonce");
            return None;
        }
    }
    // The file exists, so this pool did not create it on this call. Reading it
    // back is only sound when the pool can prove it wrote it: an existing file
    // is accepted when it is owner-only *and* this process created it earlier
    // (the common case for every process after the first, and for the same
    // process on a later call). Anything else -- a planted file, a link, a
    // world-readable one -- is refused, because a nonce read from it is not a
    // secret this pool holds.
    // SAFETY: `getuid` takes no arguments and cannot fail.
    let our_uid = unsafe { libc::getuid() };
    read_back_owned_nonce(&path, our_uid)
}

/// Create the nonce file exclusively, refusing a link and an existing name.
fn create_nonce_exclusive(path: &Path, value: &str) -> std::io::Result<()> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true) // `O_EXCL|O_CREAT`: never adopt, never truncate.
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW)
        .open(path)?;
    file.write_all(value.as_bytes())?;
    Ok(())
}

/// Read the nonce back, but only from a file this pool can prove it wrote.
///
/// The proof is ownership plus owner-only permissions: the file was created by
/// this process or by a peer pool process running as the same user, it is
/// owned by this pool's own uid, it is a regular file rather than a link or a
/// device, and nothing outside this user can read it. A planted file in a
/// shared base is readable and writable by whoever planted it, and is refused
/// rather than believed. The uid check is what separates a file this pool's
/// user created from one a *different* local user pre-created at the fixed,
/// predictable name: the two are indistinguishable by mode alone (both can be
/// `0600`), so without it a foreign-owner file would be adopted as this pool's
/// secret and could sign forged steer-log records that the consolidator then
/// renders as the orchestrator's own scope amendments.
fn read_back_owned_nonce(path: &Path, our_uid: u32) -> Option<String> {
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)
        .ok()?;
    let meta = file.metadata().ok()?;
    if !meta.file_type().is_file() {
        warn!(path = %path.display(), "Refusing the steer-log nonce: not a regular file");
        return None;
    }
    if meta.mode() & 0o077 != 0 {
        warn!(path = %path.display(), mode = format!("{:o}", meta.mode() & 0o7777), "Refusing the steer-log nonce: readable or writable beyond this user");
        return None;
    }
    // A `0600` file at the fixed name is not proof this pool wrote it: a
    // different local user can pre-create the same mode. Only a file owned by
    // this pool's own uid is this pool's voice.
    if meta.uid() != our_uid {
        warn!(path = %path.display(), uid = meta.uid(), "Refusing the steer-log nonce: owned by another user");
        return None;
    }
    let mut content = String::new();
    file.take(STEER_LOG_READ_CAP)
        .read_to_string(&mut content)
        .ok()?;
    let nonce = content.trim().to_string();
    if nonce.is_empty() {
        return None;
    }
    Some(nonce)
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
    // A record with no nonce is indistinguishable from a planted one, so it is
    // not written at all: better a lost amendment than guidance the consolidator
    // would have to refuse anyway.
    let Some(nonce) = log_nonce_in(root) else {
        warn!(worker = %worker_id, path = %path.display(), "Not recording orchestrator steer: the steer-log nonce is unavailable");
        return;
    };
    let record = SteerRecord {
        message: message.to_string(),
        sent_at: super::unix_timestamp(),
        pid: std::process::id(),
        nonce: Some(nonce),
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
    // Nothing is rendered unless this scratch root's own nonce authenticates
    // it: see [`log_nonce_in`]. A log that cannot be authenticated is not this
    // pool's voice, so it yields no amendments rather than a planted one.
    let Some(nonce) = log_nonce_in(root) else {
        warn!(worker = %worker_id, path = %path.display(), "Refusing to read orchestrator steer log: no nonce to authenticate it");
        return Vec::new();
    };
    parse_records(&content, &path, &nonce)
}

/// The messages of an already-read steer log that this scratch root wrote.
///
/// A record whose `nonce` does not match is dropped, not rendered: it is either
/// planted in the shared base or left over from a scratch root that has since
/// been recreated, and neither is this pool's voice. A malformed line is
/// dropped the same way, so one corrupt append cannot hide the amendments
/// around it.
fn parse_records(content: &str, source: &Path, nonce: &str) -> Vec<String> {
    let mut out = Vec::new();
    for line in content.lines() {
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<SteerRecord>(line) {
            Ok(record) if record.nonce.as_deref() != Some(nonce) => {
                warn!(path = %source.display(), "Skipping an orchestrator steer record this scratch root did not write")
            }
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

#[cfg(test)]
mod nonce_ownership_tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    /// A `0600` nonce file owned by a *different* user must not be adopted:
    /// mode alone cannot separate this pool's own file from one a foreign
    /// local user pre-created at the fixed, predictable name, so the owner
    /// check is what refuses it. Without it, a foreign-owner `0600` file would
    /// be read back as this pool's secret and could sign forged steer-log
    /// records that the consolidator renders as the orchestrator's own scope
    /// amendments.
    #[test]
    fn foreign_owner_0600_nonce_is_refused() {
        let dir = std::env::temp_dir().join(format!(
            "steer-nonce-foreign-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(".steer-log-nonce");
        std::fs::write(&path, "foreign-secret").unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();

        // SAFETY: `getuid` takes no arguments and cannot fail.
        let our_uid = unsafe { libc::getuid() };
        // A uid that is not the current process's owner: simulates a file
        // planted by a different local user. (The test cannot chown to another
        // real user without privileges, so it exercises the check by passing
        // the foreign uid directly.)
        let foreign_uid = our_uid.wrapping_add(1);
        assert_ne!(foreign_uid, our_uid, "test assumption: foreign uid differs");
        assert!(
            read_back_owned_nonce(&path, foreign_uid).is_none(),
            "a nonce file owned by another user must be refused"
        );
    }

    /// The pool's own `0600` file (owned by the pool's uid) is adopted, so a
    /// later consolidator process can verify the records an earlier
    /// orchestrator process wrote.
    #[test]
    fn own_uid_0600_nonce_is_accepted() {
        let dir = std::env::temp_dir().join(format!(
            "steer-nonce-own-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(".steer-log-nonce");
        std::fs::write(&path, "our-secret").unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();

        // SAFETY: `getuid` takes no arguments and cannot fail.
        let our_uid = unsafe { libc::getuid() };
        assert_eq!(
            read_back_owned_nonce(&path, our_uid).as_deref(),
            Some("our-secret"),
            "the pool's own owner-only nonce file is adopted"
        );
    }
}
