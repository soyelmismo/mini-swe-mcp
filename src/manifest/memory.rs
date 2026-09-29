//! Persistent, per-role agent memory.
//!
//! A subagent's conversation is volatile: every dispatch starts from the same
//! static [`SYSTEM_PROMPT`](crate::agent::SYSTEM_PROMPT) and nothing a previous
//! run learned survives the worktree. This module gives each *role* a durable
//! memory file at `<repo>/.agents/memory/<alias>.md` that is loaded into the
//! system prompt when the prompt is built, and that can be appended to with
//! atomic, self-contained takeaways ([`append_agent_memory`]).
//!
//! Four properties are load-bearing, and are asserted by the tests in
//! `tests.rs`:
//!
//! * **Optional by design.** [`load_agent_memory`] returns `None` — never an
//!   empty prompt section — when the file is absent, unreadable or blank. A
//!   repository without `.agents/memory/` behaves exactly as it did before this
//!   module existed.
//! * **Bounded.** Injected memory is capped at [`MAX_MEMORY_PROMPT_BYTES`].
//!   Memory is appended to forever, so an unbounded file would eventually push
//!   the instructions out of the model's attention (and, past the context
//!   window, out of the request entirely). The cap keeps the *most recent*
//!   entries, which are the ones that describe the current state of the repo.
//! * **Atomic writes.** [`append_agent_memory`] writes through a temporary
//!   file and `rename`s it into place, so a reader in the repo root (or a
//!   concurrent sync from `worktree`) never observes a half-written note.
//! * **Traversal-safe.** The alias is reduced to a conservative
//!   `[a-z0-9_-]` slug before it touches the filesystem, so a hostile `model`
//!   argument can never read or write outside `.agents/memory/`.
//! * **Symlink-safe.** The staging file is *exclusively* created
//!   ([`write_new_staging_file`]), so a pre-planted `.ninja.md.<pid>.<seq>.tmp`
//!   symlink is never followed, and a symlinked `.agents/memory` directory is
//!   rejected instead of quietly redirecting the write.
//!
//! Role memory is deliberately *not* part of the memoized catalog
//! ([`super::cache`]): memory changes on disk between renders, so it must be
//! re-read on every build instead of being interned process-wide.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

/// Directory (relative to a repository root) holding the per-role memory files.
pub const MEMORY_DIR: &str = ".agents/memory";

/// Header rendered above an injected memory block.
///
/// It is a constant so the injected text is recognisable in a transcript and so
/// an agent can tell *its own* notes apart from the static instructions.
const MEMORY_HEADER: &str = "PERSISTENT ROLE MEMORY (from .agents/memory/):";

/// Serializes the read-modify-write of [`append_agent_memory`].
///
/// `rename` alone makes each *write* atomic, but two concurrent appends to the
/// same role memory would still race: both would read the pre-append contents and
/// the second `rename` would discard the first note. One process-wide mutex makes
/// the whole append a critical section; role memories are written a handful of
/// times per task, so the contention is negligible.
static APPEND_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Hard ceiling on the number of memory bytes injected into one system prompt.
///
/// Generous enough for a long-lived memory file, small enough that the system
/// prompt stays dominated by the actual instructions.
pub const MAX_MEMORY_PROMPT_BYTES: usize = 8 * 1024;

/// Reduce a model alias to a filesystem-safe slug.
///
/// Only `[a-z0-9_-]` survives, so `.`, `/`, `\` and NUL can never appear in the
/// resulting path component: `../../etc/passwd` collapses to `-etc-passwd`.
/// Anything that sanitizes away to nothing (or that is unreasonably long) yields
/// `None`, which the callers treat as "no memory for this alias".
fn memory_slug(alias: &str) -> Option<String> {
    // Every surviving character is one ASCII byte, so the byte budget doubles as
    // a character budget and the slug is capped without re-measuring.
    const MAX_SLUG_BYTES: usize = 64;

    let mut slug = String::with_capacity(alias.len().min(MAX_SLUG_BYTES));
    for ch in alias.chars().take(MAX_SLUG_BYTES) {
        if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
            slug.push(ch.to_ascii_lowercase());
        } else {
            slug.push('-');
        }
    }
    // Trim the separator noise an alias like `"  ninja  "` would otherwise keep.
    let trimmed = slug.trim_matches('-');
    (!trimmed.is_empty()).then(|| trimmed.to_owned())
}

/// Path of the memory file backing `model_alias` inside `repo_path`.
///
/// `None` when the alias contains no usable characters (see [`memory_slug`]).
pub fn agent_memory_path(repo_path: &Path, model_alias: &str) -> Option<PathBuf> {
    memory_slug(model_alias).map(|slug| repo_path.join(MEMORY_DIR).join(format!("{slug}.md")))
}

/// Load the role memory for `model_alias`, if any.
///
/// Returns the trimmed contents of `<repo>/.agents/memory/<alias>.md`, truncated
/// to the newest [`MAX_MEMORY_PROMPT_BYTES`] when the file is longer, or `None`
/// when there is nothing worth injecting: the file is missing, is not a file, is
/// not valid UTF-8, or contains only whitespace. A memory read is *best effort* —
/// an unreadable file degrades to "no memory" rather than failing a dispatch.
pub fn load_agent_memory(repo_path: &Path, model_alias: &str) -> Option<String> {
    let path = agent_memory_path(repo_path, model_alias)?;
    let raw = std::fs::read_to_string(&path).ok()?;
    let text = truncate_to_memory_budget(raw.trim());
    (!text.is_empty()).then(|| text.to_string())
}

/// Keep at most [`MAX_MEMORY_PROMPT_BYTES`] of `text`, dropping the oldest part.
///
/// The tail of at most `MAX_MEMORY_PROMPT_BYTES` bytes is taken — the cut index
/// is rounded up to a `char` boundary first so slicing can never panic inside
/// a multi-byte code point — then any leading partial line is dropped, so a
/// truncated block starts at a line boundary and never begins mid-sentence.
fn truncate_to_memory_budget(text: &str) -> &str {
    if text.len() <= MAX_MEMORY_PROMPT_BYTES {
        return text;
    }
    // The byte budget need not fall on a `char` boundary (e.g. a cut inside a
    // multi-byte code point); `ceil_char_boundary` rounds the cut index up so the
    // slice below can never panic and stays within the byte budget, then drop
    // any leading partial line so the block starts at a line boundary.
    let tail = &text[text.ceil_char_boundary(text.len() - MAX_MEMORY_PROMPT_BYTES)..];
    match tail.find('\n') {
        Some(offset) => &tail[offset + 1..],
        None => tail,
    }
}

/// Append one atomic takeaway to the role memory of `model_alias`.
///
/// "Atomic" covers both halves of the append:
///
/// * **All-or-nothing** — the new file is staged next to the target and moved
///   into place with [`std::fs::rename`], which is atomic within a filesystem.
///   Both paths are built in the same directory, so a concurrent reader (a
///   `load_agent_memory` in the repo root, or a `worktree` artifact sync) sees
///   either the whole previous file or the whole new one, never a partial note.
/// * **Serialized** — the read-modify-write runs under [`APPEND_LOCK`], so two
///   agents of the same role finishing at the same time cannot both read the old
///   contents and have the second `rename` discard the first note.
///
/// The note is normalized to a single markdown list item, so successive notes
/// read as an ordered log. The parent directory is created on demand and the
/// `MEMORY_HEADER` is seeded on first write, which keeps an appended-only file in
/// exactly the format [`load_agent_memory`] injects.
///
/// Errors are returned rather than swallowed: a failed memory write is the one
/// memory operation whose failure the caller should surface.
pub fn append_agent_memory(repo_path: &Path, model_alias: &str, note: &str) -> Result<()> {
    let path = agent_memory_path(repo_path, model_alias)
        .with_context(|| format!("Invalid model alias for memory: {model_alias:?}"))?;
    let dir = path
        .parent()
        .context("Memory path has no parent directory")?;

    let entry = normalize_note(note)?;

    // A poisoned lock only means some *other* append panicked mid-way; the file
    // itself is still intact (publishing is a rename), so recovery is safe.
    let _guard = APPEND_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    // `create_dir_all` reports success for any already-existing path, *including*
    // a symlink to a directory somewhere else entirely, so a hostile repository
    // that symlinks `.agents/memory` would silently redirect every memory write
    // (and every staged temp file) out of the worktree. `symlink_metadata` reads
    // the link itself rather than following it, which is exactly the check needed
    // to refuse that.
    match std::fs::symlink_metadata(dir) {
        Ok(meta) if meta.is_dir() => {}
        Ok(_) => anyhow::bail!(
            "Refusing to write agent memory: {} is not a real directory",
            dir.display()
        ),
        Err(_) => std::fs::create_dir_all(dir)
            .with_context(|| format!("Failed to create memory directory {}", dir.display()))?,
    }

    // Read-then-rewrite (rather than a bare `O_APPEND` write) so the note can be
    // published with a temp-file + `rename`, which is atomic: a concurrent
    // `load_agent_memory` in the repo root — or a `worktree` artifact sync —
    // observes either the old file or the new one, never a partial write.
    // A missing file is the normal first-write case and reads as empty; any other
    // read failure (permissions, a directory in the way) is deliberately *not*
    // fatal here either, because overwriting with a fresh header is a better
    // outcome than aborting the append.
    let mut existing = std::fs::read_to_string(&path).unwrap_or_default();
    if existing.trim().is_empty() {
        existing = format!("{MEMORY_HEADER}\n");
    } else if !existing.ends_with('\n') {
        existing.push('\n');
    }
    existing.push_str(&entry);

    // A hidden, same-directory staging name. It carries the pid so two *processes*
    // appending the same role never collide, and a monotonic counter so two
    // appends in this process never collide either; `APPEND_LOCK` already
    // serializes the append itself, so this only has to be unique, not ordered.
    static APPEND_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let tmp = dir.join(format!(
        ".{}.{}.{}.tmp",
        path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("memory"),
        std::process::id(),
        APPEND_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
    ));
    write_new_staging_file(&tmp, existing.as_bytes())
        .with_context(|| format!("Failed to stage memory update in {}", tmp.display()))?;
    if let Err(e) = std::fs::rename(&tmp, &path) {
        let _ = std::fs::remove_file(&tmp);
        return Err(anyhow::Error::new(e))
            .with_context(|| format!("Failed to publish memory file {}", path.display()));
    }
    Ok(())
}

/// Create `path` and write `bytes` to it, exclusively and durably.
///
/// The staging name is derived from the target name plus a pid and a counter, so
/// it is *predictable* — which makes [`std::fs::write`] unsafe here: it opens
/// with `O_CREAT | O_TRUNC` and **follows symlinks**, so a symlink pre-planted
/// at that name (in the shared `.agents/memory` directory) would redirect the
/// staged note to any file the daemon can write. [`OpenOptions::create_new`]
/// fails with `AlreadyExists` instead of following the link, which closes the
/// race, and the collision it reports is resolved by the caller's next counter
/// value.
///
/// [`std::fs::write`] also returns as soon as the data reached the page cache, so
/// a note that has been accepted could still be lost to a crash. `sync_all`
/// closes that window: the buffer is on stable storage before the `rename` that
/// publishes it, so recovery always finds a complete file.
fn write_new_staging_file(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(bytes)?;
    file.sync_all()
}

/// Reduce a note to a single markdown list item.
///
/// Whitespace is collapsed so a multi-line note cannot break the one-entry-per-
/// line format, and an empty result is rejected rather than written as a blank
/// bullet.
fn normalize_note(note: &str) -> Result<String> {
    // Collapsed straight into the final bullet: one allocation, and a short
    // note never borrows an intermediate `Vec<&str>`.
    let mut entry = String::with_capacity(note.len() + 4);
    entry.push_str("- ");
    for word in note.split_whitespace() {
        if entry.len() > 2 {
            entry.push(' ');
        }
        entry.push_str(word);
    }
    if entry.len() == 2 {
        anyhow::bail!("Refusing to append an empty note to agent memory");
    }
    entry.push('\n');
    Ok(entry)
}

/// Render the system-prompt section for a role, memory included.
///
/// Returns `None` when the role has no memory file, so the caller can pass
/// [`crate::agent::SYSTEM_PROMPT`] through untouched and the prompt stays
/// byte-identical to the pre-memory behaviour for every repository that has not
/// opted in.
pub(crate) fn memory_prompt_section(repo_path: &Path, model_alias: &str) -> Option<String> {
    let memory = load_agent_memory(repo_path, model_alias)?;
    // Appended-only files are seeded with `MEMORY_HEADER`, and the shipped
    // defaults start with it too, so avoid emitting the header twice.
    if memory.starts_with(MEMORY_HEADER) {
        Some(format!("\n\n{memory}"))
    } else {
        Some(format!("\n\n{MEMORY_HEADER}\n{memory}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slug_is_traversal_safe() {
        assert_eq!(memory_slug("ninja").as_deref(), Some("ninja"));
        assert_eq!(memory_slug("Ninja").as_deref(), Some("ninja"));
        assert_eq!(
            memory_slug("../../etc/passwd").as_deref(),
            Some("etc-passwd")
        );
        assert_eq!(memory_slug("a/b\\c").as_deref(), Some("a-b-c"));
        assert_eq!(memory_slug("   "), None);
        assert_eq!(memory_slug("///"), None);
        assert_eq!(memory_slug(&"x".repeat(500)).map(|s| s.len()), Some(64));
    }

    #[test]
    fn truncate_never_splits_a_code_point() {
        let text = "é".repeat(MAX_MEMORY_PROMPT_BYTES);
        let cut = truncate_to_memory_budget(&text);
        assert!(cut.len() <= MAX_MEMORY_PROMPT_BYTES);
        assert!(text.ends_with(cut));
    }

    #[test]
    fn truncate_never_panics_on_a_mid_char_cut() {
        // One trailing ASCII byte shifts every cut index by one, so the budget
        // boundary lands inside a two-byte `é` instead of on it.
        let text = format!("{}a", "é".repeat(MAX_MEMORY_PROMPT_BYTES));
        assert!(text.len() > MAX_MEMORY_PROMPT_BYTES);
        let cut = truncate_to_memory_budget(&text);
        assert!(cut.len() <= MAX_MEMORY_PROMPT_BYTES);
        assert!(text.ends_with(cut));
        assert!(cut.is_char_boundary(0));
    }

    #[test]
    fn normalize_note_collapses_to_one_bullet() {
        assert_eq!(
            normalize_note("  reproduce\n  before   patching\n").expect("note"),
            "- reproduce before patching\n"
        );
        assert!(
            normalize_note(" \n\t ").is_err(),
            "a whitespace-only note is not a memory entry"
        );
    }

    /// The staging name is predictable, so the note must be staged with
    /// `O_CREAT | O_EXCL` (`create_new`), never the symlink-following
    /// `O_CREAT | O_TRUNC` that [`std::fs::write`] uses.
    #[test]
    fn staging_a_note_never_follows_a_planted_symlink() {
        let dir = std::env::temp_dir().join(format!("swe-memory-stage-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("staging dir");
        let victim = dir.join("victim.md");
        std::fs::write(&victim, "untouched\n").expect("victim");
        let link = dir.join(".note.md.1.2.tmp");
        std::os::unix::fs::symlink(&victim, &link).expect("planted symlink");

        let err = write_new_staging_file(&link, b"- note\n").expect_err("must not follow the link");
        assert_eq!(err.kind(), std::io::ErrorKind::AlreadyExists);
        assert_eq!(
            std::fs::read_to_string(&victim).expect("victim intact"),
            "untouched\n",
            "a pre-planted symlink must never receive the staged note"
        );

        // The exclusive create succeeds once the name is free.
        write_new_staging_file(&dir.join(".note.md.1.3.tmp"), b"- note\n").expect("fresh stage");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A symlinked `.agents/memory` passes `create_dir_all` (which reports
    /// success for any existing path) and would otherwise redirect every memory
    /// write outside the repository.
    #[test]
    fn append_refuses_a_symlinked_memory_directory() {
        let repo = std::env::temp_dir().join(format!("swe-memory-link-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&repo);
        std::fs::create_dir_all(repo.join(".agents")).expect(".agents");
        let elsewhere = repo.join("elsewhere");
        std::fs::create_dir_all(&elsewhere).expect("decoy");
        std::os::unix::fs::symlink(&elsewhere, repo.join(MEMORY_DIR)).expect("symlinked dir");

        let err = append_agent_memory(&repo, "ninja", "note")
            .expect_err("append must not write through a symlinked memory dir");
        assert!(
            err.to_string().contains("not a real directory"),
            "the refusal must say why: {err}"
        );
        assert!(
            std::fs::read_dir(&elsewhere)
                .expect("decoy readable")
                .next()
                .is_none(),
            "nothing may be written through the symlink"
        );
        let _ = std::fs::remove_dir_all(&repo);
    }
}
