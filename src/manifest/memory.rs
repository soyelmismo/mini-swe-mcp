//! Persistent, per-role agent memory.
//!
//! A subagent's conversation is volatile: every dispatch starts from the same
//! static [`SYSTEM_PROMPT`](crate::agent::SYSTEM_PROMPT) and nothing a previous
//! run learned survives the worktree. This module gives each *role* a durable
//! memory file at `<repo>/.agents/memory/<alias>.md` that is loaded into the
//! system prompt when the prompt is built.
//!
//! Three properties are load-bearing, and are asserted by the tests in
//! `tests.rs`:
//!
//! * **Optional by design.** [`load_agent_memory`] returns `None` — never an
//!   empty prompt section — when the file is absent, unreadable or blank. A
//!   repository without `.agents/memory/` behaves exactly as it did before this
//!   module existed.
//! * **Bounded.** Injected memory is capped at [`MAX_MEMORY_PROMPT_BYTES`].
//!   The cap keeps the *most recent* entries, which are the ones that describe
//!   the current state of the repo.
//! * **Traversal-safe.** The alias is reduced to a conservative
//!   `[a-z0-9_-]` slug before it touches the filesystem, so a hostile `model`
//!   argument can never read outside `.agents/memory/`.

use std::path::{Path, PathBuf};

/// Directory (relative to a repository root) holding the per-role memory files.
pub const MEMORY_DIR: &str = ".agents/memory";

/// Header rendered above an injected memory block.
///
/// It is a constant so the injected text is recognisable in a transcript and so
/// an agent can tell *its own* notes apart from the static instructions.
const MEMORY_HEADER: &str = "PERSISTENT ROLE MEMORY (from .agents/memory/):";

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
}
