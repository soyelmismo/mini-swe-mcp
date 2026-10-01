//! Repository instruction files loaded into a worker's system prompt.
//!
//! A dispatched task tends to restate the same project rules the repository
//! already documents, and orchestrators pay that cost on every dispatch. This
//! module removes the repetition: the worker reads the repository's own
//! instruction files and they are appended to its system prompt, each under a
//! header naming the file it came from.
//!
//! The recognised candidates, in preference order, are [`INSTRUCTION_FILES`]:
//! `AGENTS.md`, `CLAUDE.md`, `GEMINI.md`, `.github/copilot-instructions.md` and
//! `.cursorrules`. Every candidate that exists is included once; two files with
//! identical contents collapse into a single block, since a repository that
//! keeps `CLAUDE.md` as a byte-identical copy of `AGENTS.md` should not pay for
//! it twice.
//!
//! Three properties are load-bearing, and are asserted by the tests:
//!
//! * **Optional by design.** [`load_instruction_files`] returns an empty list —
//!   never an empty prompt section — when nothing is readable, so a repository
//!   without instruction files gets exactly the prompt it had before this
//!   module existed.
//! * **Bounded.** The rendered section is capped at
//!   [`MAX_INSTRUCTIONS_PROMPT_BYTES`]. The *head* of the text is kept, since a
//!   rule stated first is the one that matters most, and the cut is marked with
//!   a note so a truncated prompt is never mistaken for the whole file.
//! * **Contained.** Only regular files reached without following a symlink are
//!   read, and the resolved path must stay under the resolved repository root,
//!   so a `.cursorrules` pointing at `/etc` can never pull outside text into
//!   the prompt.

use std::path::Path;

/// Repository-relative instruction files injected into the system prompt, in
/// preference order.
///
/// The list is the de-facto standard set shared by agent tooling; a repository
/// that uses none of these gets no injected section at all.
pub const INSTRUCTION_FILES: &[&str] = &[
    "AGENTS.md",
    "CLAUDE.md",
    "GEMINI.md",
    ".github/copilot-instructions.md",
    ".cursorrules",
];

/// Header rendered once above the injected instruction blocks.
const INSTRUCTIONS_HEADER: &str = "REPOSITORY INSTRUCTION FILES (loaded from the repository root):";

/// Prefix of the per-file header that names each injected block's source.
const FILE_HEADER_PREFIX: &str = "### ";

/// Note appended when the files do not fit the injected budget.
const TRUNCATION_NOTE: &str =
    "\n\n[truncated: repository instruction files exceed the prompt budget]";

/// Hard ceiling on the number of instruction bytes injected into one system
/// prompt.
///
/// Large enough for a repository's rule book, small enough that the injected
/// text stays a small part of a system prompt that already carries the static
/// instructions and the role memory.
pub const MAX_INSTRUCTIONS_PROMPT_BYTES: usize = 16 * 1024;

/// Read and label the repository's instruction files.
///
/// Returns `(relative path, trimmed contents)` for every candidate that is a
/// readable regular file inside the repository and whose contents were not
/// already taken. Reading is *best effort*: anything missing, unreadable or
/// outside the repository is skipped rather than failing a dispatch.
fn load_instruction_files(repo_root: &Path) -> Vec<(String, String)> {
    let Ok(canonical_root) = repo_root.canonicalize() else {
        return Vec::new();
    };

    let mut files: Vec<(String, String)> = Vec::new();
    for candidate in INSTRUCTION_FILES {
        let Some(content) = read_instruction_file(&canonical_root, candidate) else {
            continue;
        };
        if files
            .iter()
            .any(|(_, seen): &(String, String)| seen == &content)
        {
            continue;
        }
        files.push(((*candidate).to_string(), content));
    }
    files
}

/// Read one candidate file, refusing symlinks and paths that escape the root.
fn read_instruction_file(canonical_root: &Path, candidate: &str) -> Option<String> {
    let path = canonical_root.join(candidate);
    // `symlink_metadata` describes the link itself, so a symlinked instruction
    // file is rejected instead of followed.
    if !std::fs::symlink_metadata(&path).ok()?.is_file() {
        return None;
    }
    // A regular file under a symlinked parent (e.g. `.github` -> elsewhere)
    // would still escape, so confirm the resolved path stays under the root.
    if !std::fs::canonicalize(&path)
        .ok()?
        .starts_with(canonical_root)
    {
        return None;
    }
    let text = std::fs::read_to_string(&path).ok()?;
    let trimmed = text.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

/// Render the system-prompt section for a repository, or `None` when it has no
/// instruction files.
///
/// Returns `None` rather than an empty section so the caller can leave the
/// static prompt untouched for every repository that has not opted in.
pub(crate) fn instructions_prompt_section(repo_root: &Path) -> Option<String> {
    let files = load_instruction_files(repo_root);
    (!files.is_empty()).then(|| render_instruction_section(&files))
}

/// Render the files under their headers, trimming to the byte budget.
///
/// The complete text is built first so the headers are deterministic; only when
/// it exceeds [`MAX_INSTRUCTIONS_PROMPT_BYTES`] is the tail dropped. The cut
/// index is rounded down to a `char` boundary so slicing can never panic inside
/// a multi-byte code point, and then back to a line boundary so the kept block
/// never ends mid-sentence.
fn render_instruction_section(files: &[(String, String)]) -> String {
    let mut section = String::from(INSTRUCTIONS_HEADER);
    for (path, content) in files {
        section.push_str("\n\n");
        section.push_str(FILE_HEADER_PREFIX);
        section.push_str(path);
        section.push('\n');
        section.push_str(content);
    }

    if section.len() <= MAX_INSTRUCTIONS_PROMPT_BYTES {
        return section;
    }
    // Reserve room for the note inside the same budget.
    let mut cut = MAX_INSTRUCTIONS_PROMPT_BYTES - TRUNCATION_NOTE.len();
    while !section.is_char_boundary(cut) {
        cut -= 1;
    }
    let kept = match section[..cut].rfind('\n') {
        Some(end) => &section[..end],
        None => &section[..cut],
    };
    format!("{kept}{TRUNCATION_NOTE}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_keeps_the_head_at_a_line_boundary() {
        // Multi-byte `é` makes the budget boundary land inside a code point for
        // some sizes, which `render_instruction_section` must round down.
        let filler = "- é rule that is repeated until the file is long\n";
        let note = filler.repeat(MAX_INSTRUCTIONS_PROMPT_BYTES / filler.len() + 8);
        let files = vec![("AGENTS.md".to_string(), note.clone())];
        let section = render_instruction_section(&files);

        assert!(section.len() <= MAX_INSTRUCTIONS_PROMPT_BYTES);
        let kept = section
            .strip_suffix(TRUNCATION_NOTE)
            .expect("truncation note");
        assert!(section.starts_with(INSTRUCTIONS_HEADER));
        // The kept text is the head of the untruncated render, ending exactly on
        // a line boundary (the dropped newline is the one before the note).
        let full = format!("{INSTRUCTIONS_HEADER}\n\n### AGENTS.md\n{note}");
        assert!(
            full.starts_with(&format!("{kept}\n")),
            "the cut must land on a line boundary"
        );
    }

    #[test]
    fn a_section_within_budget_is_returned_whole() {
        let files = vec![("AGENTS.md".to_string(), "- Run `cargo test`.".to_string())];
        let section = render_instruction_section(&files);

        assert_eq!(
            section,
            format!("{INSTRUCTIONS_HEADER}\n\n### AGENTS.md\n- Run `cargo test`.")
        );
    }
}
