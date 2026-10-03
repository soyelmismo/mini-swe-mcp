//! Integration tests for repository instruction files
//! ([`mini_swe_mcp::manifest::instructions`]).
//!
//! These go through the public library surface only, i.e. exactly the API the
//! pool runner uses when it builds a worker's system prompt:
//!
//! * [`build_system_prompt`] splices the repository's `AGENTS.md` / `CLAUDE.md`
//!   / … into the system prompt, each under a header naming the file.
//! * It is a byte-for-byte no-op for a repository without any of those files.
//! * The injected text is bounded and symlinks out of the repository are never
//!   followed.
//!
//! The rendering lives in the crate's unit tests (`src/manifest/instructions.rs`);
//! this file pins the *public* surface — the names, signatures and visibility the
//! rest of the crate (and any downstream binary) depends on.

use mini_swe_mcp::agent::SYSTEM_PROMPT;
use mini_swe_mcp::manifest::{
    INSTRUCTION_FILES, MAX_INSTRUCTIONS_PROMPT_BYTES, ModelManifest, build_system_prompt,
};
use std::path::{Path, PathBuf};

/// Scratch directory, removed on drop; doubles as a repository root.
struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        let path =
            std::env::temp_dir().join(format!("swe-instructions-it-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).expect("scratch dir");
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }

    fn write(&self, name: &str, content: &str) {
        let path = self.0.join(name);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("parent dir");
        }
        std::fs::write(path, content).expect("fixture written");
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn test_agents_and_claude_are_both_injected_with_headers() {
    let repo = Scratch::new("both");
    repo.write("AGENTS.md", "- Run the gates before reporting.\n");
    repo.write("CLAUDE.md", "- Prefer the smallest edit that works.\n");

    let prompt = build_system_prompt(&ModelManifest::default(), repo.path(), "ninja");

    assert!(
        prompt.starts_with(SYSTEM_PROMPT),
        "the static instructions must come first and stay complete"
    );
    assert!(
        prompt.contains("### AGENTS.md"),
        "AGENTS.md must be named in a header: {prompt}"
    );
    assert!(
        prompt.contains("### CLAUDE.md"),
        "CLAUDE.md must be named in a header: {prompt}"
    );
    assert!(
        prompt.contains("Run the gates before reporting."),
        "AGENTS.md contents must reach the prompt: {prompt}"
    );
    assert!(
        prompt.contains("Prefer the smallest edit that works."),
        "CLAUDE.md contents must reach the prompt: {prompt}"
    );
}

#[test]
fn test_byte_identical_instruction_files_are_injected_once() {
    let repo = Scratch::new("dedup");
    let shared = "- One copy only.\n";
    repo.write("AGENTS.md", shared);
    repo.write("CLAUDE.md", shared);

    let prompt = build_system_prompt(&ModelManifest::default(), repo.path(), "ninja");

    assert_eq!(
        prompt.matches("### AGENTS.md").count(),
        1,
        "the preferred file must be injected"
    );
    assert_eq!(
        prompt.matches("### CLAUDE.md").count(),
        0,
        "a byte-identical duplicate must be dropped"
    );
    assert_eq!(prompt.matches("- One copy only.").count(), 1);
}

#[test]
fn test_injected_instructions_are_bounded() {
    let repo = Scratch::new("bounded");
    let filler = "- a rule that is repeated until the instruction file is long\n";
    let note = filler.repeat(MAX_INSTRUCTIONS_PROMPT_BYTES / filler.len() + 8);
    repo.write("AGENTS.md", &note);

    let prompt = build_system_prompt(&ModelManifest::default(), repo.path(), "ninja");
    let injected = &prompt[SYSTEM_PROMPT.len()..];

    assert!(
        injected.len() <= MAX_INSTRUCTIONS_PROMPT_BYTES,
        "injected instructions must stay bounded, got {} bytes",
        injected.len()
    );
    assert!(
        injected.contains("[truncated"),
        "a truncated block must say so: {injected}"
    );
    assert!(injected.contains("### AGENTS.md"));
}

#[test]
fn test_no_instruction_files_leaves_the_prompt_unchanged() {
    let repo = Scratch::new("none");

    assert_eq!(
        build_system_prompt(&ModelManifest::default(), repo.path(), "ninja"),
        SYSTEM_PROMPT,
        "a repository without instruction files must keep the static prompt byte-identical"
    );
}

#[test]
fn test_symlink_out_of_the_repository_is_ignored() {
    let outside = Scratch::new("outside");
    outside.write(
        "secret.md",
        "- SECRET: never inject text from outside the repo.\n",
    );

    let repo = Scratch::new("symlink");
    std::os::unix::fs::symlink(
        outside.path().join("secret.md"),
        repo.path().join("AGENTS.md"),
    )
    .expect("symlink fixture");

    // A real file in the same repository proves the feature is working while the
    // symlink is being refused.
    repo.write("CLAUDE.md", "- In-repo rule.\n");

    let prompt = build_system_prompt(&ModelManifest::default(), repo.path(), "ninja");

    assert!(
        prompt.contains("- In-repo rule."),
        "the regular instruction file must still be injected: {prompt}"
    );
    assert!(
        !prompt.contains("SECRET"),
        "a symlink pointing outside the repository must never be followed: {prompt}"
    );
}

#[test]
fn test_recognised_files_are_the_standard_agent_instructions() {
    for expected in [
        "AGENTS.md",
        "CLAUDE.md",
        "GEMINI.md",
        ".github/copilot-instructions.md",
        ".cursorrules",
    ] {
        assert!(
            INSTRUCTION_FILES.contains(&expected),
            "{expected} must be a recognised instruction file"
        );
    }
}
