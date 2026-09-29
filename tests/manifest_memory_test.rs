//! Integration tests for persistent role memory ([`mini_swe_mcp::manifest::memory`]).
//!
//! These go through the public library surface only, i.e. exactly the API the
//! pool runner uses when it builds a worker's system prompt:
//!
//! * [`load_agent_memory`] reads `<repo>/.agents/memory/<alias>.md`.
//! * [`append_agent_memory`] records an atomic takeaway into the same file.
//! * [`build_system_prompt`] splices that memory into the system prompt, and is a
//!   byte-for-byte no-op when the repository has no memory for the role.
//!
//! The behaviour lives in the crate's unit tests (`src/manifest/tests.rs`); this
//! file pins the *public* surface — the names, signatures and visibility the rest
//! of the crate (and any downstream binary) depends on.

use mini_swe_mcp::agent::SYSTEM_PROMPT;
use mini_swe_mcp::manifest::{
    MAX_MEMORY_PROMPT_BYTES, MEMORY_DIR, ModelManifest, agent_memory_path, append_agent_memory,
    build_system_prompt, load_agent_memory,
};
use std::path::{Path, PathBuf};

/// Scratch repository root, removed on drop.
struct ScratchRepo(PathBuf);

impl ScratchRepo {
    fn new(tag: &str) -> Self {
        let path = std::env::temp_dir().join(format!("swe-memory-it-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).expect("scratch repo");
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }

    fn memory_dir(&self) -> PathBuf {
        self.0.join(MEMORY_DIR)
    }
}

impl Drop for ScratchRepo {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn test_load_agent_memory_round_trips_an_appended_takeaway() {
    let repo = ScratchRepo::new("roundtrip");

    // A fresh repository has no memory at all — the loader degrades to `None`
    // instead of failing, so a dispatch never breaks on a missing file.
    assert_eq!(load_agent_memory(repo.path(), "ninja"), None);

    append_agent_memory(
        repo.path(),
        "ninja",
        "Always run `cargo test --all-targets`.",
    )
    .expect("append creates the memory directory and file");

    assert!(repo.memory_dir().join("ninja.md").is_file());
    let memory = load_agent_memory(repo.path(), "ninja").expect("memory is readable back");
    assert!(memory.contains("Always run `cargo test --all-targets`."));

    // Roles are independent files.
    assert_eq!(load_agent_memory(repo.path(), "nerd"), None);
}

#[test]
fn test_build_system_prompt_is_a_no_op_without_memory() {
    let repo = ScratchRepo::new("noop");

    assert_eq!(
        build_system_prompt(repo.path(), "ninja"),
        SYSTEM_PROMPT,
        "a repository without role memory must keep the static prompt byte-identical"
    );
}

#[test]
fn test_build_system_prompt_injects_memory_after_the_static_prompt() {
    let repo = ScratchRepo::new("inject");
    append_agent_memory(repo.path(), "nerd", "Reproduce before you patch.").expect("append");

    let prompt = build_system_prompt(repo.path(), "nerd");
    assert!(
        prompt.starts_with(SYSTEM_PROMPT),
        "the static instructions must come first and stay complete"
    );
    assert!(
        prompt.contains("Reproduce before you patch."),
        "the role memory must reach the prompt"
    );
    assert!(
        !build_system_prompt(repo.path(), "ninja").contains("Reproduce before you patch."),
        "one role's memory must never leak into another's prompt"
    );
}

#[test]
fn test_memory_path_is_traversal_safe() {
    let hostile = agent_memory_path(Path::new("/repo"), "../../etc/passwd").expect("sanitized");
    assert_eq!(
        hostile.parent(),
        Some(Path::new("/repo").join(MEMORY_DIR).as_path()),
        "a hostile alias must stay inside the memory directory"
    );
    assert!(!hostile.to_string_lossy().contains(".."));
    assert_eq!(agent_memory_path(Path::new("/repo"), "///"), None);
}

#[test]
fn test_injected_memory_is_bounded() {
    let repo = ScratchRepo::new("bounded");
    std::fs::create_dir_all(repo.memory_dir()).expect("memory dir");
    let filler = "- a takeaway that is repeated until the memory file is large\n";
    let note = filler.repeat(MAX_MEMORY_PROMPT_BYTES / filler.len() + 8);
    std::fs::write(
        agent_memory_path(repo.path(), "ninja").expect("path"),
        &note,
    )
    .expect("fixture written");

    let memory = load_agent_memory(repo.path(), "ninja").expect("memory is loaded");
    assert!(
        memory.len() <= MAX_MEMORY_PROMPT_BYTES,
        "memory injected into a prompt must stay bounded, got {} bytes",
        memory.len()
    );
}

#[test]
fn test_alias_for_model_resolves_the_resolved_id_to_its_alias() {
    let manifest = ModelManifest::default();

    // The pool carries the resolved id; memory files are keyed by alias.
    assert_eq!(manifest.alias_for_model("combo:ninja"), "ninja");
    assert_eq!(manifest.alias_for_model("combo:nerd"), "nerd");
    assert_eq!(manifest.alias_for_model("ninja"), "ninja");
    assert_eq!(manifest.alias_for_model("vendor:unknown"), "vendor:unknown");
}
