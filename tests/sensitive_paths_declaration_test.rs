//! The repository's own `## Sensitive paths` declaration must cover the file
//! that declares it.
//!
//! `AGENTS.md`'s `## Sensitive paths` section is what decides which diffs get
//! the automatic adversarial security review, so a worker that edits it could
//! quietly shrink the list for every later round. The declaration therefore
//! names itself -- and `CLAUDE.md`, which the parser reads as a second
//! candidate -- and the matcher has to agree, so a diff that touches either
//! file is security-reviewed whatever a later edit does to the prose.
//!
//! The declaration is read from the crate root the test binary was compiled
//! from, and never written to.

use std::path::PathBuf;

use mini_swe_mcp::manifest::{matches_sensitive, parse_sensitive_paths};

/// The repository root the test binary was compiled from.
fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// The globs the repository's `AGENTS.md` declares.
fn declared() -> Vec<String> {
    let agents = std::fs::read_to_string(repo_root().join("AGENTS.md"))
        .expect("read the repository's AGENTS.md");
    parse_sensitive_paths(&agents)
}

#[test]
fn the_declaration_names_itself() {
    let globs = declared();
    assert!(
        globs.iter().any(|glob| glob == "AGENTS.md"),
        "AGENTS.md must declare itself sensitive, got {globs:?}"
    );
    assert!(
        globs.iter().any(|glob| glob == "CLAUDE.md"),
        "the parser reads CLAUDE.md as a second candidate, so it must be declared too, got {globs:?}"
    );
}

#[test]
fn a_diff_touching_the_declaration_is_security_reviewed() {
    let globs = declared();
    for path in ["AGENTS.md", "CLAUDE.md"] {
        assert!(
            matches_sensitive(path, &globs),
            "a diff touching {path} must match a declared glob"
        );
    }
    // The surfaces the declaration already named stay declared: adding a path
    // must not have displaced one.
    for path in [
        "src/hub/mod.rs",
        "src/agent/sandbox.rs",
        "src/pool/merge.rs",
    ] {
        assert!(matches_sensitive(path, &globs), "{path} must stay declared");
    }
}
