//! Markdown catalog rendering for the MCP `tools/list` payload.
//!
//! [`ModelManifest::build_tool_description`] turns the manifest into the plain
//! markdown block embedded in the `dispatch` tool description. It lives here,
//! away from manifest *discovery* (`mod.rs`) and *validation* (`validate.rs`),
//! because rendering is the one part of the package with no input other than
//! the catalog itself: it reads no environment variable, no file, and no user
//! configuration.
//!
//! It also owns [`build_system_prompt`], the one place where a role's persistent
//! memory ([`super::memory`]) is spliced into the system prompt a worker starts
//! with, so the implementer and the reviewer build their prompts identically.
//!
//! The render itself is deliberately plain: one header constant plus one
//! `writeln!` per model. It runs once per process (the MCP server precomputes
//! the `tools/list` payload in [`crate::mcp::McpServer::new`]), so there is
//! nothing to memoize and no reason to keep a cache around it.
//!
//! Its one load-bearing property, asserted by the tests in `tests.rs`, is
//! determinism: bullets are emitted in sorted-alias order (via
//! [`ModelManifest::sorted_models`]) rather than in `HashMap` iteration order, so
//! the same `models.yaml` always renders byte-identical text — rendering straight
//! out of the `HashMap` produced up to 24 different strings for the same YAML
//! across 200 parses.

use std::fmt::Write as _;
use std::path::Path;

use super::DEFAULT_ROLE;
use super::memory::memory_prompt_section;
use super::types::ModelManifest;

/// First line of the rendered catalog.
const CATALOG_HEADER: &str = "Available model aliases and their roles:\n";

impl ModelManifest {
    /// Render the catalog advertised through the MCP `tools/list` payload.
    ///
    /// The output is a header line followed by one bullet per model entry, in
    /// alias order, which makes it byte-identical for identical manifests.
    pub fn build_tool_description(&self) -> String {
        let mut desc = String::from(CATALOG_HEADER);
        for (alias, def) in self.sorted_models() {
            let role = def.role.as_deref().unwrap_or(DEFAULT_ROLE);
            let _ = writeln!(desc, "- `{alias}` (id: `{}`): {role}", def.id);
        }
        desc
    }
}

/// Build the effective system prompt for a worker of `model_alias` running in
/// `repo_path`: the crate-wide [`SYSTEM_PROMPT`](crate::agent::SYSTEM_PROMPT)
/// followed by that role's persistent memory, when the repository provides any.
///
/// This is the single point where role memory enters a conversation, so both the
/// implementer loop and the review phase get identical treatment (they previously
/// both passed the static prompt straight to `ChatMessage::text`). Memory is read
/// from disk on every build — never memoized, see [`super::memory`] — so a note
/// appended by a previous run is visible to the very next dispatch.
///
/// Returns the static prompt **unchanged** when the role has no memory file, so a
/// repository that has not opted in sees byte-identical behaviour to before
/// persistent memory existed.
pub fn build_system_prompt(repo_path: &Path, model_alias: &str) -> String {
    match memory_prompt_section(repo_path, model_alias) {
        Some(section) => format!("{}{section}", crate::agent::SYSTEM_PROMPT),
        None => crate::agent::SYSTEM_PROMPT.to_string(),
    }
}
