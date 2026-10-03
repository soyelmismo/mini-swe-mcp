//! Markdown catalog rendering for the MCP `tools/list` payload.
//!
//! [`ModelManifest::build_tool_description`] turns the manifest into the plain
//! markdown block embedded in the `dispatch` tool description. It lives here,
//! away from manifest *discovery* (`mod.rs`) and *validation* (`validate.rs`),
//! because rendering is the one part of the package with no input other than
//! the catalog itself: it reads no environment variable, no file, and no user
//! configuration.
//!
//! It also owns [`build_system_prompt`], the one place where a repository's own
//! instruction files ([`super::instructions`]), a role's persistent memory
//! ([`super::memory`]) and a model's own `instructions:` block are spliced into
//! the system prompt a worker starts with, so the implementer and the reviewer
//! build their prompts identically.
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
use super::instructions::instructions_prompt_section;
use super::memory::memory_prompt_section;
use super::types::ModelManifest;

/// First line of the rendered catalog.
const CATALOG_HEADER: &str = "Available model aliases and their roles:\n";

/// Header of the per-model instructions appended to a worker's system prompt.
pub(super) const MODEL_INSTRUCTIONS_HEADER: &str =
    "Model-specific instructions (declared for this model in models.yaml):";

/// Note appended when a model's instructions did not fit their budget, so a
/// cut block is never mistaken for the whole one.
pub(super) const MODEL_INSTRUCTIONS_TRUNCATION_NOTE: &str =
    "[truncated: this model's instructions exceed the catalog budget]";

/// Prefix of each rendered instruction, shared with the byte budget in
/// [`super::types::BULLET_PREFIX`] so the two cannot drift.
const INSTRUCTION_BULLET: &str = super::types::BULLET_PREFIX;

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
/// followed by the repository's instruction files, that role's persistent
/// memory and that model's own `instructions:` block, each when it exists.
///
/// This is the single point where repository- and role-scoped text enters a
/// conversation, so both the implementer loop and the review phase get identical
/// treatment (they previously both passed the static prompt straight to
/// `ChatMessage::text`). Both sources are read from disk on every build — never
/// memoized, see `super::instructions` and `super::memory` — so an edit made
/// between dispatches is visible to the very next one.
///
/// Returns the static prompt **unchanged** when the repository has no
/// instruction files, the role has no memory file and the model declares no
/// instructions, so an unadorned repository sees byte-identical behaviour to
/// before these injections existed.
///
/// The model passed as `model_alias` is the one whose `instructions:` block is
/// appended, which is what makes the review phase carry the *reviewer's*
/// habits rather than the implementer's: the review phase resolves its own
/// reviewer alias before calling here.
pub fn build_system_prompt(
    manifest: &ModelManifest,
    repo_path: &Path,
    model_alias: &str,
) -> String {
    let mut prompt = String::from(crate::agent::SYSTEM_PROMPT);
    if let Some(section) = instructions_prompt_section(repo_path) {
        prompt.push_str(&section);
    }
    if let Some(section) = memory_prompt_section(repo_path, model_alias) {
        prompt.push_str(&section);
    }
    if let Some(section) = model_instructions_prompt_section(manifest, model_alias) {
        prompt.push_str(&section);
    }
    prompt
}

/// Render the system-prompt section for the per-model `instructions:` block, or
/// `None` when `model_alias` declares none.
///
/// Last of the three injections, so the model-specific rules read as the most
/// specific thing in the prompt: repository instructions, then role memory, then
/// the habits this model in particular has to correct. `None` (rather than an
/// empty section) keeps the prompt byte-identical for every catalog written
/// before per-model instructions existed.
fn model_instructions_prompt_section(
    manifest: &ModelManifest,
    model_alias: &str,
) -> Option<String> {
    let block = manifest.instructions_for(model_alias)?;
    if block.is_empty() {
        return None;
    }

    let mut section = format!("\n\n{MODEL_INSTRUCTIONS_HEADER}\n");
    for entry in block.entries() {
        section.push_str(INSTRUCTION_BULLET);
        section.push_str(entry);
        section.push('\n');
    }
    if block.is_truncated() {
        section.push_str(MODEL_INSTRUCTIONS_TRUNCATION_NOTE);
        section.push('\n');
    }
    Some(section)
}
