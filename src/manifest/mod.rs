//! Model catalog served to MCP hosts, loaded from a `models.yaml`.
//!
//! Split by responsibility while keeping the historical
//! `mini_swe_mcp::manifest::*` surface identical through the re-exports below:
//!
//! * `types` — the serializable [`ModelManifest`] / [`ModelDefinition`] pair
//!   and the process-wide constants that bound them.
//! * `cache` — the bounded, process-wide memoization of rendered catalog
//!   bullets.
//! * `catalog` — the markdown catalog rendering used by the MCP `tools/list`
//!   payload ([`ModelManifest::build_tool_description`]).
//! * `memory` — the persistent per-role memory (`.agents/memory/<alias>.md`)
//!   loaded into a worker's system prompt.
//! * `rules` — the accept/reject rules for the optional declarative execution
//!   policy (`policy.network`) and the fixups that repair them.
//! * `validate` — the advisory warning rules and the fixups that repair what
//!   they report.
//! * `tests` — the package unit tests (compiled only under `cfg(test)`).
//!
//! [`ModelManifest`] itself stays here: it owns manifest discovery and
//! loading ([`ModelManifest::load`], [`ModelManifest::from_path`]) and id
//! resolution ([`ModelManifest::resolve_model`]).

use std::env;
use std::path::{Path, PathBuf};
use tracing::{error, info, warn};

use crate::config::xdg_config_dir;

mod cache;
mod catalog;
mod memory;
mod rules;
mod types;
mod validate;

#[cfg(test)]
mod tests;

pub use self::cache::{CATALOG_CACHE_CAPACITY, catalog_cache_len, clear_catalog_cache};
pub use self::catalog::build_system_prompt;
pub use self::memory::{
    MAX_MEMORY_PROMPT_BYTES, MEMORY_DIR, agent_memory_path, append_agent_memory, load_agent_memory,
};
pub use self::types::{
    BUILTIN_DEFAULT_MODEL, DEFAULT_MAX_TURNS, ExecutionPolicy, MAX_TURNS_LIMIT,
    ModelDefinition, ModelManifest, NETWORK_POLICIES, NetworkPolicy, TEMPERATURE_RANGE,
};

/// Role shown for a model that declares none.
const DEFAULT_ROLE: &str = "Autonomous subagent";

impl ModelManifest {
    pub fn load() -> Self {
        let mut candidates = Vec::with_capacity(4);

        // 1. Explicit environment variable MODELS_FILE
        if let Ok(path) = env::var("MODELS_FILE") {
            candidates.push((PathBuf::from(path), "MODELS_FILE"));
        }

        // 2. Local working directory models.yaml
        candidates.push((PathBuf::from("models.yaml"), "current directory"));

        // 3. Standard XDG config directory (~/.config/mini-swe/models.yaml)
        if let Some(dir) = xdg_config_dir() {
            candidates.push((
                dir.join("mini-swe").join("models.yaml"),
                "XDG config directory",
            ));
        }

        // 4. Alongside the executable
        if let Ok(exe) = env::current_exe()
            && let Some(parent) = exe.parent()
        {
            candidates.push((parent.join("models.yaml"), "executable directory"));
        }

        for (path, source) in candidates {
            if let Some(manifest) = Self::from_candidate(&path, source) {
                return manifest;
            }
        }

        info!("No models.yaml found; using default built-in manifest (ninja & nerd)");
        Self::default().normalized()
    }

    /// Parse and normalize a manifest from an explicit `models.yaml` path.
    ///
    /// Unlike [`ModelManifest::load`] this does not consult the environment or
    /// fall back to the built-in catalog: a caller that names a file wants that
    /// file (or a hard error), which is what makes it testable without mutating
    /// process-wide state. Warnings are logged exactly as they are on the
    /// discovery path, and the returned manifest is always normalized.
    pub fn from_path(path: &Path) -> anyhow::Result<Self> {
        let manifest = Self::from_file(path)?;
        for warning in manifest.validate() {
            warn!(path = %path.display(), "Model manifest warning: {warning}");
        }
        Ok(manifest.normalized())
    }

    fn from_candidate(path: &Path, source: &str) -> Option<Self> {
        if !path.exists() {
            return None;
        }

        match Self::from_file(path) {
            Ok(manifest) => {
                info!(path = %path.display(), "Loaded model manifest from {source}");
                for warning in manifest.validate() {
                    warn!(path = %path.display(), "Model manifest warning: {warning}");
                }
                // Validation is advisory, so fixups are applied here: a manifest
                // with warnings is still served, but as a working configuration.
                Some(manifest.normalized())
            }
            Err(e) => {
                error!(error = %e, path = %path.display(), "Failed to parse models.yaml from {source}");
                None
            }
        }
    }

    fn from_file(path: &Path) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let manifest: Self = serde_yaml::from_str(&content)?;
        warn_fs_policy(&content);
        Ok(manifest)
    }

    /// Resolve an alias **or** a full model id to `(id, temperature, max_turns)`.
    ///
    /// Alias lookup is an O(1) `HashMap` hit; a hit always wins over the id
    /// fallback so a manifest can never be shadowed by pass-through.
    ///
    /// Duplicate-`id` policy: when several aliases share one `id`, the
    /// **first alias in lexicographic (sorted) order wins**. `self.models` is a
    /// `HashMap`, whose iteration order depends on `RandomState` and therefore
    /// differs between instances, so the previous `values().find(..)` fallback
    /// returned a *different* definition for the same id on every parse. The
    /// tie-break is now stable, and [`ModelManifest::validate`] reports the
    /// collision as a non-fatal warning.
    pub fn resolve_model(&self, requested: &str) -> (String, Option<f32>, Option<usize>) {
        if let Some(def) = self.models.get(requested) {
            return (def.id.clone(), def.temperature, def.max_turns);
        }

        self.sorted_models()
            .iter()
            .find(|(_, def)| def.id == requested)
            .map_or_else(
                || (requested.to_string(), None, None),
                |(_, def)| (def.id.clone(), def.temperature, def.max_turns),
            )
    }

    /// Resolve the *alias* that owns `model`, whether `model` is already an alias
    /// or a full model id.
    ///
    /// The worker pool only carries the resolved id (see [`Self::resolve_model`]),
    /// but role memory is keyed by alias (`.agents/memory/<alias>.md`), so this is
    /// the bridge between the two. An alias hit always wins; otherwise the first
    /// alias (in sorted order) whose `id` matches is returned. Unknown models fall
    /// back to the input unchanged, so a pass-through id like `some/unknown` is
    /// still looked up verbatim as a slug and simply finds no file.
    pub fn alias_for_model(&self, model: &str) -> String {
        if self.models.contains_key(model) {
            return model.to_string();
        }
        self.sorted_models()
            .iter()
            .find(|(_, def)| def.id == model)
            .map_or_else(|| model.to_string(), |(alias, _)| (*alias).to_string())
    }

    /// Model entries in alias order.
    ///
    /// `self.models` is a `HashMap` with a randomly seeded `RandomState`, so its
    /// iteration order differs between processes *and* between instances built
    /// from the same YAML. Every user-visible derivation (catalog rendering,
    /// id resolution, warnings) goes through this helper so the output is
    /// reproducible.
    fn sorted_models(&self) -> Vec<(&str, &ModelDefinition)> {
        let mut entries: Vec<(&str, &ModelDefinition)> = self
            .models
            .iter()
            .map(|(alias, def)| (alias.as_str(), def))
            .collect();
        entries.sort_unstable_by_key(|(alias, _)| *alias);
        entries
    }
}

/// Warn once per `fs:` key in a `models.yaml`, since the filesystem policy is
/// not enforced at runtime and advertising it would be misleading.
fn warn_fs_policy(content: &str) {
    let Ok(value) = serde_yaml::from_str::<serde_yaml::Value>(content) else {
        return;
    };
    let Some(models) = value.get("models").and_then(|m| m.as_mapping()) else {
        return;
    };
    for (alias, def) in models {
        let Some(policy) = def.get("policy").and_then(|p| p.as_mapping()) else {
            continue;
        };
        if policy.contains_key(serde_yaml::Value::String("fs".to_string())) {
            warn!(
                alias = %alias.as_str().unwrap_or("?"),
                "models.yaml policy.fs is not supported and is ignored"
            );
        }
    }
}
