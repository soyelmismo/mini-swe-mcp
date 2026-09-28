use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::env;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, RwLock};
use tracing::{error, info, warn};

use crate::config::xdg_config_dir;

/// Role shown for a model that declares none.
const DEFAULT_ROLE: &str = "Autonomous subagent";

/// First line of the rendered catalog.
const CATALOG_HEADER: &str = "Available model aliases and their roles:\n";

/// Hard bound on the process-wide catalog row cache.
///
/// A manifest is a single, immutable, `Arc`-shared value in practice, so the
/// cache holds at most a handful of entries. The cap is defense in depth: it
/// keeps a pathological caller (many synthetic manifests, e.g. in tests) from
/// growing the process without bound. Public so tests can exercise the bound.
pub const CATALOG_CACHE_CAPACITY: usize = 256;

/// Identity of one rendered catalog bullet: everything a bullet depends on.
type CatalogRowKey = (String, String, String);

/// Process-wide memoization of the rendered catalog bullets.
///
/// [`ModelManifest::build_tool_description`] is re-rendered on every
/// `tools/list` request, and rendering a bullet allocates and formats a
/// `String`. The rendered bullet is a pure function of `(alias, id, role)`, so
/// it is cached here rather than inside [`ModelManifest`]: the struct stays
/// `&self`-clean, so its `Clone`/`Debug`/`Serialize` derives are untouched (no
/// `serde(skip)` plumbing, no interior mutability leaking into `Arc` clones).
static CATALOG_ROW_CACHE: LazyLock<RwLock<HashMap<CatalogRowKey, Arc<str>>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

/// Render (or reuse) the bullet for one model entry.
///
/// Cache misses build the row with `writeln!` on a single pre-sized allocation
/// instead of `format!` plus `push_str`; hits clone an `Arc<str>` instead of the
/// whole catalog `String`, which is what makes repeated `tools/list` calls cheap.
fn catalog_row(alias: &str, def: &ModelDefinition) -> String {
    let key = (
        alias.to_string(),
        def.id.clone(),
        def.role.clone().unwrap_or_default(),
    );

    if let Ok(cache) = CATALOG_ROW_CACHE.read()
        && let Some(row) = cache.get(&key)
    {
        return row.to_string();
    }

    let role = def.role.as_deref().unwrap_or(DEFAULT_ROLE);
    let mut row = String::with_capacity(48 + alias.len() + def.id.len() + role.len());
    let _ = writeln!(row, "- `{alias}` (id: `{}`): {role}", def.id);

    if let Ok(mut cache) = CATALOG_ROW_CACHE.write() {
        if cache.len() >= CATALOG_CACHE_CAPACITY {
            cache.clear();
        }
        cache.insert(key, Arc::from(row.as_str()));
    }

    row
}

/// Render aliases as `"a", "b", "c"` for a warning message.
fn quote_list(aliases: &[&str]) -> String {
    aliases
        .iter()
        .map(|alias| format!("\"{alias}\""))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Number of catalog bullets currently memoized.
///
/// Exposed for tests and diagnostics: the cache is bounded by
/// [`CATALOG_CACHE_CAPACITY`] and never grows with repeated `tools/list` calls.
pub fn catalog_cache_len() -> usize {
    CATALOG_ROW_CACHE
        .read()
        .map(|cache| cache.len())
        .unwrap_or_default()
}

/// Drop every memoized catalog bullet.
///
/// Exposed for tests and diagnostics; never needed on a serving path.
pub fn clear_catalog_cache() {
    if let Ok(mut cache) = CATALOG_ROW_CACHE.write() {
        cache.clear();
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDefinition {
    pub id: String,
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub max_turns: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelManifest {
    #[serde(default)]
    pub default: Option<String>,
    #[serde(default)]
    pub models: HashMap<String, ModelDefinition>,
}

impl Default for ModelManifest {
    fn default() -> Self {
        let mut models = HashMap::new();
        models.insert(
            "ninja".to_string(),
            ModelDefinition {
                id: "combo:ninja".to_string(),
                role: Some("Fast executor. Use for exploration, test runs, syntax fixes, and focused edits.".to_string()),
                temperature: Some(0.2),
                max_turns: Some(100),
            },
        );
        models.insert(
            "nerd".to_string(),
            ModelDefinition {
                id: "combo:nerd".to_string(),
                role: Some("Deep reasoner. Use for hard debugging, complex architecture, and multi-file refactors.".to_string()),
                temperature: Some(0.6),
                max_turns: Some(100),
            },
        );

        Self {
            default: Some("ninja".to_string()),
            models,
        }
    }
}

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
        Self::default()
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
                Some(manifest)
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

    /// Render the catalog advertised through the MCP `tools/list` payload.
    ///
    /// Bullets are emitted in alias order, which makes the output byte-identical
    /// for identical manifests (rendering straight out of the `HashMap` produced
    /// up to 24 different strings for the same YAML across 200 parses), and each
    /// bullet is memoized process-wide (see [`catalog_row`]).
    pub fn build_tool_description(&self) -> String {
        let entries = self.sorted_models();

        let total = CATALOG_HEADER.len()
            + entries
                .iter()
                .map(|(alias, def)| {
                    48 + alias.len() + def.id.len() + def.role.as_deref().map_or(0, str::len)
                })
                .sum::<usize>();
        let mut desc = String::with_capacity(total);
        desc.push_str(CATALOG_HEADER);

        for (alias, def) in entries {
            desc.push_str(&catalog_row(alias, def));
        }

        desc
    }

    /// Collect human-readable warnings about suspicious manifest entries.
    ///
    /// Validation is deliberately non-fatal: a manifest with warnings is still
    /// served so that a typo in `models.yaml` degrades gracefully instead of
    /// taking the server down. Callers surface the returned strings as warnings
    /// (see [`ModelManifest::from_candidate`]).
    pub fn validate(&self) -> Vec<String> {
        let mut warnings = Vec::new();

        if let Some(name) = &self.default
            && !self.models.contains_key(name)
        {
            warnings.push(format!("default model \"{name}\" not found in models"));
        }

        // Duplicate ids are ambiguous for id-based resolution. The policy in
        // `resolve_model` is first-alias-wins (sorted by alias), so the
        // resolution is stable, but the manifest is still ambiguous and the user
        // should know. Non-fatal by design, like every other warning here.
        let mut by_id: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
        for (alias, def) in self.sorted_models() {
            by_id.entry(def.id.as_str()).or_default().push(alias);
        }
        for (id, aliases) in &by_id {
            if aliases.len() > 1 {
                warnings.push(format!(
                    "duplicate model id \"{id}\" shared by aliases {}; resolving the full id returns the first alias",
                    quote_list(aliases)
                ));
            }
        }

        for (alias, def) in self.sorted_models() {
            if def.id.trim().is_empty() {
                warnings.push(format!("model \"{alias}\": id cannot be empty"));
            }

            if let Some(t) = def.temperature
                && (!(0.0..=2.0).contains(&t) || t.is_nan())
            {
                warnings.push(format!(
                    "model \"{alias}\": temperature {t} must be between 0.0 and 2.0"
                ));
            }

            if def.max_turns == Some(0) {
                warnings.push(format!(
                    "model \"{alias}\": max_turns must be greater than 0"
                ));
            }
        }

        warnings
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CATALOG_CACHE_CAPACITY, ModelDefinition, ModelManifest, catalog_cache_len, catalog_row,
        clear_catalog_cache,
    };

    #[test]
    fn test_default_manifest() {
        let manifest = ModelManifest::default();

        assert_eq!(manifest.default, Some("ninja".to_string()));

        let ninja = manifest
            .models
            .get("ninja")
            .expect("default manifest must contain the `ninja` model");
        assert_eq!(ninja.id, "combo:ninja");
        assert!(ninja.role.as_deref().is_some_and(|r| !r.is_empty()));
        assert_eq!(ninja.temperature, Some(0.2));
        assert_eq!(ninja.max_turns, Some(100));

        let nerd = manifest
            .models
            .get("nerd")
            .expect("default manifest must contain the `nerd` model");
        assert_eq!(nerd.id, "combo:nerd");
        assert!(nerd.role.as_deref().is_some_and(|r| !r.is_empty()));
        assert_eq!(nerd.temperature, Some(0.6));
        assert_eq!(nerd.max_turns, Some(100));

        assert_eq!(manifest.models.len(), 2);
    }

    #[test]
    fn test_resolve_model() {
        let manifest = ModelManifest::default();

        // Resolution by alias name
        assert_eq!(
            manifest.resolve_model("ninja"),
            ("combo:ninja".to_string(), Some(0.2), Some(100))
        );
        assert_eq!(
            manifest.resolve_model("nerd"),
            ("combo:nerd".to_string(), Some(0.6), Some(100))
        );

        // Resolution by full model id
        assert_eq!(
            manifest.resolve_model("combo:nerd"),
            ("combo:nerd".to_string(), Some(0.6), Some(100))
        );

        // Fallback: unknown models are passed through untouched
        assert_eq!(
            manifest.resolve_model("some/unknown-model"),
            ("some/unknown-model".to_string(), None, None)
        );

        // Empty request
        assert_eq!(manifest.resolve_model(""), (String::new(), None, None));

        // Model without overrides
        let mut models = std::collections::HashMap::new();
        models.insert(
            "plain".to_string(),
            ModelDefinition {
                id: "vendor:plain".to_string(),
                role: None,
                temperature: None,
                max_turns: None,
            },
        );
        let sparse = ModelManifest {
            default: None,
            models,
        };
        assert_eq!(
            sparse.resolve_model("plain"),
            ("vendor:plain".to_string(), None, None)
        );
    }

    #[test]
    fn test_tool_description() {
        let manifest = ModelManifest::default();
        let desc = manifest.build_tool_description();

        let mut lines = desc.lines();
        assert_eq!(
            lines.next(),
            Some("Available model aliases and their roles:")
        );

        let body: Vec<&str> = lines.collect();
        assert_eq!(body.len(), 2);

        let find = |alias: &str| {
            body.iter()
                .find(|line| line.starts_with(&format!("- `{alias}`")))
                .unwrap_or_else(|| panic!("missing bullet for alias `{alias}`"))
        };

        let ninja = find("ninja");
        assert!(ninja.contains("combo:ninja"));

        let nerd = find("nerd");
        assert!(nerd.contains("combo:nerd"));
    }

    #[test]
    fn test_tool_description_role_fallback() {
        let mut models = std::collections::HashMap::new();
        models.insert(
            "bare".to_string(),
            ModelDefinition {
                id: "vendor:bare".to_string(),
                role: None,
                temperature: Some(0.9),
                max_turns: Some(7),
            },
        );
        let manifest = ModelManifest {
            default: None,
            models,
        };

        assert_eq!(
            manifest.build_tool_description(),
            "Available model aliases and their roles:\n\
             - `bare` (id: `vendor:bare`): Autonomous subagent\n"
        );
    }

    #[test]
    fn test_catalog_row_is_memoized_and_keyed_on_every_input() {
        clear_catalog_cache();

        let a = ModelDefinition {
            id: "vendor:a".to_string(),
            role: Some("Role A.".to_string()),
            temperature: None,
            max_turns: None,
        };
        let b = ModelDefinition {
            id: "vendor:b".to_string(),
            role: Some("Role A.".to_string()),
            temperature: None,
            max_turns: None,
        };
        let no_role = ModelDefinition {
            id: "vendor:a".to_string(),
            role: None,
            temperature: None,
            max_turns: None,
        };

        assert_eq!(catalog_row("a", &a), "- `a` (id: `vendor:a`): Role A.\n");
        assert_eq!(catalog_cache_len(), 1);

        // Same key twice -> memoized, no new entry.
        assert_eq!(catalog_row("a", &a), "- `a` (id: `vendor:a`): Role A.\n");
        assert_eq!(catalog_cache_len(), 1);

        // Different id, different role and a missing role are all distinct keys.
        assert_eq!(catalog_row("a", &b), "- `a` (id: `vendor:b`): Role A.\n");
        assert_eq!(
            catalog_row("a", &no_role),
            "- `a` (id: `vendor:a`): Autonomous subagent\n"
        );
        assert_eq!(catalog_cache_len(), 3);
    }

    #[test]
    fn test_catalog_cache_stays_bounded() {
        clear_catalog_cache();

        for i in 0..(CATALOG_CACHE_CAPACITY + 8) {
            let def = ModelDefinition {
                id: format!("vendor:id{i}"),
                role: Some("Role.".to_string()),
                temperature: None,
                max_turns: None,
            };
            let row = catalog_row(&format!("alias{i}"), &def);
            assert_eq!(row, format!("- `alias{i}` (id: `vendor:id{i}`): Role.\n"));
            assert!(
                catalog_cache_len() <= CATALOG_CACHE_CAPACITY,
                "catalog cache must stay bounded, got {}",
                catalog_cache_len()
            );
        }
    }

    #[test]
    fn test_resolve_model_duplicate_id_uses_first_alias_in_sorted_order() {
        let mut models = std::collections::HashMap::new();
        models.insert(
            "z:shared".to_string(),
            ModelDefinition {
                id: "vendor:shared".to_string(),
                role: None,
                temperature: Some(0.9),
                max_turns: Some(9),
            },
        );
        models.insert(
            "a:shared".to_string(),
            ModelDefinition {
                id: "vendor:shared".to_string(),
                role: None,
                temperature: Some(0.1),
                max_turns: Some(1),
            },
        );
        let manifest = ModelManifest {
            default: None,
            models,
        };

        // "a:shared" sorts first, so it wins regardless of HashMap order.
        for _ in 0..200 {
            assert_eq!(
                manifest.resolve_model("vendor:shared"),
                ("vendor:shared".to_string(), Some(0.1), Some(1))
            );
        }

        // Alias hits always win over the id fallback.
        assert_eq!(
            manifest.resolve_model("z:shared"),
            ("vendor:shared".to_string(), Some(0.9), Some(9))
        );
    }

    #[test]
    fn test_validate_flags_duplicate_model_ids() {
        let mut models = std::collections::HashMap::new();
        for (alias, temperature) in [("a", 0.1), ("b", 0.9), ("c", 0.5)] {
            models.insert(
                alias.to_string(),
                ModelDefinition {
                    id: "vendor:shared".to_string(),
                    role: None,
                    temperature: Some(temperature),
                    max_turns: None,
                },
            );
        }
        let manifest = ModelManifest {
            default: None,
            models,
        };

        assert_eq!(
            manifest.validate(),
            vec![
                "duplicate model id \"vendor:shared\" shared by aliases \"a\", \"b\", \"c\"; resolving the full id returns the first alias"
                    .to_string()
            ]
        );
    }

    #[test]
    fn test_tool_description_is_reproducible_and_cached() {
        clear_catalog_cache();
        let manifest = ModelManifest::default();

        let first = manifest.build_tool_description();
        assert_eq!(catalog_cache_len(), 2);
        assert_eq!(manifest.build_tool_description(), first);
        assert_eq!(
            catalog_cache_len(),
            2,
            "repeat calls must not grow the cache"
        );

        // Bullets are sorted by alias, not by HashMap iteration order.
        let aliases: Vec<&str> = first
            .lines()
            .skip(1)
            .filter_map(|line| line.split('`').nth(1))
            .collect();
        assert_eq!(aliases, vec!["nerd", "ninja"]);
    }
}
