//! Model catalog served to MCP hosts, loaded from a `models.yaml`.
//!
//! The package is split by responsibility while keeping the historical
//! `mini_swe_mcp::manifest::*` surface identical through the re-exports below:
//!
//! * `types` — the serializable [`ModelManifest`] / [`ModelDefinition`] pair
//!   and the process-wide constants that bound them.
//! * `cache` — the bounded, process-wide memoization of rendered catalog
//!   bullets.
//! * `validate` — the advisory warning rules and the fixups that repair what
//!   they report.
//!
//! [`ModelManifest`] itself stays here: it owns manifest discovery and
//! loading ([`ModelManifest::load`], [`ModelManifest::from_path`]), id
//! resolution ([`ModelManifest::resolve_model`]) and catalog rendering
//! ([`ModelManifest::build_tool_description`]), the three entry points the rest
//! of the crate uses.

use std::env;
use std::path::{Path, PathBuf};
use tracing::{error, info, warn};

use crate::config::xdg_config_dir;

mod cache;
mod types;
mod validate;

pub use self::cache::{catalog_cache_len, clear_catalog_cache, CATALOG_CACHE_CAPACITY};
pub use self::types::{
    ModelDefinition, ModelManifest, BUILTIN_DEFAULT_MODEL, DEFAULT_MAX_TURNS, MAX_TURNS_LIMIT,
    TEMPERATURE_RANGE,
};
use self::cache::{CATALOG_HEADER, CATALOG_ROW_OVERHEAD, catalog_row};

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
                // Validation is advisory, so the fixups are applied here: a
                // manifest with warnings is still served, but it serves a
                // working configuration (see [`ModelManifest::normalize`]).
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
    /// bullet is memoized process-wide (see `cache::catalog_row`).
    pub fn build_tool_description(&self) -> String {
        let entries = self.sorted_models();

        let total = CATALOG_HEADER.len()
            + entries
                .iter()
                .map(|(alias, def)| {
                    CATALOG_ROW_OVERHEAD
                        + alias.len()
                        + def.id.len()
                        + def.role.as_deref().map_or(0, str::len)
                })
                .sum::<usize>();
        let mut desc = String::with_capacity(total);
        desc.push_str(CATALOG_HEADER);

        for (alias, def) in entries {
            desc.push_str(&catalog_row(alias, def));
        }

        desc
    }
}

#[cfg(test)]
mod tests {
    use super::{
        catalog_cache_len, catalog_row, clear_catalog_cache, BUILTIN_DEFAULT_MODEL,
        CATALOG_CACHE_CAPACITY, DEFAULT_MAX_TURNS, MAX_TURNS_LIMIT, ModelDefinition, ModelManifest,
    };

    fn single(definition: ModelDefinition) -> ModelManifest {
        let mut models = std::collections::HashMap::new();
        models.insert("solo".to_string(), definition);
        ModelManifest {
            default: Some("solo".to_string()),
            models,
        }
    }

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
    fn test_built_in_default_model_const() {
        assert_eq!(
            ModelManifest::default().default.as_deref(),
            Some(BUILTIN_DEFAULT_MODEL)
        );
        assert_eq!(
            DEFAULT_MAX_TURNS, 100,
            "the documented default budget is 100 turns"
        );
        assert_eq!(
            MAX_TURNS_LIMIT, 500,
            "mirrors the REQUEST_TURNS expansion cap in pool.rs"
        );
    }

    #[test]
    fn test_sanitize_temperature_clamps_and_drops_non_finite() {
        assert_eq!(ModelManifest::sanitize_temperature(None), None);
        assert_eq!(ModelManifest::sanitize_temperature(Some(0.7)), Some(0.7));
        assert_eq!(ModelManifest::sanitize_temperature(Some(2.5)), Some(2.0));
        assert_eq!(ModelManifest::sanitize_temperature(Some(-1.0)), Some(0.0));
        assert_eq!(ModelManifest::sanitize_temperature(Some(f32::NAN)), None);
        assert_eq!(
            ModelManifest::sanitize_temperature(Some(f32::INFINITY)),
            None
        );
        assert_eq!(
            ModelManifest::sanitize_temperature(Some(f32::NEG_INFINITY)),
            None
        );
    }

    #[test]
    fn test_sanitize_max_turns_never_returns_zero() {
        // `0` is `Some(0)`, not `None`: shipping it would make the worker's
        // `while step < current_max_turns` loop exit before its first turn.
        assert_eq!(
            ModelManifest::sanitize_max_turns(None, None),
            DEFAULT_MAX_TURNS
        );
        assert_eq!(
            ModelManifest::sanitize_max_turns(Some(0), None),
            DEFAULT_MAX_TURNS
        );
        assert_eq!(
            ModelManifest::sanitize_max_turns(None, Some(0)),
            DEFAULT_MAX_TURNS
        );
        // A zero request must not shadow a usable manifest budget.
        assert_eq!(ModelManifest::sanitize_max_turns(Some(0), Some(50)), 50);
        assert_eq!(ModelManifest::sanitize_max_turns(None, Some(50)), 50);
        assert_eq!(ModelManifest::sanitize_max_turns(Some(12), Some(50)), 12);
        assert_eq!(
            ModelManifest::sanitize_max_turns(Some(usize::MAX), None),
            MAX_TURNS_LIMIT
        );
    }

    #[test]
    fn test_normalize_repairs_every_fixable_warning() {
        let mut models = std::collections::HashMap::new();
        models.insert(
            "hot".to_string(),
            ModelDefinition {
                id: "combo:hot".to_string(),
                role: None,
                temperature: Some(9.0),
                max_turns: Some(0),
            },
        );
        models.insert(
            "cold".to_string(),
            ModelDefinition {
                id: "combo:cold".to_string(),
                role: None,
                temperature: Some(f32::NAN),
                max_turns: Some(usize::MAX),
            },
        );
        let manifest = ModelManifest {
            default: Some("ghost".to_string()),
            models,
        };

        assert!(
            !manifest.validate().is_empty(),
            "the fixture must start out invalid"
        );

        let normalized = manifest.normalized();
        assert_eq!(normalized.default, None, "a dangling default is dropped");
        assert_eq!(normalized.models.len(), 2, "entries are still served");

        let hot = &normalized.models["hot"];
        assert_eq!(hot.temperature, Some(2.0), "out-of-range is clamped");
        assert_eq!(
            hot.max_turns,
            Some(DEFAULT_MAX_TURNS),
            "0 becomes the default budget"
        );

        let cold = &normalized.models["cold"];
        assert_eq!(
            cold.temperature, None,
            "a non-finite temperature is dropped"
        );
        assert_eq!(
            cold.max_turns,
            Some(MAX_TURNS_LIMIT),
            "the budget is clamped"
        );

        assert!(
            normalized.validate().is_empty(),
            "everything fixable is fixed: {:?}",
            normalized.validate()
        );
    }

    #[test]
    fn test_normalize_keeps_a_resolvable_default_and_is_idempotent() {
        let manifest = single(ModelDefinition {
            id: "combo:solo".to_string(),
            role: None,
            temperature: Some(0.4),
            max_turns: Some(7),
        });

        let normalized = manifest.normalized();
        assert_eq!(normalized.default, Some("solo".to_string()));
        assert_eq!(
            normalized.resolve_model("solo"),
            ("combo:solo".to_string(), Some(0.4), Some(7))
        );
        assert_eq!(
            normalized.clone().normalize().validate(),
            normalized.validate()
        );
    }

    #[test]
    fn test_normalize_drops_a_padded_default_that_names_nothing() {
        let mut models = std::collections::HashMap::new();
        models.insert(
            "solo".to_string(),
            ModelDefinition {
                id: "combo:solo".to_string(),
                role: None,
                temperature: None,
                max_turns: None,
            },
        );
        let manifest = ModelManifest {
            default: Some("  ghost  ".to_string()),
            models,
        };

        assert!(!manifest.validate().is_empty());
        assert_eq!(manifest.normalized().default, None);
    }

    #[test]
    fn test_validate_order_is_stable_regardless_of_insertion_order() {
        // The catalog is a `HashMap`, so ordering has to come from sorting the
        // aliases rather than from the iteration order.
        let render = |aliases: &[&str]| {
            let mut models = std::collections::HashMap::new();
            for alias in aliases {
                models.insert(
                    (*alias).to_string(),
                    ModelDefinition {
                        id: "".to_string(),
                        role: None,
                        temperature: None,
                        max_turns: Some(0),
                    },
                );
            }
            ModelManifest {
                default: None,
                models,
            }
            .validate()
        };

        let forward = render(&["alpha", "beta", "gamma", "delta"]);
        let backward = render(&["delta", "gamma", "beta", "alpha"]);

        assert_eq!(forward, backward);
        assert_eq!(
            forward,
            [
                "model \"alpha\": id cannot be empty",
                "model \"alpha\": max_turns must be greater than 0; replaced with 100",
                "model \"beta\": id cannot be empty",
                "model \"beta\": max_turns must be greater than 0; replaced with 100",
                "model \"delta\": id cannot be empty",
                "model \"delta\": max_turns must be greater than 0; replaced with 100",
                "model \"gamma\": id cannot be empty",
                "model \"gamma\": max_turns must be greater than 0; replaced with 100",
            ]
        );
    }

    static TEST_CACHE_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn test_catalog_row_is_memoized_and_keyed_on_every_input() {
        let _guard = TEST_CACHE_MUTEX.lock().unwrap();
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
        let _guard = TEST_CACHE_MUTEX.lock().unwrap();
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
    fn test_tool_description_lists_aliases_in_sorted_order() {
        let mut models = std::collections::HashMap::new();
        for alias in ["zulu", "alpha", "mike"] {
            models.insert(
                alias.to_string(),
                ModelDefinition {
                    id: format!("vendor:{alias}"),
                    role: Some("Role.".to_string()),
                    temperature: None,
                    max_turns: None,
                },
            );
        }
        let manifest = ModelManifest {
            default: None,
            models,
        };

        let description = manifest.build_tool_description();
        let bullets: Vec<&str> = description.lines().skip(1).collect();
        assert_eq!(
            bullets,
            [
                "- `alpha` (id: `vendor:alpha`): Role.",
                "- `mike` (id: `vendor:mike`): Role.",
                "- `zulu` (id: `vendor:zulu`): Role.",
            ],
            "the advertised catalog must not depend on HashMap iteration order"
        );
    }

    #[test]
    fn test_tool_description_is_reproducible_and_cached() {
        let manifest = ModelManifest::default();

        let first = manifest.build_tool_description();
        let second = manifest.build_tool_description();
        assert_eq!(first, second);

        // Bullets are sorted by alias, not by HashMap iteration order.
        let aliases: Vec<&str> = first
            .lines()
            .skip(1)
            .filter_map(|line| line.split('`').nth(1))
            .collect();
        assert_eq!(aliases, vec!["nerd", "ninja"]);
    }

    /// `from_path` is the only loader that names a file explicitly, so the
    /// fixtures are written to a scratch directory and the manifest is checked
    /// for the two properties the discovery path relies on: the YAML is parsed
    /// and the fixups from `validate` are already applied.
    #[test]
    fn test_from_path_parses_and_normalizes_an_explicit_file() {
        let dir = std::env::temp_dir().join(format!(
            "swe-manifest-from-path-{}-{}",
            std::process::id(),
            catalog_cache_len() as u64 ^ DEFAULT_MAX_TURNS as u64
        ));
        std::fs::create_dir_all(&dir).expect("scratch dir");
        let path = dir.join("models.yaml");
        std::fs::write(
            &path,
            "default: ghost\nmodels:\n  hot: { id: vendor:hot, role: \"Hot.\", temperature: 9.0, max_turns: 0 }\n",
        )
        .expect("fixture written");

        let manifest = ModelManifest::from_path(&path).expect("fixture parses");

        // A dangling `default` is dropped and the out-of-range values repaired,
        // exactly as `from_candidate` would serve them.
        assert_eq!(manifest.default, None);
        let hot = &manifest.models["hot"];
        assert_eq!(hot.temperature, Some(2.0));
        assert_eq!(hot.max_turns, Some(DEFAULT_MAX_TURNS));
        assert_eq!(
            manifest.resolve_model("hot"),
            ("vendor:hot".to_string(), Some(2.0), Some(DEFAULT_MAX_TURNS))
        );

        // A malformed file is a hard error, never a silent built-in fallback.
        let broken = dir.join("broken.yaml");
        std::fs::write(&broken, "models: [not, a, mapping]").expect("fixture written");
        assert!(ModelManifest::from_path(&broken).is_err());

        // A missing file is an error too: the caller named it explicitly.
        assert!(ModelManifest::from_path(&dir.join("absent.yaml")).is_err());

        let _ = std::fs::remove_dir_all(&dir);
    }
}
