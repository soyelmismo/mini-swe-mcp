//! Unit tests for the `manifest` package.
//!
//! Kept in a dedicated file rather than inlined in `mod.rs` so the module wiring
//! there stays readable. Everything here is exercised through the package
//! surface (`super::…`), i.e. exactly the API the rest of the crate sees:
//! manifest discovery/loading, id resolution, catalog rendering, the advisory
//! `validate` rules and the `normalize` fixups.
//!
//! The catalog-row cache assertions additionally lock a process-wide mutex
//! (`TEST_CACHE_MUTEX`) because the memoization cache is a global shared by
//! every manifest instance in the process.

use super::{
    BUILTIN_DEFAULT_MODEL, CATALOG_CACHE_CAPACITY, DEFAULT_MAX_TURNS, DEFAULT_ROLE,
    MAX_MEMORY_PROMPT_BYTES, MAX_TURNS_LIMIT, MEMORY_DIR, ModelDefinition, ModelManifest,
    agent_memory_path, append_agent_memory, build_system_prompt, catalog::catalog_row,
    catalog_cache_len, clear_catalog_cache, load_agent_memory,
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
            policy: None,
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
            policy: None,
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
            policy: None,
        },
    );
    models.insert(
        "cold".to_string(),
        ModelDefinition {
            id: "combo:cold".to_string(),
            role: None,
            temperature: Some(f32::NAN),
            max_turns: Some(usize::MAX),
            policy: None,
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
        policy: None,
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
            policy: None,
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
                    policy: None,
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

#[test]
fn test_catalog_row_is_memoized_and_keyed_on_every_input() {
    let _guard = super::cache::TEST_CACHE_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    clear_catalog_cache();

    let a = ModelDefinition {
        id: "vendor:a".to_string(),
        role: Some("Role A.".to_string()),
        temperature: None,
        max_turns: None,
        policy: None,
    };
    let b = ModelDefinition {
        id: "vendor:b".to_string(),
        role: Some("Role A.".to_string()),
        temperature: None,
        max_turns: None,
        policy: None,
    };
    let no_role = ModelDefinition {
        id: "vendor:a".to_string(),
        role: None,
        temperature: None,
        max_turns: None,
        policy: None,
    };

    let first = catalog_row("a", &a);
    assert_eq!(&*first, "- `a` (id: `vendor:a`): Role A.\n");
    assert_eq!(catalog_cache_len(), 1);

    // Same key twice -> memoized, no new entry AND no re-render: the cache
    // hands back the very same `Arc`, so a hit is a refcount bump, not a copy.
    let second = catalog_row("a", &a);
    assert_eq!(&*second, "- `a` (id: `vendor:a`): Role A.\n");
    assert!(std::sync::Arc::ptr_eq(&first, &second));
    assert_eq!(catalog_cache_len(), 1);

    // Different id, different role and a missing role are all distinct keys.
    let other_id = catalog_row("a", &b);
    assert_eq!(&*other_id, "- `a` (id: `vendor:b`): Role A.\n");
    let fallback = catalog_row("a", &no_role);
    assert_eq!(&*fallback, "- `a` (id: `vendor:a`): Autonomous subagent\n");
    assert_eq!(catalog_cache_len(), 3);
}

/// The cache key is built from the **effective** role, so a model that declares
/// no role and one that spells out `DEFAULT_ROLE` verbatim render identical text
/// and therefore correctly share one entry. Keying on the raw `Option` (as the
/// previous `(String, String, String)` tuple did, via `unwrap_or_default()`)
/// would have split them into two entries holding the same bytes.
#[test]
fn test_catalog_row_shares_an_entry_with_an_explicit_default_role() {
    let _guard = super::cache::TEST_CACHE_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    clear_catalog_cache();

    let implicit = ModelDefinition {
        id: "vendor:role".to_string(),
        role: None,
        temperature: None,
        max_turns: None,
        policy: None,
    };
    let explicit = ModelDefinition {
        role: Some(DEFAULT_ROLE.to_string()),
        ..implicit.clone()
    };

    let a = catalog_row("r", &implicit);
    let b = catalog_row("r", &explicit);

    assert_eq!(a, b, "both render the same bullet");
    assert!(std::sync::Arc::ptr_eq(&a, &b), "and share one cache entry");
    assert_eq!(catalog_cache_len(), 1);
}

#[test]
fn test_catalog_cache_stays_bounded() {
    let _guard = super::cache::TEST_CACHE_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    clear_catalog_cache();

    for i in 0..(CATALOG_CACHE_CAPACITY + 8) {
        let def = ModelDefinition {
            id: format!("vendor:id{i}"),
            role: Some("Role.".to_string()),
            temperature: None,
            max_turns: None,
            policy: None,
        };
        let row = catalog_row(&format!("alias{i}"), &def);
        assert_eq!(&*row, format!("- `alias{i}` (id: `vendor:id{i}`): Role.\n"));
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
            policy: None,
        },
    );
    models.insert(
        "a:shared".to_string(),
        ModelDefinition {
            id: "vendor:shared".to_string(),
            role: None,
            temperature: Some(0.1),
            max_turns: Some(1),
            policy: None,
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
                policy: None,
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
                policy: None,
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

// ---------------------------------------------------------------------------
// Persistent role memory (`.agents/memory/<alias>.md`)
// ---------------------------------------------------------------------------

/// A scratch repository root for the memory tests, removed on drop.
struct MemoryRepo {
    path: std::path::PathBuf,
}

impl MemoryRepo {
    fn new(tag: &str) -> Self {
        // The tag keeps the two repos that share a `.agents/memory` layout apart,
        // and the pid keeps two concurrent test binaries apart.
        let path = std::env::temp_dir().join(format!(
            "swe-manifest-memory-{}-{tag}-{}",
            std::process::id(),
            CATALOG_CACHE_CAPACITY as u64
        ));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).expect("scratch repo");
        Self { path }
    }

    fn memory_dir(&self) -> std::path::PathBuf {
        self.path.join(".agents").join("memory")
    }
}

impl Drop for MemoryRepo {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

/// Memory lives at `.agents/memory/<alias>.md`, keyed by the alias and never by
/// the resolved model id.
#[test]
fn test_agent_memory_path_is_alias_keyed_and_traversal_safe() {
    let repo = std::path::Path::new("/repo");

    assert_eq!(
        agent_memory_path(repo, "ninja"),
        Some(std::path::PathBuf::from("/repo/.agents/memory/ninja.md"))
    );
    assert_eq!(
        agent_memory_path(repo, "nerd"),
        Some(std::path::PathBuf::from("/repo/.agents/memory/nerd.md"))
    );
    // A full model id sanitizes into a harmless slug instead of walking out of
    // the memory directory.
    let hostile = agent_memory_path(repo, "../../etc/passwd").expect("sanitized");
    assert_eq!(hostile.parent(), Some(repo.join(MEMORY_DIR).as_path()));
    assert!(!hostile.to_string_lossy().contains(".."));
    // Nothing usable left -> no path at all.
    assert_eq!(agent_memory_path(repo, "///"), None);
}

#[test]
fn test_load_agent_memory_reads_the_role_file() {
    let repo = MemoryRepo::new("load");
    std::fs::create_dir_all(repo.memory_dir()).expect("memory dir");
    std::fs::write(
        agent_memory_path(&repo.path, "ninja").expect("path"),
        "- Run `cargo test --all-targets` before declaring success.\n",
    )
    .expect("fixture written");

    let memory = load_agent_memory(&repo.path, "ninja").expect("memory is loaded");
    assert_eq!(
        memory,
        "- Run `cargo test --all-targets` before declaring success."
    );

    // Memory is per role: another alias of the same repo has none.
    assert_eq!(load_agent_memory(&repo.path, "nerd"), None);
}

/// The two shipped defaults must actually be loadable, or the feature would ship
/// dead: the files exist at the repo root and are readable through the loader.
#[test]
fn test_shipped_role_memory_files_are_loadable() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    for alias in ["ninja", "nerd"] {
        let memory = load_agent_memory(repo_root, alias)
            .unwrap_or_else(|| panic!("{MEMORY_DIR}/{alias}.md must ship with the crate"));
        assert!(
            memory.contains('-'),
            "{alias} memory must hold at least one takeaway"
        );
        assert!(
            !load_agent_memory(repo_root, alias).unwrap().is_empty(),
            "{alias} memory must never load as empty"
        );
    }
}

/// Missing, blank, unreadable and directory-shaped memory all degrade to `None`
/// so a repository without `.agents/memory/` behaves exactly as it did before.
#[test]
fn test_load_agent_memory_falls_back_to_none() {
    let repo = MemoryRepo::new("missing");

    // No `.agents/memory` directory at all.
    assert_eq!(load_agent_memory(&repo.path, "ninja"), None);
    // A directory exists but holds no file for this alias.
    std::fs::create_dir_all(repo.memory_dir()).expect("memory dir");
    assert_eq!(load_agent_memory(&repo.path, "ninja"), None);
    // A blank file is as good as absent: injecting it would add an empty section.
    let path = agent_memory_path(&repo.path, "ninja").expect("path");
    std::fs::write(&path, "   \n\n\t\n").expect("fixture written");
    assert_eq!(load_agent_memory(&repo.path, "ninja"), None);
    // The path is a directory, not a file.
    std::fs::remove_file(&path).expect("cleanup");
    std::fs::create_dir_all(&path).expect("dir-shaped memory path");
    assert_eq!(load_agent_memory(&repo.path, "ninja"), None);
    // An alias with no usable slug has no path to load.
    assert_eq!(load_agent_memory(&repo.path, "///"), None);
}

#[test]
fn test_append_agent_memory_creates_and_appends() {
    let repo = MemoryRepo::new("append");

    append_agent_memory(&repo.path, "ninja", "Both clippy and cargo test must pass.")
        .expect("first append");

    let memory = load_agent_memory(&repo.path, "ninja").expect("memory exists");
    assert!(
        memory.contains("Both clippy and cargo test must pass."),
        "the note must be readable back: {memory}"
    );
    // The directory is created on demand, at the documented location.
    assert!(repo.memory_dir().join("ninja.md").is_file());

    append_agent_memory(&repo.path, "ninja", "Prefer the smallest diff.").expect("second append");

    let memory = load_agent_memory(&repo.path, "ninja").expect("memory exists");
    assert!(memory.contains("Both clippy and cargo test must pass."));
    assert!(memory.contains("Prefer the smallest diff."));
    // Appends accumulate; they never overwrite.
    assert_eq!(memory.lines().filter(|l| l.starts_with("- ")).count(), 2);
    // A second role has its own, independent file.
    assert_eq!(load_agent_memory(&repo.path, "nerd"), None);
    append_agent_memory(&repo.path, "nerd", "Reproduce before you patch.").expect("nerd append");
    let nerd = load_agent_memory(&repo.path, "nerd").expect("nerd memory");
    assert!(nerd.contains("Reproduce before you patch."));
    assert!(!nerd.contains("smallest diff"));
}

/// A takeaway must land as exactly one list item: a multi-line note cannot break
/// the line-oriented format, and an empty note is refused rather than written as a
/// blank bullet that would load back as noise.
#[test]
fn test_append_agent_memory_normalizes_notes_and_rejects_empty() {
    let repo = MemoryRepo::new("normalize");

    append_agent_memory(&repo.path, "nerd", "  reproduce\n\tbefore   patching  ").expect("append");
    let memory = load_agent_memory(&repo.path, "nerd").expect("memory exists");
    assert_eq!(
        memory.lines().filter(|l| l.starts_with("- ")).count(),
        1,
        "a multi-line note must collapse into a single entry: {memory}"
    );
    assert!(memory.contains("- reproduce before patching"));

    for empty in ["", "   ", "\n\t "] {
        assert!(
            append_agent_memory(&repo.path, "nerd", empty).is_err(),
            "an empty note must be rejected, not written"
        );
    }
    // An alias with no usable slug has no memory file to write to.
    assert!(append_agent_memory(&repo.path, "///", "note").is_err());
}

/// Memory is appended to forever, so what reaches the prompt is capped: the
/// system instructions must not be squeezed out by a bloated memory file, and the
/// cut must land on a line boundary rather than mid-line.
#[test]
fn test_loaded_memory_is_bounded() {
    let repo = MemoryRepo::new("bounded");
    std::fs::create_dir_all(repo.memory_dir()).expect("memory dir");
    let mut note = String::new();
    while note.len() < MAX_MEMORY_PROMPT_BYTES * 2 {
        note.push_str("- filler takeaway that keeps the memory file growing\n");
    }
    std::fs::write(agent_memory_path(&repo.path, "ninja").expect("path"), &note)
        .expect("fixture written");

    let memory = load_agent_memory(&repo.path, "ninja").expect("memory is loaded");
    assert!(memory.len() <= MAX_MEMORY_PROMPT_BYTES);
    assert!(
        memory
            .lines()
            .all(|line| line.starts_with("- ") || line.is_empty()),
        "the cut must land on a line boundary: {memory:?}"
    );
    assert!(!memory.contains("filler takeaway") || memory.len() < note.len());
}

/// The system prompt is the only injection point, and it must be a no-op for a
/// repository without memory.
#[test]
fn test_build_system_prompt_injects_memory_only_when_present() {
    let repo = MemoryRepo::new("prompt");

    // No memory file: byte-identical to the static prompt.
    assert_eq!(
        build_system_prompt(&repo.path, "ninja"),
        crate::agent::SYSTEM_PROMPT
    );

    append_agent_memory(&repo.path, "ninja", "Verify with cargo clippy.").expect("append");
    let prompt = build_system_prompt(&repo.path, "ninja");
    assert!(prompt.starts_with(crate::agent::SYSTEM_PROMPT));
    assert!(
        prompt.contains("Verify with cargo clippy."),
        "the role memory must reach the prompt: {prompt}"
    );
    // Roles stay isolated: the nerd prompt is untouched by ninja's memory.
    assert_eq!(
        build_system_prompt(&repo.path, "nerd"),
        crate::agent::SYSTEM_PROMPT
    );
}

/// The pool carries a *resolved* model id (`combo:ninja`), while memory files are
/// keyed by alias, so `alias_for_model` is what makes the lookup work at all.
#[test]
fn test_alias_for_model_bridges_resolved_ids() {
    let manifest = ModelManifest::default();

    assert_eq!(manifest.alias_for_model("combo:ninja"), "ninja");
    assert_eq!(manifest.alias_for_model("combo:nerd"), "nerd");
    // An alias is already its own answer.
    assert_eq!(manifest.alias_for_model("ninja"), "ninja");
    // Unknown models pass through so a missing memory file is simply "none".
    assert_eq!(manifest.alias_for_model("some/unknown"), "some/unknown");

    // And the round trip actually finds the file the worker asked for.
    let repo = MemoryRepo::new("alias");
    append_agent_memory(&repo.path, "ninja", "Round trip note.").expect("append");
    let alias = manifest.alias_for_model("combo:ninja");
    assert!(build_system_prompt(&repo.path, &alias).contains("Round trip note."));
}

/// `rename` makes each write atomic and the lock serializes the
/// read-modify-write; the second property is what stops two agents finishing at
/// the same time from silently discarding each other's takeaway, so it is
/// asserted rather than assumed.
#[test]
fn test_concurrent_appends_do_not_lose_notes() {
    use std::sync::Arc;

    let repo = Arc::new(MemoryRepo::new("concurrent"));
    const THREADS: usize = 8;

    std::thread::scope(|scope| {
        for i in 0..THREADS {
            let repo = Arc::clone(&repo);
            scope.spawn(move || {
                append_agent_memory(&repo.path, "nerd", &format!("takeaway {i}"))
                    .expect("concurrent append");
            });
        }
    });

    let memory = load_agent_memory(&repo.path, "nerd").expect("memory exists");
    for i in 0..THREADS {
        assert!(
            memory.contains(&format!("takeaway {i}")),
            "every concurrent append must survive; missing takeaway {i}: {memory}"
        );
    }
    assert_eq!(
        memory.lines().filter(|l| l.starts_with("- ")).count(),
        THREADS,
        "and no append may be duplicated"
    );
    // The staging file never survives a successful append.
    let leftovers: Vec<_> = std::fs::read_dir(repo.memory_dir())
        .expect("memory dir")
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|name| name.ends_with(".tmp"))
        .collect();
    assert!(
        leftovers.is_empty(),
        "staging debris left behind: {leftovers:?}"
    );
}

// ----------
// Declarative execution policy (`policy:` in models.yaml)
// ----------

use crate::manifest::{ExecutionPolicy, FS_POLICIES, FsPolicy, NETWORK_POLICIES, NetworkPolicy};

/// Parse a single-entry manifest carrying the given `policy:` block body.
fn policy_manifest(block: &str) -> ModelManifest {
    let yaml = format!("models:\n  solo:\n    id: combo:solo\n{block}");
    serde_yaml::from_str(&yaml).expect("policy manifest must parse")
}

#[test]
fn test_a_model_without_a_policy_stays_none_and_warns_about_nothing() {
    let manifest = policy_manifest("");

    assert_eq!(
        manifest.models["solo"].policy, None,
        "an entry that never declared a policy must parse to None, so a \
         models.yaml written before policies existed behaves unchanged"
    );
    assert!(
        manifest.validate().is_empty(),
        "a manifest without policies must stay warning-free: {:?}",
        manifest.validate()
    );
    assert_eq!(
        manifest.normalized().models["solo"].policy,
        None,
        "None must survive normalization, so \"declared nothing\" stays \
         distinguishable from \"declared the default\""
    );
}

#[test]
fn test_every_declared_policy_value_is_accepted_verbatim() {
    let def = |id: &str, network: NetworkPolicy, fs: FsPolicy| ModelDefinition {
        id: id.to_string(),
        role: None,
        temperature: None,
        max_turns: None,
        policy: Some(ExecutionPolicy {
            network: Some(network),
            fs: Some(fs),
        }),
    };
    let manifest = ModelManifest {
        default: None,
        models: [
            (
                "a".to_string(),
                def("combo:a", NetworkPolicy::Offline, FsPolicy::ReadOnly),
            ),
            (
                "b".to_string(),
                def("combo:b", NetworkPolicy::Allow, FsPolicy::Full),
            ),
            (
                "c".to_string(),
                def("combo:c", NetworkPolicy::Allow, FsPolicy::WorktreeOnly),
            ),
        ]
        .into_iter()
        .collect(),
    };

    assert!(
        manifest.validate().is_empty(),
        "every documented policy value must be accepted: {:?}",
        manifest.validate()
    );
    assert_eq!(
        manifest.normalized().models["a"].policy,
        manifest.models["a"].policy,
        "an already-valid policy must survive normalization byte-for-byte"
    );
}

#[test]
fn test_policy_round_trips_through_yaml() {
    for (network, fs) in [
        ("offline", "read-only"),
        ("allow", "worktree-only"),
        ("allow", "full"),
    ] {
        let manifest = policy_manifest(&format!(
            "    policy:\n      network: {network}\n      fs: {fs}\n"
        ));
        let policy = manifest.models["solo"]
            .policy
            .clone()
            .expect("the policy block must parse");
        assert_eq!(
            policy.network.as_ref().map(NetworkPolicy::as_str),
            Some(network)
        );
        assert_eq!(policy.fs.as_ref().map(FsPolicy::as_str), Some(fs));
        assert!(
            manifest.validate().is_empty(),
            "{network}/{fs} is a documented policy and must not warn"
        );
    }
}

#[test]
fn test_an_unknown_policy_value_warns_and_repairs_to_the_restrictive_default() {
    let manifest =
        policy_manifest("    policy:\n      network: \"offine\"\n      fs: \"everything\"\n");

    let warnings = manifest.validate();
    assert_eq!(
        warnings.len(),
        2,
        "both misspelled fields must be reported: {warnings:?}"
    );
    assert!(
        warnings.iter().any(|w| w.contains("offine")),
        "the network warning must name the value the user wrote: {warnings:?}"
    );
    assert!(
        warnings.iter().any(|w| w.contains("everything")),
        "the fs warning must name the value the user wrote: {warnings:?}"
    );

    let policy = manifest.normalized().models["solo"]
        .policy
        .clone()
        .expect("still a policy after repair");
    assert_eq!(
        policy.network,
        Some(NetworkPolicy::Offline),
        "an unknown network must fall back to the restrictive default, never \
         to the permissive one: a typo must not widen a sandbox"
    );
    assert_eq!(
        policy.fs,
        Some(FsPolicy::ReadOnly),
        "an unknown fs value must fall back to the restrictive default"
    );
    assert!(
        manifest.normalized().validate().is_empty(),
        "every reported policy warning must be repaired by normalize"
    );
}

#[test]
fn test_policy_values_are_matched_case_insensitively_and_after_trimming() {
    let manifest = policy_manifest("    policy:\n      network: \"  Offline \"\n");

    assert!(
        manifest.validate().is_empty(),
        "whitespace and case must not turn a valid policy into a warning: {:?}",
        manifest.validate()
    );
    assert_eq!(
        manifest.normalized().models["solo"]
            .policy
            .as_ref()
            .and_then(|p| p.network.as_ref())
            .map(NetworkPolicy::as_str),
        Some("offline"),
        "a case-insensitive match must still normalize to the canonical spelling"
    );
}

#[test]
fn test_shipped_models_yaml_declares_an_explicit_policy_for_both_roles() {
    let manifest = ModelManifest::from_path(std::path::Path::new("models.yaml"))
        .expect("the shipped models.yaml must load");

    for alias in ["ninja", "nerd"] {
        let policy = manifest.models[alias]
            .policy
            .clone()
            .unwrap_or_else(|| panic!("{alias} must declare an explicit policy"));
        assert!(
            policy
                .network
                .as_ref()
                .is_some_and(NetworkPolicy::is_declared),
            "{alias} must declare a known network policy"
        );
        assert!(
            policy.fs.as_ref().is_some_and(FsPolicy::is_declared),
            "{alias} must declare a known fs policy"
        );
    }
    assert!(
        manifest.validate().is_empty(),
        "the shipped models.yaml must be warning-free: {:?}",
        manifest.validate()
    );
}

#[test]
fn test_the_advertised_policy_lists_match_the_parsed_grammar() {
    // `NETWORK_POLICIES` / `FS_POLICIES` are what the warning messages print, so
    // a value the parser accepts but the list omits (or vice versa) would make
    // the message either incomplete or a lie.
    for value in NETWORK_POLICIES {
        assert!(
            matches!(
                NetworkPolicy::parse(value),
                NetworkPolicy::Allow | NetworkPolicy::Offline
            ),
            "{value:?} is advertised as a network policy but does not parse as one",
        );
    }
    for value in FS_POLICIES {
        assert!(
            !matches!(FsPolicy::parse(value), FsPolicy::Other(..)),
            "{value:?} is advertised as an fs policy but does not parse as one",
        );
    }

    // Every variant the parser can produce is advertised, so a *new* mode cannot
    // be added without also updating the user-facing list.
    for variant in [NetworkPolicy::Offline, NetworkPolicy::Allow] {
        let name = variant.as_str();
        assert!(
            NETWORK_POLICIES.contains(&name),
            "{name:?} parses as a policy but is not in NETWORK_POLICIES",
        );
    }
    for variant in [FsPolicy::ReadOnly, FsPolicy::WorktreeOnly, FsPolicy::Full] {
        let name = variant.as_str();
        assert!(
            FS_POLICIES.contains(&name),
            "{name:?} parses as a policy but is not in FS_POLICIES",
        );
    }
}

#[test]
fn test_model_level_network_policies_match_the_dispatch_network_modes() {
    // `policy.network` and the `network` argument of the `worker` tool are the
    // same switch spelled in two places, so a value accepted by one and
    // rejected by the other would be a trap. They are kept as two constants (the
    // manifest must not depend on `mcp` and vice versa), so pin them here.
    assert_eq!(
        NETWORK_POLICIES,
        crate::mcp::NETWORK_MODES,
        "models.yaml policy.network and the dispatch tool advertise different values",
    );
    for mode in crate::mcp::NETWORK_MODES {
        assert!(
            matches!(
                NetworkPolicy::parse(mode),
                NetworkPolicy::Allow | NetworkPolicy::Offline
            ),
            "{mode:?} is a dispatch network mode but not a valid models.yaml policy",
        );
    }
}

#[test]
fn test_normalizing_an_undeclared_or_empty_policy_preserves_the_declaration_shape() {
    // "declared nothing" and "declared nothing useful" are different states and
    // neither is repaired into the other: a `None` policy stays `None`, and an
    // empty block stays an (empty) `Some` so the distinction survives normalize.
    let absent = policy_manifest("");
    assert_eq!(absent.models["solo"].policy, None);

    let empty = policy_manifest("    policy: {}\n");
    let policy = empty.models["solo"].policy.clone().expect("a policy block");
    assert!(
        policy.is_empty(),
        "an empty block declares nothing: {policy:?}"
    );

    let normalized = empty.normalized();
    assert_eq!(
        normalized.models["solo"].policy,
        Some(ExecutionPolicy::default()),
        "an empty block is still a declared (if empty) policy",
    );
    assert!(
        normalized.validate().is_empty(),
        "an empty policy block has nothing to warn about",
    );

    assert_eq!(
        absent.normalized().models["solo"].policy,
        None,
        "normalize must never invent a policy the manifest did not declare",
    );
}
