//! Unit tests for the `manifest` package.
//!
//! Kept in a dedicated file rather than inlined in `mod.rs` so the module wiring
//! there stays readable. Everything here is exercised through the package
//! surface (`super::…`), i.e. exactly the API the rest of the crate sees:
//! manifest discovery/loading, id resolution, catalog rendering, the advisory
//! `validate` rules and the `normalize` fixups.

use super::{
    BUILTIN_DEFAULT_MODEL, DEFAULT_MAX_TURNS, MAX_MEMORY_PROMPT_BYTES, MAX_TURNS_LIMIT, MEMORY_DIR,
    ModelDefinition, ModelManifest, agent_memory_path, build_system_prompt, load_agent_memory,
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

    let normalized = manifest.normalize();
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

    let normalized = manifest.normalize();
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
    assert_eq!(manifest.normalize().default, None);
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
fn test_tool_description_is_reproducible() {
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
        DEFAULT_MAX_TURNS as u64
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
            DEFAULT_MAX_TURNS as u64
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

    std::fs::create_dir_all(repo.memory_dir()).expect("memory dir");
    std::fs::write(
        agent_memory_path(&repo.path, "ninja").expect("path"),
        "PERSISTENT ROLE MEMORY (from .agents/memory/):\n- Verify with cargo clippy.\n",
    )
    .expect("fixture written");
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
    std::fs::create_dir_all(repo.memory_dir()).expect("memory dir");
    std::fs::write(
        agent_memory_path(&repo.path, "ninja").expect("path"),
        "PERSISTENT ROLE MEMORY (from .agents/memory/):\n- Round trip note.\n",
    )
    .expect("fixture written");
    let alias = manifest.alias_for_model("combo:ninja");
    assert!(build_system_prompt(&repo.path, &alias).contains("Round trip note."));
}

// ----------
// Declarative execution policy (`policy:` in models.yaml)
// ----------

use crate::manifest::{ExecutionPolicy, NETWORK_POLICIES, NetworkPolicy};

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
        manifest.normalize().models["solo"].policy,
        None,
        "None must survive normalization, so \"declared nothing\" stays \
         distinguishable from \"declared the default\""
    );
}

#[test]
fn test_every_declared_network_policy_value_is_accepted_verbatim() {
    let def = |id: &str, network: NetworkPolicy| ModelDefinition {
        id: id.to_string(),
        role: None,
        temperature: None,
        max_turns: None,
        policy: Some(ExecutionPolicy {
            network: Some(network),
        }),
    };
    let manifest = ModelManifest {
        default: None,
        models: [
            ("a".to_string(), def("combo:a", NetworkPolicy::Offline)),
            ("b".to_string(), def("combo:b", NetworkPolicy::Allow)),
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
        manifest.clone().normalize().models["a"].policy,
        manifest.models["a"].policy,
        "an already-valid policy must survive normalization byte-for-byte"
    );
}

#[test]
fn test_policy_round_trips_through_yaml() {
    for network in ["offline", "allow"] {
        let manifest = policy_manifest(&format!("    policy:\n      network: {network}\n"));
        let policy = manifest.models["solo"]
            .policy
            .clone()
            .expect("the policy block must parse");
        assert_eq!(
            policy.network.as_ref().map(NetworkPolicy::as_str),
            Some(network)
        );
        assert!(
            manifest.validate().is_empty(),
            "{network} is a documented policy and must not warn"
        );
    }
}

#[test]
fn test_an_unknown_network_policy_warns_and_repairs_to_the_restrictive_default() {
    let manifest = policy_manifest("    policy:\n      network: \"offine\"\n");

    let warnings = manifest.validate();
    assert_eq!(
        warnings.len(),
        1,
        "the misspelled field must be reported: {warnings:?}"
    );
    assert!(
        warnings.iter().any(|w| w.contains("offine")),
        "the network warning must name the value the user wrote: {warnings:?}"
    );

    let policy = manifest.clone().normalize().models["solo"]
        .policy
        .clone()
        .expect("still a policy after repair");
    assert_eq!(
        policy.network,
        Some(NetworkPolicy::Offline),
        "an unknown network must fall back to the restrictive default, never \
         to the permissive one: a typo must not widen a sandbox"
    );
    assert!(
        manifest.clone().normalize().validate().is_empty(),
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
        manifest.normalize().models["solo"]
            .policy
            .as_ref()
            .and_then(|p| p.network.as_ref())
            .map(NetworkPolicy::as_str),
        Some("offline"),
        "a case-insensitive match must still normalize to the canonical spelling"
    );
}

#[test]
fn test_shipped_models_yaml_declares_an_explicit_network_policy_for_both_roles() {
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
    }
    assert!(
        manifest.validate().is_empty(),
        "the shipped models.yaml must be warning-free: {:?}",
        manifest.validate()
    );
}

#[test]
fn test_the_advertised_network_policy_list_matches_the_parsed_grammar() {
    // `NETWORK_POLICIES` is what the warning messages print, so a value the
    // parser accepts but the list omits (or vice versa) would make the message
    // either incomplete or a lie.
    for value in NETWORK_POLICIES {
        assert!(
            matches!(
                NetworkPolicy::parse(value),
                NetworkPolicy::Allow | NetworkPolicy::Offline
            ),
            "{value:?} is advertised as a network policy but does not parse as one",
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

    let normalized = empty.normalize();
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
        absent.normalize().models["solo"].policy,
        None,
        "normalize must never invent a policy the manifest did not declare",
    );
}
