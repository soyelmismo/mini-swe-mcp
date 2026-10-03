//! Integration tests for [`mini_swe_mcp::manifest::ModelManifest`].
//!
//! The manifest is the user-facing model catalog that is served to MCP hosts:
//!
//! * [`ModelManifest::default`] ships the built-in `ninja` / `nerd` pair, and
//!   it is what the server falls back to when no `models.yaml` can be found.
//! * A custom manifest (parsed from YAML, i.e. exactly what a user's
//!   `models.yaml` looks like) may override the default, temperatures and
//!   turn budgets arbitrarily.
//! * [`ModelManifest::resolve_model`] accepts an alias *or* a full model id
//!   and passes unknown requests through untouched so the upstream provider
//!   can still be used.
//! * [`ModelManifest::build_tool_description`] renders the catalog that is
//!   embedded in the `dispatch` tool description.
//!
//! These are exercised through the public library surface only, i.e. the same
//! way `src/mcp/` and `src/main.rs` consume the manifest.

use mini_swe_mcp::manifest::{DEFAULT_MAX_TURNS, MAX_TURNS_LIMIT, ModelDefinition, ModelManifest};
use std::collections::HashMap;

/// Build a `ModelDefinition` with every field filled in.
fn definition(
    id: &str,
    role: Option<&str>,
    temperature: Option<f32>,
    max_turns: Option<usize>,
) -> ModelDefinition {
    ModelDefinition {
        id: id.to_string(),
        role: role.map(str::to_string),
        temperature,
        max_turns,
        policy: None,
        instructions: None,
    }
}

/// Build a manifest from YAML text, mirroring how `models.yaml` is loaded.
fn parse_manifest(yaml: &str) -> ModelManifest {
    serde_yaml::from_str(yaml).unwrap_or_else(|e| panic!("manifest YAML must parse: {e}\n{yaml}"))
}

// -------------------------------------------------------------------------
// 1. Default manifest
// -------------------------------------------------------------------------

#[test]
fn test_default_manifest_contains_ninja_and_nerd() {
    let manifest = ModelManifest::default();

    assert!(
        manifest.models.contains_key("ninja"),
        "default manifest must contain the `ninja` alias"
    );
    assert!(
        manifest.models.contains_key("nerd"),
        "default manifest must contain the `nerd` alias"
    );
    assert_eq!(
        manifest.default,
        Some("ninja".to_string()),
        "`ninja` is the built-in default model"
    );
    assert_eq!(
        manifest.models.len(),
        2,
        "default manifest exposes exactly the two built-in aliases"
    );
}

#[test]
fn test_default_manifest_model_definitions() {
    let manifest = ModelManifest::default();

    let ninja = manifest.models.get("ninja").expect("ninja entry");
    assert_eq!(ninja.id, "combo:ninja");
    assert_eq!(ninja.temperature, Some(0.2));
    assert_eq!(ninja.max_turns, Some(100));
    assert!(
        ninja.role.as_deref().is_some_and(|r| !r.trim().is_empty()),
        "ninja must document a non-empty role"
    );

    let nerd = manifest.models.get("nerd").expect("nerd entry");
    assert_eq!(nerd.id, "combo:nerd");
    assert_eq!(nerd.temperature, Some(0.6));
    assert_eq!(nerd.max_turns, Some(100));
    assert!(
        nerd.role.as_deref().is_some_and(|r| !r.trim().is_empty()),
        "nerd must document a non-empty role"
    );
}

// -------------------------------------------------------------------------
// 2. Custom manifest
// -------------------------------------------------------------------------

#[test]
fn test_custom_manifest_overrides_temperature_and_max_turns() {
    let manifest = parse_manifest(
        r#"
default: turbo
models:
  turbo:
    id: combo:turbo
    role: "Blazing fast executor."
    temperature: 0.05
    max_turns: 12
  oracle:
    id: combo:oracle
    role: "Slow, careful reasoner."
    temperature: 0.95
    max_turns: 250
"#,
    );

    assert_eq!(manifest.default, Some("turbo".to_string()));
    assert_eq!(manifest.models.len(), 2);
    assert!(
        !manifest.models.contains_key("ninja"),
        "custom manifest replaces defaults"
    );

    let turbo = manifest.models.get("turbo").expect("turbo entry");
    assert_eq!(turbo.id, "combo:turbo");
    assert_eq!(turbo.temperature, Some(0.05));
    assert_eq!(turbo.max_turns, Some(12));

    let oracle = manifest.models.get("oracle").expect("oracle entry");
    assert_eq!(oracle.id, "combo:oracle");
    assert_eq!(oracle.temperature, Some(0.95));
    assert_eq!(oracle.max_turns, Some(250));

    // The custom values must be what the server actually dispatches with.
    assert_eq!(
        manifest.resolve_model("turbo"),
        ("combo:turbo".to_string(), Some(0.05), Some(12))
    );
    assert_eq!(
        manifest.resolve_model("combo:oracle"),
        ("combo:oracle".to_string(), Some(0.95), Some(250))
    );
}

#[test]
fn test_custom_manifest_built_in_code() {
    let mut models = HashMap::new();
    models.insert(
        "solo".to_string(),
        definition("vendor:solo", Some("Only model."), Some(0.33), Some(7)),
    );

    let manifest = ModelManifest {
        default: Some("solo".to_string()),
        strongest: None,
        sensitive_paths: Vec::new(),
        review_modes: std::collections::HashMap::new(),
        models,
    };

    assert_eq!(manifest.models.len(), 1);
    assert_eq!(
        manifest.resolve_model("solo"),
        ("vendor:solo".to_string(), Some(0.33), Some(7))
    );
}

#[test]
fn test_custom_manifest_optional_fields_default_to_none() {
    // `role`, `temperature` and `max_turns` are all optional in `models.yaml`.
    let manifest = parse_manifest(
        r#"
models:
  bare:
    id: vendor:bare
"#,
    );

    assert_eq!(
        manifest.default, None,
        "a manifest without `default:` yields None"
    );
    let bare = manifest.models.get("bare").expect("bare entry");
    assert_eq!(bare.role, None);
    assert_eq!(bare.temperature, None);
    assert_eq!(bare.max_turns, None);

    // No overrides means the requester keeps control of the parameters.
    assert_eq!(
        manifest.resolve_model("bare"),
        ("vendor:bare".to_string(), None, None)
    );
}

#[test]
fn test_empty_manifest_is_valid() {
    let manifest = parse_manifest("{}");
    assert_eq!(manifest.default, None);
    assert!(manifest.models.is_empty());
    assert_eq!(
        manifest.resolve_model("ninja"),
        ("ninja".to_string(), None, None)
    );
}

// -------------------------------------------------------------------------
// 2b. Custom manifest turn-budget (`max_turns`) overrides
// -------------------------------------------------------------------------

#[test]
fn test_custom_manifest_max_turns_override_via_resolve_model() {
    let manifest = parse_manifest(
        r#"
default: quick
models:
  quick:
    id: vendor:quick
    role: "Short, focused edits."
    temperature: 0.1
    max_turns: 8
  marathon:
    id: vendor:marathon
    role: "Long refactors."
    temperature: 0.7
    max_turns: 500
  unlimited:
    id: vendor:unlimited
    role: "No explicit budget."
    temperature: 0.3
"#,
    );

    assert_eq!(manifest.default, Some("quick".to_string()));
    assert_eq!(
        manifest.models.len(),
        3,
        "custom manifest must not inherit the built-in `ninja`/`nerd` entries"
    );

    assert_eq!(
        manifest.resolve_model("quick"),
        ("vendor:quick".to_string(), Some(0.1), Some(8)),
        "per-model `max_turns` must be returned to the dispatcher"
    );
    assert_eq!(
        manifest.resolve_model("vendor:marathon"),
        ("vendor:marathon".to_string(), Some(0.7), Some(500))
    );
    assert_eq!(
        manifest.resolve_model("marathon").2,
        Some(500),
        "a large budget must not be clamped back to the built-in default"
    );
    assert_eq!(
        manifest.resolve_model("unlimited").2,
        None,
        "a model without `max_turns` must leave the turn budget unset"
    );

    let zero = parse_manifest(
        r#"
models:
  halt:
    id: vendor:halt
    max_turns: 0
"#,
    );
    assert_eq!(
        zero.resolve_model("halt"),
        ("vendor:halt".to_string(), None, Some(0))
    );
}

#[test]
fn test_model_definition_max_turns_is_returned_by_resolve_model() {
    let mut models = std::collections::HashMap::new();
    models.insert(
        "tight".to_string(),
        definition("vendor:tight", Some("Tight budget."), Some(0.2), Some(3)),
    );
    models.insert(
        "loose".to_string(),
        definition("vendor:loose", None, Some(0.8), Some(1_000)),
    );
    models.insert(
        "unset".to_string(),
        definition("vendor:unset", None, None, None),
    );
    let manifest = ModelManifest {
        default: Some("tight".to_string()),
        strongest: None,
        sensitive_paths: Vec::new(),
        review_modes: std::collections::HashMap::new(),
        models,
    };

    assert_eq!(
        manifest.resolve_model("tight"),
        ("vendor:tight".to_string(), Some(0.2), Some(3)),
        "a model definition setting `max_turns` must yield that exact value"
    );
    assert_eq!(
        manifest.resolve_model("loose"),
        ("vendor:loose".to_string(), Some(0.8), Some(1_000))
    );
    assert_eq!(
        manifest.resolve_model("unset").2,
        None,
        "omitting `max_turns` must stay `None` so the caller keeps control"
    );

    assert_eq!(
        manifest.resolve_model("not-in-manifest"),
        ("not-in-manifest".to_string(), None, None)
    );
}

// -------------------------------------------------------------------------
// 3. resolve_model
// -------------------------------------------------------------------------

#[test]
fn test_resolve_model_by_alias() {
    let manifest = ModelManifest::default();

    assert_eq!(
        manifest.resolve_model("ninja"),
        ("combo:ninja".to_string(), Some(0.2), Some(100))
    );
    assert_eq!(
        manifest.resolve_model("nerd"),
        ("combo:nerd".to_string(), Some(0.6), Some(100))
    );
}

#[test]
fn test_resolve_model_by_full_model_id() {
    let manifest = ModelManifest::default();

    // Passing the concrete id must be equivalent to passing the alias...
    assert_eq!(
        manifest.resolve_model("combo:ninja"),
        manifest.resolve_model("ninja")
    );
    assert_eq!(
        manifest.resolve_model("combo:nerd"),
        manifest.resolve_model("nerd")
    );

    // ...and must still return the manifest's overrides.
    assert_eq!(
        manifest.resolve_model("combo:nerd"),
        ("combo:nerd".to_string(), Some(0.6), Some(100))
    );
}

#[test]
fn test_resolve_model_unknown_falls_back_to_passthrough() {
    let manifest = ModelManifest::default();

    for unknown in [
        "gpt-9",
        "some/unknown-model",
        "vendor:model:not-in-manifest",
        "NINJA", // alias lookup is case-sensitive
        "",
    ] {
        let (id, temperature, max_turns) = manifest.resolve_model(unknown);
        assert_eq!(
            id, unknown,
            "unknown model {unknown:?} must be passed through"
        );
        assert_eq!(
            temperature, None,
            "unknown model {unknown:?} has no temperature override"
        );
        assert_eq!(
            max_turns, None,
            "unknown model {unknown:?} has no turn override"
        );
    }
}

#[test]
fn test_resolve_model_alias_wins_over_id_lookup() {
    // If one model's id is used as another model's alias, the alias key must
    // take precedence so a manifest can always win over pass-through.
    let mut models = HashMap::new();
    models.insert(
        "a".to_string(),
        definition("vendor:a", Some("A."), Some(0.1), Some(1)),
    );
    models.insert(
        "vendor:a".to_string(),
        definition("vendor:b", Some("B."), Some(0.9), Some(9)),
    );
    let manifest = ModelManifest {
        default: Some("a".to_string()),
        strongest: None,
        sensitive_paths: Vec::new(),
        review_modes: std::collections::HashMap::new(),
        models,
    };

    assert_eq!(
        manifest.resolve_model("a"),
        ("vendor:a".to_string(), Some(0.1), Some(1))
    );
    assert_eq!(
        manifest.resolve_model("vendor:a"),
        ("vendor:b".to_string(), Some(0.9), Some(9))
    );
}

// -------------------------------------------------------------------------
// 4. build_tool_description
// -------------------------------------------------------------------------

#[test]
fn test_build_tool_description_lists_all_aliases_and_roles() {
    let manifest = ModelManifest::default();
    let desc = manifest.build_tool_description();

    assert!(
        desc.starts_with("Available model aliases and their roles:"),
        "description must open with the catalog header: {desc}"
    );

    for (alias, def) in &manifest.models {
        let expected_role = def.role.as_deref().expect("default models have roles");
        let expected_line = format!("- `{alias}` (id: `{}`): {expected_role}", def.id);
        assert!(
            desc.contains(&expected_line),
            "description must contain {expected_line:?}:\n{desc}"
        );
    }

    // Header + one line per alias.
    assert_eq!(
        desc.lines().count(),
        manifest.models.len() + 1,
        "description must have exactly one bullet per alias:\n{desc}"
    );
    assert_eq!(
        desc.matches("- `combo:ninja`").count(),
        0,
        "bullets are keyed by alias, not by id:\n{desc}"
    );
}

#[test]
fn test_build_tool_description_uses_role_fallback_when_missing() {
    let mut models = HashMap::new();
    models.insert(
        "bare".to_string(),
        definition("vendor:bare", None, Some(0.4), Some(5)),
    );
    models.insert(
        "documented".to_string(),
        definition("vendor:documented", Some("Writes docs."), None, None),
    );
    let manifest = ModelManifest {
        default: None,
        strongest: None,
        sensitive_paths: Vec::new(),
        review_modes: std::collections::HashMap::new(),
        models,
    };

    let desc = manifest.build_tool_description();

    assert!(
        desc.contains("- `bare` (id: `vendor:bare`): Autonomous subagent\n"),
        "missing role must fall back to a generic description:\n{desc}"
    );
    assert!(
        desc.contains("- `documented` (id: `vendor:documented`): Writes docs.\n"),
        "declared role must be rendered verbatim:\n{desc}"
    );
    assert_eq!(desc.lines().count(), 3);
}

#[test]
fn test_build_tool_description_without_models_is_header_only() {
    let manifest = parse_manifest("default: ninja");
    assert_eq!(
        manifest.build_tool_description(),
        "Available model aliases and their roles:\n"
    );
}

#[test]
fn test_build_tool_description_covers_custom_manifest() {
    let manifest = parse_manifest(
        r#"
default: a
models:
  a:
    id: vendor:a
    role: "Role A."
  b:
    id: vendor:b
    role: "Role B."
"#,
    );

    let desc = manifest.build_tool_description();
    for line in [
        "- `a` (id: `vendor:a`): Role A.",
        "- `b` (id: `vendor:b`): Role B.",
    ] {
        assert!(
            desc.contains(line),
            "description must contain {line:?}:\n{desc}"
        );
    }
    assert!(
        !desc.contains("ninja"),
        "custom manifest must not leak defaults:\n{desc}"
    );
}

// ----------
// 5. validate() / normalize()
//
// `validate()` reports what is wrong with a manifest, `normalize()` is the
// fixup that makes it serveable, and the sanitizers close the two request-side
// ingresses. Each rule is a row: the manifest, the warning that must be
// reported, and the value `normalize()` must produce.
// ----------

/// One row per rule: `(case, model yaml, expected warning, temperature after
/// normalization, max_turns after normalization)`.
type WarningCase = (
    &'static str,
    &'static str,
    &'static str,
    Option<f32>,
    Option<usize>,
);

fn warning_cases() -> Vec<WarningCase> {
    vec![
        (
            "dangling default",
            "default: ghost\nmodels:\n  solo:\n    id: combo:solo\n",
            "default model \"ghost\" not found in models; it will be ignored and the built-in \
             fallback used (set DEFAULT_MODEL to override)",
            None,
            None,
        ),
        (
            "empty id",
            "models:\n  blank:\n    id: \"\"\n",
            "model \"blank\": id cannot be empty",
            None,
            None,
        ),
        (
            "whitespace-only id",
            "models:\n  padded:\n    id: \"   \"\n",
            "model \"padded\": id cannot be empty",
            None,
            None,
        ),
        (
            "temperature above range",
            "models:\n  hot:\n    id: combo:hot\n    temperature: 2.5\n",
            "model \"hot\": temperature 2.5 is outside [0, 2]; clamped to that range",
            Some(2.0),
            None,
        ),
        (
            "temperature below range",
            "models:\n  cold:\n    id: combo:cold\n    temperature: -0.1\n",
            "model \"cold\": temperature -0.1 is outside [0, 2]; clamped to that range",
            Some(0.0),
            None,
        ),
        (
            "non-finite temperature",
            "models:\n  broken:\n    id: combo:broken\n    temperature: .nan\n",
            "model \"broken\": temperature NaN is not a finite number; replaced with the provider \
             default",
            None,
            None,
        ),
        (
            "infinite temperature",
            "models:\n  endless:\n    id: combo:endless\n    temperature: .inf\n",
            "model \"endless\": temperature inf is not a finite number; replaced with the provider \
             default",
            None,
            None,
        ),
        (
            "zero max_turns",
            "models:\n  halt:\n    id: combo:halt\n    max_turns: 0\n",
            "model \"halt\": max_turns must be greater than 0; replaced with 100",
            None,
            Some(DEFAULT_MAX_TURNS),
        ),
        (
            "max_turns above the runtime limit",
            "models:\n  greedy:\n    id: combo:greedy\n    max_turns: 10000\n",
            "model \"greedy\": max_turns 10000 exceeds the runtime limit 500; clamped to that \
             limit",
            None,
            Some(MAX_TURNS_LIMIT),
        ),
    ]
}

#[test]
fn test_validate_reports_each_rule_and_normalize_repairs_it() {
    for (case, yaml, expected_warning, expected_temp, expected_turns) in warning_cases() {
        let manifest = parse_manifest(yaml);

        assert_eq!(
            manifest.validate(),
            vec![expected_warning.to_string()],
            "case `{case}` must report exactly its warning"
        );

        let normalized = manifest.normalize();
        // Every warning but the empty-`id` one has a mechanical fixup, so a
        // normalized manifest reports nothing except an unrepairable id.
        let residual: Vec<String> = normalized
            .validate()
            .into_iter()
            .filter(|w| !w.ends_with("id cannot be empty"))
            .collect();
        assert!(
            residual.is_empty(),
            "case `{case}`: normalization must repair everything fixable: {residual:?}"
        );
        assert_eq!(
            normalized.clone().normalize().validate(),
            normalized.validate(),
            "case `{case}`: normalization must be idempotent"
        );

        if expected_warning.starts_with("model \"") {
            let alias = expected_warning
                .split("model \"")
                .nth(1)
                .and_then(|rest| rest.split('"').next())
                .expect("a per-alias warning names its alias");
            let def = normalized.models.get(alias).unwrap_or_else(|| {
                panic!("case `{case}`: alias `{alias}` must survive normalization")
            });

            assert_eq!(def.temperature, expected_temp, "case `{case}`: temperature");
            assert_eq!(def.max_turns, expected_turns, "case `{case}`: max_turns");
            if expected_warning.ends_with("id cannot be empty") {
                // Nothing to substitute for a missing id: the entry is still
                // served, with the value the user wrote.
                assert_eq!(def.id.trim(), "", "case `{case}`: id is left as written");
            }
        } else {
            // The `default` rule is about `manifest.default`, not an entry.
            assert_eq!(
                normalized.default, None,
                "case `{case}`: default is dropped"
            );
            assert_eq!(
                normalized.models.len(),
                1,
                "case `{case}`: entries are kept"
            );
        }
    }
}

#[test]
fn test_validate_accepts_manifests_without_warnings() {
    // The built-in manifest, a fully populated custom one, boundary
    // temperatures, an unset budget, and the documented "models.yaml is
    // optional" shapes: none of them are validation errors.
    let sane = parse_manifest(
        r#"
default: turbo
models:
  turbo:
    id: combo:turbo
    role: "Blazing fast executor."
    temperature: 0.05
    max_turns: 12
  oracle:
    id: combo:oracle
    temperature: 1.75
    max_turns: 250
  edge:
    id: combo:edge
    temperature: 2.0
  bare:
    id: vendor:bare
"#,
    );
    let no_default = parse_manifest("models:\n  lone:\n    id: vendor:lone\n");
    let empty = parse_manifest("{}");

    for (case, manifest) in [
        ("built-in", ModelManifest::default()),
        ("custom", sane),
        ("without default", no_default),
        ("empty catalog", empty),
    ] {
        assert!(
            manifest.validate().is_empty(),
            "case `{case}` must not warn: {:?}",
            manifest.validate()
        );
    }
}

#[test]
fn test_validate_accepts_a_padded_default_that_names_an_alias() {
    // `default` is matched against the alias keys, trimmed the same way the
    // `id` check trims, so an over-indented `default:` is not dangling.
    let manifest = parse_manifest("default: \" ninja \"\nmodels:\n  ninja:\n    id: combo:ninja\n");

    assert!(
        manifest.validate().is_empty(),
        "a padded default that trims onto a known alias is fine: {:?}",
        manifest.validate()
    );
}

#[test]
fn test_validate_warnings_are_sorted_and_repeatable() {
    // Ordering is a real property, not an accident of a single-alias fixture:
    // per-alias warnings come back in sorted alias order, before the catalog is
    // normalized, and repeated calls are identical.
    let manifest = parse_manifest(
        r#"
default: ghost
models:
  delta:
    id: ""
    temperature: 9.5
    max_turns: 0
  beta:
    id: combo:beta
    temperature: .inf
  alpha:
    id: combo:alpha
    max_turns: 10000
  gamma:
    id: combo:gamma
"#,
    );

    let expected = [
        "default model \"ghost\" not found in models; it will be ignored and the built-in \
         fallback used (set DEFAULT_MODEL to override)",
        "model \"alpha\": max_turns 10000 exceeds the runtime limit 500; clamped to that limit",
        "model \"beta\": temperature inf is not a finite number; replaced with the provider \
         default",
        "model \"delta\": id cannot be empty",
        "model \"delta\": temperature 9.5 is outside [0, 2]; clamped to that range",
        "model \"delta\": max_turns must be greater than 0; replaced with 100",
    ];
    let expected: Vec<String> = expected.iter().map(|w| w.to_string()).collect();

    assert_eq!(manifest.validate(), expected);
    assert_eq!(
        manifest.validate(),
        expected,
        "repeated calls must not accumulate or reorder state"
    );
}

#[test]
fn test_normalize_keeps_serving_invalid_entries() {
    // Warnings are advisory: the offending entries stay in the catalog so a
    // user can see (and fix) what the manifest actually contains, just fixed.
    let manifest = parse_manifest(
        r#"
default: ghost
models:
  broken:
    id: combo:broken
    temperature: 5.0
    max_turns: 0
"#,
    )
    .normalize();

    assert_eq!(manifest.default, None, "the dangling default is dropped");
    assert_eq!(manifest.models.len(), 1, "the entry is still served");
    let broken = manifest.models.get("broken").expect("broken entry");
    assert_eq!(broken.temperature, Some(2.0));
    assert_eq!(broken.max_turns, Some(DEFAULT_MAX_TURNS));
    assert_eq!(
        manifest.build_tool_description().lines().count(),
        2,
        "the repaired entry is still listed in the tool description"
    );
}

#[test]
fn test_normalize_preserves_a_resolvable_default() {
    let manifest =
        parse_manifest("default: solo\nmodels:\n  solo:\n    id: combo:solo\n").normalize();

    assert_eq!(manifest.default, Some("solo".to_string()));
}

#[test]
fn test_sanitize_temperature_clamps_and_drops() {
    for (input, expected) in [
        (None, None),
        (Some(0.3), Some(0.3)),
        (Some(0.0), Some(0.0)),
        (Some(2.0), Some(2.0)),
        (Some(2.5), Some(2.0)),
        (Some(-0.1), Some(0.0)),
        (Some(f32::NAN), None),
        (Some(f32::INFINITY), None),
        (Some(f32::NEG_INFINITY), None),
    ] {
        assert_eq!(
            ModelManifest::sanitize_temperature(input),
            expected,
            "sanitize_temperature({input:?})"
        );
    }
}

#[test]
fn test_sanitize_max_turns_filters_zero_and_clamps() {
    for (requested, manifest, expected) in [
        (None, None, DEFAULT_MAX_TURNS),
        (Some(0), None, DEFAULT_MAX_TURNS),
        (None, Some(0), DEFAULT_MAX_TURNS),
        (Some(0), Some(50), 50),
        (None, Some(50), 50),
        (Some(12), Some(50), 12),
        (Some(usize::MAX), None, MAX_TURNS_LIMIT),
        (None, Some(10_000), MAX_TURNS_LIMIT),
    ] {
        assert_eq!(
            ModelManifest::sanitize_max_turns(requested, manifest),
            expected,
            "sanitize_max_turns({requested:?}, {manifest:?})"
        );
    }
}

#[test]
fn test_shipped_models_yaml_validates_cleanly() {
    // The repository's own `models.yaml` must validate cleanly, otherwise the
    // server would log warnings on every start.
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/models.yaml");
    let content = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("models.yaml must be readable: {e}"));
    let manifest = parse_manifest(&content).normalize();

    assert!(
        manifest.validate().is_empty(),
        "models.yaml must validate cleanly: {:?}",
        manifest.validate()
    );
}

// ----------
// 6. Catalog determinism
// ----------

#[test]
fn test_build_tool_description_is_deterministic_across_parses() {
    // `models` is a `HashMap` with a randomly seeded `RandomState`, so iterating
    // it directly produced a different bullet order for every instance. Parsing
    // the same YAML 200 times must now yield exactly one output.
    let yaml = r#"
default: a
models:
  a: { id: vendor:a, role: "Role A.", temperature: 0.1, max_turns: 5 }
  b: { id: vendor:b, role: "Role B.", temperature: 0.2, max_turns: 6 }
  c: { id: vendor:c, role: "Role C.", temperature: 0.3, max_turns: 7 }
  d: { id: vendor:d, role: "Role D.", temperature: 0.4, max_turns: 8 }
"#;

    let expected = parse_manifest(yaml).build_tool_description();
    for _ in 0..200 {
        assert_eq!(
            parse_manifest(yaml).build_tool_description(),
            expected,
            "identical manifests must render an identical catalog"
        );
    }
}

#[test]
fn test_build_tool_description_sorts_bullets_by_alias() {
    let manifest = parse_manifest(
        r#"
models:
  zeta: { id: vendor:zeta, role: "Z." }
  alpha: { id: vendor:alpha, role: "A." }
  mid: { id: vendor:mid, role: "M." }
"#,
    );

    let desc = manifest.build_tool_description();
    let aliases: Vec<&str> = desc
        .lines()
        .skip(1)
        .filter_map(|line| line.split('`').nth(1))
        .collect();

    assert_eq!(aliases, vec!["alpha", "mid", "zeta"]);
}

#[test]
fn test_build_tool_description_is_stable_across_manifest_instances() {
    // Two *different* instances holding the same entries (as happens when the
    // same models.yaml is parsed twice) must agree byte-for-byte.
    let yaml = r#"
models:
  a: { id: vendor:a, role: "Role A." }
  b: { id: vendor:b }
"#;

    let first = parse_manifest(yaml);
    let second = parse_manifest(yaml);
    assert_eq!(
        first.build_tool_description(),
        second.build_tool_description()
    );
}

#[test]
fn test_resolve_model_with_duplicate_ids_is_deterministic() {
    // Two aliases pointing at the same provider id with different overrides.
    // The documented policy is "first alias (sorted) wins", so the result must
    // not depend on the `HashMap` iteration order.
    let yaml = r#"
models:
  v:shared: { id: vendor:shared, temperature: 0.1 }
  z:shared: { id: vendor:shared, temperature: 0.9 }
"#;

    let expected = ("vendor:shared".to_string(), Some(0.1), None);
    for _ in 0..200 {
        assert_eq!(
            parse_manifest(yaml).resolve_model("vendor:shared"),
            expected,
            "duplicate ids must resolve to the first alias, deterministically"
        );
    }
}

#[test]
fn test_validate_warns_about_duplicate_model_ids() {
    let manifest = parse_manifest(
        r#"
models:
  a: { id: vendor:shared, role: "A.", temperature: 0.1 }
  b: { id: vendor:shared, role: "B.", temperature: 0.9 }
"#,
    );

    let warnings = manifest.validate();
    assert_eq!(
        warnings,
        vec![
            "duplicate model id \"vendor:shared\" shared by aliases \"a\", \"b\"; resolving the full id returns the first alias"
                .to_string()
        ]
    );

    // The documented policy is honoured by the resolver: "a" sorts first.
    assert_eq!(
        manifest.resolve_model("vendor:shared"),
        ("vendor:shared".to_string(), Some(0.1), None)
    );
}

#[test]
fn test_validate_duplicate_id_warning_is_deterministic() {
    let yaml = r#"
models:
  c: { id: vendor:x, role: "C." }
  a: { id: vendor:x, role: "A." }
  b: { id: vendor:y, role: "B." }
  d: { id: vendor:y, role: "D." }
"#;

    let expected = parse_manifest(yaml).validate();
    assert_eq!(
        expected.len(),
        2,
        "one warning per duplicated id: {expected:?}"
    );
    for _ in 0..200 {
        assert_eq!(parse_manifest(yaml).validate(), expected);
    }
}

#[test]
fn test_validate_does_not_warn_for_distinct_ids() {
    let manifest = parse_manifest(
        r#"
models:
  a: { id: vendor:a, role: "A." }
  b: { id: vendor:b, role: "B." }
"#,
    );
    assert!(manifest.validate().is_empty());
}

// ----------
// 5. Declarative execution policy (`policy:` block)
// ----------

#[test]
fn test_manifest_parses_optional_model_level_policy() {
    let manifest = parse_manifest(
        r#"
models:
  sealed:
    id: vendor:sealed
    role: "Isolated refactor."
    policy:
      network: "offline"
  runner:
    id: vendor:runner
    role: "Builds and tests."
    policy:
      network: "allow"
  bare:
    id: vendor:bare
"#,
    );

    let sealed = manifest.models["sealed"]
        .policy
        .as_ref()
        .expect("sealed policy");
    assert_eq!(sealed.network.as_ref().map(|n| n.as_str()), Some("offline"));

    let runner = manifest.models["runner"]
        .policy
        .as_ref()
        .expect("runner policy");
    assert_eq!(runner.network.as_ref().map(|n| n.as_str()), Some("allow"));

    assert_eq!(
        manifest.models["bare"].policy, None,
        "a pre-policy entry must still parse, with no policy declared"
    );
    assert!(
        manifest.validate().is_empty(),
        "documented policy values must not warn: {:?}",
        manifest.validate()
    );
}

#[test]
fn test_network_policy_resolves_by_alias_and_id() {
    let manifest = parse_manifest(
        r#"
models:
  sealed:
    id: vendor:sealed
    policy:
      network: "offline"
  runner:
    id: vendor:runner
    policy:
      network: "allow"
  bare:
    id: vendor:bare
"#,
    );

    // By alias.
    assert_eq!(
        manifest
            .network_policy("sealed")
            .as_ref()
            .map(|n| n.as_str()),
        Some("offline")
    );
    assert_eq!(
        manifest
            .network_policy("runner")
            .as_ref()
            .map(|n| n.as_str()),
        Some("allow")
    );
    // By full id.
    assert_eq!(
        manifest
            .network_policy("vendor:sealed")
            .as_ref()
            .map(|n| n.as_str()),
        Some("offline")
    );
    // A model with no policy block, and an unknown model, both yield `None`.
    assert_eq!(manifest.network_policy("bare"), None);
    assert_eq!(manifest.network_policy("unknown"), None);
}

#[test]
fn test_policy_is_validated_and_repaired_without_failing_the_load() {
    // A bad policy value must not make the manifest unparseable: the rest of the
    // catalog still has to be served, with the bad value named in a warning.
    let manifest = parse_manifest(
        r#"
default: sealed
models:
  sealed:
    id: vendor:sealed
    policy:
      network: "offine"
"#,
    );

    let warnings = manifest.validate();
    assert!(
        warnings.iter().any(|w| w.contains("offine")),
        "the misspelled network value must be reported: {warnings:?}"
    );

    let normalized = manifest.normalize();
    let policy = normalized.models["sealed"].policy.as_ref().expect("policy");
    assert_eq!(
        policy.network.as_ref().map(|n| n.as_str()),
        Some("offline"),
        "an unrecognised network must be repaired to the restrictive default"
    );
    assert!(
        normalized.validate().is_empty(),
        "normalize must repair everything it reports: {:?}",
        normalized.validate()
    );
}

#[test]
fn test_backward_compatible_manifest_has_no_policy_and_no_warnings() {
    // Exactly a pre-policy models.yaml: nothing here may change behaviour.
    let manifest = parse_manifest(
        r#"
default: ninja
models:
  ninja:
    id: combo:ninja
    role: "Fast subagent."
    temperature: 0.2
    max_turns: 150
  nerd:
    id: combo:nerd
    role: "Deep reasoner."
    temperature: 0.6
    max_turns: 200
"#,
    );

    assert!(manifest.models.values().all(|d| d.policy.is_none()));
    assert!(manifest.validate().is_empty(), "{:?}", manifest.validate());
    assert_eq!(
        manifest.resolve_model("ninja"),
        ("combo:ninja".to_string(), Some(0.2), Some(150)),
        "resolution must be unaffected by the absent policy"
    );
}

#[test]
fn test_shipped_models_yaml_is_valid_and_policy_annotated() {
    let manifest = ModelManifest::from_path(std::path::Path::new("models.yaml"))
        .expect("the shipped models.yaml must load");

    assert!(manifest.validate().is_empty(), "{:?}", manifest.validate());
    for alias in ["ninja", "nerd"] {
        let policy = manifest.models[alias]
            .policy
            .as_ref()
            .unwrap_or_else(|| panic!("{alias} must declare a policy"));
        assert!(
            policy.network.as_ref().is_some_and(|n| n.is_declared()),
            "{alias} must declare a known network policy"
        );
    }
}
