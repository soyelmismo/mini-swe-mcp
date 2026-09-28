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
//! way `src/mcp.rs` and `src/main.rs` consume the manifest.

use mini_swe_mcp::manifest::{
    CATALOG_CACHE_CAPACITY, ModelDefinition, ModelManifest, catalog_cache_len, clear_catalog_cache,
};
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
// 5. validate()
// ----------

#[test]
fn test_validate_default_manifest_has_no_warnings() {
    let warnings = ModelManifest::default().validate();

    assert!(
        warnings.is_empty(),
        "the built-in manifest must validate cleanly: {warnings:?}"
    );
}

#[test]
fn test_validate_custom_manifest_has_no_warnings() {
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
    temperature: 1.75
    max_turns: 250
  bare:
    id: vendor:bare
"#,
    );

    let warnings = manifest.validate();
    assert!(
        warnings.is_empty(),
        "a sane manifest must not produce warnings: {warnings:?}"
    );
}

#[test]
fn test_validate_empty_manifest_has_no_warnings() {
    // Neither a missing `default` nor an empty catalog is a validation error:
    // the server simply serves whatever the requester asked for.
    let manifest = parse_manifest("{}");

    assert!(manifest.validate().is_empty());
}

// --- default resolution ---

#[test]
fn test_validate_flags_unknown_default_model() {
    let manifest = parse_manifest(
        r#"
default: ninja
models:
  nerd:
    id: combo:nerd
"#,
    );

    assert_eq!(
        manifest.validate(),
        vec!["default model \"ninja\" not found in models".to_string()]
    );
}

#[test]
fn test_validate_accepts_default_pointing_at_existing_alias() {
    let mut models = HashMap::new();
    models.insert(
        "solo".to_string(),
        definition("vendor:solo", Some("Only model."), Some(0.5), Some(10)),
    );
    let manifest = ModelManifest {
        default: Some("solo".to_string()),
        models,
    };

    assert!(
        manifest.validate().is_empty(),
        "a default that resolves to a known alias is fine"
    );
}

#[test]
fn test_validate_without_default_reports_nothing() {
    let manifest = parse_manifest(
        r#"
models:
  lone:
    id: vendor:lone
"#,
    );

    assert!(manifest.validate().is_empty());
}

// --- empty ids ---

#[test]
fn test_validate_flags_empty_model_id() {
    let manifest = parse_manifest(
        r#"
models:
  blank:
    id: ""
"#,
    );

    assert_eq!(
        manifest.validate(),
        vec!["model \"blank\": id cannot be empty".to_string()]
    );
}

#[test]
fn test_validate_flags_whitespace_only_model_id() {
    let manifest = parse_manifest(
        r#"
models:
  padded:
    id: "   "
"#,
    );

    assert_eq!(
        manifest.validate(),
        vec!["model \"padded\": id cannot be empty".to_string()],
        "an id made of whitespace only is as unusable as an empty one"
    );
}

#[test]
fn test_validate_accepts_non_empty_model_id() {
    let mut models = HashMap::new();
    models.insert(
        "kept".to_string(),
        definition("vendor:kept", None, Some(0.3), Some(4)),
    );
    let manifest = ModelManifest {
        default: Some("kept".to_string()),
        models,
    };

    assert!(manifest.validate().is_empty());
}

// --- temperature ---

#[test]
fn test_validate_flags_out_of_range_temperatures() {
    let mut models = HashMap::new();
    models.insert(
        "hot".to_string(),
        definition("vendor:hot", None, Some(2.5), Some(10)),
    );
    models.insert(
        "cold".to_string(),
        definition("vendor:cold", None, Some(-0.1), Some(10)),
    );
    let manifest = ModelManifest {
        default: None,
        models,
    };

    let warnings = manifest.validate();
    assert_eq!(
        warnings.len(),
        2,
        "both bad temperatures must be reported: {warnings:?}"
    );

    assert!(
        warnings
            .contains(&"model \"hot\": temperature 2.5 must be between 0.0 and 2.0".to_string()),
        "unexpected warnings: {warnings:?}"
    );
    assert!(
        warnings
            .contains(&"model \"cold\": temperature -0.1 must be between 0.0 and 2.0".to_string()),
        "unexpected warnings: {warnings:?}"
    );
}

#[test]
fn test_validate_accepts_boundary_temperatures() {
    for temperature in [0.0_f32, 0.5, 1.0, 2.0] {
        let mut models = HashMap::new();
        models.insert(
            "ok".to_string(),
            definition("vendor:ok", None, Some(temperature), Some(1)),
        );
        let manifest = ModelManifest {
            default: Some("ok".to_string()),
            models,
        };

        assert!(
            manifest.validate().is_empty(),
            "temperature {temperature} is within the allowed [0.0, 2.0] range"
        );
    }
}

#[test]
fn test_validate_flags_nan_temperature() {
    let mut models = HashMap::new();
    models.insert(
        "broken".to_string(),
        definition("vendor:broken", None, Some(f32::NAN), Some(5)),
    );
    let manifest = ModelManifest {
        default: None,
        models,
    };

    let warnings = manifest.validate();
    assert_eq!(warnings.len(), 1, "NaN must be rejected: {warnings:?}");
    let warning = &warnings[0];
    assert!(
        warning.starts_with("model \"broken\": temperature "),
        "unexpected warning: {warning}"
    );
    assert!(
        warning.ends_with(" must be between 0.0 and 2.0"),
        "unexpected warning: {warning}"
    );
}

#[test]
fn test_validate_ignores_missing_temperature() {
    let mut models = HashMap::new();
    models.insert(
        "unset".to_string(),
        definition("vendor:unset", None, None, Some(3)),
    );
    let manifest = ModelManifest {
        default: Some("unset".to_string()),
        models,
    };

    assert!(
        manifest.validate().is_empty(),
        "an unset temperature means \"use the provider default\""
    );
}

// --- max_turns ---

#[test]
fn test_validate_flags_zero_max_turns() {
    let mut models = HashMap::new();
    models.insert(
        "halt".to_string(),
        definition("vendor:halt", None, Some(0.2), Some(0)),
    );
    let manifest = ModelManifest {
        default: Some("halt".to_string()),
        models,
    };

    assert_eq!(
        manifest.validate(),
        vec!["model \"halt\": max_turns must be greater than 0".to_string()]
    );
}

#[test]
fn test_validate_flags_zero_max_turns_parsed_from_yaml() {
    let manifest = parse_manifest(
        r#"
default: stop
models:
  stop:
    id: vendor:stop
    max_turns: 0
"#,
    );

    assert_eq!(
        manifest.validate(),
        vec!["model \"stop\": max_turns must be greater than 0".to_string()]
    );
}

#[test]
fn test_validate_ignores_missing_max_turns() {
    let mut models = HashMap::new();
    models.insert(
        "unbounded".to_string(),
        definition("vendor:unbounded", None, Some(0.4), None),
    );
    let manifest = ModelManifest {
        default: Some("unbounded".to_string()),
        models,
    };

    assert!(
        manifest.validate().is_empty(),
        "an unset turn budget means \"no manifest-imposed limit\""
    );
}

// --- combinations ---

#[test]
fn test_validate_reports_every_problem_in_one_pass() {
    let mut models = HashMap::new();
    // Valid entry: must never be reported.
    models.insert(
        "good".to_string(),
        definition("vendor:good", Some("Fine."), Some(0.4), Some(20)),
    );
    // Three independent problems on a single alias.
    models.insert("bad".to_string(), definition("", None, Some(3.0), Some(0)));
    let manifest = ModelManifest {
        default: Some("ghost".to_string()),
        models,
    };

    let warnings = manifest.validate();
    assert_eq!(
        warnings.len(),
        4,
        "all four problems must surface: {warnings:?}"
    );

    assert!(warnings.contains(&"default model \"ghost\" not found in models".to_string()));
    assert!(warnings.contains(&"model \"bad\": id cannot be empty".to_string()));
    assert!(
        warnings.contains(&"model \"bad\": temperature 3 must be between 0.0 and 2.0".to_string())
    );
    assert!(warnings.contains(&"model \"bad\": max_turns must be greater than 0".to_string()));
    assert!(
        !warnings.iter().any(|w| w.contains("good")),
        "a valid alias must stay silent: {warnings:?}"
    );
}

#[test]
fn test_validate_is_deterministic_for_a_single_alias() {
    let manifest = parse_manifest(
        r#"
default: missing
models:
  broken:
    id: ""
    temperature: 9.5
    max_turns: 0
"#,
    );

    let warnings = manifest.validate();
    assert_eq!(
        warnings,
        vec![
            "default model \"missing\" not found in models".to_string(),
            "model \"broken\": id cannot be empty".to_string(),
            "model \"broken\": temperature 9.5 must be between 0.0 and 2.0".to_string(),
            "model \"broken\": max_turns must be greater than 0".to_string(),
        ],
        "warnings are emitted in a stable order: default first, then per-alias checks"
    );

    // Repeated calls must not mutate or accumulate state.
    assert_eq!(manifest.validate(), warnings);
}

#[test]
fn test_validate_does_not_mutate_the_manifest() {
    let mut models = HashMap::new();
    models.insert(
        "kept".to_string(),
        definition("vendor:kept", Some("Role."), Some(0.2), Some(9)),
    );
    let manifest = ModelManifest {
        default: Some("kept".to_string()),
        models,
    };

    let before = manifest.resolve_model("kept");
    let _ = manifest.validate();
    let after = manifest.resolve_model("kept");

    assert_eq!(before, after, "validation is read-only");
    assert_eq!(before, ("vendor:kept".to_string(), Some(0.2), Some(9)));
}

#[test]
fn test_validate_does_not_filter_out_invalid_models() {
    // Warnings are advisory: the offending entries must still be served so a
    // user can see (and fix) what the manifest actually contains.
    let manifest = parse_manifest(
        r#"
default: ghost
models:
  broken:
    id: ""
    temperature: 5.0
    max_turns: 0
"#,
    );

    assert!(!manifest.validate().is_empty());

    assert!(manifest.models.contains_key("broken"));
    assert_eq!(manifest.default, Some("ghost".to_string()));
    assert_eq!(manifest.models.len(), 1);
    assert_eq!(
        manifest.build_tool_description().lines().count(),
        2,
        "the invalid entry is still listed in the tool description"
    );
}

#[test]
fn test_validate_is_available_on_the_shipped_models_yaml() {
    // The repository's own `models.yaml` must validate cleanly, otherwise the
    // server would log warnings on every start.
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/models.yaml");
    let content = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("models.yaml must be readable: {e}"));
    let manifest: ModelManifest =
        serde_yaml::from_str(&content).unwrap_or_else(|e| panic!("models.yaml must parse: {e}"));

    let warnings = manifest.validate();
    assert!(
        warnings.is_empty(),
        "models.yaml must validate cleanly: {warnings:?}"
    );
}

// ----------
// 6. Catalog memoization + determinism (audit opt_06)
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
fn test_build_tool_description_reuses_the_catalog_cache() {
    clear_catalog_cache();
    let manifest = ModelManifest::default();

    let cold = manifest.build_tool_description();
    let after_cold = catalog_cache_len();
    assert_eq!(
        after_cold,
        manifest.models.len(),
        "a cold render memoizes one row per model"
    );

    let warm = manifest.build_tool_description();
    assert_eq!(cold, warm, "cached rendering must be byte-identical");
    assert_eq!(
        catalog_cache_len(),
        after_cold,
        "repeating tools/list must not grow the cache"
    );
}

#[test]
fn test_build_tool_description_cache_key_includes_role_and_id() {
    clear_catalog_cache();
    // Same alias, different id/role: the memoized row must not be reused across
    // the variants, otherwise the catalog would serve a stale bullet.
    let variants = [
        ("vendor:one", "First."),
        ("vendor:two", "First."),
        ("vendor:one", "Second."),
    ];

    for (id, role) in variants {
        let mut models = HashMap::new();
        models.insert("shared".to_string(), definition(id, Some(role), None, None));
        let manifest = ModelManifest {
            default: None,
            models,
        };

        let expected =
            format!("Available model aliases and their roles:\n- `shared` (id: `{id}`): {role}\n");
        assert_eq!(manifest.build_tool_description(), expected);
    }
}

#[test]
fn test_build_tool_description_cache_is_bounded() {
    clear_catalog_cache();

    // More distinct rows than the cache capacity: the cache must stay bounded
    // instead of growing without limit, and still render every manifest
    // correctly.
    for i in 0..(CATALOG_CACHE_CAPACITY + 32) {
        let mut models = HashMap::new();
        models.insert(
            format!("alias{i}"),
            definition(&format!("vendor:id{i}"), Some("Role."), None, None),
        );
        let manifest = ModelManifest {
            default: None,
            models,
        };

        let desc = manifest.build_tool_description();
        assert!(
            desc.contains(&format!("- `alias{i}` (id: `vendor:id{i}`): Role.\n")),
            "row {i} must render correctly: {desc}"
        );
        assert!(
            catalog_cache_len() <= CATALOG_CACHE_CAPACITY,
            "cache must stay bounded, got {}",
            catalog_cache_len()
        );
    }
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
