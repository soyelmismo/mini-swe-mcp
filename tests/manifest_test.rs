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

use mini_swe_mcp::manifest::{ModelDefinition, ModelManifest};
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
    assert!(!manifest.models.contains_key("ninja"), "custom manifest replaces defaults");

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

    assert_eq!(manifest.default, None, "a manifest without `default:` yields None");
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
    assert_eq!(manifest.resolve_model("ninja"), ("ninja".to_string(), None, None));
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
        assert_eq!(id, unknown, "unknown model {unknown:?} must be passed through");
        assert_eq!(temperature, None, "unknown model {unknown:?} has no temperature override");
        assert_eq!(max_turns, None, "unknown model {unknown:?} has no turn override");
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
        assert!(desc.contains(line), "description must contain {line:?}:\n{desc}");
    }
    assert!(!desc.contains("ninja"), "custom manifest must not leak defaults:\n{desc}");
}
