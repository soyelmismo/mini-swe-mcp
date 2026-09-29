//! Integration tests for the process-wide catalog memoization cache.
//!
//! These live in their own test binary, separate from `manifest_test.rs`, on
//! purpose. The cache behind [`catalog_cache_len`] is a *single* process-wide
//! `static`, so its cardinality is only observable by a thread that owns it. In
//! `manifest_test.rs` every rendered-output test inserts rows into that same map
//! from parallel test threads, which makes an exact entry count inherently
//! racy — that is the flake this file removes by owning the cache exclusively.
//!
//! Within this binary the three tests still mutate shared state (`clear_catalog_cache`
//! followed by a count), so they share one mutex and it is now sufficient.

use mini_swe_mcp::manifest::{
    ModelDefinition, ModelManifest, catalog_cache_len, clear_catalog_cache, CATALOG_CACHE_CAPACITY,
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

static CACHE_TEST_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
fn test_build_tool_description_reuses_the_catalog_cache() {
    let _guard = CACHE_TEST_MUTEX.lock().unwrap();
    clear_catalog_cache();
    let manifest = ModelManifest::default();

    let cold = manifest.build_tool_description();
    let after_cold = catalog_cache_len();
    assert!(
        after_cold >= manifest.models.len(),
        "a cold render memoizes at least one row per model"
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
    let _guard = CACHE_TEST_MUTEX.lock().unwrap();
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
    let _guard = CACHE_TEST_MUTEX.lock().unwrap();
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
