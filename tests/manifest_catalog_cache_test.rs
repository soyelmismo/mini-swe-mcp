//! Integration tests for the process-wide catalog memoization cache.
//!
//! These live in their own test binary, separate from `manifest_test.rs`, on
//! purpose. The cache behind [`catalog_cache_len`] is a *single* process-wide
//! `static`, so its cardinality is only observable by a thread that owns it. In
//! `manifest_test.rs` every rendered-output test inserts rows into that same map
//! from parallel test threads, which makes an exact entry count inherently
//! racy — that is the flake this file removes by owning the cache exclusively.
//!
//! Within this binary the tests still mutate shared state (`clear_catalog_cache`
//! followed by a count), so they share one mutex and it is now sufficient.

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
        policy: None,
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

/// The cache key is a single string built from `(alias, id, role)`, so it is
/// length-prefixed rather than joined on a reserved separator. This is the
/// regression test for that decision: a collision would be a *silent*
/// correctness bug, because the catalog would advertise one model's id and
/// role under another model's alias, and no count-based assertion would see it.
///
/// The pairs below are chosen to collide under a naive `":"`/`"\0"` join
/// (`("a", "bc", "")` vs `("ab", "c", "")`, `("a", ":b", "c")` vs
/// `("a", "b:c", "")`, a field that itself looks like a `len:` prefix, ...).
/// Every one must still render its own bullet.
#[test]
fn test_cache_key_does_not_collide_on_adversarial_fields() {
    let _guard = CACHE_TEST_MUTEX.lock().unwrap();
    clear_catalog_cache();

    let triples = [
        ("a", "b", "c"),
        ("a", "bc", ""),
        ("ab", "c", ""),
        ("a", ":b", "c"),
        ("a", "b:c", ""),
        ("", "", ""),
        ("1:a", "b", "c"),
        ("a", "1:b", "c"),
        ("a", "b", "1:c"),
        (":::", ":::", ":::"),
    ];

    for (alias, id, role) in triples {
        let mut models = HashMap::new();
        // A role-less entry is the one place the renderer substitutes
        // `DEFAULT_ROLE`, so the effective role in the key is not the raw one.
        let (id, role) = if role.is_empty() {
            (id.to_string(), None)
        } else {
            (id.to_string(), Some(role.to_string()))
        };
        models.insert(
            alias.to_string(),
            definition(&id, role.as_deref(), None, None),
        );
        let manifest = ModelManifest {
            default: None,
            models,
        };

        let expected_role = role.as_deref().unwrap_or("Autonomous subagent");
        let expected = format!(
            "Available model aliases and their roles:\n- `{alias}` (id: `{id}`): {expected_role}\n"
        );
        assert_eq!(
            manifest.build_tool_description(),
            expected,
            "(`{alias}`, `{id}`, {expected_role:?}) must not alias onto another entry"
        );
    }

    // 10 distinct triples, 3 of which render the shared role fallback text but
    // under different keys, so the cache must hold one entry per triple.
    assert_eq!(
        catalog_cache_len(),
        triples.len(),
        "every distinct (alias, id, role) triple needs its own cache entry"
    );
}

/// A warm `tools/list` must not re-render a bullet: the memoized rows are
/// `Arc<str>`, so a second render allocates only the returned `String` itself
/// and copies nothing twice. The observable contract here is that repeating the
/// render is byte-identical and adds no cache entries; the allocation property
/// itself is asserted in the unit tests, where the `Arc` identity is reachable.
#[test]
fn test_repeated_renders_are_stable_and_add_no_entries() {
    let _guard = CACHE_TEST_MUTEX.lock().unwrap();
    clear_catalog_cache();

    let manifest = ModelManifest::default();
    let first = manifest.build_tool_description();
    let after_first = catalog_cache_len();

    for _ in 0..100 {
        assert_eq!(
            manifest.build_tool_description(),
            first,
            "every tools/list render must be byte-identical"
        );
    }
    assert_eq!(
        catalog_cache_len(),
        after_first,
        "100 further renders must not add a single cache entry"
    );
}
