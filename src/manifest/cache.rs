//! Process-wide memoization of rendered catalog bullets.
//!
//! [`ModelManifest::build_tool_description`] is re-rendered on every MCP
//! `tools/list` request, and rendering a bullet allocates and formats a
//! `String`. A rendered bullet is a pure function of `(alias, id, role)`, so it
//! is memoized here rather than inside the manifest: [`ModelManifest`] stays
//! `&self`-clean, keeping its `Clone`/`Debug`/`Serialize` derives untouched (no
//! `serde(skip)` plumbing, no interior mutability leaking into `Arc` clones).

use std::fmt::Write as _;
use std::sync::{Arc, OnceLock, RwLock};

use crate::cache::LruCache;

use super::types::ModelDefinition;
use super::DEFAULT_ROLE;

/// First line of the rendered catalog.
pub(super) const CATALOG_HEADER: &str = "Available model aliases and their roles:\n";

/// Rough byte cost of one rendered bullet, excluding the alias, id and role
/// text. Used to pre-size the catalog `String` so it is allocated once.
pub(super) const CATALOG_ROW_OVERHEAD: usize = 48;

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
/// The storage is the crate's O(1) LRU cache (see [`crate::cache::LruCache`]) and
/// the handle is built behind a `OnceLock`, so the map is allocated exactly once
/// per process no matter how many threads race into the first `tools/list`.
///
/// This replaces the previous `HashMap` + `if len >= CAP { clear() }` scheme,
/// which was O(n) per overflow *and* threw away the whole working set: a
/// long-lived daemon alternating between two manifests re-rendered every bullet
/// on each crossing of the cap. [`LruCache`] instead evicts exactly the
/// least-recently-used bullet in O(1), so the manifest that is actually being
/// served stays warm.
static CATALOG_ROW_CACHE: OnceLock<RwLock<LruCache<CatalogRowKey, Arc<str>>>> = OnceLock::new();

/// The process-wide catalog bullet cache, created on first use.
fn catalog_row_cache() -> &'static RwLock<LruCache<CatalogRowKey, Arc<str>>> {
    CATALOG_ROW_CACHE.get_or_init(|| {
        RwLock::new(LruCache::with_capacity(CATALOG_CACHE_CAPACITY))
    })
}

/// Render (or reuse) the bullet for one model entry.
///
/// Cache misses build the row with `writeln!` on a single pre-sized allocation
/// instead of `format!` plus `push_str`; hits clone an `Arc<str>` instead of the
/// whole catalog `String`, which is what makes repeated `tools/list` calls cheap.
pub(super) fn catalog_row(alias: &str, def: &ModelDefinition) -> String {
    let key = (
        alias.to_string(),
        def.id.clone(),
        def.role.clone().unwrap_or_default(),
    );

    // The read path refreshes recency under a *write* lock so a hot, frequently
    // re-rendered manifest keeps its bullets: an LRU that never learns which keys
    // are hot will eventually evict the hottest one. (A `peek` under a shared read
    // lock would allow parallel hits but would leave the hot key looking cold, and
    // it would then be evicted despite being re-read on every `tools/list`.)
    if let Ok(mut cache) = catalog_row_cache().write()
        && let Some(row) = cache.get(&key)
    {
        return row.to_string();
    }

    let role = def.role.as_deref().unwrap_or(DEFAULT_ROLE);
    let mut row =
        String::with_capacity(CATALOG_ROW_OVERHEAD + alias.len() + def.id.len() + role.len());
    let _ = writeln!(row, "- `{alias}` (id: `{}`): {role}", def.id);

    if let Ok(mut cache) = catalog_row_cache().write() {
        // The cache evicts its own least-recently-used entry in O(1) once it is
        // full; there is no bulk `clear()` and no capacity check here.
        cache.insert(key, Arc::from(row.as_str()));
    }

    row
}

/// Number of catalog bullets currently memoized.
///
/// Exposed for tests and diagnostics: the cache is bounded by
/// [`CATALOG_CACHE_CAPACITY`] and never grows with repeated `tools/list` calls.
pub fn catalog_cache_len() -> usize {
    catalog_row_cache()
        .read()
        .map(|cache| cache.len())
        .unwrap_or_default()
}

/// Drop every memoized catalog bullet.
///
/// Exposed for tests and diagnostics; never needed on a serving path.
pub fn clear_catalog_cache() {
    if let Ok(mut cache) = catalog_row_cache().write() {
        cache.clear();
    }
}

#[cfg(test)]
mod eviction_tests {
    use super::*;
    use crate::manifest::ModelDefinition;

    fn definition(id: String) -> ModelDefinition {
        ModelDefinition {
            id,
            role: Some("Role.".to_string()),
            temperature: None,
            max_turns: None,
        }
    }

    /// A repeatedly-rendered manifest must keep its memoized bullets across a
    /// flood of foreign rows.
    ///
    /// The old storage dropped *every* entry on the first insert that reached the
    /// cap, so this scenario re-rendered the hot manifest on each crossing. With
    /// O(1) LRU eviction only the least-recently-used foreign bullet is dropped.
    #[test]
    fn hot_manifest_rows_survive_a_flood_of_cold_rows() {
        let hot = definition("vendor:hot".to_string());
        let hot_key = ("hot-alias".to_string(), hot.id.clone(), "Role.".to_string());

        catalog_row("hot-alias", &hot);
        assert!(
            catalog_row_cache()
                .read()
                .is_ok_and(|c| c.contains_key(&hot_key)),
            "the hot bullet must be memoized"
        );

        // A realistic hot key is re-read at a steady cadence while one-shot cold
        // keys stream past. Under a bulk `clear()` policy the entire cache —
        // including the hot row — is flushed on the *first* cap crossing, so the
        // hot row misses until the next re-render. Under O(1) LRU eviction the hot
        // row is re-read often enough to stay out of the eviction path and is
        // therefore resident on every single cold step.
        const HOT_EVERY: usize = 4;
        for i in 0..(CATALOG_CACHE_CAPACITY * 4) {
            let cold = definition(format!("vendor:cold{i}"));
            catalog_row(&format!("cold-alias{i}"), &cold);
            if i.is_multiple_of(HOT_EVERY) {
                assert_eq!(
                    catalog_row("hot-alias", &hot),
                    "- `hot-alias` (id: `vendor:hot`): Role.\n"
                );
            }
            // The hot row must be resident at *every* cold step, not just on the
            // steps where it happened to be re-rendered.
            assert!(
                catalog_row_cache()
                    .read()
                    .is_ok_and(|c| c.contains_key(&hot_key)),
                "the hot bullet was evicted at cold row {i} (len={}); a hot working set must not be flushed by cold one-shots", catalog_cache_len()
            );
        }

        assert!(
            catalog_row_cache()
                .read()
                .is_ok_and(|c| c.contains_key(&hot_key)),
            "the hot bullet must survive {0} cold rows with LRU eviction",
            CATALOG_CACHE_CAPACITY * 4
        );
        assert!(
            catalog_cache_len() <= CATALOG_CACHE_CAPACITY,
            "cache must stay bounded, got {}",
            catalog_cache_len()
        );
    }

    /// Eviction must be O(1) per insert and the cache must never exceed capacity,
    /// no matter how many distinct rows are pushed through it.
    #[test]
    fn flood_of_distinct_rows_stays_bounded() {
        for i in 0..(CATALOG_CACHE_CAPACITY * 10) {
            let def = definition(format!("vendor:id{i}"));
            let alias = format!("alias{i}");
            assert_eq!(
                catalog_row(&alias, &def),
                format!("- `{alias}` (id: `vendor:id{i}`): Role.\n")
            );
            assert!(
                catalog_cache_len() <= CATALOG_CACHE_CAPACITY,
                "cache exceeded capacity at row {i}: {}",
                catalog_cache_len()
            );
        }
    }

    /// The cache is a single process-wide `RwLock`; hammering it from many
    /// threads must not panic, lose rows, or breach the capacity bound.
    #[test]
    fn concurrent_catalog_renders_are_safe_and_bounded() {
        use std::sync::Arc;
        use std::sync::Barrier;
        use std::thread;

        const THREADS: usize = 8;
        const PER_THREAD: usize = CATALOG_CACHE_CAPACITY;

        let barrier = Arc::new(Barrier::new(THREADS));
        let handles: Vec<_> = (0..THREADS)
            .map(|t| {
                let barrier = Arc::clone(&barrier);
                thread::spawn(move || {
                    barrier.wait();
                    for i in 0..PER_THREAD {
                        let def = definition(format!("vendor:t{t}id{i}"));
                        let alias = format!("alias{t}-{i}");
                        assert_eq!(
                            catalog_row(&alias, &def),
                            format!("- `{alias}` (id: `vendor:t{t}id{i}`): Role.\n")
                        );
                        assert!(catalog_cache_len() <= CATALOG_CACHE_CAPACITY);
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().expect("catalog worker thread must not panic");
        }
    }
}
