//! Process-wide memoization of the rendered catalog, its header and its bullets.
//!
//! [`ModelManifest::build_tool_description`] is re-rendered on every MCP
//! `tools/list` request, and rendering a bullet used to allocate and format a
//! fresh `String`. A rendered bullet is a pure function of `(alias, id, role)`,
//! so it is memoized here rather than inside the manifest: [`ModelManifest`]
//! stays `&self`-clean, keeping its `Clone`/`Debug`/`Serialize` derives
//! untouched (no `serde(skip)` plumbing, no interior mutability leaking into
//! `Arc` clones).
//!
//! Three properties are load-bearing here:
//!
//! * **No copy on a hit.** A cached bullet is handed back as an `Arc<str>`
//!   (a refcount bump), so a warm `tools/list` performs no per-bullet string
//!   allocation. Misses render into a single pre-sized `String` and intern it.
//! * **One key allocation per lookup, not three.** The cache is keyed by a
//!   single length-prefixed `Box<str>` built from the three inputs, and
//!   `HashMap<Box<str>, _>` is probed through `Borrow<str>` with a borrowed
//!   `&str`, so the probe never clones the stored key. Building that probe key
//!   is the only per-lookup allocation, replacing the three `String` key
//!   allocations the previous `(String, String, String)` tuple performed.
//! * **Bounded.** The cache is hard-capped at [`CATALOG_CACHE_CAPACITY`]
//!   entries and cleared wholesale when the cap is reached, so a pathological
//!   caller (many synthetic manifests, e.g. in tests) cannot grow the process
//!   without bound.

use std::fmt::Write as _;
use std::sync::{Arc, OnceLock, RwLock};

use crate::cache::LruCache;

use super::DEFAULT_ROLE;
use super::types::ModelDefinition;

/// First line of the rendered catalog.
const CATALOG_HEADER: &str = "Available model aliases and their roles:\n";

/// Rendered catalog header, materialized once per process.
///
/// The header text is a compile-time constant, so the `OnceLock` does not save
/// measurable work on its own. It earns its place by making the *static* part of
/// the catalog a single shared, allocation-owned value of the same kind as the
/// memoized bullets: [`ModelManifest::build_tool_description`] then assembles
/// the catalog from one shared header plus N shared rows, and any future header
/// that has to be *built* (a version line, a generated index) initializes here
/// exactly once instead of once per `tools/list` request.
static CATALOG_HEADER_ARC: OnceLock<Arc<str>> = OnceLock::new();

/// Shared, once-initialized render of [`CATALOG_HEADER`].
///
/// Returns `&'static` so the shared value is reachable without re-taking a lock
/// and is only ever initialized once, no matter how many threads render.
pub(super) fn catalog_header() -> &'static Arc<str> {
    CATALOG_HEADER_ARC.get_or_init(|| Arc::from(CATALOG_HEADER))
}

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

/// Identity of one rendered catalog bullet: everything a bullet depends on,
/// encoded into a single owned, length-prefixed string so the key is one
/// allocation and can be probed by `&str` without cloning it.
type CatalogRowKey = Box<str>;

/// Build the collision-free cache key for `(alias, id, role)`.
///
/// Each component is prefixed with its byte length and a `':'`, so the encoding
/// is injective: no two distinct triples can produce the same key, regardless
/// of the characters (including `:` or empty strings) inside the fields.
fn catalog_row_key(alias: &str, id: &str, role: &str) -> String {
    let mut key = String::with_capacity(alias.len() + id.len() + role.len() + 24);
    for field in [alias, id, role] {
        // The decimal length plus its own `:` bounds each field exactly, so the
        // concatenation is unambiguously decodable — no reserved separator and
        // therefore no chance two different inputs alias to the same key.
        let _ = write!(key, "{}:", field.len());
        key.push_str(field);
    }
    key
}

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
/// instead of `format!` plus `push_str`; hits clone an `Arc<str>` (a refcount
/// bump) instead of the whole catalog `String`, which is what makes repeated
/// `tools/list` calls cheap. The role a bullet renders is the *effective* role
/// (falling back to [`DEFAULT_ROLE`]), so the key and the rendered text always
/// agree.
pub(super) fn catalog_row(alias: &str, def: &ModelDefinition) -> Arc<str> {
    let role = def.role.as_deref().unwrap_or(DEFAULT_ROLE);
    let key = catalog_row_key(alias, &def.id, role).into_boxed_str();

    // The read path refreshes recency under a *write* lock so a hot, frequently
    // re-rendered manifest keeps its bullets: an LRU that never learns which keys
    // are hot will eventually evict the hottest one. (A `peek` under a shared read
    // lock would allow parallel hits but would leave the hot key looking cold, and
    // it would then be evicted despite being re-read on every `tools/list`.)
    if let Ok(mut cache) = catalog_row_cache().write()
        && let Some(row) = cache.get(&key)
    {
        return Arc::clone(row);
    }

    let mut row =
        String::with_capacity(CATALOG_ROW_OVERHEAD + alias.len() + def.id.len() + role.len());
    let _ = writeln!(row, "- `{alias}` (id: `{}`): {role}", def.id);
    let row: Arc<str> = Arc::from(row.as_str());

    if let Ok(mut cache) = catalog_row_cache().write() {
        // The cache evicts its own least-recently-used entry in O(1) once it is
        // full; there is no bulk `clear()` and no capacity check here.
        cache.insert(key, Arc::clone(&row));
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
pub(crate) static TEST_CACHE_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

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
        let _guard = TEST_CACHE_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        clear_catalog_cache();

        let hot = definition("vendor:hot".to_string());
        let hot_key = catalog_row_key("hot-alias", &hot.id, "Role.").into_boxed_str();

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
                    &*catalog_row("hot-alias", &hot),
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
        let _guard = TEST_CACHE_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        clear_catalog_cache();
        for i in 0..(CATALOG_CACHE_CAPACITY * 10) {
            let def = definition(format!("vendor:id{i}"));
            let alias = format!("alias{i}");
            assert_eq!(
                &*catalog_row(&alias, &def),
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
        let _guard = TEST_CACHE_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        clear_catalog_cache();
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
                            &*catalog_row(&alias, &def),
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
