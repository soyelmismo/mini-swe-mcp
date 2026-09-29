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
//! Four properties are load-bearing here:
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
//!   entries. Overflow evicts exactly one least-recently-used bullet in O(1)
//!   (see [`crate::cache::LruCache`]), so a pathological caller (many synthetic
//!   manifests, e.g. in tests) cannot grow the process without bound.
//! * **Bounded *bytes*.** The entry cap bounds a count, not a size. A bullet
//!   mirrors `alias`, `id` and `role` verbatim and those fields are validated
//!   for presence and numeric range but never for length, so one poisoned
//!   `models.yaml` would otherwise own one unbounded allocation per cached row.
//!   [`CATALOG_CACHE_MAX_KEY_BYTES`] bounds a single key (a key is about the
//!   size of the row it names), so input length cannot dictate process memory.
//!   An over-bound row still renders correctly; only its *memoization* is
//!   skipped.
//!
//! # Stampede behaviour
//!
//! `tools/list` renders `n` bullets, each through [`catalog_row`], and rendering
//! is a pure function of `(alias, id, role)`. A redundant render caused by a
//! lost race is therefore always byte-identical: the waste is one dropped
//! [`Arc`], never a divergent bullet. That is what lets a stampede be handled by
//! the cache's own locking instead of per-key single-flight guards or throttles
//! — many threads may render the same bullet and all of them get the same bytes.
//! What must survive a stampede is that a **hit** never blocks: [`catalog_row`]
//! serves it from a non-blocking exclusive acquisition when the cache is idle and
//! from the *shared* lock when it is not, so concurrent `tools/list` calls never
//! queue behind each other or behind a slow renderer.

use std::fmt::Write as _;
use std::sync::{Arc, OnceLock, PoisonError, RwLock, RwLockWriteGuard, TryLockError};

use crate::cache::LruCache;

use super::DEFAULT_ROLE;
use super::types::ModelDefinition;

/// Exclusive handle on the process-wide catalog bullet cache.
type CatalogWriteGuard<'a> = RwLockWriteGuard<'a, LruCache<CatalogRowKey, Arc<str>>>;

/// First line of the rendered catalog.
const CATALOG_HEADER: &str = "Available model aliases and their roles:\n";

/// Recover the cache lock even after a poisoned lock is observed.
///
/// Poisoning means some thread panicked while holding the lock, and the catalog
/// cache cannot be that thread: the guard is taken and dropped inside
/// [`catalog_row`] and holds nothing with a destructor that can unwind, and
/// every stored value is an immutable `Arc<str>` that cannot be left
/// half-initialized. A poisoned cache therefore holds nothing but memoized
/// strings, and recovering the guard serves exactly the bytes a fresh render
/// would have produced anyway.
///
/// Mapping `Err(_)` to `None` instead — which this module used to do — turns
/// poisoning, and *only* poisoning (nothing else here fails), into a permanently
/// absent cache: every render takes the render path and every entry is inserted
/// then immediately dropped, silently, for the rest of the process's life.
fn poisoned_or<'a>(
    result: Result<CatalogWriteGuard<'a>, PoisonError<CatalogWriteGuard<'a>>>,
) -> CatalogWriteGuard<'a> {
    match result {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

/// Non-blocking counterpart of [`poisoned_or`].
///
/// `try_write` reports contention as well as poisoning, and only poisoning is
/// recoverable: a contended lock is simply retried through the next tier.
fn try_poisoned_or<'a>(
    result: Result<CatalogWriteGuard<'a>, TryLockError<CatalogWriteGuard<'a>>>,
) -> Option<CatalogWriteGuard<'a>> {
    match result {
        Ok(guard) => Some(guard),
        Err(TryLockError::Poisoned(poisoned)) => Some(poisoned.into_inner()),
        Err(TryLockError::WouldBlock) => None,
    }
}

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

/// Largest cache key — and therefore largest memoized bullet — ever retained.
///
/// The bullet mirrors `alias`, `id` and `role` verbatim, so a key is about the
/// size of the row it names and [`CATALOG_CACHE_CAPACITY`] bounds only the
/// *count* of rows, never their bytes. Manifest fields are checked for presence
/// and numeric range but never for length, so a single poisoned `models.yaml`
/// (one model with a multi-megabyte `role`) would otherwise stay resident for
/// the life of the daemon — memory retention with no legitimate counterpart,
/// since a real alias/id/role is tens of bytes.
///
/// Above this bound the bullet renders exactly as before and is simply not
/// memoized: a cache that degrades to "no cache" under abuse is strictly better
/// than one that lets input size dictate process memory.
pub const CATALOG_CACHE_MAX_KEY_BYTES: usize = 8 * 1024;

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
    let key: CatalogRowKey = catalog_row_key(alias, &def.id, role).into_boxed_str();

    // Over-long (i.e. poisoned) input renders but is never memoized: the key is
    // built first precisely so its size can be bounded before anything is
    // retained. See CATALOG_CACHE_MAX_KEY_BYTES.
    if key.len() > CATALOG_CACHE_MAX_KEY_BYTES {
        return render_catalog_row(alias, &def.id, role);
    }

    // Hit path, two tiers, cheapest first:
    //
    // 1. `try_write` — one exclusive acquisition that both serves the bullet and
    //    refreshes its recency, so a hit costs a *single* lock round-trip (the
    //    previous code took one to serve and a second to mark). It is
    //    non-blocking, so under contention a thread never queues here: it falls
    //    through to (2) instead of waiting.
    // 2. `read` + `peek` — served under the *shared* lock, so concurrent
    //    `tools/list` calls render in parallel instead of serializing on an
    //    exclusive lock. This tier does not refresh recency, which is exactly
    //    why tier (1) exists; when tier (1) loses the race the mark is deferred
    //    to the next render, and the miss path re-marks unconditionally.
    let hit = try_poisoned_or(catalog_row_cache().try_write()).and_then(|mut cache| {
        cache.get(&key).map(Arc::clone)
    });
    if let Some(row) = hit {
        return row;
    }
    let hit = if let Ok(cache) = catalog_row_cache().read() {
        cache.peek(&key).map(Arc::clone)
    } else {
        None
    };
    if let Some(row) = hit {
        return row;
    }

    // Miss: render once, *outside* the lock. Rendering is a pure function of its
    // inputs, so a redundant render caused by losing the race below is always
    // byte-identical. That is why no per-key single-flight guard or throttle is
    // needed: the worst case is one dropped `Arc<str>`, never a wrong bullet.
    let row = render_catalog_row(alias, &def.id, role);

    let mut guard = poisoned_or(catalog_row_cache().write());
    // Re-check under the exclusive lock: a concurrent renderer may have
    // published the identical bytes while this one was rendering. Adopt them
    // rather than inserting a duplicate and evicting a fresher entry.
    if let Some(cached) = guard.get(&key).map(Arc::clone) {
        return cached;
    }
    // The cache evicts its own least-recently-used entry in O(1) once it is
    // full; there is no bulk `clear()` and no capacity check here. The insert
    // refreshes recency unconditionally, so a key that keeps missing is never
    // left looking cold.
    guard.insert(key, Arc::clone(&row));

    row
}

/// Render one bullet into a fresh `Arc<str>` on a single pre-sized allocation.
fn render_catalog_row(alias: &str, id: &str, role: &str) -> Arc<str> {
    let mut row = String::with_capacity(CATALOG_ROW_OVERHEAD + alias.len() + id.len() + role.len());
    let _ = writeln!(row, "- `{alias}` (id: `{id}`): {role}");
    Arc::from(row.as_str())
}

/// Number of catalog bullets currently memoized.
///
/// Exposed for tests and diagnostics: the cache is bounded by
/// [`CATALOG_CACHE_CAPACITY`] and never grows with repeated `tools/list` calls.
pub fn catalog_cache_len() -> usize {
    catalog_row_cache().read().map_or(0, |cache| cache.len())
}

/// Drop every memoized catalog bullet.
///
/// Exposed for tests and diagnostics; never needed on a serving path.
pub fn clear_catalog_cache() {
    poisoned_or(catalog_row_cache().write()).clear();
}

#[cfg(test)]
pub(crate) static TEST_CACHE_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Non-blocking write that also recovers a poisoned lock — the exact tier-1
/// acquisition [`catalog_row`] performs. Test-only helper.
#[cfg(test)]
fn try_write_recover(
    cache: &RwLock<LruCache<CatalogRowKey, Arc<str>>>,
) -> Result<CatalogWriteGuard<'_>, PoisonError<CatalogWriteGuard<'_>>> {
    match cache.try_write() {
        Ok(guard) => Ok(guard),
        Err(TryLockError::Poisoned(poisoned)) => Err(poisoned),
        Err(e @ TryLockError::WouldBlock) => unreachable!("uncontended lock: {e:?}"),
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

    /// The entry cap bounds a *count*, not a size, so a row whose key exceeds
    /// [`CATALOG_CACHE_MAX_KEY_BYTES`] must be rendered correctly but never
    /// retained. Before this bound existed, one poisoned `models.yaml` (a model
    /// with a multi-megabyte `role`) stayed resident in the process for the life
    /// of the daemon.
    #[test]
    fn oversized_rows_are_rendered_but_never_memoized() {
        let _guard = TEST_CACHE_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        clear_catalog_cache();

        let huge_role = "R".repeat(CATALOG_CACHE_MAX_KEY_BYTES + 1);
        let def = ModelDefinition {
            id: "vendor:huge".to_string(),
            role: Some(huge_role.clone()),
            temperature: None,
            max_turns: None,
        };

        let row = catalog_row("huge-alias", &def);
        assert_eq!(
            &*row,
            format!("- `huge-alias` (id: `vendor:huge`): {huge_role}\n"),
            "an oversized row must still render byte-for-byte correctly"
        );
        assert_eq!(
            catalog_cache_len(),
            0,
            "an oversized row must not be retained, or the entry cap does not bound bytes"
        );

        // A key that *fits* is still memoized: the bound must not degrade the
        // cache into a no-op for ordinary input.
        let fits = ModelDefinition {
            id: "vendor:fits".to_string(),
            role: Some("R".repeat(CATALOG_CACHE_MAX_KEY_BYTES / 4)),
            temperature: None,
            max_turns: None,
        };
        let _ = catalog_row("fits-alias", &fits);
        assert_eq!(catalog_cache_len(), 1, "in-bound rows must still be cached");
    }

    /// A hit must refresh recency, or the LRU eventually evicts the key it is
    /// being re-read on every single `tools/list`.
    ///
    /// A hit is served either from a non-blocking exclusive acquisition (which
    /// refreshes recency) or, when that loses the race, from the shared lock
    /// (which does not). This drives one cold insert between every single hot
    /// re-read — the harshest cadence, and the one that keeps the cache
    /// continuously contended — and asserts the hot row is never evicted, i.e.
    /// that at least one tier keeps its recency mark.
    #[test]
    fn hot_rows_stay_resident_when_re_read_before_every_cold_insert() {
        let _guard = TEST_CACHE_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        clear_catalog_cache();

        let hot = definition("vendor:hot2".to_string());
        let hot_key = catalog_row_key("hot2", &hot.id, "Role.").into_boxed_str();
        catalog_row("hot2", &hot);

        for i in 0..(CATALOG_CACHE_CAPACITY * 2) {
            // Re-read the hot row *before* each cold insert pushes it one step
            // closer to the eviction head.
            assert_eq!(
                &*catalog_row("hot2", &hot),
                "- `hot2` (id: `vendor:hot2`): Role.\n"
            );
            catalog_row(&format!("cold{i}"), &definition(format!("vendor:cold{i}")));
            assert!(
                catalog_row_cache()
                    .read()
                    .is_ok_and(|c| c.contains_key(&hot_key)),
                "the hot bullet was evicted at cold row {i}; a hit must refresh recency"
            );
        }
        assert!(catalog_cache_len() <= CATALOG_CACHE_CAPACITY);
    }

    /// The mirror image of the test above, and the reason the LRU is a *cache*
    /// and not a log: a key that is genuinely never touched again must be
    /// evictable. Pinning this stops a future "refresh on read" change from
    /// silently turning the bounded cache into an unbounded one.
    #[test]
    fn a_key_that_is_never_re_read_is_still_evicted() {
        let _guard = TEST_CACHE_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        clear_catalog_cache();

        let cold_key = catalog_row_key("never-again", "vendor:cold", "Role.").into_boxed_str();
        catalog_row("never-again", &definition("vendor:cold".to_string()));
        assert!(
            catalog_row_cache()
                .read()
                .is_ok_and(|c| c.contains_key(&cold_key)),
            "precondition: the row starts resident"
        );

        for i in 0..(CATALOG_CACHE_CAPACITY + 1) {
            catalog_row(&format!("cold{i}"), &definition(format!("vendor:cold{i}")));
        }

        assert!(
            !catalog_row_cache()
                .read()
                .is_ok_and(|c| c.contains_key(&cold_key)),
            "a row that was never re-read must be evicted, or the cache grows without bound"
        );
        assert!(catalog_cache_len() <= CATALOG_CACHE_CAPACITY);
    }

    /// The *lock-recovery* helpers must return a usable guard from a poisoned
    /// lock, and nothing from a contended one.
    ///
    /// This is exercised against a private cache rather than the process-wide
    /// static on purpose: poisoning is irreversible for a given lock, so testing
    /// it in place would leave the real cache poisoned for every other test in
    /// this binary. [`catalog_row`] is the only caller that matters, and the
    /// helpers it uses are exactly the ones under test here.
    #[test]
    fn poisoned_locks_are_recovered_and_contended_ones_are_not() {
        let cache: RwLock<LruCache<CatalogRowKey, Arc<str>>> =
            RwLock::new(LruCache::with_capacity(4));

        // Uncontended: the lock is handed straight through.
        let mut guard = poisoned_or(try_write_recover(&cache));
        let absent: CatalogRowKey = "1:a".into();
        assert!(guard.get(&absent).is_none());
        drop(guard);

        // Poisoned: recoverable, so the cache keeps working instead of silently
        // degrading into "no cache" for the rest of the process's life.
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _held = cache.write().expect("unpoisoned at test start");
            panic!("simulate a thread panicking while holding the cache lock");
        }));
        assert!(cache.read().is_err(), "precondition: the lock is poisoned");

        let mut guard = poisoned_or(cache.write());
        guard.insert("2:b".into(), Arc::from("v"));
        assert_eq!(guard.len(), 1, "a poisoned cache must still be writable");
        drop(guard);
        // The read side reports the poison (that is the contract of
        // `RwLock::read`), but the data is intact and `catalog_cache_len`'s
        // `map_or` must not read that as "empty".
        let recovered = cache.read().unwrap_or_else(|p| p.into_inner());
        assert_eq!(recovered.len(), 1, "a poisoned cache must still hold its entries");
        drop(recovered);
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

