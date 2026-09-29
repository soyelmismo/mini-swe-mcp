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

use std::collections::HashMap;
use std::fmt::Write as _;
use std::sync::{Arc, LazyLock, OnceLock, RwLock};

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
static CATALOG_ROW_CACHE: LazyLock<RwLock<HashMap<CatalogRowKey, Arc<str>>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

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
    let key = catalog_row_key(alias, &def.id, role);

    // Probe by borrowed `&str` (`Box<str>: Borrow<str>`) so a hit does not
    // need to clone the owned key; the owned key is only interned on a miss.
    if let Ok(cache) = CATALOG_ROW_CACHE.read()
        && let Some(row) = cache.get(key.as_str())
    {
        return Arc::clone(row);
    }

    let mut row =
        String::with_capacity(CATALOG_ROW_OVERHEAD + alias.len() + def.id.len() + role.len());
    let _ = writeln!(row, "- `{alias}` (id: `{}`): {role}", def.id);
    let row: Arc<str> = Arc::from(row.as_str());

    if let Ok(mut cache) = CATALOG_ROW_CACHE.write() {
        if cache.len() >= CATALOG_CACHE_CAPACITY {
            cache.clear();
        }
        cache.insert(key.into_boxed_str(), Arc::clone(&row));
    }

    row
}

/// Number of catalog bullets currently memoized.
///
/// Exposed for tests and diagnostics: the cache is bounded by
/// [`CATALOG_CACHE_CAPACITY`] and never grows with repeated `tools/list` calls.
pub fn catalog_cache_len() -> usize {
    CATALOG_ROW_CACHE
        .read()
        .map(|cache| cache.len())
        .unwrap_or_default()
}

/// Drop every memoized catalog bullet.
///
/// Exposed for tests and diagnostics; never needed on a serving path.
pub fn clear_catalog_cache() {
    if let Ok(mut cache) = CATALOG_ROW_CACHE.write() {
        cache.clear();
    }
}
