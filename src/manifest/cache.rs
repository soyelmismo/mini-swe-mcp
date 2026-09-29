//! Process-wide memoization of rendered catalog bullets.
//!
//! [`ModelManifest::build_tool_description`] is re-rendered on every MCP
//! `tools/list` request, and rendering a bullet allocates and formats a
//! `String`. A rendered bullet is a pure function of `(alias, id, role)`, so it
//! is memoized here rather than inside the manifest: [`ModelManifest`] stays
//! `&self`-clean, keeping its `Clone`/`Debug`/`Serialize` derives untouched (no
//! `serde(skip)` plumbing, no interior mutability leaking into `Arc` clones).

use std::collections::HashMap;
use std::fmt::Write as _;
use std::sync::{Arc, LazyLock, RwLock};

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
static CATALOG_ROW_CACHE: LazyLock<RwLock<HashMap<CatalogRowKey, Arc<str>>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

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

    if let Ok(cache) = CATALOG_ROW_CACHE.read()
        && let Some(row) = cache.get(&key)
    {
        return row.to_string();
    }

    let role = def.role.as_deref().unwrap_or(DEFAULT_ROLE);
    let mut row =
        String::with_capacity(CATALOG_ROW_OVERHEAD + alias.len() + def.id.len() + role.len());
    let _ = writeln!(row, "- `{alias}` (id: `{}`): {role}", def.id);

    if let Ok(mut cache) = CATALOG_ROW_CACHE.write() {
        if cache.len() >= CATALOG_CACHE_CAPACITY {
            cache.clear();
        }
        cache.insert(key, Arc::from(row.as_str()));
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
