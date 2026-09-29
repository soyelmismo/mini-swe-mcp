//! Markdown catalog rendering for the MCP `tools/list` payload.
//!
//! [`ModelManifest::build_tool_description`] turns the manifest into the plain
//! markdown block that is embedded in the `dispatch` tool description. It lives
//! here, away from manifest *discovery* (`mod.rs`) and manifest *validation*
//! (`validate.rs`), because rendering is the one part of the package that has no
//! input other than the catalog itself: it reads no environment variable, no
//! file, and no user configuration.
//!
//! Three properties are load-bearing and are asserted by the tests in `tests.rs`:
//!
//! * **Determinism.** Bullets are emitted in sorted-alias order (via
//!   [`ModelManifest::sorted_models`]) rather than in `HashMap` iteration order,
//!   so the same `models.yaml` always renders byte-identical text — rendering
//!   straight out of the `HashMap` produced up to 24 different strings for the
//!   same YAML across 200 parses.
//! * **Bounded cost.** The result is allocated exactly once, pre-sized from the
//!   *effective* cost of every bullet, and each individual bullet is memoized
//!   process-wide as an `Arc<str>` by [`catalog_row`] (see the `cache`
//!   submodule), so a warm render copies no bullet twice.
//! * **Single allocation.** The header is a shared, once-initialized `Arc<str>`
//!   ([`catalog_header`]) and the buffer is pre-sized with the same accounting
//!   [`catalog_row`] uses, including the role fallback, so appending never
//!   reallocates the output.

use super::DEFAULT_ROLE;
/// The one-entry-bullet renderer, re-exported from the [`cache`] submodule so
/// the whole markdown rendering path lives in this one file.
///
/// `pub(super)` on a `use` is a re-export, not a fresh binding, so the unit
/// tests in `tests.rs` can name it; it is deliberately not re-exported from
/// the package root, where it remains an implementation detail.
pub(super) use super::cache::catalog_row;
use super::cache::{CATALOG_ROW_OVERHEAD, catalog_header};
use super::types::ModelManifest;

impl ModelManifest {
    /// Render the catalog advertised through the MCP `tools/list` payload.
    ///
    /// The output is a header line followed by one bullet per model entry, in
    /// alias order, which makes it byte-identical for identical manifests. Each
    /// bullet is memoized process-wide as an `Arc<str>` (see `cache::catalog_row`)
    /// and the header as a shared `Arc<str>`, so the only per-request allocation
    /// is the returned `String` itself.
    pub fn build_tool_description(&self) -> String {
        let entries = self.sorted_models();
        let header = catalog_header();

        // Pre-size the buffer so the catalog is built with a single allocation:
        // the header plus, for every bullet, the fixed overhead plus the three
        // variable-length pieces of text it embeds. The role length must use the
        // *effective* role (the `DEFAULT_ROLE` fallback) so a manifest without
        // explicit roles still never reallocates.
        let total = header.len()
            + entries
                .iter()
                .map(|(alias, def)| {
                    CATALOG_ROW_OVERHEAD
                        + alias.len()
                        + def.id.len()
                        + def.role.as_deref().unwrap_or(DEFAULT_ROLE).len()
                })
                .sum::<usize>();
        let mut desc = String::with_capacity(total);
        desc.push_str(header);

        for (alias, def) in entries {
            // `catalog_row` returns a memoized `Arc<str>`; appending it copies the
            // bullet into the output exactly once, with no intermediate `String`.
            desc.push_str(&catalog_row(alias, def));
        }

        desc
    }
}
