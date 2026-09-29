//! Accept/reject rules for the declarative execution policy of a model.
//!
//! `models.yaml` may attach a `policy:` block to any model entry, naming the
//! egress permission (`network`) a worker of that model runs under. This module
//! is the single place that decides which spellings are *valid* and which are
//! repaired, and it is deliberately separate from [`super::validate`], which
//! reports and repairs what a manifest declared as a whole (`default`,
//! duplicate ids, turn budgets, temperatures).
//!
//! Three properties are load-bearing:
//!
//! * **Backward compatible.** Every policy field is optional, and an entry with
//!   no `policy:` block deserializes to `None` — "no policy declared", which is
//!   *not* the same as declaring a restrictive one and is not repaired into one.
//!   A manifest written before this module existed therefore parses exactly as
//!   before and behaves identically.
//! * **Lenient parse, strict report.** Deserialization never fails on an
//!   unknown string: it lands in [`NetworkPolicy::Other`] with the text
//!   preserved, so [`ModelManifest::validate`] can name the exact value the
//!   user wrote and [`ModelManifest::normalize`] can replace it. A single typo
//!   degrades to the documented default instead of taking the whole catalog
//!   down with a parse error.
//! * **Every warning has a fixup.** An unknown or misspelled policy value is
//!   repaired to the *restrictive* default, never to the permissive one, so a
//!   typo cannot silently widen a sandbox. Normalizing is idempotent, exactly
//!   like the rest of the manifest.
//!
//! [`ExecutionPolicy`]: super::ExecutionPolicy
//! [`NetworkPolicy`]: super::NetworkPolicy

use super::types::{ExecutionPolicy, ModelManifest, NETWORK_POLICIES, NetworkPolicy};

impl ModelManifest {
    /// Validate one model entry's execution policy, returning the warnings for
    /// that entry alone.
    ///
    /// Split out of [`ModelManifest::validate`] so the rules can be exercised
    /// against a single definition without building a whole manifest, and so the
    /// set of accepted values has exactly one spelling in this crate.
    pub(crate) fn validate_policy(alias: &str, policy: &ExecutionPolicy) -> Vec<String> {
        let mut warnings = Vec::new();

        if let Some(NetworkPolicy::Other(raw)) = &policy.network {
            warnings.push(format!(
                "model \"{alias}\": policy.network \"{raw}\" is not a valid network policy; \
                 expected one of: {}. Replaced with \"{}\".",
                join_known(NETWORK_POLICIES),
                NetworkPolicy::default().as_str(),
            ));
        }

        warnings
    }

    /// Repair one model entry's execution policy in place.
    ///
    /// An unknown value is replaced by the variant's `Default` — `offline` for
    /// the network. Choosing the *restrictive* side matters: a typo must never
    /// quietly hand a worker more capability than the manifest asked for.
    ///
    /// A `None` policy stays `None`: "declared nothing" is preserved so a caller
    /// can still tell it apart from "declared the default", and normalizing is
    /// therefore idempotent.
    pub(crate) fn normalize_policy(policy: &mut Option<ExecutionPolicy>) {
        let Some(policy) = policy else {
            return;
        };
        if let Some(network) = &mut policy.network
            && matches!(network, NetworkPolicy::Other(..))
        {
            *network = NetworkPolicy::default();
        }
    }
}

/// Render the accepted values as `"a", "b"` for a warning message.
fn join_known(values: &[&str]) -> String {
    values
        .iter()
        .map(|v| format!("\"{v}\""))
        .collect::<Vec<_>>()
        .join(", ")
}
