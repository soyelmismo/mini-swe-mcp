//! Validation rules and the fixups that repair what they report.
//!
//! [`ModelManifest::validate`] collects human-readable warnings about suspicious
//! manifest entries and is deliberately non-fatal: a manifest with warnings is
//! still served, so a typo in `models.yaml` degrades gracefully instead of
//! taking the server down. [`ModelManifest::normalize`] then applies the
//! mechanical fixups, and [`ModelManifest::sanitize_temperature`] /
//! [`ModelManifest::sanitize_max_turns`] are the per-value rules the rest of
//! the crate calls directly (see `mcp.rs`). The rules for the optional
//! declarative execution policy live in their own `rules` submodule, which this
//! one calls into for both the warning and the fixup.

use std::collections::BTreeMap;

use super::rules::join_known;
use super::types::{DEFAULT_MAX_TURNS, MAX_TURNS_LIMIT, ModelManifest};

impl ModelManifest {
    /// Collect human-readable warnings about suspicious manifest entries.
    ///
    /// Validation is deliberately non-fatal: a manifest with warnings is still
    /// served so that a typo in `models.yaml` degrades gracefully instead of
    /// taking the server down. Callers surface the returned strings as warnings
    /// (see `ModelManifest::from_candidate`).
    ///
    /// This is a *shape* check on a manifest, not schema validation: it is
    /// callable for any manifest, including [`ModelManifest::default`] and
    /// manifests built in code. Every warning it reports is paired with a fixup
    /// in [`ModelManifest::normalize`], so the warnings are never the only
    /// consequence of a bad value.
    ///
    /// The catalog is iterated in sorted alias order so the output is stable
    /// across runs; callers may rely on the ordering.
    pub fn validate(&self) -> Vec<String> {
        let mut warnings = Vec::new();
        let entries = self.sorted_models();

        // `default` is looked up by alias, so the lookup is trimmed the same way
        // `resolve_model` would match it: an over-indented `default:` is not a
        // dangling reference.
        if let Some(name) = &self.default
            && !self.models.contains_key(name.trim())
        {
            warnings.push(format!(
                "default model \"{name}\" not found in models; it will be ignored and the built-in \
                 fallback used (set DEFAULT_MODEL to override)"
            ));
        }

        // `strongest` is looked up by alias too, so a dangling reference would
        // silently leave the consolidator on the dispatch default.
        if let Some(name) = &self.strongest
            && !self.models.contains_key(name.trim())
        {
            warnings.push(format!(
                "strongest model \"{name}\" not found in models; it will be ignored and the \
                 dispatch default used"
            ));
        }

        // Duplicate ids are ambiguous for id-based resolution. The policy in
        // `resolve_model` is first-alias-wins (sorted by alias), so the
        // resolution is stable, but the manifest is still ambiguous and the user
        // should know. Non-fatal by design, like every other warning here.
        let mut by_id: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
        for (alias, def) in &entries {
            let id = def.id.trim();
            if !id.is_empty() {
                by_id.entry(id).or_default().push(alias);
            }
        }
        for (id, aliases) in &by_id {
            if aliases.len() > 1 {
                warnings.push(format!(
                    "duplicate model id \"{id}\" shared by aliases {}; resolving the full id returns the first alias",
                    join_known(aliases)
                ));
            }
        }
        for (alias, def) in &entries {
            if def.id.trim().is_empty() {
                warnings.push(format!("model \"{alias}\": id cannot be empty"));
            }

            if let Some(t) = def.temperature {
                if !t.is_finite() {
                    warnings.push(format!(
                        "model \"{alias}\": temperature {t} is not a finite number; replaced with \
                         the provider default"
                    ));
                } else if !super::TEMPERATURE_RANGE.contains(&t) {
                    warnings.push(format!(
                        "model \"{alias}\": temperature {t} is outside [0, 2]; clamped to that \
                         range"
                    ));
                }
            }

            match def.max_turns {
                Some(0) => warnings.push(format!(
                    "model \"{alias}\": max_turns must be greater than 0; replaced with \
                     {DEFAULT_MAX_TURNS}"
                )),
                Some(n) if n > MAX_TURNS_LIMIT => warnings.push(format!(
                    "model \"{alias}\": max_turns {n} exceeds the runtime limit \
                     {MAX_TURNS_LIMIT}; clamped to that limit"
                )),
                _ => {}
            }

            // Declarative execution policy (see the `rules` submodule). A model
            // with no `policy:` block contributes nothing, which is what keeps a
            // pre-policy manifest warning-free.
            if let Some(policy) = &def.policy {
                warnings.extend(Self::validate_policy(alias, policy));
            }
        }

        // Review modes are iterated in sorted name order so the output is
        // stable across runs. A mode with an empty checklist is an auditor
        // with nothing to say, so it is reported and dropped by `normalize`.
        // Each trigger glob is validated: an invalid glob would silently match
        // nothing, so it is reported and dropped too.
        for (name, def) in self.sorted_review_modes() {
            if def.checklist.trim().is_empty() {
                warnings.push(format!(
                    "review mode \"{name}\" has an empty checklist; it will be ignored"
                ));
            }
            for trigger in &def.triggers {
                if !super::validate_glob(trigger) {
                    warnings.push(format!(
                        "review mode \"{name}\" has an invalid trigger glob \"{trigger}\"; it will be ignored"
                    ));
                }
            }
        }

        warnings
    }

    /// Clamp `temperature` into [`TEMPERATURE_RANGE`](super::TEMPERATURE_RANGE),
    /// dropping non-finite values (which no provider accepts) in favour of the
    /// provider default.
    ///
    /// `None` in, `None` out: an unset temperature still means "use the provider
    /// default" and must stay distinguishable from a clamped one.
    pub fn sanitize_temperature(temperature: Option<f32>) -> Option<f32> {
        let t = temperature?;
        if !t.is_finite() {
            return None;
        }
        Some(t.clamp(
            *super::TEMPERATURE_RANGE.start(),
            *super::TEMPERATURE_RANGE.end(),
        ))
    }

    /// Resolve the turn budget from both ingresses, filtering a useless `0`
    /// before it can reach the worker loop.
    ///
    /// A `max_turns` of `0` is `Some(0)`, not `None`: it would defeat the
    /// `unwrap_or` fallback and make `while step < current_max_turns` false on
    /// the first check, i.e. a worker that never runs a single turn. So `0` is
    /// filtered out of *both* ingresses -- a request of `0` falls through to the
    /// manifest budget rather than clobbering it, and a manifest of `0` falls
    /// through to [`DEFAULT_MAX_TURNS`]. Anything above [`MAX_TURNS_LIMIT`] is
    /// clamped to it, matching the `REQUEST_TURNS` expansion cap in `pool.rs`.
    pub fn sanitize_max_turns(requested: Option<usize>, manifest: Option<usize>) -> usize {
        // Filter *before* combining: a `0` from either ingress must fall through
        // to the other one, not shadow it and then vanish.
        requested
            .filter(|&n| n > 0)
            .or_else(|| manifest.filter(|&n| n > 0))
            .map_or(DEFAULT_MAX_TURNS, |n| n.min(MAX_TURNS_LIMIT))
    }

    /// Apply every fixup that [`ModelManifest::validate`] reports.
    ///
    /// Each *fixable* warning is paired with a repair: an unrecognised
    /// execution policy is replaced by its restrictive default (the
    /// `normalize_policy` fixup of the `rules` submodule), invalid temperatures
    /// are clamped or dropped, unusable turn budgets are replaced with
    /// [`DEFAULT_MAX_TURNS`] (or the runtime limit), and a `default` that names
    /// no known alias is dropped so `main.rs` reaches its fallback deliberately.
    ///
    /// One warning has no mechanical fixup and is left for the user: an empty
    /// `id` has no correct value to substitute (the alias key is the only
    /// guess available), so [`ModelManifest::resolve_model`] keeps passing it
    /// through and the provider decides. Normalizing is therefore idempotent,
    /// and re-running [`ModelManifest::validate`] on the result only ever
    /// reports that remaining `id cannot be empty`.
    pub fn normalize(mut self) -> Self {
        self.default = self
            .default
            .filter(|name| self.models.contains_key(name.trim()));
        self.strongest = self
            .strongest
            .filter(|name| self.models.contains_key(name.trim()));

        for def in self.models.values_mut() {
            def.temperature = Self::sanitize_temperature(def.temperature);
            def.max_turns = def
                .max_turns
                .map(|n| Self::sanitize_max_turns(Some(n), None));
            Self::normalize_policy(&mut def.policy);
        }

        self.review_modes
            .retain(|_, def| !def.checklist.trim().is_empty());
        for def in self.review_modes.values_mut() {
            def.triggers.retain(|trigger| super::validate_glob(trigger));
        }

        self
    }
}
