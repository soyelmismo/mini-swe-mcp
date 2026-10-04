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
use super::types::{
    DEFAULT_MAX_TURNS, FRAMING_OVERHEAD, MAX_MODEL_INSTRUCTIONS_BYTES, MAX_TURNS_LIMIT,
    ModelInstructions, ModelManifest,
};

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

        // The retired `strongest:` key is ignored, but the operator should
        // hear that: the consolidator and the automatic security reviewer no
        // longer read it.
        if self.strongest_ignored {
            warnings.push(
                "`strongest:` is no longer supported and is ignored; pick the consolidator model \
                 with `--consolidate=<model>` (MCP `consolidate: \"<model>\"`) and the automatic \
                 security reviewer with the security mode's `default_model`, else both use the \
                 dispatch default"
                    .to_string(),
            );
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

            // Per-model instructions (appended to this model's system prompt).
            // A model that declares none contributes nothing, which is what
            // keeps a pre-instructions manifest warning-free.
            if let Some(instructions) = &def.instructions {
                warnings.extend(Self::validate_instructions(alias, instructions));
            }
        }

        // Review modes are iterated in sorted name order so the output is
        // stable across runs. A non-built-in mode with no checklist is an
        // auditor with nothing to say, so it is reported and dropped by
        // `normalize`; a built-in override may set only `default_model`, in
        // which case the built-in checklist stays.
        for (name, def) in self.sorted_review_modes() {
            let builtin = name.eq_ignore_ascii_case("quality")
                || name.eq_ignore_ascii_case("security");
            match def.checklist.as_deref() {
                None if !builtin => warnings.push(format!(
                    "review mode \"{name}\" has no checklist; it will be ignored"
                )),
                Some(c) if c.trim().is_empty() => warnings.push(format!(
                    "review mode \"{name}\" has an empty checklist; it will be ignored"
                )),
                _ => {}
            }
            if def.model_key_deprecated {
                warnings.push(format!(
                    "review mode \"{name}\" uses the deprecated `model:` key; rename it to \
                     `default_model:`"
                ));
            }
            // A single token that names both a mode and a model alias resolves
            // as the mode (see `ReviewMode::parse_with_manifest`), so the
            // model alias is shadowed and the operator should know.
            if self.models.contains_key(name.trim()) {
                warnings.push(format!(
                    "review mode \"{name}\" shadows the model alias \"{name}\"; \
                     `--review-after {name}` resolves as the mode"
                ));
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

    /// Validate one model entry's `instructions:` block, returning the warnings
    /// for that entry alone.
    ///
    /// Split out of [`ModelManifest::validate`] for the same reason as
    /// [`ModelManifest::validate_policy`]: the rule is expressible against a
    /// single definition, and pairing it with [`ModelManifest::normalize_instructions`]
    /// keeps the warning and its repair in one place.
    pub(crate) fn validate_instructions(
        alias: &str,
        instructions: &ModelInstructions,
    ) -> Vec<String> {
        let mut warnings = Vec::new();
        // The budget bounds the *emitted* section, not just the bullets, so the
        // warning fires exactly when [`ModelInstructions::truncate_to`] would
        // cut: a block whose bullets alone sit under the cap can still overrun
        // it once the header framing is counted.
        let len = FRAMING_OVERHEAD + instructions.rendered_len();
        if len > MAX_MODEL_INSTRUCTIONS_BYTES {
            warnings.push(format!(
                "model \"{alias}\": instructions are {len} bytes, above the \
                 {MAX_MODEL_INSTRUCTIONS_BYTES}-byte budget; the tail is dropped"
            ));
        }
        warnings
    }

    /// Repair one model entry's `instructions:` block in place.
    ///
    /// Entries that are blank once trimmed were already dropped when the block
    /// was deserialized, so the only repair left is the byte budget: the *head*
    /// is kept, because an instruction stated first is the one that corrects the
    /// habit, and the block is marked [`ModelInstructions::is_truncated`] so the
    /// prompt can say it is incomplete. A block that ends up empty is removed
    /// altogether, so "declares nothing" stays `None` and normalizing is
    /// idempotent.
    pub(crate) fn normalize_instructions(instructions: &mut Option<ModelInstructions>) {
        let Some(block) = instructions else {
            return;
        };
        if block.is_empty() {
            *instructions = None;
            return;
        }
        let mut block = block.clone();
        if block.truncate_to(MAX_MODEL_INSTRUCTIONS_BYTES) && block.is_empty() {
            *instructions = None;
        } else {
            *instructions = Some(block);
        }
    }

    /// Apply every fixup that [`ModelManifest::validate`] reports.
    ///
    /// Each *fixable* warning is paired with a repair: an unrecognised
    /// execution policy is replaced by its restrictive default (the
    /// `normalize_policy` fixup of the `rules` submodule), invalid temperatures
    /// are clamped or dropped, unusable turn budgets are replaced with
    /// [`DEFAULT_MAX_TURNS`] (or the runtime limit), an over-long
    /// `instructions:` block is cut to [`MAX_MODEL_INSTRUCTIONS_BYTES`], and a
    /// `default` that names no known alias is dropped so `main.rs` reaches its
    /// fallback deliberately.
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

        for def in self.models.values_mut() {
            def.temperature = Self::sanitize_temperature(def.temperature);
            def.max_turns = def
                .max_turns
                .map(|n| Self::sanitize_max_turns(Some(n), None));
            Self::normalize_policy(&mut def.policy);
            Self::normalize_instructions(&mut def.instructions);
        }

        // A built-in override with an empty checklist falls back to the
        // built-in one (the warning already named it); a user mode with an
        // empty or missing checklist is dropped.
        for (name, def) in self.review_modes.iter_mut() {
            let builtin = name.eq_ignore_ascii_case("quality")
                || name.eq_ignore_ascii_case("security");
            if builtin && def.checklist.as_deref().is_some_and(|c| c.trim().is_empty()) {
                def.checklist = None;
            }
        }
        self.review_modes.retain(|name, def| {
            let builtin = name.eq_ignore_ascii_case("quality")
                || name.eq_ignore_ascii_case("security");
            builtin || def.checklist.as_deref().is_some_and(|c| !c.trim().is_empty())
        });

        self
    }
}
