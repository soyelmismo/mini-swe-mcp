//! Data model and tuning constants for the model manifest.
//!
//! Owns the serializable structs of `models.yaml` ([`ModelDefinition`],
//! [`ModelManifest`], [`ModelInstructions`]) plus the process-wide constants that
//! bound them ([`BUILTIN_DEFAULT_MODEL`], [`DEFAULT_MAX_TURNS`],
//! [`MAX_TURNS_LIMIT`], [`MAX_MODEL_INSTRUCTIONS_BYTES`], [`TEMPERATURE_RANGE`]).
//! Contains no behaviour: every rule that reads or repairs these values lives
//! in the `validate` submodule. The one exception is how an `instructions:`
//! block is *split* into entries, which is deserialization itself and therefore
//! belongs next to the field it deserializes.
//!
//! The declarative execution policy ([`ExecutionPolicy`] and its
//! [`NetworkPolicy`] field) is the exception: it is *data* here too. Parsing
//! is lenient (every field is optional and an unknown string is kept verbatim
//! so a warning can name it), while the accept/reject rules for those strings
//! live in the `rules` submodule.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Model id the server falls back to when neither `DEFAULT_MODEL` nor a usable
/// `default:` in the manifest is available (see `main.rs`).
pub const BUILTIN_DEFAULT_MODEL: &str = "ninja";

/// Turn budget used when neither the request nor the manifest asks for one.
///
/// Also the value the runtime falls back to in `pool.rs` when a `REQUEST_TURNS`
/// expansion yields nothing, so the two code paths agree.
pub const DEFAULT_MAX_TURNS: usize = 100;

/// Hard ceiling on the turn budget, mirroring the `REQUEST_TURNS` expansion cap
/// in `pool.rs`: a budget above it can never be grown into, so it is clamped
/// here instead of being shipped to the worker.
pub const MAX_TURNS_LIMIT: usize = 500;

/// Accepted values of `policy.network` in `models.yaml`, in the order the
/// documentation lists them.
///
/// Kept as a constant rather than derived from the enum so the *advertised*
/// list is a deliberate, user-facing string that never drifts from what the
/// warning messages print, and so a future mode can be added without changing
/// every message.
pub const NETWORK_POLICIES: &[&str] = &["offline", "allow"];

/// Hard ceiling on the bytes of one model's `instructions:` block.
///
/// Instructions are appended to a worker's system prompt verbatim, so an
/// unbounded block would let one catalog entry crowd out the repository rules
/// and the role memory it shares a prompt with. 4 KB is enough to correct a
/// model's habits and small enough to stay a minor part of the prompt; the
/// head is kept when the budget is exceeded (see `validate.rs`).
pub const MAX_MODEL_INSTRUCTIONS_BYTES: usize = 4 * 1024;

/// Prefix every instruction is rendered under in the system prompt.
///
/// Owned here rather than duplicated in the renderer because the byte budget
/// below counts it: a bound that ignored the prefix would let the emitted
/// section overrun the very cap it claims to enforce.
pub(crate) const BULLET_PREFIX: &str = "- ";

/// Bytes the prompt spends on the framing *around* the instruction bullets
/// whether or not the block is cut: the two leading newlines, the
/// `MANDATORY DIRECTIVES FOR YOUR MODEL` header line and its trailing newline.
///
/// This is what a *complete* block adds on top of its bullets, so
/// [`ModelInstructions::truncate_to`] compares the emitted section against the
/// budget with it, not against the bullets alone: a block whose bullets fit
/// under the cap can still push the rendered section over it. Built from the
/// renderer's own text by `catalog`, so the two cannot drift apart.
pub(crate) const FRAMING_OVERHEAD: usize = 2 + super::catalog::MODEL_INSTRUCTIONS_HEADER.len() + 1;

/// Bytes the prompt spends on everything *around* the instruction bullets when
/// the block has been cut: [`FRAMING_OVERHEAD`] plus the truncation note and its
/// trailing newline.
///
/// Reserved out of [`MAX_MODEL_INSTRUCTIONS_BYTES`] by
/// [`ModelInstructions::truncate_to`] so the rendered section — not merely the
/// bullets behind it — fits the budget even after the note is appended.
pub(crate) const SECTION_OVERHEAD: usize =
    FRAMING_OVERHEAD + super::catalog::MODEL_INSTRUCTIONS_TRUNCATION_NOTE.len() + 1;

/// Inclusive bounds every sampling temperature is clamped into before it can
/// reach a provider. OpenAI-compatible endpoints reject values outside this
/// window, some silently clamp, and some ignore the field entirely.
pub const TEMPERATURE_RANGE: std::ops::RangeInclusive<f32> = 0.0..=2.0;

/// Network permission of a model, as declared by `models.yaml`.
///
/// The variants mirror the `network` enum the `dispatch` tool already accepts,
/// so a model-level declaration and a per-dispatch override spell the policy
/// the same way. Deserialization is lenient on purpose: an unrecognised string
/// is kept as [`NetworkPolicy::Other`] instead of failing the whole manifest,
/// which lets [`ModelManifest::validate`] report it and keep serving the
/// catalog.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NetworkPolicy {
    /// No egress at all: every bash step runs in its own network namespace.
    Offline,
    /// The host's normal connectivity.
    Allow,
    /// Anything else the manifest spelled out, preserved verbatim.
    ///
    /// [`NetworkPolicy::parse`] is the only constructor, so this variant is
    /// exactly "the user wrote something that is not a known policy". It is
    /// reported as a warning and repaired to [`NetworkPolicy::Offline`].
    Other(String),
}

/// The value an unrecognised policy is repaired to: the *most restrictive*
/// one, never the permissive one.
///
/// This is deliberately **not** "the policy a worker gets when the manifest
/// declares nothing" — that case keeps `None` and inherits the runtime default.
/// It is only the replacement for a value that was written down and could not be
/// understood, where guessing permissively would quietly hand a worker more
/// capability than the manifest asked for.
impl Default for NetworkPolicy {
    fn default() -> Self {
        Self::Offline
    }
}

impl<'de> Deserialize<'de> for NetworkPolicy {
    /// Delegate to [`NetworkPolicy::parse`] so the accepted spellings are
    /// defined in exactly one place.
    ///
    /// Without this, `Offline` and `" offline "` would deserialize into
    /// [`NetworkPolicy::Other`] and a perfectly valid value would be reported as
    /// a typo.
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let raw = String::deserialize(d)?;
        Ok(Self::parse(&raw))
    }
}

impl Serialize for NetworkPolicy {
    /// Always emit the canonical spelling.
    ///
    /// A manifest is normalized before it is served, so what reaches a consumer
    /// is one of `offline` / `allow`; round-tripping a value the manifest has
    /// already rejected would only re-export the typo.
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(self.as_str())
    }
}

impl NetworkPolicy {
    /// The policy as written in `models.yaml`: the canonical spelling for a
    /// known value, and the user's own text for [`NetworkPolicy::Other`].
    ///
    /// "Declares no network policy at all" is not representable here — that is
    /// an `Option::None` *field*, i.e. an absent [`ExecutionPolicy::network`],
    /// which is what keeps it distinct from declaring `offline`.
    pub fn as_str(&self) -> &str {
        match self {
            Self::Offline => "offline",
            Self::Allow => "allow",
            Self::Other(raw) => raw,
        }
    }

    /// Whether the model declares this policy explicitly.
    pub fn is_declared(&self) -> bool {
        !matches!(self, Self::Other(..))
    }

    /// Parse a manifest-supplied value, keeping anything unrecognised.
    ///
    /// Trims and lower-cases first, so `Offline` and ` offline ` are the
    /// spelling of `offline` rather than a typo the user has to hear about.
    pub fn parse(raw: &str) -> Self {
        match raw.trim().to_ascii_lowercase().as_str() {
            "offline" => Self::Offline,
            "allow" => Self::Allow,
            _ => Self::Other(raw.trim().to_string()),
        }
    }
}

/// The optional `policy:` block of one model entry.
///
/// Every field defaults to "not declared", which is what keeps the change
/// backwards compatible: a `models.yaml` written before this block existed
/// parses unchanged and behaves exactly as it did, and a model that declares a
/// policy gets exactly the fields it declared.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ExecutionPolicy {
    /// Egress permission, or `None` when the model declares none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub network: Option<NetworkPolicy>,
}

impl ExecutionPolicy {
    /// Whether the policy block declares nothing at all.
    pub fn is_empty(&self) -> bool {
        self.network.is_none()
    }
}

/// The optional `instructions:` block of one model entry: extra rules appended
/// to the system prompt of every worker that runs on this model.
///
/// Both spellings a YAML author reaches for are accepted and mean the same
/// thing — a multi-line string is one instruction per line, a list is one
/// instruction per bullet:
///
/// ```yaml
/// models:
///   small:
///     id: combo:small
///     instructions: |
///       Read whole files instead of many small ranges.
///       Run the cheap gate before the full suite.
///   # equivalently:
///   small:
///     instructions:
///       - Read whole files instead of many small ranges.
///       - Run the cheap gate before the full suite.
/// ```
///
/// The two forms are normalized into one ordered list at parse time, so nothing
/// downstream has to know which one a catalog used, and the list form is what
/// serializes back out. Blank entries are dropped here rather than rendered as
/// empty bullets, and [`Self::is_truncated`] records that a repair cut the
/// block at [`MAX_MODEL_INSTRUCTIONS_BYTES`] so the prompt can say so.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ModelInstructions {
    entries: Vec<String>,
    /// Set when [`truncate_to`](ModelInstructions::truncate_to) dropped the
    /// tail of an over-long block; never set by deserialization.
    truncated: bool,
}

impl ModelInstructions {
    /// The instructions in declaration order, one per line of the block.
    pub fn entries(&self) -> &[String] {
        &self.entries
    }

    /// How many instructions the block carries, which is what the `manifest`
    /// action reports per model.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the block carries nothing worth injecting into a prompt.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Whether a repair cut this block short at [`MAX_MODEL_INSTRUCTIONS_BYTES`].
    ///
    /// Kept as state rather than re-derived from the byte count so the prompt
    /// can mark a cut block exactly once, and so re-normalizing a cut block is
    /// idempotent.
    pub fn is_truncated(&self) -> bool {
        self.truncated
    }

    /// The bytes the entry list occupies once rendered into the prompt: every
    /// instruction as its own `- <entry>\n` bullet, which is exactly what
    /// `build_system_prompt` appends.
    ///
    /// The bullet prefix is counted because it is real payload the model reads,
    /// so a bound checked against this value bounds the emitted text and not
    /// just the strings behind it.
    pub fn rendered_len(&self) -> usize {
        self.entries
            .iter()
            .map(|entry| entry.len() + BULLET_PREFIX.len() + 1)
            .sum::<usize>()
    }

    /// Cut the block to at most `max_bytes`, keeping the head, and report whether
    /// anything was dropped.
    ///
    /// The budget is spent entry by entry, because an instruction is the unit the
    /// author wrote: dropping a whole one is comprehensible where half a
    /// sentence is not. A single entry larger than the whole budget is the one
    /// exception — it is cut at a `char` boundary so the block still carries
    /// *something* and never disappears without trace. Either way
    /// [`Self::is_truncated`] is set, and running this on an already-cut block is
    /// a no-op, which keeps normalizing idempotent.
    pub(crate) fn truncate_to(&mut self, max_bytes: usize) -> bool {
        // A *complete* block adds the framing on top of its bullets, so the
        // no-cut case must fit the section, not just the bullets: a block whose
        // bullets alone sit under the cap can still overrun the budget once the
        // header is counted.
        if FRAMING_OVERHEAD + self.rendered_len() <= max_bytes {
            return false;
        }

        // Reserve the wrapping the prompt adds around the bullets (its own
        // leading newlines, the header line and the truncation note) so the
        // *emitted* section, not just the bullets, stays inside the budget.
        let budget = max_bytes.saturating_sub(SECTION_OVERHEAD);
        let mut used = 0;
        let mut kept = 0;
        for entry in &self.entries {
            let cost = entry.len() + BULLET_PREFIX.len() + 1;
            if used + cost > budget {
                break;
            }
            used += cost;
            kept += 1;
        }
        if kept == 0 {
            // One entry bigger than the budget: keep its head, rounded down so
            // the block stays inside the budget and the slice cannot split a
            // multi-byte code point. Everything behind it is dropped — the head
            // alone already fills the budget.
            let head = budget
                .saturating_sub(BULLET_PREFIX.len() + 1)
                .min(self.entries[0].len());
            let head = floor_char_boundary(&self.entries[0], head);
            self.entries.truncate(1);
            self.entries[0] = self.entries[0][..head].to_string();
        } else {
            self.entries.truncate(kept);
        }
        self.truncated = true;
        true
    }

    /// Split raw YAML forms into one entry per line/bullet, dropping blanks.
    fn from_parts(parts: impl IntoIterator<Item = String>) -> Self {
        Self {
            entries: parts.into_iter().filter_map(normalize_entry).collect(),
            // Deserialization never cuts: an over-long block is repaired by
            // `validate::normalize_instructions`, which can warn about it.
            truncated: false,
        }
    }
}

impl Serialize for ModelInstructions {
    /// Always emit the list form, so a catalog round-trips through one spelling
    /// no matter which one it was written in.
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        self.entries.serialize(s)
    }
}

impl<'de> Deserialize<'de> for ModelInstructions {
    /// Accept a multi-line string or a list of strings, in that order.
    ///
    /// A YAML scalar is a string, so `instructions: read whole files` is one
    /// instruction; anything else that is neither string nor list of strings
    /// (a bare number, a mapping) is a schema error and fails the manifest load
    /// with the parser's own message rather than being silently dropped.
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Repr {
            Text(String),
            List(Vec<String>),
        }

        Ok(match Repr::deserialize(d)? {
            Repr::Text(text) => Self::from_parts(text.lines().map(str::to_string)),
            Repr::List(items) => Self::from_parts(items),
        })
    }
}

/// The largest index `<= index` that falls on a `char` boundary of `text`.
fn floor_char_boundary(text: &str, index: usize) -> usize {
    let mut index = index.min(text.len());
    while !text.is_char_boundary(index) {
        index -= 1;
    }
    index
}

/// Trim one raw instruction and drop it when nothing is left.
///
/// A markdown bullet marker is stripped so the list and string spellings of the
/// same instruction render identically (the prompt adds its own bullet).
fn normalize_entry(raw: String) -> Option<String> {
    let trimmed = raw.trim();
    let stripped = trimmed
        .strip_prefix("- ")
        .or_else(|| trimmed.strip_prefix("* "))
        .or_else(|| trimmed.strip_prefix("+ "))
        .unwrap_or(trimmed)
        .trim();
    (!stripped.is_empty()).then(|| stripped.to_string())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDefinition {
    pub id: String,
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub max_turns: Option<usize>,
    /// Optional declarative execution policy, `None` for every manifest
    /// written before policies existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub policy: Option<ExecutionPolicy>,
    /// Extra rules appended to the system prompt of a worker of this model,
    /// `None` for every catalog that declares none.
    ///
    /// This is how a model's habits are corrected from the catalog: the
    /// operator appends them to the prompt of a worker running that model, and
    /// the review phase uses the reviewer's own block. See
    /// [`ModelInstructions`] for the accepted spellings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<ModelInstructions>,
}

/// One review role declared by `models.yaml`'s optional `review_modes:` map.
///
/// A review mode is the user's own auditor: its `checklist` is the focus
/// instructions appended to the common review frame (inspect the diff, run
/// the dispatch's verify gate, fix real defects with a regression test, list
/// unfixed findings in REPORT risks), and the optional `model` names the
/// default reviewer for that mode. Built-in modes `quality` and `security`
/// keep their current prompts and can be overridden by declaring the same
/// name here.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewModeDefinition {
    /// The focus instructions appended to the common review frame.
    ///
    /// Non-empty is validated in `validate.rs`; an empty checklist would be an
    /// auditor with nothing to say.
    pub checklist: String,
    /// The default reviewer model for this mode, when one is declared.
    ///
    /// `None` falls back to the dispatch's own model for the review phase.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelManifest {
    #[serde(default)]
    pub default: Option<String>,
    /// Alias of the manifest's strongest tier, when one is marked.
    ///
    /// Set `strongest: <alias>` in `models.yaml` to name it. A consolidator
    /// integrates a whole round, so it runs on the deepest model the manifest
    /// declares rather than on the fast executor the dispatch default names;
    /// `--model` still overrides it per dispatch. `None` for a manifest that
    /// marks none, which keeps the dispatch default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strongest: Option<String>,
    /// Repository-relative globs whose diffs trigger an automatic adversarial
    /// security review, the manifest-side counterpart of the `## Sensitive
    /// paths` section of `AGENTS.md`.
    ///
    /// Absent means "none declared here"; the instruction-file section is
    /// still consulted, so marking paths in either place is enough.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sensitive_paths: Vec<String>,
    #[serde(default)]
    pub models: HashMap<String, ModelDefinition>,
    /// Optional review roles, keyed by mode name.
    ///
    /// Each entry is a [`ReviewModeDefinition`]. Declaring a mode named
    /// `quality` or `security` overrides the built-in prompt; any other name
    /// adds a new mode selectable via `--review-after <model>:<mode>`. Absent
    /// means only the built-in `quality` and `security` modes exist.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub review_modes: HashMap<String, ReviewModeDefinition>,
}

impl Default for ModelManifest {
    fn default() -> Self {
        let mut models = HashMap::new();
        models.insert(
            "ninja".to_string(),
            ModelDefinition {
                id: "combo:ninja".to_string(),
                role: Some("Fast executor: exploration, tests, syntax fixes, edits.".to_string()),
                temperature: Some(0.2),
                max_turns: Some(100),
                policy: Some(ExecutionPolicy {
                    // The fast executor runs tests and the toolchain, so it keeps
                    // egress.
                    network: Some(NetworkPolicy::Allow),
                }),
                instructions: None,
            },
        );
        models.insert(
            "nerd".to_string(),
            ModelDefinition {
                id: "combo:nerd".to_string(),
                role: Some(
                    "Deep reasoner: debugging, architecture, multi-file refactors.".to_string(),
                ),
                temperature: Some(0.6),
                max_turns: Some(100),
                policy: Some(ExecutionPolicy {
                    // The deep reasoner is for debugging and refactors: it reads
                    // broadly and must reach its dependencies to do so.
                    network: Some(NetworkPolicy::Allow),
                }),
                instructions: None,
            },
        );

        Self {
            default: Some(BUILTIN_DEFAULT_MODEL.to_string()),
            strongest: None,
            sensitive_paths: Vec::new(),
            models,
            review_modes: HashMap::new(),
        }
    }
}
