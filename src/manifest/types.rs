//! Data model and tuning constants for the model manifest.
//!
//! Owns the two serializable structs of `models.yaml` ([`ModelDefinition`],
//! [`ModelManifest`]) plus the four process-wide constants that bound them
//! ([`BUILTIN_DEFAULT_MODEL`], [`DEFAULT_MAX_TURNS`], [`MAX_TURNS_LIMIT`],
//! [`TEMPERATURE_RANGE`]). Contains no behaviour: every rule that reads or
//! repairs these values lives in the `validate` submodule.
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
