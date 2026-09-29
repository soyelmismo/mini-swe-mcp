//! Data model and tuning constants for the model manifest.
//!
//! Owns the two serializable structs of `models.yaml` ([`ModelDefinition`],
//! [`ModelManifest`]) plus the four process-wide constants that bound them
//! ([`BUILTIN_DEFAULT_MODEL`], [`DEFAULT_MAX_TURNS`], [`MAX_TURNS_LIMIT`],
//! [`TEMPERATURE_RANGE`]). Contains no behaviour: every rule that reads or
//! repairs these values lives in the `validate` submodule.
//!
//! The declarative execution policy ([`ExecutionPolicy`] and its
//! [`NetworkPolicy`] / [`FsPolicy`] fields) is the exception: it is *data* here
//! too. Parsing is lenient (every field is optional and an unknown string is
//! kept verbatim so a warning can name it), while the accept/reject rules for
//! those strings live in the `rules` submodule.

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

/// Accepted values of `policy.fs` in `models.yaml`, from least to most capable.
pub const FS_POLICIES: &[&str] = &["read-only", "worktree-only", "full"];

/// Inclusive bounds every sampling temperature is clamped into before it can
/// reach a provider. OpenAI-compatible endpoints reject values outside this
/// window, some silently clamp, and some ignore the field entirely.
pub const TEMPERATURE_RANGE: std::ops::RangeInclusive<f32> = 0.0..=2.0;

/// Network permission of a model, as declared by `models.yaml`.
///
/// The variants mirror the `network` enum the `dispatch` tool already accepts
/// (see [`NETWORK_MODES`](crate::mcp::NETWORK_MODES)), so a model-level
/// declaration and a per-dispatch override spell the policy the same way.
/// Deserialization is lenient on purpose: an unrecognised string is kept as
/// [`NetworkPolicy::Other`] instead of failing the whole manifest, which lets
/// [`ModelManifest::validate`] report it and keep serving the catalog.
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

/// Filesystem confinement of a model, as declared by `models.yaml`.
///
/// The three values describe exactly the writable set of a sandboxed worker
/// (see [`crate::agent::sandbox`]): the widest one is the worktree plus its
/// build directory, the narrowest is nothing at all. They are listed least to
/// most capable, and [`FS_POLICIES`] lists them in that same order.
///
/// Deliberately not [`Ord`]: the derive would rank [`FsPolicy::Other`] *above*
/// [`FsPolicy::Full`], so the order would not mean "least privilege" for exactly
/// the values that most need it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FsPolicy {
    /// Nothing may be written outside the process' own scratch state.
    ReadOnly,
    /// Only the worktree (and its git directory) may be written.
    WorktreeOnly,
    /// The worktree *and* its build/target directory.
    Full,
    /// Anything else the manifest spelled out, preserved verbatim.
    ///
    /// Reported as a warning and repaired to [`FsPolicy::ReadOnly`].
    Other(String),
}

/// The value an unrecognised filesystem policy is repaired to: the most
/// restrictive one. See [`NetworkPolicy`]'s `Default` for why it is not `Full`.
impl Default for FsPolicy {
    fn default() -> Self {
        Self::ReadOnly
    }
}

impl<'de> Deserialize<'de> for FsPolicy {
    /// Delegate to [`FsPolicy::parse`]; see [`NetworkPolicy`]'s impl for why.
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let raw = String::deserialize(d)?;
        Ok(Self::parse(&raw))
    }
}

impl Serialize for FsPolicy {
    /// Always emit the canonical spelling.
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(self.as_str())
    }
}

impl FsPolicy {
    /// The policy as written in `models.yaml`: the canonical spelling for a
    /// known value, and the user's own text for [`FsPolicy::Other`].
    ///
    /// "Declares no filesystem policy at all" is not representable here — that
    /// is an `Option::None` *field*, i.e. an absent [`ExecutionPolicy::fs`],
    /// which is what keeps it distinct from declaring `read-only`.
    pub fn as_str(&self) -> &str {
        match self {
            Self::ReadOnly => "read-only",
            Self::WorktreeOnly => "worktree-only",
            Self::Full => "full",
            Self::Other(raw) => raw,
        }
    }

    /// Whether the model declares this policy explicitly.
    pub fn is_declared(&self) -> bool {
        !matches!(self, Self::Other(..))
    }

    /// Parse a manifest-supplied value, keeping anything unrecognised.
    ///
    /// Trims and lower-cases first; `_` is accepted as a separator so both
    /// `worktree-only` and `worktree_only` name the same policy.
    pub fn parse(raw: &str) -> Self {
        match raw.trim().to_ascii_lowercase().replace('_', "-").as_str() {
            "read-only" | "readonly" => Self::ReadOnly,
            "worktree-only" => Self::WorktreeOnly,
            "full" => Self::Full,
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
    /// Writable filesystem confinement, or `None` when the model declares none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fs: Option<FsPolicy>,
}

impl ExecutionPolicy {
    /// Whether the policy block declares nothing at all.
    pub fn is_empty(&self) -> bool {
        self.network.is_none() && self.fs.is_none()
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelManifest {
    #[serde(default)]
    pub default: Option<String>,
    #[serde(default)]
    pub models: HashMap<String, ModelDefinition>,
}

impl Default for ModelManifest {
    fn default() -> Self {
        let mut models = HashMap::new();
        models.insert(
            "ninja".to_string(),
            ModelDefinition {
                id: "combo:ninja".to_string(),
                role: Some("Fast executor. Use for exploration, test runs, syntax fixes, and focused edits.".to_string()),
                temperature: Some(0.2),
                max_turns: Some(100),
                policy: Some(ExecutionPolicy {
                    // The fast executor runs tests and the toolchain, so it keeps
                    // egress and the full writable set of a sandboxed worker.
                    network: Some(NetworkPolicy::Allow),
                    fs: Some(FsPolicy::Full),
                }),
            },
        );
        models.insert(
            "nerd".to_string(),
            ModelDefinition {
                id: "combo:nerd".to_string(),
                role: Some("Deep reasoner. Use for hard debugging, complex architecture, and multi-file refactors.".to_string()),
                temperature: Some(0.6),
                max_turns: Some(100),
                policy: Some(ExecutionPolicy {
                    // The deep reasoner is for debugging and refactors: it reads
                    // broadly and must reach its dependencies to do so.
                    network: Some(NetworkPolicy::Allow),
                    fs: Some(FsPolicy::Full),
                }),
            },
        );

        Self {
            default: Some(BUILTIN_DEFAULT_MODEL.to_string()),
            models,
        }
    }
}
