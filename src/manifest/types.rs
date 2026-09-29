//! Data model and tuning constants for the model manifest.
//!
//! This module owns the two serializable structs that make up `models.yaml`
//! ([`ModelDefinition`], [`ModelManifest`]) plus the four process-wide constants
//! that bound them ([`BUILTIN_DEFAULT_MODEL`], [`DEFAULT_MAX_TURNS`],
//! [`MAX_TURNS_LIMIT`], [`TEMPERATURE_RANGE`]). It contains no behaviour: every
//! rule that reads or repairs these values lives in the `validate` submodule
//! of the manifest package.

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

/// Inclusive bounds every sampling temperature is clamped into before it can
/// reach a provider. OpenAI-compatible endpoints reject values outside this
/// window, some silently clamp, and some ignore the field entirely.
pub const TEMPERATURE_RANGE: std::ops::RangeInclusive<f32> = 0.0..=2.0;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDefinition {
    pub id: String,
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub max_turns: Option<usize>,
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
            },
        );
        models.insert(
            "nerd".to_string(),
            ModelDefinition {
                id: "combo:nerd".to_string(),
                role: Some("Deep reasoner. Use for hard debugging, complex architecture, and multi-file refactors.".to_string()),
                temperature: Some(0.6),
                max_turns: Some(100),
            },
        );

        Self {
            default: Some(BUILTIN_DEFAULT_MODEL.to_string()),
            models,
        }
    }
}
