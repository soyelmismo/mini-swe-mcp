use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use std::path::{Path, PathBuf};
use tracing::{error, info};

use crate::config::xdg_config_dir;

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
                max_turns: Some(50),
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
            default: Some("ninja".to_string()),
            models,
        }
    }
}

impl ModelManifest {
    pub fn load() -> Self {
        let mut candidates = Vec::with_capacity(4);

        // 1. Explicit environment variable MODELS_FILE
        if let Ok(path) = env::var("MODELS_FILE") {
            candidates.push((PathBuf::from(path), "MODELS_FILE"));
        }

        // 2. Local working directory models.yaml
        candidates.push((PathBuf::from("models.yaml"), "current directory"));

        // 3. Standard XDG config directory (~/.config/mini-swe/models.yaml)
        if let Some(dir) = xdg_config_dir() {
            candidates.push((
                dir.join("mini-swe").join("models.yaml"),
                "XDG config directory",
            ));
        }

        // 4. Alongside the executable
        if let Ok(exe) = env::current_exe()
            && let Some(parent) = exe.parent()
        {
            candidates.push((parent.join("models.yaml"), "executable directory"));
        }

        for (path, source) in candidates {
            if let Some(manifest) = Self::from_candidate(&path, source) {
                return manifest;
            }
        }

        info!("No models.yaml found; using default built-in manifest (ninja & nerd)");
        Self::default()
    }

    fn from_candidate(path: &Path, source: &str) -> Option<Self> {
        if !path.exists() {
            return None;
        }

        match Self::from_file(path) {
            Ok(manifest) => {
                info!(path = %path.display(), "Loaded model manifest from {source}");
                Some(manifest)
            }
            Err(e) => {
                error!(error = %e, path = %path.display(), "Failed to parse models.yaml from {source}");
                None
            }
        }
    }

    fn from_file(path: &Path) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let manifest: Self = serde_yaml::from_str(&content)?;
        Ok(manifest)
    }

    pub fn resolve_model(&self, requested: &str) -> (String, Option<f32>, Option<usize>) {
        self.models
            .get(requested)
            .or_else(|| self.models.values().find(|def| def.id == requested))
            .map_or_else(
                || (requested.to_string(), None, None),
                |def| (def.id.clone(), def.temperature, def.max_turns),
            )
    }

    pub fn build_tool_description(&self) -> String {
        let mut desc = String::from("Available model aliases and their roles:\n");

        for (alias, def) in &self.models {
            let role = def.role.as_deref().unwrap_or("Autonomous subagent");
            desc.push_str(&format!("- `{}` (id: `{}`): {}\n", alias, def.id, role));
        }

        desc
    }
}

#[cfg(test)]
mod tests {
    use super::{ModelDefinition, ModelManifest};

    #[test]
    fn test_default_manifest() {
        let manifest = ModelManifest::default();

        assert_eq!(manifest.default, Some("ninja".to_string()));

        let ninja = manifest
            .models
            .get("ninja")
            .expect("default manifest must contain the `ninja` model");
        assert_eq!(ninja.id, "combo:ninja");
        assert!(ninja.role.as_deref().is_some_and(|r| !r.is_empty()));
        assert_eq!(ninja.temperature, Some(0.2));
        assert_eq!(ninja.max_turns, Some(50));

        let nerd = manifest
            .models
            .get("nerd")
            .expect("default manifest must contain the `nerd` model");
        assert_eq!(nerd.id, "combo:nerd");
        assert!(nerd.role.as_deref().is_some_and(|r| !r.is_empty()));
        assert_eq!(nerd.temperature, Some(0.6));
        assert_eq!(nerd.max_turns, Some(100));

        assert_eq!(manifest.models.len(), 2);
    }

    #[test]
    fn test_resolve_model() {
        let manifest = ModelManifest::default();

        // Resolution by alias name
        assert_eq!(
            manifest.resolve_model("ninja"),
            ("combo:ninja".to_string(), Some(0.2), Some(50))
        );
        assert_eq!(
            manifest.resolve_model("nerd"),
            ("combo:nerd".to_string(), Some(0.6), Some(100))
        );

        // Resolution by full model id
        assert_eq!(
            manifest.resolve_model("combo:nerd"),
            ("combo:nerd".to_string(), Some(0.6), Some(100))
        );

        // Fallback: unknown models are passed through untouched
        assert_eq!(
            manifest.resolve_model("some/unknown-model"),
            ("some/unknown-model".to_string(), None, None)
        );

        // Empty request
        assert_eq!(manifest.resolve_model(""), (String::new(), None, None));

        // Model without overrides
        let mut models = std::collections::HashMap::new();
        models.insert(
            "plain".to_string(),
            ModelDefinition {
                id: "vendor:plain".to_string(),
                role: None,
                temperature: None,
                max_turns: None,
            },
        );
        let sparse = ModelManifest {
            default: None,
            models,
        };
        assert_eq!(
            sparse.resolve_model("plain"),
            ("vendor:plain".to_string(), None, None)
        );
    }

    #[test]
    fn test_tool_description() {
        let manifest = ModelManifest::default();
        let desc = manifest.build_tool_description();

        let mut lines = desc.lines();
        assert_eq!(
            lines.next(),
            Some("Available model aliases and their roles:")
        );

        let body: Vec<&str> = lines.collect();
        assert_eq!(body.len(), 2);

        let find = |alias: &str| {
            body.iter()
                .find(|line| line.starts_with(&format!("- `{alias}`")))
                .unwrap_or_else(|| panic!("missing bullet for alias `{alias}`"))
        };

        let ninja = find("ninja");
        assert!(ninja.contains("combo:ninja"));

        let nerd = find("nerd");
        assert!(nerd.contains("combo:nerd"));
    }

    #[test]
    fn test_tool_description_role_fallback() {
        let mut models = std::collections::HashMap::new();
        models.insert(
            "bare".to_string(),
            ModelDefinition {
                id: "vendor:bare".to_string(),
                role: None,
                temperature: Some(0.9),
                max_turns: Some(7),
            },
        );
        let manifest = ModelManifest {
            default: None,
            models,
        };

        assert_eq!(
            manifest.build_tool_description(),
            "Available model aliases and their roles:\n\
             - `bare` (id: `vendor:bare`): Autonomous subagent\n"
        );
    }
}
