use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use std::path::{Path, PathBuf};
use tracing::info;

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
        // 1. Explicit environment variable MODELS_FILE
        if let Ok(path) = env::var("MODELS_FILE") {
            let p = PathBuf::from(path);
            if p.exists() {
                if let Ok(manifest) = Self::from_file(&p) {
                    info!(path = %p.display(), "Loaded model manifest from MODELS_FILE");
                    return manifest;
                }
            }
        }

        // 2. Local working directory models.yaml
        let local_path = PathBuf::from("models.yaml");
        if local_path.exists() {
            if let Ok(manifest) = Self::from_file(&local_path) {
                info!(path = %local_path.display(), "Loaded model manifest from current directory");
                return manifest;
            }
        }

        // 3. Standard XDG config directory (~/.config/mini-swe/models.yaml)
        let config_dir = env::var("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .or_else(|_| env::var("HOME").map(|h| Path::new(&h).join(".config")))
            .ok();

        if let Some(dir) = config_dir {
            let xdg_path = dir.join("mini-swe").join("models.yaml");
            if xdg_path.exists() {
                if let Ok(manifest) = Self::from_file(&xdg_path) {
                    info!(path = %xdg_path.display(), "Loaded model manifest from XDG config directory");
                    return manifest;
                }
            }
        }

        // 4. Alongside the executable
        if let Ok(exe) = env::current_exe() {
            if let Some(parent) = exe.parent() {
                let exe_model_path = parent.join("models.yaml");
                if exe_model_path.exists() {
                    if let Ok(manifest) = Self::from_file(&exe_model_path) {
                        info!(path = %exe_model_path.display(), "Loaded model manifest from executable directory");
                        return manifest;
                    }
                }
            }
        }

        info!("No models.yaml found; using default built-in manifest (ninja & nerd)");
        Self::default()
    }

    fn from_file(path: &Path) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let manifest: Self = serde_yaml::from_str(&content)?;
        Ok(manifest)
    }

    pub fn resolve_model(&self, requested: &str) -> (String, Option<f32>, Option<usize>) {
        if let Some(def) = self.models.get(requested) {
            return (def.id.clone(), def.temperature, def.max_turns);
        }

        // Check if requested matches any model.id directly
        for def in self.models.values() {
            if def.id == requested {
                return (def.id.clone(), def.temperature, def.max_turns);
            }
        }

        // Fallback: use requested name directly
        (requested.to_string(), None, None)
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
