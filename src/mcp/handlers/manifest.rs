use super::*;

impl McpServer {
    pub(super) fn handle_manifest(&self) -> Result<Value> {
        Ok(json!({
            "default_model": self.manifest.default,
            "models": self.manifest.models,
            "review_modes": self.manifest.review_modes,
        }))
    }
}
