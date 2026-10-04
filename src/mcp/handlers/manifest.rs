use super::*;

impl McpServer {
    pub(super) fn handle_manifest(&self) -> Result<Value> {
        // Every mode the catalog offers, built-ins included, each tagged with
        // its source so the host can tell a built-in from a `models.yaml`
        // entry (or override).
        let mut review_modes = serde_json::Map::new();
        for (name, def, source) in self.manifest.effective_review_modes() {
            review_modes.insert(
                name,
                json!({
                    "checklist": def.checklist,
                    "default_model": def.default_model,
                    "source": source,
                }),
            );
        }
        Ok(json!({
            "default_model": self.manifest.default,
            "models": self.manifest.models,
            "review_modes": review_modes,
        }))
    }
}
