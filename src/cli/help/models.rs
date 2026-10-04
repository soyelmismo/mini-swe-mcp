/// `models`: the `models.yaml` catalog and the per-model `instructions:` block.
pub(super) const TEXT: &str = "models.yaml has three top-level keys: `default:` (the alias a dispatch that names no model runs on), `models:` (one entry per alias: an id - what the provider is called - plus optional role, temperature, max_turns, a policy: block (network: offline|allow) and an instructions: block), and `review_modes:` (one entry per review mode: a `checklist` of focus instructions plus an optional `default_model`; the built-ins `quality` and `security` are always available and a same-named entry overrides them field by field - a `default_model:` alone keeps the built-in checklist, a `checklist:` replaces it - while a user mode needs a checklist; the old `model:` spelling still works but warns). Aliases resolve from the first models.yaml found: MODELS_FILE, then ./models.yaml, then $XDG_CONFIG_HOME/mini-swe/models.yaml, then next to the executable, then the built-in catalog. A mode's reviewer is the model named in `--review-after <model>:<mode>`, else the mode's `default_model`, else the dispatch default. The instructions block is how you correct a model's habits from the catalog: it is appended to the system prompt of every worker that runs on that model, after the repository's own instruction files and the role memory. Write it as a multi-line string (one instruction per line) or as a list (one instruction per bullet); both mean the same thing. It is optional and capped at 4 KB per model - past that the tail is dropped, the cut is marked in the prompt and the load warns. The review phase uses the REVIEWER's block, so review habits never leak into the implementer's prompt. mini-swe-mcp manifest prints the resolved catalog with the instruction count per model and the review modes with their source (built-in / models.yaml); mini-swe-mcp help env lists the environment variables. Example: `review_modes: {security: {default_model: nerd}}`.";

#[cfg(test)]
mod tests {
    use crate::cli::help::topic_text;

    /// The topic has to be enough to write a correct `models.yaml` entry without
    /// reading the source: both spellings, the cap, and the reviewer's own block.
    #[test]
    fn models_topic_documents_the_instructions_block() {
        let text = topic_text("models").expect("models topic");
        for needle in [
            "models.yaml",
            "instructions:",
            "multi-line string",
            "list",
            "4 KB",
            "review phase uses the REVIEWER's block",
            "mini-swe-mcp manifest",
        ] {
            assert!(
                text.contains(needle),
                "the models topic must mention {needle}: {text}"
            );
        }
    }
}
