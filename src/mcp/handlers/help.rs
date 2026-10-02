use super::*;

/// `topic` property of the `help` action: which topic's text to
/// return, or none for the topic index.
pub(super) const TOPIC_DESCRIPTION: &str =
    "Help topic to read; omitted returns the topic index.";

impl McpServer {
    /// `help` action: the long-form guidance the CLI keeps per topic.
    ///
    /// No argument returns the topic index (the same list the CLI
    /// prints after `mini-swe-mcp help`); an unknown topic is refused
    /// with the same list, so a typo names its alternatives instead of
    /// answering with nothing. Read-only and agent-independent: help
    /// is the one topic every agent of a hub may read, whatever it
    /// dispatched.
    pub(super) fn handle_help(&self, args: &Value) -> Result<Value> {
        let topic = args
            .get("topic")
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty());
        match topic {
            None => Ok(json!({
                "status": "help",
                "topics": crate::cli::help::TOPICS,
            })),
            Some(topic) => match crate::cli::help::topic_text(topic) {
                Some(text) => Ok(json!({
                    "status": "help",
                    "topic": topic,
                    "text": text,
                })),
                None => anyhow::bail!(
                    "Unknown help topic: {topic}; topics: {}",
                    crate::cli::help::TOPICS.join(", ")
                ),
            },
        }
    }
}
