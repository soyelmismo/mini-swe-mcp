use super::*;

/// `topic` property of the `help` action: which topic's text to
/// return, or none for the topic index.
pub(crate) const TOPIC_DESCRIPTION: &str = "Help topic; omitted: index";

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pool::WorkerPool;

    fn server() -> McpServer {
        McpServer::new(
            WorkerPool::new(1, "http://localhost:1".to_string(), "test-key".to_string()),
            "ninja".to_string(),
        )
    }

    /// The index an MCP-only agent gets instead of the CLI: every topic the
    /// `help <topic>` topics accept, so nothing is unreachable without a shell.
    #[test]
    fn no_topic_returns_the_index() {
        let reply = server().handle_help(&json!({})).expect("the index");
        let topics: Vec<&str> = reply["topics"]
            .as_array()
            .expect("an array of topics")
            .iter()
            .map(|topic| topic.as_str().expect("topic names are strings"))
            .collect();
        assert_eq!(topics, crate::cli::help::TOPICS);
    }

    /// A topic returns exactly the text the CLI prints for it, so the two
    /// paths can never drift.
    #[test]
    fn a_topic_returns_the_same_text_as_the_cli() {
        let reply = server()
            .handle_help(&json!({ "topic": "workflow" }))
            .expect("the workflow topic");
        assert_eq!(reply["topic"], json!("workflow"));
        assert_eq!(
            reply["text"],
            json!(crate::cli::help::topic_text("workflow").expect("workflow topic"))
        );
    }

    /// Every indexed topic is readable, so the index never advertises one the
    /// action would refuse.
    #[test]
    fn every_indexed_topic_resolves() {
        for topic in crate::cli::help::TOPICS {
            let reply = server()
                .handle_help(&json!({ "topic": topic }))
                .unwrap_or_else(|error| panic!("topic '{topic}' must resolve: {error}"));
            assert!(
                !reply["text"].as_str().unwrap_or_default().is_empty(),
                "topic '{topic}' must not answer empty"
            );
        }
    }

    /// A typo names the alternatives rather than answering with nothing.
    #[test]
    fn an_unknown_topic_is_refused_with_the_available_topics() {
        let error = server()
            .handle_help(&json!({ "topic": "nope" }))
            .expect_err("an unknown topic must be refused");
        let message = error.to_string();
        assert!(message.contains("Unknown help topic: nope"), "{message}");
        assert!(message.contains("workflow"), "{message}");
    }
}
