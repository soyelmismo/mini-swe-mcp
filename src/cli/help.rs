//! The `--help` flag reference, next to the argv parsing it documents.

/// The `Flags:` block of `--help`, one flag per line.
///
/// `--admin` is the human operator's escape hatch: every other caller may only
/// read, steer, kill, collect and watch the workers it dispatched itself, while the
/// operator can act on any of them (H-3).
pub const HELP_FLAGS: &str = concat!(
    "      --json     Output in JSON format (default is formatted plain text)\n",
    "      --all      List every agent's workers (requires --admin)\n",
    "      --admin    Operator override: act on workers owned by any agent\n",
    "  -h, --help     Print help\n",
    "  -V, --version  Print version\n",
    "      --build-id  Print this build's identity (id and build clock)",
);

mod collect;
mod consolidate;
mod discard;
mod env;
mod identity;
mod merge;
mod review;
mod sandbox;
mod steer;
mod watch;
mod workflow;

/// Topics accepted by `mini-swe-mcp help <topic>`.
///
/// The long-form guidance the MCP tool description used to carry inline lives
/// here, one concern per topic, so an agent can fetch exactly what it needs
/// without paying for all of it in every session's context.
pub const TOPICS: &[&str] = &[
    "workflow",
    "watch",
    "steer",
    "review",
    "collect",
    "merge",
    "identity",
    "sandbox",
    "env",
    "consolidate",
    "discard",
];

/// Text of one help topic, or `None` for an unknown topic.
pub fn topic_text(topic: &str) -> Option<&'static str> {
    Some(match topic {
        "workflow" => workflow::TEXT,
        "watch" => watch::TEXT,
        "steer" => steer::TEXT,
        "review" => review::TEXT,
        "collect" => collect::TEXT,
        "merge" => merge::TEXT,
        "identity" => identity::TEXT,
        "sandbox" => sandbox::TEXT,
        "env" => env::TEXT,
        "consolidate" => consolidate::TEXT,
        "discard" => discard::TEXT,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::{HELP_FLAGS, TOPICS, topic_text};

    /// Every flag the argv parser knows is documented, so a flag can never be
    /// accepted without being discoverable.
    #[test]
    fn every_selector_is_documented() {
        for flag in [
            "--json",
            "--all",
            "--admin",
            "-h",
            "--help",
            "-V",
            "--version",
        ] {
            assert!(
                HELP_FLAGS.contains(flag),
                "'{flag}' is accepted by the parser but missing from --help: {HELP_FLAGS}"
            );
        }
    }

    /// Every documented topic resolves to non-empty text; an unknown one does
    /// not, so `help <topic>` can refuse it with the available list.
    #[test]
    fn every_topic_has_text_and_unknown_ones_do_not() {
        assert_eq!(TOPICS.len(), 11);
        for topic in TOPICS {
            let text = topic_text(topic).unwrap_or_else(|| panic!("'{topic}' has no text"));
            assert!(!text.trim().is_empty(), "'{topic}' is empty");
        }
        assert!(topic_text("nope").is_none());
    }
}
