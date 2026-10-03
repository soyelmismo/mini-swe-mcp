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

mod archive;
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
    "archive",
    "identity",
    "sandbox",
    "env",
    "consolidate",
    "discard",
];

/// The whole `--help` page, one string, so the index is composed and tested in
/// the library instead of being printed from the binary.
///
/// Every action's usage line names its own verb: a caller pasting a line back
/// into the shell never repeats it.
pub fn index() -> String {
    static BODY: &str = concat!(
    "mini-swe-mcp {{VERSION}}\n",
    "Usage: mini-swe-mcp [--stdio | [--json] [--admin] <action> [args...]]\n",
    "\\nActions:\n",
    "  {{DISPATCH_USAGE}}\n",
    "           Start a worker on its own branch; always detaches.\n",
    "  watch [<worker_id>...] [--group <g>] [--all] [--follow] [--json] [--timeout <secs>]\n",
    "           Block until the next worker event (replaying missed ones), print it and exit;\n",
    "           run it in the background and the host CLI wakes you when it ends.\n",
    "  status <worker_id> | status --line\n",
    "           Final status/diff, or a one-line pool summary for statusLine.\n",
    "  collect <worker_id> [--full] [--file <path>]\n",
    "           Final message with a per-file diff stat; --full adds the whole diff,\n",
    "           --file narrows it to one path (repeatable).\n",
    "  review <worker_id> [--diff code|all|none]\n",
    "           One compact view of a finished worker: task, verification, the code\n",
    "           diff, per-file stat, tests summarised, and whether it still merges.\n",
    "  approve <worker_id> [\\\"note\\\"]\n",
    "           Record your verdict on a completed worker (owner-only).\n",
    "  unapprove <worker_id>\n",
    "           Withdraw that approval.\n",
    "  logs <worker_id>\n",
    "           Recent commands and their output.\n",
    "  steer <worker_id> <message> [--max-turns <n>]\n",
    "           Correct a completed worker or continue a stopped one.\n",
    "  consolidate {{CONSOLIDATE_USAGE}}\n",
    "           Integrate one group's round: merge the finished branches, run the full gate\n",
    "           once, route each failure to its owner, review every diff, and report.\n",
    "  list [--all]\n",
    "           Workers you own; --all (with --admin) lists every agent's.\n",
    "  kill <worker_id>\n",
    "           Terminate a worker.\n",
    "  discard <worker_id>\n",
    "           Drop a stopped worker for good: branch, row, history, steer files and\n",
    "           worktree leftovers, with no merge. A running worker is refused; kill it\n",
    "           first.\n",
    "  merge <worker_id> [--no-delete]\n",
    "           Merge a finished worker's branch into its base branch: trial merge,\n",
    "           verify gate on the merge result, then merge --no-ff and clean up.\n",
    "  merge --approved [--group <group>]\n",
    "           Land every approved worker of a group with ONE gate on the combined\n",
    "           result: a conflicting worker is skipped, the rest merge with --no-ff.\n",
    "  archive [--group <group>] [--last <n>]\n",
    "           Final reports of already-retired workers (merge, discard, retention):\n",
    "           done/files/tests/risks, the task, the gate verdict and why it left.\n",
    "  reap\n",
    "           Evict expired terminal worker records.\n",
    "  prune\n",
    "           Clean stale worktrees and caches.\n",
    "  manifest\n",
    "           Print the models catalog.\n",
    "  monitor [--once]\n",
    "           Full-screen view of the pool.\n",
    "  supervisor [--once]\n",
    "           Health view of workers and the hub.\n",
    "  daemon\n",
    "           Run the shared hub in the foreground.\n",
    "  whoami\n",
    "           Print this session's agent identity and how it was derived.\n",
    "  help <topic>\n",
    "           Long-form guidance on one concern (see Topics below).\n",
    "\\nTopics:\n",
    "  mini-swe-mcp help <topic>   {{TOPICS}}\n",
    "\\nFlags:\n",
    "{{HELP_FLAGS}}\n",
    );
    let body = BODY
        .replace("{VERSION}", env!("CARGO_PKG_VERSION"))
        .replace("{TOPICS}", &TOPICS.join(", "))
        .replace("{HELP_FLAGS}", HELP_FLAGS);
    body
}

/// Text of one help topic, or `None` for an unknown topic.
pub fn topic_text(topic: &str) -> Option<&'static str> {
    Some(match topic {
        "workflow" => workflow::TEXT,
        "watch" => watch::TEXT,
        "steer" => steer::TEXT,
        "review" => review::TEXT,
        "collect" => collect::TEXT,
        "merge" => merge::TEXT,
        "archive" => archive::TEXT,
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
        assert_eq!(TOPICS.len(), 12);
        for topic in TOPICS {
            let text = topic_text(topic).unwrap_or_else(|| panic!("'{topic}' has no text"));
            assert!(!text.trim().is_empty(), "'{topic}' is empty");
        }
        assert!(topic_text("nope").is_none());
    }
}
