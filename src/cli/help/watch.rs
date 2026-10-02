/// `watch`: the process that wakes the orchestrator when a worker needs it.
pub(super) const TEXT: &str = "Run `mini-swe-mcp watch` exactly as printed, using the host's own background mechanism (e.g. a background shell task) - no redirection, no trailing `&`, no wrapper: it blocks until the next actionable event - completion, failure, a question, or a stall - prints it and exits, and the host wakes you with the finished task's output; missed events are replayed first. After every event, run it again. A watch with no worker ids follows every worker you own, including any dispatched after it starts (--group still filters). One watch runs per session: a second that asks for more widens the running one to the union (exits 0 with `widened the running watch`), one it already covers exits 0 with `already covered`. Claude Code sessions started with channels enabled also receive the same events as push notifications. For whole rounds, run `mini-swe-mcp watch --group <g> --all` (MCP action 'watch', `all: true` with `group` or explicit ids): it answers with ONE event for the round that lands - one compact line per worker of that round: id, outcome, verification and the report's done: line - as soon as every selected worker of it has stopped, or earlier when one of them needs input or fails, since those need you. Repeat --group as often as you like (MCP: pass `group` an array of names): several rounds running at once are then covered by that one watch, and it still answers for the first of them to finish, listing that round only; with no --group at all it covers every live group of yours. Run it again for the next round. Stalls are not reported in --all mode unless a worker goes 20 minutes without a step; the consolidator handles ordinary stalls. The round acknowledges its workers, so a later plain watch does not replay them. An agent with no shell can call the 'watch' action instead, passing timeout_secs below its host's tool deadline and calling it again on no_event.";

#[cfg(test)]
mod tests {
    use crate::cli::help::topic_text;
    /// `watch` carries the shell-less wait contract: the background command,
    /// the `timeout_secs`/`no_event` fallback, and the channel push.
    #[test]
    fn watch_topic_teaches_the_mcp_wait() {
        let text = topic_text("watch").expect("watch topic");
        for needle in [
            "mini-swe-mcp watch",
            "background shell task",
            "no redirection",
            "run it again",
            "timeout_secs",
            "no_event",
            "push notifications",
            "--group <g> --all",
            "Repeat --group",
            "all: true",
            "20 minutes",
        ] {
            assert!(
                text.contains(needle),
                "the watch topic must mention {needle}: {text}"
            );
        }
    }
}
