/// `watch`: the process that wakes the orchestrator when a worker needs it.
pub(super) const TEXT: &str = "Run `mini-swe-mcp watch` exactly as printed, using the host's own background mechanism (e.g. a background shell task) - no redirection, no trailing `&`, no wrapper: it blocks until the next actionable event - completion, failure, a question, or a stall - prints it and exits, and the host wakes you with the finished task's output; missed events are replayed first. After every event, run it again. A watch with no worker ids follows every worker you own, including any dispatched after it starts (--group still filters). One watch runs per session: a second that asks for more widens the running one to the union and exits 0 at once printing `widened the running watch (pid N) to: <selection>`, while one it already covers exits 0 printing `already covered by the running watch (pid N): <selection>`; the running watch keeps its place and its missed events, and follows the union from its next wake. The MCP 'watch' action does not wait at all: it answers at once with the shell command that waits for the same selection the call asked for, because a tool call is bounded by the client's own timeout and the abort that ends it cannot deliver an event. Two plain watches stay plain (the union of their ids); `--all` on either side moves the union to rounds. Claude Code sessions started with channels enabled also receive the same events as push notifications. For whole rounds, run `mini-swe-mcp watch --group <g> --all`: it answers with ONE event for the round that lands - one compact line per worker of that round: id, outcome, verification and the report's done: line - as soon as every selected worker of it has stopped, or earlier when one of them needs input or fails, since those need you. Repeat --group as often as you like: several rounds running at once are then covered by that one watch, and it still answers for the first of them to finish, listing that round only; with no --group at all it covers every live group of yours. Run it again for the next round. Stalls are not reported in --all mode unless a worker goes 20 minutes without a step; the consolidator handles ordinary stalls. The round acknowledges its workers, so a later plain watch does not replay them. The action translates what the call asked for into that command: the ids or `--group` names, `--all` for whole rounds, and `timeout_secs` into `--timeout <secs>`.";

#[cfg(test)]
mod tests {
    use crate::cli::help::topic_text;
    /// `watch` carries the wait contract: the background command, the deadline
    /// its `timeout_secs` translates into, and the channel push.
    #[test]
    fn watch_topic_teaches_the_mcp_wait() {
        let text = topic_text("watch").expect("watch topic");
        for needle in [
            "mini-swe-mcp watch",
            "background shell task",
            "no redirection",
            "run it again",
            "timeout_secs",
            "--timeout",
            "push notifications",
            "--group <g> --all",
            "Repeat --group",
            "20 minutes",
            "widened the running watch",
            "already covered by the running watch",
        ] {
            assert!(
                text.contains(needle),
                "the watch topic must mention {needle}: {text}"
            );
        }
    }
}
