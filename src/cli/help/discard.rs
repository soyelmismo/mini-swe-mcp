/// `discard`: drop a stopped worker you will never merge.
pub(super) const TEXT: &str = "`mini-swe-mcp discard <worker_id>` removes a stopped worker on purpose, with no merge and no verify gate: its `worker-<id>` branch, its registry row, its history JSONL, its steering mailbox and steer-source, its pinned round base, its watch acknowledgements and its worktree leftovers all go in one command. Use it for a worker that must go without ever landing - a failed consolidator whose gate can never run, a duplicate dispatch, a branch whose work was abandoned - instead of deleting the branch and every scratch file by hand. Like kill it is owner-only, so another agent's worker is refused. A running or paused worker is refused too, naming the `kill` to run first: a discard deletes unmerged work with no gate at all, so it must never be a quiet way to stop a worker that is still producing. Nothing it removes can be recovered, so read `collect <id>` first if the result is still wanted.";

#[cfg(test)]
mod tests {
    use crate::cli::help::topic_text;

    /// `discard` states the refusals and what it deletes, so an agent reads
    /// the rule before deleting unmerged work.
    #[test]
    fn discard_topic_names_the_refusals_and_the_leftovers() {
        let text = topic_text("discard").expect("discard topic");
        for needle in [
            "owner-only",
            "running or paused worker is refused",
            "kill",
            "round base",
            "no verify gate",
        ] {
            assert!(
                text.contains(needle),
                "the discard topic must mention {needle}: {text}"
            );
        }
    }
}
