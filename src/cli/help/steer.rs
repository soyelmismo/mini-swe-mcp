/// `steer`: correcting a live worker or continuing a stopped one.
pub(super) const TEXT: &str = "Send every correction and merge conflict to the same worker with `mini-swe-mcp steer <worker_id> <message>` rather than editing its branch yourself. Steering corrects a completed worker or continues any stopped one (failed, interrupted, killed): it resumes on its own worker-<id> branch with the full conversation plus this message, on a fresh turn budget (optional --max-turns, default 60). Never dispatch a replacement for a stopped worker. Review the diff and merge only when it is right. Quote the message only when it must survive the shell verbatim: every unquoted word after the worker id is the message, joined with single spaces, and no word is ever dropped.";

#[cfg(test)]
mod tests {
    use crate::cli::help::topic_text;
    /// `steer` is the one verb that continues a stopped worker, so its topic
    /// spells out every stopped state and the branch it resumes on.
    #[test]
    fn steer_topic_covers_every_stopped_state() {
        let text = topic_text("steer").expect("steer topic");
        for needle in [
            "completed",
            "failed",
            "interrupted",
            "killed",
            "worker-<id>",
            "--max-turns",
        ] {
            assert!(
                text.contains(needle),
                "the steer topic must mention {needle}: {text}"
            );
        }
    }
}
