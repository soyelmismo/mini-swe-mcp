/// `collect`: the final message, with the diff summarised unless asked for.
pub(super) const TEXT: &str = "Run `mini-swe-mcp collect <worker_id>` for a finished worker's final message. The default reply is compact - summary, verification outcome, per-file diff stat and branch - because the full diff of a large task is what makes a review expensive. Pass --full for the whole diff, or --file <path> (repeatable) for the diff of named files only. Collect ends the worker's reviewable life: prefer `review` while the worker is still live, and collect once it is done.";

#[cfg(test)]
mod tests {
    use crate::cli::help::topic_text;
    /// `collect` documents the diff scope, because the default reply no longer
    /// carries the diff at all.
    #[test]
    fn collect_topic_teaches_the_diff_scope() {
        let text = topic_text("collect").expect("collect topic");
        for needle in ["--full", "--file", "per-file diff stat"] {
            assert!(
                text.contains(needle),
                "the collect topic must mention {needle}: {text}"
            );
        }
    }
}
