/// `review`: the one compact view of a finished worker's branch.
pub(super) const TEXT: &str = "Run `mini-swe-mcp review <worker_id>` for everything needed to decide what to do next, in one bounded reply: the task's first line, whether the verify gate passed (and the tail of its output when it did not), the diff of the code files, the per-file diff stat, the test files summarised by the cases added and removed, the docs touched with their +/- counts, the summary, the revision, and whether the branch still merges cleanly into the base branch tip. The diff is bounded, and a truncation says how many bytes it dropped. `--diff code|all|none` selects the code diff (the default), the whole diff, or none; test files are summarised instead of shown, so a change's test churn never dominates the reply. The merge check runs `git merge-tree --write-tree`, so it touches no worktree, no index and no lock. A clean branch ends with the merge to run; a conflicting one ends with the `steer` that sends the conflicts back to the worker that owns them. When the result is right, `mini-swe-mcp approve <worker_id> [\"note\"]` records your verdict on the completed worker (owner-only, and it survives collect; `mini-swe-mcp unapprove <worker_id>` withdraws it), while steering the worker into a new revision clears it because the branch changed. Review is a read: unlike collect it never evicts the worker.";

#[cfg(test)]
mod tests {
    use crate::cli::help::topic_text;
    /// `review` is the verb that answers "what do I do with this branch", so
    /// its topic names the merge check, its read-only nature and the command it
    /// ends with.
    #[test]
    fn review_topic_teaches_the_compact_view() {
        let text = topic_text("review").expect("review topic");
        for needle in [
            "mini-swe-mcp review",
            "git merge-tree --write-tree",
            "never evicts",
            "steer",
            "--diff code|all|none",
            "test files",
            "mini-swe-mcp approve",
        ] {
            assert!(
                text.contains(needle),
                "the review topic must mention {needle}: {text}"
            );
        }
    }
}
