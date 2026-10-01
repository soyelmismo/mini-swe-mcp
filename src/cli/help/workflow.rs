/// `workflow`: dispatch in parallel, review, and never edit a worker's branch.
pub(super) const TEXT: &str = "Write the task as ONE focused concern with files in scope and an acceptance gate. Dispatch independent tasks in parallel - many workers at once is the intended use; each worker integrates the latest base branch and resolves conflicts before completing. Split work so two workers do not rewrite the same function at the same time. To wait, run `mini-swe-mcp watch` in the background (see `mini-swe-mcp help watch`). After completion, review the diff and run the checks. Send every correction AND any merge conflict back to the same worker with steer (see `mini-swe-mcp help steer`). Do not edit its branch yourself; merge only when it is right. Use `mini-swe-mcp merge <id>` to trial merge, verify the merge result, merge --no-ff and clean up; --no-delete keeps the branch. A running worker, a wrong checked-out branch, a dirty touched file or a conflict refuses the merge.";

#[cfg(test)]
mod tests {
    use crate::cli::help::topic_text;
    /// `workflow` keeps the orchestrator guidelines the tool description used to
    /// carry inline.
    #[test]
    fn workflow_topic_keeps_the_orchestrator_guidelines() {
        let text = topic_text("workflow").expect("workflow topic");
        for needle in [
            "ONE focused concern",
            "many workers at once is the intended use",
            "mini-swe-mcp watch",
            "merge only when it is right",
        ] {
            assert!(
                text.contains(needle),
                "the workflow topic must mention {needle}: {text}"
            );
        }
    }
}
