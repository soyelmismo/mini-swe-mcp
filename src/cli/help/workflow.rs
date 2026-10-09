/// `workflow`: the round workflow for parallel work, and the single
/// worker's path for one-off tasks.
pub(super) const TEXT: &str = "Write the task as ONE focused concern with files in scope and an acceptance gate. Many workers at once is the intended use; for parallel work the default is a ROUND: put the tasks in one group (`dispatch -f tasks.yaml --group <g> --consolidate[=<model>]`, MCP `tasks` + `group` + `consolidate`), give every worker the CHEAP gate (fmt, lint or type-check, and the tests of the files it touched - never the full suite), wait with `watch --group <g> --all`, read the consolidator's one-line-per-worker report, hand-review only the security-relevant parts, then `merge <consolidator>` - its branch already carries the whole round (see `mini-swe-mcp help consolidate`). For a one-off task, dispatch one worker, review the diff, run the checks, and merge only when it is right. Split work so two workers never rewrite the same function at the same time; each worker integrates the latest base branch and resolves conflicts before completing. To wait, run `mini-swe-mcp watch` in the background (see `mini-swe-mcp help watch`). Send every correction AND any merge conflict back to the same worker with steer (see `mini-swe-mcp help steer`) - never edit its branch yourself. Do not move the base branch while a round is consolidating. `mini-swe-mcp merge <id>` trial-merges, verifies the merge result, merges --no-ff and cleans up; --no-delete keeps the branch. A running worker, a wrong checked-out branch, a dirty touched file or a conflict refuses the merge.";

#[cfg(test)]
mod tests {
    use crate::cli::help::topic_text;
    /// `workflow` keeps the orchestrator guidelines the tool description used to
    /// carry inline, now centred on the round workflow.
    #[test]
    fn workflow_topic_keeps_the_orchestrator_guidelines() {
        let text = topic_text("workflow").expect("workflow topic");
        for needle in [
            "ONE focused concern",
            "Many workers at once is the intended use",
            "mini-swe-mcp watch",
            "merge only when it is right",
        ] {
            assert!(
                text.contains(needle),
                "the workflow topic must mention {needle}: {text}"
            );
        }
    }

    /// The round is the default for parallel work: the group dispatch,
    /// the cheap worker gate, the `--all` wait, the consolidator's
    /// report and the single branch to merge.
    #[test]
    fn workflow_topic_teaches_the_round_workflow() {
        let text = topic_text("workflow").expect("workflow topic");
        for needle in [
            "the default is a ROUND",
            "dispatch -f tasks.yaml --group <g> --consolidate",
            "CHEAP gate",
            "watch --group <g> --all",
            "consolidator's one-line-per-worker report",
            "hand-review only the security-relevant parts",
            "merge <consolidator>",
            "never edit its branch yourself",
            "Do not move the base branch while a round is consolidating",
        ] {
            assert!(
                text.contains(needle),
                "the workflow topic must mention {needle}: {text}"
            );
        }
    }
}
