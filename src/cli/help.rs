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
    "  -V, --version  Print version",
);

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
];

/// Text of one help topic, or `None` for an unknown topic.
pub fn topic_text(topic: &str) -> Option<&'static str> {
    Some(match topic {
        "workflow" => WORKFLOW,
        "watch" => WATCH,
        "steer" => STEER,
        "review" => REVIEW,
        "collect" => COLLECT,
        "merge" => MERGE,
        "identity" => IDENTITY,
        "sandbox" => SANDBOX,
        "env" => ENV,
        "consolidate" => CONSOLIDATE,
        _ => return None,
    })
}

/// `workflow`: dispatch in parallel, review, and never edit a worker's branch.
const WORKFLOW: &str = "Write the task as ONE focused concern with files in scope and an acceptance gate. Dispatch independent tasks in parallel - many workers at once is the intended use; each worker integrates the latest base branch and resolves conflicts before completing. Split work so two workers do not rewrite the same function at the same time. To wait, run `mini-swe-mcp watch` in the background (see `mini-swe-mcp help watch`). After completion, review the diff and run the checks. Send every correction AND any merge conflict back to the same worker with steer (see `mini-swe-mcp help steer`). Do not edit its branch yourself; merge only when it is right. Use `mini-swe-mcp merge <id>` to trial merge, verify the merge result, merge --no-ff and clean up; --no-delete keeps the branch. A running worker, a wrong checked-out branch, a dirty touched file or a conflict refuses the merge.";

/// `watch`: the process that wakes the orchestrator when a worker needs it.
const WATCH: &str = "Run `mini-swe-mcp watch` exactly as printed, using the host's own background mechanism (e.g. a background shell task) - no redirection, no trailing `&`, no wrapper: it blocks until the next actionable event - completion, failure, a question, or a stall - prints it and exits, and the host wakes you with the finished task's output; missed events are replayed first. After every event, run it again. A watch with no worker ids follows every worker you own, including any dispatched after it starts (--group still filters). One watch runs per session: a second is refused (exit 5) so the first is the one the next event wakes. Claude Code sessions started with channels enabled also receive the same events as push notifications. An agent with no shell can call the 'watch' action instead, passing timeout_secs below its host's tool deadline and calling it again on no_event.";

/// `steer`: correcting a live worker or continuing a stopped one.
const STEER: &str = "Send every correction and merge conflict to the same worker with `mini-swe-mcp steer <worker_id> <message>` rather than editing its branch yourself. Steering corrects a completed worker or continues any stopped one (failed, interrupted, killed): it resumes on its own worker-<id> branch with the full conversation plus this message, on a fresh turn budget (optional --max-turns, default 60). Never dispatch a replacement for a stopped worker. Review the diff and merge only when it is right.";

/// `review`: the one compact view of a finished worker's branch.
const REVIEW: &str = "Run `mini-swe-mcp review <worker_id>` for everything needed to decide what to do next, in one bounded reply: the task's first line, whether the verify gate passed (and the tail of its output when it did not), the diff of the code files, the per-file diff stat, the test files summarised by the cases added and removed, the docs touched with their +/- counts, the summary, the revision, and whether the branch still merges cleanly into the base branch tip. The diff is bounded, and a truncation says how many bytes it dropped. `--diff code|all|none` selects the code diff (the default), the whole diff, or none; test files are summarised instead of shown, so a change's test churn never dominates the reply. The merge check runs `git merge-tree --write-tree`, so it touches no worktree, no index and no lock. A clean branch ends with the merge to run; a conflicting one ends with the `steer` that sends the conflicts back to the worker that owns them. When the result is right, `mini-swe-mcp approve <worker_id> [\"note\"]` records your verdict on the completed worker (owner-only, and it survives collect; `mini-swe-mcp unapprove <worker_id>` withdraws it), while steering the worker into a new revision clears it because the branch changed. Review is a read: unlike collect it never evicts the worker.";

/// `collect`: the final message, with the diff summarised unless asked for.
const COLLECT: &str = "Run `mini-swe-mcp collect <worker_id>` for a finished worker's final message. The default reply is compact - summary, verification outcome, per-file diff stat and branch - because the full diff of a large task is what makes a review expensive. Pass --full for the whole diff, or --file <path> (repeatable) for the diff of named files only. Collect ends the worker's reviewable life: prefer `review` while the worker is still live, and collect once it is done.";

/// `merge`: land a reviewed worker's branch in one command.
const MERGE: &str = "`mini-swe-mcp merge <worker_id>` lands a finished worker's branch on the base branch recorded for it: it trial-merges, runs the verify gate on the merge result in a throwaway worktree, merges with --no-ff and then deletes the branch, its history file and its worktree leftovers. It refuses - changing nothing - while the worker still runs, while the repository has another branch checked out, while a file the merge would touch has uncommitted changes (untracked and unrelated files are left alone), or while the branch conflicts, in which case it names the conflicting files and the steer that sends them back. The gate is skipped when the branch already contains the base tip and the worker's last verify passed. `--no-delete` keeps the branch. It never pushes.";

/// `identity`: who owns a worker and which override sees everything.
const IDENTITY: &str = "A worker belongs to the agent that dispatched it: status, steer, kill, collect, logs, list and watch only ever see or act on your own workers. Your identity is derived per session and exported to the watch your shell runs, so the CLI and the MCP connection agree; `mini-swe-mcp whoami` prints it and how it was derived. The human operator's `--admin` override is the only way to act on another agent's workers (H-3).";

/// `sandbox`: worktree isolation, network policy and the verification gate.
const SANDBOX: &str = "Each dispatch runs in its own Git worktree on a worker-<id> branch, with bash steps in a sandbox. The optional network property (CLI --offline) selects the policy: offline runs every bash step in an isolated network namespace with no egress, allow (the default) keeps normal connectivity. Before a completion sentinel is honoured, the optional verify command - or one auto-detected from the repository layout - runs through the same sandboxed bash path; pass an empty string to disable the gate.";

/// `env`: the startup environment variables.
const ENV: &str = "Read at startup: OPENAI_API_KEY (required to dispatch), OPENAI_API_BASE (default https://api.openai.com/v1), DEFAULT_MODEL (the default model alias), MODELS_FILE (the models catalog), MINI_SWE_NO_DAEMON=1 (serve MCP in-process instead of through the hub), MINI_SWE_AGENT_ID (pin the session's agent identity), and MINI_SWE_WATCH_TOKEN (set by a dispatch so the watch its shell runs is attributed to your session). A .env file is loaded first.";

/// `consolidate`: the round workflow an orchestrator runs at the end of a round.
const CONSOLIDATE: &str = "A consolidated round has two kinds of worker. Dispatch the group's tasks as usual, but give every worker the CHEAP gate -- fmt, lint or type-check, and the tests of the files it touched -- never the full suite. When they are done, dispatch exactly ONE consolidator for the group: `mini-swe-mcp consolidate --group <g>` (MCP action 'consolidate', same 'group'). Set `strongest: <alias>` in models.yaml to pick the consolidator model; if absent, it uses the dispatch default. Pass `--model <m>` to override. It merges the completed branches, runs the project's FULL gate once, sends each file-attributable failure back to the worker that owns it (CONSOLIDATE_STEER) and waits for it (CONSOLIDATE_WAIT), fixes cross-worker interaction errors itself, reviews every diff, and finishes with a one-line-per-worker report. Read that report, then merge ONLY the consolidator's branch: it already carries the whole round.";

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
        assert_eq!(TOPICS.len(), 10);
        for topic in TOPICS {
            let text = topic_text(topic).unwrap_or_else(|| panic!("'{topic}' has no text"));
            assert!(!text.trim().is_empty(), "'{topic}' is empty");
        }
        assert!(topic_text("nope").is_none());
    }

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
        ] {
            assert!(
                text.contains(needle),
                "the watch topic must mention {needle}: {text}"
            );
        }
    }

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

    /// `consolidate` teaches the round workflow: the cheap worker gate, the one
    /// consolidator per group, the report, and the single branch to merge.
    #[test]
    fn consolidate_topic_teaches_the_round_workflow() {
        let text = topic_text("consolidate").expect("consolidate topic");
        for needle in [
            "CHEAP gate",
            "ONE consolidator",
            "mini-swe-mcp consolidate --group",
            "FULL gate",
            "CONSOLIDATE_STEER",
            "CONSOLIDATE_WAIT",
            "merge ONLY the consolidator's branch",
        ] {
            assert!(
                text.contains(needle),
                "the consolidate topic must mention {needle}: {text}"
            );
        }
    }

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
