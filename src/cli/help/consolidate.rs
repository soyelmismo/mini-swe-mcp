/// `consolidate`: the round workflow an orchestrator runs at the end of a round.
pub(super) const TEXT: &str = "A consolidated round has two kinds of worker. Dispatch the group's tasks as usual, but give every worker the CHEAP gate -- fmt, lint or type-check, and the tests of the files it touched -- never the full suite. When they are done, dispatch exactly ONE consolidator for the group: `mini-swe-mcp consolidate --group <g>` (MCP action 'consolidate', same 'group'). The consolidator model is `--consolidate=<model>` (MCP `consolidate: \"<model>\"`, CLI `consolidate --model <m>`), else the dispatch default; the automatic security reviewer is the security review mode's `default_model` (see `mini-swe-mcp help models`), else the dispatch default. A worker whose diff touches a path declared sensitive (AGENTS.md `## Sensitive paths`, or `sensitive_paths:` in models.yaml) is audited by an ADVERSARIAL security reviewer -- on the security mode's reviewer, never on the model that wrote the diff; `dispatch --review-after <model>:security` overrides it. Pass `--model <m>` to override. It merges the completed branches, runs the project's FULL gate once, sends each file-attributable failure back to the worker that owns it (CONSOLIDATE_STEER) and waits for it (CONSOLIDATE_WAIT), fixes cross-worker interaction errors itself, reviews every diff, and finishes with a one-line-per-worker report. Alternatively, pass `dispatch --consolidate[=<model>]` (also with -f; MCP `consolidate: true` or a model alias) to have the hub start it for you. `--consolidate-verify <cmd>` (MCP `consolidate_verify`) overrides the auto-detected full gate; The consolidator itself always runs the FULL auto-detected gate. Such a dispatch without an explicit `--verify` gives its workers the CHEAP gate auto-detected for the project (Rust: `cargo fmt --check && cargo clippy --all-targets -- -D warnings`; Node/TS: the package's `lint` and `typecheck` scripts, else `tsc --noEmit`; Python: `ruff check .` plus `mypy .` when configured; Go: `gofmt -l . && go vet ./...`; otherwise none) -- an explicit `--verify` always wins. both gates and this one are parse-checked with `sh -n` at dispatch and never run, so a quote-split command is refused there instead of reaching a consolidator. To change a round after the fact, amend it: `mini-swe-mcp consolidate --group <g> --set [--model <m>] [--verify <cmd>]` (MCP action 'consolidate' with `set: true`) rewrites the settings of your group's PENDING round through the hub -- editing `<hub_dir>/auto-consolidate.json` by hand does not, because the daemon holds the rounds in memory and rewrites that file. It refuses a round whose consolidator already ran (consumed), and any other owner's round. Omit `--model`/`--verify` to leave that half as it was; pass an empty `--verify` to clear the gate back to auto-detect. The owner/group setting survives hub restarts. The hub waits for every worker to stop and at least one to complete; failed, paused, exhausted or interrupted workers block it until resolved or explicitly stopped. The orchestrator receives those workers' events and must resolve them. A later dispatch into the same group after its consolidator started opens a new round. Wait for the round with `mini-swe-mcp watch --group <g> --all` (or the 'watch' action with `all: true`): one event once every worker of the group has stopped, or earlier when one needs input or fails. Read that report, then merge ONLY the consolidator's branch: it already carries the whole round.";

#[cfg(test)]
mod tests {
    use crate::cli::help::topic_text;
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
            "watch --group <g> --all",
            "--set",
            "sh -n",
        ] {
            assert!(
                text.contains(needle),
                "the consolidate topic must mention {needle}: {text}"
            );
        }
    }
}
