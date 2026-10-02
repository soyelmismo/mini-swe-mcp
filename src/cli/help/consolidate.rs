/// `consolidate`: the round workflow an orchestrator runs at the end of a round.
pub(super) const TEXT: &str = "A consolidated round has two kinds of worker. Dispatch the group's tasks as usual, but give every worker the CHEAP gate -- fmt, lint or type-check, and the tests of the files it touched -- never the full suite. When they are done, dispatch exactly ONE consolidator for the group: `mini-swe-mcp consolidate --group <g>` (MCP action 'consolidate', same 'group'). Set `strongest: <alias>` in models.yaml to pick the consolidator model; if absent, it uses the dispatch default. Pass `--model <m>` to override. It merges the completed branches, runs the project's FULL gate once, sends each file-attributable failure back to the worker that owns it (CONSOLIDATE_STEER) and waits for it (CONSOLIDATE_WAIT), fixes cross-worker interaction errors itself, reviews every diff, and finishes with a one-line-per-worker report. Alternatively, pass `dispatch --consolidate[=<model>]` (also with -f; MCP `consolidate: true` or a model alias) to have the hub start it for you. Such a dispatch without an explicit `--verify` gives its workers the CHEAP gate auto-detected for the project (Rust: `cargo fmt --check && cargo clippy --all-targets -- -D warnings`; Node/TS: the package's `lint` and `typecheck` scripts, else `tsc --noEmit`; Python: `ruff check .` plus `mypy .` when configured; Go: `gofmt -l . && go vet ./...`; otherwise none) -- an explicit `--verify` always wins. The consolidator itself always runs the FULL auto-detected gate; `--consolidate-verify <cmd>` (MCP `consolidate_verify`) overrides it. The owner/group setting survives hub restarts. The hub waits for every worker to stop and at least one to complete; failed, paused, exhausted or interrupted workers block it until resolved or explicitly stopped. The orchestrator receives those workers' events and must resolve them. A later dispatch into the same group after its consolidator started opens a new round. Wait for the round with `mini-swe-mcp watch --group <g> --all` (or the 'watch' action with `all: true`): one event once every worker of the group has stopped, or earlier when one needs input or fails. Read that report, then merge ONLY the consolidator's branch: it already carries the whole round.";

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
        ] {
            assert!(
                text.contains(needle),
                "the consolidate topic must mention {needle}: {text}"
            );
        }
    }
}
