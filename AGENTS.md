# AGENTS.md

Rules for any agent working in this repository. They are loaded into a worker's
system prompt at dispatch time, so keep them short and actionable.

## Layout

- Single-crate Rust workspace, edition 2024: sources under `src/`, integration
  tests under `tests/`.
- Crate-level module docs (`//!`) state which properties are load-bearing; read
  them before editing a module.
- `pool/runner` is the single agent engine: change behaviour there once, not in
  every caller.
- Integration tests under `tests/` use the public `pool`/`agent` APIs; update
  them when an API changes.
- New verbs belong in action-family files under `src/mcp/handlers/`,
  `src/cli/args/` and `src/cli/format/worker/`, with one dispatch-table arm.
  Put property descriptions beside the handler and add one ordered row in
  `src/mcp/schema.rs`; help topics get one file in `src/cli/help/` plus an index entry.

## Gates (all must pass)

- `cargo fmt --check`
- `cargo clippy --all-targets -- -D warnings`
- `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps`
- `cargo test`

## Iterating

- While working, build and run only what you touch: `cargo test --test it <module>::`
  or `cargo test <name>`, and `cargo check` instead of a full build.
- A round's consolidator runs the full suite once, on the integrated result, so
  you do not have to run it per worker while you implement.
- Run the full gates (fmt --check, clippy, rustdoc, full test) once, right
  before requesting completion: the harness reuses an identical passing run
  and runs the divergent variant itself.

## Tests must be hermetic

- Give every file, directory, daemon or registry a test touches a temporary
  location passed to the code under test; the scratch helpers live in
  `tests/it/common/`.
- Do not mutate process-global state (environment variables) in tests that run
  in parallel.
- Do not write to the real registry, hub or repository.
- Poll for a condition instead of sleeping, and never assert something that
  cannot fail.

## What deserves a test

- Test what can break and matters: security and isolation properties, data
  loss, the wire and CLI output contracts, and a regression test for a real bug
  you fixed.
- Do not test wording: prompt text, nudge or help prose, log messages, a
  constant. Assert the behaviour instead - the block is present, it is ordered,
  it reaches the right model - never a copied sentence.
- One focused test per property: no re-testing what an existing test already
  covers, and no test for a trivial or mechanical change the gate already
  covers.
- Put tests in the area's existing test module; a new module only for a new
  area.

## Working style

- Locate code by name, not by the line numbers in the task: they may be stale.
- List in your REPORT any file outside the task's scope you had to touch.
- Never modify `models.yaml`: it is the operator's model catalog, not the task's.
- Reuse existing helpers (`tests/it/common/`, module-level functions) instead of
  copying a block of logic into a second place.
- Keep the diff to the task's scope, and file each test where the area's own
  tests live (`## What deserves a test`).
- Preserve the MCP tool contract, CLI output, wire formats and security
  properties unless the task says otherwise.
- Never run `git commit`, `git stash` or `git checkout` in the sandbox: the
  harness commits the work. Use git to inspect only.
- Emit exactly one bash command per response and batch reads into it.

## Verification

- Establish the failing signal before editing, so the fix is provably a fix.
- Do not call a failure pre-existing without showing it on the unmodified code.

A worker whose diff touches one of the paths below gets an automatic
adversarial security review (`--review-after <model>:security`) before it is
reported complete. Keep the list to the surfaces where a mistake is a security
defect, not a style one; the section is parsed as one glob per line.

## Sensitive paths

- AGENTS.md
- CLAUDE.md
- src/hub/**
- src/agent/sandbox*
- src/agent/exec*
- src/worktree/guard.rs
- src/pool/merge.rs
- src/pool/revision.rs
- src/hub/identity.rs
- src/mcp/events.rs
- src/agent/intercept.rs
- src/agent/env.rs
- src/pool/steer.rs
- src/mcp/handlers/**
- src/pool/runner/whole_file.rs
