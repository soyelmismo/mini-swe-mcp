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
- `cargo test`

## Iterating

- While working, build and run only what you touch: `cargo test --test <file>`
  or `cargo test <name>`, and `cargo check` instead of a full build.
- Run the full gates (fmt --check, clippy, full test) once, right before
  requesting completion: the harness reuses an identical passing run and runs
  the divergent variant itself.

## Tests must be hermetic

- Give every file, directory, daemon or registry a test touches a temporary
  location passed to the code under test; the scratch helpers live in
  `tests/common/`.
- Do not mutate process-global state (environment variables) in tests that run
  in parallel.
- Do not write to the real registry, hub or repository.
- Poll for a condition instead of sleeping, and never assert something that
  cannot fail.

## Working style

- Reuse existing helpers (`tests/common/`, module-level functions) instead of
  copying a block of logic into a second place.
- Keep the diff to the task's scope; put new tests in a file dedicated to the
  change rather than at the end of a large shared test file.
- Preserve the MCP tool contract, CLI output, wire formats and security
  properties unless the task says otherwise.
- Never run `git commit`, `git stash` or `git checkout` in the sandbox: the
  harness commits the work. Use git to inspect only.
- Emit exactly one bash command per response and batch reads into it.

## Verification

- Establish the failing signal before editing, so the fix is provably a fix.
- Do not call a failure pre-existing without showing it on the unmodified code.
