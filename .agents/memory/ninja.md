PERSISTENT ROLE MEMORY (from .agents/memory/):
- Fast executor. Prefer the smallest edit that makes the tests green; do not refactor code you were not asked to touch.
- Establish the failing signal first (`cargo test --all-targets`, or the specific test module) before editing, so the fix is provably a fix.
- Both gates must pass before reporting completion: `cargo clippy --all-targets -- -D warnings` and `cargo test --all-targets`.
- One bash command per response; batch reads with a single `grep -rn`/`cat` instead of several round trips.
- This repository is a single-crate Rust workspace on edition 2024: sources under `src/`, integration tests under `tests/`, and the crate-level module docs (`//!`) are load-bearing.
- pool/runner is the single agent engine: change behaviour there once, not in each caller.
- Integration tests under tests/ use public pool/agent APIs; update them when removing APIs.
