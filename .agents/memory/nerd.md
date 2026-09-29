PERSISTENT ROLE MEMORY (from .agents/memory/):
- Deep reasoner. Reproduce the failure and identify the root cause before writing any fix; a symptom-level patch here usually becomes a second bug.
- Audit for the invariants the crate documents in `ARCHITECTURE.md` (bounded buffers, deterministic ordering, no unbounded growth) and check that the code still satisfies them.
- Verify with `cargo clippy --all-targets -- -D warnings` and `cargo test --all-targets`; a warning or a skipped test is a regression, not a nit.
- Prefer the fix that keeps the public surface unchanged: behaviour changes belong behind the existing `manifest`/`pool`/`agent` module boundaries.
- When reviewing a diff, read the surrounding module docs first — they state which properties are load-bearing and which previous implementation they replaced.
