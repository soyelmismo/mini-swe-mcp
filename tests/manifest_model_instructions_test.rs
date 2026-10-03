//! Integration tests for the per-model `instructions:` block of `models.yaml`.
//!
//! The block is how an operator corrects a model's habits from the catalog (a
//! small model that reads files in many small ranges is told to read whole
//! files). These go through the public library surface only, i.e. exactly the
//! API the pool runner uses:
//!
//! * [`ModelInstructions`] accepts both YAML spellings — a multi-line string and
//!   a list — and normalizes them into one ordered list of entries.
//! * [`build_system_prompt`] appends that model's block to a worker's prompt,
//!   after the repository instructions and the role memory, and leaves the
//!   prompt untouched for a model that declares none.
//! * [`ModelManifest::instructions_for`] resolves the block by alias *or* by
//!   full model id, which is what makes the review phase carry the reviewer's
//!   rules rather than the implementer's.
//! * The block is bounded: past [`MAX_MODEL_INSTRUCTIONS_BYTES`] it is warned
//!   about and cut, with the cut marked so a short block is never mistaken for
//!   the whole one.
//!
//! The behaviour lives in the crate's unit tests (`src/manifest/tests.rs`); this
//! file pins the *public* surface — the names, signatures and visibility the rest
//! of the crate (and any downstream binary) depends on.

mod common;

use common::TempDir;
use mini_swe_mcp::agent::SYSTEM_PROMPT;
use mini_swe_mcp::manifest::{
    MAX_MODEL_INSTRUCTIONS_BYTES, ModelInstructions, ModelManifest, build_system_prompt,
};
use std::path::Path;

/// Parse a `models.yaml` the way the loader does.
fn parse_manifest(yaml: &str) -> ModelManifest {
    serde_yaml::from_str::<ModelManifest>(yaml)
        .unwrap_or_else(|e| panic!("manifest YAML must parse: {e}\n{yaml}"))
        .normalize()
}

// ----------
// 1. Both YAML spellings parse into the same list
// ----------

#[test]
fn test_string_and_list_forms_parse_into_one_ordered_list() {
    let string_form = parse_manifest(
        r#"
models:
  small:
    id: combo:small
    instructions: |
      Read whole files instead of many small ranges.
      Run the cheap gate before the full suite.
"#,
    );
    let list_form = parse_manifest(
        r#"
models:
  small:
    id: combo:small
    instructions:
      - Read whole files instead of many small ranges.
      - Run the cheap gate before the full suite.
"#,
    );

    let from_string = string_form.instructions_for("small").expect("string form");
    let from_list = list_form.instructions_for("small").expect("list form");
    assert_eq!(
        from_string.entries(),
        &[
            "Read whole files instead of many small ranges.".to_string(),
            "Run the cheap gate before the full suite.".to_string(),
        ]
    );
    assert_eq!(
        from_string.entries(),
        from_list.entries(),
        "both spellings must normalize to the same ordered entries"
    );
    assert_eq!(from_list.len(), 2, "the count is what `manifest` reports");
}

/// Blank lines are dropped instead of rendered as empty bullets, and a bullet
/// marker written by hand is stripped so the two spellings read the same.
#[test]
fn test_empty_entries_are_ignored_and_bullets_are_stripped() {
    let manifest = parse_manifest(
        r#"
models:
  small:
    id: combo:small
    instructions:
      - "  "
      - "- Read whole files."
      - ""
"#,
    );

    let block = manifest.instructions_for("small").expect("block");
    assert_eq!(block.entries(), ["Read whole files.".to_string()]);
}

/// A block of nothing but blank lines declares nothing at all: it is repaired
/// away, so the model is indistinguishable from one that never declared it.
#[test]
fn test_a_blank_only_block_normalizes_to_nothing() {
    let manifest = parse_manifest(
        r#"
models:
  small:
    id: combo:small
    instructions:
      - ""
      - "   "
"#,
    );

    assert!(manifest.validate().is_empty(), "{:?}", manifest.validate());
    assert!(manifest.models["small"].instructions.is_none());
    assert!(manifest.instructions_for("small").is_none());
}

// ----------
// 2. The system prompt of a worker running that model
// ----------

#[test]
fn test_prompt_of_a_model_with_instructions_contains_them_and_one_without_does_not() {
    let repo = TempDir::new_in_tmp("model-instructions");
    let manifest = parse_manifest(
        r#"
models:
  small:
    id: combo:small
    instructions: |
      Prefer one whole-file read over a dozen small ranges.
  deep:
    id: combo:deep
"#,
    );

    let with = build_system_prompt(&manifest, repo.path(), "small");
    let without = build_system_prompt(&manifest, repo.path(), "deep");

    assert!(
        with.contains("Prefer one whole-file read over a dozen small ranges."),
        "a worker of `small` must be told to read whole files: {with}"
    );
    assert!(
        !without.contains("Prefer one whole-file read"),
        "a worker of `deep` must not inherit `small`'s rules: {without}"
    );
    // The heading names the source, so a worker can tell these rules from the
    // repository's own.
    assert!(with.contains("Model-specific instructions"), "{with}");
    // No model-specific rules declared anywhere means byte-identical behaviour
    // to before this block existed.
    let bare = ModelManifest::default();
    assert_eq!(
        build_system_prompt(&bare, repo.path(), "ninja"),
        SYSTEM_PROMPT,
        "a catalog without instructions must leave the prompt untouched"
    );
}

/// The block comes last: repository instructions, then role memory, then the
/// model-specific rules, so the most specific rules read last.
#[test]
fn test_instructions_are_appended_after_the_repository_files_and_memory() {
    let repo = TempDir::new_in_tmp("model-instructions-order");
    std::fs::write(repo.path().join("AGENTS.md"), "Run the gates.\n").expect("fixture");
    std::fs::create_dir_all(repo.path().join(".agents/memory")).expect("memory dir");
    std::fs::write(
        repo.path().join(".agents/memory/small.md"),
        "PERSISTENT ROLE MEMORY (from .agents/memory/):\n- Reproduce before you patch.\n",
    )
    .expect("fixture");
    let manifest = parse_manifest(
        r#"
models:
  small:
    id: combo:small
    instructions: Read whole files instead of many small ranges.
"#,
    );

    let prompt = build_system_prompt(&manifest, repo.path(), "small");
    let repo_rules = prompt.find("Run the gates.").expect("repository rules");
    let memory = prompt.find("Reproduce before you patch.").expect("memory");
    let specific = prompt
        .find("Read whole files instead of many small ranges.")
        .expect("model instructions");
    assert!(repo_rules < memory, "memory follows the repository rules");
    assert!(memory < specific, "model instructions come last: {prompt}");
}

// ----------
// 3. The review phase uses the reviewer's model
// ----------

/// The runner maps a resolved id back to its alias and builds the review prompt
/// from *that* alias, so a reviewer's rules never bleed into the implementer's
/// prompt. Resolving by full id is what the review phase actually does with
/// `--review-after <alias>`, once the dispatch has turned the alias into
/// `combo:deep`.
#[test]
fn test_the_reviewer_model_instructions_apply_to_the_reviewer() {
    let manifest = parse_manifest(
        r#"
models:
  small:
    id: combo:small
    instructions: Read whole files instead of many small ranges.
  deep:
    id: combo:deep
    instructions: |
      Question the assumption before you accept it.
      A passing test is not proof the bug is fixed.
"#,
    );

    let implementer = manifest.alias_for_model("combo:small");
    let reviewer = manifest.alias_for_model("combo:deep");
    let reviewer_block = manifest
        .instructions_for(&reviewer)
        .expect("reviewer block");

    assert_eq!(
        reviewer_block.entries()[0],
        "Question the assumption before you accept it."
    );
    // The implementer's own model is what the implementation prompt carries, and
    // neither block mentions the other's rule.
    assert_eq!(
        manifest
            .instructions_for(&implementer)
            .expect("implementer block")
            .entries()[0],
        "Read whole files instead of many small ranges."
    );
    assert!(
        !reviewer_block
            .entries()
            .iter()
            .any(|e| e.contains("many small ranges"))
    );
}

/// An unknown model passes through and finds no block, which keeps a
/// pass-through id (a model the catalog does not define) working unchanged.
#[test]
fn test_an_unknown_model_declares_no_instructions() {
    let manifest = parse_manifest(
        r#"
models:
  small:
    id: combo:small
    instructions: Read whole files.
"#,
    );

    assert!(manifest.instructions_for("some/unknown").is_none());
    assert!(
        ModelInstructions::default().is_empty(),
        "the default block is empty"
    );
}

// ----------
// 4. The length cap
// ----------

/// Past the budget the block is warned about and cut, and the kept head is
/// marked as truncated so the worker is never misled into thinking it has the
/// whole rulebook.
#[test]
fn test_instructions_over_the_budget_are_warned_about_and_truncated() {
    let entry = "x".repeat(MAX_MODEL_INSTRUCTIONS_BYTES);
    let yaml = format!(
        "models:\n  small:\n    id: combo:small\n    instructions:\n      - {entry}\n      - \
         never reached\n"
    );

    // The warning belongs to the un-repaired manifest: `normalize` is what cuts
    // the block, and it runs after validation on the load path.
    let raw: ModelManifest = serde_yaml::from_str(&yaml).expect("parses");
    let warnings = raw.validate();
    assert!(
        warnings
            .iter()
            .any(|w| w.contains("instructions") && w.contains("budget")),
        "an over-long block must be reported: {warnings:?}"
    );

    let raw = raw.normalize();
    let block = raw.instructions_for("small").expect("block");
    assert!(
        block.rendered_len() <= MAX_MODEL_INSTRUCTIONS_BYTES,
        "the kept block must fit the budget: {} bytes",
        block.rendered_len()
    );
    assert!(block.is_truncated(), "the cut is recorded on the block");

    let repo = TempDir::new_in_tmp("model-instructions-cap");
    let prompt = build_system_prompt(&raw, repo.path(), "small");
    assert!(!prompt.contains("never reached"), "the tail is dropped");
    assert!(prompt.contains("[truncated"), "the cut is marked: {prompt}");
}

/// Normalizing is idempotent: a second pass neither drops another entry nor
/// warns again, so a manifest repaired on load stays repaired.
#[test]
fn test_normalizing_a_truncated_block_is_idempotent() {
    let entry = "y".repeat(MAX_MODEL_INSTRUCTIONS_BYTES + 100);
    let manifest = parse_manifest(&format!(
        "models:\n  small:\n    id: combo:small\n    instructions:\n      - {entry}\n      - tail\n"
    ))
    .normalize()
    .normalize();

    let block = manifest.instructions_for("small").expect("block");
    assert_eq!(block.entries().len(), 1);
    assert!(manifest.validate().is_empty(), "{:?}", manifest.validate());
    assert_eq!(
        serde_yaml::to_string(&block).expect("serializes"),
        serde_yaml::to_string(
            manifest.models["small"]
                .instructions
                .as_ref()
                .expect("declares")
        )
        .expect("serializes"),
        "the list form is what serializes back out"
    );
}

// ----------
// 5. End to end: the review phase of a real dispatch
// ----------

/// Dispatch a worker with `--review-after` against the fake LLM and return the
/// system prompt of every request the run made, implementation turns first.
async fn system_prompts_of_a_reviewed_run(repo: &Path, manifest_yaml: &str) -> Vec<String> {
    // `echo` completes immediately; the reviewer then gets one completion turn.
    let llm = common::fake_llm::FakeLlm::spawn_gate_then_complete("echo baseline").await;
    let scratch = TempDir::new_in_tmp("model-instructions-e2e");
    let manifest = parse_manifest(manifest_yaml);
    let pool = mini_swe_mcp::pool::WorkerPool::with_scratch(
        1,
        llm.base_url().to_string(),
        "test-key".to_string(),
        mini_swe_mcp::worktree::ScratchRoot::new(scratch.path()),
    )
    .with_manifest(std::sync::Arc::new(manifest));

    let worker_id = pool
        .dispatch(
            "e2e-owner".to_string(),
            "exercise the review phase".to_string(),
            "small".to_string(),
            None,
            repo.to_path_buf(),
            12,
            Some("e2e".to_string()),
            Some("deep".to_string()),
            false,
            None,
            Vec::new(),
        )
        .await
        .expect("dispatch the worker");

    // Poll for the terminal state rather than sleeping a fixed amount: the run
    // ends when both phases have finished.
    let terminal = |s: &mini_swe_mcp::pool::WorkerState| {
        matches!(
            s,
            mini_swe_mcp::pool::WorkerState::Completed { .. }
                | mini_swe_mcp::pool::WorkerState::Failed { .. }
                | mini_swe_mcp::pool::WorkerState::Exhausted { .. }
        )
    };
    let mut finished = false;
    for _ in 0..600 {
        if let Some(s) = pool.get_worker_state(&worker_id).await
            && terminal(&s)
        {
            finished = true;
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    assert!(finished, "worker reaches a terminal state");

    llm.request_bodies()
        .await
        .iter()
        .filter_map(|body| {
            body["messages"]
                .as_array()?
                .first()
                .filter(|m| m["role"] == "system")?
                .get("content")?
                .as_str()
                .map(str::to_string)
        })
        .collect()
}

/// A git repository a worker can be dispatched against, over a uniquely named
/// scratch directory (see [`common::TempDir`]) so two runs can never collide on
/// one predictable `/tmp` path.
struct TestRepo {
    dir: std::path::PathBuf,
    _scratch: common::TempDir,
}

impl TestRepo {
    fn new(tag: &str) -> Self {
        let scratch = common::TempDir::new_in_tmp(tag);
        let dir = scratch.path().to_path_buf();
        common::git(&dir, &["init", "-b", "master"]);
        common::git(&dir, &["config", "user.name", "mini-swe-test"]);
        common::git(&dir, &["config", "user.email", "test@localhost"]);
        std::fs::write(dir.join("README.md"), "# scratch\n").expect("seed file");
        common::git(&dir, &["add", "README.md"]);
        common::git(&dir, &["commit", "-m", "baseline"]);
        Self {
            dir,
            _scratch: scratch,
        }
    }

    fn path(&self) -> &std::path::Path {
        &self.dir
    }
}

impl Drop for TestRepo {
    fn drop(&mut self) {
        mini_swe_mcp::cache::remove_build_dir_leases(&self.dir);
    }
}

/// The review phase is a second conversation with a different model, and it must
/// be corrected by *that* model's block: the implementer's rule never reaches the
/// reviewer, and the reviewer's rule never reached the implementer.
#[tokio::test(flavor = "multi_thread")]
async fn the_review_phase_carries_the_reviewer_model_instructions() {
    let repo = TestRepo::new("reviewer");
    let prompts = system_prompts_of_a_reviewed_run(
        repo.path(),
        r#"
models:
  small:
    id: combo:small
    instructions: |
      IMPLEMENTER-ONLY-RULE: prefer one whole-file read.
  deep:
    id: combo:deep
    instructions: |
      REVIEWER-ONLY-RULE: question the assumption before accepting it.
"#,
    )
    .await;

    let implementer: Vec<&String> = prompts
        .iter()
        .filter(|p| p.contains("IMPLEMENTER-ONLY-RULE"))
        .collect();
    let reviewer: Vec<&String> = prompts
        .iter()
        .filter(|p| p.contains("REVIEWER-ONLY-RULE"))
        .collect();

    assert!(
        !implementer.is_empty(),
        "the implementation prompt must carry its own model's rule: {prompts:?}"
    );
    assert!(
        !reviewer.is_empty(),
        "the review prompt must carry the reviewer's rule: {prompts:?}"
    );
    assert!(
        implementer
            .iter()
            .all(|p| !p.contains("REVIEWER-ONLY-RULE")),
        "the implementer must not be given the reviewer's rules: {implementer:?}"
    );
    assert!(
        reviewer
            .iter()
            .all(|p| !p.contains("IMPLEMENTER-ONLY-RULE")),
        "the reviewer must not inherit the implementer's rules: {reviewer:?}"
    );
}

/// The cap bounds the text that actually reaches the prompt, not just the
/// strings behind it: the `- ` bullet the renderer adds to every entry, the
/// heading and the truncation note are all part of the section.
///
/// Checking `rendered_len()` alone cannot catch a budget overrun, because the
/// framing is what makes the difference: a block of many short entries spends
/// two bytes of bullet per entry on top of the text, so a section that fits the
/// entry strings alone can still overrun the cap the docs promise.
#[test]
fn test_the_emitted_prompt_section_fits_the_documented_cap() {
    // Many short entries: the worst case for per-entry framing overhead.
    let mut yaml = String::from("models:\n  small:\n    id: combo:small\n    instructions:\n");
    for i in 0..400 {
        yaml.push_str(&format!("      - rule {i} with a little padding\n"));
    }
    let raw: ModelManifest = serde_yaml::from_str(&yaml).expect("parses");
    let manifest = raw.normalize();

    let repo = TempDir::new_in_tmp("model-instructions-budget");
    let with = build_system_prompt(&manifest, repo.path(), "small");
    // A model that declares nothing is the same prompt minus this section.
    let without = build_system_prompt(&manifest, repo.path(), "other");
    let section = with.trim_start_matches(&without);

    assert!(
        section.len() <= MAX_MODEL_INSTRUCTIONS_BYTES,
        "the section appended to the system prompt must fit the {MAX_MODEL_INSTRUCTIONS_BYTES}-byte \
         cap, but it is {} bytes: the bullet, heading and note overhead must be \
         charged to the budget",
        section.len()
    );
    assert!(
        section.contains("[truncated"),
        "a block cut to fit is marked as cut: {section}"
    );
}

/// The cap bounds the *emitted* section, not just the bullets behind it: a block
/// whose bullets total just under `MAX_MODEL_INSTRUCTIONS_BYTES` still gains the
/// `\n\n`+header framing when rendered, so it must be cut too, or the prompt
/// overruns the very budget the docs promise. This is the boundary the
/// many-short-entries case above cannot reach (it is always far over budget).
#[test]
fn test_a_block_under_the_bullet_budget_but_over_the_section_budget_is_cut() {
    let header = "Model-specific instructions (declared for this model in models.yaml):";
    // Bullets alone fit within the cap...
    let entry_len = MAX_MODEL_INSTRUCTIONS_BYTES - 2 - header.len() - 1;
    let entry = "a".repeat(entry_len);
    let yaml = format!(
        "models:\n  small:\n    id: combo:small\n    instructions:\n      - {entry}\n      - tail\n"
    );
    let raw: ModelManifest = serde_yaml::from_str(&yaml).expect("parses");
    assert!(
        raw.validate()
            .iter()
            .any(|w| w.contains("instructions") && w.contains("budget")),
        "the block must be warned about exactly when it is cut: {:?}",
        raw.validate()
    );
    let manifest = raw.normalize();

    let repo = TempDir::new_in_tmp("model-instructions-section-budget");
    let with = build_system_prompt(&manifest, repo.path(), "small");
    let without = build_system_prompt(&manifest, repo.path(), "other");
    let section = with.trim_start_matches(&without);

    assert!(
        section.len() <= MAX_MODEL_INSTRUCTIONS_BYTES,
        "the emitted section must fit the {MAX_MODEL_INSTRUCTIONS_BYTES}-byte cap, but is {} bytes",
        section.len()
    );
    assert!(
        section.contains("[truncated"),
        "a block that only fits once the framing is counted must be marked cut: {section}"
    );
}
