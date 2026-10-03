//! User-declared review modes (`review_modes:` in `models.yaml`).
//!
//! The manifest may declare its own auditors the same way it declares models:
//! each entry carries a `checklist` (focus instructions appended to the common
//! review frame) and an optional default `model`. Built-in modes `quality`
//! and `security` keep their prompts and can be overridden by declaring the
//! same name. `--review-after <model>:<mode>` accepts any declared mode; an
//! unknown mode is a dispatch error listing the available ones.

use crate::common;
use crate::common::fake_llm::FakeLlm;

use mini_swe_mcp::manifest::ModelManifest;
use mini_swe_mcp::pool::{ReviewMode, review_prompt};

// ----------
// Parsing
// ----------

fn modes_manifest(yaml: &str) -> ModelManifest {
    serde_yaml::from_str(yaml).expect("review-modes manifest must parse")
}

#[test]
fn a_manifest_without_review_modes_parses_to_empty() {
    let manifest = modes_manifest("models:\n  solo:\n    id: combo:solo\n");
    assert!(manifest.review_modes.is_empty());
    assert!(manifest.validate().is_empty());
}

#[test]
fn a_declared_mode_parses_its_checklist_and_model() {
    let manifest = modes_manifest(
        "models:\n  nerd:\n    id: combo:nerd\nreview_modes:\n  perf:\n    checklist: Check for N+1 queries.\n    model: nerd\n",
    );
    let def = manifest.review_mode("perf").expect("perf mode");
    assert_eq!(def.checklist, "Check for N+1 queries.");
    assert_eq!(def.model.as_deref(), Some("nerd"));
    assert!(manifest.validate().is_empty());
}

#[test]
fn a_declared_mode_without_a_model_has_none() {
    let manifest = modes_manifest(
        "models:\n  nerd:\n    id: combo:nerd\nreview_modes:\n  style:\n    checklist: Check naming.\n",
    );
    let def = manifest.review_mode("style").expect("style mode");
    assert_eq!(def.model, None);
    assert!(manifest.validate().is_empty());
}

// ----------
// Validation
// ----------

#[test]
fn an_empty_checklist_warns_and_is_dropped_by_normalize() {
    let manifest = modes_manifest(
        "models:\n  nerd:\n    id: combo:nerd\nreview_modes:\n  empty:\n    checklist: \"   \"\n",
    );
    let warnings = manifest.validate();
    assert!(
        warnings
            .iter()
            .any(|w| w.contains("empty") && w.contains("checklist")),
        "an empty checklist must warn: {warnings:?}"
    );
    let normalized = manifest.normalize();
    assert!(
        normalized.review_mode("empty").is_none(),
        "an empty checklist must be dropped by normalize"
    );
}

// ----------
// Mode resolution
// ----------

#[test]
fn parse_with_manifest_resolves_a_custom_mode() {
    let manifest = modes_manifest(
        "models:\n  nerd:\n    id: combo:nerd\nreview_modes:\n  perf:\n    checklist: Check for N+1 queries.\n    model: nerd\n",
    );
    let (model, mode) =
        ReviewMode::parse_with_manifest("some-reviewer:perf", &manifest).expect("perf mode");
    assert_eq!(model, "some-reviewer");
    assert_eq!(mode.name, "perf");
    assert_eq!(mode.checklist.as_deref(), Some("Check for N+1 queries."));
}

#[test]
fn parse_with_manifest_uses_the_mode_default_for_an_empty_model() {
    let manifest = modes_manifest(
        "models:\n  nerd:\n    id: combo:nerd\nreview_modes:\n  perf:\n    checklist: Check for N+1 queries.\n    model: nerd\n",
    );
    let (model, mode) = ReviewMode::parse_with_manifest(":perf", &manifest).expect("perf mode");
    assert_eq!(model, "nerd");
    assert_eq!(mode.name, "perf");
}

#[test]
fn parse_with_manifest_keeps_colon_model_ids_as_quality() {
    let manifest = ModelManifest::default();
    // `combo:nerd` is a model id, not a mode: the suffix names no mode and
    // the whole string is a known model.
    let (model, mode) = ReviewMode::parse_with_manifest("combo:nerd", &manifest).expect("model id");
    assert_eq!(model, "combo:nerd");
    assert_eq!(mode.name, "quality");
    assert_eq!(mode.checklist, None);
}

#[test]
fn an_unknown_mode_is_refused_with_the_available_ones() {
    let manifest = modes_manifest(
        "models:\n  nerd:\n    id: combo:nerd\nreview_modes:\n  perf:\n    checklist: Check.\n",
    );
    let err = ReviewMode::parse_with_manifest("reviewer:nope", &manifest)
        .expect_err("an unknown mode must be refused");
    let message = err.to_string();
    assert!(message.contains("unknown review mode"), "{message}");
    assert!(message.contains("nope"), "{message}");
    for available in ["quality", "security", "perf"] {
        assert!(
            message.contains(available),
            "{message} must list {available}"
        );
    }
}

// ----------
// The prompt
// ----------

#[test]
fn a_custom_mode_appends_its_checklist_to_the_review_frame() {
    let mode = ReviewMode {
        name: "perf".to_string(),
        checklist: Some("Check for N+1 queries.".to_string()),
    };
    let prompt = review_prompt(&mode, "speed up", Some("make check"), &[]);
    assert!(
        prompt.contains("REVIEW PHASE (perf)"),
        "a custom mode names itself: {prompt}"
    );
    assert!(
        prompt.contains("Check for N+1 queries."),
        "the checklist is appended: {prompt}"
    );
    assert!(
        prompt.contains("make check"),
        "the dispatch gate is still run: {prompt}"
    );
    assert!(
        prompt.contains("COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"),
        "the completion contract is kept: {prompt}"
    );
}

#[test]
fn overriding_security_replaces_its_checklist() {
    let manifest = modes_manifest(
        "models:\n  nerd:\n    id: combo:nerd\nreview_modes:\n  security:\n    checklist: Custom hostile review.\n",
    );
    let mode = ReviewMode::resolve_declared("security", &manifest);
    assert_eq!(mode.name, "security");
    assert_eq!(mode.checklist.as_deref(), Some("Custom hostile review."));
    assert!(
        mode.is_security(),
        "an overridden security is still the security review"
    );
    let prompt = review_prompt(&mode, "task", Some("make check"), &[]);
    assert!(
        prompt.contains("Custom hostile review."),
        "the override replaces the prompt: {prompt}"
    );
    assert!(
        !prompt.contains("ADVERSARIAL SECURITY REVIEW PHASE"),
        "the built-in prompt is replaced: {prompt}"
    );
}

// ----------
// End-to-end: the phase loop uses the mode's checklist and reviewer
// ----------

use std::sync::Arc;

use serde_json::json;

use mini_swe_mcp::pool::{WorkerPool, WorkerState};
use mini_swe_mcp::worktree::ScratchRoot;

const TEST_OWNER: &str = "test-agent";
const COMPLETION_SENTINEL: &str = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT";

/// A worker turn that runs `command` with `content` as its prose.
fn turn(call_id: &str, content: &str, command: &str) -> Vec<String> {
    common::tool_turn(call_id, content, command)
}

/// The turn a model issues when it is done.
fn completion_turn(call_id: &str, content: &str) -> Vec<String> {
    turn(call_id, content, &format!("echo {COMPLETION_SENTINEL}"))
}

fn modes_pool(base_url: &str, scratch: &common::TempDir) -> WorkerPool {
    let manifest: ModelManifest = serde_yaml::from_str(
        "models:\n  test-model:\n    id: test-model\nreview_modes:\n  perf:\n    checklist: Check for N+1 queries and missing indexes.\n    model: test-model\n",
    )
    .expect("modes manifest must parse");
    WorkerPool::with_scratch(
        1,
        base_url.to_string(),
        "test-key".to_string(),
        ScratchRoot::new(scratch.path()),
    )
    .with_manifest(Arc::new(manifest))
}

/// A custom mode selected by suffix runs its checklist on the requested model.
#[tokio::test]
async fn a_custom_mode_uses_its_checklist_and_reviewer() {
    let repo = common::TestRepo::new("custom");
    let server = FakeLlm::spawn_sse(vec![
        turn("call_write", "", "echo changed > src/ordinary.rs"),
        turn(
            "call_impl",
            "REPORT\ndone: impl\nrisks: none",
            &format!("echo {COMPLETION_SENTINEL}"),
        ),
        completion_turn("call_review", "REPORT\ndone: reviewed\nrisks: none"),
    ])
    .await;

    let scratch = common::TempDir::new_in_tmp("review-modes-custom");
    let pool = modes_pool(server.base_url(), &scratch);
    let worker_id = pool
        .dispatch(
            TEST_OWNER.to_string(),
            "exercise the custom mode".to_string(),
            "test-model".to_string(),
            None,
            repo.path().to_path_buf(),
            5,
            Some("review-modes".to_string()),
            Some("test-reviewer:perf".to_string()),
            false,
            None,
            Vec::new(),
        )
        .await
        .expect("dispatch the worker");
    let state = common::wait_for_terminal(&pool, &worker_id).await;

    let requests = server.request_bodies().await;
    assert_eq!(
        requests.len(),
        3,
        "write, implementer, and the custom review"
    );
    assert_eq!(
        requests[2]["model"],
        json!("test-reviewer"),
        "the suffix selects the mode, not the model"
    );
    let prompt = common::review_prompt_of(&server)
        .await
        .expect("a review prompt");
    assert!(
        prompt.contains("REVIEW PHASE (perf)"),
        "the custom mode names itself: {prompt}"
    );
    assert!(
        prompt.contains("Check for N+1 queries and missing indexes."),
        "the review runs the mode's checklist: {prompt}"
    );

    let WorkerState::Completed { .. } = &state else {
        panic!("worker must complete, got {state:?}")
    };
    let _ = pool.kill(&worker_id).await;
}

/// An unknown mode is a dispatch error, not a worker failure.
#[tokio::test]
async fn an_unknown_mode_is_a_dispatch_error() {
    let repo = common::TestRepo::new("unknown");
    let server = FakeLlm::spawn_sse(vec![]).await;

    let scratch = common::TempDir::new_in_tmp("review-modes-unknown");
    let pool = modes_pool(server.base_url(), &scratch);
    let err = pool
        .dispatch(
            TEST_OWNER.to_string(),
            "exercise the unknown mode".to_string(),
            "test-model".to_string(),
            None,
            repo.path().to_path_buf(),
            5,
            Some("review-modes".to_string()),
            Some("test-reviewer:nope".to_string()),
            false,
            None,
            Vec::new(),
        )
        .await
        .expect_err("an unknown mode must be a dispatch error");
    let message = err.to_string();
    assert!(message.contains("unknown review mode"), "{message}");
    assert!(message.contains("nope"), "{message}");
    assert!(
        message.contains("perf"),
        "{message} must list the declared mode"
    );
}
