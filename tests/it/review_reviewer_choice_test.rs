//! The model an *automatic* security review runs on.
//!
//! A diff that touches a declared sensitive path triggers an adversarial pass
//! nobody asked for by name, so the harness has to pick the reviewer. The
//! defect this file pins down is that pickup picking the wrong one: the fast
//! executor that implemented the change (the `combo:ninja` worker in the report)
//! audited itself, even though the operator's catalog declares a deeper tier.
//!
//! The order under test, for a sensitive diff with no `review_after`:
//!
//! 1. the security mode's `default_model`, resolved to its id;
//! 2. otherwise the dispatch's default -- the manifest `default:`, resolved,
//!    which is never the implementer's own per-dispatch `--model`;
//! 3. an explicit `--review-after <model>[:security]` always wins over both,
//!    because it is the orchestrator's own instruction.
//!
//! A quality review is not routed through the rule and keeps running on the
//! requested model.

use crate::common;
use serde_json::{Value, json};
use std::path::Path;
use std::sync::Arc;

use mini_swe_mcp::manifest::ModelManifest;
use mini_swe_mcp::pool::{WorkerPool, WorkerState};
use mini_swe_mcp::worktree::ScratchRoot;

const TEST_OWNER: &str = "test-agent";
const COMPLETION_SENTINEL: &str = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT";

/// The catalog an operator writes: a fast default and a deeper tier, named by
/// the security mode's `default_model`. `declare_default` toggles that key.
fn catalog(declare_default: bool) -> ModelManifest {
    let modes_line = if declare_default {
        "review_modes:\n  security:\n    default_model: nerd\n"
    } else {
        ""
    };
    let yaml = format!(
        "default: ninja\nmodels:\n  ninja:\n    id: combo:ninja\n  nerd:\n    id: combo:nerd\n{modes_line}"
    );
    serde_yaml::from_str(&yaml).unwrap_or_else(|e| panic!("catalog YAML must parse: {e}\n{yaml}"))
}

/// Dispatch one worker on `model` with `review_after` and wait for it to stop.
async fn dispatch_and_wait(
    base_url: &str,
    repo: &Path,
    manifest: ModelManifest,
    model: &str,
    review_after: Option<&str>,
) -> (WorkerPool, String, WorkerState, common::TempDir) {
    let scratch = common::TempDir::new_in_tmp("review-reviewer-pool");
    let pool = WorkerPool::with_scratch(
        1,
        base_url.to_string(),
        "test-key".to_string(),
        ScratchRoot::new(scratch.path()),
    )
    .with_manifest(Arc::new(manifest));
    let worker_id = pool
        .dispatch(
            TEST_OWNER.to_string(),
            "exercise the reviewer choice".to_string(),
            model.to_string(),
            None,
            repo.to_path_buf(),
            5,
            Some("review-reviewer".to_string()),
            review_after.map(str::to_string),
            false,
            None,
            Vec::new(),
        )
        .await
        .expect("dispatch the worker");
    let state = common::wait_for_terminal(&pool, &worker_id).await;
    (pool, worker_id, state, scratch)
}

/// The model each request in the run asked for, in order.
fn models_of(bodies: &[Value]) -> Vec<String> {
    bodies
        .iter()
        .map(|body| body["model"].as_str().unwrap_or_default().to_string())
        .collect()
}

/// The model the review phase asked for.
///
/// One request per turn at most, with a retry allowed for a transport failure,
/// so the *distinct* models a run asked for name the phases unambiguously:
/// the implementer's model first, then the reviewer's. A report-only retry
/// repeats the implementer's request, which changes nothing here.
fn models_asked_for(bodies: &[Value]) -> Vec<String> {
    let mut seen = Vec::new();
    for body in bodies {
        let model = body["model"].as_str().unwrap_or_default().to_string();
        if seen.last() != Some(&model) {
            seen.push(model);
        }
    }
    seen
}

/// The user message of the request that opened the review phase, if any.
fn review_prompt_of(bodies: &[Value]) -> Option<String> {
    bodies.iter().find_map(|body| {
        body["messages"]
            .as_array()?
            .iter()
            .filter(|message| message["role"] == json!("user"))
            .map(|message| message["content"].as_str().unwrap_or_default().to_string())
            .find(|content| content.contains("REVIEW PHASE"))
    })
}

/// A sensitive diff on a catalog with a security `default_model: nerd`: the
/// review runs on `combo:nerd`, not on the `combo:ninja` worker that wrote
/// the diff.
#[tokio::test]
async fn the_security_default_model_reviews_the_sensitive_diff() {
    let repo = common::TestRepo::new("secdefault");
    repo.declare_sensitive(&["src/hub/**"]);
    let llm = common::fake_llm::FakeLlm::spawn_scripted(&[
        "mkdir -p src/hub && echo changed > src/hub/mod.rs",
        &format!("echo {COMPLETION_SENTINEL}"),
        &format!("echo {COMPLETION_SENTINEL}"),
    ])
    .await;

    let (pool, worker_id, state, _scratch) = dispatch_and_wait(
        llm.base_url(),
        repo.path(),
        catalog(true),
        "combo:ninja",
        None,
    )
    .await;

    let bodies = llm.request_bodies().await;
    assert_eq!(
        models_asked_for(&bodies),
        vec!["combo:ninja".to_string(), "combo:nerd".to_string()],
        "the automatic security review must run on the security mode's default model, \
         not on the implementer's model"
    );
    let prompt = review_prompt_of(&bodies).expect("a review prompt");
    assert!(
        prompt.contains("Assume the diff is hostile"),
        "the sensitive diff must still trigger the adversarial prompt"
    );

    assert!(
        matches!(state, WorkerState::Completed { .. }),
        "worker must complete, got {state:?}"
    );
    let _ = pool.kill(&worker_id).await;
}

/// Without a security `default_model` the automatic review falls back to the
/// catalog's `default:` -- and still not to the worker's own `--model`.
#[tokio::test]
async fn without_a_security_default_model_the_default_reviews() {
    let repo = common::TestRepo::new("default");
    repo.declare_sensitive(&["src/hub/**"]);
    let llm = common::fake_llm::FakeLlm::spawn_scripted(&[
        "mkdir -p src/hub && echo changed > src/hub/mod.rs",
        &format!("echo {COMPLETION_SENTINEL}"),
        &format!("echo {COMPLETION_SENTINEL}"),
    ])
    .await;

    let (pool, worker_id, state, _scratch) = dispatch_and_wait(
        llm.base_url(),
        repo.path(),
        catalog(false),
        "combo:nerd",
        None,
    )
    .await;

    let bodies = llm.request_bodies().await;
    assert_eq!(
        models_asked_for(&bodies),
        vec!["combo:nerd".to_string(), "combo:ninja".to_string()],
        "the fallback is the catalog's `default:`, not the dispatched model"
    );
    let _ = review_prompt_of(&bodies).expect("a review prompt");

    assert!(
        matches!(state, WorkerState::Completed { .. }),
        "worker must complete, got {state:?}"
    );
    let _ = pool.kill(&worker_id).await;
}

/// An explicit `--review-after` beats the security mode's default: the
/// orchestrator's instruction is the reviewer's choice, not the harness's.
#[tokio::test]
async fn an_explicit_review_after_wins() {
    let repo = common::TestRepo::new("explicit");
    repo.declare_sensitive(&["src/hub/**"]);
    let llm = common::fake_llm::FakeLlm::spawn_scripted(&[
        "mkdir -p src/hub && echo changed > src/hub/mod.rs",
        &format!("echo {COMPLETION_SENTINEL}"),
        &format!("echo {COMPLETION_SENTINEL}"),
    ])
    .await;

    let (pool, worker_id, state, _scratch) = dispatch_and_wait(
        llm.base_url(),
        repo.path(),
        catalog(true),
        "combo:ninja",
        Some("combo:nerd"),
    )
    .await;

    let bodies = llm.request_bodies().await;
    assert_eq!(
        models_of(&bodies).last().map(String::as_str),
        Some("combo:nerd"),
        "an explicit --review-after must win over the security default: {:?}",
        models_of(&bodies)
    );
    assert!(
        matches!(state, WorkerState::Completed { .. }),
        "worker must complete, got {state:?}"
    );
    let _ = pool.kill(&worker_id).await;
}

/// A quality review keeps its own rule: with no `review_after` there is none,
/// and a requested one is not rerouted through the security default.
#[tokio::test]
async fn a_requested_quality_review_keeps_its_model() {
    let repo = common::TestRepo::new("quality");
    let llm = common::fake_llm::FakeLlm::spawn_scripted(&[
        "echo changed > src/ordinary.rs",
        &format!("echo {COMPLETION_SENTINEL}"),
        &format!("echo {COMPLETION_SENTINEL}"),
    ])
    .await;

    let (pool, worker_id, state, _scratch) = dispatch_and_wait(
        llm.base_url(),
        repo.path(),
        catalog(true),
        "combo:ninja",
        Some("combo:nerd"),
    )
    .await;

    let bodies = llm.request_bodies().await;
    assert_eq!(
        models_of(&bodies).last().map(String::as_str),
        Some("combo:nerd"),
        "a quality review stays on the requested model: {:?}",
        models_of(&bodies)
    );
    let prompt = review_prompt_of(&bodies).expect("a review prompt");
    assert!(
        prompt.contains("REVIEW PHASE (quality)") && !prompt.contains("Assume the diff is hostile"),
        "the requested quality mode must not be upgraded on an insensitive diff"
    );

    assert!(
        matches!(state, WorkerState::Completed { .. }),
        "worker must complete, got {state:?}"
    );
    let _ = pool.kill(&worker_id).await;
}
