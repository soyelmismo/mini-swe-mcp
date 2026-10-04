//! Scratch probe 2: an unknown `default_model` on the security mode.
use crate::common;
use mini_swe_mcp::manifest::ModelManifest;
use mini_swe_mcp::pool::{WorkerPool, WorkerState};
use mini_swe_mcp::worktree::ScratchRoot;
use std::path::Path;
use std::sync::Arc;

const TEST_OWNER: &str = "test-agent";
const COMPLETION_SENTINEL: &str = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT";

async fn dispatch_and_wait(
    base_url: &str,
    repo: &Path,
    manifest: ModelManifest,
    review_after: Option<&str>,
) -> (WorkerPool, String, WorkerState, common::TempDir) {
    let scratch = common::TempDir::new_in_tmp("zz-probe2-pool");
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
            "probe".to_string(),
            "combo:ninja".to_string(),
            None,
            repo.to_path_buf(),
            5,
            Some("zz-probe2".to_string()),
            review_after.map(str::to_string),
            false,
            None,
            Vec::new(),
        )
        .await
        .expect("dispatch");
    let state = common::wait_for_terminal(&pool, &worker_id).await;
    (pool, worker_id, state, scratch)
}

fn models_asked_for(bodies: &[serde_json::Value]) -> Vec<String> {
    let mut seen = Vec::new();
    for body in bodies {
        let model = body["model"].as_str().unwrap_or_default().to_string();
        if seen.last() != Some(&model) {
            seen.push(model);
        }
    }
    seen
}

fn hostile(bodies: &[serde_json::Value]) -> bool {
    bodies.iter().any(|b| {
        b["messages"].as_array().is_some_and(|m| {
            m.iter().any(|x| {
                x["content"]
                    .as_str()
                    .unwrap_or_default()
                    .contains("Assume the diff is hostile")
            })
        })
    })
}

/// Automatic sensitive-path review with a typo'd security `default_model`.
#[tokio::test]
async fn probe_typo_default_model_drops_the_automatic_security_review() {
    let repo = common::TestRepo::new("zz-typo");
    repo.declare_sensitive(&["src/hub/**"]);
    let man: ModelManifest = serde_yaml::from_str(
        "default: ninja\nmodels:\n  ninja:\n    id: combo:ninja\n  nerd:\n    id: combo:nerd\nreview_modes:\n  security:\n    default_model: neerd\n",
    )
    .expect("catalog");
    println!("warnings: {:?}", man.validate());
    let llm = common::fake_llm::FakeLlm::spawn_scripted(&[
        "mkdir -p src/hub && echo changed > src/hub/mod.rs",
        &format!("echo {COMPLETION_SENTINEL}"),
        &format!("echo {COMPLETION_SENTINEL}"),
    ])
    .await;
    let (pool, id, state, _s) = dispatch_and_wait(llm.base_url(), repo.path(), man, None).await;
    let bodies = llm.request_bodies().await;
    println!("models asked = {:?}", models_asked_for(&bodies));
    println!("hostile prompt present = {}", hostile(&bodies));
    println!("state = {state:?}");
    let _ = pool.kill(&id).await;
}

/// Requested `--review-after security` with the same typo.
#[tokio::test]
async fn probe_typo_default_model_drops_a_requested_security_review() {
    let repo = common::TestRepo::new("zz-typo2");
    let man: ModelManifest = serde_yaml::from_str(
        "default: ninja\nmodels:\n  ninja:\n    id: combo:ninja\n  nerd:\n    id: combo:nerd\nreview_modes:\n  security:\n    default_model: neerd\n",
    )
    .expect("catalog");
    let llm = common::fake_llm::FakeLlm::spawn_scripted(&[
        "echo changed > src/ordinary.rs",
        &format!("echo {COMPLETION_SENTINEL}"),
        &format!("echo {COMPLETION_SENTINEL}"),
    ])
    .await;
    let (pool, id, state, _s) =
        dispatch_and_wait(llm.base_url(), repo.path(), man, Some("security")).await;
    let bodies = llm.request_bodies().await;
    println!("models asked = {:?}", models_asked_for(&bodies));
    println!("hostile prompt present = {}", hostile(&bodies));
    println!("state = {state:?}");
    let _ = pool.kill(&id).await;
}
