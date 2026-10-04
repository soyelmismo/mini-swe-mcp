//! Scratch probe (deleted before completion).
use crate::common;
use serde_json::{Value, json};
use std::path::Path;
use std::sync::Arc;

use mini_swe_mcp::manifest::ModelManifest;
use mini_swe_mcp::pool::{WorkerPool, WorkerState};
use mini_swe_mcp::worktree::ScratchRoot;

const TEST_OWNER: &str = "test-agent";
const COMPLETION_SENTINEL: &str = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT";

fn catalog() -> ModelManifest {
    serde_yaml::from_str(
        "default: ninja\nmodels:\n  ninja:\n    id: combo:ninja\n  nerd:\n    id: combo:nerd\nreview_modes:\n  security:\n    default_model: nerd\n",
    )
    .expect("catalog")
}

async fn dispatch_and_wait(
    base_url: &str,
    repo: &Path,
    review_after: Option<&str>,
) -> (WorkerPool, String, WorkerState, common::TempDir) {
    let scratch = common::TempDir::new_in_tmp("zz-probe-pool");
    let pool = WorkerPool::with_scratch(
        1,
        base_url.to_string(),
        "test-key".to_string(),
        ScratchRoot::new(scratch.path()),
    )
    .with_manifest(Arc::new(catalog()));
    let worker_id = pool
        .dispatch(
            TEST_OWNER.to_string(),
            "probe".to_string(),
            "combo:ninja".to_string(),
            None,
            repo.to_path_buf(),
            5,
            Some("zz-probe".to_string()),
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

#[tokio::test]
async fn probe_bare_security_mode_runs_on_its_default_model() {
    let repo = common::TestRepo::new("zz-bare");
    let llm = common::fake_llm::FakeLlm::spawn_scripted(&[
        "echo changed > src/ordinary.rs",
        &format!("echo {COMPLETION_SENTINEL}"),
        &format!("echo {COMPLETION_SENTINEL}"),
    ])
    .await;
    let (_pool, _id, state, _s) =
        dispatch_and_wait(llm.base_url(), repo.path(), Some("security")).await;
    let bodies = llm.request_bodies().await;
    println!("BARE security -> models={:?} state={state:?}", models_asked_for(&bodies));
}

#[tokio::test]
async fn probe_model_colon_mode_overrides_the_model() {
    let repo = common::TestRepo::new("zz-colon");
    let llm = common::fake_llm::FakeLlm::spawn_scripted(&[
        "echo changed > src/ordinary.rs",
        &format!("echo {COMPLETION_SENTINEL}"),
        &format!("echo {COMPLETION_SENTINEL}"),
    ])
    .await;
    let (_pool, _id, state, _s) =
        dispatch_and_wait(llm.base_url(), repo.path(), Some("combo:ninja:security")).await;
    let bodies = llm.request_bodies().await;
    println!("MODEL:MODE -> models={:?} state={state:?}", models_asked_for(&bodies));
    println!("prompt has hostile: {:?}", llm.request_bodies().await.iter().any(|b| {
        b["messages"].as_array().map(|m| m.iter().any(|x| x["content"].as_str().unwrap_or_default().contains("Assume the diff is hostile"))).unwrap_or(false)
    }));
    let _ = json!(1);
}
