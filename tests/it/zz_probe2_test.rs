//! Scratch probe 5: a mode name with a colon kills the run in the phase loop.
use crate::common;
use mini_swe_mcp::manifest::ModelManifest;
use mini_swe_mcp::pool::{WorkerPool, WorkerState};
use mini_swe_mcp::worktree::ScratchRoot;
use std::path::Path;
use std::sync::Arc;

const TEST_OWNER: &str = "test-agent";
const COMPLETION_SENTINEL: &str = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT";

async fn go(base_url: &str, repo: &Path, review_after: &str) -> WorkerState {
    let manifest: ModelManifest = serde_yaml::from_str(
        "default: ninja\nmodels:\n  ninja:\n    id: combo:ninja\nreview_modes:\n  \"a:b\":\n    checklist: Check the diff.\n",
    )
    .expect("catalog");
    let scratch = common::TempDir::new_in_tmp("zz5-pool");
    let pool = WorkerPool::with_scratch(
        1, base_url.to_string(), "test-key".to_string(), ScratchRoot::new(scratch.path()),
    )
    .with_manifest(Arc::new(manifest));
    let id = pool.dispatch(
        TEST_OWNER.to_string(), "probe".to_string(), "combo:ninja".to_string(),
        None, repo.to_path_buf(), 5, Some("zz5".to_string()),
        Some(review_after.to_string()), false, None, Vec::new(),
    ).await.expect("dispatch");
    let state = common::wait_for_terminal(&pool, &id).await;
    let _ = pool.kill(&id).await;
    state
}

#[tokio::test]
async fn probe_colon_mode_name_kills_the_run() {
    let repo = common::TestRepo::new("zz5");
    let llm = common::fake_llm::FakeLlm::spawn_scripted(&[
        "echo changed > src/ordinary.rs",
        &format!("echo {COMPLETION_SENTINEL}"),
        &format!("echo {COMPLETION_SENTINEL}"),
    ]).await;
    // what dispatch.rs would store for "a:b"
    let state = go(llm.base_url(), repo.path(), ":a:b").await;
    println!("state = {state:?}");
    assert!(matches!(state, WorkerState::Completed { .. }), "the run must complete, got {state:?}");
}
