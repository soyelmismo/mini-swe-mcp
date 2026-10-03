//! A security approval is proof, not a side effect of a phase having run.
//!
//! Four properties the round's integration left unguarded:
//!
//! 1. A security review that never emitted the completion sentinel (an LLM
//!    error or budget exhaustion) must not leave an approved commit behind,
//!    because recording it would let the next revision skip the review of
//!    code nobody audited.
//! 2. An empty reviewer (`--review-after :<mode>` with no mode default) must
//!    run on the implementer's model, not send an empty model string to the
//!    provider and quietly end inconclusive.
//! 3. A requested manifest-declared mode whose phase the sensitive-path
//!    upgrade displaced must still run as its own successive phase, not be
//!    silently replaced.
//! 4. A registry-sourced approved commit that is not a plain git object id
//!    (one an attacker could plant as `--output=<path>`) must be rejected
//!    rather than spliced verbatim into a git revision argument.

mod common;

use std::path::Path;
use std::sync::Arc;

use common::fake_llm::FakeLlm;
use serde_json::json;

use mini_swe_mcp::manifest::ModelManifest;
use mini_swe_mcp::pool::{WorkerPool, WorkerRole, WorkerState, scope_for};
use mini_swe_mcp::worktree::ScratchRoot;

const TEST_OWNER: &str = "test-agent";

/// A pool with the default manifest, one slot, and the given API base.
fn default_pool(base_url: &str, scratch: &common::TempDir) -> WorkerPool {
    WorkerPool::with_scratch(
        1,
        base_url.to_string(),
        "test-key".to_string(),
        ScratchRoot::new(scratch.path()),
    )
}

/// Dispatch one worker and wait for its terminal state.
async fn dispatch_and_wait(
    pool: &WorkerPool,
    repo: &Path,
    model: &str,
    review_after: Option<String>,
) -> (String, WorkerState) {
    let worker_id = pool
        .dispatch(
            TEST_OWNER.to_string(),
            "exercise the approval proof".to_string(),
            model.to_string(),
            None,
            repo.to_path_buf(),
            5,
            Some("approval-proof".to_string()),
            review_after,
            false,
            None,
            Vec::new(),
        )
        .await
        .expect("dispatch the worker");
    let state = common::wait_for_terminal(pool, &worker_id).await;
    (worker_id, state)
}

// ================================================================
// 1. Inconclusive review → no approved commit
// ================================================================

/// A security review that never emitted the completion sentinel approved
/// nothing, so it must not leave an approval commit on the registry row.
#[tokio::test]
async fn an_inconclusive_security_review_records_no_approval() {
    let repo = common::TestRepo::new("approval-inconclusive");
    repo.declare_sensitive(&["src/hub/**"]);
    let llm = FakeLlm::spawn_sse(vec![
        common::write_turn("call_write", "src/hub/mod.rs"),
        common::completion_turn("call_impl", "REPORT\ndone: impl\nrisks: none"),
        // The reviewer keeps working and never finishes: the script runs out
        // after this turn, so the next request is left unanswered and the
        // phase ends inconclusive rather than approved.
        common::tool_turn("call_review", "still auditing", "true"),
    ])
    .await;

    let scratch = common::TempDir::new_in_tmp("review-approval-inconclusive");
    let pool = default_pool(&llm.base_url(), &scratch);
    let (worker_id, _state) = dispatch_and_wait(&pool, repo.path(), "test-model", None).await;

    let entry =
        mini_swe_mcp::pool::load_registry_entry_in(pool.scratch_root(), &worker_id)
            .expect("the worker's registry row");
    assert!(
        entry.security_review.is_some(),
        "the security review ran (it touched a sensitive path)"
    );
    assert_eq!(
        entry.security_approved_commit, None,
        "an inconclusive review must not record an approval"
    );
    let _ = pool.kill(&worker_id).await;
}

/// A completing security review records the approved commit.
#[tokio::test]
async fn a_completed_security_review_records_the_approved_commit() {
    let repo = common::TestRepo::new("approval-completed");
    repo.declare_sensitive(&["src/hub/**"]);
    let llm = FakeLlm::spawn_sse(vec![
        common::write_turn("call_write", "src/hub/mod.rs"),
        common::completion_turn("call_impl", "REPORT\ndone: impl\nrisks: none"),
        common::security_completion_turn("call_review", 0),
    ])
    .await;

    let scratch = common::TempDir::new_in_tmp("review-approval-completed");
    let pool = default_pool(&llm.base_url(), &scratch);
    let (worker_id, _state) = dispatch_and_wait(&pool, repo.path(), "test-model", None).await;

    let entry =
        mini_swe_mcp::pool::load_registry_entry_in(pool.scratch_root(), &worker_id)
            .expect("the worker's registry row");
    let approved = entry
        .security_approved_commit
        .expect("a completing review records the approval");
    assert_eq!(
        approved.len(),
        40,
        "the approval is a full hex sha: {approved}"
    );
    assert!(
        approved.bytes().all(|b| b.is_ascii_hexdigit()),
        "the approval is hex: {approved}"
    );
    let _ = pool.kill(&worker_id).await;
}

// ================================================================
// 2. Empty reviewer → implementer's model
// ================================================================

/// `--review-after :<mode>` against a mode that declares no default model
/// must run on the implementer's model, not send an empty model string.
#[tokio::test]
async fn an_empty_reviewer_runs_on_the_implementers_model() {
    let repo = common::TestRepo::new("approval-empty-reviewer");
    let llm = FakeLlm::spawn_sse(vec![
        common::tool_turn("call_write", "", "echo changed > src/ordinary.rs"),
        common::completion_turn("call_impl", "REPORT\ndone: impl\nrisks: none"),
        common::security_completion_turn("call_review", 0),
    ])
    .await;

    let scratch = common::TempDir::new_in_tmp("review-approval-empty-reviewer");
    let pool = default_pool(&llm.base_url(), &scratch);
    let (worker_id, _state) = dispatch_and_wait(
        &pool,
        repo.path(),
        "test-model",
        Some(":security".to_string()),
    )
    .await;

    let bodies = llm.request_bodies().await;
    assert_eq!(
        bodies.len(),
        3,
        "write, implementer completion, and the security review"
    );
    assert_eq!(
        bodies[2]["model"],
        json!("test-model"),
        "the empty reviewer falls back to the implementer's model"
    );
    let _ = pool.kill(&worker_id).await;
}

// ================================================================
// 3. Displaced requested mode still runs
// ================================================================

/// A requested manifest-declared mode whose phase the sensitive-path upgrade
/// displaced must still run as its own successive phase.
#[tokio::test]
async fn a_requested_mode_survives_the_sensitive_upgrade() {
    let repo = common::TestRepo::new("approval-displaced");
    repo.declare_sensitive(&["src/hub/**"]);
    let manifest: ModelManifest = serde_yaml::from_str(
        "models:\n  test-model:\n    id: test-model\nreview_modes:\n  perf:\n    checklist: Check for N+1 queries.\n    model: test-model\n",
    )
    .expect("manifest with a perf mode");
    let llm = FakeLlm::spawn_sse(vec![
        common::write_turn("call_write", "src/hub/mod.rs"),
        common::completion_turn("call_impl", "REPORT\ndone: impl\nrisks: none"),
        common::security_completion_turn("call_security", 0),
        common::completion_turn("call_perf", "REPORT\ndone: perf\nrisks: none"),
    ])
    .await;

    let scratch = common::TempDir::new_in_tmp("review-approval-displaced");
    let pool = default_pool(&llm.base_url(), &scratch)
        .with_manifest(Arc::new(manifest));
    let (worker_id, _state) = dispatch_and_wait(
        &pool,
        repo.path(),
        "test-model",
        Some("test-model:perf".to_string()),
    )
    .await;

    let bodies = llm.request_bodies().await;
    assert_eq!(
        bodies.len(),
        4,
        "write, implementer, security upgrade, and the displaced perf mode"
    );
    let prompts: Vec<String> = bodies
        .iter()
        .filter_map(|req| {
            req["messages"]
                .as_array()
                .unwrap_or(&Vec::new())
                .iter()
                .filter_map(|m| {
                    if m["role"] == "user" {
                        m["content"].as_str().map(str::to_string)
                    } else {
                        None
                    }
                })
                .find(|c| c.contains("REVIEW PHASE"))
        })
        .collect();
    assert!(
        prompts.iter().any(|p| p.contains("ADVERSARIAL SECURITY REVIEW PHASE")),
        "the sensitive diff triggers the security upgrade: {prompts:?}"
    );
    assert!(
        prompts.iter().any(|p| p.contains("REVIEW PHASE (perf)")),
        "the requested perf mode still runs: {prompts:?}"
    );
    let _ = pool.kill(&worker_id).await;
}

// ================================================================
// 4. Planted approval is not a git object id
// ================================================================

/// A registry value that is not a plain git object id must be rejected by
/// `scope_for` so it is never spliced verbatim into a git revision argument.
#[tokio::test]
async fn a_planted_approval_is_not_an_object_id() {
    let dir = common::TempDir::new_in_tmp("approval-planted");
    let repo = dir.path();
    common::git(repo, &["init", "-q", "-b", "master", "."]);
    common::git(repo, &["config", "user.email", "planted@test"]);
    common::git(repo, &["config", "user.name", "Planted Test"]);
    common::git(repo, &["add", "-A"]);
    common::git(repo, &["commit", "-q", "-m", "base"]);
    common::git(repo, &["checkout", "-q", "-b", "worker-w1"]);
    std::fs::write(repo.join("work.rs"), "// work\n").unwrap();
    common::git(repo, &["add", "-A"]);
    common::git(repo, &["commit", "-q", "-m", "work"]);
    let base = common::git(repo, &["rev-parse", "master"]).trim().to_string();

    // A value that looks like a git option, not a revision.
    let planted = "--output=pwned".to_string();
    let scope = scope_for(
        repo,
        "worker-w1",
        WorkerRole::Worker,
        &base,
        Some(planted),
        &[],
    )
    .await;
    assert_eq!(
        scope.base_commit(),
        None,
        "a non-object-id approval must widen to the full scope"
    );
    assert_eq!(scope.skip_log(), None, "a full scope never skips");
    // Without the guard, `git log --reverse --format=%H --output=pwned..worker-w1`
    // would have created this file.
    assert!(
        !repo.join("pwned..worker-w1").exists(),
        "no git option was executed"
    );
}
