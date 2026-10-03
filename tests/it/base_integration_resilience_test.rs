//! Base integration survives teardown and starts before a continued worker's first turn.
use std::path::Path;
use std::time::Duration;
use crate::common;

use mini_swe_mcp::agent::{ChatMessage, Role};
use mini_swe_mcp::pool::{WorkerHistory, WorkerPool, WorkerState, save_worker_history_in};
use mini_swe_mcp::worktree::{BaseSync, ScratchRoot, WorktreeGuard};

fn seed(repo: &Path, root: &ScratchRoot, id: &str) -> WorkerHistory {
    common::git(repo, &["init", "-b", "master"]);
    common::git(repo, &["config", "user.name", "test"]);
    common::git(repo, &["config", "user.email", "test@localhost"]);
    for name in ["a.rs", "b.rs"] {
        std::fs::write(repo.join(name), "base\n").unwrap();
    }
    common::git(repo, &["add", "."]);
    common::git(repo, &["commit", "-m", "base"]);
    let mut guard = WorktreeGuard::new_in(root, repo, id).unwrap();
    for name in ["a.rs", "b.rs"] {
        std::fs::write(guard.path.join(name), "worker\n").unwrap();
        std::fs::write(repo.join(name), "master\n").unwrap();
    }
    guard.commit_changes("worker changes").unwrap();
    common::git(repo, &["add", "."]);
    common::git(repo, &["commit", "-m", "advance master"]);
    let history = WorkerHistory {
        task: "integrate master".into(),
        group: None,
        role: mini_swe_mcp::pool::WorkerRole::Worker,
        model: "test-model".into(),
        temperature: None,
        repo_path: repo.to_string_lossy().into(),
        base_commit: guard.base_commit.clone(),
        base_branch: guard.base_branch.clone(),
        branch: guard.branch.clone(),
        network_offline: false,
        verify: None,
        client_env: Vec::new(),
        max_turns: 2,
        review_after: None,
        revision: 0,
        auto_continues: 0,
        owner: None,
        messages: vec![
            ChatMessage::text(Role::System, "Use bash."),
            ChatMessage::text(Role::User, "integrate master"),
        ],
    };
    drop(guard);
    save_worker_history_in(root, id, &history).unwrap();
    history
}

async fn stopped(pool: &WorkerPool, root: &ScratchRoot, id: &str) -> WorkerState {
    tokio::time::timeout(Duration::from_secs(20), async {
        loop {
            if let Some(state) = pool.get_worker_state(id).await
                && !matches!(
                    state,
                    WorkerState::Running { .. } | WorkerState::Paused { .. }
                )
                && !root.join(format!("swe-wt-{id}")).exists()
            {
                return state;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("worker teardown")
}

#[tokio::test]
async fn budget_exhaustion_keeps_resolved_hunks_and_pending_files_on_continuation() {
    let repo = common::TempDir::new_in_tmp("base-resilience-repo");
    let scratch = common::TempDir::new_in_tmp("base-resilience-root");
    let root = ScratchRoot::new(scratch.path());
    let history = seed(repo.path(), &root, "budget");
    let llm = common::fake_llm::FakeLlm::spawn(
        "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
        "printf 'resolved a\\n' > a.rs",
    )
    .await;
    let pool = WorkerPool::with_scratch(1, llm.base_url().into(), "key".into(), root.clone());
    pool.steer_with_budget("budget", "resolve master conflicts".into(), Some(2))
        .await
        .unwrap();
    assert!(matches!(
        stopped(&pool, &root, "budget").await,
        WorkerState::Failed { .. }
    ));
    let subject = common::git(repo.path(), &["log", "-1", "--format=%s", &history.branch]);
    assert!(subject.contains("WIP base integration"), "{subject}");
    assert!(
        subject.contains("b.rs") && !subject.contains("a.rs"),
        "{subject}"
    );
    assert_eq!(
        common::git(repo.path(), &["show", "worker-budget:a.rs"]),
        "resolved a\n"
    );
    assert!(common::git(repo.path(), &["show", "worker-budget:b.rs"]).contains("<<<<<<<"));
    let previous_messages = mini_swe_mcp::pool::load_worker_history_in(&root, "budget")
        .unwrap()
        .messages
        .len();
    pool.steer_with_budget("budget", "finish the pending merge".into(), Some(1))
        .await
        .unwrap();
    stopped(&pool, &root, "budget").await;
    let resumed = mini_swe_mcp::pool::load_worker_history_in(&root, "budget").unwrap();
    let notices: Vec<_> = resumed
        .messages
        .iter()
        .skip(previous_messages + 1)
        .take(1)
        .filter_map(|m| {
            let value = serde_json::to_value(m).unwrap();
            value["content"].as_str().map(str::to_owned)
        })
        .filter(|m| m.contains("BASE INTEGRATION") && m.contains("b.rs") && !m.contains("a.rs"))
        .collect();
    assert!(
        !notices.is_empty(),
        "remaining conflicts were not disclosed"
    );
    let mut guard =
        WorktreeGuard::reopen_in(&root, repo.path(), "budget", &history.base_commit).unwrap();
    guard.base_branch = history.base_branch;
    assert!(WorktreeGuard::merge_in_progress_at(&guard.path).unwrap());
    assert!(
        guard
            .commit_changes("must not hide pending markers")
            .is_err()
    );
    assert_eq!(
        WorktreeGuard::sync_base_at(
            &guard.path,
            repo.path(),
            &guard.branch,
            &guard.base_commit,
            Some("master")
        )
        .unwrap(),
        BaseSync::Conflicts {
            branch: "master".into(),
            files: vec!["b.rs".into()]
        }
    );
    std::fs::write(guard.path.join("b.rs"), "resolved b\n").unwrap();
    assert!(matches!(
        WorktreeGuard::sync_base_at(
            &guard.path,
            repo.path(),
            &guard.branch,
            &guard.base_commit,
            Some("master")
        )
        .unwrap(),
        BaseSync::Merged { .. }
    ));
    assert!(!WorktreeGuard::merge_in_progress_at(&guard.path).unwrap());
    assert_eq!(
        std::fs::read_to_string(guard.path.join("a.rs")).unwrap(),
        "resolved a\n"
    );
}

#[tokio::test]
async fn steer_integrates_stale_base_before_the_first_command() {
    let repo = common::TempDir::new_in_tmp("base-steer-repo");
    let scratch = common::TempDir::new_in_tmp("base-steer-root");
    let root = ScratchRoot::new(scratch.path());
    seed(repo.path(), &root, "stale");
    let llm = common::fake_llm::FakeLlm::spawn(
        "grep '^<<<<<<<' b.rs && echo CONFLICT_PRESENT_ON_FIRST_TURN",
        "true",
    )
    .await;
    let pool = WorkerPool::with_scratch(1, llm.base_url().into(), "key".into(), root.clone());
    pool.steer_with_budget("stale", "resolve conflicts with master".into(), Some(1))
        .await
        .unwrap();
    stopped(&pool, &root, "stale").await;
    let history = mini_swe_mcp::pool::load_worker_history_in(&root, "stale").unwrap();
    let encoded = serde_json::to_value(&history.messages).unwrap();
    let messages = encoded.as_array().unwrap();
    let assistant = messages
        .iter()
        .position(|m| m["role"] == "assistant")
        .unwrap();
    assert!(messages[..assistant].iter().any(|m| {
        m["content"]
            .as_str()
            .is_some_and(|s| s.contains("BASE INTEGRATION") && s.contains("b.rs"))
    }));
    assert!(messages.iter().any(|m| {
        m["role"] == "tool"
            && m["content"]
                .as_str()
                .is_some_and(|s| s.contains("CONFLICT_PRESENT_ON_FIRST_TURN"))
    }));
}

#[tokio::test]
async fn steer_leaves_an_already_integrated_branch_unchanged() {
    let repo = common::TempDir::new_in_tmp("base-current-repo");
    let scratch = common::TempDir::new_in_tmp("base-current-root");
    let root = ScratchRoot::new(scratch.path());
    let history = seed(repo.path(), &root, "current");
    let mut guard =
        WorktreeGuard::reopen_in(&root, repo.path(), "current", &history.base_commit).unwrap();
    guard.base_branch = Some("master".into());
    WorktreeGuard::sync_base_at(
        &guard.path,
        repo.path(),
        &guard.branch,
        &guard.base_commit,
        Some("master"),
    )
    .unwrap();
    for name in ["a.rs", "b.rs"] {
        std::fs::write(guard.path.join(name), "resolved\n").unwrap();
    }
    WorktreeGuard::sync_base_at(
        &guard.path,
        repo.path(),
        &guard.branch,
        &guard.base_commit,
        Some("master"),
    )
    .unwrap();
    let tip = common::git(repo.path(), &["rev-parse", &history.branch]);
    drop(guard);
    let llm = common::fake_llm::FakeLlm::spawn("true", "true").await;
    let pool = WorkerPool::with_scratch(1, llm.base_url().into(), "key".into(), root.clone());
    pool.steer_with_budget("current", "continue".into(), Some(1))
        .await
        .unwrap();
    stopped(&pool, &root, "current").await;
    assert_eq!(
        common::git(repo.path(), &["rev-parse", &history.branch]),
        tip
    );
    let history = mini_swe_mcp::pool::load_worker_history_in(&root, "current").unwrap();
    assert!(
        !serde_json::to_string(&history.messages)
            .unwrap()
            .contains("BASE INTEGRATION")
    );
}
