//! The adversarial security review: mode selection, the language-agnostic
//! prompt, and the automatic sensitive-path trigger.
//!
//! Three properties are asserted, one per defect class this phase exists
//! for:
//!
//! * **Mode selection.** `--review-after <model>:security` selects the
//!   adversarial prompt; a bare model id keeps the generic one. The
//!   split happens in [`ReviewMode::parse_model`], which the CLI and
//!   the MCP arg both feed through, so the two ingress paths cannot
//!   disagree.
//! * **No language-specific command.** The prompt runs the dispatch's
//!   own verify command, never a hardcoded `cargo` invocation, so a Go
//!   or Python repository is reviewed by its own gate.
//! * **Automatic trigger.** A worker whose diff touches a path the
//!   repository declared sensitive gets the security review even when
//!   nobody asked for a review, and a worker that touched nothing
//!   sensitive gets none.

use std::path::Path;

use crate::common::fake_llm::FakeLlm;

use mini_swe_mcp::manifest::{matches_sensitive, parse_sensitive_paths};
use mini_swe_mcp::pool::{ReviewMode, parse_findings, review_prompt};

// ----------
// Mode selection
// ----------

#[test]
fn parse_model_splits_the_security_suffix() {
    assert_eq!(
        ReviewMode::parse_model("nerd:security"),
        ("nerd".to_string(), ReviewMode::security())
    );
    // Case-insensitive, and the model may be a resolved id.
    assert_eq!(
        ReviewMode::parse_model("combo:nerd:SECURITY"),
        ("combo:nerd".to_string(), ReviewMode::security())
    );
}

#[test]
fn parse_model_keeps_colon_ids_as_quality() {
    // A model id legitimately contains `:`; only a trailing
    // `:security` is a mode marker.
    assert_eq!(
        ReviewMode::parse_model("combo:nerd"),
        ("combo:nerd".to_string(), ReviewMode::quality())
    );
    assert_eq!(
        ReviewMode::parse_model("some/unknown"),
        ("some/unknown".to_string(), ReviewMode::quality())
    );
    assert_eq!(
        ReviewMode::parse_model("nerd:"),
        ("nerd:".to_string(), ReviewMode::quality())
    );
}

// ----------
// The prompt
// ----------

#[test]
fn security_prompt_is_used_for_the_suffix_and_generic_otherwise() {
    let security = review_prompt(
        &ReviewMode::security(),
        "harden the socket",
        Some("make check"),
        &["src/hub/mod.rs".to_string()],
    );
    assert!(
        security.contains("ADVERSARIAL SECURITY REVIEW PHASE"),
        "the security mode must use the adversarial prompt"
    );
    assert!(
        security.contains("src/hub/mod.rs"),
        "the trigger names the sensitive paths the diff touched"
    );

    let quality = review_prompt(
        &ReviewMode::quality(),
        "harden the socket",
        Some("make check"),
        &[],
    );
    assert!(
        quality.contains("AUDIT & REVIEW PHASE"),
        "the default mode must keep the generic prompt"
    );
    assert!(
        !quality.contains("ADVERSARIAL"),
        "the generic prompt must not carry the adversarial checklist"
    );
}

#[test]
fn the_prompt_names_no_language_specific_command() {
    for mode in [ReviewMode::quality(), ReviewMode::security()] {
        let prompt = review_prompt(&mode, "task", Some("make check"), &[]);
        for forbidden in ["cargo test", "cargo clippy", "cargo test --all-targets"] {
            assert!(
                !prompt.contains(forbidden),
                "the {mode:?} prompt must not hardcode `{forbidden}`"
            );
        }
        assert!(
            prompt.contains("make check"),
            "the prompt must run the dispatch's own verify command"
        );
    }
}

#[test]
fn a_disabled_gate_is_stated_not_invented() {
    let prompt = review_prompt(&ReviewMode::quality(), "task", Some(""), &[]);
    assert!(
        prompt.contains("(none: this dispatch disabled the completion gate)"),
        "a disabled gate must be stated, not replaced by an invented suite"
    );
}

// ----------
// The finding count
// ----------

#[test]
fn parse_findings_reads_only_the_marker_line() {
    assert_eq!(
        parse_findings("REPORT\ndone: x\nFINDINGS: 3\nrisks: none"),
        Some(3)
    );
    assert_eq!(parse_findings("REPORT\ndone: x\nFINDINGS: 0"), Some(0));
    // A missing line is `None`, never a reassuring zero.
    assert_eq!(parse_findings("REPORT\ndone: x"), None);
    assert_eq!(parse_findings("FINDINGS: many"), None);
}

// ----------
// AGENTS.md parsing
// ----------

#[test]
fn parse_sensitive_paths_reads_one_glob_per_line() {
    let text = "\
# rules

## Sensitive paths
- src/hub/**
src/agent/sandbox*
`src/pool/merge.rs`

# a comment, ignored

## Other section
src/ignored.rs
";
    let paths = parse_sensitive_paths(text);
    assert_eq!(
        paths,
        vec![
            "src/hub/**".to_string(),
            "src/agent/sandbox*".to_string(),
            "src/pool/merge.rs".to_string(),
        ]
    );
}

#[test]
fn parse_sensitive_paths_is_empty_without_the_section() {
    assert!(parse_sensitive_paths("## Rules\n- nothing\n").is_empty());
    assert!(parse_sensitive_paths("").is_empty());
}

/// The repository declares its own sensitive surfaces in `AGENTS.md`, so
/// the automatic trigger has something to match against.
#[test]
fn this_repository_declares_its_sensitive_paths() {
    let paths = mini_swe_mcp::manifest::sensitive_paths(Path::new("."));
    for expected in [
        "src/hub/**",
        "src/agent/sandbox*",
        "src/agent/exec*",
        "src/worktree/guard.rs",
        "src/pool/merge.rs",
        "src/pool/revision.rs",
        "src/hub/identity.rs",
        "src/mcp/events.rs",
    ] {
        assert!(
            paths.iter().any(|p| p == expected),
            "AGENTS.md must declare `{expected}` as sensitive: {paths:?}"
        );
    }
    // A path the repository did not declare must not match.
    assert!(!matches_sensitive("src/cli/args.rs", &paths));
}

#[test]
fn glob_matching_crosses_segments_only_for_double_star() {
    let patterns = vec!["src/hub/**".to_string()];
    assert!(matches_sensitive("src/hub/mod.rs", &patterns));
    assert!(matches_sensitive("src/hub/sub/file.rs", &patterns));
    assert!(!matches_sensitive("src/hub", &patterns));
    assert!(!matches_sensitive("src/agent/mod.rs", &patterns));

    let patterns = vec!["src/agent/sandbox*".to_string()];
    assert!(matches_sensitive("src/agent/sandbox.rs", &patterns));
    // A single `*` stays inside one segment.
    assert!(!matches_sensitive("src/agent/sandbox/mod.rs", &patterns));

    let patterns = vec!["src/pool/merge.rs".to_string()];
    assert!(matches_sensitive("src/pool/merge.rs", &patterns));
    assert!(!matches_sensitive("src/pool/merge.rs.bak", &patterns));
}

// ----------
// The automatic trigger
// ----------

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

/// The security reviewer's completion: the sentinel plus the
/// finding count the harness reads back.
fn security_completion_turn(call_id: &str, findings: usize) -> Vec<String> {
    completion_turn(
        call_id,
        &format!("REPORT\ndone: adversarial pass\nFINDINGS: {findings}\nrisks: none"),
    )
}

/// A worker turn that writes `path` in its worktree, so the worker's diff
/// touches it. The parent directory is created first so a nested path works.
fn write_turn(call_id: &str, path: &str) -> Vec<String> {
    let parent = std::path::Path::new(path)
        .parent()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_default();
    let command = if parent.is_empty() {
        format!("echo changed > {path}")
    } else {
        format!("mkdir -p {parent} && echo changed > {path}")
    };
    turn(call_id, "", &command)
}

/// Dispatch one worker and wait for its terminal state.
async fn dispatch_and_wait(
    base_url: &str,
    repo: &Path,
    review_after: Option<String>,
) -> (WorkerPool, String, WorkerState, common::TempDir) {
    let scratch = common::TempDir::new_in_tmp("review-security-pool");
    let pool = WorkerPool::with_scratch(
        1,
        base_url.to_string(),
        "test-key".to_string(),
        ScratchRoot::new(scratch.path()),
    );
    let worker_id = pool
        .dispatch(
            TEST_OWNER.to_string(),
            "exercise the review trigger".to_string(),
            "test-model".to_string(),
            None,
            repo.to_path_buf(),
            5,
            Some("review-security".to_string()),
            review_after,
            false,
            None,
            Vec::new(),
        )
        .await
        .expect("dispatch the worker");
    let state = common::wait_for_terminal(&pool, &worker_id).await;
    (pool, worker_id, state, scratch)
}

/// A worker whose diff touches a declared sensitive path gets the
/// security review automatically, with no `review_after` at all.
#[tokio::test]
async fn a_sensitive_diff_triggers_the_security_review_automatically() {
    let repo = common::TestRepo::new("auto");
    repo.declare_sensitive(&["src/hub/**"]);
    let server = FakeLlm::spawn_sse(vec![
        write_turn("call_write", "src/hub/mod.rs"),
        turn(
            "call_impl",
            "REPORT\ndone: impl\nrisks: none",
            &format!("echo {COMPLETION_SENTINEL}"),
        ),
        security_completion_turn("call_review", 2),
    ])
    .await;

    let (pool, worker_id, state, _scratch) =
        dispatch_and_wait(server.base_url(), repo.path(), None).await;

    // The security review ran: the reviewer's turn exists, and the
    // completion event carries its finding count.
    let requests = server.request_bodies().await;
    assert_eq!(
        requests.len(),
        3,
        "write turn, implementer completion, automatic security review"
    );
    let reviewer_request = &requests[2];
    // The default catalog marks no `strongest:` tier, so the automatic
    // reviewer is the dispatch default (`ninja`), never the implementer's own
    // `test-model`. See `review_reviewer_choice_test.rs` for the rule and its
    // strongest-tier case.
    assert_eq!(
        reviewer_request["model"],
        json!("combo:ninja"),
        "the automatic review must not run on the implementer's model"
    );
    let prompt = common::review_prompt_of(&server)
        .await
        .expect("a review prompt");
    assert!(
        prompt.contains("ADVERSARIAL SECURITY REVIEW PHASE"),
        "the automatic trigger must run the adversarial prompt"
    );
    assert!(
        prompt.contains("src/hub/mod.rs"),
        "the prompt names the sensitive path the diff touched"
    );

    // The finding count rides the registry row, which is what the
    // completion event and the status read.
    let registry = mini_swe_mcp::pool::load_registry_entry_in(pool.scratch_root(), &worker_id)
        .expect("the worker's registry row");
    let security = registry
        .security_review
        .expect("the row records the security review");
    assert_eq!(security.findings, Some(2));

    let WorkerState::Completed { .. } = &state else {
        panic!("worker must complete, got {state:?}")
    };
    let _ = pool.kill(&worker_id).await;
}

/// A worker whose diff touches nothing sensitive gets no review phase
/// when nobody asked for one.
#[tokio::test]
async fn an_insensitive_diff_gets_no_review_phase() {
    let repo = common::TestRepo::new("skip");
    repo.declare_sensitive(&["src/hub/**"]);
    let server = FakeLlm::spawn_sse(vec![
        write_turn("call_write", "src/ordinary.rs"),
        turn(
            "call_impl",
            "REPORT\ndone: impl\nrisks: none",
            &format!("echo {COMPLETION_SENTINEL}"),
        ),
    ])
    .await;

    let (pool, _worker_id, state, _scratch) =
        dispatch_and_wait(server.base_url(), repo.path(), None).await;

    assert_eq!(
        server.requests(),
        2,
        "write turn and completion, but no review phase"
    );
    assert!(
        common::review_prompt_of(&server).await.is_none(),
        "no review prompt must be sent"
    );
    let WorkerState::Completed { .. } = &state else {
        panic!("worker must complete, got {state:?}")
    };
    let _ = pool.kill(&_worker_id).await;
}

/// A repository that declares no sensitive paths never triggers the
/// review, even for a diff that would have matched.
#[tokio::test]
async fn no_declared_patterns_means_no_trigger() {
    let repo = common::TestRepo::new("undeclared");
    let server = FakeLlm::spawn_sse(vec![
        write_turn("call_write", "src/hub/mod.rs"),
        turn(
            "call_impl",
            "REPORT\ndone: impl\nrisks: none",
            &format!("echo {COMPLETION_SENTINEL}"),
        ),
    ])
    .await;

    let (pool, _worker_id, state, _scratch) =
        dispatch_and_wait(server.base_url(), repo.path(), None).await;

    assert_eq!(
        server.requests(),
        2,
        "write turn and completion, but no trigger"
    );
    let WorkerState::Completed { .. } = &state else {
        panic!("worker must complete, got {state:?}")
    };
    let _ = pool.kill(&_worker_id).await;
}

/// `--review-after <model>:security` runs the adversarial prompt on the
/// requested model even when the diff touches nothing sensitive.
#[tokio::test]
async fn the_security_suffix_selects_the_adversarial_review() {
    let repo = common::TestRepo::new("suffix");
    let server = FakeLlm::spawn_sse(vec![
        write_turn("call_write", "src/ordinary.rs"),
        turn(
            "call_impl",
            "REPORT\ndone: impl\nrisks: none",
            &format!("echo {COMPLETION_SENTINEL}"),
        ),
        security_completion_turn("call_review", 0),
    ])
    .await;

    let (pool, _worker_id, state, _scratch) = dispatch_and_wait(
        server.base_url(),
        repo.path(),
        Some("test-reviewer:security".to_string()),
    )
    .await;

    let requests = server.request_bodies().await;
    assert_eq!(
        requests.len(),
        3,
        "write, implementer, and the requested review"
    );
    assert_eq!(
        requests[2]["model"],
        json!("test-reviewer"),
        "the suffix selects the mode, not the model"
    );
    let prompt = common::review_prompt_of(&server)
        .await
        .expect("a review prompt");
    assert!(prompt.contains("ADVERSARIAL SECURITY REVIEW PHASE"));
    // Nothing sensitive was touched, so the trigger adds no paths.
    assert!(!prompt.contains("declared sensitive"));

    let WorkerState::Completed { .. } = &state else {
        panic!("worker must complete, got {state:?}")
    };
    let _ = pool.kill(&_worker_id).await;
}

/// A bare `--review-after <model>` keeps the generic prompt.
#[tokio::test]
async fn a_bare_review_after_keeps_the_generic_prompt() {
    let repo = common::TestRepo::new("bare");
    let server = FakeLlm::spawn_sse(vec![
        write_turn("call_write", "src/ordinary.rs"),
        turn(
            "call_impl",
            "REPORT\ndone: impl\nrisks: none",
            &format!("echo {COMPLETION_SENTINEL}"),
        ),
        completion_turn("call_review", "REPORT\ndone: reviewed\nrisks: none"),
    ])
    .await;

    let (pool, _worker_id, state, _scratch) = dispatch_and_wait(
        server.base_url(),
        repo.path(),
        Some("test-reviewer".to_string()),
    )
    .await;

    let prompt = common::review_prompt_of(&server)
        .await
        .expect("a review prompt");
    assert!(prompt.contains("AUDIT & REVIEW PHASE"));
    assert!(!prompt.contains("ADVERSARIAL"));

    let WorkerState::Completed { .. } = &state else {
        panic!("worker must complete, got {state:?}")
    };
    let _ = pool.kill(&_worker_id).await;
}

/// A sensitive diff upgrades a requested review to the adversarial mode
/// and keeps the requested model.
#[tokio::test]
async fn a_sensitive_diff_upgrades_a_requested_review() {
    let repo = common::TestRepo::new("upgrade");
    repo.declare_sensitive(&["src/hub/**"]);
    let server = FakeLlm::spawn_sse(vec![
        write_turn("call_write", "src/hub/mod.rs"),
        turn(
            "call_impl",
            "REPORT\ndone: impl\nrisks: none",
            &format!("echo {COMPLETION_SENTINEL}"),
        ),
        security_completion_turn("call_review", 1),
    ])
    .await;

    let (pool, _worker_id, state, _scratch) = dispatch_and_wait(
        server.base_url(),
        repo.path(),
        Some("test-reviewer".to_string()),
    )
    .await;

    let requests = server.request_bodies().await;
    assert_eq!(requests.len(), 3);
    assert_eq!(requests[2]["model"], json!("test-reviewer"));
    let prompt = common::review_prompt_of(&server)
        .await
        .expect("a review prompt");
    assert!(prompt.contains("ADVERSARIAL SECURITY REVIEW PHASE"));
    assert!(prompt.contains("src/hub/mod.rs"));

    let WorkerState::Completed { .. } = &state else {
        panic!("worker must complete, got {state:?}")
    };
    let _ = pool.kill(&_worker_id).await;
}

/// The reviewer runs the dispatch's verify command, not a language-
/// specific suite: the gate the dispatch declared is what the prompt
/// names.
#[tokio::test]
async fn the_reviewer_runs_the_dispatch_verify_command() {
    let repo = common::TestRepo::new("verify");
    repo.declare_sensitive(&["src/hub/**"]);
    let server = FakeLlm::spawn_sse(vec![
        write_turn("call_write", "src/hub/mod.rs"),
        turn(
            "call_impl",
            "REPORT\ndone: impl\nrisks: none",
            &format!("echo {COMPLETION_SENTINEL}"),
        ),
        security_completion_turn("call_review", 0),
    ])
    .await;

    let scratch = common::TempDir::new_in_tmp("review-security-verify");
    let pool = WorkerPool::with_scratch(
        1,
        server.base_url().to_string(),
        "test-key".to_string(),
        ScratchRoot::new(scratch.path()),
    );
    let worker_id = pool
        .dispatch(
            TEST_OWNER.to_string(),
            "exercise the review trigger".to_string(),
            "test-model".to_string(),
            None,
            repo.path().to_path_buf(),
            5,
            Some("review-security".to_string()),
            None,
            false,
            Some("echo gate-ok".to_string()),
            Vec::new(),
        )
        .await
        .expect("dispatch the worker");
    let state = common::wait_for_terminal(&pool, &worker_id).await;

    let prompt = common::review_prompt_of(&server)
        .await
        .expect("a review prompt");
    assert!(
        prompt.contains("echo gate-ok"),
        "the prompt must name the dispatch's gate"
    );
    assert!(!prompt.contains("cargo"));

    let WorkerState::Completed { .. } = &state else {
        panic!("worker must complete, got {state:?}")
    };
    let _ = pool.kill(&worker_id).await;
}

// ----------
// The completion event and the status
// ----------

/// The completion event says a security review ran and how many findings
/// it reported; a review whose count was not reported is never shown as
/// a clean zero.
#[test]
fn the_completion_event_shows_the_security_review_and_count() {
    use mini_swe_mcp::mcp::{EventKind, Outcome, WorkerView, render_for_test};
    use mini_swe_mcp::pool::SecurityReviewOutcome;

    let view = WorkerView {
        worker_id: "w1".to_string(),
        event: Some(EventKind::Completed),
        status: "completed".to_string(),
        outcome: Outcome {
            security_review: Some(SecurityReviewOutcome { findings: Some(3) }),
            ..Outcome::default()
        },
        ..WorkerView::default()
    };
    let text = render_for_test(&view, EventKind::Completed);
    assert!(
        text.contains("Security review: 3 findings"),
        "the completion event must show the security review and its count:\n{text}"
    );

    let view = WorkerView {
        worker_id: "w2".to_string(),
        event: Some(EventKind::Completed),
        status: "completed".to_string(),
        outcome: Outcome {
            security_review: Some(SecurityReviewOutcome { findings: None }),
            ..Outcome::default()
        },
        ..WorkerView::default()
    };
    let text = render_for_test(&view, EventKind::Completed);
    assert!(
        text.contains("Security review: not reported findings"),
        "an unreported count must not read as a clean zero:\n{text}"
    );
}

/// The plain-text status shows the same fact.
#[test]
fn the_status_text_shows_the_security_review_and_count() {
    let text = mini_swe_mcp::cli::format::format_status(
        &serde_json::from_str(
            r#"{"worker_id":"w","state":{"state":"Completed","details":{
             "security_review":{"findings":4}}}}"#,
        )
        .expect("fixture must be valid JSON"),
    );
    assert!(
        text.contains("Security review: 4 findings"),
        "the status must show the security review and its count:\n{text}"
    );
}
