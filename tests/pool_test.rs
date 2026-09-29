//! Integration tests for the command-parsing helpers in
//! [`mini_swe_mcp::pool`].
//!
//! The pool intercepts bash commands issued by agents and recognises two
//! "control" commands that let a worker communicate with the orchestrator:
//!
//! * `echo REQUEST_TURNS: <n>`  -> ask for `n` more turns
//! * `echo "ASK_ORCHESTRATOR: <question>"` -> pause and escalate a question
//!
//! It also builds short, log-friendly summaries of every command it runs.
//! These tests exercise those three pure functions through the public library
//! surface (i.e. the way an external integration test would).

use mini_swe_mcp::pool::{parse_ask_orchestrator, parse_request_turns, summarize_command};

// ---------------------------------------------------------------------------
// parse_request_turns
// ---------------------------------------------------------------------------

#[test]
fn test_parse_request_turns_echo_returns_count() {
    assert_eq!(parse_request_turns("echo REQUEST_TURNS: 15", ""), Some(15));
}

#[test]
fn test_parse_request_turns_variants() {
    // `printf` is accepted just like `echo`.
    assert_eq!(parse_request_turns("printf 'REQUEST_TURNS: 15'", ""), Some(15));
    // Different counts, and surrounding whitespace is irrelevant.
    assert_eq!(parse_request_turns("echo REQUEST_TURNS: 20", ""), Some(20));
    assert_eq!(parse_request_turns("echo REQUEST_TURNS: 1", ""), Some(1));
    assert_eq!(parse_request_turns("   echo REQUEST_TURNS: 42   ", ""), Some(42));
    // Leading whitespace between the token and the number is skipped.
    assert_eq!(parse_request_turns("echo REQUEST_TURNS:   7", ""), Some(7));
    // Trailing characters after the number are ignored.
    assert_eq!(parse_request_turns("echo REQUEST_TURNS: 15 # more", ""), Some(15));
}

#[test]
fn test_parse_request_turns_non_matching_commands_return_none() {
    // Right token, wrong command.
    assert_eq!(parse_request_turns("cat file.rs", "REQUEST_TURNS: 15"), None);
    // Right command, no token.
    assert_eq!(parse_request_turns("echo nothing", "normal output"), None);
    // Unrelated commands.
    assert_eq!(parse_request_turns("ls -la", "total 12"), None);
    assert_eq!(parse_request_turns("git status", "On branch main"), None);
    // The output stream is never consulted for these control commands.
    assert_eq!(parse_request_turns("echo done", "REQUEST_TURNS: 15"), None);
    // Not an echo/printf invocation at all.
    assert_eq!(parse_request_turns("REQUEST_TURNS: 15", ""), None);
    assert_eq!(parse_request_turns("myecho REQUEST_TURNS: 15", ""), None);
}

#[test]
fn test_parse_request_turns_rejects_degenerate_numbers() {
    // Zero turns is not a meaningful request.
    assert_eq!(parse_request_turns("echo REQUEST_TURNS: 0", ""), None);
    // Missing / non-numeric value.
    assert_eq!(parse_request_turns("echo REQUEST_TURNS:", ""), None);
    assert_eq!(parse_request_turns("echo REQUEST_TURNS: abc", ""), None);
    assert_eq!(parse_request_turns("echo REQUEST_TURNS: -3", ""), None);
    // Overflows `usize`.
    assert_eq!(
        parse_request_turns("echo REQUEST_TURNS: 99999999999999999999999", ""),
        None
    );
}

#[test]
fn test_parse_request_turns_empty_input() {
    assert_eq!(parse_request_turns("", ""), None);
    assert_eq!(parse_request_turns("   \n\t ", ""), None);
}

// ---------------------------------------------------------------------------
// parse_ask_orchestrator
// ---------------------------------------------------------------------------

#[test]
fn test_parse_ask_orchestrator_double_quoted_echo() {
    assert_eq!(
        parse_ask_orchestrator("echo ASK_ORCHESTRATOR: \"Proceed?\"", ""),
        Some("Proceed?".to_string())
    );
    assert_eq!(
        parse_ask_orchestrator("echo \"ASK_ORCHESTRATOR: is this ok?\"", ""),
        Some("is this ok?".to_string())
    );
}

#[test]
fn test_parse_ask_orchestrator_single_quoted_echo() {
    assert_eq!(
        parse_ask_orchestrator("echo 'ASK_ORCHESTRATOR: should I delete old code?'", ""),
        Some("should I delete old code?".to_string())
    );
    assert_eq!(
        parse_ask_orchestrator("printf 'ASK_ORCHESTRATOR: continue?'", ""),
        Some("continue?".to_string())
    );
    // Surrounding whitespace of the command is trimmed first.
    assert_eq!(
        parse_ask_orchestrator("   echo \"ASK_ORCHESTRATOR:  hello  \"   ", ""),
        Some("hello".to_string())
    );
}

#[test]
fn test_parse_ask_orchestrator_non_matching_commands_return_none() {
    // Right token, wrong command (and the output is ignored).
    assert_eq!(
        parse_ask_orchestrator(
            "cat src/agent.rs",
            "echo 'ASK_ORCHESTRATOR: <your specific question>'"
        ),
        None
    );
    // Unrelated command and output.
    assert_eq!(parse_ask_orchestrator("ls -la", "total 12"), None);
    assert_eq!(parse_ask_orchestrator("cargo build", "Finished release"), None);
    // Right command, no token.
    assert_eq!(parse_ask_orchestrator("echo done", "all good"), None);
    // A bare command that merely mentions the token.
    assert_eq!(parse_ask_orchestrator("ASK_ORCHESTRATOR: why?", ""), None);
    assert_eq!(parse_ask_orchestrator("grepecho ASK_ORCHESTRATOR: why?", ""), None);
}

#[test]
fn test_parse_ask_orchestrator_placeholders_and_blanks_are_rejected() {
    // The literal templates documented for agents must not escalate.
    assert_eq!(
        parse_ask_orchestrator("echo 'ASK_ORCHESTRATOR: <your specific question>'", ""),
        None
    );
    assert_eq!(parse_ask_orchestrator("echo \"ASK_ORCHESTRATOR: <question>\"", ""), None);
    // Empty question.
    assert_eq!(parse_ask_orchestrator("echo ASK_ORCHESTRATOR:", ""), None);
    assert_eq!(parse_ask_orchestrator("echo \"ASK_ORCHESTRATOR:   \"", ""), None);
    assert_eq!(parse_ask_orchestrator("", ""), None);
}

// ---------------------------------------------------------------------------
// summarize_command
// ---------------------------------------------------------------------------

#[test]
fn test_summarize_command_uses_only_the_first_line() {
    // Only the first line is considered...
    let cmd = "cargo build\nrm -rf /tmp/whatever\ncurl https://example.com";
    assert_eq!(summarize_command(cmd), "cargo build");

    // ...and at most the first four whitespace separated words of it.
    let long_first_line = "git commit -m 'a message that is quite long' --amend";
    assert_eq!(summarize_command(long_first_line), "git commit -m 'a");
}

#[test]
fn test_summarize_command_collapses_whitespace() {
    assert_eq!(summarize_command("  ls   -la    /tmp  "), "ls -la /tmp");
    // Only the first line of a CRLF command is summarised.
    assert_eq!(summarize_command("echo\t\thello\r\nworld"), "echo hello");
    // A leading blank line means the first line is empty -> generic fallback.
    assert_eq!(summarize_command("\n\nls -la"), "bash");
}

#[test]
fn test_summarize_command_falls_back_to_bash() {
    assert_eq!(summarize_command(""), "bash");
    assert_eq!(summarize_command("   \n\t  \n"), "bash");
}

#[test]
fn test_summarize_command_truncates_long_summaries() {
    // Four long words -> well over the 40 byte cap, so the summary is cut to
    // at most 37 bytes plus an ellipsis.
    let cmd = "echo alpha-bravo-charlie-delta-echo-foxtrot-golf-hotel-india-juliett";
    let summary = summarize_command(cmd);
    assert!(summary.ends_with("..."), "expected truncation, got {summary:?}");
    assert_eq!(summary.len(), 40, "summary should be 37 bytes + '...'");
    assert!(summary.is_char_boundary(summary.len()));
}

#[test]
fn test_summarize_command_multibyte_utf8() {
    // A long word containing non-ASCII characters is still truncated safely.
    let cmd = "echo 'esta_es_una_palabra_extremadamente_larga_con_ñ_y_acentos_para_superar_limite'";
    let summary = summarize_command(cmd);
    assert!(summary.ends_with("..."), "expected truncation, got {summary:?}");
    assert_eq!(summary.len(), 40);
    // The visible prefix must be valid UTF-8 and made of whole characters.
    let visible = summary.trim_end_matches("...");
    assert!(std::str::from_utf8(visible.as_bytes()).is_ok());
    assert!(visible.is_char_boundary(visible.len()));
}

#[test]
fn test_summarize_command_multibyte_boundary_is_not_split() {
    // '€' is three bytes and would straddle the byte-37 cut point, so the
    // implementation must back off to a char boundary instead of slicing a
    // partial code point.
    let mut special = "a".repeat(36);
    special.push('€');
    special.push_str(" rest of command");
    let summary = summarize_command(&special);
    assert!(summary.ends_with("..."), "expected truncation, got {summary:?}");
    assert_eq!(summary, format!("{}...", "a".repeat(36)));
    assert_eq!(summary.len(), 39);
    assert!(std::str::from_utf8(summary.as_bytes()).is_ok());
}

#[test]
fn test_summarize_command_other_multibyte_boundaries() {
    // A 2-byte char straddling the cut point: 'Ж' occupies bytes 36..38, so
    // the cut backs off to byte 36.
    let mut cyrillic = "a".repeat(36);
    cyrillic.push('Ж');
    cyrillic.push_str(" tail");
    let summary = summarize_command(&cyrillic);
    assert_eq!(summary, format!("{}...", "a".repeat(36)));
    assert_eq!(summary.len(), 39);

    // The 4-byte 🦀 occupies bytes 34..38, so the cut backs off to byte 34.
    let mut crab = "b".repeat(34);
    crab.push('🦀');
    crab.push_str(" tail");
    let summary = summarize_command(&crab);
    assert_eq!(summary, format!("{}...", "b".repeat(34)));
    assert_eq!(summary.len(), 37);
    assert!(std::str::from_utf8(summary.as_bytes()).is_ok());
}

#[test]
fn test_summarize_command_short_multibyte_is_untouched() {
    // No truncation: the summary is returned verbatim.
    assert_eq!(summarize_command("echo 'ñandú café ☕'"), "echo 'ñandú café ☕'");
    assert_eq!(summarize_command("ls 日本語 ファイル"), "ls 日本語 ファイル");
    // Byte length above 40 is not reached by this one.
    let s = summarize_command("echo 'ñandú café ☕'");
    assert!(s.len() < 40);
    assert!(!s.ends_with("..."));
}

// ----------
// Steer / lock-discipline (audit opt_02_pool_locks, H-1)
// ----------

use mini_swe_mcp::pool::{WorkerPhase, WorkerPool, WorkerRecord, WorkerState};

fn running_worker(id: &str) -> WorkerRecord {
    WorkerRecord {
        id: id.to_string(),
        task: "t".into(),
        model: "m".into(),
        state: WorkerState::Running {
            step: 1,
            last_command: "ls".into(),
            started_at: 0,
        },
        logs: Vec::new(),
        pending_steer: Vec::new(),
        resume_tx: None,
        handle: None,
    }
}

fn paused_worker(id: &str, tx: tokio::sync::mpsc::Sender<String>) -> WorkerRecord {
    WorkerRecord {
        state: WorkerState::Paused {
            question: "q?".into(),
            step: 2,
            paused_at: 0,
        },
        resume_tx: Some(tx),
        ..running_worker(id)
    }
}

#[tokio::test]
async fn steer_on_running_worker_queues_without_lock_convoy() {
    let pool = WorkerPool::new(1, "http://x".into(), "k".into());
    pool.__test_insert_worker(running_worker("w1")).await;

    pool.steer("w1", "please focus".into()).await.unwrap();

    // The guidance is queued for the next turn and the worker is still running.
    let progress = pool.worker_progress("w1").await.unwrap();
    assert_eq!(progress.phase, WorkerPhase::Running);
    assert_eq!(progress.step, 1);
}

#[tokio::test]
async fn steer_does_not_hold_write_guard_across_send() {
    // Fill the capacity-1 resume channel *before* handing the sender to the
    // record, so the `send()` performed by `steer` has to await a full buffer
    // (nobody is receiving). If `steer` still held the write-guard across that
    // await -- the pre-fix behaviour -- every other pool operation (including
    // reads) would convoy behind it, and the `worker_progress` call below would
    // time out.
    let pool = WorkerPool::new(1, "http://x".into(), "k".into());
    let (tx, mut rx) = tokio::sync::mpsc::channel(1);
    tx.send("occupying the single buffer slot".into())
        .await
        .unwrap();
    pool.__test_insert_worker(paused_worker("w2", tx)).await;

    let steer_fut = pool.steer("w2", "blocked guidance".into());
    tokio::pin!(steer_fut);
    // Give the future a chance to acquire the guard and park on the full
    // channel. It cannot complete: the buffer is full and `rx` is untouched.
    assert!(
        tokio::time::timeout(std::time::Duration::from_millis(100), &mut steer_fut)
            .await
            .is_err(),
        "steer should still be pending on the full resume channel"
    );

    // The write-guard must NOT be held while that send is in flight.
    let read = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        pool.worker_progress("w2"),
    )
    .await
    .expect("read-lock acquisition deadlocked behind an in-flight steer send")
    .expect("worker still registered");
    assert_eq!(read.phase, WorkerPhase::Paused);

    // Writers must be servable too, and once the slot frees the pending send
    // completes normally -- proving the lock was simply released, not leaked.
    let drained = tokio::spawn(async move {
        let first = rx.recv().await;
        let second = rx.recv().await;
        (first, second)
    });
    tokio::time::timeout(std::time::Duration::from_secs(5), steer_fut)
        .await
        .expect("steer never completed after the buffer drained")
        .expect("steer returned an error");
    let (first, second) = drained.await.unwrap();
    assert_eq!(first.as_deref(), Some("occupying the single buffer slot"));
    assert_eq!(
        second.as_deref(),
        Some("blocked guidance"),
        "the message that had to wait for the channel must be delivered, not dropped"
    );
}

#[tokio::test]
async fn steer_reports_missing_resume_channel_instead_of_dropping_message() {
    // A paused worker whose sender was already taken (e.g. it has been resumed
    // concurrently) must surface an error rather than silently swallow the
    // orchestrator guidance and return Ok(()).
    let pool = WorkerPool::new(1, "http://x".into(), "k".into());
    let (tx, _rx) = tokio::sync::mpsc::channel(1);
    let mut w = paused_worker("w3", tx);
    w.resume_tx = None;
    pool.__test_insert_worker(w).await;

    let err = pool.steer("w3", "guidance".into()).await.unwrap_err();
    assert!(
        err.to_string().contains("w3"),
        "error should name the worker, got: {err}"
    );
}

#[tokio::test]
async fn steer_rejects_unknown_and_unsteerable_workers() {
    let pool = WorkerPool::new(1, "http://x".into(), "k".into());
    let err = pool.steer("nope", "x".into()).await.unwrap_err();
    assert!(err.to_string().contains("Worker not found"));

    let mut done = running_worker("w4");
    done.state = WorkerState::Completed {
        turns: 1,
        diff: String::new(),
        summary: String::new(),
        completed_at: 0,
        artifacts: vec![],
        branch: None,
    };
    pool.__test_insert_worker(done).await;
    let err = pool.steer("w4", "x".into()).await.unwrap_err();
    assert!(err.to_string().contains("not in a steerable state"));
}

// ----------
// Lightweight progress polling (audit opt_02_pool_locks, H-5)
// ----------

#[tokio::test]
async fn worker_progress_reports_phase_step_and_command() {
    let pool = WorkerPool::new(1, "http://x".into(), "k".into());
    pool.__test_insert_worker(running_worker("p1")).await;

    let p = pool.worker_progress("p1").await.unwrap();
    assert_eq!(p.phase, WorkerPhase::Running);
    assert_eq!(p.step, 1);
    assert_eq!(p.last_command.as_deref(), Some("ls"));
    assert!(!p.terminal);

    assert!(pool.worker_progress("missing").await.is_none());
}

#[tokio::test]
async fn worker_progress_never_clones_the_terminal_payload() {
    // A completed worker carries a large `diff`. Polling progress must report
    // the terminal phase without pulling that payload out of the record.
    let pool = WorkerPool::new(1, "http://x".into(), "k".into());
    let mut w = running_worker("p2");
    w.state = WorkerState::Completed {
        turns: 7,
        diff: "d".repeat(2 * 1024 * 1024),
        summary: "s".repeat(1024),
        completed_at: 0,
        artifacts: vec!["a".repeat(4096)],
        branch: Some("feature".into()),
    };
    pool.__test_insert_worker(w).await;

    let p = pool.worker_progress("p2").await.unwrap();
    assert_eq!(p.phase, WorkerPhase::Completed);
    assert_eq!(p.step, 7);
    assert!(p.terminal);
    assert!(p.last_command.is_none());
    assert!(p.question.is_none());

    // The full payload is still available, untouched, on the terminal path.
    match pool.get_worker_state("p2").await.unwrap() {
        WorkerState::Completed { diff, .. } => assert_eq!(diff.len(), 2 * 1024 * 1024),
        other => panic!("expected Completed, got {other:?}"),
    }
}

#[tokio::test]
async fn worker_progress_reports_paused_questions() {
    let pool = WorkerPool::new(1, "http://x".into(), "k".into());
    let (tx, _rx) = tokio::sync::mpsc::channel(1);
    pool.__test_insert_worker(paused_worker("p3", tx)).await;

    let p = pool.worker_progress("p3").await.unwrap();
    assert_eq!(p.phase, WorkerPhase::Paused);
    assert_eq!(p.step, 2);
    assert_eq!(p.question.as_deref(), Some("q?"));
    assert!(!p.terminal);
}

#[tokio::test]
async fn worker_progress_reports_failed_workers() {
    let pool = WorkerPool::new(1, "http://x".into(), "k".into());
    let mut w = running_worker("p4");
    w.state = WorkerState::Failed {
        error: "boom".into(),
        step: 3,
        failed_at: 0,
    };
    pool.__test_insert_worker(w).await;

    let p = pool.worker_progress("p4").await.unwrap();
    assert_eq!(p.phase, WorkerPhase::Failed);
    assert_eq!(p.step, 3);
    assert!(p.terminal);
}

// ----------
// Non-cloning log access (audit opt_02_pool_locks, H-3)
// ----------

#[tokio::test]
async fn take_worker_logs_moves_history_without_clearing_state() {
    use mini_swe_mcp::agent::AgentStepLog;

    let pool = WorkerPool::new(1, "http://x".into(), "k".into());
    let mut w = running_worker("l1");
    w.logs = vec![AgentStepLog {
        step: 1,
        command: "ls".into(),
        output: "out".into(),
        exit_code: Some(0),
    }];
    pool.__test_insert_worker(w).await;

    assert_eq!(pool.worker_step_count("l1").await, Some(1));
    let logs = pool.take_worker_logs("l1").await.unwrap();
    assert_eq!(logs.len(), 1);
    assert_eq!(logs[0].output, "out");
    // Moving out of the record does not evict it: terminal state stays queryable.
    assert_eq!(pool.worker_step_count("l1").await, Some(0));
    assert!(pool.get_worker_state("l1").await.is_some());
    assert!(pool.take_worker_logs("l1").await.unwrap().is_empty());
    assert!(pool.take_worker_logs("nope").await.is_none());
}

#[tokio::test]
async fn take_worker_evicts_and_returns_the_full_record() {
    let pool = WorkerPool::new(1, "http://x".into(), "k".into());
    pool.__test_insert_worker(running_worker("c1")).await;

    let collected = pool.take_worker("c1").await.unwrap();
    assert_eq!(collected.id, "c1");
    assert!(pool.get_worker_state("c1").await.is_none());
    assert!(pool.take_worker("c1").await.is_none());
    // `collect` keeps its original evict-and-return semantics.
    assert!(pool.collect("c1").await.is_none());
}
