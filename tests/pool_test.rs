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
//!
//! It also covers the bounded step-log retention window, the hard per-field
//! byte ceilings, and the emission budget that keeps a single response small
//! (see `audits/opt_07_step_log_memory.md`).

use serde_json::json;
use std::io::Write;

use mini_swe_mcp::agent::AgentStepLog;
use mini_swe_mcp::pool::{
    DEFAULT_MAX_EMITTED_LOGS, DEFAULT_MAX_RETAINED_LOGS, LogBuffer, LogRetentionPolicy,
    MAX_EMITTED_LOGS_CEILING, MAX_LOG_COMMAND_BYTES, MAX_LOG_OUTPUT_BYTES, MAX_RETAINED_LOGS_CEILING,
    RegistryStatus, WorkerMetrics, WorkerPhase, WorkerPool, WorkerRecord, WorkerRegistryEntry,
    WorkerState, build_step_log, clamp_string, emit_view, parse_ask_orchestrator,
    parse_request_turns, summarize_command,
};

/// Owner recorded for the synthetic workers these tests insert: the pool's
/// lock discipline and registry coalescing are under test, not ownership.
const TEST_OWNER: &str = "test-agent";

// ---------------------------------------------------------------------------
// parse_request_turns
// ---------------------------------------------------------------------------

#[test]
fn test_parse_request_turns_echo_returns_count() {
    assert_eq!(parse_request_turns("echo REQUEST_TURNS: 15"), Some(15));
}

#[test]
fn test_parse_request_turns_variants() {
    // `printf` is accepted just like `echo`.
    assert_eq!(parse_request_turns("printf 'REQUEST_TURNS: 15'"), Some(15));
    // Different counts, and surrounding whitespace is irrelevant.
    assert_eq!(parse_request_turns("echo REQUEST_TURNS: 20"), Some(20));
    assert_eq!(parse_request_turns("echo REQUEST_TURNS: 1"), Some(1));
    assert_eq!(parse_request_turns("   echo REQUEST_TURNS: 42   "), Some(42));
    // Leading whitespace between the token and the number is skipped.
    assert_eq!(parse_request_turns("echo REQUEST_TURNS:   7"), Some(7));
    // Trailing characters after the number are ignored.
    assert_eq!(parse_request_turns("echo REQUEST_TURNS: 15 # more"), Some(15));
}

#[test]
fn test_parse_request_turns_non_matching_commands_return_none() {
    // Right token, wrong command.
    assert_eq!(parse_request_turns("cat file.rs"), None);
    // Right command, no token.
    assert_eq!(parse_request_turns("echo nothing"), None);
    // Unrelated commands.
    assert_eq!(parse_request_turns("ls -la"), None);
    assert_eq!(parse_request_turns("git status"), None);
    // Only the command itself is ever inspected for these control sentinels.
    assert_eq!(parse_request_turns("echo done"), None);
    // Not an echo/printf invocation at all.
    assert_eq!(parse_request_turns("REQUEST_TURNS: 15"), None);
    assert_eq!(parse_request_turns("myecho REQUEST_TURNS: 15"), None);
}

#[test]
fn test_parse_request_turns_rejects_degenerate_numbers() {
    // Zero turns is not a meaningful request.
    assert_eq!(parse_request_turns("echo REQUEST_TURNS: 0"), None);
    // Missing / non-numeric value.
    assert_eq!(parse_request_turns("echo REQUEST_TURNS:"), None);
    assert_eq!(parse_request_turns("echo REQUEST_TURNS: abc"), None);
    assert_eq!(parse_request_turns("echo REQUEST_TURNS: -3"), None);
    // Overflows `usize`.
    assert_eq!(
        parse_request_turns("echo REQUEST_TURNS: 99999999999999999999999"),
        None
    );
}

#[test]
fn test_parse_request_turns_empty_input() {
    assert_eq!(parse_request_turns(""), None);
    assert_eq!(parse_request_turns("   \n\t "), None);
}

// ---------------------------------------------------------------------------
// parse_ask_orchestrator
// ---------------------------------------------------------------------------

#[test]
fn test_parse_ask_orchestrator_double_quoted_echo() {
    assert_eq!(
        parse_ask_orchestrator("echo ASK_ORCHESTRATOR: \"Proceed?\""),
        Some("Proceed?".to_string())
    );
    assert_eq!(
        parse_ask_orchestrator("echo \"ASK_ORCHESTRATOR: is this ok?\""),
        Some("is this ok?".to_string())
    );
}

#[test]
fn test_parse_ask_orchestrator_single_quoted_echo() {
    assert_eq!(
        parse_ask_orchestrator("echo 'ASK_ORCHESTRATOR: should I delete old code?'"),
        Some("should I delete old code?".to_string())
    );
    assert_eq!(
        parse_ask_orchestrator("printf 'ASK_ORCHESTRATOR: continue?'"),
        Some("continue?".to_string())
    );
    // Surrounding whitespace of the command is trimmed first.
    assert_eq!(
        parse_ask_orchestrator("   echo \"ASK_ORCHESTRATOR:  hello  \"   "),
        Some("hello".to_string())
    );
}

#[test]
fn test_parse_ask_orchestrator_non_matching_commands_return_none() {
    // Right token, wrong command.
    assert_eq!(parse_ask_orchestrator("cat src/agent.rs"), None);
    // Unrelated command.
    assert_eq!(parse_ask_orchestrator("ls -la"), None);
    assert_eq!(parse_ask_orchestrator("cargo build"), None);
    // Right command, no token.
    assert_eq!(parse_ask_orchestrator("echo done"), None);
    // A bare command that merely mentions the token.
    assert_eq!(parse_ask_orchestrator("ASK_ORCHESTRATOR: why?"), None);
    assert_eq!(parse_ask_orchestrator("grepecho ASK_ORCHESTRATOR: why?"), None);
}

#[test]
fn test_parse_ask_orchestrator_placeholders_and_blanks_are_rejected() {
    // The literal templates documented for agents must not escalate.
    assert_eq!(
        parse_ask_orchestrator("echo 'ASK_ORCHESTRATOR: <your specific question>'"),
        None
    );
    assert_eq!(parse_ask_orchestrator("echo \"ASK_ORCHESTRATOR: <question>\""), None);
    // Empty question.
    assert_eq!(parse_ask_orchestrator("echo ASK_ORCHESTRATOR:"), None);
    assert_eq!(parse_ask_orchestrator("echo \"ASK_ORCHESTRATOR:   \""), None);
    assert_eq!(parse_ask_orchestrator(""), None);
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
// Cross-process steering mailbox
// ----------

/// `SWE_TEMP_DIR` is process-global, so every test that redirects the mailbox
/// root has to run alone. `RUST_TEST_THREADS` caps the harness at two threads,
/// but the mutex is what actually serialises them (a test that raced would
/// silently observe another test's base dir).
static SWE_TEMP_DIR_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// A unique scratch root under the system temp dir, for `SWE_TEMP_DIR`.
///
/// Mirrors the worktree naming so the real `swe_base_dir()` resolution path
/// (including the `swe-wt-` prefix) is exercised rather than a stub.
fn scratch_dir(tag: &str) -> String {
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let unique = format!(
        "swe-wt-test-{tag}-{}-{}-{n}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    );
    let dir = std::env::temp_dir().join(&unique);
    std::fs::create_dir_all(&dir).expect("create scratch base dir");
    dir.to_string_lossy().into_owned()
}

/// Sets `SWE_TEMP_DIR` for its lifetime and holds the global mailbox lock.
///
/// Restoring the previous value on drop matters: the rest of the suite asserts
/// against the default base dir and would otherwise inherit a stale override.
struct ScopedTempDir {
    _guard: std::sync::MutexGuard<'static, ()>,
    previous: Option<String>,
}

impl ScopedTempDir {
    fn set(dir: &str) -> Self {
        let guard = SWE_TEMP_DIR_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let previous = std::env::var("SWE_TEMP_DIR").ok();
        // SAFETY: the mailbox lock above means no other test reads or writes
        // this variable while it is overridden, and the pool's own tests are
        // the only consumers of the base dir in this binary.
        unsafe { std::env::set_var("SWE_TEMP_DIR", dir) };
        Self {
            _guard: guard,
            previous,
        }
    }
}

impl Drop for ScopedTempDir {
    fn drop(&mut self) {
        match self.previous.take() {
            Some(v) => unsafe { std::env::set_var("SWE_TEMP_DIR", v) },
            None => unsafe { std::env::remove_var("SWE_TEMP_DIR") },
        }
    }
}

#[test]
fn steer_mailbox_uses_the_documented_path_and_json_lines() {
    let dir = scratch_dir("steer-path");
    let _scope = ScopedTempDir::set(&dir);

    // The path is a sibling of the worktree, so one `ls` shows every worker
    // and `prune` reclaims both together.
    let path = mini_swe_mcp::pool::steer_path("abc123");
    assert_eq!(path, std::path::PathBuf::from(&dir).join("swe-wt-abc123.steer"));

    mini_swe_mcp::pool::write_steer_message("abc123", "first").unwrap();
    mini_swe_mcp::pool::write_steer_message("abc123", "second").unwrap();

    // One JSON object per line, each carrying the sender pid: the format is
    // self-delimiting, so a multi-line message cannot corrupt its neighbours.
    let raw = std::fs::read_to_string(&path).unwrap();
    let lines: Vec<&str> = raw.lines().filter(|l| !l.trim().is_empty()).collect();
    assert_eq!(lines.len(), 2, "expected one line per message, got: {raw}");
    for line in &lines {
        let v: serde_json::Value = serde_json::from_str(line).unwrap();
        assert!(v["message"].is_string());
        assert_eq!(v["pid"], json!(std::process::id() as u32));
    }

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn steer_mailbox_drain_returns_messages_in_order_and_empties_the_file() {
    let dir = scratch_dir("steer-drain");
    let _scope = ScopedTempDir::set(&dir);

    mini_swe_mcp::pool::write_steer_message("d1", "one").unwrap();
    mini_swe_mcp::pool::write_steer_message("d1", "two").unwrap();
    mini_swe_mcp::pool::write_steer_message("d1", "three").unwrap();

    assert_eq!(
        mini_swe_mcp::pool::drain_steer_messages("d1"),
        vec!["one", "two", "three"],
        "arrival order must be preserved"
    );
    // Draining twice must not re-deliver: guidance is consumed exactly once.
    assert!(mini_swe_mcp::pool::drain_steer_messages("d1").is_empty());
    assert!(!mini_swe_mcp::pool::steer_path("d1").exists());

    // A worker nobody ever steered drains empty rather than erroring.
    assert!(mini_swe_mcp::pool::drain_steer_messages("never-steered").is_empty());

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn steer_mailbox_preserves_multiline_and_unicode_payloads() {
    let dir = scratch_dir("steer-multiline");
    let _scope = ScopedTempDir::set(&dir);

    // The real reason the format is JSON lines rather than raw text: a pasted
    // stack trace or a diff hunk contains newlines, and a line-oriented plain
    // format would split one message into several truncated ones.
    let patch = "diff --git a/x b/x\n-old\n+new";
    let unicode = "corrige la lógica de parseo — ñandú ☕ 日本語";
    mini_swe_mcp::pool::write_steer_message("m1", patch).unwrap();
    mini_swe_mcp::pool::write_steer_message("m1", unicode).unwrap();

    assert_eq!(
        mini_swe_mcp::pool::drain_steer_messages("m1"),
        vec![patch.to_string(), unicode.to_string()]
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn steer_mailbox_survives_a_corrupt_line_without_stranding_the_worker() {
    let dir = scratch_dir("steer-corrupt");
    let _scope = ScopedTempDir::set(&dir);

    mini_swe_mcp::pool::write_steer_message("c1", "good one").unwrap();
    // A truncated write, or a hand-edited file, leaves an unparsable line.
    std::fs::OpenOptions::new()
        .append(true)
        .open(mini_swe_mcp::pool::steer_path("c1"))
        .unwrap()
        .write_all(b"{not json at all\n")
        .unwrap();
    mini_swe_mcp::pool::write_steer_message("c1", "good two").unwrap();

    // The bad line is skipped; the guidance either side of it still arrives,
    // because a single corrupt record must not wedge the worker's turn loop.
    assert_eq!(
        mini_swe_mcp::pool::drain_steer_messages("c1"),
        vec!["good one", "good two"]
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn steer_mailbox_drain_claims_the_file_so_two_readers_never_both_win() {
    let dir = scratch_dir("steer-claim");
    let _scope = ScopedTempDir::set(&dir);

    mini_swe_mcp::pool::write_steer_message("r1", "deliver me once").unwrap();

    // The drain renames before it reads, so the implementer loop and the review
    // loop can never deliver the same message twice.
    let first = mini_swe_mcp::pool::drain_steer_messages("r1");
    let second = mini_swe_mcp::pool::drain_steer_messages("r1");
    assert_eq!(first, vec!["deliver me once"]);
    assert!(
        second.is_empty(),
        "a second drain must not re-deliver an already-consumed message"
    );

    // The claim file is removed by the drain, so no scratch is left behind.
    let leftovers: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.contains("r1"))
        .collect();
    assert!(leftovers.is_empty(), "claim file leaked: {leftovers:?}");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn steer_mailbox_late_arrival_after_a_drain_is_picked_up_next_time() {
    let dir = scratch_dir("steer-late");
    let _scope = ScopedTempDir::set(&dir);

    mini_swe_mcp::pool::write_steer_message("l1", "early").unwrap();
    assert_eq!(mini_swe_mcp::pool::drain_steer_messages("l1"), vec!["early"]);

    // A steer that lands *after* the rename recreated the original path; it
    // must wait for the next drain rather than being lost in the claim.
    mini_swe_mcp::pool::write_steer_message("l1", "late").unwrap();
    assert_eq!(mini_swe_mcp::pool::drain_steer_messages("l1"), vec!["late"]);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn removing_the_steer_file_clears_the_mailbox_and_its_claim() {
    let dir = scratch_dir("steer-remove");
    let _scope = ScopedTempDir::set(&dir);

    mini_swe_mcp::pool::write_steer_message("x1", "guidance").unwrap();
    assert!(mini_swe_mcp::pool::steer_path("x1").is_file());

    // Worker exit: a finished worker must not leave a mailbox that a future
    // worker reusing the id would pick up as phantom guidance.
    mini_swe_mcp::pool::remove_steer_file("x1");
    assert!(!mini_swe_mcp::pool::steer_path("x1").exists());
    assert!(
        mini_swe_mcp::pool::drain_steer_messages("x1").is_empty(),
        "no guidance may survive the worker's exit"
    );

    // Removing a mailbox that was never created is a no-op, not an error.
    mini_swe_mcp::pool::remove_steer_file("never-existed");

    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn the_step_loop_sees_both_local_and_cross_process_guidance() {
    // The step loop merges two sources into one injection: `pending_steer` for
    // a `steer` this process handled, and the mailbox for one it did not. Both
    // must reach the turn -- dropping either would make cross-process steering
    // unreliable exactly when the local path is also in use.
    let dir = scratch_dir("steer-merge");
    let _scope = ScopedTempDir::set(&dir);

    let pool = WorkerPool::new(1, "http://x".into(), "k".into());
    pool.__test_insert_worker(running_worker("m1")).await;

    // Local guidance (this process) and remote guidance (another process).
    pool.steer("m1", "from this process".into()).await.unwrap();
    mini_swe_mcp::pool::write_steer_message("m1", "from another process").unwrap();

    // `take_pending_steer` is the loop's read of the in-memory half.
    let mut seen = pool.take_pending_steer("m1").await;
    seen.extend(mini_swe_mcp::pool::drain_steer_messages("m1"));

    assert!(seen.contains(&"from this process".to_string()));
    assert!(seen.contains(&"from another process".to_string()));
    assert!(seen.len() == 2, "each message must be delivered once: {seen:?}");

    // A second turn finds nothing left to inject -- guidance is consumed, not
    // replayed on every subsequent turn.
    let mut again = pool.take_pending_steer("m1").await;
    again.extend(mini_swe_mcp::pool::drain_steer_messages("m1"));
    assert!(again.is_empty(), "guidance was re-delivered: {again:?}");

    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn draining_the_mailbox_for_an_unknown_worker_is_a_no_op() {
    // The loop drains on every turn of a worker that may never have been
    // steered; that must be silent, and must not create a mailbox either.
    let dir = scratch_dir("steer-drain-unknown");
    let _scope = ScopedTempDir::set(&dir);

    assert!(mini_swe_mcp::pool::drain_steer_messages("ghost").is_empty());
    assert!(!mini_swe_mcp::pool::steer_path("ghost").exists());

    let _ = std::fs::remove_dir_all(&dir);
}

// ----------
// Steer / lock-discipline (audit opt_02_pool_locks, H-1)
// ----------

fn running_worker(id: &str) -> WorkerRecord {
    WorkerRecord {
        id: id.to_string(),
        task: "t".into(),
        model: "m".into(),
        owner: TEST_OWNER.into(),
        state: WorkerState::Running {
            step: 1,
            last_command: "ls".into(),
            started_at: 0,
        },
        metrics: WorkerMetrics::default(),
        logs: LogBuffer::new(),
        pending_steer: Vec::new(),
        resume_tx: None,
        handle: None,
        revision: 0,
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
async fn steer_on_unknown_worker_falls_back_to_the_cross_process_mailbox() {
    // The old contract was `Err("Worker not found")`. It is now the *normal*
    // cross-process path: an id this pool does not own is a worker running in
    // some other `mini-swe-mcp` process, and the guidance is queued in its
    // on-disk mailbox instead of being refused.
    let dir = scratch_dir("steer-unknown-mailbox");
    let pool = WorkerPool::new(1, "http://x".into(), "k".into());
    let err_guard = ScopedTempDir::set(&dir);

    pool.steer("nope", "focus on the parser".into()).await.unwrap();

    let path = mini_swe_mcp::pool::steer_path("nope");
    assert!(path.is_file(), "the message must be queued, not dropped");
    let raw = std::fs::read_to_string(&path).unwrap();
    assert!(raw.contains("focus on the parser"));

    // ...and a worker in *this* process still takes the fast in-memory path,
    // leaving no mailbox behind.
    pool.__test_insert_worker(running_worker("local")).await;
    pool.steer("local", "in memory".into()).await.unwrap();
    assert!(!mini_swe_mcp::pool::steer_path("local").exists());

    drop(err_guard);
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn steer_on_a_finished_worker_without_history_names_the_missing_file() {
    // A finished worker with no saved conversation cannot be revised: the
    // error names the missing history file, not just the worker.
    let dir = scratch_dir("steer-finished-no-history");
    let _scope = ScopedTempDir::set(&dir);
    let pool = WorkerPool::new(1, "http://x".into(), "k".into());

    let mut done = running_worker("w4");
    done.state = WorkerState::Completed {
        turns: 1,
        diff: String::new(),
        summary: String::new(),
        completed_at: 0,
        artifacts: vec![],
        branch: None,
        verified: None,
        metrics: WorkerMetrics::default(),
        revision: 0,
    };
    pool.__test_insert_worker(done).await;
    let err = pool.steer("w4", "x".into()).await.unwrap_err();
    assert!(
        err.to_string().contains("history") || err.to_string().contains("conversation"),
        "the error must name the missing history file, got: {err}"
    );
    let _ = std::fs::remove_dir_all(&dir);
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
        verified: None,
        metrics: WorkerMetrics::default(),
        revision: 0,
    };
    pool.__test_insert_worker(w).await;

    let p = pool.worker_progress("p2").await.unwrap();
    assert_eq!(p.phase, WorkerPhase::Completed);
    assert_eq!(p.step, 7);
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
}

#[tokio::test]
async fn worker_progress_reports_failed_workers() {
    let pool = WorkerPool::new(1, "http://x".into(), "k".into());
    let mut w = running_worker("p4");
    w.state = WorkerState::Failed {
        error: "boom".into(),
        step: 3,
        failed_at: 0,
        metrics: WorkerMetrics::default(),
        revision: 0,
    };
    pool.__test_insert_worker(w).await;

    let p = pool.worker_progress("p4").await.unwrap();
    assert_eq!(p.phase, WorkerPhase::Failed);
    assert_eq!(p.step, 3);
}

#[tokio::test]
async fn collect_evicts_and_returns_the_full_record() {
    let pool = WorkerPool::new(1, "http://x".into(), "k".into());
    pool.__test_insert_worker(running_worker("c1")).await;

    let collected = pool.collect("c1").await.unwrap();
    assert_eq!(collected.id, "c1");
    assert!(pool.get_worker_state("c1").await.is_none());
    assert!(pool.collect("c1").await.is_none());
}

// ----------
// Step-log retention bounds (audit 07, R1/R2/R4/F6)
// ----------

fn entry(step: usize) -> AgentStepLog {
    build_step_log(step, "cargo test", "ok".to_string(), Some(0))
}

#[test]
fn test_log_buffer_window_is_bounded_by_entry_count() {
    let mut buf = LogBuffer::with_policy(LogRetentionPolicy {
        max_retained: 8,
        max_emitted: 4,
    });
    for step in 0..500 {
        buf.push(entry(step));
    }
    assert_eq!(buf.len(), 8, "the window must never grow past the cap");
    assert_eq!(buf.retained(), 8);
    assert_eq!(buf.dropped(), 492);
    assert_eq!(buf.total(), 500);
}

#[test]
fn test_log_buffer_memory_is_bounded_per_worker() {
    // The audit's failure scenario: 150-turn workers at ~2.1 KiB per entry.
    // With the window in place the resident payload is a hard constant.
    let mut buf = LogBuffer::new();
    for step in 0..150 {
        buf.push(build_step_log(
            step,
            "cargo build --release",
            "x".repeat(64 * 1024),
            Some(0),
        ));
    }
    assert!(buf.len() <= DEFAULT_MAX_RETAINED_LOGS);
    assert!(
        buf.iter()
            .all(|e| e.output.len() <= MAX_LOG_OUTPUT_BYTES && e.command.len() <= MAX_LOG_COMMAND_BYTES),
        "every retained entry must be clamped, which bounds the window's payload"
    );
}

#[test]
fn test_log_buffer_is_preallocated_to_the_window() {
    // R2: reserving the window removes the GeomGrow over-allocation. An
    // *unbounded* buffer would grow 100 -> 128 -> 256 -> 512 ... leaving up to
    // 41% of the backing store unused; the window is sized once, up front.
    let mut buf = LogBuffer::with_policy(LogRetentionPolicy {
        max_retained: 200,
        max_emitted: 8,
    });
    for step in 0..200 {
        buf.push(entry(step));
    }
    assert!(
        buf.len() <= 200,
        "the window can never exceed its reservation, got {}",
        buf.len()
    );
    assert_eq!(serde_json::to_value(LogBuffer::new()).unwrap(), json!([]));
}

#[test]
fn test_build_step_log_enforces_a_hard_output_ceiling() {
    // F6: the 2048-byte cap used to be a *floor* (marker was added on top).
    let log = build_step_log(1, "cargo test", "y".repeat(16_384), Some(0));
    assert!(
        log.output.len() <= MAX_LOG_OUTPUT_BYTES,
        "output was {} bytes, over the {MAX_LOG_OUTPUT_BYTES}-byte ceiling",
        log.output.len()
    );
    assert!(log.output.contains("bytes truncated"));
}

#[test]
fn test_build_step_log_enforces_a_hard_command_ceiling() {
    let log = build_step_log(1, &"c".repeat(4096), "short".to_string(), None);
    assert!(log.command.len() <= MAX_LOG_COMMAND_BYTES);
    assert_eq!(
        log.output, "short",
        "short output must pass through untouched"
    );
}

#[test]
fn test_clamp_string_respects_multi_byte_boundaries() {
    // A 3-byte code point straddling the cut must be dropped, not split.
    let mut s = "a".repeat(MAX_LOG_OUTPUT_BYTES - 1);
    s.push('€');
    s.push_str(&"b".repeat(64));
    let clamped = clamp_string(&s, MAX_LOG_OUTPUT_BYTES);
    assert!(clamped.len() <= MAX_LOG_OUTPUT_BYTES);
    // Round-tripping through JSON proves no partial code point survived.
    let json = serde_json::to_string(&clamped).unwrap();
    let back: String = serde_json::from_str(&json).unwrap();
    assert_eq!(back, clamped);
}

#[test]
fn test_emit_view_caps_a_single_response() {
    // R4: a response must never serialize the whole window.
    let mut buf = LogBuffer::new();
    for step in 0..300 {
        buf.push(entry(step));
    }
    let view = emit_view(&buf, DEFAULT_MAX_EMITTED_LOGS);
    assert_eq!(view.logs.len(), DEFAULT_MAX_EMITTED_LOGS);
    assert_eq!(view.logs_omitted, buf.retained() - DEFAULT_MAX_EMITTED_LOGS);
    assert!(
        view.logs_truncation_notice.is_some(),
        "an orchestrator must be able to tell the history is degraded"
    );
}

#[test]
fn test_emit_view_of_a_short_worker_is_complete_and_silent() {
    let mut buf = LogBuffer::new();
    for step in 0..3 {
        buf.push(entry(step));
    }
    let view = emit_view(&buf, DEFAULT_MAX_EMITTED_LOGS);
    assert_eq!(view.logs.len(), 3);
    assert_eq!(view.logs_omitted, 0);
    assert!(view.logs_truncation_notice.is_none());
}

#[test]
fn test_emit_view_serialization_shape() {
    let mut buf = LogBuffer::new();
    buf.push(entry(1));
    let value = serde_json::to_value(emit_view(&buf, DEFAULT_MAX_EMITTED_LOGS)).unwrap();
    assert_eq!(value["logs"].as_array().unwrap().len(), 1);
    assert_eq!(value["logs_omitted"], json!(0));
    assert!(
        value.get("logs_truncation_notice").is_none(),
        "a complete history must not carry a notice"
    );
}

#[test]
fn test_retention_policy_ceilings_are_enforced_in_code() {
    // The documented ceilings must exist as public constants so a caller cannot
    // accidentally configure an unbounded window.
    assert_eq!(MAX_RETAINED_LOGS_CEILING, 1000);
    assert_eq!(MAX_EMITTED_LOGS_CEILING, 500);
    // Compiled as const assertions: the defaults can never drift past the caps.
    const { assert!(DEFAULT_MAX_RETAINED_LOGS <= MAX_RETAINED_LOGS_CEILING) };
    const { assert!(DEFAULT_MAX_EMITTED_LOGS <= MAX_EMITTED_LOGS_CEILING) };
    const { assert!(MAX_LOG_OUTPUT_BYTES > 0 && MAX_LOG_COMMAND_BYTES > 0) };
}

/// End-to-end guard for the audit's failure scenario: a worker that runs many
/// turns of high-output commands must retain a *constant* amount of memory and
/// report every eviction, instead of growing an unbounded `Vec<AgentStepLog>`
/// (audit 07, §5).
#[test]
fn test_high_turn_high_output_worker_stays_bounded() {
    const TURNS: usize = 600;
    const MAX_RETAINED: usize = 20;
    const MAX_EMITTED: usize = 5;

    let mut buf = LogBuffer::with_policy(LogRetentionPolicy {
        max_retained: MAX_RETAINED,
        max_emitted: MAX_EMITTED,
    });

    // Each iteration reproduces the real worst case: a `cargo test`-sized
    // 16 KiB payload entering a ~2 KiB log entry.
    for step in 0..TURNS {
        let raw = format!("step {step}\n{}", "x".repeat(16 * 1024));
        buf.push(build_step_log(step, "cargo test --release", raw, Some(0)));
    }

    // Memory: a hard constant, identical to a 50-turn worker.
    assert_eq!(buf.retained(), MAX_RETAINED);

    // Bookkeeping: nothing is silently lost.
    assert_eq!(buf.total(), TURNS);
    assert_eq!(buf.dropped(), TURNS - buf.retained());

    // Per-entry ceiling (F6): "2048 bytes" is now a ceiling, not a floor.
    for entry in buf.iter() {
        assert!(entry.output.len() <= MAX_LOG_OUTPUT_BYTES);
        assert!(entry.command.len() <= MAX_LOG_COMMAND_BYTES);
    }

    // Per-response ceiling (R4): the response is bounded independently.
    let view = emit_view(&buf, MAX_EMITTED);
    assert_eq!(view.logs.len(), MAX_EMITTED);
    assert_eq!(view.logs_omitted, buf.retained() - MAX_EMITTED);
    let notice = view
        .logs_truncation_notice
        .as_deref()
        .expect("an orchestrator must see that the history is degraded");
    assert!(notice.contains("evicted"), "got {notice}");
    assert!(notice.contains("omitted"), "got {notice}");

    // Contrast with the pre-fix shape: the old buffer would have held all 600.
    assert!(
        buf.total() > buf.retained(),
        "the window must be a strict subset of the history"
    );
}

#[test]
fn the_exit_guard_contract_clears_the_mailbox_on_every_worker_exit_path() {
    // Requirement 3 of the feature: the worker removes its mailbox on exit.
    // The loop returns from many places (completion, bash failure, cancellation,
    // a propagated error) and a `remove_steer_file` call in each is exactly the
    // duplication that rots, so cleanup is a `Drop` guard. This pins the
    // contract that guard depends on: a mailbox left behind is always reclaimed,
    // whether the worker succeeded or failed, and never leaks a claim file.
    let dir = scratch_dir("steer-exit-guard");
    let _scope = ScopedTempDir::set(&dir);

    // A worker that ran, got steered from another process, and finished.
    mini_swe_mcp::pool::write_steer_message("g1", "guidance").unwrap();
    assert!(mini_swe_mcp::pool::steer_path("g1").is_file());
    mini_swe_mcp::pool::remove_steer_file("g1");
    assert!(!mini_swe_mcp::pool::steer_path("g1").exists());

    // Same for a worker that dies mid-run: the guard fires on the error path
    // too, so a crashed worker's guidance cannot be inherited later.
    mini_swe_mcp::pool::write_steer_message("g2", "guidance").unwrap();
    mini_swe_mcp::pool::remove_steer_file("g2");
    assert!(!mini_swe_mcp::pool::steer_path("g2").exists());

    // Cleanup is idempotent: an already-removed mailbox must not turn a
    // successful worker exit into an error (the guard's Drop ignores errors).
    mini_swe_mcp::pool::remove_steer_file("g2");
    mini_swe_mcp::pool::remove_steer_file("never-existed");

    // Nothing at all is left in the scratch base afterwards.
    let leftovers: Vec<String> = std::fs::read_dir(&dir)
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    assert!(leftovers.is_empty(), "worker exit leaked files: {leftovers:?}");

    let _ = std::fs::remove_dir_all(&dir);
}


// ----------
// Per-worker health metrics (selfimprove-I7)
// ----------

/// A registry row with every counter moved, as a finished run would write it.
fn measured_entry() -> WorkerRegistryEntry {
    WorkerRegistryEntry {
        id: "m1".into(),
        pid: 42,
        task: "t".into(),
        model: "ninja".into(),
        status: RegistryStatus::Completed,
        step: 142,
        max_turns: 150,
        last_command: "completed".into(),
        question: None,
        started_at: 1_700_000_000,
        updated_at: 1_700_000_100,
        group: Some("g".into()),
        repo_path: Some("/tmp/repo".into()),
        owner: Some(TEST_OWNER.into()),
        metrics: WorkerMetrics {
            turns_used: 142,
            extensions_granted: 4,
            extensions_refused: 2,
            repeat_blocks: 3,
            stagnation_nudges: 1,
            loop_pauses: 1,
            verify_runs: 2,
            verify_failures: 1,
            diff_files: 5,
            diff_insertions: 120,
            diff_deletions: 340,
        },
    }
}

#[test]
fn test_worker_metrics_survive_a_registry_row_round_trip() {
    let entry = measured_entry();
    let json = serde_json::to_string(&entry).expect("registry entry serializes");
    let back: WorkerRegistryEntry = serde_json::from_str(&json).expect("registry entry parses");
    assert_eq!(back.metrics, entry.metrics);
    assert_eq!(back.metrics.turns_used, 142);
    assert_eq!(back.metrics.diff_insertions, 120);
}

#[test]
fn test_a_registry_row_written_before_the_metrics_still_parses() {
    // The shape a build without counters wrote: no `metrics`, and none of the
    // optional fields either.
    let legacy = r#"{
        "id": "old1",
        "pid": 7,
        "task": "t",
        "model": "ninja",
        "status": "completed",
        "step": 12,
        "max_turns": 20,
        "last_command": "completed",
        "started_at": 1,
        "updated_at": 2
    }"#;
    let entry: WorkerRegistryEntry =
        serde_json::from_str(legacy).expect("a row without metrics must still parse");
    assert_eq!(entry.metrics, WorkerMetrics::default());
    assert!(!entry.metrics.is_recorded());
    assert!(!WorkerMetrics::default().is_recorded());
}

#[test]
fn test_a_partially_recorded_metrics_object_fills_the_rest_with_zero() {
    let partial: WorkerMetrics =
        serde_json::from_str(r#"{"repeat_blocks":3,"stagnation_nudges":1}"#).expect("parses");
    assert_eq!(partial.repeat_blocks, 3);
    assert_eq!(partial.stagnation_nudges, 1);
    assert_eq!(partial.turns_used, 0);
    assert_eq!(partial.verify_runs, 0);
    assert!(partial.is_recorded());
}

#[test]
fn test_a_completed_state_serializes_its_health_counters() {
    let state = WorkerState::Completed {
        turns: 3,
        diff: String::new(),
        summary: "s".into(),
        completed_at: 1,
        artifacts: Vec::new(),
        branch: None,
        verified: Some(true),
        metrics: measured_entry().metrics,
        revision: 0,
    };
    let json = serde_json::to_value(&state).expect("state serializes");
    assert_eq!(json["state"], "Completed");
    assert_eq!(json["details"]["metrics"]["turns_used"], 142);
    assert_eq!(json["details"]["metrics"]["verify_runs"], 2);
    assert_eq!(json["details"]["metrics"]["loop_pauses"], 1);
}

#[tokio::test]
async fn test_a_killed_worker_reports_what_the_run_had_measured() {
    // The record caches the phase loop's counters, so a kill racing the loop
    // still reports the measurements the run had made.
    let pool = WorkerPool::new(1, "http://x".into(), "k".into());
    let mut record = running_worker("f1");
    record.metrics.repeat_blocks = 2;
    record.metrics.turns_used = 9;
    pool.__test_insert_worker(record).await;

    assert!(pool.kill("f1").await);
    let Some(WorkerState::Failed { metrics, .. }) = pool.get_worker_state("f1").await else {
        panic!("a killed worker must leave a Failed state");
    };
    assert_eq!(metrics.repeat_blocks, 2);
    assert_eq!(metrics.turns_used, 9);
}

// ----------
// Revision loop (hub-H8): history persistence and steer-after-finish
// ----------

/// Build a minimal saved history for `id`: a system prompt, a task and one
/// completed exchange, plus the relaunch facts a revision needs.
fn sample_history(repo_path: &std::path::Path, base_commit: &str, branch: &str) -> mini_swe_mcp::pool::WorkerHistory {
    use mini_swe_mcp::agent::{ChatMessage, Role};
    mini_swe_mcp::pool::WorkerHistory {
        task: "fix the parser".to_string(),
        group: Some("backend".to_string()),
        model: "test-model".to_string(),
        temperature: None,
        repo_path: repo_path.to_string_lossy().to_string(),
        base_commit: base_commit.to_string(),
        branch: branch.to_string(),
        network_offline: false,
        verify: None,
        max_turns: 10,
        review_after: None,
        revision: 0,
        owner: Some("test-owner".to_string()),
        messages: vec![
            ChatMessage::text(Role::System, "system prompt"),
            ChatMessage::text(Role::User, "TASK:\nfix the parser"),
            ChatMessage::assistant_with_tool_calls(
                Some("running ls".to_string()),
                vec![mini_swe_mcp::agent::ToolCall {
                    id: "call_1".to_string(),
                    r#type: "function".to_string(),
                    function: mini_swe_mcp::agent::ToolCallFn {
                        name: "bash".to_string(),
                        arguments: "{\"command\":\"ls\"}".to_string(),
                    },
                }],
            ),
            ChatMessage::tool_result("call_1".to_string(), "file list"),
        ],
    }
}

/// A scratch git repository with one commit, for revision tests that need a
/// real branch to re-attach to.
fn scratch_repo(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "swe-rev-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create scratch repo");
    let run = |args: &[&str]| {
        let out = std::process::Command::new("git")
            .current_dir(&dir)
            .args(args)
            .output()
            .unwrap_or_else(|e| panic!("git {args:?}: {e}"));
        assert!(out.status.success(), "git {args:?}: {}", String::from_utf8_lossy(&out.stderr));
    };
    run(&["init", "-b", "master"]);
    run(&["config", "user.name", "mini-swe-test"]);
    run(&["config", "user.email", "test@localhost"]);
    std::fs::write(dir.join("README.md"), "# scratch\n").expect("seed file");
    run(&["add", "README.md"]);
    run(&["commit", "-m", "baseline"]);
    dir
}

#[test]
fn history_file_round_trips_and_rejects_an_unreplayable_conversation() {
    let dir = scratch_dir("history-roundtrip");
    let _scope = ScopedTempDir::set(&dir);

    let repo = scratch_repo("history-roundtrip");
    let history = sample_history(&repo, "abc123", "worker-rev1");
    mini_swe_mcp::pool::save_worker_history("rev1", &history).expect("save history");

    // Atomic write, owner-only permissions, beside the mailbox.
    let path = mini_swe_mcp::pool::history_path("rev1");
    assert!(path.is_file(), "history file must exist at {path:?}");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(&path).expect("stat history").permissions().mode() & 0o777;
        assert_eq!(mode, 0o600, "history file must be owner-only, got {mode:o}");
    }

    let loaded = mini_swe_mcp::pool::load_worker_history("rev1").expect("reload history");
    assert_eq!(loaded.task, "fix the parser");
    assert_eq!(loaded.branch, "worker-rev1");
    assert_eq!(loaded.messages.len(), 4);
    assert!(mini_swe_mcp::pool::is_replayable(&loaded.messages));

    // A conversation without the system prompt is not replayable.
    let mut broken = loaded.messages.clone();
    broken.remove(0);
    assert!(!mini_swe_mcp::pool::is_replayable(&broken));
    assert!(!mini_swe_mcp::pool::is_replayable(&[]));

    mini_swe_mcp::pool::remove_worker_history("rev1");
    assert!(!path.exists(), "removal must delete the history file");
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&repo);
}

/// `prune` retires the saved conversation of a worker whose branch is gone
/// (merged and deleted) and keeps one whose branch can still be revised.
#[test]
fn prune_retires_histories_whose_branch_is_gone() {
    let dir = scratch_dir("history-orphans");
    let _scope = ScopedTempDir::set(&dir);
    let repo = scratch_repo("history-orphans");
    let git = |args: &[&str]| {
        let out = std::process::Command::new("git")
            .current_dir(&repo)
            .args(args)
            .output()
            .expect("run git");
        assert!(out.status.success(), "git {args:?}");
    };
    git(&["branch", "worker-alive"]);
    mini_swe_mcp::pool::save_worker_history("alive", &sample_history(&repo, "abc", "worker-alive"))
        .expect("save the revisable history");
    mini_swe_mcp::pool::save_worker_history("merged", &sample_history(&repo, "abc", "worker-merged"))
        .expect("save the orphaned history");

    assert_eq!(mini_swe_mcp::pool::prune_orphan_histories(&repo), 1);
    assert!(mini_swe_mcp::pool::history_path("alive").is_file(), "a live branch keeps its history");
    assert!(!mini_swe_mcp::pool::history_path("merged").exists(), "a deleted branch loses it");

    mini_swe_mcp::pool::remove_worker_history("alive");
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&repo);
}

/// A finished worker the reaper dropped (no record, no registry row) is
/// still owned: its saved conversation names the agent that may revise it.
#[tokio::test]
async fn a_reaped_worker_is_owned_by_the_agent_its_history_names() {
    let dir = scratch_dir("history-owner");
    let _scope = ScopedTempDir::set(&dir);
    let repo = scratch_repo("history-owner");
    let mut history = sample_history(&repo, "abc", "worker-reaped");
    history.owner = Some("agent-x".to_string());
    mini_swe_mcp::pool::save_worker_history("reaped", &history).expect("save history");

    let pool = WorkerPool::new(1, "http://x".into(), "k".into());
    assert_eq!(
        pool.worker_owner("reaped").await,
        Some(mini_swe_mcp::pool::WorkerOwner::Agent("agent-x".to_string()))
    );

    mini_swe_mcp::pool::remove_worker_history("reaped");
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&repo);
}

#[tokio::test]
async fn steer_on_a_completed_worker_revises_on_the_same_branch() {
    // A finished worker steered with corrections restarts on its preserved
    // branch with the revision message appended to the reloaded history.
    let dir = scratch_dir("steer-revision");
    let _scope = ScopedTempDir::set(&dir);
    let repo = scratch_repo("steer-revision");

    // Create the preserved branch the finished run left behind.
    let branch = "worker-revwork";
    let run = |args: &[&str]| {
        let out = std::process::Command::new("git")
            .current_dir(&repo)
            .args(args)
            .output()
            .unwrap_or_else(|e| panic!("git {args:?}: {e}"));
        assert!(out.status.success(), "git {args:?}: {}", String::from_utf8_lossy(&out.stderr));
    };
    run(&["checkout", "-b", branch]);
    std::fs::write(repo.join("fix.txt"), "fix\n").expect("worker change");
    run(&["add", "fix.txt"]);
    run(&["commit", "-m", "worker(revwork): fix the parser"]);
    let rev_out = std::process::Command::new("git")
        .current_dir(&repo)
        .args(["rev-parse", "HEAD~1"])
        .output()
        .expect("rev-parse");
    let base = String::from_utf8_lossy(&rev_out.stdout).trim().to_string();
    run(&["checkout", "master"]);

    let pool = WorkerPool::new(1, "http://x".into(), "k".into());
    let mut done = running_worker("revwork");
    done.state = WorkerState::Completed {
        turns: 2,
        diff: "fix".to_string(),
        summary: "fixed the parser".to_string(),
        completed_at: 0,
        artifacts: Vec::new(),
        branch: Some(branch.to_string()),
        verified: None,
        metrics: WorkerMetrics::default(),
        revision: 0,
    };
    pool.__test_insert_worker(done).await;
    // The finished run's history file is what the revision reloads.
    let mut history = sample_history(&repo, &base, branch);
    history.branch = branch.to_string();
    mini_swe_mcp::pool::save_worker_history("revwork", &history).expect("save history");

    pool.steer("revwork", "also handle empty input".into())
        .await
        .expect("steer on a completed worker must start a revision, not error");

    // The record is Running again on the same id, with the revision counted.
    let progress = pool.worker_progress("revwork").await.expect("worker still tracked");
    assert_eq!(progress.phase, WorkerPhase::Running);
    // Two steers that both saw the worker finished race into `revise`; the
    // loser must be refused instead of launching a second loop on the branch.
    let err = pool
        .revise("revwork", "a racing correction".into(), None)
        .await
        .expect_err("a running revision must not be revised again");
    assert!(
        err.to_string().contains("already running (revision 1 in progress)"),
        "{err}"
    );

    // The revision reloaded the history and appended the request: kill the
    // worker (no LLM is running in this test) and inspect the file the next
    // revision would read -- i.e. the messages the loop started with. The
    // loop owns them now, so assert on the registry row + branch instead.
    let wt_path = mini_swe_mcp::worktree::swe_base_dir().join("swe-wt-revwork");
    // Give the spawned revision task a moment to check out the branch.
    // The directory appears before `git worktree add` writes its `.git` link,
    // so wait for the link: only then does the checkout name its branch.
    for _ in 0..200 {
        if wt_path.join(".git").exists() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    assert!(wt_path.join(".git").exists(), "the revision must re-create the worktree");
    let out = std::process::Command::new("git")
        .current_dir(&wt_path)
        .args(["rev-parse", "--abbrev-ref", "HEAD"])
        .output()
        .expect("rev-parse in worktree");
    assert_eq!(
        String::from_utf8_lossy(&out.stdout).trim(),
        branch,
        "the revision must resume on the same branch"
    );
    // The previous commit survived the re-attach.
    assert!(
        wt_path.join("fix.txt").is_file(),
        "the preserved branch's checkpoints must survive the revision"
    );

    pool.kill("revwork").await;
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&repo);
}

#[tokio::test]
async fn collect_keeps_the_history_so_a_collected_worker_stays_revisable() {
    // Collection evicts the record but the worker becomes registry-only, not
    // unrevisable: the history file must survive it (only prune retires
    // it), so a later steer can still revise the same id and branch.
    let dir = scratch_dir("collect-keeps-history");
    let _scope = ScopedTempDir::set(&dir);
    let repo = scratch_repo("collect-keeps-history");
    let pool = WorkerPool::new(1, "http://x".into(), "k".into());
    let mut done = running_worker("keep1");
    done.state = WorkerState::Completed {
        turns: 1,
        diff: String::new(),
        summary: "done".to_string(),
        completed_at: 0,
        artifacts: Vec::new(),
        branch: Some("worker-keep1".to_string()),
        verified: None,
        metrics: WorkerMetrics::default(),
        revision: 0,
    };
    pool.__test_insert_worker(done).await;
    let history = sample_history(&repo, "abc123", "worker-keep1");
    mini_swe_mcp::pool::save_worker_history("keep1", &history).expect("save history");
    pool.collect("keep1").await.expect("collect the worker");
    assert!(
        mini_swe_mcp::pool::history_path("keep1").is_file(),
        "collect must not delete the history file; only prune retires it"
    );
    mini_swe_mcp::pool::remove_worker_history("keep1");
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&repo);
}

#[tokio::test]
async fn steer_on_a_finished_worker_without_a_branch_is_a_clear_error() {
    // The branch the orchestrator reviewed is gone: the error names the
    // branch, not just the worker.
    let dir = scratch_dir("steer-missing-branch");
    let _scope = ScopedTempDir::set(&dir);
    let repo = scratch_repo("steer-missing-branch");

    let pool = WorkerPool::new(1, "http://x".into(), "k".into());
    let mut done = running_worker("gonework");
    done.state = WorkerState::Completed {
        turns: 1,
        diff: String::new(),
        summary: "done".to_string(),
        completed_at: 0,
        artifacts: Vec::new(),
        branch: Some("worker-gonework".to_string()),
        verified: None,
        metrics: WorkerMetrics::default(),
        revision: 0,
    };
    pool.__test_insert_worker(done).await;
    // No `worker-gonework` branch was ever created in the scratch repo.
    let history = sample_history(&repo, "abc123", "worker-gonework");
    mini_swe_mcp::pool::save_worker_history("gonework", &history).expect("save history");

    let err = pool.steer("gonework", "fix it".into()).await.unwrap_err();
    assert!(
        err.to_string().contains("worker-gonework"),
        "the error must name the missing branch, got: {err}"
    );
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&repo);
}

// ----------
// Hub H-5a: event-driven waits, coalesced registry writes, one HTTP client
// ----------

/// A synthetic state change wakes a subscribed waiter without any tick.
#[tokio::test]
async fn change_subscription_fires_on_every_state_change() {
    let pool = WorkerPool::new(1, "http://x".into(), "k".into());
    pool.__test_insert_worker(running_worker("h5a-sub")).await;
    let mut changes = pool.subscribe_changes();
    // The subscription starts at the current generation, so only the state
    // change below may resolve it.
    let changed = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        changes.changed(),
    );
    pool.__test_set_worker_state(
        "h5a-sub",
        WorkerState::Paused {
            question: "q?".into(),
            step: 1,
            paused_at: 0,
        },
    )
    .await;
    changed.await.expect("the waiter must observe the change").expect("watch open");
}

/// Step-only registry updates coalesce; a status transition writes at once.
#[tokio::test]
async fn step_only_registry_updates_coalesce_to_one_write() {
    let dir = scratch_dir("h5a-reg");
    let _scope = ScopedTempDir::set(&dir);
    let pool = WorkerPool::new(1, "http://x".into(), "k".into());
    let meta = mini_swe_mcp::pool::WorkerMeta {
        id: "h5a-reg".into(),
        task: "t".into(),
        group: None,
        repo_path: None,
        owner: TEST_OWNER.into(),
        started_at: 0,
        pid: std::process::id(),
        metrics: WorkerMetrics::default(),
    };
    let row_path =
        std::path::PathBuf::from(&dir).join("swe-registry").join("h5a-reg.json");
    let mtime = || std::fs::metadata(&row_path).ok().and_then(|m| m.modified().ok());

    // First write always lands (there is no row yet to coalesce with)...
    pool.__test_save_status(&meta, "m", RegistryStatus::Running, 1, 10, "ls", None);
    assert!(row_path.exists(), "the first registry row must be written");
    let first = mtime();
    // ...but N rapid step-only updates behind the throttle window do not.
    for step in 2..=10usize {
        pool.__test_save_status(&meta, "m", RegistryStatus::Running, step, 10, "ls", None);
    }
    assert_eq!(mtime(), first, "rapid step updates must coalesce to one file write");

    // Past the throttle window a step update lands again, so the final state
    // can never be stuck behind the throttle.
    pool.__test_reset_registry_throttle("h5a-reg");
    pool.__test_save_status(&meta, "m", RegistryStatus::Running, 11, 10, "ls", None);
    assert!(mtime() >= first, "a step update past the window must be written");
    let stepped = mtime();

    // A status transition is never throttled.
    pool.__test_save_status(&meta, "m", RegistryStatus::Paused, 11, 10, "ls", Some("q?".into()));
    let entry: WorkerRegistryEntry =
        serde_json::from_str(&std::fs::read_to_string(&row_path).expect("row readable"))
            .expect("row parses");
    assert_eq!(entry.status, RegistryStatus::Paused);
    assert_eq!(entry.question.as_deref(), Some("q?"));
    assert!(mtime() >= stepped, "a status transition must write immediately");

    let _ = std::fs::remove_dir_all(&dir);
}

/// Every runner reuses the single process-wide HTTP client.
#[test]
fn agent_runners_share_one_http_client() {
    use mini_swe_mcp::agent::AgentRunner;
    let _first = AgentRunner::new("http://x".into(), "k".into(), "m".into(), None);
    let builds = AgentRunner::__test_shared_client_builds();
    assert_eq!(builds, 1, "the first runner builds the one shared client");
    // Further runners — even against another base URL — must not build again:
    // they clone the shared client, so connections and TLS sessions are reused.
    let _second = AgentRunner::new("http://y".into(), "k".into(), "m".into(), None);
    let _third = AgentRunner::new("http://z".into(), "k".into(), "m".into(), None);
    assert_eq!(
        AgentRunner::__test_shared_client_builds(),
        builds,
        "runners must reuse one client so connections and TLS sessions are shared"
    );
}

// ----------
// Heavy-command admission controller
// ----------

/// A slot is granted while the host is idle, and the job count divides the
/// cores over the builds running after the grant.
#[tokio::test]
async fn admission_grants_with_job_count() {
    use mini_swe_mcp::pool::{AdmissionController, HostSample};
    let gate = AdmissionController::new(4, 4, 2048, 1536);
    gate.__test_set_host_sample(Some(HostSample {
        mem_available_mb: Some(12_000),
        load1: Some(1.0),
    }));
    let first = gate.acquire().await;
    assert_eq!(first.jobs(), 4, "the first build owns the machine");
    assert_eq!(gate.running_heavy(), 1);
    let second = gate.acquire().await;
    assert_eq!(second.jobs(), 2, "two builds split the cores");
    drop(first);
    drop(second);
    assert_eq!(gate.running_heavy(), 0);
}

/// Two queued waiters are granted in the order they arrived: when one slot
/// frees, the waiter queued first is the one admitted.
#[tokio::test]
async fn admission_waiters_are_granted_fifo() {
    use mini_swe_mcp::pool::{AdmissionController, HostSample};
    use std::sync::{Arc, Mutex};
    let gate = AdmissionController::new(1, 4, 2048, 1536);
    gate.__test_set_host_sample(Some(HostSample {
        mem_available_mb: Some(12_000),
        load1: Some(0.5),
    }));
    let held = gate.acquire().await;

    let order = Arc::new(Mutex::new(Vec::new()));
    let spawn_waiter = |tag: &'static str| {
        let gate = gate.clone();
        let order = order.clone();
        tokio::spawn(async move {
            let _permit = gate.acquire().await;
            order.lock().expect("order lock poisoned").push(tag);
        })
    };
    let first = spawn_waiter("first");
    // The first waiter must be queued before the second arrives.
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    let second = spawn_waiter("second");
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    assert_eq!(gate.waiting(), 2, "both waiters must be queued");

    drop(held);
    first.await.expect("first waiter completes");
    // The first waiter drains before the second is even eligible.
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    second.await.expect("second waiter completes");

    assert_eq!(
        *order.lock().expect("order lock poisoned"),
        vec!["first", "second"],
        "the waiter queued first must be granted first"
    );
}

/// A waiter dropped while queued (its worker killed) leaves the line: the
/// request behind it is admitted instead of waiting forever behind a ghost.
#[tokio::test]
async fn admission_a_cancelled_waiter_releases_its_place() {
    use mini_swe_mcp::pool::{AdmissionController, HostSample};
    let gate = AdmissionController::new(1, 4, 2048, 1536);
    gate.__test_set_host_sample(Some(HostSample {
        mem_available_mb: Some(12_000),
        load1: Some(0.5),
    }));
    let held = gate.acquire().await;

    let doomed = tokio::spawn({
        let gate = gate.clone();
        async move {
            let _permit = gate.acquire().await;
        }
    });
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    let survivor = tokio::spawn({
        let gate = gate.clone();
        async move { gate.acquire().await.jobs() }
    });
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    assert_eq!(gate.waiting(), 2);

    doomed.abort();
    let _ = doomed.await;
    assert_eq!(gate.waiting(), 1, "the aborted request must leave the queue");
    drop(held);
    let jobs = tokio::time::timeout(std::time::Duration::from_secs(5), survivor)
        .await
        .expect("the survivor must not wait behind the aborted request")
        .expect("survivor task completes");
    assert_eq!(jobs, 4);
}

/// Admitting the first build never waits, even when the host is saturated:
/// the progress guarantee keeps a busy machine from deadlocking.
#[tokio::test]
async fn admission_first_build_never_blocks() {
    use mini_swe_mcp::pool::{AdmissionController, HostSample};
    let gate = AdmissionController::new(2, 4, 2048, 1536);
    gate.__test_set_host_sample(Some(HostSample {
        mem_available_mb: Some(64),
        load1: Some(99.0),
    }));
    let permit = tokio::time::timeout(std::time::Duration::from_secs(5), gate.acquire())
        .await
        .expect("the first build must be admitted without waiting");
    assert_eq!(gate.running_heavy(), 1);
    drop(permit);
}

/// A second build waits while memory is short, then is admitted once memory
/// frees up.
#[tokio::test]
async fn admission_memory_shortage_waits_then_admits() {
    use mini_swe_mcp::pool::{AdmissionController, HostSample};
    let gate = AdmissionController::new(2, 4, 2048, 1536);
    gate.__test_set_host_sample(Some(HostSample {
        mem_available_mb: Some(12_000),
        load1: Some(0.5),
    }));
    let _held = gate.acquire().await;
    gate.__test_set_host_sample(Some(HostSample {
        mem_available_mb: Some(100),
        load1: Some(0.5),
    }));
    let waiter = tokio::spawn({
        let gate = gate.clone();
        async move { gate.acquire().await }
    });
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    assert_eq!(gate.waiting(), 1, "the second build must wait on memory");
    gate.__test_set_host_sample(Some(HostSample {
        mem_available_mb: Some(12_000),
        load1: Some(0.5),
    }));
    // The waiter re-evaluates every 2 s at the latest.
    let permit = tokio::time::timeout(std::time::Duration::from_secs(5), waiter)
        .await
        .expect("the waiter must be admitted once memory frees")
        .expect("waiter task completes");
    drop(permit);
}

// ----------
// Per-agent ownership (H-3)
// ----------

/// A synthetic record owned by `owner`, running.
fn owned_worker(id: &str, owner: &str) -> WorkerRecord {
    WorkerRecord {
        owner: owner.to_string(),
        ..running_worker(id)
    }
}

/// The in-memory view: each agent sees only its own workers, and `owner`
/// travels with every row.
#[tokio::test]
async fn listing_is_scoped_to_the_owning_agent() {
    let dir = scratch_dir("h3-list-inmemory");
    let _scope = ScopedTempDir::set(&dir);
    let pool = WorkerPool::new(4, "http://x".into(), "k".into());
    pool.__test_insert_worker(owned_worker("h3-mine", "agent-a")).await;
    pool.__test_insert_worker(owned_worker("h3-theirs", "agent-b")).await;

    let ids = |rows: Vec<serde_json::Value>| -> Vec<String> {
        rows.iter()
            .map(|row| row["id"].as_str().expect("id").to_string())
            .collect()
    };
    let all = ids(pool.list_workers().await);
    assert_eq!(all.len(), 2, "both workers: {all:?}");

    let mine = pool.list_workers_of("agent-a").await;
    assert_eq!(ids(mine.clone()), vec!["h3-mine".to_string()]);
    assert_eq!(mine[0]["owner"], "agent-a");
    let theirs = pool.list_workers_of("agent-b").await;
    assert_eq!(ids(theirs), vec!["h3-theirs".to_string()]);
    assert!(
        pool.list_workers_of("agent-c").await.is_empty(),
        "an agent with no workers sees none"
    );
}

/// The cross-process view: a worker of another connection lives in the
/// registry, and the same scoping applies to those rows.
#[tokio::test]
async fn listing_is_scoped_across_processes_through_the_registry() {
    let dir = scratch_dir("h3-list-registry");
    let _scope = ScopedTempDir::set(&dir);
    let mut row = measured_entry();
    row.id = "h3-reg".to_string();
    row.status = RegistryStatus::Running;
    row.owner = Some("agent-a".to_string());
    mini_swe_mcp::pool::save_registry_entry(&row);
    let pool = WorkerPool::new(4, "http://x".into(), "k".into());

    let mine = pool.list_workers_of("agent-a").await;
    assert_eq!(mine.len(), 1, "the owner's own row: {mine:?}");
    assert_eq!(mine[0]["id"], "h3-reg");
    assert_eq!(mine[0]["owner"], "agent-a");
    assert!(
        pool.list_workers_of("agent-b").await.is_empty(),
        "another agent must not see it in the default list"
    );
    let all = pool.list_workers().await;
    assert_eq!(all.len(), 1, "scope=all sees every row: {all:?}");
}

/// Ownership survives a restart: a worker this pool never dispatched is still
/// attributed to the agent that did, and a row written before ownership was
/// tracked is attributed to nobody.
#[tokio::test]
async fn worker_ownership_falls_back_to_the_registry_row() {
    let dir = scratch_dir("h3-owner-fallback");
    let _scope = ScopedTempDir::set(&dir);
    let pool = WorkerPool::new(4, "http://x".into(), "k".into());

    assert_eq!(pool.worker_owner("h3-nobody").await, None);

    let mut row = measured_entry();
    row.id = "h3-foreign".to_string();
    row.status = RegistryStatus::Running;
    row.owner = Some("agent-a".to_string());
    mini_swe_mcp::pool::save_registry_entry(&row);
    assert_eq!(
        pool.worker_owner("h3-foreign").await,
        Some(mini_swe_mcp::pool::WorkerOwner::Agent("agent-a".to_string()))
    );

    row.id = "h3-ancient".to_string();
    row.owner = None;
    mini_swe_mcp::pool::save_registry_entry(&row);
    assert_eq!(
        pool.worker_owner("h3-ancient").await,
        Some(mini_swe_mcp::pool::WorkerOwner::Unattributed),
        "a row with no owner belongs to nobody"
    );
    assert_eq!(
        mini_swe_mcp::pool::registry_owner_label(&row),
        mini_swe_mcp::pool::UNATTRIBUTED_OWNER
    );
}

/// A registry row written before `owner` existed still parses, so an upgrade
/// cannot turn an old worker's row into a hard read error.
#[test]
fn a_registry_row_without_an_owner_still_parses() {
    let row: WorkerRegistryEntry = serde_json::from_str(
        r#"{"id":"legacy","pid":1,"task":"t","model":"m","status":"running","step":1,
            "max_turns":5,"last_command":"ls","started_at":1,"updated_at":2}"#,
    )
    .expect("a pre-ownership row must parse");
    assert_eq!(row.owner, None);
    assert_eq!(row.status, RegistryStatus::Running);
}

/// The per-agent cap counts only that agent's *running* workers, which is what
/// makes it a fairness gate rather than a global limit.
#[tokio::test]
async fn the_per_agent_cap_counts_only_that_agents_running_workers() {
    let dir = scratch_dir("h3-cap-count");
    let _scope = ScopedTempDir::set(&dir);
    let pool = WorkerPool::new(8, "http://x".into(), "k".into());
    pool.__test_insert_worker(owned_worker("h3-a1", "agent-a")).await;
    pool.__test_insert_worker(owned_worker("h3-a2", "agent-a")).await;
    pool.__test_insert_worker(owned_worker("h3-b1", "agent-b")).await;
    // A finished worker of the same agent no longer occupies the cap.
    pool.__test_insert_worker(WorkerRecord {
        state: WorkerState::Completed {
            turns: 1,
            diff: String::new(),
            summary: String::new(),
            completed_at: 0,
            artifacts: Vec::new(),
            branch: None,
            verified: None,
            revision: 0,
            metrics: WorkerMetrics::default(),
        },
        ..owned_worker("h3-a3", "agent-a")
    })
    .await;

    assert_eq!(pool.active_workers_of("agent-a").await, vec!["h3-a1", "h3-a2"]);
    assert_eq!(pool.active_workers_of("agent-b").await, vec!["h3-b1"]);
    assert!(pool.active_workers_of("agent-c").await.is_empty());
}
