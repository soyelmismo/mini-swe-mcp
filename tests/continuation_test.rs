//! Continuing a stopped worker: the durable log, the CONTINUE prefix, cold
//! continuation, the steer reply and the transient-failure guarantee.
//!
//! Every test here pins one property of the principle that a worker which
//! stopped for *any* reason is continued with `steer` on the same id and
//! branch, and that `failed` only means "this cannot continue by itself".

use std::path::Path;

use mini_swe_mcp::agent::{ChatMessage, Role};
use mini_swe_mcp::pool::{
    RegistryStatus, WorkerHistory, WorkerPool, append_history_message, history_log_path,
    load_worker_history, save_registry_entry,
};

/// Serializes the tests that override `SWE_TEMP_DIR`: the variable is
/// process-global, so only one test may point it at its own scratch dir.
static SWE_TEMP_DIR_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// A scratch base dir with `SWE_TEMP_DIR` pointed at it for this test only.
struct Scratch {
    _guard: std::sync::MutexGuard<'static, ()>,
    dir: std::path::PathBuf,
}

impl Scratch {
    fn new(tag: &str) -> Self {
        let _guard = SWE_TEMP_DIR_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "swe-cont-{tag}-{}-{}-{n}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create scratch dir");
        // SAFETY: this test owns the variable for its whole lifetime and no
        // other test in this binary reads it concurrently.
        unsafe { std::env::set_var("SWE_TEMP_DIR", &dir) };
        Self { _guard, dir }
    }

    fn path(&self) -> &Path {
        &self.dir
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        unsafe { std::env::remove_var("SWE_TEMP_DIR") };
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// A minimal replayable conversation for `worker_id`.
fn history(worker_id: &str, repo: &Path) -> WorkerHistory {
    WorkerHistory {
        task: "fix the parser".to_string(),
        group: None,
        model: "test-model".to_string(),
        temperature: None,
        repo_path: repo.to_string_lossy().to_string(),
        base_commit: "base".to_string(),
        base_branch: Some("master".to_string()),
        branch: format!("worker-{worker_id}"),
        network_offline: false,
        verify: None,
        max_turns: 10,
        review_after: None,
        revision: 1,
        auto_continues: 0,
        owner: None,
        messages: vec![
            ChatMessage::text(Role::System, "system prompt"),
            ChatMessage::text(Role::User, "TASK:\nfix the parser"),
            ChatMessage::text(Role::Assistant, "I will look at the parser."),
            ChatMessage::text(Role::User, "ok"),
        ],
    }
}

/// A git repo with a `master` branch and a `worker-<id>` branch, as dispatch
/// leaves behind.
fn repo_with_branch(tag: &str, id: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "swe-cont-repo-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let git = |args: &[&str]| {
        let out = std::process::Command::new("git")
            .current_dir(&dir)
            .args(args)
            .output()
            .unwrap();
        assert!(out.status.success(), "git {args:?}: {}", String::from_utf8_lossy(&out.stderr));
    };
    git(&["init", "--initial-branch=master"]);
    git(&["config", "user.email", "t@t"]);
    git(&["config", "user.name", "t"]);
    std::fs::write(dir.join("a.txt"), "base\n").unwrap();
    git(&["add", "."]);
    git(&["commit", "-m", "base"]);
    git(&["checkout", "-b", &format!("worker-{id}")]);
    std::fs::write(dir.join("a.txt"), "work\n").unwrap();
    git(&["add", "."]);
    git(&["commit", "-m", "work"]);
    dir
}

/// A registry row for `id` in `status`, as a stopped run leaves behind.
fn row(id: &str, repo: &Path, status: RegistryStatus) -> mini_swe_mcp::pool::WorkerRegistryEntry {
    mini_swe_mcp::pool::WorkerRegistryEntry {
        id: id.to_string(),
        pid: std::process::id(),
        task: "fix the parser".to_string(),
        group: None,
        model: "test-model".to_string(),
        status,
        step: 4,
        max_turns: 10,
        last_command: "cargo test".into(),
        question: None,
        repo_path: Some(repo.to_string_lossy().to_string()),
        started_at: 0,
        updated_at: 0,
        owner: None,
        metrics: Default::default(),
        base_branch: Some("master".into()),
        base_commit: Some("base".into()),
        revision: 1,
        auto_continues: 0,
    }
}

#[test]
fn the_append_only_log_survives_a_torn_last_line() {
    let scratch = Scratch::new("torn");
    let repo = repo_with_branch("torn", "torn1");
    let meta = history("torn1", &repo);

    // One line per message, as the turn loop pushes them.
    for msg in &meta.messages {
        append_history_message("torn1", &meta, msg).expect("append");
    }

    // Simulate a crash mid-append: the last line is torn.
    let path = history_log_path("torn1");
    let raw = std::fs::read_to_string(&path).unwrap();
    let lines: Vec<&str> = raw.lines().collect();
    assert_eq!(lines.len(), 5, "metadata plus four messages");
    eprintln!("LINES={} PATH={}", lines.len(), path.display());
    let torn = format!("{}\n{}", lines[..4].join("\n"), &lines[4][..lines[4].len() / 2]);
    std::fs::write(&path, torn).unwrap();

    // The reload keeps everything before the torn line and never fails.
    let reloaded = load_worker_history("torn1").expect("a torn last line must not fail the reload");
    assert_eq!(
        reloaded.messages.len(),
        3,
        "the torn line is dropped, the messages before it are kept"
    );
    assert_eq!(reloaded.branch, "worker-torn1");
    assert_eq!(reloaded.task, "fix the parser");

    // A continuation can still be built from what survived.
    assert!(mini_swe_mcp::pool::is_replayable(&reloaded.messages));
    let _ = std::fs::remove_dir_all(&repo);
    drop(scratch);
}

#[test]
fn the_log_is_append_only_and_never_rewritten() {
    let scratch = Scratch::new("append");
    let repo = repo_with_branch("append", "app1");
    let meta = history("app1", &repo);
    let path = history_log_path("app1");

    append_history_message("app1", &meta, &meta.messages[0]).unwrap();
    let first = std::fs::read_to_string(&path).unwrap();
    append_history_message("app1", &meta, &meta.messages[1]).unwrap();
    let second = std::fs::read_to_string(&path).unwrap();

    assert!(
        second.starts_with(&first),
        "appending must not rewrite what is already there"
    );
    assert_eq!(second.lines().count(), first.lines().count() + 1);
    let _ = std::fs::remove_dir_all(&repo);
    drop(scratch);
}

#[test]
fn a_legacy_whole_file_history_is_still_read() {
    let scratch = Scratch::new("legacy");
    let repo = repo_with_branch("legacy", "leg1");
    let meta = history("leg1", &repo);
    let raw = serde_json::to_string(&meta).unwrap();
    std::fs::write(
        scratch.path().join("swe-wt-leg1.history.json"),
        raw.as_bytes(),
    )
    .unwrap();

    let loaded = load_worker_history("leg1").expect("the legacy whole-file form is still read");
    assert_eq!(loaded.messages.len(), 4);
    assert!(!history_log_path("leg1").exists(), "no log is created by a read");
    let _ = std::fs::remove_dir_all(&repo);
    drop(scratch);
}
