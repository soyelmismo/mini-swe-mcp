//! The orchestrator steer log is read on every round manifest build, so an
//! unbounded read of it turns a chatty (or hostile) log into an unbounded
//! amount of work and memory on a path that has to answer promptly.
//!
//! The log lives beside the worktrees under the shared scratch base (`/var/tmp`
//! by default), which any local user can write, so the reader must bound what
//! it will read rather than trusting the file to stay small.

mod common;

use common::{IsolatedPool, TempDir, git, unique_suffix};
use mini_swe_mcp::pool::{RegistryStatus, WorkerRegistryEntry, WorkerRole, save_registry_entry_in};

const OWNER: &str = "agent-a";
const GROUP: &str = "round-logbound";

/// Plant a steer log holding `count` records of `size` bytes each and return its
/// on-disk size.
fn plant_log(scratch: &TempDir, worker_id: &str, count: usize, size: usize) -> u64 {
    let path = scratch
        .path()
        .join(format!("swe-wt-{worker_id}.steer-log.jsonl"));
    let mut payload = String::new();
    for i in 0..count {
        payload.push_str(
            &serde_json::json!({
                "message": format!("steer {i}: {}", "s".repeat(size)),
                "sent_at": 1_u64,
                "pid": 1_u32,
            })
            .to_string(),
        );
        payload.push('\n');
    }
    std::fs::write(&path, &payload).unwrap();
    std::fs::metadata(&path).unwrap().len()
}

fn file_row(repo: &TempDir, pool: &IsolatedPool, worker_id: &str) {
    // A real branch carrying one commit, as a finished worker of the round has:
    // without it the row is not "ready" and never reaches the manifest.
    git(repo.path(), &["checkout", "-q", "master"]);
    git(
        repo.path(),
        &["checkout", "-q", "-b", &format!("worker-{worker_id}")],
    );
    std::fs::write(repo.path().join(format!("{worker_id}.txt")), "done\n").unwrap();
    git(repo.path(), &["add", "."]);
    git(
        repo.path(),
        &[
            "-c",
            "user.name=w",
            "-c",
            "user.email=w@x",
            "commit",
            "-m",
            "work",
        ],
    );
    git(repo.path(), &["checkout", "-q", "master"]);

    let entry = WorkerRegistryEntry {
        task: "t".into(),
        status: RegistryStatus::Completed,
        step: 1,
        last_command: "completed".into(),
        group: Some(GROUP.into()),
        role: WorkerRole::Worker,
        repo_path: Some(repo.path().to_string_lossy().to_string()),
        base_branch: Some("master".into()),
        ..WorkerRegistryEntry::test_row(worker_id, OWNER)
    };
    save_registry_entry_in(&pool.root(), &entry);
}

/// A log far larger than anything the prompt needs must not be read whole: the
/// manifest build reads at most a bounded prefix, so its work and memory stay
/// proportional to the budget rather than to the file.
#[tokio::test]
async fn an_oversized_steer_log_is_not_read_whole() {
    let repo = TempDir::new_in_tmp("steer-log-bound-repo");
    git(repo.path(), &["init", "-b", "master"]);
    git(repo.path(), &["config", "user.name", "mini-swe-test"]);
    git(repo.path(), &["config", "user.email", "test@localhost"]);
    std::fs::write(repo.path().join("README.md"), "# baseline\n").unwrap();
    git(repo.path(), &["add", "README.md"]);
    git(repo.path(), &["commit", "-m", "baseline"]);

    let pool = IsolatedPool::new(4, "steer-log-bound");
    let worker = format!("w1-{}", unique_suffix("w"));
    file_row(&repo, &pool, &worker);

    // ~16 MiB planted by anything that can write the scratch base.
    let planted = plant_log(&pool.scratch, &worker, 20_000, 800);
    assert!(
        planted > 16 * 1024 * 1024,
        "the planted log must be far over the budget to prove a bound, got {planted}"
    );

    let manifest = pool.pool.round_manifest(OWNER, GROUP, repo.path()).await;
    let shown: usize = manifest
        .ready
        .iter()
        .chain(manifest.not_ready.iter())
        .map(|w| w.steers.iter().map(|s| s.len()).sum::<usize>())
        .sum();

    assert_eq!(
        manifest.ready.len() + manifest.not_ready.len(),
        1,
        "the worker must actually be in the manifest, or this proves nothing"
    );

    // The invariant is that the reader's cost tracks its read cap, not the file:
    // a huge log must not cost materially more than a small one of the same
    // shape. Expressed as a ratio so it does not pin the cap to a literal.
    let small_repo = TempDir::new_in_tmp("steer-log-bound-small");
    git(small_repo.path(), &["init", "-b", "master"]);
    git(small_repo.path(), &["config", "user.name", "mini-swe-test"]);
    git(
        small_repo.path(),
        &["config", "user.email", "test@localhost"],
    );
    std::fs::write(small_repo.path().join("README.md"), "# baseline\n").unwrap();
    git(small_repo.path(), &["add", "README.md"]);
    git(small_repo.path(), &["commit", "-m", "baseline"]);

    let small_pool = IsolatedPool::new(4, "steer-log-bound-small-pool");
    let small_worker = format!("w2-{}", unique_suffix("w"));
    file_row(&small_repo, &small_pool, &small_worker);
    plant_log(&small_pool.scratch, &small_worker, 20, 800);

    let small = small_pool
        .pool
        .round_manifest(OWNER, GROUP, small_repo.path())
        .await;
    let small_shown: usize = small
        .ready
        .iter()
        .chain(small.not_ready.iter())
        .map(|w| w.steers.iter().map(|s| s.len()).sum::<usize>())
        .sum();

    assert!(
        shown <= small_shown.saturating_mul(8) + 1024 * 1024,
        "a {planted}-byte log must not cost materially more to read than a small one: \
         {shown} bytes vs {small_shown}"
    );
}

/// A record cut by the read cap is a torn line, not guidance: it must be
/// dropped whole rather than half-parsed into a fragment the prompt would
/// present as the orchestrator's words.
#[tokio::test]
async fn a_record_torn_by_the_read_cap_is_dropped_not_half_shown() {
    let repo = TempDir::new_in_tmp("steer-log-torn-repo");
    git(repo.path(), &["init", "-b", "master"]);
    git(repo.path(), &["config", "user.name", "mini-swe-test"]);
    git(repo.path(), &["config", "user.email", "test@localhost"]);
    std::fs::write(repo.path().join("README.md"), "# baseline\n").unwrap();
    git(repo.path(), &["add", "README.md"]);
    git(repo.path(), &["commit", "-m", "baseline"]);

    let pool = IsolatedPool::new(4, "steer-log-torn");
    let worker = format!("w3-{}", unique_suffix("w"));
    file_row(&repo, &pool, &worker);

    // Two intact records, then a third whose closing brace is missing: exactly
    // what a log cut mid-append looks like.
    let path = pool
        .scratch
        .path()
        .join(format!("swe-wt-{worker}.steer-log.jsonl"));
    let mut payload = String::new();
    payload.push_str(
        &serde_json::json!({"message": "INTACT-STEER", "sent_at": 1_u64, "pid": 1_u32}).to_string(),
    );
    payload.push('\n');
    payload.push_str(
        &serde_json::json!({"message": "SECOND-INTACT-STEER", "sent_at": 2_u64, "pid": 1_u32})
            .to_string(),
    );
    payload.push('\n');
    payload.push_str("{\"message\": \"TORN-STEER-NO-CLOSE");
    std::fs::write(&path, payload).unwrap();

    let manifest = pool.pool.round_manifest(OWNER, GROUP, repo.path()).await;
    let steers: Vec<&String> = manifest
        .ready
        .iter()
        .chain(manifest.not_ready.iter())
        .flat_map(|w| w.steers.iter())
        .collect();

    assert!(
        steers.iter().any(|s| s.contains("INTACT-STEER")),
        "the whole records before the torn one must survive: {steers:?}"
    );
    assert!(
        !steers.iter().any(|s| s.contains("TORN-STEER")),
        "a torn trailing record must never reach the prompt: {steers:?}"
    );
}
