//! The differential verify gate and the side-effect audit around it.
//!
//! A verify gate that only ever runs in the canonical sandbox environment
//! proves the suite is hermetic *against the sandbox*, not against the world
//! the code is later verified in: the orchestrator's shell, CI, a developer
//! machine. A suite that reads an ambient variable, inherits an identity
//! variable, assumes the host timezone or dispatches into the real repository
//! passes here and fails there, and the worker never learns why.
//!
//! So when the gate passes in the canonical environment (variant A) the same
//! command runs once more in variant B: the canonical environment plus the
//! dispatcher's filtered ambient variables, a fresh different `HOME` and
//! `TMPDIR`, and a `TZ` shifted far from the host's. Same sandbox, same
//! worktree, same verify command - only the environment differs.
//!
//! Around both runs the module audits what the suite left behind: refs and
//! worktrees in the shared repository, processes the reap sweep had to kill,
//! and new files in the repository's main checkout. Anything found refuses
//! completion with the exact list, and the harness cleans it up.
//!
//! Nothing here assumes a language or a test runner: the gate is whatever
//! command the dispatch or the auto-detection chose, and it is replayed
//! verbatim.

use std::path::{Path, PathBuf};

/// Disables variant B when set to `0`.
///
/// The differential run doubles the verify cost, so an operator who has
/// already made the suite hermetic can turn it off; every other value (and an
/// unset variable) leaves it on.
pub const DISABLE_ENV: &str = "WORKER_DIVERGENT_VERIFY";

/// Timezones variant B shifts to, in the order they are considered.
///
/// One is the far east of the date line, the other the far west: between them
/// every host timezone is at least twelve hours away from one of them.
const SHIFTED_ZONES: &[&str] = &["Pacific/Kiritimati", "Etc/GMT+12"];

/// Bound on the number of entries one side-effect list may carry.
///
/// A suite that creates thousands of refs must not turn the refusal into an
/// unbounded message, so the list is truncated and the truncation is stated.
const SIDE_EFFECT_REPORT_LIMIT: usize = 20;

/// Bound on the number of files walked in the main checkout.
///
/// The walk is a safety net against a suite that writes into the repository it
/// was dispatched against, not a repository indexer; a huge tree is reported
/// truncated rather than walked to the end.
const CHECKOUT_WALK_LIMIT: usize = 50_000;

/// Whether variant B should run for this worker.
pub fn enabled() -> bool {
    std::env::var(DISABLE_ENV).ok().as_deref() != Some("0")
}

/// The `TZ` value farthest from the host's, as a zone name.
///
/// The host's offset is read once from `date`, so no timezone database has to
/// be linked in; when it cannot be read the easternmost zone is used, which is
/// the safer default because it is never the host's zone on a machine that
/// could not report one.
pub fn shifted_timezone() -> String {
    let host_offset = host_utc_offset_minutes();
    let mut best = SHIFTED_ZONES[0];
    let mut best_distance = i32::MIN;
    for zone in SHIFTED_ZONES {
        let distance = zone_offset_minutes(zone)
            .map(|offset| (offset - host_offset).abs())
            .unwrap_or(i32::MAX);
        if distance > best_distance {
            best_distance = distance;
            best = zone;
        }
    }
    best.to_string()
}

/// The host's UTC offset in minutes, or `0` when it cannot be determined.
fn host_utc_offset_minutes() -> i32 {
    std::process::Command::new("date")
        .arg("+%z")
        .output()
        .ok()
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .and_then(|raw| parse_offset(raw.trim()))
        .unwrap_or(0)
}

/// The UTC offset `zone` implies, in minutes.
fn zone_offset_minutes(zone: &str) -> Option<i32> {
    std::process::Command::new("date")
        .args(["-d", "TZ=now", "+%z"])
        .env("TZ", zone)
        .output()
        .ok()
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .and_then(|raw| parse_offset(raw.trim()))
}

/// Parse `date +%z` (`+HHMM`, `-HHMM`) into minutes east of UTC.
fn parse_offset(raw: &str) -> Option<i32> {
    let (sign, rest) = match raw.as_bytes().first()? {
        b'+' => (1, &raw[1..]),
        b'-' => (-1, &raw[1..]),
        _ => return None,
    };
    if rest.len() < 4 || !rest.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let hours: i32 = rest[..2].parse().ok()?;
    let minutes: i32 = rest[2..].parse().ok()?;
    Some(sign * (hours * 60 + minutes))
}

/// Variant B's environment layered on top of the canonical one.
///
/// The canonical environment is what [`crate::agent::env`] already gives every
/// command; this adds the dispatcher's filtered ambient variables and then the
/// three deliberate divergences. `HOME` and `TMPDIR` point at fresh directories
/// inside the worktree, so they stay inside the sandbox's writable area and are
/// created before the command runs.
pub fn divergent_environment(
    worktree: &Path,
    client_env: &[(String, String)],
) -> Vec<(String, String)> {
    let mut env: Vec<(String, String)> = client_env.to_vec();
    // `HOME`/`TMPDIR` last: a dispatcher's own value must not win over the
    // deliberate divergence, which is the whole point of the second run.
    let home = divergent_dir(worktree, "home");
    let tmp = divergent_dir(worktree, "tmp");
    let _ = std::fs::create_dir_all(&home);
    let _ = std::fs::create_dir_all(&tmp);
    env.push(("HOME".to_string(), home.to_string_lossy().into_owned()));
    env.push(("TMPDIR".to_string(), tmp.to_string_lossy().into_owned()));
    env.push(("TMP".to_string(), tmp.to_string_lossy().into_owned()));
    env.push(("TEMP".to_string(), tmp.to_string_lossy().into_owned()));
    env.push(("TZ".to_string(), shifted_timezone()));
    // The deliberate divergences are appended last, so a dispatcher's own
    // HOME/TMPDIR/TZ is shadowed rather than honoured: the point of the second
    // run is that these three differ. The last occurrence wins, which is what
    // the export sequence below produces.
    let mut last: Vec<(String, String)> = Vec::with_capacity(env.len());
    for (name, value) in env {
        if let Some(slot) = last.iter_mut().find(|(existing, _)| *existing == name) {
            slot.1 = value;
        } else {
            last.push((name, value));
        }
    }
    last
}

/// A fresh per-worktree directory for variant B, inside the worker's private
/// scratch (writable in the sandbox, removed at teardown).
///
/// It is kept SHORT on purpose: suites create Unix sockets under `TMPDIR`, and
/// a socket path must fit in 108 bytes, so a deep variant-B `TMPDIR` made
/// otherwise hermetic suites fail for a reason that is not theirs.
fn divergent_dir(worktree: &Path, name: &str) -> PathBuf {
    crate::worktree::scratch_dir(worktree).join(format!("vb-{name}"))
}

/// The names variant B changes relative to the canonical environment.
pub fn divergent_names(env: &[(String, String)]) -> Vec<String> {
    let mut names: Vec<String> = env.iter().map(|(name, _)| name.clone()).collect();
    names.sort();
    names.dedup();
    names
}

// ----------
// Side-effect audit
// ----------

/// What the repository looked like before the gate ran.
#[derive(Debug, Default)]
pub struct SideEffectBaseline {
    refs: Vec<String>,
    worktrees: Vec<String>,
    files: Vec<String>,
}

/// What the gate left behind, in the order it should be reported.
#[derive(Debug, Default)]
pub struct SideEffects {
    pub new_refs: Vec<String>,
    pub new_worktrees: Vec<String>,
    pub new_files: Vec<String>,
    /// Processes the reap sweep had to kill, as `pid (command)`.
    pub killed_processes: Vec<String>,
    pub truncated: bool,
}

impl SideEffects {
    /// Whether the gate left anything behind at all.
    pub fn is_empty(&self) -> bool {
        self.new_refs.is_empty()
            && self.new_worktrees.is_empty()
            && self.new_files.is_empty()
            && self.killed_processes.is_empty()
    }
}

/// Snapshot the shared repository before the gate runs.
pub fn snapshot(repo_root: &Path) -> SideEffectBaseline {
    SideEffectBaseline {
        refs: refs_of(repo_root),
        worktrees: worktrees_of(repo_root),
        files: checkout_files(repo_root),
    }
}

/// Diff the repository against `baseline` and sweep the worker's processes.
///
/// The sweep is the same one the harness runs on teardown
/// ([`crate::agent::reap`]), so a process it had to kill is both reported and
/// cleaned up here rather than left for the next worker to inherit.
pub fn audit(
    repo_root: &Path,
    worktree: &Path,
    worker_id: &str,
    baseline: &SideEffectBaseline,
) -> SideEffects {
    let mut effects = SideEffects {
        new_refs: added(&baseline.refs, &refs_of(repo_root))
            .into_iter()
            .filter(|reference| !is_hub_managed_ref(reference))
            .collect(),
        new_worktrees: added(&baseline.worktrees, &worktrees_of(repo_root))
            .into_iter()
            .filter(|worktree| !is_hub_managed_worktree(worktree))
            .collect(),
        new_files: added(&baseline.files, &checkout_files(repo_root)),
        ..SideEffects::default()
    };
    effects.truncated = effects.new_refs.len() > SIDE_EFFECT_REPORT_LIMIT
        || effects.new_worktrees.len() > SIDE_EFFECT_REPORT_LIMIT
        || effects.new_files.len() > SIDE_EFFECT_REPORT_LIMIT;
    effects.killed_processes = reap_worker_processes(worktree, worker_id);
    effects.truncated |= effects.killed_processes.len() > SIDE_EFFECT_REPORT_LIMIT;
    effects.new_refs.truncate(SIDE_EFFECT_REPORT_LIMIT);
    effects.new_worktrees.truncate(SIDE_EFFECT_REPORT_LIMIT);
    effects.new_files.truncate(SIDE_EFFECT_REPORT_LIMIT);
    effects.killed_processes.truncate(SIDE_EFFECT_REPORT_LIMIT);
    effects
}

/// Remove what the gate left behind.
///
/// Refs and worktrees are only REPORTED, never deleted: the repository is
/// shared with the hub (which creates a `worker-<id>` branch and worktree for
/// every dispatch) and with the operator, so something that appeared during
/// this gate cannot be attributed to this suite with certainty, and deleting
/// it destroyed a concurrently dispatched worker's checkout. Processes are
/// handled by the reap sweep, which only touches this worker's directories.
pub fn cleanup(_repo_root: &Path, _effects: &SideEffects) {}

/// A ref the hub itself creates for a worker (`refs/heads/worker-<id>`).
fn is_hub_managed_ref(reference: &str) -> bool {
    reference.starts_with("refs/heads/worker-")
}

/// A worktree the hub itself creates for a worker (`.../swe-wt-<id>`).
fn is_hub_managed_worktree(path: &str) -> bool {
    Path::new(path)
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.starts_with("swe-wt-"))
}

/// Every ref in the shared repository, as `refs/...` names.
fn refs_of(repo_root: &Path) -> Vec<String> {
    let output = crate::worktree::git(
        repo_root,
        "for-each-ref",
        &["for-each-ref", "--format=%(refname)"],
    );
    match output {
        Ok(out) if out.status.success() => lines_of(&out.stdout),
        _ => Vec::new(),
    }
}

/// Every registered worktree path in the shared repository.
fn worktrees_of(repo_root: &Path) -> Vec<String> {
    let output = crate::worktree::git(
        repo_root,
        "worktree list",
        &["worktree", "list", "--porcelain"],
    );
    match output {
        Ok(out) if out.status.success() => out
            .stdout
            .split(|&b| b == b'\n')
            .filter_map(|line| {
                let line = String::from_utf8_lossy(line);
                line.strip_prefix("worktree ").map(str::to_string)
            })
            .collect(),
        _ => Vec::new(),
    }
}

/// The untracked, non-ignored files of the repository's main checkout,
/// repository-relative - what a suite could have left behind there.
///
/// Ignored paths (build output such as `target/`, caches) are excluded: other
/// processes - the operator's own builds, compiler caches - write there all
/// the time, so they say nothing about the worker. Asking git also keeps the
/// audit bounded and independent of the language's build layout.
fn checkout_files(repo_root: &Path) -> Vec<String> {
    let Ok(output) = crate::worktree::git(
        repo_root,
        "status untracked",
        &["status", "--porcelain", "-z", "--untracked-files=all"],
    ) else {
        return Vec::new();
    };
    if !output.status.success() {
        return Vec::new();
    }
    let mut files: Vec<String> = output
        .stdout
        .split(|byte| *byte == 0)
        .filter_map(|entry| entry.strip_prefix(b"?? "))
        .map(|path| String::from_utf8_lossy(path).into_owned())
        .take(CHECKOUT_WALK_LIMIT)
        .collect();
    files.sort();
    files
}

/// The processes the reap sweep had to kill for this worker.
fn reap_worker_processes(worktree: &Path, worker_id: &str) -> Vec<String> {
    let dirs = crate::agent::reap::worker_dirs(worktree);
    crate::agent::reap::sweep_worker_processes_named(worker_id, &dirs)
        .into_iter()
        .map(|proc| format!("{} ({})", proc.pid, proc.command))
        .collect()
}

/// The entries of `after` that `before` did not have, sorted.
fn added(before: &[String], after: &[String]) -> Vec<String> {
    let known: std::collections::HashSet<&str> = before.iter().map(String::as_str).collect();
    let mut fresh: Vec<String> = after
        .iter()
        .filter(|entry| !known.contains(entry.as_str()))
        .cloned()
        .collect();
    fresh.sort();
    fresh
}

/// One line of a command's stdout, trimmed of the trailing newline.
fn lines_of(raw: &[u8]) -> Vec<String> {
    String::from_utf8_lossy(raw)
        .lines()
        .map(str::to_string)
        .collect()
}

/// The refusal pushed back to the model when variant B fails.
///
/// It names what differs, states that `HOME`/`TMPDIR`/`TZ` differ, and carries
/// a bounded tail of the failing output - the model has to be able to act on
/// it, and it must not be able to read a credential out of it either.
pub fn divergence_refusal(
    verify: &str,
    differing: &[String],
    exit: Option<i32>,
    output: &str,
) -> String {
    let names = if differing.is_empty() {
        "none beyond HOME/TMPDIR/TZ".to_string()
    } else {
        differing.join(", ")
    };
    let tail = crate::agent::sandbox::truncate_output(output);
    format!(
        "COMPLETION REFUSED: verification passes in a clean environment but fails in the orchestrator's environment; \
variables that differ: {names}; HOME/TMPDIR/TZ differ; failing output: {tail}. \
Make the tests independent of the environment.\n\
(The same command `{verify}` was re-run with the dispatcher's ambient variables, a fresh HOME and TMPDIR and a shifted TZ; exit {exit:?}.)"
    )
}

/// The refusal pushed back to the model when the gate left side effects.
pub fn side_effect_refusal(effects: &SideEffects) -> String {
    let mut items = Vec::new();
    for reference in &effects.new_refs {
        items.push(format!("created ref {reference}"));
    }
    for worktree in &effects.new_worktrees {
        items.push(format!("left worktree {worktree}"));
    }
    for process in &effects.killed_processes {
        items.push(format!("left process {process}"));
    }
    for file in &effects.new_files {
        items.push(format!(
            "created file {file} in the repository's main checkout"
        ));
    }
    let truncation = if effects.truncated {
        " (list truncated; there is more)"
    } else {
        ""
    };
    format!(
        "COMPLETION REFUSED: your tests must clean up after themselves: {}{truncation}. \
The harness removed what it could; make the suite leave the repository, its refs and its processes exactly as it found them.",
        items.join(", ")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Only what git would show as new counts: ignored build output written by
    /// other processes (the operator's builds, compiler caches) is not a leak.
    #[test]
    fn the_checkout_audit_ignores_ignored_paths() {
        let repo = std::env::temp_dir().join(format!("audit-ignored-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&repo);
        std::fs::create_dir_all(&repo).expect("create repo");
        let git = |args: &[&str]| {
            let out = std::process::Command::new("git")
                .current_dir(&repo)
                .args(args)
                .output()
                .expect("run git");
            assert!(out.status.success(), "git {args:?}");
        };
        git(&["init", "-q", "-b", "master"]);
        std::fs::write(repo.join(".gitignore"), "target/\n").expect("write .gitignore");
        git(&[
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t",
            "add",
            ".gitignore",
        ]);
        git(&[
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t",
            "commit",
            "-q",
            "-m",
            "seed",
        ]);
        let before = checkout_files(&repo);

        std::fs::create_dir_all(repo.join("target/debug")).expect("create target");
        std::fs::write(repo.join("target/debug/artifact"), "x").expect("write ignored file");
        std::fs::write(repo.join("left-behind.txt"), "x").expect("write untracked file");

        assert_eq!(
            added(&before, &checkout_files(&repo)),
            vec!["left-behind.txt".to_string()]
        );
        let _ = std::fs::remove_dir_all(&repo);
    }

    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "divergent-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create scratch dir");
        dir
    }

    fn git(dir: &Path, args: &[&str]) {
        let out = std::process::Command::new("git")
            .current_dir(dir)
            .args(args)
            .output()
            .unwrap_or_else(|e| panic!("git {args:?}: {e}"));
        assert!(
            out.status.success(),
            "git {args:?} failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    fn repo(tag: &str) -> PathBuf {
        let dir = scratch(tag);
        git(&dir, &["init", "-b", "master"]);
        git(&dir, &["config", "user.name", "t"]);
        git(&dir, &["config", "user.email", "t@localhost"]);
        std::fs::write(dir.join("seed.txt"), "seed\n").unwrap();
        git(&dir, &["add", "seed.txt"]);
        git(&dir, &["commit", "-m", "baseline"]);
        dir
    }

    #[test]
    fn a_created_ref_is_reported_but_never_deleted() {
        let dir = repo("ref");
        let baseline = snapshot(&dir);
        git(&dir, &["branch", "leftover-branch"]);
        let effects = audit(&dir, &dir, "worker-test", &baseline);
        assert_eq!(
            effects.new_refs,
            vec!["refs/heads/leftover-branch".to_string()],
            "the audit must name a ref that appeared during the gate"
        );
        assert!(
            !effects.is_empty(),
            "a created ref is a side effect that must refuse completion"
        );
        cleanup(&dir, &effects);
        assert!(
            snapshot(&dir)
                .refs
                .contains(&"refs/heads/leftover-branch".to_string()),
            "refs are reported, never deleted: the repository is shared"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_created_checkout_file_is_audited() {
        let dir = repo("file");
        let baseline = snapshot(&dir);
        std::fs::write(dir.join("stray.txt"), "stray\n").unwrap();
        let effects = audit(&dir, &dir, "worker-test", &baseline);
        assert_eq!(
            effects.new_files,
            vec!["stray.txt".to_string()],
            "a new file in the main checkout is a side effect"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The hub creates `worker-<id>` branches and `swe-wt-<id>` worktrees for
    /// workers dispatched while this gate runs: they are not this suite's side
    /// effects, and nothing is ever deleted on their account.
    #[test]
    fn concurrent_hub_branches_are_neither_reported_nor_deleted() {
        let dir = repo("concurrent");
        let baseline = snapshot(&dir);
        let out = std::process::Command::new("git")
            .current_dir(&dir)
            .args(["branch", "worker-concurrent1"])
            .output()
            .expect("create a hub-style branch");
        assert!(out.status.success());
        let effects = audit(&dir, &dir, "worker-test", &baseline);
        assert!(effects.new_refs.is_empty(), "{effects:?}");
        cleanup(&dir, &effects);
        let out = std::process::Command::new("git")
            .current_dir(&dir)
            .args(["rev-parse", "--verify", "refs/heads/worker-concurrent1"])
            .output()
            .expect("check the branch");
        assert!(
            out.status.success(),
            "the concurrent worker's branch must survive"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_untouched_repository_has_no_side_effects() {
        let dir = repo("clean");
        let baseline = snapshot(&dir);
        let effects = audit(&dir, &dir, "worker-test", &baseline);
        assert!(
            effects.is_empty(),
            "an untouched repository must not refuse completion: {effects:?}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_side_effect_refusal_lists_everything_it_found() {
        let effects = SideEffects {
            new_refs: vec!["refs/heads/worker-x".to_string()],
            new_files: vec!["stray.txt".to_string()],
            ..SideEffects::default()
        };
        let refusal = side_effect_refusal(&effects);
        assert!(
            refusal.contains("created ref refs/heads/worker-x")
                && refusal.contains("created file stray.txt"),
            "the refusal must list the exact leftovers, got {refusal:?}"
        );
    }

    #[test]
    fn the_divergence_refusal_names_the_variables_and_the_bounded_tail() {
        let refusal = divergence_refusal(
            "pytest -q",
            &["HOME".to_string(), "TZ".to_string()],
            Some(1),
            "assert 1 == 2\n",
        );
        assert!(
            refusal.contains("HOME, TZ")
                && refusal.contains("HOME/TMPDIR/TZ differ")
                && refusal.contains("assert 1 == 2")
                && refusal.contains("Make the tests independent of the environment"),
            "the refusal must name what differs and carry the failing output, got {refusal:?}"
        );
    }

    #[test]
    fn the_divergent_environment_shifts_home_tmpdir_and_tz() {
        let guard = crate::test_support::TestScratch::own(scratch("env"));
        let worktree = guard.path().to_path_buf();
        let env = divergent_environment(
            &worktree,
            &[("SWE_DIVERGENT_PROBE".to_string(), "set".to_string())],
        );
        let names = divergent_names(&env);
        for name in ["HOME", "TMPDIR", "TZ", "SWE_DIVERGENT_PROBE"] {
            assert!(
                names.iter().any(|n| n == name),
                "{name} must be in {names:?}"
            );
        }
        let tz = env
            .iter()
            .find(|(name, _)| name == "TZ")
            .map(|(_, value)| value.clone())
            .expect("TZ is always set");
        assert!(
            SHIFTED_ZONES.contains(&tz.as_str()),
            "TZ must be one of the far zones, got {tz:?}"
        );
        let home = env
            .iter()
            .find(|(name, _)| name == "HOME")
            .map(|(_, value)| value.clone())
            .expect("HOME is always set");
        assert!(
            Path::new(&home).starts_with(crate::worktree::scratch_dir(&worktree)),
            "the divergent HOME must stay inside the worker's private scratch, got {home:?}"
        );
        let _ = std::fs::remove_dir_all(&worktree);
    }

    #[test]
    fn a_dispatcher_variable_cannot_override_the_deliberate_divergence() {
        let guard = crate::test_support::TestScratch::own(scratch("override"));
        let worktree = guard.path().to_path_buf();
        let env = divergent_environment(
            &worktree,
            &[
                ("HOME".to_string(), "/dispatchers/home".to_string()),
                ("TZ".to_string(), "UTC".to_string()),
            ],
        );
        let home = env
            .iter()
            .find(|(name, _)| name == "HOME")
            .map(|(_, value)| value.clone())
            .expect("HOME");
        assert_ne!(
            home, "/dispatchers/home",
            "the deliberate divergence must win over the dispatcher's own value"
        );
        let _ = std::fs::remove_dir_all(&worktree);
    }

    #[test]
    fn the_disable_switch_reads_the_environment() {
        crate::agent::env::with_env_lock(|| {
            // SAFETY: serialized against every other test that reads the
            // process environment.
            unsafe {
                std::env::set_var(DISABLE_ENV, "0");
            }
            assert!(!enabled(), "WORKER_DIVERGENT_VERIFY=0 disables variant B");
            unsafe {
                std::env::set_var(DISABLE_ENV, "1");
            }
            assert!(enabled(), "any other value leaves variant B on");
            unsafe {
                std::env::remove_var(DISABLE_ENV);
            }
            assert!(enabled(), "an unset variable leaves variant B on");
        });
    }
}
