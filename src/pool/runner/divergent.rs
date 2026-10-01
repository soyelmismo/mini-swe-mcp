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
    env
}

/// A fresh per-worktree directory for variant B, beside the canonical one.
fn divergent_dir(worktree: &Path, name: &str) -> PathBuf {
    worktree.join("target").join("divergent").join(name)
}

/// Wrap `verify` so it runs with `env` layered on top of the canonical one.
///
/// The command is replayed verbatim behind an `export` of the divergent
/// variables: the worker's own bash path still applies the sandbox, the
/// timeout, the guardrails and the build environment, so variant B is the same
/// execution as variant A with a different environment and nothing else.
pub fn divergent_command(verify: &str, env: &[(String, String)]) -> String {
    let mut wrapped = String::new();
    for (name, value) in env {
        wrapped.push_str("export ");
        wrapped.push_str(&shell_quote(name));
        wrapped.push('=');
        wrapped.push_str(&shell_quote(value));
        wrapped.push_str("; ");
    }
    wrapped.push_str(verify);
    wrapped
}

/// Single-quote `value` for `bash -c`, so a divergent value is never
/// re-interpreted by the shell that runs the gate.
fn shell_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', r"'\''"))
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
        new_refs: added(&baseline.refs, &refs_of(repo_root)),
        new_worktrees: added(&baseline.worktrees, &worktrees_of(repo_root)),
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

/// Remove what the gate left behind: the refs it created and the processes it
/// left running. The harness owns this cleanup so the next worker - and the
/// repository the orchestrator reviews - starts clean.
pub fn cleanup(repo_root: &Path, effects: &SideEffects) {
    for reference in &effects.new_refs {
        let _ = crate::worktree::git(repo_root, "update-ref", &["update-ref", "-d", reference]);
    }
    for worktree in &effects.new_worktrees {
        let _ = crate::worktree::git(
            repo_root,
            "worktree remove",
            &["worktree", "remove", "--force", worktree],
        );
    }
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
    let output = crate::worktree::git(repo_root, "worktree list", &["worktree", "list", "--porcelain"]);
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

/// Every file in the repository's main checkout, repository-relative.
///
/// `.git` is skipped: it is where refs and worktrees already live, and it is
/// audited separately above. The walk is depth-first and bounded, so a suite that
/// writes a deep or very large tree cannot stall the gate.
fn checkout_files(repo_root: &Path) -> Vec<String> {
    let mut files = Vec::new();
    let mut stack = vec![repo_root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        if files.len() >= CHECKOUT_WALK_LIMIT {
            break;
        }
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let name = entry.file_name();
            if name == ".git" {
                continue;
            }
            if path.is_dir() {
                stack.push(path);
            } else if let Ok(relative) = path.strip_prefix(repo_root) {
                files.push(relative.to_string_lossy().into_owned());
            }
        }
    }
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
        items.push(format!("created file {file} in the repository's main checkout"));
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
