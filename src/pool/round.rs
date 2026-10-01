//! The round manifest: what a consolidator is told when it is dispatched.
//!
//! One consolidator integrates a whole round, so it cannot discover the round
//! by exploring: it is handed the group's state at dispatch time, computed here
//! from the shared registry and the repository's own branches. The manifest
//! answers four questions and nothing else:
//!
//! * which of the caller's workers in the group finished with a branch that is
//!   not merged into the base branch yet (the ones to integrate),
//! * which are still running or stopped (reported separately as *not ready*, so
//!   the consolidator never merges a branch nobody finished),
//! * which files each of them touched (`git diff --name-only base...branch`),
//! * which of those files more than one worker touched — the interaction points
//!   the consolidator has to resolve itself.
//!
//! Everything is derived from read-only git probes on the repository the group
//! works in, and every list is sorted by worker id, so the same round always
//! renders byte-identical text.
//!
//! A group is one repository: the first row (in id order) that names a
//! `repo_path` decides which repository is probed, and its `base_branch` is the
//! base every branch is measured against.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use super::registry::RegistryStatus;
use super::revision::detect_base_branch;

/// One of the caller's workers in the group, as the registry knows it.
#[derive(Debug, Clone)]
pub struct RoundRow {
    pub id: String,
    pub task: String,
    pub status: RegistryStatus,
    /// Whether the worker's own verify gate passed, when this process still
    /// holds the state that recorded it.
    pub verified: Option<bool>,
}

/// One worker of the round, as the consolidator sees it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoundWorker {
    pub id: String,
    /// Lifecycle name (`Completed`, `Running`, ...), or `no branch` when the
    /// worker left no `worker-<id>` branch behind.
    pub state: String,
    pub verified: Option<bool>,
    /// First line of the worker's task.
    pub task: String,
    /// Files the branch touched, relative to the base branch.
    pub files: Vec<String>,
}

/// The round a consolidator is dispatched into.
#[derive(Debug, Clone, Default)]
pub struct RoundManifest {
    pub group: String,
    /// Base branch the branches are measured against, when one is known.
    pub base_branch: Option<String>,
    /// Finished workers whose branch is not merged yet: the ones to integrate.
    pub ready: Vec<RoundWorker>,
    /// Everyone else in the group with a branch still unmerged.
    pub not_ready: Vec<RoundWorker>,
    /// Files touched by more than one listed worker, with the workers that
    /// touched each.
    pub interaction_points: Vec<(String, Vec<String>)>,
}

impl RoundManifest {
    /// Whether the round holds anything to integrate.
    pub fn has_ready(&self) -> bool {
        !self.ready.is_empty()
    }

    /// Render the manifest as the text embedded in the consolidator's task.
    ///
    /// One block per section, workers in id order, so the model reads the same
    /// shape every round and the orchestrator can diff two rounds by eye.
    pub fn render(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(
            out,
            "ROUND MANIFEST group={} base={}",
            self.group,
            self.base_branch.as_deref().unwrap_or("unknown")
        );
        out.push_str("ready (completed, branch not yet merged):\n");
        if self.ready.is_empty() {
            out.push_str("  (none)\n");
        }
        for worker in &self.ready {
            push_worker(&mut out, worker);
        }
        out.push_str("not ready:\n");
        if self.not_ready.is_empty() {
            out.push_str("  (none)\n");
        }
        for worker in &self.not_ready {
            push_worker(&mut out, worker);
        }
        out.push_str("interaction points (touched by more than one worker):\n");
        if self.interaction_points.is_empty() {
            out.push_str("  (none)\n");
        }
        for (file, workers) in &self.interaction_points {
            let _ = writeln!(out, "  {file}: {}", workers.join(", "));
        }
        out
    }
}

/// One worker's lines: identity, state, verification, task heading, files.
fn push_worker(out: &mut String, worker: &RoundWorker) {
    let _ = writeln!(
        out,
        "  {} {} verified={} task=\"{}\"",
        worker.id,
        worker.state,
        verified_label(worker.verified),
        worker.task
    );
    if worker.files.is_empty() {
        out.push_str("    files: (none)\n");
        return;
    }
    let _ = writeln!(out, "    files: {}", worker.files.join(", "));
}

/// The manifest's spelling of a worker's verification outcome.
fn verified_label(verified: Option<bool>) -> &'static str {
    match verified {
        Some(true) => "yes",
        Some(false) => "no",
        None => "unknown",
    }
}

/// Build the round manifest for `group` from the caller's rows.
///
/// `repo` is the repository the group works in; `base_branch` overrides the
/// branch the rows name, which is what a dispatch that already resolved one
/// passes. Every git probe runs on the blocking pool: the caller is an async
/// handler, and the probes are the only part of this that touches the disk.
pub async fn build(
    group: &str,
    base_branch: Option<String>,
    repo: Option<PathBuf>,
    rows: Vec<RoundRow>,
) -> RoundManifest {
    let mut rows = rows;
    rows.sort_by(|a, b| a.id.cmp(&b.id));
    let mut manifest = RoundManifest {
        group: group.to_string(),
        base_branch: None,
        ready: Vec::new(),
        not_ready: Vec::new(),
        interaction_points: Vec::new(),
    };
    let Some(repo) = repo.filter(|path| path.is_dir()) else {
        // Without a repository there is no branch to inspect, so nothing can
        // be called ready: the dispatch refuses rather than sending a
        // consolidator after branches it cannot see.
        manifest.not_ready = rows
            .into_iter()
            .map(|row| RoundWorker {
                id: row.id,
                state: row.status.display_name().to_string(),
                verified: row.verified,
                task: crate::mcp::handlers::first_line(&row.task),
                files: Vec::new(),
            })
            .collect();
        return manifest;
    };

    let base_branch = base_branch.or_else(|| detect_base_branch(&repo));
    manifest.base_branch = base_branch.clone();
    let probes: Vec<(RoundRow, bool, bool, Vec<String>)> = tokio::task::spawn_blocking(move || {
        rows.into_iter()
            .map(|row| {
                let branch = format!("worker-{}", row.id);
                let exists = branch_exists(&repo, &branch);
                let merged = exists
                    && base_branch
                        .as_deref()
                        .is_some_and(|base| is_ancestor(&repo, &branch, base));
                let files = match base_branch.as_deref() {
                    Some(base) if exists && !merged => touched_files(&repo, base, &branch),
                    _ => Vec::new(),
                };
                (row, exists, merged, files)
            })
            .collect()
    })
    .await
    .unwrap_or_default();

    let mut by_file: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (row, exists, merged, files) in probes {
        if merged {
            // Already integrated into the base branch: not this round's work.
            continue;
        }
        let state = if exists {
            row.status.display_name().to_string()
        } else {
            "no branch".to_string()
        };
        let worker = RoundWorker {
            id: row.id.clone(),
            state,
            verified: row.verified,
            task: crate::mcp::handlers::first_line(&row.task),
            files: files.clone(),
        };
        if exists && row.status == RegistryStatus::Completed {
            for file in &files {
                by_file.entry(file.clone()).or_default().push(row.id.clone());
            }
            manifest.ready.push(worker);
        } else {
            manifest.not_ready.push(worker);
        }
    }
    manifest.interaction_points = by_file
        .into_iter()
        .filter(|(_, workers)| workers.len() > 1)
        .collect();
    manifest
}

/// Whether the repository still carries `branch`.
fn branch_exists(repo: &Path, branch: &str) -> bool {
    git_ok(repo, &["rev-parse", "--verify", "--quiet", branch])
}

/// Whether `branch` is already contained in `base`.
fn is_ancestor(repo: &Path, branch: &str, base: &str) -> bool {
    git_ok(repo, &["merge-base", "--is-ancestor", branch, base])
}

/// Files `branch` changed relative to its merge-base with `base`.
///
/// The three-dot range is the one `review` measures a branch with, so the
/// manifest and the review view agree on what a worker touched.
fn touched_files(repo: &Path, base: &str, branch: &str) -> Vec<String> {
    let range = format!("{base}...{branch}");
    let Ok(output) = crate::worktree::git(repo, "diff", &["diff", "--name-only", &range]) else {
        return Vec::new();
    };
    if !output.status.success() {
        return Vec::new();
    }
    let mut files: Vec<String> = String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(str::to_string)
        .collect();
    files.sort();
    files.dedup();
    files
}

/// Run a git probe that answers with its exit status alone.
fn git_ok(repo: &Path, args: &[&str]) -> bool {
    crate::worktree::git(repo, args[0], args)
        .is_ok_and(|output| output.status.success())
}
