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
    /// Steers the orchestrator sent this worker after dispatch, in arrival
    /// order. Read from the worker's steer log, so a worker this process never
    /// steered is still listed.
    pub steers: Vec<String>,
}

/// One worker of the round, as the consolidator sees it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoundWorker {
    pub id: String,
    /// Lifecycle name (`Completed`, `Running`, ...), or `no branch` when the
    /// worker left no `worker-<id>` branch behind.
    pub state: String,
    pub verified: Option<bool>,
    /// First line of the worker's task, as the compact manifest renders it.
    pub task: String,
    /// The worker's whole task, as the registry recorded it, for the
    /// consolidator's scope review. Bounded when rendered (see
    /// [`RoundManifest::render_full_tasks`]), never part of the compact
    /// [`RoundManifest::render`].
    pub full_task: String,
    /// What the orchestrator steered the worker *after* dispatch, in arrival
    /// order: the scope amendments that supersede `full_task` where they
    /// conflict. Bounded when rendered (see
    /// [`RoundManifest::render_full_tasks`]), never part of the compact
    /// [`RoundManifest::render`].
    pub steers: Vec<String>,
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

    /// The task a consolidator is dispatched with.
    ///
    /// Three parts, in the order the model needs them: what the round holds,
    /// the one gate it must run, and the procedure it follows (see
    /// [`crate::agent::CONSOLIDATOR_INSTRUCTIONS`]). The gate is spelled out
    /// because "run the full gate" is only actionable once it names a command.
    pub fn task_text(&self, verify: Option<&str>) -> String {
        let gate = verify.unwrap_or("(none detected: run the project's own checks)");
        let mut out = format!(
            "Consolidate the round in group {}: integrate the finished branches, run the \
             full gate once, route what you cannot own back to its owner, review every diff, \
             and report.\n\n{}\n",
            self.group,
            self.render(),
        );
        let full_tasks = self.render_full_tasks();
        if !full_tasks.is_empty() {
            let _ = writeln!(out, "{full_tasks}");
        }
        let _ = write!(
            out,
            "Full gate for this round: `{gate}`\n\n{}",
            crate::agent::CONSOLIDATOR_INSTRUCTIONS
        );
        out
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

    /// The full task of every worker of the round, bounded, in a section
    /// deliberately separate from the compact [`Self::render`].
    ///
    /// The compact manifest keeps each task's first line so the orchestrator
    /// can diff two rounds by eye; the consolidator judges whether each diff
    /// respected its task's scope and so needs the whole text. Every worker
    /// listed (ready first, then not ready) contributes one entry: its whole
    /// task bounded by `FULL_TASK_BUDGET`, then the orchestrator steers it
    /// received after dispatch (rendered by `render_steers`), which amend that
    /// task. Both are cut with a `[truncated]` marker, and the whole section —
    /// the omitted-count footer included, when one is written — bounded by
    /// `FULL_TASKS_BUDGET` bytes, so
    /// a verbose round cannot flood the prompt; work the budget leaves
    /// out is counted rather than silently dropped.
    ///
    /// Returns an empty string when the round lists no worker, so a caller can
    /// splice it unconditionally.
    pub fn render_full_tasks(&self) -> String {
        if self.ready.is_empty() && self.not_ready.is_empty() {
            return String::new();
        }
        let mut out = String::from(
            "FULL TASKS OF THE ROUND'S WORKERS (judge each worker's diff against the \
             worker's own text below, as amended by the orchestrator steers that follow \
             it; the manifest above keeps only each task's first line):\n",
        );
        // The section ends with the omitted-count footer whenever any
        // worker is left out, so the footer's bytes are part of the
        // budget: entries stop `FOOTER_BUDGET` early, at a whole
        // worker, and the finished section (footer included) is at
        // most `FULL_TASKS_BUDGET` bytes.
        let footer = format!(
            "({} further worker task(s) omitted: section budget reached)\n",
            self.ready.len() + self.not_ready.len()
        );
        let footer = footer.len().min(FOOTER_BUDGET);
        let mut remaining = FULL_TASKS_BUDGET
            .saturating_sub(out.len())
            .saturating_sub(footer);
        let mut omitted = 0usize;
        for worker in self.ready.iter().chain(self.not_ready.iter()) {
            let task = bound_task(&worker.full_task, FULL_TASK_BUDGET);
            let entry = format!(
                "### {}:\n{task}{}\n",
                worker.id,
                render_steers(&worker.steers)
            );
            if entry.len() > remaining {
                omitted += 1;
                continue;
            }
            out.push_str(&entry);
            remaining -= entry.len();
        }
        if omitted > 0 {
            let _ = writeln!(
                out,
                "({omitted} further worker task(s) omitted: section budget reached)"
            );
        }
        out
    }
}

/// Byte budget for one worker's full task in [`RoundManifest::render_full_tasks`],
/// truncation marker included.
const FULL_TASK_BUDGET: usize = 4 * 1024;

/// Overall byte budget for the full-task section, so a round of many verbose
/// workers cannot flood the consolidator's prompt. The section never
/// exceeds it, omitted-count footer included.
const FULL_TASKS_BUDGET: usize = 16 * 1024;

/// Ceiling for the omitted-count footer the section ends with. The
/// count is bounded by the round's worker list, so the real footer is
/// rarely this long; reserving the ceiling keeps the entry budget
/// valid for every count.
const FOOTER_BUDGET: usize = 128;

/// Marker appended to a task cut to [`FULL_TASK_BUDGET`].
const TASK_TRUNCATION_MARKER: &str = "\n[truncated]";

/// Bound `task` to `budget` bytes, appending [`TASK_TRUNCATION_MARKER`] when it
/// is cut, never splitting a UTF-8 code point and never exceeding `budget`.
fn bound_task(task: &str, budget: usize) -> String {
    if task.len() <= budget {
        return task.to_string();
    }
    if budget <= TASK_TRUNCATION_MARKER.len() {
        return String::new();
    }
    let cut = task.floor_char_boundary(budget - TASK_TRUNCATION_MARKER.len());
    format!("{}{TASK_TRUNCATION_MARKER}", &task[..cut])
}

/// What the orchestrator told a worker after dispatch, as the amendment block
/// that follows that worker's task.
///
/// Rendered after the original task because it *is* an amendment to it: the
/// consolidator reads the task, then the orchestrator's later steering, and
/// judges the diff against the pair. Each steer is one numbered item so the
/// order the worker received them in stays visible (a later steer can retract
/// an earlier one), and the whole block is bounded by [`STEERS_BUDGET`] with
/// the shared truncation marker, so a chatty orchestrator cannot push a real
/// amendment out of the section.
///
/// Returns an empty string for a worker nobody steered, so the common case
/// costs nothing in the section's byte budget.
fn render_steers(steers: &[String]) -> String {
    if steers.is_empty() {
        return String::new();
    }
    let mut out = String::from(
        "ORCHESTRATOR STEERS AFTER DISPATCH (these SUPERSEDE the task above wherever they \
         conflict):\n",
    );
    for (i, steer) in steers.iter().enumerate() {
        // A steer is the orchestrator's own words: rendered verbatim, so the
        // consolidator judges the decision the user actually approved.
        out.push_str(&format!("{}. {}\n", i + 1, steer.trim()));
    }
    bound_task(&out, STEERS_BUDGET)
}

/// Byte budget for one worker's orchestrator-steer amendment block,
/// truncation marker included.
const STEERS_BUDGET: usize = 2 * 1024;

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
                task: first_line(&row.task),
                full_task: row.task,
                steers: row.steers,
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
            task: first_line(&row.task),
            full_task: row.task,
            steers: row.steers,
            files: files.clone(),
        };
        if exists && row.status == RegistryStatus::Completed {
            for file in &files {
                by_file
                    .entry(file.clone())
                    .or_default()
                    .push(row.id.clone());
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

/// The first line of a task that says something: an agent writes a heading and
/// a body, and only the heading belongs in a compact view.
pub(crate) fn first_line(text: &str) -> String {
    text.lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .unwrap_or_default()
        .to_string()
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
    crate::worktree::git(repo, args[0], args).is_ok_and(|output| output.status.success())
}

#[cfg(test)]
mod round_tests {
    use super::*;

    /// A worker row for the section's bounds, with the compact
    /// `task` derived from the full text the way `build` derives it.
    fn worker(id: &str, full_task: &str) -> RoundWorker {
        RoundWorker {
            id: id.to_string(),
            state: "Completed".to_string(),
            verified: None,
            task: first_line(full_task),
            full_task: full_task.to_string(),
            steers: Vec::new(),
            files: Vec::new(),
        }
    }

    /// The footer is reserved space, not a suffix: entries stop
    /// `FOOTER_BUDGET` early, so the finished section — footer
    /// included — never exceeds `FULL_TASKS_BUDGET`.
    ///
    /// The crafting is exact: full 4 KiB tasks fill the entry
    /// budget whole, then one task sized to fill the remaining
    /// bytes exactly, then a task only the unreserved
    /// implementation still admits, then one more worker neither
    /// admits. Without the reservation that middle task was
    /// admitted and the omitted-count footer was appended outside
    /// the budget, past the cap.
    #[test]
    fn the_full_task_section_reserves_the_footer_before_admitting_entries() {
        // The header's own bytes, measured from a rendered section
        // so the test follows the header's text.
        let probe = RoundManifest {
            group: "g".to_string(),
            base_branch: None,
            ready: vec![worker("w0", "probe")],
            not_ready: Vec::new(),
            interaction_points: Vec::new(),
        };
        let header_len = probe
            .render_full_tasks()
            .find("### ")
            .expect("the section opens with its header");

        // One `  ### wN:\n<task>\n` entry: a two-character id
        // around a task bounded by the per-worker budget.
        let full_entry = FULL_TASK_BUDGET + "### w0:\n".len() + "\n".len();
        let reserved = FULL_TASKS_BUDGET
            .saturating_sub(header_len)
            .saturating_sub(FOOTER_BUDGET);
        let full_entries = reserved / full_entry;
        let leftover = reserved - full_entries * full_entry;
        assert!(
            leftover >= "### w0:\n".len() + "\n".len(),
            "the crafted round needs room for the exact filler: {leftover}"
        );

        let mut ready = Vec::new();
        for i in 0..full_entries {
            // A task of exactly the per-worker budget: embedded
            // whole, never truncated.
            ready.push(worker(
                &format!("w{}", i + 1),
                &"a".repeat(FULL_TASK_BUDGET),
            ));
        }
        // The task that fills the reserved bytes exactly.
        let filler = "b".repeat(leftover - "### w0:\n".len() - "\n".len());
        ready.push(worker(&format!("w{}", full_entries + 1), &filler));
        // A task the unreserved budget still admits (it leaves
        // `FOOTER_BUDGET` bytes free) but the reserved one does
        // not: the regression's tell-tale.
        let smuggled = "c".repeat(FOOTER_BUDGET - "### w0:\n".len() - "\n".len());
        ready.push(worker(&format!("w{}", full_entries + 2), &smuggled));
        // One more worker neither implementation admits, so both
        // write the omitted-count footer the section is bounded by.
        ready.push(worker(
            &format!("w{}", full_entries + 3),
            &"d".repeat(FOOTER_BUDGET),
        ));

        let manifest = RoundManifest {
            group: "g".to_string(),
            base_branch: None,
            ready,
            not_ready: Vec::new(),
            interaction_points: Vec::new(),
        };
        let section = manifest.render_full_tasks();
        assert!(
            section.contains(&filler),
            "the task filling the reserved bytes is embedded whole: {section}"
        );
        assert!(
            !section.contains(&smuggled),
            "the entry only the unreserved budget admits must be omitted: {section}"
        );
        assert!(
            section.contains("omitted"),
            "the workers the budget left out are counted: {section}"
        );
        assert!(
            section.len() <= FULL_TASKS_BUDGET,
            "the footer is reserved space, not an addition: {} bytes over {}",
            section.len().saturating_sub(FULL_TASKS_BUDGET),
            FULL_TASKS_BUDGET,
        );
    }

    /// A section with nothing omitted carries no footer and stays
    /// bounded too.
    #[test]
    fn the_full_task_section_without_omissions_has_no_footer() {
        let small = RoundManifest {
            group: "g".to_string(),
            base_branch: None,
            ready: vec![worker("w1", "one task\nbody")],
            not_ready: Vec::new(),
            interaction_points: Vec::new(),
        };
        let section = small.render_full_tasks();
        assert_eq!(
            section,
            "FULL TASKS OF THE ROUND'S WORKERS (judge each worker's diff against the worker's own text below, as amended by the orchestrator steers that follow it; the manifest above keeps only each task's first line):\n### w1:\none task\nbody\n",
            "{section}"
        );
        assert!(section.len() <= FULL_TASKS_BUDGET);
    }

    /// The per-worker bound holds exactly at the budget, and a cut
    /// never splits a code point: a task one byte over keeps its
    /// marker, and a multi-byte task is cut before the code point,
    /// not inside it.
    #[test]
    fn a_task_is_bounded_per_worker_without_splitting_a_code_point() {
        let exact = "a".repeat(FULL_TASK_BUDGET);
        assert_eq!(bound_task(&exact, FULL_TASK_BUDGET), exact);
        let over = format!("{exact}b");
        let bounded = bound_task(&over, FULL_TASK_BUDGET);
        assert!(
            bounded.ends_with(TASK_TRUNCATION_MARKER),
            "an oversized task must be marked: {bounded}"
        );
        assert!(bounded.len() <= FULL_TASK_BUDGET, "{bounded}");

        // A 3-byte code point straddling the cut: the head ends
        // before it, so the result is whole characters plus the
        // marker, never a split code point.
        let head = "a".repeat(FULL_TASK_BUDGET - TASK_TRUNCATION_MARKER.len() - 2);
        let multi = format!("{head}{}", "\u{2022}".repeat(6));
        assert_eq!(
            bound_task(&multi, FULL_TASK_BUDGET),
            format!("{head}{TASK_TRUNCATION_MARKER}"),
            "the cut must land before the bullet, not inside it"
        );
    }
}
