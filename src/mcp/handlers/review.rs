use super::*;

impl McpServer {
    /// `review` action: one compact view of a worker's branch, ending with the
    /// command that acts on it.
    ///
    /// Reviewing used to mean `status`, plus `collect` (the whole diff, however
    /// large), plus `logs`, plus a hand-run `git merge-tree`. This is the same
    /// answer in one bounded payload: the task's first line, what verification
    /// said, the per-file diff stat, the revision, and whether the branch still
    /// merges cleanly into the base branch tip. Read-only — unlike `collect` it
    /// never evicts the worker.
    pub(super) async fn handle_review(
        &self,
        args: &Value,
        ctx: &crate::mcp::server::ConnectionContext,
    ) -> Result<Value> {
        let wid = self.resolve_worker_id(args, "review", ctx).await?;
        self.require_owner(&wid, ctx).await?;
        let scope = Self::get_review_diff_scope(args)?;
        let state = self.pool.get_worker_state(&wid).await;
        let entry = crate::pool::load_registry_entry_in(self.pool.scratch_root(), &wid);
        if state.is_none() && entry.is_none() {
            anyhow::bail!("Worker not found: {wid}");
        }
        let (summary, verified, state_branch, report) = completed_fields(state.as_ref());
        let report = report.or_else(|| entry.as_ref().and_then(|entry| entry.report.clone()));
        let branch = state_branch.unwrap_or_else(|| format!("worker-{wid}"));
        // The registry row is the only cross-process record of where the
        // worker's repository is and which branch it integrates with.
        let repo = entry
            .as_ref()
            .and_then(|entry| entry.repo_path.as_deref())
            .map(std::path::Path::new)
            .filter(|path| path.is_dir())
            .map(std::path::Path::to_path_buf);
        let base_branch = entry
            .as_ref()
            .and_then(|entry| entry.base_branch.clone())
            .or_else(|| {
                repo.as_deref()
                    .and_then(crate::pool::revision::detect_base_branch)
            });
        // A live worker's diff is the exact text it produced; a collected one
        // has only its branch left, so its change is measured from that.
        let live_diff = match &state {
            Some(crate::pool::WorkerState::Completed { diff, .. })
            | Some(crate::pool::WorkerState::Exhausted { diff, .. }) => Some(diff.clone()),
            _ => None,
        };
        let probe = repo
            .clone()
            .zip(base_branch.clone())
            .map(|(repo, base)| (repo, base, branch.clone()));
        let (stats, summaries, merge, raw_diff) = match tokio::task::spawn_blocking(move || {
            let Some((repo, base, branch)) = probe else {
                // No repository recorded: a live worker still carries its own
                // diff, so the change can be classified even without git.
                return match live_diff {
                    Some(diff) => {
                        let summaries = diff_file_summaries(&diff);
                        let stats = diff_file_stats(&diff);
                        (stats, summaries, None, diff)
                    }
                    None => (Vec::new(), Vec::new(), None, String::new()),
                };
            };
            let merge = merge_check(&repo, &base, &branch);
            let text = match &live_diff {
                Some(diff) => diff.clone(),
                None => branch_diff(&repo, &base, &branch),
            };
            let stats = match &live_diff {
                Some(diff) => diff_file_stats(diff),
                None => branch_file_stats(&repo, &base, &branch),
            };
            let summaries = diff_file_summaries(&text);
            (stats, summaries, merge, text)
        })
        .await
        {
            Ok(probed) => probed,
            Err(err) => {
                tracing::warn!("review probe for worker {wid} could not run: {err}");
                (Vec::new(), Vec::new(), None, String::new())
            }
        };
        // A worker that never verified is exactly the one whose verify output
        // the orchestrator needs; a verified one has nothing to show.
        let verify_tail = if verified == Some(true) {
            None
        } else {
            self.pool
                .get_worker_logs(&wid)
                .await
                .and_then(|logs| crate::mcp::events::verify_tail_of(&logs.tail(VERIFY_TAIL_STEPS)))
        };
        // Test files are summarised rather than shown, so the diff stays code
        // by default; `all` shows everything and `none` withholds it.
        let shown = match scope {
            ReviewDiffScope::None => String::new(),
            ReviewDiffScope::All => raw_diff,
            ReviewDiffScope::Code => diff_of_kind(&raw_diff, PathKind::Code),
        };
        let diff = match scope {
            ReviewDiffScope::None => Value::Null,
            _ => Value::String(crate::agent::truncate_output(&shown)),
        };
        Ok(json!({
            "worker_id": wid,
            "owner": self.owner_of(&wid).await,
            "task": first_line(entry.as_ref().map(|entry| entry.task.as_str()).unwrap_or_default()),
            "state": state_name(state.as_ref(), entry.as_ref()),
            "verified": verified,
            "verify_tail": verify_tail,
            "approved": entry.as_ref().and_then(|entry| entry.approved.clone()),
            "diff_scope": scope.name(),
            "diff": diff,
            "diff_stat": diff_stat_value(&stats),
            "test_files": test_files_value(&summaries),
            "docs": docs_value(&summaries),
            "summary": summary,
            "report": report,
            "revision": revision_of(state.as_ref(), entry.as_ref()),
            "branch": branch,
            "merge": merge,
            "next_command": if let Some(turns) = exhausted_turns(state.as_ref(), entry.as_ref()) {
                crate::pool::exhausted_continue_command(&wid, turns)
            } else {
                next_command(&wid, &branch, merge.as_ref())
            },
        }))
    }

    /// Parse the optional `diff` argument of `review`: which part of the change
    /// to show. Defaults to the code diff, which is what a reviewer reads.
    pub(super) fn get_review_diff_scope(args: &Value) -> Result<ReviewDiffScope> {
        match args.get("diff") {
            None => Ok(ReviewDiffScope::Code),
            Some(value) => match value.as_str() {
                Some("code") => Ok(ReviewDiffScope::Code),
                Some("all") => Ok(ReviewDiffScope::All),
                Some("none") => Ok(ReviewDiffScope::None),
                _ => anyhow::bail!("'diff' must be one of: code, all, none"),
            },
        }
    }

    /// `approve` action: the orchestrator signs off a completed worker.
    ///
    /// Owner-only like `collect`. The verdict is written to the registry row so
    /// it outlives the in-memory record `collect` evicts; steering the worker
    /// into a new revision drops it, because a changed branch needs a new
    /// review.
    pub(super) async fn handle_approve(
        &self,
        args: &Value,
        ctx: &crate::mcp::server::ConnectionContext,
    ) -> Result<Value> {
        let wid = self.resolve_worker_id(args, "approve", ctx).await?;
        self.require_owner(&wid, ctx).await?;
        let note = args
            .get("message")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|note| !note.is_empty())
            .map(str::to_owned);
        let approved = self.pool.approve(&wid, note).await?;
        Ok(json!({
            "worker_id": wid,
            "status": "approved",
            "approved": approved,
        }))
    }

    /// `unapprove` action: withdraw a completed worker's approval.
    pub(super) async fn handle_unapprove(
        &self,
        args: &Value,
        ctx: &crate::mcp::server::ConnectionContext,
    ) -> Result<Value> {
        let wid = self.resolve_worker_id(args, "unapprove", ctx).await?;
        self.require_owner(&wid, ctx).await?;
        self.pool.unapprove(&wid).await?;
        Ok(json!({
            "worker_id": wid,
            "status": "unapproved",
            "approved": Value::Null,
        }))
    }
}

/// Which part of a worker's change `review` returns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ReviewDiffScope {
    /// Only the code diff; test files are summarised and docs listed.
    Code,
    /// The whole diff, test and doc churn included.
    All,
    /// No diff at all.
    None,
}

impl ReviewDiffScope {
    /// The wire name of the scope.
    fn name(self) -> &'static str {
        match self {
            Self::Code => "code",
            Self::All => "all",
            Self::None => "none",
        }
    }
}

/// How many step logs a review looks back through for a failed verify.
const VERIFY_TAIL_STEPS: usize = 8;

/// The fields a finished worker carries: its summary, whether the gate
/// verified it (always `None` for an exhausted worker, which never verified),
/// and the branch it leaves behind.
pub(super) fn completed_fields(
    state: Option<&crate::pool::WorkerState>,
) -> (
    Option<String>,
    Option<bool>,
    Option<String>,
    Option<crate::pool::WorkerReport>,
) {
    match state {
        Some(crate::pool::WorkerState::Completed {
            summary,
            verified,
            branch,
            report,
            ..
        }) => (
            Some(summary.clone()),
            *verified,
            branch.clone(),
            report.clone(),
        ),
        Some(crate::pool::WorkerState::Exhausted {
            summary,
            branch,
            report,
            ..
        }) => (Some(summary.clone()), None, branch.clone(), report.clone()),
        _ => (None, None, None, None),
    }
}

/// The turn count of an exhausted worker, from its live state or its registry
/// row; `None` for a worker that did not stop on its turn budget.
fn exhausted_turns(
    state: Option<&crate::pool::WorkerState>,
    entry: Option<&crate::pool::WorkerRegistryEntry>,
) -> Option<usize> {
    match state {
        Some(crate::pool::WorkerState::Exhausted { turns, .. }) => Some(*turns),
        _ => entry
            .filter(|entry| entry.status == crate::pool::RegistryStatus::Exhausted)
            .map(|entry| entry.step),
    }
}

/// The lifecycle name of a worker, from its live state when it still has one
/// and from its registry row otherwise.
pub(super) fn state_name(
    state: Option<&crate::pool::WorkerState>,
    entry: Option<&crate::pool::WorkerRegistryEntry>,
) -> &'static str {
    match state {
        Some(crate::pool::WorkerState::Running { .. }) => "Running",
        Some(crate::pool::WorkerState::Paused { .. }) => "Paused",
        Some(crate::pool::WorkerState::Completed { .. }) => "Completed",
        Some(crate::pool::WorkerState::Failed { .. }) => "Failed",
        Some(crate::pool::WorkerState::Exhausted { .. }) => "Exhausted",
        None => entry.map_or("Unknown", |entry| entry.status.display_name()),
    }
}

/// The revision a worker reached: the live state carries it, and a collected
/// worker's registry row is the only other record of it.
pub(super) fn revision_of(
    state: Option<&crate::pool::WorkerState>,
    entry: Option<&crate::pool::WorkerRegistryEntry>,
) -> usize {
    match state {
        Some(crate::pool::WorkerState::Completed { revision, .. })
        | Some(crate::pool::WorkerState::Failed { revision, .. })
        | Some(crate::pool::WorkerState::Exhausted { revision, .. }) => *revision,
        _ => entry.map_or(0, |entry| entry.revision),
    }
}

/// The command that acts on this review: merge a clean branch, or send the
/// conflicts back to the worker that owns them.
pub(super) fn next_command(wid: &str, branch: &str, merge: Option<&MergeCheck>) -> String {
    let Some(merge) = merge else {
        // No answer was possible, so the merge itself still has to be checked.
        return format!("git merge-tree --write-tree <base-branch> {branch}");
    };
    if merge.clean == Some(true) {
        return format!("git merge {branch}");
    }
    if !merge.conflicts.is_empty() {
        return format!(
            "mini-swe-mcp steer {wid} \"resolve the merge conflicts with {}: {}\"",
            merge.base_branch,
            merge.conflicts.join(", ")
        );
    }
    format!("git merge {branch}")
}

/// Whether `branch` still merges cleanly into the tip of `base_branch`.
///
/// `git merge-tree --write-tree` (git >= 2.38) performs the merge in memory: it
/// writes the resulting tree object and answers through its exit status, so the
/// probe touches no worktree, no index and no lock — whichever way it answers,
/// nothing on disk changes. Exit 0 is a clean merge; exit 1 lists the conflicted
/// files after the tree oid; anything else means git refused (an unknown option
/// on an older git, a ref that does not exist).
pub(super) fn merge_check(
    repo: &std::path::Path,
    base_branch: &str,
    branch: &str,
) -> Option<MergeCheck> {
    let mut check = MergeCheck {
        base_branch: base_branch.to_string(),
        clean: None,
        conflicts: Vec::new(),
        error: None,
    };
    let output = match crate::worktree::git(
        repo,
        "merge-tree",
        &["merge-tree", "--write-tree", base_branch, branch],
    ) {
        Ok(output) => output,
        Err(err) => {
            check.error = Some(err.to_string());
            return Some(check);
        }
    };
    match output.status.code() {
        Some(0) => check.clean = Some(true),
        Some(1) => {
            check.clean = Some(false);
            // The conflicted file list follows the tree oid, one
            // `<mode> <oid> <stage>\t<path>` line per stage of a path, and
            // stops at the blank line that introduces the informational
            // messages. A path therefore appears once per stage it conflicts
            // in, so the list is deduplicated in the order git reported it.
            for path in String::from_utf8_lossy(&output.stdout)
                .lines()
                .skip(1)
                .take_while(|line| !line.is_empty())
                .filter_map(|line| line.rsplit('\t').next().map(str::to_string))
            {
                if !check.conflicts.contains(&path) {
                    check.conflicts.push(path);
                }
            }
        }
        _ => {
            check.error = Some(format!(
                "git merge-tree --write-tree failed: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            ));
        }
    }
    Some(check)
}

/// The answer to "does this branch still merge into the base branch tip?".
#[derive(Debug, serde::Serialize)]
pub(super) struct MergeCheck {
    pub(super) base_branch: String,
    /// `None` when git could not answer at all.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) clean: Option<bool>,
    pub(super) conflicts: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) error: Option<String>,
}

/// The merge-base commit `branch` shares with `base_branch`, or `None` when
/// git cannot name one.
pub(super) fn merge_base(
    repo: &std::path::Path,
    base_branch: &str,
    branch: &str,
) -> Option<String> {
    let output =
        crate::worktree::git(repo, "merge-base", &["merge-base", base_branch, branch]).ok()?;
    if !output.status.success() {
        return None;
    }
    let base = String::from_utf8_lossy(&output.stdout).trim().to_string();
    (!base.is_empty()).then_some(base)
}

/// Per-file diff stat of `branch` against its merge-base with `base_branch`.
///
/// The branch is what the orchestrator merges, so it is measured against the
/// base tip it will land on — the same range `git diff --shortstat` reports.
pub(super) fn branch_file_stats(
    repo: &std::path::Path,
    base_branch: &str,
    branch: &str,
) -> Vec<DiffFileStat> {
    let Some(base) = merge_base(repo, base_branch, branch) else {
        return Vec::new();
    };
    let range = format!("{base}...{branch}");
    let Ok(output) = crate::worktree::git(repo, "diff --numstat", &["diff", "--numstat", &range])
    else {
        return Vec::new();
    };
    if !output.status.success() {
        return Vec::new();
    }
    parse_numstat(&String::from_utf8_lossy(&output.stdout))
}

/// The whole unified diff of `branch` against its merge-base with
/// `base_branch`, so a collected worker's code diff can still be shown.
pub(super) fn branch_diff(repo: &std::path::Path, base_branch: &str, branch: &str) -> String {
    let Some(base) = merge_base(repo, base_branch, branch) else {
        return String::new();
    };
    let range = format!("{base}...{branch}");
    match crate::worktree::git(repo, "diff", &["diff", &range]) {
        Ok(output) if output.status.success() => {
            String::from_utf8_lossy(&output.stdout).into_owned()
        }
        _ => String::new(),
    }
}

/// Read `git diff --numstat` output: `<added>\t<deleted>\t<path>` per line,
/// with `-` for a binary file, which counts as neither added nor deleted.
pub(super) fn parse_numstat(text: &str) -> Vec<DiffFileStat> {
    text.lines()
        .filter_map(|line| {
            let mut fields = line.split('\t');
            let insertions = numstat_count(fields.next()?)?;
            let deletions = numstat_count(fields.next()?)?;
            let path = fields.next()?;
            Some(DiffFileStat {
                path: normalize_diff_path(path),
                insertions,
                deletions,
            })
        })
        .collect()
}

/// One `--numstat` count: a number, or `0` for the `-` a binary file carries.
pub(super) fn numstat_count(field: &str) -> Option<usize> {
    Some(if field == "-" {
        0
    } else {
        field.parse::<usize>().ok()?
    })
}

/// One file's share of a diff, as `git diff --numstat` reports it. The shared
/// [`crate::pool::FileStat`] keeps the review payload and the completion event
/// reading the same shape.
pub(super) type DiffFileStat = crate::pool::FileStat;

/// Which part of a change a path belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PathKind {
    Code,
    Test,
    Doc,
}

/// Directory components that mark a file as a test wherever it lives.
const TEST_DIRS: &[&str] = &["tests", "test", "__tests__", "spec"];
/// Filename tails that mark a file as a test whatever its directory.
const TEST_NAME_TAILS: &[&str] = &["_test.", "_spec.", ".test.", ".spec."];
/// Directory components that mark a file as documentation.
const DOC_DIRS: &[&str] = &["docs"];
/// Filename tails that mark a file as documentation.
const DOC_EXTENSIONS: &[&str] = &[".md", ".rst", ".txt"];

/// Classify one diff path as code, test or doc.
///
/// One small table, applied in order: a test directory or a `*_test.*` style
/// name wins over a documentation extension, so `tests/README.md` is a test.
pub(super) fn classify_path(path: &str) -> PathKind {
    let file = path.rsplit('/').next().unwrap_or(path);
    let has_dir = |dir: &str| path.split('/').any(|part| part == dir);
    if TEST_DIRS.iter().any(|dir| has_dir(dir))
        || TEST_NAME_TAILS.iter().any(|tail| file.contains(tail))
        || (file.starts_with("test_") && file.ends_with(".py"))
    {
        return PathKind::Test;
    }
    if DOC_DIRS.iter().any(|dir| has_dir(dir))
        || DOC_EXTENSIONS.iter().any(|ext| file.ends_with(ext))
    {
        return PathKind::Doc;
    }
    PathKind::Code
}

/// Declarations that each start one test case, language-agnostically.
///
/// `#[test]`/`#[tokio::test]` are the Rust forms, `fn test_`/`def test_` the
/// function forms, `it(`/`test(`/`describe(` the JS ones and `@Test`/
/// `func Test` the JVM and Swift/Go ones.
const TEST_CASE_PATTERNS: &[&str] = &[
    "#[test]",
    "#[tokio::test]",
    "fn test_",
    "def test_",
    "it(",
    "test(",
    "describe(",
    "@Test",
    "func Test",
];

/// How many test cases one changed line declares.
pub(super) fn count_test_cases(line: &str) -> usize {
    TEST_CASE_PATTERNS
        .iter()
        .filter(|pattern| contains_at_word_start(line, pattern))
        .count()
}

/// Whether `line` contains `pattern` at a word start, so `it(` does not match
/// inside an identifier such as `unit(`.
pub(super) fn contains_at_word_start(line: &str, pattern: &str) -> bool {
    let mut from = 0;
    while let Some(at) = line[from..].find(pattern) {
        let idx = from + at;
        let boundary = idx == 0
            || !line[..idx]
                .chars()
                .next_back()
                .is_some_and(|c| c.is_alphanumeric() || c == '_');
        if boundary {
            return true;
        }
        from = idx + pattern.len();
    }
    false
}

/// One file's share of a diff, including the test cases its changed lines
/// declare and how the path classifies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct DiffFileSummary {
    pub(super) path: String,
    pub(super) insertions: usize,
    pub(super) deletions: usize,
    pub(super) added_cases: usize,
    pub(super) removed_cases: usize,
    pub(super) kind: PathKind,
}

/// Per-file summaries of a unified diff.
///
/// Only hunk lines are counted, and a hunk starts at its `@@` header, so the
/// `---`/`+++` headers and an added line that itself starts with `+` are never
/// mistaken for a change.
pub(super) fn diff_file_summaries(diff: &str) -> Vec<DiffFileSummary> {
    crate::pool::diff_sections_of(diff)
        .into_iter()
        .map(|(path, section)| {
            let kind = classify_path(&path);
            let mut insertions = 0;
            let mut deletions = 0;
            let mut added_cases = 0;
            let mut removed_cases = 0;
            let mut in_hunks = false;
            for line in section.lines() {
                if line.starts_with("@@") {
                    in_hunks = true;
                } else if in_hunks && line.starts_with('+') {
                    insertions += 1;
                    added_cases += count_test_cases(line.get(1..).unwrap_or_default());
                } else if in_hunks && line.starts_with('-') {
                    deletions += 1;
                    removed_cases += count_test_cases(line.get(1..).unwrap_or_default());
                }
            }
            DiffFileSummary {
                path,
                insertions,
                deletions,
                added_cases,
                removed_cases,
                kind,
            }
        })
        .collect()
}

/// Per-file `(path, insertions, deletions)` of a unified diff.
pub(super) fn diff_file_stats(diff: &str) -> Vec<DiffFileStat> {
    crate::pool::file_stats_of_diff(diff)
}

/// The sections of `diff` whose path classifies as `kind`, rejoined.
pub(super) fn diff_of_kind(diff: &str, kind: PathKind) -> String {
    crate::pool::diff_sections_of(diff)
        .into_iter()
        .filter(|(path, _)| classify_path(path) == kind)
        .map(|(_, section)| section)
        .collect()
}

/// The `test_files` payload: one entry per test file, naming the test cases
/// its added and removed lines declare.
pub(super) fn test_files_value(summaries: &[DiffFileSummary]) -> Value {
    Value::Array(
        summaries
            .iter()
            .filter(|summary| summary.kind == PathKind::Test)
            .map(|summary| {
                json!({
                    "path": summary.path,
                    "added_cases": summary.added_cases,
                    "removed_cases": summary.removed_cases,
                })
            })
            .collect(),
    )
}

/// The `docs` payload: one entry per documentation file with its +/ counts.
pub(super) fn docs_value(summaries: &[DiffFileSummary]) -> Value {
    Value::Array(
        summaries
            .iter()
            .filter(|summary| summary.kind == PathKind::Doc)
            .map(|summary| {
                json!({
                    "path": summary.path,
                    "insertions": summary.insertions,
                    "deletions": summary.deletions,
                })
            })
            .collect(),
    )
}

/// The sections of `diff` that touch any of `files`, rejoined.
pub(super) fn diff_of_files(diff: &str, files: &[String]) -> String {
    crate::pool::diff_sections_of(diff)
        .into_iter()
        .filter(|(path, _)| {
            files
                .iter()
                .any(|file| crate::pool::same_diff_path(file, path))
        })
        .map(|(_, section)| section)
        .collect()
}

/// The `diff_stat` payload: the totals plus the per-file counts.
pub(super) fn diff_stat_value(stats: &[DiffFileStat]) -> Value {
    let insertions: usize = stats.iter().map(|stat| stat.insertions).sum();
    let deletions: usize = stats.iter().map(|stat| stat.deletions).sum();
    json!({
        "files": stats.len(),
        "insertions": insertions,
        "deletions": deletions,
        "per_file": stats
            .iter()
            .map(|stat| json!({
                "path": stat.path,
                "insertions": stat.insertions,
                "deletions": stat.deletions,
            }))
            .collect::<Vec<_>>(),
    })
}

pub(in crate::mcp) const DIFF_DESCRIPTION: &str = "Diff scope for 'review': 'code' (default) hides tests, 'all' shows everything, 'none' hides it.";

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::AgentStepLog;
    /// A diff split into sections keeps every file, and the counts come from the
    /// hunks alone: an added line that itself starts with `+` and a removed line
    /// that starts with `--` are content, not headers.
    #[test]
    fn diff_sections_count_hunk_lines_only() {
        let diff = concat!(
            "diff --git a/one.rs b/one.rs\n",
            "index 111..222 100644\n",
            "--- a/one.rs\n",
            "+++ b/one.rs\n",
            "@@ -1,3 +1,4 @@\n",
            " context\n",
            "-removed\n",
            "++added line that starts with a plus\n",
            "--removed line that starts with two dashes\n",
            "diff --git a/two.rs b/two.rs\n",
            "new file mode 100644\n",
            "index 000..333\n",
            "--- /dev/null\n",
            "+++ b/two.rs\n",
            "@@ -0,0 +1,2 @@\n",
            "+first\n",
            "+second\n",
        );
        let stats = diff_file_stats(diff);
        assert_eq!(
            stats,
            vec![
                DiffFileStat {
                    path: "one.rs".to_string(),
                    insertions: 1,
                    deletions: 2,
                },
                DiffFileStat {
                    path: "two.rs".to_string(),
                    insertions: 2,
                    deletions: 0,
                },
            ]
        );
        let stat = diff_stat_value(&stats);
        assert_eq!(stat["files"], 2);
        assert_eq!(stat["insertions"], 3);
        assert_eq!(stat["deletions"], 2);
        assert_eq!(stat["per_file"][1]["path"], "two.rs");
    }

    /// A binary file carries no line counts, but it is still a file that changed.
    #[test]
    fn a_binary_file_counts_as_a_file_with_no_lines() {
        let diff = concat!(
            "diff --git a/logo.png b/logo.png\n",
            "index 111..222 100644\n",
            "Binary files a/logo.png and b/logo.png differ\n",
        );
        assert_eq!(
            diff_file_stats(diff),
            vec![DiffFileStat {
                path: "logo.png".to_string(),
                insertions: 0,
                deletions: 0,
            }]
        );
    }

    /// `files` selects sections by path, and a bare name still finds the file
    /// inside a directory.
    #[test]
    fn diff_of_files_matches_a_path_or_a_component_suffix() {
        let diff = concat!(
            "diff --git a/src/a.rs b/src/a.rs\n",
            "--- a/src/a.rs\n",
            "+++ b/src/a.rs\n",
            "@@ -1 +1 @@\n",
            "-old\n",
            "+new\n",
            "diff --git a/b.rs b/b.rs\n",
            "--- a/b.rs\n",
            "+++ b/b.rs\n",
            "@@ -1 +1 @@\n",
            "-old\n",
            "+new\n",
        );
        // Six lines: the `diff --git` header, the two path headers, the hunk
        // header and the two changed lines.
        assert_eq!(
            diff_of_files(diff, &["src/a.rs".to_string()])
                .lines()
                .count(),
            6
        );
        assert_eq!(
            diff_of_files(diff, &["a.rs".to_string()]).lines().count(),
            6
        );
        assert_eq!(
            diff_of_files(diff, &["b.rs".to_string()]).lines().count(),
            6
        );
        assert!(diff_of_files(diff, &["nope.rs".to_string()]).is_empty());
    }

    /// `git diff --numstat` is read as the same per-file stat the in-memory diff
    /// produces, and a binary file (`-`) counts as neither.
    #[test]
    fn numstat_is_read_as_the_same_per_file_stat() {
        assert_eq!(
            parse_numstat("3\t1\tsrc/a.rs\n-\t-\tlogo.png\n"),
            vec![
                DiffFileStat {
                    path: "src/a.rs".to_string(),
                    insertions: 3,
                    deletions: 1,
                },
                DiffFileStat {
                    path: "logo.png".to_string(),
                    insertions: 0,
                    deletions: 0,
                },
            ]
        );
    }

    /// The next command acts on the merge answer: merge a clean branch, send the
    /// conflicts back to the worker that owns them.
    #[test]
    fn next_command_follows_the_merge_answer() {
        let clean = MergeCheck {
            base_branch: "master".to_string(),
            clean: Some(true),
            conflicts: Vec::new(),
            error: None,
        };
        assert_eq!(
            next_command("w1", "worker-w1", Some(&clean)),
            "git merge worker-w1"
        );

        let mut conflicting = clean;
        conflicting.clean = Some(false);
        conflicting.conflicts = vec!["a.rs".to_string(), "b.rs".to_string()];
        assert_eq!(
            next_command("w1", "worker-w1", Some(&conflicting)),
            "mini-swe-mcp steer w1 \"resolve the merge conflicts with master: a.rs, b.rs\""
        );

        let unknown = MergeCheck {
            base_branch: "master".to_string(),
            clean: None,
            conflicts: Vec::new(),
            error: Some("git merge-tree failed".to_string()),
        };
        assert_eq!(
            next_command("w1", "worker-w1", Some(&unknown)),
            "git merge worker-w1"
        );
        assert_eq!(
            next_command("w1", "worker-w1", None),
            "git merge-tree --write-tree <base-branch> worker-w1"
        );
    }

    /// A review carries the first line of the task and the tail of the verify
    /// that failed, never the whole log.
    #[test]
    fn the_review_view_is_bounded() {
        assert_eq!(first_line("Fix the parser\nand its docs"), "Fix the parser");
        assert_eq!(first_line("   \n  padded  "), "padded");
        assert_eq!(first_line("  \n\n"), "");

        let build = AgentStepLog {
            step: 1,
            command: "cargo build".to_string(),
            output: "warning: unused".to_string(),
            exit_code: Some(0),
        };
        let verify = AgentStepLog {
            step: 2,
            command: "[verify] cargo test".to_string(),
            output: "test a ... FAILED\nassertion failed".to_string(),
            exit_code: Some(101),
        };
        let logs = vec![&build, &verify];
        assert_eq!(
            crate::mcp::events::verify_tail_of(&logs).expect("a failed verify must be shown"),
            "test a ... FAILED\nassertion failed"
        );
        assert_eq!(crate::mcp::events::verify_tail_of(&logs[..1]), None);
    }
    /// The path classifier is one table: a test directory or a test-style name
    /// beats a documentation extension, so `tests/README.md` is a test.
    #[test]
    fn classify_path_sorts_code_tests_and_docs() {
        for (path, kind) in [
            ("src/parser.rs", PathKind::Code),
            ("src/main.py", PathKind::Code),
            ("tests/integration.rs", PathKind::Test),
            ("test/cli_test.go", PathKind::Test),
            ("app/__tests__/x.js", PathKind::Test),
            ("spec/models_spec.rb", PathKind::Test),
            ("src/serde_test.rs", PathKind::Test),
            ("scripts/test_smoke.py", PathKind::Test),
            ("web/widget.spec.ts", PathKind::Test),
            ("src/thing.test.ts", PathKind::Test),
            ("README.md", PathKind::Doc),
            ("docs/design.rst", PathKind::Doc),
            ("notes.txt", PathKind::Doc),
            ("tests/README.md", PathKind::Test),
        ] {
            assert_eq!(classify_path(path), kind, "wrong kind for {path}");
        }
    }

    /// The counter is language-agnostic: every documented declaration earns one
    /// case, and an identifier that merely contains a pattern earns none.
    #[test]
    fn count_test_cases_reads_every_declaration() {
        for line in [
            "#[test]",
            "    #[tokio::test]",
            "fn test_parses() {",
            "def test_parses(self):",
            "it(`adds`)",
            "test('adds', () => {})",
            "describe('parser', () => {",
            "    @Test",
            "func TestParse(t *testing.T) {",
        ] {
            assert_eq!(count_test_cases(line), 1, "missed a case in {line:?}");
        }
        for line in [
            "// a comment about tests",
            "fn parses() {",
            "let unit = 1;",
            "let submit = 2;",
        ] {
            assert_eq!(count_test_cases(line), 0, "false positive in {line:?}");
        }
    }

    /// Test-case churn is measured from the changed hunk lines only, and each
    /// summary keeps the path's classification.
    #[test]
    fn diff_summaries_count_cases_and_classify() {
        let diff = concat!(
            "diff --git a/src/a.rs b/src/a.rs\n",
            "--- a/src/a.rs\n",
            "+++ b/src/a.rs\n",
            "@@ -1 +1 @@\n",
            "-old\n",
            "+new\n",
            "diff --git a/tests/a_test.rs b/tests/a_test.rs\n",
            "--- a/tests/a_test.rs\n",
            "+++ b/tests/a_test.rs\n",
            "@@ -1,2 +1,3 @@\n",
            "-#[test]\n",
            "-fn test_old() {}\n",
            "+#[test]\n",
            "+fn test_new() {}\n",
            "+#[test]\n",
        );
        let summaries = diff_file_summaries(diff);
        assert_eq!(summaries.len(), 2);
        assert_eq!(summaries[0].kind, PathKind::Code);
        assert_eq!(summaries[0].added_cases, 0);
        let test = &summaries[1];
        assert_eq!(test.kind, PathKind::Test);
        assert_eq!(test.added_cases, 3);
        assert_eq!(test.removed_cases, 2);
        assert_eq!(diff_of_kind(diff, PathKind::Code).lines().count(), 6);
    }

    /// `review --diff` accepts exactly the three documented scopes and defaults
    /// to the code diff.
    #[test]
    fn review_diff_scope_defaults_to_code_and_rejects_typos() {
        assert_eq!(
            McpServer::get_review_diff_scope(&json!({})).expect("absent is the default"),
            ReviewDiffScope::Code
        );
        assert_eq!(
            McpServer::get_review_diff_scope(&json!({ "diff": "all" })).unwrap(),
            ReviewDiffScope::All
        );
        assert_eq!(
            McpServer::get_review_diff_scope(&json!({ "diff": "none" })).unwrap(),
            ReviewDiffScope::None
        );
        assert!(McpServer::get_review_diff_scope(&json!({ "diff": "everything" })).is_err());
    }
}
