//! The multi-phase review auditor: an independent agent that re-runs the
//! quality gates over the implementation phase's work.
//!
//! Once the implementation loop stops, [`run_review_phase`] checkpoints the
//! worktree in git and then drives a *second* agent (a different model) over
//! the same worktree. The reviewer is deliberately not the implementation
//! agent: its contract is to inspect the diff, run the suite the dispatch
//! named, fix whatever it finds, and only then emit the completion sentinel.
//!
//! Two built-in modes share this engine ([`ReviewMode`]), plus any the
//! manifest declares (`review_modes:` in `models.yaml`):
//!
//! * `quality` — today's generic audit: run the gate, inspect the diff, fix
//!   real defects, re-run the gate.
//! * `security` — an adversarial pass over the same diff. Its checklist is
//!   about what an unprivileged local user, another agent or a lying model
//!   text can do with every new path, socket, file, environment variable and
//!   IPC message. It is selected by the `:security` suffix of `--review-after`
//!   (`review_after`), and it is also what the automatic sensitive-path
//!   trigger runs -- on the mode's default reviewer when the manifest names
//!   one, else on the manifest's strongest tier, since the model that wrote a
//!   sensitive diff must not be the model that audits it ([`ReviewerChoice`]).
//! * A manifest-declared mode — its `checklist` appended to the common review
//!   frame, selected by `--review-after <model>:<mode>`.
//!
//! The security review audits only what has not been audited yet. Every commit
//! an earlier security review approved is recorded on the worker's registry row
//! ([`SecurityScope`]), so a revision — and a consolidator, whose diff is the
//! union of branches each already reviewed — is handed the diff since that
//! commit, not the whole branch again: 22 full reviews of 9 workers become one
//! review per real change. With nothing new since the approval, the review is
//! skipped and the skip is logged rather than paid for.
//!
//! It runs under its own turn budget ([`ReviewPhase::review_max_turns`],
//! resolved from the model manifest) and its own message history, so the
//! reviewer's context never mixes with the implementer's.

use anyhow::Result;
use tracing::info;

use std::fmt::Write as _;
use std::path::Path;

use crate::agent::{AgentRunner, ChatMessage, Role};
use crate::manifest::{ModelManifest, build_system_prompt};
use crate::worktree::WorktreeGuard;

use super::super::WorkerPool;
use super::super::registry::{RegistryStatus, WorkerMeta, WorkerRole};
use super::turn::{LlmErrorPolicy, ProgressWatch, TurnConfig, TurnEngine, TurnOutcome};

/// Which model runs an automatic security review.
///
/// A sensitive-path diff is audited by the *strongest* reviewer the manifest
/// declares, not by whatever model happened to implement it: the fast executor
/// that wrote the change is the one model whose blind spots the audit exists to
/// catch, so reusing it as the reviewer silently downgrades the pass to a
/// self-review.
///
/// The order is fixed and logged (see [`select_security_reviewer`]):
///
/// 1. an explicit `--review-after <model>[:security]` -- the orchestrator's
///    own instruction, which always wins;
/// 2. the manifest's `strongest:` tier, resolved to its id;
/// 3. the dispatch's default model, supplied by the caller as
///    `RunConfig::default_model`.
///
/// A quality review keeps its own rule and is not routed through here: it runs
/// on the requested model, and the sensitive-path upgrade swaps only the mode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ReviewerChoice {
    /// The model id the review phase runs.
    pub(crate) model: String,
    /// Why this model, for the hub log line.
    pub(crate) reason: &'static str,
}

/// Resolve the reviewer for an automatic (no `review_after`) security review.
///
/// An explicit `--review-after` never reaches here: the caller keeps that
/// reviewer and only upgrades its mode, so the orchestrator's own instruction
/// wins over the manifest's tier.
pub(crate) fn select_security_reviewer(
    manifest: &ModelManifest,
    default_model: &str,
) -> ReviewerChoice {
    match manifest.strongest_alias() {
        Some(alias) => ReviewerChoice {
            model: manifest.resolve_model(alias).0,
            reason: "manifest's strongest tier",
        },
        None => ReviewerChoice {
            model: default_model.to_string(),
            reason: "manifest declares no strongest tier; dispatch default",
        },
    }
}

/// Which auditor runs over the finished implementation.
///
/// The default is the historical quality review; the `security` mode is the
/// adversarial variant. The modes differ only in the prompt they hand the
/// reviewer: the engine, the turn budget, the network policy and the worktree
/// are identical, so a security review is not a second worker to reason about.
///
/// A manifest may declare its own review modes (`review_modes:` in
/// `models.yaml`), each with a `checklist` and an optional default `model`.
/// Declaring a mode named `quality` or `security` overrides the built-in
/// prompt; any other name adds a new mode selectable via
/// `--review-after <model>:<mode>`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ReviewMode {
    /// The mode's name: `quality`, `security`, or a manifest-declared mode.
    pub name: String,
    /// The checklist for a manifest-declared mode (or a built-in overridden by
    /// the manifest). `None` for the built-in quality/security modes, which
    /// build their own full prompt.
    pub checklist: Option<String>,
}

impl ReviewMode {
    /// The suffix that selects the adversarial mode in
    /// `--review-after <model>:<mode>`.
    ///
    /// Spelled here and matched in [`ReviewMode::parse_model`] so the CLI, the
    /// MCP arg and the prompt cannot disagree on the spelling.
    pub const SECURITY_SUFFIX: &'static str = "security";

    /// The suffix that selects the generic audit in
    /// `--review-after <model>:<mode>`.
    pub const QUALITY_SUFFIX: &'static str = "quality";

    /// The generic audit: gates, diff, real defects, gates again.
    pub fn quality() -> Self {
        Self {
            name: Self::QUALITY_SUFFIX.to_string(),
            checklist: None,
        }
    }

    /// The adversarial audit: hostile inputs, untrusted model text, crash and
    /// handover windows, deletion proof, test meaning.
    pub fn security() -> Self {
        Self {
            name: Self::SECURITY_SUFFIX.to_string(),
            checklist: None,
        }
    }

    /// Whether this is the adversarial security review (by name).
    ///
    /// A manifest that overrides `security` with its own checklist is still
    /// the security review, so the finding count is still recorded.
    pub fn is_security(&self) -> bool {
        self.name.eq_ignore_ascii_case(Self::SECURITY_SUFFIX)
    }

    /// Split `--review-after <model>[:mode]` into the model and the mode.
    ///
    /// A model id legitimately contains `:` (`combo:nerd`), so only a suffix
    /// equal to a built-in mode (`security`, `quality`) is a mode marker: the
    /// *last* `:` segment is inspected and removed only when it names one.
    /// Everything else — `combo:nerd`, `some/unknown`, a trailing `:` — stays
    /// the model verbatim, which is what keeps today's `--review-after` values
    /// parsing exactly as they did. Manifest-declared modes are resolved by
    /// [`crate::manifest::ModelManifest::parse_review_after`], which has the
    /// catalog to validate them against.
    pub fn parse_model(requested: &str) -> (String, Self) {
        let trimmed = requested.trim();
        match trimmed.rsplit_once(':') {
            Some((model, suffix))
                if !model.is_empty() && suffix.eq_ignore_ascii_case(Self::SECURITY_SUFFIX) =>
            {
                (model.to_string(), Self::security())
            }
            Some((model, suffix))
                if !model.is_empty() && suffix.eq_ignore_ascii_case(Self::QUALITY_SUFFIX) =>
            {
                (model.to_string(), Self::quality())
            }
            _ => (trimmed.to_string(), Self::quality()),
        }
    }

    /// The name a status line, an event or a log prints.
    pub fn as_str(&self) -> &str {
        &self.name
    }

    /// Parse `--review-after <model>[:<mode>]` against the manifest's declared
    /// modes.
    ///
    /// A trailing `:<mode>` names the mode when `<mode>` is available
    /// (a built-in or a manifest-declared `review_modes:` entry); the mode's
    /// checklist is resolved here, and an empty model part falls back to the
    /// mode's declared default reviewer. A string with no mode suffix is the
    /// reviewer with the default `quality` mode. A `:<suffix>` that names no
    /// available mode is a dispatch error listing the available ones — unless
    /// the whole string is a known model (an alias or an id), in which case it
    /// is that model with the default mode, which is what keeps a
    /// colon-containing model id like `combo:nerd` parsing as a model.
    pub fn parse_with_manifest(
        requested: &str,
        manifest: &crate::manifest::ModelManifest,
    ) -> anyhow::Result<(String, Self)> {
        let trimmed = requested.trim();
        if let Some((model, suffix)) = trimmed.rsplit_once(':')
            && !suffix.trim().is_empty()
        {
            let mode_name = suffix.trim();
            if manifest.is_review_mode(mode_name) {
                let mode = Self::resolve_declared(mode_name, manifest);
                // An empty model part uses the mode's default reviewer; a mode
                // without one falls back to the empty string, which the caller
                // resolves against the implementer's model.
                let reviewer = if model.trim().is_empty() {
                    mode_default_reviewer(mode_name, manifest).unwrap_or_default()
                } else {
                    model.trim().to_string()
                };
                return Ok((reviewer, mode));
            }
            // Not a declared mode: if the whole string is a known model, it is
            // the reviewer with the default mode. Otherwise the suffix was
            // meant as a mode and it does not exist.
            if !is_known_model(trimmed, manifest) {
                anyhow::bail!(
                    "unknown review mode \"{mode_name}\"; available modes: {}",
                    manifest.available_review_modes().join(", ")
                );
            }
        }
        Ok((trimmed.to_string(), Self::quality()))
    }

    /// Resolve a declared mode name to its [`ReviewMode`], filling in the
    /// manifest's checklist when one is declared.
    ///
    /// A manifest entry named `quality` or `security` overrides the built-in
    /// prompt; any other declared name builds a custom mode around its
    /// checklist.
    pub fn resolve_declared(name: &str, manifest: &crate::manifest::ModelManifest) -> Self {
        if let Some(def) = manifest.review_mode(name) {
            return Self {
                name: name.to_string(),
                checklist: Some(def.checklist.clone()),
            };
        }
        if name.eq_ignore_ascii_case(Self::SECURITY_SUFFIX) {
            return Self::security();
        }
        Self::quality()
    }
}

/// The default reviewer a manifest-declared mode names, if any.
pub fn mode_default_reviewer(
    name: &str,
    manifest: &crate::manifest::ModelManifest,
) -> Option<String> {
    manifest
        .review_mode(name)
        .and_then(|def| def.model.clone())
        .map(|m| m.trim().to_string())
        .filter(|m| !m.is_empty())
}

/// Whether `requested` names a model the manifest knows: an alias, or a full
/// id owned by some alias.
fn is_known_model(requested: &str, manifest: &crate::manifest::ModelManifest) -> bool {
    if manifest.models.contains_key(requested) {
        return true;
    }
    manifest
        .models
        .values()
        .any(|def| def.id.trim() == requested)
}

/// Build the reviewer's opening message.
///
/// `verify` is the *dispatch's* completion gate (or the auto-detected cheap
/// gate of a consolidated round), never a language-specific command invented
/// here: a Go, Python or Rust repository is reviewed by running the suite the
/// dispatch already declared. When the dispatch disabled verification there is
/// no gate to re-run, and the prompt says so instead of naming one.
///
/// `sensitive` names the diff's touched files the repository declared
/// sensitive, which is why the security review was triggered; it is empty for
/// a quality review or for a security review the orchestrator asked for by
/// hand.
pub fn review_prompt(
    mode: &ReviewMode,
    task: &str,
    verify: Option<&str>,
    sensitive: &[String],
) -> String {
    let gate = verify
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .unwrap_or(DISABLED_GATE);
    // A manifest-declared checklist overrides the built-in prompt of the same
    // name; any other declared mode runs the common review frame with its own
    // focus instructions appended.
    if let Some(checklist) = mode.checklist.as_deref() {
        return custom_prompt(&mode.name, task, gate, checklist);
    }
    if mode.is_security() {
        return security_prompt(task, gate, sensitive);
    }
    quality_prompt(task, gate)
}

/// What the prompt says when the dispatch disabled the completion gate.
///
/// Naming no command is the honest answer: a reviewer that invented a suite
/// would run one the dispatch never agreed to.
const DISABLED_GATE: &str = "(none: this dispatch disabled the completion gate)";

/// The generic audit: gates, diff, real defects, gates again.
fn quality_prompt(task: &str, gate: &str) -> String {
    format!(
        "AUDIT & REVIEW PHASE:\nThe previous subagent implemented the following task:\n{}\n\n\
        YOUR OBJECTIVE AS THE INDEPENDENT REVIEWER:\n\
        1. First run the completion gate on the checkpoint and see it pass: `{}`. Then run the tests of the files you touch.\n\
        2. Inspect the whole diff since the base commit plus the working tree: run `git status`, `git diff HEAD~1` (or `git log -1 -p`) and `git diff`.\n\
        3. Fix real problems only: regressions, edge cases, dead code, orphan imports, or missed requirements.\n\
        4. Re-run the gate and the tests you touched and see them pass before completing.\n\
        5. When verified and 100% clean, execute:\n\
           echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
        task, gate
    )
}

/// The adversarial audit.
///
/// Every item is a question the diff has to answer, not a style preference:
/// the three defects this mode exists to catch (a predictable socket
/// directory with no owner check, a retirement that deleted a completed
/// worker's branch on a report that said "fixed", a test that mirrored the
/// bug it should catch) are all instances of one of them.
fn security_prompt(task: &str, gate: &str, sensitive: &[String]) -> String {
    let mut prompt = String::from(
        "ADVERSARIAL SECURITY REVIEW PHASE:\n\
         The previous subagent implemented the following task:\n",
    );
    let _ = write!(prompt, "{task}\n\n");
    if !sensitive.is_empty() {
        let _ = write!(
            prompt,
            "This diff touches paths the repository declared sensitive: {}.\n\n",
            sensitive.join(", ")
        );
    }
    prompt.push_str(
        "YOUR OBJECTIVE AS THE ADVERSARIAL REVIEWER:\n\
         Assume the diff is hostile until you have proved otherwise. Work through the checklist below against the *actual* diff (`git status`, `git diff HEAD~1` or `git log -1 -p`, `git diff`), and judge it as an attacker who is already an unprivileged local user or another agent in the same harness.\n\
         1. NEW RESOURCE: for every path, socket, file, directory, environment variable, lock and IPC message the diff creates or opens -- who else can reach it? Check ownership, permissions (0600/0700 vs world-writable), predictable names (`/tmp/<fixed>`, a pid-less, random-less path), time-of-check/time-of-use races, symlink and hardlink tricks, and a name an attacker can pre-create. A temporary path must be created exclusively with the right owner and mode, and verified, not assumed.\n\
         2. UNTRUSTED TEXT: model-written text (a report, a summary, a question, a verdict like \"fixed\" or \"merged\") is data, never proof. Find every decision that deletes, merges, retires, routes or grants on the strength of such text and ask what a lying or stale value would do. A destructive branch needs positive evidence (a git object, an exit status, a file that really exists), not a word.\n\
         3. CRASH AND HANDOVER: for each new operation, what is left behind if the process is killed, the daemon hands over, the worker is revised or the network drops in the middle? Look for a guard, a lock, a permit, a temp file or a job slot that is only released on the success path, and for a half-written state a later pass would trust.\n\
         4. DELETION: what exactly does each new deletion, retirement or cleanup remove, and what positive proof gates it? A path that can delete an unmerged branch, a worktree, a report or a registry row on an absence of evidence is a defect.\n\
         5. TEST MEANING: would each new or changed test fail if the code were wrong? A test that mirrors the implementation (asserts the same literal, the same branch, the same constant), that cannot fail, or that only asserts a happy path, is not a regression test.\n\
         6. FIX, DO NOT LIST AWAY: fix every real defect you find, with a regression test that fails without the fix. Run the completion gate `");
    prompt.push_str(gate);
    prompt.push_str(
        "` and the tests of the files you touched, and see them pass.\n\
         7. Report honestly: in your REPORT block list every finding you did NOT fix in the `risks:` line, one line each. Fixing nothing real is a valid outcome; claiming a clean bill of health you did not check is not.\n\
         8. Before the completion sentinel, print a line `FINDINGS: <n>` giving the total number of findings you found, fixed or listed. Print `FINDINGS: 0` when you found none; the harness shows this count in the completion event and the status.\n\
         9. When the gate and the tests pass and every finding is either fixed or listed, execute:\n\
            echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
    );
    prompt
}

/// A manifest-declared review mode: the common review frame with the mode's
/// own focus instructions appended.
///
/// The frame is the same contract every reviewer honours — inspect the diff,
/// run the dispatch's verify gate, fix real defects with a regression test,
/// list unfixed findings in REPORT risks — and the `checklist` is what the
/// manifest author added on top of it.
fn custom_prompt(name: &str, task: &str, gate: &str, checklist: &str) -> String {
    format!(
        "REVIEW PHASE ({name}):\nThe previous subagent implemented the following task:\n{task}\n\n\
        YOUR OBJECTIVE AS THE INDEPENDENT REVIEWER:\n\
        1. First run the completion gate on the checkpoint and see it pass: `{gate}`. Then run the tests of the files you touch.\n\
        2. Inspect the whole diff since the base commit plus the working tree: run `git status`, `git diff HEAD~1` (or `git log -1 -p`) and `git diff`.\n\
        3. Focus instructions for this review mode:\n{checklist}\n\
        4. Fix real problems only: regressions, edge cases, dead code, orphan imports, or missed requirements, with a regression test that fails without the fix.\n\
        5. Re-run the gate and the tests you touched and see them pass before completing.\n\
        6. In your REPORT block list every finding you did NOT fix in the `risks:` line, one line each.\n\
        7. When verified and 100% clean, execute:\n\
           echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
    )
}

impl SecurityScope {
    /// The commits this review adds beyond what an earlier security review
    /// already approved, oldest first. Empty for a [`SecurityScope::Full`]
    /// review, which is measured against the worker's base commit rather than
    /// against a list, and for a scope whose review must be skipped.
    pub fn reviewed_commits(&self) -> Vec<String> {
        match self {
            Self::Full => Vec::new(),
            Self::Since { commits, .. } => commits.clone(),
        }
    }

    /// The commits earlier security reviews approved, oldest first, which the
    /// reviewer is given for context so it neither re-reviews them nor mistakes
    /// them for new code.
    pub fn approved_commits(&self) -> Vec<String> {
        match self {
            Self::Full => Vec::new(),
            Self::Since { approved, .. } => approved.clone(),
        }
    }

    /// The files this review has to judge: the paths the covered commits
    /// touched, or the consolidator's own commits' paths. Empty for a full
    /// scope, whose files are measured against the worker's base commit by the
    /// caller that owns it.
    pub async fn reviewed_files(&self, repo: &Path) -> Vec<String> {
        match self {
            Self::Full => Vec::new(),
            // A consolidator's scope is its own commits, so its files are too:
            // measuring `base..branch` would name every merged worker's files,
            // which is exactly the re-audit this scope exists to avoid.
            Self::Since {
                merged,
                branch,
                commits,
                ..
            } if !merged.is_empty() && !commits.is_empty() => own_files(repo, branch, merged).await,
            Self::Since { base, branch, .. } => files_since(repo, base, branch).await,
        }
    }

    /// The diff of exactly what this review covers: the changes its own commits
    /// made, which is what a consolidator's security review sees of its own
    /// work -- never the merged worker branches it did not write.
    pub async fn reviewed_diff(&self, repo: &Path) -> String {
        match self {
            Self::Full => String::new(),
            Self::Since {
                commits, merged, ..
            } if commits.is_empty() => String::new(),
            // A consolidator reviews the patch of its own commits only: the
            // merges that carried the reviewed worker branches in are not its
            // work, and handing them to the reviewer would audit those branches
            // a second time.
            Self::Since {
                commits, merged, ..
            } if !merged.is_empty() => {
                // The own commits are a contiguous run on this branch, so the
                // patch between the parent of the first and the last is exactly
                // the consolidator's own work, with the merged branches'
                // contribution already inside the merge commit it recorded.
                match (commits.first(), commits.last()) {
                    (Some(first), Some(last)) => {
                        incremental_diff(repo, &format!("{first}^..{last}"), true).await
                    }
                    _ => String::new(),
                }
            }
            // A worker's revision: everything its branch added after the last
            // approval, which is exactly the unaudited range.
            Self::Since { base, branch, .. } => {
                incremental_diff(repo, &format!("{base}..{branch}"), true).await
            }
        }
    }

    /// The base commit the incremental diff is measured from. `None` for a full
    /// review, which measures from the worker's own base instead.
    pub fn base_commit(&self) -> Option<&str> {
        match self {
            Self::Full => None,
            Self::Since { base, .. } => Some(base.as_str()),
        }
    }

    /// The log line that states the decision, so a skipped review is as
    /// legible in the hub log as one that ran.
    pub fn decision_log(&self) -> String {
        match self {
            Self::Full => "reviewing the whole diff since the base commit".to_string(),
            Self::Since { base, commits, .. } => {
                format!("reviewing {n} commits since {base}", n = commits.len())
            }
        }
    }

    /// The skip line for this scope, when the review must not run because
    /// nothing unaudited changed; `None` when there is something to review.
    pub fn skip_log(&self) -> Option<String> {
        match self {
            // Only a commit list that is empty *and* a clean working tree is
            // evidence that nothing is left unaudited. An agent that edited a
            // sensitive file without committing it has changed the code, and the
            // commit range alone would not see it.
            Self::Since {
                base,
                commits,
                uncommitted,
                ..
            } if commits.is_empty() && !*uncommitted => Some(format!(
                "security review skipped: no sensitive change since {base}"
            )),
            _ => None,
        }
    }
}

/// The merged worker branches a consolidator may leave out of its own audit.
///
/// Dropping a merged branch is a claim that the code it carries was already
/// reviewed at its own approved commit. `integrated` proves only that the merge
/// happened, so the approval has to be checked separately: a worker merged
/// without one -- never reviewed, or reviewed before the field existed -- keeps
/// its branch in scope and is audited by the consolidator rather than by nobody.
///
/// `approvals` maps a worker id to the commit its security review approved, so
/// the rule is one predicate the pipeline and the tests read alike.
pub fn approved_merged_branches(
    integrated: &[String],
    approvals: impl Fn(&str) -> Option<String>,
) -> Vec<String> {
    integrated
        .iter()
        .filter(|id| approvals(id).is_some())
        .map(|id| format!("worker-{id}"))
        .collect()
}

/// The review the pipeline will actually run, given the scope's skip decision,
/// what the dispatch asked for and whether the scope has no sensitive change
/// (`true` = no sensitive change → no automatic trigger).
///
/// `strongest` is the model to audit with when the manifest marks no tier of its
/// own; it is only consulted for the automatic sensitive-path trigger, so it is
/// the last argument and callers that never reach it may pass any model.
///
/// Two rules carry the security weight:
///
/// * A requested review is never downgraded. `--review-after <m>:security` names
///   the adversarial audit, and an optimisation that skipped work must not
///   quietly turn it into a generic quality pass -- that would weaken a gate the
///   caller asked for by name, on the strength of a bookkeeping field.
/// * The automatic trigger still defers. With nothing unaudited since the last
///   approval and no explicit request, the sensitive-path trigger does not fire
///   a second time over the same code.
pub fn plan_review(
    skip: bool,
    requested: Option<(String, ReviewMode)>,
    sensitive_is_empty: bool,
    strongest: &str,
    security_mode: &ReviewMode,
) -> Option<(String, ReviewMode)> {
    match (skip, requested, sensitive_is_empty) {
        (true, Some((model, mode)), _) => Some((model, mode)),
        (true, None, _) => None,
        // A requested review on a sensitive diff is upgraded to the adversarial
        // mode; the requested model still runs it.
        (false, Some((model, _)), false) => Some((model, security_mode.clone())),
        (false, Some((model, wanted)), true) => Some((model, wanted)),
        // No requested review, but the diff is sensitive: trigger the security
        // review on the manifest's strongest tier.
        (false, None, false) => Some((strongest.to_string(), security_mode.clone())),
        (false, None, true) => None,
    }
}

/// Whether `value` is a plain git object id: 40 or 64 hex digits.
///
/// The approved commit is a string the registry round-trips and this module
/// splices verbatim into git revision arguments (`git log <base>..<branch>`). A
/// value that is not a plain object id -- one a writer with access to the
/// registry file could plant, such as `--output=<path>` -- would be parsed by
/// git as an option rather than a revision, so anything else is not an
/// approval: [`scope_for`] widens to the whole diff instead of trusting it.
fn is_object_id(value: &str) -> bool {
    (value.len() == 40 || value.len() == 64) && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

/// Decide a worker's security review scope from a repository path and a branch.
///
/// The same decision [`security_scope`] makes inside the phase loop, over the
/// facts a caller already has, so the rule — audit only what no earlier security
/// review covered — has one implementation whether the caller is the loop or a
/// reader asking what a worker would be reviewed over.
///
/// * `approved` is the commit an earlier security review approved on this
///   branch, if any.
/// * `merged` names the worker branches a consolidator integrated; a
///   consolidator's scope excludes them.
/// * `role` decides the exclusion; a plain worker is never treated as a
///   consolidator even if it names branches.
pub async fn scope_for(
    repo: &Path,
    branch: &str,
    role: WorkerRole,
    base_commit: &str,
    approved: Option<String>,
    merged: &[String],
) -> SecurityScope {
    match role {
        WorkerRole::Consolidate => SecurityScope::Since {
            base: base_commit.to_string(),
            branch: branch.to_string(),
            approved: Vec::new(),
            commits: own_commits(repo, branch, merged).await,
            merged: merged.to_vec(),
            uncommitted: has_uncommitted_changes(repo).await,
        },
        WorkerRole::Worker => match approved.filter(|sha| is_object_id(sha)) {
            // An approval the repository cannot resolve -- a pruned branch, a
            // rewritten history, a value that is not a plain object id at all
            // -- is not evidence that nothing changed since. Reading it that
            // way would skip a real audit, so the scope widens to the whole
            // diff: re-auditing is the recoverable mistake.
            Some(base) => match commits_since(repo, &base, branch).await {
                Some(commits) => {
                    let already = vec![base.clone()];
                    SecurityScope::Since {
                        base,
                        branch: branch.to_string(),
                        approved: already,
                        commits,
                        merged: Vec::new(),
                        uncommitted: has_uncommitted_changes(repo).await,
                    }
                }
                None => SecurityScope::Full,
            },
            None => SecurityScope::Full,
        },
    }
}

/// Whether the working tree carries a change its branch tip does not: staged,
/// unstaged or intent-to-add.
///
/// An agent edits before it commits and the harness checkpoints only after the
/// review phase, so "no new commit" is not "no new change". Treating a dirty
/// tree as nothing to audit would skip the review of exactly the edits the run
/// is about to hand on, so the honest answer is that something is unaudited.
async fn has_uncommitted_changes(repo: &Path) -> bool {
    let path = repo.to_path_buf();
    tokio::task::spawn_blocking(move || {
        let _ = crate::worktree::git(&path, "add", &["add", "-N", "."]);
        crate::worktree::git(&path, "status", &["status", "--porcelain"])
            .ok()
            .filter(|out| out.status.success())
            .is_some_and(|out| !out.stdout.iter().all(u8::is_ascii_whitespace))
    })
    .await
    .unwrap_or(true)
}

/// Decide what one run's security review has to cover, from the worktree's own
/// facts.
///
/// A thin adapter over [`scope_for`], which holds the rule; this only supplies
/// the branch, base commit and merged worker branches the guard already carries,
/// so the phase loop and any other reader cannot drift apart on what a security
/// review covers.
pub(super) async fn security_scope(
    worktree: &WorktreeGuard,
    role: WorkerRole,
    approved_commit: Option<String>,
    merged_branches: &[String],
) -> SecurityScope {
    scope_for(
        &worktree.path,
        &worktree.branch,
        role,
        &worktree.base_commit,
        approved_commit,
        merged_branches,
    )
    .await
}

/// The reviewer's opening message for a review that covers only what came after
/// an earlier approval.
///
/// The reviewer is auditing a revision, not the whole branch: the diff since
/// `base` is what it has never seen, while the commits earlier security reviews
/// already approved are named so it neither re-reviews them nor mistakes them
/// for new code. It keeps the adversarial checklist -- only the scope shrinks.
fn incremental_prompt(task: &str, base: &str, approved: &[String], diff: &str) -> String {
    let mut prompt = String::from(
        "ADVERSARIAL SECURITY REVIEW PHASE (incremental):\n\
         The previous subagent implemented the following task:\n",
    );
    let _ = write!(prompt, "{task}\n\n");
    let _ = write!(
        prompt,
        "An earlier security review already approved the branch at commit {base}, \
         so this pass covers only what came after it.\n\
         Commits already security-reviewed and approved: {}.\n\n",
        if approved.is_empty() {
            base.to_string()
        } else {
            approved.join(", ")
        }
    );
    prompt.push_str(
        "YOUR OBJECTIVE AS THE ADVERSARIAL REVIEWER:\n\
         Review ONLY the incremental diff below (`git diff {base}..HEAD`, \
         plus the working tree with `git diff`). The earlier commits were \
         already adversarially reviewed; re-litigating them wastes the budget. \
         Ask the same questions of the new lines as of any other diff: who else \
         can reach every path, socket, file, environment variable and IPC message \
         they create; what model-written text is trusted as proof; what is left \
         behind on a crash, handover or revision; what each deletion removes and \
         what gates it; whether each changed test would fail if the code were wrong.\n\
         Fix every real defect you find, with a regression test that fails without the fix.\n\
         1. Report honestly: list every finding you did NOT fix in the `risks:` line.\n\
         2. Print a line `FINDINGS: <n>` with the total number of findings you found.\n\
         When done, execute:\n\
         \x20  echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n\n\
         INCREMENTAL DIFF SINCE {base}:\n",
    );
    if diff.trim().is_empty() {
        prompt.push_str("(no committed change since the approved commit; the working tree diff is the change)\n");
    } else {
        prompt.push_str(diff);
        if !diff.ends_with('\n') {
            prompt.push('\n');
        }
    }
    prompt
}

/// The commit the worktree's checked-out branch points at, or `None` when git
/// cannot resolve it. Recorded as the commit a security review approved.
pub(super) async fn head_commit_of(path: &Path) -> Option<String> {
    let path = path.to_path_buf();
    tokio::task::spawn_blocking(move || {
        let output = crate::worktree::git(&path, "rev-parse", &["rev-parse", "HEAD"]).ok()?;
        output
            .status
            .success()
            .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
            .filter(|head| !head.is_empty())
    })
    .await
    .unwrap_or(None)
}

/// The repository-relative files a worker's working tree changed against its
/// base, uncommitted changes included.
///
/// The automatic sensitive-path trigger runs before the review phase's
/// checkpoint, so the tail may still be uncommitted: intent-to-add stages the
/// new files and the diff against the base then names every path, committed or
/// not, exactly as the worker's final diff does. A git failure yields an empty
/// list, which reads as "nothing sensitive" and simply skips the trigger
/// rather than failing the worker.
pub(super) async fn touched_files(
    path: &Path,
    base_commit: &str,
    base_branch: Option<&str>,
) -> Vec<String> {
    let path = path.to_path_buf();
    let base_commit = base_commit.to_string();
    let base_branch = base_branch.map(str::to_string);
    tokio::task::spawn_blocking(move || {
        let base = WorktreeGuard::diff_base_at(&path, &base_commit, base_branch.as_deref())
            .unwrap_or_else(|_| "HEAD".to_string());
        let _ = crate::worktree::git(&path, "add", &["add", "-N", "."]);
        let Ok(output) = crate::worktree::git(&path, "diff", &["diff", "--name-only", &base])
        else {
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
    })
    .await
    .unwrap_or_default()
}

/// The commits `branch` added after `base`, oldest first.
///
/// Empty when `base` is unknown to the repository (a pruned branch, a rewritten
/// history) or is not an ancestor, because the honest answer is then "nothing
/// is known to be approved" -- and the caller reviews the whole diff rather than
/// treat the gap as an approval.
/// `None` when git cannot resolve the range at all, which is not the same as an
/// empty commit list and must not be read as an approval.
pub(super) async fn commits_since(path: &Path, base: &str, branch: &str) -> Option<Vec<String>> {
    let path = path.to_path_buf();
    let base = base.to_string();
    let branch = branch.to_string();
    tokio::task::spawn_blocking(move || {
        let range = format!("{base}..{branch}");
        let Ok(output) =
            crate::worktree::git(&path, "log", &["log", "--reverse", "--format=%H", &range])
        else {
            // git itself could not run: the range's history is unknown, and an
            // unknown history is not an approval.
            return None;
        };
        output.status.success().then(|| commits_in(output.stdout))
    })
    .await
    .ok()
    .flatten()
}

/// The files a consolidator's own commits changed: the same exclusion of the
/// worker branches it merged, but yielding the paths those own commits touched
/// instead of the commits themselves. This is the consolidator's sensitive-path
/// probe -- the merged worker branches are not its to re-audit.
pub(super) async fn own_files(path: &Path, branch: &str, merged: &[String]) -> Vec<String> {
    let path = path.to_path_buf();
    let branch = branch.to_string();
    let merged = merged.to_vec();
    tokio::task::spawn_blocking(move || {
        // Every commit of every merged branch, so `git log ... --not` leaves
        // exactly the consolidator's own non-merge commits.
        let mut exclusions: Vec<String> = merged
            .iter()
            .filter(|merged| *merged != &branch)
            .flat_map(|merged| rev_list(&path, merged))
            .collect();
        exclusions.sort();
        exclusions.dedup();
        let mut args: Vec<String> = vec![
            "log".to_string(),
            "--no-merges".to_string(),
            "--name-only".to_string(),
            "--format=".to_string(),
            branch.clone(),
            "--not".to_string(),
        ];
        args.extend(exclusions);
        let borrowed: Vec<&str> = args.iter().map(String::as_str).collect();
        crate::worktree::git(&path, "log", &borrowed)
            .ok()
            .filter(|out| out.status.success())
            .map(|out| {
                let mut files: Vec<String> = String::from_utf8_lossy(&out.stdout)
                    .lines()
                    .map(str::trim)
                    .filter(|line| !line.is_empty())
                    .map(str::to_string)
                    .collect();
                files.sort();
                files.dedup();
                files
            })
            .unwrap_or_default()
    })
    .await
    .unwrap_or_default()
}

/// The files the commits `base..branch` changed, sorted and deduplicated.
///
/// This is the sensitive-path probe for an incremental review: it answers "what
/// did this run change that nobody has reviewed yet" without ever widening back
/// to the whole branch diff. A range git cannot resolve yields an empty list,
/// which the caller treats as "nothing new" rather than "everything".
pub(super) async fn files_since(path: &Path, base: &str, branch: &str) -> Vec<String> {
    let path = path.to_path_buf();
    let base = base.to_string();
    let branch = branch.to_string();
    tokio::task::spawn_blocking(move || {
        let _ = crate::worktree::git(&path, "add", &["add", "-N", "."]);
        // The commit range names what the branch added; the working tree names
        // what the run edited but has not committed yet. Both are unaudited, so
        // the probe is the union: measuring only the range would hide an
        // uncommitted sensitive edit from the very decision meant to catch it.
        let mut files: Vec<String> = Vec::new();
        for range in [format!("{base}..{branch}"), branch.clone()] {
            let Ok(output) = crate::worktree::git(&path, "diff", &["diff", "--name-only", &range])
            else {
                continue;
            };
            if !output.status.success() {
                continue;
            }
            files.extend(
                String::from_utf8_lossy(&output.stdout)
                    .lines()
                    .map(str::trim)
                    .filter(|line| !line.is_empty())
                    .map(str::to_string),
            );
        }
        files.sort();
        files.dedup();
        files
    })
    .await
    .unwrap_or_default()
}

/// Commits on `branch` that none of `merged` contains: a consolidator's own
/// work -- its interaction fixes and conflict resolutions -- with the worker
/// branches it integrated, each already reviewed at its own approved commit,
/// left out.
///
/// The exclusion is a set difference over whole histories rather than over merge
/// commit parents, so a worker branch that is itself an ancestor of another
/// merged one cannot smuggle that ancestor's commits back into the review.
pub(super) async fn own_commits(path: &Path, branch: &str, merged: &[String]) -> Vec<String> {
    let path = path.to_path_buf();
    let branch = branch.to_string();
    let merged = merged.to_vec();
    tokio::task::spawn_blocking(move || {
        let all = rev_list(&path, &branch);
        if all.is_empty() {
            return Vec::new();
        }
        let exclusions: Vec<String> = merged
            .iter()
            .filter(|merged_branch| *merged_branch != &branch)
            .flat_map(|merged_branch| rev_list(&path, merged_branch))
            .collect();
        if exclusions.is_empty() {
            return all;
        }
        // `--not` flips every revision after it, so the merged branches' own
        // commits are listed as `^sha` and subtracted from the branch.
        let mut args: Vec<String> = vec![
            "rev-list".to_string(),
            "--reverse".to_string(),
            "--no-merges".to_string(),
            branch.clone(),
            "--not".to_string(),
        ];
        args.extend(exclusions);
        let borrowed: Vec<&str> = args.iter().map(String::as_str).collect();
        crate::worktree::git(&path, "rev-list", &borrowed)
            .ok()
            .filter(|out| out.status.success())
            .map(|out| commits_in(out.stdout))
            .unwrap_or(all)
    })
    .await
    .unwrap_or_default()
}

/// Commits reachable from `r#ref`, or empty when git cannot resolve it.
fn rev_list(path: &Path, r#ref: &str) -> Vec<String> {
    crate::worktree::git(path, "rev-list", &["rev-list", r#ref])
        .ok()
        .filter(|out| out.status.success())
        .map(|out| commits_in(out.stdout))
        .unwrap_or_default()
}

/// Commit ids out of a `git rev-list`/`git log --format=%H` body.
fn commits_in(stdout: Vec<u8>) -> Vec<String> {
    String::from_utf8_lossy(&stdout)
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(str::to_string)
        .collect()
}

/// The diff of the reviewed commits: what the listed commits changed since the
/// earlier approval, uncommitted changes included.
///
/// A commit range never contains work the run has not committed yet, so the
/// working-tree diff is appended when asked for: the reviewer is told to audit
/// `git diff` as well, and handing it a range that silently omits the edits the
/// run is about to hand on would make that instruction a lie.
pub(super) async fn incremental_diff(path: &Path, range: &str, working_tree: bool) -> String {
    let path = path.to_path_buf();
    let range = range.to_string();
    tokio::task::spawn_blocking(move || {
        let _ = crate::worktree::git(&path, "add", &["add", "-N", "."]);
        let mut diff = crate::worktree::git(&path, "diff", &["diff", &range])
            .ok()
            .filter(|out| out.status.success())
            .map(|out| String::from_utf8_lossy(&out.stdout).into_owned())
            .unwrap_or_default();
        if working_tree {
            let uncommitted = crate::worktree::git(&path, "diff", &["diff"])
                .ok()
                .filter(|out| out.status.success())
                .map(|out| String::from_utf8_lossy(&out.stdout).into_owned())
                .unwrap_or_default();
            if !uncommitted.is_empty() {
                if !diff.is_empty() && !diff.ends_with('\n') {
                    diff.push('\n');
                }
                if !diff.is_empty() {
                    diff.push_str("\n--- uncommitted working tree changes, also unaudited ---\n");
                }
                diff.push_str(&uncommitted);
            }
        }
        diff
    })
    .await
    .unwrap_or_default()
}

/// Parse the security reviewer's `FINDINGS: <n>` line.
///
/// The adversarial prompt asks the reviewer to state the total number of
/// findings (fixed or listed) on its own line so the completion event and the
/// status can show a count without depending on free-form prose. A missing or
/// unparseable line yields `None`, which a display renders as "findings not
/// reported" rather than a reassuring zero.
pub fn parse_findings(text: &str) -> Option<usize> {
    text.lines().find_map(|line| {
        let rest = line.trim().strip_prefix("FINDINGS:")?;
        rest.trim().parse::<usize>().ok()
    })
}

/// Everything the review phase needs, and the step counter it hands back.
///
/// The counter flows out through [`ReviewPhaseOutcome::step`] because the
/// reviewer's turns are interleaved with the implementer's in a single
/// monotonic sequence that the registry and the completion record both report.
pub struct ReviewPhase<'a> {
    pub worker_id: String,
    pub reviewer_model: String,
    pub temperature: Option<f32>,
    pub task: String,
    /// The configured turn budget; `0` means "ask the manifest for the
    /// reviewer model's budget instead".
    pub max_turns: usize,
    /// The implementation loop's running turn budget, needed to report the
    /// combined `max_turns` while the worker is marked `reviewing`.
    pub current_max_turns: usize,
    pub repo_path_str: String,
    /// The implementer's turn counter entering the review phase.
    pub step: usize,
    /// The dispatch's declared network policy, applied to the reviewer's own
    /// bash steps too: a worker declared `offline` must not regain egress just
    /// because a second agent takes over the worktree.
    pub network_offline: bool,
    /// The same registry row the implementer wrote to, so the reviewer's turns
    /// land in one continuous set of health counters instead of a second run's
    /// worth.
    pub meta: &'a mut WorkerMeta,
    /// Which auditor runs: the generic quality review, or the adversarial
    /// security review. Defaults to the `quality` mode.
    pub mode: ReviewMode,
    /// The dispatch's completion gate, re-run by the reviewer instead of a
    /// language-specific suite invented in the prompt. `None` when the
    /// dispatch disabled the gate.
    pub verify: Option<String>,
    /// Touched files the repository declared sensitive, when the security
    /// review was triggered by one of them. Empty for a hand-requested
    /// review.
    pub sensitive: Vec<String>,
    /// Which commits this security review covers: the whole diff since the base
    /// for a first run, or only what came after an earlier approval.
    pub scope: SecurityScope,
}

/// What the review phase hands back: the combined step counter and whether the
/// reviewer itself emitted the completion sentinel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReviewPhaseOutcome {
    /// The combined turn counter, for the caller to fold back into its own
    /// loop state.
    pub step: usize,
    /// Whether the reviewer completed. `false` when the reviewer gave up
    /// quietly or ran out of turns, so the whole run must be read as stopped
    /// rather than done.
    pub completed: bool,
    /// The security review that ran, when the mode was
    /// `security`. Carries the finding count the reviewer
    /// reported, for the completion event and the status.
    pub security: Option<SecurityReviewOutcome>,
}

/// The recorded result of one adversarial security review.
///
/// `findings` is what the reviewer reported on its `FINDINGS:` line; an absent
/// line is `None`, never a silent zero, so a status never presents "not
/// reported" as a clean audit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SecurityReviewOutcome {
    pub findings: Option<usize>,
}

/// Which commits one security review has to look at.
///
/// The whole diff since the base commit is the security review of a first run.
/// A revision, and a consolidator integrating branches that were already
/// reviewed at their own approved commits, have an approval to start from: only
/// what came after it is unaudited, so only that is handed to the reviewer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SecurityScope {
    /// No approval to start from: the reviewer sees everything since the base.
    Full,
    /// An earlier security review approved `base`; this review covers the
    /// commits after it, plus the approved commit itself named for context.
    Since {
        /// Commit the earlier security review approved.
        base: String,
        /// The branch tip this review covers: every probe below measures the
        /// unaudited range against it, so an incremental scope needs no
        /// second base of its own.
        branch: String,
        /// The commits that earlier security reviews approved, oldest first.
        approved: Vec<String>,
        /// The unaudited commits this review covers, oldest first.
        commits: Vec<String>,
        /// Branches already reviewed at their own approved commits, which a
        /// consolidator's own commits must be measured without.
        merged: Vec<String>,
        /// Whether the working tree carries a change the covered commits do
        /// not: an agent edits before it commits, so an empty commit list is
        /// not by itself evidence that nothing is left unaudited.
        uncommitted: bool,
    },
}

impl WorkerPool {
    /// Run the review auditor over the finished implementation.
    ///
    /// Returns the combined turn count. Errors propagate: a failed review bash
    /// command is a real failure, while a *failed LLM step* is not — the
    /// reviewer gives up quietly and the implementation result stands.
    pub(super) async fn run_review_phase(
        &self,
        worktree: &mut WorktreeGuard,
        review: ReviewPhase<'_>,
    ) -> Result<ReviewPhaseOutcome> {
        let ReviewPhase {
            worker_id,
            reviewer_model,
            temperature,
            task,
            max_turns,
            current_max_turns,
            repo_path_str,
            mut step,
            network_offline,
            meta,
            mode,
            verify,
            sensitive,
            scope,
        } = review;

        info!(
            worker = %worker_id,
            reviewer = %reviewer_model,
            mode = mode.as_str(),
            "Implementation finished; starting multi-phase review pipeline"
        );

        // Checkpoint phase 1 implementation changes in git. `commit_changes`
        // shells out to git, so it runs off the runtime thread.
        {
            let path = worktree.path.clone();
            let message = format!(
                "worker({}): implementation phase completed (checkpoint)",
                worker_id
            );
            let committed = tokio::task::spawn_blocking(move || {
                crate::worktree::WorktreeGuard::commit_all(&path, &message)
            })
            .await
            .unwrap_or(Ok(false));
            if committed.unwrap_or(false) {
                worktree.preserve_branch = true;
            }
        }
        info!(
            worker = %worker_id,
            "{}", scope.decision_log()
        );

        let review_prompt = review_prompt(&mode, &task, verify.as_deref(), &sensitive);
        // A security review of a revision gets the unaudited diff rather than
        // the whole branch: what it must judge is what came after the earlier
        // approval, and the earlier approvals are named for context.
        let security_prompt = match (&scope, mode.is_security()) {
            (
                incremental @ SecurityScope::Since {
                    commits, approved, ..
                },
                true,
            ) if !commits.is_empty() => {
                let diff = incremental.reviewed_diff(&worktree.path).await;
                Some(incremental_prompt(
                    &task,
                    incremental.base_commit().unwrap_or_default(),
                    approved,
                    &diff,
                ))
            }
            _ => None,
        };
        let review_prompt = security_prompt.unwrap_or(review_prompt);

        let reviewer_runner = AgentRunner::new(
            self.api_base.clone(),
            self.api_key.clone(),
            reviewer_model.clone(),
            temperature,
        )
        .with_network_offline(network_offline)
        // The reviewer shares the worker's job table, so a job it backgrounds
        // is confined and stopped exactly like the implementer's.
        .with_jobs(self.job_handle(&worker_id));

        let manifest = self.manifest();
        let (_, _, reviewer_manifest_turns) = manifest.resolve_model(&reviewer_model);

        // The reviewer gets *its own* role memory, keyed by the reviewer alias, so
        // review lessons never bleed into the implementer's prompt (and vice versa).
        // `reviewer_model` may already be a resolved id (`combo:nerd`), so it is
        // mapped back to its alias first; an unknown id passes through unchanged and
        // simply finds no memory file.
        let reviewer_alias = manifest.alias_for_model(&reviewer_model);
        let mut review_messages = vec![
            ChatMessage::text(
                Role::System,
                build_system_prompt(Path::new(&repo_path_str), &reviewer_alias),
            ),
            ChatMessage::text(Role::User, review_prompt),
        ];
        let review_max_turns = if max_turns > 0 {
            max_turns
        } else {
            reviewer_manifest_turns.unwrap_or(current_max_turns)
        };

        self.save_status(
            meta,
            &reviewer_model,
            RegistryStatus::Reviewing,
            step,
            current_max_turns + review_max_turns,
            "starting review phase",
            None,
        );

        let mut review_step = 0;
        let mut last_assistant_text = String::new();
        let mut consecutive_no_cmd = 0;
        let mut watch = ProgressWatch::default();
        // The reviewer's completion approves the audit; it stores no report, so
        // the engine's report state is a sink here.
        let mut report = None;
        let mut report_asked = false;
        let mut report_text = String::new();
        // The reviewer stores no report, so its verdicts are a sink too: only
        // the implementer's completion is a round's record of its workers.
        let mut verdicts = None;
        let mut combined_max_turns = current_max_turns + review_max_turns;

        while review_step < review_max_turns {
            review_step += 1;
            step += 1;
            let max_turns_for_config = current_max_turns + review_max_turns;
            let turn_config = TurnConfig {
                label_prefix: "[review] ",
                steer_prefix: "ORCHESTRATOR GUIDANCE:\n",
                apply_sentinels: false,
                // Inspecting the diff and re-running the gate is the reviewer's
                // job in both modes, so a turn that changes nothing is the
                // review, not a stall: the read-only escalation stays off.
                read_only_exempt: true,
                llm_error_policy: LlmErrorPolicy::EndQuietly,
                status: RegistryStatus::Reviewing,
                model: &reviewer_model,
                max_turns: max_turns_for_config,
                task: &task,
                temperature,
                review_after: None,
                network_offline,
            };
            let mut engine = TurnEngine {
                pool: self,
                worktree,
                runner: &reviewer_runner,
                worker_id: &worker_id,
                meta,
                messages: &mut review_messages,
                unsaved_messages: Vec::new(),
                step: &mut step,
                current_max_turns: &mut combined_max_turns,
                last_assistant_text: &mut last_assistant_text,
                consecutive_no_cmd: &mut consecutive_no_cmd,
                verify: None,
                client_env: &[],
                dispatch_max_turns: max_turns,
                watch: &mut watch,
                report: &mut report,
                report_asked: &mut report_asked,
                report_text: &mut report_text,
                verdicts: &mut verdicts,
            };
            match engine.run_turn(&turn_config).await? {
                TurnOutcome::Completed { .. } => {
                    info!(
                        worker = %worker_id,
                        step = review_step,
                        "Reviewer completed and approved changes"
                    );
                    // The engine still holds the `&mut` borrow of the last
                    // assistant text, so the text is cloned through it.
                    let text = engine.last_assistant_text.clone();
                    let security = mode.is_security().then(|| SecurityReviewOutcome {
                        findings: parse_findings(&text),
                    });
                    return Ok(ReviewPhaseOutcome {
                        step,
                        completed: true,
                        security,
                    });
                }
                TurnOutcome::Continue | TurnOutcome::NoCommand => {}
                // The reviewer gave up quietly (an LLM error under
                // `EndQuietly`): the audit is inconclusive, not approved.
                TurnOutcome::EndReview => {
                    // The reviewer gave up quietly: the audit is
                    // inconclusive, but a security review that ran is still
                    // recorded, with no count rather than a reassuring zero.
                    let text = engine.last_assistant_text.clone();
                    let security = mode.is_security().then(|| SecurityReviewOutcome {
                        findings: parse_findings(&text),
                    });
                    return Ok(ReviewPhaseOutcome {
                        step,
                        completed: false,
                        security,
                    });
                }
            }
        }

        // The budget ran out with no completion sentinel.
        let security = mode.is_security().then(|| SecurityReviewOutcome {
            findings: parse_findings(&last_assistant_text),
        });
        Ok(ReviewPhaseOutcome {
            step,
            completed: false,
            security,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest(yaml: &str) -> ModelManifest {
        serde_yaml::from_str(yaml).expect("manifest YAML must parse")
    }

    #[test]
    fn the_strongest_tier_is_resolved_to_its_id() {
        let manifest = manifest(
            "default: ninja\nstrongest: nerd\nmodels:\n  ninja:\n    id: combo:ninja\n  nerd:\n    id: combo:nerd\n",
        );
        let choice = select_security_reviewer(&manifest, "combo:default");
        assert_eq!(choice.model, "combo:nerd");
        assert!(choice.reason.contains("strongest"), "{}", choice.reason);
    }

    #[test]
    fn an_unmarked_catalog_falls_back_to_the_dispatch_default() {
        let manifest = manifest("default: ninja\nmodels:\n  ninja:\n    id: combo:ninja\n");
        let choice = select_security_reviewer(&manifest, "combo:default");
        assert_eq!(choice.model, "combo:default");
        assert!(choice.reason.contains("no strongest"), "{}", choice.reason);
    }

    #[test]
    fn a_dangling_strongest_key_is_ignored() {
        let manifest =
            manifest("default: ninja\nstrongest: absent\nmodels:\n  ninja:\n    id: combo:ninja\n");
        let choice = select_security_reviewer(&manifest, "combo:default");
        assert_eq!(
            choice.model, "combo:default",
            "a `strongest:` alias the catalog does not define must fall back"
        );
    }
}
