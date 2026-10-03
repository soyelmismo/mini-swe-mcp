//! The multi-phase review auditor: an independent agent that re-runs the
//! quality gates over the implementation phase's work.
//!
//! Once the implementation loop stops, [`run_review_phase`] checkpoints the
//! worktree in git and then drives a *second* agent (a different model) over
//! the same worktree. The reviewer is deliberately not the implementation
//! agent: its contract is to inspect the diff, run the suite the dispatch
//! named, fix whatever it finds, and only then emit the completion sentinel.
//!
//! Two modes share this engine ([`ReviewMode`]):
//!
//! * [`ReviewMode::Quality`] — today's generic audit: run the gate, inspect
//!   the diff, fix real defects, re-run the gate.
//! * [`ReviewMode::Security`] — an adversarial pass over the same diff. Its
//!   checklist is about what an unprivileged local user, another agent or a
//!   lying model text can do with every new path, socket, file, environment
//!   variable and IPC message. It is selected by the `:security` suffix of
//!   `--review-after` (`review_after`), and it is also what the automatic
//!   sensitive-path trigger runs.
//!
//! It runs under its own turn budget ([`ReviewPhase::review_max_turns`],
//! resolved from the model manifest) and its own message history, so the
//! reviewer's context never mixes with the implementer's.

use anyhow::Result;
use tracing::info;

use std::fmt::Write as _;
use std::path::Path;

use crate::agent::{AgentRunner, ChatMessage, Role};
use crate::manifest::build_system_prompt;
use crate::worktree::WorktreeGuard;

use super::super::WorkerPool;
use super::super::registry::{RegistryStatus, WorkerMeta};
use super::turn::{LlmErrorPolicy, ProgressWatch, TurnConfig, TurnEngine, TurnOutcome};

/// Which auditor runs over the finished implementation.
///
/// The default is the historical quality review; [`ReviewMode::Security`] is
/// the adversarial variant. The two differ only in the prompt they hand the
/// reviewer: the engine, the turn budget, the network policy and the worktree
/// are identical, so a security review is not a second worker to reason about.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReviewMode {
    /// The generic audit: gates, diff, real defects, gates again.
    #[default]
    Quality,
    /// The adversarial audit: hostile inputs, untrusted model text, crash and
    /// handover windows, deletion proof, test meaning.
    Security,
}

impl ReviewMode {
    /// The suffix that selects this mode in `--review-after <model>:<mode>`.
    ///
    /// Spelled here and matched in [`ReviewMode::parse_model`] so the CLI, the
    /// MCP arg and the prompt cannot disagree on the spelling.
    pub const SECURITY_SUFFIX: &'static str = "security";

    /// Split `--review-after <model>[:security]` into the model and the mode.
    ///
    /// A model id legitimately contains `:` (`combo:nerd`), so only a suffix
    /// equal to [`ReviewMode::SECURITY_SUFFIX`] is a mode marker: the *last*
    /// `:` segment is inspected and removed only when it names a known mode.
    /// Everything else — `combo:nerd`, `some/unknown`, a trailing `:` — stays
    /// the model verbatim, which is what keeps today's `--review-after` values
    /// parsing exactly as they did.
    pub fn parse_model(requested: &str) -> (String, Self) {
        let trimmed = requested.trim();
        match trimmed.rsplit_once(':') {
            Some((model, suffix))
                if !model.is_empty() && suffix.eq_ignore_ascii_case(Self::SECURITY_SUFFIX) =>
            {
                (model.to_string(), Self::Security)
            }
            _ => (trimmed.to_string(), Self::Quality),
        }
    }

    /// The name a status line, an event or a log prints.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Quality => "quality",
            Self::Security => "security",
        }
    }
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
    mode: ReviewMode,
    task: &str,
    verify: Option<&str>,
    sensitive: &[String],
) -> String {
    let gate = verify
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .unwrap_or(DISABLED_GATE);
    match mode {
        ReviewMode::Quality => quality_prompt(task, gate),
        ReviewMode::Security => security_prompt(task, gate, sensitive),
    }
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
    /// security review. Defaults to [`ReviewMode::Quality`].
    pub mode: ReviewMode,
    /// The dispatch's completion gate, re-run by the reviewer instead of a
    /// language-specific suite invented in the prompt. `None` when the
    /// dispatch disabled the gate.
    pub verify: Option<String>,
    /// Touched files the repository declared sensitive, when the security
    /// review was triggered by one of them. Empty for a hand-requested
    /// review.
    pub sensitive: Vec<String>,
}

/// What the review phase hands back: the combined step counter and whether the
/// reviewer itself emitted the completion sentinel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReviewPhaseOutcome {
    /// The combined turn counter, for the caller to fold back into its own
    /// loop state.
    pub step: usize,
    /// Whether the reviewer completed. `false` when the reviewer gave up
    /// quietly or ran out of turns, so the whole run must be read as stopped
    /// rather than done.
    pub completed: bool,
    /// The security review that ran, when the mode was
    /// [`ReviewMode::Security`]. Carries the finding count the reviewer
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

        let review_prompt = review_prompt(mode, &task, verify.as_deref(), &sensitive);

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

        // The reviewer gets *its own* role memory and *its own* manifest
        // `instructions:` block, both keyed by the reviewer alias, so review
        // habits never bleed into the implementer's prompt (and vice versa).
        // `reviewer_model` may already be a resolved id (`combo:nerd`), so it is
        // mapped back to its alias first; an unknown id passes through unchanged and
        // simply finds no memory file and no instructions.
        let reviewer_alias = manifest.alias_for_model(&reviewer_model);
        let mut review_messages = vec![
            ChatMessage::text(
                Role::System,
                build_system_prompt(manifest, Path::new(&repo_path_str), &reviewer_alias),
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
                    let security = (mode == ReviewMode::Security).then(|| SecurityReviewOutcome {
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
                    let security = (mode == ReviewMode::Security).then(|| SecurityReviewOutcome {
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
        let security = (mode == ReviewMode::Security).then(|| SecurityReviewOutcome {
            findings: parse_findings(&last_assistant_text),
        });
        Ok(ReviewPhaseOutcome {
            step,
            completed: false,
            security,
        })
    }
}
