//! Live terminal supervisor and dashboard monitor for `mini-swe-mcp`.
//!
//! Interactive, in-place overwriting TUI that monitors running agent swarms
//! and isolated worker pods without spamming log lines.
//!
//! # Layout contract
//!
//! The dashboard is a table: every worker occupies exactly one *bounded* row,
//! so a narrow terminal wraps nothing and a wide one is not filled with dead
//! space. Bounding is driven by [`Layout`], a pure struct derived from the
//! terminal width (see [`Layout::for_terminal_width`]). All measurement runs on
//! the visible (ANSI-stripped) text, never on escape sequences, so colour never
//! shifts a column.
//!
//! # Grouping contract
//!
//! Rows are grouped **by repository** (`WorkerRegistryEntry::repo_path`,
//! [`DEFAULT_REPO_KEY`] when absent), never by the swarm tag: a supervisor
//! watches *one worktree at a time*, so grouping by repository yields exactly
//! one coherent table per repository. The optional swarm/domain tag is demoted
//! to a `[tag]` prefix inside the task column, where it still shows but can no
//! longer fracture the dashboard into many tiny tables.
//!
//! Each repository heading carries its own counters (`total`, then every
//! non-zero state), folded in by `RepoGroup::push` on the same pass that groups
//! the rows. The global counters above stay the cross-repository view, so one
//! repository finishing early is visible both in its own table and in the
//! fleet-wide strip.

use crate::config::env_parse;
use crate::pool::{RegistryStatus, WorkerRegistryEntry, load_all_registry_entries, unix_timestamp};
use anyhow::Result;
use std::collections::BTreeMap;
use std::io::{IsTerminal, Write};

/// Human-readable elapsed time: `05s`, `01m 05s`, `01h 01m`.
fn format_duration(secs: u64) -> String {
    if secs < 60 {
        format!("{secs:02}s")
    } else if secs < 3600 {
        format!("{:02}m {:02}s", secs / 60, secs % 60)
    } else {
        format!("{:02}h {:02}m", secs / 3600, (secs % 3600) / 60)
    }
}

/// Width assumed when the real terminal size cannot be determined (non-tty
/// output, `MONITOR_WIDTH` override, or a TUI that has not reported its size).
pub const DEFAULT_TERMINAL_WIDTH: usize = 120;

/// Narrowest usable terminal: below this the fixed ID/PID columns would leave
/// nothing for the task text, so the row degrades to a two-line layout.
pub const MIN_TERMINAL_WIDTH: usize = 60;

/// Grouping key used for workers whose `repo_path` was never recorded.
pub const DEFAULT_REPO_KEY: &str = "local";

/// Narrowest task column that still shows a command prefix plus its ellipsis.
pub const MIN_OP_WIDTH: usize = 12;

/// Widest task column, even on very wide terminals.
pub const MAX_OP_WIDTH: usize = 60;

/// Indent of a stacked row's continuation lines.
const STACKED_INDENT: usize = 2;

/// Visible width of the ID column (a full UUID is never shown).
const ID_WIDTH: usize = 8;
/// Visible width of the PID column.
const PID_WIDTH: usize = 7;
/// Visible width of the status column: `RUNNING  ` is 9, `◆ REVIEWING` is 11.
const STATUS_WIDTH: usize = 11;
/// Visible width of the model column.
const MODEL_WIDTH: usize = 8;
/// Visible width of the uptime column (`01m 05s` is the widest format).
const UPTIME_WIDTH: usize = 7;
/// Visible width of the bare `step/max` counter.
const TURNS_BASE: usize = 9;
/// Filled/empty cells inside the progress bar, brackets excluded.
const PROGRESS_CELLS: usize = 8;
/// Width of the progress bar with its brackets.
const BAR_WIDTH: usize = PROGRESS_CELLS + 2;
/// Spaces between adjacent columns.
const GAP: usize = 2;
/// Spaces between the last column and the task text.
const LAST_GAP: usize = 1;

/// Fixed width of a row without the progress bar.
const FIXED_CORE: usize = ID_WIDTH
    + PID_WIDTH
    + STATUS_WIDTH
    + TURNS_BASE
    + MODEL_WIDTH
    + UPTIME_WIDTH
    + 5 * GAP
    + LAST_GAP;

/// Fixed width of a row including the progress bar.
const FIXED_FULL: usize = FIXED_CORE + BAR_WIDTH + 1;

/// Whether a row is laid out on one line or as a stack of short lines.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowShape {
    /// Every column on one line: `ID PID STATUS TURNS MODEL UPTIME task`.
    Inline {
        /// Whether the turns column carries a progress bar.
        progress: bool,
    },
    /// Too narrow for one line: two label lines plus wrapped task lines.
    Stacked,
}

/// Visible column widths, derived from the terminal width.
///
/// The task column is the only elastic one: it absorbs whatever the fixed
/// columns leave behind and is clamped into [`MIN_OP_WIDTH`]..=[`MAX_OP_WIDTH`].
/// That clamp is what makes the cap on long commands *dynamic* -- a 400-column
/// terminal still truncates at [`MAX_OP_WIDTH`] instead of printing a 3 KB
/// command, while an 80-column terminal truncates at ~19 characters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Layout {
    /// Total terminal width this layout was computed for.
    pub total: usize,
    /// Visible width of the ID column.
    pub id: usize,
    /// Visible width of the PID column.
    pub pid: usize,
    /// Visible width of the status column.
    pub status: usize,
    /// Visible width of the turns column (bar included when shown).
    pub turns: usize,
    /// Visible width of the model column.
    pub model: usize,
    /// Visible width of the uptime column.
    pub uptime: usize,
    /// Maximum visible width of the last-op/task text.
    pub op: usize,
    /// How a worker row is laid out at this width.
    pub shape: RowShape,
}

impl Layout {
    /// Derive the responsive layout for a terminal of `term_width` columns.
    ///
    /// Three tiers by available width: **inline + progress bar** when both fit;
    /// **inline, no bar** when one line still fits a usable task column (the
    /// bar is dropped first, since `step/max` carries the same information);
    /// **stacked** otherwise — two short label lines and a wrapped task, so no
    /// line overflows. Never panics and never returns a zero-width task column;
    /// widths below [`MIN_TERMINAL_WIDTH`] are raised to that floor.
    pub fn for_terminal_width(term_width: usize) -> Self {
        let total = term_width.max(MIN_TERMINAL_WIDTH);

        // Tier 1: the progress bar is a bonus, not a requirement.
        if total >= FIXED_FULL + MIN_OP_WIDTH {
            return Self::inline(total, true);
        }
        // Tier 2: one line, counter only.
        if total >= FIXED_CORE + MIN_OP_WIDTH {
            return Self::inline(total, false);
        }
        // Tier 3: stack it; the task wraps at the full usable width.
        Self {
            total,
            id: ID_WIDTH,
            pid: PID_WIDTH,
            status: STATUS_WIDTH,
            turns: TURNS_BASE,
            model: MODEL_WIDTH,
            uptime: UPTIME_WIDTH,
            op: total.saturating_sub(STACKED_INDENT).max(1),
            shape: RowShape::Stacked,
        }
    }

    /// Build an inline layout with the task column taking the slack.
    fn inline(total: usize, progress: bool) -> Self {
        let turns = if progress {
            BAR_WIDTH + 1 + TURNS_BASE
        } else {
            TURNS_BASE
        };
        let fixed = FIXED_CORE + if progress { BAR_WIDTH + 1 } else { 0 };
        Self {
            total,
            id: ID_WIDTH,
            pid: PID_WIDTH,
            status: STATUS_WIDTH,
            turns,
            model: MODEL_WIDTH,
            uptime: UPTIME_WIDTH,
            op: total.saturating_sub(fixed).clamp(MIN_OP_WIDTH, MAX_OP_WIDTH),
            shape: RowShape::Inline { progress },
        }
    }

    /// Visible width of everything left of the task text.
    ///
    /// Used to draw the rule that closes the header, so the rule is exactly
    /// as wide as the rows it sits above.
    pub fn fixed_width(&self) -> usize {
        self.id
            + self.pid
            + self.status
            + self.turns
            + self.model
            + self.uptime
            + 5 * GAP
            + LAST_GAP
    }

    /// Whether the progress bar is rendered in the turns column.
    pub fn shows_progress(&self) -> bool {
        matches!(self.shape, RowShape::Inline { progress: true })
    }
}

// ---------------------------------------------------------------------------
// Text measurement helpers (width-aware, never byte-based)
// ---------------------------------------------------------------------------

/// Number of columns a string occupies, ignoring SGR escape sequences.
///
/// ANSI escapes are zero-width on screen but take bytes in `String`, so any
/// byte-length arithmetic over a coloured cell would drift. Keeps all padding
/// and capping decisions on the visible text.
pub fn visible_width(s: &str) -> usize {
    let mut width = 0usize;
    let mut in_escape = false;
    for ch in s.chars() {
        if in_escape {
            // A truncated row can end mid-CSI; those bytes stay invisible so
            // the width of a damaged cell is still the text it actually shows.
            if ch == 'm' {
                in_escape = false;
            }
        } else if ch == '\x1b' {
            in_escape = true;
        } else {
            width += 1;
        }
    }
    width
}

/// Split `s` into at most `max_len` visible columns, appending `…` when it had
/// to cut.
///
/// Result is at most `max_len` visible columns wide, ellipsis included. Escapes
/// inside the kept prefix are preserved verbatim (invisible) but do not consume
/// budget; an unterminated escape sequence is dropped so a truncated colour
/// never leaks into the next row.
pub fn truncate_visible(s: &str, max_len: usize) -> String {
    if max_len == 0 {
        return String::new();
    }
    if visible_width(s) <= max_len {
        return s.to_string();
    }

    // The ellipsis is paid for out of the budget, so the result is *never*
    // wider than `max_len` -- that is what makes the column cap exact.
    let keep = max_len - 1;
    let mut out = String::new();
    let mut width = 0usize;
    let mut chars = s.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\x1b' {
            // Copy the escape verbatim; it occupies no visible column.
            out.push(ch);
            for esc in chars.by_ref() {
                out.push(esc);
                if esc == 'm' {
                    break;
                }
            }
        } else if width < keep {
            width += 1;
            out.push(ch);
        } else {
            break;
        }
    }
    // A truncated tail never carries an SGR reset, so nothing leaks; the
    // ellipsis is the only visible addition.
    out.push('\u{2026}');
    out
}

/// Right-pad `s` with spaces to `width` visible columns (no-op when wider).
pub fn pad_visible(s: &str, width: usize) -> String {
    let visible = visible_width(s);
    if visible >= width {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len() + (width - visible));
    out.push_str(s);
    out.extend(std::iter::repeat_n(' ', width - visible));
    out
}

/// Split a long task into `op_width`-sized chunks for wrapped rendering.
fn wrap_visible(s: &str, op_width: usize) -> Vec<String> {
    let mut lines = Vec::new();
    let mut current = String::new();
    for word in s.split_whitespace() {
        if !current.is_empty() && visible_width(&current) + 1 + visible_width(word) > op_width {
            lines.push(std::mem::take(&mut current));
        }
        if !current.is_empty() {
            current.push(' ');
        }
        if visible_width(word) > op_width {
            // A single oversized word: hard-split it so no line overflows.
            let mut head = String::new();
            for ch in word.chars() {
                if head.chars().count() == op_width {
                    lines.push(std::mem::take(&mut head));
                }
                head.push(ch);
            }
            current = head;
        } else {
            current.push_str(word);
        }
    }
    if !current.is_empty() || lines.is_empty() {
        lines.push(current);
    }
    lines
}

// ---------------------------------------------------------------------------
// Row cells
// ---------------------------------------------------------------------------

/// Visual indicator for the review phase.
///
/// A reviewing worker must not look like a running one: the implementer has
/// finished and an independent reviewer is auditing the diff. `REVIEWING` is
/// padded to the column width and, when colour is on, carries a bright marker
/// plus the `[REVIEW]` tag that [`review_tag`] also prepends to the task.
fn review_badge(use_color: bool) -> &'static str {
    if use_color {
        "\x1b[1;35m\u{25c6} REVIEWING\x1b[0m"
    } else {
        "\u{25c6} REVIEWING"
    }
}

/// Tag marking a row as being in the review phase.
fn review_tag(use_color: bool) -> &'static str {
    if use_color {
        "\x1b[1;35m[REVIEW]\x1b[0m"
    } else {
        "[REVIEW]"
    }
}

/// Progress indicator for the turn budget: a filled/empty bar plus `step/max`.
///
/// The bar makes "how far along is this worker" readable at a glance; the
/// numbers stay for precision.
fn progress_bar(step: usize, max_turns: usize) -> String {
    // `checked_div` keeps a zero budget (never written by the pool, but a
    // hand-edited registry file can hold it) from panicking the whole TUI.
    let filled = (step.min(max_turns) * PROGRESS_CELLS)
        .checked_div(max_turns)
        .unwrap_or(PROGRESS_CELLS)
        .min(PROGRESS_CELLS);
    let mut bar = String::with_capacity(BAR_WIDTH);
    bar.push('[');
    for i in 0..PROGRESS_CELLS {
        bar.push(if i < filled { '#' } else { '.' });
    }
    bar.push(']');
    bar
}

/// The turns column: a progress bar followed by the exact `step/max` counter.
///
/// The bar answers "how far along" at a glance, the counter "exactly how far".
/// The bar is dropped when the terminal is too narrow to show both without
/// wrapping.
fn turns_cell(step: usize, max_turns: usize, layout: &Layout) -> String {
    let counter = format!("{step}/{max_turns}");
    let cell = match layout.shows_progress() {
        true => format!("{} {counter}", progress_bar(step, max_turns)),
        false => counter,
    };
    if cell.chars().count() >= layout.turns {
        return cell;
    }
    format!("{}{cell}", " ".repeat(layout.turns - cell.chars().count()))
}

/// Status cell for a worker: badge text plus colour, padded by the caller.
///
/// `reviewing` is deliberately distinct from `running`: the implementer is
/// done and an independent reviewer is auditing the diff.
fn status_cell(status: RegistryStatus, use_color: bool) -> &'static str {
    if status == RegistryStatus::Reviewing {
        return review_badge(use_color);
    }
    if use_color {
        match status {
            RegistryStatus::Running => "\x1b[1;32mRUNNING  \x1b[0m",
            RegistryStatus::Paused => "\x1b[1;33mPAUSED  \x1b[0m",
            RegistryStatus::Completed => "\x1b[1;34mDONE    \x1b[0m",
            RegistryStatus::Failed => "\x1b[1;31mFAILED  \x1b[0m",
            _ => "\x1b[2;37mSTOPPED \x1b[0m",
        }
    } else {
        match status {
            RegistryStatus::Running => "RUNNING ",
            RegistryStatus::Paused => "PAUSED  ",
            RegistryStatus::Completed => "DONE    ",
            RegistryStatus::Failed => "FAILED  ",
            _ => "STOPPED ",
        }
    }
}

/// The fixed columns of a worker row (everything left of the task text).
fn row_prefix(w: &WorkerRegistryEntry, layout: &Layout, use_color: bool, now: u64) -> String {
    let id = truncate_visible(&w.id, layout.id);
    let pid = format!("{:<width$}", w.pid, width = PID_WIDTH);
    let duration_secs = if w.status.is_terminal() {
        w.updated_at.saturating_sub(w.started_at)
    } else {
        now.saturating_sub(w.started_at)
    };
    let uptime = format_duration(duration_secs);

    let status = status_cell(w.status, use_color);

    let turns = turns_cell(w.step, w.max_turns, layout);
    let model = pad_visible(&w.model, layout.model);
    let uptime = pad_visible(&uptime, layout.uptime);

    let mut prefix = String::new();
    prefix.push_str(&pad_visible(&id, layout.id));
    prefix.push_str(&" ".repeat(GAP));
    prefix.push_str(&pid);
    prefix.push_str(&" ".repeat(GAP));
    prefix.push_str(&pad_visible(status, layout.status));
    prefix.push_str(&" ".repeat(GAP));
    prefix.push_str(&turns);
    prefix.push_str(&" ".repeat(GAP));
    prefix.push_str(&model);
    prefix.push_str(&" ".repeat(GAP));
    prefix.push_str(&uptime);
    prefix.push_str(&" ".repeat(LAST_GAP));
    prefix
}

/// Lay a worker out as a stack of short lines for narrow terminals.
///
/// Two label lines carry the fixed columns and the task text wraps below them,
/// so nothing is chopped to a single character and no line overflows.
fn stack_row(
    w: &WorkerRegistryEntry,
    layout: &Layout,
    use_color: bool,
    now: u64,
    op: &str,
) -> String {
    let duration_secs = if w.status.is_terminal() {
        w.updated_at.saturating_sub(w.started_at)
    } else {
        now.saturating_sub(w.started_at)
    };
    let status = status_cell(w.status, use_color);
    let first = format!(
        "{}  {}  {}",
        pad_visible(&truncate_visible(&w.id, layout.id), layout.id),
        pad_visible(&format!("{:<width$}", w.pid, width = layout.pid), layout.pid),
        turns_cell(w.step, w.max_turns, layout),
    );
    let mut second = format!(
        "{}  {}  {}",
        pad_visible(status, layout.status),
        pad_visible(&w.model, layout.model),
        pad_visible(&format_duration(duration_secs), layout.uptime),
    );
    // Health counters ride the label line, which has room the inline row does
    // not — but only whole: a cell that would overflow the terminal is
    // dropped rather than clipped, so the row stays readable.
    let health = w.metrics.repeat_nudge_cell();
    if visible_width(&second) + 2 + visible_width(&health) <= layout.total {
        second = format!("{}  {health}", second.trim_end());
    }
    let mut out = String::new();
    out.push_str(&first);
    out.push('\n');
    out.push_str(&second);
    out.push('\n');
    out.push_str("  ");
    out.push_str(op);
    out.push('\n');
    for extra in wrap_visible(op, layout.op).into_iter().skip(1) {
        out.push_str(&" ".repeat(STACKED_INDENT));
        out.push_str(&extra);
        out.push('\n');
    }
    out
}

/// The last-op / task cell, capped to `layout.op` visible columns.
fn op_cell(w: &WorkerRegistryEntry, layout: &Layout, use_color: bool) -> String {
    let first_line = w.task.lines().next().unwrap_or("").trim();
    let detail = if let Some(ref q) = w.question {
        format!("ASK: {q}")
    } else if !w.last_command.is_empty()
        && w.last_command != "completed"
        && w.last_command != "initializing"
    {
        format!("{} — {}", w.last_command, first_line)
    } else {
        first_line.to_string()
    };

    let tag = if w.status == RegistryStatus::Reviewing {
        format!("{} ", review_tag(use_color))
    } else {
        match w.group.as_deref() {
            Some(g) if !g.trim().is_empty() => format!("[{}] ", g.trim()),
            _ => String::new(),
        }
    };
    let detail = format!("{tag}{detail}");
    truncate_visible(&detail, layout.op)
}

// ---------------------------------------------------------------------------
// Dashboard rendering
// ---------------------------------------------------------------------------

/// One repository's slice of the dashboard: its workers plus the counters
/// shown in its heading.
///
/// Counters are folded in while grouping (one pass over the entries), so the
/// heading never needs a second scan and the row loop never needs a
/// per-status branch of its own.
struct RepoGroup<'a> {
    workers: Vec<&'a WorkerRegistryEntry>,
    active: usize,
    paused: usize,
    reviewing: usize,
    completed: usize,
    failed: usize,
    stopped: usize,
}

impl<'a> RepoGroup<'a> {
    /// An empty group, ready for [`RepoGroup::push`].
    fn new() -> Self {
        Self {
            workers: Vec::new(),
            active: 0,
            paused: 0,
            reviewing: 0,
            completed: 0,
            failed: 0,
            stopped: 0,
        }
    }

    /// Append one worker and fold its status into the counters.
    fn push(&mut self, entry: &'a WorkerRegistryEntry) {
        match entry.status {
            RegistryStatus::Running => self.active += 1,
            RegistryStatus::Paused => self.paused += 1,
            RegistryStatus::Reviewing => self.reviewing += 1,
            RegistryStatus::Completed => self.completed += 1,
            RegistryStatus::Failed => self.failed += 1,
            RegistryStatus::Stopped => self.stopped += 1,
        }
        self.workers.push(entry);
    }

    /// Heading counters: `total` first, then every non-zero state.
    ///
    /// A quiet repository reads `total: 4`, a busy one
    /// `total: 4 | active: 2 | reviewing: 1 | completed: 1`. Every state is
    /// covered, so no worker is hidden behind a dropped counter; zeros are
    /// dropped so an idle worktree spends its heading on `total` alone.
    fn summary_items(&self) -> Vec<(&'static str, String, &'static str)> {
        [
            ("total", self.workers.len(), BOLD),
            ("active", self.active, GREEN),
            ("paused", self.paused, YELLOW),
            ("reviewing", self.reviewing, MAGENTA),
            ("completed", self.completed, BLUE),
            ("failed", self.failed, RED),
            ("stopped", self.stopped, DIM),
        ]
        .into_iter()
        .filter(|(_, count, _)| *count > 0)
        .map(|(label, count, color)| (label, count.to_string(), color))
        .collect()
    }
}

/// Render the supervisor dashboard at the default width.
pub fn render_dashboard(entries: &[WorkerRegistryEntry], now: u64, use_color: bool) -> String {
    render_dashboard_with_width(entries, now, use_color, DEFAULT_TERMINAL_WIDTH)
}

/// Render the supervisor dashboard for a specific terminal width.
///
/// Pure: identical inputs always produce identical output, which is what makes
/// the width-dependent behaviour unit-testable.
pub fn render_dashboard_with_width(
    entries: &[WorkerRegistryEntry],
    now: u64,
    use_color: bool,
    term_width: usize,
) -> String {
    let layout = Layout::for_terminal_width(term_width);
    let mut out = String::new();

    let mut active = 0;
    let mut paused = 0;
    let mut completed = 0;
    let mut failed = 0;
    let mut stopped = 0;
    let mut reviewing = 0;

    // Group entries by repository path: a supervisor supervises one worktree
    // at a time, so this yields one coherent table per repository, each with
    // its own counters. `BTreeMap` keeps repositories in path order, so the
    // dashboard never reshuffles between ticks.
    let mut repos: BTreeMap<&str, RepoGroup<'_>> = BTreeMap::new();
    for entry in entries {
        match entry.status {
            RegistryStatus::Running => active += 1,
            RegistryStatus::Paused => paused += 1,
            RegistryStatus::Reviewing => reviewing += 1,
            RegistryStatus::Completed => completed += 1,
            RegistryStatus::Failed => failed += 1,
            RegistryStatus::Stopped => stopped += 1,
        }
        let repo = entry.repo_path.as_deref().unwrap_or(DEFAULT_REPO_KEY);
        repos.entry(repo).or_insert_with(RepoGroup::new).push(entry);
    }

    let total = entries.len();
    let time_str = {
        let secs = now % 60;
        let mins = (now / 60) % 60;
        let hours = (now / 3600) % 24;
        format!("{hours:02}:{mins:02}:{secs:02}")
    };

    // Header bar, sized to the terminal so the rule never wraps.
    let rule_width = layout.total.min(96);
    let title = "── MINI-SWE SWARM SUPERVISOR ";
    let tail = format!("{time_str} ──");
    let dashes = rule_width
        .saturating_sub(visible_width(title))
        .saturating_sub(visible_width(&tail));
    let header = format!("{title}{}{tail}", "─".repeat(dashes));
    if use_color {
        out.push_str(&format!("\x1b[1;36m{header}\x1b[0m\n"));
    } else {
        out.push_str(&header);
        out.push('\n');
    }
    // The counter strip folds onto as many lines as the terminal needs instead
    // of wrapping mid-item on a narrow window.
    out.push_str(&stats_lines(
        &[
            ("Active", active.to_string(), GREEN),
            ("Paused", paused.to_string(), YELLOW),
            ("Reviewing", reviewing.to_string(), MAGENTA),
            ("Completed", completed.to_string(), BLUE),
            ("Failed", failed.to_string(), RED),
            ("Stopped", stopped.to_string(), DIM),
            ("Total", total.to_string(), BOLD),
            ("Repos", repos.len().to_string(), BOLD),
        ],
        layout.total,
        use_color,
    ));
    out.push_str(&"\u{2501}".repeat(rule_width));
    out.push_str("\n\n");

    if entries.is_empty() {
        let reg_path = crate::pool::registry_dir();
        out.push_str(&format!(
            "No active or recent workers found in {}.\n",
            reg_path.display()
        ));
        out.push_str("Waiting for workers to dispatch... (Press Ctrl+C to exit)\n");
        return out;
    }

    for (repo_path, group) in repos {
        // The heading carries the path and every per-repo counter, so the rows
        // below it stay one clean table with no domain sub-sections. The
        // summary is joined on one line and the whole heading is then capped to
        // the terminal, so a long path costs counters, never a wrapped header.
        let summary = summary_line(&group.summary_items(), use_color);
        let heading = truncate_visible(&format!("[REPO: {repo_path}]  {summary}"), layout.total);
        if use_color {
            out.push_str(&format!("\x1b[1;36m{heading}\x1b[0m\n"));
        } else {
            out.push_str(&heading);
            out.push('\n');
        }

        match layout.shape {
            RowShape::Inline { .. } => {
                out.push_str(&header_line(&layout));
                out.push_str(&"\u{2500}".repeat(layout.fixed_width() + layout.op));
            }
            RowShape::Stacked => out.push_str(&stacked_header_line(&layout)),
        }
        out.push('\n');

        for w in &group.workers {
            let op = op_cell(w, &layout, use_color);
            match layout.shape {
                RowShape::Inline { .. } => {
                    let prefix = row_prefix(w, &layout, use_color, now);
                    out.push_str(&prefix);
                    out.push_str(&pad_visible(&op, layout.op));
                    out.push('\n');
                }
                RowShape::Stacked => {
                    out.push_str(&stack_row(w, &layout, use_color, now, &op));
                }
            }
        }
        out.push('\n');
    }

    out.push_str("Press Ctrl+C to exit monitor.\n");
    out
}

/// Column header matching the row layout.
fn header_line(layout: &Layout) -> String {
    let mut line = String::new();
    let cols = [
        ("ID", layout.id),
        ("PID", layout.pid),
        ("STATUS", layout.status),
        ("TURNS", layout.turns),
        ("MODEL", layout.model),
        ("UPTIME", layout.uptime),
    ];
    for (i, (label, width)) in cols.iter().enumerate() {
        line.push_str(&pad_visible(label, *width));
        if i + 1 == cols.len() {
            line.push_str(&" ".repeat(LAST_GAP));
        } else {
            line.push_str(&" ".repeat(GAP));
        }
    }
    line.push_str("LAST OP / TASK\n");
    line
}

/// Header for the stacked (narrow terminal) row shape.
fn stacked_header_line(layout: &Layout) -> String {
    let first = format!(
        "{}  {}  {}",
        pad_visible("ID", layout.id),
        pad_visible("PID", layout.pid),
        pad_visible("TURNS", layout.turns)
    );
    let second = format!(
        "{}  {}  {}",
        pad_visible("STATUS", layout.status),
        pad_visible("MODEL", layout.model),
        pad_visible("UPTIME", layout.uptime)
    );
    format!("{first}\n{second}\nLAST OP / TASK\n")
}

// ---------------------------------------------------------------------------
// Colour helpers
// ---------------------------------------------------------------------------

/// Bold, used for the totals.
const BOLD: &str = "\x1b[1m";
/// Bold green: active work.
const GREEN: &str = "\x1b[1;32m";
/// Bold yellow: paused.
const YELLOW: &str = "\x1b[1;33m";
/// Bold magenta: reviewing.
const MAGENTA: &str = "\x1b[1;35m";
/// Bold blue: completed.
const BLUE: &str = "\x1b[1;34m";
/// Bold red: failed.
const RED: &str = "\x1b[1;31m";
/// Dim white: stopped.
const DIM: &str = "\x1b[2;37m";

/// Render one `label: value` item, coloured when requested.
fn item_text(label: &str, value: &str, color: &str, use_color: bool) -> String {
    if use_color {
        format!("{label}: {color}{value}\x1b[0m")
    } else {
        format!("{label}: {value}")
    }
}

/// Join `label: value` items onto a single line for a repository heading.
///
/// Unlike [`stats_lines`] this never folds and never appends a newline: the
/// heading must stay one line so the table header sits directly under it, and
/// the caller caps the whole heading with [`truncate_visible`] afterwards.
fn summary_line(items: &[(&str, String, &'static str)], use_color: bool) -> String {
    let mut out = String::new();
    for (i, (label, value, color)) in items.iter().enumerate() {
        if i > 0 {
            out.push_str("  |  ");
        }
        out.push_str(&item_text(label, value, color, use_color));
    }
    out
}

/// Render the `label: value | label: value` counter strip, folded to `width`.
///
/// Folding keeps a narrow terminal from wrapping mid-item, which is what makes
/// the whole header readable at any width. Measurement always runs on the plain
/// form, so an escape never counts against the budget.
fn stats_lines(items: &[(&str, String, &'static str)], width: usize, use_color: bool) -> String {
    const SEP: &str = "  |  ";
    let sep_width = visible_width(SEP);
    let mut out = String::new();
    let mut line_width = 0usize;
    for (i, (label, value, color)) in items.iter().enumerate() {
        let plain = format!("{label}: {value}");
        let item_width = visible_width(&plain);
        if line_width > 0 && line_width + sep_width + item_width > width {
            out.push('\n');
            line_width = 0;
        }
        if line_width > 0 {
            out.push_str(SEP);
            line_width += sep_width;
        }
        out.push_str(&item_text(label, value, color, use_color));
        line_width += item_width;
        if i + 1 == items.len() {
            out.push('\n');
        }
    }
    if !out.ends_with('\n') {
        out.push('\n');
    }
    out
}

/// Best-effort terminal width in columns.
pub fn terminal_width() -> Option<usize> {
    if let Some(parsed) = env_parse::<usize>("MONITOR_WIDTH").filter(|&w| w > 0) {
        return Some(parsed);
    }
    terminal_size_via_tty()
}

/// Terminal width from `/dev/tty` (or `COLUMNS`), if it can be read.
#[cfg(unix)]
fn terminal_size_via_tty() -> Option<usize> {
    use std::os::fd::AsRawFd;

    let file = std::fs::OpenOptions::new().read(true).open("/dev/tty").ok()?;
    let mut size = libc::winsize {
        ws_row: 0,
        ws_col: 0,
        ws_xpixel: 0,
        ws_ypixel: 0,
    };
    let rc = unsafe { libc::ioctl(file.as_raw_fd(), libc::TIOCGWINSZ, &mut size) };
    if rc == 0 && size.ws_col > 0 {
        return Some(size.ws_col as usize);
    }
    columns_env()
}

#[cfg(not(unix))]
fn terminal_size_via_tty() -> Option<usize> {
    columns_env()
}

/// `COLUMNS` fallback shared by both platforms.
fn columns_env() -> Option<usize> {
    env_parse::<usize>("COLUMNS").filter(|c| *c > 0)
}

pub async fn run_monitor(once: bool) -> Result<()> {
    let is_tty = std::io::stdout().is_terminal();
    if once || !is_tty {
        let entries = load_all_registry_entries();
        let now = unix_timestamp();
        let output = render_dashboard_with_width(
            &entries,
            now,
            is_tty,
            terminal_width().unwrap_or(DEFAULT_TERMINAL_WIDTH),
        );
        println!("{output}");
        return Ok(());
    }

    struct CursorGuard;
    impl Drop for CursorGuard {
        fn drop(&mut self) {
            println!("\x1b[?25h");
            let _ = std::io::stdout().flush();
        }
    }

    // Hide cursor during interactive monitoring
    print!("\x1b[?25l");
    let _ = std::io::stdout().flush();
    let _guard = CursorGuard;

    loop {
        let entries = load_all_registry_entries();
        let now = unix_timestamp();
        // Re-read the width every tick so a terminal resize is picked up live.
        let output = render_dashboard_with_width(
            &entries,
            now,
            true,
            terminal_width().unwrap_or(DEFAULT_TERMINAL_WIDTH),
        );

        {
            let mut stdout = std::io::stdout().lock();
            let _ = write!(stdout, "\x1b[H\x1b[J{output}");
            let _ = stdout.flush();
        }

        tokio::select! {
            _ = tokio::signal::ctrl_c() => {
                break;
            }
            _ = tokio::time::sleep(std::time::Duration::from_millis(1000)) => {}
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pool::WorkerMetrics;

    /// Builder for registry rows: the tests below touch every field, so a
    /// positional 8-argument helper would be unreadable.
    struct Row {
        id: &'static str,
        status: RegistryStatus,
        step: usize,
        max_turns: usize,
        command: String,
        task: String,
        group: Option<&'static str>,
        repo: Option<&'static str>,
        metrics: WorkerMetrics,
        updated_at: u64,
    }

    impl Row {
        fn new(id: &'static str) -> Self {
            Self {
                id,
                status: RegistryStatus::Running,
                step: 1,
                max_turns: 100,
                command: String::new(),
                task: String::new(),
                group: None,
                repo: None,
                metrics: WorkerMetrics::default(),
                updated_at: 1050,
            }
        }

        fn status(mut self, status: RegistryStatus) -> Self {
            self.status = status;
            self
        }

        fn turns(mut self, step: usize, max_turns: usize) -> Self {
            self.step = step;
            self.max_turns = max_turns;
            self
        }

        fn command(mut self, command: &str) -> Self {
            self.command = command.to_string();
            self
        }

        fn task(mut self, task: &str) -> Self {
            self.task = task.to_string();
            self
        }

        fn group(mut self, group: &'static str) -> Self {
            self.group = Some(group);
            self
        }

        fn repo(mut self, repo: &'static str) -> Self {
            self.repo = Some(repo);
            self
        }

        fn updated_at(mut self, updated_at: u64) -> Self {
            self.updated_at = updated_at;
            self
        }

        fn metrics(mut self, repeat_blocks: usize, stagnation_nudges: usize) -> Self {
            self.metrics.repeat_blocks = repeat_blocks;
            self.metrics.stagnation_nudges = stagnation_nudges;
            self
        }

        fn build(self) -> WorkerRegistryEntry {
            WorkerRegistryEntry {
                id: self.id.into(),
                pid: 1234,
                task: self.task,
                model: "ninja".into(),
                status: self.status,
                step: self.step,
                max_turns: self.max_turns,
                last_command: self.command,
                question: None,
                started_at: 1000,
                updated_at: self.updated_at,
                group: self.group.map(str::to_string),
                repo_path: self.repo.map(str::to_string),
                owner: None,
                metrics: self.metrics,
            }
        }
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(5), "05s");
        assert_eq!(format_duration(65), "01m 05s");
        assert_eq!(format_duration(3665), "01h 01m");
    }

    #[test]
    fn test_render_dashboard_empty() {
        let text = render_dashboard(&[], 1000, false);
        assert!(text.contains("MINI-SWE SWARM SUPERVISOR"));
        assert!(text.contains("No active or recent workers found"));
    }

    #[test]
    fn test_render_dashboard_groups_by_repository() {
        let entries = vec![
            Row::new("a1b2c3d4e5")
                .task("Fix architecture")
                .command("cargo test")
                .turns(10, 200)
                .group("audits")
                .repo("/home/dev/proj-a")
                .build(),
            Row::new("b2c3d4e5f6")
                .task("Fix something else")
                .command("ask")
                .turns(5, 100)
                .status(RegistryStatus::Paused)
                .repo("/home/dev/proj-b")
                .build(),
            Row::new("c3d4e5f6a7")
                .task("Third task")
                .command("cargo build")
                .turns(3, 100)
                .group("audits")
                .repo("/home/dev/proj-a")
                .build(),
            Row::new("d4e5f6a7b8")
                .task("No repo recorded")
                .command("completed")
                .turns(9, 100)
                .status(RegistryStatus::Completed)
                .build(),
        ];

        let text = render_dashboard(&entries, 1060, false);

        // One unified table per repository, with all of its workers together,
        // each heading carrying that repository's own counters.
        assert!(
            text.contains("[REPO: /home/dev/proj-a]  total: 2  |  active: 2"),
            "missing per-repo summary:\n{text}"
        );
        assert!(
            text.contains("[REPO: /home/dev/proj-b]  total: 1  |  paused: 1"),
            "missing per-repo summary:\n{text}"
        );
        assert!(
            text.contains("[REPO: local]  total: 1  |  completed: 1"),
            "missing per-repo summary:\n{text}"
        );
        // A repository with nothing but terminal workers spends its heading on
        // `total` alone, never on the states that are zero.
        assert!(
            text.contains("[REPO: local]  total: 1  |  completed: 1\n"),
            "zero counters must be dropped:\n{text}"
        );
        assert!(!text.contains("[SWARM:"));
        assert_eq!(text.matches("LAST OP / TASK").count(), 3);

        // ...and the swarm tag demoted to a [tag] prefix in the task column.
        let proj_a = &text[text.find("[REPO: /home/dev/proj-a]").unwrap()
            ..text.find("[REPO: /home/dev/proj-b]").unwrap()];
        assert!(proj_a.contains("[audits] cargo test"));
        // The ID column is 8 columns wide, so a full UUID is capped there.
        assert!(proj_a.contains("a1b2c3d\u{2026}"), "missing worker a:\n{proj_a}");
        assert!(proj_a.contains("c3d4e5f\u{2026}"), "missing worker c:\n{proj_a}");
        assert!(!proj_a.contains("b2c3d4e"), "proj-b worker leaked into proj-a");

        // The counters are global, above every repository table.
        assert!(text.contains("Active: 2"));
        assert!(text.contains("Repos: 3"));
    }

    #[test]
    fn test_repository_grouping_ignores_domain_tags() {
        // Three domain tags, one repository: the tags must not sub-partition
        // the dashboard, they only prefix the task column of their own row.
        let entries: Vec<WorkerRegistryEntry> = [
            ("aud001", RegistryStatus::Running, "audits"),
            ("perf001", RegistryStatus::Running, "perf"),
            ("sec001", RegistryStatus::Completed, "sec"),
        ]
        .iter()
        .map(|(id, status, group)| {
            Row::new(id)
                .status(*status)
                .task("Task for the domain")
                .command("cargo test")
                .turns(3, 100)
                .group(group)
                .repo("/home/dev/proj-a")
                .build()
        })
        .collect();

        let text = render_dashboard(&entries, 1060, false);

        // A single table holds all three workers of the repository...
        assert_eq!(text.matches("[REPO: /home/dev/proj-a]").count(), 1);
        assert_eq!(text.matches("LAST OP / TASK").count(), 1);
        // ...with one summary counting every state at once...
        assert!(
            text.contains("[REPO: /home/dev/proj-a]  total: 3  |  active: 2  |  completed: 1"),
            "per-repo summary must consolidate the states:\n{text}"
        );
        // ...and the domain tags demoted to a per-row prefix.
        for tag in ["[audits]", "[perf]", "[sec]"] {
            assert!(text.contains(tag), "missing {tag} prefix:\n{text}");
        }
    }

    #[test]
    fn test_render_dashboard_completed_uptime_is_frozen() {
        let entries = vec![Row::new("done01")
            .status(RegistryStatus::Completed)
            .task("Finished task")
            .command("completed")
            .repo("local")
            .updated_at(1065) // Ran 65s (started_at = 1000)
            .build()];

        // Rendered long after completion (now = 5000) the uptime must stay 65s.
        let text = render_dashboard(&entries, 5000, false);
        assert!(text.contains("01m 05s"), "expected frozen duration 01m 05s, got:\n{text}");
    }

    #[test]
    fn test_reviewing_status_is_indicated() {
        let entries = vec![
            Row::new("rev123")
                .status(RegistryStatus::Reviewing)
                .turns(42, 120)
                .task("Audit the diff")
                .command("[review] cargo clippy")
                .group("audits")
                .repo("/repo/x")
                .build(),
            Row::new("run456")
                .task("Implement feature")
                .command("cargo test")
                .repo("/repo/x")
                .build(),
        ];

        let plain = render_dashboard(&entries, 1100, false);
        assert!(plain.contains("REVIEWING"), "missing review badge:\n{plain}");
        assert!(plain.contains("[REVIEW]"), "missing review tag:\n{plain}");
        assert!(plain.contains("Reviewing: 1"), "reviewing not counted:\n{plain}");
        // The review marker must not be mistaken for a running row.
        assert!(plain.contains("\u{25c6} REVIEWING"));
        assert!(!plain.contains("[audits]"), "review tag replaces the group tag");

        let colored = render_dashboard(&entries, 1100, true);
        assert!(colored.contains("\u{25c6}"), "missing review marker:\n{colored}");
        assert!(colored.contains("1;35m"));
    }

    #[test]
    fn test_turn_progress_indicator() {
        // 5 of 10 turns through an 8-cell bar: 4 filled, 4 empty.
        assert_eq!(progress_bar(5, 10), "[####....]");
        assert_eq!(progress_bar(0, 10), "[........]");
        assert_eq!(progress_bar(10, 10), "[########]");
        // Clamped, never panicking on a zero budget or over-budget counter.
        assert_eq!(progress_bar(3, 0), "[########]");
        assert_eq!(progress_bar(99, 10), "[########]");

        let text = render_dashboard(
            &[Row::new("prog001")
                .task("Progress row")
                .command("cargo test")
                .turns(5, 10)
                .repo("local")
                .build()],
            1060,
            false,
        );
        let row = text.lines().find(|l| l.starts_with("prog001")).unwrap();
        assert!(row.contains("[####....] 5/10"), "bad turns cell:\n{row}");

        // A width too narrow for bar + counter keeps the counter and drops the bar.
        let narrow = Layout::for_terminal_width(80);
        assert!(!narrow.shows_progress(), "80 cols must drop the bar");
        let cell = turns_cell(5, 10, &narrow);
        assert!(cell.contains("5/10") && !cell.contains('#'), "bad cell: {cell:?}");
    }

    #[test]
    fn test_layout_scales_with_terminal_width() {
        let narrow = Layout::for_terminal_width(80);
        let wide = Layout::for_terminal_width(240);

        assert_eq!(narrow.total, 80);
        assert_eq!(wide.total, 240);
        assert!(narrow.op < wide.op, "op column must grow with the terminal");
        // Wide terminals earn the progress bar; medium ones do not.
        assert!(wide.shows_progress());
        assert!(!narrow.shows_progress());
        // Both stay inside the dynamic cap.
        assert!(narrow.op <= MAX_OP_WIDTH && wide.op <= MAX_OP_WIDTH);
        assert!(narrow.op >= MIN_OP_WIDTH && wide.op >= MIN_OP_WIDTH);
    }

    #[test]
    fn test_layout_is_bounded_for_extreme_widths() {
        // Below the usable floor everything is raised, never a zero-width cell.
        for width in [0usize, 1, 10, 40] {
            let layout = Layout::for_terminal_width(width);
            assert_eq!(layout.total, MIN_TERMINAL_WIDTH);
            assert!(layout.op >= 1);
        }
        // Too narrow for one line: stack it.
        let cramped = Layout::for_terminal_width(MIN_TERMINAL_WIDTH);
        assert_eq!(cramped.shape, RowShape::Stacked);

        // A huge terminal must not become a huge row: the cap holds.
        let huge = Layout::for_terminal_width(usize::MAX);
        assert_eq!(huge.op, MAX_OP_WIDTH);
        assert!(huge.shows_progress());
        assert_eq!(huge.shape, RowShape::Inline { progress: true });
    }

    #[test]
    fn test_long_command_is_capped_by_width() {
        let long_command = format!("echo {}", "x".repeat(400));
        let entries = vec![Row::new("long001")
            .task("A task with a very long description that should also be capped")
            .command(&long_command)
            .turns(1, 10)
            .repo("local")
            .build()];

        let narrow = render_dashboard_with_width(&entries, 1060, false, 84);
        let narrow_row = narrow.lines().find(|l| l.starts_with("long001")).unwrap();
        assert!(
            narrow_row.contains('\u{2026}'),
            "narrow render must ellipsize the long command:\n{narrow_row}"
        );
        assert!(narrow_row.chars().count() <= 84, "row overflows 84 cols:\n{narrow_row}");

        let wide = render_dashboard_with_width(&entries, 1060, false, 240);
        let wide_row = wide.lines().find(|l| l.starts_with("long001")).unwrap();
        assert!(
            wide_row.contains('\u{2026}'),
            "even wide renders stay capped:\n{wide_row}"
        );
        assert!(wide_row.chars().count() <= 240, "row overflows 240 cols:\n{wide_row}");
        // The cap is dynamic: a wider terminal reveals more of the command.
        assert!(
            wide_row.chars().count() > narrow_row.chars().count(),
            "wider terminal must reveal more text"
        );
        // A very wide terminal must not become 400 columns of command.
        let huge = render_dashboard_with_width(&entries, 1060, false, 400);
        let huge_row = huge.lines().find(|l| l.starts_with("long001")).unwrap();
        assert!(visible_width(huge_row) <= FIXED_FULL + MAX_OP_WIDTH);
        assert_eq!(visible_width(huge_row), visible_width(wide_row));
    }

    #[test]
    fn test_visible_width_ignores_ansi_escapes() {
        assert_eq!(visible_width("plain"), 5);
        assert_eq!(visible_width("\x1b[1;32mRUNNING\x1b[0m"), 7);
        assert_eq!(
            pad_visible("\x1b[1;32mRUN\x1b[0m", 6),
            "\x1b[1;32mRUN\x1b[0m   "
        );
    }

    #[test]
    fn test_truncate_visible_keeps_escapes_and_closes_them() {
        // The ellipsis is paid for out of the budget, so the cap is exact.
        assert_eq!(truncate_visible("abcdef", 4), "abc\u{2026}");
        assert_eq!(visible_width(&truncate_visible("abcdef", 4)), 4);
        assert_eq!(truncate_visible("abc", 8), "abc");
        assert_eq!(truncate_visible("abc", 0), "");

        let colored = truncate_visible("\x1b[1;32mRUNNING\x1b[0m", 4);
        assert_eq!(visible_width(&colored), 4);
        assert!(colored.starts_with("\x1b[1;32m"));
        assert!(!colored.contains("\x1b[0m"), "trailing reset must be dropped");
        assert!(colored.ends_with('\u{2026}'));
    }

    #[test]
    fn test_wrap_visible_respects_width() {
        let text = "alpha beta gamma delta epsilon";
        for width in [6usize, 8, 12] {
            for line in wrap_visible(text, width) {
                assert!(
                    line.chars().count() <= width,
                    "line '{line}' exceeds width {width}"
                );
            }
        }
        assert_eq!(wrap_visible("", 10), vec![String::new()]);
        assert_eq!(wrap_visible("short", 10), vec!["short".to_string()]);
    }

    #[test]
    fn test_colored_rows_stay_aligned() {
        let entries: Vec<WorkerRegistryEntry> = ["c1", "c2", "c3", "c4"]
            .iter()
            .zip([
                RegistryStatus::Running,
                RegistryStatus::Reviewing,
                RegistryStatus::Completed,
                RegistryStatus::Failed,
            ])
            .map(|(id, status)| {
                Row::new(id)
                    .status(status)
                    .task("T")
                    .command("cargo test")
                    .repo("r")
                    .build()
            })
            .collect();

        let width = 240;
        let text = render_dashboard_with_width(&entries, 1100, true, width);
        let rows: Vec<&str> = text
            .lines()
            .filter(|l| ["c1", "c2", "c3", "c4"].iter().any(|id| l.starts_with(id)))
            .collect();
        assert_eq!(rows.len(), 4);
        for row in &rows {
            assert!(
                visible_width(row) <= width,
                "colored row overflows: {row:?} -> {}",
                visible_width(row)
            );
        }
        // The review badge and its marker must not shift any column.
        assert_eq!(visible_width(rows[0]), visible_width(rows[1]));
        assert_eq!(visible_width(rows[1]), visible_width(rows[2]));
    }

    #[test]
    fn test_every_line_fits_every_width() {
        let entries: Vec<WorkerRegistryEntry> = [
            ("w1", RegistryStatus::Running, "/home/dev/very-long-project-name-here", "[audits]"),
            ("w2", RegistryStatus::Reviewing, "/home/dev/very-long-project-name-here", "[audits]"),
            ("w3", RegistryStatus::Failed, "local", ""),
        ]
        .iter()
        .map(|(id, status, repo, group)| {
            let mut row = Row::new(id)
                .status(*status)
                .task("Refactor the authentication middleware into smaller cohesive pieces")
                .command("cargo test --all --verbose")
                .turns(37, 250)
                .repo(repo);
            if !group.is_empty() {
                row = row.group(group);
            }
            row.build()
        })
        .collect();

        for width in [60usize, 70, 80, 96, 120, 160, 240, 400] {
            for use_color in [false, true] {
                let text = render_dashboard_with_width(&entries, 1100, use_color, width);
                for line in text.lines() {
                    assert!(
                        visible_width(line) <= width,
                        "line overflows {width} cols ({}): {line:?}",
                        visible_width(line)
                    );
                }
            }
        }
    }

    /// The stacked row has room the inline row does not, so the repeat/nudge
    /// counts ride its label line — and never overflow it.
    #[test]
    fn test_stacked_row_carries_the_repeat_and_nudge_counts() {
        let entries = vec![Row::new("health01")
            .task("Refactor the authentication middleware into smaller pieces")
            .command("cargo test --all")
            .metrics(3, 1)
            .repo("local")
            .build()];

        let text = render_dashboard_with_width(&entries, 1060, false, MIN_TERMINAL_WIDTH);
        assert!(
            text.contains("3 repeats, 1 nudge"),
            "the stacked row must show the health counters:\n{text}"
        );
        for line in text.lines() {
            assert!(
                visible_width(line) <= MIN_TERMINAL_WIDTH,
                "line overflows the terminal width: {line:?}"
            );
        }

        // The inline row has no room for a new column, so it shows none.
        let inline = render_dashboard_with_width(&entries, 1060, false, 200);
        assert!(!inline.contains("repeats"), "{inline}");
    }

    #[test]
    fn test_narrow_terminal_stacks_rows() {
        let entries = vec![Row::new("wrap001")
            .task("Refactor the authentication middleware into smaller cohesive pieces")
            .command("cargo test --all")
            .repo("local")
            .build()];

        let text = render_dashboard_with_width(&entries, 1060, false, MIN_TERMINAL_WIDTH);
        let layout = Layout::for_terminal_width(MIN_TERMINAL_WIDTH);
        assert_eq!(layout.shape, RowShape::Stacked);
        for line in text.lines() {
            assert!(
                visible_width(line) <= MIN_TERMINAL_WIDTH,
                "line overflows the terminal width: {line:?}"
            );
        }
        // All fixed columns survive, and so does the task text.
        assert!(text.contains("wrap001") && text.contains("1234") && text.contains("ninja"));
        assert!(text.contains("Refactor"), "task text must survive stacking:\n{text}");
        assert!(text.contains("cargo test --all"), "command must survive:\n{text}");
    }
}
