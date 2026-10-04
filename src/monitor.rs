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

/// Compact statusLine summary: live workers plus terminal rows from the last
/// five minutes. Reviewing workers count as running; failures remain visible.
pub fn format_status_line(entries: &[WorkerRegistryEntry], now: u64) -> String {
    let mut counts = [0usize; 5];
    for entry in entries {
        if entry.status.is_terminal()
            && now.saturating_sub(entry.updated_at) > crate::pool::DEFAULT_TERMINAL_TTL_SECS
        {
            continue;
        }
        let index = match entry.status {
            RegistryStatus::Running | RegistryStatus::Reviewing => 0,
            RegistryStatus::Paused => 1,
            RegistryStatus::Completed => 2,
            RegistryStatus::Failed => 3,
            RegistryStatus::Exhausted | RegistryStatus::Stopped => 4,
            RegistryStatus::Interrupted => 5,
        };
        counts[index] += 1;
    }
    let parts: Vec<_> = counts
        .into_iter()
        .zip([
            "running",
            "needs input",
            "done",
            "failed",
            "stopped",
            "interrupted",
        ])
        .filter(|(count, _)| *count > 0)
        .map(|(count, label)| format!("{count} {label}"))
        .collect();
    if parts.is_empty() {
        String::new()
    } else {
        format!("⚙ {}", parts.join(" · "))
    }
}

/// Print the registry-only status line, ignoring unavailable rows and broken
/// output pipes so a statusLine command never disrupts the host UI.
pub fn print_status_line() {
    let output = format_status_line(
        &crate::pool::load_registry_entries_read_only(),
        unix_timestamp(),
    );
    if !output.is_empty() {
        let _ = writeln!(std::io::stdout().lock(), "{output}");
    }
}

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
            op: total
                .saturating_sub(fixed)
                .clamp(MIN_OP_WIDTH, MAX_OP_WIDTH),
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
        pad_visible(
            &format!("{:<width$}", w.pid, width = layout.pid),
            layout.pid
        ),
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
    exhausted: usize,
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
            exhausted: 0,
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
            RegistryStatus::Exhausted => self.exhausted += 1,
            RegistryStatus::Stopped => self.stopped += 1,
            RegistryStatus::Interrupted => self.stopped += 1,
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
            ("exhausted", self.exhausted, DIM),
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
    let mut exhausted = 0;
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
            RegistryStatus::Exhausted => exhausted += 1,
            RegistryStatus::Stopped => stopped += 1,
            RegistryStatus::Interrupted => stopped += 1,
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
            ("Exhausted", exhausted.to_string(), DIM),
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

    let file = std::fs::OpenOptions::new()
        .read(true)
        .open("/dev/tty")
        .ok()?;
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

// ----------
// Interactive mini-TUI
// ----------

/// Maximum number of turns kept in memory per worker detail view.
///
/// The history log is append-only and can grow without bound over a long
/// worker, so the detail view keeps only the newest turns in memory and trims
/// the rest as it reads. Bounded by construction; the reader never grows past
/// this even when the file on disk is much longer.
pub const MAX_TURNS_IN_MEMORY: usize = 500;

/// A single parsed turn of a worker's history log, as the detail view shows it.
///
/// One assistant turn plus its tool result (or the user-role output that
/// answers a code-block turn) collapses into one row: the step number, the
/// command it ran, the exit code the harness recorded, and the last few lines
/// of output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TurnView {
    /// 1-based step number this turn belongs to.
    pub step: usize,
    /// The bash command the assistant ran (empty for a no-command turn).
    pub command: String,
    /// The exit code recorded by the harness, when the output named one.
    pub exit_code: Option<i32>,
    /// The last few lines of the tool output, newest last.
    pub output_lines: Vec<String>,
}

/// Incremental reader over a worker's append-only history log.
///
/// Remembers the byte offset it has consumed and parses only the lines that
/// were appended since the last read, so a refresh never re-reads the whole
/// file. Keeps at most [`MAX_TURNS_IN_MEMORY`] turns; older ones are dropped.
/// A torn final line (a crash mid-append) is skipped and the offset advances
/// past it, so the next read picks up after it.
#[derive(Debug, Clone, Default)]
pub struct HistoryReader {
    /// Byte offset the reader has consumed so far.
    pub offset: u64,
    /// Parsed turns, newest last, capped at [`MAX_TURNS_IN_MEMORY`].
    pub turns: Vec<TurnView>,
}

/// Number of output lines kept per turn.
const TURN_TAIL_LINES: usize = 5;

/// Turns one PgUp/PgDn moves the detail view: a page, not a line, so a long
/// history stays navigable without repeating the key.
const PAGE_TURNS: usize = 10;

impl HistoryReader {
    /// Read every line appended to `path` since the last read and fold the new
    /// turns into `self.turns`.
    ///
    /// The file is opened read-only and only the bytes after `self.offset` are
    /// read, so a refresh is proportional to what changed, never to the whole
    /// log. A missing file (a worker with no history yet) is not an error.
    pub fn read_incremental(&mut self, path: &std::path::Path) -> std::io::Result<()> {
        use std::io::{Read, Seek, SeekFrom};
        let mut file = match std::fs::File::open(path) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(e) => return Err(e),
        };
        let len = file.metadata()?.len();
        if len < self.offset {
            // The file shrank (a rewrite replaced the log): start over.
            self.offset = 0;
            self.turns.clear();
        }
        if len <= self.offset {
            return Ok(());
        }
        file.seek(SeekFrom::Start(self.offset))?;
        let mut buf = String::new();
        // The offset follows what was *read*, not the `len` stat'd above: the
        // writer appends between the two calls, and an offset taken from the
        // stale length would re-parse those bytes on the next refresh and
        // duplicate their turns.
        let read = file.read_to_string(&mut buf)? as u64;
        self.offset = self.offset.saturating_add(read);
        self.consume_lines(&buf);
        Ok(())
    }

    /// Fold freshly-read lines into `self.turns`, capping the memory.
    fn consume_lines(&mut self, buf: &str) {
        let mut pending: Option<TurnView> = None;
        let mut step = self.turns.last().map(|t| t.step).unwrap_or(0);
        for line in buf.lines() {
            if line.trim().is_empty() {
                continue;
            }
            if let Some(turn) = parse_turn_line(line, step) {
                step = turn.step;
                if let Some(done) = pending.take() {
                    self.turns.push(done);
                }
                pending = Some(turn);
            } else if let Some(output) = parse_tool_output(line)
                && let Some(turn) = pending.as_mut()
            {
                turn.exit_code = output.0;
                turn.output_lines = output.1;
            }
            // A torn line (a crash mid-append) parses as neither; the offset
            // already advanced past it, so the next read continues after it.
        }
        if let Some(done) = pending.take() {
            self.turns.push(done);
        }
        if self.turns.len() > MAX_TURNS_IN_MEMORY {
            let excess = self.turns.len() - MAX_TURNS_IN_MEMORY;
            self.turns.drain(..excess);
        }
    }
}

/// Parse one JSONL line of a history log into a [`TurnView`], if it is a turn.
///
/// The metadata line (line one) and any non-turn message parse to `None`. A
/// turn is an assistant message that ran a command: its tool result carries
/// the `COMMAND OUTPUT (exit code: N)` prefix and the output tail. `step` is
/// the running step counter the caller maintains; it advances only when a turn
/// is found.
fn parse_turn_line(line: &str, step: usize) -> Option<TurnView> {
    let value: serde_json::Value = serde_json::from_str(line).ok()?;
    if value.get("role")?.as_str()? != "assistant" {
        return None;
    }
    // The command is the first bash tool call's `command` argument, or the
    // assistant's prose when no tool call named one.
    let command = value
        .get("tool_calls")
        .and_then(|calls| calls.as_array())
        .and_then(|calls| {
            calls.iter().find_map(|call| {
                if call.get("function")?.get("name")?.as_str()? != "bash" {
                    return None;
                }
                let args = call.get("function")?.get("arguments")?.as_str()?;
                serde_json::from_str::<serde_json::Value>(args)
                    .ok()
                    .and_then(|a| {
                        a.get("command")
                            .and_then(|c| c.as_str())
                            .map(str::to_string)
                    })
            })
        })
        .unwrap_or_else(|| {
            value
                .get("content")
                .and_then(|c| c.as_str())
                .unwrap_or("")
                .trim()
                .to_string()
        });
    Some(TurnView {
        step: step + 1,
        command,
        exit_code: None,
        output_lines: Vec::new(),
    })
}

/// Parse a tool-result line into its exit code and output tail.
///
/// The harness records every executed command as `COMMAND OUTPUT (exit code:
/// N)` followed by the output; the detail view shows only the last few lines
/// so one noisy turn cannot push the rest off screen.
fn parse_tool_output(line: &str) -> Option<(Option<i32>, Vec<String>)> {
    let value: serde_json::Value = serde_json::from_str(line).ok()?;
    let role = value.get("role")?.as_str()?;
    let content = value.get("content")?.as_str()?;
    let prefix = "COMMAND OUTPUT (exit code: ";
    let Some(rest) = content.strip_prefix(prefix) else {
        // A tool result without the harness prefix (an error, a block note):
        // still worth showing under the turn, without an exit code. User-role
        // lines are instructions, not output, so only tool lines qualify.
        if role != "tool" {
            return None;
        }
        return Some((None, tail_lines(content)));
    };
    let (code_text, output) = rest.split_once(')')?;
    let exit_code = code_text.trim().parse::<i32>().ok();
    // A code-block (prose) turn is answered by a user message carrying the
    // same harness prefix, so both roles attach output to the pending turn.
    if role != "tool" && role != "user" {
        return None;
    }
    Some((exit_code, tail_lines(output)))
}

/// Keep the last [`TURN_TAIL_LINES`] non-blank lines of a tool output.
fn tail_lines(output: &str) -> Vec<String> {
    output
        .lines()
        .map(str::trim_end)
        .filter(|l| !l.is_empty())
        .map(str::to_string)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .take(TURN_TAIL_LINES)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect()
}

/// A key the interactive loop can receive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Key {
    Up,
    Down,
    PageUp,
    PageDown,
    Enter,
    Esc,
    /// A printable character (e.g. `q`, `g`, `f`, `j`, `k`).
    Char(char),
}

/// What a keypress does to the UI state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    /// Move the selection up one worker (list).
    MoveUp,
    /// Move the selection down one worker (list).
    MoveDown,
    /// Open the selected worker's detail view (list).
    OpenDetail,
    /// Go back to the list (detail).
    Back,
    /// Quit the monitor entirely (list).
    Quit,
    /// Toggle group collapse/expand (list).
    ToggleGroups,
    /// Toggle follow mode (detail).
    ToggleFollow,
    /// Scroll the detail view up one page.
    ScrollUp,
    /// Scroll the detail view down one page.
    ScrollDown,
    /// No state change (e.g. `g`/`f` in the wrong view).
    None,
}

/// The two views of the mini-TUI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum View {
    List,
    Detail,
}

/// The interactive UI state.
#[derive(Debug, Clone)]
pub struct UiState {
    /// Which view is active.
    pub view: View,
    /// Index of the selected worker within the flat list of visible workers.
    pub selection: usize,
    /// Whether workers are shown under their group headings.
    pub groups_expanded: bool,
    /// Whether the detail view sticks to the newest turn.
    pub follow: bool,
    /// Scroll offset (in turns) of the detail view.
    pub scroll: usize,
}

impl Default for UiState {
    fn default() -> Self {
        Self {
            view: View::List,
            selection: 0,
            groups_expanded: true,
            follow: true,
            scroll: 0,
        }
    }
}

impl UiState {
    /// Apply a key to the state and return what it did.
    ///
    /// Pure and deterministic: the interactive loop feeds raw keys in and the
    /// resulting action drives the redraw. `worker_count` is the number of
    /// visible workers in the list, used to clamp the selection.
    pub fn apply_key(&mut self, key: Key, worker_count: usize) -> Action {
        match self.view {
            View::List => match key {
                Key::Up => {
                    if self.selection > 0 {
                        self.selection -= 1;
                        Action::MoveUp
                    } else {
                        Action::None
                    }
                }
                Key::Down => {
                    if worker_count > 0 && self.selection + 1 < worker_count {
                        self.selection += 1;
                        Action::MoveDown
                    } else {
                        Action::None
                    }
                }
                Key::Enter => {
                    if worker_count > 0 {
                        self.view = View::Detail;
                        self.follow = true;
                        self.scroll = 0;
                        Action::OpenDetail
                    } else {
                        Action::None
                    }
                }
                Key::Esc | Key::Char('q') => Action::Quit,
                Key::Char('g') => {
                    self.groups_expanded = !self.groups_expanded;
                    Action::ToggleGroups
                }
                _ => Action::None,
            },
            View::Detail => match key {
                // Ctrl+C arrives as `q` from `read_key`: it steps back here
                // and quits from the list, so two presses always exit.
                Key::Esc | Key::Char('q') => {
                    self.view = View::List;
                    Action::Back
                }
                Key::PageUp => {
                    self.scroll = self.scroll.saturating_sub(PAGE_TURNS);
                    Action::ScrollUp
                }
                Key::PageDown => {
                    self.scroll = self.scroll.saturating_add(PAGE_TURNS);
                    Action::ScrollDown
                }
                Key::Char('f') => {
                    self.follow = !self.follow;
                    Action::ToggleFollow
                }
                _ => Action::None,
            },
        }
    }
}

/// Parse a raw byte sequence from the terminal into a [`Key`].
///
/// Handles the arrow keys and PgUp/PgDn escape sequences plus `j`/`k` as
/// Up/Down aliases; everything else maps to its character or `None` for
/// control bytes the UI ignores.
pub fn parse_key(bytes: &[u8]) -> Option<Key> {
    match bytes {
        [0x1b, b'[', b'A'] => Some(Key::Up),
        [0x1b, b'[', b'B'] => Some(Key::Down),
        [0x1b, b'[', b'5', b'~'] => Some(Key::PageUp),
        [0x1b, b'[', b'6', b'~'] => Some(Key::PageDown),
        [b'\r'] | [b'\n'] => Some(Key::Enter),
        [0x1b] => Some(Key::Esc),
        [b'j'] => Some(Key::Down),
        [b'k'] => Some(Key::Up),
        [b] if b.is_ascii_graphic() || *b == b' ' => Some(Key::Char(*b as char)),
        _ => None,
    }
}

/// Fit one worker to a single compact line of at most `width` visible columns.
///
/// Layout: `glyph id model step/max elapsed op` with the group demoted to a
/// `[group]` prefix inside the op column. The op text is truncated to whatever
/// the fixed columns leave behind, so no line ever overflows `width`.
/// Completed and retired-soon workers are dimmed.
/// Strip terminal escape sequences and control characters from untrusted text.
///
/// A worker's task, its last command, its pause question and above all its tool
/// output are model- or repository-written text, and the monitor draws them into
/// a raw-mode alternate-screen terminal. Rendered verbatim, an embedded
/// `\x1b[?1049l` (leave the alternate screen), a screen clear, a cursor move or
/// an OSC sequence could redraw, hide or replace what the operator is looking at
/// -- the monitor would then show a state the worker chose, not the state the
/// pool is in. The monitor's own styling is applied around the sanitized text,
/// never by it.
fn sanitize_text(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(ch) = chars.next() {
        match ch {
            '\x1b' => match chars.peek() {
                // CSI: parameters and intermediates, then one final byte.
                Some('[') => {
                    chars.next();
                    while let Some(c) = chars.next() {
                        if ('@'..='~').contains(&c) {
                            break;
                        }
                    }
                }
                // OSC: a string terminated by BEL or by ST (ESC \).
                Some(']') => {
                    chars.next();
                    while let Some(c) = chars.next() {
                        if c == '\x07' {
                            break;
                        }
                        if c == '\x1b' && chars.peek() == Some(&'\\') {
                            chars.next();
                            break;
                        }
                    }
                }
                // Any other two-byte escape (charset selection, and so on).
                Some(_) => {
                    chars.next();
                }
                None => {}
            },
            '\n' | '\t' => out.push(ch),
            // Every other C0 control, DEL and the C1 range is invisible at best
            // and a terminal command at worst; none of them is content.
            c if (c as u32) < 0x20 || c == '\u{7f}' || ('\u{80}'..='\u{9f}').contains(&c) => {}
            c => out.push(c),
        }
    }
    out
}

pub fn fit_compact_row(w: &WorkerRegistryEntry, now: u64, width: usize) -> String {
    let glyph = status_glyph(w.status).to_string();
    let id = pad_visible(&truncate_visible(&sanitize_text(&w.id), 8), 8);
    let model = pad_visible(&truncate_visible(&sanitize_text(&w.model), 8), 8);
    let duration_secs = if w.status.is_terminal() {
        w.updated_at.saturating_sub(w.started_at)
    } else {
        now.saturating_sub(w.started_at)
    };
    let elapsed = format_duration(duration_secs);
    let turns = format!("{}/{}", w.step, w.max_turns);

    let group = w
        .group
        .as_deref()
        .filter(|g| !g.trim().is_empty())
        .map(|g| format!("[{}] ", sanitize_text(g.trim())));
    let op_text = if let Some(ref q) = w.question {
        format!("ASK: {}", sanitize_text(q))
    } else if !w.last_command.is_empty()
        && w.last_command != "completed"
        && w.last_command != "initializing"
    {
        sanitize_text(&w.last_command)
    } else {
        sanitize_text(w.task.lines().next().unwrap_or("").trim())
    };
    let op_text = format!("{}{op_text}", group.unwrap_or_default());

    let fixed = visible_width(&glyph)
        + 1
        + 8
        + 1
        + 8
        + 1
        + visible_width(&turns)
        + 1
        + visible_width(&elapsed)
        + 1;
    let op_width = width.saturating_sub(fixed).max(1);
    let op = truncate_visible(&op_text, op_width);

    let line = format!("{glyph} {id} {model} {turns} {elapsed} {op}");
    if dim_worker(w, now) {
        format!("\x1b[2m{line}\x1b[0m")
    } else {
        line
    }
}

/// Whether a worker row is dimmed: terminal workers, plus live workers that
/// stopped reporting (no update within the terminal TTL, so the sweeper will
/// soon retire them).
fn dim_worker(w: &WorkerRegistryEntry, now: u64) -> bool {
    if w.status.is_terminal() {
        return true;
    }
    now.saturating_sub(w.updated_at) > crate::pool::DEFAULT_TERMINAL_TTL_SECS
}

/// The one-character status glyph for a worker.
pub fn status_glyph(status: RegistryStatus) -> char {
    match status {
        RegistryStatus::Running => '\u{25cf}',
        RegistryStatus::Paused => '\u{25cb}',
        RegistryStatus::Reviewing => '\u{25c6}',
        RegistryStatus::Completed => '\u{2713}',
        RegistryStatus::Failed => '\u{2717}',
        RegistryStatus::Exhausted => '\u{26a0}',
        RegistryStatus::Stopped => '\u{25a0}',
        RegistryStatus::Interrupted => '\u{25b2}',
    }
}

/// One line of the key hint shown at the bottom of the interactive views.
pub fn key_hint(view: View) -> &'static str {
    match view {
        View::List => "Up/Down or j/k move - Enter detail - g groups - q quit",
        View::Detail => "PgUp/PgDn scroll - f follow - Esc/q back",
    }
}

// ----------
// Interactive terminal loop (raw mode, alternate screen, key input)
// ----------

/// Raw-mode terminal guard: enters raw mode plus the alternate screen on
/// creation and restores cooked mode, the cursor and the main screen on drop,
/// so every exit path (normal, error, panic unwind) leaves a usable terminal.
struct TerminalGuard {
    orig: libc::termios,
}

impl TerminalGuard {
    /// Enter raw mode on stdin and switch to the alternate screen.
    ///
    /// Fails when stdin is not a TTY or termios cannot be read, in which case
    /// the caller falls back to the non-interactive rendering.
    fn enter() -> std::io::Result<Self> {
        let mut orig: libc::termios = unsafe { std::mem::zeroed() };
        if unsafe { libc::tcgetattr(0, &mut orig) } != 0 {
            return Err(std::io::Error::last_os_error());
        }
        let mut raw = orig;
        raw.c_iflag &= !(libc::BRKINT | libc::ICRNL | libc::INPCK | libc::ISTRIP | libc::IXON);
        raw.c_oflag &= !libc::OPOST;
        raw.c_cflag |= libc::CS8;
        raw.c_lflag &= !(libc::ECHO | libc::ICANON | libc::IEXTEN | libc::ISIG);
        raw.c_cc[libc::VMIN] = 1;
        raw.c_cc[libc::VTIME] = 0;
        if unsafe { libc::tcsetattr(0, libc::TCSANOW, &raw) } != 0 {
            return Err(std::io::Error::last_os_error());
        }
        {
            let mut stdout = std::io::stdout().lock();
            let _ = write!(stdout, "\x1b[?1049h\x1b[?25l");
            let _ = stdout.flush();
        }
        Ok(Self { orig })
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        unsafe {
            libc::tcsetattr(0, libc::TCSANOW, &self.orig);
        }
        {
            let mut stdout = std::io::stdout().lock();
            let _ = write!(stdout, "\x1b[?25h\x1b[?1049l");
            let _ = stdout.flush();
        }
    }
}

/// Install a panic hook that restores the terminal before unwinding further.
///
/// [`TerminalGuard`] already restores on unwind, but a panic on another thread
/// would otherwise leave raw mode behind; the hook makes the restore
/// unconditional and then runs the previous hook.
fn install_panic_hook(orig: libc::termios) {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        unsafe {
            libc::tcsetattr(0, libc::TCSANOW, &orig);
        }
        {
            let mut stdout = std::io::stdout().lock();
            let _ = write!(stdout, "\x1b[?25h\x1b[?1049l");
            let _ = stdout.flush();
        }
        previous(info);
    }));
}

/// One selectable line of the compact list: a group heading or a worker row.
enum ListLine<'a> {
    Header(String),
    Worker(&'a WorkerRegistryEntry),
}

/// Build the compact list lines for `entries`, grouped by repository.
///
/// Each group gets a one-line header with its counts
/// (`running/done/failed`, every non-zero state) followed by one compact row
/// per worker. When `expanded` is false only the headers are returned. Returns
/// the lines plus the worker ids in display order, so the selection index maps
/// to a worker.
fn build_list_lines<'a>(
    entries: &'a [WorkerRegistryEntry],
    now: u64,
    width: usize,
    expanded: bool,
) -> (Vec<ListLine<'a>>, Vec<&'a WorkerRegistryEntry>) {
    let mut repos: BTreeMap<&str, RepoGroup<'a>> = BTreeMap::new();
    for entry in entries {
        let repo = entry.repo_path.as_deref().unwrap_or(DEFAULT_REPO_KEY);
        repos.entry(repo).or_insert_with(RepoGroup::new).push(entry);
    }
    let mut lines = Vec::new();
    let mut order = Vec::new();
    for (repo_path, group) in &repos {
        let summary = summary_line(&group.summary_items(), true);
        let heading = truncate_visible(&format!("[{}]  {summary}", sanitize_text(repo_path)), width);
        lines.push(ListLine::Header(format!("\x1b[1;36m{heading}\x1b[0m")));
        if expanded {
            for w in &group.workers {
                order.push(*w);
                lines.push(ListLine::Worker(w));
            }
        }
    }
    let _ = now;
    (lines, order)
}

/// Render the compact list view into `height` rows at `width` columns.
///
/// The selected worker's row is highlighted; the view scrolls so the selection
/// stays visible. Ends with the key hint line.
fn render_list(
    entries: &[WorkerRegistryEntry],
    state: &UiState,
    now: u64,
    width: usize,
    height: usize,
) -> String {
    let (lines, order) = build_list_lines(entries, now, width, state.groups_expanded);
    let selected_id = order.get(state.selection).map(|w| w.id.as_str());
    // Line index of the selected worker, for scroll-into-view.
    let mut selected_line = 0usize;
    for (i, line) in lines.iter().enumerate() {
        if let ListLine::Worker(w) = line
            && Some(w.id.as_str()) == selected_id
        {
            selected_line = i;
        }
    }
    let body_height = height.saturating_sub(2).max(1);
    let start = if lines.len() <= body_height {
        0
    } else {
        selected_line
            .saturating_sub(body_height.saturating_sub(1))
            .min(lines.len() - body_height)
    };
    let mut out = String::new();
    for line in lines.iter().skip(start).take(body_height) {
        match line {
            ListLine::Header(h) => {
                out.push_str(&truncate_visible(h, width));
                out.push('\n');
            }
            ListLine::Worker(w) => {
                let row = fit_compact_row(w, now, width);
                if Some(w.id.as_str()) == selected_id {
                    out.push_str(&format!("\x1b[7m{row}\x1b[0m\n"));
                } else {
                    out.push_str(&row);
                    out.push('\n');
                }
            }
        }
    }
    out.push_str(&truncate_visible(key_hint(View::List), width));
    out.push('\n');
    out
}

/// Render the detail view for one worker: its REPORT/question/status header on
/// top, then its turns newest-last. Follow mode sticks to the newest turn;
/// otherwise `state.scroll` turns are skipped from the end.
fn render_detail(
    entry: &WorkerRegistryEntry,
    reader: &HistoryReader,
    state: &UiState,
    now: u64,
    width: usize,
    height: usize,
) -> String {
    let mut out = String::new();
    let status_name = entry.status.display_name();
    out.push_str(&truncate_visible(
        &format!(
            "{} {} {} {}/{}",
            status_glyph(entry.status),
            entry.id,
            status_name,
            entry.step,
            entry.max_turns
        ),
        width,
    ));
    out.push('\n');
    out.push_str(&truncate_visible(
        &format!(
            "task: {}",
            sanitize_text(entry.task.lines().next().unwrap_or(""))
        ),
        width,
    ));
    out.push('\n');
    if let Some(ref q) = entry.question {
        out.push_str(&truncate_visible(&format!("question: {}", sanitize_text(q)), width));
        out.push('\n');
    }
    if let Some(ref report) = entry.report
        && !report.is_empty()
    {
        out.push_str(&truncate_visible(
            &format!(
                "report: {} | files: {} | tests: {} | risks: {}",
                sanitize_text(&report.done),
                sanitize_text(&report.files),
                sanitize_text(&report.tests),
                sanitize_text(&report.risks)
            ),
            width,
        ));
        out.push('\n');
    }
    let header_lines = out.lines().count();
    let body_height = height.saturating_sub(header_lines + 1).max(1);
    // Window over turns first (scroll is in turns), then fit the flattened
    // lines to the body: follow mode sticks to the newest turn.
    let skip = if state.follow {
        0
    } else {
        state.scroll.min(reader.turns.len().saturating_sub(1))
    };
    let end = reader.turns.len().saturating_sub(skip);
    let mut turn_lines: Vec<String> = Vec::new();
    for turn in &reader.turns[..end] {
        let code = turn
            .exit_code
            .map(|c| format!(" (exit {c})"))
            .unwrap_or_default();
        turn_lines.push(truncate_visible(
            &format!("#{} {}{}", turn.step, sanitize_text(&turn.command), code),
            width,
        ));
        for line in &turn.output_lines {
            turn_lines.push(truncate_visible(&format!("  {}", sanitize_text(line)), width));
        }
    }
    if turn_lines.is_empty() {
        turn_lines.push("(no turns recorded yet)".to_string());
    }
    let start = turn_lines.len().saturating_sub(body_height);
    for line in turn_lines.iter().skip(start) {
        out.push_str(line);
        out.push('\n');
    }
    let _ = now;
    out.push_str(&truncate_visible(key_hint(View::Detail), width));
    out.push('\n');
    out
}
/// Read one keypress from stdin, gathering a full escape sequence.
///
/// Arrow keys and PgUp/PgDn arrive as multi-byte sequences; after a lone ESC
/// byte a short `poll` decides whether more bytes follow (sequence) or the
/// user pressed Esc alone. Returns `None` on EOF or an unreadable stdin.
fn read_key() -> Option<Key> {
    use std::io::Read;
    use std::os::fd::AsRawFd;
    let stdin = std::io::stdin();
    let fd = stdin.as_raw_fd();
    let mut buf = [0u8; 32];
    let n = stdin.lock().read(&mut buf).ok()?;
    if n == 0 {
        return None;
    }
    let mut bytes = buf[..n].to_vec();
    // A lone ESC may be the start of a sequence: wait briefly for the rest.
    if bytes == [0x1b] {
        let mut pfd = libc::pollfd {
            fd,
            events: libc::POLLIN,
            revents: 0,
        };
        if unsafe { libc::poll(&mut pfd, 1, 50) } > 0 && pfd.revents & libc::POLLIN != 0 {
            let extra = stdin.lock().read(&mut buf).ok().unwrap_or(0);
            bytes.extend_from_slice(&buf[..extra]);
        }
    }
    if bytes == [0x03] {
        // Ctrl+C in raw mode arrives as a byte, not a signal: it steps back
        // in the detail view and quits from the list, like `q`.
        return Some(Key::Char('q'));
    }
    // `j`/`k` move in the list; elsewhere they are plain characters.
    parse_key(&bytes)
}

/// Terminal height in rows, defaulting when it cannot be read.
fn terminal_height() -> usize {
    #[cfg(unix)]
    {
        use std::os::fd::AsRawFd;
        if let Ok(file) = std::fs::OpenOptions::new().read(true).open("/dev/tty") {
            let mut size = libc::winsize {
                ws_row: 0,
                ws_col: 0,
                ws_xpixel: 0,
                ws_ypixel: 0,
            };
            if unsafe { libc::ioctl(file.as_raw_fd(), libc::TIOCGWINSZ, &mut size) } == 0
                && size.ws_row > 0
            {
                return size.ws_row as usize;
            }
        }
    }
    24
}

/// Run the interactive mini-TUI: compact list plus worker detail.
///
/// Enters raw mode and the alternate screen, then redraws at most once per
/// second plus on every keypress, so idle CPU stays near zero. The monitor is
/// strictly read-only: it never writes registry rows, histories or the hub.
async fn run_interactive() -> Result<()> {
    let guard = TerminalGuard::enter()?;
    install_panic_hook(guard.orig);
    let _guard = guard;

    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Key>();
    std::thread::spawn(move || {
        while let Some(key) = read_key() {
            if tx.send(key).is_err() {
                break;
            }
        }
    });

    let mut sigterm = {
        #[cfg(unix)]
        {
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()).ok()
        }
        #[cfg(not(unix))]
        {
            None::<tokio::signal::unix::Signal>
        }
    };

    let mut state = UiState::default();
    let mut readers: std::collections::HashMap<String, HistoryReader> =
        std::collections::HashMap::new();
    // Scratch root is resolved once: history files live beside the registry.
    let scratch = crate::worktree::ScratchRoot::from_env();

    loop {
        let entries = load_all_registry_entries();
        let now = unix_timestamp();
        let width = terminal_width().unwrap_or(DEFAULT_TERMINAL_WIDTH);
        let height = terminal_height();

        // Clamp the selection to the visible workers.
        let (_, order) = build_list_lines(&entries, now, width, state.groups_expanded);
        if !order.is_empty() {
            state.selection = state.selection.min(order.len() - 1);
        }
        // Drop readers for workers that left the registry, so a long session
        // never accumulates histories for workers that are gone.
        readers.retain(|id, _| entries.iter().any(|e| e.id == *id));

        let output = match state.view {
            View::List => render_list(&entries, &state, now, width, height),
            View::Detail => {
                let selected = order.get(state.selection);
                match selected {
                    Some(entry) => {
                        let path = crate::pool::history_log_path_in(&scratch, &entry.id);
                        let reader = readers.entry(entry.id.clone()).or_default();
                        let _ = reader.read_incremental(&path);
                        if state.follow {
                            state.scroll = 0;
                        } else {
                            state.scroll = state.scroll.min(reader.turns.len().saturating_sub(1));
                        }
                        render_detail(entry, reader, &state, now, width, height)
                    }
                    None => render_list(&entries, &state, now, width, height),
                }
            }
        };
        {
            let mut stdout = std::io::stdout().lock();
            let _ = write!(stdout, "\x1b[H\x1b[J{output}");
            let _ = stdout.flush();
        }

        enum Wake {
            Key(Key),
            Tick,
            Done,
        }
        let wake = async {
            {
                if let Some(sig) = sigterm.as_mut() {
                    tokio::select! {
                        biased;
                        _ = sig.recv() => Wake::Done,
                        _ = tokio::signal::ctrl_c() => Wake::Done,
                        key = rx.recv() => match key {
                            Some(k) => Wake::Key(k),
                            None => Wake::Done,
                        },
                        _ = tokio::time::sleep(std::time::Duration::from_secs(1)) => Wake::Tick,
                    }
                } else {
                    tokio::select! {
                        biased;
                        _ = tokio::signal::ctrl_c() => Wake::Done,
                        key = rx.recv() => match key {
                            Some(k) => Wake::Key(k),
                            None => Wake::Done,
                        },
                        _ = tokio::time::sleep(std::time::Duration::from_secs(1)) => Wake::Tick,
                    }
                }
            }
        };
        match wake.await {
            Wake::Done => break,
            Wake::Tick => {}
            Wake::Key(key) => {
                let (_, order) = build_list_lines(&entries, now, width, state.groups_expanded);
                if state.apply_key(key, order.len()) == Action::Quit {
                    break;
                }
            }
        }
    }
    Ok(())
}

pub async fn run_monitor(once: bool) -> Result<()> {
    let is_tty = std::io::stdout().is_terminal();
    let stdin_tty = std::io::stdin().is_terminal();
    if !once && is_tty && stdin_tty {
        // Interactive mini-TUI; any terminal setup failure falls back to the
        // plain rendering below, so a broken `$TERM` never breaks the monitor.
        if run_interactive().await.is_ok() {
            return Ok(());
        }
    }
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
                pid: 1234,
                task: self.task,
                model: "ninja".into(),
                status: self.status,
                step: self.step,
                max_turns: self.max_turns,
                last_command: self.command,
                started_at: 1000,
                updated_at: self.updated_at,
                group: self.group.map(str::to_string),
                repo_path: self.repo.map(str::to_string),
                owner: None,
                metrics: self.metrics,
                ..WorkerRegistryEntry::test_row(self.id, "")
            }
        }
    }

    fn line_entry(id: &str, status: RegistryStatus, updated_at: u64) -> WorkerRegistryEntry {
        WorkerRegistryEntry {
            pid: 1234,
            model: "ninja".into(),
            status,
            step: 1,
            max_turns: 100,
            started_at: 1000,
            updated_at,
            owner: None,
            ..WorkerRegistryEntry::test_row(id, "")
        }
    }

    #[test]
    fn test_status_line_omits_zero_parts_and_empty_state() {
        let now = 10_000;
        assert_eq!(format_status_line(&[], now), "");
        let entries = vec![
            line_entry("a", RegistryStatus::Running, now),
            line_entry("b", RegistryStatus::Reviewing, now),
            line_entry("c", RegistryStatus::Running, now),
            line_entry("d", RegistryStatus::Paused, now),
            line_entry("e", RegistryStatus::Completed, now),
            line_entry("f", RegistryStatus::Completed, now),
        ];
        assert_eq!(
            format_status_line(&entries, now),
            "⚙ 3 running · 1 needs input · 2 done"
        );
    }

    #[test]
    fn test_status_line_covers_failed_and_expires_terminal_rows() {
        use crate::pool::DEFAULT_TERMINAL_TTL_SECS;
        let now = 10_000;
        let fresh_failed = line_entry("a", RegistryStatus::Failed, now - 10);
        let stale_failed = line_entry(
            "b",
            RegistryStatus::Failed,
            now - DEFAULT_TERMINAL_TTL_SECS - 1,
        );
        let live = line_entry("c", RegistryStatus::Running, now);
        assert_eq!(
            format_status_line(&[fresh_failed, stale_failed.clone(), live], now),
            "⚙ 1 running · 1 failed"
        );
        // Terminal rows age out exactly at the shared TTL boundary.
        assert_eq!(
            format_status_line(&[stale_failed], now),
            "",
            "old terminal rows must not wake a statusLine"
        );
        let entries = vec![
            line_entry("d", RegistryStatus::Paused, now),
            line_entry("e", RegistryStatus::Completed, now),
            line_entry("f", RegistryStatus::Stopped, now),
        ];
        assert_eq!(
            format_status_line(&entries, now),
            "⚙ 1 needs input · 1 done · 1 stopped"
        );
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
        assert!(
            proj_a.contains("a1b2c3d\u{2026}"),
            "missing worker a:\n{proj_a}"
        );
        assert!(
            proj_a.contains("c3d4e5f\u{2026}"),
            "missing worker c:\n{proj_a}"
        );
        assert!(
            !proj_a.contains("b2c3d4e"),
            "proj-b worker leaked into proj-a"
        );

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
        let entries = vec![
            Row::new("done01")
                .status(RegistryStatus::Completed)
                .task("Finished task")
                .command("completed")
                .repo("local")
                .updated_at(1065) // Ran 65s (started_at = 1000)
                .build(),
        ];

        // Rendered long after completion (now = 5000) the uptime must stay 65s.
        let text = render_dashboard(&entries, 5000, false);
        assert!(
            text.contains("01m 05s"),
            "expected frozen duration 01m 05s, got:\n{text}"
        );
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
        assert!(
            plain.contains("REVIEWING"),
            "missing review badge:\n{plain}"
        );
        assert!(plain.contains("[REVIEW]"), "missing review tag:\n{plain}");
        assert!(
            plain.contains("Reviewing: 1"),
            "reviewing not counted:\n{plain}"
        );
        // The review marker must not be mistaken for a running row.
        assert!(plain.contains("\u{25c6} REVIEWING"));
        assert!(
            !plain.contains("[audits]"),
            "review tag replaces the group tag"
        );

        let colored = render_dashboard(&entries, 1100, true);
        assert!(
            colored.contains("\u{25c6}"),
            "missing review marker:\n{colored}"
        );
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
        assert!(
            cell.contains("5/10") && !cell.contains('#'),
            "bad cell: {cell:?}"
        );
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
        let entries = vec![
            Row::new("long001")
                .task("A task with a very long description that should also be capped")
                .command(&long_command)
                .turns(1, 10)
                .repo("local")
                .build(),
        ];

        let narrow = render_dashboard_with_width(&entries, 1060, false, 84);
        let narrow_row = narrow.lines().find(|l| l.starts_with("long001")).unwrap();
        assert!(
            narrow_row.contains('\u{2026}'),
            "narrow render must ellipsize the long command:\n{narrow_row}"
        );
        assert!(
            narrow_row.chars().count() <= 84,
            "row overflows 84 cols:\n{narrow_row}"
        );

        let wide = render_dashboard_with_width(&entries, 1060, false, 240);
        let wide_row = wide.lines().find(|l| l.starts_with("long001")).unwrap();
        assert!(
            wide_row.contains('\u{2026}'),
            "even wide renders stay capped:\n{wide_row}"
        );
        assert!(
            wide_row.chars().count() <= 240,
            "row overflows 240 cols:\n{wide_row}"
        );
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
        assert!(
            !colored.contains("\x1b[0m"),
            "trailing reset must be dropped"
        );
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
            (
                "w1",
                RegistryStatus::Running,
                "/home/dev/very-long-project-name-here",
                "[audits]",
            ),
            (
                "w2",
                RegistryStatus::Reviewing,
                "/home/dev/very-long-project-name-here",
                "[audits]",
            ),
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
        let entries = vec![
            Row::new("health01")
                .task("Refactor the authentication middleware into smaller pieces")
                .command("cargo test --all")
                .metrics(3, 1)
                .repo("local")
                .build(),
        ];

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
        let entries = vec![
            Row::new("wrap001")
                .task("Refactor the authentication middleware into smaller cohesive pieces")
                .command("cargo test --all")
                .repo("local")
                .build(),
        ];

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
        assert!(
            text.contains("Refactor"),
            "task text must survive stacking:\n{text}"
        );
        assert!(
            text.contains("cargo test --all"),
            "command must survive:\n{text}"
        );
    }

    /// The list/detail/back/quit/scroll/follow state machine answers every key
    /// with the action its view owns and ignores the keys of the other view.
    #[test]
    fn test_key_state_machine_covers_list_detail_back_quit_scroll_follow() {
        let mut state = UiState::default();
        assert_eq!(state.view, View::List);

        // List: move within bounds, clamp at both ends.
        assert_eq!(state.apply_key(Key::Down, 3), Action::MoveDown);
        assert_eq!(state.selection, 1);
        assert_eq!(state.apply_key(Key::Up, 3), Action::MoveUp);
        assert_eq!(state.selection, 0);
        assert_eq!(state.apply_key(Key::Up, 3), Action::None);
        state.selection = 2;
        assert_eq!(state.apply_key(Key::Down, 3), Action::None);

        // List: groups toggle, follow/scroll keys are ignored.
        assert_eq!(state.apply_key(Key::Char('g'), 3), Action::ToggleGroups);
        assert!(!state.groups_expanded);
        assert_eq!(state.apply_key(Key::Char('g'), 3), Action::ToggleGroups);
        assert!(state.groups_expanded);
        assert_eq!(state.apply_key(Key::Char('f'), 3), Action::None);
        assert_eq!(state.apply_key(Key::PageDown, 3), Action::None);

        // List: Enter opens the detail, Esc/q quit; Enter with no workers does
        // nothing.
        assert_eq!(state.apply_key(Key::Enter, 0), Action::None);
        assert_eq!(state.apply_key(Key::Enter, 3), Action::OpenDetail);
        assert_eq!(state.view, View::Detail);
        assert!(state.follow);
        assert_eq!(state.scroll, 0);

        // Detail: scroll pages, follow toggles, movement keys are ignored.
        assert_eq!(state.apply_key(Key::PageDown, 3), Action::ScrollDown);
        assert_eq!(state.scroll, PAGE_TURNS);
        assert_eq!(state.apply_key(Key::PageDown, 3), Action::ScrollDown);
        assert_eq!(state.scroll, 2 * PAGE_TURNS);
        assert_eq!(state.apply_key(Key::PageUp, 3), Action::ScrollUp);
        assert_eq!(state.scroll, PAGE_TURNS);
        // Scrolling past the newest turn clamps instead of wrapping.
        assert_eq!(state.apply_key(Key::PageUp, 3), Action::ScrollUp);
        assert_eq!(state.apply_key(Key::PageUp, 3), Action::ScrollUp);
        assert_eq!(state.scroll, 0);
        assert_eq!(state.apply_key(Key::Char('f'), 3), Action::ToggleFollow);
        assert!(!state.follow);
        assert_eq!(state.apply_key(Key::Up, 3), Action::None);
        assert_eq!(state.apply_key(Key::Char('g'), 3), Action::None);

        // Detail: Esc/q go back; list: Esc/q quit.
        assert_eq!(state.apply_key(Key::Esc, 3), Action::Back);
        assert_eq!(state.view, View::List);
        assert_eq!(state.apply_key(Key::Char('q'), 3), Action::Quit);
        let mut detail = UiState {
            view: View::Detail,
            ..UiState::default()
        };
        assert_eq!(detail.apply_key(Key::Char('q'), 3), Action::Back);
    }

    /// Raw terminal bytes map to keys: arrows, PgUp/PgDn, Enter, Esc, j/k as
    /// Up/Down aliases, and control bytes map to nothing.
    #[test]
    fn test_parse_key_maps_sequences_and_aliases() {
        assert_eq!(parse_key(&[0x1b, b'[', b'A']), Some(Key::Up));
        assert_eq!(parse_key(&[0x1b, b'[', b'B']), Some(Key::Down));
        assert_eq!(parse_key(&[0x1b, b'[', b'5', b'~']), Some(Key::PageUp));
        assert_eq!(parse_key(&[0x1b, b'[', b'6', b'~']), Some(Key::PageDown));
        assert_eq!(parse_key(b"\r"), Some(Key::Enter));
        assert_eq!(parse_key(&[0x1b]), Some(Key::Esc));
        assert_eq!(parse_key(b"j"), Some(Key::Down));
        assert_eq!(parse_key(b"k"), Some(Key::Up));
        assert_eq!(parse_key(b"q"), Some(Key::Char('q')));
        assert_eq!(parse_key(&[0x03]), None);
        assert_eq!(parse_key(&[0x1b, b'[', b'Z']), None);
    }

    /// History lines fold into turns: an assistant tool call opens a turn and
    /// the tool result attaches its exit code plus the output tail.
    #[test]
    fn test_history_reader_parses_turns_incrementally() {
        let dir = std::env::temp_dir().join(format!(
            "monitor-test-{}-{}",
            std::process::id(),
            unix_timestamp()
        ));
        std::fs::create_dir_all(&dir).expect("scratch dir");
        let path = dir.join("swe-wt-w1.history.jsonl");

        let meta = serde_json::json!({"task": "t", "model": "m", "repo_path": "r",
            "base_commit": "c", "branch": "b", "network_offline": false,
            "max_turns": 10, "revision": 0, "messages": []});
        let assistant = |command: &str| {
            serde_json::json!({"role": "assistant", "content": "working",
                "tool_calls": [{"id": "call_1", "type": "function",
                    "function": {"name": "bash", "arguments":
                        serde_json::to_string(&serde_json::json!({"command": command}))
                            .expect("args")}}]})
            .to_string()
        };
        let tool = |code: i32, output: &str| {
            serde_json::json!({"role": "tool", "tool_call_id": "call_1",
                "content": format!("COMMAND OUTPUT (exit code: {code})\n{output}")})
            .to_string()
        };
        std::fs::write(
            &path,
            format!(
                "{}\n{}\n{}\n",
                meta,
                assistant("cargo test"),
                tool(0, "ok\nline2\nline3\nline4\nline5\nline6\nline7")
            ),
        )
        .expect("write history");

        let mut reader = HistoryReader::default();
        reader.read_incremental(&path).expect("read");
        assert_eq!(reader.turns.len(), 1);
        assert_eq!(reader.turns[0].step, 1);
        assert_eq!(reader.turns[0].command, "cargo test");
        assert_eq!(reader.turns[0].exit_code, Some(0));
        // Only the last few output lines survive per turn.
        assert_eq!(reader.turns[0].output_lines.len(), 5);
        assert_eq!(reader.turns[0].output_lines.last().unwrap(), "line7");
        let offset = reader.offset;
        assert!(offset > 0);

        // A second read with nothing appended parses nothing new.
        reader.read_incremental(&path).expect("re-read");
        assert_eq!(reader.turns.len(), 1);
        assert_eq!(reader.offset, offset);

        // Appended lines parse from the remembered offset; a torn final line
        // is skipped without failing the read.
        {
            use std::io::Write;
            let mut file = std::fs::OpenOptions::new()
                .append(true)
                .open(&path)
                .expect("append");
            writeln!(file, "{}", assistant("cargo build")).expect("turn");
            writeln!(file, "{}", tool(1, "boom")).expect("result");
            writeln!(file, "{{\"role\": \"assistant\", \"broken\"").expect("torn");
        }
        reader.read_incremental(&path).expect("incremental");
        assert_eq!(reader.turns.len(), 2);
        assert_eq!(reader.turns[1].step, 2);
        assert_eq!(reader.turns[1].command, "cargo build");
        assert_eq!(reader.turns[1].exit_code, Some(1));
        assert_eq!(reader.turns[1].output_lines, vec!["boom".to_string()]);
        assert!(reader.offset > offset);

        std::fs::remove_dir_all(&dir).ok();
    }

    /// The reader caps its memory: far more turns than fit stay on disk.
    #[test]
    fn test_history_reader_keeps_at_most_the_last_500_turns() {
        let mut reader = HistoryReader::default();
        let mut buf = String::new();
        for i in 0..(MAX_TURNS_IN_MEMORY + 50) {
            buf.push_str(
                &serde_json::json!({"role": "assistant", "content": format!("turn {i}")})
                    .to_string(),
            );
            buf.push('\n');
        }
        reader.consume_lines(&buf);
        assert_eq!(reader.turns.len(), MAX_TURNS_IN_MEMORY);
        assert_eq!(
            reader.turns.last().unwrap().command,
            format!("turn {}", MAX_TURNS_IN_MEMORY + 49)
        );
        assert_eq!(reader.turns.first().unwrap().step, 51);
    }

    /// One worker fits one line at any width: the op column absorbs the slack
    /// and the line never overflows, dimmed for terminal workers.
    #[test]
    fn test_compact_row_fits_any_width() {
        let live = Row::new("a1b2c3d4")
            .task("Refactor the authentication middleware into smaller pieces")
            .command("cargo test --all --verbose -- --nocapture")
            .turns(37, 250)
            .group("audits")
            .build();
        let done = Row::new("d4e5f6a7b8")
            .status(RegistryStatus::Completed)
            .task("Finished work")
            .command("completed")
            .turns(9, 100)
            .build();

        for width in [40usize, 60, 80, 120, 200] {
            let row = fit_compact_row(&live, 1060, width);
            assert!(
                visible_width(&row) <= width,
                "live row overflows {width}: {row:?}"
            );
            assert!(row.contains("a1b2c3d4"), "id lost:\n{row}");
            assert!(row.contains("37/250"), "step/max lost:\n{row}");
            if width >= 80 {
                assert!(row.contains("[audits]"), "group lost:\n{row}");
            }

            let finished = fit_compact_row(&done, 1060, width);
            assert!(
                visible_width(&finished) <= width,
                "done row overflows {width}: {finished:?}"
            );
            assert!(
                finished.starts_with("\x1b[2m"),
                "completed workers must dim:\n{finished:?}"
            );
        }
        // The live row carries no dimming, but a live worker that stopped
        // reporting (retired-soon) dims like a terminal one.
        assert!(!fit_compact_row(&live, 1060, 120).starts_with("\x1b[2m"));
        let stale = Row::new("stale001")
            .task("Gone quiet")
            .command("cargo test")
            .turns(3, 100)
            .updated_at(100)
            .build();
        assert!(
            fit_compact_row(&stale, 10_000, 120).starts_with("\x1b[2m"),
            "a worker quiet past the TTL must dim"
        );
    }
}
