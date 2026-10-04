//! Live terminal supervisor and dashboard monitor for `mini-swe-mcp`.
//!
//! Interactive, in-place overwriting TUI that monitors running agent swarms
//! and isolated worker pods without spamming log lines.
//!
//! # Layout contract
//!
//! The dashboard is a bordered box: every visible line -- borders included --
//! is at most the terminal width, so a narrow terminal wraps nothing and a wide
//! one is not filled with dead space. The width comes from [`terminal_width`]
//! (TIOCGWINSZ, else `COLUMNS`, else [`DEFAULT_TERMINAL_WIDTH`]); the renderer
//! clips every line as a final safety net, whatever the content.
//!
//! All measurement runs on the visible (ANSI-stripped) text, never on escape
//! sequences, so colour never shifts a column. Colour is applied only when the
//! output is a TTY and `NO_COLOR` is unset; the plain non-TTY output uses the
//! same layout without ANSI codes.
//!
//! # Row contract
//!
//! Each worker is one compact line: `glyph id model step/max elapsed op`. The
//! columns drop, in order, as the terminal narrows: elapsed first, then the
//! model alias, then `step/max`; the glyph, the id and the op always stay. A
//! small progress bar precedes `step/max` only at 100+ columns. The op (or the
//! task headline when idle) takes whatever width the fixed columns leave and is
//! truncated with `…`.
//!
//! # Grouping contract
//!
//! Rows are grouped by round/group (`WorkerRegistryEntry::group`), each group
//! headed by one line with its per-status counts. There is no PID column here
//! (the PID lives in the detail view), and a repository line is shown only when
//! more than one repository is present.

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

/// Width assumed when the real terminal size cannot be determined (non-tty
/// output, `MONITOR_WIDTH` override, or a TUI that has not reported its size).
pub const DEFAULT_TERMINAL_WIDTH: usize = 80;

/// Grouping key used for workers whose `repo_path` was never recorded.
pub const DEFAULT_REPO_KEY: &str = "local";

/// Visible width of the ID column (a full UUID is never shown).
const ID_WIDTH: usize = 8;
/// Visible width of the model-alias column.
const MODEL_WIDTH: usize = 7;
/// Visible width of the `step/max` column.
const TURNS_WIDTH: usize = 6;
/// Filled/empty cells inside the progress bar, brackets excluded.
const PROGRESS_CELLS: usize = 8;
/// Width of the progress bar with its brackets.
const BAR_WIDTH: usize = PROGRESS_CELLS + 2;
/// Terminals at or above this width get a progress bar before `step/max`.
const PROGRESS_THRESHOLD: usize = 100;
/// Narrowest op column that still shows a command prefix plus its ellipsis.
const MIN_OP_WIDTH: usize = 12;

// ----------
// Text measurement helpers (width-aware, never byte-based)
// ----------

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

// ----------
// Row cells
// ----------

/// Bold, used for worker ids.
const C_BOLD: &str = "\x1b[1m";
/// Green: running.
const C_GREEN: &str = "\x1b[32m";
/// Cyan: reviewing.
const C_CYAN: &str = "\x1b[36m";
/// Yellow: paused or asking.
const C_YELLOW: &str = "\x1b[33m";
/// Red: failed or exhausted.
const C_RED: &str = "\x1b[31m";
/// Dim: done, retired states, model aliases.
const C_DIM: &str = "\x1b[2m";
/// SGR reset.
const C_RESET: &str = "\x1b[0m";

/// The one-character status glyph for a worker.
pub fn status_glyph(status: RegistryStatus) -> char {
    match status {
        RegistryStatus::Running => '\u{25cf}',
        RegistryStatus::Paused => '\u{23f8}',
        RegistryStatus::Reviewing => '\u{25c6}',
        RegistryStatus::Completed => '\u{2713}',
        RegistryStatus::Failed => '\u{2717}',
        RegistryStatus::Exhausted => '\u{2717}',
        RegistryStatus::Stopped => '\u{25a0}',
        RegistryStatus::Interrupted => '\u{25b2}',
    }
}

/// SGR colour of a status glyph: running green, reviewing cyan, paused yellow,
/// failed/exhausted red, everything terminal dim.
fn glyph_colour(status: RegistryStatus) -> &'static str {
    match status {
        RegistryStatus::Running => C_GREEN,
        RegistryStatus::Reviewing => C_CYAN,
        RegistryStatus::Paused => C_YELLOW,
        RegistryStatus::Failed | RegistryStatus::Exhausted => C_RED,
        RegistryStatus::Completed | RegistryStatus::Stopped | RegistryStatus::Interrupted => C_DIM,
    }
}

/// A coloured `●2`-style counter, or the plain form when colour is off.
fn glyph_count(status: RegistryStatus, count: usize, use_color: bool) -> String {
    let glyph = status_glyph(status);
    if use_color {
        format!("{}{glyph}{C_RESET}{count}", glyph_colour(status))
    } else {
        format!("{glyph}{count}")
    }
}

/// Model alias with any provider prefix stripped (`combo:nerd` renders `nerd`).
fn model_alias(model: &str) -> &str {
    match model.split_once(':') {
        Some((_, rest)) if !rest.is_empty() => rest,
        _ => model,
    }
}

/// Progress indicator for the turn budget: a filled/empty bar.
fn progress_bar(step: usize, max_turns: usize) -> String {
    // `checked_div` keeps a zero budget (never written by the pool, but a
    // hand-edited registry file can hold it) from panicking the renderer.
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

/// Human-readable elapsed time: `16m`, `05s`, `01h 05m`.
fn format_elapsed(secs: u64) -> String {
    if secs < 60 {
        format!("{:02}s", secs)
    } else if secs < 3600 {
        format!("{:02}m", secs / 60)
    } else {
        format!("{:02}h {:02}m", secs / 3600, (secs % 3600) / 60)
    }
}

/// The op cell: the question when the worker asks, else the last command, else
/// the task headline. The group tag is demoted to a `[group]` prefix here.
fn op_text(w: &WorkerRegistryEntry) -> String {
    let first_line = w.task.lines().next().unwrap_or("").trim();
    let detail = if let Some(ref q) = w.question {
        format!("ASK: {q}")
    } else if !w.last_command.is_empty()
        && w.last_command != "completed"
        && w.last_command != "initializing"
    {
        w.last_command.clone()
    } else {
        first_line.to_string()
    };
    match w.group.as_deref() {
        Some(g) if !g.trim().is_empty() => format!("[{}] {detail}", g.trim()),
        _ => detail,
    }
}

/// Fit one worker to a single compact line of at most `inner` visible columns.
///
/// Layout: `glyph id model step/max elapsed op`. Columns drop, in order, as
/// the terminal narrows: elapsed first, then the model alias, then `step/max`;
/// the glyph, the id and the op always stay. A progress bar precedes `step/max`
/// only at [`PROGRESS_THRESHOLD`]+ columns. The op takes whatever the fixed
/// columns leave and is truncated with `…`, so the row never overflows `inner`.
fn compact_row(w: &WorkerRegistryEntry, now: u64, inner: usize, use_color: bool) -> String {
    let glyph = status_glyph(w.status);
    let id = truncate_visible(&w.id, ID_WIDTH);
    let model = model_alias(&w.model);
    let turns = format!("{}/{}", w.step, w.max_turns);
    let duration_secs = if w.status.is_terminal() {
        w.updated_at.saturating_sub(w.started_at)
    } else {
        now.saturating_sub(w.started_at)
    };
    let elapsed = format_elapsed(duration_secs);
    let detail = op_text(w);

    let show_bar = inner + 4 >= PROGRESS_THRESHOLD;
    let base = 1 + 1 + ID_WIDTH + 1;
    let turns_w = visible_width(&turns).max(TURNS_WIDTH);
    let turns_cost = turns_w + 1 + if show_bar { BAR_WIDTH + 1 } else { 0 };
    let model_cost = MODEL_WIDTH + 1;
    let elapsed_cost = visible_width(&elapsed) + 1;

    // Drop order is elapsed, then model, then step/max: each column is shown
    // only when every column it outranks still fits, and the op keeps a
    // readable minimum width.
    let show_turns = base + turns_cost + MIN_OP_WIDTH <= inner;
    let show_model = show_turns && base + turns_cost + model_cost + MIN_OP_WIDTH <= inner;
    let show_elapsed =
        show_model && base + turns_cost + model_cost + elapsed_cost + MIN_OP_WIDTH <= inner;

    let mut prefix = String::new();
    if use_color {
        prefix.push_str(glyph_colour(w.status));
    }
    prefix.push(glyph);
    if use_color {
        prefix.push_str(C_RESET);
    }
    prefix.push(' ');
    if use_color {
        prefix.push_str(C_BOLD);
    }
    prefix.push_str(&pad_visible(&id, ID_WIDTH));
    if use_color {
        prefix.push_str(C_RESET);
    }
    prefix.push(' ');
    if show_model {
        if use_color {
            prefix.push_str(C_DIM);
        }
        prefix.push_str(&pad_visible(
            &truncate_visible(model, MODEL_WIDTH),
            MODEL_WIDTH,
        ));
        if use_color {
            prefix.push_str(C_RESET);
        }
        prefix.push(' ');
    }
    if show_turns {
        if show_bar {
            prefix.push_str(&progress_bar(w.step, w.max_turns));
            prefix.push(' ');
        }
        prefix.push_str(&pad_visible(&turns, turns_w));
        prefix.push(' ');
    }
    if show_elapsed {
        prefix.push_str(&elapsed);
        prefix.push(' ');
    }
    let op_width = inner.saturating_sub(visible_width(&prefix));
    prefix.push_str(&truncate_visible(&detail, op_width));
    truncate_visible(&prefix, inner)
}

// ----------
// Bordered box
// ----------

/// Clock for the header: `HH:MM` in the local day.
fn header_clock(now: u64) -> String {
    format!("{:02}:{:02}", (now / 3600) % 24, (now / 60) % 60)
}

/// Wrap `content` in side borders, padded (and clipped) to exactly `width`.
///
/// The final clip guarantees the rule: every visible line, borders included,
/// is at most `width`, whatever the content.
fn box_line(content: &str, width: usize) -> String {
    let inner = width.saturating_sub(4).max(1);
    let body = pad_visible(&truncate_visible(content, inner), inner);
    truncate_visible(&format!("│ {body} │"), width.max(1))
}

/// Top border: `╭─ mini-swe ─ N workers ─ counts ─…─ HH:MM ─╮`, filled to
/// exactly `width`. The header carries the title, the total, the per-status
/// counts and the clock on one bordered line.
fn box_top(
    title: &str,
    total: &str,
    counts: &str,
    clock: &str,
    width: usize,
    use_color: bool,
) -> String {
    let mut prefix = format!("╭─ {title} ─ {total}");
    if !counts.is_empty() {
        prefix.push_str(&format!(" ─ {counts}"));
    }
    let tail = format!(" {clock} ─╮");
    let dashes = width
        .saturating_sub(visible_width(&prefix) + visible_width(&tail) + 1)
        .max(1);
    let mut line = format!("{prefix} {}{tail}", "─".repeat(dashes));
    if use_color {
        line = format!("{C_BOLD}{line}{C_RESET}");
    }
    truncate_visible(&line, width.max(1))
}

/// Bottom border: `╰─ hint ─…─╯`, filled to exactly `width`.
fn box_bottom(hint: &str, width: usize, use_color: bool) -> String {
    let prefix = format!("╰─ {hint} ─");
    let dashes = width.saturating_sub(visible_width(&prefix) + 1).max(1);
    let mut line = format!("{prefix}{}╯", "─".repeat(dashes));
    if use_color {
        line = format!("{C_DIM}{line}{C_RESET}");
    }
    truncate_visible(&line, width.max(1))
}

// ----------
// Grouping and dashboard rendering
// ----------

/// Grouping key for a worker: its round/group, else its repository, else the
/// default key. The header line carries this key plus the group's counts.
fn group_key(entry: &WorkerRegistryEntry) -> String {
    if let Some(g) = entry.group.as_deref()
        && !g.trim().is_empty()
    {
        return g.trim().to_string();
    }
    if let Some(r) = entry.repo_path.as_deref()
        && !r.trim().is_empty()
    {
        return r.trim().to_string();
    }
    DEFAULT_REPO_KEY.to_string()
}

/// Per-status counts for the header: running, reviewing, paused and done when
/// non-zero, failed always (so `✗0` reads as the all-clear).
fn header_counts(entries: &[WorkerRegistryEntry], use_color: bool) -> String {
    let mut running = 0;
    let mut reviewing = 0;
    let mut paused = 0;
    let mut done = 0;
    let mut failed = 0;
    for e in entries {
        match e.status {
            RegistryStatus::Running => running += 1,
            RegistryStatus::Reviewing => reviewing += 1,
            RegistryStatus::Paused => paused += 1,
            RegistryStatus::Completed | RegistryStatus::Stopped | RegistryStatus::Interrupted => {
                done += 1
            }
            RegistryStatus::Failed | RegistryStatus::Exhausted => failed += 1,
        }
    }
    let mut parts = Vec::new();
    if running > 0 {
        parts.push(glyph_count(RegistryStatus::Running, running, use_color));
    }
    if reviewing > 0 {
        parts.push(glyph_count(RegistryStatus::Reviewing, reviewing, use_color));
    }
    if paused > 0 {
        parts.push(glyph_count(RegistryStatus::Paused, paused, use_color));
    }
    if done > 0 {
        parts.push(glyph_count(RegistryStatus::Completed, done, use_color));
    }
    parts.push(glyph_count(RegistryStatus::Failed, failed, use_color));
    parts.join(" ")
}

/// Per-status counts for one group header: every non-zero state, glyph form.
fn group_counts(workers: &[&WorkerRegistryEntry], use_color: bool) -> String {
    let mut running = 0;
    let mut reviewing = 0;
    let mut paused = 0;
    let mut done = 0;
    let mut failed = 0;
    for w in workers {
        match w.status {
            RegistryStatus::Running => running += 1,
            RegistryStatus::Reviewing => reviewing += 1,
            RegistryStatus::Paused => paused += 1,
            RegistryStatus::Completed | RegistryStatus::Stopped | RegistryStatus::Interrupted => {
                done += 1
            }
            RegistryStatus::Failed | RegistryStatus::Exhausted => failed += 1,
        }
    }
    let mut parts = Vec::new();
    if running > 0 {
        parts.push(glyph_count(RegistryStatus::Running, running, use_color));
    }
    if reviewing > 0 {
        parts.push(glyph_count(RegistryStatus::Reviewing, reviewing, use_color));
    }
    if paused > 0 {
        parts.push(glyph_count(RegistryStatus::Paused, paused, use_color));
    }
    if done > 0 {
        parts.push(glyph_count(RegistryStatus::Completed, done, use_color));
    }
    if failed > 0 {
        parts.push(glyph_count(RegistryStatus::Failed, failed, use_color));
    }
    parts.join(" ")
}

/// The list content lines (group headers plus worker rows), each already at
/// most `width - 4` visible columns. Shared by the plain and the interactive
/// renderers so both views keep the same layout.
fn list_content_lines(
    entries: &[WorkerRegistryEntry],
    now: u64,
    width: usize,
    use_color: bool,
) -> Vec<String> {
    let inner = width.saturating_sub(4).max(1);
    let mut groups: BTreeMap<String, Vec<&WorkerRegistryEntry>> = BTreeMap::new();
    for entry in entries {
        groups.entry(group_key(entry)).or_default().push(entry);
    }
    let repos: std::collections::BTreeSet<&str> = entries
        .iter()
        .map(|e| e.repo_path.as_deref().unwrap_or(DEFAULT_REPO_KEY))
        .collect();
    let show_repo = repos.len() > 1;
    let mut lines = Vec::new();
    for (name, workers) in &groups {
        if show_repo {
            let repo = workers
                .first()
                .and_then(|w| w.repo_path.as_deref())
                .unwrap_or(DEFAULT_REPO_KEY);
            lines.push(truncate_visible(&format!("repo: {repo}"), inner));
        }
        let counts = group_counts(workers, use_color);
        let header = if counts.is_empty() {
            name.clone()
        } else {
            format!("{name}  {counts}")
        };
        lines.push(truncate_visible(&header, inner));
        for w in workers {
            lines.push(compact_row(w, now, inner, use_color));
        }
    }
    lines
}

/// Render the supervisor dashboard at the default width.
pub fn render_dashboard(entries: &[WorkerRegistryEntry], now: u64, use_color: bool) -> String {
    render_dashboard_with_width(entries, now, use_color, DEFAULT_TERMINAL_WIDTH)
}

/// Render the supervisor dashboard for a specific terminal width.
///
/// Pure: identical inputs always produce identical output, which is what makes
/// the width-dependent behaviour unit-testable. The plain non-TTY output uses
/// this same bordered layout, without ANSI codes unless `use_color` is set.
pub fn render_dashboard_with_width(
    entries: &[WorkerRegistryEntry],
    now: u64,
    use_color: bool,
    term_width: usize,
) -> String {
    let width = term_width.max(20);
    let mut out = String::new();
    let total = format!("{} workers", entries.len());
    let counts = header_counts(entries, use_color);
    out.push_str(&box_top(
        "mini-swe",
        &total,
        &counts,
        &header_clock(now),
        width,
        use_color,
    ));
    out.push('\n');
    if entries.is_empty() {
        out.push_str(&box_line("No active or recent workers.", width));
        out.push('\n');
        return out;
    }
    for line in list_content_lines(entries, now, width, use_color) {
        out.push_str(&box_line(&line, width));
        out.push('\n');
    }
    out
}

/// Colours are on only when the output is a TTY and `NO_COLOR` is unset.
pub fn use_color_for_tty(is_tty: bool) -> bool {
    is_tty && std::env::var_os("NO_COLOR").is_none()
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
        file.read_to_string(&mut buf)?;
        self.offset = len;
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
/// Layout: `glyph id model step/max elapsed op`, columns dropping in the
/// stated order as the terminal narrows. Delegates to [`compact_row`] so the
/// interactive list and the plain dashboard share one layout; plain (no ANSI
/// codes) like the non-TTY output.
pub fn fit_compact_row(w: &WorkerRegistryEntry, now: u64, width: usize) -> String {
    compact_row(w, now, width.saturating_sub(4).max(1), false)
}

/// One line of the key hint shown at the bottom of the interactive views.
pub fn key_hint(view: View) -> &'static str {
    match view {
        View::List => "\u{2191}\u{2193} select  \u{23ce} turns  g groups  q quit",
        View::Detail => "PgUp/PgDn scroll  f follow  Esc/q back",
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

/// Build the compact list lines for `entries`, grouped by round/group.
///
/// Each group gets a one-line header with its per-status counts followed by
/// one compact row per worker. When `expanded` is false only the headers are
/// returned. Returns the lines plus the worker ids in display order, so the
/// selection index maps to a worker.
fn build_list_lines<'a>(
    entries: &'a [WorkerRegistryEntry],
    now: u64,
    width: usize,
    expanded: bool,
) -> (Vec<ListLine<'a>>, Vec<&'a WorkerRegistryEntry>) {
    let content = list_content_lines(entries, now, width, false);
    let mut lines = Vec::new();
    let mut order = Vec::new();
    let mut group_names: Vec<String> = Vec::new();
    let mut groups: BTreeMap<String, Vec<&'a WorkerRegistryEntry>> = BTreeMap::new();
    for entry in entries {
        groups.entry(group_key(entry)).or_default().push(entry);
    }
    let repos: std::collections::BTreeSet<&str> = entries
        .iter()
        .map(|e| e.repo_path.as_deref().unwrap_or(DEFAULT_REPO_KEY))
        .collect();
    let show_repo = repos.len() > 1;
    let _ = content;
    for (name, workers) in &groups {
        group_names.push(name.clone());
        if show_repo {
            let repo = workers
                .first()
                .and_then(|w| w.repo_path.as_deref())
                .unwrap_or(DEFAULT_REPO_KEY);
            lines.push(ListLine::Header(format!("repo: {repo}")));
        }
        lines.push(ListLine::Header(name.clone()));
        if expanded {
            for w in workers {
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
/// stays visible. The whole view is a bordered box; the header and the key-hint
/// line are the top and bottom borders, and the body scrolls between them.
fn render_list(
    entries: &[WorkerRegistryEntry],
    state: &UiState,
    now: u64,
    width: usize,
    height: usize,
) -> String {
    let (lines, order) = build_list_lines(entries, now, width, state.groups_expanded);
    let selected_id = order.get(state.selection).map(|w| w.id.as_str());
    let use_color = use_color_for_tty(true);
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
    let total = format!("{} workers", entries.len());
    let counts = header_counts(entries, use_color);
    let mut out = String::new();
    out.push_str(&box_top(
        "mini-swe",
        &total,
        &counts,
        &header_clock(now),
        width,
        use_color,
    ));
    out.push('\n');
    for line in lines.iter().skip(start).take(body_height) {
        match line {
            ListLine::Header(h) => out.push_str(&box_line(h, width)),
            ListLine::Worker(w) => {
                let row = compact_row(w, now, width.saturating_sub(4).max(1), use_color);
                if Some(w.id.as_str()) == selected_id {
                    out.push_str(&box_line(&format!("\x1b[7m{row}\x1b[0m"), width));
                } else {
                    out.push_str(&box_line(&row, width));
                }
            }
        }
        out.push('\n');
    }
    out.push_str(&box_bottom(key_hint(View::List), width, use_color));
    out.push('\n');
    out
}

/// Render the detail view for one worker: its REPORT/question/status header on
/// top, then its turns newest-last. Follow mode sticks to the newest turn;
/// otherwise `state.scroll` turns are skipped from the end. The view is a
/// bordered box following the same width rules as the list.
fn render_detail(
    entry: &WorkerRegistryEntry,
    reader: &HistoryReader,
    state: &UiState,
    now: u64,
    width: usize,
    height: usize,
) -> String {
    let use_color = use_color_for_tty(true);
    let status_name = entry.status.display_name();
    let title = format!(
        "{} {} {}",
        status_glyph(entry.status),
        entry.id,
        status_name
    );
    let total = format!("{}/{}", entry.step, entry.max_turns);
    let mut out = String::new();
    out.push_str(&box_top(
        &title,
        &total,
        "",
        &header_clock(now),
        width,
        use_color,
    ));
    out.push('\n');
    let mut body: Vec<String> = Vec::new();
    body.push(format!("task: {}", entry.task.lines().next().unwrap_or("")));
    if let Some(ref q) = entry.question {
        body.push(format!("question: {q}"));
    }
    if let Some(ref report) = entry.report
        && !report.is_empty()
    {
        body.push(format!(
            "report: {} | files: {} | tests: {} | risks: {}",
            report.done, report.files, report.tests, report.risks
        ));
    }
    let body_height = height.saturating_sub(3).max(1);
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
        turn_lines.push(format!("#{} {}{}", turn.step, turn.command, code));
        for line in &turn.output_lines {
            turn_lines.push(format!("  {line}"));
        }
    }
    if turn_lines.is_empty() {
        turn_lines.push("(no turns recorded yet)".to_string());
    }
    let start = turn_lines.len().saturating_sub(body_height);
    for line in turn_lines.iter().skip(start) {
        body.push(line.clone());
    }
    for line in body.iter().take(body_height) {
        out.push_str(&box_line(line, width));
        out.push('\n');
    }
    let _ = now;
    out.push_str(&box_bottom(key_hint(View::Detail), width, use_color));
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
            use_color_for_tty(is_tty),
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
            use_color_for_tty(true),
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
    use serde_json::json;

    /// Builder for registry rows: the tests below touch every field, so a
    /// positional 8-argument helper would be unreadable.
    struct Row {
        id: String,
        status: RegistryStatus,
        step: usize,
        max_turns: usize,
        command: String,
        task: String,
        group: Option<String>,
        repo: Option<String>,
        metrics: WorkerMetrics,
        updated_at: u64,
    }

    impl Row {
        fn new(id: &str) -> Self {
            Self {
                id: id.to_string(),
                status: RegistryStatus::Running,
                step: 0,
                max_turns: 10,
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
            self.group = Some(group.to_string());
            self
        }

        fn repo(mut self, repo: &'static str) -> Self {
            self.repo = Some(repo.to_string());
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
                id: self.id,
                pid: 1234,
                task: self.task,
                model: "combo:ninja".into(),
                status: self.status,
                step: self.step,
                max_turns: self.max_turns,
                last_command: self.command,
                question: None,
                started_at: 1000,
                updated_at: self.updated_at,
                group: self.group,
                role: crate::pool::WorkerRole::Worker,
                repo_path: self.repo,
                owner: None,
                metrics: self.metrics,
                base_branch: None,
                base_commit: None,
                head_commit: None,
                revision: 0,
                auto_continues: 0,
                report: None,
                approved: None,
                verified: None,
                security_review: None,
                security_approved_commit: None,
                integrated: Vec::new(),
                absorbed: Vec::new(),
                verdicts: None,
                keep_branch: false,
            }
        }
    }

    /// A fixed set of rows in every status, used by the width-contract tests.
    fn sample_entries() -> Vec<WorkerRegistryEntry> {
        vec![
            Row::new("925633bb")
                .status(RegistryStatus::Running)
                .turns(15, 250)
                .command("CONSOLIDATE_WAIT ea0f")
                .task("Consolidate the round")
                .group("round52")
                .repo("/repo/x")
                .build(),
            Row::new("ea0ff2c4")
                .status(RegistryStatus::Reviewing)
                .turns(35, 120)
                .command("review: cat >> tests/")
                .task("Review the diff")
                .group("round52")
                .repo("/repo/x")
                .build(),
            Row::new("ccbc2be1")
                .status(RegistryStatus::Running)
                .turns(12, 60)
                .command("cargo fmt --check && cargo test")
                .task("Format and test")
                .group("round50")
                .repo("/repo/x")
                .build(),
            Row::new("29a760ed")
                .status(RegistryStatus::Completed)
                .turns(70, 60)
                .command("completed")
                .task("Keep the hub log small")
                .group("round50")
                .repo("/repo/x")
                .build(),
        ]
    }

    #[test]
    fn test_render_dashboard_empty() {
        let text = render_dashboard(&[], 1000, false);
        assert!(text.contains("0 workers"), "{text}");
        assert!(text.contains("No active or recent workers."), "{text}");
        assert!(text.contains("╭"), "missing top border:\n{text}");
    }

    /// Every visible line -- borders included -- is at most the terminal width
    /// at 40, 60, 80 and 120 columns, and the id and op stay present at 40.
    #[test]
    fn test_every_line_fits_every_width() {
        let entries = sample_entries();
        for width in [40usize, 60, 80, 120] {
            for use_color in [false, true] {
                let text = render_dashboard_with_width(&entries, 1060, use_color, width);
                for line in text.lines() {
                    assert!(
                        visible_width(line) <= width,
                        "line overflows {width} cols ({}): {line:?}",
                        visible_width(line)
                    );
                }
                if width == 40 {
                    // The id and the op always stay, even at the narrowest.
                    assert!(text.contains("925633bb"), "id lost at 40:\n{text}");
                    assert!(text.contains("CONSOLI"), "op lost at 40:\n{text}");
                }
            }
        }
    }

    /// Columns drop in the stated order as the terminal narrows: elapsed first,
    /// then the model alias, then `step/max`; glyph + id + op always stay.
    #[test]
    fn test_columns_drop_in_order() {
        let entries = sample_entries();
        let wide = render_dashboard_with_width(&entries, 1060, false, 120);
        assert!(wide.contains("15/250"), "step/max missing at 120:\n{wide}");
        assert!(
            wide.contains("ninja"),
            "model alias missing at 120:\n{wide}"
        );
        assert!(wide.contains("01m"), "elapsed missing at 120:\n{wide}");

        let mid = render_dashboard_with_width(&entries, 1060, false, 60);
        // At 60 every column still fits.
        assert!(mid.contains("15/250"), "step/max missing at 60:\n{mid}");
        assert!(mid.contains("ninja"), "model alias missing at 60:\n{mid}");
        assert!(mid.contains("01m"), "elapsed missing at 60:\n{mid}");

        let narrow = render_dashboard_with_width(&entries, 1060, false, 40);
        // Elapsed then model drop first; step/max still fits; id + op stay.
        assert!(narrow.contains("925633bb"), "id lost at 40:\n{narrow}");
        assert!(narrow.contains("CONSOLI"), "op lost at 40:\n{narrow}");
        assert!(
            narrow.contains("15/250"),
            "step/max should stay at 40:\n{narrow}"
        );
        assert!(
            !narrow.contains("ninja"),
            "model should be dropped at 40:\n{narrow}"
        );
        assert!(
            !narrow.contains("01m"),
            "elapsed should be dropped at 40:\n{narrow}"
        );

        let tiny = render_dashboard_with_width(&entries, 1060, false, 25);
        // Narrowest: step/max drops too; glyph + id + op always stay.
        assert!(tiny.contains("925633bb"), "id lost at 25:\n{tiny}");
        assert!(
            !tiny.contains("15/250"),
            "step/max should be dropped at 25:\n{tiny}"
        );
    }

    /// The model alias strips the provider prefix (`combo:ninja` renders `ninja`).
    #[test]
    fn test_model_alias_strips_provider_prefix() {
        assert_eq!(model_alias("combo:ninja"), "ninja");
        assert_eq!(model_alias("openai:gpt-4o"), "gpt-4o");
        assert_eq!(model_alias("plain"), "plain");
        assert_eq!(model_alias(":"), ":");
    }

    /// NO_COLOR disables every escape sequence in the coloured renderer.
    #[test]
    fn test_no_color_disables_escapes() {
        let entries = sample_entries();
        let colored = render_dashboard_with_width(&entries, 1060, true, 80);
        assert!(colored.contains("\x1b["), "expected colour:\n{colored}");
        let plain = render_dashboard_with_width(&entries, 1060, false, 80);
        assert!(
            !plain.contains("\x1b["),
            "NO_COLOR must disable escapes:\n{plain}"
        );
        // The plain layout is the same, just without ANSI codes.
        assert_eq!(visible_width(&colored), visible_width(&plain));
    }

    /// The header carries the title, the per-status counts and the clock on one
    /// bordered line; the counts use the status glyphs.
    #[test]
    fn test_header_counts_and_glyphs() {
        let entries = sample_entries();
        let text = render_dashboard(&entries, 1060, false);
        assert!(text.contains("mini-swe"), "missing title:\n{text}");
        assert!(text.contains("4 workers"), "missing total:\n{text}");
        // running 2, reviewing 1, done 1, failed 0 (always shown).
        assert!(text.contains("\u{25cf}2"), "running count:\n{text}");
        assert!(text.contains("\u{25c6}1"), "reviewing count:\n{text}");
        assert!(text.contains("\u{2713}1"), "done count:\n{text}");
        assert!(text.contains("\u{2717}0"), "failed count:\n{text}");
        // Group headers carry their own per-status counts.
        assert!(text.contains("round52"), "missing group header:\n{text}");
        assert!(text.contains("round50"), "missing group header:\n{text}");
    }

    /// The status glyphs match the task's legend.
    #[test]
    fn test_status_glyphs_match_legend() {
        assert_eq!(status_glyph(RegistryStatus::Running), '\u{25cf}');
        assert_eq!(status_glyph(RegistryStatus::Reviewing), '\u{25c6}');
        assert_eq!(status_glyph(RegistryStatus::Paused), '\u{23f8}');
        assert_eq!(status_glyph(RegistryStatus::Completed), '\u{2713}');
        assert_eq!(status_glyph(RegistryStatus::Failed), '\u{2717}');
        assert_eq!(status_glyph(RegistryStatus::Exhausted), '\u{2717}');
    }

    /// The progress bar precedes step/max only at 100+ columns.
    #[test]
    fn test_progress_bar_only_wide() {
        let entries = sample_entries();
        let wide = render_dashboard_with_width(&entries, 1060, false, 120);
        assert!(
            wide.contains('[') && wide.contains('#'),
            "bar missing at 120:\n{wide}"
        );
        let narrow = render_dashboard_with_width(&entries, 1060, false, 80);
        assert!(
            !narrow.contains('#'),
            "bar must not show below 100 cols:\n{narrow}"
        );
    }

    /// The repository line appears only when more than one repository is shown.
    #[test]
    fn test_repo_line_only_when_multiple_repos() {
        let single = vec![
            Row::new("a1").repo("/repo/x").group("r1").task("T").build(),
            Row::new("a2").repo("/repo/x").group("r1").task("T").build(),
        ];
        let one = render_dashboard(&single, 1060, false);
        assert!(
            !one.contains("repo:"),
            "no repo line for a single repo:\n{one}"
        );

        let multi = vec![
            Row::new("a1").repo("/repo/x").group("r1").task("T").build(),
            Row::new("a2").repo("/repo/y").group("r1").task("T").build(),
        ];
        let two = render_dashboard(&multi, 1060, false);
        assert!(
            two.contains("repo:"),
            "repo line missing for two repos:\n{two}"
        );
    }

    /// A live worker's elapsed keeps counting; a terminal one freezes.
    #[test]
    fn test_elapsed_frozen_for_terminal_rows() {
        let done = Row::new("done01")
            .status(RegistryStatus::Completed)
            .task("Finished")
            .command("completed")
            .repo("local")
            .updated_at(1065)
            .build();
        let text = render_dashboard(&[done], 5000, false);
        assert!(text.contains("01m"), "frozen elapsed expected:\n{text}");
    }

    /// Colour stays aligned: every coloured row fits and the visible widths of
    /// the status variants match.
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
                    .group("g")
                    .build()
            })
            .collect();
        let width = 120;
        let text = render_dashboard_with_width(&entries, 1100, true, width);
        for line in text.lines() {
            assert!(
                visible_width(line) <= width,
                "colored line overflows: {line:?} -> {}",
                visible_width(line)
            );
        }
    }

    /// One worker fits one line at any width: the op column absorbs the slack.
    #[test]
    fn test_compact_row_fits_any_width() {
        let live = Row::new("a1b2c3d4")
            .task("Refactor the authentication middleware into smaller pieces")
            .command("cargo test --all --verbose -- --nocapture")
            .turns(37, 250)
            .group("audits")
            .build();
        for width in [40usize, 60, 80, 120, 200] {
            let row = fit_compact_row(&live, 1060, width);
            assert!(
                visible_width(&row) <= width,
                "live row overflows {width}: {row:?}"
            );
            assert!(row.contains("a1b2c3d4"), "id lost:\n{row}");
            if width >= 80 {
                assert!(row.contains("37/250"), "step/max lost:\n{row}");
            }
        }
    }

    /// The key state machine covers list/detail/back/quit/scroll/follow.
    #[test]
    fn test_key_state_machine_covers_list_detail_back_quit_scroll_follow() {
        let mut state = UiState::default();
        assert_eq!(state.view, View::List);
        assert_eq!(state.apply_key(Key::Down, 3), Action::MoveDown);
        assert_eq!(state.selection, 1);
        assert_eq!(state.apply_key(Key::Up, 3), Action::MoveUp);
        assert_eq!(state.selection, 0);
        assert_eq!(state.apply_key(Key::Up, 3), Action::None);
        state.selection = 2;
        assert_eq!(state.apply_key(Key::Down, 3), Action::None);
        assert_eq!(state.apply_key(Key::Char('g'), 3), Action::ToggleGroups);
        assert!(!state.groups_expanded);
        assert_eq!(state.apply_key(Key::Char('g'), 3), Action::ToggleGroups);
        assert!(state.groups_expanded);
        assert_eq!(state.apply_key(Key::Enter, 0), Action::None);
        assert_eq!(state.apply_key(Key::Enter, 3), Action::OpenDetail);
        assert_eq!(state.view, View::Detail);
        assert!(state.follow);
        assert_eq!(state.apply_key(Key::PageDown, 3), Action::ScrollDown);
        assert_eq!(state.scroll, PAGE_TURNS);
        assert_eq!(state.apply_key(Key::Esc, 3), Action::Back);
        assert_eq!(state.view, View::List);
        assert_eq!(state.apply_key(Key::Char('q'), 3), Action::Quit);
    }

    /// Raw terminal bytes map to keys.
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
    }

    /// History lines fold into turns.
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

    /// The reader caps its memory.
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
}
