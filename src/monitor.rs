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
//! The plain non-TTY output has no window to query, so its width comes from
//! [`plain_width`]: `MONITOR_WIDTH`, else `COLUMNS` (which a shell exports for
//! a redirected command too), else [`DEFAULT_TERMINAL_WIDTH`].
//!
//! All measurement runs on the visible (ANSI-stripped) text, never on escape
//! sequences, so colour never shifts a column. Colour is applied only when the
//! output is a TTY and `NO_COLOR` is unset; the plain non-TTY output uses the
//! same layout without ANSI codes.
//!
//! # Redraw contract
//!
//! The interactive views render to a list of lines, and the private
//! `write_frame` is the single place that puts them on the terminal. Raw mode
//! clears OPOST, so a bare `\n` no longer returns the carriage and every line
//! would start where the previous one ended; `write_frame` therefore joins the
//! lines with `\r\n`. A frame is one `write_all` plus one flush, drawn without
//! clearing the screen first: the cursor moves home, every line erases its own
//! tail and one clear-to-end-of-screen removes the leftovers, so the redraw
//! does not flicker. An unchanged frame is not written at all.
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
//! The choice is made once per frame, not once per row: the columns shown and
//! their widths are decided from the terminal width and the widest values in
//! the whole frame, so every row puts its op at the same column and the table
//! never comes out ragged. A row whose own `step/max` or elapsed cell is
//! narrower is padded out to the frame's width, not allowed to shift the op.
//!
//! The op is the protected column: it keeps at least 24 visible characters,
//! and a column is dropped rather than let the op fall below that. What the
//! glyph already shows is never repeated in the op, so a narrow row spends its
//! columns on the work rather than on its label.
//!
//! # Grouping contract
//!
//! Rows are grouped by round/group (`WorkerRegistryEntry::group`), each group
//! headed by one line with its per-status counts. A row under its header
//! therefore carries no `[group]` tag of its own; only the flat view, which has
//! no header above it, does. There is no PID column here (the PID lives in the
//! detail view), and a repository line is shown only when more than one
//! repository is present.

use crate::config::env_parse_from;
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
/// Columns the op keeps before the dropping logic starts removing columns
/// around it: enough for a real command prefix at any usable width. A shorter
/// op simply renders whole, so this is a floor, never a truncation target.
const MIN_OP_WIDTH: usize = 24;

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
                    for c in chars.by_ref() {
                        if ('@'..='~').contains(&c) {
                            break;
                        }
                    }
                }
                // OSC: a string terminated by BEL or by ST (ESC \\).
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

/// The elapsed cell for one worker: still counting while the worker lives,
/// frozen at its last update once it is terminal.
fn elapsed_cell(w: &WorkerRegistryEntry, now: u64) -> String {
    let secs = if w.status.is_terminal() {
        w.updated_at.saturating_sub(w.started_at)
    } else {
        now.saturating_sub(w.started_at)
    };
    format_elapsed(secs)
}

/// The `step/max` cell for one worker.
fn turns_cell(w: &WorkerRegistryEntry) -> String {
    format!("{}/{}", w.step, w.max_turns)
}

/// Prefix the review engine stamps onto a reviewing worker's command.
const REVIEW_PREFIX: &str = "[review] ";

/// The op cell: the question when the worker asks, else the last command, else
/// the task headline.
///
/// The `[group]` tag is only added for a flat (ungrouped) row: a row printed
/// under its group's header line already shows the group in the line above, so
/// repeating it there is pure noise. The `[review] ` prefix the review engine
/// stamps onto a command is dropped for a reviewing worker, whose glyph (`◆`)
/// already says the same thing. Every cell is passed through [`sanitize_text`],
/// so an escape sequence in a task headline, a question or a command can never
/// reach the rendered row.
fn op_text(w: &WorkerRegistryEntry, show_group: bool) -> String {
    let first_line = sanitize_text(w.task.lines().next().unwrap_or("").trim());
    let mut detail = if let Some(ref q) = w.question {
        format!("ASK: {}", sanitize_text(q))
    } else if !w.last_command.is_empty()
        && w.last_command != "completed"
        && w.last_command != "initializing"
    {
        sanitize_text(&w.last_command)
    } else {
        first_line
    };
    if w.status == RegistryStatus::Reviewing {
        detail = detail
            .strip_prefix(REVIEW_PREFIX)
            .unwrap_or(&detail)
            .to_string();
    }
    match w.group.as_deref() {
        Some(g) if show_group && !g.trim().is_empty() => {
            format!("[{}] {detail}", sanitize_text(g.trim()))
        }
        _ => detail,
    }
}

/// The columns a whole frame shows, and the width each one occupies.
///
/// Decided once per frame from the terminal width and the widest value in the
/// frame, never per row: that is what keeps every row's op in the same column.
#[derive(Clone, Copy)]
struct FrameLayout {
    show_model: bool,
    show_turns: bool,
    show_elapsed: bool,
    show_bar: bool,
    /// Width of the `step/max` column: the widest value in the frame, so
    /// `9/9` is padded out beside `15/250` instead of shifting the op left.
    turns_w: usize,
    /// Width of the elapsed column: `02h 05m` is wider than `14m`, and the
    /// narrow cell is padded to match rather than pulling its op left.
    elapsed_w: usize,
}

impl FrameLayout {
    /// Choose the frame's columns for `inner` visible columns of a box body.
    ///
    /// The fixed part is the glyph, the id and their separators. The optional
    /// columns drop in the established order -- elapsed first, then the model
    /// alias, then `step/max` -- and each is shown only when every column it
    /// outranks plus a protected op of at least [`MIN_OP_WIDTH`] still fits.
    /// The progress bar rides along with `step/max` at [`PROGRESS_THRESHOLD`]+.
    fn for_width(inner: usize, widest_turns: usize, widest_elapsed: usize) -> Self {
        let show_bar = inner + 4 >= PROGRESS_THRESHOLD;
        let base = 1 + 1 + ID_WIDTH + 1;
        let turns_w = widest_turns.max(TURNS_WIDTH);
        let turns_cost = turns_w + 1 + if show_bar { BAR_WIDTH + 1 } else { 0 };
        let model_cost = MODEL_WIDTH + 1;
        let elapsed_cost = widest_elapsed + 1;

        let show_turns = base + turns_cost + MIN_OP_WIDTH <= inner;
        let show_model = show_turns && base + turns_cost + model_cost + MIN_OP_WIDTH <= inner;
        let show_elapsed =
            show_model && base + turns_cost + model_cost + elapsed_cost + MIN_OP_WIDTH <= inner;
        FrameLayout {
            show_model,
            show_turns,
            show_elapsed,
            show_bar,
            turns_w,
            elapsed_w: widest_elapsed,
        }
    }
}

/// Choose the frame's columns from every row that will be drawn.
///
/// `rows` must be exactly the rows the frame renders, so a row it does not
/// show never narrows the columns of the rows that remain. The scan takes the
/// widest `step/max` and elapsed cell in the frame and hands them to
/// [`FrameLayout::for_width`], which applies the drop order once.
fn frame_layout_for<'a>(
    rows: impl IntoIterator<Item = &'a WorkerRegistryEntry>,
    now: u64,
    inner: usize,
) -> FrameLayout {
    let mut widest_turns = 0usize;
    let mut widest_elapsed = 0usize;
    for w in rows {
        widest_turns = widest_turns.max(visible_width(&turns_cell(w)));
        widest_elapsed = widest_elapsed.max(visible_width(&elapsed_cell(w, now)));
    }
    FrameLayout::for_width(inner, widest_turns, widest_elapsed)
}

/// Fit one worker to a single compact line of at most `inner` visible columns.
///
/// Layout: `glyph id model step/max elapsed op`, with the optional columns and
/// their widths taken from `layout` -- the decision made once for the whole
/// frame -- so every row of a frame starts its op at the same column. A row
/// whose own value is narrower than the frame's column is padded out to it.
///
/// `show_group` prints the `[group]` tag in the op; callers pass `false` for a
/// row printed under its own group header, which already names the group.
fn compact_row(
    w: &WorkerRegistryEntry,
    now: u64,
    inner: usize,
    use_color: bool,
    show_group: bool,
    layout: &FrameLayout,
) -> String {
    let glyph = status_glyph(w.status);
    let id = truncate_visible(&w.id, ID_WIDTH);
    let model = model_alias(&w.model);
    let turns = turns_cell(w);
    let elapsed = elapsed_cell(w, now);
    let detail = op_text(w, show_group);

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
    if layout.show_model {
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
    if layout.show_turns {
        if layout.show_bar {
            prefix.push_str(&progress_bar(w.step, w.max_turns));
            prefix.push(' ');
        }
        prefix.push_str(&pad_visible(&turns, layout.turns_w));
        prefix.push(' ');
    }
    if layout.show_elapsed {
        prefix.push_str(&pad_visible(&elapsed, layout.elapsed_w));
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
    // One decision for the whole frame: every row gets the same columns and the
    // same column widths, so their ops all start in the same place.
    let layout = frame_layout_for(entries.iter(), now, inner);
    let mut lines = Vec::new();
    for (name, workers) in &groups {
        if show_repo {
            let repo = workers
                .first()
                .and_then(|w| w.repo_path.as_deref())
                .unwrap_or(DEFAULT_REPO_KEY);
            // The same sanitizing the interactive list applies: a heading is
            // drawn verbatim into a terminal, so nothing the registry row
            // carries may carry a terminal command of its own.
            lines.push(truncate_visible(
                &format!("repo: {}", sanitize_text(repo)),
                inner,
            ));
        }
        let counts = group_counts(workers, use_color);
        let header = if counts.is_empty() {
            sanitize_text(name)
        } else {
            format!("{}  {counts}", sanitize_text(name))
        };
        lines.push(truncate_visible(&header, inner));
        for w in workers {
            lines.push(compact_row(w, now, inner, use_color, false, &layout));
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
    if let Some(parsed) = monitor_width_override() {
        return Some(parsed);
    }
    terminal_size_via_tty()
}

/// Width for the plain (non-TTY) rendering.
///
/// A pipe has no window to ask, so the ioctl path is skipped entirely: the
/// explicit `MONITOR_WIDTH` override wins, then `COLUMNS` (which a shell
/// exports for a redirected command just as it does for an interactive one),
/// then [`DEFAULT_TERMINAL_WIDTH`]. Without this, a `mini-swe monitor | less`
/// always rendered 80 columns wide no matter what the user had set.
pub fn plain_width() -> usize {
    plain_width_from(&|key| std::env::var(key).ok())
}

/// Pure core of [`plain_width`], parameterized over the environment lookup so
/// the precedence rules can be tested without mutating process state.
fn plain_width_from(lookup: &dyn Fn(&str) -> Option<String>) -> usize {
    monitor_width_override_from(lookup)
        .or_else(|| columns_env_from(lookup))
        .unwrap_or(DEFAULT_TERMINAL_WIDTH)
}

/// The `MONITOR_WIDTH` override, ignored when unset, unparsable or zero.
fn monitor_width_override() -> Option<usize> {
    monitor_width_override_from(&|key| std::env::var(key).ok())
}

/// Pure core of [`monitor_width_override`], parameterized over the environment
/// lookup so it is testable without mutating process state.
fn monitor_width_override_from(lookup: &dyn Fn(&str) -> Option<String>) -> Option<usize> {
    env_parse_from::<usize>("MONITOR_WIDTH", lookup).filter(|&w| w > 0)
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
    columns_env_from(&|key| std::env::var(key).ok())
}

/// Pure core of [`columns_env`], parameterized over the environment lookup.
fn columns_env_from(lookup: &dyn Fn(&str) -> Option<String>) -> Option<usize> {
    env_parse_from::<usize>("COLUMNS", lookup).filter(|c| *c > 0)
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
    /// How long the turn's command took, in seconds, when the history records
    /// it. The append-only log carries no timestamp, so this stays `None`
    /// unless a future writer records one; the separator omits it then.
    pub duration_secs: Option<u64>,
    /// Whether this turn belongs to the review phase (a `[review]`-prefixed
    /// command), shown on the separator as `(review)`.
    pub review: bool,
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

/// Output lines a collapsed turn block shows: the last few, so one noisy
/// turn cannot push the rest of the view off screen.
const TURN_TAIL_LINES: usize = 5;

/// Output lines the reader keeps per turn. Enter expands a turn to everything
/// it kept, which is more than the collapsed [`TURN_TAIL_LINES`] preview but
/// still bounded, so a long session's memory stays proportional to the turns
/// shown rather than to the output the commands produced.
const TURN_KEPT_LINES: usize = 20;

/// Bytes the first read of a history log is bounded to.
///
/// The log is append-only and can already be long when the operator first
/// opens a worker, so the first read takes only its newest tail and every
/// later read continues from where that one stopped. A refresh stays
/// proportional to what changed, never to the whole log.
const FIRST_READ_BYTES: u64 = 1024 * 1024;

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
        // The first open of a long history reads only its newest tail; every
        // later read starts where the previous one stopped, on a line boundary.
        let (start, jumped) = if self.offset == 0 && self.turns.is_empty() && len > FIRST_READ_BYTES
        {
            (len - FIRST_READ_BYTES, true)
        } else {
            (self.offset, false)
        };
        file.seek(SeekFrom::Start(start))?;
        let mut bytes = Vec::new();
        // The offset follows what was *read*, not the `len` stat'd above: the
        // writer appends between the two calls, and an offset taken from the
        // stale length would re-parse those bytes on the next refresh and
        // duplicate their turns.
        let read = file.read_to_end(&mut bytes)? as u64;
        self.offset = start.saturating_add(read);
        // Bytes, not `read_to_string`: the writer appends whole lines, but a
        // read that lands mid-append can end inside a multi-byte character, and
        // a UTF-8 error there would stall this reader for the rest of the
        // worker's life. A torn character becomes a replacement character and
        // the offset still advances past it.
        let mut buf = String::from_utf8_lossy(&bytes).into_owned();
        if jumped {
            // The tail begins mid-line: drop the partial line the seek landed
            // in, so it is never parsed as a turn of its own. The newline is
            // located in the *bytes*, because the lossy conversion above can
            // have shifted the string's byte positions.
            match bytes.iter().position(|b| *b == b'\n') {
                Some(pos) => {
                    self.offset = start + pos as u64 + 1;
                    buf = String::from_utf8_lossy(&bytes[pos + 1..]).into_owned();
                }
                None => buf.clear(),
            }
        }
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
    let review = command.starts_with(REVIEW_PREFIX);
    Some(TurnView {
        step: step + 1,
        command,
        exit_code: None,
        output_lines: Vec::new(),
        duration_secs: None,
        review,
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

/// Keep the last [`TURN_KEPT_LINES`] non-blank lines of a tool output.
fn tail_lines(output: &str) -> Vec<String> {
    output
        .lines()
        .map(str::trim_end)
        .filter(|l| !l.is_empty())
        .map(str::to_string)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .take(TURN_KEPT_LINES)
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
    /// Expand/collapse the selected turn's full output (detail).
    ToggleExpand,
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
    /// Index of the turn the detail view's selection cursor points at
    /// (0 = newest). Enter expands/collapses its output.
    pub selected_turn: usize,
    /// Steps of the turns whose full (width-clipped) output is expanded.
    pub expanded: std::collections::HashSet<usize>,
}

impl Default for UiState {
    fn default() -> Self {
        Self {
            view: View::List,
            selection: 0,
            groups_expanded: true,
            follow: true,
            scroll: 0,
            selected_turn: 0,
            expanded: std::collections::HashSet::new(),
        }
    }
}

impl UiState {
    /// Apply a key to the state and return what it did.
    ///
    /// Pure and deterministic: the interactive loop feeds raw keys in and the
    /// resulting action drives the redraw. `worker_count` is the number of
    /// visible workers in the list, used to clamp the selection;
    /// `selected_step` is the step number of the turn the detail view's cursor
    /// is on, which Enter expands or collapses (the state machine cannot know
    /// it: the turns live in the caller's reader).
    pub fn apply_key(
        &mut self,
        key: Key,
        worker_count: usize,
        selected_step: Option<usize>,
    ) -> Action {
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
                Key::Up => {
                    self.selected_turn = self.selected_turn.saturating_add(1);
                    Action::MoveUp
                }
                Key::Down => {
                    self.selected_turn = self.selected_turn.saturating_sub(1);
                    Action::MoveDown
                }
                Key::Enter => match selected_step {
                    // The caller names the selected turn by its step number,
                    // which is stable while new turns arrive; the expansion
                    // follows the turn, not its position on screen.
                    Some(step) => {
                        if !self.expanded.insert(step) {
                            self.expanded.remove(&step);
                        }
                        Action::ToggleExpand
                    }
                    None => Action::None,
                },
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
/// stated order as the terminal narrows. Delegates to the shared `compact_row`
/// helper so the interactive list and the plain dashboard share one layout;
/// plain (no ANSI codes) like the non-TTY output. This is the flat view: it
/// keeps the `[group]` tag, since no group header precedes it.
pub fn fit_compact_row(w: &WorkerRegistryEntry, now: u64, width: usize) -> String {
    let inner = width.saturating_sub(4).max(1);
    compact_row(
        w,
        now,
        inner,
        false,
        true,
        &frame_layout_for([w], now, inner),
    )
}

/// One line of the key hint shown at the bottom of the interactive views.
pub fn key_hint(view: View) -> &'static str {
    match view {
        View::List => "\u{2191}\u{2193} select  \u{23ce} turns  g groups  q quit",
        View::Detail => {
            "\u{2191}\u{2193} turn  \u{23ce} expand  PgUp/PgDn scroll  f follow  Esc/q back"
        }
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
    ///
    /// Raw mode clears OPOST, so the terminal no longer turns a bare `\n`
    /// into a line feed plus carriage return: every line the monitor writes
    /// would start where the previous one ended. [`write_frame`] therefore
    /// joins every frame with `\r\n` -- the one place that decides the line
    /// ending -- and this function must not re-enable OPOST without that
    /// contract changing with it.
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
    let mut lines = Vec::new();
    let mut order = Vec::new();
    let mut groups: BTreeMap<String, Vec<&'a WorkerRegistryEntry>> = BTreeMap::new();
    for entry in entries {
        groups.entry(group_key(entry)).or_default().push(entry);
    }
    let repos: std::collections::BTreeSet<&str> = entries
        .iter()
        .map(|e| e.repo_path.as_deref().unwrap_or(DEFAULT_REPO_KEY))
        .collect();
    let show_repo = repos.len() > 1;
    for (name, workers) in &groups {
        if show_repo {
            let repo = workers
                .first()
                .and_then(|w| w.repo_path.as_deref())
                .unwrap_or(DEFAULT_REPO_KEY);
            lines.push(ListLine::Header(format!("repo: {}", sanitize_text(repo))));
        }
        let counts = group_counts(workers, false);
        let header = if counts.is_empty() {
            sanitize_text(name)
        } else {
            format!("{}  {counts}", sanitize_text(name))
        };
        let inner = width.saturating_sub(4).max(1);
        lines.push(ListLine::Header(truncate_visible(&header, inner)));
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
) -> Vec<String> {
    let (lines, order) = build_list_lines(entries, now, width, state.groups_expanded);
    // Decided once over the rows this frame shows, so the rows on screen line
    // up with each other rather than each picking its own columns.
    let inner = width.saturating_sub(4).max(1);
    let layout = frame_layout_for(order.iter().copied(), now, inner);
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
    let mut out = Vec::new();
    out.push(box_top(
        "mini-swe",
        &total,
        &counts,
        &header_clock(now),
        width,
        use_color,
    ));
    for line in lines.iter().skip(start).take(body_height) {
        match line {
            ListLine::Header(h) => out.push(box_line(h, width)),
            ListLine::Worker(w) => {
                let row = compact_row(w, now, inner, use_color, false, &layout);
                if Some(w.id.as_str()) == selected_id {
                    out.push(box_line(&format!("\x1b[7m{row}\x1b[0m"), width));
                } else {
                    out.push(box_line(&row, width));
                }
            }
        }
    }
    out.push(box_bottom(key_hint(View::List), width, use_color));
    out
}

/// Render the detail view for one worker: its REPORT/question/status header on
/// top, then its turns newest-last as clearly separated blocks. Follow mode
/// sticks to the newest turn; otherwise `state.scroll` turns are skipped from
/// the end. The view is a bordered box following the same width rules as the
/// list, and every line is clipped to the terminal width.
fn render_detail(
    entry: &WorkerRegistryEntry,
    reader: &HistoryReader,
    state: &UiState,
    now: u64,
    width: usize,
    height: usize,
) -> Vec<String> {
    let use_color = use_color_for_tty(true);
    let status_name = entry.status.display_name();
    let title = format!(
        "{} {} {} {}",
        status_glyph(entry.status),
        entry.id,
        model_alias(&entry.model),
        status_name
    );
    let total = format!("{}/{}", entry.step, entry.max_turns);
    let mut out = vec![box_top(
        &title,
        &total,
        "",
        &header_clock(now),
        width,
        use_color,
    )];

    // The turns, newest last, that the view may show.
    let skip = if state.follow {
        0
    } else {
        state.scroll.min(reader.turns.len().saturating_sub(1))
    };
    let end = reader.turns.len().saturating_sub(skip);
    let turns = &reader.turns[..end];

    // Render each turn's block: a separator line carrying the turn number, a
    // coloured ✓/✗ with the exit code, the duration when known and the review
    // phase, then the bold command and (for the tail) the dim output lines.
    // The newest turn is at the bottom of the list, so we walk newest-last.
    let mut blocks: Vec<String> = Vec::new();
    // `selected_turn` counts back from the newest turn the reader holds, so it
    // names the turn at that index from the end of `reader.turns`. `turns` is a
    // prefix of `reader.turns`, so the enumerate index below is the same index.
    let selected_idx = reader
        .turns
        .len()
        .saturating_sub(1)
        .saturating_sub(state.selected_turn);
    for (off, turn) in turns.iter().enumerate().rev() {
        let is_selected = off == selected_idx;
        blocks.push(turn_separator(turn, width, use_color, is_selected));
        let indent = "  ";
        // The command is bold after `$ ` and the output lines are dim, but only
        // when colour is on: NO_COLOR must leave the frame free of escapes.
        let cmd = if use_color {
            format!(
                "{indent}$ {}{}{C_RESET}",
                C_BOLD,
                sanitize_text(&turn.command)
            )
        } else {
            format!("{indent}$ {}", sanitize_text(&turn.command))
        };
        blocks.push(box_line(&cmd, width));
        let output_lines: Vec<String> = if state.expanded.contains(&turn.step) {
            // Fully expanded (still width-clipped): every output line.
            turn.output_lines.iter().map(|l| sanitize_text(l)).collect()
        } else {
            // Collapsed: the newest TURN_TAIL_LINES output lines only.
            let skip = turn.output_lines.len().saturating_sub(TURN_TAIL_LINES);
            turn.output_lines
                .iter()
                .skip(skip)
                .map(|l| sanitize_text(l))
                .collect()
        };
        for line in output_lines {
            let body = if use_color {
                format!("{indent}{C_DIM}{line}{C_RESET}")
            } else {
                format!("{indent}{line}")
            };
            blocks.push(box_line(&body, width));
        }
    }
    if blocks.is_empty() {
        blocks.push(box_line("(no turns recorded yet)", width));
    }

    // The header block (task/question/report) stays pinned at the top; the
    // turn blocks scroll beneath it.
    let mut header: Vec<String> = Vec::new();
    header.push(box_line(
        &format!(
            "task: {}",
            sanitize_text(entry.task.lines().next().unwrap_or(""))
        ),
        width,
    ));
    if let Some(ref q) = entry.question {
        header.push(box_line(&format!("question: {}", sanitize_text(q)), width));
    }
    if let Some(ref report) = entry.report
        && !report.is_empty()
    {
        header.push(box_line(
            &format!(
                "report: {} | files: {} | tests: {} | risks: {}",
                sanitize_text(&report.done),
                sanitize_text(&report.files),
                sanitize_text(&report.tests),
                sanitize_text(&report.risks)
            ),
            width,
        ));
    }

    let body_height = height.saturating_sub(2).max(1);
    // The header is pinned; the turn blocks fill what remains.
    let header_height = header.len().min(body_height);
    for l in header.iter().take(header_height) {
        out.push(l.clone());
    }
    let turn_height = body_height.saturating_sub(header_height);
    let start = blocks.len().saturating_sub(turn_height);
    for l in blocks.iter().skip(start).take(turn_height) {
        out.push(l.clone());
    }
    out.push(box_bottom(key_hint(View::Detail), width, use_color));
    out
}

/// The separator line of one turn block: `── turn N · ✓ exit 0 · 4s ───…`.
///
/// Carries the turn number, a coloured ✓/✗ with the exit code, the duration
/// when the history recorded one and `(review)` when the turn belongs to the
/// review phase. The line is clipped to the box width like every other line.
fn turn_separator(turn: &TurnView, width: usize, use_color: bool, selected: bool) -> String {
    let mark = if turn.exit_code == Some(0) {
        "\u{2713}"
    } else {
        "\u{2717}"
    };
    let mark = if use_color {
        let colour = if turn.exit_code == Some(0) {
            C_GREEN
        } else {
            C_RED
        };
        format!("{colour}{mark}{C_RESET}")
    } else {
        mark.to_string()
    };
    let code = turn
        .exit_code
        .map(|c| format!(" exit {c}"))
        .unwrap_or_default();
    let duration = turn
        .duration_secs
        .map(|d| format!(" \u{b7} {}", format_elapsed(d)))
        .unwrap_or_default();
    let review = if turn.review { " (review)" } else { "" };
    let cursor = if selected { " \u{25b8}" } else { "" };
    let inner = width.saturating_sub(4).max(1);
    let prefix = format!(
        "── turn {} \u{b7} {mark}{code}{duration}{review}{cursor}",
        turn.step
    );
    let dashes = inner.saturating_sub(visible_width(&prefix)).max(1);
    let line = format!("{prefix} {}", "─".repeat(dashes));
    box_line(&line, width)
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
    // `j`/`k` are the arrow keys: they move the worker selection in the list
    // and the turn cursor in the detail view.
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

/// Write one interactive frame to the terminal without flicker.
///
/// The frame is a list of already-rendered lines. The cursor moves home, each
/// line is written followed by `\x1b[K` (erase to end of line) so a shorter
/// line never leaves a stale tail, then `\x1b[J` once at the end clears any
/// leftover rows. The lines are joined with `\r\n` -- raw mode clears OPOST,
/// so a bare `\n` would not return the carriage and every line would start
/// where the previous one ended -- and the whole frame is one `write_all` plus
/// one flush. When the frame equals the previous one nothing is written at all.
fn write_frame(stdout: &mut dyn Write, frame: &[String], last: &mut Option<String>) {
    let joined = frame.join("\r\n");
    if last.as_deref() == Some(joined.as_str()) {
        return;
    }
    let mut buf = String::from("\x1b[H");
    for line in frame {
        buf.push_str(line);
        buf.push_str("\x1b[K\r\n");
    }
    buf.push_str("\x1b[J");
    let _ = stdout.write_all(buf.as_bytes());
    let _ = stdout.flush();
    *last = Some(joined);
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
    let mut last_frame: Option<String> = None;
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
                        // The turn cursor cannot point past the oldest turn.
                        state.selected_turn = state
                            .selected_turn
                            .min(reader.turns.len().saturating_sub(1));
                        render_detail(entry, reader, &state, now, width, height)
                    }
                    None => render_list(&entries, &state, now, width, height),
                }
            }
        };
        {
            let mut stdout = std::io::stdout().lock();
            write_frame(&mut stdout, &output, &mut last_frame);
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
                // The step of the turn the detail cursor is on: `selected_turn`
                // counts back from the newest, so it names a reader turn only
                // while it stays inside the history the reader holds.
                let selected_step = order.get(state.selection).and_then(|entry| {
                    readers.get(&entry.id).and_then(|reader| {
                        reader
                            .turns
                            .len()
                            .checked_sub(1)?
                            .checked_sub(state.selected_turn)
                            .map(|idx| reader.turns[idx].step)
                    })
                });
                if state.apply_key(key, order.len(), selected_step) == Action::Quit {
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
        // A pipe has no window to query, so its width comes from `COLUMNS` (or
        // `MONITOR_WIDTH`), never from an ioctl. A one-shot run on a real
        // terminal still has a window, so it keeps the ioctl path.
        let width = if is_tty {
            terminal_width().unwrap_or(DEFAULT_TERMINAL_WIDTH)
        } else {
            plain_width()
        };
        let output = render_dashboard_with_width(&entries, now, use_color_for_tty(is_tty), width);
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

    let mut last_frame: Option<String> = None;
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
            let frame: Vec<String> = output.lines().map(str::to_string).collect();
            write_frame(&mut stdout, &frame, &mut last_frame);
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
        // At 60 every column still fits beside the protected op.
        assert!(mid.contains("15/250"), "step/max missing at 60:\n{mid}");
        assert!(mid.contains("ninja"), "model alias missing at 60:\n{mid}");
        assert!(mid.contains("01m"), "elapsed missing at 60:\n{mid}");

        // Elapsed drops first: at 54 the model alias still fits, the clock does not.
        let no_elapsed = render_dashboard_with_width(&entries, 1060, false, 54);
        assert!(
            no_elapsed.contains("ninja"),
            "model alias should stay at 54:\n{no_elapsed}"
        );
        assert!(
            !no_elapsed.contains("01m"),
            "elapsed should be dropped at 54:\n{no_elapsed}"
        );

        // Then the model: at 50 step/max still fits, the alias does not.
        let no_model = render_dashboard_with_width(&entries, 1060, false, 50);
        assert!(
            no_model.contains("15/250"),
            "step/max should stay at 50:\n{no_model}"
        );
        assert!(
            !no_model.contains("ninja"),
            "model should be dropped at 50:\n{no_model}"
        );
        assert!(
            !no_model.contains("01m"),
            "elapsed should be dropped at 50:\n{no_model}"
        );

        let narrow = render_dashboard_with_width(&entries, 1060, false, 40);
        // Narrower still: step/max drops too; id + op stay.
        assert!(narrow.contains("925633bb"), "id lost at 40:\n{narrow}");
        assert!(narrow.contains("CONSOLI"), "op lost at 40:\n{narrow}");
        assert!(
            !narrow.contains("15/250"),
            "step/max should be dropped at 40:\n{narrow}"
        );
        assert!(
            !narrow.contains("ninja"),
            "model should be dropped at 40:\n{narrow}"
        );

        let tiny = render_dashboard_with_width(&entries, 1060, false, 25);
        // Narrowest: glyph + id + op always stay.
        assert!(tiny.contains("925633bb"), "id lost at 25:\n{tiny}");
        assert!(
            !tiny.contains("15/250"),
            "step/max should be dropped at 25:\n{tiny}"
        );
    }

    /// Every row of a frame starts its op at the same column, at 40, 60, 80 and
    /// 120: the visible columns and their widths are chosen once for the whole
    /// frame, so a table where one row keeps a column its neighbour dropped
    /// (a ragged table) cannot happen.
    #[test]
    fn test_every_row_of_a_frame_aligns_its_op_column() {
        // Deliberately mixed rows: a wide and a narrow `step/max` value, and
        // -- the one that mattered -- a wide (`02h 05m`) and a narrow (`01m`)
        // elapsed cell, so a per-row column decision disagrees with its
        // neighbour and the table comes out ragged.
        let entries = vec![
            Row::new("a1b2c3d4")
                .status(RegistryStatus::Running)
                .turns(15, 250)
                .command("Consolidate the round in group round57")
                .group("round57")
                .build(),
            Row::new("bb1a5885")
                .status(RegistryStatus::Completed)
                .turns(71, 150)
                .command("Polish the monitor columns")
                .group("round57")
                .updated_at(1060)
                .build(),
            Row::new("c0ffee00")
                .status(RegistryStatus::Failed)
                .turns(9, 9)
                .command("cargo test --lib monitor")
                .group("round57")
                .updated_at(1400)
                .build(),
        ];
        // Two hours after `started_at`, so the live row's elapsed is the wide
        // `02h 05m` cell while the frozen terminal rows keep a narrow one.
        let now = 1000 + 2 * 3600 + 5 * 60;
        // The op keeps at least MIN_OP_WIDTH columns, so its first 10
        // characters survive truncation at every width and locate the column.
        let ops: Vec<&str> = entries.iter().map(|e| &e.last_command[..10]).collect();
        for width in [40usize, 60, 80, 120] {
            let text = render_dashboard_with_width(&entries, now, false, width);
            let columns: Vec<usize> = ops
                .iter()
                .map(|op| {
                    text.lines()
                        .find_map(|line| line.find(op))
                        .unwrap_or_else(|| panic!("no row showing {op:?} at {width}:\n{text}"))
                })
                .collect();
            assert!(
                columns.windows(2).all(|w| w[0] == w[1]),
                "rows disagree on the op column at {width}: {columns:?}\n{text}"
            );
        }
    }

    /// The op is the protected column: at 50 columns it keeps at least
    /// [`MIN_OP_WIDTH`] visible characters, so the narrow layout spends its
    /// columns on what the worker is doing rather than on the label around it.
    #[test]
    fn test_op_keeps_its_minimum_width_at_50_columns() {
        let reviewing = Row::new("925633bb")
            .status(RegistryStatus::Reviewing)
            .turns(38, 120)
            .command("[review] ls -d target 2>/dev/null")
            .task("Review the diff")
            .group("round52")
            .build();
        let text = render_dashboard_with_width(&[reviewing], 1060, false, 50);
        let row = text
            .lines()
            .find(|l| l.contains("925633bb"))
            .unwrap_or_else(|| panic!("no row for the worker:\n{text}"));
        assert!(visible_width(row) <= 50, "row overflows 50: {row:?}");
        // The group tag and the redundant review prefix are gone; what is left
        // is the glyph, the id, step/max and a full-width op.
        assert!(!row.contains("round52"), "group tag kept: {row:?}");
        assert!(!row.contains("[review]"), "review prefix kept: {row:?}");
        assert!(row.contains("38/120"), "step/max missing: {row:?}");
        assert!(
            row.contains("ls -d target 2>/dev/null"),
            "op should fit whole at 50 cols: {row:?}"
        );
    }

    /// A row that sits under its group's header line carries no `[group]` tag:
    /// the header above already names the group, so the tag is a repeated
    /// column that only narrows the op.
    #[test]
    fn test_grouped_rows_carry_no_group_tag() {
        let entries = sample_entries();
        let text = render_dashboard_with_width(&entries, 1060, false, 80);
        let mut seen_groups = 0;
        for line in text.lines() {
            if line.contains("925633bb") {
                assert!(
                    !line.contains("round52"),
                    "grouped row repeats its group tag: {line:?}"
                );
            }
            if line.contains("ccbc2be1") {
                assert!(
                    !line.contains("round50"),
                    "grouped row repeats its group tag: {line:?}"
                );
            }
            // The headers themselves still name the group.
            if line.contains("round52") || line.contains("round50") {
                seen_groups += 1;
            }
        }
        assert_eq!(seen_groups, 2, "group headers lost:\n{text}");

        // The flat (ungrouped) view keeps the tag: it has no header to read.
        let flat = fit_compact_row(&entries[0], 1060, 80);
        assert!(
            flat.contains("[round52]"),
            "flat view must keep the group tag: {flat:?}"
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

    /// The interactive list view's group headers carry the same per-status
    /// counts as the plain dashboard, so both views stay consistent.
    #[test]
    fn test_interactive_list_group_headers_carry_counts() {
        let entries = sample_entries();
        let (lines, _) = build_list_lines(&entries, 1060, 80, true);
        let headers: Vec<String> = lines
            .iter()
            .filter_map(|l| match l {
                ListLine::Header(h) => Some(h.clone()),
                ListLine::Worker(_) => None,
            })
            .collect();
        assert!(
            headers
                .iter()
                .any(|h| h.contains("round52") && h.contains("\u{25cf}1")),
            "round52 header must carry counts: {headers:?}"
        );
        assert!(
            headers
                .iter()
                .any(|h| h.contains("round50") && h.contains("\u{2713}1")),
            "round50 header must carry counts: {headers:?}"
        );
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

    /// The plain (non-TTY) rendering honours `COLUMNS` instead of always
    /// falling back to 80, and `MONITOR_WIDTH` still overrides both.
    #[test]
    fn test_plain_width_honours_columns_without_a_tty() {
        use std::collections::HashMap;
        // A synthetic environment lookup, so the precedence rules are tested
        // without mutating the process-global environment.
        let lookup = |vars: &[(&str, &str)]| {
            let map: HashMap<String, String> = vars
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect();
            move |key: &str| map.get(key).cloned()
        };

        // COLUMNS is honoured when MONITOR_WIDTH is unset.
        let env = lookup(&[("COLUMNS", "50")]);
        assert_eq!(plain_width_from(&env), 50, "COLUMNS must be honoured");

        // Zero and unparsable values are ignored, as they are everywhere.
        let env = lookup(&[("COLUMNS", "0")]);
        assert_eq!(plain_width_from(&env), DEFAULT_TERMINAL_WIDTH, "COLUMNS=0");
        let env = lookup(&[("COLUMNS", "wide")]);
        assert_eq!(
            plain_width_from(&env),
            DEFAULT_TERMINAL_WIDTH,
            "COLUMNS=wide"
        );

        // MONITOR_WIDTH wins over COLUMNS.
        let env = lookup(&[("COLUMNS", "50"), ("MONITOR_WIDTH", "72")]);
        assert_eq!(plain_width_from(&env), 72, "MONITOR_WIDTH must win");

        // Neither set: the default.
        let env = lookup(&[]);
        assert_eq!(
            plain_width_from(&env),
            DEFAULT_TERMINAL_WIDTH,
            "neither set"
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
        assert_eq!(state.apply_key(Key::Down, 3, None), Action::MoveDown);
        assert_eq!(state.selection, 1);
        assert_eq!(state.apply_key(Key::Up, 3, None), Action::MoveUp);
        assert_eq!(state.selection, 0);
        assert_eq!(state.apply_key(Key::Up, 3, None), Action::None);
        state.selection = 2;
        assert_eq!(state.apply_key(Key::Down, 3, None), Action::None);
        assert_eq!(
            state.apply_key(Key::Char('g'), 3, None),
            Action::ToggleGroups
        );
        assert!(!state.groups_expanded);
        assert_eq!(
            state.apply_key(Key::Char('g'), 3, None),
            Action::ToggleGroups
        );
        assert!(state.groups_expanded);
        assert_eq!(state.apply_key(Key::Enter, 0, None), Action::None);
        assert_eq!(state.apply_key(Key::Enter, 3, None), Action::OpenDetail);
        assert_eq!(state.view, View::Detail);
        assert!(state.follow);
        assert_eq!(state.apply_key(Key::PageDown, 3, None), Action::ScrollDown);
        assert_eq!(state.scroll, PAGE_TURNS);
        assert_eq!(state.apply_key(Key::Esc, 3, None), Action::Back);
        assert_eq!(state.view, View::List);
        assert_eq!(state.apply_key(Key::Char('q'), 3, None), Action::Quit);
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
        // Only the last TURN_KEPT_LINES output lines survive per turn; this
        // output is shorter than the bound, so all of it stays.
        assert_eq!(reader.turns[0].output_lines.len(), 7);
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
    /// The sanitizer strips every terminal command and control character a
    /// worker's text could smuggle in, keeping only real content and layout.
    #[test]
    fn sanitize_text_strips_escapes_and_keeps_content() {
        assert_eq!(
            sanitize_text("\x1b[?1049l\x1b[2J\x1b]52;c;QUJD\x07\x1b]0;pwned\x07visible"),
            "visible",
            "alt-screen, clear, clipboard and title escapes must all go"
        );
        assert_eq!(sanitize_text("\x1bBkept"), "kept", "two-byte escape");
        assert_eq!(sanitize_text("cut\x1b[31;"), "cut", "unterminated CSI");
        assert_eq!(sanitize_text("cut\x1b]0;no end"), "cut", "unterminated OSC");
        assert_eq!(
            sanitize_text("a\tb\nc"),
            "a\tb\nc",
            "layout characters stay"
        );
        assert_eq!(
            sanitize_text("a\u{7f}b\u{9b}c\u{1}d"),
            "abcd",
            "DEL, C1 and C0 controls go"
        );
        assert_eq!(sanitize_text("caf\u{e9}"), "caf\u{e9}", "text is untouched");
    }

    /// A hostile worker cannot inject terminal escapes into the interactive
    /// list or detail views: its task, command, question, report and output are
    /// all sanitized before they reach the raw-mode screen.
    #[test]
    fn the_tui_renders_no_escapes_from_a_worker() {
        let hostile = "\x1b[?1049l\x1b[2J\x1b]0;pwned\x07";
        let mut entry = Row::new("hostile1")
            .task(&format!("do it{hostile}"))
            .command(&format!("cargo test{hostile}"))
            .repo("local")
            .build();
        entry.question = Some(format!("pause?{hostile}"));
        entry.report = Some(crate::pool::WorkerReport {
            done: format!("changed{hostile}"),
            files: "src/a.rs".to_string(),
            tests: "cargo test".to_string(),
            risks: "none".to_string(),
        });

        // The flat view: this row is rendered on its own, not under a group
        // header, so the `[group]` tag (absent here) would be part of it.
        let layout = frame_layout_for([&entry], 1060, 200);
        let row = compact_row(&entry, 1060, 200, false, true, &layout);
        assert!(
            !row.contains('\x1b'),
            "a live row carries no styling of its own, so no escape may survive: {row:?}"
        );
        assert!(row.contains("ASK: pause?"), "the question must show: {row}");

        let mut reader = HistoryReader::default();
        reader.turns.push(TurnView {
            step: 1,
            command: format!("cargo test{hostile}"),
            exit_code: Some(0),
            output_lines: vec![format!("out{hostile}")],
            duration_secs: None,
            review: false,
        });
        let detail = render_detail(&entry, &reader, &UiState::default(), 1060, 120, 24).join("\n");
        assert!(
            !detail.contains("\x1b[?1049l") && !detail.contains("]0;pwned"),
            "no worker-chosen escape may reach the terminal: {detail:?}"
        );
        assert!(
            detail.contains("out"),
            "the output must still show: {detail}"
        );

        // The headings show the group and repo the row carries, too -- in the
        // interactive list and in the plain dashboard alike, and both draw
        // into a terminal. A second repository makes the `repo:` line render,
        // so the repo path is exercised beside the group name.
        let mut grouped = entry;
        grouped.group = Some(format!("round52{hostile}"));
        grouped.repo_path = Some(format!("local{hostile}"));
        let mut second = Row::new("hostile2")
            .task("second")
            .command("true")
            .repo("other")
            .build();
        second.group = grouped.group.clone();
        let grouped_slice = [grouped, second];
        let headings: Vec<String> = build_list_lines(&grouped_slice, 1060, 200, true)
            .0
            .iter()
            .filter_map(|line| match line {
                ListLine::Header(h) => Some(h.clone()),
                ListLine::Worker(_) => None,
            })
            .chain(
                render_dashboard_with_width(&grouped_slice, 1060, false, 1060)
                    .lines()
                    .map(str::to_string),
            )
            .collect();
        assert!(
            headings.iter().all(|h| !h.contains('\x1b')),
            "no escape may survive into a heading: {headings:?}"
        );
        assert!(
            headings.iter().any(|h| h.contains("round52")),
            "the group heading must still show: {headings:?}"
        );
        assert!(
            headings.iter().any(|h| h.contains("repo:")),
            "the repo heading must still show: {headings:?}"
        );
    }

    /// The first read of a long history log takes only its newest tail, so the
    /// detail view opens fast and never parses a whole huge file.
    #[test]
    fn the_first_read_of_a_long_history_is_bounded_to_its_newest_tail() {
        let dir = std::env::temp_dir().join(format!(
            "monitor-tail-{}-{}",
            std::process::id(),
            unix_timestamp()
        ));
        std::fs::create_dir_all(&dir).expect("scratch dir");
        let path = dir.join("swe-wt-big.history.jsonl");

        let turn = |command: &str| {
            serde_json::json!({"role": "assistant", "content": "working",
                "tool_calls": [{"id": "c1", "type": "function",
                    "function": {"name": "bash",
                        "arguments": serde_json::to_string(
                            &serde_json::json!({"command": command})).expect("args")}}]})
            .to_string()
        };
        let result = serde_json::json!({"role": "tool", "tool_call_id": "c1",
            "content": format!("COMMAND OUTPUT (exit code: 0)\n{}", "x".repeat(250_000))})
        .to_string();
        let mut body = String::new();
        for i in 1..=6 {
            body.push_str(&turn(&format!("step {i}")));
            body.push('\n');
            body.push_str(&result);
            body.push('\n');
        }
        std::fs::write(&path, &body).expect("write the long history");
        assert!(
            std::fs::metadata(&path).expect("stat").len() > FIRST_READ_BYTES,
            "the test needs a log past the bound"
        );

        let mut reader = HistoryReader::default();
        reader.read_incremental(&path).expect("read");
        assert!(
            reader.turns.len() < 6,
            "the first read must not have parsed the whole file: {} turns",
            reader.turns.len()
        );
        assert_eq!(
            reader.turns.last().map(|t| t.command.as_str()),
            Some("step 6"),
            "the newest turn must be the one the view ends on"
        );
        assert_ne!(
            reader.turns.first().map(|t| t.command.as_str()),
            Some("step 1"),
            "the oldest turns are the ones the bound drops"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A read that lands inside a multi-byte character must not stall the
    /// reader: the torn character becomes a replacement and the offset still
    /// advances, so the next read continues after it.
    #[test]
    fn a_torn_multi_byte_character_does_not_stall_the_reader() {
        let dir = std::env::temp_dir().join(format!(
            "monitor-torn-{}-{}",
            std::process::id(),
            unix_timestamp()
        ));
        std::fs::create_dir_all(&dir).expect("scratch dir");
        let path = dir.join("swe-wt-torn.history.jsonl");

        let turn = |command: &str| {
            serde_json::json!({"role": "assistant", "content": "working",
                "tool_calls": [{"id": "c1", "type": "function",
                    "function": {"name": "bash",
                        "arguments": serde_json::to_string(
                            &serde_json::json!({"command": command})).expect("args")}}]})
            .to_string()
        };
        std::fs::write(&path, format!("{}\n", turn("cargo test"))).expect("seed");
        {
            use std::io::Write;
            let mut file = std::fs::OpenOptions::new()
                .append(true)
                .open(&path)
                .expect("append");
            // Cut after the first byte of the two-byte character, so the file
            // ends inside it.
            let prefix = "{\"role\":\"assistant\",\"content\":\"caf\u{e9}";
            let cut = prefix.len() - 1;
            file.write_all(&prefix.as_bytes()[..cut])
                .expect("torn write");
        }

        let mut reader = HistoryReader::default();
        reader
            .read_incremental(&path)
            .expect("a torn read must not fail");
        assert_eq!(reader.turns.len(), 1, "only the complete turn parses");
        let after_torn = reader.offset;

        // The rest of the line arrives; the reader moves on instead of stalling.
        {
            use std::io::Write;
            let mut file = std::fs::OpenOptions::new()
                .append(true)
                .open(&path)
                .expect("append");
            writeln!(file, " done\"}}").expect("finish the torn line");
            writeln!(file, "{}", turn("cargo build")).expect("next turn");
        }
        reader.read_incremental(&path).expect("read on");
        assert!(reader.offset > after_torn, "the offset must advance");
        assert_eq!(
            reader.turns.last().map(|t| t.command.as_str()),
            Some("cargo build"),
            "the turn after the torn line must be parsed: {:?}",
            reader.turns
        );
        std::fs::remove_dir_all(&dir).ok();
    }
    /// Strip ANSI escape sequences, so an assertion holds whether or not the
    /// renderer coloured the frame (`NO_COLOR` decides that at runtime).
    fn strip_escapes(s: &str) -> String {
        let mut out = String::new();
        let mut in_escape = false;
        for ch in s.chars() {
            if in_escape {
                if ch == 'm' {
                    in_escape = false;
                }
            } else if ch == '\x1b' {
                in_escape = true;
            } else {
                out.push(ch);
            }
        }
        out
    }

    /// A counting writer so a test can assert how many times a frame is written.
    struct CountingWriter {
        buf: Vec<u8>,
        writes: usize,
    }
    impl std::io::Write for CountingWriter {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.buf.extend_from_slice(buf);
            self.writes += 1;
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    /// The interactive frame joins its lines with `\r\n` (raw mode clears
    /// OPOST, so a bare `\n` would not return the carriage) and an unchanged
    /// frame produces no write at all.
    #[test]
    fn test_write_frame_joins_with_crlf_and_skips_unchanged() {
        let mut w = CountingWriter {
            buf: Vec::new(),
            writes: 0,
        };
        let mut last: Option<String> = None;
        let frame = vec!["line one".to_string(), "line two".to_string()];

        write_frame(&mut w, &frame, &mut last);
        assert_eq!(w.writes, 1, "one write per changed frame");
        let text = String::from_utf8(w.buf.clone()).expect("utf8");
        assert!(
            text.contains("\r\n"),
            "frame must join lines with CRLF: {text:?}"
        );
        assert!(
            !text.contains("\n") || text.contains("\r\n"),
            "no bare LF may separate interactive lines: {text:?}"
        );
        assert!(text.contains("\x1b[H"), "frame must move the cursor home");
        assert!(text.contains("\x1b[K"), "each line erases to end of line");
        assert!(
            text.contains("\x1b[J"),
            "one clear-to-end-of-screen at the end"
        );

        // An identical frame writes nothing.
        write_frame(&mut w, &frame, &mut last);
        assert_eq!(w.writes, 1, "an unchanged frame must not be written");

        // A changed frame writes again.
        let changed = vec!["line one".to_string(), "line changed".to_string()];
        write_frame(&mut w, &changed, &mut last);
        assert_eq!(w.writes, 2, "a changed frame must be written");
    }

    /// Every rendered line of the interactive list is within the terminal width
    /// at 40, 60 and 100, borders included.
    #[test]
    fn test_interactive_list_lines_fit_width() {
        let entries = sample_entries();
        for width in [40usize, 60, 100] {
            let frame = render_list(&entries, &UiState::default(), 1060, width, 20);
            for line in &frame {
                assert!(
                    visible_width(line) <= width,
                    "list line overflows {width} cols ({}): {line:?}",
                    visible_width(line)
                );
            }
        }
    }

    /// Every rendered line of the detail view is within the terminal width at
    /// 40, 60 and 100, and a turn block carries its separator with the turn
    /// number and exit code.
    #[test]
    fn test_interactive_detail_lines_fit_width_and_turn_blocks() {
        let entry = Row::new("925633bb")
            .status(RegistryStatus::Running)
            .turns(15, 250)
            .command("cargo test --lib monitor")
            .task("Consolidate the round")
            .group("round52")
            .build();
        let mut reader = HistoryReader::default();
        reader.turns.push(TurnView {
            step: 12,
            command: "cargo test --lib monitor 2>&1 | tail -5".to_string(),
            exit_code: Some(0),
            output_lines: vec![
                "test result: ok. 41 passed; 0 failed".to_string(),
                "another output line that is long".to_string(),
            ],
            duration_secs: Some(4),
            review: false,
        });
        reader.turns.push(TurnView {
            step: 13,
            command: "cargo clippy --all-targets -- -D warnings".to_string(),
            exit_code: Some(101),
            output_lines: vec!["error: unused variable `width`".to_string()],
            duration_secs: Some(2),
            review: true,
        });

        for width in [40usize, 60, 100] {
            let frame = render_detail(&entry, &reader, &UiState::default(), 1060, width, 24);
            for line in &frame {
                assert!(
                    visible_width(line) <= width,
                    "detail line overflows {width} cols ({}): {line:?}",
                    visible_width(line)
                );
            }
            let text = strip_escapes(&frame.join("\n"));
            assert!(
                text.contains("turn 12") && text.contains("exit 0"),
                "turn 12 separator must carry the number and exit code: {text}"
            );
            assert!(
                text.contains("turn 13") && text.contains("exit 101"),
                "turn 13 separator must carry the number and exit code: {text}"
            );
            if width >= 100 {
                assert!(
                    text.contains("(review)"),
                    "a review turn must be marked (review): {text}"
                );
            }
            if width >= 60 {
                assert!(
                    text.contains("$ cargo test --lib monitor 2>&1 | tail -5"),
                    "the command must show after `$ `: {text}"
                );
                assert!(
                    text.contains("test result: ok. 41 passed; 0 failed"),
                    "the output tail must show: {text}"
                );
            }
        }
    }

    /// Up/Down select a turn and Enter expands/collapses its full output in the
    /// detail view; follow clamps the selection to the newest turn.
    #[test]
    fn test_detail_turn_selection_and_expand_collapse() {
        let mut state = UiState {
            view: View::Detail,
            ..UiState::default()
        };
        assert_eq!(state.apply_key(Key::Up, 0, None), Action::MoveUp);
        assert_eq!(state.selected_turn, 1);
        assert_eq!(state.apply_key(Key::Down, 0, None), Action::MoveDown);
        assert_eq!(state.selected_turn, 0);
        // Enter expands the turn the cursor is on (step 12); with no turn
        // selected there is nothing to expand.
        assert_eq!(state.apply_key(Key::Enter, 0, None), Action::None);
        assert!(
            state.expanded.is_empty(),
            "nothing to expand without a turn"
        );
        assert_eq!(
            state.apply_key(Key::Enter, 0, Some(12)),
            Action::ToggleExpand
        );
        assert!(state.expanded.contains(&12), "turn 12 must expand");
        // Enter again collapses it.
        assert_eq!(
            state.apply_key(Key::Enter, 0, Some(12)),
            Action::ToggleExpand
        );
        assert!(!state.expanded.contains(&12), "turn 12 must collapse");
    }

    /// Enter expands a turn to every output line the reader kept, which is more
    /// than the collapsed preview, so the expansion is a real behaviour change.
    #[test]
    fn test_expanding_a_turn_shows_more_output_lines() {
        let entry = Row::new("925633bb")
            .status(RegistryStatus::Running)
            .turns(15, 250)
            .command("cargo test")
            .task("T")
            .group("g")
            .build();
        let mut reader = HistoryReader::default();
        let lines: Vec<String> = (0..TURN_KEPT_LINES)
            .map(|i| format!("output line {i}"))
            .collect();
        reader.turns.push(TurnView {
            step: 1,
            command: "cargo test".to_string(),
            exit_code: Some(0),
            output_lines: lines,
            duration_secs: None,
            review: false,
        });

        // Collapsed: only the newest TURN_TAIL_LINES show.
        let collapsed =
            render_detail(&entry, &reader, &UiState::default(), 1060, 120, 40).join("\n");
        assert!(
            collapsed.contains("output line 19"),
            "newest must show: {collapsed}"
        );
        assert!(
            !collapsed.contains("output line 0"),
            "oldest must be hidden when collapsed: {collapsed}"
        );

        // Expanded: every kept line shows.
        let mut state = UiState::default();
        state.expanded.insert(1);
        let expanded = render_detail(&entry, &reader, &state, 1060, 120, 40).join("\n");
        assert!(
            expanded.contains("output line 0"),
            "expanded must show oldest: {expanded}"
        );
        assert!(
            expanded.contains("output line 19"),
            "expanded must show newest: {expanded}"
        );
    }
    /// The turn cursor starts on the newest turn: its separator carries the
    /// `\u{25b8}` marker, the older turn's does not. The cursor counts back from
    /// the newest turn, not from the oldest line on screen.
    #[test]
    fn test_turn_cursor_marks_the_newest_turn_by_default() {
        let entry = Row::new("925633bb")
            .status(RegistryStatus::Running)
            .turns(15, 250)
            .command("cargo test")
            .task("T")
            .group("g")
            .build();
        let mut reader = HistoryReader::default();
        for step in [7usize, 8] {
            reader.turns.push(TurnView {
                step,
                command: format!("cmd {step}"),
                exit_code: Some(0),
                output_lines: vec![format!("out {step}")],
                duration_secs: None,
                review: false,
            });
        }
        let frame = render_detail(&entry, &reader, &UiState::default(), 1060, 80, 24);
        let newest = frame
            .iter()
            .find(|l| strip_escapes(l).contains("turn 8"))
            .unwrap_or_else(|| panic!("newest turn missing: {frame:?}"));
        assert!(
            strip_escapes(newest).contains('\u{25b8}'),
            "the cursor must mark the newest turn: {newest:?}"
        );
        let oldest = frame
            .iter()
            .find(|l| strip_escapes(l).contains("turn 7"))
            .unwrap_or_else(|| panic!("oldest turn missing: {frame:?}"));
        assert!(
            !strip_escapes(oldest).contains('\u{25b8}'),
            "the cursor must not mark the oldest turn: {oldest:?}"
        );
    }
}
