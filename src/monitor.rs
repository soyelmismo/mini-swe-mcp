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
        RegistryStatus::Completed | RegistryStatus::Stopped | RegistryStatus::Interrupted => {
            C_DIM
        }
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
        let heading = truncate_visible(&format!("[{repo_path}]  {summary}"), width);
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
        &format!("task: {}", entry.task.lines().next().unwrap_or("")),
        width,
    ));
    out.push('\n');
    if let Some(ref q) = entry.question {
        out.push_str(&truncate_visible(&format!("question: {q}"), width));
        out.push('\n');
    }
    if let Some(ref report) = entry.report
        && !report.is_empty()
    {
        out.push_str(&truncate_visible(
            &format!(
                "report: {} | files: {} | tests: {} | risks: {}",
                report.done, report.files, report.tests, report.risks
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
            &format!("#{} {}{}", turn.step, turn.command, code),
            width,
        ));
        for line in &turn.output_lines {
            turn_lines.push(truncate_visible(&format!("  {line}"), width));
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
