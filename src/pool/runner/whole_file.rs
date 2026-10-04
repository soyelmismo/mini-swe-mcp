//! Show a small file whole when a worker asks for a slice of it.
//!
//! A worker that reads a 300-line source file with `sed -n '1,120p'` pays a
//! turn for the slice and another for the rest, and the run that motivated
//! this did it seventeen times in sixty-one turns. The models are told to read
//! files whole; some of them slice anyway, so the harness closes the gap here,
//! for every role and every model.
//!
//! ## What qualifies
//!
//! One command, one file, one range: `sed -n 'A,Bp' FILE`, `sed -n 'A,B p'
//! FILE`, `head -n N FILE`, `head -N FILE` or `tail -n N FILE`, optionally
//! preceded by `cd <dir> &&`. The file must resolve to a regular file inside
//! the worker's worktree small enough to be worth showing whole
//! ([`WHOLE_FILE_MAX_LINES`] lines, [`WHOLE_FILE_MAX_BYTES`] bytes). Everything
//! else keeps its normal output: a pipeline, a second file, a glob, another
//! flag, a missing file, a failed command, or content that is not UTF-8.
//!
//! ## Why the read is a harness read
//!
//! The replacement text is produced by the harness, outside the sandbox, so it
//! must never show a byte the sandboxed command could not have shown. Three
//! rules hold that line, and each is enforced on the path rather than assumed:
//!
//! * the worktree root is canonicalized, and the file's resolved path must stay
//!   under it, so a `..` escape, an absolute path outside the worktree, or a
//!   symlink pointing out of the worktree is refused rather than followed;
//! * the file is opened with `O_NOFOLLOW`, so the final component cannot be
//!   swapped for a link between the check and the read, and the opened handle's
//!   own type must be a regular file -- a directory, a FIFO or a device is
//!   refused;
//! * the substitution happens only after the sandboxed command itself exited
//!   zero, so a failed read keeps its error and the model sees what happened.
//!
//! ## Why it does not loop
//!
//! Showing the same file whole on every slice would spend the token saving the
//! harness spends. [`WholeFileGuard`] remembers the step and the size and mtime
//! each shown file had, and a slice of an unchanged file inside
//! [`WHOLE_FILE_REPEAT_TURNS`] turns keeps the ranged output with a note
//! pointing at the step that already showed the file.

use std::collections::VecDeque;
use std::fs::OpenOptions;
use std::io::Read;
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::path::{Path, PathBuf};

use tracing::info;

/// Largest file the harness will show whole, in lines.
///
/// A file under this bound is small enough that the whole of it costs one
/// answer where two slices cost two turns; past it, the model's own judgement
/// about which lines it needs is cheaper than a few hundred lines it did not.
const WHOLE_FILE_MAX_LINES: usize = 600;

/// Largest file the harness will show whole, in bytes: the line bound alone
/// would admit one enormous line, which is the case the whole-file answer is
/// meant to prevent.
const WHOLE_FILE_MAX_BYTES: u64 = 48 * 1024;

/// Turns within which a slice of an already-shown, unchanged file keeps its
/// range instead of the whole file again.
pub(super) const WHOLE_FILE_REPEAT_TURNS: usize = 8;

/// What the harness should answer a range read with.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum WholeFileReply {
    /// Nothing to do: the command is not a plain range read, the file is not
    /// small enough, or it could not be read safely. The command's own output
    /// stands, byte for byte.
    Keep,
    /// Replace the observation with the whole file, numbered like `cat -n`.
    Whole { text: String },
    /// Keep the ranged output, but say where the whole file was already shown.
    AlreadyShown { step: usize },
}

/// Decide what to answer `command`, which exited zero inside `worktree`.
///
/// Called only on an exit-zero step: a failed command keeps its error, so the
/// model reads what went wrong rather than a file the harness read on its
/// behalf.
pub(super) fn whole_file_reply(
    command: &str,
    worktree: &Path,
    guard: &mut WholeFileGuard,
    step: usize,
) -> WholeFileReply {
    let Some(read) = parse_range_read(command) else {
        return WholeFileReply::Keep;
    };
    let Ok(root) = worktree.canonicalize() else {
        return WholeFileReply::Keep;
    };
    let Some(resolved) = resolve_inside(&root, read.dir.as_deref(), &read.display) else {
        return WholeFileReply::Keep;
    };
    let Some((stamp, text)) = read_small_regular_file(&resolved) else {
        return WholeFileReply::Keep;
    };
    let line_count = text.lines().count();
    if line_count > WHOLE_FILE_MAX_LINES {
        return WholeFileReply::Keep;
    }
    let spelled = read.display;
    if let Some(shown_at) = guard.recent_show(&resolved, stamp, step) {
        info!(
            file = %spelled,
            step,
            shown_at,
            "Range read of a file already shown whole; keeping the ranged output"
        );
        return WholeFileReply::AlreadyShown { step: shown_at };
    }
    guard.record(resolved, stamp, step);
    info!(
        file = %spelled,
        lines = line_count,
        "Replacing a small range read with the whole file"
    );
    WholeFileReply::Whole {
        text: numbered(&spelled, line_count, &text),
    }
}

/// Render the whole-file answer: one harness line saying what happened, then the
/// file numbered like `cat -n`, so the line numbers the model asked for are the
/// ones it gets.
fn numbered(spelled: &str, lines: usize, text: &str) -> String {
    let mut out = format!(
        "[harness: {spelled} has {lines} lines; showing the whole file instead of the requested range]\n"
    );
    for (i, line) in text.lines().enumerate() {
        out.push_str(&format!("{:>6}\t{}\n", i + 1, line));
    }
    out
}

/// The note that replaces a repeat slice, appended to the normal output.
pub(super) fn already_shown_note(step: usize) -> String {
    format!("[harness: the whole file was shown at step {step} and is unchanged]")
}

/// A range read of one file, as the command spelled it.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct RangeRead {
    /// The path as written, which the harness note quotes back verbatim so the
    /// model sees its own spelling rather than a rewritten one.
    pub display: String,
    /// The directory the command changed into, relative to the worktree root,
    /// when it spelled a `cd <dir> &&` prefix.
    pub dir: Option<String>,
}

/// Parse a plain range read of one file, or `None` for anything else.
///
/// Deliberately narrow: only the spellings that cost a turn to read whole are
/// recognized, and anything that could mean more than "show me these lines of
/// this one file" -- a pipeline, several files, a glob, another flag, a
/// redirection -- is left to the command's own output. A glob never survives
/// the parse, because a shell-expanded name is never the literal the parser
/// sees, and an unexpanded `*` is not a path that can resolve to one file.
fn parse_range_read(command: &str) -> Option<RangeRead> {
    // The `cd <dir> &&` prefix is taken off textually, before tokenizing,
    // because `&&` is a metacharacter the tokenizer refuses everywhere else:
    // this is the one place the harness accepts a separator, and only as the
    // two-word `cd` form that changes where the file is looked for while
    // naming no other work.
    let (dir, rest) = match command.split_once("&&") {
        Some((head, rest)) => {
            let head = shell_words(head.trim())?;
            match head.as_slice() {
                [cd, dir] if cd == "cd" => (Some(dir.clone()), rest.trim()),
                _ => return None,
            }
        }
        None => (None, command),
    };
    let words = shell_words(rest)?;
    let (verb, rest) = words.split_first()?;
    let display = match verb.as_str() {
        "sed" => sed_range(rest)?,
        "head" | "tail" => {
            let rest = split_operands(rest);
            // One operand: several files is not one range read of one file.
            if rest.len() != 1 {
                return None;
            }
            rest[0].clone()
        }
        _ => return None,
    };
    Some(RangeRead { display, dir })
}

/// Drop a `head`/`tail` count, in either spelling, leaving the operands.
///
/// `head -n 20` and `head -20` both name a count, and a bare `head` names its
/// own default of ten; either way the operands are what is left, and they must
/// be exactly one file.
fn split_operands(words: &[String]) -> Vec<String> {
    match words {
        [first, second, rest @ ..] if first == "-n" && is_number(second) => rest.to_vec(),
        [one, rest @ ..] if one.len() > 1 && one.starts_with('-') && is_number(&one[1..]) => {
            rest.to_vec()
        }
        rest => rest.to_vec(),
    }
}

/// The `sed` range spelling: `-n 'A,Bp'` or `-n 'A,B p'`, one file.
///
/// Only the silent flag: `sed '1,10p'` without `-n` prints the whole file
/// *and* the range, which is not a slice read of anything. The script must be
/// the range and its `p` alone, so `1,10{s/x/y/}` and `10p;11p` are not it.
fn sed_range(words: &[String]) -> Option<String> {
    let [flag, script, file] = words else {
        return None;
    };
    if flag != "-n" {
        return None;
    }
    // `1,5p` and `1,5 p` are the same script; only the range and the `p` may
    // be there, so the bounds are trimmed and must be plain counts.
    let bounds = script.strip_suffix('p')?.trim_end();
    let bounds: Vec<&str> = bounds.split(',').collect();
    if !(1..=2).contains(&bounds.len()) || !bounds.iter().copied().all(is_number) {
        return None;
    }
    Some(file.clone())
}

/// Whether `text` is a plain decimal count.
fn is_number(text: &str) -> bool {
    !text.is_empty() && text.bytes().all(|b| b.is_ascii_digit())
}

/// Split a command into words the way the shell does for this one shape:
/// whitespace-separated, with single and double quotes removed.
///
/// Returns `None` if the command carries anything this does not model -- a
/// pipeline, a redirection, a subshell, a glob -- so no other command can be
/// mistaken for a plain range read of one file.
fn shell_words(command: &str) -> Option<Vec<String>> {
    let mut words = Vec::new();
    let mut current = String::new();
    let mut has_word = false;
    let mut chars = command.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            ' ' | '\t' | '\n' => {
                if has_word {
                    words.push(std::mem::take(&mut current));
                    has_word = false;
                }
            }
            '\'' | '"' => {
                let quote = c;
                has_word = true;
                loop {
                    let inner = chars.next()?;
                    if inner == quote {
                        break;
                    }
                    current.push(inner);
                }
            }
            '|' | '&' | ';' | '>' | '<' | '(' | ')' | '`' | '$' | '*' | '?' | '{' | '}' => {
                return None;
            }
            '\\' => {
                // An escape keeps the next character literal; the escape itself
                // is not part of the word.
                has_word = true;
                chars.next()?;
            }
            _ => {
                has_word = true;
                current.push(c);
            }
        }
    }
    if has_word {
        words.push(current);
    }
    Some(words)
}

/// Resolve `file`, as the command spelled it, to a real path inside `root`.
///
/// A relative name is taken against the `cd` directory when the command named
/// one and against the worktree root otherwise; an absolute name is taken as
/// written. Either way the result must resolve *under* the canonical root, so
/// `..` escapes, an absolute path outside the worktree, and a symlink whose
/// target lies outside it are all refused rather than followed.
fn resolve_inside(root: &Path, dir: Option<&str>, file: &str) -> Option<PathBuf> {
    let base: PathBuf = match dir {
        Some(dir) => root.join(dir),
        None => root.to_path_buf(),
    };
    let candidate = base.join(file);
    let canonical = candidate.canonicalize().ok()?;
    canonical.starts_with(root).then_some(canonical)
}

/// Read a small regular file through a handle that cannot be a link.
///
/// `O_NOFOLLOW` and `O_NONBLOCK` are on the open itself, not a check that
/// precedes it: a `symlink_metadata`/`read_to_string` pair would have a window
/// in which a regular file is replaced by a link or a FIFO, and `O_NONBLOCK`
/// keeps a planted FIFO from blocking in `open(2)` before any type check runs.
fn read_small_regular_file(path: &Path) -> Option<(FileStamp, String)> {
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)
        .ok()?;
    let meta = file.metadata().ok()?;
    if !meta.file_type().is_file() || meta.len() > WHOLE_FILE_MAX_BYTES {
        return None;
    }
    let mut text = String::new();
    // `read_to_string` fails on invalid UTF-8, which is what keeps binary
    // content out of a whole-file answer: the model's own command would have
    // printed something, but the harness must not guess what that was.
    file.take(WHOLE_FILE_MAX_BYTES + 1)
        .read_to_string(&mut text)
        .ok()?;
    Some((FileStamp::of(&meta), text))
}

/// What a shown file looked like when the harness replaced a slice with it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FileStamp {
    len: u64,
    mtime: i64,
    mtime_nsec: i64,
}

impl FileStamp {
    fn of(meta: &std::fs::Metadata) -> Self {
        Self {
            len: meta.len(),
            mtime: meta.mtime(),
            mtime_nsec: meta.mtime_nsec(),
        }
    }
}

/// Remember the files the harness has already shown whole, so a worker that
/// slices one file a dozen times is answered once and pointed at the rest.
///
/// Keyed by canonical path, valued by the step and the file's size and mtime
/// when it was shown: a file that has changed is worth showing again, and one
/// that has not is not. Only the stamp is kept, never content: this guard
/// exists to avoid repeating a payload, so it must not hold one.
#[derive(Default)]
pub(super) struct WholeFileGuard {
    shown: VecDeque<(PathBuf, FileStamp, usize)>,
}

impl WholeFileGuard {
    /// Record that `path` looked like `stamp` when it was shown whole at
    /// `step`, evicting the oldest entry past the window so a long run cannot
    /// grow this without bound.
    fn record(&mut self, path: PathBuf, stamp: FileStamp, step: usize) {
        self.shown.retain(|(p, s, _)| !(p == &path && *s == stamp));
        self.shown.push_back((path, stamp, step));
        while self.shown.len() > WHOLE_FILE_REPEAT_TURNS {
            self.shown.pop_front();
        }
    }

    /// The step at which `path`, unchanged since, was last shown whole and
    /// still inside the repeat window, or `None` when it was never shown, has
    /// changed since, or was shown long enough ago that showing it again is
    /// cheaper than pointing at it.
    fn recent_show(&self, path: &Path, stamp: FileStamp, step: usize) -> Option<usize> {
        self.shown
            .iter()
            .rev()
            .find(|(p, s, shown_at)| {
                p == path
                    && *s == stamp
                    && step.saturating_sub(*shown_at) <= WHOLE_FILE_REPEAT_TURNS
            })
            .map(|(_, _, shown_at)| *shown_at)
    }
}
