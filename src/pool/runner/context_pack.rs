//! A bounded context pack appended to the worker's opening message.
//!
//! Most non-trivial workers spend their first dozen turns re-discovering what
//! the task text already names: the files it mentions and the symbols it
//! quotes. The pack is the harness's answer, computed from the task alone
//! before the first turn -- every named repository path with a short
//! top-level outline, then every backticked identifier located with
//! `git grep -n -w`.
//!
//! Everything here is deliberately bounded and best-effort, because a task is
//! free text and a repository is arbitrarily large:
//!
//! * the whole pack is capped at [`PACK_CAP_BYTES`], and the least relevant
//!   lines (the identifier hits, which come last) are dropped first;
//! * each file outline keeps at most [`OUTLINE_CAP`] item lines;
//! * each identifier keeps at most [`HITS_PER_IDENTIFIER`] hits, and at most
//!   [`MAX_IDENTIFIERS`] identifiers are located at all.
//!
//! A task that names nothing yields no pack. The heading says the pack is
//! generated, so a worker treats it as a map to verify, not as ground truth.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use regex::Regex;

/// Hard ceiling on the rendered pack, in bytes.
pub const PACK_CAP_BYTES: usize = 6 * 1024;

/// Most top-level item lines kept for one named file.
pub const OUTLINE_CAP: usize = 15;

/// Most `git grep` hits shown for one located identifier.
const HITS_PER_IDENTIFIER: usize = 3;

/// Most identifiers located, bounding both the work and the output.
const MAX_IDENTIFIERS: usize = 8;

/// Most repository paths outlined, bounding the work and the output.
const MAX_PATHS: usize = 12;

/// Most files the fallback scanner reads before giving up.
const MAX_SCAN_FILES: usize = 5_000;

/// The heading the pack appears under, so it cannot be mistaken for the task.
const HEADING: &str = "Where things are (generated, may be incomplete):";

/// Directory names never descended into by the fallback scanner.
const SKIP_DIRS: &[&str] = &[
    ".git",
    "target",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "dist",
    "build",
];

/// Extensions treated as repository source (and text) files.
const SOURCE_EXTENSIONS: &[&str] = &[
    "rs", "py", "ts", "tsx", "js", "jsx", "go", "md", "yaml", "yml", "toml", "json", "sh", "c",
    "h", "cc", "cpp", "hpp", "java", "rb",
];

/// Build the context pack for `task`, resolved against the worker's `root`.
///
/// Returns `None` when the task names no path that exists and no identifier
/// that can be located: an empty pack would only be noise in the prompt.
pub fn context_pack(task: &str, root: &Path) -> Option<String> {
    let paths = extract_paths(task, root);
    let identifiers = extract_identifiers(task);
    let located = locate_identifiers(root, &identifiers);
    if paths.is_empty() && located.is_empty() {
        return None;
    }
    Some(render(task, root, &paths, &located))
}

/// Every repository-relative path the task names and `root` actually contains.
pub fn extract_paths(task: &str, root: &Path) -> Vec<PathBuf> {
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for m in token_regex().find_iter(task) {
        if out.len() >= MAX_PATHS {
            break;
        }
        let token = m
            .as_str()
            .trim_matches(|c: char| matches!(c, '.' | ',' | ';' | ':' | '"' | '\''));
        if token.is_empty() || !path_like(token) || !seen.insert(token.to_string()) {
            continue;
        }
        if token_contains_escape(token) {
            continue;
        }
        if root.join(token).is_file() {
            out.push(PathBuf::from(token));
        }
    }
    out
}

/// Every backticked identifier and `path::to::item` the task quotes.
pub fn extract_identifiers(task: &str) -> Vec<String> {
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for cap in backtick_regex().captures_iter(task) {
        let raw = cap.get(1).map(|m| m.as_str().trim()).unwrap_or("");
        if !is_identifier(raw) || !seen.insert(raw.to_string()) {
            continue;
        }
        out.push(raw.to_string());
        if out.len() >= MAX_IDENTIFIERS {
            return out;
        }
    }
    for m in path_item_regex().find_iter(task) {
        if out.len() >= MAX_IDENTIFIERS {
            break;
        }
        let raw = m.as_str();
        if !is_identifier(raw) || !seen.insert(raw.to_string()) {
            continue;
        }
        out.push(raw.to_string());
    }
    out
}

/// The top-level item lines of one file, as `"<line>: <text>"`, capped at
/// `OUTLINE_CAP` with a trailing count of what was omitted.
///
/// The language is chosen from the extension and falls back to a generic
/// "non-indented declaration-looking line" rule for anything else.
pub fn outline_file(path: &Path) -> Vec<String> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    let mut out = Vec::new();
    let mut omitted = 0usize;
    for (i, line) in text.lines().enumerate() {
        if !is_outline_line(ext, line) {
            continue;
        }
        if out.len() >= OUTLINE_CAP {
            omitted += 1;
            continue;
        }
        out.push(format!("{}: {}", i + 1, line.trim_end()));
    }
    if omitted > 0 {
        out.push(format!("... {omitted} more top-level items"));
    }
    out
}

// ----------
// Task extraction
// ----------

/// A whitespace-free token that can carry a path: letters, digits and `/._-`.
fn token_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"[A-Za-z0-9_./-]+").expect("path token regex must compile"))
}

fn backtick_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"`([^`\n]{1,120})`").expect("backticked span regex must compile"))
}

fn path_item_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"[A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)+")
            .expect("path::item regex must compile")
    })
}

/// Whether `token` could name a file outside the worktree.
///
/// The task is free text and the token is joined onto the worktree root, so an
/// absolute token replaces the root outright and any `..` segment climbs out
/// of it. The pack is prompt text the worker is told to act on, so a token
/// that can point anywhere on disk is a prompt-injection read primitive: the
/// outline would carry that file's top-level lines into the first message.
/// Refuse both, and let the identifier locator (which only greps inside the
/// checkout) carry genuinely useful absolute-looking names instead.
fn token_contains_escape(token: &str) -> bool {
    let path = Path::new(token);
    if path.is_absolute() {
        return true;
    }
    path.components().any(|c| {
        matches!(
            c,
            std::path::Component::ParentDir
                | std::path::Component::RootDir
                | std::path::Component::Prefix(_)
        )
    })
}

fn path_like(token: &str) -> bool {
    if token.contains('/') {
        return true;
    }
    token
        .rsplit_once('.')
        .is_some_and(|(_, ext)| SOURCE_EXTENSIONS.contains(&ext))
}

fn is_identifier(raw: &str) -> bool {
    !raw.is_empty()
        && raw.len() <= 120
        && raw.split("::").all(|part| {
            let mut chars = part.chars();
            chars
                .next()
                .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
                && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
        })
}

// ----------
// Location
// ----------

/// Locate each identifier's first few hits, dropping identifiers nothing hits.
///
/// Uses `git grep -n -w` inside the worktree when `root` is a checkout, and a
/// bounded filesystem walk otherwise (a test fixture is not a git repository).
fn locate_identifiers(root: &Path, identifiers: &[String]) -> Vec<(String, Vec<String>)> {
    let use_git = root.join(".git").exists();
    identifiers
        .iter()
        .filter_map(|ident| {
            let hits = if use_git {
                git_grep(root, ident)
            } else {
                scan_dir(root, ident)
            };
            (!hits.is_empty()).then(|| (ident.clone(), hits))
        })
        .collect()
}

fn git_grep(root: &Path, ident: &str) -> Vec<String> {
    let Ok(out) = crate::worktree::git(root, "grep", &["grep", "-n", "-w", "--", ident]) else {
        return Vec::new();
    };
    if !out.status.success() {
        // `git grep` exits 1 for "no matches"; any other status is a failure.
        return Vec::new();
    }
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .take(HITS_PER_IDENTIFIER)
        .map(str::to_string)
        .collect()
}

/// The git-free fallback: an ordered, bounded walk of `root`.
fn scan_dir(root: &Path, ident: &str) -> Vec<String> {
    let mut hits = Vec::new();
    let mut visited = 0usize;
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        if hits.len() >= HITS_PER_IDENTIFIER || visited >= MAX_SCAN_FILES {
            break;
        }
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        let mut paths: Vec<PathBuf> = entries.flatten().map(|e| e.path()).collect();
        paths.sort();
        for path in paths {
            if hits.len() >= HITS_PER_IDENTIFIER {
                break;
            }
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if path.is_dir() {
                if !SKIP_DIRS.contains(&name) {
                    stack.push(path);
                }
                continue;
            }
            visited += 1;
            if visited > MAX_SCAN_FILES || !is_source_file(&path) {
                continue;
            }
            let Ok(text) = std::fs::read_to_string(&path) else {
                continue;
            };
            for (i, line) in text.lines().enumerate() {
                if contains_word(line, ident) {
                    let rel = path.strip_prefix(root).unwrap_or(&path);
                    hits.push(format!("{}:{}: {}", rel.display(), i + 1, line.trim()));
                    if hits.len() >= HITS_PER_IDENTIFIER {
                        break;
                    }
                }
            }
        }
    }
    hits
}

fn is_source_file(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|ext| SOURCE_EXTENSIONS.contains(&ext))
}

/// Whether `needle` occurs in `hay` bounded by non-word bytes on both sides.
fn contains_word(hay: &str, needle: &str) -> bool {
    let bytes = hay.as_bytes();
    let mut from = 0usize;
    while from < bytes.len() {
        let Some(pos) = hay[from..].find(needle) else {
            return false;
        };
        let start = from + pos;
        let end = start + needle.len();
        let before_ok = start == 0 || !is_word_byte(bytes[start - 1]);
        let after_ok = end >= bytes.len() || !is_word_byte(bytes[end]);
        if before_ok && after_ok {
            return true;
        }
        from = start + 1;
    }
    false
}

fn is_word_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

// ----------
// Outline classification
// ----------

fn is_outline_line(ext: &str, line: &str) -> bool {
    if line.is_empty() || line.starts_with(char::is_whitespace) {
        return false;
    }
    let trimmed = line.trim_end();
    if trimmed.starts_with("//")
        || trimmed.starts_with('#')
        || trimmed.starts_with("/*")
        || trimmed.starts_with('*')
    {
        return false;
    }
    match ext {
        "rs" => rust_item(trimmed),
        "py" => starts_with_any(trimmed, &["def ", "class ", "async def "]),
        "ts" | "tsx" | "js" | "jsx" => starts_with_any(
            trimmed,
            &[
                "function ",
                "class ",
                "export ",
                "interface ",
                "async function ",
                "declare ",
                "const ",
                "let ",
                "type ",
            ],
        ),
        "go" => starts_with_any(trimmed, &["func ", "type ", "var ", "const "]),
        _ => generic_decl(trimmed),
    }
}

fn rust_item(line: &str) -> bool {
    let stripped = strip_prefixes(
        line,
        &[
            "pub(crate) ",
            "pub(super) ",
            "pub(self) ",
            "pub ",
            "async ",
            "unsafe ",
            "default ",
            "extern \"C\" ",
        ],
    );
    stripped == "impl"
        || stripped.starts_with("impl ")
        || stripped.starts_with("impl<")
        || starts_with_any(
            stripped,
            &[
                "fn ",
                "struct ",
                "enum ",
                "trait ",
                "mod ",
                "const ",
                "static ",
                "type ",
                "union ",
                "macro_rules!",
            ],
        )
}

fn generic_decl(line: &str) -> bool {
    let first = line.as_bytes()[0];
    if !(first.is_ascii_alphabetic() || first == b'_') {
        return false;
    }
    line.contains('(') || line.contains('=') || line.ends_with(':') || line.ends_with('{')
}

fn starts_with_any(line: &str, prefixes: &[&str]) -> bool {
    prefixes.iter().any(|p| line.starts_with(p))
}

fn strip_prefixes<'a>(mut line: &'a str, prefixes: &[&str]) -> &'a str {
    loop {
        let mut changed = false;
        for p in prefixes {
            if let Some(rest) = line.strip_prefix(p) {
                line = rest;
                changed = true;
            }
        }
        if !changed {
            return line;
        }
    }
}

// ----------
// Rendering
// ----------

fn render(
    _task: &str,
    root: &Path,
    paths: &[PathBuf],
    located: &[(String, Vec<String>)],
) -> String {
    let mut lines = vec![HEADING.to_string(), String::new()];
    if !paths.is_empty() {
        lines.push("Paths named in the task:".to_string());
        for path in paths {
            lines.push(format!("- {}", path.display()));
            let outline = outline_file(&root.join(path));
            if outline.is_empty() {
                lines.push("  (no top-level items found)".to_string());
            } else {
                for item in outline {
                    lines.push(format!("  {item}"));
                }
            }
        }
    }
    if !located.is_empty() {
        lines.push(String::new());
        lines.push("Identifiers located:".to_string());
        for (ident, hits) in located {
            lines.push(format!("- `{ident}`"));
            for hit in hits {
                lines.push(format!("  {hit}"));
            }
        }
    }
    cap_lines(&lines)
}

/// Join `lines` under [`PACK_CAP_BYTES`], dropping from the end (the
/// identifier hits are least relevant and come last).
fn cap_lines(lines: &[String]) -> String {
    let mut out = String::new();
    let mut shown = 0usize;
    for line in lines {
        if out.len() + line.len() >= PACK_CAP_BYTES {
            break;
        }
        out.push_str(line);
        out.push('\n');
        shown += 1;
    }
    if shown < lines.len() {
        let note = "(context pack truncated)";
        if out.len() + note.len() < PACK_CAP_BYTES {
            out.push_str(note);
            out.push('\n');
        }
    }
    out
}
