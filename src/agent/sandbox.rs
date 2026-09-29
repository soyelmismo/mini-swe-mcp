use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// Byte budget above which captured command output is truncated.
pub const TRUNCATE_LIMIT: usize = 16_384;
/// Bytes retained from the *start* of over-budget output (floored to a
/// character boundary).
pub const TRUNCATE_HEAD: usize = 12_288;
/// Bytes retained from the *end* of over-budget output (ceiled to a character
/// boundary).
pub const TRUNCATE_TAIL: usize = 4_096;

/// Literal text of the marker inserted in place of the discarded middle,
/// excluding the decimal byte count that is rendered between the two halves.
const TRUNCATE_MARKER: &str = "\n... [Truncated ";
/// Literal text of the second half of the marker, after the byte count.
const TRUNCATE_MARKER_SUFFIX: &str = " bytes] ...\n";
/// Upper bound on the decimal digits of a `usize` (2^64 - 1 has 20 digits).
/// Used to size the result buffer up front so the marker needs no allocation.
const USIZE_MAX_DIGITS: usize = 20;

/// Bound command output to [`TRUNCATE_LIMIT`] bytes, keeping the head and the
/// tail of the text and reporting how many bytes were discarded.
///
/// Both cut points are snapped to UTF-8 character boundaries (`floor` for the
/// head, `ceil` for the tail), so no character is ever split and
/// `head + dropped + tail == input.len()` holds exactly.
///
/// The result is assembled **once** into an exactly-sized `String`: the marker
/// is pushed directly (the byte count is rendered into a stack buffer) and the
/// 4 KiB tail is never copied through an intermediate allocation.
pub fn truncate_output(combined: &str) -> String {
    let total = combined.len();
    if total <= TRUNCATE_LIMIT {
        return combined.to_string();
    }

    let head_end = combined.floor_char_boundary(TRUNCATE_HEAD);
    let tail_start = combined.ceil_char_boundary(total - TRUNCATE_TAIL);
    let dropped = total - (head_end + (total - tail_start));

    let mut out = String::with_capacity(
        head_end + TRUNCATE_MARKER.len() + USIZE_MAX_DIGITS + TRUNCATE_MARKER_SUFFIX.len()
            + (total - tail_start),
    );
    let mut digits = [0u8; USIZE_MAX_DIGITS];
    let count = render_decimal(&mut digits, dropped);

    out.push_str(&combined[..head_end]);
    out.push_str(TRUNCATE_MARKER);
    out.push_str(count);
    out.push_str(TRUNCATE_MARKER_SUFFIX);
    out.push_str(&combined[tail_start..]);
    debug_assert!(
        out.capacity() >= out.len(),
        "result buffer must be sized up front"
    );
    out
}

/// Render `value` as decimal ASCII digits into `buf` and return the used
/// prefix as a `&str`.
fn render_decimal(buf: &mut [u8; USIZE_MAX_DIGITS], mut value: usize) -> &str {
    debug_assert!(value > 0, "the marker is only emitted with a dropped region");
    let mut idx = buf.len();
    while value > 0 {
        idx -= 1;
        buf[idx] = b'0' + u8::try_from(value % 10).expect("remainder is a single digit");
        value /= 10;
    }
    std::str::from_utf8(&buf[idx..]).expect("ASCII digits are valid UTF-8")
}

/// Validate that a subagent command does not attempt to escape the worktree
/// or trigger runaway recursive scans of root or home filesystems.
pub fn validate_bash_command(command: &str) -> Result<(), &'static str> {
    let trimmed = command.trim();

    // 1. Block recursive searches starting at root, home, or system directories
    const FORBIDDEN_SEARCHES: &[&str] = &[
        "find / ",
        "find / -",
        "find /\"",
        "find /'",
        "find ~",
        "find /home",
        "find /root",
        "find /etc",
        "find /var",
        "find /usr",
        "grep -rn / ",
        "grep -r / ",
    ];

    for token in FORBIDDEN_SEARCHES {
        if trimmed.contains(token) {
            return Err(
                "Scanning root '/' or system directories is forbidden. Confine searches to the current repository ($PWD).",
            );
        }
    }

    // 2. Block escaping to parent or root directory via cd
    const FORBIDDEN_CDS: &[&str] = &[
        "cd / ", "cd /;", "cd /&&", "cd /||", "cd /home", "cd ~", "cd $HOME", "cd /root",
    ];

    for token in FORBIDDEN_CDS {
        if trimmed.contains(token) || trimmed.ends_with("cd /") {
            return Err(
                "Navigating outside the repository with 'cd' is forbidden. All files are in $PWD.",
            );
        }
    }

    Ok(())
}

/// Distinguish CPU-heavy commands (compilations, test runners) from
/// lightweight exploration commands (git status, cat, ls, grep, etc.).
pub fn is_heavy_command(command: &str) -> bool {
    let lower = command.to_lowercase();
    if lower.starts_with("cargo") || lower.contains("cargo ") || lower.contains("cargo\t") {
        return true;
    }
    if lower == "make"
        || lower.starts_with("make ")
        || lower.contains(" make ")
        || lower.contains(" make\t")
    {
        return true;
    }
    const HEAVY_PATTERNS: &[&str] = &[
        "rustc", "pytest", "unittest", "cmake", "ninja", "gcc", "g++", "clang", "npm ", "yarn ",
        "pnpm ", "mvn ", "gradle", "go test", "go build",
    ];
    HEAVY_PATTERNS.iter().any(|pattern| lower.contains(pattern))
}

/// Check if the bubblewrap (`bwrap`) sandbox utility is available on this system.
pub fn has_bwrap() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        std::process::Command::new("bwrap")
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    })
}

/// If a worktree's `.git` is a gitdir reference pointing to a parent git directory,
/// locate both the common `.git` directory and the specific worktree gitdir.
pub fn find_git_dirs(worktree_dir: &Path) -> Option<(PathBuf, Option<PathBuf>)> {
    let dot_git = worktree_dir.join(".git");
    if dot_git.is_file()
        && let Ok(content) = std::fs::read_to_string(&dot_git)
        && let Some(gitdir_line) = content.lines().find(|l| l.starts_with("gitdir: "))
    {
        let raw_path = gitdir_line.trim_start_matches("gitdir: ").trim();
        let gitdir_path = PathBuf::from(raw_path);
        for ancestor in gitdir_path.ancestors() {
            if ancestor.file_name().and_then(|n| n.to_str()) == Some(".git") {
                return Some((ancestor.to_path_buf(), Some(gitdir_path)));
            }
        }
    }
    None
}

/// Backwards-compatible helper returning only the common `.git` root.
pub fn find_git_common_dir(worktree_dir: &Path) -> Option<PathBuf> {
    find_git_dirs(worktree_dir).map(|(common, _)| common)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truncate_output_utf8_boundary() {
        let mut s = "a".repeat(TRUNCATE_HEAD - 1);
        s.push('€'); // bytes 12287..12290
        s.push_str(&"b".repeat(9000));
        let truncated = truncate_output(&s);
        assert!(truncated.contains("... [Truncated"));
    }

    #[test]
    fn test_truncate_output_constants_match_the_budget() {
        const { assert!(TRUNCATE_HEAD + TRUNCATE_TAIL == TRUNCATE_LIMIT) };
        const { assert!(TRUNCATE_TAIL < TRUNCATE_LIMIT && TRUNCATE_HEAD < TRUNCATE_LIMIT) };
    }

    #[test]
    fn test_validate_bash_command_blocks_escapes() {
        // Blocked: root find
        assert!(validate_bash_command("find / -name 'foo'").is_err());
        assert!(validate_bash_command("find ~ -name 'foo'").is_err());
        assert!(validate_bash_command("find /home -name 'foo'").is_err());

        // Blocked: cd to root or home
        assert!(validate_bash_command("cd / && ls").is_err());
        assert!(validate_bash_command("cd /home && ls").is_err());
        assert!(validate_bash_command("cd ~").is_err());
        assert!(validate_bash_command("cd /").is_err());

        // Allowed: within worktree
        assert!(validate_bash_command("find . -name 'foo'").is_ok());
        assert!(validate_bash_command("find src -type f").is_ok());
        assert!(validate_bash_command("cd src && cargo test").is_ok());
        assert!(validate_bash_command("grep -rn 'WorkerState' src/").is_ok());
    }

    #[test]
    fn test_is_heavy_command() {
        assert!(is_heavy_command("cargo build"));
        assert!(is_heavy_command("cargo test --all"));
        assert!(is_heavy_command("cargo"));
        assert!(is_heavy_command("pytest tests/"));
        assert!(is_heavy_command("make -j4"));
        assert!(is_heavy_command("make"));
        assert!(is_heavy_command("gcc -O3 main.c"));

        assert!(!is_heavy_command("git status"));
        assert!(!is_heavy_command("git diff HEAD"));
        assert!(!is_heavy_command("ls -la"));
        assert!(!is_heavy_command("cat src/agent.rs"));
        assert!(!is_heavy_command("find . -name '*.rs'"));
        assert!(!is_heavy_command(
            "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
        ));
    }

    #[test]
    fn test_has_bwrap_returns_boolean() {
        let _ = has_bwrap();
    }

    #[test]
    fn test_find_git_common_dir_on_regular_dir() {
        let tmp = std::env::temp_dir();
        assert_eq!(find_git_common_dir(&tmp), None);
    }
}
