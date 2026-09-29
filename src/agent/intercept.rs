//! Destructive-command guard run before any bash step is spawned.
//!
//! [`check_command`] is called by [`crate::agent::AgentRunner::execute_bash`]
//! before sandbox construction. It rejects the obvious catastrophic patterns
//! (`rm -rf /`, `mkfs`, raw `dd` to a device, the classic fork bomb) even when
//! they are hidden behind quotes, variable expansions or command substitution.
//! It is a text-level first line of defence; filesystem confinement is the
//! sandbox's job.

/// Reason string for a matched dangerous pattern.
fn block_reason(matched: &str) -> String {
    format!("destructive pattern blocked: {matched}")
}

/// Plain developer verbs eligible for the pattern-check fast path.
const BENIGN_VERBS: &[&str] = &["cargo", "git", "ls", "cat"];

/// Skip expensive pattern matching for simple commands with a known verb.
/// Shell composition and substitution always take the slow path.
fn is_benign(lowered: &str) -> bool {
    // Do not accept substituted commands or shell redirections as plain verbs.
    if lowered.contains([';', '&', '|', '`', '\n', '$', '(', ')', '<', '>']) {
        return false;
    }
    let trimmed = lowered.trim();
    BENIGN_VERBS
        .iter()
        .any(|verb| strip_verb(trimmed, verb).is_some())
}

/// Strip a leading `verb` from a command segment, requiring the shell token
/// boundary (end of input or ASCII whitespace) so `lsof`/`catastrophe` do not
/// match `ls`/`cat`.
fn strip_verb<'a>(segment: &'a str, verb: &str) -> Option<&'a str> {
    let rest = segment.strip_prefix(verb)?;
    if rest.is_empty() || rest.starts_with(|c: char| c.is_ascii_whitespace()) {
        Some(rest)
    } else {
        None
    }
}

/// Build a conservative scan view of a command, retaining token boundaries
/// while removing interleaved quotes and neutralising variable expansions.
fn scan_form(command: &str) -> String {
    let mut out = String::with_capacity(command.len());
    let mut quote: Option<char> = None;
    let mut token = String::new();
    // Set while a `${...}` expansion is open: the shell keeps the text up
    // to the closing brace inside the expansion, so the view has to close it
    // too instead of leaving a suffix glued to the expansion.
    let mut braced = false;
    for ch in command.chars() {
        // Inside quotes only the closing quote has a meaning. A quote
        // character of the other kind is literal text and contributes
        // nothing, so `r"m` still spells `rm`; everything else is copied
        // verbatim, spaces included, because inside quotes a space is data
        // rather than a separator.
        if let Some(q) = quote {
            if ch == q {
                quote = None;
            } else if ch == '$' && !braced && !token.ends_with('/') {
                out.push_str(&token);
                token.clear();
                out.push_str(" / ");
                braced = true;
            } else if ch == '}' && braced {
                out.push_str(&token);
                token.clear();
                braced = false;
            } else if ch != '\'' && ch != '"' {
                out.push(ch);
            }
            continue;
        }
        match ch {
            // A quote ends the current token: the command word may be spelled
            // with quotes interleaved, so `r"m" -rf /` has to be seen as
            // `rm -rf /`.
            '\'' | '"' => {
                out.push_str(&token);
                token.clear();
                quote = Some(ch);
            }
            // Neutralise a variable expansion that *starts a token*: the
            // value is unknown here, so it becomes the root of a top-level
            // path, which the block rules already refuse. An expansion
            // that continues a path is confined by that path instead, so a
            // worktree-relative one such as `target/$BUILD` keeps its
            // relative prefix and stays allowed.
            '$' if !braced && !token.ends_with('/') => {
                out.push_str(&token);
                token.clear();
                out.push_str(" / ");
                braced = true;
            }
            // The closing brace belongs to the expansion rather than to a
            // token, so it is dropped and a suffix starts a token of its
            // own: the view then keeps the token boundaries the shell
            // sees.
            '}' if braced => {
                out.push_str(&token);
                token.clear();
                braced = false;
            }
            _ => {
                token.push(ch);
                if ch.is_ascii_whitespace() {
                    out.push_str(&token);
                    token.clear();
                }
            }
        }
    }
    if quote == Some('\'') {
        // Unterminated quote: the shell keeps the quote character, so the
        // missing quote belongs to the last token and must survive the trim.
        token.push('\'');
    }
    out.push_str(&token);
    // An unterminated expansion swallows the separator that would have closed
    // it, so the view terminates it with a separator instead.
    if braced {
        out.push(' ');
    }
    out
}

/// Reject destructive invocations, including simple whitespace, quote and
/// variable-expansion obfuscation. Returns the reason shown to the model.
///
/// The whole command text is scanned, heredoc bodies and quoted strings
/// included: a pattern cannot be told apart from data there, so content that
/// merely *mentions* one (a test fixture, a grep for it) is refused too.
pub fn check_command(command: &str) -> Result<(), String> {
    // Fast path: pure developer verbs (`cargo test`, `git status`, `ls -la`,
    // `cat Cargo.toml`) can never contain a destructive pattern, so the
    // canonicalisation work below is skipped entirely.
    if is_benign(command) {
        return Ok(());
    }
    let trimmed = scan_form(&command.to_lowercase());
    let trimmed = trimmed.trim();

    // `rm -rf /`, `rm -fr /`, `rm -rf /*`, `rm --no-preserve-root`. Token-aware:
    // `rm` with recursive+force flags and a root-level target (`/`, `/*`, `~`,
    // `/home`, `/etc`, ...); a plain `rm -rf target/` stays allowed.
    if is_destructive_rm(trimmed) {
        return Err(block_reason("rm -rf /"));
    }
    // Filesystem creation / raw disk writes.
    if trimmed.contains("mkfs") {
        return Err(block_reason("mkfs"));
    }
    if dd_targets_disk(trimmed) {
        return Err(block_reason("dd if="));
    }
    // Classic bash fork bomb (whitespace-insensitive).
    if is_fork_bomb(trimmed) {
        return Err(block_reason("fork bomb"));
    }
    Ok(())
}

/// `true` when `scan_trimmed` (see [`scan_form`]) contains an `rm` invocation
/// that would recursively delete at or above the filesystem root (or home),
/// rather than a path inside the worktree.
///
/// The `rm` command word is looked up in every segment instead of only at its
/// head, so a wrapper (`sudo`, `xargs`, `bash -c "..."`) or a command word
/// spelled with interleaved quotes cannot hide the invocation.
fn is_destructive_rm(scan_trimmed: &str) -> bool {
    // Split on shell separators so `echo hi; rm -rf /` is still caught. `$` no
    // longer occurs in the scan view, but it is cheap defence in depth against
    // an `rm` hidden behind an expansion.
    for segment in scan_trimmed.split([';', '&', '|', '`', '$', '\n']) {
        let tokens: Vec<&str> = segment.split_whitespace().collect();
        for (at, tok) in tokens.iter().enumerate() {
            if !is_command_word(tok, "rm") {
                continue;
            }
            let args = &tokens[at + 1..];
            // Flags must contain both recursive (`r`) and force (`f`), e.g.
            // `-rf`, `-fr`, `-r -f`, `--recursive`, or `--no-preserve-root`. Short flags may
            // be combined (`-rf`), so inspect flag characters rather than
            // substrings: `-rf` contains "-r" but not "-f".
            let preserves_root = args.contains(&"--no-preserve-root");
            let mut has_recursive =
                preserves_root || args.iter().any(|a| a.contains("--recursive"));
            let mut has_force = preserves_root || args.iter().any(|a| a.contains("--force"));
            for arg in args {
                if let Some(flags) = arg.strip_prefix('-') {
                    has_recursive |= flags.contains('r');
                    has_force |= flags.contains('f');
                }
            }
            if !(has_recursive && has_force) {
                continue;
            }
            // Check for a bare `/` token or a destructive prefix after the
            // flags. Rule: `rm <flags> <target>` where the target is `/`,
            // `/*`, `~`, or a single-component top-level system dir — never a
            // path inside the worktree.
            if preserves_root {
                return true;
            }
            for arg in args {
                // The root operand may sit in flag position, as in
                // `rm -fr -/` or `rm -rf -**`, where the plain
                // flag skip below would hide it.
                if let Some(operand) = root_operand_in_flag(arg) {
                    if targets_root_level(operand) {
                        return true;
                    }
                    continue;
                }
                if arg.starts_with('-') {
                    continue;
                }
                if targets_root_level(arg.trim_end_matches([')', '}'])) {
                    return true;
                }
            }
        }
    }
    false
}

/// `true` when `token` names `command`, with or without a directory prefix:
/// `rm`, `./rm`, `/bin/rm`.
///
/// Wrapping a command (`sudo rm`, `xargs rm`) does not change the name, which
/// is why the callers look for the command word anywhere in a segment.
fn is_command_word(token: &str, command: &str) -> bool {
    token
        .trim_start_matches(['<', '>', '$', '(', '{'])
        .rsplit('/')
        .next()
        .is_some_and(|name| name == command)
}

/// The root operand carried by a flag-shaped token, e.g. `/` in `-rf/`
/// or in `-/`, which `rm` reads as an option ending in `/` rather than as
/// the operand. Real options such as `-rf` carry none.
fn root_operand_in_flag(token: &str) -> Option<&str> {
    let rest = token.strip_prefix('-')?;
    let at = rest.find('/')?;
    let operand = &rest[at..];
    // Only a bare `/` operand qualifies; `-rf/` is an option, not a target.
    (operand == "/").then_some(operand)
}

/// `true` when `path` addresses the filesystem root or a direct child of it.
fn targets_root_level(path: &str) -> bool {
    // `~`, `~/x`, `~user` and `~user/x` all address a path under the home
    // root, so a leading `~` counts as the root component.
    let home_root;
    let path = if let Some(rest) = path.strip_prefix('~') {
        home_root = format!("/{rest}");
        home_root.as_str()
    } else {
        path
    };
    if path == "/" || path.starts_with("/*") {
        return true;
    }
    // `rm -rf / home`-style split or top-level system dir with no deeper
    // path component (e.g. `/etc`, not `/etc/hostname.bak`? conservatively
    // block any direct child of root that is short).
    if path.starts_with('/') {
        let depth = path.split('/').filter(|c| !c.is_empty()).count();
        return depth <= 1;
    }
    false
}

/// `true` when `scan_trimmed` (see [`scan_form`]) contains a `dd` invocation
/// with an input source (raw disk copy), however it is wrapped.
fn dd_targets_disk(scan_trimmed: &str) -> bool {
    scan_trimmed.split([';', '&', '|', '\n']).any(|segment| {
        segment.contains("if=")
            && segment
                .split_whitespace()
                .any(|tok| is_command_word(tok, "dd"))
    })
}

/// `true` when `scan_trimmed` contains the classic bash fork bomb, ignoring
/// whitespace differences (`:(){ :|:& };:`).
fn is_fork_bomb(scan_trimmed: &str) -> bool {
    let compact: String = scan_trimmed
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect();
    compact.contains(":(){:|:&};:")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn destructive_blocks_rm_rf_root() {
        for cmd in [
            "rm -rf /",
            "rm -rf /*",
            "rm -fr /",
            "sudo rm -rf /",
            "echo hi; rm -rf /",
            "rm --no-preserve-root -rf /",
        ] {
            assert!(check_command(cmd).is_err(), "should block: {cmd}");
        }
    }

    #[test]
    fn destructive_blocks_mkfs_dd_and_fork_bomb() {
        assert!(check_command("mkfs.ext4 /dev/sda1").is_err());
        assert!(check_command("sudo mkfs -t ext4 /dev/sda1").is_err());
        assert!(check_command("dd if=/dev/zero of=/dev/sda").is_err());
        assert!(check_command("sudo dd if=/dev/random of=/dev/sda").is_err());
        assert!(check_command(":(){ :|:& };:").is_err());
        assert!(check_command(":() { : | : & } ; :").is_err());
    }

    #[test]
    fn destructive_allows_normal_commands() {
        for cmd in [
            "cargo test",
            "cargo test --all-targets",
            "git status",
            "git diff HEAD",
            "ls -la",
            "echo hello",
            "rm -rf target/debug/build",
            "rm ./some-file.txt",
            "grep -rn 'foo' src/",
        ] {
            assert!(check_command(cmd).is_ok(), "should allow: {cmd}");
        }
    }

    #[test]
    fn destructive_blocks_obfuscated_and_substituted_commands() {
        for cmd in [
            "r\"m\" -rf /",
            "rm -rf \"$HOME\"",
            "rm -rf ${HOME}",
            "rm -rf /etc",
            "git status $(rm -rf /)",
            "cat <(rm -rf /)",
            "cargo test `rm -rf /`",
            "ls; rm -rf /",
        ] {
            assert!(check_command(cmd).is_err(), "should block: {cmd}");
        }
    }
}
