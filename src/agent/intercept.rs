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
/// Only what would run is judged: command words (see [`invocations`]) and
/// heredoc bodies a shell executes. Data that merely *mentions* a pattern - a
/// source file or fixture written through a heredoc, a `grep` or `echo`
/// argument - is allowed, so work on this very guard is not blocked by it.
pub fn check_command(command: &str) -> Result<(), String> {
    // Fast path: pure developer verbs (`cargo test`, `git status`, `ls -la`,
    // `cat Cargo.toml`) can never contain a destructive pattern, so the
    // canonicalisation work below is skipped entirely.
    if is_benign(command) {
        return Ok(());
    }
    let executable = strip_data_heredocs(command);
    let trimmed = scan_form(&executable.to_lowercase());
    let trimmed = trimmed.trim();
    let commands = invocations(trimmed);

    // `rm -rf /`, `rm -fr /`, `rm -rf /*`, `rm --no-preserve-root`. Token-aware:
    // `rm` with recursive+force flags and a root-level target (`/`, `/*`, `~`,
    // `/home`, `/etc`, ...); a plain `rm -rf target/` stays allowed.
    if commands.iter().any(|(word, args)| command_name(word) == "rm" && is_destructive_rm(args)) {
        return Err(block_reason("rm -rf /"));
    }
    // Filesystem creation / raw disk writes.
    if commands.iter().any(|(word, _)| {
        let name = command_name(word);
        name == "mkfs" || name.starts_with("mkfs.")
    }) {
        return Err(block_reason("mkfs"));
    }
    if commands
        .iter()
        .any(|(word, args)| command_name(word) == "dd" && args.iter().any(|a| a.contains("if=")))
    {
        return Err(block_reason("dd if="));
    }
    // Classic bash fork bomb (whitespace-insensitive).
    if is_fork_bomb(trimmed) {
        return Err(block_reason("fork bomb"));
    }
    Ok(())
}

/// Commands whose heredoc body is executed as shell code.
const SHELLS: &[&str] = &["bash", "sh", "zsh", "dash", "ksh", "eval", "source"];

/// The command with every heredoc body removed unless a shell executes it.
///
/// `cat > src/x.rs <<'EOF' ... EOF` writes *data*: a test fixture or a source
/// file that merely mentions `rm -rf /` must not be refused. A body stays in
/// the scanned text only when a shell word appears on the heredoc's line
/// (`bash <<EOF`, `cat <<EOF | sh`), because then it runs. Programs such as
/// `python3 - <<EOF` are out of reach of a shell-text guard either way; the
/// sandbox is what confines them.
pub(crate) fn strip_data_heredocs(command: &str) -> String {
    let mut out = String::with_capacity(command.len());
    let mut lines = command.lines();
    while let Some(line) = lines.next() {
        out.push_str(line);
        out.push('\n');
        let Some(delimiter) = heredoc_delimiter(line) else {
            continue;
        };
        let executes = line
            .split(|c: char| c.is_whitespace() || matches!(c, '|' | ';' | '&' | '('))
            .any(|token| SHELLS.contains(&command_name(token)));
        for body in lines.by_ref() {
            let done = body.trim() == delimiter;
            if executes || done {
                out.push_str(body);
                out.push('\n');
            }
            if done {
                break;
            }
        }
    }
    out
}

/// The delimiter word of a heredoc (`<<EOF`, `<<-'EOF'`, `<< "END"`) opened on
/// `line`, ignoring here-strings (`<<<`).
fn heredoc_delimiter(line: &str) -> Option<String> {
    let mut from = 0;
    while let Some(pos) = line[from..].find("<<") {
        let at = from + pos + 2;
        if line[at..].starts_with('<') {
            from = at + 1;
            continue;
        }
        let delimiter: String = line[at..]
            .trim_start_matches('-')
            .trim_start()
            .trim_start_matches(['\'', '"'])
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        if !delimiter.is_empty() {
            return Some(delimiter);
        }
        from = at;
    }
    None
}

/// Commands that run another command, each with its flags that take a
/// separate argument (skipped together with that argument).
const WRAPPERS: &[(&str, &[&str])] = &[
    ("sudo", &["-u", "-g", "-C", "-D", "-h", "-p", "-U", "-r", "-t"]),
    ("doas", &["-u", "-C"]),
    ("env", &["-u", "-C", "-S"]),
    ("nice", &["-n"]),
    ("ionice", &["-c", "-n", "-p"]),
    ("timeout", &["-s", "-k"]),
    ("xargs", &["-I", "-L", "-n", "-P", "-d", "-a", "-E", "-s"]),
    ("stdbuf", &["-i", "-o", "-e"]),
    ("taskset", &["-c"]),
    ("chrt", &[]),
    ("nohup", &[]),
    ("time", &[]),
    ("exec", &[]),
    ("command", &[]),
    ("builtin", &[]),
    ("setsid", &[]),
];

/// The bare program name of a command token: `/bin/rm`, `./rm`, `(rm` -> `rm`.
fn command_name(token: &str) -> &str {
    token
        .trim_start_matches(['<', '>', '$', '(', '{'])
        .rsplit('/')
        .next()
        .unwrap_or(token)
}

/// `NAME=value` before a command word (an environment assignment).
fn is_assignment(token: &str) -> bool {
    token.split_once('=').is_some_and(|(name, _)| {
        !name.is_empty() && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
    })
}

/// A wrapper's positional number: `timeout 30`, `timeout 2.5m`, `taskset 0x3`.
fn is_numeric_arg(token: &str) -> bool {
    let digits = token.trim_end_matches(['s', 'm', 'h', 'd']);
    digits.parse::<f64>().is_ok() || token.starts_with("0x")
}

/// Every simple command of the scan view as `(command word, arguments)`.
///
/// The command word is the first token of each command after separators,
/// subshells and substitutions, skipping assignments, wrappers (`sudo`,
/// `xargs`, `nice -n 10`, ...) and `sh -c`. Arguments are data: `grep -rn
/// "rm -rf /" src/` mentions a pattern without running it.
///
/// Text piped into a shell that reads its program from stdin (`echo ... | sh`)
/// *is* code, so in that case every token is considered a command word, which
/// is the conservative reading.
fn invocations(scan_trimmed: &str) -> Vec<(&str, Vec<&str>)> {
    let segments: Vec<Vec<&str>> = scan_trimmed
        .split([';', '&', '|', '`', '\n', '(', ')'])
        .map(|segment| segment.split_whitespace().collect())
        .collect();
    let piped_into_shell = segments.iter().any(|tokens| {
        tokens.first().is_some_and(|first| SHELLS.contains(&command_name(first)))
            && tokens[1..].iter().all(|t| t.starts_with('-') && !t.contains('c'))
    });

    let mut found = Vec::new();
    for tokens in &segments {
        if piped_into_shell {
            for (at, tok) in tokens.iter().enumerate() {
                found.push((*tok, tokens[at + 1..].to_vec()));
            }
            continue;
        }
        let mut at = 0;
        let mut wrapper_flags: Option<&[&str]> = None;
        while let Some(&tok) = tokens.get(at) {
            if let Some(flags) = wrapper_flags {
                if flags.contains(&tok) {
                    at += 2;
                    continue;
                }
                if tok.starts_with('-') || is_numeric_arg(tok) {
                    at += 1;
                    continue;
                }
            }
            if is_assignment(tok) {
                at += 1;
                continue;
            }
            let name = command_name(tok);
            if let Some((_, flags)) = WRAPPERS.iter().find(|(wrapper, _)| *wrapper == name) {
                wrapper_flags = Some(flags);
                at += 1;
                continue;
            }
            // `sh -c "<cmd>"`: the quoted program is the command.
            if SHELLS.contains(&name)
                && tokens[at + 1..].first().is_some_and(|t| t.starts_with('-') && t.contains('c'))
            {
                wrapper_flags = Some(&[]);
                at += 1;
                continue;
            }
            found.push((tok, tokens[at + 1..].to_vec()));
            break;
        }
    }
    found
}

/// `true` when `args` (an `rm` invocation's arguments) would recursively delete
/// at or above the filesystem root (or home) rather than inside the worktree.
fn is_destructive_rm(args: &[&str]) -> bool {
    // Flags must contain both recursive (`r`) and force (`f`), e.g. `-rf`,
    // `-fr`, `-r -f`, `--recursive`, or `--no-preserve-root`. Short flags may be
    // combined (`-rf`), so inspect flag characters rather than substrings.
    let preserves_root = args.contains(&"--no-preserve-root");
    let mut has_recursive = preserves_root || args.iter().any(|a| a.contains("--recursive"));
    let mut has_force = preserves_root || args.iter().any(|a| a.contains("--force"));
    for arg in args {
        if let Some(flags) = arg.strip_prefix('-') {
            has_recursive |= flags.contains('r');
            has_force |= flags.contains('f');
        }
    }
    if !(has_recursive && has_force) {
        return false;
    }
    if preserves_root {
        return true;
    }
    args.iter().any(|arg| {
        // The root operand may sit in flag position (`rm -fr -/`), where the
        // plain flag skip below would hide it.
        if let Some(operand) = root_operand_in_flag(arg) {
            return targets_root_level(operand);
        }
        !arg.starts_with('-') && targets_root_level(arg.trim_end_matches([')', '}']))
    })
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

    /// Content written through a heredoc is data, not a command: workers must
    /// be able to write source files and fixtures that mention the patterns.
    /// Pieces are joined at runtime so this test file itself stays editable.
    #[test]
    fn heredoc_data_mentioning_patterns_is_allowed() {
        let rm_root = ["rm", "-rf", "/"].join(" ");
        let mkfs = ["mk", "fs"].concat();
        for cmd in [
            format!("cat > src/x.rs <<'EOF'\nassert!(blocks(\"{rm_root}\"));\nlet s = \"{mkfs}.ext4\";\nEOF"),
            format!("python3 - <<'PY'\nfixture = '{rm_root}'\nPY"),
            format!("cat > t.sh <<-EOF\n\t{mkfs} /dev/sda1\n\tEOF\ncargo test"),
            format!("grep -rn {mkfs} src/"),
            format!("git log --grep '{mkfs}'"),
        ] {
            assert!(check_command(&cmd).is_ok(), "should allow: {cmd}");
        }
    }

    /// The worktree guardrail shares the executable view: a file that
    /// documents a forbidden search can be written.
    #[test]
    fn guardrail_ignores_heredoc_data() {
        let search = ["find", "/home", "-name", "x"].join(" ");
        let cmd = format!("cat > notes.md <<'EOF'\nnever run: {search}\nEOF");
        let view = strip_data_heredocs(&cmd);
        assert!(crate::agent::validate_bash_command(&view).is_ok(), "{view:?}");
        assert!(crate::agent::validate_bash_command(&search).is_err());
    }

    /// A heredoc a shell executes is still scanned, and so is whatever runs
    /// after the heredoc ends.
    #[test]
    fn heredoc_fed_to_a_shell_or_followed_by_a_command_is_scanned() {
        let rm_root = ["rm", "-rf", "/"].join(" ");
        let mkfs = ["mk", "fs"].concat();
        for cmd in [
            format!("bash <<'EOF'\n{rm_root}\nEOF"),
            format!("cat <<EOF | sh\n{rm_root}\nEOF"),
            format!("cat > f <<'EOF'\ndata\nEOF\n{mkfs}.ext4 /dev/sda1"),
            format!("/sbin/{mkfs} -t ext4 /dev/sda1"),
        ] {
            assert!(check_command(&cmd).is_err(), "should block: {cmd}");
        }
    }

    /// A pattern that is only an *argument* is not an invocation.
    #[test]
    fn mentions_in_arguments_are_allowed() {
        let rm_root = ["rm", "-rf", "/"].join(" ");
        for cmd in [
            format!("echo '{rm_root}'"),
            format!("grep -rn \"{rm_root}\" src/"),
            format!("git commit -m 'guard {rm_root}'"),
            "grep -rn 'dd if=' src/".to_string(),
        ] {
            assert!(check_command(&cmd).is_ok(), "should allow: {cmd}");
        }
    }

    /// Wrappers, assignments and `sh -c` do not hide the real command, and
    /// text piped into a shell is treated as code.
    #[test]
    fn wrapped_and_piped_invocations_are_blocked() {
        let rm_root = ["rm", "-rf", "/"].join(" ");
        for cmd in [
            format!("sudo -u root {rm_root}"),
            format!("timeout 30 {rm_root}"),
            format!("nice -n 10 {rm_root}"),
            format!("env FOO=1 {rm_root}"),
            format!("FOO=1 {rm_root}"),
            format!("find . | xargs -I{{}} {rm_root}"),
            format!("sh -c '{rm_root}'"),
            format!("echo '{rm_root}' | sh"),
            format!("echo '{rm_root}' | bash -s"),
        ] {
            assert!(check_command(&cmd).is_err(), "should block: {cmd}");
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
