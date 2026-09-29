//! Pre-execution command interceptor middleware.
//!
//! Interceptors run inside [`crate::agent::AgentRunner::execute_bash`] before
//! any sandbox construction or process spawn. Each inspects the requested
//! command and votes [`InterceptDecision::Allow`], [`InterceptDecision::Block`]
//! or [`InterceptDecision::Rewrite`]. The [`CommandPipeline`] chains them in
//! registration order: the first `Block` wins, while `Rewrite` updates the
//! effective command seen by the remaining interceptors and by the executor.

use anyhow::Result;
use std::time::Instant;

/// Verdict of a single [`CommandInterceptor`] for one command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InterceptDecision {
    /// Run the (possibly rewritten) command.
    Allow,
    /// Refuse to run; the inner string is the human-readable reason shown to
    /// the model.
    Block(String),
    /// Run `String` instead of the original command.
    Rewrite(String),
}

impl InterceptDecision {
    /// `true` for [`InterceptDecision::Allow`].
    #[must_use]
    pub fn is_allow(&self) -> bool {
        matches!(self, Self::Allow)
    }

    /// `true` for [`InterceptDecision::Block`].
    #[must_use]
    pub fn is_block(&self) -> bool {
        matches!(self, Self::Block(_))
    }
}

/// Pre-execution hook inspecting a bash command before it is spawned.
pub trait CommandInterceptor: Send + Sync {
    /// Inspect `command` and vote on whether it may run.
    fn intercept(&self, command: &str) -> Result<InterceptDecision>;

    /// Stable name used in telemetry / debug logs.
    fn name(&self) -> &'static str;
}

/// Blocks destructive invocations before they reach the executor, including
/// simple whitespace, quote and variable-expansion obfuscation. The fast path
/// only accepts plain developer commands without shell substitution or chaining.
#[derive(Debug, Default, Clone, Copy)]
pub struct DestructiveCommandInterceptor;

impl DestructiveCommandInterceptor {
    /// Reason string for a matched dangerous pattern.
    fn block_reason(matched: &str) -> String {
        format!("destructive pattern blocked: {matched}")
    }
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

impl CommandInterceptor for DestructiveCommandInterceptor {
    fn intercept(&self, command: &str) -> Result<InterceptDecision> {
        // Fast path: pure developer verbs (`cargo test`, `git status`, `ls -la`,
        // `cat Cargo.toml`) can never contain a destructive pattern, so the
        // canonicalisation work below is skipped entirely.
        if is_benign(command) {
            return Ok(InterceptDecision::Allow);
        }
        let trimmed = scan_form(&command.to_lowercase());
        let trimmed = trimmed.trim();

        // `rm -rf /`, `rm -fr /`, `rm -rf /*`, `rm --no-preserve-root`, bare `rm -rf /`.
        // Match token-aware: look for `rm` followed by recursive+force flags and
        // a root-level target (`/`, `/*`, `~`, `/home`, `/etc`, ...). A plain
        // `rm -rf target/` inside the worktree must stay allowed.
        if is_destructive_rm(trimmed) {
            return Ok(InterceptDecision::Block(Self::block_reason("rm -rf /")));
        }

        // Filesystem creation / raw disk writes.
        if trimmed.contains("mkfs") {
            return Ok(InterceptDecision::Block(Self::block_reason("mkfs")));
        }
        // `dd if=... of=/dev/...` — tokenise loosely: any `dd` with `if=`.
        if dd_targets_disk(trimmed) {
            return Ok(InterceptDecision::Block(Self::block_reason("dd if=")));
        }

        // Classic bash fork bomb `:(){ :|:& };:` (whitespace-insensitive).
        if is_fork_bomb(trimmed) {
            return Ok(InterceptDecision::Block(Self::block_reason("fork bomb")));
        }

        Ok(InterceptDecision::Allow)
    }

    fn name(&self) -> &'static str {
        "destructive"
    }
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

/// Passive interceptor that records when a command was first seen and how it
/// is classified (heavy vs. light). It never blocks or rewrites.
#[derive(Debug, Clone, Copy)]
pub struct TelemetryInterceptor;

impl Default for TelemetryInterceptor {
    fn default() -> Self {
        Self
    }
}

/// Lightweight classification mirroring [`super::sandbox::is_heavy_command`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandClass {
    /// Builds, test suites, package installs — long wall-clock budget.
    Heavy,
    /// Exploration commands (`git status`, `ls`, `cat`, ...).
    Light,
}

impl CommandClass {
    /// Classify `command` without side effects.
    #[must_use]
    pub fn classify(command: &str) -> Self {
        if super::sandbox::is_heavy_command(command) {
            Self::Heavy
        } else {
            Self::Light
        }
    }

    /// Stable string used in telemetry logs.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Heavy => "heavy",
            Self::Light => "light",
        }
    }
}

impl CommandInterceptor for TelemetryInterceptor {
    fn intercept(&self, command: &str) -> Result<InterceptDecision> {
        let started = Instant::now();
        let class = CommandClass::classify(command);
        tracing::debug!(
            interceptor = self.name(),
            classification = class.as_str(),
            command_len = command.len(),
            ?started,
            "intercepting command"
        );
        Ok(InterceptDecision::Allow)
    }

    fn name(&self) -> &'static str {
        "telemetry"
    }
}

/// Ordered chain of [`CommandInterceptor`]s executed before a command runs.
///
/// Semantics: interceptors see the *effective* command in registration order.
/// A `Rewrite` replaces the effective command for the remaining interceptors;
/// the first `Block` short-circuits the chain. [`CommandPipeline::process`]
/// returns the final verdict: `Block` if any interceptor blocked, `Rewrite`
/// with the last rewritten command if any rewrote, else `Allow`.
pub struct CommandPipeline {
    interceptors: Vec<Box<dyn CommandInterceptor>>,
}

impl Default for CommandPipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl CommandPipeline {
    /// Empty pipeline (allows everything).
    #[must_use]
    pub fn new() -> Self {
        Self {
            interceptors: Vec::new(),
        }
    }

    /// Production pipeline: destructive guard first, telemetry second.
    #[must_use]
    pub fn default_pipeline() -> Self {
        Self::new()
            .with(DestructiveCommandInterceptor)
            .with(TelemetryInterceptor)
    }

    /// Append an interceptor and return the pipeline (builder style).
    #[must_use]
    pub fn with(mut self, interceptor: impl CommandInterceptor + 'static) -> Self {
        self.interceptors.push(Box::new(interceptor));
        self
    }

    /// Push an interceptor onto an existing pipeline.
    pub fn add(&mut self, interceptor: impl CommandInterceptor + 'static) {
        self.interceptors.push(Box::new(interceptor));
    }

    /// Number of registered interceptors.
    #[must_use]
    pub fn len(&self) -> usize {
        self.interceptors.len()
    }

    /// `true` when no interceptors are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.interceptors.is_empty()
    }

    /// Run the chain over `command`.
    ///
    /// Returns `Ok(Block)` on the first block, otherwise `Ok(Rewrite)` if at
    /// least one interceptor rewrote, else `Ok(Allow)`.
    pub fn process(&self, command: &str) -> Result<InterceptDecision> {
        let mut effective = command.to_string();
        let mut rewritten = false;
        for interceptor in &self.interceptors {
            match interceptor.intercept(&effective)? {
                InterceptDecision::Allow => {}
                InterceptDecision::Block(reason) => {
                    return Ok(InterceptDecision::Block(reason));
                }
                InterceptDecision::Rewrite(next) => {
                    effective = next;
                    rewritten = true;
                }
            }
        }
        if rewritten {
            Ok(InterceptDecision::Rewrite(effective))
        } else {
            Ok(InterceptDecision::Allow)
        }
    }

    /// Run the chain and resolve the command the executor should use.
    ///
    /// Returns `(effective_command, blocked_reason)`: `blocked_reason` is
    /// `Some` when the chain voted `Block`, otherwise `None` and the effective
    /// (possibly rewritten) command.
    pub fn resolve(&self, command: &str) -> Result<(String, Option<String>)> {
        match self.process(command)? {
            InterceptDecision::Allow => Ok((command.to_string(), None)),
            InterceptDecision::Rewrite(next) => Ok((next, None)),
            InterceptDecision::Block(reason) => Ok((command.to_string(), Some(reason))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn destructive_blocks_rm_rf_root() {
        let i = DestructiveCommandInterceptor;
        for cmd in [
            "rm -rf /",
            "rm -rf /*",
            "rm -fr /",
            "sudo rm -rf /",
            "echo hi; rm -rf /",
            "rm --no-preserve-root -rf /",
        ] {
            assert!(i.intercept(cmd).unwrap().is_block(), "should block: {cmd}");
        }
    }

    #[test]
    fn destructive_blocks_mkfs_dd_and_fork_bomb() {
        let i = DestructiveCommandInterceptor;
        assert!(i.intercept("mkfs.ext4 /dev/sda1").unwrap().is_block());
        assert!(i
            .intercept("sudo mkfs -t ext4 /dev/sda1")
            .unwrap()
            .is_block());
        assert!(i
            .intercept("dd if=/dev/zero of=/dev/sda")
            .unwrap()
            .is_block());
        assert!(i
            .intercept("sudo dd if=/dev/random of=/dev/sda")
            .unwrap()
            .is_block());
        assert!(i.intercept(":(){ :|:& };:").unwrap().is_block());
        assert!(i.intercept(":() { : | : & } ; :").unwrap().is_block());
    }

    #[test]
    fn destructive_allows_normal_commands() {
        let i = DestructiveCommandInterceptor;
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
            assert_eq!(
                i.intercept(cmd).unwrap(),
                InterceptDecision::Allow,
                "should allow: {cmd}"
            );
        }
    }

    #[test]
    fn destructive_blocks_obfuscated_and_substituted_commands() {
        let i = DestructiveCommandInterceptor;
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
            assert!(i.intercept(cmd).unwrap().is_block(), "should block: {cmd}");
        }
    }

    #[test]
    fn telemetry_never_blocks_and_classifies() {
        let t = TelemetryInterceptor;
        assert_eq!(t.intercept("cargo test").unwrap(), InterceptDecision::Allow);
        assert_eq!(t.intercept("git status").unwrap(), InterceptDecision::Allow);
        assert_eq!(CommandClass::classify("cargo test"), CommandClass::Heavy);
        assert_eq!(CommandClass::classify("git status"), CommandClass::Light);
        assert_eq!(CommandClass::Heavy.as_str(), "heavy");
        assert_eq!(CommandClass::Light.as_str(), "light");
    }

    #[test]
    fn pipeline_short_circuits_on_block() {
        let pipe = CommandPipeline::default_pipeline();
        let decision = pipe.process("rm -rf /").unwrap();
        assert!(decision.is_block());
        assert_eq!(
            pipe.process("cargo test").unwrap(),
            InterceptDecision::Allow
        );
        assert_eq!(
            pipe.process("git status").unwrap(),
            InterceptDecision::Allow
        );
    }

    #[test]
    fn pipeline_applies_rewrites_in_order() {
        struct Prefix;
        impl CommandInterceptor for Prefix {
            fn intercept(&self, command: &str) -> Result<InterceptDecision> {
                Ok(InterceptDecision::Rewrite(format!("echo {command}")))
            }
            fn name(&self) -> &'static str {
                "prefix"
            }
        }
        let pipe = CommandPipeline::new().with(Prefix);
        assert_eq!(
            pipe.process("hi").unwrap(),
            InterceptDecision::Rewrite("echo hi".to_string())
        );
        let (effective, blocked) = pipe.resolve("hi").unwrap();
        assert_eq!(effective, "echo hi");
        assert_eq!(blocked, None);

        let (_, blocked) = CommandPipeline::default_pipeline()
            .resolve("rm -rf /")
            .unwrap();
        assert!(blocked.is_some());
    }

    #[test]
    fn empty_pipeline_allows_everything() {
        let pipe = CommandPipeline::new();
        assert!(pipe.is_empty());
        assert_eq!(pipe.len(), 0);
        assert_eq!(pipe.process("rm -rf /").unwrap(), InterceptDecision::Allow);
        assert_eq!(CommandPipeline::default().len(), 0);
    }
}
