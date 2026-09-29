//! Pre-execution command interceptor middleware.
//!
//! Interceptors run inside [`crate::agent::AgentRunner::execute_bash`] before
//! any sandbox construction or process spawn. Each interceptor inspects the
//! requested command and votes [`InterceptDecision::Allow`],
//! [`InterceptDecision::Block`] or [`InterceptDecision::Rewrite`]. The
//! [`CommandPipeline`] chains them in registration order: the first `Block`
//! wins, while `Rewrite` updates the effective command seen by the remaining
//! interceptors and by the executor.

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

/// Blocks destructive patterns that must never reach the executor, even
/// inside the bubblewrap sandbox (the sandbox bind-mounts the worktree
/// read-write, so `rm -rf` inside it still destroys user data).
#[derive(Debug, Default, Clone, Copy)]
pub struct DestructiveCommandInterceptor;

impl DestructiveCommandInterceptor {
    /// Reason string for a matched dangerous pattern.
    fn block_reason(matched: &str) -> String {
        format!("destructive pattern blocked: {matched}")
    }

    /// Case-insensitive substring search of the lowercased command.
    fn contains_lower(lowered: &str, needle: &str) -> bool {
        lowered.contains(needle)
    }
}

impl CommandInterceptor for DestructiveCommandInterceptor {
    fn intercept(&self, command: &str) -> Result<InterceptDecision> {
        let lowered = command.to_lowercase();
        let trimmed = lowered.trim();

        // `rm -rf /`, `rm -fr /`, `rm -rf /*`, `rm --no-preserve-root`, bare `rm -rf /`.
        // Match token-aware: look for `rm` followed by recursive+force flags and
        // a root-level target (`/`, `/*`, `~`, `/home`, `/etc`, ...). A plain
        // `rm -rf target/` inside the worktree must stay allowed.
        if is_destructive_rm(trimmed) {
            return Ok(InterceptDecision::Block(Self::block_reason("rm -rf /")));
        }

        // Filesystem creation / raw disk writes.
        if Self::contains_lower(trimmed, "mkfs") {
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

/// `true` when `command` is an `rm` invocation that would recursively delete
/// at or above the filesystem root (or home), rather than a path inside the
/// worktree.
fn is_destructive_rm(lowered_trimmed: &str) -> bool {
    // Split on shell separators so `echo hi; rm -rf /` is still caught.
    for segment in lowered_trimmed.split([';', '&', '|', '`', '$', '\n']) {
        let seg = segment.trim();
        // Strip leading `sudo` wrappers.
        let seg = seg.strip_prefix("sudo").map_or(seg, |s| s.trim_start());
        if !(seg.starts_with("rm ") || seg.starts_with("rm\t") || seg == "rm") {
            continue;
        }
        // Flags must contain both recursive (`r`) and force (`f`), e.g.
        // `-rf`, `-fr`, `-r -f`, `--recursive --force`, or `--no-preserve-root`.
        // Short flags may be combined (`-rf`), so inspect flag characters
        // rather than substrings: `-rf` contains "-r" but not "-f".
        let mut has_recursive = seg.contains("--recursive") || seg.contains("--no-preserve-root");
        let mut has_force = seg.contains("--force") || seg.contains("--no-preserve-root");
        for tok in seg.split_whitespace() {
            if tok.starts_with("--") {
                continue; // handled above
            }
            if tok.starts_with('-') && tok.len() > 1 {
                let flags = &tok[1..];
                if flags.contains('r') {
                    has_recursive = true;
                }
                if flags.contains('f') {
                    has_force = true;
                }
            }
        }
        if !(has_recursive && has_force) {
            continue;
        }
        // Check for a bare `/` token or a destructive prefix after the flags.
        // Rule: `rm <flags> <target>` where the target is `/`, `/*`, `~`, or
        // a single-component top-level system dir — never a path inside the
        // worktree.
        if seg.contains("--no-preserve-root") {
            return true;
        }
        // Look at tokens after `rm`.
        let tokens: Vec<&str> = seg.split_whitespace().collect();
        for tok in tokens.iter().skip(1) {
            // Skip flag tokens.
            if tok.starts_with('-') {
                continue;
            }
            let t = tok.trim_matches(|c| c == '"' || c == '\'');
            if t == "/" || t == "/*" || t == "~" || t == "$home" || t == "${home}" {
                return true;
            }
            if t.starts_with("/*") {
                return true;
            }
            // `rm -rf / home`-style split or top-level system dir with no
            // deeper path component (e.g. `/etc`, not `/etc/hostname.bak`?
            // conservatively block any direct child of root that is short).
            if t.starts_with('/') && !t.starts_with("/tmp/") && !t.starts_with("/dev/null") {
                let depth = t.split('/').filter(|s| !s.is_empty()).count();
                if depth <= 1 {
                    return true;
                }
            }
            if *tok == "/" {
                return true;
            }
        }
    }
    false
}

/// `true` when `command` invokes `dd` with an input source (raw disk copy).
fn dd_targets_disk(lowered_trimmed: &str) -> bool {
    for segment in lowered_trimmed.split([';', '&', '|', '\n']) {
        let seg = segment.trim();
        let seg = seg.strip_prefix("sudo").map_or(seg, |s| s.trim_start());
        if (seg.starts_with("dd ") || seg.starts_with("dd\t") || seg == "dd") && seg.contains("if=")
        {
            return true;
        }
    }
    false
}

/// `true` when `command` contains the classic bash fork bomb, ignoring
/// whitespace differences (`:(){ :|:& };:`).
fn is_fork_bomb(lowered_trimmed: &str) -> bool {
    let compact: String = lowered_trimmed
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
        assert!(
            i.intercept("sudo mkfs -t ext4 /dev/sda1")
                .unwrap()
                .is_block()
        );
        assert!(
            i.intercept("dd if=/dev/zero of=/dev/sda")
                .unwrap()
                .is_block()
        );
        assert!(
            i.intercept("sudo dd if=/dev/random of=/dev/sda")
                .unwrap()
                .is_block()
        );
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
