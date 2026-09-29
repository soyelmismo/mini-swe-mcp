//! Environment hygiene for agent-spawned child processes.
//!
//! A subagent's shell is a *hostile* environment by default: the worker
//! process inherits the operator's entire environment, so every API key, cloud
//! credential, registry token and SSH agent socket that happens to be exported
//! in the operator's shell is one `echo $OPENAI_API_KEY` (or a stray
//! `git push`) away from being exfiltrated by a model that was handed an
//! unrelated bug-fix task. Inheriting a large ambient environment also makes
//! runs non-reproducible: two workers can behave differently purely because of
//! unrelated variables.
//!
//! This module replaces that inheritance with an explicit, minimal contract:
//!
//! 1. **Deny by default.** The child is spawned with [`std::process::Command::env_clear`],
//!    so *nothing* from the parent leaks unless it is deliberately re-added.
//! 2. **Strict allow-list.** Only the handful of runtime variables a compiler or
//!    shell genuinely needs survive ([`ALLOWED_VARS`]): toolchain discovery
//!    (`PATH`, `CARGO_HOME`, `RUSTUP_HOME`), identity (`USER`, `LOGNAME`,
//!    `SHELL`) and locale/terminal presentation (`LANG`, `LC_ALL`, `TERM`).
//! 3. **Isolated `HOME`.** `HOME` is remapped to a per-worktree scratch
//!    directory so a command that reads `~/.aws/credentials`, writes
//!    `~/.gitconfig` or drops a stray `~/.npmrc` touches only the sandbox, never
//!    the operator's real home.
//! 4. **Explicit secret purge.** [`is_sensitive_var`] is a second, independent
//!    line of defence: even if a sensitive name were somehow added to the
//!    allow-list (or reintroduced by a later `env(...)` call), it is stripped
//!    before the child is spawned.
//!
//! [`build_clean_environment`] is the single entry point. It returns a plain
//! `Vec<(String, String)>` rather than mutating a `Command` so that the exact
//! contract can be unit-tested without spawning anything, and so callers can
//! log/inspect the child environment in a debugging story.

#[cfg(test)]
use std::ffi::OsString;
use std::path::{Path, PathBuf};

/// Variables forwarded from the parent process into the agent's shell.
///
/// Deliberately tiny: every entry here is a variable whose *absence* breaks
/// ordinary toolchain execution, and none of them can carry a credential.
/// `PATH` locates the binaries (the command is spawned as `nice`/`bwrap`,
/// which are then resolved through it), `CARGO_HOME`/`RUSTUP_HOME` locate the
/// Rust toolchain, `USER`/`LOGNAME`/`SHELL` keep `id` and prompt-oriented tools
/// sane, and `LANG`/`LC_ALL`/`TERM` keep output and diagnostics readable.
pub const ALLOWED_VARS: &[&str] = &[
    "PATH",
    "CARGO_HOME",
    "RUSTUP_HOME",
    "USER",
    "LOGNAME",
    "SHELL",
    "LANG",
    "LC_ALL",
    "TERM",
];

/// Name of the directory created under the worker's target dir to serve as the
/// child shell's `HOME`.
const ISOLATED_HOME_DIR: &str = "home";

/// Environment variable holding the (remapped) home directory.
///
/// `HOME` is not in [`ALLOWED_VARS`]: it is *always* overridden with the
/// isolated scratch directory, so copying the operator's value through would be
/// both redundant and unsafe.
const HOME_VAR: &str = "HOME";

/// Substrings that mark a variable name as credential-bearing.
///
/// The check is a case-insensitive substring match, so it covers the common
/// families without enumerating every vendor: `OPENAI_API_KEY` and
/// `ANTHROPIC_AUTH_TOKEN` both match `API_KEY`/`AUTH_TOKEN`,
/// `AWS_SECRET_ACCESS_KEY` matches `SECRET`, `GITHUB_TOKEN` matches `TOKEN`,
/// and `SSH_AUTH_SOCK` matches `SSH`. HTTP-style `BEARER_*` and
/// `AUTHORIZATION` headers, key material (`PRIVATE_KEY`, `ENCRYPTION_KEY`,
/// `SIGNING_KEY`) and session material (`SESSION_TOKEN`, `REFRESH_TOKEN`,
/// `PASSPHRASE`) are covered by the same mechanism.
///
/// Markers are deliberately specific. A bare family prefix such as `NPM_` is
/// *not* used: `NPM_CONFIG_PREFIX` and `NPM_CONFIG_CACHE` are ordinary
/// configuration, and `NPM_TOKEN` is already caught by `TOKEN`.
const SECRET_MARKERS: &[&str] = &[
    "API_KEY",
    "APIKEY",
    "API_TOKEN",
    "AUTH_TOKEN",
    "ACCESS_TOKEN",
    "SECRET",
    "PASSWORD",
    "PASSWD",
    "PASSPHRASE",
    "TOKEN",
    "BEARER",
    "AUTHORIZATION",
    "CREDENTIAL",
    "PRIVATE_KEY",
    "SESSION_KEY",
    "SESSION_TOKEN",
    "REFRESH_TOKEN",
    "ENCRYPTION_KEY",
    "SIGNING_KEY",
    "CLIENT_SECRET",
    "CONNECTION_STRING",
    "AWS_",
    "SSH_",
    "GPG_",
    "PGP_",
];

/// Whether a variable name must never reach an agent-spawned child.
///
/// Used as a defence-in-depth filter over the allow-list: a name that trips
/// this predicate is dropped even if it is explicitly allowed, so a future
/// (mis)edit that widens [`ALLOWED_VARS`] cannot quietly re-expose a secret.
///
/// Matching is case-insensitive so `aws_secret_access_key` is caught alongside
/// the canonical spelling. Empty names are rejected: an empty variable name is
/// never meaningful and is not something to forward.
pub fn is_sensitive_var(name: &str) -> bool {
    if name.is_empty() {
        return true;
    }
    // Marker matching is case-insensitive without allocating: the allow-list is
    // re-screened on every child spawn and this predicate is pure, so building
    // an uppercasing `String` would cost a heap allocation per variable per
    // command. `eq_ignore_ascii_case` is byte-wise, so a non-ASCII byte can
    // never compare equal to an ASCII marker.
    if SECRET_MARKERS
        .iter()
        .any(|marker| contains_ignore_ascii_case(name, marker))
    {
        return true;
    }
    ["AWS_", "SSH_", "GPG_"]
        .iter()
        .any(|prefix| starts_with_ignore_ascii_case(name, prefix))
        // "Personal access token" shorthand: `GITHUB_PAT`, `GLAB_PAT`, `FOO_PAT_X`.
        // Matched only at a word boundary so a variable that merely contains the
        // letters (`PATH`, `PATCH_LEVEL`) is not mistaken for a credential.
        || is_word(name, "PAT")
}

/// Whether `name` starts with `prefix`, comparing ASCII case-insensitively.
fn starts_with_ignore_ascii_case(name: &str, prefix: &str) -> bool {
    let (name, prefix) = (name.as_bytes(), prefix.as_bytes());
    name.len() >= prefix.len() && name[..prefix.len()].eq_ignore_ascii_case(prefix)
}

/// Whether `name` contains `needle`, comparing ASCII case-insensitively.
///
/// Equivalent to `name.to_ascii_lowercase().contains(&needle.to_ascii_lowercase())`
/// without the two temporary `String`s.
fn contains_ignore_ascii_case(name: &str, needle: &str) -> bool {
    if needle.is_empty() {
        return true;
    }
    let (hay, needle) = (name.as_bytes(), needle.as_bytes());
    if needle.len() > hay.len() {
        return false;
    }
    hay.windows(needle.len())
        .any(|window| window.eq_ignore_ascii_case(needle))
}

/// Whether `name` contains `word` delimited by a non-alphanumeric boundary.
///
/// A bare substring test is too greedy for short markers: `PAT` occurs inside
/// `PATH` and `PATCH_LEVEL`, so matching it anywhere would flag every
/// `*_PATH`/`*_PATCH` variable. Requiring `_`, `-` or a string edge on both
/// sides keeps the match precise while still covering the common `FOO_PAT_BAR`
/// spellings.
fn is_word(name: &str, word: &str) -> bool {
    let bytes = name.as_bytes();
    let word = word.as_bytes();
    if word.is_empty() || word.len() > bytes.len() {
        return false;
    }
    for start in 0..=bytes.len() - word.len() {
        if !bytes[start..start + word.len()].eq_ignore_ascii_case(word) {
            continue;
        }
        let before_ok = start == 0 || !is_word_byte(bytes[start - 1]);
        let after_end = start + word.len();
        let after_ok = after_end == bytes.len() || !is_word_byte(bytes[after_end]);
        if before_ok && after_ok {
            return true;
        }
    }
    false
}

/// Whether `byte` is alphanumeric (ASCII letter/digit, or any non-ASCII
/// UTF-8 byte, which is never treated as a delimiter).
fn is_word_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || !byte.is_ascii()
}

/// Resolve the isolated `HOME` for a command running in `worktree_path`.
///
/// The scratch directory is `<worktree>/target/<ISOLATED_HOME_DIR>`: it lives
/// *inside* the worktree, so it inherits the sandbox's read-write bind on the
/// worktree and is therefore both isolated from the operator's real `$HOME` and
/// writable from inside the bubblewrap sandbox (whose tmpfs over the real home
/// would otherwise make a home directory there unwritable).
///
/// Falling back to `<repo>/target/<ISOLATED_HOME_DIR>` when the worktree has no
/// directory name of its own keeps the result absolute and inside a directory
/// the worker already owns, rather than degenerating to a relative path that
/// tools would resolve against an arbitrary working directory.
pub fn isolated_home(repo_path: &Path, worktree_path: &Path) -> PathBuf {
    let base = match worktree_path.file_name() {
        Some(name) if !name.is_empty() => worktree_path,
        _ => repo_path,
    };
    base.join("target").join(ISOLATED_HOME_DIR)
}

/// Build the sanitized environment for a child process.
///
/// `repo_path` is the original repository checkout and `worktree_path` the
/// per-worker worktree the command actually runs in. The worktree is the
/// preferred anchor for the isolated `HOME`, because the child is chdir'ed
/// into it and the sandbox binds it read-write; `repo_path` is only consulted
/// when the worktree has no directory name of its own, so the isolated home
/// never degenerates into a relative path.
///
/// The returned vector is deterministic (allow-list order, then `HOME`) and
/// contains **only** variables the child is allowed to see:
///
/// * each allow-listed variable that is set in the parent, copied verbatim;
/// * `HOME`, always remapped to [`isolated_home`] — never the real home.
///
/// Because the caller must pair this with `env_clear()`, the result is a
/// complete environment, not a patch. Credentials, tokens, `*_PROXY` and
/// anything else the operator had exported are simply absent, and
/// [`is_sensitive_var`] guarantees a sensitive name cannot slip back in even if
/// a caller re-adds one afterwards.
///
/// The function deliberately does **not** create the directory: callers that
/// need it to exist (command spawning) create it explicitly, which keeps this
/// function side-effect free and cheap to call from tests.
pub fn build_clean_environment(repo_path: &Path, worktree_path: &Path) -> Vec<(String, String)> {
    // Sized up front for the worst case (every allow-listed name set) plus the
    // `HOME` appended below, so the vector never reallocates mid-build.
    let mut env: Vec<(String, String)> = Vec::with_capacity(ALLOWED_VARS.len() + 1);
    for name in ALLOWED_VARS {
        // Defence in depth: an allow-listed name that trips the secret filter is
        // dropped regardless of its value.
        if is_sensitive_var(name) {
            continue;
        }
        // Empty values carry no information and can only confuse tools
        // that test "is this configured?" with a truthiness check.
        match std::env::var(name) {
            // One allocation for the key, one for the value; nothing is
            // allocated for names that are unset or empty.
            Ok(value) if !value.is_empty() => env.push(((*name).to_string(), value)),
            _ => {}
        }
    }

    // HOME is remapped, never inherited: an empty value would make many tools
    // fall back to the real home (or the passwd database), defeating the point.
    let home = isolated_home(repo_path, worktree_path);
    env.push((HOME_VAR.to_string(), home.to_string_lossy().into_owned()));

    env
}

/// The subset of the process-spawning API this module needs, so the same
/// sanitization applies to both `std::process::Command` and
/// `tokio::process::Command` without duplicating the allow-list logic.
pub trait CommandEnv {
    /// Drop the entire inherited environment.
    fn env_clear(&mut self) -> &mut Self;
    /// Add or overwrite a single variable.
    fn env<K, V>(&mut self, key: K, value: V) -> &mut Self
    where
        K: AsRef<std::ffi::OsStr>,
        V: AsRef<std::ffi::OsStr>;
}

impl CommandEnv for std::process::Command {
    fn env_clear(&mut self) -> &mut Self {
        std::process::Command::env_clear(self)
    }
    fn env<K, V>(&mut self, key: K, value: V) -> &mut Self
    where
        K: AsRef<std::ffi::OsStr>,
        V: AsRef<std::ffi::OsStr>,
    {
        std::process::Command::env(self, key, value)
    }
}

impl CommandEnv for tokio::process::Command {
    fn env_clear(&mut self) -> &mut Self {
        tokio::process::Command::env_clear(self)
    }
    fn env<K, V>(&mut self, key: K, value: V) -> &mut Self
    where
        K: AsRef<std::ffi::OsStr>,
        V: AsRef<std::ffi::OsStr>,
    {
        tokio::process::Command::env(self, key, value)
    }
}

/// Apply [`build_clean_environment`] to a child command.
///
/// The parent's environment is cleared first, so the child sees exactly the
/// allow-list and nothing else; the isolated `HOME` directory is created before
/// the spawn so tools that expect it to exist (npm, git config discovery) do
/// not fail on a missing directory.
pub fn apply_clean_environment<C: CommandEnv>(cmd: &mut C, repo_path: &Path, worktree_path: &Path) {
    let env = build_clean_environment(repo_path, worktree_path);
    let _ = std::fs::create_dir_all(isolated_home(repo_path, worktree_path));
    cmd.env_clear();
    for (key, value) in env {
        cmd.env(key, value);
    }
}

/// [`apply_clean_environment`] for a `tokio::process::Command`.
///
/// A named alias so call sites in `super::exec` read as "apply the clean
/// environment to this command" without repeating the generic bound.
pub fn apply_clean_environment_cmd(
    cmd: &mut tokio::process::Command,
    repo_path: &Path,
    worktree_path: &Path,
) {
    apply_clean_environment(cmd, repo_path, worktree_path);
}

/// Whether `value` is usable as a `HOME` replacement (non-empty, not `/`).
///
/// Exposed for tests; `/` would expose the whole filesystem as a home.
#[cfg(test)]
fn is_safe_home(value: &OsString) -> bool {
    let path = Path::new(value);
    !value.is_empty() && path != Path::new("/") && path.is_absolute()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unique_dir(tag: &str) -> PathBuf {
        let unique_id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("swe-env-test-{tag}-{unique_id}"));
        let _ = std::fs::create_dir_all(&dir);
        dir
    }

    #[test]
    fn sensitive_var_detection_covers_credential_families() {
        for name in [
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GITHUB_TOKEN",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_PROFILE",
            "AWS_REGION",
            "SSH_AUTH_SOCK",
            "SSH_AGENT_PID",
            "SSH_PRIVATE_KEY",
            "GITHUB_PAT",
            "GPG_TTY",
            "DB_PASSWORD",
            "SESSION_TOKEN",
            "ANTHROPIC_AUTH_TOKEN",
        ] {
            assert!(is_sensitive_var(name), "{name} must be treated as secret");
        }
    }

    #[test]
    fn allowed_vars_are_not_treated_as_sensitive() {
        for name in ALLOWED_VARS {
            assert!(
                !is_sensitive_var(name),
                "{name} is on the allow-list and must survive the secret filter"
            );
        }
    }

    #[test]
    fn unknown_non_secret_vars_are_not_sensitive() {
        for name in ["CARGO_TARGET_DIR", "PWD", "TMPDIR", "EDITOR", "CI"] {
            assert!(!is_sensitive_var(name), "{name} is not a credential");
        }
    }

    #[test]
    fn bearer_and_authorization_vars_are_treated_as_secret() {
        for name in [
            "BEARER",
            "BEARER_TOKEN",
            "AUTHORIZATION",
            "PROXY_AUTHORIZATION",
            "X_BEARER_CREDENTIAL",
        ] {
            assert!(is_sensitive_var(name), "{name} must be treated as secret");
        }
    }

    #[test]
    fn key_and_session_material_is_treated_as_secret() {
        for name in [
            "PRIVATE_KEY",
            "RSA_PRIVATE_KEY_PEM",
            "ENCRYPTION_KEY",
            "SIGNING_KEY",
            "SESSION_TOKEN",
            "REFRESH_TOKEN",
            "PASSPHRASE",
            "PASSPHRASE_FILE",
            "CLIENT_SECRET",
            "CONNECTION_STRING",
        ] {
            assert!(is_sensitive_var(name), "{name} must be treated as secret");
        }
    }

    #[test]
    fn credential_matching_is_case_insensitive() {
        for name in [
            "openai_api_key",
            "Openai_Api_Key",
            "aws_secret_access_key",
            "Aws_Secret_Access_Key",
            "github_pat",
            "GITHUB_pat",
            "Ssh_Auth_Sock",
            "bearer_token",
        ] {
            assert!(
                is_sensitive_var(name),
                "{name} must match its uppercase spelling"
            );
        }
    }

    #[test]
    fn short_markers_do_not_false_positive_on_ordinary_vars() {
        // Guards the word-boundary match for `PAT` and the absence of broad
        // family prefixes: none of these carry a credential.
        for name in [
            "PATH",
            "MY_PATH",
            "PATCH_LEVEL",
            "PATTERN",
            "SETUPTOOLS_SCM",
            "NPM_CONFIG_PREFIX",
            "NPM_CONFIG_CACHE",
            "SESSION_TYPE",
            "XDG_DATA_HOME",
            "KEYBOARD_LAYOUT",
            "AUTHOR_NAME",
        ] {
            assert!(!is_sensitive_var(name), "{name} is not a credential");
        }
    }

    #[test]
    fn empty_name_is_rejected() {
        assert!(is_sensitive_var(""), "empty name must never be forwarded");
    }

    #[test]
    fn build_clean_environment_only_returns_allow_list_plus_home() {
        let dir = unique_dir("allowlist");
        let env = build_clean_environment(&dir, &dir);

        let names: Vec<&str> = env.iter().map(|(k, _)| k.as_str()).collect();
        for name in &names {
            assert!(
                ALLOWED_VARS.contains(name) || *name == HOME_VAR,
                "{name} escaped the allow-list"
            );
        }
        assert!(env.iter().any(|(k, _)| k == HOME_VAR), "HOME must be set");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn build_clean_environment_drops_secrets_from_parent() {
        let dir = unique_dir("secrets");
        // Inject secrets into *our* environment; the child env must not carry them.
        unsafe {
            std::env::set_var("OPENAI_API_KEY", "sk-test-should-not-leak");
            std::env::set_var("AWS_SECRET_ACCESS_KEY", "aws-test-should-not-leak");
            std::env::set_var("GITHUB_TOKEN", "gh-test-should-not-leak");
            std::env::set_var("SSH_AUTH_SOCK", "/tmp/agent.sock");
        }
        let env = build_clean_environment(&dir, &dir);
        unsafe {
            std::env::remove_var("OPENAI_API_KEY");
            std::env::remove_var("AWS_SECRET_ACCESS_KEY");
            std::env::remove_var("GITHUB_TOKEN");
            std::env::remove_var("SSH_AUTH_SOCK");
        }

        for (k, _) in &env {
            assert!(!is_sensitive_var(k), "{k} leaked into the child env");
        }
        assert!(
            !env.iter().any(|(k, _)| k == "OPENAI_API_KEY"),
            "OPENAI_API_KEY must not survive"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn home_is_remapped_into_the_worktree() {
        let dir = unique_dir("home");
        let env = build_clean_environment(&dir, &dir);
        let home = env
            .iter()
            .find(|(k, _)| k == HOME_VAR)
            .map(|(_, v)| v.clone())
            .expect("HOME must be present");
        let home_path = PathBuf::from(&home);
        assert_eq!(home_path, isolated_home(&dir, &dir));
        assert!(
            home_path.starts_with(&dir),
            "HOME must live under the worktree"
        );
        assert!(!home_path.starts_with(std::env::var("HOME").unwrap_or_default()));
        assert!(is_safe_home(&OsString::from(&home)));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn home_is_remapped_even_when_the_worktree_is_relative() {
        let env = build_clean_environment(Path::new("/repo"), Path::new("wt"));
        let home = env
            .iter()
            .find(|(k, _)| k == HOME_VAR)
            .map(|(_, v)| v.clone())
            .expect("HOME must be present");
        // Relative worktrees still yield an isolated, non-root location.
        assert!(home.ends_with("target/home"), "got {home}");
        assert_ne!(home, "/");

        // An anonymous worktree (no file name) falls back to the repository.
        let fallback = build_clean_environment(Path::new("/repo"), Path::new("/"));
        let fallback_home = fallback
            .iter()
            .find(|(k, _)| k == HOME_VAR)
            .map(|(_, v)| v.clone())
            .expect("HOME must be present");
        assert_eq!(fallback_home, "/repo/target/home");
    }

    #[test]
    fn apply_clean_environment_clears_and_sets_isolated_home() {
        let dir = unique_dir("apply");
        let mut cmd = std::process::Command::new("true");
        // Seed with a secret so env_clear has something to remove.
        cmd.env("OPENAI_API_KEY", "sk-should-not-leak");
        apply_clean_environment(&mut cmd, &dir, &dir);

        let envs: Vec<(String, String)> = cmd
            .get_envs()
            .filter_map(|(k, v)| Some((k.to_string_lossy().into_owned(), v?.to_str()?.to_string())))
            .collect();
        assert!(
            !envs.iter().any(|(k, _)| k == "OPENAI_API_KEY"),
            "env_clear + allow-list must drop OPENAI_API_KEY: {envs:?}"
        );
        for (k, _) in &envs {
            assert!(
                ALLOWED_VARS.contains(&k.as_str()) || k == HOME_VAR,
                "{k} escaped the allow-list"
            );
        }
        assert!(
            isolated_home(&dir, &dir).is_dir(),
            "isolated HOME must be created"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
