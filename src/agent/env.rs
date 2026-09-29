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
//!    shell genuinely needs survive ([`ALLOWED_VARS`]): binary discovery
//!    (`PATH`), identity (`USER`, `LOGNAME`, `SHELL`) and locale/terminal
//!    presentation (`LANG`, `LC_ALL`, `TERM`).
//! 3. **Toolchain cache forwarding.** `CARGO_HOME` and `RUSTUP_HOME`
//!    ([`TOOLCHAIN_VARS`]) are *resolved* rather than copied — `CARGO_HOME`
//!    falls back to the host's `~/.cargo` when the parent did not set it
//!    ([`host_cargo_home`]). This is what keeps a sandboxed worktree able to
//!    build offline: `HOME` is remapped to an empty scratch directory, so with
//!    no explicit `CARGO_HOME` Cargo would find an empty registry and try to
//!    reach `index.crates.io` for crates the operator already has cached. These
//!    variables name *directories*, not secrets, and are still re-screened by
//!    [`is_sensitive_var`] on the way out.
//! 4. **Isolated `HOME`.** `HOME` is remapped to a per-worktree scratch
//!    directory so a command that reads `~/.aws/credentials`, writes
//!    `~/.gitconfig` or drops a stray `~/.npmrc` touches only the sandbox, never
//!    the operator's real home.
//! 5. **Explicit secret purge.** [`is_sensitive_var`] is a second, independent
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
#[cfg(test)]
use std::sync::{Mutex, MutexGuard};

/// Variables copied verbatim from the parent process into the agent's shell.
///
/// Deliberately tiny: every entry here is a variable whose *absence* breaks
/// ordinary execution, and none of them can carry a credential. `PATH` locates
/// the binaries (the command is spawned as `nice`/`bwrap`, which are then
/// resolved through it), `USER`/`LOGNAME`/`SHELL` keep `id` and prompt-oriented
/// tools sane, and `LANG`/`LC_ALL`/`TERM` keep output and diagnostics readable.
///
/// The Rust toolchain locations are deliberately **not** here: `CARGO_HOME` and
/// `RUSTUP_HOME` need discovery rather than a verbatim copy (see
/// [`host_cargo_home`]), so they are forwarded by [`build_clean_environment`]
/// through [`TOOLCHAIN_VARS`]. Keeping them in both lists would make the output
/// order depend on which list happened to run first.
pub const ALLOWED_VARS: &[&str] = &["PATH", "USER", "LOGNAME", "SHELL", "LANG", "LC_ALL", "TERM"];

/// Toolchain cache locations forwarded to the agent's shell, in output order.
///
/// Unlike [`ALLOWED_VARS`] these are *resolved* rather than copied: a name unset
/// in the parent falls back to a host default where one can be discovered on the
/// filesystem, and is otherwise omitted. The child then uses the same populated
/// registry the operator's shell would, instead of the empty isolated `HOME` a
/// plain `env_clear` leaves behind.
pub const TOOLCHAIN_VARS: &[&str] = &[CARGO_HOME_VAR, RUSTUP_HOME_VAR];

/// Cargo's configuration/registry directory (the crates cache, plus the
/// registry credentials and config that live alongside it).
pub const CARGO_HOME_VAR: &str = "CARGO_HOME";

/// `rustup`'s toolchain directory (the installed compilers and shims).
pub const RUSTUP_HOME_VAR: &str = "RUSTUP_HOME";

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
        // BEARER_* is a credential; e.g. UNBEARERABLE is not.
        || is_word(name, "BEARER")
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
/// A bare substring test is too greedy for markers: `PAT` occurs inside
/// `PATH` and `PATCH_LEVEL`, and `BEARER` occurs inside `UNBEARERABLE`.
/// Requiring non-alphanumeric delimiters or a string edge on both sides
/// keeps the match precise while still covering `FOO_PAT_BAR` and `BEARER_TOKEN`.
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

/// Resolve the host's Cargo directory for forwarding to a sandboxed child.
///
/// `CARGO_HOME` is where Cargo keeps the registry cache, so it is what makes an
/// *offline* `cargo build`/`cargo test` possible inside a sandboxed worktree.
/// The child cannot discover it on its own: `HOME` is remapped to a per-worktree
/// scratch directory, so Cargo's own `$HOME/.cargo` default resolves to an empty
/// directory and an already-cached dependency set becomes an apparent network
/// fetch.
///
/// Resolution order (see [`resolve_cargo_home`] for the pure policy):
///
/// 1. `CARGO_HOME` as set in the parent, forwarded verbatim so a non-default
///    install layout keeps working;
/// 2. `$HOME/.cargo`, **only when it actually exists** — a host with no Cargo
///    directory has no cache to forward, and pointing at a non-existent path
///    would be worse than omitting the variable and letting Cargo apply its own
///    default resolution;
/// 3. otherwise nothing, and the variable is omitted entirely.
///
/// Only the *path* is forwarded. `$CARGO_HOME/credentials.toml`, registry tokens
/// in `$CARGO_HOME/config.toml` and any `CARGO_REGISTRY_TOKEN` stay unreachable,
/// because the variable is the only thing this module emits: the allow-list
/// re-screen in [`build_clean_environment`] still runs over the result, and no
/// credential-bearing *name* is introduced by this lookup.
pub fn host_cargo_home() -> Option<PathBuf> {
    resolve_cargo_home(
        std::env::var_os(CARGO_HOME_VAR).as_deref().map(Path::new),
        std::env::var_os("HOME").as_deref().map(Path::new),
    )
}

/// The pure core of [`host_cargo_home`], with the parent's `CARGO_HOME` and
/// `HOME` passed in instead of read from the process environment.
///
/// Split out so the resolution *policy* — explicit wins, `~/.cargo` only when it
/// exists, nothing otherwise — is unit-testable against synthetic layouts (a
/// home that does have a `.cargo`, one that does not, an unset variable) rather
/// than only against whatever the machine running the tests happens to have. It
/// also keeps the `is_dir` filesystem probe out of the "forwarded verbatim" path,
/// so that path allocates nothing.
pub fn resolve_cargo_home(explicit: Option<&Path>, home: Option<&Path>) -> Option<PathBuf> {
    if let Some(explicit) = explicit.filter(|value| !value.as_os_str().is_empty()) {
        return Some(explicit.to_path_buf());
    }
    let dot_cargo = home?.join(".cargo");
    dot_cargo.is_dir().then_some(dot_cargo)
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
/// The returned vector is deterministic (allow-list order, then
/// [`TOOLCHAIN_VARS`] order, then `HOME`) and contains **only** variables the
/// child is allowed to see:
///
/// * each allow-listed variable that is set in the parent, copied verbatim;
/// * each toolchain cache location that could be resolved — `CARGO_HOME` from
///   the host, falling back to `~/.cargo` when it exists ([`host_cargo_home`]),
///   and `RUSTUP_HOME` when the parent sets it. This is what lets a sandboxed
///   worktree build against the host's crates cache offline: `HOME` is remapped
///   to an empty scratch directory, so with no explicit `CARGO_HOME` Cargo
///   would find an empty registry and try the network;
/// * `HOME`, always remapped to [`isolated_home`] — never the real home.
///
/// The toolchain variables name *directories*, not secrets: no credential-
/// bearing variable name is introduced, and every name is re-screened by
/// [`is_sensitive_var`] on the way out, so a credential cannot ride along in a
/// cache path.
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
    // Sized up front for the worst case (every allow-listed name set, every
    // toolchain cache resolved) plus the `HOME` appended below, so the vector
    // never reallocates mid-build.
    let mut env: Vec<(String, String)> =
        Vec::with_capacity(ALLOWED_VARS.len() + TOOLCHAIN_VARS.len() + 1);
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

    // Toolchain caches are resolved, not merely copied: a name the parent did
    // not set is still forwarded when a host default can be discovered, so a
    // sandboxed `cargo build` finds the operator's crates cache and resolves
    // dependencies offline instead of asking `index.crates.io` for them.
    //
    // The secret filter is re-applied to every name, including the ones
    // resolved here, so widening the resolution below can never introduce a
    // credential-bearing name.
    for name in TOOLCHAIN_VARS {
        if is_sensitive_var(name) {
            continue;
        }
        let resolved = if *name == CARGO_HOME_VAR {
            host_cargo_home()
        } else {
            // No `~/.rustup` fallback: a host with no rustup has no toolchain
            // directory to share, and an invented path would be a silent lie.
            std::env::var_os(name)
                .filter(|value| !value.is_empty())
                .map(PathBuf::from)
        };
        if let Some(value) = resolved {
            env.push(((*name).to_string(), value.to_string_lossy().into_owned()));
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

/// Serializes tests that mutate the process environment.
///
/// `std::env::set_var` is process-global, and the harness runs unit tests on
/// parallel threads, so a test that overrides `HOME`/`CARGO_HOME` would otherwise
/// be observed by an unrelated test running at the same instant — in this module
/// *and* in `exec`, which spawns real children that read the same variables.
/// Holding this lock for the whole mutating test, rather than just around the
/// `set_var` calls, is what makes those tests atomic against their neighbours.
///
/// Exposed as a crate-visible test seam because sibling modules cannot reach a
/// test-private item.
#[cfg(test)]
pub(crate) static ENV_TEST_LOCK: Mutex<()> = Mutex::new(());

/// Acquire the environment lock, ignoring poisoning.
///
/// A panic in one mutating test leaves the mutex poisoned, but the invariant the
/// lock protects (a restored environment) is re-established by each test on its
/// own path, so a stale poison must not cascade into unrelated failures.
#[cfg(test)]
pub(crate) fn env_test_guard() -> MutexGuard<'static, ()> {
    ENV_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Run `body` with exclusive access to the process environment.
///
/// The lock is a std `Mutex`, so `body` must be synchronous: holding it across an
/// `.await` would park a runtime thread and can deadlock a multi-threaded
/// runtime, which is exactly what `clippy::await_holding_lock` rejects. A test
/// that needs both an environment override and an await therefore drives the
/// future to completion inside this closure (see the `runtime.block_on` call in
/// `exec`).
#[cfg(test)]
pub(crate) fn with_env_lock<T>(body: impl FnOnce() -> T) -> T {
    let _guard = env_test_guard();
    body()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::OnceLock;

    /// The process-wide real `HOME`, captured once on first use so a test that
    /// overrides it can put it back without depending on when it ran.
    static ORIGINAL_HOME_VALUE: OnceLock<String> = OnceLock::new();

    /// The real `HOME`, captured before any test overrides it.
    fn original_home() -> String {
        ORIGINAL_HOME_VALUE
            .get_or_init(|| std::env::var("HOME").unwrap_or_default())
            .clone()
    }

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
            "UNBEARERABLE",
            "UNBEARERABLE_MODE",
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
                ALLOWED_VARS.contains(name) || TOOLCHAIN_VARS.contains(name) || *name == HOME_VAR,
                "{name} escaped the allow-list"
            );
        }
        assert!(env.iter().any(|(k, _)| k == HOME_VAR), "HOME must be set");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn build_clean_environment_drops_secrets_from_parent() {
        let _guard = env_test_guard();
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
        let _guard = env_test_guard();
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
                ALLOWED_VARS.contains(&k.as_str())
                    || TOOLCHAIN_VARS.contains(&k.as_str())
                    || k == HOME_VAR,
                "{k} escaped the allow-list"
            );
        }
        assert!(
            isolated_home(&dir, &dir).is_dir(),
            "isolated HOME must be created"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The toolchain-cache contract: with the parent's `CARGO_HOME` unset, a
    /// host that *does* have a `~/.cargo` gets that directory forwarded, so a
    /// cargo command in a sandboxed worktree sees the populated registry instead
    /// of the empty isolated `HOME`.
    #[test]
    fn cargo_home_falls_back_to_the_host_dot_cargo() {
        let _guard = env_test_guard();
        let dir = unique_dir("cargo-home-fallback");
        let fake_home = dir.join("fake-home");
        std::fs::create_dir_all(fake_home.join(".cargo")).expect("create fake ~/.cargo");
        // The parent has no `CARGO_HOME`, so the fallback is the only thing that
        // can populate the variable. `HOME` is remapped independently, so this
        // synthetic layout is the *host's*, not the child's.
        // SAFETY: serialized against every other test that reads or writes the
        // process environment.
        unsafe {
            std::env::remove_var("CARGO_HOME");
            std::env::set_var("HOME", &fake_home);
        }

        let env = build_clean_environment(&dir, &dir);
        let cargo_home = env
            .iter()
            .find(|(k, _)| k == CARGO_HOME_VAR)
            .map(|(_, v)| v.clone());
        // SAFETY: as above; restore before asserting so a failure cannot leak.
        unsafe { std::env::set_var("HOME", original_home()) };

        let cargo_home = cargo_home.expect("CARGO_HOME must be forwarded from the host ~/.cargo");
        assert_eq!(cargo_home, fake_home.join(".cargo").to_string_lossy());
        // The whole point: the cache is the host's, not the sandbox's HOME.
        assert_ne!(
            PathBuf::from(&cargo_home),
            isolated_home(&dir, &dir).join(".cargo"),
            "the cache must not point back into the sandboxed HOME"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// No host cache means no forwarded variable: inventing a path that does not
    /// exist would be worse than staying quiet and letting Cargo apply its own
    /// default resolution.
    #[test]
    fn cargo_home_is_omitted_when_the_host_has_none() {
        let _guard = env_test_guard();
        let dir = unique_dir("cargo-home-absent");
        let fake_home = dir.join("fake-home");
        std::fs::create_dir_all(&fake_home).expect("create empty fake home");
        // SAFETY: serialized against every other test that reads or writes the
        // process environment.
        unsafe {
            std::env::remove_var("CARGO_HOME");
            std::env::set_var("HOME", &fake_home);
        }

        let env = build_clean_environment(&dir, &dir);
        // SAFETY: as above; restore before asserting.
        unsafe { std::env::set_var("HOME", original_home()) };

        assert!(
            !env.iter().any(|(k, _)| k == CARGO_HOME_VAR),
            "a host with no ~/.cargo must not get a CARGO_HOME at all"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// [`resolve_cargo_home`] is the policy, tested directly against synthetic
    /// layouts so every branch is covered whether or not the machine running the
    /// suite happens to have a `~/.cargo`.
    #[test]
    fn cargo_home_resolution_covers_every_branch() {
        let dir = unique_dir("cargo-home-pure");
        let home = dir.join("home");
        std::fs::create_dir_all(home.join(".cargo")).expect("create ~/.cargo");
        let with_cargo = home.join(".cargo");
        let without_cargo = dir.join("empty-home");
        std::fs::create_dir_all(&without_cargo).expect("create empty home");

        // Explicit wins over the fallback, and is not probed on disk.
        assert_eq!(
            resolve_cargo_home(Some(Path::new("/opt/cargo")), Some(&home)),
            Some(PathBuf::from("/opt/cargo"))
        );
        // An explicit but *empty* value is treated as unset, not as a directory
        // whose name is "".
        assert_eq!(
            resolve_cargo_home(Some(Path::new("")), Some(&home)),
            Some(with_cargo.clone())
        );
        // No explicit value: `~/.cargo` when it exists.
        assert_eq!(resolve_cargo_home(None, Some(&home)), Some(with_cargo));
        // `~/.cargo` missing, or `HOME` missing entirely: nothing to forward.
        assert_eq!(resolve_cargo_home(None, Some(&without_cargo)), None);
        assert_eq!(resolve_cargo_home(None, None), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// An explicit host `CARGO_HOME` wins over the `~/.cargo` fallback and is
    /// forwarded verbatim, so a non-default install layout keeps working.
    #[test]
    fn an_explicit_cargo_home_is_forwarded_verbatim() {
        let _guard = env_test_guard();
        let dir = unique_dir("cargo-home-explicit");
        let custom = dir.join("custom-cargo");
        std::fs::create_dir_all(&custom).expect("create custom cargo home");
        // SAFETY: serialized against every other test that reads or writes the
        // process environment.
        unsafe { std::env::set_var("CARGO_HOME", &custom) };
        let env = build_clean_environment(&dir, &dir);
        unsafe { std::env::remove_var("CARGO_HOME") };

        assert_eq!(
            env.iter()
                .find(|(k, _)| k == CARGO_HOME_VAR)
                .map(|(_, v)| v.clone()),
            Some(custom.to_string_lossy().into_owned()),
            "an explicit CARGO_HOME must be forwarded verbatim"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `RUSTUP_HOME` is forwarded like `CARGO_HOME`: set in the parent means
    /// verbatim, and unset means simply absent (there is no `~/.rustup`
    /// fallback, because a host with no toolchain manager is not an error).
    #[test]
    fn rustup_home_is_forwarded_when_set_and_absent_otherwise() {
        let _guard = env_test_guard();
        let dir = unique_dir("rustup-home");
        // SAFETY: serialized against every other test that reads or writes the
        // process environment.
        unsafe { std::env::set_var("RUSTUP_HOME", "/opt/rustup") };
        let set_env = build_clean_environment(&dir, &dir);
        unsafe { std::env::remove_var("RUSTUP_HOME") };
        let unset_env = build_clean_environment(&dir, &dir);

        assert_eq!(
            set_env
                .iter()
                .find(|(k, _)| k == RUSTUP_HOME_VAR)
                .map(|(_, v)| v.clone())
                .as_deref(),
            Some("/opt/rustup")
        );
        assert!(
            !unset_env.iter().any(|(k, _)| k == RUSTUP_HOME_VAR),
            "an unset RUSTUP_HOME must stay absent, not be invented"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The toolchain paths are cache *locations*, not credentials: they must
    /// survive the secret filter, or the forwarding would be dropped.
    #[test]
    fn toolchain_cache_vars_are_not_treated_as_secrets() {
        for name in [CARGO_HOME_VAR, RUSTUP_HOME_VAR] {
            assert!(
                !is_sensitive_var(name),
                "{name} is a cache location, not a credential"
            );
        }
    }

    /// Whatever the host's toolchain layout looks like, the forwarded result
    /// must still be exactly the allow-list plus `HOME` and carry no secret.
    #[test]
    fn toolchain_forwarding_keeps_the_allow_list_and_scrubbing() {
        let dir = unique_dir("toolchain-scrub");
        let env = build_clean_environment(&dir, &dir);
        for (k, _) in &env {
            assert!(
                !is_sensitive_var(k),
                "{k} leaked into the child env through toolchain forwarding"
            );
            assert!(
                ALLOWED_VARS.contains(&k.as_str())
                    || TOOLCHAIN_VARS.contains(&k.as_str())
                    || k == HOME_VAR,
                "{k} escaped the allow-list"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
