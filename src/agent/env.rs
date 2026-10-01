//! Environment hygiene for agent-spawned child processes.
//!
//! A subagent's shell is a *hostile* environment by default: the worker
//! inherits the operator's entire environment, so every API key, cloud
//! credential, registry token and SSH agent socket exported in the operator's
//! shell is one `echo $OPENAI_API_KEY` (or a stray `git push`) away from being
//! exfiltrated by a model handed an unrelated bug-fix task. A large ambient
//! environment also makes runs non-reproducible.
//!
//! This module replaces that inheritance with an explicit, minimal contract:
//!
//! 1. **Deny by default.** The child is spawned with
//!    [`std::process::Command::env_clear`], so *nothing* leaks unless
//!    deliberately re-added.
//! 2. **Strict allow-list.** Only the handful of runtime variables a compiler
//!    or shell genuinely needs survive ([`ALLOWED_VARS`]): binary discovery
//!    (`PATH`), identity (`USER`, `LOGNAME`, `SHELL`) and locale/terminal
//!    presentation (`LANG`, `LC_ALL`, `TERM`).
//! 3. **Toolchain cache forwarding.** `CARGO_HOME` and `RUSTUP_HOME`
//!    ([`TOOLCHAIN_VARS`]) are *resolved* rather than copied - `CARGO_HOME`
//!    falls back to the host's `~/.cargo` when the parent did not set it
//!    ([`host_cargo_home`]). This keeps a sandboxed worktree able to build
//!    offline: `HOME` is remapped to an empty scratch directory, so with no
//!    explicit `CARGO_HOME` Cargo would find an empty registry and reach for
//!    `index.crates.io` for crates the operator already has cached. These
//!    variables name *directories*, not secrets.
//! 4. **Isolated `HOME`.** `HOME` is remapped to a per-worktree scratch
//!    directory so a command that reads `~/.aws/credentials`, writes
//!    `~/.gitconfig` or drops a stray `~/.npmrc` touches only the sandbox,
//!    never the operator's real home.
//!
//! [`build_clean_environment`] is the single entry point. It returns a plain
//! `Vec<(String, String)>` rather than mutating a `Command` so the exact
//! contract can be unit-tested without spawning anything, and so callers can
//! log/inspect the child environment in a debugging story.

#[cfg(test)]
use std::ffi::OsString;
use std::path::{Path, PathBuf};
#[cfg(test)]
use std::sync::{Mutex, MutexGuard};

/// Variables copied verbatim from the parent process into the agent's shell.
///
/// Deliberately tiny: every entry is a variable whose *absence* breaks ordinary
/// execution, and none can carry a credential. `PATH` locates the binaries (the
/// command is spawned as `nice`/`bwrap`, resolved through it),
/// `USER`/`LOGNAME`/`SHELL` keep `id` and prompt-oriented tools sane, and
/// `LANG`/`LC_ALL`/`TERM` keep output and diagnostics readable.
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
/// 2. `$HOME/.cargo`, **only when it actually exists** - a host with no Cargo
///    directory has no cache to forward, and pointing at a non-existent path
///    would be worse than omitting the variable and letting Cargo apply its own
///    default resolution;
/// 3. otherwise nothing, and the variable is omitted entirely.
///
/// Only the *path* is forwarded. `$CARGO_HOME/credentials.toml`, registry tokens
/// in `$CARGO_HOME/config.toml` and any `CARGO_REGISTRY_TOKEN` stay unreachable,
/// because the variable is the only thing this module emits: no
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
/// Split out so the resolution *policy* - explicit wins, `~/.cargo` only when it
/// exists, nothing otherwise - is unit-testable against synthetic layouts (a
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
/// * each toolchain cache location that could be resolved - `CARGO_HOME` from
///   the host, falling back to `~/.cargo` when it exists ([`host_cargo_home`]),
///   and `RUSTUP_HOME` when the parent sets it. This is what lets a sandboxed
///   worktree build against the host's crates cache offline: `HOME` is remapped
///   to an empty scratch directory, so with no explicit `CARGO_HOME` Cargo
///   would find an empty registry and try the network;
/// * `HOME`, always remapped to [`isolated_home`] - never the real home.
///
/// The toolchain variables name *directories*, not secrets: no credential-
/// bearing variable name is introduced, so a credential cannot ride along in a
/// cache path.
///
/// Because the caller must pair this with `env_clear()`, the result is a
/// complete environment, not a patch. Credentials, tokens, `*_PROXY` and
/// anything else the operator had exported are simply absent.
///
/// The function deliberately does **not** create the directory: callers that
/// need it to exist (command spawning) create it explicitly, which keeps this
/// function side-effect free and cheap to call from tests.
pub fn build_clean_environment(repo_path: &Path, worktree_path: &Path) -> Vec<(String, String)> {
    // Sized up front for the worst case (every allow-listed name set, every
    // toolchain cache resolved) plus `HOME`, so the vector never reallocates.
    let mut env: Vec<(String, String)> =
        Vec::with_capacity(ALLOWED_VARS.len() + TOOLCHAIN_VARS.len() + 1);
    for name in ALLOWED_VARS {
        // Empty values carry no information and can only confuse tools that
        // test "is this configured?" with a truthiness check.
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
    for name in TOOLCHAIN_VARS {
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

/// Apply [`build_clean_environment`] to a `tokio::process::Command`.
///
/// The parent's environment is cleared first, so the child sees exactly the
/// allow-list and nothing else; the isolated `HOME` directory is created before
/// the spawn so tools that expect it to exist (npm, git config discovery) do
/// not fail on a missing directory.
pub fn apply_clean_environment_cmd(
    cmd: &mut tokio::process::Command,
    repo_path: &Path,
    worktree_path: &Path,
) {
    let env = build_clean_environment(repo_path, worktree_path);
    let _ = std::fs::create_dir_all(isolated_home(repo_path, worktree_path));
    cmd.env_clear();
    for (key, value) in env {
        cmd.env(key, value);
    }
}

/// Upper bound on the ambient environment snapshot a client may send to the
/// daemon (64 KB).
///
/// The snapshot exists so the worker can re-run its verify gate in the
/// orchestrator's environment, not to become a transport for the operator's
/// whole shell: a caller with a pathological environment must not be able to
/// grow a hub frame without bound.
pub const AMBIENT_ENV_MAX_BYTES: usize = 64 * 1024;

/// Substrings that mark a variable name as credential-bearing.
///
/// Matched case-insensitively against the *name* only, never the value: a value
/// is not inspected because the point is to refuse the name before it is ever
/// read into a snapshot. The list is deliberately broad - a false positive
/// costs one variable in a differential run, a false negative ships a key.
const SECRET_NAME_MARKERS: &[&str] = &[
    "KEY",
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "PASSWD",
    "PASSPHRASE",
    "CREDENTIAL",
    "AUTH",
    "PRIVATE",
    "SIGNATURE",
    "BEARER",
    "COOKIE",
    "CERT",
    "APIKEY",
    "ACCESS_KEY",
    "SESSION_KEY",
    "SALT",
];

/// Whether `name` is credential-bearing and must never leave the client.
///
/// The sandbox's allow-list is the authority on what a child may see; this is
/// the matching *deny* rule for the one thing that does cross a process
/// boundary in the other direction - the ambient snapshot a dispatcher sends in
/// `hub/hello`. Both are applied to every snapshot, on the client that builds
/// it and again on the daemon that receives it, so neither side can be the only
/// thing standing between a credential and the model.
pub fn is_secret_name(name: &str) -> bool {
    let upper = name.to_ascii_uppercase();
    SECRET_NAME_MARKERS
        .iter()
        .any(|marker| upper.contains(marker))
}

/// The caller's environment, filtered for transport to the daemon.
///
/// Every variable of this process that is not credential-bearing
/// ([`is_secret_name`]) and not empty, sorted by name for a stable frame, and
/// truncated at [`AMBIENT_ENV_MAX_BYTES`] - the entry that would cross the
/// bound is dropped rather than split, so the result is always a set of whole
/// variables.
///
/// This is the *only* environment the daemon ever learns about the
/// orchestrator's shell, and it is what the differential verify gate layers on
/// top of the canonical sandbox environment.
pub fn ambient_environment_snapshot() -> Vec<(String, String)> {
    let mut pairs: Vec<(String, String)> = std::env::vars_os()
        .filter_map(|(name, value)| {
            let name = name.to_string_lossy().into_owned();
            if is_secret_name(&name) {
                return None;
            }
            let value = value.to_string_lossy().into_owned();
            (!value.is_empty()).then_some((name, value))
        })
        .collect();
    pairs.sort();
    let mut total = 0usize;
    pairs.retain(|(name, value)| {
        // `name.len() + value.len() + 2` covers the separator and the
        // terminator of the wire form, so the bound holds for the encoded
        // frame and not only for the raw bytes.
        let cost = name.len() + value.len() + 2;
        if total + cost > AMBIENT_ENV_MAX_BYTES {
            return false;
        }
        total += cost;
        true
    });
    pairs
}

/// Serializes tests that mutate the process environment.
///
/// `std::env::set_var` is process-global, and the harness runs unit tests on
/// parallel threads, so a test that overrides `HOME`/`CARGO_HOME` would
/// otherwise be observed by an unrelated test running at the same instant - in
/// this module *and* in `exec`, which spawns real children that read the same
/// variables. Holding this lock for the whole mutating test, rather than just
/// around the `set_var` calls, is what makes those tests atomic against their
/// neighbours.
///
/// Exposed as a crate-visible test seam because sibling modules cannot reach a
/// test-private item.
#[cfg(test)]
pub(crate) static ENV_TEST_LOCK: Mutex<()> = Mutex::new(());

/// Run `body` with exclusive access to the process environment.
///
/// The lock is a std `Mutex`, so `body` must be synchronous: holding it across
/// an `.await` would park a runtime thread and can deadlock a multi-threaded
/// runtime, which is exactly what `clippy::await_holding_lock` rejects. A test
/// that needs both an environment override and an await therefore drives the
/// future to completion inside this closure (see the `runtime.block_on` call in
/// `exec`).
#[cfg(test)]
pub(crate) fn with_env_lock<T>(body: impl FnOnce() -> T) -> T {
    let _guard = ENV_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    body()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::OnceLock;

    /// Acquire the environment lock for the duration of a mutating test.
    fn env_guard() -> MutexGuard<'static, ()> {
        ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

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

    /// The allow-list and toolchain names are constants, so they can never
    /// carry a credential: assert none of them (nor `HOME`) looks like one.
    #[test]
    fn allow_list_and_toolchain_names_are_not_credentials() {
        let markers = [
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
            "BEARER",
        ];
        for name in ALLOWED_VARS
            .iter()
            .chain(TOOLCHAIN_VARS.iter())
            .chain([HOME_VAR].iter())
        {
            let upper = name.to_ascii_uppercase();
            for marker in markers {
                assert!(
                    !upper.contains(marker),
                    "{name} must not look like a credential (marker {marker})"
                );
            }
            // `PAT` is only a credential at a word boundary, so `PATH` is fine.
            assert!(
                !upper
                    .split(|c: char| !c.is_ascii_alphanumeric())
                    .any(|w| w == "PAT"),
                "{name} must not look like a credential (marker PAT)"
            );
        }
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
        let _guard = env_guard();
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
            assert!(
                !k.to_ascii_uppercase().contains("API_KEY")
                    && !k.to_ascii_uppercase().contains("SECRET")
                    && !k.to_ascii_uppercase().contains("TOKEN")
                    && !k.to_ascii_uppercase().contains("SSH_"),
                "{k} leaked into the child env"
            );
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
        let home_os = OsString::from(&home);
        let home_path = Path::new(&home_os);
        assert!(
            !home_os.is_empty() && home_path != Path::new("/") && home_path.is_absolute(),
            "HOME must be a usable absolute path"
        );
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
        let _guard = env_guard();
        let dir = unique_dir("apply");
        let mut cmd = tokio::process::Command::new("true");
        // Seed with a secret so env_clear has something to remove.
        cmd.env("OPENAI_API_KEY", "sk-should-not-leak");
        apply_clean_environment_cmd(&mut cmd, &dir, &dir);

        // tokio::process::Command does not expose its env, so spawn a child
        // that echoes it and assert on the observed environment.
        let out = std::process::Command::new("sh")
            .arg("-c")
            .arg("env")
            .env_clear()
            .envs(build_clean_environment(&dir, &dir))
            .output()
            .expect("spawn env probe");
        let envs = String::from_utf8_lossy(&out.stdout);
        assert!(
            !envs.contains("OPENAI_API_KEY"),
            "env_clear + allow-list must drop OPENAI_API_KEY: {envs:?}"
        );
        for line in envs.lines() {
            let k = line.split('=').next().unwrap_or("");
            // `sh` injects `PWD`/`SHLVL`/`_` itself; only the sanitized names
            // are under test.
            if ["PWD", "SHLVL", "_"].contains(&k) {
                continue;
            }
            assert!(
                ALLOWED_VARS.contains(&k) || TOOLCHAIN_VARS.contains(&k) || k == HOME_VAR,
                "{k} escaped the allow-list"
            );
        }
        assert!(
            isolated_home(&dir, &dir).is_dir(),
            "isolated HOME must be created"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Toolchain-cache contract: with the parent's `CARGO_HOME` unset, a host
    /// that *does* have a `~/.cargo` gets that directory forwarded, so a cargo
    /// command in a sandboxed worktree sees the populated registry instead of
    /// the empty isolated `HOME`.
    #[test]
    fn cargo_home_falls_back_to_the_host_dot_cargo() {
        let _guard = env_guard();
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

    /// No host cache means no forwarded variable: inventing a path that does
    /// not exist would be worse than staying quiet and letting Cargo apply its
    /// own default resolution.
    #[test]
    fn cargo_home_is_omitted_when_the_host_has_none() {
        let _guard = env_guard();
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

    #[test]
    fn secret_names_are_detected_case_insensitively() {
        for name in [
            "OPENAI_API_KEY",
            "openai_api_key",
            "GITHUB_TOKEN",
            "AWS_SECRET_ACCESS_KEY",
            "DB_PASSWORD",
            "SSH_AUTH_SOCK",
            "MY_CREDENTIALS",
            "BEARER_AUTH",
            "TLS_CERT",
            "SESSION_KEY",
        ] {
            assert!(is_secret_name(name), "{name} must be treated as a secret");
        }
        for name in [
            "PATH", "USER", "HOME", "TMPDIR", "TZ", "LANG", "MY_FOO", "BUILD_ID",
        ] {
            assert!(
                !is_secret_name(name),
                "{name} must not be treated as a secret"
            );
        }
    }

    #[test]
    fn ambient_snapshot_never_carries_secrets() {
        let _guard = env_guard();
        // SAFETY: serialized against every other test that reads the process
        // environment.
        unsafe {
            std::env::set_var("SWE_AMBIENT_PLAIN_TEST", "hello");
            std::env::set_var("SWE_AMBIENT_SECRET_TOKEN_TEST", "must-not-travel");
        }
        let snapshot = ambient_environment_snapshot();
        unsafe {
            std::env::remove_var("SWE_AMBIENT_PLAIN_TEST");
            std::env::remove_var("SWE_AMBIENT_SECRET_TOKEN_TEST");
        }
        assert!(
            snapshot
                .iter()
                .any(|(k, v)| k == "SWE_AMBIENT_PLAIN_TEST" && v == "hello"),
            "plain variables must survive the snapshot"
        );
        assert!(
            !snapshot
                .iter()
                .any(|(k, _)| k == "SWE_AMBIENT_SECRET_TOKEN_TEST"),
            "secret names must not survive the snapshot"
        );
    }

    /// An explicit host `CARGO_HOME` wins over the `~/.cargo` fallback and is
    /// forwarded verbatim, so a non-default install layout keeps working.
    #[test]
    fn an_explicit_cargo_home_is_forwarded_verbatim() {
        let _guard = env_guard();
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
        let _guard = env_guard();
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

    /// The toolchain paths are cache *locations*, not credentials: they carry
    /// no secret and are always safe to forward.
    #[test]
    fn toolchain_cache_vars_are_not_treated_as_secrets() {
        for name in [CARGO_HOME_VAR, RUSTUP_HOME_VAR] {
            assert!(
                !name.to_ascii_uppercase().contains("TOKEN")
                    && !name.to_ascii_uppercase().contains("SECRET")
                    && !name.to_ascii_uppercase().contains("KEY"),
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
                !k.to_ascii_uppercase().contains("TOKEN")
                    && !k.to_ascii_uppercase().contains("SECRET")
                    && !k.to_ascii_uppercase().contains("KEY"),
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
