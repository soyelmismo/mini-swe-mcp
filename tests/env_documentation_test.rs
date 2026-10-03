//! Every environment variable the crate reads must be documented for operators
//! or explicitly declared a test-only hook.
//!
//! An operator's only map of these settings is `.env.example` and `help env`.
//! A variable added to the code and forgotten in both is invisible: the
//! feature exists, the default is invisible, and the knob nobody knows about is
//! the knob nobody sets. This test scans `src/` for the ways a name is read --
//! the [`env_parse`] helper, `std::env::var` / `var_os`, and the `*_ENV` /
//! `*_VAR` string constants that keep the name in one place -- and requires each
//! name to be either documented in `.env.example` or named here as a test-only
//! hook that operators are deliberately not told about.
//!
//! The scan is source-level and reads the tree read-only; it never spawns
//! anything and never mutates the environment, so it is hermetic.
//!
//! [`env_parse`]: mini_swe_mcp::config::env_parse

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

/// The crate root the test binary was compiled from.
fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Environment variables that exist for the test suite, not for operators.
///
/// These steer a test into a state a real deployment never reaches. Naming one
/// in `.env.example` would document a knob that has no production meaning, so
/// each has to justify itself here instead.
const TEST_ONLY_HOOKS: &[(&str, &str)] = &[
    (
        "LL_TARGET",
        "the Landlock probe's target directory; only a landlock test sets it",
    ),
    (
        "LL_WORKTREE",
        "the Landlock probe's worktree; only a landlock test sets it",
    ),
    (
        "MINI_SWE_EXEC_LANDLOCK_PROBE",
        "makes the exec sandbox re-exec itself as a Landlock probe; a test-only seam",
    ),
    (
        "MINI_SWE_FAKE_BUILD_TS",
        "stamps a fake build timestamp so the client-newer-than-hub warning can be exercised",
    ),
    (
        "MINI_SWE_FAKE_VERSION",
        "reports a fake --version so the version mismatch path can be exercised",
    ),
    (
        "MINI_SWE_HUB_RECOVERY_DELAY_MS",
        "delays hub recovery so a reconnecting client can be raced against it",
    ),
];

/// Host variables the crate reads or forwards, not operator configuration.
///
/// These name the operator's own machine: the shell's search path, the home and
/// toolchain directories a sandbox forwards, the terminal and locale it
/// presents, and the standard build/debug toggles it forwards untouched. None is
/// a setting of this server, and documenting them in `.env.example` would read
/// as advice to set them.
const HOST_VARS: &[&str] = &[
    "CARGO_HOME",
    "CARGO_INCREMENTAL",
    "CARGO_PROFILE_DEV_DEBUG",
    "CARGO_PROFILE_TEST_DEBUG",
    "COLUMNS",
    "HOME",
    "LANG",
    "LC_ALL",
    "LOGNAME",
    "PATH",
    "RUSTUP_HOME",
    "SHELL",
    "TERM",
    "USER",
    "WORKER_BUILD_DEBUG",
];

/// A name is a candidate when it is all-caps, so a local `let Some(x) = ...`
/// binding and a `Some(...)` value never look like a variable.
fn looks_like_env_name(candidate: &str) -> bool {
    !candidate.is_empty()
        && candidate
            .bytes()
            .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit() || b == b'_')
}

/// Every `.rs` file under `src/`, sorted so a failure names the same first file
/// on every run.
fn source_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = std::fs::read_dir(dir).unwrap_or_else(|e| panic!("read {}: {e}", dir.display()));
    for entry in entries {
        let path = entry.expect("read a src/ directory entry").path();
        if path.is_dir() {
            source_files(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(path);
        }
    }
    out.sort();
}

/// The `*_ENV` / `*_VAR` constants declared in `source`, mapped to the name they
/// hold, so a call site that passes the constant is resolved to a string.
///
/// Only the declarations the crate itself writes are matched, and the value has
/// to be a string literal: a constant that is not an environment name (a prefix,
/// a command) simply does not match and is not reported.
fn declared_constants(source: &str) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    for line in source.lines() {
        let line = line.trim();
        // A doc comment or prose mentioning a name is not a declaration.
        if !line.starts_with("const ") && !line.starts_with("pub const ") {
            continue;
        }
        let Some((name, value)) = line.split_once(": &str = \"") else {
            continue;
        };
        let (ident, value) = name.rsplit_once(' ').unwrap_or((name, value));
        let Some(value) = value.split('"').next() else {
            continue;
        };
        if ident.ends_with("_ENV") || ident.ends_with("_VAR") {
            out.insert(ident.to_string(), value.to_string());
        }
    }
    out
}

/// The environment variable names `source` reads, with the ones that resolve
/// through a constant already substituted.
///
/// Comments are stripped first, so a name that only appears in a doc comment
/// (which is how most of them are *described*) is not mistaken for a read.
fn env_names_read_in(source: &str) -> BTreeSet<String> {
    let code: String = source
        .lines()
        .filter(|line| {
            let trimmed = line.trim();
            !trimmed.starts_with("//")
        })
        .collect::<Vec<_>>()
        .join("\n");
    let constants = declared_constants(&code);
    let mut names = BTreeSet::new();

    for name in literal_arguments(&code) {
        if looks_like_env_name(&name) {
            names.insert(name);
        }
    }
    // Call sites that pass a constant resolve through the declarations above.
    for ident in call_identifiers(&code) {
        if let Some(name) = constants.get(&ident)
            && looks_like_env_name(name)
        {
            names.insert(name.clone());
        }
    }
    names
}

/// The string literal each environment read passes, with a turbofish stripped
/// (`env_parse::<u64>("NAME")`).
///
/// Deliberately narrow: it only looks inside the parentheses of the crate's own
/// read paths, so an unrelated string constant cannot be reported as a variable.
fn literal_arguments(code: &str) -> Vec<String> {
    const READERS: &[&str] = &["env_parse", "env::var", "env::var_os"];
    let bytes = code.as_bytes();
    let mut out = Vec::new();
    let mut index = 0;
    while index < code.len() {
        let rest = &code[index..];
        let Some(reader) = READERS.iter().find(|r| rest.starts_with(**r)) else {
            index += 1;
            continue;
        };
        let mut cursor = index + reader.len();
        // Step over `::<T>`, then whitespace, then the opening parenthesis.
        while cursor < bytes.len() && (bytes[cursor] == b':' || bytes[cursor].is_ascii_whitespace())
        {
            cursor += 1;
        }
        if bytes.get(cursor) == Some(&b'(') {
            if let Some((argument, _)) = read_string_literal(&code[cursor + 1..]) {
                out.push(argument);
            }
        }
        index += reader.len();
    }
    out
}

/// The identifier each environment read passes instead of a literal
/// (`env_parse(POLL_ENV)`), for a constant the caller resolves elsewhere.
fn call_identifiers(code: &str) -> Vec<String> {
    const READERS: &[&str] = &["env_parse", "env::var", "env::var_os"];
    let bytes = code.as_bytes();
    let mut out = Vec::new();
    let mut index = 0;
    while index < code.len() {
        let rest = &code[index..];
        let Some(reader) = READERS.iter().find(|r| rest.starts_with(**r)) else {
            index += 1;
            continue;
        };
        let mut cursor = index + reader.len();
        while cursor < bytes.len() && (bytes[cursor] == b':' || bytes[cursor].is_ascii_whitespace())
        {
            cursor += 1;
        }
        if bytes.get(cursor) == Some(&b'(') {
            let argument = &code[cursor + 1..];
            let trimmed = argument.trim_start().trim_start_matches(|c: char| {
                c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_'
            });
            if trimmed.len() < argument.len() {
                out.push(trimmed[..ident_len(trimmed)].to_string());
            }
        }
        index += reader.len();
    }
    out
}

/// The length of the leading all-caps identifier in `text`.
fn ident_len(text: &str) -> usize {
    text.bytes()
        .take_while(|b| b.is_ascii_uppercase() || b.is_ascii_digit() || *b == b'_')
        .count()
}

/// The first `"..."` literal in `text`, with no escape in it (a name never has
/// one).
fn read_string_literal(text: &str) -> Option<(String, usize)> {
    let start = text.find('"')?;
    let rest = &text[start + 1..];
    let end = rest.find('"')?;
    if rest[..end].contains('\\') {
        return None;
    }
    Some((rest[..end].to_string(), start + 1 + end + 1))
}

/// The names `.env.example` documents.
fn documented_names() -> BTreeSet<String> {
    let text =
        std::fs::read_to_string(repo_root().join(".env.example")).expect("read .env.example");
    text.split(|c: char| !(c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_'))
        .filter(|word| looks_like_env_name(word))
        .map(str::to_string)
        .collect()
}

/// Every environment name the crate reads, with the file that reads it.
fn read_names_by_file() -> BTreeMap<String, Vec<String>> {
    let mut files = Vec::new();
    source_files(&repo_root().join("src"), &mut files);
    let mut by_name: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for path in files {
        let source = std::fs::read_to_string(&path).expect("read a src/ file");
        for name in env_names_read_in(&source) {
            let relative = path
                .strip_prefix(repo_root())
                .unwrap_or(&path)
                .display()
                .to_string();
            by_name.entry(name).or_default().push(relative);
        }
    }
    by_name
}

/// The names the crate reads that no operator is told about.
fn undocumented_names() -> BTreeSet<String> {
    let documented = documented_names();
    read_names_by_file()
        .into_keys()
        .filter(|name| !documented.contains(name))
        .filter(|name| !TEST_ONLY_HOOKS.iter().any(|(hook, _)| hook == name))
        .filter(|name| !HOST_VARS.contains(&name.as_str()))
        .collect()
}

#[test]
fn every_env_var_read_is_documented_or_a_declared_test_hook() {
    let undocumented = undocumented_names();
    let read = read_names_by_file();
    assert!(
        undocumented.is_empty(),
        "these environment variables are read by the code but an operator is \
         never told about them: add each to .env.example (and to `help env`) \
         as an operator setting, or to TEST_ONLY_HOOKS in this test if it is a \
         test-only hook. {} undocumented: {undocumented:?}; the crate reads \
         {} names in total",
        undocumented.len(),
        read.len(),
    );
}

/// The guard is load-bearing in both directions: a name that disappears from
/// the code must not stay in the allowlist forever, and the allowlist must not
/// be a place to park real operator settings.
#[test]
fn the_test_only_allowlist_is_exactly_the_declared_hooks() {
    let read = read_names_by_file();
    for (hook, why) in TEST_ONLY_HOOKS {
        assert!(
            read.contains_key(*hook),
            "{hook} is allowlisted as a test-only hook but nothing in src/ reads \
             it any more; remove it from TEST_ONLY_HOOKS (it was documented as: \
             {why})"
        );
        assert!(
            !documented_names().contains(*hook),
            "{hook} is a test-only hook and must not be documented for operators"
        );
    }
}
