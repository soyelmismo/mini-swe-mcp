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

/// Whether `b` can appear in an environment variable name.
fn is_ident_byte(b: u8) -> bool {
    b.is_ascii_uppercase() || b.is_ascii_digit() || b == b'_'
}

/// A name is a candidate when it is all-caps, so a local `let Some(x) = ...`
/// binding and a `Some(...)` value never look like a variable.
fn looks_like_env_name(candidate: &str) -> bool {
    !candidate.is_empty() && candidate.bytes().all(is_ident_byte)
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
        let trimmed = line.trim();
        // A doc comment or prose mentioning a name is not a declaration.
        if !trimmed.starts_with("const ") && !trimmed.starts_with("pub const ") {
            continue;
        }
        // `const NAME_ENV: &str = "VALUE";` — the name is the last token before
        // the `:`, and the value is the literal right of `= "`. A doc comment
        // joined to the same logical line is split off by `line`, and a constant
        // that is not an environment name carries no `_ENV`/`_VAR` suffix, so it
        // is skipped.
        let Some((ident, after_colon)) = trimmed.split_once(':') else {
            continue;
        };
        let ident = ident.split_whitespace().last().unwrap_or(ident);
        let Some(value) = after_colon
            .split_once("= \"")
            .and_then(|(_, v)| v.split('"').next())
        else {
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

    for (argument, quoted) in read_arguments(&code) {
        if quoted {
            // A literal argument: the name is right there.
            if let Some((name, _)) = read_string_literal(&argument)
                && looks_like_env_name(&name)
            {
                names.insert(name);
            }
            continue;
        }
        // A bare argument: the name lives in a `*_ENV` / `*_VAR` declaration,
        // which is exactly what `constants` resolved above.
        if let Some(name) = constants.get(argument.trim())
            && looks_like_env_name(name)
        {
            names.insert(name.clone());
        }
    }
    names
}

/// What every environment read in `code` passes as its first argument: the
/// text between the `(` and the matching `)`, and whether it was a quoted
/// string literal or a bare identifier.
///
/// The scan walks `char_indices`, so every index it slices on is a character
/// boundary even in a source file that carries non-ASCII text in a string
/// literal.
fn read_arguments(code: &str) -> Vec<(String, bool)> {
    // Longest first: `env::var_os` also starts with `env::var`.
    // Both the crate-local `env::var` and the fully-qualified `std::env::var`
    // are matched; the search is a prefix, so `std::env::var` also matches the
    // bare form once `std::env::var` is listed first.
    const READERS: &[&str] = &[
        "env_parse",
        "std::env::var_os",
        "env::var_os",
        "std::env::var",
        "env::var",
    ];
    let offsets: Vec<usize> = code
        .char_indices()
        .map(|(index, _)| index)
        .chain(std::iter::once(code.len()))
        .collect();
    let bytes = code.as_bytes();
    let mut out = Vec::new();
    let mut cursor = 0;
    while cursor + 1 < offsets.len() {
        let start = offsets[cursor];
        let reader = READERS
            .iter()
            .find(|reader| code[start..].starts_with(**reader));
        let Some(reader) = reader else {
            cursor += 1;
            continue;
        };
        // Step over an optional turbofish (`::<u64>`), and any whitespace, to
        // the `(`. A turbofish is `:<...>`, so the scan consumes a `:` only
        // while the next bytes are `<...>`; a `:` that is not the start of a
        // turbofish (none is expected here) would stop the search, which is
        // correct because these readers are never followed by another `:`.
        let mut at = start + reader.len();
        let open = loop {
            if at >= bytes.len() {
                break None;
            }
            if bytes[at] == b'(' {
                break Some(at);
            }
            if bytes[at].is_ascii_whitespace() {
                at += 1;
                continue;
            }
            // A turbofish is `::<T>`; consume both leading colons and the
            // `<...>` that follows, so the scan reaches the `(` either way.
            if bytes[at] == b':' {
                let mut at2 = at;
                while at2 < bytes.len() && bytes[at2] == b':' {
                    at2 += 1;
                }
                if bytes.get(at2) == Some(&b'<') {
                    match code[at2..].find('>') {
                        Some(close) => {
                            at = at2 + close + 1;
                            continue;
                        }
                        None => break None,
                    }
                }
            }
            break None;
        };
        let Some(open) = open else {
            cursor += 1;
            continue;
        };
        // The argument runs to the first `)` or `,` at the top level of the
        // call, which for these single-argument reads is the closing paren.
        let end = code[open + 1..]
            .find([')', ','])
            .map(|offset| open + 1 + offset)
            .unwrap_or(bytes.len());
        let argument = code[open + 1..end].trim();
        // A quoted string literal is a name written inline; a bare token is a
        // `*_ENV` / `*_VAR` constant resolved through `declared_constants`.
        let quoted = argument.starts_with('"') && argument.ends_with('"');
        out.push((argument.to_string(), quoted));
        cursor = offsets
            .iter()
            .position(|offset| *offset >= end)
            .unwrap_or(offsets.len() - 1);
    }
    out
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
