//! The bounded context pack in the worker's first message.
//!
//! The pack is the harness's substitute for the worker's first dozen
//! read-only turns: it names the paths the task mentions and locates the
//! symbols it quotes, bounded, and it is absent when the task names nothing.

use std::fs;

use mini_swe_mcp::pool::{context_pack, extract_identifiers, extract_paths, outline_file};

use crate::common::TempDir;

const RUST_SAMPLE: &str = r#"//! Sample module.

use std::path::Path;

pub struct Worker {
    id: String,
}

impl Worker {
    pub fn new(id: String) -> Self {
        Self { id }
    }
}

pub enum Phase {
    Idle,
    Running,
}

pub trait Runner {
    fn run(&self);
}

pub mod nested;

pub const MAX_TURNS: usize = 45;

fn helper() {}
"#;

const PYTHON_SAMPLE: &str = r#"""Sample module."""

import os


class Worker:
    def __init__(self, worker_id):
        self.worker_id = worker_id

    def run(self):
        pass


def dispatch(task):
    return task


async def collect(worker_id):
    return worker_id
"#;

#[test]
fn the_extractor_finds_named_paths_and_backticked_identifiers() {
    let tmp = TempDir::new_in_tmp("context-pack-extract");
    let src = tmp.subdir("src");
    fs::write(src.join("worker.rs"), RUST_SAMPLE).unwrap();

    let task = "Edit `src/worker.rs`: the `Worker` struct needs a `run` method. See `Phase`.";
    let paths = extract_paths(task, tmp.path());
    assert_eq!(paths, vec![std::path::PathBuf::from("src/worker.rs")]);

    let idents = extract_identifiers(task);
    assert!(idents.contains(&"Worker".to_string()), "got {idents:?}");
    assert!(idents.contains(&"Phase".to_string()), "got {idents:?}");
}

#[test]
fn the_outliner_lists_rust_top_level_items_with_line_numbers() {
    let tmp = TempDir::new_in_tmp("context-pack-rust");
    let src = tmp.subdir("src");
    let file = src.join("worker.rs");
    fs::write(&file, RUST_SAMPLE).unwrap();

    let outline = outline_file(&file);
    let joined = outline.join("\n");
    for expected in [
        "pub struct Worker",
        "impl Worker",
        "pub enum Phase",
        "pub trait Runner",
        "pub mod nested",
        "pub const MAX_TURNS",
        "fn helper",
    ] {
        assert!(
            joined.contains(expected),
            "missing {expected} in:\n{joined}"
        );
    }
    // Line numbers are present and 1-based: the struct starts at line 5.
    assert!(
        outline
            .iter()
            .any(|l| l.starts_with("5:") && l.contains("pub struct Worker")),
        "got {outline:?}"
    );
    // Indented method lines are not top-level items.
    assert!(!joined.contains("pub fn new"), "got:\n{joined}");
}

#[test]
fn the_outliner_lists_python_top_level_items() {
    let tmp = TempDir::new_in_tmp("context-pack-python");
    let file = tmp.path().join("worker.py");
    fs::write(&file, PYTHON_SAMPLE).unwrap();

    let outline = outline_file(&file);
    let joined = outline.join("\n");
    for expected in ["class Worker", "def dispatch", "async def collect"] {
        assert!(
            joined.contains(expected),
            "missing {expected} in:\n{joined}"
        );
    }
    // Methods are indented, so they are not top-level items.
    assert!(!joined.contains("def __init__"), "got:\n{joined}");
}

#[test]
fn the_pack_names_paths_and_locates_identifiers() {
    let tmp = TempDir::new_in_tmp("context-pack-render");
    let src = tmp.subdir("src");
    fs::write(src.join("worker.rs"), RUST_SAMPLE).unwrap();

    let task = "Edit `src/worker.rs`: give `Worker` a `run` method.";
    let pack = context_pack(task, tmp.path()).expect("the task names things, so a pack");
    assert!(
        pack.starts_with("Where things are (generated, may be incomplete):"),
        "got:\n{pack}"
    );
    assert!(pack.contains("- src/worker.rs"), "got:\n{pack}");
    assert!(pack.contains("pub struct Worker"), "got:\n{pack}");
    assert!(pack.contains("`Worker`"), "got:\n{pack}");
    // The identifier hit points at the struct's line in the named file.
    assert!(pack.contains("src/worker.rs:5:"), "got:\n{pack}");
}

#[test]
fn the_pack_uses_git_grep_inside_a_checkout() {
    let tmp = TempDir::new_in_tmp("context-pack-git");
    let src = tmp.subdir("src");
    fs::write(src.join("worker.rs"), RUST_SAMPLE).unwrap();
    common::git(tmp.path(), &["init", "-q"]);
    common::git(tmp.path(), &["add", "-A"]);
    common::git(
        tmp.path(),
        &[
            "-c",
            "user.email=t@t",
            "-c",
            "user.name=t",
            "commit",
            "-qm",
            "init",
        ],
    );

    let task = "What calls `helper`?";
    let pack = context_pack(task, tmp.path()).expect("the identifier is locatable");
    assert!(pack.contains("src/worker.rs:"), "got:\n{pack}");
    assert!(pack.contains("fn helper"), "got:\n{pack}");
}

#[test]
fn the_pack_is_capped() {
    let tmp = TempDir::new_in_tmp("context-pack-cap");
    let src = tmp.subdir("src");
    // A large file with many top-level items, so the outline alone overflows.
    let mut big = String::new();
    for i in 0..200 {
        big.push_str(&format!("pub fn function_number_{i}() {{}}\n"));
    }
    fs::write(src.join("big.rs"), &big).unwrap();
    // Several identifiers, each with several hits.
    let task = "Edit `src/big.rs`: `function_number_1`, `function_number_2`, `function_number_3`, \
                `function_number_4`, `function_number_5`, `function_number_6`, `function_number_7`, \
                `function_number_8`, `function_number_9`, `function_number_10`.";
    let pack = context_pack(task, tmp.path()).expect("the task names things");
    assert!(
        pack.len() <= mini_swe_mcp::pool::PACK_CAP_BYTES,
        "pack is {} bytes",
        pack.len()
    );
}

#[test]
fn a_task_naming_nothing_yields_no_pack() {
    let tmp = TempDir::new_in_tmp("context-pack-empty");
    let src = tmp.subdir("src");
    fs::write(src.join("worker.rs"), RUST_SAMPLE).unwrap();

    let pack = context_pack("Fix the flaky retry logic, please.", tmp.path());
    assert!(pack.is_none(), "got:\n{pack:?}");
}

/// A task that names a path escaping the worktree must not put it in the
/// pack: `root.join(token)` on `../secret` or an absolute path reads outside
/// the checkout, so the extractor only keeps paths that resolve inside it.
#[test]
fn traversal_paths_outside_the_worktree_are_ignored() {
    let tmp = TempDir::new_in_tmp("context-pack-traversal");
    let src = tmp.subdir("src");
    fs::write(src.join("worker.rs"), RUST_SAMPLE).unwrap();
    // A real file outside the worktree the token points at, so the test fails
    // if the extractor ever reads through the escape.
    let outside = tmp
        .path()
        .parent()
        .expect("the scratch base has a parent")
        .join("context-pack-outside-secret.txt");
    fs::write(&outside, "TOP SECRET OUTSIDE THE WORKTREE\n").unwrap();

    let task = "Read ../context-pack-outside-secret.txt and src/worker.rs for context.";
    let paths = extract_paths(task, tmp.path());
    assert!(
        !paths.iter().any(|p| p.to_string_lossy().contains("..")),
        "no escaping path may be extracted: {paths:?}"
    );
    assert_eq!(paths, vec![std::path::PathBuf::from("src/worker.rs")]);

    let pack = context_pack(task, tmp.path()).expect("the in-tree path still packs");
    assert!(
        !pack.contains("TOP SECRET"),
        "the pack must not carry bytes from outside the worktree:\n{pack}"
    );
    assert!(pack.contains("src/worker.rs"), "got:\n{pack}");

    let _ = std::fs::remove_file(&outside);
}

/// Absolute paths and interior `..` segments must not escape the worktree
/// either: `root.join("/abs")` replaces the base and `root.join("a/../../b")`
/// resolves outside it, so either token would put an arbitrary file's outline
/// into the pack.
#[test]
fn absolute_and_interior_dotdot_paths_are_ignored() {
    let tmp = TempDir::new_in_tmp("context-pack-absescape");
    let src = tmp.subdir("src");
    fs::write(src.join("worker.rs"), RUST_SAMPLE).unwrap();
    // A credentials-shaped file outside the worktree, so a leak is visible.
    let outside_dir = tmp
        .path()
        .parent()
        .expect("the scratch base has a parent")
        .join("context-pack-absescape-outside");
    fs::create_dir_all(&outside_dir).unwrap();
    let secret = outside_dir.join("creds.ini");
    fs::write(&secret, "aws_secret_access_key = TOPSECRETLEAK\n").unwrap();

    let abs_token = secret.to_string_lossy().to_string();
    let task = format!("Read {abs_token} and src/worker.rs for context.");
    let paths = extract_paths(&task, tmp.path());
    assert_eq!(
        paths,
        vec![std::path::PathBuf::from("src/worker.rs")],
        "an absolute path outside the worktree must never be extracted"
    );
    let pack = context_pack(&task, tmp.path()).expect("the in-tree path still packs");
    assert!(
        !pack.contains("TOPSECRETLEAK"),
        "the pack must not carry bytes from an absolute path:\n{pack}"
    );

    // Interior `..`: lexically inside-looking, resolved outside.
    let task2 = "Read src/../../context-pack-absescape-outside/creds.ini for context.";
    let paths2 = extract_paths(task2, tmp.path());
    assert!(
        paths2.is_empty(),
        "an interior `..` escape must never be extracted: {paths2:?}"
    );
    let _ = std::fs::remove_dir_all(&outside_dir);
}
