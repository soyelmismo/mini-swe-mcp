//! The ` (deleted)` strip and the wait for a replacement.
//!
//! A rebuilt binary is the normal case for a long-lived daemon, so both are
//! covered without spawning anything: a client that respawned the marked path
//! would fail with `No such file or directory (os error 2)`.

use super::*;
use std::ffi::OsString;
use std::os::unix::ffi::OsStringExt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// A `(deleted)` path is stripped to the path the build wrote over.
#[test]
fn a_deleted_suffix_is_stripped() {
    let stripped = strip_deleted(PathBuf::from(
        "/home/u/target/release/mini-swe-mcp (deleted)",
    ));
    assert_eq!(
        stripped,
        PathBuf::from("/home/u/target/release/mini-swe-mcp")
    );
}

/// A path the kernel never marked is handed on unchanged.
#[test]
fn a_live_path_is_untouched() {
    let live = PathBuf::from("/home/u/target/release/mini-swe-mcp");
    assert_eq!(strip_deleted(live.clone()), live);
}

/// The marker is stripped only from the end of the path, so a directory whose
/// own name ends in ` (deleted)` keeps it and fails loudly instead of quietly
/// pointing somewhere else.
#[test]
fn the_suffix_is_only_stripped_from_the_end() {
    let odd = PathBuf::from("/tmp/build (deleted)/mini-swe-mcp");
    assert_eq!(strip_deleted(odd.clone()), odd);
}

/// A client whose exe path carries the suffix spawns from the stripped path,
/// once the build has written it. The path is injected, which is the only way
/// a test can present the kernel's view of a replaced binary.
#[test]
fn a_client_exe_path_carrying_the_suffix_spawns_from_the_stripped_path() {
    let scratch = std::env::temp_dir().join(format!("exe-path-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&scratch).expect("scratch dir");
    let replacement = scratch.join("mini-swe-mcp");
    // Exactly what the kernel reports while cargo holds the gap between unlink
    // and write: the file is gone and the marker says why.
    let deleted = PathBuf::from(OsString::from_vec(
        format!("{} (deleted)", replacement.display()).into_bytes(),
    ));

    let spawned = Arc::new(AtomicBool::new(false));
    let seen = spawned.clone();
    let build = {
        let replacement = replacement.clone();
        std::thread::spawn(move || {
            std::fs::write(&replacement, b"#!/bin/sh\nexit 0\n").expect("write the replacement");
        })
    };
    // The same wait `connect_or_spawn` performs, with the spawn stand-in so the
    // test observes the path instead of a detached daemon.
    let waiter = std::thread::spawn({
        let deleted = deleted.clone();
        move || {
            let path = await_executable(strip_deleted(deleted), WAIT_FOR_REPLACEMENT)
                .expect("the replacement appears");
            seen.store(true, Ordering::SeqCst);
            path
        }
    });
    build.join().expect("build thread");
    let path = waiter.join().expect("waiter thread");

    assert_eq!(
        path, replacement,
        "the spawn must use the live path, not the marked one"
    );
    assert!(path.is_file());
    assert!(
        spawned.load(Ordering::SeqCst),
        "the spawn happens once the file is there"
    );
    std::fs::remove_dir_all(&scratch).ok();
}

/// A path that stays gone is named in the error, so the client can say what it
/// could not start instead of reporting a bare `ENOENT`.
#[test]
fn a_path_that_never_appears_is_reported_by_name() {
    let missing = std::env::temp_dir().join(format!("exe-path-{}", uuid::Uuid::new_v4()));
    let error = await_executable(missing.clone(), Duration::from_millis(20))
        .expect_err("a path that never appears cannot be spawned");
    assert!(error.contains(&missing.display().to_string()), "{error}");
    assert!(error.contains("does not exist"), "{error}");
}
