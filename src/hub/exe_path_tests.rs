//! The ` (deleted)` strip and the wait for a replacement.
//!
//! A rebuilt binary is the normal case for a long-lived daemon, so both are
//! covered without spawning a daemon: a client that respawned the marked path
//! would fail with `No such file or directory (os error 2)`.

use super::*;
use std::ffi::OsString;
use std::os::unix::ffi::OsStringExt;
use std::os::unix::fs::PermissionsExt;

/// The path the kernel reports for a binary a build already replaced, at the
/// moment it holds: unlinked, marked, and not yet written again.
fn marked(replacement: &std::path::Path) -> PathBuf {
    PathBuf::from(OsString::from_vec(
        format!("{} (deleted)", replacement.display()).into_bytes(),
    ))
}

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
/// pointing at some other directory.
#[test]
fn the_suffix_is_only_stripped_from_the_end() {
    let odd = PathBuf::from("/tmp/build (deleted)/mini-swe-mcp");
    assert_eq!(strip_deleted(odd.clone()), odd);
}

/// A client whose exe path carries the suffix spawns from the stripped path.
/// The path is injected, which is the only way a test can present the kernel's
/// view of a replaced binary, and the build lands mid-wait the way `cargo`
/// writes the replacement in place.
#[test]
fn a_client_exe_path_carrying_the_suffix_spawns_from_the_stripped_path() {
    let scratch = std::env::temp_dir().join(format!("exe-path-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&scratch).expect("scratch dir");
    let replacement = scratch.join("mini-swe-mcp");
    let deleted = marked(&replacement);

    let build = {
        let replacement = replacement.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(30));
            std::fs::write(&replacement, b"#!/bin/sh\nexit 0\n").expect("write the replacement");
            std::fs::set_permissions(&replacement, std::fs::Permissions::from_mode(0o700))
                .expect("make the replacement executable");
        })
    };
    // The same resolution `connect_or_spawn` performs, on the injected path.
    let client = std::thread::spawn(move || executable_from(deleted));
    let resolved = client
        .join()
        .expect("resolver thread")
        .expect("the replacement appears");
    build.join().expect("build thread");

    assert_eq!(
        resolved, replacement,
        "the spawn must use the live path, not the marked one"
    );
    // And the resolved path is what the client's `spawn(2)` gets, end to end:
    // running the marked string instead is the ENOENT this fixes.
    let started = std::process::Command::new(&resolved)
        .arg("daemon")
        .status()
        .expect("the resolved path is executable");
    assert!(started.success());
    assert!(
        std::process::Command::new(format!("{} (deleted)", resolved.display()))
            .arg("daemon")
            .status()
            .is_err(),
        "the marked path is exactly what used to fail to spawn"
    );
    std::fs::remove_dir_all(&scratch).ok();
}

/// A path that stays gone is named in the error, so the client can say which
/// executable it could not start instead of reporting a bare `ENOENT`.
#[test]
fn a_path_that_never_appears_is_reported_by_name() {
    let missing = std::env::temp_dir().join(format!("exe-path-{}", uuid::Uuid::new_v4()));
    let error = await_executable(missing.clone(), Duration::from_millis(20))
        .expect_err("a path that never appears cannot be spawned")
        .to_string();
    assert!(error.contains(&missing.display().to_string()), "{error}");
    assert!(error.contains("does not exist"), "{error}");
}
