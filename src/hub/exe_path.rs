//! The one place that turns [`std::env::current_exe`] into a spawnable path.
//!
//! `cargo build` replaces a binary in place, so a process started from the
//! older one — a long-lived daemon handing over, or any client that has to
//! respawn it — sees `/proc/self/exe` (and therefore `current_exe()`) as
//! `<path> (deleted)` while `<path>` already holds the newer build. Executing
//! that literal string fails with `No such file or directory (os error 2)`, so
//! every spawn path strips the kernel's marker first and runs the path, which
//! is what makes the replacement the new binary instead of another copy of the
//! old one.
//!
//! Stripping is not enough on its own: cargo unlinks the old binary before it
//! writes the new one, so a process that reacts in that window finds no file
//! either. [`executable`] therefore waits a bounded while for the path to come
//! back, and only then says which executable it could not find.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// The kernel's marker for a `/proc/self/exe` whose inode no longer has a name.
const DELETED_SUFFIX: &str = " (deleted)";

/// How long [`executable`] waits for a stripped path to exist.
///
/// Long enough to cover a build writing the replacement in place, short enough
/// that a genuinely gone binary fails fast instead of hanging the client that
/// needed a hub.
const WAIT_FOR_REPLACEMENT: Duration = Duration::from_millis(500);

/// How often [`executable`] re-checks the path while waiting.
const RETRY_INTERVAL: Duration = Duration::from_millis(10);

/// This executable's path with the kernel's ` (deleted)` marker removed.
///
/// `None` only when `current_exe()` itself fails, the one case with no path to
/// strip. The result is not required to exist: the binary may genuinely be
/// gone, which [`executable`] turns into a readable error.
pub fn current_exe_path() -> Option<PathBuf> {
    strip_deleted(std::env::current_exe().ok()?)
}

/// The path a client or handover should run to start a hub, waiting out a
/// build that is mid-replace.
///
/// Unlike [`current_exe_path`] this requires the stripped path to be a file:
/// that is the difference between a client that starts the hub and one whose
/// `spawn(2)` fails with `No such file or directory (os error 2)`.
pub fn executable() -> Result<PathBuf, String> {
    let path =
        current_exe_path().ok_or_else(|| "Could not read this process's own path".to_string())?;
    await_executable(path, WAIT_FOR_REPLACEMENT)
}

/// `path` once it names a file, polling for up to `wait`.
///
/// A path that never appears is reported by name, so the failure says which
/// executable is missing instead of leaking the raw `spawn(2)` error.
fn await_executable(path: PathBuf, wait: Duration) -> Result<PathBuf, String> {
    let deadline = Instant::now() + wait;
    loop {
        if path.is_file() {
            return Ok(path);
        }
        if Instant::now() >= deadline {
            return Err(format!(
                "Hub daemon executable {} does not exist: a build replaced this binary and has not written the new one yet",
                path.display()
            ));
        }
        std::thread::sleep(RETRY_INTERVAL);
    }
}

/// Drop the kernel's ` (deleted)` marker from a `/proc/self/exe` path.
///
/// Purely lexical, so it applies to a path that no longer exists and is the
/// single definition of what the marker is. A path without the marker is
/// returned unchanged.
fn strip_deleted(exe: PathBuf) -> PathBuf {
    use std::ffi::OsString;
    use std::os::unix::ffi::{OsStrExt, OsStringExt};
    let bytes = exe.as_os_str().as_bytes();
    match bytes.strip_suffix(DELETED_SUFFIX.as_bytes()) {
        Some(stripped) => PathBuf::from(OsString::from_vec(stripped.to_vec())),
        None => exe,
    }
}

#[cfg(test)]
#[path = "exe_path_tests.rs"]
mod tests;
