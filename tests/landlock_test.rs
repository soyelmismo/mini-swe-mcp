//! Integration coverage for the Landlock filesystem confinement.
//!
//! Exercises the public surface an external consumer sees: the sandbox module
//! is registered, the opt-out knob is nameable, and applying a domain to a
//! *missing* worktree is reported rather than silently skipped.

use std::path::Path;

use mini_swe_mcp::agent::sandbox::{DISABLE_LANDLOCK_ENV, apply_landlock_sandbox};

#[test]
fn sandbox_module_is_exported_with_its_opt_out_knob() {
    assert_eq!(DISABLE_LANDLOCK_ENV, "SWE_DISABLE_LANDLOCK");
    // Referencing the function is enough: it must be nameable from outside.
    let _f: fn(&Path, &Path) -> anyhow::Result<()> = apply_landlock_sandbox;
}

#[test]
fn a_missing_worktree_is_an_error_for_external_callers() {
    let missing = std::env::temp_dir().join("landlock-it-does-not-exist");
    assert!(!missing.exists());
    let target = std::env::temp_dir();
    let err = apply_landlock_sandbox(&missing, &target).unwrap_err();
    assert!(format!("{err:#}").contains("does not exist"), "{err:#}");
}
