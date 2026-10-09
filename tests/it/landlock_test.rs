//! Integration coverage for the Landlock filesystem confinement.
//!
//! Exercises the public surface an external consumer sees: the sandbox module
//! is registered, the opt-out knob is nameable, and applying a domain to a
//! *missing* worktree is reported rather than silently skipped.

use std::path::{Path, PathBuf};

use mini_swe_mcp::agent::AgentRunner;
use mini_swe_mcp::agent::sandbox::{
    DISABLE_LANDLOCK_ENV, LandlockPlan, build_landlock_plan, has_bwrap,
};

#[test]
fn sandbox_module_is_exported_with_its_opt_out_knob() {
    assert_eq!(DISABLE_LANDLOCK_ENV, "SWE_DISABLE_LANDLOCK");
    // Referencing the plan builder is enough: it must be nameable from outside.
    let _f: fn(&Path, &Path, bool) -> anyhow::Result<Option<LandlockPlan>> = build_landlock_plan;
}

#[test]
fn a_missing_worktree_is_an_error_for_external_callers() {
    let missing = std::env::temp_dir().join("landlock-it-does-not-exist");
    assert!(!missing.exists());
    let target = std::env::temp_dir();
    let err = build_landlock_plan(&missing, &target, false).unwrap_err();
    assert!(format!("{err:#}").contains("does not exist"), "{err:#}");
}

// ----------
// Autonomic confinement: the pre_exec wiring in `exec.rs`
// ----------

/// Sentinel that turns this test binary into a confinement probe.
///
/// `landlock_restrict_self` is irreversible and confines *the calling process*,
/// so the probe that installs a domain cannot share an address space with the
/// rest of the suite. The parent re-executes this same binary with the
/// sentinel set, exactly as the crate's own Landlock end-to-end test does.
const EXEC_PROBE_ENV: &str = "MINI_SWE_LANDLOCK_EXEC_PROBE";
const RO_PROBE_ENV: &str = "MINI_SWE_LANDLOCK_RO_PROBE";

/// A sandboxed runner pointed at a throwaway endpoint; no request is ever made.
fn runner() -> AgentRunner {
    AgentRunner::new(
        "http://127.0.0.1:1".to_string(),
        "not-a-real-key".to_string(),
        "test-model".to_string(),
        None,
    )
}

/// Directories unique to this test run, removed on drop.
struct Roots {
    base: PathBuf,
    worktree: PathBuf,
    target: PathBuf,
}

impl Roots {
    fn new(tag: &str) -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let base =
            std::env::temp_dir().join(format!("swe-landlock-{tag}-{}-{nanos}", std::process::id()));
        let worktree = base.join("worktree");
        let target = base.join("target");
        std::fs::create_dir_all(&worktree).expect("create worktree");
        std::fs::create_dir_all(&target).expect("create target dir");
        Self {
            base,
            worktree,
            target,
        }
    }
}

impl Drop for Roots {
    fn drop(&mut self) {
        // A step run against a declared root makes the runner derive its
        // private `swe-tmp-<leaf>` scratch next to the scratch *base*, so
        // removing this directory alone would leave one `swe-tmp-worktree` and
        // `swe-tmp-target` entry behind in the real base, filed there by every
        // run of this suite.
        for root in [&self.worktree, &self.target] {
            mini_swe_mcp::worktree::remove_target_dirs(root);
        }
        let _ = std::fs::remove_dir_all(&self.base);
    }
}

/// The plan builder is nameable from outside and reports "no plan" rather than
/// failing where the kernel cannot confine the process.
///
/// This is the contract `exec.rs` relies on: a host without Landlock must
/// produce a *skip*, never an error, because a worker that cannot be confined
/// has to run anyway.
#[test]
fn the_plan_builder_is_exported_and_degrades_instead_of_failing() {
    let roots = Roots::new("plan");
    let built: anyhow::Result<Option<LandlockPlan>> =
        build_landlock_plan(&roots.worktree, &roots.target, false);
    let plan = built.expect("a Landlock-capable kernel must not fail to build a plan");

    // Either a plan, or a documented skip. Both are acceptable; a panic is not.
    if let Some(plan) = plan {
        assert!(plan.rule_count() > 0, "a plan must carry rules");
        assert_ne!(
            plan.handled_access(),
            0,
            "a plan handling no rights would deny the worker everything"
        );
        eprintln!(
            "landlock enforced in this test run ({} rules)",
            plan.rule_count()
        );
    } else {
        eprintln!("skipping enforcement assertions: this kernel has no Landlock");
    }
}

/// A missing root is reported rather than turned into an empty plan.
#[test]
fn a_missing_root_is_reported_rather_than_producing_an_empty_plan() {
    let roots = Roots::new("missing");
    let missing = roots.worktree.join("nope");
    let err = build_landlock_plan(&missing, &roots.target, false).unwrap_err();
    assert!(
        format!("{err:#}").contains("does not exist"),
        "a missing worktree must be reported, got: {err:#}"
    );
}

/// End-to-end through the real execution path: with bubblewrap out of `PATH`,
/// a worker step is still confined by the kernel.
///
/// The child re-runs this binary with a `PATH` that cannot resolve `bwrap`, so
/// `has_bwrap()` is false in it and `execute_bash` must take the Landlock
/// branch. The child then asserts, with real syscalls, that a file outside the
/// domain is unreadable while the worktree stays writable - and, crucially,
/// that the *parent* test process is still unconfined afterwards.
#[test]
fn a_worker_step_is_confined_without_bubblewrap() {
    // Child mode: run the probe and exit; never fall through into the suite.
    if std::env::var_os(EXEC_PROBE_ENV).is_some() {
        run_exec_confined_probe();
    }

    let roots = Roots::new("e2e");

    // A file outside the domain: a sibling of the worktree, which the policy
    // never grants. No assumption about $HOME or the host's layout.
    let secret = roots.base.join("outside-the-domain");
    std::fs::write(&secret, b"PRIVATE").expect("seed a file outside the domain");

    // A PATH that resolves `bash` but not `bwrap`. The crate spawns
    // `nice`/`bash` by bare name, so a real directory is needed; hiding only
    // `bwrap` is enough to flip the branch.
    let sanitized = bin_dir_without_bwrap(&roots);
    if !has_command(&sanitized, "bash") {
        eprintln!("skipping: could not build a PATH without bwrap");
        return;
    }

    let exe = std::env::current_exe().expect("test binary path");
    let out = std::process::Command::new(&exe)
        .arg("--exact")
        .arg("landlock_test::a_worker_step_is_confined_without_bubblewrap")
        .arg("--nocapture")
        .env(EXEC_PROBE_ENV, "1")
        .env("PATH", &sanitized)
        // An inherited override (this suite running inside a worker step)
        // would replace the probe's own target with one that encloses the
        // file outside its domain.
        .env_remove("CARGO_TARGET_DIR")
        .env("LL_WORKTREE", &roots.worktree)
        .env("LL_TARGET", &roots.target)
        .env("LL_OUTSIDE", &secret)
        .env("SWE_DISABLE_LANDBOX", "1")
        .output()
        .unwrap_or_else(|e| panic!("failed to re-run {}: {e}", exe.display()));

    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "the confinement probe failed\nstdout: {stdout}\nstderr: {stderr}"
    );

    if stdout.contains("NO_LANDLOCK") {
        eprintln!("skipping enforcement: this kernel does not support Landlock");
    } else {
        assert!(
            stdout.contains("VERDICT OK"),
            "the worker step was not confined as expected\nstdout: {stdout}\nstderr: {stderr}"
        );
    }

    // The parent is untouched. If the hook had confined this process, the
    // suite could not continue - and this read is the cheap, explicit proof.
    assert_eq!(
        std::fs::read(&secret).expect("the parent must keep its own access"),
        b"PRIVATE",
        "confining a worker must never confine the daemon that spawned it"
    );
}

/// Child half of [`a_worker_step_is_confined_without_bubblewrap`].
fn run_exec_confined_probe() -> ! {
    let var = |name: &str| -> PathBuf {
        std::env::var_os(name)
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                eprintln!("probe needs {name}");
                std::process::exit(2);
            })
    };
    let worktree = var("LL_WORKTREE");
    let target = var("LL_TARGET");
    let outside = var("LL_OUTSIDE");

    // Confirm the probe is really on the Landlock branch: the parent stripped
    // bwrap from PATH, so `has_bwrap()` must be false here. If it is not, the
    // assertions below would pass for the wrong reason (bwrap confined the
    // child) and the test would be worthless.
    if has_bwrap() {
        eprintln!("PROBE ERROR: bwrap is still reachable, the Landlock branch was not taken");
        std::process::exit(3);
    }

    // No plan means this kernel cannot confine at all; that is the documented
    // degradation, not a failure.
    match build_landlock_plan(&worktree, &target, false) {
        Ok(None) => {
            println!("NO_LANDLOCK");
            std::process::exit(0);
        }
        Ok(Some(_)) => {}
        Err(e) => {
            eprintln!("PROBE ERROR: could not build a plan: {e:#}");
            std::process::exit(1);
        }
    }

    let r = runner();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("build a runtime");

    // 1. A read outside the domain must be denied.
    //
    // The command is `cat X; echo rc=$?` on purpose: the shell's exit status is
    // that of the *last* command, so it is `echo` that decides it and the
    // step still "succeeds". Asserting on the status would therefore measure
    // nothing; the denial is visible in the `rc=` the probe prints and in
    // cat's own stderr, and both are checked here.
    let (out, code) = runtime
        .block_on(r.execute_bash(
            &worktree,
            &format!("cat {}; echo cat_rc=$?", outside.display()),
        ))
        .expect("a denied read is output, not a spawn error");
    assert_eq!(code, Some(0), "the step itself must still succeed: {out:?}");
    assert!(
        out.contains("cat_rc=1"),
        "a confined worker must not read a file outside its domain: {out:?}"
    );
    // The file's contents must not appear anywhere in the captured output.
    assert!(
        !out.contains("PRIVATE"),
        "the denied file leaked into the worker's output: {out:?}"
    );

    // 2. The worktree must still be writable - a sandbox that denies the
    //    agent its own worktree is not a sandbox, it is an outage.
    let (out, code) = runtime
        .block_on(r.execute_bash(&worktree, "printf built > artifact.txt; echo rc=$?"))
        .expect("a worktree write must not be a spawn error");
    assert_eq!(code, Some(0), "the worktree must stay writable: {out:?}");
    assert!(out.contains("rc=0"), "{out:?}");
    assert!(
        worktree.join("artifact.txt").exists(),
        "the worktree write must have really landed: {out:?}"
    );

    // 3. A plain command still works, with its output intact.
    let (out, code) = runtime
        .block_on(r.execute_bash(&worktree, "echo still-alive"))
        .expect("a confined command must still run");
    assert_eq!(code, Some(0), "{out:?}");
    assert!(out.contains("still-alive"), "{out:?}");

    println!("VERDICT OK");
    // libtest's stdout capture is never flushed after `process::exit`, so the
    // verdict is written straight to fd 1 by the runner above; exit cleanly.
    std::process::exit(0);
}

/// A directory holding symlinks to every executable in `PATH` except `bwrap`.
///
/// The crate resolves `bwrap`, `nice` and `bash` by bare name, so the child
/// needs a working `PATH` - it just must not contain bubblewrap. Symlinks keep
/// this cheap and avoid copying megabytes of toolchain.
fn bin_dir_without_bwrap(roots: &Roots) -> PathBuf {
    let dest = roots.base.join("bin");
    std::fs::create_dir_all(&dest).expect("create the shim PATH");
    for dir in std::env::var_os("PATH")
        .unwrap_or_default()
        .to_string_lossy()
        .split(':')
    {
        if dir.is_empty() {
            continue;
        }
        let Ok(entries) = std::fs::read_dir(dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name == "bwrap" {
                continue; // This is the whole point of the shim.
            }
            let link = dest.join(name.as_ref());
            if link.symlink_metadata().is_ok() {
                continue; // First match on PATH wins, as in a real shell.
            }
            let _ = std::os::unix::fs::symlink(entry.path(), &link);
        }
    }
    dest
}

/// Whether `name` resolves inside `dir`, the way `Command::spawn` would.
fn has_command(dir: &Path, name: &str) -> bool {
    dir.join(name).exists()
}

/// An external read-only fixture is readable under Landlock confinement, but
/// cannot be modified or executed, and sibling files outside the mount stay denied.
/// Non-secret environment variables are forwarded, while secret-bearing ones are filtered.
#[test]
fn an_external_readonly_mount_is_readable_not_writable_under_confinement() {
    if std::env::var_os(RO_PROBE_ENV).is_some() {
        run_readonly_mount_confined_probe();
    }

    let roots = Roots::new("ro-e2e");

    let fixture_base = std::env::temp_dir().join(format!(
        "swe-ro-fixture-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&fixture_base).expect("create fixture dir");
    let fixture = fixture_base.join("fixture_exec.sh");
    let dir_fixture = fixture_base.join("dir_assets");
    std::fs::create_dir_all(&dir_fixture).expect("create dir fixture");
    let asset = dir_fixture.join("asset.txt");
    let outside = fixture_base.join("sibling_secret.txt");

    std::fs::write(&fixture, b"#!/bin/sh\necho VULN_EXECUTED\n").expect("write fixture");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&fixture).expect("metadata").permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&fixture, perms).expect("chmod +x fixture");
    }
    std::fs::write(&asset, b"ASSET_DATA_BYTES").expect("write asset");
    std::fs::write(&outside, b"SIBLING_SECRET").expect("write outside secret");

    let canonical_fixture = fixture.canonicalize().expect("canonical fixture");
    let canonical_dir = dir_fixture.canonicalize().expect("canonical dir");
    let canonical_outside = outside.canonicalize().expect("canonical outside");

    let sanitized = bin_dir_without_bwrap(&roots);
    if !has_command(&sanitized, "bash") {
        eprintln!("skipping: could not build a PATH without bwrap");
        let _ = std::fs::remove_dir_all(&fixture_base);
        return;
    }

    let exe = std::env::current_exe().expect("test binary path");
    let mounts_json = format!(
        r#"["{}", "{}"]"#,
        canonical_fixture.display(),
        canonical_dir.display()
    );
    let forward_json = r#"["TEST_FORWARD_OK", "TEST_EMPTY_VAR", "TEST_ABSENT_VAR"]"#.to_string();

    let out = std::process::Command::new(&exe)
        .arg("--exact")
        .arg("landlock_test::an_external_readonly_mount_is_readable_not_writable_under_confinement")
        .arg("--nocapture")
        .env(RO_PROBE_ENV, "1")
        .env("PATH", &sanitized)
        .env_remove("CARGO_TARGET_DIR")
        .env("LL_WORKTREE", &roots.worktree)
        .env("LL_TARGET", &roots.target)
        .env("LL_FIXTURE", &canonical_fixture)
        .env("LL_DIR", &canonical_dir)
        .env("LL_OUTSIDE", &canonical_outside)
        .env(
            mini_swe_mcp::agent::sandbox::READONLY_MOUNTS_ENV,
            &mounts_json,
        )
        .env(mini_swe_mcp::agent::env::FORWARD_ENV_VAR, &forward_json)
        .env("TEST_FORWARD_OK", "forward_success")
        .env("TEST_EMPTY_VAR", "")
        .output()
        .expect("spawn readonly probe");

    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    let _ = std::fs::remove_dir_all(&fixture_base);

    assert!(
        out.status.success(),
        "readonly probe failed\nstdout: {stdout}\nstderr: {stderr}"
    );

    if stdout.contains("NO_LANDLOCK") {
        eprintln!("skipping enforcement: this kernel does not support Landlock");
    } else {
        assert!(
            stdout.contains("VERDICT OK"),
            "readonly mount probe failed\nstdout: {stdout}\nstderr: {stderr}"
        );
    }
}

fn run_readonly_mount_confined_probe() -> ! {
    let var = |name: &str| -> PathBuf {
        std::env::var_os(name)
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                eprintln!("probe needs {name}");
                std::process::exit(2);
            })
    };
    let worktree = var("LL_WORKTREE");
    let target = var("LL_TARGET");
    let fixture = var("LL_FIXTURE");
    let dir = var("LL_DIR");
    let outside = var("LL_OUTSIDE");

    if has_bwrap() {
        eprintln!("PROBE ERROR: bwrap is still reachable");
        std::process::exit(3);
    }

    match build_landlock_plan(&worktree, &target, false) {
        Ok(None) => {
            println!("NO_LANDLOCK");
            std::process::exit(0);
        }
        Ok(Some(_)) => {}
        Err(e) => {
            eprintln!("PROBE ERROR: could not build a plan: {e:#}");
            std::process::exit(1);
        }
    }

    let mounts = mini_swe_mcp::agent::sandbox::parse_readonly_mounts(&worktree, &target)
        .expect("parse readonly mounts");
    let operator_env =
        mini_swe_mcp::agent::env::operator_forwarded_vars().expect("parse operator forward env");
    let r = runner()
        .with_readonly_mounts(mounts)
        .with_operator_env(operator_env);
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("build a runtime");

    // 1. Reading the mounted fixture file succeeds.
    let (out, code) = runtime
        .block_on(r.execute_bash(
            &worktree,
            &format!("cat {}; echo cat_rc=$?", fixture.display()),
        ))
        .expect("read fixture");
    assert_eq!(code, Some(0));
    assert!(
        out.contains("cat_rc=0"),
        "fixture must be readable: {out:?}"
    );
    assert!(
        out.contains("VULN_EXECUTED"),
        "fixture content must match: {out:?}"
    );

    // 2. Writing to the mounted fixture file is DENIED.
    let (out, _) = runtime
        .block_on(r.execute_bash(
            &worktree,
            &format!("printf overwrite > {}; echo write_rc=$?", fixture.display()),
        ))
        .expect("write to readonly fixture");
    assert!(
        !out.contains("write_rc=0"),
        "writing to readonly fixture must be denied: {out:?}"
    );

    // 3. Direct execution (execve) of the mounted fixture is DENIED under Landlock.
    let (out, code) = runtime
        .block_on(r.execute_bash(
            &worktree,
            &format!("{}; echo exec_rc=$?", fixture.display()),
        ))
        .expect("exec fixture");
    assert!(
        !out.contains("exec_rc=0") || !out.contains("VULN_EXECUTED"),
        "direct execve of readonly fixture must be denied: {out:?}"
    );
    assert!(
        code != Some(0) || !out.contains("VULN_EXECUTED"),
        "direct execution must not succeed: code={code:?}, out={out:?}"
    );

    // 3b. Userland interpreter data-reading execution (sh /path/to/script) succeeds
    // because reading bytes is permitted; this clarifies the direct-execve vs data-read distinction.
    let (out, code) = runtime
        .block_on(r.execute_bash(
            &worktree,
            &format!("sh {}; echo interp_rc=$?", fixture.display()),
        ))
        .expect("interpret fixture");
    assert_eq!(code, Some(0));
    assert!(
        out.contains("interp_rc=0") && out.contains("VULN_EXECUTED"),
        "interpreter reading bytes is permitted: {out:?}"
    );

    // 4. Reading directory asset succeeds, writing to directory is DENIED.
    let asset_path = dir.join("asset.txt");
    let (out, code) = runtime
        .block_on(r.execute_bash(
            &worktree,
            &format!("cat {}; echo asset_rc=$?", asset_path.display()),
        ))
        .expect("read asset");
    assert_eq!(code, Some(0));
    assert!(
        out.contains("asset_rc=0") && out.contains("ASSET_DATA_BYTES"),
        "directory asset must be readable: {out:?}"
    );

    let (out, _) = runtime
        .block_on(r.execute_bash(
            &worktree,
            &format!(
                "printf overwrite > {}; echo asset_write_rc=$?",
                asset_path.display()
            ),
        ))
        .expect("write to readonly asset");
    assert!(
        !out.contains("asset_write_rc=0"),
        "overwriting deep asset in readonly directory must be denied: {out:?}"
    );

    let new_asset = dir.join("new_file.txt");
    let (out, _) = runtime
        .block_on(r.execute_bash(
            &worktree,
            &format!("touch {}; echo touch_rc=$?", new_asset.display()),
        ))
        .expect("write to readonly dir");
    assert!(
        !out.contains("touch_rc=0"),
        "writing new file into readonly directory must be denied: {out:?}"
    );

    // 5. Reading sibling outside the mount is DENIED.
    let (out, _) = runtime
        .block_on(r.execute_bash(
            &worktree,
            &format!("cat {}; echo cat_rc=$?", outside.display()),
        ))
        .expect("read outside sibling");
    assert!(
        !out.contains("cat_rc=0"),
        "sibling secret must stay denied: {out:?}"
    );
    assert!(
        !out.contains("SIBLING_SECRET"),
        "secret must not leak: {out:?}"
    );

    // 6. Scoped non-secret env reaches command, preserving defined-empty and omitting absent.
    let (out, _) = runtime
        .block_on(r.execute_bash(
            &worktree,
            "echo F=$TEST_FORWARD_OK EMPTY_SET=${TEST_EMPTY_VAR+set} ABSENT_SET=${TEST_ABSENT_VAR+set}",
        ))
        .expect("check env");
    assert!(
        out.contains("F=forward_success"),
        "forwarded non-secret env must reach child: {out:?}"
    );
    assert!(
        out.contains("EMPTY_SET=set"),
        "defined empty env var must be set in child environment: {out:?}"
    );
    assert!(
        out.contains("ABSENT_SET="),
        "absent env var must remain unset in child environment: {out:?}"
    );

    // 7. Operator environment precedence over client overlay.
    let r_tamper = r.clone().with_extra_env(vec![(
        "TEST_FORWARD_OK".to_string(),
        "tampered_value".to_string(),
    )]);
    let (out, _) = runtime
        .block_on(r_tamper.execute_bash(&worktree, "echo F=$TEST_FORWARD_OK"))
        .expect("check env tamper");
    assert!(
        out.contains("F=forward_success") && !out.contains("tampered_value"),
        "operator env must have immutable precedence over client overlay: {out:?}"
    );

    println!("VERDICT OK");
    std::process::exit(0);
}

#[tokio::test]
async fn bubblewrap_backend_fails_closed_when_readonly_mounts_configured() {
    let roots = Roots::new("ro-bwrap-failclosed");
    let mut cmd = tokio::process::Command::new("true");
    let err = mini_swe_mcp::agent::exec::__test_apply_sandbox_args(
        &mut cmd,
        &roots.worktree,
        &roots.target,
        std::slice::from_ref(&roots.worktree),
    )
    .expect_err("bwrap must fail closed with mounts");
    assert!(
        err.to_string()
            .contains("bubblewrap backend cannot enforce noexec isolation"),
        "unexpected error message: {err:#}"
    );
}
