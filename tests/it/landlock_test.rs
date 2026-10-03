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
        .arg("a_worker_step_is_confined_without_bubblewrap")
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
