//! Shared toolchain/compiler cache wiring.
//!
//! Resolves the shared cache root, lays out the well-known package/compiler
//! cache directories (`kache`, `uv`, `pip`, the Node toolchain and Go), and
//! turns that into (a) `bubblewrap` bind mounts and (b) child-process
//! environment variables. The directory layout is computed lazily once per
//! process and ensured to exist before use. The memoized tool probes
//! (`has_kache` / `has_sccache`) answer "is this compiler wrapper installed?"
//! at most once per process.
//!
//! Rust builds are the disk-heaviest consumer. Worker commands get
//! `RUSTC_WRAPPER=kache` but with executable caching off
//! (`KACHE_CACHE_EXECUTABLES=0`), because re-caching the crate's own test
//! binaries - which the per-worker target directory already holds - doubles
//! each write and lets the resulting store GC evict the third-party
//! dependencies a hit would have saved. `agent::exec::apply_build_env`
//! complements that by dropping debug info and incremental state from those
//! directories (`CARGO_PROFILE_DEV_DEBUG=0`, `CARGO_PROFILE_TEST_DEBUG=0`,
//! `CARGO_INCREMENTAL=0`).
//!
//! The per-repository build slots those directories live in are leased
//! exclusively ([`BuildDirLease`]) and bounded twice over: an idle slot past
//! `MINI_SWE_TARGET_SLOT_MAX_GIB` is emptied when it is leased, and a slot
//! nobody has used for `HUB_TARGET_TTL_HOURS` is removed by the sweep the daemon
//! starts on boot and repeats every five minutes.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// Root directory where shared compiler/package caches reside across workers.
pub fn shared_cache_root() -> PathBuf {
    if let Ok(dir) = std::env::var("SWE_CACHE_DIR") {
        PathBuf::from(dir)
    } else {
        crate::worktree::swe_base_dir().join("swe-cache")
    }
}

/// Paths to known package/compiler caches inside `shared_cache_root()`.
#[derive(Debug, Clone)]
pub struct CacheDirs {
    pub root: PathBuf,
    pub kache: PathBuf,
    pub uv: PathBuf,
    pub pip: PathBuf,
    pub npm: PathBuf,
    pub yarn: PathBuf,
    pub pnpm_home: PathBuf,
    pub pnpm_store: PathBuf,
    pub go_build: PathBuf,
    pub go_mod: PathBuf,
    /// Maven's local repository, `~/.m2/repository` by default.
    pub maven: PathBuf,
    /// Gradle's user home, `~/.gradle` by default: caches plus wrapper dists.
    pub gradle: PathBuf,
}

impl CacheDirs {
    pub fn new() -> Self {
        let root = shared_cache_root();
        Self {
            kache: root.join("kache"),
            uv: root.join("python").join("uv"),
            pip: root.join("python").join("pip"),
            npm: root.join("node").join("npm"),
            yarn: root.join("node").join("yarn"),
            pnpm_home: root.join("node").join("pnpm"),
            pnpm_store: root.join("node").join("pnpm-store"),
            go_build: root.join("go").join("build"),
            go_mod: root.join("go").join("mod"),
            maven: root.join("java").join("m2"),
            gradle: root.join("java").join("gradle"),
            root,
        }
    }

    /// Ensure all default cache subdirectories exist on disk.
    pub fn ensure_dirs(&self) {
        let _ = std::fs::create_dir_all(&self.root);
        let _ = std::fs::create_dir_all(&self.kache);
        let _ = std::fs::create_dir_all(&self.uv);
        let _ = std::fs::create_dir_all(&self.pip);
        let _ = std::fs::create_dir_all(&self.npm);
        let _ = std::fs::create_dir_all(&self.yarn);
        let _ = std::fs::create_dir_all(&self.pnpm_home);
        let _ = std::fs::create_dir_all(&self.pnpm_store);
        let _ = std::fs::create_dir_all(&self.go_build);
        let _ = std::fs::create_dir_all(&self.go_mod);
        let _ = std::fs::create_dir_all(&self.maven);
        let _ = std::fs::create_dir_all(&self.gradle);
    }
}

impl Default for CacheDirs {
    fn default() -> Self {
        Self::new()
    }
}

/// The process-wide cache directory layout, computed once and ensured to exist.
///
/// [`CacheDirs::new`] is cheap but `ensure_dirs` performs ten `create_dir_all`
/// calls, and both `append_bwrap_cache_args` and `apply_shared_cache_env` need
/// the same layout per command. Computing it once per process keeps the dirs
/// (and their existence) shared across every command.
fn cache_dirs() -> &'static CacheDirs {
    static DIRS: OnceLock<CacheDirs> = OnceLock::new();
    DIRS.get_or_init(|| {
        let dirs = CacheDirs::new();
        dirs.ensure_dirs();
        dirs
    })
}

/// Custom cache binds parsed from `SWE_SHARED_CACHES` (e.g. "/host/a:/guest/a,/host/b").
pub fn parse_custom_cache_binds(raw: &str) -> Vec<(PathBuf, PathBuf)> {
    raw.split(',')
        .filter_map(|entry| {
            let entry = entry.trim();
            if entry.is_empty() {
                return None;
            }
            if let Some((host, guest)) = entry.split_once(':') {
                Some((PathBuf::from(host.trim()), PathBuf::from(guest.trim())))
            } else {
                let p = PathBuf::from(entry);
                Some((p.clone(), p))
            }
        })
        .collect()
}

/// Check if `kache` is available in PATH or standard user binary paths.
///
/// Memoized so the (potentially `fork`+`exec`-bound) probe runs at most once
/// per process.
pub fn has_kache() -> bool {
    if let Some(home) = std::env::var_os("HOME").map(PathBuf::from)
        && home.join(".local/bin/kache").is_file()
    {
        return true;
    }
    crate::agent::sandbox::binary_available("kache")
}

/// Check if `sccache` is available in PATH.
///
/// Memoized so the probe runs at most once per process.
pub fn has_sccache() -> bool {
    crate::agent::sandbox::binary_available("sccache")
}

/// Append bubblewrap arguments for mounting the shared cache root, user tool caches,
/// and any custom cache binds specified in `SWE_SHARED_CACHES`.
pub fn append_bwrap_cache_args(cmd: &mut tokio::process::Command, home: Option<&Path>) {
    let dirs = cache_dirs();

    // 1. Bind global shared cache directory read-write
    let root_str = dirs.root.to_string_lossy();
    cmd.args(["--bind", &root_str, &root_str]);

    // 2. If host home has ~/.cache/kache (or kache is used), bind it so kache works without tmpfs discarding
    if let Some(h) = home {
        let user_kache = h.join(".cache").join("kache");
        let _ = std::fs::create_dir_all(&user_kache);
        let kache_str = user_kache.to_string_lossy();
        cmd.args(["--bind", &kache_str, &kache_str]);
    }

    // 3. User-defined custom cache binds via SWE_SHARED_CACHES
    if let Ok(custom) = std::env::var("SWE_SHARED_CACHES") {
        for (host, guest) in parse_custom_cache_binds(&custom) {
            let _ = std::fs::create_dir_all(&host);
            let h_str = host.to_string_lossy();
            let g_str = guest.to_string_lossy();
            cmd.args(["--bind", &h_str, &g_str]);
        }
    }
}

/// Append `extra` to the whitespace-separated value of `name` on `cmd`.
///
/// Two layers configure the same variables: `exec::apply_build_env` sets the
/// granted job count (`MAVEN_OPTS=-T4`) and this module adds the shared cache
/// location. Overwriting either would silently drop the other, so the value is
/// read back from the command and concatenated in call order.
fn append_env_value(cmd: &mut tokio::process::Command, name: &str, extra: &str) {
    let existing = cmd
        .as_std()
        .get_envs()
        .find(|(key, _)| *key == std::ffi::OsStr::new(name))
        .and_then(|(_, value)| value.map(|value| value.to_string_lossy().into_owned()))
        .unwrap_or_default();
    let joined = if existing.is_empty() {
        extra.to_string()
    } else {
        format!("{existing} {extra}")
    };
    cmd.env(name, joined);
}

/// Apply universal cache environment variables to the child command.
pub fn apply_shared_cache_env(cmd: &mut tokio::process::Command) {
    let dirs = cache_dirs();

    // Python (uv, pip)
    cmd.env("UV_CACHE_DIR", &dirs.uv);
    cmd.env("PIP_CACHE_DIR", &dirs.pip);

    // Node / JS / TS (npm, yarn, pnpm)
    cmd.env("npm_config_cache", &dirs.npm);
    cmd.env("YARN_CACHE_FOLDER", &dirs.yarn);
    cmd.env("PNPM_HOME", &dirs.pnpm_home);
    cmd.env("PNPM_STORE_DIR", &dirs.pnpm_store);

    // Go (build and modules)
    cmd.env("GOCACHE", &dirs.go_build);
    cmd.env("GOMODCACHE", &dirs.go_mod);

    // JVM. `MAVEN_OPTS` carries the local repository because `MAVEN_ARGS` is
    // not read by every launcher, and `GRADLE_USER_HOME` relocates the whole
    // Gradle user home, caches and wrapper dists included. The repository is
    // appended to whatever `apply_build_env` already put in `MAVEN_OPTS`
    // (its `-T` job cap), so neither setting is lost.
    append_env_value(
        cmd,
        "MAVEN_OPTS",
        &format!("-Dmaven.repo.local={}", dirs.maven.display()),
    );
    cmd.env("GRADLE_USER_HOME", &dirs.gradle);

    // Ensure toolchain paths (~/.local/bin, ~/.cargo/bin) are in PATH
    if let Some(home) = std::env::var_os("HOME").map(PathBuf::from) {
        let current_path = std::env::var("PATH").unwrap_or_default();
        let local_bin = home.join(".local/bin").to_string_lossy().to_string();
        let cargo_bin = home.join(".cargo/bin").to_string_lossy().to_string();
        let mut parts: Vec<&str> = current_path.split(':').collect();
        let mut new_parts = Vec::new();
        if !parts.contains(&local_bin.as_str()) {
            new_parts.push(local_bin.as_str());
        }
        if !parts.contains(&cargo_bin.as_str()) {
            new_parts.push(cargo_bin.as_str());
        }
        new_parts.append(&mut parts);
        cmd.env("PATH", new_parts.join(":"));
    }

    // Rust / C / C++ compiler wrapper
    if std::env::var("SWE_DISABLE_KACHE").as_deref() != Ok("1")
        && std::env::var("KACHE_DISABLED").as_deref() != Ok("1")
    {
        let wrapper = if has_kache() {
            Some("kache")
        } else if has_sccache() {
            Some("sccache")
        } else {
            None
        };
        apply_rust_compiler_cache_env(cmd, wrapper, std::env::var_os(KACHE_CACHE_EXECUTABLES_VAR));
    }
}

/// kache's switch for caching binaries and test executables.
const KACHE_CACHE_EXECUTABLES_VAR: &str = "KACHE_CACHE_EXECUTABLES";

/// Size cap for one leased build slot, in GiB. `0` disables the cap.
const SLOT_MAX_GIB_ENV: &str = "MINI_SWE_TARGET_SLOT_MAX_GIB";

/// Wire the Rust compiler cache wrapper onto a child command.
///
/// `wrapper` is the installed wrapper, if any. `cache_executables` is the
/// ambient [`KACHE_CACHE_EXECUTABLES_VAR`], forwarded verbatim so an operator's
/// explicit choice survives the cleared child environment; when the operator
/// set nothing, kache's executable caching is turned off
/// ([`kache_cache_executables`]).
fn apply_rust_compiler_cache_env(
    cmd: &mut tokio::process::Command,
    wrapper: Option<&str>,
    cache_executables: Option<std::ffi::OsString>,
) {
    let Some(wrapper) = wrapper else {
        return;
    };
    cmd.env("RUSTC_WRAPPER", wrapper);
    if wrapper == "kache" {
        cmd.env(
            KACHE_CACHE_EXECUTABLES_VAR,
            kache_cache_executables(cache_executables),
        );
    }
}

/// The `KACHE_CACHE_EXECUTABLES` value a worker command gets.
///
/// kache's own default caches *binaries and test executables*, which are
/// exactly the artefacts a per-worker target directory already holds: caching
/// them writes each crate test binary twice and grows the store past its cap,
/// where the resulting GC evicts the third-party dependencies that a hit would
/// have saved. Worker commands therefore opt out; an operator's explicit value
/// wins verbatim.
fn kache_cache_executables(ambient: Option<std::ffi::OsString>) -> String {
    match ambient {
        Some(value) => value.to_string_lossy().into_owned(),
        None => "0".to_string(),
    }
}

/// Stable key of the canonical repository root, including linked worktrees.
pub(crate) fn repo_key(repo: &Path) -> anyhow::Result<String> {
    let root = crate::agent::sandbox::find_git_common_dir(repo)
        .and_then(|git| git.parent().map(Path::to_path_buf))
        .unwrap_or_else(|| repo.to_path_buf())
        .canonicalize()?;
    // FNV-1a is fixed across processes and Rust versions, unlike DefaultHasher.
    let mut hash = 0xcbf29ce484222325u64;
    for byte in root.as_os_str().as_encoded_bytes() {
        hash = (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3);
    }
    Ok(format!("{hash:016x}"))
}

/// `n`th build directory of a repository's pool. Directories are indexed by
/// lease order, not by admission slot: a live worker holds one exclusively for
/// its whole lifetime, so two workers of one repository can never share a dir.
pub(crate) fn build_dir(repo: &Path, index: usize) -> anyhow::Result<PathBuf> {
    Ok(crate::worktree::swe_base_dir().join(format!("swe-target-{}-{index}", repo_key(repo)?)))
}

/// Remove every build directory leased for `repo`.
///
/// A temporary repository is deleted with the worker that used it, but its
/// leased build directories are filed under the *repository's* key in the
/// scratch base and would otherwise outlive it: a [`BuildDirLease`] only marks
/// the directory reusable, it never deletes it. Removing a temporary repo
/// therefore has to take its leases with it.
pub fn remove_build_dir_leases(repo: &Path) {
    let Ok(key) = repo_key(repo) else {
        return;
    };
    let Ok(entries) = std::fs::read_dir(crate::worktree::swe_base_dir()) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if build_dir_repo(&path) == Some(key.as_str()) {
            let _ = std::fs::remove_dir_all(&path);
        }
    }
}

fn build_dir_repo(path: &Path) -> Option<&str> {
    let name = path.file_name()?.to_str()?.strip_prefix("swe-target-")?;
    let (repo, index) = name.split_once('-')?;
    (repo.len() == 16
        && repo.bytes().all(|b| b.is_ascii_hexdigit())
        && !index.is_empty()
        && index.bytes().all(|b| b.is_ascii_digit()))
    .then_some(repo)
}

/// Directories left by the retired pool naming `swe-target-<repo-key>-slot<k>`.
/// No worker leases them any more, so the sweep reclaims an idle one at once.
fn is_legacy_slot_dir(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
        return false;
    };
    let Some(name) = name.strip_prefix("swe-target-") else {
        return false;
    };
    let Some((repo, slot)) = name.split_once("-slot") else {
        return false;
    };
    repo.len() == 16
        && repo.bytes().all(|b| b.is_ascii_hexdigit())
        && !slot.is_empty()
        && slot.bytes().all(|b| b.is_ascii_digit())
}

/// A build directory the sweep may reclaim: a leased pool dir or a legacy slot
/// dir.
fn is_sweepable_dir(path: &Path) -> bool {
    build_dir_repo(path).is_some() || is_legacy_slot_dir(path)
}

#[derive(Debug, Clone)]
struct TargetEntry {
    dir: PathBuf,
    last_used: std::time::SystemTime,
    size: u64,
    idle: bool,
}

/// TTL is repository-wide; the size cap evicts idle dirs in deterministic LRU
/// order. Active bytes count toward the cap but can never be evicted. A legacy
/// slot dir is never leased again, so an idle one goes regardless of the TTL.
fn target_evictions(
    entries: &[TargetEntry],
    now: std::time::SystemTime,
    ttl: std::time::Duration,
    max_bytes: u64,
) -> Vec<PathBuf> {
    let mut latest = std::collections::BTreeMap::new();
    let mut active_repos = std::collections::BTreeSet::new();
    for entry in entries {
        if let Some(repo) = build_dir_repo(&entry.dir) {
            let at = latest.entry(repo).or_insert(entry.last_used);
            *at = (*at).max(entry.last_used);
            if !entry.idle {
                active_repos.insert(repo);
            }
        }
    }
    let mut total = entries
        .iter()
        .fold(0u64, |sum, e| sum.saturating_add(e.size));
    let mut removed = Vec::new();
    for entry in entries {
        if entry.idle && is_legacy_slot_dir(&entry.dir) {
            removed.push(entry.dir.clone());
            total = total.saturating_sub(entry.size);
        }
    }
    let mut ordered: Vec<_> = entries
        .iter()
        .filter(|e| e.idle && !is_legacy_slot_dir(&e.dir))
        .collect();
    ordered.sort_by(|a, b| {
        a.last_used
            .cmp(&b.last_used)
            .then_with(|| a.dir.cmp(&b.dir))
    });
    for entry in &ordered {
        let expired = build_dir_repo(&entry.dir)
            .and_then(|repo| latest.get(repo))
            .is_some_and(|at| {
                !active_repos.contains(build_dir_repo(&entry.dir).expect("build dir repo"))
                    && now.duration_since(*at).unwrap_or_default() >= ttl
            });
        if expired {
            removed.push(entry.dir.clone());
            total = total.saturating_sub(entry.size);
        }
    }
    for entry in ordered {
        if total <= max_bytes {
            break;
        }
        if !removed.contains(&entry.dir) {
            removed.push(entry.dir.clone());
            total = total.saturating_sub(entry.size);
        }
    }
    removed
}

fn lock_file(path: &Path) -> std::io::Result<std::fs::File> {
    std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)
}

#[cfg(unix)]
fn flock(file: &std::fs::File, exclusive: bool, nonblocking: bool) -> std::io::Result<()> {
    use std::os::fd::AsRawFd;
    let flags = if exclusive {
        libc::LOCK_EX
    } else {
        libc::LOCK_SH
    } | if nonblocking { libc::LOCK_NB } else { 0 };
    // SAFETY: flock only reads the live descriptor and integer flags.
    if unsafe { libc::flock(file.as_raw_fd(), flags) } != 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

#[cfg(not(unix))]
fn flock(_file: &std::fs::File, _exclusive: bool, _nonblocking: bool) -> std::io::Result<()> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "Target leases require Unix",
    ))
}

fn build_dir_lock_path(dir: &Path) -> PathBuf {
    dir.join(".swe-target.lease")
}

/// Exclusive lease on one build directory of a repository's pool.
///
/// The `flock` on the directory's lock file is held for the guard's whole
/// lifetime, so no second worker -- in this process or in another one -- can
/// lease the same directory while this worker is live, and the sweep sees the
/// directory as busy. Dropping the guard releases the directory for a later
/// worker, which then inherits its warm dependency cache -- unless the directory
/// outgrew `MINI_SWE_TARGET_SLOT_MAX_GIB`, in which case it is emptied first,
/// since cargo would otherwise have accumulated every artifact set the slot ever
/// built.
pub struct BuildDirLease {
    dir: PathBuf,
    lock: std::fs::File,
}

impl BuildDirLease {
    /// Lease the lowest-indexed free directory of `repo`, creating a new one
    /// when every directory the repository already has is live.
    pub fn acquire(repo: &Path) -> std::io::Result<Self> {
        let base = crate::worktree::swe_base_dir();
        std::fs::create_dir_all(&base)?;
        // The sweep lock keeps eviction from removing a directory between the
        // probe below and the lock that proves it free.
        let global = lock_file(&base.join(".swe-target-sweep.lock"))?;
        flock(&global, true, false)?;
        let mut index = 0usize;
        let lease = loop {
            let dir = build_dir(repo, index).map_err(std::io::Error::other)?;
            std::fs::create_dir_all(&dir)?;
            let lock = lock_file(&build_dir_lock_path(&dir))?;
            if flock(&lock, true, true).is_ok() {
                break Self { dir, lock };
            }
            index += 1;
        };
        drop(global);
        lease.touch()?;
        // The slot is leased now, so it is idle and over-cap contents are dead
        // weight: the sweep only evicts whole slots, which throws away the
        // warm dependency cache too.
        enforce_slot_cap(&lease.dir, slot_max_bytes());

        start_target_sweep();
        Ok(lease)
    }

    /// The directory this worker builds in.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Refresh the directory's last-used stamp for the sweep's LRU order.
    fn touch(&self) -> std::io::Result<()> {
        self.lock.set_modified(std::time::SystemTime::now())
    }
}

impl Drop for BuildDirLease {
    fn drop(&mut self) {
        let _ = self.touch();
    }
}

fn target_size(dir: &Path) -> std::io::Result<u64> {
    // Stream each level and never follow symlinks into another worker's data.
    let mut stack = vec![std::fs::read_dir(dir)?];
    let mut size = 0u64;
    while let Some(level) = stack.last_mut() {
        match level.next() {
            Some(entry) => {
                let entry = entry?;
                if entry.file_type()?.is_dir() {
                    if stack.len() >= 128 {
                        return Err(std::io::Error::other(
                            "Build target directory nesting exceeds scan limit",
                        ));
                    }
                    stack.push(std::fs::read_dir(entry.path())?);
                } else {
                    size = size.saturating_add(entry.metadata()?.len());
                }
            }
            None => {
                stack.pop();
            }
        }
    }
    Ok(size)
}

/// Bytes one leased build slot may hold before it is emptied, from
/// [`SLOT_MAX_GIB_ENV`] (default 4 GiB). `0` disables the cap.
fn slot_max_bytes() -> u64 {
    crate::config::env_parse::<u64>(SLOT_MAX_GIB_ENV)
        .unwrap_or(4)
        .saturating_mul(1024 * 1024 * 1024)
}

/// Empty a leased build slot that grew past `max_bytes`.
///
/// Cargo never deletes a stale artifact: every rustc, dependency or flag change
/// and every renamed test target leaves a full extra set behind, so a slot that
/// stays in one pool for weeks grows without bound (39 GiB across six slots was
/// measured on one host). The caller holds the slot's exclusive lease, so no
/// other worker can be building in it; the compile cache restores the
/// third-party crates and the cost is one rebuild of this crate. A disabled cap,
/// a slot within it and a slot that cannot be emptied are all left alone.
fn enforce_slot_cap(dir: &Path, max_bytes: u64) {
    if max_bytes == 0 {
        return;
    }
    // Metadata only: the walk never reads a file's contents, so a slot with
    // tens of thousands of artifacts costs one `stat` each.
    let Ok(size) = target_size(dir) else {
        return;
    };
    if size <= max_bytes {
        return;
    }
    match empty_dir_except(dir, &build_dir_lock_path(dir)) {
        Ok(()) => {
            tracing::info!(path = %dir.display(), bytes = size, "Emptied oversized build target")
        }
        Err(error) => {
            tracing::warn!(%error, path = %dir.display(), "Failed to empty oversized build target")
        }
    }
}

/// Remove every entry of `dir` except `keep`, so the next build starts empty.
///
/// The lease file lives inside the directory and the worker that leased it
/// still holds its `flock`: deleting it would drop that worker's claim and let
/// a second worker lease the directory while it builds.
fn empty_dir_except(dir: &Path, keep: &Path) -> std::io::Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        // `file_type` does not follow symlinks, so a link inside the slot is
        // removed as a link rather than as whatever it points at.
        let path = entry.path();
        if path == keep {
            continue;
        }
        if entry.file_type()?.is_dir() {
            std::fs::remove_dir_all(&path)?;
        } else {
            std::fs::remove_file(&path)?;
        }
    }
    Ok(())
}

fn sweep_targets(base: &Path, ttl: std::time::Duration, max_bytes: u64) -> std::io::Result<()> {
    let global = lock_file(&base.join(".swe-target-sweep.lock"))?;
    flock(&global, true, false)?;
    let mut entries = Vec::new();
    let mut idle_leases = Vec::new();
    for dir in std::fs::read_dir(base)?.flatten() {
        if !dir.file_type()?.is_dir() || !is_sweepable_dir(&dir.path()) {
            continue;
        }
        let path = dir.path();
        let lease = lock_file(&build_dir_lock_path(&path))?;
        let idle = flock(&lease, true, true).is_ok();
        let last_used = lease.metadata()?.modified()?;
        // A failed scan cannot safely participate in size eviction.
        let Ok(size) = target_size(&path) else {
            continue;
        };
        entries.push(TargetEntry {
            dir: path,
            last_used,
            size,
            idle,
        });
        idle_leases.push(lease);
    }
    let now = std::time::SystemTime::now();
    for path in target_evictions(&entries, now, ttl, max_bytes) {
        if let Err(error) = std::fs::remove_dir_all(&path) {
            // The lock file lives inside the directory, so the global lock keeps
            // acquisitions out until the whole name is gone.
            tracing::warn!(%error, path = %path.display(), "Failed to evict idle build target");
        }
    }
    Ok(())
}

pub(crate) fn start_target_sweep() {
    static START: std::sync::Once = std::sync::Once::new();
    START.call_once(|| {
        std::thread::spawn(|| {
            loop {
                let ttl = crate::config::env_parse::<u64>("HUB_TARGET_TTL_HOURS").unwrap_or(24);
                let cap = crate::config::env_parse::<u64>("HUB_TARGET_MAX_GB").unwrap_or(40);
                // Build dirs are created only in the configured base; legacy
                // per-worktree targets in other temp roots belong to worktree prune.
                for base in [crate::worktree::swe_base_dir()] {
                    if let Err(error) = sweep_targets(
                        &base,
                        std::time::Duration::from_secs(ttl.saturating_mul(3600)),
                        cap.saturating_mul(1024 * 1024 * 1024),
                    ) {
                        tracing::debug!(%error, "Build target sweep unavailable");
                    }
                }
                std::thread::sleep(std::time::Duration::from_secs(300));
            }
        });
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_dir_naming_is_stable_and_bounded() {
        let _lock = crate::agent::env::ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let repo = crate::worktree::swe_base_dir();
        let first = super::repo_key(&repo).expect("Repository root must be usable");
        let second = super::repo_key(&repo).expect("Repository root must be usable");
        assert_eq!(first, second, "The same repository must keep one key");
        assert_eq!(first.len(), 16, "got: {first:?}");
        let base = crate::worktree::swe_base_dir();
        let dir0 = super::build_dir(&repo, 0).expect("Build dir must resolve");
        let dir1 = super::build_dir(&repo, 1).expect("Build dir must resolve");
        assert_eq!(
            dir0.parent(),
            Some(base.as_path()),
            "{dir0:?} must live in the base"
        );
        assert_eq!(
            dir1.parent(),
            Some(base.as_path()),
            "{dir1:?} must live in the base"
        );
        assert_ne!(dir0, dir1, "Two workers must never share a build dir");
        let name0 = dir0
            .file_name()
            .expect("Named target")
            .to_str()
            .expect("UTF-8 target");
        assert!(
            name0.starts_with("swe-target-") && name0.contains(&first) && name0.ends_with("-0"),
            "got: {name0:?}"
        );
    }

    #[test]
    fn test_target_eviction_prefers_idle_lru_under_the_cap() {
        use std::time::{Duration, SystemTime};
        fn entry(dir: &str, age_secs: u64, size: u64, idle: bool) -> super::TargetEntry {
            super::TargetEntry {
                dir: PathBuf::from(dir),
                last_used: SystemTime::now() - Duration::from_secs(age_secs),
                size,
                idle,
            }
        }
        let now = SystemTime::now();
        let ttl = Duration::from_secs(24 * 3600);
        let shared = entry(
            "/base/swe-target-0123456789abcdef-0",
            25 * 3600 + 60,
            10,
            false,
        );
        let busy = entry(
            "/base/swe-target-aaaaaaaaaaaaaaaa-0",
            25 * 3600 + 60,
            10,
            false,
        );
        let old_idle = entry(
            "/base/swe-target-bbbbbbbbbbbbbbbb-0",
            25 * 3600 + 60,
            10,
            true,
        );
        let trimmed = super::target_evictions(
            &[shared.clone(), busy.clone(), old_idle.clone()],
            now,
            ttl,
            u64::MAX,
        );
        assert_eq!(
            trimmed,
            vec![old_idle.dir.clone()],
            "Only an idle dir in a repository unused past the TTL is evicted"
        );
        // A warm cap keeps the newest idle dirs and evicts least-recently-used first.
        let mut entries = Vec::new();
        for (index, age_secs) in [(0, 400), (1, 300), (2, 200), (3, 100)] {
            entries.push(entry(
                &format!("/base/swe-target-cccccccccccccccc-{index}"),
                age_secs,
                10,
                true,
            ));
        }
        let trimmed = super::target_evictions(&entries, now, ttl, 25);
        assert_eq!(
            trimmed,
            vec![entries[0].dir.clone(), entries[1].dir.clone()],
            "The cap must remove the oldest idle dirs first, deterministically"
        );
        assert!(!trimmed.contains(&entries[2].dir) && !trimmed.contains(&entries[3].dir));
    }

    #[test]
    fn canonical_repo_aliases_share_a_key() {
        let root = std::env::current_dir().unwrap();
        assert_eq!(repo_key(&root).unwrap(), repo_key(&root.join(".")).unwrap());
        if let Some(common) = crate::agent::sandbox::find_git_common_dir(&root) {
            assert_eq!(
                repo_key(&root).unwrap(),
                repo_key(common.parent().unwrap()).unwrap()
            );
        }
        #[cfg(unix)]
        {
            let base = crate::worktree::swe_base_dir()
                .join(format!("swe-key-test-{}", uuid::Uuid::new_v4()));
            std::fs::create_dir_all(&base).unwrap();
            let alias = base.join("alias");
            std::os::unix::fs::symlink(&root, &alias).unwrap();
            assert_eq!(repo_key(&root).unwrap(), repo_key(&alias).unwrap());
            std::fs::remove_dir_all(base).unwrap();
        }
    }

    #[test]
    fn ttl_is_repo_wide_and_cap_never_evicts_busy_dirs() {
        use std::time::{Duration, UNIX_EPOCH};
        let entries = vec![
            TargetEntry {
                dir: "/base/swe-target-0123456789abcdef-0".into(),
                last_used: UNIX_EPOCH,
                size: 10,
                idle: true,
            },
            TargetEntry {
                dir: "/base/swe-target-0123456789abcdef-1".into(),
                last_used: UNIX_EPOCH + Duration::from_secs(90),
                size: 10,
                idle: true,
            },
        ];
        let now = UNIX_EPOCH + Duration::from_secs(100);
        assert!(target_evictions(&entries, now, Duration::from_secs(20), 30).is_empty());
        let mut busy = entries.clone();
        busy[1].idle = false;
        assert!(target_evictions(&busy, now, Duration::ZERO, 30).is_empty());
        assert_eq!(
            target_evictions(&busy, now, Duration::ZERO, 0),
            vec![busy[0].dir.clone()]
        );
        let mut tied = entries.clone();
        tied[1].last_used = UNIX_EPOCH;
        tied.reverse();
        assert_eq!(
            target_evictions(&tied, now, Duration::from_secs(200), 10),
            vec![entries[0].dir.clone()]
        );
    }

    #[cfg(unix)]
    #[test]
    fn sweep_preserves_leased_targets_and_removes_idle_targets() {
        crate::agent::env::with_env_lock(|| {
            let base = crate::worktree::swe_base_dir()
                .join(format!("swe-sweep-test-{}", uuid::Uuid::new_v4()));
            std::fs::create_dir_all(&base).unwrap();
            let previous = std::env::var("SWE_TEMP_DIR").ok();
            // SAFETY: the environment lock is held for the whole closure.
            unsafe { std::env::set_var("SWE_TEMP_DIR", &base) };
            let lease = BuildDirLease::acquire(&base).unwrap();
            let target = lease.dir().to_path_buf();
            std::fs::write(target.join("artifact"), b"build").unwrap();
            sweep_targets(&base, std::time::Duration::ZERO, 0).unwrap();
            assert!(
                target.join("artifact").exists(),
                "A leased dir must never be swept"
            );
            drop(lease);
            sweep_targets(&base, std::time::Duration::ZERO, 0).unwrap();
            assert!(
                !target.exists(),
                "A released dir must be swept once it is idle"
            );
            // SAFETY: the environment lock is still held.
            unsafe {
                match previous {
                    Some(value) => std::env::set_var("SWE_TEMP_DIR", value),
                    None => std::env::remove_var("SWE_TEMP_DIR"),
                }
            }
            let _ = std::fs::remove_dir_all(base);
        });
    }

    /// A repository of this test's own, so the pool of build dirs keyed to it
    /// is this test's alone.
    fn slot_cap_repo(tag: &str) -> PathBuf {
        let repo = std::env::temp_dir().join(format!(
            "swe-slot-cap-repo-{tag}-{}",
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&repo).unwrap();
        repo
    }

    /// Run `body` with the slot cap a worker would see, then restore the
    /// environment even if the body panics. The variable is process-global, so
    /// this holds the shared environment lock; the slots themselves live in the
    /// scratch base under the repository's own key.
    fn with_slot_cap(cap_gib: &str, body: impl FnOnce()) {
        struct Restore(Option<String>);
        impl Drop for Restore {
            fn drop(&mut self) {
                // SAFETY: the caller holds the environment lock.
                unsafe {
                    match self.0.take() {
                        Some(value) => std::env::set_var(SLOT_MAX_GIB_ENV, value),
                        None => std::env::remove_var(SLOT_MAX_GIB_ENV),
                    }
                }
            }
        }
        crate::agent::env::with_env_lock(|| {
            let restore = Restore(std::env::var(SLOT_MAX_GIB_ENV).ok());
            // SAFETY: the environment lock is held for the whole scope.
            unsafe { std::env::set_var(SLOT_MAX_GIB_ENV, cap_gib) };
            body();
            drop(restore);
        });
    }

    /// A file that *looks* like `bytes` of build output without occupying them:
    /// `set_len` leaves the blocks unallocated, so the GiB cap can be exercised
    /// without writing a gigabyte.
    fn seed_stale_artifact(dir: &Path, bytes: u64) -> PathBuf {
        let stale = dir.join("debug").join("deps").join("stale");
        std::fs::create_dir_all(stale.parent().unwrap()).unwrap();
        std::fs::File::create(&stale)
            .unwrap()
            .set_len(bytes)
            .unwrap();
        stale
    }

    /// One gibibyte plus a byte: the smallest size that breaks a 1 GiB cap.
    const OVER_ONE_GIB: u64 = (1 << 30) + 1;

    /// Cargo keeps every superseded artifact set, so a slot that stays in one
    /// pool for weeks grows without bound (39 GiB across six slots was measured
    /// on one host). The next worker to lease it must find it emptied rather
    /// than pay for that disk, at the cost of one rebuild of this crate.
    #[cfg(unix)]
    #[test]
    fn an_idle_slot_over_the_cap_is_emptied_when_it_is_leased() {
        let repo = slot_cap_repo("trim");
        with_slot_cap("1", || {
            let first = BuildDirLease::acquire(&repo).unwrap();
            let dir = first.dir().to_path_buf();
            // The worker ends: the slot is idle again, as it is between two.
            drop(first);
            let stale = seed_stale_artifact(&dir, OVER_ONE_GIB);

            let second = BuildDirLease::acquire(&repo).unwrap();
            assert_eq!(
                second.dir(),
                dir.as_path(),
                "the over-cap slot is the one a worker must still get"
            );
            assert!(
                dir.is_dir(),
                "only the contents go, or the worker's target directory is gone"
            );
            assert!(
                !stale.exists() && !dir.join("debug").exists(),
                "an idle slot over the cap must be emptied on lease"
            );
            assert!(
                build_dir_lock_path(&dir).is_file(),
                "the lease file must survive: it is what the worker still holds"
            );
            drop(second);
            let _ = std::fs::remove_dir_all(&dir);
        });
        let _ = std::fs::remove_dir_all(&repo);
    }

    /// The cap is enforced on the slot the caller holds and on nothing else:
    /// emptying a slot another worker is building in would delete artifacts
    /// under a live build.
    #[cfg(unix)]
    #[test]
    fn a_slot_another_worker_holds_is_never_emptied() {
        let repo = slot_cap_repo("held");
        with_slot_cap("1", || {
            let held = BuildDirLease::acquire(&repo).unwrap();
            let dir = held.dir().to_path_buf();
            let stale = seed_stale_artifact(&dir, OVER_ONE_GIB);
            // A second worker of the repository leases the next free slot, so
            // the cap runs over the base again: a cap that ignored the lease
            // lock would strike here instead of here alone.
            let other = BuildDirLease::acquire(&repo).unwrap();
            assert_ne!(other.dir(), dir);
            assert!(
                stale.exists() && held.dir().join("debug").is_dir(),
                "a slot a live worker holds must never be emptied"
            );
            drop((held, other));
            let _ = std::fs::remove_dir_all(&dir);
        });
        let _ = std::fs::remove_dir_all(&repo);
    }

    /// `MINI_SWE_TARGET_SLOT_MAX_GIB=0` turns the cap off, for a host whose
    /// build slots are worth more than the rebuild.
    #[cfg(unix)]
    #[test]
    fn a_zero_cap_leaves_an_over_cap_slot_alone() {
        let repo = slot_cap_repo("off");
        with_slot_cap("0", || {
            let first = BuildDirLease::acquire(&repo).unwrap();
            let dir = first.dir().to_path_buf();
            drop(first);
            let stale = seed_stale_artifact(&dir, OVER_ONE_GIB);

            let second = BuildDirLease::acquire(&repo).unwrap();
            assert_eq!(second.dir(), dir.as_path());
            assert!(
                stale.exists(),
                "a zero cap must leave the slot exactly as it was"
            );
            drop(second);
            let _ = std::fs::remove_dir_all(&dir);
        });
        let _ = std::fs::remove_dir_all(&repo);
    }

    #[cfg(unix)]
    #[test]
    fn sweep_evicts_legacy_slot_dirs_regardless_of_ttl() {
        let base = crate::worktree::swe_base_dir()
            .join(format!("swe-legacy-slot-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&base).unwrap();
        let key = "0123456789abcdef";
        let legacy = base.join(format!("swe-target-{key}-slot3"));
        let current = base.join(format!("swe-target-{key}-0"));
        std::fs::create_dir_all(&legacy).unwrap();
        std::fs::create_dir_all(&current).unwrap();
        std::fs::write(legacy.join("artifact"), b"stale").unwrap();
        std::fs::write(current.join("artifact"), b"warm").unwrap();
        assert!(is_legacy_slot_dir(&legacy) && !is_legacy_slot_dir(&current));
        assert!(is_sweepable_dir(&legacy) && is_sweepable_dir(&current));
        // Hold the current dir's lease so the sweep must keep it even under a
        // cap of zero bytes.
        let lease = lock_file(&build_dir_lock_path(&current)).unwrap();
        flock(&lease, true, true).unwrap();
        sweep_targets(&base, std::time::Duration::from_secs(24 * 3600), 0).unwrap();
        assert!(
            !legacy.exists(),
            "A legacy slot dir must be evicted without waiting for the TTL"
        );
        assert!(
            current.join("artifact").exists(),
            "A leased current dir must be kept"
        );
        drop(lease);
        sweep_targets(&base, std::time::Duration::ZERO, 0).unwrap();
        assert!(
            !current.exists(),
            "A released current dir must still follow the usual rules"
        );
        let _ = std::fs::remove_dir_all(base);
    }

    #[test]
    fn test_shared_cache_root_default() {
        let root = shared_cache_root();
        assert!(root.ends_with("swe-cache"));
    }

    #[test]
    fn test_cache_dirs_creation() {
        let dirs = cache_dirs();
        assert!(dirs.root.is_dir());
        assert!(dirs.uv.is_dir());
        assert!(dirs.pip.is_dir());
        assert!(dirs.npm.is_dir());
        assert!(dirs.go_build.is_dir());
    }

    #[test]
    fn test_parse_custom_cache_binds() {
        let binds = parse_custom_cache_binds("/host/a:/guest/a, /host/b");
        assert_eq!(binds.len(), 2);
        assert_eq!(
            binds[0],
            (PathBuf::from("/host/a"), PathBuf::from("/guest/a"))
        );
        assert_eq!(
            binds[1],
            (PathBuf::from("/host/b"), PathBuf::from("/host/b"))
        );

        let empty = parse_custom_cache_binds("  , , ");
        assert!(empty.is_empty());
    }

    #[test]
    fn test_apply_shared_cache_env() {
        let mut cmd = tokio::process::Command::new("true");
        apply_shared_cache_env(&mut cmd);
    }

    /// Every ecosystem cache the child is pointed at lives under the shared
    /// cache root, so one Landlock/bwrap grant covers them all and no cache
    /// directory is ever granted beside a credential file.
    #[test]
    fn test_every_ecosystem_cache_lives_under_the_shared_root() {
        let dirs = cache_dirs();
        for cache in [
            &dirs.kache,
            &dirs.uv,
            &dirs.pip,
            &dirs.npm,
            &dirs.yarn,
            &dirs.pnpm_home,
            &dirs.pnpm_store,
            &dirs.go_build,
            &dirs.go_mod,
            &dirs.maven,
            &dirs.gradle,
        ] {
            assert!(
                cache.starts_with(&dirs.root),
                "{} must live under the shared cache root",
                cache.display()
            );
        }
        assert!(dirs.maven.is_dir(), "the Maven cache must be created");
        assert!(dirs.gradle.is_dir(), "the Gradle cache must be created");
    }

    /// The JVM caches are redirected through their own variables, and the
    /// Maven repository is *appended* to the job cap `apply_build_env` sets
    /// rather than replacing it.
    #[test]
    fn test_shared_cache_env_points_the_jvm_tools_at_the_shared_caches() {
        let mut cmd = tokio::process::Command::new("true");
        cmd.env("MAVEN_OPTS", "-T4");
        apply_shared_cache_env(&mut cmd);

        let dirs = cache_dirs();
        let env = |name: &str| {
            cmd.as_std()
                .get_envs()
                .find(|(key, _)| *key == std::ffi::OsStr::new(name))
                .and_then(|(_, value)| value.map(|v| v.to_string_lossy().into_owned()))
                .expect("the variable must be set")
        };
        assert_eq!(env("GRADLE_USER_HOME"), dirs.gradle.to_string_lossy());
        let maven_opts = env("MAVEN_OPTS");
        assert!(
            maven_opts.contains("-T4"),
            "the granted job cap must survive: {maven_opts}"
        );
        assert!(
            maven_opts.contains(&format!("-Dmaven.repo.local={}", dirs.maven.display())),
            "the shared local repository must be added: {maven_opts}"
        );
    }

    /// kache's executable caching doubles the write of a worker's own test
    /// binaries and evicts the dependency cache, so worker commands opt out;
    /// an operator-exported value is forwarded verbatim.
    #[test]
    fn test_kache_does_not_cache_executables_by_default() {
        let env = |cmd: &tokio::process::Command, name: &str| {
            cmd.as_std()
                .get_envs()
                .find(|(key, _)| *key == std::ffi::OsStr::new(name))
                .and_then(|(_, value)| value.map(|v| v.to_string_lossy().into_owned()))
        };

        let mut cmd = tokio::process::Command::new("true");
        apply_rust_compiler_cache_env(&mut cmd, Some("kache"), None);
        assert_eq!(env(&cmd, "RUSTC_WRAPPER").as_deref(), Some("kache"));
        assert_eq!(env(&cmd, KACHE_CACHE_EXECUTABLES_VAR).as_deref(), Some("0"));

        // The operator's explicit choice is respected, not overridden.
        let mut cmd = tokio::process::Command::new("true");
        apply_rust_compiler_cache_env(&mut cmd, Some("kache"), Some("1".into()));
        assert_eq!(env(&cmd, KACHE_CACHE_EXECUTABLES_VAR).as_deref(), Some("1"));

        // sccache has no equivalent knob, and no wrapper sets nothing.
        let mut cmd = tokio::process::Command::new("true");
        apply_rust_compiler_cache_env(&mut cmd, Some("sccache"), None);
        assert_eq!(env(&cmd, KACHE_CACHE_EXECUTABLES_VAR), None);
        let mut cmd = tokio::process::Command::new("true");
        apply_rust_compiler_cache_env(&mut cmd, None, None);
        assert_eq!(env(&cmd, "RUSTC_WRAPPER"), None);
    }
}
