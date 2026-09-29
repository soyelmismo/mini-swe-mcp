//! Shared toolchain/compiler cache wiring.
//!
//! Resolves the shared cache root, lays out the well-known package/compiler
//! cache directories (`kache`, `uv`, `pip`, the Node toolchain and Go), and
//! turns that into (a) `bubblewrap` bind mounts and (b) child-process
//! environment variables. The directory layout is computed lazily once per
//! process and ensured to exist before use. The memoized tool probes
//! (`has_kache` / `has_sccache`) answer "is this compiler wrapper installed?"
//! at most once per process.

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
        if has_kache() {
            cmd.env("RUSTC_WRAPPER", "kache");
        } else if has_sccache() {
            cmd.env("RUSTC_WRAPPER", "sccache");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(binds[0], (PathBuf::from("/host/a"), PathBuf::from("/guest/a")));
        assert_eq!(binds[1], (PathBuf::from("/host/b"), PathBuf::from("/host/b")));

        let empty = parse_custom_cache_binds("  , , ");
        assert!(empty.is_empty());
    }

    #[test]
    fn test_apply_shared_cache_env() {
        let mut cmd = tokio::process::Command::new("true");
        apply_shared_cache_env(&mut cmd);
    }
}
