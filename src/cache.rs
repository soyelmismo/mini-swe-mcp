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
pub fn has_kache() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        if let Some(home) = std::env::var_os("HOME").map(PathBuf::from)
            && home.join(".local/bin/kache").is_file()
        {
            return true;
        }
        std::process::Command::new("kache")
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    })
}

/// Check if `sccache` is available in PATH.
pub fn has_sccache() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        std::process::Command::new("sccache")
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    })
}

/// Append bubblewrap arguments for mounting the shared cache root, user tool caches,
/// and any custom cache binds specified in `SWE_SHARED_CACHES`.
pub fn append_bwrap_cache_args(cmd: &mut tokio::process::Command, home: Option<&Path>) {
    let dirs = CacheDirs::new();
    dirs.ensure_dirs();

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
    let dirs = CacheDirs::new();

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
        let dirs = CacheDirs::new();
        dirs.ensure_dirs();
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
