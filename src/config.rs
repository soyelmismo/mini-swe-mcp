use std::env;
use std::path::PathBuf;

/// Resolve the agent configuration directory.
///
/// Prefers `XDG_CONFIG_HOME` (when non-blank), else `$HOME/.config`. Returns
/// `None` when neither variable yields a usable value.
pub fn xdg_config_dir() -> Option<PathBuf> {
    xdg_config_dir_from(
        env::var("XDG_CONFIG_HOME").ok().as_deref(),
        env::var("HOME").ok().as_deref(),
    )
}

/// Pure core of [`xdg_config_dir`], parameterized over the two environment
/// variables so the fallback rules can be tested without mutating (unsafe in
/// edition 2024) process state. Private: no consumer outside this module.
fn xdg_config_dir_from(xdg: Option<&str>, home: Option<&str>) -> Option<PathBuf> {
    if let Some(xdg) = xdg.filter(|s| !s.trim().is_empty()) {
        Some(PathBuf::from(xdg))
    } else {
        home.filter(|s| !s.trim().is_empty())
            .map(|h| PathBuf::from(h).join(".config"))
    }
}

/// Parse the environment variable `name` into `T`.
///
/// The value is trimmed first, and an unset, blank or unparsable value all
/// yield `None` so every caller can express its own default with a single
/// `unwrap_or` / `filter` chain.
pub fn env_parse<T: std::str::FromStr>(name: &str) -> Option<T> {
    env_parse_from(name, &|key| env::var(key).ok())
}

/// Parse the variable `name` from an explicit `lookup` instead of the process environment.
///
/// `lookup` is the process environment in production (`|key| std::env::var(key).ok()`)
/// and a synthetic map in tests, so parsing is testable without mutating
/// process-global state. [`env_parse`] delegates here unchanged.
pub fn env_parse_from<T: std::str::FromStr>(
    name: &str,
    lookup: &dyn Fn(&str) -> Option<String>,
) -> Option<T> {
    let raw = lookup(name)?;
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    trimmed.parse().ok()
}

/// Worker loops allowed at once; heavy commands have a separate resource gate.
pub fn max_concurrent_workers() -> usize {
    env_parse("MAX_CONCURRENT_WORKERS").unwrap_or(128)
}

/// Process-wide in-flight LLM request limit. Zero or unset means unlimited.
pub(crate) fn llm_concurrency() -> usize {
    env_parse("HUB_LLM_CONCURRENCY").unwrap_or(0)
}

/// The available CPU cores (falls back to 2).
///
/// The admission controller divides its job count over these, so the count is
/// read once here rather than probed per decision.
pub fn cores() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(2)
}

/// Half the available CPU cores, never below one (falls back to 2).
pub fn half_the_cores() -> usize {
    (cores() / 2).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A custom `XDG_CONFIG_HOME` wins outright and is returned verbatim,
    /// even when `HOME` is also set.
    #[test]
    fn test_custom_xdg_config_dir_is_returned() {
        let dir = xdg_config_dir_from(Some("/custom/xdg"), Some("/home/user"));
        assert_eq!(
            dir,
            Some(PathBuf::from("/custom/xdg")),
            "a non-empty XDG_CONFIG_HOME must be used as-is"
        );

        // A relative-looking value is still honoured verbatim; no joining occurs.
        let relative = xdg_config_dir_from(Some("relative/xdg"), Some("/home/user"));
        assert_eq!(relative, Some(PathBuf::from("relative/xdg")));

        // It is returned as a single path component-free value, not joined with HOME.
        let with_home = xdg_config_dir_from(Some("/custom/xdg"), None);
        assert_eq!(with_home, Some(PathBuf::from("/custom/xdg")));
    }

    /// When `XDG_CONFIG_HOME` is absent, fall back to `$HOME/.config`.
    #[test]
    fn test_missing_xdg_config_dir_falls_back_to_home() {
        let dir = xdg_config_dir_from(None, Some("/home/user"));
        assert_eq!(
            dir,
            Some(PathBuf::from("/home/user").join(".config")),
            "missing XDG_CONFIG_HOME should fall back to $HOME/.config"
        );
    }

    /// Blank/whitespace-only `XDG_CONFIG_HOME` values are ignored, so the
    /// `$HOME/.config` fallback still applies.
    #[test]
    fn test_blank_xdg_config_dir_falls_back_to_home() {
        for blank in ["", " ", "   ", "\t", "\n", " \t\n "] {
            let dir = xdg_config_dir_from(Some(blank), Some("/home/user"));
            assert_eq!(
                dir,
                Some(PathBuf::from("/home/user").join(".config")),
                "whitespace-only XDG_CONFIG_HOME {blank:?} should fall back to $HOME/.config"
            );
        }
    }

    /// With neither variable usable there is nothing to resolve, so `None`.
    #[test]
    fn test_no_xdg_and_no_home_returns_none() {
        assert_eq!(xdg_config_dir_from(None, None), None);
        assert_eq!(xdg_config_dir_from(Some(""), None), None);
        assert_eq!(xdg_config_dir_from(Some("  \t "), None), None);
        assert_eq!(xdg_config_dir_from(None, Some("")), None);
        assert_eq!(xdg_config_dir_from(None, Some("   ")), None);
        assert_eq!(xdg_config_dir_from(Some(""), Some("")), None);
        assert_eq!(xdg_config_dir_from(Some("  "), Some("\t\n ")), None);
    }

    /// `env_parse_from` returns `None` when the variable is unset.
    #[test]
    fn test_env_parse_unset() {
        let lookup = |_: &str| None;
        assert_eq!(env_parse_from::<usize>("ANYTHING", &lookup), None);
        // The wrapper still reports `None` for a name the process environment
        // does not define.
        assert_eq!(
            env_parse::<usize>("MINI_SWE_ENV_PARSE_UNSET_TEST_XYZ"),
            None
        );
    }

    /// `env_parse_from` returns `None` for blank/whitespace-only values.
    #[test]
    fn test_env_parse_blank() {
        for blank in ["", " ", "   ", "\t", "\n", " \t\n "] {
            let owned = blank.to_string();
            let lookup = |_: &str| Some(owned.clone());
            assert_eq!(
                env_parse_from::<usize>("ANYTHING", &lookup),
                None,
                "blank {blank:?} should be None"
            );
        }
    }

    /// `env_parse_from` returns `None` for unparsable values.
    #[test]
    fn test_env_parse_unparsable() {
        let lookup = |_: &str| Some("not-a-number".to_string());
        assert_eq!(env_parse_from::<usize>("ANYTHING", &lookup), None);
    }

    /// `env_parse_from` parses a valid value, trimming surrounding whitespace.
    #[test]
    fn test_env_parse_valid() {
        let lookup = |_: &str| Some("  42  ".to_string());
        assert_eq!(env_parse_from::<usize>("ANYTHING", &lookup), Some(42));
    }
}
