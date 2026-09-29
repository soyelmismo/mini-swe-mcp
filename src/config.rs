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
}
