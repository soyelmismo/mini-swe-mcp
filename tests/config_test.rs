//! Integration tests for [`mini_swe_mcp::config::xdg_config_dir_from`].
//!
//! The helper resolves the directory used to look up the agent configuration
//! file. It is a pure function of the two environment variables it is handed,
//! which makes it directly testable without mutating the process environment.

use mini_swe_mcp::config::xdg_config_dir_from;
use std::path::PathBuf;

/// 1. A custom `XDG_CONFIG_HOME` wins outright and is returned verbatim,
///    even when `HOME` is also set.
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

/// 2. When `XDG_CONFIG_HOME` is absent, fall back to `$HOME/.config`.
#[test]
fn test_missing_xdg_config_dir_falls_back_to_home() {
    let dir = xdg_config_dir_from(None, Some("/home/user"));
    assert_eq!(
        dir,
        Some(PathBuf::from("/home/user").join(".config")),
        "missing XDG_CONFIG_HOME should fall back to $HOME/.config"
    );
}

/// 3. Blank/whitespace-only `XDG_CONFIG_HOME` values are ignored, so the
///    `$HOME/.config` fallback still applies.
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

/// 4. With neither variable usable there is nothing to resolve, so `None`.
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
