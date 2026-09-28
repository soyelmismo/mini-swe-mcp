use std::env;
use std::path::PathBuf;

pub fn xdg_config_dir() -> Option<PathBuf> {
    xdg_config_dir_from(
        env::var("XDG_CONFIG_HOME").ok().as_deref(),
        env::var("HOME").ok().as_deref(),
    )
}

pub fn xdg_config_dir_from(xdg: Option<&str>, home: Option<&str>) -> Option<PathBuf> {
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

    #[test]
    fn test_xdg_config_dir_with_xdg_env() {
        let dir = xdg_config_dir_from(Some("/custom/xdg"), Some("/home/user"));
        assert_eq!(dir, Some(PathBuf::from("/custom/xdg")));
    }

    #[test]
    fn test_xdg_config_dir_fallback_home() {
        let dir = xdg_config_dir_from(None, Some("/home/user"));
        assert_eq!(dir, Some(PathBuf::from("/home/user/.config")));

        let dir_empty_xdg = xdg_config_dir_from(Some("   "), Some("/home/user"));
        assert_eq!(dir_empty_xdg, Some(PathBuf::from("/home/user/.config")));
    }

    #[test]
    fn test_xdg_config_dir_none_when_both_empty() {
        assert_eq!(xdg_config_dir_from(None, None), None);
        assert_eq!(xdg_config_dir_from(Some(""), Some("")), None);
    }
}
