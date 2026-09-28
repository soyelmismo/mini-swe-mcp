use std::env;
use std::path::PathBuf;

pub fn xdg_config_dir() -> Option<PathBuf> {
    env::var("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|_| env::var("HOME").map(|home| PathBuf::from(home).join(".config")))
        .ok()
}
