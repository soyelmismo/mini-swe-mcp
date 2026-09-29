//! Shared wall-clock helper.
//!
//! Its own module so the state module (terminal-record TTLs) and the pool /
//! runner modules (registry timestamps) can use it without a cyclic dependency.

use std::time::{SystemTime, UNIX_EPOCH};

pub fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
