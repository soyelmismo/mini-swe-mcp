//! Stamp every build with an identity the hub handshake can compare.
//!
//! `CARGO_PKG_VERSION` alone cannot tell a rebuilt binary from the one already
//! running the hub: both say `0.1.0`, so a `cargo build` looked like a no-op to
//! the version handshake. These two values can. `MINI_SWE_BUILD_TS` is the
//! build's clock in unix nanoseconds, comparable across builds, and
//! `MINI_SWE_BUILD_ID` is a token unique to this one, so the pair identifies a
//! build and orders it. Both are compiled into every target of the package.

use std::time::{SystemTime, UNIX_EPOCH};

/// FNV-1a over the timestamp: an opaque id beside the clock it came from.
fn build_id(nanos: u128) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in nanos.to_le_bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    // Any source change is a new build, so restamp instead of reusing the
    // cached script output.
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=Cargo.toml");
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |since| since.as_nanos());
    println!("cargo:rustc-env=MINI_SWE_BUILD_TS={nanos}");
    println!("cargo:rustc-env=MINI_SWE_BUILD_ID={:016x}", build_id(nanos));
}
