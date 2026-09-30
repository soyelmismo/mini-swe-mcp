//! The `--help` flag reference, next to the argv parsing it documents.

/// The `Flags:` block of `--help`, one flag per line.
///
/// `--admin` is the human operator's escape hatch: every other caller may only
/// read, steer, kill, collect and watch the workers it dispatched itself, while the
/// operator can act on any of them (H-3).
pub const HELP_FLAGS: &str = concat!(
    "      --json     Output in JSON format (default is formatted plain text)\n",
    "      --all      List every agent's workers (requires --admin)\n",
    "      --admin    Operator override: act on workers owned by any agent\n",
    "  -h, --help     Print help\n",
    "  -V, --version  Print version",
);

#[cfg(test)]
mod tests {
    use super::HELP_FLAGS;

    /// Every flag the argv parser knows is documented, so a flag can never be
    /// accepted without being discoverable.
    #[test]
    fn every_selector_is_documented() {
        for flag in [
            "--json",
            "--all",
            "--admin",
            "-h",
            "--help",
            "-V",
            "--version",
        ] {
            assert!(
                HELP_FLAGS.contains(flag),
                "'{flag}' is accepted by the parser but missing from --help: {HELP_FLAGS}"
            );
        }
    }
}
