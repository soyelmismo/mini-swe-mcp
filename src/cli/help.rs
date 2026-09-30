//! The `--help` flag reference, next to the argv parsing it documents.

/// The `Flags:` block of `--help`, one flag per line.
///
/// `--admin` is the human operator's escape hatch: every other caller may only
/// steer, kill, collect and wait on the workers it dispatched itself, while the
/// operator can act on any of them (H-3).
pub const HELP_FLAGS: &str = "\
      --json     Output in JSON format (default is formatted plain text)
      --all      List every agent's workers, not just this one's
      --admin    Operator override: act on workers owned by any agent
  -h, --help     Print help
  -V, --version  Print version";
