//! One identity per agent session: the host process that owns it.
//!
//! An agent session is the HOST process — the `claude`, `opencode` or `agy`
//! process an operator started. It spawns both the MCP stdio proxy and the
//! agent's shell commands, so naming it gives those two callers one identity
//! with no configuration at all: the MCP connection that dispatched a worker
//! and the `mini-swe-mcp watch` its shell runs afterwards see the same
//! workers, while two hosts never see each other's.
//!
//! The walk starts at the client process's *parent* and steps over the shells
//! and wrappers a host puts between itself and the binary it launches
//! (`mini-swe-mcp <- bash <- claude`), then names the first process left:
//! `host:<comm>:<pid>:<starttime>`. The start time is field 22 of
//! `/proc/<pid>/stat`, so a pid the kernel recycled never collides with the
//! agent that held it before.
//!
//! The client computes its own identity and sends it in `hub/hello`. The
//! daemon treats it as a coordination identity, exactly like the
//! `MINI_SWE_AGENT_ID` that outranks it: `SO_PEERCRED` is still what restricts
//! the hub to one uid, so a client can only misname which of that user's
//! agents it is.

use std::collections::BTreeMap;
use std::fmt;

/// Processes the ancestor walk steps over: the shells, and the small wrappers a
/// host puts between itself and the binary it launches.
///
/// The list is deliberately explicit and short. `sudo` is absent on purpose:
/// the walk stops there, because a privilege boundary is not something an
/// agent identity should be walked across.
const WRAPPERS: &[&str] = &[
    "sh", "bash", "dash", "zsh", "fish", "env", "nice", "setsid", "script", "stdbuf", "timeout",
    "xargs", "tail", "tee",
];

/// Ancestors the walk reads before giving up: a bound, so a corrupt or
/// pathological `/proc` cannot spin here.
const MAX_DEPTH: usize = 32;

/// One process in an ancestry, as `/proc/<pid>/stat` reports it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Process {
    /// Field 1: the pid.
    pub pid: u32,
    /// Field 2: the executable name, truncated by the kernel to 15 bytes.
    pub comm: String,
    /// Field 4: the parent pid.
    pub ppid: u32,
    /// Field 22: the start time in clock ticks since boot.
    pub starttime: u64,
}

/// A synthetic ancestry: pid → process. The seam [`resolve`] is tested through.
pub type Ancestry = BTreeMap<u32, Process>;

/// The host process an agent session belongs to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Host {
    pub comm: String,
    pub pid: u32,
    pub starttime: u64,
}

impl fmt::Display for Host {
    /// `host:<comm>:<pid>:<starttime>`: the start time is what keeps a reused
    /// pid from inheriting the previous agent's workers.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "host:{}:{}:{}", self.comm, self.pid, self.starttime)
    }
}

/// A host, plus the wrappers the walk stepped over to reach it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Resolution {
    pub host: Host,
    /// The wrappers skipped, nearest first.
    pub skipped: Vec<String>,
}

/// Where a client's agent identity came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Source {
    /// `MINI_SWE_AGENT_ID`: the operator named the agent.
    Override,
    /// The host process, plus the wrappers the walk stepped over.
    Host { host: Host, skipped: Vec<String> },
    /// No host process to name, so the caller's transport identity stands.
    Fallback,
}

/// What a client presents as its agent identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Identity {
    pub id: String,
    pub source: Source,
}

impl Identity {
    /// How the identity was derived, in one sentence for `whoami`.
    pub fn explain(&self) -> String {
        match &self.source {
            Source::Override => "MINI_SWE_AGENT_ID".to_string(),
            Source::Host { host, skipped } if skipped.is_empty() => {
                format!("host process {} (pid {})", host.comm, host.pid)
            }
            Source::Host { host, skipped } => format!(
                "host process {} (pid {}), skipping {}",
                host.comm,
                host.pid,
                skipped.join(", ")
            ),
            Source::Fallback => format!("no host process found; using '{}'", self.id),
        }
    }
}

/// Walk `ancestry` from `pid`'s parent and name the first ancestor that is
/// neither a shell nor a wrapper.
///
/// The client process itself is never the host: it is the thing the host
/// launched, so the walk starts one level up. `None` means every ancestor was
/// a wrapper, or the chain is unreadable — the caller then keeps the transport
/// identity it would have used anyway.
pub fn resolve(ancestry: &Ancestry, pid: u32) -> Option<Resolution> {
    let mut skipped = Vec::new();
    // The client itself is seeded as visited: a cycle in `/proc` must not walk
    // back and name the process that asked.
    let mut visited = vec![pid];
    let mut cursor = ancestry.get(&pid)?.ppid;
    for _ in 0..MAX_DEPTH {
        let process = ancestry.get(&cursor)?;
        // pid 1 is the system's init, not anybody's agent host: naming it would
        // make every session of the machine one agent.
        if cursor <= 1 || visited.contains(&cursor) {
            return None;
        }
        visited.push(cursor);
        if WRAPPERS.contains(&process.comm.as_str()) {
            skipped.push(process.comm.clone());
            cursor = process.ppid;
            continue;
        }
        return Some(Resolution {
            host: Host {
                comm: process.comm.clone(),
                pid: cursor,
                starttime: process.starttime,
            },
            skipped,
        });
    }
    None
}

/// One process as `/proc/<pid>/stat` reports it, or `None` when it is gone.
///
/// `comm` is parenthesised and may itself contain spaces or parentheses, so
/// the fields are read after the *last* `)`: field 4 is the parent pid and
/// field 22 the start time.
pub fn process(pid: u32) -> Option<Process> {
    let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    let open = stat.find('(')?;
    let close = stat.rfind(')')?;
    let comm = stat.get(open + 1..close)?.to_string();
    // Field 1 is the pid and field 2 the parenthesised comm, so the fields
    // after the comm start at field 3.
    let fields: Vec<&str> = stat[close + 1..].split_whitespace().collect();
    let ppid = fields.get(4 - 3)?.parse().ok()?;
    let starttime = fields.get(22 - 3)?.parse().ok()?;
    Some(Process {
        pid,
        comm,
        ppid,
        starttime,
    })
}

/// Read `pid`'s ancestry from `/proc`, bounded by [`MAX_DEPTH`] and guarded
/// against a cycle.
pub fn ancestry(pid: u32) -> Ancestry {
    let mut table = Ancestry::new();
    let mut cursor = pid;
    for _ in 0..MAX_DEPTH {
        let Some(process) = process(cursor) else {
            break;
        };
        let parent = process.ppid;
        table.insert(cursor, process);
        if parent <= 1 || parent == cursor || table.contains_key(&parent) {
            break;
        }
        cursor = parent;
    }
    table
}

/// The host process of `pid`, or `None` when `/proc` names none.
pub fn host_of(pid: u32) -> Option<Host> {
    resolve(&ancestry(pid), pid).map(|resolution| resolution.host)
}

/// The host process of this process.
pub fn host_identity() -> Option<Host> {
    host_of(std::process::id())
}

/// Resolve this process's agent identity the way the hub will, in precedence
/// order: the operator's `MINI_SWE_AGENT_ID`, then the host process, then
/// `fallback` for a client whose ancestry names no host.
pub fn identity(fallback: &str) -> Identity {
    let override_id = std::env::var("MINI_SWE_AGENT_ID").ok();
    identity_of(std::process::id(), override_id.as_deref(), fallback)
}

/// The identity of `pid`: `override_id` when the operator named the agent, else
/// the host process, else `fallback`.
pub fn identity_of(pid: u32, override_id: Option<&str>, fallback: &str) -> Identity {
    if let Some(agent_id) = override_id.filter(|id| !id.is_empty()) {
        return Identity {
            id: agent_id.to_string(),
            source: Source::Override,
        };
    }
    match resolve(&ancestry(pid), pid) {
        Some(resolution) => Identity {
            id: resolution.host.to_string(),
            source: Source::Host {
                host: resolution.host,
                skipped: resolution.skipped,
            },
        },
        None => Identity {
            id: fallback.to_string(),
            source: Source::Fallback,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(pid: u32, comm: &str, ppid: u32, starttime: u64) -> Process {
        Process {
            pid,
            comm: comm.to_string(),
            ppid,
            starttime,
        }
    }

    /// `mini-swe-mcp <- bash <- claude`: the walk steps over the shell and
    /// names the host, with the pid and start time that keep it unique.
    #[test]
    fn the_walk_skips_shells_and_names_the_host() {
        let mut table = Ancestry::new();
        table.insert(900, row(900, "mini-swe-mcp", 800, 10));
        table.insert(800, row(800, "bash", 700, 5));
        table.insert(700, row(700, "claude", 1, 2));
        let resolution = resolve(&table, 900).expect("claude is the host");
        assert_eq!(
            resolution.host,
            Host {
                comm: "claude".to_string(),
                pid: 700,
                starttime: 2
            }
        );
        assert_eq!(resolution.skipped, ["bash"]);
        assert_eq!(resolution.host.to_string(), "host:claude:700:2");
    }

    /// Every wrapper in the list is stepped over, however many are stacked.
    #[test]
    fn every_listed_wrapper_is_skipped() {
        for wrapper in WRAPPERS {
            let mut table = Ancestry::new();
            table.insert(20, row(20, "mini-swe-mcp", 19, 1));
            table.insert(19, row(19, wrapper, 18, 1));
            table.insert(18, row(18, wrapper, 17, 1));
            table.insert(17, row(17, "agy", 1, 1));
            let resolution = resolve(&table, 20).unwrap_or_else(|| panic!("{wrapper} skipped"));
            assert_eq!(
                resolution.host.comm, "agy",
                "{wrapper} must be stepped over"
            );
            assert_eq!(resolution.skipped, [*wrapper, *wrapper]);
        }
    }

    /// A chain of nothing but wrappers names no host, so the caller keeps its
    /// transport identity instead of inventing a shared one.
    #[test]
    fn a_chain_of_only_wrappers_names_no_host() {
        let mut table = Ancestry::new();
        table.insert(30, row(30, "mini-swe-mcp", 29, 1));
        table.insert(29, row(29, "sh", 28, 1));
        table.insert(28, row(28, "bash", 27, 1));
        table.insert(27, row(27, "env", 1, 1));
        assert_eq!(resolve(&table, 30), None);
    }

    /// pid 1 is init, not an agent host: naming it would merge every session
    /// on the machine into one identity.
    #[test]
    fn init_is_not_a_host() {
        let mut table = Ancestry::new();
        table.insert(40, row(40, "mini-swe-mcp", 39, 1));
        table.insert(39, row(39, "sh", 1, 1));
        table.insert(1, row(1, "systemd", 0, 1));
        assert_eq!(resolve(&table, 40), None);
    }

    /// An unknown pid, and a cycle, both answer `None` rather than looping.
    #[test]
    fn an_unreadable_or_cyclic_ancestry_names_no_host() {
        assert_eq!(resolve(&Ancestry::new(), 404), None);
        let mut table = Ancestry::new();
        table.insert(50, row(50, "mini-swe-mcp", 51, 1));
        table.insert(51, row(51, "bash", 50, 1));
        assert_eq!(resolve(&table, 50), None);
    }

    /// The real `/proc` of this process: the walk reads a start time, which is
    /// what makes the identity survive a pid being reused.
    #[test]
    fn this_process_has_a_host_with_a_start_time() {
        let me = super::process(std::process::id()).expect("read /proc/self/stat");
        assert!(me.starttime > 0, "field 22 is the start time: {me:?}");
        assert!(
            !WRAPPERS.contains(&me.comm.as_str()),
            "the test binary is not a wrapper: {me:?}"
        );
        let host = host_identity().expect("the test binary has a host");
        assert!(host.starttime > 0, "{host}");
        assert_eq!(host.to_string().split(':').count(), 4, "{host}");
    }

    /// `MINI_SWE_AGENT_ID` outranks the host, and a blank value does not: it is
    /// not an identity, so the host process answers instead.
    #[test]
    fn the_override_outranks_the_host() {
        let resolved = identity_of(std::process::id(), Some("orchestrator-7"), "cli");
        assert_eq!(resolved.id, "orchestrator-7");
        assert_eq!(resolved.source, Source::Override);
        assert_eq!(resolved.explain(), "MINI_SWE_AGENT_ID");

        let resolved = identity_of(std::process::id(), Some(""), "cli");
        assert_ne!(resolved.id, "", "a blank override is not an identity");
        assert!(
            matches!(resolved.source, Source::Host { .. } | Source::Fallback),
            "{resolved:?}"
        );
    }

    /// The host identity names the host process and the wrappers it stepped
    /// over, which is what `whoami` prints.
    #[test]
    fn the_host_identity_explains_itself() {
        let identity = Identity {
            id: "host:claude:700:2".to_string(),
            source: Source::Host {
                host: Host {
                    comm: "claude".to_string(),
                    pid: 700,
                    starttime: 2,
                },
                skipped: vec!["sh".to_string(), "bash".to_string()],
            },
        };
        assert_eq!(
            identity.explain(),
            "host process claude (pid 700), skipping sh, bash"
        );
        let bare = Identity {
            id: "host:claude:700:2".to_string(),
            source: Source::Host {
                host: Host {
                    comm: "claude".to_string(),
                    pid: 700,
                    starttime: 2,
                },
                skipped: Vec::new(),
            },
        };
        assert_eq!(bare.explain(), "host process claude (pid 700)");
        let fallback = Identity {
            id: "cli".to_string(),
            source: Source::Fallback,
        };
        assert_eq!(fallback.explain(), "no host process found; using 'cli'");
    }
}
