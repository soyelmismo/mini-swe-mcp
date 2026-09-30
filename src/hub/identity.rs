//! One identity per agent session: the host process that owns it, and the
//! session running inside it.
//!
//! An agent session is the HOST process — the `claude`, `opencode` or `agy`
//! process an operator started. It spawns both the MCP stdio proxy and the
//! agent's shell commands, so naming it gives those two callers one identity
//! with no configuration at all: the MCP connection that dispatched a worker
//! and the `mini-swe-mcp watch` its shell runs afterwards see the same
//! workers, while two hosts never see each other's.
//!
//! One host process is not always one session, though: opencode v2 runs a tab
//! per session inside ONE process, over ONE shared MCP connection. So when the
//! session is known it qualifies the host — `host:<comm>:<pid>:<start>/
//! session:<id>` — and two sessions of one host are two agents. The session is
//! read per MCP call from `CallToolRequest.params._meta.sessionID` (never
//! cached on the connection, which several sessions share) and from the
//! environment for the handshakes (see [`SESSION_ENV_VARS`]).
//!
//! The walk starts at the client process's *parent* and steps over the shells
//! and wrappers a host puts between itself and the binary it launches
//! (`mini-swe-mcp <- bash <- claude`), then names the first process left:
//! `host:<comm>:<pid>:<starttime>`. The start time is field 22 of
//! `/proc/<pid>/stat`, so a pid the kernel recycled never collides with the
//! agent that held it before. A service manager is never a host: a process
//! daemonized with `setsid -f` is reparented to the user's `systemd`, which
//! would otherwise give every detached process of that user one identity.
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

/// Processes that are never an agent host, however the walk reaches them.
///
/// A command started with `setsid -f` (or any other daemonizing fork) is
/// reparented to the per-user service manager, so naming it would give every
/// detached process of this user one shared identity — the same hole a single
/// shared `cli` identity would be. The walk stops here and the next identity
/// rule answers instead. The list is explicit: guessing at "looks like a
/// daemon" would either miss the per-user manager or refuse a real host.
const SERVICE_MANAGERS: &[&str] = &[
    "systemd",
    "init",
    "launchd",
    "openrc",
    "openrc-init",
    "runit",
    "runsvdir",
    "s6-svscan",
    "supervisord",
    "tini",
    "docker-init",
];

/// Environment variables that name the session inside the host process, in
/// precedence order: the first one that is set wins.
///
/// `CLAUDE_CODE_SESSION_ID` is exported to both the shell commands and the MCP
/// server of a Claude Code session; `OPENCODE_SESSION_ID` is opencode v2's
/// spelling. `MINI_SWE_SESSION_ID` is the generic one for any other host. None
/// of them reaches a plain `bash` tool call, which is what a watch token is
/// for (see [`crate::hub::WatchTokens`]).
pub const SESSION_ENV_VARS: &[&str] = &[
    "CLAUDE_CODE_SESSION_ID",
    "OPENCODE_SESSION_ID",
    "MINI_SWE_SESSION_ID",
];

/// Environment variable carrying a watch token: the caller is then exactly the
/// identity that token was minted for (see [`crate::hub::WatchTokens`]).
pub const WATCH_TOKEN_ENV: &str = "MINI_SWE_WATCH_TOKEN";

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
    /// The host process *and* the session running inside it: a host that runs
    /// several sessions is not one agent.
    Session {
        host: Host,
        skipped: Vec<String>,
        session: String,
    },
    /// A session the environment named, with no host process to attach it to.
    SessionOnly { session: String },
    /// A `MINI_SWE_WATCH_TOKEN`: the caller is exactly the identity that token
    /// was minted for, which is how a shell that cannot know its session still
    /// acts as the session that dispatched a worker.
    Token,
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
            Source::Session {
                host,
                skipped,
                session,
            } => format!(
                "host process {} (pid {}){} in session {}",
                host.comm,
                host.pid,
                if skipped.is_empty() {
                    String::new()
                } else {
                    format!(", skipping {}", skipped.join(", "))
                },
                session
            ),
            Source::SessionOnly { session } => {
                format!("session {session} (no host process to attach it to)")
            }
            Source::Token => "MINI_SWE_WATCH_TOKEN".to_string(),
            Source::Fallback => format!("no host process found; using '{}'", self.id),
        }
    }
}

/// The agent identity a hub connection presents: the host process the client
/// walked, qualified by the session running inside it.
///
/// The one place the `host:…/session:…` spelling lives, so the client that
/// computes its own identity and the daemon that combines the two fields of a
/// `hub/hello` can never drift apart. `None` when neither is known, which
/// leaves the caller's transport identity to answer.
pub fn qualify(host: Option<&str>, session: Option<&str>) -> Option<String> {
    let host = host.filter(|host| !host.is_empty());
    let session = session.filter(|session| !session.is_empty());
    match (host, session) {
        (Some(host), Some(session)) => Some(format!("{host}/session:{session}")),
        (Some(host), None) => Some(host.to_string()),
        (None, Some(session)) => Some(format!("session:{session}")),
        (None, None) => None,
    }
}

/// The session this process runs in, when its host named one: the first of
/// [`SESSION_ENV_VARS`] that is set to a non-empty value.
pub fn session_from_env() -> Option<String> {
    SESSION_ENV_VARS
        .iter()
        .find_map(|var| std::env::var(var).ok())
        .filter(|session| !session.is_empty())
}

/// Whether `comm` names a service manager rather than an agent host.
pub fn is_service_manager(comm: &str) -> bool {
    SERVICE_MANAGERS.contains(&comm)
}

/// Walk `ancestry` from `pid`'s parent and name the first ancestor that is
/// neither a shell, nor a wrapper, nor a service manager.
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
        // A service manager is not a host either: a process daemonized with
        // `setsid -f` is reparented to the user's `systemd`, so naming it
        // would give every detached process of this user one identity.
        if is_service_manager(&process.comm) {
            return None;
        }
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
/// order: the operator's `MINI_SWE_AGENT_ID`, then a `MINI_SWE_WATCH_TOKEN`,
/// then the host process plus the session the environment names, then
/// `fallback` for a client whose ancestry names no host.
pub fn identity(fallback: &str) -> Identity {
    let override_id = std::env::var("MINI_SWE_AGENT_ID").ok();
    identity_of(std::process::id(), override_id.as_deref(), fallback)
}

/// The identity of `pid`, reading the session and the watch token from this
/// process's environment.
pub fn identity_of(pid: u32, override_id: Option<&str>, fallback: &str) -> Identity {
    identity_with_session(pid, override_id, session_from_env().as_deref(), fallback)
}

/// The identity of `pid` with the session supplied by the caller.
///
/// The seam the session split is tested through: the environment is
/// process-global, so a test cannot set `CLAUDE_CODE_SESSION_ID` for one case
/// and clear it for the next.
pub fn identity_with_session(
    pid: u32,
    override_id: Option<&str>,
    session: Option<&str>,
    fallback: &str,
) -> Identity {
    if let Some(agent_id) = override_id.filter(|id| !id.is_empty()) {
        return Identity {
            id: agent_id.to_string(),
            source: Source::Override,
        };
    }
    // A watch token outranks the host: it is the only thing that ties a shell
    // that cannot know its session to the session that dispatched a worker.
    if let Some(identity) = watch_token_identity() {
        return Identity {
            id: identity,
            source: Source::Token,
        };
    }
    let session = session.filter(|session| !session.is_empty());
    match (resolve(&ancestry(pid), pid), session) {
        (Some(resolution), Some(session)) => Identity {
            id: format!("{}/session:{session}", resolution.host),
            source: Source::Session {
                host: resolution.host,
                skipped: resolution.skipped,
                session: session.to_string(),
            },
        },
        (Some(resolution), None) => Identity {
            id: resolution.host.to_string(),
            source: Source::Host {
                host: resolution.host,
                skipped: resolution.skipped,
            },
        },
        (None, Some(session)) => Identity {
            id: format!("session:{session}"),
            source: Source::SessionOnly {
                session: session.to_string(),
            },
        },
        (None, None) => Identity {
            id: fallback.to_string(),
            source: Source::Fallback,
        },
    }
}

/// The identity a `MINI_SWE_WATCH_TOKEN` in this process's environment names,
/// resolved against the hub's token store.
///
/// `None` when no token is set, or the hub does not know it: an unknown token
/// is not an identity, so the next rule answers rather than the caller being
/// locked out.
fn watch_token_identity() -> Option<String> {
    let token = std::env::var(WATCH_TOKEN_ENV)
        .ok()
        .filter(|token| !token.is_empty())?;
    super::watch_token_identity(&token)
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

    /// A chain that ends at the per-user service manager names no host: a
    /// command daemonized with `setsid -f` is reparented to `systemd`, and
    /// every detached process of this user would otherwise share that one
    /// identity.
    #[test]
    fn a_chain_that_ends_at_a_service_manager_names_no_host() {
        let mut table = Ancestry::new();
        table.insert(700, row(700, "mini-swe-mcp", 730, 40));
        table.insert(730, row(730, "systemd", 1, 12));
        assert_eq!(
            resolve(&table, 700),
            None,
            "the user's service manager is not an agent host"
        );
    }

    /// Every listed service manager stops the walk, whatever sits below it.
    #[test]
    fn every_listed_service_manager_stops_the_walk() {
        for manager in SERVICE_MANAGERS {
            let mut table = Ancestry::new();
            table.insert(60, row(60, "mini-swe-mcp", 59, 1));
            table.insert(59, row(59, manager, 1, 1));
            assert_eq!(
                resolve(&table, 60),
                None,
                "{manager} must not be named as a host"
            );
        }
    }

    /// A service manager below a shell is still not a host: the walk steps over
    /// the shell and stops at the manager instead of naming it.
    #[test]
    fn a_service_manager_below_a_shell_is_not_a_host() {
        let mut table = Ancestry::new();
        table.insert(70, row(70, "mini-swe-mcp", 69, 1));
        table.insert(69, row(69, "bash", 730, 1));
        table.insert(730, row(730, "systemd", 1, 1));
        assert_eq!(resolve(&table, 70), None);
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

    /// A session qualifies the host: two sessions of one host process are two
    /// agents, which is what opencode v2's tabs need.
    #[test]
    fn a_session_qualifies_the_host() {
        let resolved = identity_with_session(std::process::id(), None, Some("tab-7"), "cli");
        assert_eq!(
            resolved.id,
            format!("{}/session:tab-7", host_of(std::process::id()).unwrap())
        );
        assert_eq!(
            resolved.source,
            Source::Session {
                host: host_of(std::process::id()).unwrap(),
                skipped: Vec::new(),
                session: "tab-7".to_string(),
            }
        );
        assert!(
            resolved.explain().contains("session tab-7"),
            "whoami must name the session: {}",
            resolved.explain()
        );
    }

    /// A session with no host process to attach it to keeps its own identity,
    /// and a host with no session keeps today's host identity.
    #[test]
    fn a_session_without_a_host_is_still_an_identity() {
        let mut table = Ancestry::new();
        table.insert(80, row(80, "mini-swe-mcp", 79, 1));
        table.insert(79, row(79, "systemd", 1, 1));
        let resolved = identity_with_session(80, None, Some("tab-9"), "cli");
        assert_eq!(resolved.id, "session:tab-9");
        assert_eq!(
            resolved.source,
            Source::SessionOnly {
                session: "tab-9".to_string()
            }
        );

        let bare = identity_with_session(80, None, None, "cli");
        assert_eq!(bare.id, "cli");
        assert_eq!(bare.source, Source::Fallback);
    }

    /// `MINI_SWE_AGENT_ID` outranks the session, and a blank session is not
    /// one: neither splits nor invents an identity.
    #[test]
    fn the_override_outranks_the_session() {
        let resolved = identity_with_session(
            std::process::id(),
            Some("orchestrator-7"),
            Some("tab-7"),
            "cli",
        );
        assert_eq!(resolved.id, "orchestrator-7");
        assert_eq!(resolved.source, Source::Override);

        let host_only = identity_with_session(std::process::id(), None, Some(""), "cli");
        assert!(
            matches!(host_only.source, Source::Host { .. }),
            "{:?}",
            host_only.source
        );
        assert_eq!(
            host_only.id,
            host_of(std::process::id()).unwrap().to_string()
        );
    }

    /// The session variables are read in one fixed order, and the first one
    /// that is set wins.
    #[test]
    fn the_session_variables_have_one_precedence_order() {
        assert_eq!(
            SESSION_ENV_VARS,
            [
                "CLAUDE_CODE_SESSION_ID",
                "OPENCODE_SESSION_ID",
                "MINI_SWE_SESSION_ID"
            ]
        );
    }

    /// `qualify` is the one place the `host:…/session:…` spelling lives, so the
    /// client and the daemon cannot drift apart.
    #[test]
    fn qualify_spells_the_qualified_identity_once() {
        assert_eq!(
            qualify(Some("host:claude:700:2"), Some("tab-7")).as_deref(),
            Some("host:claude:700:2/session:tab-7")
        );
        assert_eq!(
            qualify(Some("host:claude:700:2"), None).as_deref(),
            Some("host:claude:700:2")
        );
        assert_eq!(
            qualify(None, Some("tab-7")).as_deref(),
            Some("session:tab-7")
        );
        assert_eq!(qualify(None, None), None);
        assert_eq!(
            qualify(Some("host:claude:700:2"), Some("")).as_deref(),
            Some("host:claude:700:2"),
            "a blank session is not a session"
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
