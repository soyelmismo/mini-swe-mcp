//! The hub daemon: one process, one
//! [`WorkerPool`](crate::pool::WorkerPool), many connections.
//!
//! Lifecycle, in order: take the exclusive `flock` on `hub.lock` (a second
//! daemon on the same directory gives up instead of stealing the socket),
//! clear a stale `hub.sock`, bind and serve. Every accepted connection is
//! checked with `SO_PEERCRED` and then handed to
//! [`McpServer::serve_connection`], so the wire behaviour is byte-identical to
//! the stdio server. The daemon exits when it has had no connection and no
//! live worker for [`HubConfig::idle_secs`], or on `SIGTERM`/`SIGINT`.

use std::collections::BTreeMap;
use std::os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt, PermissionsExt};
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::Mutex;
use tracing::{debug, error, info, warn};

/// Raise this process's soft `RLIMIT_NOFILE` to the hard limit, so ~100 worker
/// tasks each owning pipes, sockets and subprocess fds never bump into the
/// login-shell default (typically 1024).
pub fn raise_nofile_limit() {
    use std::mem::MaybeUninit;
    // SAFETY: `getrlimit`/`setrlimit` take `RLIMIT_NOFILE` and a valid rlimit
    // pointer, and nothing else in this process touches the fd limit.
    unsafe {
        let mut limits = MaybeUninit::<libc::rlimit>::uninit();
        let get = libc::getrlimit(libc::RLIMIT_NOFILE, limits.as_mut_ptr());
        if get != 0 {
            warn!(error = ?std::io::Error::last_os_error(), "Could not read RLIMIT_NOFILE");
            return;
        }
        let limits = limits.assume_init();
        if limits.rlim_cur >= limits.rlim_max {
            debug!(
                soft = limits.rlim_cur,
                hard = limits.rlim_max,
                "RLIMIT_NOFILE already raised"
            );
            return;
        }
        let raised = libc::rlimit {
            rlim_cur: limits.rlim_max,
            rlim_max: limits.rlim_max,
        };
        let old = limits.rlim_cur;
        if libc::setrlimit(libc::RLIMIT_NOFILE, &raised) == 0 {
            debug!(
                old,
                new = raised.rlim_cur,
                "Raised RLIMIT_NOFILE soft limit to the hard limit"
            );
        } else {
            warn!(
                old,
                hard = limits.rlim_max,
                error = ?std::io::Error::last_os_error(),
                "Could not raise RLIMIT_NOFILE"
            );
        }
    }
}

use crate::mcp::McpServer;

/// Default idle window before the daemon exits on its own.
const DEFAULT_IDLE_SECS: u64 = 600;

/// How often the idle watchdog re-checks the daemon's liveness.
const IDLE_POLL_INTERVAL: Duration = Duration::from_millis(500);

/// Artificial delay before startup recovery, from the environment.
///
/// It exists only so a test can hold recovery open long enough to observe a
/// client being served while it runs; production never sets it.
fn recovery_delay() -> Duration {
    crate::config::env_parse::<u64>("MINI_SWE_HUB_RECOVERY_DELAY_MS")
        .map(Duration::from_millis)
        .unwrap_or_default()
}

/// How long a starting daemon waits for a predecessor to release `hub.lock`
/// after the predecessor removed its socket.
const LOCK_WAIT: Duration = Duration::from_secs(5);

/// How often that wait re-probes the lock and the socket.
const LOCK_POLL_INTERVAL: Duration = Duration::from_millis(50);

/// The file the watch tokens are kept in, inside the hub directory.
const WATCH_TOKENS_FILE: &str = "watch-tokens.json";

/// Rows kept in the token file: a bound, so a long-lived hub cannot grow it
/// without limit.
const MAX_TOKEN_ROWS: usize = 512;

/// Size above which `hub.log` is rotated before it is opened for append.
///
/// The log only exists to answer "what did the hub do"; it is not an archive.
/// A live hub reached 6.6 MB over 51.7k lines with nothing to stop it, so the
/// daemon now keeps one generation: past this cap the current file becomes
/// `hub.log.1` and a fresh `hub.log` takes its place. 8 MiB is far more than a
/// working operator reads and small enough to keep the directory tidy.
pub const LOG_ROTATE_BYTES: u64 = 8 * 1024 * 1024;

/// How often a running daemon re-checks the log's size and rotates it.
///
/// Rotation only helps if a long-lived daemon performs it: the cap is checked
/// whenever the log is opened for append, and the appends a daemon makes on its
/// own are rare compared to the tracing a worker writes to the same file
/// continuously. Ten minutes bounds how long the file can sit past the cap.
pub const LOG_ROTATE_INTERVAL: Duration = Duration::from_secs(600);

/// The three files a hub directory holds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HubPaths {
    dir: PathBuf,
}

impl HubPaths {
    /// Paths rooted at a previously validated hub directory.
    pub fn new(dir: PathBuf) -> Self {
        Self { dir }
    }

    /// The hub directory itself.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The Unix socket clients connect to, when it lives on the filesystem.
    ///
    /// A Unix socket path must fit in `sun_path` (108 bytes on Linux). When
    /// `<hub dir>/hub.sock` is longer - a deep `SWE_HUB_DIR` or `TMPDIR` - the
    /// socket moves to a short private directory derived from the hub dir
    /// (`/tmp/mswe-<uid>-<hash>`, created 0700 and owner-checked like the hub
    /// dir). See [`HubPaths::endpoint`] for the case where no such directory
    /// can be created either.
    pub fn socket(&self) -> PathBuf {
        match self.endpoint() {
            HubEndpoint::Path(path) => path,
            HubEndpoint::Abstract(_) => self.dir.join("hub.sock"),
        }
    }

    /// The fallback directory [`HubPaths::socket`] would file its socket in,
    /// or `None` when the socket fits in the hub directory.
    ///
    /// Answering "where would the socket go" must not *make* anything: the
    /// endpoint a caller inspects is usually one it never binds, and a
    /// fallback directory created for such a probe outlives the process that
    /// made it -- an empty `/tmp/mswe-<uid>-<hash>` nobody ever owned. Only the
    /// binder creates the directory, and it removes it with the socket (see
    /// `FallbackSocketGuard`).
    pub fn fallback_dir(&self) -> Option<PathBuf> {
        let natural = self.dir.join("hub.sock");
        if natural.as_os_str().len() < MAX_SOCKET_PATH {
            return None;
        }
        Some(fallback_socket_dir(&self.dir))
    }

    /// Where the hub listens: a filesystem socket when one fits, otherwise
    /// (no short writable directory, e.g. inside a sandbox that denies `/tmp`)
    /// a Linux abstract-namespace socket named after the hub dir. Daemon and
    /// clients derive the same endpoint; access stays restricted to this user
    /// by the daemon's `SO_PEERCRED` check.
    pub fn endpoint(&self) -> HubEndpoint {
        let natural = self.dir.join("hub.sock");
        if natural.as_os_str().len() < MAX_SOCKET_PATH {
            return HubEndpoint::Path(natural);
        }
        fallback_endpoint(&self.dir)
    }

    /// The lock file serialising daemons on this directory.
    pub fn lock(&self) -> PathBuf {
        self.dir.join("hub.lock")
    }

    /// The daemon's log file.
    ///
    /// It is bounded, not archived: past [`LOG_ROTATE_BYTES`] it becomes
    /// `hub.log.1` (replacing the generation before it) and a fresh
    /// `hub.log` takes over, both when a client opens it for the daemon it
    /// spawns and on a [`LOG_ROTATE_INTERVAL`] timer while the daemon runs. At
    /// most two generations exist at any time.
    pub fn log(&self) -> PathBuf {
        self.dir.join("hub.log")
    }

    /// The watch-token store: one token per agent identity (see
    /// [`WatchTokens`]).
    pub fn watch_tokens(&self) -> PathBuf {
        self.dir.join(WATCH_TOKENS_FILE)
    }
}

/// Resolve the hub directory.
///
/// `$SWE_HUB_DIR` wins; otherwise `<swe_base_dir()>/mini-swe-hub-<uid>`. The
/// directory is created with mode 0700 when missing, and refused when it
/// exists but is not owned by this uid or is group/world accessible.
pub fn hub_dir() -> Result<PathBuf> {
    if let Some(dir) = std::env::var_os("SWE_HUB_DIR").filter(|v| !v.is_empty()) {
        return hub_dir_in(PathBuf::from(dir));
    }
    let dir = crate::worktree::swe_base_dir().join(format!("mini-swe-hub-{}", current_uid()));
    hub_dir_in(dir)
}

/// [`hub_dir`] for a directory the caller names, with the same creation and
/// refusal rules.
///
/// The binary resolves `$SWE_HUB_DIR` in [`hub_dir`]; this is the seam that
/// lets an in-process caller (a test, an embedder) check a directory it holds
/// without the process-wide variable every other test in the same process
/// would inherit.
pub fn hub_dir_in(dir: PathBuf) -> Result<PathBuf> {
    harden_hub_dir(dir)
}

/// Where a hub listens; see [`HubPaths::endpoint`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HubEndpoint {
    /// A filesystem socket.
    Path(PathBuf),
    /// A Linux abstract-namespace socket with this name.
    Abstract(String),
}

impl std::fmt::Display for HubEndpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HubEndpoint::Path(path) => write!(f, "{}", path.display()),
            HubEndpoint::Abstract(name) => write!(f, "@{name}"),
        }
    }
}

/// Connect to a hub endpoint.
///
/// The dial is followed by a credential check, because the fallback socket name
/// is predictable and `/tmp` is shared: a local attacker can squat either a
/// filesystem socket or an abstract name. Refusing the peer before the client
/// sends an identity or a task means a squat can only deny service, never
/// impersonate the hub. `HubServer::serve` performs the mirror check on the
/// accepted side, so both directions of the connection are verified.
pub async fn connect_endpoint(endpoint: &HubEndpoint) -> std::io::Result<UnixStream> {
    let stream = match endpoint {
        HubEndpoint::Path(path) => UnixStream::connect(path).await?,
        HubEndpoint::Abstract(name) => {
            use std::os::linux::net::SocketAddrExt;
            let addr = std::os::unix::net::SocketAddr::from_abstract_name(name.as_bytes())?;
            let stream = std::os::unix::net::UnixStream::connect_addr(&addr)?;
            stream.set_nonblocking(true)?;
            UnixStream::from_std(stream)?
        }
    };
    ensure_peer_is_self(stream.peer_cred()?.uid())?;
    Ok(stream)
}

/// Refuse a hub connection whose peer is not this user.
///
/// `SO_PEERCRED` is what makes a shared fallback name safe: the daemon already
/// drops foreign peers, and this keeps the client from ever speaking to a
/// daemon another user planted on `/tmp/mswe-<uid>-<hash>` or on an abstract
/// name.
fn ensure_peer_is_self(uid: u32) -> std::io::Result<()> {
    if uid == current_uid() {
        return Ok(());
    }
    let me = current_uid();
    Err(std::io::Error::new(
        std::io::ErrorKind::PermissionDenied,
        format!("hub peer is uid {uid}, but this process is uid {me}"),
    ))
}

/// Removes a hub socket that lives outside its hub directory, taking the short
/// fallback directory that holds it with it.
///
/// The daemon drops this guard on every exit path, so a caller that aborts the
/// `run` future (rather than letting it shut down) still leaves no socket or
/// fallback directory behind.
struct FallbackSocketGuard {
    socket: PathBuf,
    fallback_dir: Option<PathBuf>,
}

impl FallbackSocketGuard {
    fn new(socket: &Path, hub_dir: &Path) -> Self {
        let fallback_dir = socket
            .parent()
            .filter(|parent| *parent != hub_dir)
            .map(Path::to_path_buf);
        Self {
            socket: socket.to_path_buf(),
            fallback_dir,
        }
    }
}

impl Drop for FallbackSocketGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.socket);
        if let Some(dir) = &self.fallback_dir {
            let _ = std::fs::remove_dir(dir);
        }
    }
}

/// Bind a hub endpoint (a filesystem socket is restricted to 0600).
fn bind_endpoint(endpoint: &HubEndpoint) -> Result<UnixListener> {
    match endpoint {
        HubEndpoint::Path(path) => {
            let listener = UnixListener::bind(path)
                .with_context(|| format!("Could not bind hub socket {}", path.display()))?;
            std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
                .with_context(|| format!("Could not restrict {} to 0600", path.display()))?;
            Ok(listener)
        }
        HubEndpoint::Abstract(name) => {
            use std::os::linux::net::SocketAddrExt;
            let addr = std::os::unix::net::SocketAddr::from_abstract_name(name.as_bytes())?;
            let listener = std::os::unix::net::UnixListener::bind_addr(&addr)
                .with_context(|| format!("Could not bind abstract hub socket @{name}"))?;
            listener.set_nonblocking(true)?;
            Ok(UnixListener::from_std(listener)?)
        }
    }
}

/// The socket's file name, inside the hub directory or the fallback one.
const SOCKET_NAME: &str = "hub.sock";

/// Longest socket path used as-is, below Linux's 108-byte `sun_path`.
const MAX_SOCKET_PATH: usize = 100;

/// The name of the short fallback directory a too-deep hub directory's socket
/// moves into: `mswe-<uid>-<hash>`, private to this user and derived from the
/// hub directory so daemon and clients pick the same one.
fn fallback_socket_key(dir: &Path) -> String {
    format!(
        "mswe-{}-{:016x}",
        current_uid(),
        fnv1a(dir.as_os_str().as_encoded_bytes())
    )
}

/// Whether a fallback socket directory may be used, or can be created here.
///
/// The fallback name is predictable (`mswe-<uid>-<hash>`) and `/tmp` is shared,
/// so an existing directory is trusted only when it is a real directory --
/// never a symlink -- owned by this user and private to it. Anything else (a
/// foreign or group/world-accessible directory, a symlink) is refused, and the
/// hub listens in the abstract namespace instead: a directory another user
/// planted must never be the socket a client dials into. A directory that is
/// absent is probed: the creation the daemon would perform is attempted and
/// undone, so a sandbox that denies `/tmp` falls back to the abstract socket
/// while an ordinary probe still leaves nothing behind. The daemon re-runs
/// [`harden_hub_dir`] before it binds, so a directory planted between this
/// check and the bind is refused rather than used.
fn fallback_dir_is_creatable(dir: &Path) -> bool {
    match std::fs::symlink_metadata(dir) {
        // Present: accept only a real directory, ours, and private to us.
        Ok(meta) => {
            meta.is_dir()
                && !meta.file_type().is_symlink()
                && meta.uid() == current_uid()
                && (meta.permissions().mode() & 0o077) == 0
        }
        // Absent: try the creation this hub would perform and undo it, so the
        // answer is the real one for a shared `/tmp` without leaking a
        // directory. Mode 0o700 keeps a racing `harden_hub_dir` happy.
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            match std::fs::DirBuilder::new().mode(0o700).create(dir) {
                Ok(()) => {
                    let _ = std::fs::remove_dir(dir);
                    true
                }
                Err(_) => false,
            }
        }
        Err(_) => false,
    }
}

/// Where a too-deep hub directory listens: the short fallback socket, or the
/// abstract socket a sandbox that denies `/tmp` forces.
///
/// This is the single decision point, so the daemon that binds and the clients
/// that dial always agree -- and it leaves nothing behind: an existing
/// fallback directory is only trusted when it is this user's and private, an
/// absent one is probed and undone by [`fallback_dir_is_creatable`], and the
/// directory that is finally used is created by `harden_hub_dir` and removed
/// again by `FallbackSocketGuard`, so resolving an endpoint leaves no empty
/// `/tmp/mswe-<uid>-<hash>` behind.
fn fallback_endpoint(dir: &Path) -> HubEndpoint {
    let fallback = fallback_socket_dir(dir);
    if fallback_dir_is_creatable(&fallback) {
        HubEndpoint::Path(fallback.join("hub.sock"))
    } else {
        HubEndpoint::Abstract(fallback_socket_key(dir))
    }
}

/// The short fallback directory for `dir`, whether or not it exists yet.
///
/// Purely a path computation: callers that only inspect an endpoint must not
/// create anything (see [`HubPaths::fallback_dir`]), while the daemon and the
/// clients that bind there create it through [`harden_hub_dir`].
fn fallback_socket_dir(dir: &Path) -> PathBuf {
    PathBuf::from("/tmp").join(fallback_socket_key(dir))
}

/// FNV-1a: a stable short name for a hub dir's fallback socket directory.
fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// This process's uid.
fn current_uid() -> u32 {
    // SAFETY: `getuid` takes no arguments and cannot fail.
    unsafe { libc::getuid() }
}

/// Ensure `dir` exists, is owned by us, and is private to us.
fn harden_hub_dir(dir: PathBuf) -> Result<PathBuf> {
    let meta = match std::fs::symlink_metadata(&dir) {
        Ok(meta) => meta,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            std::fs::create_dir_all(&dir)
                .with_context(|| format!("Could not create hub directory {}", dir.display()))?;
            std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700))
                .with_context(|| format!("Could not restrict {} to 0700", dir.display()))?;
            return Ok(dir);
        }
        Err(e) => {
            return Err(e)
                .with_context(|| format!("Could not inspect hub directory {}", dir.display()));
        }
    };

    if !meta.is_dir() {
        anyhow::bail!("Hub path {} is not a directory", dir.display());
    }
    if meta.uid() != current_uid() {
        anyhow::bail!(
            "Hub directory {} is not owned by this user; refusing to use it",
            dir.display()
        );
    }
    let mode = meta.permissions().mode() & 0o777;
    if mode & 0o077 != 0 {
        anyhow::bail!(
            "Hub directory {} is group/world accessible (mode {mode:o}); refusing to use it",
            dir.display()
        );
    }
    Ok(dir)
}

/// Everything the daemon needs that is not the shared server itself.
#[derive(Debug, Clone)]
pub struct HubConfig {
    paths: HubPaths,
    idle_secs: u64,
    respawn: bool,
}

impl HubConfig {
    /// Configuration for an explicit hub directory and idle window.
    pub fn new(paths: HubPaths, idle_secs: u64) -> Self {
        Self {
            paths,
            idle_secs,
            respawn: false,
        }
    }

    /// The resolved hub files.
    pub fn paths(&self) -> &HubPaths {
        &self.paths
    }

    /// Idle window after which the daemon exits by itself.
    pub fn idle_secs(&self) -> u64 {
        self.idle_secs
    }
}

/// The running daemon: one shared [`McpServer`] behind a Unix socket.
#[derive(Clone)]
pub struct HubServer {
    server: Arc<McpServer>,
    config: HubConfig,
    next_conn_id: Arc<AtomicU64>,
    open_conns: Arc<Mutex<u64>>,
}

impl HubServer {
    /// Whether the daemon continues interrupted workers at startup.
    ///
    /// `HUB_AUTO_RESUME=0` opts out: every interrupted worker then stays
    /// interrupted, and only an explicit `steer <id> "..."` moves it.
    fn auto_resume_enabled() -> bool {
        std::env::var("HUB_AUTO_RESUME")
            .map(|v| v != "0")
            .unwrap_or(true)
    }

    /// A daemon serving `server` on the socket described by `config`.
    pub fn new(server: Arc<McpServer>, config: HubConfig) -> Self {
        Self {
            server,
            config,
            next_conn_id: Arc::new(AtomicU64::new(1)),
            open_conns: Arc::new(Mutex::new(0)),
        }
    }

    /// Run until idle shutdown or a termination signal.
    ///
    /// Returns `Ok(false)` when another daemon already holds the lock, so the
    /// caller can report it and exit 0 without disturbing the live daemon.
    /// Continue every `interrupted` worker that has a history, at most
    /// [`MAX_AUTO_CONTINUES`] times per worker.
    ///
    /// Returns how many workers were continued. `HUB_AUTO_RESUME=0` disables
    /// it, leaving every interrupted worker for the orchestrator to steer.
    async fn auto_resume_interrupted(&self) -> usize {
        if !Self::auto_resume_enabled() {
            info!("HUB_AUTO_RESUME=0; interrupted workers stay interrupted");
            return 0;
        }
        let pool = self.server.pool();
        let candidates = pool.interrupted_workers().await;
        let mut resumed = 0;
        for id in candidates {
            // A predecessor that outlived the bounded shutdown wait is still
            // tearing this worker's worktree down in its own process. Wait for
            // its `.teardown` marker to clear (bounded) before recreating the
            // checkout, or the two remove/create calls race exactly as H18.
            if !self.teardown_settled(&id).await {
                continue;
            }
            let budget = pool.auto_continue_budget(&id).await;
            if budget == 0 {
                info!(
                    worker = %id,
                    "Interrupted worker has spent its automatic continuations; leaving it for the orchestrator"
                );
                continue;
            }
            let message = "the hub restarted".to_string();
            // No explicit budget: the continuation resumes the run's own
            // ceiling and step counter, so the handover costs it nothing but
            // the turns it had already spent.
            match pool.continue_worker(&id, message, None).await {
                Ok(_) => {
                    // Count it before anything else can, so a worker the hub
                    // keeps losing stops being restarted after the cap.
                    pool.count_auto_continue(&id).await;
                    resumed += 1;
                    info!(worker = %id, "Auto-continued interrupted worker");
                }
                Err(e) => {
                    warn!(worker = %id, error = %e, "Could not auto-continue an interrupted worker");
                }
            }
        }
        resumed
    }

    /// Wait, bounded, for a worker's predecessor teardown to finish.
    ///
    /// Returns `true` when the worker's worktree is safe to recreate: no
    /// `.teardown` marker, or one whose owning process is gone. Returns `false`
    /// when the marker still names a live process at the deadline, so the
    /// worker stays interrupted for the orchestrator instead of racing a
    /// teardown that would delete the checkout out from under the recreation.
    async fn teardown_settled(&self, id: &str) -> bool {
        let path = self
            .server
            .pool()
            .scratch_root()
            .join(format!("swe-wt-{id}"));
        if !crate::worktree::WorktreeGuard::teardown_pending(&path) {
            return true;
        }
        // One clear line: the predecessor is provably slow, and the operator
        // needs to know why this worker is not being continued yet.
        info!(worker = %id, "Predecessor is still tearing this worker's worktree down; waiting for it");
        let deadline = tokio::time::Instant::now() + self.server.pool().teardown_wait();
        loop {
            tokio::time::sleep(Duration::from_millis(50)).await;
            if !crate::worktree::WorktreeGuard::teardown_pending(&path) {
                return true;
            }
            if tokio::time::Instant::now() >= deadline {
                warn!(
                    worker = %id,
                    wait_secs = self.server.pool().teardown_wait().as_secs(),
                    "Predecessor teardown did not finish in time; leaving the worker interrupted",
                );
                return false;
            }
        }
    }

    /// Recover the previous hub's workers and auto-continue them, then open
    /// the recovery gate.
    ///
    /// Runs concurrently with the accept loop: the gate stays closed until the
    /// pool reflects the recovered registry, so a worker-state request that
    /// arrives first waits instead of reading a half-recovered pool.
    async fn recover_and_resume(&self) {
        let delay = recovery_delay();
        if !delay.is_zero() {
            info!(?delay, "Delaying startup recovery (test hook)");
            tokio::time::sleep(delay).await;
        }
        // Recover the workers this pool owns: the pool's scratch root is where
        // its registry rows and worktrees live, so recovery must look there
        // (in production it is the same root `ScratchRoot::from_env` resolves,
        // but a test pool over a scratch root must not read the real one).
        let recovery_root = self.server.pool().scratch_root().clone();
        match tokio::task::spawn_blocking(move || {
            crate::pool::recover_orphaned_workers_in(&recovery_root)
        })
        .await
        {
            Ok(recovered) => {
                info!(workers = recovered, "Recovered orphaned hub workers");
                append_log(
                    &self.config.paths().log(),
                    &format!("recovered {recovered} orphaned workers"),
                );
            }
            Err(e) => error!(error = %e, "Hub recovery task failed"),
        }

        // Every already-integrated worker leaves nothing behind: its branch is
        // in the base branch, so branch, row, history, mailbox, steer-source and
        // watch acknowledgements go now. Runs before the resumed workers are
        // listed, so a worker that is still awaiting integration stays visible
        // while an integrated one is gone.
        let root = self.server.pool().scratch_root().clone();
        let ack_dir = self.config.paths().dir().to_path_buf();
        match tokio::task::spawn_blocking(move || {
            crate::pool::sweep_retired_workers_in(&root, Some(&ack_dir))
        })
        .await
        {
            Ok(sweep)
                if !sweep.workers.is_empty()
                    || !sweep.orphan_workers.is_empty()
                    || sweep.orphans > 0 =>
            {
                // The file edit above is not enough on its own: the event
                // router may already hold the store in memory, and its next
                // `persist` would rewrite the very entries just removed. Forget
                // them through the router, on its own lock, so memory and file
                // agree whichever was loaded first.
                for id in sweep.workers.iter().chain(sweep.orphan_workers.iter()) {
                    self.server.forget_retired_worker(id).await;
                }
                info!(
                    workers = sweep.workers.len(),
                    orphans = sweep.orphans,
                    "Retired integrated workers and orphan leftovers"
                );
                append_log(
                    &self.config.paths().log(),
                    &format!(
                        "retired {} integrated worker(s) and {} orphan file(s)",
                        sweep.workers.len(),
                        sweep.orphans
                    ),
                );
            }
            Ok(_) => {}
            Err(e) => error!(error = %e, "Retirement sweep failed"),
        }

        // Every interrupted worker with a surviving conversation is continued
        // automatically: it stopped because the hub did, not because it could
        // not go on. Capped per worker so a worker the hub keeps losing is left
        // to the orchestrator instead of being restarted forever.
        let resumed = self.auto_resume_interrupted().await;
        if resumed > 0 {
            info!(workers = resumed, "Auto-continued interrupted workers");
            append_log(
                &self.config.paths().log(),
                &format!("auto-continued {resumed} workers"),
            );
        }
        self.server.finish_recovery();
    }

    /// Wait out a predecessor that removed `hub.sock` but still holds
    /// `hub.lock`.
    ///
    /// Teardown removes the socket before it drops the lock, so a starter that
    /// arrives in that window sees a busy lock and no listener. Retry for a
    /// bounded time; `Ok(None)` means a hub is genuinely running or a
    /// predecessor outlasted [`LOCK_WAIT`].
    async fn wait_for_lock(&self) -> Result<Option<HubLock>> {
        let path = self.config.paths().lock();
        let deadline = tokio::time::Instant::now() + LOCK_WAIT;
        let mut waited = false;
        let mut warned_slow = false;
        let slow_deadline = tokio::time::Instant::now() + Duration::from_secs(1);
        loop {
            if let Some(lock) = acquire_lock(&path)? {
                if waited {
                    info!("Predecessor released hub.lock; continuing startup");
                }
                return Ok(Some(lock));
            }
            if !waited {
                info!("hub.lock held by a predecessor; waiting for it to finish teardown");
                waited = true;
            } else if tokio::time::Instant::now() + LOCK_POLL_INTERVAL >= deadline {
                warn!(
                    "Predecessor did not release hub.lock within {}s; not starting",
                    LOCK_WAIT.as_secs()
                );
            } else if !warned_slow && tokio::time::Instant::now() >= slow_deadline {
                // One clear line once the predecessor is provably slow.
                warn!("Predecessor still holding hub.lock; waiting for its teardown to finish");
                warned_slow = true;
            }
            // A predecessor owns the socket here, so this daemon never binds.
            // `endpoint()` resolves it without creating the fallback directory
            // a hub that will not run would otherwise leave behind.
            if connect_endpoint(&self.config.paths().endpoint())
                .await
                .is_ok()
            {
                return Ok(None);
            }
            if tokio::time::Instant::now() >= deadline {
                return Ok(None);
            }
            tokio::time::sleep(LOCK_POLL_INTERVAL).await;
        }
    }

    pub async fn run(&self) -> Result<bool> {
        raise_nofile_limit();
        let paths = self.config.paths();
        let lock = match acquire_lock(&paths.lock())? {
            Some(lock) => lock,
            None => match self.wait_for_lock().await? {
                Some(lock) => lock,
                None => {
                    info!("hub already running");
                    return Ok(false);
                }
            },
        };

        let mut socket = paths.socket();
        // A socket left behind by a killed daemon would make `bind` fail with
        // "address already in use"; the lock proves nobody owns it now.
        let _ = std::fs::remove_file(&socket);
        // A socket the hub directory is too deep to hold is filed in a short
        // private directory of its own. The daemon creates it here -- the one
        // place that is about to bind that socket -- and then rebuilds the guard
        // over *that* directory, so it removes the socket and the directory
        // together. Creating it after the socket and guard were resolved would
        // leave both behind on a host that could create the directory all along.
        let endpoint = paths.endpoint();
        if let HubEndpoint::Path(path) = &endpoint
            && let Some(fallback) = path.parent().filter(|parent| *parent != paths.dir())
        {
            harden_hub_dir(fallback.to_path_buf()).with_context(|| {
                format!(
                    "Could not create the short hub socket directory {}",
                    fallback.display()
                )
            })?;
            let _ = std::fs::remove_file(path);
            socket = fallback.join(SOCKET_NAME);
        }
        let _socket_cleanup = FallbackSocketGuard::new(&socket, paths.dir());
        let listener = bind_endpoint(&endpoint)?;

        info!(socket = %endpoint, idle_secs = self.config.idle_secs(), "Hub daemon listening");
        append_log(&paths.log(), "listening");

        // Recovery - the git salvage of every orphaned worktree - can take
        // seconds with many workers, so it runs concurrently with serving
        // instead of before the socket exists. The gate keeps worker-state
        // requests correct meanwhile: they wait until the pool is recovered.
        self.server.begin_recovery();
        let recovering = self.clone();
        let recovery_task = tokio::spawn(async move { recovering.recover_and_resume().await });

        let events = self
            .server
            .start_hub_events(Some(self.config.paths().dir()))
            .await;
        let auto_consolidate = self
            .server
            .start_auto_consolidate(paths.dir().to_path_buf())
            .await?;
        let mut shutdown = self.server.subscribe_shutdown();

        let reaper = crate::pool::spawn_reaper(
            (*self.server.pool()).clone(),
            Some(paths.dir().to_path_buf()),
        );
        let idle_watcher = self.clone();
        let mut idle_task = tokio::spawn(async move { idle_watcher.watch_idle().await });
        let log_watcher = self.clone();
        let log_task = tokio::spawn(async move { log_watcher.watch_log_rotation().await });

        // OTA-style handover (H17): the daemon notices its own rebuilt
        // executable and arms the planned handover itself, instead of waiting
        // for a newer client to ask. Only a daemon that can respawn itself
        // watches: a handover leaves the hub unserved until the replacement
        // binds, so a daemon with no way to start one would strand its workers.
        let auto_handover = self.clone();
        let auto_handover_task = tokio::spawn(async move {
            auto_handover.watch_executable().await;
        });

        let mut term =
            match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
                Ok(sig) => Some(sig),
                Err(e) => {
                    warn!(error = %e, "Could not install SIGTERM handler");
                    None
                }
            };

        loop {
            tokio::select! {
                accepted = listener.accept() => match accepted {
                    Ok((stream, _)) => self.serve(stream),
                    Err(e) => error!(error = %e, "Failed accepting a hub connection"),
                },
                _ = tokio::signal::ctrl_c() => {
                    info!("Received SIGINT, shutting down hub daemon");
                    break;
                }
                _ = async {
                    match term.as_mut() {
                        Some(sig) => {
                            sig.recv().await;
                        }
                        None => std::future::pending::<()>().await,
                    }
                } => {
                    info!("Received SIGTERM, shutting down hub daemon");
                    break;
                }
                _ = shutdown.changed() => {
                    info!("Hub shutdown requested by client");
                    break;
                }
                _ = &mut idle_task => {
                    info!(idle_secs = self.config.idle_secs(), "Hub daemon idle, shutting down");
                    break;
                }
            }
        }

        idle_task.abort();
        log_task.abort();
        auto_handover_task.abort();
        recovery_task.abort();
        reaper.abort();
        events.abort();
        auto_consolidate.abort();
        // Stop accepting before teardown: a starter that arrives while the
        // old daemon is still cleaning up must wait on `hub.lock`, not
        // mistake the still-bound socket for a live hub and give up.
        let _ = std::fs::remove_file(&socket);
        // A socket moved to a short fallback directory takes it along.
        if let Some(parent) = socket.parent()
            && parent != paths.dir()
        {
            let _ = std::fs::remove_dir(parent);
        }
        drop(listener);
        drop(_socket_cleanup);
        let killed = self.server.pool().kill_all().await;
        if killed > 0 {
            info!(workers = killed, "Terminated workers on hub shutdown");
        }
        append_log(&paths.log(), "stopped");
        drop(lock);
        // A handover leaves the hub unserved until a client dials it again, and
        // the workers it just interrupted are auto-continued by whichever
        // daemon binds the socket next, so the daemon that stepped aside starts
        // that daemon itself rather than waiting for one.
        if self.config.respawn && self.server.handover_requested() {
            respawn_daemon(paths);
        }
        debug!("Hub daemon stopped");
        Ok(true)
    }

    /// Arm the planned handover when this daemon's own executable is
    /// replaced by a newer build.
    ///
    /// `HUB_AUTO_HANDOVER=0` opts out, and a daemon with no way to respawn
    /// itself never watches: a handover leaves the hub unserved until the
    /// replacement binds, so stepping aside without one would strand the
    /// workers.
    async fn watch_executable(&self) {
        if !self.config.respawn || !super::auto_handover::enabled() {
            return;
        }
        let Some(path) = super::exe_path::current_exe_path() else {
            return;
        };
        let running = self.server.build_identity();
        let deadline = super::client::handover_deadline(None);
        super::auto_handover::watch(
            path,
            running,
            self.server.clone(),
            deadline,
            self.config.paths().log().to_path_buf(),
        )
        .await;
    }

    /// Keep the log bounded while the daemon runs.
    ///
    /// The cap is checked whenever anything opens the log for append, but a
    /// quiet hub appends little and a worker's tracing writes to the same file
    /// through the daemon's inherited stderr, so only a timer catches a log
    /// that outgrew the cap between two hub events.
    async fn watch_log_rotation(&self) {
        let path = self.config.paths().log();
        loop {
            tokio::time::sleep(LOG_ROTATE_INTERVAL).await;
            let path = path.clone();
            match tokio::task::spawn_blocking(move || rotate_log(&path)).await {
                Ok(Ok(_)) => {}
                Ok(Err(e)) => debug!(error = %e, "Hub log rotation failed; trying again later"),
                Err(e) => warn!(error = %e, "Hub log rotation task failed; trying again later"),
            }
        }
    }

    /// Resolve when the daemon has had no open connection and no live
    /// worker for the whole idle window.
    ///
    /// `last_active` is refreshed on every tick that sees work, so one idle
    /// tick can never end the daemon: the window must elapse uninterrupted.
    async fn watch_idle(&self) {
        let mut last_active = std::time::Instant::now();
        loop {
            tokio::time::sleep(IDLE_POLL_INTERVAL).await;
            let busy = *self.open_conns.lock().await > 0
                || self.server.pool().active_worker_count().await > 0;
            if busy {
                last_active = std::time::Instant::now();
            } else if last_active.elapsed() >= Duration::from_secs(self.config.idle_secs()) {
                return;
            }
        }
    }

    /// Verify the peer and hand the connection to the shared server.
    fn serve(&self, stream: UnixStream) {
        // `SO_PEERCRED`: only this user's processes may drive the pool.
        let uid = match stream.peer_cred() {
            Ok(cred) => cred.uid(),
            Err(e) => {
                warn!(error = %e, "Dropping hub connection with no peer credentials");
                return;
            }
        };
        if uid != current_uid() {
            warn!(peer_uid = uid, "Dropping hub connection from another user");
            return;
        }

        let id = self.next_conn_id.fetch_add(1, Ordering::Relaxed);
        let server = self.server.clone();
        let open_conns = self.open_conns.clone();
        // Every connection mints from this daemon's own token store, so the
        // tokens a dispatch hands out are the ones this daemon resolves and
        // they survive its restart.
        let tokens = Arc::new(WatchTokens::new(self.config.paths().dir().to_path_buf()));
        tokio::spawn(async move {
            *open_conns.lock().await += 1;
            let (reader, writer) = stream.into_split();
            let mut shutdown = server.subscribe_shutdown();
            let res = tokio::select! {
                _ = shutdown.changed() => Ok(()),
                res = server.serve_connection(
                    tokio::io::BufReader::new(reader),
                    writer,
                    crate::mcp::ConnectionContext::hub_connection(id).with_watch_tokens(tokens),
                ) => res,
            };
            *open_conns.lock().await -= 1;
            if let Err(e) = res {
                debug!(connection = id, error = %e, "Hub connection ended");
            }
        });
    }
}

/// Start a replacement daemon from this executable, detached exactly the way a
/// client auto-starts one (see [`crate::hub::client::connect_or_spawn`]).
///
/// The path comes from [`crate::hub::exe_path`], the same helper the client
/// spawns through: `cargo` replaces the binary in place, so this daemon's
/// `/proc/self/exe` reads `<path> (deleted)` while `path` itself already holds
/// the newer build, and a build caught mid-write leaves no file at all for a
/// moment. Respawning the path is what makes the replacement the new binary
/// instead of another copy of this one.
fn respawn_daemon(paths: &HubPaths) {
    let exe = match super::exe_path::executable() {
        Ok(exe) => exe,
        Err(error) => {
            warn!(%error, "No executable to hand over to; the next client will start the hub");
            return;
        }
    };
    if let Err(e) = super::client::spawn_daemon(paths, &exe) {
        warn!(error = %e, "Could not start replacement daemon");
    }
}

/// Rotate `hub.log` once it is larger than [`LOG_ROTATE_BYTES`].
///
/// Called before a caller opens the log for append, and on a timer while the
/// daemon runs. A log at or below the cap is left alone and `Ok(false)` says
/// so; past it, the current file is renamed to `hub.log.1` -- replacing an older
/// generation, so at most two files exist -- and `Ok(true)` says a rotation
/// happened. The caller then opens `path`, which is now a new empty file.
///
/// The rename replaces the previous generation atomically, so a reader holding
/// the old inode keeps reading it while the directory already names the newer
/// one. Failures are the caller's to report: a hub that cannot rotate still
/// logs, because losing the log is worse than keeping a big one.
pub(crate) fn rotate_log(path: &Path) -> std::io::Result<bool> {
    let size = match std::fs::metadata(path) {
        Ok(meta) => meta.len(),
        // No log yet, or an unreadable one: nothing to rotate. The caller
        // still opens it, which is what creates it.
        Err(_) => return Ok(false),
    };
    if size <= LOG_ROTATE_BYTES {
        return Ok(false);
    }
    let previous = rotated_log_path(path);
    // Replace any earlier generation in one step. A leftover `hub.log.1` from a
    // crash between the two calls is replaced here rather than accumulating.
    match std::fs::rename(path, &previous) {
        Ok(()) => {
            info!(
                path = %path.display(),
                bytes = size,
                kept = %previous.display(),
                "Rotated an oversized hub log"
            );
            // Best effort: a caller whose stderr is that file (the daemon a
            // client spawned) has to follow the rotation, or every line after
            // it would keep filling the generation that was just retired.
            if let Err(e) = repoint_stderr_at(path, &previous) {
                debug!(error = %e, "Stderr was not the rotated hub log; leaving it alone");
            }
            Ok(true)
        }
        Err(e) => {
            warn!(
                error = %e,
                path = %path.display(),
                "Could not rotate an oversized hub log; keeping the current one"
            );
            Err(e)
        }
    }
}

/// Where a rotation moves the log it replaces: the same directory, the same
/// stem, one `.1` generation.
pub fn rotated_log_path(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_os_string();
    name.push(".1");
    PathBuf::from(name)
}

/// Whether an open descriptor and a `stat` of a path name the same file.
///
/// Device *and* inode together, never the inode alone: inode numbers are only
/// unique within one filesystem, so a stderr that lives on another mount can
/// collide with the rotated log's inode. Matching on the inode alone would
/// then redirect a stderr the caller never pointed at the log -- and silently,
/// because the descriptor is the process's own output.
fn is_same_file(fd_dev: u64, fd_ino: u64, path_dev: u64, path_ino: u64) -> bool {
    fd_dev == path_dev && fd_ino == path_ino
}

/// Point this process's stderr at `fresh` when it is the file `rotated` names.
///
/// The daemon's tracing goes to stderr, which the client handed it as an open
/// handle on `hub.log` (see [`crate::hub::client::spawn_daemon`]). A rename
/// does not move that handle, so without this the log would keep growing under
/// the name `hub.log.1` and a fresh `hub.log` would stay empty -- the cap
/// would then be enforced by nobody.
///
/// Only the daemon's own stderr is redirected: a caller that logs to the
/// terminal (an in-process daemon in a test, an embedder) leaves stderr alone,
/// which the inode comparison below decides rather than an assumption. `fresh`
/// is opened with the same flags and mode the client opens the log with.
fn repoint_stderr_at(fresh: &Path, rotated: &Path) -> std::io::Result<()> {
    // SAFETY: `fstat` fills a `stat` for a descriptor this process owns.
    let mut stderr_stat: libc::stat = unsafe { std::mem::zeroed() };
    if unsafe { libc::fstat(libc::STDERR_FILENO, &mut stderr_stat) } != 0 {
        return Err(std::io::Error::last_os_error());
    }
    let rotated_meta = std::fs::metadata(rotated)?;
    if !is_same_file(
        stderr_stat.st_dev as u64,
        stderr_stat.st_ino as u64,
        rotated_meta.dev(),
        rotated_meta.ino(),
    ) {
        return Ok(());
    }
    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW)
        .open(fresh)?;
    // SAFETY: `dup2` atomically replaces fd 2 with `file`'s descriptor. Neither
    // is a Rust-owned handle at this point (fd 2 is inherited), and the file
    // outlives the call.
    if unsafe { libc::dup2(file.as_raw_fd(), libc::STDERR_FILENO) } == -1 {
        return Err(std::io::Error::last_os_error());
    }
    // `dup2` clears FD_CLOEXEC on the new descriptor; the hub spawns workers,
    // which must not inherit its log.
    // SAFETY: `fcntl` sets a descriptor flag on fd 2, this process's stderr.
    unsafe { libc::fcntl(libc::STDERR_FILENO, libc::F_SETFD, libc::FD_CLOEXEC) };
    Ok(())
}

/// Append one timestamped line to the hub log; failures are traced, never fatal.
pub(crate) fn append_log(path: &Path, event: &str) {
    use std::fmt::Write as _;
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let mut line = String::new();
    let _ = writeln!(line, "{now} pid={} {event}", std::process::id());
    use std::io::Write as _;
    // The open below is this function's only handle on the log, so it is where
    // the cap is enforced: past it the append lands in a fresh file.
    let _ = rotate_log(path);
    if let Err(e) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW)
        .open(path)
        .and_then(|mut f| f.write_all(line.as_bytes()))
    {
        warn!(error = %e, path = %path.display(), "Could not append to hub log");
    }
}

/// Take the exclusive, non-blocking lock on `path`.
///
/// `Ok(None)` means another daemon already holds it. The lock is advisory and
/// released when the returned guard (or the process) goes away.
fn acquire_lock(path: &Path) -> Result<Option<HubLock>> {
    let file = std::fs::OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(path)
        .with_context(|| format!("Could not open hub lock {}", path.display()))?;
    // SAFETY: `flock` takes the fd and an operation flag; no pointers.
    let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    if rc == 0 {
        return Ok(Some(HubLock { file }));
    }
    let err = std::io::Error::last_os_error();
    if err.kind() == std::io::ErrorKind::WouldBlock {
        return Ok(None);
    }
    Err(err).with_context(|| format!("Could not lock {}", path.display()))
}

/// Whether some process currently holds the exclusive lock on `path`.
///
/// The probe takes the same non-blocking `flock` as [`acquire_lock`] and
/// releases it at once, so a client can observe the holder without keeping the
/// hub locked.
pub fn hub_lock_held(path: &Path) -> Result<bool> {
    match acquire_lock(path)? {
        Some(lock) => {
            drop(lock);
            Ok(false)
        }
        None => Ok(true),
    }
}

/// The held `flock`, released on drop.
struct HubLock {
    file: std::fs::File,
}

impl Drop for HubLock {
    fn drop(&mut self) {
        // SAFETY: unlocking the same fd this guard locked.
        unsafe {
            libc::flock(self.file.as_raw_fd(), libc::LOCK_UN);
        }
    }
}

/// Build the daemon for `dir` and run it to completion.
///
/// The caller supplies the shared [`McpServer`] so the daemon never builds a
/// second pool; `idle_secs` overrides `HUB_IDLE_SECS`.
pub async fn run_daemon(
    server: Arc<McpServer>,
    dir: PathBuf,
    idle_secs: Option<u64>,
) -> Result<bool> {
    let paths = HubPaths { dir };
    let idle = idle_secs
        .or_else(|| crate::config::env_parse("HUB_IDLE_SECS"))
        .unwrap_or(DEFAULT_IDLE_SECS);
    let mut config = HubConfig::new(paths, idle);
    config.respawn = true;
    HubServer::new(server, config).run().await
}

/// Watch tokens: one unguessable token per agent identity, kept in the hub
/// directory.
///
/// A shell cannot know its session: the agent's `bash` tool sees none of
/// [`SESSION_ENV_VARS`](crate::hub::identity::SESSION_ENV_VARS), so a
/// `mini-swe-mcp watch` it runs would fall back to the bare host identity and
/// miss the workers its own session dispatched. Every dispatch and steer answer
/// therefore carries a `watch_command` naming a token bound to the caller's
/// identity; presenting that token in `hub/hello` makes the caller exactly that
/// identity — never `admin`, never another agent's.
///
/// The tokens live in the hub directory with mode 0600, so they survive a
/// daemon restart, and are never listed: a caller only ever learns the token
/// minted for its own identity, and a lookup goes by token and nothing else.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WatchTokens {
    dir: PathBuf,
}

/// One token row: the token itself, and the clock it was minted at, so a stale
/// row can be dropped once the file grows.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct TokenRow {
    token: String,
    created: u64,
}

/// The token file as it is written: identity → row, ordered so the file is
/// byte-stable for the same contents.
#[derive(Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
struct TokenStore {
    rows: BTreeMap<String, TokenRow>,
}

impl TokenStore {
    /// Drop the rows that can never be presented again: a `host:` identity
    /// whose process is gone.
    fn retain_live(&mut self) {
        self.rows.retain(|identity, _| identity_alive(identity));
    }

    /// Drop the oldest rows until the file is back inside [`MAX_TOKEN_ROWS`].
    fn cap_rows(&mut self) {
        if self.rows.len() <= MAX_TOKEN_ROWS {
            return;
        }
        let mut by_age: Vec<(String, u64)> = self
            .rows
            .iter()
            .map(|(identity, row)| (identity.clone(), row.created))
            .collect();
        by_age.sort_by_key(|(_, created)| *created);
        let excess = self.rows.len() - MAX_TOKEN_ROWS;
        for (identity, _) in by_age.into_iter().take(excess) {
            self.rows.remove(&identity);
        }
    }
}

/// Whether `identity` can still be presented by a live process.
///
/// A `host:<comm>:<pid>:<starttime>` row whose pid is gone — or whose start
/// time no longer matches, so the kernel recycled the pid — is dead weight in
/// the token file. Sessions, overrides and fallback identities name no process,
/// so they are kept.
fn identity_alive(identity: &str) -> bool {
    let Some(rest) = identity.strip_prefix("host:") else {
        return true;
    };
    let host = rest.split("/session:").next().unwrap_or(rest);
    let mut fields = host.split(':');
    let (Some(_comm), Some(pid), Some(starttime)) = (fields.next(), fields.next(), fields.next())
    else {
        return true;
    };
    let (Ok(pid), Ok(starttime)) = (pid.parse::<u32>(), starttime.parse::<u64>()) else {
        return true;
    };
    match crate::hub::identity::process(pid) {
        Some(process) => process.starttime == starttime,
        None => false,
    }
}

/// The held `flock` on the token file, released on drop.
struct TokenLock {
    file: std::fs::File,
}

impl Drop for TokenLock {
    fn drop(&mut self) {
        // SAFETY: unlocking the same fd this guard locked.
        unsafe {
            libc::flock(self.file.as_raw_fd(), libc::LOCK_UN);
        }
    }
}

impl WatchTokens {
    /// The store kept in `dir`, the hub directory.
    pub fn new(dir: PathBuf) -> Self {
        Self { dir }
    }

    /// The token bound to `identity`, minting one the first time it is asked
    /// for. `None` when the store cannot be read or written.
    pub fn token_for(&self, identity: &str) -> Option<String> {
        let _lock = self.lock(true)?;
        let mut store = self.read();
        if let Some(row) = store.rows.get(identity) {
            return Some(row.token.clone());
        }
        let token = mint_token()?;
        // Dead rows go first, so a long-lived hub cannot grow the file without
        // limit; the caller's own row is live by definition — it is the
        // identity asking right now.
        store.retain_live();
        store.rows.insert(
            identity.to_string(),
            TokenRow {
                token: token.clone(),
                created: now_secs(),
            },
        );
        store.cap_rows();
        self.write(&store)?;
        Some(token)
    }

    /// The identity `token` was minted for, or `None` when this store does not
    /// know it: an unknown token is not an identity, so the caller falls back
    /// to the next rule instead of being locked out.
    pub fn identity_of(&self, token: &str) -> Option<String> {
        let _lock = self.lock(false)?;
        self.read()
            .rows
            .iter()
            .find(|(_, row)| row.token == token)
            .map(|(identity, _)| identity.clone())
    }

    /// The token file, locked exclusively so two connections minting at once
    /// cannot lose each other's row.
    fn lock(&self, create: bool) -> Option<TokenLock> {
        let mut options = std::fs::OpenOptions::new();
        options.read(true).write(true);
        if create {
            options.create(true).mode(0o600);
        }
        let file = options.open(self.dir.join(WATCH_TOKENS_FILE)).ok()?;
        // SAFETY: `flock` takes the fd and an operation flag; no pointers.
        let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX) };
        if rc != 0 {
            return None;
        }
        Some(TokenLock { file })
    }

    /// The store on disk; an absent or unreadable file is an empty store.
    fn read(&self) -> TokenStore {
        let Ok(text) = std::fs::read_to_string(self.dir.join(WATCH_TOKENS_FILE)) else {
            return TokenStore::default();
        };
        serde_json::from_str(&text).unwrap_or_default()
    }

    /// Write the store back, replacing the file so a reader never sees a
    /// half-written row.
    fn write(&self, store: &TokenStore) -> Option<()> {
        let text = serde_json::to_string(store).ok()?;
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(self.dir.join(WATCH_TOKENS_FILE))
            .ok()?;
        std::io::Write::write_all(&mut file, text.as_bytes()).ok()
    }
}

/// The identity a `MINI_SWE_WATCH_TOKEN` names, read from the hub directory's
/// token store. `None` when the token is unknown, so `whoami` can say so
/// instead of inventing an identity.
pub fn watch_token_identity(token: &str) -> Option<String> {
    WatchTokens::new(hub_dir().ok()?).identity_of(token)
}

/// 32 hexadecimal characters of kernel randomness: unguessable, and short
/// enough to paste into a shell command.
fn mint_token() -> Option<String> {
    use std::fmt::Write as _;
    use std::io::Read as _;

    let mut bytes = [0u8; 16];
    std::fs::File::open("/dev/urandom")
        .ok()?
        .read_exact(&mut bytes)
        .ok()?;
    let mut token = String::with_capacity(32);
    for byte in bytes {
        let _ = write!(token, "{byte:02x}");
    }
    Some(token)
}

/// The wall clock in whole seconds, for a token row's age.
fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A descriptor only names the rotated log when the device matches too.
    ///
    /// Inode numbers are unique per filesystem, not globally, so a stderr on
    /// another mount can share the log's inode. Treating that as the same file
    /// would hand the process's own stderr to `dup2` and redirect it into the
    /// log without anyone asking.
    #[test]
    fn a_descriptor_is_matched_to_a_file_by_device_and_inode() {
        assert!(
            is_same_file(7, 42, 7, 42),
            "the same device and inode are the same file"
        );
        assert!(
            !is_same_file(7, 42, 9, 42),
            "a colliding inode on another device is a different file"
        );
        assert!(
            !is_same_file(7, 42, 7, 43),
            "a different inode on the same device is a different file"
        );
    }

    /// A log at or below the cap is left exactly as it is; the cap is what
    /// rotates, and a log under it must survive untouched.
    #[test]
    fn a_log_below_the_cap_is_not_rotated() {
        let scratch = crate::test_support::TestScratch::new("hub-log-under-cap");
        let log = scratch.path().join("hub.log");
        std::fs::write(&log, "under the cap").expect("seed the log");

        assert!(!rotate_log(&log).expect("rotate"));
        assert!(log.is_file(), "the log must survive below the cap");
        assert!(
            !rotated_log_path(&log).exists(),
            "no generation may be created below the cap"
        );
        assert_eq!(
            std::fs::read_to_string(&log).expect("read the log"),
            "under the cap",
            "a log below the cap must keep its content"
        );
    }

    /// Past the cap the log is retired to `hub.log.1` and the caller opens a
    /// fresh one; a second oversized generation replaces the first instead of
    /// piling up, so the directory never holds more than two.
    #[test]
    fn an_oversized_log_rotates_and_keeps_one_previous_generation() {
        let scratch = crate::test_support::TestScratch::new("hub-log-rotate");
        let log = scratch.path().join("hub.log");
        let previous = rotated_log_path(&log);
        let oversize = vec![b'x'; (LOG_ROTATE_BYTES + 1) as usize];
        std::fs::write(&log, &oversize).expect("seed the oversized log");

        assert!(rotate_log(&log).expect("rotate the oversized log"));
        assert_eq!(
            std::fs::read(&previous).expect("read the rotated generation"),
            oversize,
            "the whole oversized log must be kept as hub.log.1"
        );
        // The caller reopens the path: it is now a new, empty file.
        std::fs::write(&log, b"fresh\n").expect("start the new generation");
        assert!(previous.is_file(), "the previous generation must survive");

        // A second oversized log replaces that generation rather than adding
        // a second one.
        std::fs::write(&log, vec![b'y'; (LOG_ROTATE_BYTES + 1) as usize]).expect("seed again");
        assert!(rotate_log(&log).expect("rotate the second time"));
        assert_eq!(
            std::fs::metadata(&previous)
                .expect("stat the generation")
                .len(),
            LOG_ROTATE_BYTES + 1,
            "hub.log.1 must hold the newest oversized log"
        );
        // Rotation leaves the name free for the caller's fresh file; nothing
        // else is created, so a second generation can never accumulate.
        assert!(
            !log.exists(),
            "rotation leaves the log name free for the caller's fresh file"
        );
        let kept: Vec<_> = std::fs::read_dir(scratch.path())
            .expect("list the hub dir")
            .map(|entry| entry.expect("dir entry").file_name())
            .collect();
        assert_eq!(
            kept,
            vec![std::ffi::OsString::from("hub.log.1")],
            "only the retired generation may remain, found {kept:?}"
        );
    }

    /// The abstract-namespace fallback (used when no short writable directory
    /// exists, e.g. inside a sandbox) binds and accepts like a path socket.
    #[tokio::test]
    async fn an_abstract_endpoint_binds_and_connects() {
        let endpoint = HubEndpoint::Abstract(format!("mswe-test-{}", std::process::id()));
        let listener = bind_endpoint(&endpoint).expect("bind the abstract socket");
        let accept = tokio::spawn(async move { listener.accept().await.map(|_| ()) });
        connect_endpoint(&endpoint)
            .await
            .expect("connect to the abstract socket");
        accept
            .await
            .expect("accept task")
            .expect("accept the connection");
    }

    /// The short fallback directory is created by the daemon that binds it and
    /// taken away with the socket, so nothing is ever left holding an empty one.
    ///
    /// Everything else resolves the endpoint without touching the filesystem,
    /// which is what makes the resolution side-effect free; this pins the other
    /// half of that contract.
    #[tokio::test]
    async fn binding_a_fallback_socket_creates_a_directory_the_guard_then_removes() {
        let scratch = crate::test_support::TestScratch::new("hub-bind-fallback");
        let hub_dir = scratch.path().join("hub");
        std::fs::create_dir_all(&hub_dir).expect("create the hub dir");
        // The fallback directory stands in for `/tmp/mswe-<uid>-<hash>`, so
        // this test never writes outside its scratch.
        let fallback = scratch.path().join("mswe-fallback");
        let socket = fallback.join("hub.sock");
        let guard = FallbackSocketGuard::new(&socket, &hub_dir);

        // The daemon's sequence: create the directory, then bind into it.
        harden_hub_dir(fallback.clone()).expect("create the fallback socket directory");
        let listener =
            bind_endpoint(&HubEndpoint::Path(socket.clone())).expect("bind the fallback socket");
        assert!(fallback.is_dir());
        assert!(socket.exists());
        drop(listener);
        drop(guard);
        assert!(
            !fallback.exists(),
            "the guard must remove the fallback directory with the socket"
        );
        assert!(!socket.exists(), "the guard must remove the socket");
    }

    /// Resolving the endpoint of a hub directory too deep for `sun_path` names
    /// a short fallback directory under `/tmp` -- and must not leave one behind.
    ///
    /// Callers routinely ask where the socket would go and never bind it (the
    /// tests' `socket()` probes, a client deciding whether a daemon is
    /// listening). A `socket()` call that created `/tmp/mswe-<uid>-<hash>` would
    /// leave an empty directory behind for every such probe, under a `TMPDIR`
    /// deep enough to force the fallback.
    #[test]
    fn resolving_a_deep_hub_endpoint_creates_no_fallback_directory() {
        let scratch = crate::test_support::TestScratch::new("hub-deep-endpoint");
        let deep = scratch
            .path()
            .join("a-rather-long-directory-name-to-push-the-socket-path")
            .join("past-the-unix-socket-path-limit-of-one-hundred-and-eight-bytes");
        std::fs::create_dir_all(&deep).expect("create the deep hub dir");
        let paths = HubPaths::new(deep.clone());

        // The premise: this hub directory is too deep to hold its own socket.
        assert!(
            deep.join("hub.sock").as_os_str().len() >= MAX_SOCKET_PATH,
            "the hub dir must exceed the socket path limit for this test to mean anything"
        );

        // Asking where the socket would go creates nothing ...
        let fallback = paths
            .fallback_dir()
            .expect("a deep hub dir needs a fallback dir");
        // The directory the endpoint resolution names must not exist, whether or
        // not this host can create one: a resolvable `/tmp` is what turned every
        // such probe into an empty leftover directory.
        let _ = paths.endpoint();
        assert!(
            !fallback.exists(),
            "fallback_dir() must only compute the path, not create {}",
            fallback.display()
        );
        let _ = paths.socket();
        assert!(
            !fallback.exists(),
            "probing the socket left {} behind",
            fallback.display()
        );
        // Nor does a client checking whether a daemon is already up, nor the
        // lock waiter probing for a predecessor: both dial the fallback socket
        // through `endpoint()`, which resolves it without creating it.
        let _ = paths.socket();
        assert!(
            !fallback.exists(),
            "a probe of the endpoint left {} behind",
            fallback.display()
        );
    }

    /// A fallback directory this user owns and kept private is trusted; a
    /// squatted, group/world-accessible or symlinked one is not, so a client
    /// never dials a socket another user planted under the shared name.
    #[test]
    fn a_foreign_open_or_symlinked_fallback_directory_is_refused() {
        let scratch = crate::test_support::TestScratch::new("hub-fallback-trust");

        let owned = scratch.path().join("owned");
        std::fs::create_dir(&owned).expect("create the owned dir");
        std::fs::set_permissions(&owned, std::fs::Permissions::from_mode(0o700))
            .expect("private mode");
        assert!(fallback_dir_is_creatable(&owned));

        let open = scratch.path().join("open");
        std::fs::create_dir(&open).expect("create the open dir");
        std::fs::set_permissions(&open, std::fs::Permissions::from_mode(0o777)).expect("open mode");
        assert!(
            !fallback_dir_is_creatable(&open),
            "a group/world-accessible fallback directory must be refused"
        );

        let link = scratch.path().join("link");
        std::os::unix::fs::symlink(&owned, &link).expect("create the symlink");
        assert!(
            !fallback_dir_is_creatable(&link),
            "a symlinked fallback directory must be refused"
        );

        // Absent: probed by creating and undoing, so nothing is left behind.
        let absent = scratch.path().join("absent");
        assert!(fallback_dir_is_creatable(&absent));
        assert!(
            !absent.exists(),
            "the creatability probe must not leave {} behind",
            absent.display()
        );
    }

    /// The peer-uid seam accepts this process's own uid and refuses any other,
    /// so a squatted socket can deny service but never impersonate the hub.
    #[test]
    fn a_foreign_peer_uid_is_refused() {
        assert!(ensure_peer_is_self(current_uid()).is_ok());
        let foreign = current_uid().wrapping_add(1);
        let error = ensure_peer_is_self(foreign).expect_err("a foreign uid is refused");
        assert_eq!(error.kind(), std::io::ErrorKind::PermissionDenied);
        assert!(
            error.to_string().contains("hub peer is uid"),
            "the refusal must say who the peer is: {error}"
        );
    }
}
