//! The hub daemon: one process, one [`WorkerPool`], many connections.
//!
//! Lifecycle, in order: take the exclusive `flock` on `hub.lock` (a second
//! daemon on the same directory gives up instead of stealing the socket),
//! clear a stale `hub.sock`, bind and serve. Every accepted connection is
//! checked with `SO_PEERCRED` and then handed to
//! [`McpServer::serve_connection`], so the wire behaviour is byte-identical to
//! the stdio server. The daemon exits when it has had no connection and no
//! live worker for [`HubConfig::idle_secs`], or on `SIGTERM`/`SIGINT`.

use std::collections::BTreeMap;
use std::os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt};
use std::os::unix::process::CommandExt;
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
        let key = format!(
            "mswe-{}-{:016x}",
            current_uid(),
            fnv1a(self.dir.as_os_str().as_encoded_bytes())
        );
        match harden_hub_dir(PathBuf::from("/tmp").join(&key)) {
            Ok(dir) => HubEndpoint::Path(dir.join("hub.sock")),
            Err(_) => HubEndpoint::Abstract(key),
        }
    }

    /// The lock file serialising daemons on this directory.
    pub fn lock(&self) -> PathBuf {
        self.dir.join("hub.lock")
    }

    /// The daemon's log file.
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
        return harden_hub_dir(PathBuf::from(dir));
    }
    let dir = crate::worktree::swe_base_dir().join(format!("mini-swe-hub-{}", current_uid()));
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
pub async fn connect_endpoint(endpoint: &HubEndpoint) -> std::io::Result<UnixStream> {
    match endpoint {
        HubEndpoint::Path(path) => UnixStream::connect(path).await,
        HubEndpoint::Abstract(name) => {
            use std::os::linux::net::SocketAddrExt;
            let addr = std::os::unix::net::SocketAddr::from_abstract_name(name.as_bytes())?;
            let stream = std::os::unix::net::UnixStream::connect_addr(&addr)?;
            stream.set_nonblocking(true)?;
            UnixStream::from_std(stream)
        }
    }
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

/// Longest socket path used as-is, below Linux's 108-byte `sun_path`.
const MAX_SOCKET_PATH: usize = 100;

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
}

impl HubConfig {
    /// Configuration for an explicit hub directory and idle window.
    pub fn new(paths: HubPaths, idle_secs: u64) -> Self {
        Self { paths, idle_secs }
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
            let budget = pool.auto_continue_budget(&id).await;
            if budget == 0 {
                info!(
                    worker = %id,
                    "Interrupted worker has spent its automatic continuations; leaving it for the orchestrator"
                );
                continue;
            }
            let message = "the hub restarted".to_string();
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
        match tokio::task::spawn_blocking(crate::pool::recover_orphaned_workers).await {
            Ok(recovered) => {
                info!(workers = recovered, "Recovered orphaned hub workers");
                append_log(
                    &self.config.paths().log(),
                    &format!("recovered {recovered} orphaned workers"),
                );
            }
            Err(e) => error!(error = %e, "Hub recovery task failed"),
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
        loop {
            if let Some(lock) = acquire_lock(&path)? {
                return Ok(Some(lock));
            }
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

        // A socket left behind by a killed daemon would make `bind` fail with
        // "address already in use"; the lock proves nobody owns it now.
        let socket = paths.socket();
        // A socket the hub dir is too deep to hold is filed in a short private
        // directory of its own. Tie that directory's life to this future so an
        // abort (a test dropping the daemon task) removes it, not only the
        // graceful shutdown path below.
        let _socket_cleanup = FallbackSocketGuard::new(&socket, paths.dir());
        let _ = std::fs::remove_file(&socket);
        let endpoint = paths.endpoint();
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

        let events = self.server.start_hub_events().await;
        let mut shutdown = self.server.subscribe_shutdown();

        let reaper = crate::pool::spawn_reaper((*self.server.pool()).clone());
        let idle_watcher = self.clone();
        let mut idle_task = tokio::spawn(async move { idle_watcher.watch_idle().await });

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
        recovery_task.abort();
        reaper.abort();
        events.abort();
        let killed = self.server.pool().kill_all().await;
        if killed > 0 {
            info!(workers = killed, "Terminated workers on hub shutdown");
        }
        let _ = std::fs::remove_file(&socket);
        // A socket moved to a short fallback directory takes it along.
        if let Some(parent) = socket.parent()
            && parent != paths.dir()
        {
            let _ = std::fs::remove_dir(parent);
        }
        append_log(&paths.log(), "stopped");
        drop(lock);
        // A handover leaves the hub unserved until a client dials it again, and
        // the workers it just interrupted are auto-continued by whichever
        // daemon binds the socket next, so the daemon that stepped aside starts
        // that daemon itself rather than waiting for one.
        if self.server.handover_requested() {
            respawn_daemon(paths);
        }
        debug!("Hub daemon stopped");
        Ok(true)
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
            let res = server
                .serve_connection(
                    tokio::io::BufReader::new(reader),
                    writer,
                    crate::mcp::ConnectionContext::hub_connection(id).with_watch_tokens(tokens),
                )
                .await;
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
/// `cargo` replaces the binary in place, so the running daemon's
/// `/proc/self/exe` reads `<path> (deleted)` while `path` itself already holds
/// the newer build: respawning the path is what makes the replacement the new
/// binary instead of another copy of this one.
fn respawn_daemon(paths: &HubPaths) {
    let Some(exe) = current_exe_path() else {
        warn!("No executable to hand over to; the next client will start the hub");
        return;
    };
    let log = match std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW)
        .open(paths.log())
    {
        Ok(log) => log,
        Err(e) => {
            warn!(error = %e, "Could not open the hub log for the replacement daemon");
            return;
        }
    };
    let mut command = std::process::Command::new(&exe);
    command
        .arg("daemon")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(log);
    // SAFETY: setsid is async-signal-safe and touches no Rust state after fork.
    // SAFETY: setsid is async-signal-safe and touches no Rust state after fork.
    unsafe {
        command.pre_exec(|| {
            if libc::setsid() == -1 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    match tokio::process::Command::from(command).spawn() {
        Ok(mut child) => {
            info!(exe = %exe.display(), "Started the replacement hub daemon");
            append_log(&paths.log(), &format!("handover to {}", exe.display()));
            tokio::spawn(async move {
                let _ = child.wait().await;
            });
        }
        Err(e) => {
            warn!(error = %e, exe = %exe.display(), "Could not start the replacement hub daemon")
        }
    }
}

/// This executable's path with the kernel's ` (deleted)` suffix removed.
///
/// `None` when the path no longer exists, which means there is nothing to
/// respawn and the next client starts the hub instead.
fn current_exe_path() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let text = exe.to_string_lossy();
    let path = PathBuf::from(text.strip_suffix(" (deleted)").unwrap_or(&text));
    path.is_file().then_some(path)
}

/// Append one timestamped line to the hub log; failures are traced, never fatal.
fn append_log(path: &Path, event: &str) {
    use std::fmt::Write as _;
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let mut line = String::new();
    let _ = writeln!(line, "{now} pid={} {event}", std::process::id());
    use std::io::Write as _;
    if let Err(e) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
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
    HubServer::new(server, HubConfig::new(paths, idle))
        .run()
        .await
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
}
