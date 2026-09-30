//! The hub daemon: one process, one [`WorkerPool`], many connections.
//!
//! Lifecycle, in order: take the exclusive `flock` on `hub.lock` (a second
//! daemon on the same directory gives up instead of stealing the socket),
//! clear a stale `hub.sock`, bind and serve. Every accepted connection is
//! checked with `SO_PEERCRED` and then handed to
//! [`McpServer::serve_connection`], so the wire behaviour is byte-identical to
//! the stdio server. The daemon exits when it has had no connection and no
//! live worker for [`HubConfig::idle_secs`], or on `SIGTERM`/`SIGINT`.

use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use anyhow::{Context, Result};
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

    /// The Unix socket clients connect to.
    pub fn socket(&self) -> PathBuf {
        self.dir.join("hub.sock")
    }

    /// The lock file serialising daemons on this directory.
    pub fn lock(&self) -> PathBuf {
        self.dir.join("hub.lock")
    }

    /// The daemon's log file.
    pub fn log(&self) -> PathBuf {
        self.dir.join("hub.log")
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
    pub async fn run(&self) -> Result<bool> {
        raise_nofile_limit();
        let paths = self.config.paths();
        let lock = acquire_lock(&paths.lock())?;
        let Some(lock) = lock else {
            info!("hub already running");
            return Ok(false);
        };

        // A socket left behind by a killed daemon would make `bind` fail with
        // "address already in use"; the lock proves nobody owns it now.
        let socket = paths.socket();
        let _ = std::fs::remove_file(&socket);
        let listener = UnixListener::bind(&socket)
            .with_context(|| format!("Could not bind hub socket {}", socket.display()))?;
        std::fs::set_permissions(&socket, std::fs::Permissions::from_mode(0o600))
            .with_context(|| format!("Could not restrict {} to 0600", socket.display()))?;

        info!(socket = %socket.display(), idle_secs = self.config.idle_secs(), "Hub daemon listening");
        append_log(&paths.log(), "listening");

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
        reaper.abort();
        events.abort();
        let killed = self.server.pool().kill_all().await;
        if killed > 0 {
            info!(workers = killed, "Terminated workers on hub shutdown");
        }
        let _ = std::fs::remove_file(&socket);
        append_log(&paths.log(), "stopped");
        drop(lock);
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
        tokio::spawn(async move {
            *open_conns.lock().await += 1;
            let (reader, writer) = stream.into_split();
            let res = server
                .serve_connection(
                    tokio::io::BufReader::new(reader),
                    writer,
                    crate::mcp::ConnectionContext::hub_connection(id),
                )
                .await;
            *open_conns.lock().await -= 1;
            if let Err(e) = res {
                debug!(connection = id, error = %e, "Hub connection ended");
            }
        });
    }
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
