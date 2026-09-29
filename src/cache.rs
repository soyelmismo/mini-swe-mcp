//! Shared toolchain/compiler caches and the process-wide, bounded
//! least-recently-used (LRU) storage they are memoized in.
//!
//! This module has two responsibilities:
//!
//! 1. **Filesystem cache wiring** — resolve the shared cache root, lay out the
//!    well-known package/compiler cache directories (`kache`, `uv`, `pip`, the
//!    Node toolchain and Go), and turn that into (a) `bubblewrap` bind mounts and
//!    (b) child-process environment variables. The `OnceLock`-memoized tool
//!    probes (`has_kache` / `has_sccache`) answer "is this compiler wrapper
//!    installed?" at most once per process.
//!
//! 2. **Bounded LRU storage** — [`LruCache`], a small, dependency-free,
//!    thread-safe cache with **O(1) amortized** `get` / `insert` / eviction, plus
//!    a [`OnceLock`]-backed registry ([`shared_cache`]) that materializes each
//!    named cache exactly once per process and hands out a shared handle.
//!
//! # Why an explicit LRU instead of a `HashMap` + `clear()`
//!
//! The classic bounded-memoization shortcut — `if map.len() >= CAP { map.clear() }`
//! — is *not* O(1): a full `clear()` walks and drops every entry, and it throws
//! away hot entries that will be immediately re-requested. [`LruCache`] instead
//! evicts exactly the least-recently-*used* entry (the front of a recency
//! `VecDeque`) in O(1) by popping a single key and removing its map entry,
//! leaving the working set warm.
//!
//! # Complexity
//!
//! * `get`, `insert`, `remove`, `peek`, eviction: **O(1) amortized**.
//! * `clear`, `len`: `clear` is O(n) by definition; `len` is O(1).
//!
//! # Thread safety
//!
//! [`LruCache`] is **not** internally synchronized: it is a plain value that
//! `&mut` methods operate on. [`SharedCache`] wraps one in an `RwLock` and is
//! the type to use across threads.
//!
//! On a `SharedCache`, [`SharedCache::peek`] runs under a *shared* read lock and
//! therefore many threads can read concurrently, while `get` / `insert` /
//! `remove` take the exclusive lock.
//!
//! # `peek` vs `get`: the recency trade-off
//!
//! `peek` never refreshes recency, and that is a real semantic difference, not
//! just a locking one. A key that is *read* only through `peek` ages as if it
//! were never touched, so a hot key can still be evicted by a stream of cold
//! inserts. Use `peek` for membership checks that must not influence eviction;
//! use `get` whenever the key's popularity should keep it alive. The catalog
//! cache in `src/manifest/cache.rs` reads through `get` for exactly this reason.

use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock, RwLock};

// ---------- Bounded LRU storage ----------

/// A bounded, thread-safe, least-recently-used cache.
///
/// Keys are ordered by recency: the most recently used is at the **back** of an
/// internal `VecDeque`, the least recently used at the **front**. The map
/// associates each key with its value and (optionally) a generation counter used
/// to discard stale recency records when a key is re-inserted.
///
/// When the cache is full, inserting a new key evicts exactly one
/// least-recently-used entry in O(1) instead of clearing the whole map, keeping
/// the hot working set resident.
///
/// # Examples
///
/// ```
/// use mini_swe_mcp::cache::LruCache;
/// let mut c = LruCache::with_capacity(2);
/// c.insert("a", 1);
/// c.insert("b", 2);
/// // Touching "a" makes "b" the least-recently-used entry.
/// let _ = c.get(&"a");
/// c.insert("c", 3); // evicts "b", not "a".
/// assert_eq!(c.get(&"a"), Some(&1));
/// assert_eq!(c.get(&"b"), None);
/// assert_eq!(c.get(&"c"), Some(&3));
/// ```
#[derive(Debug)]
pub struct LruCache<K, V>
where
    K: std::hash::Hash + Eq + Clone,
{
    /// Hard upper bound on the number of live entries.
    capacity: usize,
    /// Value store: the authoritative `key -> (value, generation)` mapping.
    map: HashMap<K, (V, u64)>,
    /// Recency order as `(key, generation)` pairs, least-recently-used first.
    ///
    /// A key is *not* removed from the middle of this deque on access (that
    /// would be O(n)). Instead, access pushes a new record carrying a fresh,
    /// monotonically increasing generation. A record is **live** only if its
    /// generation matches the generation currently stored in `map` for that key;
    /// every superseded record is stale and is skipped (and dropped) during
    /// eviction. This makes the least-recently-used entry unambiguous and
    /// eviction exactly O(1).
    order: VecDeque<(K, u64)>,
    /// Monotonic counter handing out the next generation number.
    next_gen: u64,
}

impl<K, V> LruCache<K, V>
where
    K: std::hash::Hash + Eq + Clone,
{
    /// Create an empty cache holding at most `capacity` entries.
    ///
    /// A `capacity` of `0` is treated as `1`: a zero-capacity cache can hold
    /// nothing, which turns every `insert` into a no-op and every `get` into a
    /// miss — almost never what a caller means, and a silent foot-gun. Clamping
    /// to `1` keeps the cache useful and its invariants simple.
    pub fn with_capacity(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        Self {
            capacity,
            map: HashMap::with_capacity(capacity),
            order: VecDeque::with_capacity(capacity),
            // Generations start at 1 so a generation of 0 is never "live" for a
            // key that has not been inserted yet.
            next_gen: 1,
        }
    }

    /// Allocate the next recency generation. Wrapping is astronomically unlikely
    /// (2^64 accesses); on the (theoretical) wrap we skip 0 to keep "0 means
    /// unset" invariant.
    fn next_generation(&mut self) -> u64 {
        let g = self.next_gen;
        self.next_gen = self.next_gen.wrapping_add(1).max(1);
        g
    }

    /// Number of live entries. O(1).
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Whether the cache holds no entries. O(1).
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Configured maximum number of entries.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Look up `key` **without** affecting recency.
    ///
    /// This is the only read that is safe under a *shared* lock in a
    /// synchronized wrapper, because it never needs to reorder `order`. Use it
    /// when you only need membership and want to keep concurrency maximal.
    pub fn peek(&self, key: &K) -> Option<&V> {
        self.map.get(key).map(|(v, _)| v)
    }

    /// Whether `key` is present, without affecting recency. O(1).
    pub fn contains_key(&self, key: &K) -> bool {
        self.map.contains_key(key)
    }

    /// Look up `key`, marking it most-recently-used on a hit. O(1) amortized.
    ///
    /// Recency is recorded by pushing a fresh `(key)` record onto the back of
    /// `order`. Older records for the same key become *stale* and are skipped (and
    /// dropped) later during eviction; this keeps `get` O(1) without needing to
    /// remove an arbitrary element from the middle of a `VecDeque` (which would
    /// be O(n)).
    pub fn get(&mut self, key: &K) -> Option<&V> {
        if !self.map.contains_key(key) {
            return None;
        }
        self.touch(key);
        self.map.get(key).map(|(v, _)| v)
    }

    /// Mark `key` as most-recently-used by appending a fresh-generation record.
    fn touch(&mut self, key: &K) {
        let generation = self.next_generation();
        if let Some((_, current)) = self.map.get_mut(key) {
            *current = generation;
            self.order.push_back((key.clone(), generation));
        }
    }

    /// Insert `key -> value`, evicting the least-recently-used entry if the
    /// cache would exceed its capacity. O(1) amortized.
    ///
    /// Returns the previous value for `key`, if any. An existing key is refreshed
    /// (its value replaced and its recency updated) without evicting anything.
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        let generation = self.next_generation();
        let previous = self
            .map
            .insert(key.clone(), (value, generation))
            .map(|(v, _)| v);
        self.order.push_back((key.clone(), generation));

        // Grow (only when this is a new key), then evict least-recently-used
        // until back within capacity. Re-inserting an existing key replaced it
        // in place, so `map.len()` did not grow and nothing is evicted.
        while self.map.len() > self.capacity {
            self.evict_lru();
        }
        previous
    }

    /// Evict the single least-recently-used entry in O(1).
    ///
    /// Pops recency records from the front, discarding any whose key is no
    /// longer present (a stale duplicate left by a previous `get`/`insert`), until
    /// it finds one that maps to a live entry and removes that entry.
    fn evict_lru(&mut self) {
        while let Some((key, generation)) = self.order.pop_front() {
            // Only a record whose generation still matches the map entry is the
            // key's *current* recency mark. Superseded (stale) records are
            // discarded; the first live record names the least-recently-used key.
            if self
                .map
                .get(&key)
                .is_some_and(|(_, current)| *current == generation)
            {
                self.map.remove(&key);
                break;
            }
        }
    }

    /// Remove `key` if present, returning its value. O(1) amortized.
    ///
    /// Recency records for `key` are left in `order` and cleaned up lazily during
    /// eviction, which keeps this O(1) instead of O(n).
    pub fn remove(&mut self, key: &K) -> Option<V> {
        self.map.remove(key).map(|(v, _)| v)
    }

    /// Drop every entry and every recency record. O(n).
    pub fn clear(&mut self) {
        self.map.clear();
        self.order.clear();
    }
}

impl<K, V> Default for LruCache<K, V>
where
    K: std::hash::Hash + Eq + Clone,
{
    fn default() -> Self {
        Self::with_capacity(1)
    }
}

// ---------- Thread-safe shared cache ----------

/// A thread-safe handle to an [`LruCache`].
///
/// The inner cache lives behind an `RwLock`. [`SharedCache::get`] and
/// [`SharedCache::peek`] take the shared (read) lock; [`SharedCache::insert`]
/// takes the exclusive (write) lock.
///
/// # Why `get` and `peek` differ
///
/// The recency `VecDeque` is mutated on access, so a recency-updating [`get`]
/// *must* take the write lock. [`peek`] leaves recency untouched and therefore
/// runs under a shared lock, letting many threads read concurrently. When
/// recency fidelity is not required (e.g. a "is this present?" check), prefer
/// [`SharedCache::peek`].
///
/// # Caveat
///
/// `peek` reads are invisible to the eviction policy. A key that is only ever
/// looked up with `peek` will eventually be evicted even if it is by far the
/// hottest key in the cache. Route hot reads through [`SharedCache::get`] so the
/// recency ordering reflects real usage.
#[derive(Debug)]
pub struct SharedCache<K, V>
where
    K: std::hash::Hash + Eq + Clone,
{
    inner: RwLock<LruCache<K, V>>,
}

impl<K, V> SharedCache<K, V>
where
    K: std::hash::Hash + Eq + Clone,
{
    /// Wrap `cache` in a synchronizing handle.
    pub fn new(cache: LruCache<K, V>) -> Self {
        Self {
            inner: RwLock::new(cache),
        }
    }

    /// Construct a shared cache with the given `capacity`.
    pub fn with_capacity(capacity: usize) -> Self {
        Self::new(LruCache::with_capacity(capacity))
    }

    /// Number of live entries. O(1). Takes the shared lock.
    pub fn len(&self) -> usize {
        self.inner.read().map(|c| c.len()).unwrap_or(0)
    }

    /// Whether the cache holds no entries. O(1).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Configured maximum number of entries. O(1).
    pub fn capacity(&self) -> usize {
        self.inner.read().map(|c| c.capacity()).unwrap_or(0)
    }

    /// Look up `key` without affecting recency, under the shared lock. O(1).
    pub fn peek(&self, key: &K) -> Option<V>
    where
        V: Clone,
    {
        self.inner.read().ok().and_then(|c| c.peek(key).cloned())
    }

    /// Look up `key`, refreshing its recency, under the write lock. O(1) amortized.
    pub fn get(&self, key: &K) -> Option<V>
    where
        V: Clone,
    {
        let mut guard = self.inner.write().ok()?;
        guard.get(key).cloned()
    }

    /// Insert `key -> value`, evicting the LRU entry at capacity. O(1) amortized.
    pub fn insert(&self, key: K, value: V) {
        if let Ok(mut guard) = self.inner.write() {
            guard.insert(key, value);
        }
    }

    /// Remove `key`, returning its value. O(1) amortized.
    pub fn remove(&self, key: &K) -> Option<V> {
        self.inner.write().ok().and_then(|mut c| c.remove(key))
    }

    /// Drop every entry. O(n).
    pub fn clear(&self) {
        if let Ok(mut guard) = self.inner.write() {
            guard.clear();
        }
    }
}

/// A process-wide named cache: `String` keys, `String` values.
pub type StringCache = SharedCache<String, String>;

/// Capacity used by the built-in named caches created via [`shared_cache`].
const DEFAULT_CACHE_CAPACITY: usize = 256;

type CacheRegistry = HashMap<&'static str, Arc<StringCache>>;

/// Lazily built, process-wide registry of named [`StringCache`]s.
///
/// A `OnceLock` is the right primitive here (rather than re-creating a cache per
/// caller or a `Mutex<HashMap>` that must be locked on every lookup) because the
/// registry itself never changes after the first access: the set of cache *names*
/// is fixed at compile time. `get_or_init` guarantees the registry is built
/// exactly once even under concurrent first access, with no `unsafe` and no
/// lock-ordering concerns. The inner `RwLock` protects the (mutable) registry map,
/// while the `OnceLock` guarantees the registry storage itself is built once.
static CACHE_REGISTRY: OnceLock<RwLock<CacheRegistry>> = OnceLock::new();

fn cache_registry() -> &'static RwLock<CacheRegistry> {
    CACHE_REGISTRY.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Get (or lazily create) the process-wide [`StringCache`] registered under
/// `name`, with `capacity` as its eviction bound.
///
/// This is the "modernize storage with `OnceLock`" entry point: the registry is
/// itself behind a `OnceLock`, so no matter how many threads and callers race
/// here, each named cache is created exactly once and thereafter shared through
/// an `Arc`. A cache's capacity is fixed on first creation; later calls for the
/// same `name` ignore `capacity` and return the already-built cache.
///
/// # Examples
///
/// ```
/// use mini_swe_mcp::cache::shared_cache;
/// let cache = shared_cache("demo", 8);
/// cache.insert("k".to_string(), "v".to_string());
/// assert_eq!(cache.get(&"k".to_string()), Some("v".to_string()));
/// ```
pub fn shared_cache(name: &'static str, capacity: usize) -> Arc<StringCache> {
    let registry = cache_registry();
    // Fast path: already-created cache, take the shared lock.
    if let Ok(entries) = registry.read()
        && let Some(cache) = entries.get(name)
    {
        return Arc::clone(cache);
    }
    // Slow path: create exactly once under the write lock, re-checking in case
    // another thread inserted `name` between the read and the write lock.
    let mut entries = registry.write().unwrap_or_else(|e| e.into_inner());
    Arc::clone(
        entries
            .entry(name)
            .or_insert_with(|| Arc::new(StringCache::with_capacity(capacity))),
    )
}

/// A process-wide cache pre-registered with [`DEFAULT_CACHE_CAPACITY`].
pub fn default_shared_cache(name: &'static str) -> Arc<StringCache> {
    shared_cache(name, DEFAULT_CACHE_CAPACITY)
}

// ---------- Filesystem cache wiring ----------

/// Root directory where shared compiler/package caches reside across workers.
pub fn shared_cache_root() -> PathBuf {
    if let Ok(dir) = std::env::var("SWE_CACHE_DIR") {
        PathBuf::from(dir)
    } else {
        crate::worktree::swe_base_dir().join("swe-cache")
    }
}

/// Paths to known package/compiler caches inside `shared_cache_root()`.
#[derive(Debug, Clone)]
pub struct CacheDirs {
    pub root: PathBuf,
    pub kache: PathBuf,
    pub uv: PathBuf,
    pub pip: PathBuf,
    pub npm: PathBuf,
    pub yarn: PathBuf,
    pub pnpm_home: PathBuf,
    pub pnpm_store: PathBuf,
    pub go_build: PathBuf,
    pub go_mod: PathBuf,
}

impl CacheDirs {
    pub fn new() -> Self {
        let root = shared_cache_root();
        Self {
            kache: root.join("kache"),
            uv: root.join("python").join("uv"),
            pip: root.join("python").join("pip"),
            npm: root.join("node").join("npm"),
            yarn: root.join("node").join("yarn"),
            pnpm_home: root.join("node").join("pnpm"),
            pnpm_store: root.join("node").join("pnpm-store"),
            go_build: root.join("go").join("build"),
            go_mod: root.join("go").join("mod"),
            root,
        }
    }

    /// Ensure all default cache subdirectories exist on disk.
    pub fn ensure_dirs(&self) {
        let _ = std::fs::create_dir_all(&self.root);
        let _ = std::fs::create_dir_all(&self.kache);
        let _ = std::fs::create_dir_all(&self.uv);
        let _ = std::fs::create_dir_all(&self.pip);
        let _ = std::fs::create_dir_all(&self.npm);
        let _ = std::fs::create_dir_all(&self.yarn);
        let _ = std::fs::create_dir_all(&self.pnpm_home);
        let _ = std::fs::create_dir_all(&self.pnpm_store);
        let _ = std::fs::create_dir_all(&self.go_build);
        let _ = std::fs::create_dir_all(&self.go_mod);
    }
}

impl Default for CacheDirs {
    fn default() -> Self {
        Self::new()
    }
}

/// Custom cache binds parsed from `SWE_SHARED_CACHES` (e.g. "/host/a:/guest/a,/host/b").
pub fn parse_custom_cache_binds(raw: &str) -> Vec<(PathBuf, PathBuf)> {
    raw.split(',')
        .filter_map(|entry| {
            let entry = entry.trim();
            if entry.is_empty() {
                return None;
            }
            if let Some((host, guest)) = entry.split_once(':') {
                Some((PathBuf::from(host.trim()), PathBuf::from(guest.trim())))
            } else {
                let p = PathBuf::from(entry);
                Some((p.clone(), p))
            }
        })
        .collect()
}

/// Check if `kache` is available in PATH or standard user binary paths.
///
/// Memoized with a `OnceLock` so the (potentially `fork`+`exec`-bound) probe runs
/// at most once per process.
pub fn has_kache() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        if let Some(home) = std::env::var_os("HOME").map(PathBuf::from)
            && home.join(".local/bin/kache").is_file()
        {
            return true;
        }
        std::process::Command::new("kache")
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    })
}

/// Check if `sccache` is available in PATH.
///
/// Memoized with a `OnceLock` so the probe runs at most once per process.
pub fn has_sccache() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        std::process::Command::new("sccache")
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    })
}

/// Append bubblewrap arguments for mounting the shared cache root, user tool caches,
/// and any custom cache binds specified in `SWE_SHARED_CACHES`.
pub fn append_bwrap_cache_args(cmd: &mut tokio::process::Command, home: Option<&Path>) {
    let dirs = CacheDirs::new();
    dirs.ensure_dirs();

    // 1. Bind global shared cache directory read-write
    let root_str = dirs.root.to_string_lossy();
    cmd.args(["--bind", &root_str, &root_str]);

    // 2. If host home has ~/.cache/kache (or kache is used), bind it so kache works without tmpfs discarding
    if let Some(h) = home {
        let user_kache = h.join(".cache").join("kache");
        let _ = std::fs::create_dir_all(&user_kache);
        let kache_str = user_kache.to_string_lossy();
        cmd.args(["--bind", &kache_str, &kache_str]);
    }

    // 3. User-defined custom cache binds via SWE_SHARED_CACHES
    if let Ok(custom) = std::env::var("SWE_SHARED_CACHES") {
        for (host, guest) in parse_custom_cache_binds(&custom) {
            let _ = std::fs::create_dir_all(&host);
            let h_str = host.to_string_lossy();
            let g_str = guest.to_string_lossy();
            cmd.args(["--bind", &h_str, &g_str]);
        }
    }
}

/// Apply universal cache environment variables to the child command.
pub fn apply_shared_cache_env(cmd: &mut tokio::process::Command) {
    let dirs = CacheDirs::new();

    // Python (uv, pip)
    cmd.env("UV_CACHE_DIR", &dirs.uv);
    cmd.env("PIP_CACHE_DIR", &dirs.pip);

    // Node / JS / TS (npm, yarn, pnpm)
    cmd.env("npm_config_cache", &dirs.npm);
    cmd.env("YARN_CACHE_FOLDER", &dirs.yarn);
    cmd.env("PNPM_HOME", &dirs.pnpm_home);
    cmd.env("PNPM_STORE_DIR", &dirs.pnpm_store);

    // Go (build and modules)
    cmd.env("GOCACHE", &dirs.go_build);
    cmd.env("GOMODCACHE", &dirs.go_mod);

    // Ensure toolchain paths (~/.local/bin, ~/.cargo/bin) are in PATH
    if let Some(home) = std::env::var_os("HOME").map(PathBuf::from) {
        let current_path = std::env::var("PATH").unwrap_or_default();
        let local_bin = home.join(".local/bin").to_string_lossy().to_string();
        let cargo_bin = home.join(".cargo/bin").to_string_lossy().to_string();
        let mut parts: Vec<&str> = current_path.split(':').collect();
        let mut new_parts = Vec::new();
        if !parts.contains(&local_bin.as_str()) {
            new_parts.push(local_bin.as_str());
        }
        if !parts.contains(&cargo_bin.as_str()) {
            new_parts.push(cargo_bin.as_str());
        }
        new_parts.append(&mut parts);
        cmd.env("PATH", new_parts.join(":"));
    }

    // Rust / C / C++ compiler wrapper
    if std::env::var("SWE_DISABLE_KACHE").as_deref() != Ok("1")
        && std::env::var("KACHE_DISABLED").as_deref() != Ok("1")
    {
        if has_kache() {
            cmd.env("RUSTC_WRAPPER", "kache");
        } else if has_sccache() {
            cmd.env("RUSTC_WRAPPER", "sccache");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Barrier;
    use std::thread;

    // ---- LruCache: capacity / eviction ----

    #[test]
    fn lru_insert_and_get_roundtrip() {
        let mut c = LruCache::with_capacity(4);
        assert!(c.is_empty());
        assert_eq!(c.insert("a", 1), None);
        assert_eq!(c.insert("b", 2), None);
        assert_eq!(c.len(), 2);
        assert_eq!(c.get(&"a"), Some(&1));
        assert_eq!(c.get(&"b"), Some(&2));
        assert_eq!(c.get(&"missing"), None);
    }

    #[test]
    fn lru_capacity_zero_is_clamped_to_one() {
        let c = LruCache::<u8, u8>::with_capacity(0);
        assert_eq!(c.capacity(), 1, "zero capacity must clamp to 1");
    }

    #[test]
    fn lru_default_capacity_is_one() {
        let c = LruCache::<u8, u8>::default();
        assert_eq!(c.capacity(), 1);
    }

    #[test]
    fn lru_capacity_overflow_evicts_least_recently_used_in_order() {
        // Fill 3 slots: a, b, c. Touch "a" so it becomes MRU.
        let mut c = LruCache::with_capacity(3);
        c.insert("a", 1);
        c.insert("b", 2);
        c.insert("c", 3);
        assert_eq!(c.get(&"a"), Some(&1)); // a is now MRU; LRU order is b, c, a

        // Insert a 4th distinct key: must evict the LRU ("b"), not the whole map.
        c.insert("d", 4);
        assert_eq!(c.len(), 3, "capacity must never be exceeded");
        assert_eq!(c.get(&"b"), None, "the least-recently-used key is evicted");
        assert_eq!(c.get(&"a"), Some(&1), "the touched key survives");
        assert_eq!(c.get(&"c"), Some(&3));
        assert_eq!(c.get(&"d"), Some(&4));
    }

    #[test]
    fn lru_strict_ldr_order_over_many_inserts() {
        // Sequentially touch 0..N-1 in order, so the LRU is always the oldest.
        let n = 64;
        let mut c = LruCache::with_capacity(8);
        for i in 0..n {
            c.insert(i, i);
        }
        // After touching each in order and exceeding capacity, the retained
        // window is the last `cap` keys (56..=63).
        assert_eq!(c.len(), 8);
        for i in 0..(n - 8) {
            assert_eq!(c.get(&i), None, "key {i} should have been evicted");
        }
        for i in (n - 8)..n {
            assert_eq!(c.get(&i), Some(&i), "key {i} should be retained");
        }
    }

    #[test]
    fn lru_reinsert_updates_value_without_eviction() {
        let mut c = LruCache::with_capacity(2);
        c.insert("a", 1);
        c.insert("b", 2);
        // Updating "a" returns the old value and must not evict "b".
        assert_eq!(c.insert("a", 10), Some(1));
        assert_eq!(c.len(), 2, "re-inserting an existing key does not evict");
        assert_eq!(c.get(&"a"), Some(&10));
        assert_eq!(c.get(&"b"), Some(&2), "other key survives the update");
    }

    #[test]
    fn lru_peek_does_not_affect_recency() {
        let mut c = LruCache::with_capacity(2);
        c.insert("a", 1);
        c.insert("b", 2);
        // peek must NOT refresh "a", so "a" stays the LRU and is evicted.
        assert_eq!(c.peek(&"a"), Some(&1));
        c.insert("c", 3);
        assert_eq!(c.peek(&"a"), None, "peek must not protect a key from eviction");
        assert_eq!(c.peek(&"b"), Some(&2));
    }

    #[test]
    fn lru_contains_key_and_remove() {
        let mut c = LruCache::with_capacity(4);
        c.insert("a", 1);
        assert!(c.contains_key(&"a"));
        assert!(!c.contains_key(&"zz"));
        assert_eq!(c.remove(&"a"), Some(1));
        assert!(!c.contains_key(&"a"));
        assert_eq!(c.remove(&"a"), None, "removing twice yields None");
        assert!(c.is_empty());
    }

    #[test]
    fn lru_remove_then_overflow_skips_stale_recency() {
        // Removing a key leaves a stale recency record; overflow eviction must
        // skip that stale record rather than misbehaving.
        let mut c = LruCache::with_capacity(3);
        c.insert("a", 1);
        c.insert("b", 2);
        c.insert("c", 3);
        c.remove(&"a"); // "a" gone, but "a" is still at the recency front
        c.insert("d", 4);
        c.insert("e", 5);
        assert_eq!(c.len(), 3);
        // "b" and "c" should be the survivors plus the newest.
        assert_eq!(c.peek(&"a"), None);
        assert!(c.contains_key(&"e"));
    }

    #[test]
    fn lru_clear_empties_both_stores() {
        let mut c = LruCache::with_capacity(4);
        c.insert("a", 1);
        c.insert("b", 2);
        c.clear();
        assert!(c.is_empty());
        assert_eq!(c.len(), 0);
        assert_eq!(c.get(&"a"), None);
        // A cleared cache still respects its capacity on refill (cap 4 here).
        c.insert("x", 1);
        c.insert("y", 2);
        c.insert("z", 3);
        c.insert("w", 4);
        assert_eq!(c.len(), 4, "all four fit within capacity");
        c.insert("v", 5); // overflow -> evict LRU ("x")
        assert_eq!(c.len(), 4);
        assert_eq!(c.peek(&"x"), None, "cleared+refilled cache still evicts LRU");
        assert_eq!(c.peek(&"v"), Some(&5));
    }

    #[test]
    fn lru_stays_bounded_under_much_larger_workload() {
        // Capacity overflow with far more inserts than slots.
        let mut c = LruCache::with_capacity(16);
        for i in 0..10_000usize {
            c.insert(i, i * 2);
            assert!(
                c.len() <= 16,
                "cache exceeded capacity mid-workload: {}",
                c.len()
            );
        }
        assert_eq!(c.len(), 16);
        // The most recent 16 keys survive.
        for i in (10_000 - 16)..10_000 {
            assert_eq!(c.get(&i), Some(&(i * 2)));
        }
    }

    /// Textbook LRU reference: a `Vec` of keys ordered least-recently-used first.
    /// Used to cross-check the real implementation's eviction decisions.
    #[derive(Default)]
    struct RefLru {
        order: Vec<i32>,
    }

    impl RefLru {
        fn touch(&mut self, key: i32) {
            if let Some(pos) = self.order.iter().position(|k| *k == key) {
                self.order.remove(pos);
            }
            self.order.push(key);
        }

        fn insert(&mut self, key: i32, capacity: usize) {
            let is_new = !self.order.contains(&key);
            self.touch(key);
            if is_new {
                while self.order.len() > capacity {
                    self.order.remove(0);
                }
            }
        }
    }

    #[test]
    fn lru_matches_reference_implementation_on_random_operations() {
        // Deterministic pseudo-random sequence (no extra dependency): a small
        // LCG keeps the test reproducible while still exploring many
        // get/insert/remove interleavings that break naive LRU implementations.
        const CAP: usize = 12;
        const OPS: usize = 20_000;
        const KEY_SPACE: i32 = 40;

        let mut cache = LruCache::with_capacity(CAP);
        let mut reference = RefLru::default();
        let mut rng: u64 = 0x2545_F491_4F6C_DD1D;

        for op in 0..OPS {
            rng = rng
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let key = (rng >> 33) as i32 % KEY_SPACE;

            match (rng >> 17) % 10 {
                // get
                0..=4 => {
                    assert_eq!(
                        cache.get(&key),
                        reference.order.contains(&key).then_some(&(key * 10)),
                        "get({key}) disagreed at op {op}"
                    );
                    if reference.order.contains(&key) {
                        reference.touch(key);
                    }
                }
                // insert
                5..=7 => {
                    let expected_prev = reference.order.contains(&key).then_some(key * 10);
                    assert_eq!(
                        cache.insert(key, key * 10),
                        expected_prev,
                        "insert({key}) previous value disagreed at op {op}"
                    );
                    reference.insert(key, CAP);
                }
                // remove
                8 => {
                    assert_eq!(
                        cache.remove(&key),
                        reference.order.contains(&key).then_some(key * 10),
                        "remove({key}) disagreed at op {op}"
                    );
                    if let Some(pos) = reference.order.iter().position(|k| *k == key) {
                        reference.order.remove(pos);
                    }
                }
                // peek: must not change recency
                _ => {
                    assert_eq!(
                        cache.peek(&key),
                        reference.order.contains(&key).then_some(&(key * 10)),
                        "peek({key}) disagreed at op {op}"
                    );
                }
            }

            assert!(
                cache.len() <= CAP,
                "capacity exceeded at op {op}: {} > {CAP}",
                cache.len()
            );
            // Membership must match the reference exactly after every operation.
            for k in 0..KEY_SPACE {
                assert_eq!(
                    cache.contains_key(&k),
                    reference.order.contains(&k),
                    "membership of {k} disagreed after op {op}"
                );
            }
        }
    }

    #[test]
    fn lru_retains_hot_set_where_a_bulk_clear_would_drop_it() {
        // The whole point of O(1) LRU eviction over `if len >= CAP { clear() }`:
        // a repeatedly-accessed hot key must survive a flood of one-shot keys.
        const HOT: u32 = u32::MAX;
        let mut cache = LruCache::with_capacity(8);
        cache.insert(HOT, 0u32);

        for i in 0..1_000u32 {
            cache.insert(i, i);
            let _ = cache.get(&HOT);
        }

        assert_eq!(
            cache.get(&HOT),
            Some(&0),
            "the hot key must survive 1000 cold insertions; a bulk clear() would \
             have dropped it on the 8th insert"
        );
        assert_eq!(cache.len(), 8);
    }

    #[test]
    fn lru_retains_a_hot_key_under_cold_traffic_on_the_get_path() {
        // Same property as the catalog cache: with a recency-refreshing `get`
        // read path, a key that is re-read on a steady cadence must never be
        // evicted, no matter how many one-shot keys stream past.
        const HOT: u32 = u32::MAX;
        const CAP: usize = 256;
        let mut cache = LruCache::with_capacity(CAP);
        cache.insert(HOT, 0u32);

        for i in 0..(CAP * 4) {
            let i = i as u32;
            cache.insert(i, i);
            if i.is_multiple_of(4) {
                assert_eq!(cache.get(&HOT), Some(&0), "hot key must be readable");
            }
            assert!(
                cache.contains_key(&HOT),
                "hot key was evicted by LRU at cold key {i}"
            );
            assert!(cache.len() <= CAP);
        }
    }

    // ---- SharedCache: thread safety ----

    #[test]
    fn shared_cache_insert_get_peek_remove() {
        let c = StringCache::with_capacity(4);
        c.insert("k".to_string(), "v".to_string());
        assert_eq!(c.get(&"k".to_string()), Some("v".to_string()));
        assert_eq!(c.peek(&"k".to_string()), Some("v".to_string()));
        assert_eq!(c.len(), 1);
        assert_eq!(c.remove(&"k".to_string()), Some("v".to_string()));
        assert!(c.is_empty());
    }

    #[test]
    fn shared_cache_concurrent_inserts_are_all_visible_and_bounded() {
        const CAP: usize = 32;
        const THREADS: usize = 8;
        const PER_THREAD: usize = 200;

        let cache = Arc::new(StringCache::with_capacity(CAP));
        let barrier = Arc::new(Barrier::new(THREADS));

        let handles: Vec<_> = (0..THREADS)
            .map(|t| {
                let cache = Arc::clone(&cache);
                let barrier = Arc::clone(&barrier);
                thread::spawn(move || {
                    barrier.wait();
                    for i in 0..PER_THREAD {
                        cache.insert(format!("t{t}-k{i}"), format!("v{i}"));
                        // Exercise the read path concurrently too.
                        let _ = cache.peek(&format!("t{t}-k{i}"));
                        let _ = cache.get(&format!("t{t}-k{i}"));
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().expect("worker thread must not panic");
        }

        assert!(
            cache.len() <= CAP,
            "shared cache exceeded capacity under concurrency: {} > {CAP}",
            cache.len()
        );
    }

    #[test]
    fn shared_cache_concurrent_hammer_keeps_len_consistent() {
        // Many threads repeatedly insert the SAME key; len must never exceed 1
        // and must settle at exactly 1, proving the read-modify-write of the map
        // is serialized by the RwLock (no lost updates / torn state).
        let cache = Arc::new(StringCache::with_capacity(8));
        let barrier = Arc::new(Barrier::new(8));
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let cache = Arc::clone(&cache);
                let barrier = Arc::clone(&barrier);
                thread::spawn(move || {
                    barrier.wait();
                    for i in 0..1000 {
                        cache.insert("shared".to_string(), format!("{i}"));
                        let got = cache.get(&"shared".to_string());
                        assert!(got.is_some(), "key must always be present");
                        assert!(cache.len() <= 1, "single key => len <= 1");
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().expect("worker thread must not panic");
        }
        assert_eq!(cache.len(), 1);
    }

    // ---- OnceLock-backed shared_cache registry ----

    #[test]
    fn shared_cache_registry_returns_same_instance_for_same_name() {
        let a = shared_cache("registry-identity-test", 4);
        let b = shared_cache("registry-identity-test", 999); // capacity ignored
        assert!(
            Arc::ptr_eq(&a, &b),
            "same name must yield the identical shared cache"
        );
        // Capacity is fixed on first creation.
        assert_eq!(a.capacity(), 4);
    }

    #[test]
    fn shared_cache_registry_distinct_names_are_distinct_instances() {
        let a = shared_cache("registry-distinct-a", 4);
        let b = shared_cache("registry-distinct-b", 4);
        assert!(!Arc::ptr_eq(&a, &b), "different names are different caches");
    }

    #[test]
    fn shared_cache_registry_is_thread_safe_under_concurrent_access() {
        // Race many threads on the SAME name; they must all get one instance.
        const THREADS: usize = 16;
        let barrier = Arc::new(Barrier::new(THREADS));
        let handles: Vec<_> = (0..THREADS)
            .map(|_| {
                let barrier = Arc::clone(&barrier);
                thread::spawn(move || {
                    barrier.wait();
                    shared_cache("registry-race-test", 6)
                })
            })
            .collect();

        let mut handles = handles.into_iter();
        let first = handles.next().expect("at least one thread").join().expect("thread 0");
        for (i, h) in handles.enumerate() {
            let other = h.join().expect("worker thread must not panic");
            assert!(
                Arc::ptr_eq(&first, &other),
                "concurrent shared_cache(\"registry-race-test\") diverged on thread {i}"
            );
        }
        assert_eq!(first.capacity(), 6);
    }

    #[test]
    fn default_shared_cache_has_default_capacity() {
        let c = default_shared_cache("default-capacity-test");
        assert_eq!(c.capacity(), DEFAULT_CACHE_CAPACITY);
    }

    // ---- Filesystem cache wiring (pre-existing behaviour, kept green) ----

    #[test]
    fn test_shared_cache_root_default() {
        let root = shared_cache_root();
        assert!(root.ends_with("swe-cache"));
    }

    #[test]
    fn test_cache_dirs_creation() {
        let dirs = CacheDirs::new();
        dirs.ensure_dirs();
        assert!(dirs.root.is_dir());
        assert!(dirs.uv.is_dir());
        assert!(dirs.pip.is_dir());
        assert!(dirs.npm.is_dir());
        assert!(dirs.go_build.is_dir());
    }

    #[test]
    fn test_parse_custom_cache_binds() {
        let binds = parse_custom_cache_binds("/host/a:/guest/a, /host/b");
        assert_eq!(binds.len(), 2);
        assert_eq!(binds[0], (PathBuf::from("/host/a"), PathBuf::from("/guest/a")));
        assert_eq!(binds[1], (PathBuf::from("/host/b"), PathBuf::from("/host/b")));

        let empty = parse_custom_cache_binds("  , , ");
        assert!(empty.is_empty());
    }

    #[test]
    fn test_apply_shared_cache_env() {
        let mut cmd = tokio::process::Command::new("true");
        apply_shared_cache_env(&mut cmd);
    }
}
