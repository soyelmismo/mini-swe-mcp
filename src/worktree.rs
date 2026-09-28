use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::process::Command;
use tracing::{error, info};

pub struct WorktreeGuard {
    pub path: PathBuf,
    pub branch: String,
    pub repo_root: PathBuf,
    pub keep: bool,
    pub preserve_branch: bool,
}

fn git(dir: &Path, operation: &str, args: &[&str]) -> Result<std::process::Output> {
    Command::new("git")
        .current_dir(dir)
        .args(args)
        .output()
        .with_context(|| format!("Failed to execute git {operation}"))
}

pub fn swe_base_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("SWE_TEMP_DIR") {
        PathBuf::from(dir)
    } else {
        let var_tmp = PathBuf::from("/var/tmp");
        if var_tmp.is_dir() {
            var_tmp
        } else {
            std::env::temp_dir()
        }
    }
}

impl WorktreeGuard {
    pub fn new(repo_root: &Path, worker_id: &str) -> Result<Self> {
        let branch = format!("worker-{}", worker_id);
        let path = swe_base_dir().join(format!("swe-wt-{}", worker_id));

        // Ensure target directory and branch don't exist
        if path.exists() {
            let _ = std::fs::remove_dir_all(&path);
        }
        let _ = git(repo_root, "worktree prune", &["worktree", "prune"]);
        let _ = git(repo_root, "branch -D", &["branch", "-D", &branch]);

        info!(repo = %repo_root.display(), branch = %branch, path = %path.display(), "Creating git worktree");

        let path_str = path
            .to_str()
            .context("Worktree path contains invalid UTF-8")?;

        let output = git(
            repo_root,
            "worktree add",
            &[
                "worktree",
                "add",
                "-b",
                &branch,
                path_str,
                "HEAD",
            ],
        )?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("git worktree add failed: {}", stderr);
        }

        // Write host PID to sibling file so running worktrees are never pruned by concurrent instances
        let pid_file = format!("{}.pid", path.to_string_lossy());
        let _ = std::fs::write(&pid_file, std::process::id().to_string());

        Ok(Self {
            path,
            branch,
            repo_root: repo_root.to_path_buf(),
            keep: false,
            preserve_branch: false,
        })
    }

    pub fn get_diff(&self) -> Result<String> {
        // Stage untracked files intent-to-add so git diff captures new files as well
        let _ = git(&self.path, "add", &["add", "-N", "."]);

        let output = git(&self.path, "diff", &["diff", "HEAD"])?;

        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }

    /// Sync report and audit directories (audits, reports, .agents, artifacts)
    /// from the worktree back into the repository root.
    pub fn sync_artifacts(&self) -> Result<Vec<String>> {
        const ARTIFACT_DIRS: &[&str] = &["audits", "reports", ".agents", "artifacts"];
        let mut synced = Vec::new();

        for dir in ARTIFACT_DIRS {
            let src_dir = self.path.join(dir);
            if src_dir.is_dir() {
                let dest_dir = self.repo_root.join(dir);
                let _ = copy_dir_all(&src_dir, &dest_dir, &mut synced, &self.path);
            }
        }

        Ok(synced)
    }

    /// Commit all dirty changes in the worktree to preserve work in git history,
    /// marking the branch to be retained upon worktree cleanup.
    pub fn commit_changes(&mut self, message: &str) -> Result<Option<String>> {
        // Stage all changes (both tracked and untracked)
        let _ = git(&self.path, "add", &["add", "-A"]);

        // Check if there are changes to commit
        let status = git(&self.path, "status", &["status", "--porcelain"])?;
        if status.stdout.is_empty() {
            return Ok(None);
        }

        // Commit with fallback credentials so lack of git config never errors
        let _ = git(
            &self.path,
            "commit",
            &[
                "-c",
                "user.name=mini-swe",
                "-c",
                "user.email=mini-swe@localhost",
                "commit",
                "-m",
                message,
            ],
        )?;

        self.preserve_branch = true;
        Ok(Some(self.branch.clone()))
    }
}

fn copy_dir_all(
    src: &Path,
    dst: &Path,
    collected: &mut Vec<String>,
    worktree_root: &Path,
) -> std::io::Result<()> {
    if !src.exists() {
        return Ok(());
    }
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let ft = entry.file_type()?;
        if ft.is_symlink() {
            continue;
        }
        let src_path = entry.path();
        let dst_path = dst.join(entry.file_name());
        if ft.is_dir() {
            copy_dir_all(&src_path, &dst_path, collected, worktree_root)?;
        } else if ft.is_file() {
            std::fs::copy(&src_path, &dst_path)?;
            if let Ok(rel) = src_path.strip_prefix(worktree_root) {
                let rel_str = rel.to_string_lossy().to_string();
                if !collected.contains(&rel_str) {
                    collected.push(rel_str);
                }
            }
        }
    }
    Ok(())
}

pub fn is_process_alive(pid: u32) -> bool {
    #[cfg(target_os = "linux")]
    {
        Path::new(&format!("/proc/{pid}")).exists()
    }
    #[cfg(not(target_os = "linux"))]
    {
        Command::new("kill")
            .args(["-0", &pid.to_string()])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }
}

fn prune_worktree_if_stale(
    repo_root: &Path,
    wt: &str,
    br: &str,
    active_branches: &mut Vec<String>,
) {
    if !br.starts_with("worker-") {
        return;
    }
    let wt_path = Path::new(wt);
    let pid_file = format!("{wt}.pid");
    let is_stale = if !wt_path.exists() {
        true
    } else if let Ok(content) = std::fs::read_to_string(&pid_file) {
        content
            .trim()
            .parse::<u32>()
            .is_ok_and(|pid| !is_process_alive(pid))
    } else {
        // Directory exists but has no pid file: do not delete, assume active
        false
    };

    if is_stale {
        info!(path = %wt, branch = %br, "Pruning zombie subagent worktree");
        let _ = git(repo_root, "worktree remove", &["worktree", "remove", "--force", wt]);
        let _ = git(repo_root, "branch -D", &["branch", "-D", br]);
        if wt_path.exists() {
            let _ = std::fs::remove_dir_all(wt_path);
        }
        let _ = std::fs::remove_file(&pid_file);
        if let Some(wt_name) = wt_path.file_name().and_then(|n| n.to_str()) {
            for base in [swe_base_dir(), std::env::temp_dir()] {
                let target_dir = base.join(format!("swe-target-{wt_name}"));
                if target_dir.exists() {
                    let _ = std::fs::remove_dir_all(&target_dir);
                }
            }
        }
    } else {
        active_branches.push(br.to_string());
    }
}

pub fn prune_stale_worktrees(repo_root: &Path) {
    let _ = git(repo_root, "worktree prune", &["worktree", "prune"]);

    // 1. Prune registered worktrees whose owner process is dead or directory is missing
    let mut active_branches = Vec::new();
    if let Ok(output) = git(repo_root, "worktree list", &["worktree", "list", "--porcelain"]) {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut current_wt: Option<String> = None;
        let mut current_branch: Option<String> = None;

        for line in stdout.lines() {
            if let Some(path) = line.strip_prefix("worktree ") {
                current_wt = Some(path.to_string());
            } else if let Some(branch) = line.strip_prefix("branch refs/heads/") {
                current_branch = Some(branch.to_string());
            } else if line.is_empty()
                && let (Some(wt), Some(br)) = (current_wt.take(), current_branch.take())
            {
                prune_worktree_if_stale(repo_root, &wt, &br, &mut active_branches);
            }
        }
        if let (Some(wt), Some(br)) = (current_wt, current_branch) {
            prune_worktree_if_stale(repo_root, &wt, &br, &mut active_branches);
        }
    }

    // 2. Prune orphaned worker-* branches that have no registered worktrees AND are merged into HEAD
    if let Ok(output) = git(repo_root, "branch --list", &["branch", "--list", "worker-*"]) {
        let stdout = String::from_utf8_lossy(&output.stdout);
        for line in stdout.lines() {
            let branch = line.trim().trim_start_matches('*').trim_start_matches('+').trim();
            if branch.starts_with("worker-") && !active_branches.iter().any(|b| b == branch) {
                let is_merged = git(
                    repo_root,
                    "merge-base",
                    &["merge-base", "--is-ancestor", branch, "HEAD"],
                )
                .map(|o| o.status.success())
                .unwrap_or(false);

                if is_merged {
                    info!(branch = %branch, "Pruning merged or empty worker branch");
                    let _ = git(repo_root, "branch -D", &["branch", "-D", branch]);
                } else {
                    info!(branch = %branch, "Preserving unmerged worker branch with commits");
                }
            }
        }
    }

    // 3. Prune orphaned swe-wt-* directories and .pid files in base dirs whose processes are dead
    for base in [swe_base_dir(), std::env::temp_dir()] {
        if let Ok(entries) = std::fs::read_dir(&base) {
            for entry in entries.flatten() {
                let p = entry.path();
                if let Some(name) = p.file_name().and_then(|n| n.to_str())
                    && name.starts_with("swe-wt-")
                {
                    if p.is_dir() {
                        let pid_file = format!("{}.pid", p.to_string_lossy());
                        let is_stale = if let Ok(content) = std::fs::read_to_string(&pid_file) {
                            content
                                .trim()
                                .parse::<u32>()
                                .is_ok_and(|pid| !is_process_alive(pid))
                        } else {
                            false
                        };
                        if is_stale {
                            let _ = std::fs::remove_dir_all(&p);
                            let _ = std::fs::remove_file(&pid_file);
                        }
                    } else if name.ends_with(".pid")
                        && let Ok(content) = std::fs::read_to_string(&p)
                    {
                        let is_stale = content
                            .trim()
                            .parse::<u32>()
                            .map_or(true, |pid| !is_process_alive(pid));
                        if is_stale {
                            let _ = std::fs::remove_file(&p);
                        }
                    }
                }
            }
        }

        // 4. Prune orphaned swe-target-* directories in base dirs whose worktrees are gone
        if let Ok(entries) = std::fs::read_dir(&base) {
            for entry in entries.flatten() {
                let p = entry.path();
                if let Some(name) = p.file_name().and_then(|n| n.to_str())
                    && let Some(wt_name) = name.strip_prefix("swe-target-")
                {
                    let wt_path = base.join(wt_name);
                    if !wt_path.exists() {
                        let _ = std::fs::remove_dir_all(&p);
                    }
                }
            }
        }
    }

    let _ = git(repo_root, "worktree prune", &["worktree", "prune"]);
}

impl Drop for WorktreeGuard {
    fn drop(&mut self) {
        if self.keep {
            info!(path = %self.path.display(), "Preserving worktree");
            return;
        }

        // Sync report/audit artifacts to repo root before cleanup
        let _ = self.sync_artifacts();

        info!(path = %self.path.display(), branch = %self.branch, "Cleaning up git worktree");

        let path_str = self.path.to_string_lossy();
        let _ = git(
            &self.repo_root,
            "worktree remove",
            &["worktree", "remove", "--force", &path_str],
        );
        let _ = git(
            &self.repo_root,
            "worktree prune",
            &["worktree", "prune"],
        );
        if self.preserve_branch {
            info!(branch = %self.branch, "Preserving worker branch with committed changes");
        } else {
            let _ = git(
                &self.repo_root,
                "branch -D",
                &["branch", "-D", &self.branch],
            );
        }

        if self.path.exists()
            && let Err(e) = std::fs::remove_dir_all(&self.path)
        {
            error!(error = %e, path = %self.path.display(), "Failed to delete leftover worktree directory");
        }

        let pid_file = format!("{}.pid", self.path.to_string_lossy());
        let _ = std::fs::remove_file(&pid_file);

        let dir_name = self.path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if !dir_name.is_empty() {
            for base in [swe_base_dir(), std::env::temp_dir()] {
                let target_dir = base.join(format!("swe-target-{dir_name}"));
                if target_dir.exists() {
                    let _ = std::fs::remove_dir_all(&target_dir);
                }
            }
        }
    }
}
