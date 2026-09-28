use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::process::Command;
use tracing::{error, info};

pub struct WorktreeGuard {
    pub path: PathBuf,
    pub branch: String,
    pub repo_root: PathBuf,
    pub keep: bool,
}

impl WorktreeGuard {
    pub fn new(repo_root: &Path, worker_id: &str) -> Result<Self> {
        let branch = format!("worker-{}", worker_id);
        let path = std::env::temp_dir().join(format!("swe-wt-{}", worker_id));

        // Ensure target directory doesn't exist
        if path.exists() {
            let _ = std::fs::remove_dir_all(&path);
        }

        info!(repo = %repo_root.display(), branch = %branch, path = %path.display(), "Creating git worktree");

        let output = Command::new("git")
            .current_dir(repo_root)
            .args(["worktree", "add", "-b", &branch, path.to_str().unwrap(), "HEAD"])
            .output()
            .context("Failed to execute git worktree add")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("git worktree add failed: {}", stderr);
        }

        Ok(Self {
            path,
            branch,
            repo_root: repo_root.to_path_buf(),
            keep: false,
        })
    }

    pub fn get_diff(&self) -> Result<String> {
        // Stage untracked files intent-to-add so git diff captures new files as well
        let _ = Command::new("git")
            .current_dir(&self.path)
            .args(["add", "-N", "."])
            .output();

        let output = Command::new("git")
            .current_dir(&self.path)
            .args(["diff", "HEAD"])
            .output()
            .context("Failed to execute git diff")?;

        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }
}

impl Drop for WorktreeGuard {
    fn drop(&mut self) {
        if self.keep {
            info!(path = %self.path.display(), "Preserving worktree");
            return;
        }

        info!(path = %self.path.display(), branch = %self.branch, "Cleaning up git worktree");

        let _ = Command::new("git")
            .current_dir(&self.repo_root)
            .args(["worktree", "remove", "--force", self.path.to_str().unwrap()])
            .output();

        let _ = Command::new("git")
            .current_dir(&self.repo_root)
            .args(["branch", "-D", &self.branch])
            .output();

        if self.path.exists()
            && let Err(e) = std::fs::remove_dir_all(&self.path) {
                error!(error = %e, path = %self.path.display(), "Failed to delete leftover worktree directory");
            }
    }
}
