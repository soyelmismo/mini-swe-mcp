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

fn git(dir: &Path, operation: &str, args: &[&str]) -> Result<std::process::Output> {
    Command::new("git")
        .current_dir(dir)
        .args(args)
        .output()
        .with_context(|| format!("Failed to execute git {operation}"))
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

        Ok(Self {
            path,
            branch,
            repo_root: repo_root.to_path_buf(),
            keep: false,
        })
    }

    pub fn get_diff(&self) -> Result<String> {
        // Stage untracked files intent-to-add so git diff captures new files as well
        let _ = git(&self.path, "add", &["add", "-N", "."]);

        let output = git(&self.path, "diff", &["diff", "HEAD"])?;

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

        let path_str = self.path.to_string_lossy();
        let _ = git(
            &self.repo_root,
            "worktree remove",
            &["worktree", "remove", "--force", &path_str],
        );
        let _ = git(
            &self.repo_root,
            "branch -D",
            &["branch", "-D", &self.branch],
        );

        if self.path.exists()
            && let Err(e) = std::fs::remove_dir_all(&self.path)
        {
            error!(error = %e, path = %self.path.display(), "Failed to delete leftover worktree directory");
        }
    }
}
