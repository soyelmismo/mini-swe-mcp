fn main() {
    let p = mini_swe_mcp::pool::WorkerPool::new(4, "x".into(), "y".into());
    println!("WorkerPool::new root = {}", p.scratch_root().path().display());
    println!("from_env root       = {}", mini_swe_mcp::worktree::ScratchRoot::from_env().path().display());
    println!("swe_base_dir        = {}", mini_swe_mcp::worktree::swe_base_dir().display());
}
