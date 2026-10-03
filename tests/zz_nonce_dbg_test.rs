mod common;
#[test]
fn dbg_nonce() {
    let d = common::TempDir::new_in_tmp("nonce-dbg");
    let root = mini_swe_mcp::worktree::ScratchRoot::new(d.path());
    let first = mini_swe_mcp::pool::__test_log_nonce_in(&root);
    use std::os::unix::fs::MetadataExt;
    let meta = std::fs::metadata(d.path().join(".steer-log-nonce")).unwrap();
    eprintln!("first={first:?} mode={:o}", meta.mode() & 0o7777);
    let second = mini_swe_mcp::pool::__test_log_nonce_in(&root);
    eprintln!("second={second:?} same={}", first == second);
}
