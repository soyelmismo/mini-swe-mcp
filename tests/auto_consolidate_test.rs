//! Automatic round consolidation uses the dispatch contract for CLI and MCP.
mod common;

use mini_swe_mcp::cli::args::tool_args;
use serde_json::json;

#[test]
fn dispatch_accepts_auto_consolidation_flags() {
    for (flag, expected) in [("--consolidate", json!(true)), ("--consolidate=nerd", json!("nerd"))] {
        let argv = ["mini-swe-mcp", "dispatch", "task", "--group", "round", flag,
            "--consolidate-verify", "cargo test"]
            .map(str::to_string);
        let args = tool_args("dispatch", &argv, true).unwrap().unwrap();
        assert_eq!(args.get("consolidate"), Some(&expected));
        assert_eq!(args.get("consolidate_verify"), Some(&json!("cargo test")));
    }
}
