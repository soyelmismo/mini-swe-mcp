//! Temporary probe: trace where an em dash is lost.
use mini_swe_mcp::cli::watch::{render_round_line, render_with};
use mini_swe_mcp::pool::{clamp_string, parse_report};
use serde_json::json;

const DONE: &str = "Fixed the last gate failure — `cargo fmt --check` — ñ 日本 🙂";

#[test]
fn probe_parse_report_keeps_utf8() {
    let msg = format!("REPORT\ndone: {DONE}\nfiles: a.rs\n");
    let report = parse_report(&msg).expect("parses");
    println!("report.done = {:?}", report.done);
    assert_eq!(report.done, DONE);
}

#[test]
fn probe_clamp_keeps_utf8() {
    assert_eq!(clamp_string(DONE, 200), DONE);
}

#[test]
fn probe_round_line_keeps_utf8() {
    let w = json!({"worker_id":"w1","outcome":"completed","verified":true,"done":DONE,"question":null,"branch":"b","time_since_last_step":0});
    let line = render_round_line(&w);
    println!("round line = {line:?}");
    assert!(line.contains(DONE), "{line}");
}

#[test]
fn probe_compact_and_verbose_keep_utf8() {
    let v = json!({
        "worker_id":"w1","event":"completed","model":"m","owner":"o","group":"g","branch":"b",
        "revision":0,"step":1,"max_turns":3,"elapsed":1,"verified":true,
        "diff_stat":{"files":1,"insertions":2,"deletions":0},"per_file":[],
        "report":{"done":DONE,"files":"a.rs","tests":"t","risks":"none"},
        "task":"t","last_ops":[],"status":"completed","moved_counters":[]
    });
    for verbose in [false, true] {
        let out = render_with(&v, verbose);
        println!("verbose={verbose}: {out:?}");
        assert!(out.contains(DONE), "lost in verbose={verbose}: {out}");
    }
}
