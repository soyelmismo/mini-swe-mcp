use mini_swe_mcp::agent::AgentRunner;
fn runner() -> AgentRunner {
    AgentRunner::new("http://127.0.0.1:1".into(), "k".into(), "m".into(), None)
}
#[tokio::test]
async fn zz_bisect_unconfined_curl() {
    let d = std::env::temp_dir().join(format!("zz-bisect-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    let on = runner();
    for cmd in [
        "echo SANDBOX=$SWE_DISABLE_SANDBOX; curl -sS --max-time 3 https://example.com -o /dev/null; echo rc=$?",
        "cat /etc/resolv.conf 2>&1; echo rc=$?",
        "ls -la /etc/resolv.conf 2>&1",
    ] {
        let (out, code) = on.execute_bash(&d, cmd).await.unwrap();
        let tail = out.lines().rev().take(4).collect::<Vec<_>>().join(" | ");
        println!("code={code:?} :: {tail}");
    }
    std::fs::remove_dir_all(&d).ok();
}
