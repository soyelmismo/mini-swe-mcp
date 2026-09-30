use mini_swe_mcp::worktree::swe_base_dir;
use std::process::{Command, Stdio};

#[test]
fn probe_helper() {
    let dir = swe_base_dir().join("swe-probe-h5");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let pid_file = dir.join("child.pid");
    let out = Command::new("bash")
        .args(["-c", &format!("setsid bash -c 'sleep 300 & echo $! > {}; wait' _helper &", pid_file.display())])
        .current_dir(&dir)
        .stdin(Stdio::null())
        .output()
        .unwrap();
    println!("helper status={:?}", out.status.code());
    std::thread::sleep(std::time::Duration::from_millis(500));
    let text = std::fs::read_to_string(&pid_file).unwrap_or_default();
    let child: u32 = text.trim().parse().unwrap_or(0);
    println!("child={child}");
    let mut pid = child;
    for _ in 0..10 {
        let status = std::fs::read_to_string(format!("/proc/{pid}/status")).unwrap_or_default();
        let ppid = status.lines().find(|l| l.starts_with("PPid:")).and_then(|l| l.split_whitespace().nth(1)).unwrap_or("0").to_string();
        let comm = std::fs::read_to_string(format!("/proc/{pid}/comm")).unwrap_or_default();
        println!("  pid={pid} comm={} ppid={ppid}", comm.trim());
        let ppid: u32 = ppid.parse().unwrap_or(0);
        if ppid == 0 || ppid == pid { break; }
        pid = ppid;
    }
    let _ = std::fs::remove_dir_all(&dir);
}
