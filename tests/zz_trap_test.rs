//! Probe: install the real kernel confinement, trap SIGSYS, run curl DNS.
use std::sync::atomic::{AtomicI32, Ordering};
static TRAPPED_NR: AtomicI32 = AtomicI32::new(-1);
extern "C" fn on_sigsys(_sig: libc::c_int, info: *mut libc::siginfo_t, _u: *mut libc::c_void) {
    let nr = unsafe { *(info as *const libc::c_int).add(3) };
    TRAPPED_NR.store(nr, Ordering::SeqCst);
    unsafe { libc::_exit(200) };
}
#[test]
fn zz_trap_which_syscall() {
    if std::env::var_os("ZZ_TRAP_CHILD").is_none() {
        let exe = std::env::current_exe().unwrap();
        let out = std::process::Command::new(exe)
            .arg("--exact").arg("zz_trap_which_syscall").arg("--nocapture")
            .env("ZZ_TRAP_CHILD", "1").output().unwrap();
        println!("CHILD STATUS: {}", out.status);
        println!("CHILD OUT: {}", String::from_utf8_lossy(&out.stdout));
        println!("CHILD ERR: {}", String::from_utf8_lossy(&out.stderr));
        return;
    }
    // Child = confined worker: prepare dirs, build confinement like exec.rs.
    let base = std::env::temp_dir().join(format!("zz-trap-{}", std::process::id()));
    let wt = base.join("wt"); let tg = base.join("tg");
    std::fs::create_dir_all(&wt).unwrap(); std::fs::create_dir_all(&tg).unwrap();
    unsafe {
        let mut sa: libc::sigaction = std::mem::zeroed();
        sa.sa_sigaction = on_sigsys as usize;
        sa.sa_flags = libc::SA_SIGINFO;
        libc::sigemptyset(&mut sa.sa_mask);
        libc::sigaction(libc::SIGSYS, &sa, std::ptr::null_mut());
    }
    let conf = mini_swe_mcp::agent::sandbox::KernelConfinement::prepare(&wt, &tg, true).unwrap().unwrap();
    unsafe { conf.apply_for_test().unwrap(); }
    println!("confinement installed, running curl...");
    let st = std::process::Command::new("curl")
        .args(["-sS","--max-time","3","https://example.com","-o","/dev/null"])
        .status().unwrap();
    use std::os::unix::process::ExitStatusExt;
    println!("curl signal={:?} code={:?} trapped_nr={}", st.signal(), st.code(), TRAPPED_NR.load(Ordering::SeqCst));
    std::fs::remove_dir_all(&base).ok();
}
