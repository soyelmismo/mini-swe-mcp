//! Confirm clone3 is the killer and see who calls it.
use std::sync::atomic::{AtomicI32, Ordering};
static TRAPPED: AtomicI32 = AtomicI32::new(-1);
extern "C" fn on_sigsys(_s: libc::c_int, info: *mut libc::siginfo_t, _u: *mut libc::c_void) {
    let nr = unsafe { *(info as *const libc::c_int).add(3) };
    TRAPPED.store(nr, Ordering::SeqCst);
    unsafe { libc::_exit(200) };
}
#[test]
fn zz_clone3_trap() {
    unsafe {
        let mut sa: libc::sigaction = std::mem::zeroed();
        sa.sa_sigaction = on_sigsys as usize;
        sa.sa_flags = libc::SA_SIGINFO;
        libc::sigemptyset(&mut sa.sa_mask);
        libc::sigaction(libc::SIGSYS, &sa, std::ptr::null_mut());
    }
    // Install a filter that TRAPs clone3 (delivers SIGSYS to handler with nr).
    let action_name = std::env::var("ZZ_ACTION").unwrap_or("enosys".into());
    let action = match action_name.as_str() {
        "eperm" => libc::SECCOMP_RET_ERRNO | (libc::EPERM as u32),
        "enosys" => libc::SECCOMP_RET_ERRNO | (libc::ENOSYS as u32),
        _ => panic!("bad action"),
    };
    let prog = mini_swe_mcp::agent::sandbox::SeccompFilter::build_action_clone3_for_test(action);
    unsafe { prog.apply().unwrap(); }
    eprintln!("filter installed action={action_name}");
    let st = std::process::Command::new("true").status();
    println!("status={st:?} trapped={}", TRAPPED.load(Ordering::SeqCst));
}
