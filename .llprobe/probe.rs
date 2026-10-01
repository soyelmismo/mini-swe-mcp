use std::ffi::CString;
use std::os::raw::{c_char, c_int, c_long};
extern "C" {
    fn syscall(num: c_long, ...) -> c_long;
    fn open(path: *const c_char, flags: c_int, ...) -> c_int;
    fn close(fd: c_int) -> c_int;
    fn read(fd: c_int, buf: *mut u8, n: usize) -> isize;
    fn prctl(option: c_int, ...) -> c_int;
}
const SYS_CREATE: c_long = 444;
const SYS_ADD: c_long = 445;
const SYS_RESTRICT: c_long = 446;
const CREATE_VERSION: u32 = 1;
const RULE_PATH_BENEATH: u32 = 1;
const PR_SET_NO_NEW_PRIVS: c_int = 38;
const EXEC: u64 = 1 << 0;
const READ_FILE: u64 = 1 << 2;
const READ_DIR: u64 = 1 << 3;
#[repr(C)]
struct Attr { handled: u64 }
#[repr(C)]
struct Rule { allowed: u64, parent_fd: c_int }
fn try_read(path: &str) -> bool {
    let c = CString::new(path).unwrap();
    let fd = unsafe { open(c.as_ptr(), 0) };
    if fd < 0 { return false; }
    let mut buf = [0u8; 64];
    let n = unsafe { read(fd, buf.as_mut_ptr(), buf.len()) };
    unsafe { close(fd) };
    n > 0
}
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let grant = &args[1];
    let inside = &args[2];
    let outside = &args[3];
    let attr = Attr { handled: EXEC | READ_FILE | READ_DIR };
    let fd = unsafe { syscall(SYS_CREATE, &attr as *const Attr as *const u8, std::mem::size_of::<Attr>() as c_long, CREATE_VERSION as c_long) };
    if fd < 0 { println!("create failed errno={:?}", std::io::Error::last_os_error()); std::process::exit(3); }
    let c = CString::new(grant.as_str()).unwrap();
    let parent = unsafe { open(c.as_ptr(), 0o1000000 | 0o2000000) };
    assert!(parent >= 0, "open grant dir failed");
    let rule = Rule { allowed: EXEC | READ_FILE | READ_DIR, parent_fd: parent };
    let ret = unsafe { syscall(SYS_ADD, fd, RULE_PATH_BENEATH as c_long, &rule as *const Rule as *const u8, 0) };
    assert!(ret == 0, "add rule failed");
    unsafe { close(parent) };
    assert_eq!(unsafe { prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) }, 0);
    let ret = unsafe { syscall(SYS_RESTRICT, fd, 0, 0, 0) };
    assert!(ret == 0, "restrict failed");
    println!("inside={} outside={}", try_read(inside), try_read(outside));
}
