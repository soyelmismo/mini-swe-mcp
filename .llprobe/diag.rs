use std::os::raw::{c_char, c_int, c_long};
extern "C" {
    fn syscall(num: c_long, ...) -> c_long;
    fn open(path: *const c_char, flags: c_int, ...) -> c_int;
}
#[repr(C)]
struct Attr { handled: u64 }
fn main() {
    let attr = Attr { handled: (1<<0)|(1<<2)|(1<<3) };
    let sz = std::mem::size_of::<Attr>() as c_long;
    for (label, size, flags) in [("size+noflag", sz, 0), ("size+version", sz, 1), ("zero+version", 0, 1), ("zero+noflag", 0, 0)] {
        let fd = unsafe { syscall(444, &attr as *const Attr as *const u8, size, flags) };
        println!("{label}: fd={fd} errno={:?}", std::io::Error::last_os_error());
        if fd >= 0 { unsafe { libc_close(fd as c_int) }; }
    }
    // probe ABI
    let abi = unsafe { syscall(444, 0 as *const u8, 0, 1) };
    println!("abi query = {abi} errno={:?}", std::io::Error::last_os_error());
    let _ = open;
}
extern "C" { fn libc_close(fd: c_int) -> c_int; }
