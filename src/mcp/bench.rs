//! TEMPORARY measurement harness (deleted before the final commit).
use super::protocol::{
    INITIALIZE_RESULT, JsonRpcResponse, PreSerializedResult, RawText, RequestId, code, parse_frame,
};
use serde_json::{Value, json};
use std::borrow::Cow;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Counting;

static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);
static ON: AtomicUsize = AtomicUsize::new(0);

unsafe impl std::alloc::GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
        if ON.load(Ordering::Relaxed) == 1 {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
            BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        unsafe { std::alloc::System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
        unsafe { std::alloc::System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: std::alloc::Layout, new_size: usize) -> *mut u8 {
        if ON.load(Ordering::Relaxed) == 1 {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
            BYTES.fetch_add(new_size, Ordering::Relaxed);
        }
        unsafe { std::alloc::System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static A: Counting = Counting;

fn measure<T>(label: &str, iters: u32, mut f: impl FnMut() -> T) {
    for _ in 0..5 {
        std::hint::black_box(f());
    }
    ALLOCS.store(0, Ordering::Relaxed);
    BYTES.store(0, Ordering::Relaxed);
    ON.store(1, Ordering::Relaxed);
    let start = std::time::Instant::now();
    for _ in 0..iters {
        std::hint::black_box(f());
    }
    let elapsed = start.elapsed();
    ON.store(0, Ordering::Relaxed);
    println!(
        "{label:<34} allocs/iter {:>5.1}  bytes/iter {:>8.1}  ns/iter {:>9.1}",
        ALLOCS.load(Ordering::Relaxed) as f64 / iters as f64,
        BYTES.load(Ordering::Relaxed) as f64 / iters as f64,
        elapsed.as_nanos() as f64 / iters as f64,
    );
}

fn owned(raw: &str) -> Cow<'static, RequestId<'static>> {
    Cow::Owned(RequestId(Cow::Owned(String::from(raw))))
}

fn payload() -> Value {
    json!({
        "worker_id": "w-1",
        "state": "Completed",
        "steps": 12,
        "logs": [{"op": "bash", "output": "ok", "n": 1}],
        "summary": "did the thing"
    })
}

#[test]
fn bench_protocol() {
    let payload = payload();
    let tools_list = json!({"tools":[{"name":"worker","description":"d","inputSchema":{"type":"object","properties":{"action":{"type":"string","enum":["a","b"]}},"required":["action"]}}]});
    let ping = JsonRpcResponse::ok(owned("7"), json!({}));
    let ping_frame = ping.to_frame().unwrap();
    let method_not_found =
        parse_frame(r#"{"jsonrpc":"2.0","id":3,"method":"does/not/exist"}"#).expect("frame");

    println!("--- frame bytes ---");
    println!("ping           : {}", ping_frame.trim_end());
    println!(
        "method not found: {}",
        JsonRpcResponse::method_not_found(method_not_found.id_or_null(), method_not_found.method)
            .to_frame()
            .unwrap()
            .trim_end()
    );
    println!(
        "initialize      : {}",
        JsonRpcResponse::ok(owned("1"), (*INITIALIZE_RESULT).clone())
            .to_frame()
            .unwrap()
            .trim_end()
    );
    println!(
        "tools/call      : {}",
        JsonRpcResponse::ok(
            owned("1"),
            serde_json::to_value(RawText(PreSerializedResult::text(payload.clone()))).unwrap()
        )
        .to_frame()
        .unwrap()
    );

    println!("--- bench ---");
    measure("parse ping frame", 20000, || {
        parse_frame(r#"{"jsonrpc":"2.0","id":7,"method":"ping"}"#).expect("parse")
    });
    measure("parse tools/call frame", 20000, || {
        parse_frame(
            r#"{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"worker","arguments":{"action":"reap"}}}"#,
        )
        .expect("parse")
    });
    measure("ping -> to_frame", 20000, || ping.to_frame().unwrap());
    measure("initialize -> to_frame", 20000, || {
        JsonRpcResponse::ok(owned("1"), (*INITIALIZE_RESULT).clone()).to_frame().unwrap()
    });
    measure("tools/list clone+to_frame", 20000, || {
        JsonRpcResponse::ok(owned("7"), tools_list.clone()).to_frame().unwrap()
    });
    measure("tools/call -> to_frame", 20000, || {
        JsonRpcResponse::ok(
            owned("7"),
            serde_json::to_value(RawText(PreSerializedResult::text(payload.clone()))).unwrap(),
        )
        .to_frame()
        .unwrap()
    });
    measure("err method-not-found", 20000, || {
        let req = parse_frame(r#"{"jsonrpc":"2.0","id":3,"method":"does/not/exist"}"#).unwrap();
        JsonRpcResponse::method_not_found(req.id_or_null(), req.method)
            .to_frame()
            .unwrap()
    });
    measure("err tool (-32000)", 20000, || {
        JsonRpcResponse::err(
            owned("3"),
            code::SERVER_ERROR,
            Cow::Owned("'worker_id' is required for action 'logs'".to_string()),
        )
        .to_frame()
        .unwrap()
    });
    measure("err parse (-32700)", 20000, || {
        use super::protocol::FrameRejection;
        FrameRejection::malformed(
            &serde_json::from_str::<Value>(r#"{"jsonrpc": "2.0", "id": 7, "method": "#).unwrap_err(),
        )
        .into_frame()
    });
    measure("err invalid request (-32600)", 20000, || {
        parse_frame(r#"{"id":1}"#).unwrap_err().into_frame()
    });
}
