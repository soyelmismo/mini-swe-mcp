//! JSON-RPC 2.0 wire types for the MCP stdio transport.
//!
//! Everything that reaches the wire is produced here. Requests are parsed with
//! a serde-derived struct (the `id` stays a [`RawValue`] so it is echoed byte
//! for byte), and responses are serialized into a single newline-terminated
//! frame buffer.

use std::borrow::Cow;
use std::sync::LazyLock;

use serde::Deserialize;
use serde::Deserializer as _;
use serde::de::IgnoredAny;
use serde::Serialize;
use serde_json::{Value, json};
use serde_json::value::RawValue;

/// The `jsonrpc` member of every frame, per JSON-RPC 2.0 §6.
pub(super) const JSONRPC_VERSION: &str = "2.0";

/// Reserved `error.code` values (JSON-RPC 2.0 §5.1) plus the
/// implementation-defined code tool failures are reported under.
pub(super) mod code {
    /// Invalid JSON was received.
    pub(crate) const PARSE_ERROR: i64 = -32700;
    /// Valid JSON but not a valid Request object.
    pub(crate) const INVALID_REQUEST: i64 = -32600;
    /// The requested method does not exist or is unavailable.
    pub(crate) const METHOD_NOT_FOUND: i64 = -32601;
    /// Implementation-defined server error (JSON-RPC reserves -32000..-32099).
    pub(crate) const SERVER_ERROR: i64 = -32000;
}

/// Output buffer covering every frame the server emits except large tool
/// payloads: `ping`, `initialize`, progress notifications and error frames fit
/// within it, so they cost exactly one allocation.
const FRAME_CAPACITY: usize = 256;

/// Largest inbound JSON-RPC frame [`parse_frame`] will look at, in bytes.
///
/// A stdio MCP peer is a program on the same host, but the transport is a pipe
/// an operator can point at anything: a frame is the one value that grows with
/// the sender's appetite. Refusing the line up front is cheaper than any bound
/// placed later — the parse never runs and the rejection frame is a fixed
/// 256-byte buffer.
///
/// The ceiling sits well above the largest frame this server legitimately
/// receives: [`crate::mcp::schema::build_tools_list`] describes one tool with
/// an enum of ten actions, and `tools/call` carries a worker's arguments; both
/// are kilobytes at most, and the agent side already caps its own payloads far
/// lower ([`crate::agent::MAX_TOOL_ARGUMENT_BYTES`], 64 KiB). A megabyte leaves
/// two orders of magnitude of headroom while bounding a single frame's memory.
pub(super) const MAX_FRAME_BYTES: usize = 1024 * 1024;

/// Frame emitted when an outgoing frame fails to serialize; `id` is `null`,
/// which JSON-RPC 2.0 §5 permits when the id cannot be determined.
pub(super) const INTERNAL_ERROR_FRAME: &str =
    "{\"jsonrpc\":\"2.0\",\"id\":null,\"error\":{\"code\":-32603,\"message\":\"Internal error\"}}\n";

/// Diagnosis for a frame that carries no usable `method`.
const NO_METHOD: &str = "Invalid Request: missing or non-string \"method\" member";

/// Diagnosis for a line refused on length alone, quoting the ceiling as a byte
/// count so an operator can size a client against it.
///
/// The figure is written out rather than derived: `stringify!` would emit the
/// constant's *name*, not its value. A test below fails if the message and
/// [`MAX_FRAME_BYTES`] ever disagree.
const FRAME_TOO_LARGE: &str = "Invalid Request: frame exceeds the 1048576-byte limit";

// ----------
// Incoming frames
// ----------

/// An incoming JSON-RPC request; `id: None` marks a notification.
///
/// `id` is kept as a [`RawValue`] so it is echoed byte for byte (JSON-RPC 2.0
/// §4 — the server may not round `9007199254740993` or re-quote a string id).
#[derive(Debug, Deserialize)]
pub(super) struct JsonRpcRequest {
    /// The method being invoked.
    pub(super) method: String,
    /// The request id, the client's own JSON text. `None` (absent or `null`)
    /// marks a notification, which the server must not answer.
    pub(super) id: Option<Box<RawValue>>,
    /// Method parameters, materialized only on the `tools/call` path.
    pub(super) params: Option<Value>,
}

impl JsonRpcRequest {
    /// The id to echo back, or `None` — serialized as `null`, which JSON-RPC
    /// 2.0 §5 asks for when the real one cannot be determined.
    pub(super) fn id_or_null(&self) -> Option<&RawValue> {
        self.id.as_deref()
    }
}

/// Why a frame could not be turned into a request.
#[derive(Debug)]
pub(super) enum FrameRejection {
    /// The line is not valid JSON (`-32700`). The message carries serde's own
    /// diagnosis — line, column and cause — the only thing an operator can act
    /// on.
    Malformed(String),
    /// Valid JSON but not a JSON-RPC Request object (`-32600`): not an object,
    /// or no string `method`.
    InvalidRequest(&'static str),
    /// The line is longer than [`MAX_FRAME_BYTES`] (`-32600`), rejected before
    /// it is parsed.
    FrameTooLarge,
}

impl FrameRejection {
    /// The `-32700` reply for a line `serde_json` refused to parse, carrying
    /// serde's own diagnosis. Well-formed JSON that is not a Request object is
    /// `-32600` instead (JSON-RPC 2.0 §5.1); [`parse_frame`] tells the two
    /// apart with a syntax-only first pass.
    pub(super) fn malformed(error: &serde_json::Error) -> Self {
        Self::Malformed(format!("Parse error: {error}"))
    }

    /// Renders the rejection as a complete stdio frame (`…\n`).
    pub(super) fn into_frame(self) -> String {
        let (code, message) = match self {
            Self::Malformed(message) => (code::PARSE_ERROR, Cow::Owned(message)),
            Self::InvalidRequest(message) => (code::INVALID_REQUEST, Cow::Borrowed(message)),
            // Rejected before it was parsed, so the honest code is `-32600`:
            // nothing about the line's syntax was ever decided, and a `-32700`
            // would claim the bytes were not JSON when they may well be.
            Self::FrameTooLarge => (code::INVALID_REQUEST, Cow::Borrowed(FRAME_TOO_LARGE)),
        };
        JsonRpcResponse::err(None, code, message)
            .to_frame()
            .unwrap_or_else(|_| String::from(INTERNAL_ERROR_FRAME))
    }
}

/// Parse one newline-delimited stdio frame into a request.
///
/// Unknown members are ignored rather than rejected (MCP peers legitimately add
/// their own) and a UTF-8 BOM is tolerated, but everything JSON-RPC 2.0
/// requires is enforced: the frame must be an object and must carry a string
/// `method`. `params` and `id` may be absent, `null`, or any JSON value.
pub(super) fn parse_frame(frame: &str) -> Result<JsonRpcRequest, FrameRejection> {
    // Checked first, on the raw line and before the BOM strip, so the bound
    // covers every byte the peer sent and an oversized frame never reaches a
    // parser.
    if frame.len() > MAX_FRAME_BYTES {
        return Err(FrameRejection::FrameTooLarge);
    }
    let frame = frame.strip_prefix('\u{feff}').unwrap_or(frame);

    // Pass 1 — is this JSON at all? `IgnoredAny` walks the frame without
    // materializing a single value and fails on exactly the inputs that are not
    // valid JSON, which is what `-32700` is reserved for (JSON-RPC 2.0 §5.1).
    //
    // This pass is what lets the two reserved codes be told apart honestly.
    // Without it the two cases are indistinguishable in a single pass: serde
    // reports both "this is not JSON" and "this JSON is not a Request object"
    // through the same deserializer, and `classify()` only splits them for some
    // shapes (`{"id":1` is a truncation, `{"id":1}` is a semantic complaint).
    let mut syntax = serde_json::Deserializer::from_str(frame);
    syntax
        .deserialize_ignored_any(IgnoredAny)
        .map_err(|error| FrameRejection::malformed(&error))?;
    syntax
        .end()
        .map_err(|error| FrameRejection::malformed(&error))?;

    // Pass 2 — build the request. Pass 2 can only fail where pass 1 already
    // proved the frame is JSON, and it can only do so because the JSON is not a
    // Request object.
    let mut request = serde_json::Deserializer::from_str(frame);
    let parsed = JsonRpcRequest::deserialize(&mut request)
        .map_err(|_| FrameRejection::InvalidRequest(NO_METHOD))?;
    request
        .end()
        .map_err(|_| FrameRejection::InvalidRequest(NO_METHOD))?;
    Ok(parsed)
}

// ----------
// Outgoing frames
// ----------

/// An outgoing JSON-RPC 2.0 response envelope.
///
/// Exactly one of `result`/`error` is serialized (see [`Body`]), so the two
/// mutually exclusive `Option`s a hand-rolled envelope struct would need cannot
/// produce a frame that is neither, or both — which the spec forbids.
#[derive(Debug)]
pub(super) struct JsonRpcResponse {
    /// The id to echo, in the encoding the client used; `None` serializes as
    /// `null`, which JSON-RPC 2.0 §5 permits when the id cannot be determined.
    id: Option<Box<RawValue>>,
    body: Body,
}

/// The mutually exclusive payload of a JSON-RPC response.
#[derive(Debug)]
enum Body {
    /// A successful `result` the caller built as a value.
    Result(Value),
    /// An `error` object: a reserved `code` plus a message.
    Error {
        code: i64,
        message: Cow<'static, str>,
    },
}

impl JsonRpcResponse {
    /// Successful JSON-RPC 2.0 response whose `result` the caller owns.
    pub(super) fn ok(id: Option<Box<RawValue>>, result: Value) -> Self {
        Self {
            id,
            body: Body::Result(result),
        }
    }

    /// Successful `tools/call` response carrying a tool payload.
    ///
    /// The payload is pretty-printed into the `text` of a single content block,
    /// exactly the shape MCP clients expect.
    pub(super) fn tool_call(id: Option<Box<RawValue>>, payload: Value) -> Self {
        let text = serde_json::to_string_pretty(&payload)
            .unwrap_or_else(|_| String::from("null"));
        Self {
            id,
            body: Body::Result(json!({
                "content": [{ "type": "text", "text": text }]
            })),
        }
    }

    /// JSON-RPC 2.0 error response carrying a reserved `code` and a message.
    pub(super) fn err(id: Option<Box<RawValue>>, code: i64, message: Cow<'static, str>) -> Self {
        Self {
            id,
            body: Body::Error { code, message },
        }
    }

    /// `-32601 Method not found` for a method the server does not expose; the
    /// rejected name is quoted.
    pub(super) fn method_not_found(id: Option<Box<RawValue>>, method: Cow<'static, str>) -> Self {
        Self::err(
            id,
            code::METHOD_NOT_FOUND,
            Cow::Owned(format!("Method not found: {method}")),
        )
    }

    /// Serialize the envelope into a single frame buffer, newline included.
    pub(super) fn to_frame(&self) -> Result<String, serde_json::Error> {
        let mut frame = Vec::with_capacity(FRAME_CAPACITY);
        let mut serializer = serde_json::Serializer::new(&mut frame);
        self.serialize(&mut serializer)?;
        frame.push(b'\n');
        Ok(String::from_utf8(frame).expect("JSON-RPC frames are UTF-8"))
    }
}

/// The `error` member of a response: a reserved `code` plus a message.
struct ErrorPayload {
    code: i64,
    message: String,
}

impl Serialize for ErrorPayload {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut error = serializer.serialize_struct("Error", 2)?;
        error.serialize_field("code", &self.code)?;
        error.serialize_field("message", &self.message)?;
        error.end()
    }
}

impl Serialize for JsonRpcResponse {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut frame = serializer.serialize_struct("JsonRpcResponse", 3)?;
        frame.serialize_field("jsonrpc", JSONRPC_VERSION)?;
        frame.serialize_field("id", &self.id)?;
        match &self.body {
            Body::Result(result) => frame.serialize_field("result", result)?,
            Body::Error { code, message } => {
                frame.serialize_field(
                    "error",
                    &ErrorPayload {
                        code: *code,
                        message: message.to_string(),
                    },
                )?;
            }
        }
        frame.end()
    }
}

// MCP envelopes built once per process
// ----------

/// The `initialize` result, materialized once per process.
///
/// A handshake is rare, so this trades a permanent `Value` for a response path
/// with no `json!` tree to build and no per-call map churn. `LazyLock` is the
/// std spelling of "once per process"; `src/mcp/server.rs` clones the document
/// out of it when a handshake arrives.
pub(super) static INITIALIZE_RESULT: LazyLock<Value> = LazyLock::new(|| {
    let wire = concat!(
        r#"{"protocolVersion":"2024-11-05","#,
        r#""capabilities":{"tools":{"listChanged":false}},"#,
        r#""serverInfo":{"name":"mini-swe-mcp","version":""#,
        env!("CARGO_PKG_VERSION"),
        r#""}}"#,
    );
    serde_json::from_str::<Value>(wire).expect("the initialize result must be valid JSON")
});

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Unknown members are ignored rather than rejected: MCP peers add their own.
    #[test]
    fn unknown_members_are_tolerated() {
        let req = parse_frame(r#"{"jsonrpc":"2.0","id":1,"method":"ping","extra":{"a":1}}"#)
            .expect("unknown members must not fail the frame");
        assert_eq!(req.method, "ping");
    }

    /// A UTF-8 BOM (some hosts prepend one) is not part of the JSON text.
    #[test]
    fn a_leading_byte_order_mark_is_stripped() {
        let req = parse_frame("\u{feff}{\"id\":1,\"method\":\"ping\"}").expect("BOM + frame");
        assert_eq!(req.method, "ping");
    }

    /// Notifications are requests without a usable `id`: absent or `null`.
    #[test]
    fn notifications_have_no_id() {
        for line in [
            r#"{"jsonrpc":"2.0","method":"ping"}"#,
            r#"{"id":null,"method":"ping"}"#,
        ] {
            let req = parse_frame(line).expect("notification");
            assert!(req.id.is_none(), "`{line}` must be a notification");
        }
        assert!(
            parse_frame(r#"{"id":0,"method":"ping"}"#)
                .expect("zero is an id")
                .id
                .is_some(),
            "id 0 is a request, not a notification"
        );
    }

    /// JSON-RPC 2.0 §5.1 splits the two rejections apart: `-32700` for input
    /// that is not JSON, `-32600` for JSON that is not a Request object.
    #[test]
    fn rejections_use_the_codes_the_spec_reserves() {
        for (line, expected) in [
            (r#"{"jsonrpc": "2.0", "id": 7, "method": "#, code::PARSE_ERROR),
            (r#"{"id":1"#, code::PARSE_ERROR),
            (r#"{"method":"ping"} trailing"#, code::PARSE_ERROR),
            (r#"{bad}"#, code::PARSE_ERROR),
            (r#"{"id":1}"#, code::INVALID_REQUEST),
            (r#"{"method":7}"#, code::INVALID_REQUEST),
            ("42", code::INVALID_REQUEST),
            ("[1,2]", code::INVALID_REQUEST),
        ] {
            let frame = parse_frame(line).expect_err("must be rejected").into_frame();
            let value: Value = serde_json::from_str(&frame).expect("frame is JSON");
            assert_eq!(value["jsonrpc"], json!(JSONRPC_VERSION));
            assert!(
                frame.contains(r#""id":null,"#),
                "an unparsed id must report null: {frame}"
            );
            assert_eq!(value["error"]["code"], json!(expected), "for {line}");
            assert!(value.get("result").is_none(), "errors carry no result");
            assert!(
                !value["error"]["message"].as_str().unwrap_or_default().is_empty(),
                "the message must stay diagnosable"
            );
        }

        // The `-32600` wording is a constant, so it stays byte-identical.
        let expected = format!(
            "{{\"jsonrpc\":\"2.0\",\"id\":null,\"error\":{{\"code\":{},\"message\":\"{}\"}}}}\n",
            code::INVALID_REQUEST,
            NO_METHOD.replace('"', "\\\"")
        );
        assert_eq!(
            parse_frame(r#"{"id":1}"#).expect_err("no method").into_frame(),
            expected
        );
    }

    /// The ceiling is a real bound, not decoration: a frame one byte past it is
    /// refused before it is parsed, and the refusal is the spec-shaped
    /// `-32600` with an `id` of `null` (nothing was read, so nothing to echo).
    #[test]
    fn oversized_frames_are_refused_before_they_are_parsed() {
        // A syntactically perfect request that is simply too large: if the
        // guard ran after parsing, this would succeed.
        let padding = "x".repeat(MAX_FRAME_BYTES);
        let oversize = format!(r#"{{"id":1,"method":"ping","params":"{padding}"}}"#);
        assert!(
            oversize.len() > MAX_FRAME_BYTES,
            "the fixture must exceed the bound to mean anything"
        );

        let frame = parse_frame(&oversize)
            .expect_err("an oversized frame must be refused")
            .into_frame();
        let value: Value = serde_json::from_str(&frame).expect("frame is JSON");
        assert_eq!(value["jsonrpc"], json!(JSONRPC_VERSION));
        assert_eq!(value["id"], json!(null), "nothing was read to echo");
        assert_eq!(value["error"]["code"], json!(code::INVALID_REQUEST));
        assert_eq!(value["error"]["message"], json!(FRAME_TOO_LARGE));
    }

    /// The bound is inclusive: a frame of exactly [`MAX_FRAME_BYTES`] bytes is
    /// served, so a client that sizes itself to the advertised limit is never
    /// rejected for being one byte under.
    #[test]
    fn the_limit_is_inclusive_and_large_frames_still_parse() {
        // `{"id":1,"method":"ping","params":"<pad>"}` — pad to land on the bound.
        let envelope = r#"{"id":1,"method":"ping","params":""#;
        let tail = r#""}"#;
        let pad = MAX_FRAME_BYTES - envelope.len() - tail.len();
        let line = format!("{envelope}{}{tail}", "x".repeat(pad));
        assert_eq!(line.len(), MAX_FRAME_BYTES, "the fixture must sit on the bound");

        let req = parse_frame(&line).expect("a frame at the bound is served");
        assert_eq!(req.method, "ping");
        assert_eq!(req.id.unwrap().get(), "1");
    }

    /// The message quotes the bound it enforces. The two are written separately
    /// (`stringify!` cannot build a value from a constant), so this is what
    /// keeps the wire text from drifting away from the limit.
    #[test]
    fn the_diagnostic_quotes_the_bound_it_enforces() {
        assert_eq!(
            MAX_FRAME_BYTES,
            1024 * 1024,
            "the message text is written against this figure"
        );
        assert!(
            FRAME_TOO_LARGE.contains(&MAX_FRAME_BYTES.to_string()),
            "`{FRAME_TOO_LARGE}` must quote {} so a client can size itself",
            MAX_FRAME_BYTES
        );
    }

    /// The `ok` / `err` constructors emit spec-shaped JSON-RPC 2.0 envelopes.
    #[test]
    fn json_rpc_constructors_are_spec_shaped() {
        let ok = JsonRpcResponse::ok(None, json!({ "a": 1 })).to_frame();
        assert_eq!(ok.expect("frame"), "{\"jsonrpc\":\"2.0\",\"id\":null,\"result\":{\"a\":1}}\n");

        // A value the server built itself, serialized as raw JSON text.
        let req = parse_frame(r#"{"id":"server-1","method":"ping"}"#).expect("parse");
        let err = JsonRpcResponse::err(
            req.id_or_null().map(|v| v.to_owned()),
            code::METHOD_NOT_FOUND,
            Cow::Borrowed("nope"),
        )
            .to_frame()
            .expect("frame");
        let value: Value = serde_json::from_str(&err).expect("frame is JSON");
        assert_eq!(value["jsonrpc"], json!(JSONRPC_VERSION));
        assert_eq!(value["id"], json!("server-1"));
        assert_eq!(
            value["error"],
            json!({ "code": -32601, "message": "nope" })
        );
        assert!(value.get("result").is_none());
    }

    /// `-32601` quotes the method it rejected.
    #[test]
    fn method_not_found_quotes_the_requested_method() {
        let line = r#"{"id":3,"method":"does/not/exist"}"#;
        let req = parse_frame(line).expect("parse");
        let response = JsonRpcResponse::method_not_found(req.id_or_null().map(|v| v.to_owned()), Cow::Owned(req.method));
        assert_eq!(
            response.to_frame().expect("frame"),
            concat!(
                r#"{"jsonrpc":"2.0","id":3,"error":{"code":-32601,"#,
                r#""message":"Method not found: does/not/exist"}}"#,
                "\n",
            )
        );
    }

    /// JSON-RPC 2.0 §4: the id is echoed with the client's own spelling and
    /// its own type — a string id comes back as that string, a number as that
    /// number, and no id is rounded on the way through.
    #[test]
    fn ids_are_echoed_verbatim() {
        for id in ["7", "0", "-3", r#""call-1""#, "1.50", "9007199254740993"] {
            let frame = frame_of(id);
            assert_eq!(
                frame,
                format!("{{\"jsonrpc\":\"2.0\",\"id\":{id},\"result\":{{}}}}\n"),
                "the id must come back exactly as it was sent: {frame}"
            );
        }
    }

    /// Every frame is exactly one newline-terminated JSON document.
    #[test]
    fn frames_are_newline_terminated_documents() {
        let response = JsonRpcResponse::ok(None, json!({}));
        let frame = response.to_frame().expect("a `Value` payload always serializes");
        assert!(frame.ends_with('\n'));
        assert_eq!(frame.matches('\n').count(), 1, "one line per frame: {frame:?}");
        let parsed: Value = serde_json::from_str(&frame).expect("the frame is one JSON document");
        assert_eq!(parsed["result"], json!({}));
    }

    /// Tool payloads are embedded as pretty-printed JSON text.
    #[test]
    fn tool_call_embeds_the_payload_as_pretty_text() {
        let req = parse_frame(r#"{"id":7,"method":"ping"}"#).expect("parse");
        let envelope = JsonRpcResponse::tool_call(req.id_or_null().map(|v| v.to_owned()), json!({ "worker_id": "w-1" }))
            .to_frame()
            .expect("frame");
        let wire: Value = serde_json::from_str(&envelope).expect("frame");
        assert_eq!(wire["id"], json!(7));
        let text = wire["result"]["content"][0]["text"]
            .as_str()
            .expect("text content");
        assert_eq!(wire["result"]["content"][0]["type"], json!("text"));
        assert_eq!(
            serde_json::from_str::<Value>(text).expect("the text is the payload"),
            json!({ "worker_id": "w-1" })
        );
        assert!(text.contains('\n') && text.contains("  "), "still pretty-printed: {text}");
    }

    /// A `tools/call` response keeps the pretty-printed payload *and* its
    /// newlines.
    #[test]
    fn tool_results_survive_the_value_slot_intact() {
        let payload = json!({ "status": "reaped", "worker_ids": [] });
        let req = parse_frame(r#"{"id":7,"method":"ping"}"#).expect("parse");
        let response = JsonRpcResponse::tool_call(req.id_or_null().map(|v| v.to_owned()), payload);
        let frame = response.to_frame().expect("frame");

        let wire: Value = serde_json::from_str(&frame).expect("frame");
        assert_eq!(wire["id"], json!(7));
        let text = wire["result"]["content"][0]["text"]
            .as_str()
            .expect("text content");
        assert_eq!(wire["result"]["content"][0]["type"], json!("text"));
        assert_eq!(
            serde_json::from_str::<Value>(text).expect("the text is the payload"),
            json!({ "status": "reaped", "worker_ids": [] })
        );
        assert!(text.contains('\n') && text.contains("  "), "still pretty-printed: {text}");
    }

    /// The handshake document is process-constant and spec-shaped.
    #[test]
    fn initialize_result_is_the_advertised_handshake() {
        let initialize: &Value = &INITIALIZE_RESULT;
        assert_eq!(initialize["protocolVersion"], json!("2024-11-05"));
        assert_eq!(initialize["serverInfo"]["name"], json!("mini-swe-mcp"));
        assert_eq!(
            initialize["serverInfo"]["version"],
            json!(env!("CARGO_PKG_VERSION")),
            "the advertised version is the package version"
        );
        assert_eq!(initialize["capabilities"]["tools"]["listChanged"], json!(false));
    }

    /// An id as it arrives on the wire, for the round-trip assertions below.
    fn frame_of(id: &str) -> String {
        let line = format!(r#"{{"id":{id},"method":"ping"}}"#);
        let req = parse_frame(&line).expect("parse");
        JsonRpcResponse::ok(req.id_or_null().map(|v| v.to_owned()), json!({}))
            .to_frame()
            .expect("frame")
    }
}
