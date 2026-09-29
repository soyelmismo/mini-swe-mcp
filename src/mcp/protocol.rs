//! JSON-RPC 2.0 wire types for the MCP stdio transport.
//!
//! Everything that reaches the wire is produced here, so this module carries
//! three rules for the hot path (a stdio daemon spends its time serialising
//! frames, not waiting for them):
//!
//! 1. **Borrow the frame you were handed.** [`parse_frame`] hands back a
//!    request whose `method` is a [`Cow::Borrowed`] slice of the line the
//!    reader already owns and whose `id` and `params` are [`RawValue`]s — the
//!    client's own bytes — so a well-formed request costs *zero* allocations
//!    to parse: no `String` for the method, no `Value` tree for the id, and
//!    `params` is promoted to a tree only on the `tools/call` path, the one
//!    place that indexes into it. Only a method name that carries a JSON
//!    escape (`Cow::Owned`) pays for an unescape.
//! 2. **Stream into one exactly sized buffer.** [`JsonRpcResponse::to_frame`]
//!    writes the envelope — payload included — straight into one buffer that
//!    already reserves room for the frame's trailing newline. That replaces
//!    `to_string(..) + "\n"`, which serializes into a temporary string and then
//!    reallocates and copies the whole frame to append a single byte. The
//!    `tools/call` payload is written by the single-pass
//!    [`PreSerializedResult`] serializer instead of being materialized as a
//!    pretty-printed `String` and escaped a second time (audit 07, F4).
//! 3. **Borrow the error text.** The messages for the common JSON-RPC errors
//!    ([`FrameRejection`]) are borrowed from the frame being answered, and a
//!    `-32601` quotes the method it rejects, so a rejection allocates nothing
//!    beyond its output buffer and the name it quotes.
//! 4. **Refuse an oversized frame before touching it.** [`parse_frame`] checks
//!    [`MAX_FRAME_BYTES`] on the raw line and answers `-32600` for anything
//!    larger, so a peer cannot make the server parse, borrow and then clone an
//!    arbitrarily large payload. The bound is on the *frame*, which is the one
//!    allocation a peer can actually grow without limit, and it is checked
//!    against a constant so the rejection itself allocates nothing.
//!
//! Spec compliance is enforced by construction rather than by convention: the
//! `jsonrpc` member is the constant [`JSONRPC_VERSION`], an envelope carries
//! exactly one of `result`/`error` (see [`Body`]), the reserved `error.code`
//! values live in [`code`], and notifications — the one case where a server
//! must stay silent — are recognised by [`JsonRpcRequest::id`] instead of by a
//! remembered string comparison.

use std::borrow::Cow;
use std::sync::LazyLock;

use serde::Deserializer as _;
use serde::de::{self, IgnoredAny, Visitor};
use serde::Serialize;
use serde_json::Value;
use serde_json::value::RawValue;

/// The `jsonrpc` member of every frame, per JSON-RPC 2.0 §6.
pub(super) const JSONRPC_VERSION: &str = "2.0";

/// Reserved `error.code` values (JSON-RPC 2.0 §5.1) and the
/// implementation-defined code tool failures are reported under.
pub(super) mod code {
    /// Invalid JSON was received.
    pub(crate) const PARSE_ERROR: i64 = -32700;
    /// The payload is valid JSON but not a valid Request object.
    pub(crate) const INVALID_REQUEST: i64 = -32600;
    /// The requested method does not exist or is unavailable.
    pub(crate) const METHOD_NOT_FOUND: i64 = -32601;
    /// Implementation-defined server error. JSON-RPC reserves -32000..-32099
    /// for exactly this range.
    pub(crate) const SERVER_ERROR: i64 = -32000;
}

/// Output buffer size that already covers every frame the server emits except
/// large tool payloads: `ping`, `initialize`, progress notifications and every
/// error frame fit within it, so they cost exactly one allocation.
const FRAME_CAPACITY: usize = 256;

/// Largest inbound JSON-RPC frame [`parse_frame`] will look at, in bytes.
///
/// A stdio MCP peer is a program on the same host, but the transport is a pipe
/// an operator can point at anything: a frame is the one value that grows with
/// the sender's appetite, and every downstream consumer copies or borrows it.
/// Refusing the line up front is cheaper than any bound placed later — the
/// parse never runs, no `Value` tree is built, and the rejection frame is a
/// fixed 256-byte buffer.
///
/// The ceiling is set well above the largest frame this server legitimately
/// receives. [`crate::mcp::schema::build_tools_list`] describes one tool with
/// an enum of ten actions, and `tools/call` carries a worker's arguments; both
/// are kilobytes at most, and the agent side already caps its own payloads
/// far lower ([`crate::agent::MAX_TOOL_ARGUMENT_BYTES`], 64 KiB). A megabyte
/// leaves two orders of magnitude of headroom while still bounding a single
/// frame's memory at something a client cannot use as an amplifier.
pub(super) const MAX_FRAME_BYTES: usize = 1024 * 1024;

/// Why a frame could not be serialized. Kept as its own type (rather than
/// swallowed) so no caller can accidentally drop a response without noticing.
#[derive(Debug)]
pub(super) enum ProtocolWarning {
    /// An outgoing frame failed to serialize. `serde_json::Value` payloads
    /// cannot fail to serialize, so this is a belt-and-braces path: the caller
    /// substitutes the spec-shaped `-32603 Internal error` frame rather than
    /// leaving the client waiting.
    UnserializableFrame,
}

/// Frame emitted on [`ProtocolWarning::UnserializableFrame`]; `id` is `null`,
/// which JSON-RPC 2.0 §5 permits when the id cannot be determined.
pub(super) const INTERNAL_ERROR_FRAME: &str =
    "{\"jsonrpc\":\"2.0\",\"id\":null,\"error\":{\"code\":-32603,\"message\":\"Internal error\"}}\n";

/// Diagnosis for a frame that carries no usable `method`. Borrowed, so the
/// `-32600` reply allocates nothing beyond its output buffer.
const NO_METHOD: &str = "Invalid Request: missing or non-string \"method\" member";

/// Diagnosis for a line refused on length alone, quoting the ceiling as a byte
/// count so an operator can size a client against it. Borrowed from the
/// binary, so an oversized frame costs only the reply's own buffer.
///
/// The figure is written out rather than derived: `stringify!` would emit the
/// constant's *name*, not its value, so it cannot build this text. A test below
/// fails if the message and [`MAX_FRAME_BYTES`] ever disagree.
const FRAME_TOO_LARGE: &str = "Invalid Request: frame exceeds the 1048576-byte limit";

// ---------------------------------------------------------------------------
// Incoming frames
// ---------------------------------------------------------------------------

/// An incoming JSON-RPC request; `id: None` marks a notification.
///
/// Every field borrows from the line the reader owns, so a request can be
/// answered on the spot (see [`JsonRpcResponse`]) without copying the frame.
#[derive(Debug)]
pub(super) struct JsonRpcRequest<'a> {
    /// The method being invoked, borrowed from the frame unless it carries a
    /// JSON escape (`"tools\u002fcall"`), which has to be unescaped.
    pub(super) method: Cow<'a, str>,
    /// The request id, the client's own JSON text. `None` (absent or `null`)
    /// marks a notification, which the server must not answer.
    pub(super) id: Option<&'a RawValue>,
    /// Method parameters, the client's own JSON text; parsed only on the
    /// `tools/call` path, the one place that indexes into them.
    pub(super) params: Option<&'a RawValue>,
}

impl<'a> JsonRpcRequest<'a> {
    /// The id to echo back, or `None` — serialized as `null`, which JSON-RPC
    /// 2.0 §5 asks for when the real one cannot be determined.
    pub(super) fn id_or_null(&self) -> Option<&'a RawValue> {
        self.id
    }

    /// The method parameters as a value, if the frame carried any.
    ///
    /// This is the one place a frame is materialized: the `tools/call` handler
    /// indexes into `name`, `arguments` and `_meta`, which a raw slice cannot
    /// answer. Every other method keeps its params as raw text and never pays.
    pub(super) fn params_value(&self) -> Option<Value> {
        serde_json::from_str(self.params?.get()).ok()
    }
}

/// Why a frame could not be turned into a request.
#[derive(Debug)]
pub(super) enum FrameRejection {
    /// The line is not valid JSON (`-32700`). The message carries serde's own
    /// diagnosis — line, column and cause — because that is the only thing an
    /// operator can act on. This is the one JSON-RPC error whose text cannot
    /// be borrowed: `serde_json::Error` owns its diagnosis and hands it back by
    /// reference only for the borrow's lifetime.
    Malformed(String),
    /// The line is valid JSON but not a JSON-RPC Request object (`-32600`):
    /// not an object, or no string `method`.
    InvalidRequest(&'static str),
    /// The line is longer than [`MAX_FRAME_BYTES`] (`-32600`), rejected before
    /// it is parsed. The text is a borrowed constant, so a flood of oversized
    /// frames costs one fixed output buffer each and no diagnostics string.
    FrameTooLarge,
}

impl FrameRejection {
    /// The `-32700` reply for a line `serde_json` refused to parse.
    ///
    /// JSON-RPC 2.0 §5.1 splits the two cases apart: `-32700` is reserved for
    /// input that is not valid JSON, while well-formed JSON that is not a
    /// Request object is `-32600`. `serde_json` reports exactly that split
    /// through [`serde_json::error::Category`], so classifying the failure
    /// costs no formatting and no allocation.
    pub(super) fn malformed(error: &serde_json::Error) -> Self {
        Self::Malformed(format!("Parse error: {error}"))
    }

    /// Renders the rejection as a complete stdio frame (`…\n`).
    ///
    /// Never fails: the envelope is a constant plus an integer plus a borrowed
    /// message, so the spec-shaped fallback is only a safety net.
    pub(super) fn into_frame(self) -> String {
        let (code, message) = match self {
            Self::Malformed(message) => (code::PARSE_ERROR, Cow::Owned(message)),
            Self::InvalidRequest(message) => (code::INVALID_REQUEST, Cow::Borrowed(message)),
            // Rejected before it was parsed, so the honest code is `-32600`:
            // nothing about the line's syntax was ever decided, and a `-32700`
            // would claim the bytes were not JSON when they may well be.
            Self::FrameTooLarge => (code::INVALID_REQUEST, Cow::Borrowed(FRAME_TOO_LARGE)),
        };
        JsonRpcResponse::err(None, code, message).to_frame()
            .unwrap_or_else(|_| String::from(INTERNAL_ERROR_FRAME))
    }
}

/// Parse one newline-delimited stdio frame into a request borrowing from
/// `frame`.
///
/// Unknown members are ignored rather than rejected (MCP peers legitimately add
/// members of their own, and `serde`'s derived structs reject them by default)
/// and a UTF-8 BOM is tolerated, but everything JSON-RPC 2.0 requires is
/// enforced: the frame must be an object and must carry a string `method`.
/// `params` and `id` may be absent, `null`, or any JSON value, and are handed
/// back exactly as written.
pub(super) fn parse_frame(frame: &str) -> Result<JsonRpcRequest<'_>, FrameRejection> {
    // Checked first, on the raw line and before the BOM strip, so the bound
    // covers every byte the peer sent and an oversized frame never reaches a
    // parser. `len()` is a field read on a `&str`, so the guard is free.
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

    // Pass 2 — build the request view. Every field is borrowed from `frame`, so
    // a well-formed frame is materialized exactly once, by nobody: what comes
    // back is a set of slices into the line the reader already holds.
    let mut request = serde_json::Deserializer::from_str(frame);
    // Pass 2 can only fail where pass 1 already proved the frame is JSON, and
    // it can only do so because the JSON is not a Request object.
    let parsed = request
        .deserialize_map(RequestVisitor)
        .map_err(|_| FrameRejection::InvalidRequest(NO_METHOD))?;
    request
        .end()
        .map_err(|_| FrameRejection::InvalidRequest(NO_METHOD))?;
    Ok(parsed)
}

/// A string borrowed from the frame when it is escape-free, unescaped into an
/// owned `String` only when it carries a JSON escape.
///
/// `serde` implements `Deserialize` for `&'de str` (borrowing through
/// `visit_borrowed_str`), but not for `Cow<'de, str>` — the blanket impl
/// always materializes `Owned`. This visitor recovers the borrow: it asks for
/// `&str` first and falls back to `String` only when the frame's escapes force
/// it, so an escaped method name is answered instead of rejected.
///
/// It reads both the `method` value and the member names, which are the two
/// strings the frame parser has to name in order to route it.
struct MethodName<'de>(Cow<'de, str>);

impl<'de> serde::Deserialize<'de> for MethodName<'de> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct MethodVisitor<'de>(std::marker::PhantomData<&'de ()>);

        impl<'de> Visitor<'de> for MethodVisitor<'de> {
            type Value = MethodName<'de>;

            // The visitor serves both the `method` value and the member names,
            // so the expectation is worded for either.
            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("a JSON-RPC method name or member name")
            }

            fn visit_borrowed_str<E: de::Error>(self, name: &'de str) -> Result<Self::Value, E> {
                Ok(MethodName(Cow::Borrowed(name)))
            }

            fn visit_str<E: de::Error>(self, name: &str) -> Result<Self::Value, E> {
                Ok(MethodName(Cow::Owned(name.to_owned())))
            }

            fn visit_string<E: de::Error>(self, name: String) -> Result<Self::Value, E> {
                Ok(MethodName(Cow::Owned(name)))
            }
        }

        deserializer.deserialize_str(MethodVisitor(std::marker::PhantomData))
    }
}

/// Streams the members the server needs out of one map access.
///
/// `parse_frame` cannot delegate to `#[derive(Deserialize)]` for two reasons:
/// the `Cow`-based request is generic over `'de`, and the derived struct would
/// hand out a freshly allocated `String` method plus a copied `Value` tree.
struct RequestVisitor;

impl<'de> Visitor<'de> for RequestVisitor {
    type Value = JsonRpcRequest<'de>;

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("a JSON-RPC 2.0 request object")
    }

    fn visit_map<A: de::MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
        let mut method: Option<Cow<'de, str>> = None;
        let mut id: Option<&'de RawValue> = None;
        let mut params: Option<&'de RawValue> = None;

        // `MethodName` doubles as the key reader: it is the same
        // borrowed-when-escape-free string deserializer, and asking for it here
        // is what keeps member names allocation-free. `Cow<str>` would have
        // copied *every* key into a `String` — including the four known ones —
        // on every frame.
        while let Some(key) = map.next_key::<MethodName<'de>>()?.0 {
            match key.as_ref() {
                // A key that carries a JSON escape (`"tools\u002fcall"`)
                // arrives owned but still compares equal, so such a frame is
                // answered instead of rejected.
                "method" => method = Some(map.next_value::<MethodName<'de>>()?.0),
                // Raw text: the id is echoed byte for byte (§4 — the server may
                // not round `9007199254740993` or re-quote a string id).
                "id" => id = map.next_value::<Option<&'de RawValue>>()?,
                "params" => params = map.next_value::<Option<&'de RawValue>>()?,
                // Unknown members (`jsonrpc`, `_meta`, host extensions, …) are
                // dropped without being parsed into a `Value`.
                _ => {
                    map.next_value::<IgnoredAny>()?;
                }
            }
        }

        Ok(JsonRpcRequest {
            method: method.ok_or_else(|| de::Error::custom(NO_METHOD))?,
            id,
            params,
        })
    }
}

// ---------------------------------------------------------------------------
// Outgoing frames
// ---------------------------------------------------------------------------

/// An outgoing JSON-RPC 2.0 response envelope.
///
/// Exactly one of `result`/`error` is serialized (see [`Body`]), so the two
/// mutually exclusive `Option`s a hand-rolled envelope struct would need cannot
/// produce a frame that is neither, or both — which the spec forbids.
#[derive(Debug)]
pub(super) struct JsonRpcResponse<'a> {
    /// The id to echo, in the encoding the client used; `None` serializes as
    /// `null`, which JSON-RPC 2.0 §5 permits when the id cannot be determined.
    id: Option<&'a RawValue>,
    body: Body<'a>,
}

/// The mutually exclusive payload of a JSON-RPC response.
#[derive(Debug)]
enum Body<'a> {
    /// A successful `result` the caller built as a value.
    Result(Value),
    /// A successful `tools/call` result: the payload is pretty-printed and
    /// escaped straight into the frame buffer, never materialized as a `String`
    /// first (audit 07, F4).
    ToolCall(PreSerializedResult),
    /// An `error` object: a reserved `code` plus a message that is borrowed
    /// from the frame whenever possible.
    Error {
        code: i64,
        message: Cow<'a, str>,
    },
}

impl<'a> JsonRpcResponse<'a> {
    /// Successful JSON-RPC 2.0 response whose `result` the caller owns.
    pub(super) fn ok(id: Option<&'a RawValue>, result: Value) -> Self {
        Self {
            id,
            body: Body::Result(result),
        }
    }

    /// Successful `tools/call` response carrying a tool payload.
    ///
    /// [`PreSerializedResult`] embeds the payload as the `text` of a single
    /// content block, pretty-printed by [`PayloadWriter`] and escaped by
    /// `collect_str` while the frame is written — the payload is materialized
    /// once, and never as an escaped `String` copy (audit 07, F4).
    pub(super) fn tool_call(id: Option<&'a RawValue>, payload: Value) -> Self {
        Self {
            id,
            body: Body::ToolCall(PreSerializedResult::text(payload)),
        }
    }

    /// JSON-RPC 2.0 error response carrying a reserved `code` and a message
    /// borrowed from the frame being answered.
    pub(super) fn err(id: Option<&'a RawValue>, code: i64, message: Cow<'a, str>) -> Self {
        Self {
            id,
            body: Body::Error { code, message },
        }
    }

    /// `-32601 Method not found` for a method the server does not expose; the
    /// rejected name is quoted straight from the frame.
    pub(super) fn method_not_found(id: Option<&'a RawValue>, method: Cow<'a, str>) -> Self {
        Self::err(
            id,
            code::METHOD_NOT_FOUND,
            Cow::Owned(format!("Method not found: {method}")),
        )
    }

    /// Serialize the envelope into a single frame buffer, newline included.
    ///
    /// The buffer is reserved up front, so the common frames (`ping`, tool
    /// errors, progress notifications) never reallocate, and the frame reaches
    /// the writer task without the extra `serialized + "\n"` copy.
    pub(super) fn to_frame(&self) -> Result<String, ProtocolWarning> {
        let mut frame = String::with_capacity(FRAME_CAPACITY);
        // The default (compact) formatter writes straight into the frame
        // buffer through the `io::Write` bridge, so no intermediate `String`
        // is produced and the trailing newline costs no reallocation.
        let mut serializer = serde_json::Serializer::new(StrSink(&mut frame));
        if let Err(error) = self.serialize(&mut serializer) {
            // A `Value` payload cannot fail to serialize, so this is
            // unreachable in practice; it is handled instead of unwrapped
            // because a daemon must not abort mid-stream.
            tracing::error!(%error, "Failed to serialize a JSON-RPC response");
            return Err(ProtocolWarning::UnserializableFrame);
        }
        frame.push('\n');
        Ok(frame)
    }
}

/// The `error` member of a response: a reserved `code` plus a message that is
/// borrowed from the frame whenever possible.
struct ErrorPayload<'a> {
    code: i64,
    message: &'a str,
}

impl Serialize for ErrorPayload<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut error = serializer.serialize_struct("Error", 2)?;
        error.serialize_field("code", &self.code)?;
        error.serialize_field("message", self.message)?;
        error.end()
    }
}

impl Serialize for JsonRpcResponse<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut frame = serializer.serialize_struct("JsonRpcResponse", 3)?;
        frame.serialize_field("jsonrpc", JSONRPC_VERSION)?;
        frame.serialize_field("id", &self.id)?;
        match &self.body {
            Body::Result(result) => frame.serialize_field("result", result)?,
            Body::ToolCall(result) => frame.serialize_field("result", result)?,
            Body::Error { code, message } => {
                frame.serialize_field(
                    "error",
                    &ErrorPayload {
                        code: *code,
                        message: message.as_ref(),
                    },
                )?;
            }
        }
        frame.end()
    }
}

// MCP envelopes built once per process
// ---------------------------------------------------------------------------

// The handshake is spelled out as a wire literal above: `protocolVersion` is the
// MCP revision this server speaks (2024-11-05) and `serverInfo.name` is the
// crate, whose version is the package version by definition, so neither is worth
// a second definition that could drift.

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

// ---------------------------------------------------------------------------
// MCP `tools/call` result
// ---------------------------------------------------------------------------

/// A JSON-RPC `result` whose single text content is a tool payload.
///
/// `tools/call` embeds the tool payload as a JSON *string* inside the envelope
/// (`{"content":[{"type":"text","text":"<json>"}]}`). Building that with `json!`
/// requires pretty-printing the payload into a `String` first, which the outer
/// frame then escapes and re-serializes — a second full materialization of the
/// same bytes. `PreSerializedResult` keeps the payload as a `Value` and lets the
/// envelope write it straight into the frame buffer, so the payload is
/// materialized once (audit 07, F4).
#[derive(Debug)]
pub(super) struct PreSerializedResult {
    pub(super) content: [PreSerializedContent; 1],
}

/// The one `{"type": "text", "text": …}` content block a tool reply carries.
#[derive(Debug)]
pub(super) struct PreSerializedContent {
    kind: &'static str,
    pub(super) payload: Value,
}

impl PreSerializedResult {
    /// Wrap a tool payload in the single text content block MCP clients expect.
    pub(super) fn text(payload: Value) -> Self {
        Self {
            content: [PreSerializedContent {
                kind: "text",
                payload,
            }],
        }
    }
}

impl Serialize for PreSerializedResult {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut st = serializer.serialize_struct("Result", 1)?;
        st.serialize_field("content", &self.content)?;
        st.end()
    }
}

impl Serialize for PreSerializedContent {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut st = serializer.serialize_struct("Content", 2)?;
        st.serialize_field("type", self.kind)?;
        // The `text` field is a JSON string; the payload is written into it
        // directly instead of via an intermediate `to_string_pretty` result.
        st.serialize_field("text", &PayloadAsText(&self.payload))?;
        st.end()
    }
}

/// Writes a `Value` into a serialized string field without a separate `String`
/// allocation for the envelope to copy.
struct PayloadAsText<'a>(&'a Value);

impl Serialize for PayloadAsText<'_> {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        serializer.collect_str(&PayloadWriter(self.0))
    }
}

/// Feeds a `Value` to `collect_str` through `Display`, so the payload is written
/// into the target string buffer in one pass.
///
/// Pretty-printing is preserved: MCP clients read this field as human-readable
/// tool output, and `collect_str` streams the formatting straight into the
/// frame's own buffer, so the previous `to_string_pretty` `String` is gone.
struct PayloadWriter<'a>(&'a Value);

impl std::fmt::Display for PayloadWriter<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Render pretty JSON into an internal buffer using a `Write` sink that
        // forwards to the formatter, so the value is produced once and written
        // through `collect_str` without a second, escaped envelope copy.
        let mut sink = FmtSink { f };
        let mut ser = serde_json::Serializer::with_formatter(
            &mut sink,
            serde_json::ser::PrettyFormatter::new(),
        );
        self.0.serialize(&mut ser).map_err(|_| std::fmt::Error)
    }
}

/// `std::io::Write` adapter over the frame buffer.
///
/// `String` is a `fmt::Write`, not an `io::Write`, so `serde_json`'s byte
/// oriented sink has to be bridged. Escaping never inserts a lone surrogate
/// (`char::encode_utf8` emits U+FFFD instead), which makes the checked
/// `from_utf8` unreachable in practice.
struct StrSink<'a>(&'a mut String);

impl std::io::Write for StrSink<'_> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let text = std::str::from_utf8(buf).map_err(|_| std::io::Error::other("non-utf8 json"))?;
        self.0.push_str(text);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Minimal `std::io::Write` adapter over a `fmt::Formatter`, used to drive
/// `serde_json`'s `PrettyFormatter` while `collect_str` streams into the
/// frame's own buffer.
struct FmtSink<'a, 'b> {
    f: &'a mut std::fmt::Formatter<'b>,
}

impl std::io::Write for FmtSink<'_, '_> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let s = std::str::from_utf8(buf).map_err(|_| std::io::Error::other("non-utf8 json"))?;
        self.f
            .write_str(s)
            .map_err(|_| std::io::Error::other("fmt error"))?;
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// A frame the reader can answer without copying; the `id` and `params`
    /// are the client's own bytes, not freshly built trees.
    #[test]
    fn parsed_frames_borrow_the_line() {
        let line = r#"{"jsonrpc":"2.0","id":7,"method":"ping","params":{"action":"reap"}}"#;
        let req = parse_frame(line).expect("a spec-shaped frame must parse");

        assert_eq!(req.method, "ping");
        assert!(matches!(req.method, Cow::Borrowed(_)), "method must borrow");
        // The id is the client's own text, so every digit of it survives.
        assert_eq!(req.id.unwrap().get(), "7");
        assert_eq!(
            req.params_value().and_then(|params| params["action"].as_str().map(str::to_owned)),
            Some("reap".to_owned())
        );
    }

    /// A method name that carries a JSON escape has to be unescaped to be
    /// represented, so it materializes. A `/` is a JSON escape even though it
    /// reads as itself, which is why `tools/call` is the escaping method.
    #[test]
    fn escaped_members_fall_back_to_owned_values() {
        let req = parse_frame(r#"{"id":"c-1","method":"tools\u002fcall"}"#).expect("parse");
        assert!(
            matches!(req.method, Cow::Owned(_)),
            "an escaped method name has to be unescaped to be represented"
        );
        assert_eq!(req.method, "tools/call");
        assert_eq!(req.id.unwrap().get(), r#""c-1""#, "the raw id keeps its quotes");
    }

    /// Member names borrow the line too. The key reader is the same
    /// borrowed-when-escape-free visitor the method name uses, so a spec-shaped
    /// frame allocates no `String` for its keys — the `Cow<str>` key type it
    /// replaced copied every one of them.
    #[test]
    fn member_names_are_borrowed_from_the_line() {
        let line = r#"{"jsonrpc":"2.0","id":1,"method":"ping"}"#;
        let req = parse_frame(line).expect("frame");
        // Nothing owned: the method and the raw id both point into `line`
        // rather than into a fresh `String`, which is what proves the key
        // reader and the value reader both borrowed.
        assert!(matches!(req.method, Cow::Borrowed(_)));
        // `rfind` picks the id's `1` rather than the `1` inside `2.0`, so the
        // expected offset is unambiguous even if the fixture grows.
        let id_offset = line.rfind('1').expect("the id is in the line");
        assert_eq!(
            req.id
                .map(|id| id.get().as_ptr() as usize - line.as_ptr() as usize),
            Some(id_offset),
            "the id must point into the line, not into a fresh allocation"
        );
    }

    /// The raw params survive exactly as written (quoted strings stay quoted,
    /// numbers stay numbers), and an absent `params` member stays absent.
    #[test]
    fn params_stay_raw_until_indexed() {
        let req = parse_frame(r#"{"id":1,"method":"ping","params":{"s":"x\ny"}}"#).expect("parse");
        assert_eq!(
            req.params.unwrap().get(),
            r#"{"s":"x\ny"}"#,
            "params are raw text: escapes are not unescaped until indexed"
        );
        assert_eq!(
            parse_frame(r#"{"id":7,"method":"ping"}"#)
                .expect("params are optional")
                .params_value(),
            None
        );
    }

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
        let err = JsonRpcResponse::err(
            id(r#"{"id":"server-1","method":"ping"}"#),
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

    /// `-32601` quotes the method it rejected, borrowed from the frame.
    #[test]
    fn method_not_found_quotes_the_requested_method() {
        let line = r#"{"id":3,"method":"does/not/exist"}"#;
        let req = parse_frame(line).expect("parse");
        let response = JsonRpcResponse::method_not_found(req.id_or_null(), req.method);
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

    /// Every frame is exactly one newline-terminated JSON document: the buffer
    /// is reserved up front, so the newline must not cost a second copy.
    #[test]
    fn frames_are_newline_terminated_documents() {
        let response = JsonRpcResponse::ok(None, json!({}));
        let frame = response.to_frame().expect("a `Value` payload always serializes");
        assert!(frame.ends_with('\n'));
        assert_eq!(frame.matches('\n').count(), 1, "one line per frame: {frame:?}");
        let parsed: Value = serde_json::from_str(&frame).expect("the frame is one JSON document");
        assert_eq!(parsed["result"], json!({}));
    }

    /// Tool payloads are embedded as pretty-printed JSON text, exactly as
    /// before, and the envelope no longer materializes an escaped copy of it.
    #[test]
    fn pre_serialized_result_embeds_the_payload_as_text() {
        let envelope =
            serde_json::to_value(PreSerializedResult::text(json!({ "worker_id": "w-1" })))
                .expect("tool results must serialise");
        assert_eq!(
            envelope,
            json!({
                "content": [{
                    "type": "text",
                    "text": "{\n  \"worker_id\": \"w-1\"\n}"
                }]
            })
        );
    }

    /// A `tools/call` response keeps the pretty-printed payload *and* its
    /// newlines: `collect_str` escapes the payload once, while the frame is
    /// written, instead of the outer serializer escaping an already-escaped
    /// `String`.
    #[test]
    fn tool_results_survive_the_value_slot_intact() {
        let payload = json!({ "status": "reaped", "worker_ids": [] });
        let response =
            JsonRpcResponse::tool_call(id(r#"{"id":7,"method":"ping"}"#), payload);
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

    /// A raw id value, for constructors that answer frames the test built.
    fn id(line: &'static str) -> Option<&'static RawValue> {
        parse_frame(line).expect("frame").id
    }

    /// An id as it arrives on the wire, for the round-trip assertions below.
    fn frame_of(id: &str) -> String {
        let line = format!(r#"{{"id":{id},"method":"ping"}}"#);
        let req = parse_frame(&line).expect("parse");
        JsonRpcResponse::ok(req.id_or_null(), json!({}))
            .to_frame()
            .expect("frame")
    }
}

