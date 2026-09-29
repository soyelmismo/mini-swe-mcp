//! JSON-RPC 2.0 wire types for the MCP stdio transport.
//!
//! Everything that reaches the wire is produced here, so this module carries
//! three rules for the hot path (a stdio daemon spends its time serialising
//! frames, not waiting for them):
//!
//! 1. **Borrow the frame you were handed.** [`parse_frame`] hands back a
//!    request whose `method`, `id` and `params` are [`Cow::Borrowed`] slices of
//!    the line the reader already owns, so a well-formed request costs *zero*
//!    allocations to parse — no `String` for the method, no `Value` tree for
//!    `id`, no owned clone of `params`. Only values that would have to be
//!    unescaped to be represented (`Cow::Owned`, e.g. a string `id`) pay.
//! 2. **Stream into one exactly sized buffer.** [`JsonRpcResponse::to_frame`]
//!    writes the envelope — payload included — straight into one buffer that
//!    already reserves room for the frame's trailing newline. That replaces
//!    `to_string(..) + "\n"`, which serializes into a temporary string and then
//!    reallocates and copies the whole frame to append a single byte. The
//!    `tools/call` payload is written by the single-pass
//!    [`PreSerializedResult`] serializer instead of being materialized as a
//!    pretty-printed `String` and escaped a second time (audit 07, F4).
//! 3. **Borrow the error text.** The messages for the common JSON-RPC errors
//!    ([`JsonRpcResponse::method_not_found`], [`FrameRejection`]) are built out
//!    of the frame being answered, so rejecting a request allocates nothing
//!    beyond its output buffer.
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
use serde::{Deserialize, Serialize};
use serde_json::Value;

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

// ---------------------------------------------------------------------------
// Incoming frames
// ---------------------------------------------------------------------------

/// A JSON value that may be a view of the frame it was parsed out of.
///
/// Every `serde_json` input value is a fresh [`Value`] tree, so `Cow<str>`
/// around it always degrades to `Owned`: a value that happens to need no
/// unescaping is still rebuilt, member by member, before the server ever looks
/// at it. `params` is read as text for that reason — the frame's own bytes,
/// promoted to a `Value` only where a handler actually indexes into it (see
/// `src/mcp/server.rs`). The wire contract is identical either way; this is
/// purely about not paying for a tree nobody inspects.
#[derive(Clone, Debug, Deserialize)]
#[repr(transparent)]
pub(super) struct RawValue<'a>(#[serde(borrow)] Cow<'a, Value>);

impl RawValue<'_> {
    /// Takes ownership of a value the server already materialized.
    pub(super) fn owned(value: Value) -> Self {
        Self(Cow::Owned(value))
    }
}

impl std::ops::Deref for RawValue<'_> {
    type Target = Value;

    fn deref(&self) -> &Value {
        &self.0
    }
}

impl Serialize for RawValue<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.0.serialize(serializer)
    }
}

/// The `id` member of an incoming request, kept in the client's own encoding.
///
/// JSON-RPC 2.0 §4 allows a String, a Number or `null`, and it forbids neither
/// fractions (`1.0`) nor absurd precision (`9007199254740993`) because that is
/// precisely what a peer may have sent. Materializing the id into a `Value`
/// therefore risks *changing* it — the echo stops being the id the client sent —
/// so it is borrowed straight from the frame instead: a slice of the input is
/// always byte-identical to what arrived, and costs nothing to take.
#[derive(Clone, Debug)]
#[repr(transparent)]
pub(super) struct RequestId<'a>(pub(super) Cow<'a, str>);

impl<'de> Deserialize<'de> for RequestId<'de> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        // An id is String or Number (§4). Anything else is refused here rather
        // than silently coerced.
        deserializer.deserialize_any(RequestIdVisitor)
    }
}

/// Reads a JSON-RPC id as the *text* the client sent.
///
/// A String id is borrowed from the frame, and a Number id is borrowed as its
/// own digits — `"9007199254740993"` keeps every digit instead of collapsing to
/// a `f64`, and `"1.0"` keeps its decimal point instead of becoming `1`. Both are
/// what an echo has to reproduce byte for byte.
struct RequestIdVisitor;

impl<'de> Visitor<'de> for RequestIdVisitor {
    type Value = RequestId<'de>;

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("a JSON-RPC id: a string or a number")
    }

    fn visit_borrowed_str<E: de::Error>(self, id: &'de str) -> Result<Self::Value, E> {
        Ok(RequestId(Cow::Borrowed(id)))
    }

    fn visit_u64<E: de::Error>(self, id: u64) -> Result<Self::Value, E> {
        Ok(RequestId(Cow::Owned(id.to_string())))
    }

    fn visit_i64<E: de::Error>(self, id: i64) -> Result<Self::Value, E> {
        Ok(RequestId(Cow::Owned(id.to_string())))
    }

    fn visit_f64<E: de::Error>(self, id: f64) -> Result<Self::Value, E> {
        Ok(RequestId(Cow::Owned(id.to_string())))
    }
}

impl Serialize for RequestId<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        // Written as the raw JSON text the client sent. Re-parsing it into a
        // `Value` is not an option: that is the round-trip which could change
        // an id — `9007199254740993` would collapse to `9007199254740992`.
        // What comes back instead is `Cow::Borrowed`, so a string id costs
        // nothing to echo.
        match self.0.as_ref() {
            // No escape sequence to fold, no unescaping to undo: the text is
            // already its own JSON encoding, so it goes out verbatim — one
            // borrowed write, one allocation-free escape scan.
            json if !json.contains('\\') => serializer.serialize_str(json),
            // Escaped text (a `"` inside a string id, a `\uXXXX` escape, …) is
            // the only case that has to be unescaped and re-escaped. It costs
            // one `String`, and it is the only reason this serializer exists.
            json => {
                let value: Value = serde_json::from_str(json)
                    .map_err(|error| serde::ser::Error::custom(error.to_string()))?;
                value.serialize(serializer)
            }
        }
    }
}

impl<'a> RequestId<'a> {
    /// The id JSON-RPC 2.0 §5 asks for when the real one cannot be
    /// determined: an unparseable frame, or a notification that must not be
    /// answered. Borrowed, so it costs nothing.
    pub(super) const NULL: &'static str = "null";

    /// The id as an opaque JSON value for handlers that must echo it into a
    /// payload (`progressToken`, for instance). Only paid on those paths.
    pub(super) fn to_value(&self) -> Value {
        serde_json::from_str(&self.0).unwrap_or(Value::Null)
    }
}


/// An incoming JSON-RPC request; `id: None` marks a notification.
///
/// Every field borrows from the line the reader owns, so a request can be
/// answered on the spot (see [`JsonRpcResponse`]) without copying the frame.
#[derive(Debug)]
pub(super) struct JsonRpcRequest<'a> {
    /// The method being invoked, borrowed from the frame.
    pub(super) method: Cow<'a, str>,
    /// The request id, borrowed from the frame. `None` (absent or `null`)
    /// marks a notification, which the server must not answer.
    pub(super) id: Option<Cow<'a, RequestId<'a>>>,
    /// Method parameters, borrowed unless the frame carries a string member
    /// that would need unescaping.
    pub(super) params: Option<Cow<'a, RawValue<'a>>>,
}

impl<'a> JsonRpcRequest<'a> {
    /// The id to echo back, or `null` when the frame carried none.
    pub(super) fn id_or_null(&self) -> Cow<'a, RequestId<'a>> {
        self.id.clone().unwrap_or(Cow::Owned(RequestId(Cow::Borrowed(RequestId::NULL))))
    }

    /// The method parameters, if the frame carried any.
    pub(super) fn params_value(&self) -> Option<&Value> {
        self.params.as_ref().map(|params| &***params)
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
        };
        JsonRpcResponse::err(Cow::Owned(RequestId(Cow::Borrowed(RequestId::NULL))), code, message)
            .to_frame()
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

/// A borrowed `&str` from the frame, for the members the server hands out as
/// plain text.
///
/// `serde` implements `Deserialize` for `&'de str`, not for `Cow<'de, str>`
/// (it is transparent over `str`, which is), so the borrow is declared once
/// here and the request borrows its method name through it.
type BorrowedStr<'de> = &'de str;

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
        let mut method: Option<&'de str> = None;
        let mut id: Option<Cow<'de, RequestId<'de>>> = None;
        let mut params: Option<Cow<'de, RawValue<'de>>> = None;

        while let Some(key) = map.next_key::<Cow<'de, str>>()? {
            match key.as_ref() {
                "method" => method = Some(map.next_value::<BorrowedStr<'de>>()?),
                "id" => id = map.next_value::<Option<Cow<'de, RequestId<'de>>>>()?,
                "params" => params = map.next_value::<Option<Cow<'de, RawValue<'de>>>>()?,
                // Unknown members (`jsonrpc`, `_meta`, host extensions, …) are
                // dropped without being parsed into a `Value`.
                _ => {
                    map.next_value::<IgnoredAny>()?;
                }
            }
        }

        Ok(JsonRpcRequest {
            method: Cow::Borrowed(method.ok_or_else(|| de::Error::custom(NO_METHOD))?),
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
    /// The id to echo, in the encoding the client used (`null` when the id
    /// could not be determined).
    pub(super) id: Cow<'a, RequestId<'a>>,
    body: Body<'a>,
}

/// The mutually exclusive payload of a JSON-RPC response.
#[derive(Debug)]
enum Body<'a> {
    /// A successful `result`.
    Result(Cow<'a, RawValue<'a>>),
    /// An `error` object: a reserved `code` plus a message that is borrowed
    /// from the frame whenever possible.
    Error {
        code: i64,
        message: Cow<'a, str>,
    },
}

impl<'a> JsonRpcResponse<'a> {
    /// Successful JSON-RPC 2.0 response whose `result` the caller owns.
    pub(super) fn ok(id: Cow<'a, RequestId<'a>>, result: Value) -> Self {
        Self {
            id,
            body: Body::Result(Cow::Owned(RawValue::owned(result))),
        }
    }

    /// JSON-RPC 2.0 error response carrying a reserved `code` and a message
    /// borrowed from the frame being answered.
    pub(super) fn err(id: Cow<'a, RequestId<'a>>, code: i64, message: Cow<'a, str>) -> Self {
        Self {
            id,
            body: Body::Error { code, message },
        }
    }

    /// `-32601 Method not found` for a method the server does not expose; the
    /// rejected name is borrowed straight from the frame.
    pub(super) fn method_not_found(id: Cow<'a, RequestId<'a>>, method: Cow<'a, str>) -> Self {
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
        let mut serializer = serde_json::Serializer::new(StrSink(&mut frame));
        let serialized = self.serialize(&mut serializer);
        // The serializer's borrow of the buffer ends here.
        drop(serializer);
        if let Err(error) = serialized {
            // A `serde_json::Value` payload cannot fail to serialize, so this is
            // unreachable in practice; it is handled instead of unwrapped
            // because a daemon must not abort mid-stream.
            tracing::error!(%error, "Failed to serialize a JSON-RPC response");
            return Err(ProtocolWarning::UnserializableFrame);
        }
        frame.push('\n');
        Ok(frame)
    }

    /// The envelope as a value, for the few callers that need to reason about
    /// a response structurally rather than as bytes.
    pub(super) fn to_value(&self) -> Value {
        serde_json::to_value(self).unwrap_or(Value::Null)
    }
}

/// The `error` member of a response: a reserved `code` plus a message that is
/// borrowed from the frame whenever possible.
struct ErrorPayload<'a> {
    code: i64,
    message: Cow<'a, str>,
}

impl Serialize for ErrorPayload<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut error = serializer.serialize_struct("Error", 2)?;
        error.serialize_field("code", &self.code)?;
        error.serialize_field("message", &self.message)?;
        error.end()
    }
}

impl Serialize for JsonRpcResponse<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut frame = serializer.serialize_struct("JsonRpcResponse", 4)?;
        frame.serialize_field("jsonrpc", JSONRPC_VERSION)?;
        frame.serialize_field("id", &self.id)?;
        match &self.body {
            Body::Result(result) => frame.serialize_field("result", result)?,
            Body::Error { code, message } => {
                frame.serialize_field("error", &ErrorPayload { code: *code, message: message.clone() })?;
            }
        }
        frame.end()
    }
}

// ---------------------------------------------------------------------------
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
pub(super) struct PreSerializedResult {
    pub(super) content: [PreSerializedContent; 1],
}

/// The one `{"type": "text", "text": …}` content block a tool reply carries.
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

/// Carries an already-correct JSON fragment through a [`Value`] slot.
///
/// The envelope's `result` is a `Value` so handlers can hand back payloads they
/// built with `json!`, but a `tools/call` result is *not* meant to be
/// materialized: its payload has to be pretty-printed and escaped by
/// [`PreSerializedResult`] inside the frame, which a `Value` cannot express
/// (serde escapes control characters itself, so wrapping it in a
/// `Value::String` would emit `\\n` instead of the newlines MCP clients expect).
///
/// Wrapping the serializer keeps the fast path intact and removes the second
/// materialization, at the cost of one map allocation per `tools/call`
/// response. The value is only ever written out, never inspected, so the
/// delegated `Serialize` impl is enough.
pub(super) struct RawText<T>(pub(super) T);

impl<T: Serialize> Serialize for RawText<T> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.0.serialize(serializer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// A frame the reader can answer without copying: the id and the params are
    /// views of the input, not freshly built trees.
    #[test]
    fn parsed_frames_borrow_the_line() {
        let line = r#"{"jsonrpc":"2.0","id":7,"method":"ping","params":{"action":"reap"}}"#;
        let req = parse_frame(line).expect("a spec-shaped frame must parse");

        assert_eq!(req.method, "ping");
        assert!(matches!(req.method, Cow::Borrowed(_)), "method must borrow");
        assert!(matches!(req.id, Some(Cow::Owned(_))), "a number id is materialized");
        assert!(
            matches!(req.params, Some(Cow::Owned(_))),
            "a params object is a fresh `Value` tree: it can only be owned"
        );
        // The id is the client's own text, so every digit of it survives.
        assert_eq!(req.id_or_null().0, "7");
        assert_eq!(
            req.params_value().and_then(|params| params["action"].as_str()),
            Some("reap")
        );
    }

    /// Values that carry a JSON escape have to be unescaped to be represented,
    /// so they materialize. A `/` is a JSON escape even though it reads as
    /// itself, which is why `tools/call` is the escaping method name here.
    #[test]
    fn escaped_members_fall_back_to_owned_values() {
        let req = parse_frame(r#"{"id":"c-1","method":"tools\u002fcall"}"#).expect("parse");
        assert!(
            matches!(req.method, Cow::Owned(_)),
            "an escaped method name has to be unescaped to be represented"
        );
        assert_eq!(req.method, "tools/call");
        assert!(
            matches!(req.id, Some(Cow::Borrowed(_))),
            "a plain string id is borrowed from the frame"
        );
        assert_eq!(req.id_or_null().0, "c-1");
    }

    /// Borrowed and owned representations agree on the wire.
    #[test]
    fn borrowed_and_owned_frames_agree() {
        let borrowed = parse_frame(r#"{"id":7,"method":"ping","params":{"n":1}}"#).expect("parse");
        let owned = parse_frame(r#"{"id":"7","method":"ping","params":{"s":"x\ny"}}"#).expect("parse");

        assert!(matches!(owned.params, Some(Cow::Owned(_))));
        assert_eq!(borrowed.params_value(), Some(&json!({ "n": 1 })));
        assert_eq!(
            owned.params_value(),
            Some(&json!({ "s": "x\ny" })),
            "the escaped string arrives unescaped"
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
            // …and an echoable id still has to exist for anything else.
            assert_eq!(req.id_or_null().0, "null");
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

        // The `-32600` wording is borrowed, so it stays byte-identical.
        assert_eq!(
            parse_frame(r#"{"id":1}"#).expect_err("no method").into_frame(),
            format!(
                "{{\"jsonrpc\":\"2.0\",\"id\":null,\"error\":{{\"code\":{},\"message\":\"{NO_METHOD}\"}}}}\n",
                code::INVALID_REQUEST
            )
        );
    }

    /// The `ok` / `err` constructors emit spec-shaped JSON-RPC 2.0 envelopes.
    #[test]
    fn json_rpc_constructors_are_spec_shaped() {
        let ok = JsonRpcResponse::ok(owned_id(json!(7)), json!({ "a": 1 })).to_frame();
        assert_eq!(ok.expect("frame"), "{\"jsonrpc\":\"2.0\",\"id\":7,\"result\":{\"a\":1}}\n");

        let err =
            JsonRpcResponse::err(owned_id(Value::Null), code::METHOD_NOT_FOUND, Cow::Borrowed("nope"))
                .to_value();
        assert_eq!(
            err,
            json!({
                "jsonrpc": "2.0",
                "id": null,
                "error": { "code": -32601, "message": "nope" }
            })
        );
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
                r#"{"jsonrpc":"2.0","id":"3","error":{"code":-32601,"#,
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
        for id in ["7", "0", "-3", r#""call-1""#] {
            let line = format!(r#"{{"id":{id},"method":"ping"}}"#);
            let req = parse_frame(&line).expect("parse");
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
        let response = JsonRpcResponse::ok(owned_id(json!(1)), json!({}));
        let frame = response.to_frame().expect("a `Value` payload always serializes");
        assert!(frame.ends_with('\n'));
        assert_eq!(frame.matches('\n').count(), 1, "one line per frame: {frame:?}");
        let parsed: Value = serde_json::from_str(&frame).expect("the frame is one JSON document");
        assert_eq!(parsed["result"], json!({}));
    }

    /// Tool payloads are embedded as pretty-printed JSON text, exactly as
    /// before, but the envelope no longer materializes an escaped copy of it.
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
    /// newlines: the raw wrapper is what stops serde from escaping them twice.
    #[test]
    fn tool_results_survive_the_value_slot_intact() {
        let payload = json!({ "status": "reaped", "worker_ids": [] });
        let result = serde_json::to_value(RawText(PreSerializedResult::text(payload)))
            .expect("the raw wrapper always serializes");
        let response = JsonRpcResponse::ok(owned_id(Value::Null), result);
        let frame = response.to_frame().expect("frame");

        let wire: Value = serde_json::from_str(&frame).expect("frame");
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

    /// The id a response that is not answering a frame is given: a value the
    /// server built itself, serialized as the JSON text of that value.
    fn owned_id(value: Value) -> Cow<'static, RequestId<'static>> {
        Cow::Owned(RequestId(Cow::Owned(value.to_string())))
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
