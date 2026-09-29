//! JSON-RPC 2.0 wire types for the MCP stdio transport: the request/response
//! envelopes and the single-pass `tools/call` result serializer described on
//! [`PreSerializedResult`].

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

/// An incoming JSON-RPC request; `id: None` marks a notification.
#[derive(Debug, Deserialize)]
pub(super) struct JsonRpcRequest {
    pub(super) id: Option<Value>,
    pub(super) method: String,
    pub(super) params: Option<Value>,
}

/// An outgoing JSON-RPC 2.0 response envelope.
#[derive(Debug, Serialize)]
pub(super) struct JsonRpcResponse {
    pub(super) jsonrpc: &'static str,
    pub(super) id: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) error: Option<Value>,
}

impl JsonRpcResponse {
    /// Successful JSON-RPC 2.0 response.
    pub(super) fn ok(id: Option<Value>, result: Value) -> Self {
        Self {
            jsonrpc: "2.0",
            id,
            result: Some(result),
            error: None,
        }
    }

    /// JSON-RPC 2.0 error response carrying an application-level `code`.
    pub(super) fn err(id: Option<Value>, code: i64, message: impl Into<String>) -> Self {
        Self {
            jsonrpc: "2.0",
            id,
            result: None,
            error: Some(json!({
                "code": code,
                "message": message.into()
            })),
        }
    }
}

/// A JSON-RPC `result` whose single text content is a tool payload.
///
/// `tools/call` embeds the tool payload as a JSON *string* inside the envelope
/// (`{"content":[{"type":"text","text":"<json>"}]}`). Building that with `json!`
/// requires pretty-printing the payload into a `String` first, which the outer
/// `to_string` then escapes and re-serializes — a second full materialization
/// of the same bytes. `PreSerializedResult` keeps the payload as a `Value` and
/// lets the envelope serializer write it directly, so the payload is
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

/// Writes a `Value` into a serialized string field without a separate
/// `String` allocation for the envelope to copy.
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
/// envelope's own buffer, so the previous `to_string_pretty` `String` is gone.
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

/// Minimal `std::io::Write` adapter over a `fmt::Formatter`, used to drive
/// `serde_json`'s `PrettyFormatter` while `collect_str` streams into the
/// envelope's own buffer.
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

    /// The `ok` / `err` constructors emit spec-shaped JSON-RPC 2.0 envelopes.
    #[test]
    fn json_rpc_constructors_are_spec_shaped() {
        let ok = serde_json::to_value(JsonRpcResponse::ok(Some(json!(7)), json!({ "a": 1 })))
            .expect("ok responses must serialise");
        assert_eq!(
            ok,
            json!({ "jsonrpc": "2.0", "id": 7, "result": { "a": 1 } })
        );

        let err = serde_json::to_value(JsonRpcResponse::err(None, -32601, "nope"))
            .expect("error responses must serialise");
        assert_eq!(
            err,
            json!({
                "jsonrpc": "2.0",
                "id": null,
                "error": { "code": -32601, "message": "nope" }
            })
        );
    }

    /// The tool payload is streamed into the envelope's `text` field as a
    /// pretty-printed JSON string, without an intermediate `String` copy.
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
}
