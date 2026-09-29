//! The advertised `worker` tool contract: verb list, property table, schema.
//!
//! Everything an MCP client sees in `tools/list` derives from the tables below,
//! so the schema can never drift from what the dispatcher in
//! [`crate::mcp::handlers`] actually implements.

use serde_json::{Map, Value, json};
use std::borrow::Cow;

use crate::manifest::ModelManifest;

/// Every verb accepted by the single `worker` tool.
///
/// The *only* place the action list is spelled out: the `tools/list` schema
/// enum derives from it, the dispatcher matches on it, and the CLI's
/// "did you mean …?" hint reuses it. Adding a verb touches one constant.
pub const WORKER_ACTIONS: &[&str] = &[
    "dispatch", "status", "steer", "collect", "logs", "list", "kill", "reap", "manifest", "prune",
];

/// Declared network policy for a dispatched worker.
///
/// `offline` runs every bash step inside an isolated network namespace (no
/// egress), `allow` keeps the host's connectivity. This is the advertised
/// `network` enum; [`NETWORK_DEFAULT`] is what a dispatch without the property
/// gets.
pub const NETWORK_MODES: &[&str] = &["offline", "allow"];

/// Policy applied when a `tools/call` omits the optional `network` property.
///
/// Backwards compatible: a client that never heard of the property keeps the
/// connected behaviour it had before isolation existed.
pub const NETWORK_DEFAULT: &str = "allow";

/// Description of the `worker` tool itself.
const WORKER_TOOL_DESCRIPTION: &str = "Manage autonomous SWE mini-agents. Dispatches subagents in isolated Git worktrees, checks progress, injects steering instructions, retrieves git diffs, or inspects models.";

/// Where the `description` of an `inputSchema` property comes from.
enum DescriptionSource {
    /// A compile-time constant baked into [`WORKER_PROPERTIES`].
    Static(&'static str),
    /// Rendered at runtime; currently only the `model` property, whose text is
    /// produced by [`ModelManifest::build_tool_description`].
    Dynamic,
}

/// The `inputSchema` of the `worker` tool, expressed as data.
///
/// Each row is `(json_name, json_type, description)` — the same name the
/// handlers read back with `args.get(json_name)`. Building the schema from this
/// table keeps the advertised contract next to the dispatch table instead of a
/// 60-line `json!` literal.
const WORKER_PROPERTIES: &[(&str, &str, DescriptionSource)] = &[
    (
        "action",
        "string",
        DescriptionSource::Static(
            "Action to perform: 'dispatch' (spawn subagent), 'status' (check step & progress), 'steer' (inject follow-up instruction), 'collect' (get final diff), 'logs' (inspect a live worker's bounded step history without collecting it), 'list' (list all workers), 'kill' (terminate worker), 'reap' (evict expired terminal worker records), 'manifest' (models catalog), 'prune' (clean stale worktrees). For unattended tracking, poll 'status' or pass wait:true; avoid short-interval busy-waiting.",
        ),
    ),
    (
        "task",
        "string",
        DescriptionSource::Static("Task description or bug to fix. Required for 'dispatch'."),
    ),
    (
        "repo_path",
        "string",
        DescriptionSource::Static(
            "Absolute path to repository root (alias: 'path'). Required for 'dispatch'.",
        ),
    ),
    (
        "path",
        "string",
        DescriptionSource::Static("Alias for repo_path."),
    ),
    (
        "model",
        "string",
        // Dynamic: renders the manifest's alias -> role catalogue.
        DescriptionSource::Dynamic,
    ),
    (
        "worker_id",
        "string",
        DescriptionSource::Static(
            "Target worker ID (alias: 'id'). Required for 'status', 'steer', 'collect', 'logs', and 'kill'.",
        ),
    ),
    (
        "id",
        "string",
        DescriptionSource::Static("Alias for worker_id."),
    ),
    (
        "message",
        "string",
        DescriptionSource::Static(
            "Steering guidance or follow-up instruction. Required for 'steer'.",
        ),
    ),
    (
        "wait",
        "boolean",
        DescriptionSource::Static(
            "If true, blocks until worker completes and returns final diff immediately. Optional for 'dispatch' (default: false). Recommended for unattended single-worker runs to avoid manual polling loops.",
        ),
    ),
    (
        "max_turns",
        "integer",
        DescriptionSource::Static("Maximum bash exploration turns (overrides manifest default)."),
    ),
    (
        "temperature",
        "number",
        DescriptionSource::Static("Model sampling temperature (overrides manifest default)."),
    ),
    (
        "review_after",
        "string",
        DescriptionSource::Static(
            "Optional reviewer model (e.g. 'nerd') to automatically audit and finalize the worktree after implementation completes, using a fresh context window.",
        ),
    ),
    (
        "network",
        "string",
        DescriptionSource::Static(
            "Declarative network policy for the worker: 'offline' runs every bash step in an isolated network namespace with no egress (useful for pure refactor/analysis tasks), 'allow' keeps normal connectivity. Optional for 'dispatch' (default: 'allow').",
        ),
    ),
];

/// Render one table row as a JSON Schema property object.
fn property_schema(name: &str, json_type: &str, description: &str) -> Value {
    let mut schema = Map::new();
    schema.insert("type".to_string(), Value::String(json_type.to_string()));
    schema.insert(
        "description".to_string(),
        Value::String(description.to_string()),
    );
    if name == "action" {
        schema.insert(
            "enum".to_string(),
            Value::Array(
                WORKER_ACTIONS
                    .iter()
                    .map(|action| Value::String((*action).to_string()))
                    .collect(),
            ),
        );
    }
    if name == "network" {
        schema.insert(
            "enum".to_string(),
            Value::Array(
                NETWORK_MODES
                    .iter()
                    .map(|mode| Value::String((*mode).to_string()))
                    .collect(),
            ),
        );
        schema.insert("default".to_string(), Value::String(NETWORK_DEFAULT.to_string()));
    }
    if name == "max_turns" {
        schema.insert("minimum".to_string(), Value::from(1));
        schema.insert(
            "maximum".to_string(),
            Value::from(crate::manifest::MAX_TURNS_LIMIT),
        );
    }
    if name == "temperature" {
        schema.insert(
            "minimum".to_string(),
            Value::from(*crate::manifest::TEMPERATURE_RANGE.start()),
        );
        schema.insert(
            "maximum".to_string(),
            Value::from(*crate::manifest::TEMPERATURE_RANGE.end()),
        );
    }
    Value::Object(schema)
}

/// Build the whole `tools/list` result from the tables above.
///
/// Only the `model` description is dynamic (rendered from the model manifest);
/// every other key is a compile-time constant.
pub(super) fn build_tools_list(manifest: &ModelManifest) -> Value {
    let model_description = manifest.build_tool_description();
    let mut properties = Map::new();
    for (name, json_type, source) in WORKER_PROPERTIES {
        let description = match source {
            DescriptionSource::Static(text) => Cow::Borrowed(*text),
            DescriptionSource::Dynamic => Cow::Borrowed(model_description.as_str()),
        };
        properties.insert(
            (*name).to_string(),
            property_schema(name, json_type, description.as_ref()),
        );
    }

    json!({
        "tools": [
            {
                "name": "worker",
                "description": WORKER_TOOL_DESCRIPTION,
                "inputSchema": {
                    "type": "object",
                    "properties": Value::Object(properties),
                    "required": ["action"]
                }
            }
        ]
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn worker_schema(tools_list: &Value) -> &Value {
        tools_list["tools"]
            .as_array()
            .and_then(|tools| {
                tools
                    .iter()
                    .find(|tool| tool["name"] == "worker")
                    .map(|tool| &tool["inputSchema"])
            })
            .expect("tools/list must expose the 'worker' tool")
    }

    /// The advertised `action` enum is the dispatch table, not a second copy.
    #[test]
    fn action_enum_is_derived_from_the_dispatch_table() {
        let tools_list = build_tools_list(&ModelManifest::default());
        let schema = worker_schema(&tools_list);
        let actions: Vec<&str> = schema["properties"]["action"]["enum"]
            .as_array()
            .expect("the action property must carry an enum")
            .iter()
            .map(|value| value.as_str().expect("enum entries must be strings"))
            .collect();

        assert_eq!(actions, WORKER_ACTIONS.to_vec());
    }

    /// The property table *is* the schema: every row is typed and documented.
    #[test]
    fn property_table_renders_every_row() {
        let tools_list = build_tools_list(&ModelManifest::default());
        let schema = worker_schema(&tools_list);
        let properties = schema["properties"]
            .as_object()
            .expect("the input schema must expose a properties object");

        assert_eq!(properties.len(), WORKER_PROPERTIES.len());
        for (name, json_type, _) in WORKER_PROPERTIES {
            let property = &properties[*name];
            assert_eq!(property["type"], *json_type, "wrong type for '{name}'");
            assert!(
                property["description"]
                    .as_str()
                    .is_some_and(|description| !description.is_empty()),
                "'{name}' needs a non-empty description"
            );
        }
        assert_eq!(schema["required"], json!(["action"]));
    }

    /// The `network` property advertises exactly the accepted policies and the
    /// documented default, so a client never has to guess the vocabulary.
    #[test]
    fn network_property_advertises_its_enum_and_default() {
        let tools_list = build_tools_list(&ModelManifest::default());
        let schema = worker_schema(&tools_list);
        let network = &schema["properties"]["network"];

        assert_eq!(network["type"], json!("string"));
        assert_eq!(
            network["enum"],
            json!(["offline", "allow"]),
            "the network enum is the documented policy vocabulary"
        );
        assert_eq!(network["default"], json!(NETWORK_DEFAULT));
        assert_eq!(NETWORK_DEFAULT, "allow");
        assert!(
            network["description"]
                .as_str()
                .is_some_and(|text| text.contains("offline")),
            "the description must document what 'offline' does"
        );
    }

    /// `network` is optional: a dispatch that omits it must still validate
    /// against the advertised schema.
    #[test]
    fn network_is_not_required() {
        let tools_list = build_tools_list(&ModelManifest::default());
        let schema = worker_schema(&tools_list);
        assert_eq!(schema["required"], json!(["action"]));
        assert!(
            !schema["required"]
                .as_array()
                .expect("required is an array")
                .iter()
                .any(|entry| entry == "network"),
            "network must stay optional so existing callers are unaffected"
        );
    }
}
