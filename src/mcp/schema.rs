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
    "dispatch", "status", "steer", "watch", "collect", "review", "logs", "list", "kill", "reap",
    "manifest", "prune", "merge",
];

/// Declared network policy for a dispatched worker.
///
/// `offline` runs every bash step inside an isolated network namespace (no
/// egress), `allow` keeps the host's connectivity. This is the advertised
/// `network` enum; [`NETWORK_DEFAULT`] is what a dispatch without the property
/// gets. Derived from [`crate::manifest::NETWORK_POLICIES`] so the MCP
/// vocabulary and the manifest vocabulary can never drift apart.
pub const NETWORK_MODES: &[&str] = crate::manifest::NETWORK_POLICIES;

/// Accepted values of the `list` `scope` property: the caller's own workers,
/// or every agent's. [`LIST_SCOPE_ALL`] is the only way to look at another
/// agent's rows and it requires the admin override; every other verb refuses
/// a foreign worker regardless (H-3).
pub const LIST_SCOPES: &[&str] = &["mine", "all"];

/// `scope` value that lists every agent's workers.
pub const LIST_SCOPE_ALL: &str = "all";

/// Policy applied when a `tools/call` omits the optional `network` property.
///
/// Backwards compatible: a client that never heard of the property keeps the
/// connected behaviour it had before isolation existed.
pub const NETWORK_DEFAULT: &str = "allow";

/// Description of the `worker` tool itself.
///
/// Kept to the rules an agent needs to call the tool correctly; the longer
/// guidance lives in `mini-swe-mcp help <topic>` (see [`crate::cli::help`]).
const WORKER_TOOL_DESCRIPTION: &str = "Manage autonomous SWE mini-agents in isolated Git worktrees. Wait with `mini-swe-mcp watch` in the background, or the 'watch' action bounded by 'timeout_secs' when you have no shell. You only see or act on your own workers; the admin override excepted. `mini-swe-mcp help <topic>` covers workflow, watch, steer, review, collect, merge, identity, sandbox, env.";

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
        DescriptionSource::Static("Action to perform; see `mini-swe-mcp help <topic>`."),
    ),
    (
        "task",
        "string",
        DescriptionSource::Static(
            "ONE focused concern: the files in scope and the acceptance gate. Required for 'dispatch'.",
        ),
    ),
    (
        "tasks",
        "array",
        DescriptionSource::Static(
            "Batch dispatch: list of {task, model?, ...} objects, one worker each; top-level values are defaults.",
        ),
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
            "Target worker ID (alias: 'id'); a unique prefix of 3+ characters or 'last' works. Required for 'status', 'steer', 'watch', 'collect', 'review', 'logs', 'kill'.",
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
            "Correction or follow-up for 'steer', which resumes the worker on its own branch with its full context (optional 'max_turns' sets the fresh budget). Required for 'steer'; also continues a stopped worker: never dispatch a replacement.",
        ),
    ),
    (
        "worker_ids",
        "array",
        DescriptionSource::Static(
            "Worker IDs to watch; each accepts the same prefixes and 'last' as 'worker_id'. Omitted watches every worker you own.",
        ),
    ),
    (
        "group",
        "string",
        DescriptionSource::Static(
            "Only workers of this group. Optional for 'watch' and 'merge --approved'.",
        ),
    ),
    (
        "role",
        "string",
        DescriptionSource::Static(
            "'consolidate': integrate this group's completed workers (requires 'group')",
        ),
    ),
    (
        "timeout_secs",
        "integer",
        DescriptionSource::Static(
            "Deadline in seconds for the blocking 'watch' action; on expiry it returns {status:'no_event'} so you can call it again. Omit to wait indefinitely.",
        ),
    ),
    (
        "max_turns",
        "integer",
        DescriptionSource::Static(
            "Maximum bash exploration turns (overrides the manifest default). On 'steer', the fresh budget when continuing a stopped worker.",
        ),
    ),
    (
        "temperature",
        "number",
        DescriptionSource::Static("Model sampling temperature (overrides the manifest default)."),
    ),
    (
        "review_after",
        "string",
        DescriptionSource::Static(
            "Optional reviewer model (e.g. 'nerd') that audits and finalizes the worktree after implementation.",
        ),
    ),
    (
        "verify",
        "string",
        DescriptionSource::Static(
            "Optional shell command run before a completion sentinel is honoured (e.g. 'cargo test'). Omit to auto-detect; pass an empty string to disable the gate.",
        ),
    ),
    (
        "scope",
        "string",
        DescriptionSource::Static(
            "Listing scope for 'list': 'mine' (default) or 'all' (every agent's; needs the admin override).",
        ),
    ),
    (
        "full",
        "boolean",
        DescriptionSource::Static("The whole diff. Optional for 'collect'."),
    ),
    (
        "files",
        "array",
        DescriptionSource::Static("Paths whose diff to return. Optional for 'collect'."),
    ),
    (
        "network",
        "string",
        DescriptionSource::Static(
            "Network policy: 'offline' isolates every bash step with no egress, 'allow' (default) keeps connectivity.",
        ),
    ),
    (
        "approved",
        "boolean",
        DescriptionSource::Static(
            "Merge every approved worker of the caller (optionally narrowed by 'group') with one              verify gate on the combined result. Optional for 'merge'.",
        ),
    ),
    (
        "keep_branch",
        "boolean",
        DescriptionSource::Static("Keep the branch."),
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
        schema.insert(
            "default".to_string(),
            Value::String(NETWORK_DEFAULT.to_string()),
        );
    }
    if name == "scope" {
        schema.insert(
            "enum".to_string(),
            Value::Array(
                LIST_SCOPES
                    .iter()
                    .map(|scope| Value::String((*scope).to_string()))
                    .collect(),
            ),
        );
        schema.insert(
            "default".to_string(),
            Value::String(LIST_SCOPES[0].to_string()),
        );
    }
    if name == "max_turns" {
        schema.insert("minimum".to_string(), Value::from(1));
        schema.insert(
            "maximum".to_string(),
            Value::from(crate::manifest::MAX_TURNS_LIMIT),
        );
    }
    if name == "timeout_secs" {
        schema.insert("minimum".to_string(), Value::from(0));
    }
    if name == "worker_ids" || name == "files" {
        schema.insert("items".to_string(), json!({ "type": "string" }));
    }
    if name == "tasks" {
        schema.insert(
            "items".to_string(),
            json!({
                "type": "object",
                "properties": {
                    "task": { "type": "string" },
                    "model": { "type": "string" },
                    "repo_path": { "type": "string" },
                    "max_turns": { "type": "integer" },
                    "verify": { "type": "string" },
                    "group": { "type": "string" },
                    "network": { "type": "string" },
                },
                "required": ["task"],
            }),
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

    /// Pre-trim size of the whole `tools/list` payload, in bytes, measured
    /// before the descriptions were shortened. The budget is 60% of it, i.e.
    /// at least a 40% cut.
    const TOOLS_LIST_BASELINE_BYTES: usize = 6990;

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

    /// The `scope` property advertises the two documented list scopes, so a
    /// client never has to guess the vocabulary.
    #[test]
    fn scope_property_advertises_the_list_scopes() {
        let tools_list = build_tools_list(&ModelManifest::default());
        let schema = worker_schema(&tools_list);
        let scope = &schema["properties"]["scope"];

        assert_eq!(scope["type"], json!("string"));
        assert_eq!(scope["enum"], json!(["mine", "all"]));
        assert_eq!(scope["default"], json!("mine"));
        assert!(LIST_SCOPES.contains(&LIST_SCOPE_ALL));
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

    /// Batch dispatch is advertised: `tasks` is an array of task objects, each
    /// requiring `task`, and it stays optional like every other dispatch
    /// property.
    #[test]
    fn tasks_property_advertises_the_batch_contract() {
        let tools_list = build_tools_list(&ModelManifest::default());
        let schema = worker_schema(&tools_list);
        let tasks = &schema["properties"]["tasks"];

        assert_eq!(tasks["type"], json!("array"));
        assert_eq!(tasks["items"]["type"], json!("object"));
        assert_eq!(tasks["items"]["required"], json!(["task"]));
        for key in [
            "task",
            "model",
            "repo_path",
            "max_turns",
            "verify",
            "group",
            "network",
        ] {
            assert!(
                tasks["items"]["properties"].get(key).is_some(),
                "the items schema must document '{key}': {tasks}"
            );
        }
    }

    /// The tool description stays a calling contract: the waiting rule, the
    /// ownership rule, and a one-line pointer to the long-form topics.
    #[test]
    fn tool_description_names_the_rules_and_points_at_help_topics() {
        let tools_list = build_tools_list(&ModelManifest::default());
        let description = tools_list["tools"]
            .as_array()
            .and_then(|tools| tools.iter().find(|tool| tool["name"] == "worker"))
            .and_then(|tool| tool["description"].as_str())
            .expect("tools/list must expose the 'worker' tool description");

        for needle in [
            "mini-swe-mcp watch",
            "mini-swe-mcp help <topic>",
            "own workers",
        ] {
            assert!(
                description.contains(needle),
                "the tool description must mention {needle}: {description}"
            );
        }
        for topic in crate::cli::help::TOPICS {
            assert!(
                description.contains(topic),
                "the tool description must point at the '{topic}' topic: {description}"
            );
        }
    }

    /// Regression budget: this payload is context every MCP agent pays on
    /// every session, so it must stay at least 40% below the pre-trim size.
    #[test]
    fn tools_list_stays_within_its_context_budget() {
        let tools_list = build_tools_list(&ModelManifest::default());
        let bytes = serde_json::to_vec(&tools_list)
            .expect("tools/list serialises")
            .len();
        assert!(
            bytes * 10 <= TOOLS_LIST_BASELINE_BYTES * 6,
            "tools/list grew to {bytes} bytes; budget is 60% of the {TOOLS_LIST_BASELINE_BYTES}-byte pre-trim payload"
        );
    }

    /// `message` stays a short call contract; the full list of stopped states
    /// it can continue lives in the `steer` help topic (tested in `cli::help`).
    #[test]
    fn message_description_points_at_steer() {
        let tools_list = build_tools_list(&ModelManifest::default());
        let schema = worker_schema(&tools_list);
        let text = schema["properties"]["message"]["description"]
            .as_str()
            .expect("the message property needs a description");

        assert!(text.contains("steer"), "{text}");
        assert!(text.contains("own branch"), "{text}");
        assert!(text.contains("never dispatch a replacement"), "{text}");
    }
}
