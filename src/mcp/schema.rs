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
    "manifest", "prune",
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
            "Action to perform: 'dispatch' (spawn subagent; the reply carries 'watch_command', the exact shell command that waits on YOUR workers -- run it in the background, since a shell cannot know your session), 'status' (check step & progress), 'steer' (correct a completed worker, or continue any stopped one -- failed, interrupted, killed -- on its own id and branch with its full context; never dispatch a replacement for a stopped worker), 'watch' (block until one of your workers produces an event -- completion, failure, a question, or a stall -- and replay the ones you missed; this action is only for agents with no shell, so prefer running `mini-swe-mcp watch` in the background, and as the fallback pass 'timeout_secs' below your host's tool deadline and call it again on 'no_event'), 'collect' (final message plus a per-file diff stat; add 'full' for the whole diff, or 'files' for named paths only), 'review' (one compact view of a finished worker: task, verification, per-file diff stat, and whether its branch still merges cleanly into the base branch tip, ending with the command that acts on it), 'logs' (inspect a live worker's bounded step history without collecting it), 'list' (list all workers), 'kill' (terminate worker), 'reap' (evict expired terminal worker records), 'manifest' (models catalog), 'prune' (clean stale worktrees). A worker belongs to the agent that dispatched it: 'status', 'steer', 'kill', 'collect', 'review', 'logs', 'list' and 'watch' only ever see or act on your own workers; the admin override sees everything.",
        ),
    ),
    (
        "task",
        "string",
        DescriptionSource::Static(
            "ONE focused concern, naming the files in scope and the acceptance gate. Dispatch independent tasks in parallel: many workers at once is the intended use, and each integrates the latest base branch before completing. Split work so two workers do not rewrite the same function at once. Required for 'dispatch'.",
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
            "Target worker ID (alias: 'id'). Any unique prefix of at least 3 characters, or 'last' for your most recently dispatched worker, is accepted; the response always names the full ID. Required for 'status', 'steer', 'watch', 'collect', 'review', 'logs', and 'kill'.",
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
            "Send every correction and merge conflict to the same worker rather than editing its branch yourself. Steering guidance or follow-up instruction. Required for 'steer'. Steering corrects a completed worker or continues any stopped one (failed, interrupted, killed): it resumes on its own worker-<id> branch with the full conversation plus this message, on a fresh turn budget. Optional 'max_turns' sets that budget. Never dispatch a replacement for a stopped worker.",
        ),
    ),
    (
        "worker_ids",
        "array",
        DescriptionSource::Static(
            "Worker IDs to watch, each accepting the same prefixes and 'last' as 'worker_id'. Optional for 'watch': omitted watches every one of your own running or paused workers.",
        ),
    ),
    (
        "group",
        "string",
        DescriptionSource::Static("Only watch workers of this group. Optional for 'watch'."),
    ),
    (
        "timeout_secs",
        "integer",
        DescriptionSource::Static(
            "Client-side deadline in seconds for the blocking 'watch' action, which is a fallback for agents with no shell: prefer running `mini-swe-mcp watch` in the background. When the deadline expires before an event arrives, the call returns {status:'no_event'} instead of blocking, so the agent can simply call 'watch' again. Omit it to wait indefinitely. Hosts with a short tool deadline (opencode ~120 s, Antigravity CLI ~180 s, Hermes 300 s) should pass a value below their own limit, e.g. 90.",
        ),
    ),
    (
        "max_turns",
        "integer",
        DescriptionSource::Static(
            "Maximum bash exploration turns (overrides manifest default). Optional for 'dispatch'; on 'steer' it is the fresh turn budget when continuing a stopped worker (completed, failed, interrupted or killed; default 60), and is ignored for running or paused workers.",
        ),
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
        "verify",
        "string",
        DescriptionSource::Static(
            "Optional shell command run through the same sandboxed bash path before a completion sentinel is honoured (e.g. 'cargo clippy --all-targets -- -D warnings && cargo test'). When absent, the harness auto-detects from the repository layout (Cargo.toml, package.json with a test script, or pyproject.toml/pytest.ini). Pass an empty string to disable the gate. Optional for 'dispatch'.",
        ),
    ),
    (
        "scope",
        "string",
        DescriptionSource::Static(
            "Listing scope for 'list': omitted or 'mine' returns only the calling agent's workers, 'all' returns every agent's and requires the admin override. Optional for 'list' (default: 'mine').",
        ),
    ),
    (
        "full",
        "boolean",
        DescriptionSource::Static(
            "Return the whole diff instead of the default per-file diff stat. Optional for 'collect' (default: false); ignored when 'files' is given.",
        ),
    ),
    (
        "files",
        "array",
        DescriptionSource::Static(
            "Paths whose diff to return, so a review of one file never carries the rest. Optional for 'collect': omitted returns no diff at all, and 'full' overrides it.",
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
                "description": format!("{} {}", WORKER_TOOL_DESCRIPTION, crate::cli::watch::WORKFLOW),
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

    /// The tool description teaches the transport-neutral wait: the `watch`
    /// action over MCP (bounded by `timeout_secs`, re-called on `no_event`),
    /// the shell command, and the channel push notifications.
    #[test]
    fn tool_description_teaches_the_mcp_wait() {
        let tools_list = build_tools_list(&ModelManifest::default());
        let description = tools_list["tools"]
            .as_array()
            .and_then(|tools| tools.iter().find(|tool| tool["name"] == "worker"))
            .and_then(|tool| tool["description"].as_str())
            .expect("tools/list must expose the 'worker' tool description");

        for needle in [
            "mini-swe-mcp watch",
            "timeout_secs",
            "no_event",
            "push notifications",
        ] {
            assert!(
                description.contains(needle),
                "the tool description must mention {needle}: {description}"
            );
        }
    }

    /// `message` teaches that steer corrects a completed worker or continues
    /// any stopped one on its own branch; it never dispatches a replacement.
    #[test]
    fn message_description_covers_every_stopped_state() {
        let tools_list = build_tools_list(&ModelManifest::default());
        let schema = worker_schema(&tools_list);
        let text = schema["properties"]["message"]["description"]
            .as_str()
            .expect("the message property needs a description");

        for needle in [
            "completed",
            "failed",
            "interrupted",
            "killed",
            "worker-<id>",
            "max_turns",
        ] {
            assert!(
                text.contains(needle),
                "the message description must mention {needle}: {text}"
            );
        }
        assert!(
            !text.contains("finished (completed/failed)"),
            "steer continues any stopped state, not just finished ones: {text}"
        );
    }
}
