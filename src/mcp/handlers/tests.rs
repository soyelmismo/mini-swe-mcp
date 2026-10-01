use super::*;

/// An omitted `timeout_secs` means "wait indefinitely"; a non-integer one
/// is a hard error, because a dropped deadline is the unbounded hang the
/// argument exists to prevent.
#[test]
fn timeout_absent_is_unbounded_and_a_non_integer_is_rejected() {
    assert_eq!(
        McpServer::get_timeout(&json!({ "action": "wait" }), "wait")
            .expect("an absent deadline is not an error"),
        None
    );
    assert_eq!(
        McpServer::get_timeout(&json!({ "timeout_secs": 90 }), "wait")
            .expect("a whole number of seconds is accepted"),
        Some(std::time::Duration::from_secs(90))
    );
    let err = McpServer::get_timeout(&json!({ "timeout_secs": "90" }), "wait")
        .expect_err("a string deadline must not be accepted");
    assert!(
        err.to_string()
            .contains("'timeout_secs' must be a non-negative integer"),
        "{err}"
    );
}

/// An omitted `network` yields `None`: the caller falls back to the
/// manifest policy, then the runtime default.
#[test]
fn network_absent_yields_none() {
    let args = json!({ "action": "dispatch", "task": "t" });
    assert_eq!(
        McpServer::get_network_offline(&args, "dispatch").expect("absent is not an error"),
        None,
        "an omitted network policy must defer to the manifest/default"
    );
}

/// An explicit `offline` is the opt-in that turns isolation on, and
/// `allow` is the explicit spelling of the default.
#[test]
fn network_offline_and_allow_are_both_accepted() {
    assert_eq!(
        McpServer::get_network_offline(&json!({ "network": "offline" }), "dispatch")
            .expect("offline must be accepted"),
        Some(true)
    );
    assert_eq!(
        McpServer::get_network_offline(&json!({ "network": "allow" }), "dispatch")
            .expect("allow must be accepted"),
        Some(false)
    );
}

/// An unknown policy (or a non-string) is rejected instead of silently
/// falling back: a caller that asked for isolation must never silently get
/// connectivity instead.
#[test]
fn an_unknown_network_policy_is_a_hard_error() {
    let err = McpServer::get_network_offline(&json!({ "network": "offine" }), "dispatch")
        .expect_err("a typo must not be accepted");
    assert!(
        err.to_string().contains("not a valid 'network' policy"),
        "{err}"
    );

    let err = McpServer::get_network_offline(&json!({ "network": true }), "dispatch")
        .expect_err("a non-string network must not be accepted");
    assert!(err.to_string().contains("must be a string"), "{err}");
}

/// An explicit argument wins over the manifest policy.
#[test]
fn explicit_network_argument_wins_over_manifest() {
    let manifest = ModelManifest::default();
    // ninja declares `allow` in the built-in manifest.
    assert!(
        McpServer::resolve_network_policy(
            &json!({ "network": "offline" }),
            "dispatch",
            &manifest,
            "combo:ninja",
        )
        .expect("explicit offline must win"),
        "an explicit offline must override the manifest's allow"
    );
    assert!(
        !McpServer::resolve_network_policy(
            &json!({ "network": "allow" }),
            "dispatch",
            &manifest,
            "combo:ninja",
        )
        .expect("explicit allow must win"),
        "an explicit allow must override the manifest's allow"
    );
}

/// When the argument is omitted, the resolved model's manifest policy
/// applies.
#[test]
fn manifest_policy_applies_when_argument_omitted() {
    let manifest = ModelManifest::default();
    // ninja declares `allow` in the built-in manifest.
    assert!(
        !McpServer::resolve_network_policy(&json!({}), "dispatch", &manifest, "combo:ninja",)
            .expect("manifest policy must apply"),
        "ninja's manifest policy is allow"
    );
}

/// A manifest that declares `offline` isolates the worker when the
/// argument is omitted.
#[test]
fn manifest_offline_policy_isolates_when_argument_omitted() {
    let mut manifest = ModelManifest::default();
    manifest.models.insert(
        "sealed".to_string(),
        crate::manifest::ModelDefinition {
            id: "vendor:sealed".to_string(),
            role: None,
            temperature: None,
            max_turns: None,
            policy: Some(crate::manifest::ExecutionPolicy {
                network: Some(NetworkPolicy::Offline),
            }),
        },
    );
    assert!(
        McpServer::resolve_network_policy(&json!({}), "dispatch", &manifest, "vendor:sealed",)
            .expect("manifest offline must apply"),
        "a model declaring offline must isolate the worker"
    );
}

/// The `watch_command` is only useful to a caller with no watch running: once
/// this identity holds the hub's one watch slot the field is dropped, while a
/// watch held by another identity does not suppress it.
#[tokio::test]
async fn watch_command_is_omitted_only_while_the_callers_own_watch_runs() {
    use crate::mcp::server::ConnectionContext;
    use crate::pool::{LogBuffer, WorkerMetrics, WorkerRecord, WorkerState};

    let dir = std::env::temp_dir().join(format!("mini-swe-watch-line-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create the token directory");

    let server = McpServer::new(
        crate::pool::WorkerPool::new(1, "http://localhost:1".to_string(), "test-key".to_string()),
        "ninja".to_string(),
    );
    server
        .pool
        .__test_insert_worker(WorkerRecord {
            id: "watch-line-1".to_string(),
            task: "probe".to_string(),
            model: "test".to_string(),
            owner: "orchestrator".to_string(),
            state: WorkerState::Running {
                step: 1,
                last_command: "probe".to_string(),
                started_at: 0,
            },
            metrics: WorkerMetrics::default(),
            logs: LogBuffer::new(),
            pending_steer: Vec::new(),
            resume_tx: None,
            handle: None,
            revision: 0,
        })
        .await;

    let mut ctx = ConnectionContext::hub_connection(1).with_watch_tokens(std::sync::Arc::new(
        crate::hub::WatchTokens::new(dir.clone()),
    ));
    ctx.agent_id = Some("orchestrator".to_string());
    async fn steer(server: &McpServer, ctx: &ConnectionContext) -> Value {
        server
            .execute_tool_for(
                "worker",
                json!({"action": "steer", "worker_id": "watch-line-1", "message": "go"}),
                ctx,
            )
            .await
            .expect("steer a seeded worker")
    }

    // No watch running: the answer hands the caller the command.
    let no_watch = steer(&server, &ctx).await;
    assert!(
        no_watch["watch_command"].is_string(),
        "a caller with no watch must be told how to start one: {no_watch}"
    );

    // Another identity's watch is not this caller's: the command stays.
    let other = server
        .hub_events
        .lock()
        .await
        .begin_watch("someone-else", 2, None)
        .expect("another identity claims its own slot");
    let foreign = steer(&server, &ctx).await;
    assert!(
        foreign["watch_command"].is_string(),
        "another identity's watch must not silence this caller: {foreign}"
    );
    drop(other);

    // This caller's own watch: the hub enforces one watch per session and the
    // running one delivers the next event, so the command is redundant.
    let own = server
        .hub_events
        .lock()
        .await
        .begin_watch("orchestrator", 1, None)
        .expect("this identity claims the slot");
    let watching = steer(&server, &ctx).await;
    assert!(
        watching.get("watch_command").is_none(),
        "a running watch makes the command redundant: {watching}"
    );
    drop(own);

    let _ = std::fs::remove_dir_all(&dir);
}

/// When neither the argument nor the manifest declares a policy, the
/// runtime default (`allow`) applies.
#[test]
fn runtime_default_applies_when_nothing_declared() {
    let manifest = ModelManifest::default();
    // An unknown model has no manifest entry, so no policy is declared.
    assert!(
            !McpServer::resolve_network_policy(
                &json!({}),
                "dispatch",
                &manifest,
                "some/unknown-model",
            )
            .expect("default must apply"),
            "an undeclared policy must fall back to allow"
        );
}
