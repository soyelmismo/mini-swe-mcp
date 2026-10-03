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
            instructions: None,
        },
    );
    assert!(
        McpServer::resolve_network_policy(&json!({}), "dispatch", &manifest, "vendor:sealed",)
            .expect("manifest offline must apply"),
        "a model declaring offline must isolate the worker"
    );
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
