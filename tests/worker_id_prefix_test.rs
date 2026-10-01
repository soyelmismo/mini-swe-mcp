//! Short worker ids (ux-U2): any unique prefix of at least three characters,
//! and `last` for the caller's most recently dispatched worker.
//!
//! Every id-taking verb goes through the same resolver, and resolution never
//! leaves the caller's own workers (H-3), so a prefix that only matches
//! another agent's worker reads as "not found" rather than leaking its id.

mod common;
use common::IsolatedPool;

use mini_swe_mcp::mcp::{ConnectionContext, McpServer};
use mini_swe_mcp::pool::{
    LogBuffer, WorkerIdLookup, WorkerMetrics, WorkerRecord, WorkerRegistryEntry, WorkerState,
    save_registry_entry_in,
};
use serde_json::json;

/// A connection that announced `agent`, as a hub client would.
fn agent_context(agent: &str) -> ConnectionContext {
    ConnectionContext {
        agent_id: Some(agent.to_string()),
        ..ConnectionContext::hub_connection(3)
    }
}

/// A running synthetic worker owned by `owner`.
fn running_worker(id: &str, owner: &str) -> WorkerRecord {
    WorkerRecord {
        id: id.to_string(),
        task: "task".to_string(),
        model: "ninja".to_string(),
        owner: owner.to_string(),
        state: WorkerState::Running {
            step: 1,
            last_command: "cargo test".to_string(),
            started_at: 0,
        },
        metrics: WorkerMetrics::default(),
        logs: LogBuffer::new(),
        pending_steer: Vec::new(),
        resume_tx: None,
        handle: None,
        revision: 0,
    }
}

/// A running registry row owned by `owner`, dispatched at `started_at`.
fn registry_row(id: &str, owner: &str, started_at: u64) -> WorkerRegistryEntry {
    WorkerRegistryEntry {
        model: "ninja".to_string(),
        step: 1,
        max_turns: 20,
        last_command: "cargo test".to_string(),
        started_at,
        updated_at: started_at,
        ..WorkerRegistryEntry::test_row(id, owner)
    }
}

/// A pool plus server over it, with no LLM anywhere in sight.
fn server() -> (IsolatedPool, McpServer) {
    let owned = IsolatedPool::new(4, "ux-u2");
    let server = McpServer::new(owned.pool.clone(), "ninja".to_string());
    (owned, server)
}

#[tokio::test]
async fn a_unique_prefix_resolves_and_the_reply_names_the_full_id() {
    let (owned, server) = server();
    owned
        .pool
        .__test_insert_worker(running_worker("aaa11111", "agent-a"))
        .await;
    owned
        .pool
        .__test_insert_worker(running_worker("bbb22222", "agent-a"))
        .await;
    let ctx = agent_context("agent-a");

    let status = server
        .execute_tool_for(
            "worker",
            json!({ "action": "status", "worker_id": "aaa" }),
            &ctx,
        )
        .await
        .expect("a unique prefix must resolve");
    assert_eq!(status["worker_id"], "aaa11111");

    // A longer prefix, and `id` as the argument name, resolve the same way.
    let status = server
        .execute_tool_for("worker", json!({ "action": "status", "id": "bbb2" }), &ctx)
        .await
        .expect("a unique prefix under `id` must resolve");
    assert_eq!(status["worker_id"], "bbb22222");

    // A mutating verb reports the full id it acted on, not the prefix.
    let killed = server
        .execute_tool_for(
            "worker",
            json!({ "action": "kill", "worker_id": "aaa1" }),
            &ctx,
        )
        .await
        .expect("kill must resolve the prefix");
    assert_eq!(killed["worker_id"], "aaa11111");
    assert_eq!(killed["killed"], true);
}

#[tokio::test]
async fn an_ambiguous_prefix_lists_only_the_callers_matches() {
    let (owned, server) = server();
    owned
        .pool
        .__test_insert_worker(running_worker("abcd1111", "agent-a"))
        .await;
    owned
        .pool
        .__test_insert_worker(running_worker("abcd2222", "agent-a"))
        .await;
    // Shares the prefix, but another agent owns it and must stay invisible.
    owned
        .pool
        .__test_insert_worker(running_worker("abcd3333", "agent-b"))
        .await;

    let error = server
        .execute_tool_for(
            "worker",
            json!({ "action": "status", "worker_id": "abcd" }),
            &agent_context("agent-a"),
        )
        .await
        .expect_err("an ambiguous prefix must be refused");
    let message = error.to_string();
    assert!(message.contains("ambiguous"), "{message}");
    assert!(
        message.contains("abcd1111") && message.contains("abcd2222"),
        "both of the caller's matches must be listed: {message}"
    );
    assert!(
        !message.contains("abcd3333"),
        "another agent's worker must not be listed: {message}"
    );
}

#[tokio::test]
async fn a_prefix_shorter_than_three_characters_is_not_found() {
    let (owned, server) = server();
    owned
        .pool
        .__test_insert_worker(running_worker("abcd1111", "agent-a"))
        .await;

    let error = server
        .execute_tool_for(
            "worker",
            json!({ "action": "status", "worker_id": "ab" }),
            &agent_context("agent-a"),
        )
        .await
        .expect_err("a two-character prefix must not resolve");
    assert!(
        error.to_string().contains("Worker not found: ab"),
        "{error}"
    );
}

#[tokio::test]
async fn a_prefix_of_only_another_agents_worker_is_not_found() {
    let (owned, server) = server();
    owned
        .pool
        .__test_insert_worker(running_worker("aaa11111", "agent-a"))
        .await;
    owned
        .pool
        .__test_insert_worker(running_worker("bbb22222", "agent-b"))
        .await;
    let ctx = agent_context("agent-a");

    let error = server
        .execute_tool_for(
            "worker",
            json!({ "action": "status", "worker_id": "bbb" }),
            &ctx,
        )
        .await
        .expect_err("a foreign prefix must not resolve");
    assert!(
        error.to_string().contains("Worker not found: bbb"),
        "{error}"
    );

    // The full id of a foreign worker is still recognised, and refused with
    // the owning agent's name rather than a bare "not found".
    let error = server
        .execute_tool_for(
            "worker",
            json!({ "action": "status", "worker_id": "bbb22222" }),
            &ctx,
        )
        .await
        .expect_err("a foreign full id must still be refused");
    assert!(
        error.to_string().contains("belongs to agent agent-b"),
        "{error}"
    );
}

#[tokio::test]
async fn last_names_the_callers_most_recently_dispatched_worker() {
    let (owned, server) = server();
    owned
        .pool
        .__test_insert_worker(running_worker("aaa11111", "agent-a"))
        .await;
    owned
        .pool
        .__test_insert_worker(running_worker("bbb22222", "agent-a"))
        .await;
    // Dispatched later, but by another agent: not `agent-a`'s last.
    owned
        .pool
        .__test_insert_worker(running_worker("ccc33333", "agent-b"))
        .await;
    let ctx = agent_context("agent-a");

    let status = server
        .execute_tool_for(
            "worker",
            json!({ "action": "status", "worker_id": "last" }),
            &ctx,
        )
        .await
        .expect("last must resolve");
    assert_eq!(status["worker_id"], "bbb22222");

    owned
        .pool
        .__test_insert_worker(running_worker("ddd44444", "agent-a"))
        .await;
    let status = server
        .execute_tool_for(
            "worker",
            json!({ "action": "status", "worker_id": "last" }),
            &ctx,
        )
        .await
        .expect("last must follow the newest dispatch");
    assert_eq!(status["worker_id"], "ddd44444");

    // `last` is per caller: agent-b's own newest is still ccc33333.
    let status = server
        .execute_tool_for(
            "worker",
            json!({ "action": "status", "worker_id": "last" }),
            &agent_context("agent-b"),
        )
        .await
        .expect("last resolves for every caller");
    assert_eq!(status["worker_id"], "ccc33333");
}

#[tokio::test]
async fn last_with_no_workers_of_the_caller_is_not_found() {
    let (_owned, server) = server();
    let error = server
        .execute_tool_for(
            "worker",
            json!({ "action": "status", "worker_id": "last" }),
            &agent_context("agent-a"),
        )
        .await
        .expect_err("last with none of the caller's workers must not resolve");
    assert!(
        error.to_string().contains("Worker not found: last"),
        "{error}"
    );
}

#[tokio::test]
async fn a_registry_only_worker_resolves_by_prefix_and_by_last() {
    let (owned, server) = server();
    let root = owned.root();
    save_registry_entry_in(&root, &registry_row("eee55555", "agent-a", 100));
    save_registry_entry_in(&root, &registry_row("eee66666", "agent-b", 200));
    let ctx = agent_context("agent-a");

    let status = server
        .execute_tool_for(
            "worker",
            json!({ "action": "status", "worker_id": "eee5" }),
            &ctx,
        )
        .await
        .expect("a registry-only prefix must resolve");
    assert_eq!(status["worker_id"], "eee55555");

    // The registry's `started_at` orders `last` too, and the newer row of
    // another agent is not a candidate.
    let status = server
        .execute_tool_for(
            "worker",
            json!({ "action": "status", "worker_id": "last" }),
            &ctx,
        )
        .await
        .expect("last must consider the registry");
    assert_eq!(status["worker_id"], "eee55555");
}

#[tokio::test]
async fn watch_accepts_a_prefix_and_last() {
    let (owned, server) = server();
    owned
        .pool
        .__test_insert_worker(running_worker("abc99999", "agent-a"))
        .await;
    let ctx = agent_context("agent-a");

    let by_prefix = server
        .execute_tool_for(
            "worker",
            json!({ "action": "watch", "worker_id": "abc9", "timeout_secs": 0 }),
            &ctx,
        )
        .await
        .expect("watch must resolve the prefix");
    assert_eq!(by_prefix["status"], "no_event");
    assert_eq!(by_prefix["watching"], json!(["abc99999"]));

    let by_last = server
        .execute_tool_for(
            "worker",
            json!({ "action": "watch", "worker_id": "last", "timeout_secs": 0 }),
            &ctx,
        )
        .await
        .expect("watch must resolve last");
    assert_eq!(by_last["status"], "no_event");
    assert_eq!(by_last["watching"], json!(["abc99999"]));
}

#[tokio::test]
async fn lookup_reports_ambiguity_with_the_callers_ids_only() {
    let owned = IsolatedPool::new(2, "ux-u2-lookup");
    owned
        .pool
        .__test_insert_worker(running_worker("abcd1111", "agent-a"))
        .await;
    owned
        .pool
        .__test_insert_worker(running_worker("abcd2222", "agent-a"))
        .await;
    owned
        .pool
        .__test_insert_worker(running_worker("abcd3333", "agent-b"))
        .await;

    assert_eq!(
        owned.pool.lookup_worker_id("abcd", "agent-a").await,
        WorkerIdLookup::Ambiguous(vec!["abcd1111".to_string(), "abcd2222".to_string()])
    );
    assert_eq!(
        owned.pool.lookup_worker_id("abc", "agent-b").await,
        WorkerIdLookup::Resolved("abcd3333".to_string())
    );
}
