//! Automatic round consolidation uses the dispatch contract for CLI and MCP.
mod common;

use mini_swe_mcp::cli::args::tool_args;
use serde_json::json;

#[test]
fn dispatch_accepts_auto_consolidation_flags() {
    for (flag, expected) in [
        ("--consolidate", json!(true)),
        ("--consolidate=nerd", json!("nerd")),
    ] {
        let argv = [
            "mini-swe-mcp",
            "dispatch",
            "task",
            "--group",
            "round",
            flag,
            "--consolidate-verify",
            "cargo test",
        ]
        .map(str::to_string);
        let args = tool_args("dispatch", &argv, true).unwrap().unwrap();
        assert_eq!(args.get("consolidate"), Some(&expected));
        assert_eq!(args.get("consolidate_verify"), Some(&json!("cargo test")));
    }
}

use mini_swe_mcp::manifest::ModelManifest;
use mini_swe_mcp::mcp::{ConnectionContext, McpServer};
use mini_swe_mcp::pool::{RegistryStatus, WorkerPool, WorkerRole, load_all_registry_entries_in};
use mini_swe_mcp::worktree::ScratchRoot;
use std::sync::Arc;
use std::time::Duration;
use tokio::io::AsyncWriteExt;
use tokio::net::TcpListener;
use tokio::sync::Semaphore;

/// Slow workers park before their first response, so release is deterministic
/// and a fast worker cannot accidentally finish after the slow one.
async fn scripted_workers() -> (String, Arc<Semaphore>, tokio::task::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let gate = Arc::new(Semaphore::new(0));
    let release = gate.clone();
    let task = tokio::spawn(async move {
        loop {
            let (mut socket, _) = listener.accept().await.unwrap();
            let gate = release.clone();
            tokio::spawn(async move {
                let body = common::fake_llm::read_request(&mut socket).await.unwrap();
                let request: serde_json::Value = serde_json::from_str(&body).unwrap();
                let model = request["model"].as_str().unwrap();
                let turn = request["messages"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .filter(|m| m["role"] == "tool")
                    .count();
                if model == "slow" && turn == 0 {
                    gate.acquire().await.unwrap().forget();
                }
                let command = if turn == 0 {
                    format!("echo contribution > {model}.txt")
                } else {
                    "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT".into()
                };
                let delta = json!({"choices":[{"delta":{"tool_calls":[{
                    "index":0,"id":format!("call-{turn}"),"function":{
                        "name":"bash","arguments":json!({"command":command}).to_string()
                    }
                }]}}]});
                let reply = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\ndata: {delta}\n\ndata: [DONE]\n\n"
                );
                socket.write_all(reply.as_bytes()).await.unwrap();
            });
        }
    });
    (url, gate, task)
}

fn server(root: &ScratchRoot, url: &str) -> McpServer {
    let pool = WorkerPool::with_scratch(8, url.into(), "test-key".into(), root.clone())
        .with_manifest(Arc::new(ModelManifest::default()));
    McpServer::new(pool, "integrator".into())
}

async fn wait_status(root: &ScratchRoot, id: &str, status: RegistryStatus) {
    tokio::time::timeout(Duration::from_secs(30), async {
        loop {
            if load_all_registry_entries_in(root)
                .iter()
                .any(|e| e.id == id && e.status == status)
            {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap_or_else(|_| {
        panic!(
            "worker {id} did not reach {status:?}: {:?}",
            load_all_registry_entries_in(root)
        )
    });
}

async fn wait_consolidators(root: &ScratchRoot, count: usize) -> Vec<String> {
    tokio::time::timeout(Duration::from_secs(30), async {
        loop {
            let ids: Vec<_> = load_all_registry_entries_in(root)
                .into_iter()
                .filter(|e| e.role == WorkerRole::Consolidate)
                .map(|e| e.id)
                .collect();
            assert!(ids.len() <= count, "duplicate consolidator: {ids:?}");
            if ids.len() == count {
                break ids;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("automatic consolidator dispatched")
}

#[tokio::test]
async fn consolidates_once_after_last_worker_and_survives_restart() {
    let scratch = common::TempDir::new_in_tmp("auto-round");
    let repo = scratch.subdir("repo");
    common::git(&repo, &["init", "-b", "main"]);
    common::git(&repo, &["config", "user.email", "test@example.test"]);
    common::git(&repo, &["config", "user.name", "Test"]);
    std::fs::write(repo.join("README"), "base\n").unwrap();
    common::git(&repo, &["add", "."]);
    common::git(&repo, &["commit", "-m", "base"]);
    let root = ScratchRoot::new(scratch.subdir("workers"));
    let hub = scratch.subdir("hub");
    let (url, slow, llm) = scripted_workers().await;
    let first = server(&root, &url);
    let events = first.start_hub_events(Some(&hub)).await;
    let scheduler = first.start_auto_consolidate(hub.clone()).await.unwrap();
    let mut ctx = ConnectionContext::stdio();
    ctx.agent_id = Some("round-owner".into());
    let result = first
        .execute_tool_for(
            "worker",
            json!({
                "action":"dispatch","repo_path":repo,"group":"round",
                "consolidate":true,"consolidate_verify":"", "verify":"", "max_turns":6,
                "tasks":[{"task":"fast contribution","model":"fast"},
                         {"task":"slow contribution","model":"slow"}]
            }),
            &ctx,
        )
        .await
        .unwrap();
    assert_eq!(result["failed"], 0, "{result}");
    let fast_id = result["workers"][0]["worker_id"].as_str().unwrap();
    let slow_id = result["workers"][1]["worker_id"].as_str().unwrap();
    wait_status(&root, fast_id, RegistryStatus::Completed).await;
    assert!(
        load_all_registry_entries_in(&root)
            .iter()
            .all(|e| e.role != WorkerRole::Consolidate)
    );
    assert_eq!(
        first.pool().get_worker_state(slow_id).await.unwrap().name(),
        "Running"
    );

    // Stop only the scheduler and reconstruct the daemon's server over the same
    // explicit hub and registry. The still-running worker keeps its live owner.
    scheduler.abort();
    let _ = scheduler.await;
    events.abort();
    let second = server(&root, &url);
    let events = second.start_hub_events(Some(&hub)).await;
    let scheduler = second.start_auto_consolidate(hub.clone()).await.unwrap();
    slow.add_permits(1);
    wait_status(&root, slow_id, RegistryStatus::Completed).await;
    let ids = wait_consolidators(&root, 1).await;
    wait_status(&root, &ids[0], RegistryStatus::Completed).await;
    let stored: serde_json::Value =
        serde_json::from_slice(&std::fs::read(hub.join("auto-consolidate.json")).unwrap()).unwrap();
    assert_eq!(stored[0]["consumed"], true);
    scheduler.abort();
    let _ = scheduler.await;
    events.abort();

    let third = server(&root, &url);
    let scheduler = third.start_auto_consolidate(hub.clone()).await.unwrap();
    let next = third
        .execute_tool_for(
            "worker",
            json!({"action":"dispatch", "task":"next round",
                "repo_path":repo,"group":"round","model":"slow","verify":"","max_turns":6
            }),
            &ctx,
        )
        .await
        .unwrap();
    assert_eq!(
        load_all_registry_entries_in(&root)
            .iter()
            .filter(|e| e.role == WorkerRole::Consolidate)
            .count(),
        1
    );
    slow.add_permits(1);
    wait_status(
        &root,
        next["worker_id"].as_str().unwrap(),
        RegistryStatus::Completed,
    )
    .await;
    let ids = wait_consolidators(&root, 2).await;
    for id in ids {
        wait_status(&root, &id, RegistryStatus::Completed).await;
    }
    scheduler.abort();
    events.abort();
    llm.abort();
}
