//! Live terminal supervisor and dashboard monitor for `mini-swe-mcp`.
//!
//! Provides an interactive, in-place overwriting TUI that monitors running
//! agent swarms and isolated worker pods without spamming log lines.

use crate::pool::{WorkerRegistryEntry, load_all_registry_entries, unix_timestamp};
use anyhow::Result;
use std::collections::BTreeMap;
use std::io::{IsTerminal, Write};

fn format_duration(secs: u64) -> String {
    if secs < 60 {
        format!("{secs:02}s")
    } else if secs < 3600 {
        format!("{:02}m {:02}s", secs / 60, secs % 60)
    } else {
        format!("{:02}h {:02}m", secs / 3600, (secs % 3600) / 60)
    }
}

fn truncate_str(s: &str, max_len: usize) -> &str {
    if s.len() <= max_len {
        s
    } else {
        let cut = s.floor_char_boundary(max_len.saturating_sub(3));
        &s[..cut]
    }
}

pub fn render_dashboard(entries: &[WorkerRegistryEntry], now: u64, use_color: bool) -> String {
    let mut out = String::new();

    let mut active = 0;
    let mut paused = 0;
    let mut completed = 0;
    let mut failed = 0;
    let mut stopped = 0;

    // Group entries by swarm/group name
    let mut groups: BTreeMap<&str, Vec<&WorkerRegistryEntry>> = BTreeMap::new();
    for entry in entries {
        match entry.status.as_str() {
            "running" => active += 1,
            "paused" => paused += 1,
            "completed" => completed += 1,
            "failed" => failed += 1,
            _ => stopped += 1,
        }
        let grp = entry.group.as_deref().unwrap_or("default");
        groups.entry(grp).or_default().push(entry);
    }

    let total = entries.len();
    let time_str = {
        let secs = now % 60;
        let mins = (now / 60) % 60;
        let hours = (now / 3600) % 24;
        format!("{hours:02}:{mins:02}:{secs:02}")
    };

    // Header bar
    if use_color {
        out.push_str(&format!(
            "\x1b[1;36m── MINI-SWE SWARM SUPERVISOR ─────────────────────────────────── \x1b[1;37m{time_str}\x1b[0m ──\n"
        ));
        out.push_str(&format!(
            "Active: \x1b[1;32m{active}\x1b[0m  |  Paused: \x1b[1;33m{paused}\x1b[0m  |  Completed: \x1b[1;34m{completed}\x1b[0m  |  Failed: \x1b[1;31m{failed}\x1b[0m  |  Stopped: \x1b[2;37m{stopped}\x1b[0m  |  Total: \x1b[1m{total}\x1b[0m\n"
        ));
    } else {
        out.push_str(&format!(
            "── MINI-SWE SWARM SUPERVISOR ─────────────────────────────────── {time_str} ──\n"
        ));
        out.push_str(&format!(
            "Active: {active}  |  Paused: {paused}  |  Completed: {completed}  |  Failed: {failed}  |  Stopped: {stopped}  |  Total: {total}\n"
        ));
    }
    out.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n");

    if entries.is_empty() {
        out.push_str("No active or recent workers found in /tmp/swe-registry.\n");
        out.push_str("Waiting for workers to dispatch... (Press Ctrl+C to exit)\n");
        return out;
    }

    for (group_name, group_workers) in groups {
        let count = group_workers.len();
        if use_color {
            out.push_str(&format!("\x1b[1;35m[SWARM: {group_name}]\x1b[0m ({count} workers)\n"));
        } else {
            out.push_str(&format!("[SWARM: {group_name}] ({count} workers)\n"));
        }

        out.push_str("ID        PID     STATUS     TURNS     UPTIME   MODEL    LAST OP / TASK\n");
        out.push_str("─────────────────────────────────────────────────────────────────────────────────────\n");

        for w in group_workers {
            let id = if w.id.len() > 8 { &w.id[..8] } else { &w.id };
            let pid = format!("{:<7}", w.pid);
            let turns = format!("{}/{}", w.step, w.max_turns);
            let uptime = format_duration(now.saturating_sub(w.started_at));
            let model = &w.model;

            let status_colored = if use_color {
                match w.status.as_str() {
                    "running" => "\x1b[1;32mRUNNING \x1b[0m",
                    "paused" => "\x1b[1;33mPAUSED  \x1b[0m",
                    "completed" => "\x1b[1;34mDONE    \x1b[0m",
                    "failed" => "\x1b[1;31mFAILED  \x1b[0m",
                    _ => "\x1b[2;37mSTOPPED \x1b[0m",
                }
            } else {
                match w.status.as_str() {
                    "running" => "RUNNING ",
                    "paused" => "PAUSED  ",
                    "completed" => "DONE    ",
                    "failed" => "FAILED  ",
                    _ => "STOPPED ",
                }
            };

            let first_line = w.task.lines().next().unwrap_or("").trim();
            let op_task = if let Some(ref q) = w.question {
                format!("ASK: {q}")
            } else if !w.last_command.is_empty()
                && w.last_command != "completed"
                && w.last_command != "initializing"
            {
                format!("{} — {}", w.last_command, first_line)
            } else {
                first_line.to_string()
            };

            let op_clean = if op_task.len() > 36 {
                format!("{}...", truncate_str(&op_task, 36))
            } else {
                op_task
            };

            out.push_str(&format!(
                "{:<8}  {pid} {status_colored} {:<9} {:<8} {:<8} {op_clean}\n",
                id, turns, uptime, model
            ));
        }
        out.push('\n');
    }

    out.push_str("Press Ctrl+C to exit monitor.\n");
    out
}

pub async fn run_monitor(once: bool) -> Result<()> {
    let is_tty = std::io::stdout().is_terminal();
    if once || !is_tty {
        let entries = load_all_registry_entries();
        let now = unix_timestamp();
        let output = render_dashboard(&entries, now, is_tty);
        println!("{output}");
        return Ok(());
    }

    struct CursorGuard;
    impl Drop for CursorGuard {
        fn drop(&mut self) {
            println!("\x1b[?25h");
            let _ = std::io::stdout().flush();
        }
    }

    // Hide cursor during interactive monitoring
    print!("\x1b[?25l");
    let _ = std::io::stdout().flush();
    let _guard = CursorGuard;

    loop {
        let entries = load_all_registry_entries();
        let now = unix_timestamp();
        let output = render_dashboard(&entries, now, true);

        {
            let mut stdout = std::io::stdout().lock();
            let _ = write!(stdout, "\x1b[H\x1b[J{output}");
            let _ = stdout.flush();
        }

        tokio::select! {
            _ = tokio::signal::ctrl_c() => {
                break;
            }
            _ = tokio::time::sleep(std::time::Duration::from_millis(1000)) => {}
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(5), "05s");
        assert_eq!(format_duration(65), "01m 05s");
        assert_eq!(format_duration(3665), "01h 01m");
    }

    #[test]
    fn test_render_dashboard_empty() {
        let text = render_dashboard(&[], 1000, false);
        assert!(text.contains("MINI-SWE SWARM SUPERVISOR"));
        assert!(text.contains("No active or recent workers found"));
    }

    #[test]
    fn test_render_dashboard_with_groups() {
        let entries = vec![
            WorkerRegistryEntry {
                id: "worker01".into(),
                pid: 1234,
                task: "[audits] Fix architecture".into(),
                model: "ninja".into(),
                status: "running".into(),
                step: 10,
                max_turns: 200,
                last_command: "cargo test".into(),
                question: None,
                started_at: 1000,
                updated_at: 1050,
                group: Some("audits".into()),
            },
            WorkerRegistryEntry {
                id: "worker02".into(),
                pid: 1235,
                task: "Fix something else".into(),
                model: "ninja".into(),
                status: "paused".into(),
                step: 5,
                max_turns: 100,
                last_command: "ask".into(),
                question: Some("Need guidance".into()),
                started_at: 1000,
                updated_at: 1050,
                group: None,
            },
        ];

        let text = render_dashboard(&entries, 1060, false);
        assert!(text.contains("[SWARM: audits] (1 workers)"));
        assert!(text.contains("[SWARM: default] (1 workers)"));
        assert!(text.contains("worker01"));
        assert!(text.contains("worker02"));
        assert!(text.contains("Active: 1"));
        assert!(text.contains("Paused: 1"));
    }
}
