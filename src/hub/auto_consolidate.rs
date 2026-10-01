//! Durable, owner-scoped automatic rounds. Short synchronous transactions only:
//! no state guard is held while dispatching or waiting on a worker. A dispatch
//! guard suppresses eligibility during a whole batch, including cancellation.
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct Round {
    pub owner: String,
    pub group: String,
    pub repo: PathBuf,
    pub model: Option<String>,
    pub verify: Option<String>,
    pub generation: u64,
    pub consumed: bool,
    pub baseline: Vec<String>,
}

pub(crate) struct AutoConsolidate {
    file: PathBuf,
    state: Mutex<State>,
}
struct State {
    rows: Vec<Round>,
    dispatches: usize,
    launching: bool,
}
pub(crate) struct DispatchGuard(Arc<AutoConsolidate>);
impl Drop for DispatchGuard {
    fn drop(&mut self) {
        self.0.state.lock().unwrap().dispatches -= 1;
    }
}
impl AutoConsolidate {
    pub fn open(dir: PathBuf) -> Result<Arc<Self>> {
        let file = dir.join("auto-consolidate.json");
        let rows = match std::fs::read(&file) {
            Ok(bytes) => serde_json::from_slice(&bytes)?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Vec::new(),
            Err(e) => return Err(e.into()),
        };
        Ok(Arc::new(Self { file, state: Mutex::new(State { rows, dispatches: 0, launching: false }) }))
    }
    fn save(&self, rows: &[Round]) -> Result<()> {
        let tmp = self.file.with_extension("tmp");
        std::fs::write(&tmp, serde_json::to_vec(rows)?)?;
        std::fs::rename(tmp, &self.file)?;
        Ok(())
    }
    pub fn dispatch_guard(self: &Arc<Self>) -> DispatchGuard {
        self.state.lock().unwrap().dispatches += 1;
        DispatchGuard(self.clone())
    }
    pub fn record(&self, mut round: Round, enable: bool) -> Result<()> {
        let mut state = self.state.lock().unwrap();
        if let Some(old) = state.rows.iter_mut().find(|r| r.owner == round.owner && r.group == round.group) {
            if old.consumed {
                round.generation = old.generation.wrapping_add(1);
                if !enable {
                    round.model = old.model.clone();
                    round.verify = old.verify.clone();
                }
                *old = round;
            } else if enable {
                old.model = round.model;
                old.verify = round.verify;
            }
        } else if enable {
            anyhow::ensure!(state.rows.len() < 512, "Too many automatic consolidation groups");
            state.rows.push(round);
        }
        self.save(&state.rows)
    }
    pub fn candidates(&self) -> Vec<Round> {
        let state = self.state.lock().unwrap();
        if state.dispatches != 0 || state.launching { return Vec::new(); }
        state.rows.iter().filter(|r| !r.consumed).cloned().collect()
    }
    pub fn claim(self: &Arc<Self>, round: &Round) -> Option<LaunchGuard> {
        let mut state = self.state.lock().unwrap();
        if state.dispatches != 0 || state.launching { return None; }
        if !state.rows.iter().any(|r| r.owner == round.owner && r.group == round.group && r.generation == round.generation && !r.consumed) { return None; }
        state.launching = true;
        Some(LaunchGuard(self.clone()))
    }
    pub fn consume(&self, round: &Round) -> Result<()> {
        let mut state = self.state.lock().unwrap();
        if let Some(row) = state.rows.iter_mut().find(|r| r.owner == round.owner && r.group == round.group && r.generation == round.generation) {
            row.consumed = true;
        }
        self.save(&state.rows)
    }
}
pub(crate) struct LaunchGuard(Arc<AutoConsolidate>);
impl Drop for LaunchGuard {
    fn drop(&mut self) { self.0.state.lock().unwrap().launching = false; }
}
