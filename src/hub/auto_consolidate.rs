//! Durable, owner-scoped automatic rounds. Short synchronous transactions only:
//! no state guard is held while dispatching or waiting on a worker. A dispatch
//! guard suppresses eligibility during a whole batch, including cancellation.
//! Launch claims are owner/group-scoped: same-round dispatches wait for the
//! claim, but unrelated dispatches never wait for a consolidator's admission.
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
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
    pub consolidators: Vec<String>,
}

pub(crate) struct AutoConsolidate {
    file: PathBuf,
    state: Mutex<State>,
    changed: tokio::sync::Notify,
}
struct State {
    rows: Vec<Round>,
    dispatches: HashMap<(String, String), usize>,
    launching: HashSet<(String, String)>,
}
pub(crate) struct DispatchGuard(Arc<AutoConsolidate>, Vec<(String, String)>);
impl Drop for DispatchGuard {
    fn drop(&mut self) {
        let mut state = self.0.state.lock().unwrap();
        for key in &self.1 {
            let count = state.dispatches.get_mut(key).unwrap();
            *count -= 1;
            if *count == 0 {
                state.dispatches.remove(key);
            }
        }
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
        Ok(Arc::new(Self {
            file,
            changed: tokio::sync::Notify::new(),
            state: Mutex::new(State {
                rows,
                dispatches: HashMap::new(),
                launching: HashSet::new(),
            }),
        }))
    }
    fn save(&self, rows: &[Round]) -> Result<()> {
        let tmp = self.file.with_extension("tmp");
        std::fs::write(&tmp, serde_json::to_vec(rows)?)?;
        std::fs::rename(tmp, &self.file)?;
        Ok(())
    }
    pub async fn dispatch_guard(self: &Arc<Self>, keys: Vec<(String, String)>) -> DispatchGuard {
        loop {
            let notified = self.changed.notified();
            tokio::pin!(notified);
            // Register before checking the claim, so its release cannot be lost.
            notified.as_mut().enable();
            {
                let mut state = self.state.lock().unwrap();
                if keys.iter().all(|key| !state.launching.contains(key)) {
                    for key in &keys {
                        *state.dispatches.entry(key.clone()).or_default() += 1;
                    }
                    return DispatchGuard(self.clone(), keys);
                }
            }
            notified.await;
        }
    }
    pub fn record(&self, mut round: Round, enable: bool) -> Result<()> {
        let mut state = self.state.lock().unwrap();
        if let Some(old) = state
            .rows
            .iter_mut()
            .find(|r| r.owner == round.owner && r.group == round.group)
        {
            if old.consumed
                || round
                    .consolidators
                    .iter()
                    .any(|id| !old.consolidators.contains(id))
            {
                round.generation = old.generation.wrapping_add(1);
                if !enable {
                    round.model = old.model.clone();
                    round.verify = old.verify.clone();
                }
                *old = round;
            } else {
                old.generation = old.generation.wrapping_add(1);
                if enable {
                    old.model = round.model;
                    old.verify = round.verify;
                }
            }
        } else if enable {
            round.baseline.clear();
            anyhow::ensure!(
                state.rows.len() < 512,
                "Too many automatic consolidation groups"
            );
            state.rows.push(round);
        }
        self.save(&state.rows)
    }
    pub fn candidates(&self) -> Vec<Round> {
        let state = self.state.lock().unwrap();
        state
            .rows
            .iter()
            .filter(|r| {
                let key = (r.owner.clone(), r.group.clone());
                !r.consumed
                    && !state.dispatches.contains_key(&key)
                    && !state.launching.contains(&key)
            })
            .cloned()
            .collect()
    }
    pub fn claim(self: &Arc<Self>, round: &Round) -> Option<LaunchGuard> {
        let mut state = self.state.lock().unwrap();
        let key = (round.owner.clone(), round.group.clone());
        if state.dispatches.contains_key(&key) || state.launching.contains(&key) {
            return None;
        }
        if !state.rows.iter().any(|r| {
            r.owner == round.owner
                && r.group == round.group
                && r.generation == round.generation
                && !r.consumed
        }) {
            return None;
        }
        state.launching.insert(key.clone());
        Some(LaunchGuard(self.clone(), key))
    }
    pub fn consume(&self, round: &Round) -> Result<()> {
        let mut state = self.state.lock().unwrap();
        if let Some(row) = state.rows.iter_mut().find(|r| {
            r.owner == round.owner && r.group == round.group && r.generation == round.generation
        }) {
            row.consumed = true;
        }
        self.save(&state.rows)
    }
}
pub(crate) struct LaunchGuard(Arc<AutoConsolidate>, (String, String));
impl Drop for LaunchGuard {
    fn drop(&mut self) {
        self.0.state.lock().unwrap().launching.remove(&self.1);
        self.0.changed.notify_waiters();
    }
}

#[cfg(test)]
#[path = "auto_consolidate_tests.rs"]
mod tests;
