//! Worker slots rotate across queued owners, with FIFO order within each owner.
//!
//! Queue state contains only live waiters. A cancellation ticket removes its
//! request on drop, and a granted permit releases its slot on every exit path.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex, MutexGuard};
use tokio::sync::Notify;

#[derive(Default)]
struct Rotation {
    owners: VecDeque<(String, VecDeque<u64>)>,
}

impl Rotation {
    fn enqueue(&mut self, owner: &str, id: u64) {
        if let Some((_, queue)) = self.owners.iter_mut().find(|(name, _)| name == owner) {
            queue.push_back(id);
        } else {
            self.owners
                .push_back((owner.to_owned(), VecDeque::from([id])));
        }
    }

    /// Pure decision: the oldest request of the owner at the front runs next.
    fn next(&self) -> Option<u64> {
        self.owners.front()?.1.front().copied()
    }

    fn grant(&mut self) {
        let (owner, mut queue) = self.owners.pop_front().expect("queued owner");
        queue.pop_front();
        if !queue.is_empty() {
            self.owners.push_back((owner, queue));
        }
    }

    fn remove(&mut self, id: u64) {
        for (_, queue) in &mut self.owners {
            queue.retain(|queued| *queued != id);
        }
        self.owners.retain(|(_, queue)| !queue.is_empty());
    }
}

struct Gate {
    running: usize,
    next_id: u64,
    rotation: Rotation,
}

struct Inner {
    max: usize,
    gate: Mutex<Gate>,
    wake: Notify,
}

#[derive(Clone)]
pub(super) struct FairScheduler {
    inner: Arc<Inner>,
}

impl FairScheduler {
    pub(super) fn new(max: usize) -> Self {
        Self {
            inner: Arc::new(Inner {
                max,
                gate: Mutex::new(Gate {
                    running: 0,
                    next_id: 0,
                    rotation: Rotation::default(),
                }),
                wake: Notify::new(),
            }),
        }
    }

    fn lock_gate(&self) -> MutexGuard<'_, Gate> {
        self.inner.gate.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub(super) async fn acquire(&self, owner: &str) -> WorkerSlot {
        let id = {
            let mut gate = self.lock_gate();
            let id = gate.next_id;
            gate.next_id = gate.next_id.checked_add(1).expect("worker ticket overflow");
            gate.rotation.enqueue(owner, id);
            id
        };
        let mut ticket = QueueTicket {
            scheduler: self,
            id,
            queued: true,
        };
        loop {
            // Register before checking so a concurrent release cannot be missed.
            let wake = self.inner.wake.notified();
            tokio::pin!(wake);
            wake.as_mut().enable();
            let granted = {
                let mut gate = self.lock_gate();
                if gate.running < self.inner.max && gate.rotation.next() == Some(id) {
                    gate.rotation.grant();
                    gate.running += 1;
                    true
                } else {
                    false
                }
            };
            if granted {
                ticket.queued = false;
                self.inner.wake.notify_waiters();
                return WorkerSlot {
                    scheduler: self.clone(),
                };
            }
            wake.await;
        }
    }
}

struct QueueTicket<'a> {
    scheduler: &'a FairScheduler,
    id: u64,
    queued: bool,
}

impl Drop for QueueTicket<'_> {
    fn drop(&mut self) {
        if self.queued {
            self.scheduler.lock_gate().rotation.remove(self.id);
            self.scheduler.inner.wake.notify_waiters();
        }
    }
}

pub(super) struct WorkerSlot {
    scheduler: FairScheduler,
}

impl Drop for WorkerSlot {
    fn drop(&mut self) {
        self.scheduler.lock_gate().running -= 1;
        self.scheduler.inner.wake.notify_waiters();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::future::Future;
    use std::task::Poll;

    async fn assert_pending(mut future: std::pin::Pin<&mut impl Future>) {
        std::future::poll_fn(|cx| {
            assert!(future.as_mut().poll(cx).is_pending());
            Poll::Ready(())
        })
        .await;
    }

    #[test]
    fn pure_rotation_is_fifo_within_owners_and_round_robin_between_them() {
        let mut rotation = Rotation::default();
        for (owner, id) in [("a", 0), ("a", 1), ("a", 2), ("a", 3), ("b", 4), ("b", 5)] {
            rotation.enqueue(owner, id);
        }
        let mut order = Vec::new();
        while let Some(id) = rotation.next() {
            order.push(id);
            rotation.grant();
        }
        assert_eq!(order, [0, 4, 1, 5, 2, 3]);
        assert!(rotation.owners.is_empty());
    }

    #[tokio::test]
    async fn two_slots_interleave_backlogs_and_never_overgrant() {
        let scheduler = FairScheduler::new(2);
        let first = scheduler.acquire("a").await;
        let second = scheduler.acquire("a").await;
        let mut a3 = Box::pin(scheduler.acquire("a"));
        let mut a4 = Box::pin(scheduler.acquire("a"));
        let mut b1 = Box::pin(scheduler.acquire("b"));
        let mut b2 = Box::pin(scheduler.acquire("b"));
        for future in [&mut a3, &mut a4, &mut b1, &mut b2] {
            assert_pending(future.as_mut()).await;
        }
        drop(first);
        // B cannot jump ahead of A's next turn even if B is polled first.
        assert_pending(b1.as_mut()).await;
        let third = a3.await;
        assert_pending(a4.as_mut()).await;
        drop(second);
        let fourth = b1.await;
        assert_pending(b2.as_mut()).await;
        drop(third);
        let fifth = a4.await;
        drop(fourth);
        let sixth = b2.await;
        assert_eq!(scheduler.lock_gate().running, 2);
        drop((fifth, sixth));
        let gate = scheduler.lock_gate();
        assert_eq!(gate.running, 0);
        assert!(gate.rotation.owners.is_empty());
    }

    #[tokio::test]
    async fn dropped_waiters_remove_empty_owners_and_preserve_fifo() {
        let scheduler = FairScheduler::new(1);
        let held = scheduler.acquire("running").await;
        let mut cancelled = Box::pin(scheduler.acquire("a"));
        let mut b1 = Box::pin(scheduler.acquire("b"));
        let mut b2 = Box::pin(scheduler.acquire("b"));
        let mut c1 = Box::pin(scheduler.acquire("c"));
        for future in [&mut cancelled, &mut b1, &mut b2, &mut c1] {
            assert_pending(future.as_mut()).await;
        }
        drop(cancelled);
        drop(b2);
        assert_eq!(scheduler.lock_gate().rotation.owners.len(), 2);
        drop(held);
        assert_pending(c1.as_mut()).await;
        let b = b1.await;
        drop(b);
        drop(c1.await);
        assert!(scheduler.lock_gate().rotation.owners.is_empty());
    }

    #[tokio::test]
    async fn release_wakes_a_registered_waiter() {
        let scheduler = FairScheduler::new(1);
        let held = scheduler.acquire("a").await;
        let next = tokio::spawn({
            let scheduler = scheduler.clone();
            async move { scheduler.acquire("b").await }
        });
        tokio::time::timeout(std::time::Duration::from_secs(2), async {
            while scheduler.lock_gate().rotation.next().is_none() {
                tokio::task::yield_now().await;
            }
            drop(held);
            drop(next.await.unwrap());
        })
        .await
        .expect("release must wake the waiter without polling");
        assert_eq!(scheduler.lock_gate().running, 0);
    }
}
