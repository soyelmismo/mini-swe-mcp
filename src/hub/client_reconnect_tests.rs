//! Which failures count as "the daemon is gone", and what follows from that.
//!
//! A handover cuts a watch's connection in whichever way the kernel felt like:
//! an orderly EOF from a graceful stop, a reset or broken pipe from an abrupt
//! one, and a refused or absent socket while the replacement starts. All of them
//! must answer with the same chase, and the chase must end — with an
//! explanation — once its budget is spent.

use super::*;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// A transport error the daemon's teardown produced, wrapped the way a real
/// call wraps it: a context above the `io::Error`.
fn wrapped(error: std::io::Error) -> anyhow::Error {
    anyhow::Error::new(error).context("Hub request failed")
}

/// A stand-in for a dial: the chase is what is under test here, not the socket
/// it opens, so each attempt reports what the test scheduled for it.
#[derive(Clone)]
struct Dial {
    next: Arc<Mutex<VecDeque<Result<()>>>>,
    attempts: Arc<AtomicUsize>,
}

impl Dial {
    /// A dial whose attempts answer with `answers`, in order. Once they run
    /// out it refuses, so a chase that outruns its script fails loudly.
    fn new(answers: impl IntoIterator<Item = Result<()>>) -> Self {
        Self {
            next: Arc::new(Mutex::new(answers.into_iter().collect())),
            attempts: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// How many times the chase dialled.
    fn attempts(&self) -> usize {
        self.attempts.load(Ordering::SeqCst)
    }
}

impl Reconnect for Dial {
    type Output = ();

    async fn attempt(&mut self) -> Result<()> {
        self.attempts.fetch_add(1, Ordering::SeqCst);
        self.next
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .pop_front()
            .unwrap_or_else(|| Err(anyhow::anyhow!("the chase dialled past its script")))
    }
}

/// Every way a daemon can disappear reads as "gone away".
#[test]
fn every_transport_loss_reads_as_the_daemon_going_away() {
    for error in [
        wrapped(std::io::Error::from(std::io::ErrorKind::ConnectionReset)),
        wrapped(std::io::Error::from(std::io::ErrorKind::BrokenPipe)),
        wrapped(std::io::Error::from(std::io::ErrorKind::ConnectionRefused)),
        wrapped(std::io::Error::from(std::io::ErrorKind::NotFound)),
        // A clean EOF is reported by the client as its own message, with no
        // `io::Error` behind it.
        anyhow::anyhow!("Hub closed the connection"),
        // Bare, the way a failed write surfaces: nothing is wrapped.
        anyhow::Error::new(std::io::Error::from(std::io::ErrorKind::ConnectionReset)),
        anyhow::Error::new(std::io::Error::from(std::io::ErrorKind::BrokenPipe)),
        // The dialler gave up before the replacement ever bound its socket.
        anyhow::anyhow!("Hub did not start within 5 seconds; inspect hub.log"),
    ] {
        assert!(
            daemon_went_away(&error),
            "must follow the daemon: {error:#}"
        );
    }
}

/// A refusal that is the daemon's answer, not its absence, is not a reconnect.
#[test]
fn a_refused_request_is_not_the_daemon_going_away() {
    for error in [
        anyhow::anyhow!("a watch is already running for your session"),
        anyhow::anyhow!("worker w1 belongs to agent other"),
        anyhow::anyhow!("Method not found"),
        anyhow::anyhow!("Hub response exceeds 32 MiB"),
        anyhow::anyhow!("Could not start hub daemon").context("init"),
    ] {
        assert!(
            !daemon_went_away(&error),
            "must be reported, not retried: {error:#}"
        );
    }
}

/// The chase keeps dialling until the daemon answers, however many of the
/// failures were the replacement coming up.
#[tokio::test]
async fn a_chase_dials_again_until_a_daemon_answers() {
    let dial = Dial::new([
        Err(wrapped(std::io::Error::from(
            std::io::ErrorKind::ConnectionRefused,
        ))),
        // The replacement has not bound its socket yet.
        Err(anyhow::Error::new(std::io::Error::from(
            std::io::ErrorKind::NotFound,
        ))),
        Ok(()),
    ]);
    let answer = follow_until::<Dial>(Duration::from_secs(5), dial.clone()).await;
    assert!(answer.is_ok(), "the chase ends on the daemon that answers");
    assert_eq!(
        dial.attempts(),
        3,
        "every failure that means the daemon is gone is retried"
    );
}

/// A failure that is not the daemon's absence ends the chase at once.
#[tokio::test]
async fn a_chase_reports_a_refusal_without_dialling_again() {
    let dial = Dial::new([Err(anyhow::anyhow!(
        "a watch is already running for your session"
    ))]);
    let error = follow_until::<Dial>(Duration::from_secs(5), dial.clone())
        .await
        .expect_err("a refusal is the caller's answer");
    assert!(
        error.to_string().contains("already running"),
        "{error:#}"
    );
    assert_eq!(
        dial.attempts(),
        1,
        "one refusal must not be dialled into a busy daemon"
    );
}

/// A hub that never comes back ends the watch with an explanation, not a hang.
#[tokio::test]
async fn a_chase_gives_up_once_its_budget_is_spent() {
    let dial = Dial::new(std::iter::repeat_with(|| {
        Err(anyhow::Error::new(std::io::Error::from(
            std::io::ErrorKind::ConnectionRefused,
        )))
    }));
    let error = follow_until::<Dial>(Duration::from_millis(120), dial.clone())
        .await
        .expect_err("the budget ends the chase");
    let message = error.to_string();
    assert!(message.contains("did not come back"), "{message}");
    assert!(
        message.contains("Connection refused"),
        "the cause is reported with it: {message}"
    );
    assert!(
        dial.attempts() > 1,
        "the budget must be spent retrying: {}",
        dial.attempts()
    );
}

/// The default budget is the one a handover has to fit in, and an override is
/// clamped at both ends: a client can neither give up instantly nor wait
/// forever.
#[test]
fn the_budget_is_the_default_and_its_override_is_bounded() {
    assert_eq!(reconnect_deadline(), Duration::from_secs(DEFAULT_RECONNECT_SECS));
    assert_eq!(reconnect_secs(Some(0)), MIN_RECONNECT_SECS);
    assert_eq!(
        reconnect_secs(Some(MAX_RECONNECT_SECS + 1)),
        MAX_RECONNECT_SECS
    );
    assert_eq!(reconnect_secs(Some(5)), 5);
}
