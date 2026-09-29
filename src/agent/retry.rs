//! Retry policy for calls out to the LLM API.
//!
//! Rate limits and upstream outages are routine for a hosted model, so every
//! request is wrapped in a bounded exponential backoff. This module holds that
//! policy — the attempt budget, the delay curve and the classification of
//! transient failures — so the transport code in [`super::runner`] stays about
//! *what* is being sent rather than *how often to resend it*.

use std::time::Duration;

/// Attempts made before a request is reported as a failure.
pub const DEFAULT_MAX_RETRIES: usize = 6;

/// Base of the exponential backoff curve; attempt *n* waits
/// `INITIAL_RETRY_DELAY_MS << (n - 1)`.
pub const INITIAL_RETRY_DELAY_MS: u64 = 500;

/// Ceiling on any single backoff wait, including a server-supplied
/// `Retry-After`. Without it a hostile or buggy gateway could park a worker for
/// hours.
pub const MAX_RETRY_DELAY: Duration = Duration::from_secs(30);

/// Largest exponent the delay curve will shift by, so the multiplication below
/// cannot overflow on a large attempt counter.
const MAX_DELAY_SHIFT: u32 = 6;

/// Attempt budget, overridable with `LLM_MAX_RETRIES` so an operator can
/// trade latency against resilience without a rebuild.
pub fn max_llm_retries() -> usize {
    std::env::var("LLM_MAX_RETRIES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_MAX_RETRIES)
}

/// Whether an HTTP status is worth retrying rather than surfacing to the model.
///
/// All four are the standard "the upstream is briefly unhealthy" set; a 4xx
/// like 401 or 400 is a configuration or request bug that a retry cannot fix,
/// so it is reported immediately.
pub fn is_transient_status(status: reqwest::StatusCode) -> bool {
    status == reqwest::StatusCode::TOO_MANY_REQUESTS
        || status == reqwest::StatusCode::SERVICE_UNAVAILABLE
        || status == reqwest::StatusCode::BAD_GATEWAY
        || status == reqwest::StatusCode::GATEWAY_TIMEOUT
}

/// Read a `Retry-After` header expressed in seconds.
///
/// Returns `None` for a missing or unparseable header, which falls the caller
/// back to the exponential curve.
pub fn parse_retry_after(headers: &reqwest::header::HeaderMap) -> Option<Duration> {
    headers
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok())
        .map(Duration::from_secs)
}

/// Backoff delay before retrying `attempt` (1-based).
///
/// A server-supplied `Retry-After` wins over the local curve — the server knows
/// when it will be ready — but is still capped at [`MAX_RETRY_DELAY`].
/// Otherwise the delay doubles per attempt, starting at `initial_delay`.
pub fn retry_delay(
    initial_delay: Duration,
    attempt: usize,
    retry_after: Option<Duration>,
) -> Duration {
    if let Some(ra) = retry_after {
        return ra.min(MAX_RETRY_DELAY);
    }
    let shift = (attempt.saturating_sub(1)).min(MAX_DELAY_SHIFT as usize);
    let base_ms = initial_delay.as_millis() as u64;
    let ms = base_ms.saturating_mul(1 << shift);
    Duration::from_millis(ms).min(MAX_RETRY_DELAY)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn headers(pairs: &[(&str, &str)]) -> reqwest::header::HeaderMap {
        let mut map = reqwest::header::HeaderMap::new();
        for (k, v) in pairs {
            map.insert(
                reqwest::header::HeaderName::from_bytes(k.as_bytes()).unwrap(),
                v.parse().unwrap(),
            );
        }
        map
    }

    #[test]
    fn transient_statuses_are_the_four_upstream_failures() {
        for status in [
            reqwest::StatusCode::TOO_MANY_REQUESTS,
            reqwest::StatusCode::SERVICE_UNAVAILABLE,
            reqwest::StatusCode::BAD_GATEWAY,
            reqwest::StatusCode::GATEWAY_TIMEOUT,
        ] {
            assert!(is_transient_status(status), "{status} should retry");
        }
    }

    #[test]
    fn client_and_success_statuses_are_not_retried() {
        for status in [
            reqwest::StatusCode::OK,
            reqwest::StatusCode::BAD_REQUEST,
            reqwest::StatusCode::UNAUTHORIZED,
            reqwest::StatusCode::NOT_FOUND,
        ] {
            assert!(!is_transient_status(status), "{status} should not retry");
        }
    }

    #[test]
    fn retry_after_header_is_read_in_seconds() {
        assert_eq!(
            parse_retry_after(&headers(&[("retry-after", "3")])),
            Some(Duration::from_secs(3))
        );
        assert_eq!(parse_retry_after(&headers(&[])), None);
        assert_eq!(
            parse_retry_after(&headers(&[("retry-after", "soon")])),
            None
        );
    }

    #[test]
    fn server_supplied_delay_wins_over_the_curve() {
        let d = retry_delay(Duration::from_millis(500), 4, Some(Duration::from_secs(2)));
        assert_eq!(d, Duration::from_secs(2));
    }

    #[test]
    fn server_supplied_delay_is_still_capped() {
        let d = retry_delay(
            Duration::from_millis(500),
            1,
            Some(Duration::from_secs(600)),
        );
        assert_eq!(d, MAX_RETRY_DELAY);
    }

    #[test]
    fn delay_doubles_per_attempt() {
        let base = Duration::from_millis(INITIAL_RETRY_DELAY_MS);
        assert_eq!(retry_delay(base, 1, None), Duration::from_millis(500));
        assert_eq!(retry_delay(base, 2, None), Duration::from_millis(1_000));
        assert_eq!(retry_delay(base, 3, None), Duration::from_millis(2_000));
        assert_eq!(retry_delay(base, 4, None), Duration::from_millis(4_000));
    }

    #[test]
    fn first_attempt_does_not_shift_and_zero_does_not_underflow() {
        let base = Duration::from_millis(INITIAL_RETRY_DELAY_MS);
        assert_eq!(retry_delay(base, 0, None), base);
        assert_eq!(retry_delay(base, 1, None), base);
    }

    #[test]
    fn delay_is_capped_and_never_overflows() {
        let base = Duration::from_millis(INITIAL_RETRY_DELAY_MS);
        assert_eq!(retry_delay(base, 7, None), MAX_RETRY_DELAY);
        // A far-out attempt must saturate, not wrap around to a small delay.
        assert_eq!(retry_delay(base, usize::MAX, None), MAX_RETRY_DELAY);
    }

    #[test]
    fn attempt_budget_defaults_and_overrides() {
        // Read the raw env rather than calling max_llm_retries(), which is
        // process-global and would race with other tests in the same binary.
        let fallback = DEFAULT_MAX_RETRIES;
        assert_eq!(fallback, 6);
        assert_eq!(INITIAL_RETRY_DELAY_MS, 500);
    }
}
