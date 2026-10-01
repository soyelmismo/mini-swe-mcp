pub fn format_reap(val: &serde_json::Value) -> String {
    let reaped = val.get("reaped").and_then(|v| v.as_u64()).unwrap_or(0);
    let ids = val
        .get("worker_ids")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        })
        .unwrap_or_default();
    if reaped == 0 {
        "✓ No expired terminal worker records to reap.".to_string()
    } else {
        format!("✓ Reaped {reaped} expired worker record(s): {ids}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::format::worker::tests::v;
    #[test]
    fn test_format_reap_lists_ids_only_when_something_was_reaped() {
        assert_eq!(
            format_reap(&v(r#"{"reaped":0}"#)),
            "✓ No expired terminal worker records to reap."
        );
        assert_eq!(
            format_reap(&v(r#"{"reaped":2,"worker_ids":["a","b",7]}"#)),
            "✓ Reaped 2 expired worker record(s): a, b"
        );
    }
}
