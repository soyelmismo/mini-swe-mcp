//! "Did you mean …?" support for mistyped CLI actions.
//!
//! A typo in a verb is the most common CLI failure mode, so an unknown action
//! never fails silently: [`suggest_action`] first tries a prefix match and then
//! falls back to a small Levenshtein distance. [`available_actions`] is the
//! single source of truth for what the binary accepts — the MCP tool verbs plus
//! the CLI-only verbs.

use crate::mcp::WORKER_ACTIONS;

/// CLI-only verbs: actions the binary handles directly instead of dispatching
/// through the `worker` tool.
const CLI_ONLY_ACTIONS: &[&str] = &["monitor", "supervisor", "watch"];

/// Everything the CLI accepts: the tool's own actions (single-sourced from the
/// MCP server) plus the CLI-only verbs.
pub fn available_actions() -> Vec<&'static str> {
    WORKER_ACTIONS
        .iter()
        .copied()
        .chain(CLI_ONLY_ACTIONS.iter().copied())
        .collect()
}

/// Classic iterative Levenshtein edit distance over `char`s (not bytes, so
/// multi-byte input cannot skew the comparison).
fn levenshtein(a: &str, b: &str) -> usize {
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut curr = vec![0; b.len() + 1];

    for (i, ca) in a.chars().enumerate() {
        curr[0] = i + 1;
        for (j, cb) in b.chars().enumerate() {
            let cost = if ca == cb { 0 } else { 1 };
            curr[j + 1] = (prev[j + 1] + 1)
                .min(curr[j] + 1)
                .min(prev[j] + cost);
        }
        prev.clone_from_slice(&curr);
    }
    prev[b.len()]
}

/// Suggest the closest known action for an unknown one, if any is close enough.
///
/// Two tiers, in order: a case-insensitive prefix of at least 3 characters, then
/// a Levenshtein distance of at most 2 (ties resolve to the first candidate in
/// the caller's ordering). Returns `None` when the input is too short or too
/// far from everything.
pub fn suggest_action<'a>(unknown: &str, candidates: &[&'a str]) -> Option<&'a str> {
    let unknown_lower = unknown.to_lowercase();
    // 1. Prefix match (min len 3 to avoid false positives)
    if unknown_lower.len() >= 3
        && let Some(&m) = candidates.iter().find(|&&c| c.starts_with(&unknown_lower))
    {
        return Some(m);
    }
    // 2. Levenshtein edit distance <= 2
    candidates
        .iter()
        .map(|&c| (c, levenshtein(&unknown_lower, c)))
        .filter(|&(_, dist)| dist <= 2)
        .min_by_key(|&(_, dist)| dist)
        .map(|(c, _)| c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_available_actions_covers_tool_verbs_and_cli_only_verbs() {
        let actions = available_actions();
        for verb in WORKER_ACTIONS {
            assert!(actions.contains(verb), "missing worker verb {verb}");
        }
        assert!(actions.contains(&"monitor"));
        assert!(actions.contains(&"supervisor"));
        // No duplicates: a CLI-only verb must not shadow a tool verb.
        let mut seen = actions.clone();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), actions.len(), "duplicate action: {actions:?}");
    }

    #[test]
    fn test_suggest_action_prefers_a_prefix_match() {
        let actions = available_actions();
        assert_eq!(suggest_action("stat", &actions), Some("status"));
        assert_eq!(suggest_action("DISPAT", &actions), Some("dispatch"));
        assert_eq!(suggest_action("super", &actions), Some("supervisor"));
    }

    #[test]
    fn test_suggest_action_falls_back_to_edit_distance() {
        let actions = available_actions();
        // No shared prefix, so only the Levenshtein tier can answer these.
        assert_eq!(suggest_action("killl", &actions), Some("kill"));
        assert_eq!(suggest_action("sttaus", &actions), Some("status"));
    }

    #[test]
    fn test_suggest_action_rejects_input_far_from_every_candidate() {
        let actions = available_actions();
        assert_eq!(suggest_action("completely-unrelated", &actions), None);
        // A single character is below the 3-character prefix floor and is more
        // than two edits away from every verb, so nothing is suggested.
        assert_eq!(suggest_action("k", &actions), None);
    }

    #[test]
    fn test_suggest_action_can_still_match_a_two_character_input_by_distance() {
        // Two characters cannot prefix-match, but they are within two edits of
        // `kill`, so the Levenshtein tier still answers.
        assert_eq!(suggest_action("ki", &available_actions()), Some("kill"));
    }

    #[test]
    fn test_suggest_action_on_an_empty_candidate_list() {
        assert_eq!(suggest_action("anything", &[]), None);
    }

    #[test]
    fn test_levenshtein_is_symmetric_and_correct_on_known_pairs() {
        assert_eq!(levenshtein("", ""), 0);
        assert_eq!(levenshtein("abc", "abc"), 0);
        assert_eq!(levenshtein("", "abc"), 3);
        assert_eq!(levenshtein("kitten", "sitting"), 3);
        assert_eq!(levenshtein("sitting", "kitten"), 3);
        // Multi-byte input is compared per char, not per byte.
        assert_eq!(levenshtein("naïve", "naïve"), 0);
        assert_eq!(levenshtein("naïve", "naive"), 1);
    }
}
