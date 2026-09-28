//! Integration tests for the pure helpers in [`mini_swe_mcp::agent`].
//!
//! Two pieces of logic are covered here, both exercised through the public
//! library surface (i.e. the way an external integration test would):
//!
//! * [`truncate_output`] — bounds command output to a 16 KiB budget while
//!   keeping the *head* and *tail* of the text. Because the split points are
//!   arbitrary byte offsets, the function must snap them to valid UTF-8
//!   character boundaries; otherwise a multi-byte character straddling a cut
//!   would panic (or, if it were done with lossy decoding, produce `U+FFFD`).
//! * [`AgentRunner::extract_command`] — the fallback parser that recovers a
//!   bash command from a markdown fenced code block when the model did not
//!   emit a structured `tool_calls` response.

use mini_swe_mcp::agent::{truncate_output, AgentRunner};

/// The byte budget above which output is truncated, and the sizes of the
/// head/tail slices that are retained when it is.
const LIMIT: usize = 16384;
const HEAD: usize = 12288;
const TAIL: usize = 4096;

fn runner() -> AgentRunner {
    AgentRunner::new(
        "http://localhost".to_string(),
        "test-key".to_string(),
        "test-model".to_string(),
        None,
    )
}

/// Extract the discarded-byte count from a `... [Truncated N bytes] ...` marker.
fn parse_marker(marker: &str) -> usize {
    marker
        .strip_prefix("... [Truncated ")
        .and_then(|m| m.strip_suffix(" bytes] ..."))
        .expect("marker must have the documented shape")
        .parse()
        .expect("marker must contain a byte count")
}

/// Split truncated output into `(head, marker_line, tail)`.
fn split_truncated(out: &str) -> (&str, &str, &str) {
    let idx = out
        .find("... [Truncated ")
        .unwrap_or_else(|| panic!("expected a truncation marker, got {out}"));
    // The marker is introduced by a leading '\n' that is *not* part of the head.
    let head = &out[..idx - 1];
    let rest = &out[idx..];
    let end = rest.find('\n').expect("marker line must be newline-terminated");
    (head, &rest[..end], &rest[end + 1..])
}

// ---------------------------------------------------------------------------
// truncate_output
// ---------------------------------------------------------------------------

/// 1. Output at or below the 16 KiB budget is returned byte-for-byte
///    unchanged — no marker is inserted and nothing is trimmed.
#[test]
fn test_truncate_output_leaves_short_output_untouched() {
    for len in [0, 1, 100, LIMIT - 1, LIMIT] {
        let input = "a".repeat(len);
        let out = truncate_output(&input);
        assert_eq!(
            out, input,
            "output of {len} bytes is within the {LIMIT}-byte budget and must be unchanged"
        );
        assert!(
            !out.contains("[Truncated"),
            "no truncation marker expected for {len} bytes"
        );
    }
}

/// 2. Just over the budget, the function keeps a 12 KiB head and a 4 KiB tail
///    around an explicit marker, and reports exactly how many bytes vanished.
#[test]
fn test_truncate_output_keeps_head_and_tail_with_16k_split() {
    let input = "a".repeat(LIMIT + 1);
    let out = truncate_output(&input);
    let (head, marker, tail) = split_truncated(&out);

    // The head is the first HEAD bytes verbatim...
    assert_eq!(head, "a".repeat(HEAD), "head must be the first {HEAD} bytes");
    // ...and the tail is the last TAIL bytes verbatim.
    assert_eq!(
        tail, "b".repeat(0) + &"a".repeat(TAIL),
        "tail must be the last {TAIL} bytes"
    );
    // Exactly one byte fell into the gap, and the marker says so.
    assert_eq!(marker, "... [Truncated 1 bytes] ...");

    // For large inputs, head + gap + tail is strictly smaller than the original.
    let large_input = "a".repeat(20_000);
    assert!(truncate_output(&large_input).len() < large_input.len(), "truncation must shrink large outputs");
    assert_eq!(out.chars().count(), out.len(), "output must be pure ASCII");
}

/// 3. The retained bytes are always the *edges* of the input: a large middle
///    region is dropped, but the beginning and end survive intact, and the
///    reported byte count is the size of the discarded middle.
#[test]
fn test_truncate_output_drops_only_the_middle() {
    let head_marker = "HEAD-SENTINEL";
    let tail_marker = "TAIL-SENTINEL";
    let input = format!("{head_marker}{}{tail_marker}", "x".repeat(100_000));

    let out = truncate_output(&input);
    let (head, marker, tail) = split_truncated(&out);

    assert!(
        head.starts_with(head_marker),
        "the beginning of the output must be preserved"
    );
    assert!(
        tail.ends_with(tail_marker),
        "the end of the output must be preserved"
    );
    assert!(
        out.len() < input.len(),
        "the middle of the output must be discarded"
    );

    let dropped = parse_marker(marker);
    assert_eq!(
        dropped,
        input.len() - (HEAD + TAIL),
        "the marker must report the exact number of discarded bytes"
    );
}

/// 4. UTF-8 boundary safety: a multi-byte character sitting exactly on the
///    head cut is never split. The head is *floored* to the previous boundary,
///    so the retained head may be one character shorter than the budget.
#[test]
fn test_truncate_output_never_splits_multibyte_head_char() {
    // Position the 3-byte '€' (U+20AC) so that the nominal head cut at byte
    // 12288 lands strictly inside it, at every offset that can straddle it.
    for leading in [12286, 12287] {
        let input = format!("{}€{}", "a".repeat(leading), "b".repeat(9000));
        assert!(
            !input.is_char_boundary(HEAD),
            "test setup: byte {HEAD} must fall inside the multi-byte char"
        );

        let out = truncate_output(&input);
        let (head, marker, tail) = split_truncated(&out);

        // No lossy-decoding artefacts anywhere in the result.
        assert!(
            !out.contains('\u{FFFD}'),
            "a multi-byte char was split: found U+FFFD in {out}"
        );
        // The head was floored back to a real character boundary.
        assert!(
            head.chars().all(|c| c == 'a'),
            "head must consist only of whole ASCII chars, got {head}"
        );
        assert!(
            head.len() <= HEAD,
            "head must not exceed the {HEAD}-byte budget"
        );
        // The tail is untouched ASCII and still ends the string.
        assert!(
            tail.ends_with(&"b".repeat(TAIL)),
            "tail must be the last {TAIL} bytes"
        );
        // The byte count is self-consistent:
        // head + dropped + tail == input length.
        let dropped = parse_marker(marker);
        assert_eq!(
            head.len() + dropped + tail.len(),
            input.len(),
            "head + dropped + tail must reconstruct the input length"
        );
    }
}

/// 5. The same guarantee applies to the *tail* cut: `ceil_char_boundary` moves
///    the tail start forward so a character straddling it is dropped whole and
///    kept, rather than being split in half.
#[test]
fn test_truncate_output_tail_cut_keeps_whole_chars() {
    // Build input whose len-4096 tail cut lands inside a 3-byte character.
    let input = format!("{}€{}", "a".repeat(20_000), "b".repeat(4095));
    let cut = input.len() - TAIL;
    assert!(
        !input.is_char_boundary(cut),
        "test setup: tail cut at byte {cut} must fall inside a multi-byte char"
    );

    let out = truncate_output(&input);
    assert!(
        !out.contains('\u{FFFD}'),
        "a multi-byte char was split: found U+FFFD in {out}"
    );
    let (head, _marker, tail) = split_truncated(&out);

    // The head is the full 12 KiB budget (the straddle is far away from it).
    assert_eq!(head.len(), HEAD, "head must be the full {HEAD}-byte budget");
    // The tail starts at a real character boundary, i.e. never mid-character.
    assert!(
        tail.starts_with("a") || tail.starts_with("b"),
        "tail must start on a character boundary, got {tail}"
    );
    // The trailing sentinel survives untouched.
    assert!(
        tail.ends_with(&"b".repeat(4095)),
        "the end of the output must be preserved verbatim"
    );
}

/// 6. Pathological inputs must not panic: many multi-byte chars, a 4-byte
///    emoji straddling the cut, and an all-emoji string.
#[test]
fn test_truncate_output_handles_pathological_multibyte_input() {
    const EMOJI: char = '\u{1F680}'; // 4 bytes
    let mut buf = [0u8; 4];
    let emoji: &str = EMOJI.encode_utf8(&mut buf);

    // A 4-byte char landing exactly on the head boundary.
    let input = format!("{}{}{}", "a".repeat(12286), emoji, "b".repeat(9000));
    let out = truncate_output(&input);
    assert!(!out.contains('\u{FFFD}'), "4-byte char must not be split");
    let (head, _, _) = split_truncated(&out);
    assert!(head.len() <= HEAD);

    // 4-byte chars straddling the head cut at every possible offset.
    for leading in 12284..=12289 {
        let input = format!("{}{}{}", "a".repeat(leading), emoji, "b".repeat(9000));
        let out = truncate_output(&input);
        assert!(
            !out.contains('\u{FFFD}'),
            "straddling 4-byte char at {leading} must not be split"
        );
    }

    // An all-emoji string: every cut is deep inside multi-byte chars.
    let input = emoji.repeat(10_000);
    let out = truncate_output(&input);
    assert!(!out.contains('\u{FFFD}'), "all-emoji input must not be split");
    let (head, _, tail) = split_truncated(&out);
    assert!(
        head.chars().all(|c| c == EMOJI),
        "all-emoji head must contain only whole emoji, got {head}"
    );
    assert!(
        tail.chars().all(|c| c == EMOJI),
        "all-emoji tail must contain only whole emoji, got {tail}"
    );
    assert!(!head.is_empty() && !tail.is_empty());

    // A 2-byte char exactly on the boundary, the case the unit test in
    // `src/agent.rs` covers.
    let input = format!("{}€{}", "a".repeat(12287), "b".repeat(9000));
    let out = truncate_output(&input);
    assert!(out.contains("... [Truncated "), "input exceeds the budget");
    assert!(!out.contains('\u{FFFD}'), "2-byte char must not be split");
}

/// 7. `execute_bash` feeds output through `truncate_output`, so a stream that
///    mixes ASCII and multi-byte characters stays valid UTF-8 end to end.
#[test]
fn test_truncate_output_preserves_multibyte_document() {
    let line = "compilando: café ☕ — ñandú 日本語 🚀\n";
    let input = line.repeat(3000); // ~100 KB of multi-byte text
    assert!(input.len() > LIMIT);

    let out = truncate_output(&input);
    assert!(!out.contains('\u{FFFD}'), "no character may be split");
    // Head and tail both still end on a whole line of the original document.
    let (head, _, tail) = split_truncated(&out);
    assert!(
        input.starts_with(head) && head.len() <= HEAD,
        "head must be an unmodified prefix of the input"
    );
    assert!(
        input.ends_with(tail) && tail.len() <= TAIL + line.len(),
        "tail must be an unmodified suffix of the input"
    );
}

// ---------------------------------------------------------------------------
// AgentRunner::extract_command
// ---------------------------------------------------------------------------

/// 8. A single-line bash block is extracted verbatim.
#[test]
fn test_extract_command_single_line_block() {
    assert_eq!(
        runner().extract_command("Run this:\n```bash\necho hello\n```"),
        Some("echo hello".to_string())
    );

    // Leading prose is discarded; only the block body is returned.
    assert_eq!(
        runner().extract_command("Sure!\n\n```bash\nls -la\n```\n\nDone."),
        Some("ls -la".to_string())
    );

    // Surrounding whitespace inside the block is trimmed.
    assert_eq!(
        runner().extract_command("```bash\n\n  echo padded  \n\n```"),
        Some("echo padded".to_string())
    );
}

/// 9. A multi-line block keeps its internal newlines and indentation.
#[test]
fn test_extract_command_multi_line_block() {
    assert_eq!(
        runner().extract_command("```bash\ncd /tmp\nls -la\n```"),
        Some("cd /tmp\nls -la".to_string())
    );

    // A longer script, including blank lines and nested quoting.
    let script = "set -euo pipefail\n\nfor f in *.rs; do\n  echo \"building $f\"\n  cargo build --quiet\ndone";
    let reply = format!("Here is the plan:\n```bash\n{script}\n```");
    assert_eq!(runner().extract_command(&reply), Some(script.to_string()));

    // `sh` is accepted as well as `bash`.
    assert_eq!(
        runner().extract_command("```sh\nmake test\n```"),
        Some("make test".to_string())
    );
}

/// 10. Indented (nested) markdown blocks. An indented *opening* fence matches,
///     and the block body is returned as-is — the regex does not de-indent
///     continuation lines, so only the first line loses its indentation.
#[test]
fn test_extract_command_indented_block() {
    // Opening fence indented by 3 spaces, closing fence at column 0.
    let reply = "1. Run:\n   ```bash\n   ls -la\n   git status\n```";
    assert_eq!(
        runner().extract_command(reply),
        Some("ls -la\n   git status".to_string()),
        "the body is returned with surrounding whitespace trimmed"
    );

    // A 4-space indent (the markdown "code block" style) behaves the same way.
    let reply = "Steps:\n    ```bash\n    echo one\n    echo two\n```";
    assert_eq!(
        runner().extract_command(reply),
        Some("echo one\n    echo two".to_string())
    );

    // An indented opening fence with an indented closing fence is NOT matched:
    // the regex requires `\n``` at the close, so a fence preceded by spaces
    // does not terminate the block. This documents real, current behaviour.
    let reply = "text:\n    ```bash\n    echo x\n    ```\n";
    assert_eq!(
        runner().extract_command(reply),
        None,
        "an indented closing fence does not match the extractor regex"
    );
}

/// 11. Structural edge cases of the extractor.
#[test]
fn test_extract_command_edge_cases() {
    // No fenced block at all.
    assert_eq!(runner().extract_command("No command here."), None);
    // An empty body: the regex requires a newline before the closing fence.
    assert_eq!(runner().extract_command("```bash\n```"), None);
    // Only the *first* block is returned.
    assert_eq!(
        runner().extract_command("```bash\nfirst\n```\nand\n```bash\nsecond\n```"),
        Some("first".to_string())
    );
    // A language tag that is neither `bash` nor `sh` is ignored.
    assert_eq!(runner().extract_command("```rust\nfn main() {}\n```"), None);
    // Trailing spaces after the info string are tolerated.
    assert_eq!(
        runner().extract_command("```bash   \necho z\n```"),
        Some("echo z".to_string())
    );
    // A block with four backticks still captures the inner command because the first three match.
    assert_eq!(
        runner().extract_command("````bash\necho inner\n````"),
        Some("echo inner".to_string()),
    );
}
