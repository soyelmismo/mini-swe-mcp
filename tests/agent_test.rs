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
//! * [`extract_command`] — the fallback parser that recovers a
//!   bash command from a markdown fenced code block when the model did not
//!   emit a structured `tool_calls` response.
//! * [`ChatMessage`] / [`Role`] — the outbound conversation wire contract. The
//!   role is a validated enum and the fields are private, so the serialized
//!   shape of each message kind is the only observable behaviour left to pin.

use mini_swe_mcp::agent::{
    ChatMessage, Role, TRUNCATE_HEAD as HEAD, TRUNCATE_LIMIT as LIMIT, TRUNCATE_TAIL as TAIL,
    ToolCall, ToolCallFn, extract_command, truncate_output,
};

// The byte budget above which output is truncated, and the sizes of the
// head/tail slices that are retained when it is, are defined once in
// `src/agent.rs` and re-exported here, so the numbers cannot drift apart.
const _: () = assert!(HEAD + TAIL == LIMIT);

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
    let end = rest
        .find('\n')
        .expect("marker line must be newline-terminated");
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
    assert_eq!(
        head,
        "a".repeat(HEAD),
        "head must be the first {HEAD} bytes"
    );
    // ...and the tail is the last TAIL bytes verbatim.
    assert_eq!(
        tail,
        "b".repeat(0) + &"a".repeat(TAIL),
        "tail must be the last {TAIL} bytes"
    );
    // Exactly one byte fell into the gap, and the marker says so.
    assert_eq!(marker, "... [Truncated 1 bytes] ...");

    // For large inputs, head + gap + tail is strictly smaller than the original.
    let large_input = "a".repeat(20_000);
    assert!(
        truncate_output(&large_input).len() < large_input.len(),
        "truncation must shrink large outputs"
    );
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
    for leading in [HEAD - 2, HEAD - 1] {
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
    let input = format!("{}€{}", "a".repeat(20_000), "b".repeat(TAIL - 1));
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
        tail.ends_with(&"b".repeat(TAIL - 1)),
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
    let input = format!("{}{}{}", "a".repeat(HEAD - 2), emoji, "b".repeat(9000));
    let out = truncate_output(&input);
    assert!(!out.contains('\u{FFFD}'), "4-byte char must not be split");
    let (head, _, _) = split_truncated(&out);
    assert!(head.len() <= HEAD);

    // 4-byte chars straddling the head cut at every possible offset.
    for leading in (HEAD - 4)..=(HEAD + 1) {
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
    assert!(
        !out.contains('\u{FFFD}'),
        "all-emoji input must not be split"
    );
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
    let input = format!("{}€{}", "a".repeat(HEAD - 1), "b".repeat(9000));
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
// extract_command
// ---------------------------------------------------------------------------

/// 8. A single-line bash block is extracted verbatim.
#[test]
fn test_extract_command_single_line_block() {
    assert_eq!(
        extract_command("Run this:\n```bash\necho hello\n```"),
        Some("echo hello".to_string())
    );

    // Leading prose is discarded; only the block body is returned.
    assert_eq!(
        extract_command("Sure!\n\n```bash\nls -la\n```\n\nDone."),
        Some("ls -la".to_string())
    );

    // Surrounding whitespace inside the block is trimmed.
    assert_eq!(
        extract_command("```bash\n\n  echo padded  \n\n```"),
        Some("echo padded".to_string())
    );
}

/// 9. A multi-line block keeps its internal newlines and indentation.
#[test]
fn test_extract_command_multi_line_block() {
    assert_eq!(
        extract_command("```bash\ncd /tmp\nls -la\n```"),
        Some("cd /tmp\nls -la".to_string())
    );

    // A longer script, including blank lines and nested quoting.
    let script = "set -euo pipefail\n\nfor f in *.rs; do\n  echo \"building $f\"\n  cargo build --quiet\ndone";
    let reply = format!("Here is the plan:\n```bash\n{script}\n```");
    assert_eq!(extract_command(&reply), Some(script.to_string()));

    // `sh` is accepted as well as `bash`.
    assert_eq!(
        extract_command("```sh\nmake test\n```"),
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
        extract_command(reply),
        Some("ls -la\n   git status".to_string()),
        "the body is returned with surrounding whitespace trimmed"
    );

    // A 4-space indent (the markdown "code block" style) behaves the same way.
    let reply = "Steps:\n    ```bash\n    echo one\n    echo two\n```";
    assert_eq!(
        extract_command(reply),
        Some("echo one\n    echo two".to_string())
    );

    // An indented opening fence with an indented closing fence is NOT matched:
    // the regex requires `\n``` at the close, so a fence preceded by spaces
    // does not terminate the block. This documents real, current behaviour.
    let reply = "text:\n    ```bash\n    echo x\n    ```\n";
    assert_eq!(
        extract_command(reply),
        None,
        "an indented closing fence does not match the extractor regex"
    );
}

/// 11. Structural edge cases of the extractor.
#[test]
fn test_extract_command_edge_cases() {
    // No fenced block at all.
    assert_eq!(extract_command("No command here."), None);
    // An empty body: the regex requires a newline before the closing fence.
    assert_eq!(extract_command("```bash\n```"), None);
    // Only the *first* block is returned.
    assert_eq!(
        extract_command("```bash\nfirst\n```\nand\n```bash\nsecond\n```"),
        Some("first".to_string())
    );
    // A language tag that is neither `bash` nor `sh` is ignored.
    assert_eq!(extract_command("```rust\nfn main() {}\n```"), None);
    // Trailing spaces after the info string are tolerated.
    assert_eq!(
        extract_command("```bash   \necho z\n```"),
        Some("echo z".to_string())
    );
    // A block with four backticks still captures the inner command because the first three match.
    assert_eq!(
        extract_command("````bash\necho inner\n````"),
        Some("echo inner".to_string()),
    );
}

// ----------
// ChatMessage wire contract (audits/overeng_01_agent_structs.md §1, §4)
// ----------

fn to_value(msg: &ChatMessage) -> serde_json::Value {
    serde_json::from_str(&serde_json::to_string(msg).expect("ChatMessage must serialize"))
        .expect("serialized ChatMessage must be valid JSON")
}

/// The role is now a `Role` enum, so the three lowercase wire strings the chat
/// API accepts are pinned by construction rather than by stringly-typed callers.
#[test]
fn test_role_wire_strings_round_trip() {
    for (role, expected) in [
        (Role::System, "\"system\""),
        (Role::User, "\"user\""),
        (Role::Assistant, "\"assistant\""),
        (Role::Tool, "\"tool\""),
    ] {
        assert_eq!(serde_json::to_string(&role).unwrap(), expected);
        let back: Role = serde_json::from_str(expected).unwrap();
        assert_eq!(back, role);
    }
}

/// A plain `system`/`user`/`assistant` message carries `content` and omits every
/// tool-only field.
#[test]
fn test_text_message_wire_shape() {
    let msg = ChatMessage::text(Role::User, "TASK: do it");
    assert_eq!(msg.role(), Role::User);
    assert_eq!(msg.content(), Some("TASK: do it"));

    let v = to_value(&msg);
    assert_eq!(v["role"], "user");
    assert_eq!(v["content"], "TASK: do it");
    assert!(v.get("tool_calls").is_none(), "got {v}");
    assert!(v.get("tool_call_id").is_none(), "got {v}");
}

/// A tool result is identified by its `tool_call_id` and never advertises
/// `tool_calls` — the pairing the API requires after an assistant tool call.
#[test]
fn test_tool_result_message_wire_shape() {
    let msg = ChatMessage::tool_result("call_abc".to_string(), "output text");
    assert_eq!(msg.role(), Role::Tool);

    let v = to_value(&msg);
    assert_eq!(v["role"], "tool");
    assert_eq!(v["tool_call_id"], "call_abc");
    assert_eq!(v["content"], "output text");
    assert!(v.get("tool_calls").is_none(), "got {v}");
}

/// An assistant turn that only calls a tool omits `content` entirely, and an
/// empty `tool_calls` vector is normalised away rather than sent as `[]`.
#[test]
fn test_assistant_with_tool_calls_wire_shape() {
    let tc = ToolCall {
        id: "call_1".to_string(),
        r#type: "function".to_string(),
        function: ToolCallFn {
            name: "bash".to_string(),
            arguments: r#"{"command":"ls"}"#.to_string(),
        },
    };

    let msg = ChatMessage::assistant_with_tool_calls(None, vec![tc]);
    assert_eq!(msg.role(), Role::Assistant);
    let v = to_value(&msg);
    assert_eq!(v["role"], "assistant");
    assert!(v.get("content").is_none(), "got {v}");
    assert!(v.get("tool_call_id").is_none(), "got {v}");
    assert_eq!(v["tool_calls"][0]["id"], "call_1");
    assert_eq!(v["tool_calls"][0]["type"], "function");
    assert_eq!(v["tool_calls"][0]["function"]["name"], "bash");
    assert_eq!(
        v["tool_calls"][0]["function"]["arguments"],
        r#"{"command":"ls"}"#
    );

    // Prose plus an empty call list must not emit `"tool_calls": []`.
    let empty = ChatMessage::assistant_with_tool_calls(Some("prose".into()), Vec::new());
    let v = to_value(&empty);
    assert_eq!(v["content"], "prose");
    assert!(v.get("tool_calls").is_none(), "got {v}");
}

/// 12. The extractor's regex is deliberately *capture-free*: the body is sliced
///     out of the full match by hand. These cases pin the hand-slicing
///     arithmetic, in particular the boundaries (an empty/whitespace body, and
///     a body that is exactly the fence delimiters).
#[test]
fn test_extract_command_body_slicing_boundaries() {
    // A whitespace-only body is non-empty between the fences, so it is returned
    // as the empty string after trimming (not `None`).
    assert_eq!(extract_command("```bash\n   \n```"), Some(String::new()));
    assert_eq!(extract_command("```bash\n\n\n```"), Some(String::new()));

    // A literally empty body has no newline before the closing fence, so the
    // pattern cannot match at all.
    assert_eq!(extract_command("```bash\n```"), None);
    assert_eq!(extract_command("```sh\n```"), None);

    // Single-line body: exactly one byte between the opening and closing newline.
    assert_eq!(extract_command("```bash\nx\n```"), Some("x".to_string()));

    // The body scan is lazy, so the *first* closing fence terminates the block
    // (a later block is simply not reached).
    assert_eq!(
        extract_command("```bash\nx\n```bash\ny\n```"),
        Some("x".to_string())
    );
}

/// 13. The info-string gap is `[ \t\r\n]*` (ASCII whitespace), a strict subset
///     of the original `\s*`, which also matched Unicode whitespace. This test
///     documents that tightening: an ASCII gap (spaces, tabs, CR, LF) still
///     opens a block, a non-ASCII gap does not.
#[test]
fn test_extract_command_info_string_whitespace_is_ascii_only() {
    // The ASCII gap sits between the info string and the newline that ends the
    // opening line, so every combination still opens a block.
    for gap in ["", " ", "   ", "\t", "\r", " \t\r", " \t\r\n"] {
        let reply = format!("```bash{gap}\necho gap\n```");
        assert_eq!(
            extract_command(&reply),
            Some("echo gap".to_string()),
            "ASCII whitespace {gap:?} must still open a block"
        );
    }

    // An empty gap does not open a block at all: the pattern requires a
    // newline after the info string, so "```bashecho gap" is not a fence.
    assert_eq!(extract_command("```bashecho gap\n```"), None);

    // U+00A0 NO-BREAK SPACE after the info string: previously matched by `\s`,
    // now intentionally does not, so the block is not recognised.
    assert_eq!(extract_command("```bash\u{a0}\necho nbsp\n```"), None);
}

// ----------
// Golden-oracle differential tests
// ----------

/// The pre-optimisation implementation of [`truncate_output`], kept verbatim as
/// a golden oracle. The optimised version must stay byte-identical to it.
fn truncate_output_reference(combined: &str) -> String {
    if combined.len() > LIMIT {
        let head_end = combined.floor_char_boundary(HEAD);
        let tail_start = combined.ceil_char_boundary(combined.len().saturating_sub(TAIL));
        let truncated = format!(
            "\n... [Truncated {} bytes] ...\n{}",
            combined.len() - (head_end + (combined.len() - tail_start)),
            &combined[tail_start..]
        );
        format!("{}{}", &combined[..head_end], truncated)
    } else {
        combined.to_string()
    }
}

/// The pre-optimisation implementation of `extract_command`, using the original
/// capture-based regex, kept as a golden oracle.
fn extract_command_reference(text: &str) -> Option<String> {
    static REF: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    let rx = REF.get_or_init(|| {
        regex::Regex::new(r"```(?:bash|sh)\s*\n([\s\S]*?)\n```")
            .expect("reference regex must compile")
    });
    rx.captures(text)
        .and_then(|cap| cap.get(1))
        .map(|m| m.as_str().trim().to_string())
}

/// A tiny deterministic xorshift PRNG, so the pseudo-random sweeps below are
/// reproducible from run to run.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// 14. `truncate_output` is byte-identical to the original two-`format!`
///     implementation across every length around the budget, around the cut
///     points, and for multi-byte characters straddling both cuts.
#[test]
fn test_truncate_output_matches_reference_implementation() {
    const EMOJI: &str = "\u{1F680}";

    // 14a. Every short length, and both sides of the budget threshold.
    for n in 0..40 {
        let input = "a".repeat(n);
        assert_eq!(truncate_output(&input), truncate_output_reference(&input));
    }
    for n in [
        LIMIT - 1,
        LIMIT,
        LIMIT + 1,
        LIMIT + 2,
        LIMIT + 6,
        2 * LIMIT,
        100_000,
    ] {
        let input = "a".repeat(n);
        assert_eq!(
            truncate_output(&input),
            truncate_output_reference(&input),
            "mismatch at length {n}"
        );
    }

    // 14b. Multi-digit dropped byte counts (large input).
    let big = "z".repeat(1_000_000);
    assert_eq!(truncate_output(&big), truncate_output_reference(&big));

    // 14c. 2-, 3- and 4-byte characters straddling the head cut, at every
    //      offset that can put the cut inside the character.
    for ch in [
        '\u{20ac}',
        EMOJI.chars().next().unwrap(),
        '\u{65e5}',
        '\u{800}',
    ] {
        for leading in (HEAD - 10)..=(HEAD + 10) {
            let input = format!("{}c{}", "a".repeat(leading), ch) + &"b".repeat(9_000);
            assert_eq!(
                truncate_output(&input),
                truncate_output_reference(&input),
                "head-cut straddle: char {ch:?} at {leading}"
            );
        }
    }

    // 14d. Multi-byte characters straddling the tail cut.
    for ch in ['\u{20ac}', EMOJI.chars().next().unwrap()] {
        for extra in 0..40 {
            let input = format!("{}c{}", "a".repeat(20_000), ch) + &"b".repeat(TAIL - 6 + extra);
            assert_eq!(
                truncate_output(&input),
                truncate_output_reference(&input),
                "tail-cut straddle: char {ch:?} with {extra} extra bytes"
            );
        }
    }

    // 14e. All-emoji and all-CJK documents: every cut is deep inside
    //      multi-byte characters.
    for input in [
        EMOJI.repeat(10_000),
        "\u{65e5}\u{672c}\u{8a9e}".repeat(5_000),
    ] {
        assert_eq!(truncate_output(&input), truncate_output_reference(&input));
    }

    // 14f. Pseudo-random mixed-width documents.
    let alphabet = [
        'a',
        'b',
        ' ',
        '\n',
        '\u{e9}',
        '\u{20ac}',
        '\u{65e5}',
        EMOJI.chars().next().unwrap(),
        '\u{800}',
    ];
    let mut rng = Rng(0x2545_F491_4F6C_DD1D);
    for _ in 0..400 {
        let target = rng.below(40_000);
        let mut input = String::with_capacity(target * 4);
        while input.len() < target {
            input.push(alphabet[rng.below(alphabet.len())]);
        }
        assert_eq!(truncate_output(&input), truncate_output_reference(&input));
    }
}

/// 15. `extract_command` is byte-identical to the original capture-based
///     implementation. The one intentional divergence — the `[ \t\r\n]*`
///     info-string class being a strict subset of `\s*` — is checked for
///     explicitly and is excluded from the oracle sweep below (which uses
///     ASCII-only inputs, where the two languages coincide).
#[test]
fn test_extract_command_matches_reference_on_edge_cases() {
    let cases = [
        "Run this:\n```bash\necho hello\n```",
        "Sure!\n\n```bash\nls -la\n```\n\nDone.",
        "```bash\n\n  echo padded  \n\n```",
        "```bash\ncd /tmp\nls -la\n```",
        "```sh\nmake test\n```",
        "1. Run:\n   ```bash\n   ls -la\n   git status\n```",
        "Steps:\n    ```bash\n    echo one\n    echo two\n```",
        "text:\n    ```bash\n    echo x\n    ```\n",
        "No command here.",
        "```bash\n```",
        "```bash\nfirst\n```\nand\n```bash\nsecond\n```",
        "```rust\nfn main() {}\n```",
        "```bash   \necho z\n```",
        "````bash\necho inner\n````",
        "```bash\n   \n```",
        "```bash\n\n\n```",
        "```bash\necho hi\n```\n```sh\nls\n```",
        "text ```bash inline\nx\n``` more",
        "```bash\r\necho crlf\r\n```",
        "```BASH\necho upper\n```",
        "```bash extra info\necho x\n```",
        "no trailing newline ```bash\nx\n```",
        "```sh\n```",
        "```bash\n```bash\n```",
        "```bash\n\u{1F680}\n```",
        "```bash\nx\n\n```",
        "```bash\na\n```trailing",
        "",
        "a",
        "```",
        "```bash",
        "```bash\n",
    ];
    for case in cases {
        assert_eq!(
            extract_command(case),
            extract_command_reference(case),
            "divergence on {case:?}"
        );
    }
}

/// 16. The same oracle, swept over pseudo-random token soup. The token
///     alphabet is ASCII-only so that the documented `\s*` -> `[ \t\r\n]*`
///     tightening cannot fire; the two patterns describe the same language
///     there, and the sweep proves the hand-slicing never drifts from the
///     capture-based original.
#[test]
fn test_extract_command_matches_reference_on_token_soup() {
    const TOKENS: &[&str] = &[
        "```bash",
        "```sh",
        "```rust",
        "```",
        "\n",
        " ",
        "echo hi",
        "ls",
        "\t",
        "\r\n",
        "a",
        "```bash\nx",
        "x\n```",
        "```bashx",
        "``` bash",
        "```sh ",
        "    ",
        "```BASH",
    ];
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    for _ in 0..2_000 {
        let parts = rng.below(8) + 1;
        let mut input = String::new();
        for _ in 0..parts {
            input.push_str(TOKENS[rng.below(TOKENS.len())]);
        }
        assert_eq!(
            extract_command(&input),
            extract_command_reference(&input),
            "divergence on {input:?}"
        );
    }
}
