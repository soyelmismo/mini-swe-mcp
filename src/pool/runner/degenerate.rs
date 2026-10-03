//! Detecting a model whose reasoning (or prose) has collapsed into noise.
//!
//! A thinking-mode provider's `reasoning_content` is replayed verbatim with
//! every assistant turn, so a degenerate value does not stay in one turn: it
//! is echoed back to the model, which repeats it, and the run burns its whole
//! budget on a worker that is not thinking. The observed failure was 55
//! consecutive turns whose reasoning was byte-identical, and a value made of
//! one repeated character is the same failure seen in a single turn.
//!
//! The detector answers one question -- is this text *content* or *filler* --
//! and it is deliberately conservative: a legitimate reasoning block can be
//! long, can repeat a word, and can contain runs of one character (a divider,
//! an indented code block, a table rule), so only a text that is *almost
//! entirely* one short repeated pattern counts as degenerate.
//!
//! Nothing here is used to drop reasoning from the *durable* history: the log
//! keeps what the model actually said. The guard only stops replaying it to
//! the provider and, when it persists, hands the decision to the orchestrator.

/// An assistant text long enough to judge.
///
/// Below this, the byte-identical-across-turns rule is the only one that can
/// fire: a three-character answer cannot be shown to be filler, and a real
/// short answer ("yes", "ok", a path) must never be called degenerate.
pub(crate) const DEGENERATE_MIN_BYTES: usize = 16;

/// Consecutive turns whose reasoning may be identical before it counts as
/// degenerate.
pub(crate) const DEGENERATE_REPEAT_TURNS: usize = 3;

/// How much of a text must be made of the repeated pattern for it to be
/// filler. A 90% share leaves room for the punctuation, line breaks and the
/// handful of real words a collapsed model still emits between runs.
const DEGENERATE_COVERAGE_PCT: usize = 90;

/// Longest n-gram the filler scan looks for: `!!!!!` is five characters, and
/// a pattern longer than this is prose repeating itself rather than noise.
const MAX_NGRAM: usize = 8;

/// The text the worker is told when its reasoning came back degenerate.
///
/// It names the symptom and asks for the one thing the run needs next -- a
/// reason tied to the output it just saw -- rather than "try again", which a
/// collapsed model answers with the same filler.
pub(crate) fn degenerate_nudge() -> String {
    "Your previous reasoning was empty/degenerate; think step by step about the last command output before acting".to_string()
}

/// Whether `text` is degenerate: one short pattern repeated until it fills the
/// whole field.
///
/// Two shapes are caught, and both are the observed failure:
///
/// * a single repeated character (`"!!!!"`, 32 of them in the run that
///   prompted this module) -- the common case, matched on a trimmed text made
///   of one distinct character;
/// * a short n-gram repeated end to end (`"ababab…"`, `"hmm hmm hmm…"`),
///   found by testing every period from 1 to [`MAX_NGRAM`] and counting how
///   much of the text the period actually covers.
///
/// A legitimate reasoning block is safe: a long code block, a numbered list or
/// a paragraph of prose repeats no period densely enough to reach the
/// coverage bound, so it is read as content.
pub(crate) fn is_degenerate(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.len() < DEGENERATE_MIN_BYTES {
        return false;
    }
    repeated_char(trimmed) || repeated_ngram(trimmed)
}

/// Whether `text` is one character repeated, allowing the whitespace and
/// punctuation a streamed field picks up around it.
///
/// Two distinct characters inside a long field is already prose, so only a
/// single distinct non-space character counts.
fn repeated_char(text: &str) -> bool {
    let mut distinct = [Option::<char>::None; 2];
    let mut count = 0usize;
    for c in text.chars() {
        if c.is_whitespace() {
            continue;
        }
        if !distinct.contains(&Some(c)) {
            if count == distinct.len() {
                return false;
            }
            distinct[count] = Some(c);
            count += 1;
        }
    }
    // One distinct character across a field this long is filler; a field of
    // two characters alternating is caught by the n-gram scan below.
    count == 1
}

/// Whether some n-gram of up to [`MAX_NGRAM`] bytes covers
/// [`DEGENERATE_COVERAGE_PCT`]% of `text`.
///
/// The text is compared as *bytes*: an n-gram is a period, and a period only
/// makes sense on whole characters. A multi-byte character whose bytes are
/// compared directly would report a false period (the UTF-8 continuation bytes
/// of one character repeat every three bytes), so a text that is not ASCII is
/// left to the repeated-character rule rather than scanned byte-wise.
fn repeated_ngram(text: &str) -> bool {
    let bytes = text.as_bytes();
    if !bytes.is_ascii() || bytes.len() < DEGENERATE_MIN_BYTES {
        return false;
    }
    let covered = coverage_needed(bytes.len());
    (1..=MAX_NGRAM.min(bytes.len() / 2)).any(|period| {
        let mut matching = 0usize;
        for (i, byte) in bytes.iter().enumerate() {
            if i >= period && *byte == bytes[i - period] {
                matching += 1;
            }
        }
        matching * 100 >= covered
    })
}

/// The `matching * 100` threshold for a text of `len` bytes.
///
/// Written as a multiplication so the 90% bound is exact where a division
/// would round a short text up into a false positive.
fn coverage_needed(len: usize) -> usize {
    len * DEGENERATE_COVERAGE_PCT
}

/// The question the orchestrator is asked when the degeneracy persisted.
///
/// It names the value the model keeps sending (trimmed and bounded, so a
/// 64 KiB run of one character cannot take the pause record with it) and the
/// number of turns it has been going on for, because the decision -- replace
/// the model, steer the worker, or retire it -- is the orchestrator's.
pub(crate) fn degenerate_pause_question(turns: usize, sample: &str) -> String {
    const SAMPLE_BYTES: usize = 64;
    let sample: String = sample.trim().chars().take(SAMPLE_BYTES).collect();
    format!(
        "Degenerate reasoning: {turns} consecutive assistant turns produced no usable reasoning (last value: {sample:?}). The worker is echoing filler back to the provider and is not making progress; replace the model, steer it, or retire it."
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The value the run that prompted this module produced on ~55 consecutive
    /// turns: 32 identical characters.
    const OBSERVED: &str = "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!";

    #[test]
    fn the_observed_value_is_degenerate() {
        assert_eq!(OBSERVED.len(), 32);
        assert!(is_degenerate(OBSERVED));
    }

    #[test]
    fn a_shorter_run_of_one_character_is_still_degenerate() {
        assert!(is_degenerate(&"!".repeat(DEGENERATE_MIN_BYTES)));
        assert!(is_degenerate(&"a".repeat(4 * 1024)));
    }

    #[test]
    fn a_short_field_is_never_judged_filler() {
        // A real short answer must survive: three characters cannot be shown
        // to be filler, and the byte-identical rule is what covers it.
        assert!(!is_degenerate(&"!".repeat(DEGENERATE_MIN_BYTES - 1)));
        assert!(!is_degenerate("ok"));
        assert!(!is_degenerate(""));
    }

    #[test]
    fn ordinary_reasoning_is_not_degenerate() {
        assert!(!is_degenerate(
            "Let me look at the repository first: the failing test builds the engine twice, so I will check the constructor before changing the loop."
        ));
    }

    #[test]
    fn a_long_legitimate_code_block_with_repeated_characters_is_not_degenerate() {
        // A divider, an indented block and a table rule are all runs of one
        // character inside real content; none of them fills the field.
        let mut text = String::from("fn main() {\n");
        for _ in 0..40 {
            text.push_str("    let value = calculate(1, 2, 3);\n");
        }
        text.push_str(&"-".repeat(80));
        text.push_str("\n=====\n");
        text.push_str(&"=".repeat(60));
        assert!(text.len() > 1024);
        assert!(!is_degenerate(&text));
    }

    #[test]
    fn a_repeated_ngram_is_degenerate() {
        assert!(is_degenerate(&"ab".repeat(64)));
        assert!(is_degenerate(&"hmm ".repeat(32)));
    }

    #[test]
    fn prose_that_repeats_a_word_sparsely_is_not_degenerate() {
        let text = "the build failed ".to_string() + &"the build failed again ".repeat(4);
        assert!(!is_degenerate(&text));
    }

    #[test]
    fn a_multi_byte_run_is_judged_by_its_characters_not_its_bytes() {
        // One repeated multi-byte character is filler; a sentence in the same
        // script is not, and neither is misread as a byte-level period.
        assert!(is_degenerate(&"日".repeat(64)));
        assert!(!is_degenerate(&"日本語のテキストが続きます。".repeat(4)));
    }

    #[test]
    fn the_nudge_reads_as_one_instruction() {
        assert_eq!(
            degenerate_nudge(),
            "Your previous reasoning was empty/degenerate; think step by step about the last command output before acting"
        );
    }

    #[test]
    fn the_pause_question_names_the_problem_and_bounds_the_sample() {
        let question = degenerate_pause_question(5, &"!".repeat(8 * 1024));
        assert!(question.contains("Degenerate reasoning"), "{question}");
        assert!(question.contains("5 consecutive"), "{question}");
        assert!(question.len() < 1024, "the sample must be bounded: {question}");
    }

    #[test]
    fn a_new_test_must_fail_without_the_change() {
        // Pin the one property that makes the guard safe to ship: prose is
        // content. Removing the coverage bound would make this fail.
        assert!(!is_degenerate("I will run the test suite and read the failure."));
    }
}
