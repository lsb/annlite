//! Tokenizer conformance.
//!
//! These are the same reference vectors `tools/tests/test_tokenization.py` asserts.
//! The corpus is tokenized offline by the Python implementation and queries are
//! tokenized in the browser by this one; if they disagree, every similarity is
//! computed between vectors drawn from two different input distributions.

use annlite_core::tokenize::WordPiece;

fn tok() -> WordPiece {
    let text = std::fs::read_to_string("../../models/tokenizers/bert-base-uncased-vocab.txt")
        .expect("vocabulary not found; run from the crate directory");
    WordPiece::from_vocab_text(&text, true).unwrap()
}

#[test]
fn vocab_matches_model_embedding_table() {
    assert_eq!(tok().len(), 30522);
}

#[test]
fn special_token_ids() {
    let t = tok();
    assert_eq!((t.pad_id, t.unk_id, t.cls_id, t.sep_id), (0, 100, 101, 102));
}

#[test]
fn known_sentence_matches_reference_ids() {
    assert_eq!(
        tok().encode("a man is playing a guitar on stage", 256),
        vec![101, 1037, 2158, 2003, 2652, 1037, 2858, 2006, 2754, 102]
    );
}

#[test]
fn punctuation_splits_from_words() {
    assert_eq!(tok().tokenize("don't"), vec![2123, 1005, 1056]);
}

#[test]
fn lowercasing_and_accent_stripping() {
    let t = tok();
    assert_eq!(t.tokenize("Café"), t.tokenize("cafe"));
    assert_eq!(t.tokenize("NAÏVE"), t.tokenize("naive"));
}

#[test]
fn subword_decomposition_uses_continuation_pieces() {
    let t = tok();
    let ids = t.tokenize("unaffable");
    assert!(ids.len() > 1);
    assert!(!ids.contains(&t.unk_id));
    assert!(ids.iter().any(|&i| t.token(i).starts_with("##")));
}

#[test]
fn overlong_word_becomes_unk() {
    let t = tok();
    assert_eq!(t.tokenize(&"a".repeat(200)), vec![t.unk_id]);
}

#[test]
fn encode_truncates_and_keeps_special_tokens() {
    let t = tok();
    let ids = t.encode(&vec!["hello"; 500].join(" "), 32);
    assert_eq!(ids.len(), 32);
    assert_eq!(ids[0], t.cls_id);
    assert_eq!(*ids.last().unwrap(), t.sep_id);
}

#[test]
fn whitespace_variants_are_equivalent() {
    let t = tok();
    assert_eq!(t.tokenize("hello world"), t.tokenize("hello\t\nworld"));
}
