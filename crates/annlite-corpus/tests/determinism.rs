//! The benchmark results are only comparable if these properties hold.

use annlite_corpus::{docs, WORDS_PER_DOC};

fn toy_vocab(n: usize) -> Vec<String> {
    (0..n).map(|i| format!("w{i:05}")).collect()
}

fn gen(vocab: &[String], n: usize) -> Vec<String> {
    let mut buf = Vec::new();
    docs::generate(vocab, n, &mut buf).unwrap();
    String::from_utf8(buf).unwrap().lines().map(str::to_string).collect()
}

#[test]
fn generation_is_reproducible() {
    let v = toy_vocab(500);
    assert_eq!(gen(&v, 40), gen(&v, 40));
}

#[test]
fn smaller_corpora_are_prefixes_of_larger_ones() {
    // Scale comparisons depend on this: the 100-doc corpus must be the first 100
    // documents of the 1M corpus, not an independent sample.
    let v = toy_vocab(500);
    let big = gen(&v, 100);
    for n in [1, 7, 10, 33, 99] {
        assert_eq!(gen(&v, n), big[..n], "corpus of {n} is not a prefix of the larger one");
    }
}

#[test]
fn documents_have_exact_length_and_no_repeats() {
    let v = toy_vocab(500);
    for doc in gen(&v, 60) {
        let words: Vec<&str> = doc.split(' ').collect();
        assert_eq!(words.len(), WORDS_PER_DOC);
        let mut uniq = words.clone();
        uniq.sort_unstable();
        uniq.dedup();
        assert_eq!(uniq.len(), WORDS_PER_DOC, "term frequency must be 1 for every term");
    }
}

#[test]
fn document_at_matches_sequential_generation() {
    // The query builder reconstructs single documents; it must agree with the corpus.
    let v = toy_vocab(500);
    let all = gen(&v, 60);
    for i in [0usize, 1, 9, 10, 11, 59] {
        assert_eq!(docs::document_at(&v, i).join(" "), all[i], "document {i} disagrees");
    }
}

#[test]
fn rounds_are_independent_not_rotations() {
    // Each round reshuffles the whole vocabulary. If rounds were a single stream,
    // document 0 of round 1 would echo the tail of round 0.
    let v = toy_vocab(500);
    let all = gen(&v, 30);
    let per_round = v.len() / WORDS_PER_DOC;
    assert_eq!(per_round, 10);
    assert_ne!(all[0], all[per_round]);
}
