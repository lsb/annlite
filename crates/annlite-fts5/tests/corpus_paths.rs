//! The two ways of naming a corpus must land on the same code path.
//!
//! `--scale 10k` is shorthand; `--docs/--queries/--name` is the general form. If the
//! shorthand ever resolved to different files, or the general form silently indexed a
//! corpus's escape sequences as text, the page counts and the quality figures would be
//! measuring different collections while claiming to be one comparison.

use annlite_fts5::{index::decode_line, scale_by_name, Corpus, CorpusFormat};
use std::path::{Path, PathBuf};

#[test]
fn a_named_scale_resolves_to_the_generated_file_names() {
    let c = Corpus::scale(scale_by_name("10k").unwrap(), Path::new("data"));
    assert_eq!(c.docs, PathBuf::from("data/corpus/docs-10k.txt"));
    assert_eq!(c.queries, PathBuf::from("data/corpus/queries-10k.jsonl"));
    assert_eq!(c.manifest, PathBuf::from("data/corpus/docs-10k.manifest.json"));
    assert_eq!(c.db, PathBuf::from("data/db/fts5-10k.db"));
    assert_eq!(c.format, CorpusFormat::Words);
    assert_eq!(c.n_docs_expected, Some(10_000));
}

#[test]
fn an_arbitrary_file_pair_finds_its_manifest_beside_the_documents() {
    let c = Corpus::files(
        "code",
        PathBuf::from("data/corpus/code-docs.txt"),
        PathBuf::from("data/corpus/code-queries.jsonl"),
        Path::new("data"),
    );
    assert_eq!(c.manifest, PathBuf::from("data/corpus/code-docs.manifest.json"));
    assert_eq!(c.db, PathBuf::from("data/db/fts5-code.db"));
    // An arbitrary corpus carries no promised document count; the build reports what
    // it actually read.
    assert_eq!(c.n_docs_expected, None);
}

#[test]
fn escaped_lines_are_decoded_only_for_the_corpus_that_writes_them() {
    let line = r"def f():\n    return 1";
    assert_eq!(decode_line(line, CorpusFormat::Code), "def f():\n    return 1");
    // The word corpora contain no escapes, and running the replacement over them
    // anyway would be a silent corruption waiting for the first backslash.
    assert_eq!(decode_line(line, CorpusFormat::Words), line);
    // A doubled backslash is one backslash.
    assert_eq!(decode_line(r"a\\b", CorpusFormat::Code), r"a\b");
    // Untouched input is borrowed, not copied: at a million lines that matters.
    assert!(matches!(
        decode_line("plain text", CorpusFormat::Code),
        std::borrow::Cow::Borrowed(_)
    ));
}

#[test]
fn a_corpus_format_pairs_its_line_decoding_with_its_term_split() {
    // The two halves travel together on purpose; a mismatch indexes one collection
    // and queries another.
    assert_eq!(
        CorpusFormat::parse("code").unwrap().term_split(),
        annlite_fts5::query::TermSplit::Alphanumeric
    );
    assert!(CorpusFormat::parse("code").unwrap().escaped_lines());
    assert_eq!(
        CorpusFormat::parse("words").unwrap().term_split(),
        annlite_fts5::query::TermSplit::Whitespace
    );
    assert!(!CorpusFormat::parse("words").unwrap().escaped_lines());
    assert!(CorpusFormat::parse("prose").is_none());
}
