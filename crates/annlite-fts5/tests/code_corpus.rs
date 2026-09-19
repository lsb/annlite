//! The arbitrary-corpus path, end to end on a miniature code corpus.
//!
//! The two conventions the code corpus adds — escaped document lines and prose
//! queries — each fail *silently* if they are skipped. The index still builds, the
//! queries still run, and the numbers are simply wrong. So both are checked here by
//! their effect on retrieval rather than on a string.
//!
//! Kept out of `end_to_end.rs` because that file's page counter records through a
//! process-global VFS, and nothing here needs it.

use annlite_fts5::{db, index, query, CorpusFormat};
use rusqlite::Connection;
use std::io::Write;

fn mini_corpus(dir: &std::path::Path) -> std::path::PathBuf {
    let path = dir.join("code-docs.txt");
    let mut f = std::fs::File::create(&path).unwrap();
    // As `tools/corpus/code.py` writes them: one function per line, newlines escaped.
    writeln!(f, r"def load_module(name):\n    return _bootstrap(name)").unwrap();
    writeln!(f, r"def unrelated(x):\n    return x + 1").unwrap();
    path
}

fn hits(conn: &Connection, text: &str, split: query::TermSplit) -> Vec<i64> {
    let t = index::TABLE;
    let sql = format!(
        "SELECT rowid FROM {t} WHERE {t} MATCH ?1 ORDER BY bm25({t}) LIMIT ?2"
    );
    let expr = query::match_expression_with(text, split).unwrap();
    conn.prepare(&sql)
        .unwrap()
        .query_map(rusqlite::params![expr, 10i64], |r| r.get(0))
        .unwrap()
        .map(Result::unwrap)
        .collect()
}

#[test]
fn an_escaped_corpus_is_indexed_as_the_source_it_encodes() {
    let dir = std::env::temp_dir().join(format!("annlite-code-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let corpus = mini_corpus(&dir);

    let escaped = dir.join("escaped.db");
    let decoded = dir.join("decoded.db");
    index::build(&escaped, &corpus, CorpusFormat::Words, false).unwrap();
    let report = index::build(&decoded, &corpus, CorpusFormat::Code, false).unwrap();
    assert_eq!(report.n_docs, 2);

    // `...(name):\n    return ...` tokenizes as `name`, `nreturn` while the escape is
    // left in place, so the word `return` is simply not in the index. This is the
    // failure the flag exists to prevent, and it is invisible in every other statistic.
    let c = Connection::open(&escaped).unwrap();
    assert!(hits(&c, "return", query::TermSplit::Alphanumeric).is_empty());
    assert_eq!(hits(&c, "nreturn", query::TermSplit::Alphanumeric).len(), 2);
    drop(c);

    let c = Connection::open(&decoded).unwrap();
    assert_eq!(hits(&c, "return", query::TermSplit::Alphanumeric).len(), 2);
    assert!(hits(&c, "nreturn", query::TermSplit::Alphanumeric).is_empty());

    // A docstring, asked as a docstring: punctuation is dropped and `load_module`
    // splits the way the tokenizer split it in the document, so the right function
    // ranks first. Whitespace splitting would search for the literal term
    // `find_module().` and match nothing.
    let text = "**DEPRECATED** Load a module, given information returned by load_module().";
    assert_eq!(hits(&c, text, query::TermSplit::Alphanumeric).first(), Some(&0));
    assert!(hits(&c, text, query::TermSplit::Whitespace).is_empty());
    drop(c);

    // Both databases are still well-formed FTS5 at the project's page size; the
    // difference is in what they contain, not in how they are built.
    let c = Connection::open(&decoded).unwrap();
    db::assert_fts5(&c).unwrap();
    let page_size: u32 = c.query_row("PRAGMA page_size", [], |r| r.get(0)).unwrap();
    assert_eq!(page_size, db::PAGE_SIZE);
    drop(c);

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_query_set_without_k_still_loads() {
    // `data/corpus/code-queries.jsonl` carries no `k` and no `random` queries, so a
    // loader that demanded either would reject the corpus this path exists for.
    let dir = std::env::temp_dir().join(format!("annlite-code-q-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("q.jsonl");
    std::fs::write(
        &path,
        "{\"qid\":0,\"kind\":\"docstring\",\"text\":\"Load a module.\",\"source_doc\":7,\
         \"name\":\"load_module\",\"module\":\"imp.py\"}\n",
    )
    .unwrap();
    let qs = query::load(&path).unwrap();
    assert_eq!(qs.len(), 1);
    assert_eq!(qs[0].k, 0);
    assert_eq!(qs[0].source_doc, Some(7));
    assert!(qs[0].is_known_item(), "a query with a gold document is a known-item query");
    std::fs::remove_dir_all(&dir).ok();
}
