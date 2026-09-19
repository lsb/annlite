//! End-to-end checks on a miniature corpus: the FTS5 build, the OR semantics on real
//! data, and the page counter itself.
//!
//! The page counter is the measurement this baseline exists for, so it is verified
//! rather than trusted: a query must read at least one page, must not read a page
//! number outside the file, and its count must agree with SQLite's own
//! `SQLITE_DBSTATUS_CACHE_MISS`.
//!
//! All of it lives in one test function on purpose. The recorder inside the VFS is
//! process-global (a VFS is), so two tests recording at the same time would interleave
//! their reads.

use annlite_fts5::{db, index, pages, query, vfs};
use rusqlite::Connection;
use std::io::Write;

fn tmpdir() -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("annlite-fts5-test-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[test]
fn build_query_and_page_counting() {
    let dir = tmpdir();
    let corpus = dir.join("docs.txt");
    let dbp = dir.join("mini.db");
    {
        let mut f = std::fs::File::create(&corpus).unwrap();
        // Document 0 holds all three terms, 1 holds two, 2 holds one, 3 holds none.
        writeln!(f, "alpha beta gamma delta").unwrap();
        writeln!(f, "alpha beta epsilon zeta").unwrap();
        writeln!(f, "alpha eta theta iota").unwrap();
        writeln!(f, "kappa lambda mu nu").unwrap();
    }

    let report = index::build(&dbp, &corpus, true).unwrap();
    assert_eq!(report.n_docs, 4);
    assert!(report.stats.page_count > 0);
    assert_eq!(report.stats.page_size, db::PAGE_SIZE);
    // dbstat must see the FTS5 shadow tables, which is what makes the size breakdown
    // in the results meaningful.
    let names: Vec<String> =
        report.stats.per_table.unwrap().into_iter().map(|t| t.name).collect();
    assert!(names.iter().any(|n| n == "docs_data"), "dbstat saw {names:?}");

    let conn = Connection::open(&dbp).unwrap();
    let sql = format!(
        "SELECT rowid, bm25({t}) FROM {t} WHERE {t} MATCH ?1 ORDER BY bm25({t}) LIMIT ?2",
        t = index::TABLE
    );
    let expr = query::match_expression("alpha beta gamma").unwrap();
    let mut st = conn.prepare(&sql).unwrap();
    let hits: Vec<i64> = st
        .query_map(rusqlite::params![expr, 10i64], |r| r.get(0))
        .unwrap()
        .map(Result::unwrap)
        .collect();

    // OR semantics: the document with one of the three terms is still a candidate,
    // and BM25 puts the document with all three first. An implicit-AND query would
    // have returned only document 0 — that is the behaviour this baseline avoids.
    assert_eq!(hits.first(), Some(&0));
    assert_eq!(hits.len(), 3, "expected partial matches to be retrieved, got {hits:?}");
    assert!(!hits.contains(&3));
    drop(st);
    drop(conn);

    // --- the page counter --------------------------------------------------------
    let queries = vec![
        query::Query { qid: 0, kind: "known_item".into(), k: 3, text: "alpha beta gamma".into(), source_doc: Some(0) },
        query::Query { qid: 1, kind: "random".into(), k: 1, text: "kappa".into(), source_doc: None },
    ];
    let page_size = db::PAGE_SIZE as u64;
    let map = {
        let c = Connection::open(&dbp).unwrap();
        let pc: i64 = c.query_row("PRAGMA page_count", [], |r| r.get(0)).unwrap();
        db::page_table_map(&c, pc as u64).unwrap()
    };
    let measured = pages::measure(&dbp, &queries, &sql, 10, page_size, Some(&map)).unwrap();
    assert_eq!(measured.len(), 2);
    let total_pages = report.stats.page_count as usize;
    for m in &measured {
        assert!(m.distinct_pages > 0, "a query that returns rows must read pages");
        assert!(m.distinct_pages <= total_pages, "read more distinct pages than the file has");
        assert!(m.read_calls >= m.distinct_pages);
        assert!(m.contiguous_runs <= m.distinct_pages);
        assert_eq!(
            m.cache_miss as usize, m.distinct_pages,
            "VFS xRead count and DBSTATUS_CACHE_MISS disagree"
        );
        // Attribution must account for every page, and must name the inverted index.
        assert_eq!(m.by_table.iter().map(|(_, c)| c).sum::<usize>(), m.distinct_pages);
        assert!(
            m.by_table.iter().any(|(n, _)| n == "docs_data"),
            "no inverted-index pages attributed: {:?}", m.by_table
        );
    }

    // Opening and preparing alone already costs pages; those are the fixed cost of a
    // session, reported separately from per-query cost.
    let base = pages::open_prepare_baseline(&dbp, &sql).unwrap();
    assert!(base.pages(page_size).len() > 0);
    assert!(base.bytes() > 0);

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn read_traces_map_to_pages_and_runs() {
    // A trace is synthesised here so the mapping is checked independently of SQLite's
    // behaviour: offsets 0 (the 100-byte header read), page 2, page 3 and page 9.
    let mut t = vfs::ReadTrace::default();
    t.reads.push(vfs::Read { offset: 0, len: 100 });
    t.reads.push(vfs::Read { offset: 4096, len: 4096 });
    t.reads.push(vfs::Read { offset: 8192, len: 4096 });
    t.reads.push(vfs::Read { offset: 4096 * 8, len: 4096 });
    // Page numbers are 1-based, matching PRAGMA page_count and dbstat.
    assert_eq!(t.pages(4096).into_iter().collect::<Vec<_>>(), vec![1, 2, 3, 9]);
    // {1,2,3} and {9}: two range requests for a client that coalesces neighbours.
    assert_eq!(t.contiguous_runs(4096), 2);
    assert_eq!(t.read_calls(), 4);
    assert_eq!(t.bytes(), 100 + 3 * 4096);

    // A re-read of the same page is one page but two calls.
    let mut t = vfs::ReadTrace::default();
    t.reads.push(vfs::Read { offset: 4096, len: 4096 });
    t.reads.push(vfs::Read { offset: 4096, len: 4096 });
    assert_eq!(t.pages(4096).len(), 1);
    assert_eq!(t.read_calls(), 2);
}
