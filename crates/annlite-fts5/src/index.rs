//! Building the FTS5 index from a corpus file.

use crate::db;
use anyhow::{Context, Result};
use rusqlite::Connection;
use serde::Serialize;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::time::Instant;

/// Name of the FTS5 table; also the name BM25 is called with.
pub const TABLE: &str = "docs";

/// Pragmas applied before the bulk load, recorded in the results so a run is
/// self-describing.
///
/// Rationale, since each one moves the numbers:
///
/// * `page_size=4096` — the unit a range client fetches (see [`db::PAGE_SIZE`]). Must
///   precede table creation.
/// * `journal_mode=OFF` — this database is a build artifact that `make` reproduces
///   from the corpus in seconds, so paying for crash recovery during the load would
///   measure durability we do not need. It roughly halves write volume. A rollback
///   journal would also leave the *queried* file identical, so this affects build time
///   only, not any later measurement.
/// * `synchronous=OFF` — same argument: no fsync barriers during a rebuildable load.
/// * `cache_size=-262144` (256 MiB) — FTS5 writes segments, then reads and rewrites
///   them when it merges. With a small cache those merges become read-modify-write
///   against the file; 256 MiB keeps the working set of a merge in memory and is the
///   single biggest lever on build throughput.
/// * `temp_store=MEMORY` — merges use temporary storage; keep it off disk.
///
/// Deliberately *not* used: `PRAGMA mmap_size`. The query phase must see every page
/// fetch as an `xRead`, and memory-mapped reads bypass the VFS entirely.
pub const BULK_LOAD_PRAGMAS: &[(&str, &str)] = &[
    ("page_size", "4096"),
    ("journal_mode", "OFF"),
    ("synchronous", "OFF"),
    ("cache_size", "-262144"),
    ("temp_store", "MEMORY"),
];

#[derive(Serialize, Clone, Debug)]
pub struct BuildReport {
    pub n_docs: u64,
    pub corpus_bytes: u64,
    pub build_secs: f64,
    pub docs_per_sec: f64,
    pub mib_per_sec: f64,
    pub stats: db::DbStats,
    pub bytes_per_doc: f64,
    pub pragmas: Vec<(String, String)>,
}

/// Create the database and load `corpus` into an FTS5 table, one row per line.
///
/// The row's `rowid` is the 0-based line number, which is the document id the query
/// sets refer to, so retrieval results need no translation table.
///
/// The load runs inside a single transaction. FTS5 buffers postings in memory and
/// flushes a segment when that buffer fills; committing per row would instead force a
/// segment per row and turn the load into a merge storm. Streaming the corpus line by
/// line keeps peak memory independent of corpus size — at 1M documents the file is
/// 454 MB and reading it whole would be a needless resident copy.
pub fn build(db_path: &Path, corpus: &Path, with_dbstat: bool) -> Result<BuildReport> {
    if db_path.exists() {
        std::fs::remove_file(db_path)?;
    }
    if let Some(dir) = db_path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let corpus_bytes = std::fs::metadata(corpus)
        .with_context(|| format!("corpus {} is missing; run `make corpora queries`", corpus.display()))?
        .len();

    let conn = Connection::open(db_path)?;
    db::assert_fts5(&conn)?;
    for (k, v) in BULK_LOAD_PRAGMAS {
        conn.pragma_update(None, k, *v)?;
    }
    conn.execute_batch(&format!("CREATE VIRTUAL TABLE {TABLE} USING fts5(body)"))?;

    let mut reader = BufReader::with_capacity(1 << 20, std::fs::File::open(corpus)?);
    let mut line = String::new();
    let mut n_docs: u64 = 0;

    let t0 = Instant::now();
    conn.execute_batch("BEGIN")?;
    {
        let mut ins = conn.prepare(&format!("INSERT INTO {TABLE}(rowid, body) VALUES (?1, ?2)"))?;
        loop {
            line.clear();
            if reader.read_line(&mut line)? == 0 {
                break;
            }
            let body = line.trim_end_matches('\n');
            if body.is_empty() {
                continue;
            }
            ins.execute(rusqlite::params![n_docs as i64, body])?;
            n_docs += 1;
            if n_docs % 100_000 == 0 {
                eprintln!("  … {n_docs} documents inserted ({:.1}s)", t0.elapsed().as_secs_f64());
            }
        }
    }
    conn.execute_batch("COMMIT")?;
    let build_secs = t0.elapsed().as_secs_f64();

    let stats = db::stats(&conn, db_path, with_dbstat)?;
    conn.close().map_err(|(_, e)| e)?;

    Ok(BuildReport {
        n_docs,
        corpus_bytes,
        build_secs,
        docs_per_sec: n_docs as f64 / build_secs,
        mib_per_sec: corpus_bytes as f64 / build_secs / (1024.0 * 1024.0),
        bytes_per_doc: stats.file_bytes as f64 / n_docs.max(1) as f64,
        stats,
        pragmas: BULK_LOAD_PRAGMAS.iter().map(|(a, b)| (a.to_string(), b.to_string())).collect(),
    })
}

#[derive(Serialize, Clone, Debug)]
pub struct OptimizeReport {
    pub optimize_secs: f64,
    pub stats: db::DbStats,
}

/// Run `INSERT INTO docs(docs) VALUES('optimize')`.
///
/// FTS5 normally leaves the index as several b-tree segments, and a query must visit
/// the doclist of every segment for every term. `optimize` merges them into one. On a
/// local disk that mostly saves CPU; over HTTP it is the difference between one seek
/// per segment per term and one, which is why it is measured as its own phase with
/// query latency and page counts taken on both sides of it.
pub fn optimize(db_path: &Path, with_dbstat: bool) -> Result<OptimizeReport> {
    let conn = Connection::open(db_path)?;
    for (k, v) in BULK_LOAD_PRAGMAS.iter().filter(|(k, _)| *k != "page_size") {
        conn.pragma_update(None, k, *v)?;
    }
    let t0 = Instant::now();
    conn.execute_batch(&format!("INSERT INTO {TABLE}({TABLE}) VALUES('optimize')"))?;
    let optimize_secs = t0.elapsed().as_secs_f64();
    let stats = db::stats(&conn, db_path, with_dbstat)?;
    conn.close().map_err(|(_, e)| e)?;
    Ok(OptimizeReport { optimize_secs, stats })
}

/// Number of FTS5 segments in the index, read from the structure record.
///
/// Reported before and after `optimize` because it is the mechanism behind the change
/// in page counts, not just a side effect of it.
pub fn segment_count(conn: &Connection) -> Result<i64> {
    // %_data row 1 (the 'structure' record) is FTS5-internal, so rather than decode
    // it, count the distinct segment ids visible in the segment-id namespace of the
    // %_idx shadow table, which has one row per segment b-tree level entry.
    Ok(conn.query_row(
        &format!("SELECT count(DISTINCT segid) FROM {TABLE}_idx"),
        [],
        |r| r.get(0),
    )?)
}
