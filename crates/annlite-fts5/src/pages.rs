//! Per-query page-access measurement.
//!
//! The project's binding constraint is HTTP round-trips, so the question is not how
//! long a query takes on a warm local cache but **which pages of the file it has to
//! read**. Two independent instruments are used on every query and their answers are
//! reported side by side; agreement between them is the evidence that either is right.
//!
//! * **`xRead` interception** ([`crate::vfs`]) — the exact byte ranges SQLite asked the
//!   operating system for, mapped to 4 KiB pages.
//! * **`SQLITE_DBSTATUS_CACHE_MISS`** — SQLite's own count of pages the pager could not
//!   serve from its cache.
//!
//! ### Making the cache cold without losing the schema
//!
//! A page only reaches `xRead` if the pager does not already hold it, so measuring a
//! second query on a warm connection would report near zero. The measurement therefore
//! issues `PRAGMA shrink_memory` before each query, which calls
//! `sqlite3_db_release_memory()` and drops the clean pages of the pager cache. The
//! connection and its prepared statement survive, so the *schema* is not re-parsed —
//! what is measured is the pages the query itself needs, plus the page-1 header re-read
//! that SQLite performs at the start of every read transaction.
//!
//! A second, blunter method is used as a cross-check on a sample: open a brand-new
//! connection per query. That includes schema loading, so its counts are higher by a
//! roughly constant amount; the `open_prepare_baseline` field records exactly how much,
//! measured by opening and preparing without running anything.
//!
//! ### What these numbers are not
//!
//! * They are **logical** reads. The operating system's cache is still warm, so they do
//!   not say anything about disk I/O; they say what a client with no local copy of the
//!   file would have to fetch.
//! * A real `sql.js-httpvfs` client fetches fixed-size *blocks* (often much larger than
//!   a page) and caches them across queries, so `distinct_pages` is an upper bound on
//!   its round-trips for a cold cache and `contiguous_runs` a lower bound.
//! * The first query of a session additionally pays for the schema; here that cost is
//!   reported separately rather than amortised into every query.

use crate::db;
use crate::vfs;
use anyhow::Result;
use rusqlite::{Connection, OpenFlags};
use serde::Serialize;
use std::path::Path;

/// What one query cost in reads.
#[derive(Serialize, Clone, Debug)]
pub struct PageOutcome {
    pub qid: usize,
    pub kind: String,
    pub k: usize,
    /// Distinct 4 KiB pages touched: the pessimistic round-trip count (one request
    /// per page).
    pub distinct_pages: usize,
    /// Maximal runs of consecutive pages: the optimistic round-trip count for a client
    /// that coalesces neighbouring pages into one range request.
    pub contiguous_runs: usize,
    /// `xRead` calls, including any page read more than once.
    pub read_calls: usize,
    pub bytes_read: u64,
    /// SQLite's own cache-miss count for the same query.
    pub cache_miss: i32,
    /// Distinct pages broken down by the table that owns them.
    pub by_table: Vec<(String, usize)>,
}

/// Open a read-only connection through the counting VFS.
pub fn open_counting(db_path: &Path) -> Result<Connection> {
    let vfs_name = vfs::register().map_err(anyhow::Error::msg)?;
    let conn = Connection::open_with_flags_and_vfs(
        db_path,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_URI | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        vfs_name,
    )?;
    // Belt and braces: the shim's iVersion already forbids mmap, and this makes the
    // intent explicit for anyone reading a results file.
    conn.pragma_update(None, "mmap_size", 0)?;
    // Large enough that no page is evicted *within* one query; shrink_memory still
    // clears it between queries.
    conn.pragma_update(None, "cache_size", -262_144)?;
    Ok(conn)
}

/// Pages read purely to open a connection and prepare the query statement.
///
/// This is the fixed cost the first query of a browser session pays before it can look
/// anything up: the database header, the schema table, and FTS5's own configuration
/// rows. Reported on its own so it is not silently folded into every query.
pub fn open_prepare_baseline(db_path: &Path, sql: &str) -> Result<vfs::ReadTrace> {
    vfs::record_start();
    let conn = open_counting(db_path)?;
    let _stmt = conn.prepare(sql)?;
    Ok(vfs::record_take())
}

/// Run every query with a cold pager cache, recording what each one read.
pub fn measure(
    db_path: &Path,
    queries: &[crate::query::Query],
    sql: &str,
    // Bound to the statement's LIMIT; SQLite reads -1 as "no limit".
    limit: i64,
    split: crate::query::TermSplit,
    page_size: u64,
    table_map: Option<&(Vec<u16>, Vec<String>)>,
) -> Result<Vec<PageOutcome>> {
    let conn = open_counting(db_path)?;
    let mut stmt = conn.prepare(sql)?;
    let mut out = Vec::with_capacity(queries.len());
    for q in queries {
        let Some(expr) = crate::query::match_expression_with(&q.text, split) else { continue };
        conn.execute_batch("PRAGMA shrink_memory")?;
        let _ = db::take_cache_miss(&conn);
        vfs::record_start();
        let mut rows = stmt.query(rusqlite::params![expr, limit])?;
        while let Some(r) = rows.next()? {
            let _: i64 = r.get(0)?;
        }
        drop(rows);
        let trace = vfs::record_take();
        let cache_miss = db::take_cache_miss(&conn);
        let touched = trace.pages(page_size);
        let by_table = match table_map {
            Some((map, names)) => {
                let mut counts = vec![0usize; names.len()];
                for p in &touched {
                    let idx = *map.get(*p as usize).unwrap_or(&0) as usize;
                    counts[idx] += 1;
                }
                names.iter().cloned().zip(counts).filter(|(_, c)| *c > 0).collect()
            }
            None => Vec::new(),
        };
        out.push(PageOutcome {
            qid: q.qid,
            kind: q.kind.clone(),
            k: q.k,
            by_table,
            distinct_pages: touched.len(),
            contiguous_runs: trace.contiguous_runs(page_size),
            read_calls: trace.read_calls(),
            bytes_read: trace.bytes(),
            cache_miss,
        });
    }
    Ok(out)
}

/// Cross-check: a fresh connection per query, so nothing at all is cached.
///
/// Returns distinct pages per query including connection setup, for the first
/// `sample` queries.
pub fn measure_fresh_connection(
    db_path: &Path,
    queries: &[crate::query::Query],
    sql: &str,
    limit: i64,
    split: crate::query::TermSplit,
    page_size: u64,
    sample: usize,
) -> Result<Vec<usize>> {
    let mut out = Vec::new();
    for q in queries.iter().take(sample) {
        let Some(expr) = crate::query::match_expression_with(&q.text, split) else { continue };
        vfs::record_start();
        {
            let conn = open_counting(db_path)?;
            let mut stmt = conn.prepare(sql)?;
            let mut rows = stmt.query(rusqlite::params![expr, limit])?;
            while let Some(r) = rows.next()? {
                let _: i64 = r.get(0)?;
            }
        }
        out.push(vfs::record_take().pages(page_size).len());
    }
    Ok(out)
}
