//! Connection setup, capability checks and file/page statistics.

use anyhow::{Context, Result};
use rusqlite::Connection;
use serde::Serialize;

/// Page size used for every database this crate builds.
///
/// 4096 bytes is SQLite's own default since 3.12 and, more to the point, it is the
/// unit a range-request client fetches. Fixing it explicitly keeps the page counts in
/// this baseline comparable with the ANN indexes, which will be measured in pages of
/// the same size. It must be set before the first table is created; afterwards only a
/// VACUUM can change it.
pub const PAGE_SIZE: u32 = 4096;

/// Fail loudly, and with the fix, if the linked SQLite has no FTS5.
///
/// `rusqlite`'s `bundled` feature compiles `-DSQLITE_ENABLE_FTS5`, but a build that
/// silently linked a system SQLite without it would otherwise fail much later with an
/// opaque "no such module" during table creation.
pub fn assert_fts5(conn: &Connection) -> Result<()> {
    let flagged: i32 = conn
        .query_row("SELECT sqlite_compileoption_used('ENABLE_FTS5')", [], |r| r.get(0))
        .context("querying sqlite_compileoption_used")?;
    let created = conn
        .execute_batch("CREATE VIRTUAL TABLE temp.fts5_probe USING fts5(x); DROP TABLE temp.fts5_probe;")
        .is_ok();
    anyhow::ensure!(
        flagged == 1 && created,
        "this build of SQLite {} has no FTS5 (compileoption_used(ENABLE_FTS5)={flagged}, \
         probe table created={created}).\n  Fix: build with `rusqlite = {{ features = [\"bundled\"] }}` \
         so libsqlite3-sys compiles the amalgamation with -DSQLITE_ENABLE_FTS5, or link a system \
         SQLite that has FTS5 enabled.",
        sqlite_version(conn)?
    );
    Ok(())
}

pub fn sqlite_version(conn: &Connection) -> Result<String> {
    Ok(conn.query_row("SELECT sqlite_version()", [], |r| r.get(0))?)
}

/// UTC timestamp from SQLite itself, so results carry a time without another crate.
pub fn utc_now(conn: &Connection) -> Result<String> {
    Ok(conn.query_row("SELECT strftime('%Y-%m-%dT%H:%M:%SZ','now')", [], |r| r.get(0))?)
}

#[derive(Serialize, Clone, Debug)]
pub struct DbStats {
    pub file_bytes: u64,
    pub page_size: u32,
    pub page_count: u64,
    pub freelist_count: u64,
    /// Pages and payload bytes per underlying table, from the `dbstat` virtual table.
    /// Absent when the scan was skipped.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub per_table: Option<Vec<TableStats>>,
}

#[derive(Serialize, Clone, Debug)]
pub struct TableStats {
    pub name: String,
    pub pages: u64,
    pub payload_bytes: u64,
}

pub fn stats(conn: &Connection, path: &std::path::Path, with_dbstat: bool) -> Result<DbStats> {
    let page_size: u32 = conn.query_row("PRAGMA page_size", [], |r| r.get(0))?;
    let page_count: i64 = conn.query_row("PRAGMA page_count", [], |r| r.get(0))?;
    let freelist: i64 = conn.query_row("PRAGMA freelist_count", [], |r| r.get(0))?;
    let per_table = if with_dbstat {
        // dbstat walks every b-tree page in the file, so it is a full read of the
        // database; cheap at 100 documents, minutes at 1M. It is the only way to
        // attribute pages to the FTS5 shadow tables without guessing.
        let mut st = conn.prepare(
            "SELECT name, count(*), sum(pgsize) FROM dbstat GROUP BY name ORDER BY 2 DESC",
        )?;
        let rows = st
            .query_map([], |r| {
                Ok(TableStats {
                    name: r.get(0)?,
                    pages: r.get::<_, i64>(1)? as u64,
                    payload_bytes: r.get::<_, i64>(2)? as u64,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Some(rows)
    } else {
        None
    };
    Ok(DbStats {
        file_bytes: std::fs::metadata(path)?.len(),
        page_size,
        page_count: page_count as u64,
        freelist_count: freelist as u64,
        per_table,
    })
}

/// Map every page of the file to the table it belongs to, via `dbstat`.
///
/// Page counts alone say how much a query costs; this says *what* it is paying for.
/// FTS5 spreads one logical table over several shadow tables — `%_data` holds the
/// inverted index, `%_content` the stored documents, `%_docsize` the per-document
/// lengths BM25 needs — and they have very different access patterns. Pages that
/// `dbstat` does not list are on the freelist or are lock/pointer-map pages; they are
/// labelled `unallocated`.
///
/// Returns a vector indexed by 1-based page number, holding an index into the returned
/// name table.
pub fn page_table_map(conn: &Connection, page_count: u64) -> Result<(Vec<u16>, Vec<String>)> {
    let mut names: Vec<String> = vec!["unallocated".to_string()];
    let mut map = vec![0u16; page_count as usize + 1];
    let mut st = conn.prepare("SELECT name, pageno FROM dbstat")?;
    let mut rows = st.query([])?;
    let mut last: Option<(String, u16)> = None;
    while let Some(r) = rows.next()? {
        let name: String = r.get(0)?;
        let pageno: i64 = r.get(1)?;
        let id = match &last {
            Some((n, id)) if *n == name => *id,
            _ => {
                let id = match names.iter().position(|n| *n == name) {
                    Some(i) => i as u16,
                    None => {
                        names.push(name.clone());
                        (names.len() - 1) as u16
                    }
                };
                last = Some((name, id));
                id
            }
        };
        if pageno >= 0 && (pageno as usize) < map.len() {
            map[pageno as usize] = id;
        }
    }
    Ok((map, names))
}

/// `SQLITE_DBSTATUS_CACHE_MISS` for this connection, resetting the counter.
///
/// Used as an independent cross-check on the VFS read counter: it is SQLite's own
/// tally of pages the pager had to fetch because they were not already cached. For
/// the CACHE_* verbs only `pCurrent` is meaningful and `resetFlg` zeroes it, so each
/// call returns the misses since the previous call.
pub fn take_cache_miss(conn: &Connection) -> i32 {
    let mut cur: i32 = 0;
    let mut hi: i32 = 0;
    unsafe {
        rusqlite::ffi::sqlite3_db_status(
            conn.handle(),
            rusqlite::ffi::SQLITE_DBSTATUS_CACHE_MISS,
            &mut cur,
            &mut hi,
            1,
        );
    }
    cur
}
