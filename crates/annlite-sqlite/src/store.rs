//! Writing and reading an annlite index in a SQLite database.
//!
//! Tables are deliberately ordinary: a rowid table of fixed-size blobs, no virtual
//! table and no loadable extension. That is what lets the result be read by a stock
//! SQLite build, including the `sql.js-httpvfs` WASM one in a browser, without
//! shipping a custom binary. The index lives in the *layout*, not in the engine.
//!
//! Node ids are the rowids, which is the whole point: SQLite stores a rowid table's
//! leaf pages in rowid order, so choosing ids chooses pages.

use annlite_core::layout::{Ordering, Permutation};
use annlite_core::pq::ProductQuantizer;
use annlite_core::vamana::Vamana;
use annlite_core::vectors::Vectors;
use anyhow::{Context, Result};
use rusqlite::{params, Connection};

use crate::format::RecordFormat;

pub const PAGE_BYTES: usize = 4096;

pub struct IndexMeta {
    pub dim: usize,
    pub m: usize,
    pub r: usize,
    pub count: usize,
    pub medoid_new_id: u32,
    pub ordering: String,
}

/// Create the schema. `page_size` must be set before any table exists, hence the
/// pragma ordering here.
pub fn init_schema(conn: &Connection) -> Result<()> {
    conn.pragma_update(None, "page_size", PAGE_BYTES as i64)?;
    conn.pragma_update(None, "journal_mode", "OFF")?;
    conn.pragma_update(None, "synchronous", "OFF")?;
    conn.execute_batch(
        r#"
        CREATE TABLE IF NOT EXISTS annlite_meta (
            key   TEXT PRIMARY KEY,
            value BLOB NOT NULL
        );
        -- Fixed-size records in rowid order. Nothing else may be inserted into this
        -- table: a variable-length row would break the id-to-page arithmetic.
        CREATE TABLE IF NOT EXISTS annlite_nodes (
            id  INTEGER PRIMARY KEY,
            rec BLOB NOT NULL
        );
        -- Full-precision vectors, used only to rerank a shallow candidate pool.
        -- Kept in a separate table so traversal never pulls them into a page it
        -- would otherwise have used for adjacency.
        CREATE TABLE IF NOT EXISTS annlite_vectors (
            id INTEGER PRIMARY KEY,
            v  BLOB NOT NULL
        );
        CREATE TABLE IF NOT EXISTS annlite_docs (
            id   INTEGER PRIMARY KEY,
            body TEXT
        );
        "#,
    )?;
    Ok(())
}

/// Write an index, renumbering nodes through `perm`.
///
/// Rows are inserted in ascending new id so SQLite fills leaf pages sequentially
/// rather than splitting them, which is what makes the page of node `i` predictable.
#[allow(clippy::too_many_arguments)]
pub fn write_index(
    conn: &mut Connection,
    graph: &Vamana,
    pq: &ProductQuantizer,
    codes: &[u8],
    vectors: &Vectors,
    perm: &Permutation,
    ordering: Ordering,
    docs: Option<&[String]>,
) -> Result<IndexMeta> {
    let fmt = RecordFormat { m: pq.m, r: graph.params.r };
    let n = graph.len();
    anyhow::ensure!(perm.len() == n, "permutation covers {} nodes, graph has {n}", perm.len());

    init_schema(conn)?;
    let tx = conn.transaction()?;
    {
        let mut node_stmt = tx.prepare("INSERT INTO annlite_nodes(id, rec) VALUES (?1, ?2)")?;
        let mut vec_stmt = tx.prepare("INSERT INTO annlite_vectors(id, v) VALUES (?1, ?2)")?;
        let mut doc_stmt = tx.prepare("INSERT INTO annlite_docs(id, body) VALUES (?1, ?2)")?;

        for new_id in 0..n {
            let old = perm.old_id_of[new_id] as usize;
            // Neighbour ids are rewritten into the new numbering; storing old ids
            // would make every hop an indirection through a translation table.
            let neighbors: Vec<u32> = graph
                .neighbors(old as u32)
                .iter()
                .map(|&o| perm.new_id_of[o as usize])
                .collect();
            let code = &codes[old * fmt.m..(old + 1) * fmt.m];
            node_stmt.execute(params![new_id as i64, fmt.encode(code, &neighbors)])?;

            let v = vectors.row(old);
            let mut bytes = Vec::with_capacity(v.len() * 4);
            for x in v {
                bytes.extend_from_slice(&x.to_le_bytes());
            }
            vec_stmt.execute(params![new_id as i64, bytes])?;

            if let Some(d) = docs {
                doc_stmt.execute(params![new_id as i64, &d[old]])?;
            }
        }
    }

    let mut codebook = Vec::with_capacity(pq.centroids.len() * 4);
    for x in &pq.centroids {
        codebook.extend_from_slice(&x.to_le_bytes());
    }
    let medoid_new = perm.new_id_of[graph.medoid as usize];
    {
        let mut meta = tx.prepare("INSERT OR REPLACE INTO annlite_meta(key, value) VALUES (?1, ?2)")?;
        for (k, v) in [
            ("dim", pq.dim.to_string()),
            ("m", pq.m.to_string()),
            ("dsub", pq.dsub.to_string()),
            ("r", fmt.r.to_string()),
            ("count", n.to_string()),
            ("medoid", medoid_new.to_string()),
            ("ordering", format!("{ordering:?}")),
            ("record_bytes", fmt.len().to_string()),
        ] {
            meta.execute(params![k, v.as_bytes()])?;
        }
        meta.execute(params!["pq_centroids", codebook])?;
    }
    tx.commit()?;

    Ok(IndexMeta {
        dim: pq.dim,
        m: pq.m,
        r: fmt.r,
        count: n,
        medoid_new_id: medoid_new,
        ordering: format!("{ordering:?}"),
    })
}

/// Measured page usage of `annlite_nodes`, from SQLite's own `dbstat`.
///
/// Reported rather than computed, because the arithmetic estimate in
/// [`RecordFormat::records_per_page`] ignores details of cell packing that only the
/// engine knows.
pub fn node_page_stats(conn: &Connection) -> Result<(usize, f64)> {
    let pages: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM dbstat WHERE name='annlite_nodes' AND pagetype='leaf'",
            [],
            |r| r.get(0),
        )
        .context("querying dbstat; the build must have SQLITE_ENABLE_DBSTAT_VTAB")?;
    let rows: i64 = conn.query_row("SELECT COUNT(*) FROM annlite_nodes", [], |r| r.get(0))?;
    Ok((pages as usize, if pages > 0 { rows as f64 / pages as f64 } else { 0.0 }))
}
