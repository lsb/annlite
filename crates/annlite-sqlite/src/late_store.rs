//! Storing a `LateIndex` in SQLite, with page cost attributable per stage.
//!
//! # The problem the dense format does not have
//!
//! The Vamana format in `format.rs` gets its page arithmetic for free: every record
//! is the same size, so node `i` lives at byte `i * record_bytes` and the page
//! holding it follows from its id (RESEARCH_LOG.md section 11.3). A late-interaction
//! document breaks that. Its record is a list of one centroid id per token, and on
//! this corpus that is a mean of 147 tokens and a maximum of 830 -- a 5.6x spread.
//! There is no record size that is both correct and predictable.
//!
//! Three ways out were considered:
//!
//! * **An offsets table.** `SELECT off FROM late_offsets WHERE id = ?` before the
//!   codes can be requested. Rejected: that is one extra *dependent* round-trip per
//!   candidate document, which is precisely FTS5's `%_docsize` pattern -- the thing
//!   section 9.4 measured at a 79x page penalty and the reason this project exists.
//! * **Length-bucketed padding.** Pad every document to its bucket's maximum so the
//!   arithmetic returns. Rejected on measured numbers: on this corpus's length
//!   distribution, padding every document to the maximum costs **5.64x**, four
//!   quantile buckets still cost **2.00x**, and sixteen buckets -- sixteen arenas
//!   and a per-document bucket lookup -- still cost **1.22x**. The directory below
//!   costs **1.007x**. Against a compressed index of 646 bytes per document
//!   (section 15.2), padding gives back a large fraction of what PLAID just won,
//!   and buys nothing the directory does not already give.
//! * **A resident offsets directory**, which is what this module does.
//!
//! # The form chosen
//!
//! Each variable-length collection is stored as one contiguous **arena**: a single
//! BLOB holding every document's codes end to end, in document order. Alongside it,
//! in the metadata that a client downloads once, sits an **offsets directory** of
//! `n + 1` `u32` token offsets. Document `i` is then the byte range
//! `[off[i] * 4, off[i + 1] * 4)` of the arena, and the pages covering that range
//! follow by arithmetic exactly as they do for a fixed-size record.
//!
//! The directory costs **4 bytes per document** -- 13 KB on this corpus (0.7% of the
//! code arena it indexes), 4 MB at a million documents. That is the same order as
//! the PQ codebook, and it is fetched once per session in one sequential range
//! rather than once per candidate. The trade is explicit: a small fixed resident
//! cost in exchange for keeping the id-to-page arithmetic that the whole measurement
//! depends on.
//!
//! The cost that is *not* paid in bytes is that the directory has to be present
//! before the first byte range can be issued. A client with a cold cache therefore
//! pays one extra sequential fetch -- the resident bundle of centroid table and both
//! directories, 111 KB to 405 KB depending on `k` -- before it can start. That is
//! the same shape of fixed cost as the dense index's resident PQ codes (section
//! 11.5), and it amortises over a session rather than over a query.
//!
//! The same directory serves the full token vectors, at `dim * 4` bytes per token
//! instead of 4, so stage 3 needs no second directory.
//!
//! # Why a BLOB rather than a row per document
//!
//! A row per document would be read with plain `SELECT ... WHERE id = ?`, but the
//! page it lands on is then decided by SQLite's cell packing, and finding it costs a
//! descent through the b-tree's interior pages. An arena is read with
//! `sqlite3_blob_open` plus a byte offset -- core SQLite, present in every stock
//! build including the WASM one, no virtual table and no loadable extension -- and
//! it maps a byte range onto pages the client can request directly. SQLite stores an
//! oversized BLOB as a chain of overflow pages allocated consecutively during a
//! single insert, so a byte range is a *run* of consecutive pages, which is the
//! shape that coalesces into one HTTP request.
//!
//! That contiguity is asserted rather than assumed: [`Arena::open`] reads the real
//! page numbers out of `dbstat` and [`Arena::is_contiguous`] reports whether they
//! are consecutive.

use annlite_core::late::{LateIndex, MultiVector};
use anyhow::{Context, Result};
use rusqlite::{params, Connection};

pub const PAGE_BYTES: usize = 4096;

/// Bytes of an overflow page available to payload: the first four are the pointer
/// to the next page in the chain.
pub const OVERFLOW_USABLE: usize = PAGE_BYTES - 4;

/// What a stored late index says about itself.
pub struct LateDbMeta {
    pub dim: usize,
    pub k: usize,
    pub count: usize,
    pub tokens: usize,
    /// Bytes a client must hold before it can issue a single byte range: the
    /// centroid table plus both offsets directories.
    pub resident_bytes: usize,
    pub write_seconds: f64,
}

pub fn init_late_schema(conn: &Connection) -> Result<()> {
    conn.pragma_update(None, "page_size", PAGE_BYTES as i64)?;
    conn.pragma_update(None, "journal_mode", "OFF")?;
    conn.pragma_update(None, "synchronous", "OFF")?;
    conn.execute_batch(
        r#"
        -- Small and resident: scalars, the centroid table, and the two offsets
        -- directories. One sequential download, then no further lookups.
        CREATE TABLE IF NOT EXISTS late_meta (
            key   TEXT PRIMARY KEY,
            value BLOB NOT NULL
        );
        -- Stage 1. Document ids per centroid, u32 little-endian, ascending within a
        -- centroid, centroids in id order. Sliced by late_meta.posting_offsets.
        CREATE TABLE IF NOT EXISTS late_postings (
            id   INTEGER PRIMARY KEY CHECK (id = 0),
            data BLOB NOT NULL
        );
        -- Stage 2. One u32 centroid id per token, documents end to end in id order.
        -- Sliced by late_meta.doc_offsets. This is the compressed index: four bytes
        -- per token against dim*4 for the vector it stands in for.
        CREATE TABLE IF NOT EXISTS late_codes (
            id   INTEGER PRIMARY KEY CHECK (id = 0),
            data BLOB NOT NULL
        );
        -- Stage 3. Full float32 token vectors, same document order, same directory
        -- scaled by dim. Kept in its own table so that reading one document's
        -- vectors can never evict or share a page with its codes -- the two stages
        -- are supposed to be separately attributable, and a shared page would make
        -- them not be.
        CREATE TABLE IF NOT EXISTS late_tokens (
            id   INTEGER PRIMARY KEY CHECK (id = 0),
            data BLOB NOT NULL
        );
        "#,
    )?;
    Ok(())
}

/// Write the index and the exact token vectors it compresses.
///
/// Arenas are inserted one at a time into a fresh database so that each one's
/// overflow chain is allocated as a single consecutive run of pages; interleaving
/// the inserts would scatter them and silently inflate every `contiguous_runs`
/// figure the benchmark reports.
pub fn write_late_index(
    conn: &mut Connection,
    idx: &LateIndex,
    docs: &[MultiVector],
) -> Result<LateDbMeta> {
    anyhow::ensure!(idx.len() == docs.len(), "index has {} documents, corpus has {}", idx.len(), docs.len());
    let t0 = std::time::Instant::now();
    init_late_schema(conn)?;

    let offsets = idx.doc_offsets().to_vec();
    anyhow::ensure!(
        *offsets.last().unwrap_or(&0) as usize == idx.total_tokens(),
        "offsets directory does not cover every token"
    );

    // Postings arena, plus its own directory. `k` is in the thousands, so this
    // directory is a few kilobytes and rides along with the centroid table.
    let mut posting_offsets: Vec<u32> = Vec::with_capacity(idx.k + 1);
    let mut postings_arena: Vec<u8> = Vec::new();
    posting_offsets.push(0);
    for c in 0..idx.k as u32 {
        for &d in idx.postings(c) {
            postings_arena.extend_from_slice(&d.to_le_bytes());
        }
        posting_offsets.push((postings_arena.len() / 4) as u32);
    }

    let mut codes_arena: Vec<u8> = Vec::with_capacity(idx.total_tokens() * 4);
    for d in 0..idx.len() as u32 {
        for &c in idx.doc_codes(d) {
            codes_arena.extend_from_slice(&c.to_le_bytes());
        }
    }

    let mut tokens_arena: Vec<u8> = Vec::with_capacity(idx.total_tokens() * idx.dim * 4);
    for d in docs {
        for x in &d.data {
            tokens_arena.extend_from_slice(&x.to_le_bytes());
        }
    }
    anyhow::ensure!(
        tokens_arena.len() == codes_arena.len() / 4 * idx.dim * 4,
        "token arena and code arena describe different corpora"
    );

    let tx = conn.transaction()?;
    tx.execute("INSERT OR REPLACE INTO late_postings(id, data) VALUES (0, ?1)", params![postings_arena])?;
    tx.execute("INSERT OR REPLACE INTO late_codes(id, data) VALUES (0, ?1)", params![codes_arena])?;
    tx.execute("INSERT OR REPLACE INTO late_tokens(id, data) VALUES (0, ?1)", params![tokens_arena])?;

    let centroid_blob = f32_le(&idx.centroids);
    let doc_off_blob = u32_le(&offsets);
    let post_off_blob = u32_le(&posting_offsets);
    {
        let mut stmt = tx.prepare("INSERT OR REPLACE INTO late_meta(key, value) VALUES (?1, ?2)")?;
        for (k, v) in [
            ("dim", idx.dim.to_string()),
            ("k", idx.k.to_string()),
            ("count", idx.len().to_string()),
            ("tokens", idx.total_tokens().to_string()),
        ] {
            stmt.execute(params![k, v.as_bytes()])?;
        }
        stmt.execute(params!["centroids", centroid_blob.as_slice()])?;
        stmt.execute(params!["doc_offsets", doc_off_blob.as_slice()])?;
        stmt.execute(params!["posting_offsets", post_off_blob.as_slice()])?;
    }
    tx.commit()?;

    Ok(LateDbMeta {
        dim: idx.dim,
        k: idx.k,
        count: idx.len(),
        tokens: idx.total_tokens(),
        resident_bytes: centroid_blob.len() + doc_off_blob.len() + post_off_blob.len(),
        write_seconds: t0.elapsed().as_secs_f64(),
    })
}

fn f32_le(xs: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(xs.len() * 4);
    for x in xs {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

fn u32_le(xs: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(xs.len() * 4);
    for x in xs {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

/// Where an arena's bytes physically are, as SQLite actually laid them out.
///
/// The first `local_bytes` of the BLOB sit in the b-tree leaf cell; the rest live in
/// a chain of overflow pages. Mapping a byte offset to a page is therefore two
/// cases, and this struct holds the measured page numbers rather than predicting
/// them, so the mapping is anchored in the file that exists.
#[derive(Clone, Debug)]
pub struct Arena {
    pub table: String,
    pub len: usize,
    pub leaf_page: usize,
    pub local_bytes: usize,
    /// Absolute page numbers of the overflow chain, in chain order.
    pub overflow: Vec<usize>,
}

impl Arena {
    pub fn open(conn: &Connection, table: &str) -> Result<Self> {
        let len: i64 = conn
            .query_row(&format!("SELECT length(data) FROM {table} WHERE id = 0"), [], |r| r.get(0))
            .with_context(|| format!("reading {table}"))?;
        let len = len as usize;

        // dbstat's `path` for an overflow page is the b-tree page's path followed by
        // '+' and the zero-based index in the chain, hex and zero-padded, so sorting
        // by it recovers chain order regardless of how pages were allocated.
        let mut stmt = conn
            .prepare(
                "SELECT pageno, pagetype, payload, path FROM dbstat WHERE name = ?1 ORDER BY path",
            )
            .context("querying dbstat; the build must have SQLITE_ENABLE_DBSTAT_VTAB")?;
        let rows = stmt.query_map([table], |r| {
            Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?, r.get::<_, i64>(2)?))
        })?;

        let mut leaf_page = 0usize;
        let mut leaf_payload = 0usize;
        let mut overflow = Vec::new();
        for row in rows {
            let (pageno, pagetype, payload) = row?;
            match pagetype.as_str() {
                "leaf" => {
                    anyhow::ensure!(leaf_page == 0, "{table} has more than one leaf page");
                    leaf_page = pageno as usize;
                    leaf_payload = payload as usize;
                }
                "overflow" => overflow.push(pageno as usize),
                _ => {}
            }
        }
        anyhow::ensure!(leaf_page != 0, "{table} has no leaf page");

        let header = record_header_bytes(len);
        let local_bytes = leaf_payload.saturating_sub(header);
        let expected = len.saturating_sub(local_bytes).div_ceil(OVERFLOW_USABLE);
        anyhow::ensure!(
            expected == overflow.len(),
            "{table}: {} bytes with {local_bytes} local predicts {expected} overflow pages, \
             dbstat reports {}",
            len,
            overflow.len()
        );

        Ok(Self { table: table.to_string(), len, leaf_page, local_bytes, overflow })
    }

    /// Absolute page numbers covering `[off, off + n)`, ascending, deduplicated.
    pub fn pages_for(&self, off: usize, n: usize) -> Vec<usize> {
        if n == 0 {
            return Vec::new();
        }
        let end = (off + n).min(self.len);
        let mut out = Vec::new();
        if off < self.local_bytes {
            out.push(self.leaf_page);
        }
        if end > self.local_bytes {
            let a = off.max(self.local_bytes) - self.local_bytes;
            let b = end - self.local_bytes - 1;
            for i in (a / OVERFLOW_USABLE)..=(b / OVERFLOW_USABLE) {
                out.push(self.overflow[i]);
            }
        }
        out
    }

    /// Whether the overflow chain occupies consecutive pages. When it does, a byte
    /// range is one range request; when it does not, it is several, and every
    /// `contiguous_runs` figure for this arena is inflated accordingly.
    pub fn is_contiguous(&self) -> bool {
        self.overflow.windows(2).all(|w| w[1] == w[0] + 1)
    }

    pub fn pages(&self) -> usize {
        self.overflow.len() + 1
    }
}

/// Bytes SQLite spends on the record header of a `(INTEGER PRIMARY KEY, BLOB)` row.
///
/// The rowid alias is stored as serial type 0 and occupies no body bytes, so the
/// header is its own length, one byte for that zero, and the blob's serial type
/// `12 + 2 * len`. Needed exactly, because it is the offset at which the blob's
/// bytes start inside the leaf cell and therefore decides which page byte zero is on.
fn record_header_bytes(blob_len: usize) -> usize {
    let serial = 12 + 2 * blob_len as u64;
    let body = 1 + varint_len(serial);
    let mut header = body + 1;
    if varint_len(header as u64) > 1 {
        header = body + varint_len(header as u64);
    }
    header
}

fn varint_len(mut v: u64) -> usize {
    if v >= 1 << 56 {
        return 9;
    }
    let mut n = 1;
    while v >= 0x80 {
        v >>= 7;
        n += 1;
    }
    n
}

/// Bytes a table occupies on disk, from `dbstat`. Used to report the size of the
/// part of the file a configuration actually reads, rather than the whole file.
pub fn table_bytes(conn: &Connection, table: &str) -> Result<usize> {
    let n: i64 = conn.query_row(
        "SELECT COALESCE(SUM(pgsize), 0) FROM dbstat WHERE name = ?1",
        [table],
        |r| r.get(0),
    )?;
    Ok(n as usize)
}
