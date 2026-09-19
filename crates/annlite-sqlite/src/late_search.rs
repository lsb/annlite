//! Staged late-interaction retrieval out of SQLite, counting what each stage costs.
//!
//! The three stages of section 12 touch three different arenas, and the point of
//! measuring them separately is that they are not remotely the same size. Stage 1
//! reads posting lists, stage 2 reads four bytes per token, stage 3 reads
//! `dim * 4` bytes per token -- a 48x ratio between the last two per document read.
//! The dense index found that reranking was the expensive half (RESEARCH_LOG.md
//! section 11.5); attributing pages per stage is what lets the same question be
//! asked here rather than assumed.
//!
//! Each stage is also exactly **one dependent round-trip**. Stage 1 knows every
//! centroid it wants the moment the resident centroid table has been scored against
//! the query, so all of its ranges are issuable at once; stage 2 knows every
//! candidate as soon as stage 1 returns; stage 3 knows the pool as soon as stage 2
//! has sorted. A whole query is therefore two hops without reranking and three with
//! -- against the dense traversal's eight-plus, because there is no graph to walk.
//! That is the structural difference between the two systems, and it is why `hops`
//! is reported separately from pages.

use annlite_core::late::{centroid_maxsim, maxsim, probe_centroids, MultiVector};
use anyhow::Result;
use rusqlite::{Connection, DatabaseName};
use std::collections::HashSet;

use crate::late_store::{table_bytes, Arena, PAGE_BYTES};

/// An index opened out of a database, with the resident part in memory.
pub struct LateDb {
    pub dim: usize,
    pub k: usize,
    pub count: usize,
    pub tokens: usize,
    /// `k * dim` centroid components. Resident, like the PQ codebook.
    pub centroids: Vec<f32>,
    /// `count + 1` token offsets. The directory that makes a variable-length
    /// document addressable by arithmetic -- see `late_store`.
    pub doc_offsets: Vec<u32>,
    /// `k + 1` document-id offsets into the postings arena.
    pub posting_offsets: Vec<u32>,
    pub postings_arena: Arena,
    pub codes_arena: Arena,
    pub tokens_arena: Arena,
}

impl LateDb {
    pub fn open(conn: &Connection) -> Result<Self> {
        let scalar = |key: &str| -> Result<usize> {
            let v: Vec<u8> =
                conn.query_row("SELECT value FROM late_meta WHERE key = ?1", [key], |r| r.get(0))?;
            Ok(String::from_utf8(v)?.parse()?)
        };
        let blob = |key: &str| -> Result<Vec<u8>> {
            Ok(conn.query_row("SELECT value FROM late_meta WHERE key = ?1", [key], |r| r.get(0))?)
        };
        let dim = scalar("dim")?;
        let k = scalar("k")?;
        let count = scalar("count")?;
        let tokens = scalar("tokens")?;

        let centroids: Vec<f32> = blob("centroids")?
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        anyhow::ensure!(centroids.len() == k * dim, "centroid table is the wrong shape");
        let doc_offsets = read_u32(&blob("doc_offsets")?);
        anyhow::ensure!(doc_offsets.len() == count + 1, "offsets directory is the wrong length");
        anyhow::ensure!(
            *doc_offsets.last().unwrap() as usize == tokens,
            "offsets directory stops at {} of {tokens} tokens",
            doc_offsets.last().unwrap()
        );
        let posting_offsets = read_u32(&blob("posting_offsets")?);
        anyhow::ensure!(posting_offsets.len() == k + 1, "postings directory is the wrong length");

        Ok(Self {
            dim,
            k,
            count,
            tokens,
            centroids,
            doc_offsets,
            posting_offsets,
            postings_arena: Arena::open(conn, "late_postings")?,
            codes_arena: Arena::open(conn, "late_codes")?,
            tokens_arena: Arena::open(conn, "late_tokens")?,
        })
    }

    /// Bytes a client holds before it can issue any byte range at all.
    pub fn resident_bytes(&self) -> usize {
        (self.centroids.len() + self.doc_offsets.len() + self.posting_offsets.len()) * 4
    }

    pub fn doc_tokens(&self, doc: u32) -> usize {
        (self.doc_offsets[doc as usize + 1] - self.doc_offsets[doc as usize]) as usize
    }

    /// Bytes of the file a configuration actually reads.
    ///
    /// Without reranking the token arena is dead weight and would not be shipped, so
    /// it is excluded rather than quietly inflating bytes-per-document by 44x. Both
    /// figures come from `dbstat`, not from arithmetic.
    pub fn index_bytes(&self, conn: &Connection, with_rerank: bool) -> Result<usize> {
        let mut total = table_bytes(conn, "late_meta")?
            + table_bytes(conn, "late_postings")?
            + table_bytes(conn, "late_codes")?;
        if with_rerank {
            total += table_bytes(conn, "late_tokens")?;
        }
        Ok(total)
    }

    fn query_centroid_table(&self, query: &MultiVector) -> Vec<f32> {
        let mut qc = vec![0f32; query.tokens() * self.k];
        for (i, q) in query.iter().enumerate() {
            for c in 0..self.k {
                qc[i * self.k + c] =
                    annlite_core::vectors::dot(q, &self.centroids[c * self.dim..(c + 1) * self.dim]);
            }
        }
        qc
    }
}

fn read_u32(b: &[u8]) -> Vec<u32> {
    b.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// What one stage cost a client that holds no copy of the file.
#[derive(Clone, Debug, Default)]
pub struct StageCost {
    /// Byte ranges requested: posting lists, documents' codes, documents' vectors.
    pub reads: usize,
    /// Distinct pages those ranges covered: the round-trip count for a client that
    /// fetches a page at a time and caches within a query.
    pub distinct_pages: usize,
    /// Maximal runs of consecutive pages: the round-trip count for a client that
    /// coalesces adjacent pages into one range request.
    pub contiguous_runs: usize,
    /// Payload bytes of the ranges requested. A page-granular client transfers
    /// `distinct_pages * 4096` instead; both are reported so neither has to be
    /// inferred from the other.
    pub bytes: usize,
}

#[derive(Clone, Debug, Default)]
pub struct LateSearchCost {
    pub postings: StageCost,
    pub centroid: StageCost,
    pub rerank: StageCost,
    /// Rounds that must be serial. Stages are dependent on each other and on nothing
    /// else, so this is 2 without reranking and 3 with it, for any corpus size.
    pub hops: usize,
    pub distinct_pages: usize,
    pub contiguous_runs: usize,
    pub bytes: usize,
    pub candidates: usize,
}

pub struct LateSearchResult {
    /// The full ranking the pipeline produced: the reranked pool first, then the
    /// remaining candidates in centroid order. Callers truncate; quality is measured
    /// against the whole thing so that a gold document at rank 57 is not silently
    /// turned into "absent" by a `LIMIT`.
    pub results: Vec<(u32, f32)>,
    pub cost: LateSearchCost,
}

/// One query through all three stages, reading everything through byte ranges.
///
/// `docs` is not a parameter: stage 3 reads the token vectors out of the database,
/// which is the entire reason the token arena is stored. Passing them in memory
/// would make the expensive stage free and the measurement meaningless.
pub fn late_search(
    conn: &Connection,
    db: &LateDb,
    query: &MultiVector,
    n_probe: usize,
    candidate_depth: usize,
    rerank: usize,
) -> Result<LateSearchResult> {
    let qc = db.query_centroid_table(query);
    let mut cost = LateSearchCost::default();

    let postings_blob = conn.blob_open(DatabaseName::Main, "late_postings", "data", 0, true)?;
    let codes_blob = conn.blob_open(DatabaseName::Main, "late_codes", "data", 0, true)?;

    // --- Stage 1: candidate generation ------------------------------------
    // Every range here is known once the resident centroid table has been scored,
    // so they issue together: one dependent round-trip.
    let mut pages1: HashSet<usize> = HashSet::new();
    let mut read_centroids: HashSet<u32> = HashSet::new();
    let mut seen = vec![false; db.count];
    let mut cands: Vec<u32> = Vec::new();
    let mut buf: Vec<u8> = Vec::new();
    for i in 0..query.tokens() {
        for c in probe_centroids(&qc[i * db.k..(i + 1) * db.k], n_probe) {
            let (a, b) = (db.posting_offsets[c as usize], db.posting_offsets[c as usize + 1]);
            let (off, len) = (a as usize * 4, (b - a) as usize * 4);
            // A centroid probed by two query tokens is fetched once; a client caches
            // within a query, and counting it twice would overstate the stage.
            if read_centroids.insert(c) {
                cost.postings.reads += 1;
                cost.postings.bytes += len;
                pages1.extend(db.postings_arena.pages_for(off, len));
                buf.resize(len, 0);
                if len > 0 {
                    postings_blob.read_at_exact(&mut buf, off)?;
                }
                for chunk in buf.chunks_exact(4) {
                    let d = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    if !seen[d as usize] {
                        seen[d as usize] = true;
                        cands.push(d);
                    }
                }
            } else {
                // Already resident from this query; re-walk it for the candidate
                // order without charging for the bytes.
                let mut local = vec![0u8; len];
                if len > 0 {
                    postings_blob.read_at_exact(&mut local, off)?;
                }
                for chunk in local.chunks_exact(4) {
                    let d = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    if !seen[d as usize] {
                        seen[d as usize] = true;
                        cands.push(d);
                    }
                }
            }
        }
    }
    cost.hops += 1;
    finish(&mut cost.postings, pages1);
    cost.candidates = cands.len();

    // --- Stage 2: centroid interaction ------------------------------------
    let mut pages2: HashSet<usize> = HashSet::new();
    let mut scored: Vec<(u32, f32)> = Vec::with_capacity(cands.len());
    for &d in &cands {
        let (a, b) = (db.doc_offsets[d as usize], db.doc_offsets[d as usize + 1]);
        let (off, len) = (a as usize * 4, (b - a) as usize * 4);
        cost.centroid.reads += 1;
        cost.centroid.bytes += len;
        pages2.extend(db.codes_arena.pages_for(off, len));
        buf.resize(len, 0);
        if len > 0 {
            codes_blob.read_at_exact(&mut buf, off)?;
        }
        let codes = read_u32(&buf);
        scored.push((d, centroid_maxsim(&qc, db.k, query.tokens(), &codes)));
    }
    if !cands.is_empty() {
        cost.hops += 1;
    }
    finish(&mut cost.centroid, pages2);
    scored.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
    scored.truncate(candidate_depth);

    // --- Stage 3: exact MaxSim over the surviving pool ---------------------
    let mut results = scored;
    if rerank > 0 && !results.is_empty() {
        let tokens_blob = conn.blob_open(DatabaseName::Main, "late_tokens", "data", 0, true)?;
        let depth = rerank.min(results.len());
        let mut pages3: HashSet<usize> = HashSet::new();
        let mut exact: Vec<(u32, f32)> = Vec::with_capacity(depth);
        for &(d, _) in results.iter().take(depth) {
            let (a, b) = (db.doc_offsets[d as usize], db.doc_offsets[d as usize + 1]);
            let stride = db.dim * 4;
            let (off, len) = (a as usize * stride, (b - a) as usize * stride);
            cost.rerank.reads += 1;
            cost.rerank.bytes += len;
            pages3.extend(db.tokens_arena.pages_for(off, len));
            buf.resize(len, 0);
            if len > 0 {
                tokens_blob.read_at_exact(&mut buf, off)?;
            }
            let data: Vec<f32> = buf
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let doc = MultiVector { data, dim: db.dim };
            exact.push((d, maxsim(query, &doc)));
        }
        cost.hops += 1;
        finish(&mut cost.rerank, pages3);
        exact.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        // The pool is reordered exactly; everything below it keeps its centroid
        // ranking, which is what a real system returns and what success@100 has to
        // be measured against.
        let tail: Vec<(u32, f32)> = results[depth..].to_vec();
        results = exact;
        results.extend(tail);
    }

    // Arenas live in disjoint page ranges, so the totals are sums; they are taken as
    // a union anyway rather than relying on that.
    cost.distinct_pages =
        cost.postings.distinct_pages + cost.centroid.distinct_pages + cost.rerank.distinct_pages;
    cost.contiguous_runs =
        cost.postings.contiguous_runs + cost.centroid.contiguous_runs + cost.rerank.contiguous_runs;
    cost.bytes = cost.postings.bytes + cost.centroid.bytes + cost.rerank.bytes;

    Ok(LateSearchResult { results, cost })
}

/// Turn a stage's page set into counted pages and coalesced runs.
fn finish(stage: &mut StageCost, pages: HashSet<usize>) {
    let mut v: Vec<usize> = pages.into_iter().collect();
    v.sort_unstable();
    stage.distinct_pages = v.len();
    stage.contiguous_runs =
        v.windows(2).filter(|w| w[1] != w[0] + 1).count() + usize::from(!v.is_empty());
}

/// Bytes a page-granular client transfers for a given page count.
pub fn page_bytes(pages: usize) -> usize {
    pages * PAGE_BYTES
}
