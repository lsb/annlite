//! Late-interaction retrieval, PLAID-style.
//!
//! A late-interaction model represents a document as one vector *per token* rather
//! than one per document, and scores with MaxSim: every query token takes its best
//! match among the document's tokens, and those maxima are summed.
//!
//! ```text
//! score(q, d) = sum over query tokens i of  max over doc tokens j of  <q_i, d_j>
//! ```
//!
//! This is strictly more expressive than a single dot product — a document can match
//! one part of a query strongly without diluting that evidence into an average — and
//! it is why late interaction beats dense retrieval on term-heavy queries.
//!
//! It is also ruinously expensive stored naively. A 50-word document tokenizes to
//! roughly 120 tokens; at 48 dimensions of float32 that is 23 KB per document, so a
//! million documents would be 23 GB. Nothing about that is servable to a browser.
//!
//! PLAID's answer, followed here, is that token vectors are extremely redundant
//! across a corpus. Cluster all of them into `k` centroids; a token is then a
//! centroid id plus a small residual. Retrieval proceeds in stages that get
//! progressively more expensive and progressively more selective:
//!
//! 1. **Candidate generation.** For each query token, find its nearest centroids and
//!    collect the documents containing them, through an inverted list keyed by
//!    centroid. No document data is read.
//! 2. **Centroid interaction.** Score candidates by MaxSim over *centroids alone*,
//!    substituting each token's centroid for the token. Still no residuals.
//! 3. **Full MaxSim.** Decompress residuals for the surviving few and score exactly.
//!
//! The staging matters here for the same reason it matters in `pq`: stages 1 and 2
//! need only data that is either resident (the centroid table) or sequential (the
//! inverted lists), and only stage 3 touches scattered per-document bytes.

use crate::vectors::{dot, Vectors};
use anyhow::Result;

/// A document as a sequence of L2-normalised token vectors.
pub struct MultiVector {
    pub data: Vec<f32>,
    pub dim: usize,
}

impl MultiVector {
    pub fn tokens(&self) -> usize {
        self.data.len() / self.dim
    }

    pub fn token(&self, i: usize) -> &[f32] {
        &self.data[i * self.dim..(i + 1) * self.dim]
    }

    pub fn iter(&self) -> impl Iterator<Item = &[f32]> {
        self.data.chunks_exact(self.dim)
    }
}

/// Exact MaxSim between a query and a document.
///
/// The reference implementation: every approximation in this module is scored
/// against this, so it is written for obviousness rather than speed.
pub fn maxsim(query: &MultiVector, doc: &MultiVector) -> f32 {
    query
        .iter()
        .map(|q| doc.iter().map(|d| dot(q, d)).fold(f32::NEG_INFINITY, f32::max))
        .filter(|s| s.is_finite())
        .sum()
}

/// A corpus of documents compressed to centroid ids.
pub struct LateIndex {
    pub dim: usize,
    /// `k * dim` centroid components.
    pub centroids: Vec<f32>,
    pub k: usize,
    /// Centroid id per token, concatenated over documents.
    codes: Vec<u32>,
    /// `offsets[i]..offsets[i+1]` is document `i`'s span of `codes`.
    offsets: Vec<u32>,
    /// Documents containing each centroid, sorted. The inverted list of stage 1.
    postings: Vec<Vec<u32>>,
}

impl LateIndex {
    /// Build from documents' token vectors.
    ///
    /// `k` follows PLAID's guidance of roughly the square root of the total token
    /// count: enough centroids that a cluster is small, few enough that the centroid
    /// table stays resident.
    pub fn build(docs: &[MultiVector], k: usize, iters: usize, seed: u64) -> Result<Self> {
        anyhow::ensure!(!docs.is_empty(), "cannot build over an empty corpus");
        let dim = docs[0].dim;
        anyhow::ensure!(docs.iter().all(|d| d.dim == dim), "documents disagree on dimension");

        let total: usize = docs.iter().map(|d| d.tokens()).sum();
        anyhow::ensure!(total > 0, "corpus has no tokens");
        let k = k.min(total).max(1);

        let mut flat = Vec::with_capacity(total * dim);
        for d in docs {
            flat.extend_from_slice(&d.data);
        }
        let pool = Vectors { data: flat, dim };
        let assign = crate::pq::kmeans_assign(&pool, k, iters, seed);
        let centroids = crate::pq::kmeans_centroids(&pool, k, iters, seed);

        let mut codes = Vec::with_capacity(total);
        let mut offsets = Vec::with_capacity(docs.len() + 1);
        let mut postings: Vec<Vec<u32>> = vec![Vec::new(); k];
        let mut cursor = 0usize;
        offsets.push(0);
        for (doc_id, d) in docs.iter().enumerate() {
            for _ in 0..d.tokens() {
                let c = assign[cursor];
                codes.push(c);
                postings[c as usize].push(doc_id as u32);
                cursor += 1;
            }
            offsets.push(codes.len() as u32);
        }
        for p in postings.iter_mut() {
            p.dedup();
        }
        Ok(Self { dim, centroids, k, codes, offsets, postings })
    }

    pub fn len(&self) -> usize {
        self.offsets.len() - 1
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn doc_codes(&self, doc: u32) -> &[u32] {
        let (a, b) = (self.offsets[doc as usize] as usize, self.offsets[doc as usize + 1] as usize);
        &self.codes[a..b]
    }

    fn centroid(&self, c: u32) -> &[f32] {
        &self.centroids[c as usize * self.dim..(c as usize + 1) * self.dim]
    }

    /// Bytes a compressed corpus occupies, versus storing raw token vectors.
    pub fn footprint(&self) -> (usize, usize) {
        let compressed = self.codes.len() * 4 + self.centroids.len() * 4;
        let raw = self.codes.len() * self.dim * 4;
        (compressed, raw)
    }

    /// Stage 1 and 2: candidates from the inverted lists, ranked by centroid MaxSim.
    ///
    /// `n_probe` is how many centroids each query token contributes. Raising it
    /// widens the candidate net at the cost of reading more postings.
    pub fn candidates(&self, query: &MultiVector, n_probe: usize, top: usize) -> Vec<(u32, f32)> {
        // Query-to-centroid scores, reused by both stages.
        let mut qc = vec![0f32; query.tokens() * self.k];
        for (i, q) in query.iter().enumerate() {
            for c in 0..self.k {
                qc[i * self.k + c] = dot(q, self.centroid(c as u32));
            }
        }

        let mut seen = vec![false; self.len()];
        let mut cands: Vec<u32> = Vec::new();
        for i in 0..query.tokens() {
            let row = &qc[i * self.k..(i + 1) * self.k];
            let mut order: Vec<u32> = (0..self.k as u32).collect();
            let probe = n_probe.min(self.k);
            order.select_nth_unstable_by(probe - 1, |&a, &b| {
                row[b as usize].total_cmp(&row[a as usize])
            });
            for &c in &order[..probe] {
                for &d in &self.postings[c as usize] {
                    if !seen[d as usize] {
                        seen[d as usize] = true;
                        cands.push(d);
                    }
                }
            }
        }

        // Stage 2: MaxSim with each token replaced by its centroid. Needs only the
        // document's centroid ids and the resident query-centroid table.
        let mut scored: Vec<(u32, f32)> = cands
            .into_iter()
            .map(|d| {
                let codes = self.doc_codes(d);
                let s: f32 = (0..query.tokens())
                    .map(|i| {
                        let row = &qc[i * self.k..(i + 1) * self.k];
                        codes
                            .iter()
                            .map(|&c| row[c as usize])
                            .fold(f32::NEG_INFINITY, f32::max)
                    })
                    .filter(|s| s.is_finite())
                    .sum();
                (d, s)
            })
            .collect();
        scored.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        scored.truncate(top);
        scored
    }

    /// Full pipeline: generate candidates, then rescore the top `rerank` exactly.
    pub fn search(
        &self,
        docs: &[MultiVector],
        query: &MultiVector,
        k: usize,
        n_probe: usize,
        candidate_depth: usize,
        rerank: usize,
    ) -> Vec<(u32, f32)> {
        let mut cands = self.candidates(query, n_probe, candidate_depth);
        if rerank > 0 {
            let depth = rerank.min(cands.len());
            let mut exact: Vec<(u32, f32)> = cands[..depth]
                .iter()
                .map(|&(d, _)| (d, maxsim(query, &docs[d as usize])))
                .collect();
            exact.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
            cands = exact;
        }
        cands.truncate(k);
        cands
    }
}
