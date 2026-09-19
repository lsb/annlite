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
        Self::build_sampled(docs, k, iters, seed, 0)
    }

    /// As [`LateIndex::build`], but fitting centroids on at most `train_sample`
    /// tokens rather than all of them.
    ///
    /// Lloyd's algorithm costs `tokens * k * dim` per iteration, which at half a
    /// million tokens and a few thousand centroids is tens of billions of operations
    /// per pass. Centroid positions converge from a sample long before the cost of
    /// using every token is justified, and every token is still assigned afterwards,
    /// so the index itself is complete. The sample is taken by stride rather than at
    /// random, which keeps it deterministic and spreads it across the corpus instead
    /// of favouring the first few documents.
    pub fn build_sampled(
        docs: &[MultiVector],
        k: usize,
        iters: usize,
        seed: u64,
        train_sample: usize,
    ) -> Result<Self> {
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
        let centroids = if train_sample > 0 && train_sample < total {
            let stride = (total / train_sample).max(1);
            let mut sample = Vec::with_capacity(train_sample * dim);
            for i in (0..total).step_by(stride).take(train_sample) {
                sample.extend_from_slice(pool.row(i));
            }
            crate::pq::kmeans_centroids(&Vectors { data: sample, dim }, k, iters, seed)
        } else {
            crate::pq::kmeans_centroids(&pool, k, iters, seed)
        };
        let assign = assign_to(&pool, &centroids, k, dim);

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

    /// Documents containing centroid `c`, ascending. Stage 1's inverted list.
    ///
    /// Exposed because the SQLite form stores exactly these bytes and has to be
    /// checkable against them; a posting list written under a different ordering
    /// still returns plausible-looking candidates.
    pub fn postings(&self, c: u32) -> &[u32] {
        &self.postings[c as usize]
    }

    /// `offsets[i]..offsets[i + 1]` in token units. The directory a storage form
    /// needs to find a variable-length document without a second lookup.
    pub fn doc_offsets(&self) -> &[u32] {
        &self.offsets
    }

    pub fn total_tokens(&self) -> usize {
        self.codes.len()
    }

    /// Score every query token against every centroid: `tokens * k` entries.
    ///
    /// Both remaining stages read from this table and nothing else that is
    /// query-dependent, which is what makes stages 1 and 2 free of per-candidate
    /// lookups -- section 9.4's lesson, applied to late interaction.
    pub fn query_centroid_table(&self, query: &MultiVector) -> Vec<f32> {
        let mut qc = vec![0f32; query.tokens() * self.k];
        for (i, q) in query.iter().enumerate() {
            for c in 0..self.k {
                qc[i * self.k + c] = dot(q, self.centroid(c as u32));
            }
        }
        qc
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
        let qc = self.query_centroid_table(query);

        let mut seen = vec![false; self.len()];
        let mut cands: Vec<u32> = Vec::new();
        for i in 0..query.tokens() {
            for c in probe_centroids(&qc[i * self.k..(i + 1) * self.k], n_probe) {
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
            .map(|d| (d, centroid_maxsim(&qc, self.k, query.tokens(), self.doc_codes(d))))
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

/// The `n_probe` centroids a single query token probes, best score first among the
/// selected set but otherwise unordered.
///
/// Factored out rather than inlined because the SQLite storage form must select the
/// *same* centroids as the in-memory index; two independently written selections
/// that disagree only on ties would produce two candidate pools that are almost the
/// same, which is the kind of difference a quality number hides rather than reveals.
pub fn probe_centroids(row: &[f32], n_probe: usize) -> Vec<u32> {
    let k = row.len();
    let probe = n_probe.clamp(1, k);
    let mut order: Vec<u32> = (0..k as u32).collect();
    order.select_nth_unstable_by(probe - 1, |&a, &b| row[b as usize].total_cmp(&row[a as usize]));
    order.truncate(probe);
    order
}

/// Stage 2's score: MaxSim with every document token replaced by its centroid.
///
/// `qc` is the `tokens * k` table from [`LateIndex::query_centroid_table`]. The
/// document contributes only its centroid ids, which is the entire point -- four
/// bytes per token instead of `dim * 4`.
pub fn centroid_maxsim(qc: &[f32], k: usize, tokens: usize, codes: &[u32]) -> f32 {
    (0..tokens)
        .map(|i| {
            let row = &qc[i * k..(i + 1) * k];
            codes.iter().map(|&c| row[c as usize]).fold(f32::NEG_INFINITY, f32::max)
        })
        .filter(|s| s.is_finite())
        .sum()
}

/// Nearest centroid for every vector in `pool`.
fn assign_to(pool: &Vectors, centroids: &[f32], k: usize, dim: usize) -> Vec<u32> {
    (0..pool.len())
        .map(|i| {
            let p = pool.row(i);
            let mut best = 0u32;
            let mut best_d = f32::INFINITY;
            for c in 0..k {
                let d = crate::vectors::sqeuclidean(p, &centroids[c * dim..(c + 1) * dim]);
                if d < best_d {
                    best_d = d;
                    best = c as u32;
                }
            }
            best
        })
        .collect()
}
