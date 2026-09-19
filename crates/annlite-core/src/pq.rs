//! Product quantization.
//!
//! A 384-dimensional float32 embedding costs 1,536 bytes. At a million documents
//! that is 1.5 GB, which is not a thing a browser can download to answer one query.
//! PQ splits each vector into `m` contiguous subvectors, clusters each subspace
//! independently into 256 centroids, and stores one byte per subspace. With
//! `m = 64` over 384 dimensions — six dimensions per subquantizer — a document costs
//! **64 bytes**, a 24x reduction, and a million of them fit in 64 MB.
//!
//! Scoring uses *asymmetric* distance computation: the query stays in full precision
//! and only the documents are quantized. The alternative, quantizing the query too,
//! would throw away precision on the one vector we have exactly, for no saving. ADC
//! also makes scoring cheap. Because inner product decomposes over a partition of
//! the dimensions,
//!
//! ```text
//! <q, x> = sum over subspaces m of <q_m, x_m> ~= sum over m of <q_m, centroid[m][code[m]]>
//! ```
//!
//! the `m x 256` table of `<q_m, centroid>` values can be precomputed once per query,
//! after which each document costs `m` table lookups and `m` additions — no
//! multiplications and no access to the original vectors at all.
//!
//! Clustering minimises squared Euclidean distance rather than maximising inner
//! product. That is the right objective even though scoring is by inner product:
//! the ADC error is bounded by how far a subvector sits from its centroid, and
//! k-means minimises exactly that quantity.

use crate::vectors::{sqeuclidean, Vectors};
use anyhow::Result;
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Centroids per subquantizer. 256 is what makes a code fit in exactly one byte.
pub const CENTROIDS: usize = 256;

#[derive(Clone)]
pub struct ProductQuantizer {
    /// Full vector dimension.
    pub dim: usize,
    /// Number of subquantizers; also the bytes per encoded document.
    pub m: usize,
    /// Dimensions per subquantizer.
    pub dsub: usize,
    /// `m * CENTROIDS * dsub` floats, indexed `[subspace][centroid][component]`.
    pub centroids: Vec<f32>,
}

impl ProductQuantizer {
    /// Train on `training` vectors with `m` subquantizers.
    ///
    /// `seed` fixes k-means++ initialisation, so a rebuild from the same training set
    /// yields byte-identical codes. Without that the index would differ between runs
    /// and recall numbers would not be comparable.
    pub fn train(training: &Vectors, m: usize, iters: usize, seed: u64) -> Result<Self> {
        let dim = training.dim;
        anyhow::ensure!(m > 0 && dim % m == 0, "dimension {dim} is not divisible by m={m}");
        anyhow::ensure!(!training.is_empty(), "cannot train on an empty set");
        let dsub = dim / m;
        let n = training.len();
        if n < CENTROIDS {
            // Fewer training points than centroids means some centroids would be
            // arbitrary. Allowed, but the caller should know the codebook is degenerate.
            eprintln!(
                "warning: training PQ on {n} vectors for {CENTROIDS} centroids per subspace; \
                 codebook will be under-determined"
            );
        }

        let mut centroids = vec![0f32; m * CENTROIDS * dsub];
        let mut sub = vec![0f32; n * dsub];
        for s in 0..m {
            // Gather subspace s: contiguous columns [s*dsub, (s+1)*dsub) of every row.
            for i in 0..n {
                let src = &training.row(i)[s * dsub..(s + 1) * dsub];
                sub[i * dsub..(i + 1) * dsub].copy_from_slice(src);
            }
            let book = kmeans(&sub, n, dsub, CENTROIDS, iters, seed ^ (s as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            centroids[s * CENTROIDS * dsub..(s + 1) * CENTROIDS * dsub].copy_from_slice(&book);
        }
        Ok(Self { dim, m, dsub, centroids })
    }

    #[inline]
    fn book(&self, subspace: usize) -> &[f32] {
        &self.centroids[subspace * CENTROIDS * self.dsub..(subspace + 1) * CENTROIDS * self.dsub]
    }

    /// Encode one vector to `m` bytes.
    pub fn encode(&self, v: &[f32]) -> Vec<u8> {
        let mut code = vec![0u8; self.m];
        self.encode_into(v, &mut code);
        code
    }

    pub fn encode_into(&self, v: &[f32], out: &mut [u8]) {
        debug_assert_eq!(v.len(), self.dim);
        debug_assert_eq!(out.len(), self.m);
        for s in 0..self.m {
            let piece = &v[s * self.dsub..(s + 1) * self.dsub];
            let book = self.book(s);
            let mut best = 0usize;
            let mut best_d = f32::INFINITY;
            for c in 0..CENTROIDS {
                let d = sqeuclidean(piece, &book[c * self.dsub..(c + 1) * self.dsub]);
                if d < best_d {
                    best_d = d;
                    best = c;
                }
            }
            out[s] = best as u8;
        }
    }

    /// Encode a whole matrix into a packed `n * m` byte array.
    pub fn encode_all(&self, vectors: &Vectors) -> Vec<u8> {
        let mut out = vec![0u8; vectors.len() * self.m];
        for (i, row) in vectors.rows().enumerate() {
            self.encode_into(row, &mut out[i * self.m..(i + 1) * self.m]);
        }
        out
    }

    /// Approximate the original vector from its code. Only for diagnostics —
    /// scoring never needs this, which is the point of ADC.
    pub fn decode(&self, code: &[u8]) -> Vec<f32> {
        let mut out = vec![0f32; self.dim];
        for s in 0..self.m {
            let c = code[s] as usize;
            out[s * self.dsub..(s + 1) * self.dsub]
                .copy_from_slice(&self.book(s)[c * self.dsub..(c + 1) * self.dsub]);
        }
        out
    }

    /// Build the per-query lookup table of `<q_subspace, centroid>` inner products.
    /// Computed once per query; afterwards a document scores in `m` lookups.
    pub fn score_table(&self, query: &[f32]) -> ScoreTable {
        debug_assert_eq!(query.len(), self.dim);
        let mut table = vec![0f32; self.m * CENTROIDS];
        for s in 0..self.m {
            let q = &query[s * self.dsub..(s + 1) * self.dsub];
            let book = self.book(s);
            for c in 0..CENTROIDS {
                let cent = &book[c * self.dsub..(c + 1) * self.dsub];
                table[s * CENTROIDS + c] = q.iter().zip(cent).map(|(a, b)| a * b).sum();
            }
        }
        ScoreTable { table, m: self.m }
    }
}

/// Precomputed query-to-centroid inner products for one query.
pub struct ScoreTable {
    table: Vec<f32>,
    m: usize,
}

impl ScoreTable {
    /// Approximate `<query, document>` from the document's code alone.
    #[inline]
    pub fn score(&self, code: &[u8]) -> f32 {
        debug_assert_eq!(code.len(), self.m);
        let mut acc = 0f32;
        for (s, &c) in code.iter().enumerate() {
            acc += self.table[s * CENTROIDS + c as usize];
        }
        acc
    }
}

/// Lloyd's algorithm with k-means++ seeding over `n` points of `d` dimensions.
fn kmeans(data: &[f32], n: usize, d: usize, k: usize, iters: usize, seed: u64) -> Vec<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut centroids = vec![0f32; k * d];

    // k-means++: seed the first centroid uniformly, then each subsequent one with
    // probability proportional to its squared distance from the nearest chosen
    // centroid. Plain random seeding leaves duplicate and dead centroids, which in
    // PQ show up directly as wasted code points.
    let first = (rng.next_u64() % n as u64) as usize;
    centroids[..d].copy_from_slice(&data[first * d..(first + 1) * d]);
    let mut closest: Vec<f32> = (0..n)
        .map(|i| sqeuclidean(&data[i * d..(i + 1) * d], &centroids[..d]))
        .collect();

    for c in 1..k {
        let total: f64 = closest.iter().map(|&x| x as f64).sum();
        let pick = if total <= 0.0 {
            // All remaining points coincide with a chosen centroid; any index works.
            (rng.next_u64() % n as u64) as usize
        } else {
            let target = (rng.next_u64() as f64 / u64::MAX as f64) * total;
            let mut acc = 0f64;
            let mut chosen = n - 1;
            for (i, &w) in closest.iter().enumerate() {
                acc += w as f64;
                if acc >= target {
                    chosen = i;
                    break;
                }
            }
            chosen
        };
        centroids[c * d..(c + 1) * d].copy_from_slice(&data[pick * d..(pick + 1) * d]);
        for i in 0..n {
            let dist = sqeuclidean(&data[i * d..(i + 1) * d], &centroids[c * d..(c + 1) * d]);
            if dist < closest[i] {
                closest[i] = dist;
            }
        }
    }

    let mut assign = vec![0u32; n];
    let mut sums = vec![0f64; k * d];
    let mut counts = vec![0u32; k];
    for _ in 0..iters {
        let mut moved = false;
        for i in 0..n {
            let p = &data[i * d..(i + 1) * d];
            let mut best = 0usize;
            let mut best_d = f32::INFINITY;
            for c in 0..k {
                let dist = sqeuclidean(p, &centroids[c * d..(c + 1) * d]);
                if dist < best_d {
                    best_d = dist;
                    best = c;
                }
            }
            if assign[i] != best as u32 {
                assign[i] = best as u32;
                moved = true;
            }
        }
        sums.iter_mut().for_each(|x| *x = 0.0);
        counts.iter_mut().for_each(|x| *x = 0);
        for i in 0..n {
            let c = assign[i] as usize;
            counts[c] += 1;
            for j in 0..d {
                sums[c * d + j] += data[i * d + j] as f64;
            }
        }
        for c in 0..k {
            if counts[c] == 0 {
                // An empty cluster is a wasted code point. Re-seed it onto the point
                // furthest from its own centroid, which is where the quantization
                // error is currently worst.
                let mut worst = 0usize;
                let mut worst_d = -1f32;
                for i in 0..n {
                    let a = assign[i] as usize;
                    let dist = sqeuclidean(&data[i * d..(i + 1) * d], &centroids[a * d..(a + 1) * d]);
                    if dist > worst_d {
                        worst_d = dist;
                        worst = i;
                    }
                }
                centroids[c * d..(c + 1) * d].copy_from_slice(&data[worst * d..(worst + 1) * d]);
                assign[worst] = c as u32;
                moved = true;
                continue;
            }
            for j in 0..d {
                centroids[c * d + j] = (sums[c * d + j] / counts[c] as f64) as f32;
            }
        }
        if !moved {
            break; // Converged: no point changed cluster.
        }
    }
    centroids
}
