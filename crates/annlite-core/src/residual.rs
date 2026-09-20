//! Residual quantization for late-interaction token vectors.
//!
//! Section 15.2 measured the gap this closes. Clustering token vectors into
//! centroids compresses the *candidate-generation* data by 43.7x, but a centroid id
//! alone is too coarse to rank with: scoring against centroids reaches success@1 of
//! 0.240 where exact MaxSim reaches 0.454. Recovering that meant reranking against
//! the original float32 vectors, which put the stored index back at 29,196 bytes per
//! document — cheap and weak, or accurate and enormous, with nothing between.
//!
//! ColBERTv2's answer, implemented here: keep the centroid id *and* a few bits per
//! dimension of the residual `v - centroid[c]`. Reconstruction is
//! `centroid + dequantized residual`, so reranking happens in the compressed domain
//! and never reads a full vector.
//!
//! Bucket cutoffs are quantiles of the residual distribution **pooled across all
//! dimensions**, not fitted per dimension. That is what ColBERTv2 does and it is not
//! merely a simplification: residuals here are differences from a nearby centroid, so
//! their spread is dominated by how far tokens sit from their cluster centre rather
//! than by which axis is being measured. A per-dimension codebook would spend `dim`
//! times the table for a distribution that is nearly the same in every direction.
//!
//! At 48 dimensions a token costs `dim * bits / 8` bytes: 6 at one bit, 12 at two,
//! 24 at four, against 192 uncompressed.

use anyhow::Result;

/// Quantizer for residual components, shared by every dimension and every token.
#[derive(Clone, Debug)]
pub struct ResidualCodec {
    pub bits: u8,
    /// Reconstruction value per bucket; `2^bits` of them.
    pub centers: Vec<f32>,
    /// Upper edges between buckets; `centers.len() - 1` of them, ascending.
    cutoffs: Vec<f32>,
}

impl ResidualCodec {
    /// Fit from a sample of residual components.
    ///
    /// Buckets are equal-mass rather than equal-width: residual distributions are
    /// sharply peaked at zero, so equal-width buckets would spend most of their code
    /// space on tails that hold almost no mass.
    pub fn train(sample: &[f32], bits: u8) -> Result<Self> {
        anyhow::ensure!((1..=8).contains(&bits), "bits must be in 1..=8, got {bits}");
        anyhow::ensure!(!sample.is_empty(), "cannot fit a codec on an empty sample");
        let n_buckets = 1usize << bits;

        let mut sorted: Vec<f32> = sample.to_vec();
        sorted.sort_by(f32::total_cmp);

        // Cutoff i is the quantile at (i+1)/n_buckets; the center of a bucket is the
        // median of the mass that falls inside it, which is what minimises
        // reconstruction error for that bucket under absolute deviation.
        let at = |q: f64| -> f32 {
            let idx = ((sorted.len() - 1) as f64 * q).round() as usize;
            sorted[idx.min(sorted.len() - 1)]
        };
        let cutoffs: Vec<f32> = (1..n_buckets)
            .map(|i| at(i as f64 / n_buckets as f64))
            .collect();
        let centers: Vec<f32> = (0..n_buckets)
            .map(|i| at((i as f64 + 0.5) / n_buckets as f64))
            .collect();
        Ok(Self { bits, centers, cutoffs })
    }

    pub fn buckets(&self) -> usize {
        self.centers.len()
    }

    /// Bytes one token's residual occupies at `dim` dimensions.
    pub fn bytes_per_token(&self, dim: usize) -> usize {
        (dim * self.bits as usize).div_ceil(8)
    }

    #[inline]
    fn bucket_of(&self, x: f32) -> u8 {
        // Ascending cutoffs, so the first one the value does not exceed names it.
        match self.cutoffs.iter().position(|&c| x <= c) {
            Some(i) => i as u8,
            None => (self.centers.len() - 1) as u8,
        }
    }

    /// Pack `residual` into `out`, which must be [`Self::bytes_per_token`] long.
    pub fn encode_into(&self, residual: &[f32], out: &mut [u8]) {
        debug_assert_eq!(out.len(), self.bytes_per_token(residual.len()));
        out.fill(0);
        let bits = self.bits as usize;
        for (i, &x) in residual.iter().enumerate() {
            let code = self.bucket_of(x) as usize;
            let bit = i * bits;
            // A code may straddle a byte boundary at 3, 5, 6 or 7 bits.
            for b in 0..bits {
                if code >> b & 1 == 1 {
                    let p = bit + b;
                    out[p / 8] |= 1 << (p % 8);
                }
            }
        }
    }

    /// Reconstruct into `out`, which must be `dim` long.
    pub fn decode_into(&self, packed: &[u8], out: &mut [f32]) {
        let bits = self.bits as usize;
        for (i, slot) in out.iter_mut().enumerate() {
            let bit = i * bits;
            let mut code = 0usize;
            for b in 0..bits {
                let p = bit + b;
                if packed[p / 8] >> (p % 8) & 1 == 1 {
                    code |= 1 << b;
                }
            }
            *slot = self.centers[code];
        }
    }

    /// Bytes the codec itself occupies when stored.
    pub fn table_bytes(&self) -> usize {
        (self.centers.len() + self.cutoffs.len()) * 4 + 1
    }
}

/// Residuals for a whole corpus, packed token by token.
pub struct ResidualStore {
    pub codec: ResidualCodec,
    pub dim: usize,
    data: Vec<u8>,
}

impl ResidualStore {
    /// Quantize every token's difference from its assigned centroid.
    pub fn build(
        tokens: &[f32],
        codes: &[u32],
        centroids: &[f32],
        dim: usize,
        bits: u8,
        train_sample: usize,
    ) -> Result<Self> {
        let n = codes.len();
        anyhow::ensure!(tokens.len() == n * dim, "token buffer does not match code count");

        // Fit on a strided sample: the distribution is the same everywhere and
        // sorting every residual component of a large corpus is needless work.
        let stride = if train_sample > 0 && train_sample < n { (n / train_sample).max(1) } else { 1 };
        let mut sample = Vec::with_capacity(n / stride * dim);
        for i in (0..n).step_by(stride) {
            let c = codes[i] as usize;
            for d in 0..dim {
                sample.push(tokens[i * dim + d] - centroids[c * dim + d]);
            }
        }
        let codec = ResidualCodec::train(&sample, bits)?;

        let per = codec.bytes_per_token(dim);
        let mut data = vec![0u8; n * per];
        let mut residual = vec![0f32; dim];
        for i in 0..n {
            let c = codes[i] as usize;
            for d in 0..dim {
                residual[d] = tokens[i * dim + d] - centroids[c * dim + d];
            }
            codec.encode_into(&residual, &mut data[i * per..(i + 1) * per]);
        }
        Ok(Self { codec, dim, data })
    }

    pub fn bytes(&self) -> usize {
        self.data.len()
    }

    /// Reconstruct token `i` as `centroid + residual`.
    pub fn reconstruct_into(&self, i: usize, code: u32, centroids: &[f32], out: &mut [f32]) {
        let per = self.codec.bytes_per_token(self.dim);
        self.codec.decode_into(&self.data[i * per..(i + 1) * per], out);
        let base = code as usize * self.dim;
        for d in 0..self.dim {
            out[d] += centroids[base + d];
        }
    }
}
