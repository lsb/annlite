//! Reading the embedding matrices produced by `tools/embed`.
//!
//! The on-disk format is deliberately headerless: little-endian float32, row-major,
//! with metadata in a separate JSON sidecar. Row `i` therefore begins at byte
//! `i * dim * 4` and nothing has to be parsed to find it. That arithmetic is what
//! lets the browser client turn "I need vector 91,332" into a single HTTP range
//! request, and it is why the format is not `.npy`.

use anyhow::{Context, Result};
use std::path::Path;

/// A row-major matrix of embeddings held in memory.
pub struct Vectors {
    pub data: Vec<f32>,
    pub dim: usize,
}

impl Vectors {
    pub fn load(path: &Path, dim: usize) -> Result<Self> {
        let bytes = std::fs::read(path)
            .with_context(|| format!("reading embeddings {}", path.display()))?;
        Self::from_bytes(&bytes, dim)
    }

    pub fn from_bytes(bytes: &[u8], dim: usize) -> Result<Self> {
        anyhow::ensure!(dim > 0, "dimension must be positive");
        anyhow::ensure!(
            bytes.len() % (dim * 4) == 0,
            "embedding file of {} bytes is not a whole number of {dim}-d float32 rows",
            bytes.len()
        );
        let mut data = vec![0f32; bytes.len() / 4];
        for (slot, chunk) in data.iter_mut().zip(bytes.chunks_exact(4)) {
            *slot = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        Ok(Self { data, dim })
    }

    pub fn len(&self) -> usize {
        self.data.len() / self.dim
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn row(&self, i: usize) -> &[f32] {
        &self.data[i * self.dim..(i + 1) * self.dim]
    }

    pub fn rows(&self) -> impl Iterator<Item = &[f32]> {
        self.data.chunks_exact(self.dim)
    }
}

/// Inner product. Embeddings leave the encoder L2-normalised, so this is cosine
/// similarity and larger is better — the opposite orientation from a distance.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Squared Euclidean distance. Used for k-means, which minimises exactly this.
#[inline]
pub fn sqeuclidean(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Exact brute-force top-`k` by inner product. This is the ground truth every
/// approximate index is scored against, so it is deliberately the dumbest possible
/// implementation: nothing here should be clever enough to be wrong.
pub fn exact_top_k(vectors: &Vectors, query: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut scored: Vec<(u32, f32)> = vectors
        .rows()
        .enumerate()
        .map(|(i, r)| (i as u32, dot(query, r)))
        .collect();
    let k = k.min(scored.len());
    if k == 0 {
        return Vec::new();
    }
    let nth = k - 1;
    scored.select_nth_unstable_by(nth, |a, b| b.1.total_cmp(&a.1));
    scored.truncate(k);
    scored.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
    scored
}
