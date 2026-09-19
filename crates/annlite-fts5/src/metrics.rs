//! Latency and retrieval-quality summaries.
//!
//! Kept free of SQLite so the definitions can be unit-tested against hand-worked
//! examples, and so the ANN crates can reuse exactly the same arithmetic when their
//! numbers are set beside this baseline. A quality comparison is only meaningful if
//! both sides compute success@k and MRR the same way.

use serde::Serialize;

/// Nearest-rank percentile: the smallest value at or above which `q` of the sample
/// lies. No interpolation, because interpolating between two latencies invents a
/// measurement that never happened.
pub fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let rank = (q * sorted.len() as f64).ceil().max(1.0) as usize;
    sorted[rank.min(sorted.len()) - 1]
}

#[derive(Serialize, Clone, Debug, Default)]
pub struct Dist {
    pub n: usize,
    pub mean: f64,
    pub median: f64,
    pub p95: f64,
    pub p99: f64,
    pub max: f64,
    pub min: f64,
}

impl Dist {
    pub fn of(values: &[f64]) -> Dist {
        if values.is_empty() {
            return Dist::default();
        }
        let mut v = values.to_vec();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Dist {
            n: v.len(),
            mean: v.iter().sum::<f64>() / v.len() as f64,
            median: percentile(&v, 0.50),
            p95: percentile(&v, 0.95),
            p99: percentile(&v, 0.99),
            max: *v.last().unwrap(),
            min: v[0],
        }
    }
}

/// Cutoffs at which known-item quality is reported.
pub const CUTOFFS: [usize; 3] = [1, 10, 100];

/// Quality over a set of known-item queries.
///
/// `ranks` holds the 1-based rank of the gold document for each query, or `None` when
/// it was not retrieved at all within the depth the search was run to. A query whose
/// gold document was missed still counts in the denominator — dropping it would
/// flatter the system by measuring only the queries it answered.
#[derive(Serialize, Clone, Debug)]
pub struct KnownItemQuality {
    pub n_queries: usize,
    /// success@k for each cutoff in [`CUTOFFS`].
    pub success_at: Vec<(usize, f64)>,
    /// MRR@k for each cutoff: mean of 1/rank, counting 0 when rank > k.
    pub mrr_at: Vec<(usize, f64)>,
    /// Queries where the gold document was never returned.
    pub n_unretrieved: usize,
}

pub fn known_item_quality(ranks: &[Option<usize>]) -> KnownItemQuality {
    let n = ranks.len();
    let denom = n.max(1) as f64;
    let mut success_at = Vec::new();
    let mut mrr_at = Vec::new();
    for &k in CUTOFFS.iter() {
        let hits = ranks.iter().filter(|r| matches!(r, Some(rank) if *rank <= k)).count();
        success_at.push((k, hits as f64 / denom));
        let mrr: f64 = ranks
            .iter()
            .map(|r| match r {
                Some(rank) if *rank <= k => 1.0 / *rank as f64,
                _ => 0.0,
            })
            .sum::<f64>()
            / denom;
        mrr_at.push((k, mrr));
    }
    KnownItemQuality {
        n_queries: n,
        success_at,
        mrr_at,
        n_unretrieved: ranks.iter().filter(|r| r.is_none()).count(),
    }
}
