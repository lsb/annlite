//! FTS5 baseline for annlite.
//!
//! Every ANN index in this project is compared against SQLite's own full-text search,
//! so this crate measures FTS5 the way the ANN indexes will be measured: build cost,
//! query latency, retrieval quality, and — the number that decides whether an index is
//! usable from a browser — how many distinct database pages a single query touches.
//! See [`vfs`] for how the page counts are obtained and what they do and do not mean.

pub mod db;
pub mod index;
pub mod metrics;
pub mod pages;
pub mod query;
pub mod vfs;

/// The three corpus scales, as named in `data/corpus/`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Scale {
    pub name: &'static str,
    pub n_docs: usize,
}

pub const SCALES: [Scale; 3] = [
    Scale { name: "100", n_docs: 100 },
    Scale { name: "10k", n_docs: 10_000 },
    Scale { name: "1m", n_docs: 1_000_000 },
];

pub fn scale_by_name(name: &str) -> Option<Scale> {
    SCALES.iter().copied().find(|s| s.name == name)
}
