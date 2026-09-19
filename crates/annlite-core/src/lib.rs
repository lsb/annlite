//! Vector index algorithms for annlite.
//!
//! Everything here is built for a reader that pays a network round-trip per page it
//! has not already fetched. That constraint shapes choices that would look odd for
//! an in-memory index: compressed document representations so a candidate list fits
//! in few pages, and graph layouts chosen for locality rather than for the shortest
//! possible search path.

pub mod hnsw;
pub mod layout;
pub mod pq;
pub mod vamana;
pub mod vectors;
