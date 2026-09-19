//! The on-disk record format.
//!
//! One design decision dominates this module, and it comes straight out of the FTS5
//! baseline. Measuring that baseline showed a 79x gap between *finding* candidates
//! (20 pages) and *scoring* them (1,538 pages), because BM25 needs a per-document
//! length that lives in a separate rowid-keyed table — one random lookup per match.
//! The lesson was: co-locate whatever the scorer needs with the traversal data, or
//! make the scorer need nothing per candidate.
//!
//! This format does both. A node's record holds its PQ code *and* its adjacency
//! list, and PQ's asymmetric distance computation scores a document from its own
//! code alone against a table held in memory. So a single page read yields both the
//! score of every node on that page and the ids to hop to next. There is no second
//! lookup, and the traversal never touches the full vectors at all.
//!
//! ```text
//! record := pq_code[m]  degree:u16  neighbours[r]:u32   (little-endian)
//! ```
//!
//! The record is fixed-size, which is what lets node `i` be found at a computed
//! offset instead of through an index, and what makes the page a node lives on a
//! function of its id alone — see `annlite_core::layout`.

use anyhow::Result;

/// Fixed-size node record: PQ code plus adjacency.
#[derive(Clone, Copy, Debug)]
pub struct RecordFormat {
    /// PQ code length in bytes; also the number of subquantizers.
    pub m: usize,
    /// Maximum out-degree.
    pub r: usize,
}

impl RecordFormat {
    pub fn len(&self) -> usize {
        self.m + 2 + self.r * 4
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Records that fit a page, allowing for SQLite's per-cell overhead.
    ///
    /// Each row in a rowid table costs a 2-byte cell pointer plus varints for the
    /// rowid and the payload length — about 10 bytes at these sizes — and the page
    /// carries an 8-byte header. This is an estimate, and the measured figure from
    /// `dbstat` is what the benchmarks report; it exists so a format can be sized
    /// before building anything.
    pub fn records_per_page(&self, page_bytes: usize) -> usize {
        ((page_bytes - 8) / (self.len() + 10)).max(1)
    }

    pub fn encode(&self, code: &[u8], neighbors: &[u32]) -> Vec<u8> {
        debug_assert_eq!(code.len(), self.m);
        debug_assert!(neighbors.len() <= self.r);
        let mut out = vec![0u8; self.len()];
        out[..self.m].copy_from_slice(code);
        let deg = neighbors.len() as u16;
        out[self.m..self.m + 2].copy_from_slice(&deg.to_le_bytes());
        for (i, &n) in neighbors.iter().enumerate() {
            let off = self.m + 2 + i * 4;
            out[off..off + 4].copy_from_slice(&n.to_le_bytes());
        }
        out
    }

    pub fn decode<'a>(&self, rec: &'a [u8]) -> Result<(&'a [u8], Vec<u32>)> {
        anyhow::ensure!(
            rec.len() == self.len(),
            "record is {} bytes, expected {}",
            rec.len(),
            self.len()
        );
        let code = &rec[..self.m];
        let deg = u16::from_le_bytes([rec[self.m], rec[self.m + 1]]) as usize;
        anyhow::ensure!(deg <= self.r, "record claims degree {deg} above the maximum {}", self.r);
        let neighbors = (0..deg)
            .map(|i| {
                let off = self.m + 2 + i * 4;
                u32::from_le_bytes([rec[off], rec[off + 1], rec[off + 2], rec[off + 3]])
            })
            .collect();
        Ok((code, neighbors))
    }
}
