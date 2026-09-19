//! Querying an index out of SQLite, counting what a network client would pay.
//!
//! The search deliberately reads through SQL rather than through a memory-mapped
//! structure, because that is what a browser client does. Every node record fetched
//! is attributed to a page, so a query reports not only its results but the number
//! of round-trips a client holding no copy of the file would have needed.

use annlite_core::layout::PageLayout;
use annlite_core::pq::{ProductQuantizer, CENTROIDS};
use anyhow::Result;
use rusqlite::Connection;
use std::collections::HashSet;

use crate::format::RecordFormat;
use crate::store::PAGE_BYTES;

pub struct Index {
    pub fmt: RecordFormat,
    pub pq: ProductQuantizer,
    pub medoid: u32,
    pub count: usize,
    pub dim: usize,
}

impl Index {
    pub fn open(conn: &Connection) -> Result<Self> {
        let get = |k: &str| -> Result<String> {
            let v: Vec<u8> =
                conn.query_row("SELECT value FROM annlite_meta WHERE key=?1", [k], |r| r.get(0))?;
            Ok(String::from_utf8(v)?)
        };
        let dim: usize = get("dim")?.parse()?;
        let m: usize = get("m")?.parse()?;
        let dsub: usize = get("dsub")?.parse()?;
        let r: usize = get("r")?.parse()?;
        let count: usize = get("count")?.parse()?;
        let medoid: u32 = get("medoid")?.parse()?;

        let raw: Vec<u8> = conn.query_row(
            "SELECT value FROM annlite_meta WHERE key='pq_centroids'",
            [],
            |r| r.get(0),
        )?;
        anyhow::ensure!(
            raw.len() == m * CENTROIDS * dsub * 4,
            "codebook is {} bytes, expected {}",
            raw.len(),
            m * CENTROIDS * dsub * 4
        );
        let centroids: Vec<f32> = raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        Ok(Self {
            fmt: RecordFormat { m, r },
            pq: ProductQuantizer { dim, m, dsub, centroids },
            medoid,
            count,
            dim,
        })
    }

    pub fn page_layout(&self) -> PageLayout {
        PageLayout { page_bytes: PAGE_BYTES, record_bytes: self.fmt.len() }
    }
}

#[derive(Debug, Default, Clone)]
pub struct SearchCost {
    /// Node records read. One per node, never more: a record carries both the
    /// score and the adjacency, so nothing has to be looked up twice.
    pub nodes_read: usize,
    /// Distinct pages those records occupied.
    pub distinct_pages: usize,
    /// Rounds of the greedy loop. These are *dependent*: the next cannot be issued
    /// until the current returns, so this is the latency-critical count, whereas
    /// nodes within one round can be fetched in parallel.
    pub hops: usize,
    /// Full vectors fetched to rerank.
    pub vectors_read: usize,
    pub rerank_pages: usize,
    /// Maximal runs of consecutive *vector* pages. Reranking reads a different table
    /// from a candidate list that is in score order, not id order, so its reads are
    /// scattered and must be counted as their own request run rather than folded into
    /// the traversal's.
    pub rerank_runs: usize,
    /// Maximal runs of consecutive node pages; see [`SearchCost::contiguous_runs_estimate`].
    pub contiguous_runs: usize,
}

/// Distinct pages and the number of maximal runs of consecutive ones.
///
/// The pair is the span a client's round-trip count lies in: one request per page if
/// it fetches pages, one per run if it coalesces neighbours into a range.
fn pages_and_runs(pages: HashSet<usize>) -> (usize, usize) {
    let mut sorted: Vec<usize> = pages.into_iter().collect();
    sorted.sort_unstable();
    let runs = sorted.windows(2).filter(|w| w[1] != w[0] + 1).count()
        + usize::from(!sorted.is_empty());
    (sorted.len(), runs)
}

/// All PQ codes held client-side, fetched once per session.
///
/// With codes resident, traversal fetches a node record only when it *expands* that
/// node, because scoring its neighbours no longer requires reading them. That turns
/// the per-query cost from "one fetch per scored node" into "one fetch per expanded
/// node", which measurement showed to be roughly an order of magnitude fewer.
pub struct ResidentCodes {
    pub codes: Vec<u8>,
    pub m: usize,
}

impl ResidentCodes {
    /// Fetch the whole code blob. One sequential range request.
    pub fn load(conn: &Connection, idx: &Index) -> Result<Self> {
        let codes: Vec<u8> =
            conn.query_row("SELECT codes FROM annlite_codeblob WHERE id = 0", [], |r| r.get(0))?;
        anyhow::ensure!(
            codes.len() == idx.count * idx.fmt.m,
            "code blob is {} bytes, expected {}",
            codes.len(),
            idx.count * idx.fmt.m
        );
        Ok(Self { codes, m: idx.fmt.m })
    }

    pub fn bytes(&self) -> usize {
        self.codes.len()
    }

    #[inline]
    fn code(&self, id: u32) -> &[u8] {
        &self.codes[id as usize * self.m..(id as usize + 1) * self.m]
    }
}

pub struct SearchResult {
    pub results: Vec<(u32, f32)>,
    pub cost: SearchCost,
}

/// Beam search over the stored graph, scoring candidates with PQ, then optionally
/// reranking the best `rerank` of them against full vectors.
///
/// `beam` controls how many nodes are expanded per round. Widening it costs nothing
/// in round-trips — the whole frontier is one batched fetch — while narrowing the
/// number of rounds. That trade is backwards from an in-memory index, where beam
/// width is pure extra work, and it is the reason this is a beam search rather than
/// the single-node greedy walk that construction uses.
pub fn search(
    conn: &Connection,
    idx: &Index,
    query: &[f32],
    k: usize,
    l: usize,
    beam: usize,
    rerank: usize,
) -> Result<SearchResult> {
    let table = idx.pq.score_table(query);
    let layout = idx.page_layout();
    let mut cost = SearchCost::default();
    let mut pages: HashSet<usize> = HashSet::new();

    let mut stmt = conn.prepare_cached("SELECT rec FROM annlite_nodes WHERE id = ?1")?;

    // Read one node record: its PQ score and its adjacency come from the same
    // bytes, so this is the only fetch a traversal step ever needs.
    fn read_node(
        stmt: &mut rusqlite::CachedStatement<'_>,
        fmt: &RecordFormat,
        table: &annlite_core::pq::ScoreTable,
        layout: &PageLayout,
        id: u32,
        cost: &mut SearchCost,
        pages: &mut HashSet<usize>,
    ) -> Result<(f32, Vec<u32>)> {
        let rec: Vec<u8> = stmt.query_row([id as i64], |r| r.get(0))?;
        cost.nodes_read += 1;
        pages.insert(layout.page_of(id));
        let (code, neighbors) = fmt.decode(&rec)?;
        Ok((table.score(code), neighbors))
    }

    let (s0, nb) = read_node(
        &mut stmt, &idx.fmt, &table, &layout, idx.medoid, &mut cost, &mut pages,
    )?;
    let mut list: Vec<(u32, f32, bool)> = vec![(idx.medoid, s0, true)];
    let mut seen: HashSet<u32> = HashSet::from([idx.medoid]);
    let mut pending: Vec<u32> = nb;
    cost.hops += 1;

    let l = l.max(k).max(1);
    let mut cached: std::collections::HashMap<u32, Vec<u32>> = std::collections::HashMap::new();
    loop {
        // Score everything discovered in the previous round. In a browser these are
        // one batched range request, not many.
        let mut added = false;
        for id in std::mem::take(&mut pending) {
            if !seen.insert(id) {
                continue;
            }
            let (score, neighbors) = read_node(
                &mut stmt, &idx.fmt, &table, &layout, id, &mut cost, &mut pages,
            )?;
            // The adjacency arrived with the score, so expanding this node later
            // costs no further read. Keeping it is what makes `nodes_read` equal
            // the number of distinct nodes rather than counting re-fetches.
            cached.insert(id, neighbors);
            list.push((id, score, false));
            added = true;
        }
        if added {
            list.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
            list.truncate(l);
        }

        // Expand the best `beam` unexpanded entries together.
        let frontier: Vec<u32> = list
            .iter_mut()
            .filter(|e| !e.2)
            .take(beam)
            .map(|e| {
                e.2 = true;
                e.0
            })
            .collect();
        if frontier.is_empty() {
            break;
        }
        cost.hops += 1;
        for id in frontier {
            if let Some(neighbors) = cached.get(&id) {
                let next: Vec<u32> =
                    neighbors.iter().copied().filter(|n| !seen.contains(n)).collect();
                pending.extend(next);
            }
        }
    }
    (cost.distinct_pages, cost.contiguous_runs) = pages_and_runs(pages);

    let mut results: Vec<(u32, f32)> = list.iter().map(|e| (e.0, e.1)).collect();


    if rerank > 0 {
        // PQ recovers ~99% of the true top 10 into a top-100 pool but orders it
        // correctly only ~61% of the time (RESEARCH_LOG.md section 7.3), so the pool
        // is rescored exactly. This is the expensive half over a network: full
        // vectors are 1.5 KB each and scattered.
        let depth = rerank.min(results.len());
        let mut vstmt = conn.prepare_cached("SELECT v FROM annlite_vectors WHERE id = ?1")?;
        let vec_layout = PageLayout { page_bytes: PAGE_BYTES, record_bytes: idx.dim * 4 + 10 };
        let mut vpages: HashSet<usize> = HashSet::new();
        let mut rescored = Vec::with_capacity(depth);
        for &(id, _) in results.iter().take(depth) {
            let raw: Vec<u8> = vstmt.query_row([id as i64], |r| r.get(0))?;
            cost.vectors_read += 1;
            vpages.insert(vec_layout.page_of(id));
            let v: Vec<f32> = raw
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            rescored.push((id, annlite_core::vectors::dot(query, &v)));
        }
        (cost.rerank_pages, cost.rerank_runs) = pages_and_runs(vpages);
        rescored.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        results = rescored;
    }

    results.truncate(k);
    Ok(SearchResult { results, cost })
}

impl SearchCost {
    /// Round-trips for a client that coalesces adjacent pages into one range
    /// request. The true figure sits between this and `distinct_pages`, depending on
    /// the client's block size.
    pub fn contiguous_runs_estimate(&self) -> usize {
        self.contiguous_runs
    }
}

/// Beam search with the PQ codes already client-side.
///
/// The difference from [`search`] is only in *when* a record is read. Candidates are
/// scored from resident codes, so a record is fetched only for a node the search
/// decides to expand. Results are identical to [`search`] given the same parameters;
/// the costs are not.
pub fn search_resident(
    conn: &Connection,
    idx: &Index,
    codes: &ResidentCodes,
    query: &[f32],
    k: usize,
    l: usize,
    beam: usize,
    rerank: usize,
) -> Result<SearchResult> {
    let table = idx.pq.score_table(query);
    let layout = idx.page_layout();
    let mut cost = SearchCost::default();
    let mut pages: HashSet<usize> = HashSet::new();
    let mut stmt = conn.prepare_cached("SELECT rec FROM annlite_nodes WHERE id = ?1")?;

    let l = l.max(k).max(1);
    let mut list: Vec<(u32, f32, bool)> =
        vec![(idx.medoid, table.score(codes.code(idx.medoid)), false)];
    let mut seen: HashSet<u32> = HashSet::from([idx.medoid]);

    loop {
        let frontier: Vec<u32> = list
            .iter_mut()
            .filter(|e| !e.2)
            .take(beam.max(1))
            .map(|e| {
                e.2 = true;
                e.0
            })
            .collect();
        if frontier.is_empty() {
            break;
        }
        cost.hops += 1;

        let mut discovered: Vec<u32> = Vec::new();
        for id in frontier {
            let rec: Vec<u8> = stmt.query_row([id as i64], |r| r.get(0))?;
            cost.nodes_read += 1;
            pages.insert(layout.page_of(id));
            let (_, neighbors) = idx.fmt.decode(&rec)?;
            for nb in neighbors {
                if seen.insert(nb) {
                    discovered.push(nb);
                }
            }
        }
        if discovered.is_empty() {
            continue;
        }
        // Scoring costs nothing over the network: the codes are already here.
        for id in discovered {
            list.push((id, table.score(codes.code(id)), false));
        }
        list.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        list.truncate(l);
    }

    (cost.distinct_pages, cost.contiguous_runs) = pages_and_runs(pages);

    let mut results: Vec<(u32, f32)> = list.iter().map(|e| (e.0, e.1)).collect();
    if rerank > 0 {
        let depth = rerank.min(results.len());
        let mut vstmt = conn.prepare_cached("SELECT v FROM annlite_vectors WHERE id = ?1")?;
        let vec_layout = PageLayout { page_bytes: PAGE_BYTES, record_bytes: idx.dim * 4 + 10 };
        let mut vpages: HashSet<usize> = HashSet::new();
        let mut rescored = Vec::with_capacity(depth);
        for &(id, _) in results.iter().take(depth) {
            let raw: Vec<u8> = vstmt.query_row([id as i64], |r| r.get(0))?;
            cost.vectors_read += 1;
            vpages.insert(vec_layout.page_of(id));
            let v: Vec<f32> = raw
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            rescored.push((id, annlite_core::vectors::dot(query, &v)));
        }
        (cost.rerank_pages, cost.rerank_runs) = pages_and_runs(vpages);
        rescored.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        results = rescored;
    }
    results.truncate(k);
    Ok(SearchResult { results, cost })
}
