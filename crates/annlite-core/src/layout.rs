//! Node ordering and page-locality measurement.
//!
//! This module holds what the project is actually about. A graph index stored in
//! SQLite is a table of fixed-size records; SQLite lays a rowid table out in rowid
//! order, so record `i` lands on page `i * record_size / page_size`. Which page a
//! node occupies is therefore decided entirely by the id it was given, and ids are
//! ours to choose.
//!
//! That choice is worth more than any amount of query-time cleverness. A greedy
//! search hops from a node to its neighbours; if neighbours carry nearby ids they
//! share a page, and a hop that would have been a network round-trip becomes free.
//! The FTS5 baseline already demonstrated the size of the effect from the other
//! direction: `VACUUM` left page *counts* untouched while collapsing 2,380 scattered
//! reads into 133 contiguous ones, purely by reordering.
//!
//! Vamana is the index this applies to, because its records are fixed-size and its
//! graph is flat. HNSW's variable per-node degree makes the offset of node `i`
//! depend on every node before it.

use crate::vectors::Vectors;
use std::collections::VecDeque;

/// How node ids are assigned before the index is written out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ordering {
    /// Insertion order: whatever order documents arrived in. The baseline, and for
    /// this corpus effectively random with respect to similarity.
    Identity,
    /// Breadth-first from the medoid over the graph itself. Puts a node near the
    /// neighbours it was reached through, which is exactly the traversal a query
    /// performs.
    Bfs,
    /// Group by k-means cluster, ordering clusters by size. Similar vectors get
    /// adjacent ids without reference to the graph, so it also works for a flat scan.
    Cluster,
}

/// A permutation: `new_id_of[old] = new`, and `old_id_of[new] = old`.
pub struct Permutation {
    pub new_id_of: Vec<u32>,
    pub old_id_of: Vec<u32>,
}

impl Permutation {
    pub fn identity(n: usize) -> Self {
        Self { new_id_of: (0..n as u32).collect(), old_id_of: (0..n as u32).collect() }
    }

    /// Build from a listing of old ids in their intended new order.
    pub fn from_order(old_id_of: Vec<u32>) -> Self {
        let mut new_id_of = vec![0u32; old_id_of.len()];
        for (new, &old) in old_id_of.iter().enumerate() {
            new_id_of[old as usize] = new as u32;
        }
        Self { new_id_of, old_id_of }
    }

    pub fn len(&self) -> usize {
        self.old_id_of.len()
    }

    pub fn is_empty(&self) -> bool {
        self.old_id_of.is_empty()
    }
}

/// Breadth-first order over an adjacency function, starting from `start`.
///
/// Nodes unreachable from `start` are appended in their original order, so the
/// result is always a full permutation even if the graph is disconnected.
pub fn bfs_order(n: usize, start: u32, neighbors: impl Fn(u32) -> Vec<u32>) -> Permutation {
    let mut seen = vec![false; n];
    let mut out = Vec::with_capacity(n);
    let mut queue = VecDeque::new();
    queue.push_back(start);
    seen[start as usize] = true;
    while let Some(node) = queue.pop_front() {
        out.push(node);
        for nb in neighbors(node) {
            if !seen[nb as usize] {
                seen[nb as usize] = true;
                queue.push_back(nb);
            }
        }
    }
    for (i, &s) in seen.iter().enumerate() {
        if !s {
            out.push(i as u32);
        }
    }
    Permutation::from_order(out)
}

/// Cluster order: assign each vector to the nearest of `k` centroids found by
/// k-means, then list clusters largest first with members contiguous.
pub fn cluster_order(vectors: &Vectors, k: usize, iters: usize, seed: u64) -> Permutation {
    let assign = crate::pq::kmeans_assign(vectors, k, iters, seed);
    let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); k];
    for (i, &c) in assign.iter().enumerate() {
        buckets[c as usize].push(i as u32);
    }
    buckets.sort_by_key(|b| std::cmp::Reverse(b.len()));
    Permutation::from_order(buckets.into_iter().flatten().collect())
}

/// How an index's records map onto pages.
#[derive(Clone, Copy, Debug)]
pub struct PageLayout {
    pub page_bytes: usize,
    pub record_bytes: usize,
}

impl PageLayout {
    /// Records per page, ignoring any per-page header. Overstates density slightly,
    /// which is stated rather than silently corrected for: the comparison between
    /// orderings is unaffected because every ordering shares the assumption.
    pub fn records_per_page(&self) -> usize {
        (self.page_bytes / self.record_bytes).max(1)
    }

    pub fn page_of(&self, id: u32) -> usize {
        id as usize / self.records_per_page()
    }

    pub fn pages_for(&self, n: usize) -> usize {
        n.div_ceil(self.records_per_page())
    }
}

/// What a traversal costs a network client.
#[derive(Clone, Copy, Debug, Default)]
pub struct AccessCost {
    /// Nodes whose records had to be read.
    pub nodes: usize,
    /// Distinct pages touched: the round-trip count for a client that fetches one
    /// page per request and caches within a query.
    pub distinct_pages: usize,
    /// Maximal runs of consecutive pages: the round-trip count for a client that
    /// coalesces adjacent pages into one range request. The realistic figure sits
    /// between this and `distinct_pages`.
    pub contiguous_runs: usize,
}

/// Page cost of visiting `visited` under a layout, after `perm` has renumbered nodes.
pub fn access_cost(visited: &[u32], perm: &Permutation, layout: PageLayout) -> AccessCost {
    let mut pages: Vec<usize> = visited
        .iter()
        .map(|&old| layout.page_of(perm.new_id_of[old as usize]))
        .collect();
    pages.sort_unstable();
    pages.dedup();

    let runs = pages
        .windows(2)
        .filter(|w| w[1] != w[0] + 1)
        .count()
        + usize::from(!pages.is_empty());

    AccessCost { nodes: visited.len(), distinct_pages: pages.len(), contiguous_runs: runs }
}
