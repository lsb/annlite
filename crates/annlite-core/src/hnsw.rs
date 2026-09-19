//! Hierarchical Navigable Small World graphs.
//!
//! HNSW builds a hierarchy of proximity graphs. The sparse upper layers act as
//! express lanes that carry a search most of the way across the space in a few hops;
//! the dense bottom layer, which holds every point, refines the result. Search is
//! greedy best-first at each layer, entering the next layer down at the best node
//! found in the one above.
//!
//! The graph is built over **exact** vectors even though search may score with PQ
//! codes. Graph quality depends on getting neighbour relationships right, and
//! quantization error at build time compounds: a bad edge is permanent, while a bad
//! score at query time only costs one comparison. Build exactly once, then choose
//! the scoring precision per query.
//!
//! A note on what this costs over a network, since that is the project's real
//! constraint: HNSW's access pattern is close to adversarial for an HTTP-backed
//! reader. Node ids are assigned in insertion order and neighbour lists point
//! essentially anywhere, so consecutive hops land on unrelated pages. `vamana`
//! addresses exactly this; HNSW is here as the baseline that shows why it is needed.

use crate::vectors::{dot, Vectors};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Similarity expressed as a distance so that smaller is better, which is what the
/// search heaps assume. Embeddings are L2-normalised, so `dot` is cosine in
/// `[-1, 1]` and this maps it to `[0, 2]` monotonically.
#[inline]
pub fn distance(a: &[f32], b: &[f32]) -> f32 {
    1.0 - dot(a, b)
}

/// A `(distance, id)` pair ordered by distance. `f32` is not `Ord`, so ordering goes
/// through `total_cmp`, which is a total order even across NaN.
#[derive(Copy, Clone, Debug, PartialEq)]
struct Candidate {
    dist: f32,
    id: u32,
}
impl Eq for Candidate {}
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist.total_cmp(&other.dist).then(self.id.cmp(&other.id))
    }
}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Reversed ordering, to turn Rust's max-heap into a min-heap.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct MinFirst(Candidate);
impl Ord for MinFirst {
    fn cmp(&self, other: &Self) -> Ordering {
        other.0.cmp(&self.0)
    }
}
impl PartialOrd for MinFirst {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Visit tracking that is reused across searches.
///
/// A fresh `vec![false; n]` per layer search would make construction quadratic in
/// the corpus size -- at a million documents the allocation and zeroing alone would
/// dominate everything else. Instead the buffer is allocated once and "cleared" by
/// bumping a generation counter, so marking and testing stay O(1) and resetting is
/// free.
struct VisitedSet {
    stamp: Vec<u32>,
    generation: u32,
}

impl VisitedSet {
    fn new(n: usize) -> Self {
        Self { stamp: vec![0; n], generation: 0 }
    }

    fn clear(&mut self) {
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 {
            // Wrapped around: the only moment the buffer must actually be zeroed,
            // once every 4 billion searches.
            self.stamp.iter_mut().for_each(|s| *s = 0);
            self.generation = 1;
        }
    }

    #[inline]
    fn test_and_set(&mut self, id: u32) -> bool {
        let slot = &mut self.stamp[id as usize];
        if *slot == self.generation {
            true
        } else {
            *slot = self.generation;
            false
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct HnswParams {
    /// Neighbours per node on layers above 0.
    pub m: usize,
    /// Neighbours per node on layer 0. Conventionally `2 * m`: the bottom layer
    /// carries every point and needs the extra connectivity to stay navigable.
    pub m0: usize,
    /// Search breadth during construction. Higher builds a better graph, slower.
    pub ef_construction: usize,
    pub seed: u64,
}

impl Default for HnswParams {
    fn default() -> Self {
        Self { m: 16, m0: 32, ef_construction: 200, seed: 0x5EED }
    }
}

pub struct Hnsw {
    pub params: HnswParams,
    /// `layers[l][id]` holds node `id`'s neighbours on layer `l`, empty above its level.
    layers: Vec<Vec<Vec<u32>>>,
    /// Top layer each node appears on.
    levels: Vec<u8>,
    entry: Option<u32>,
    len: usize,
    /// Reused across every layer search; see [`VisitedSet`].
    visited: VisitedSet,
}

impl Hnsw {
    /// Build over every row of `vectors`.
    pub fn build(vectors: &Vectors, params: HnswParams) -> Self {
        let mut idx = Self {
            params,
            layers: vec![vec![Vec::new(); vectors.len()]],
            levels: vec![0; vectors.len()],
            entry: None,
            len: 0,
            visited: VisitedSet::new(vectors.len()),
        };
        let mut rng = ChaCha8Rng::seed_from_u64(params.seed);
        // Level assignment: geometric with mean 1/ln(M), the standard choice that
        // makes each layer about M times sparser than the one below it.
        let ml = 1.0 / (params.m as f64).ln();
        for i in 0..vectors.len() {
            let level = {
                let u: f64 = rng.gen_range(f64::MIN_POSITIVE..1.0);
                ((-u.ln() * ml).floor() as usize).min(31) as u8
            };
            idx.insert(vectors, i as u32, level);
        }
        idx
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn neighbors(&self, layer: usize, id: u32) -> &[u32] {
        &self.layers[layer][id as usize]
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn level_of(&self, id: u32) -> u8 {
        self.levels[id as usize]
    }

    fn max_degree(&self, layer: usize) -> usize {
        if layer == 0 { self.params.m0 } else { self.params.m }
    }

    fn insert(&mut self, vectors: &Vectors, id: u32, level: u8) {
        self.levels[id as usize] = level;
        while self.layers.len() <= level as usize {
            self.layers.push(vec![Vec::new(); vectors.len()]);
        }

        let Some(entry) = self.entry else {
            self.entry = Some(id);
            self.len = 1;
            return;
        };

        let q = vectors.row(id as usize);
        let mut cur = entry;
        let mut cur_d = distance(q, vectors.row(cur as usize));

        // Descend the express lanes greedily down to the node's own top layer.
        let top = self.layers.len() - 1;
        for layer in ((level as usize + 1)..=top).rev() {
            (cur, cur_d) = self.greedy_descend(vectors, q, layer, cur, cur_d);
        }

        // Entry points for the next layer down. Algorithm 1 of the HNSW paper carries
        // the entire candidate set from one layer into the next rather than only its
        // nearest member. Measured effect here is nil, because the vast majority of
        // nodes live only on layer 0 and so run this loop exactly once; it is done
        // this way for conformance with the algorithm, not for a recall gain.
        let mut entries: Vec<u32> = vec![cur];

        for layer in (0..=(level as usize).min(top)).rev() {
            let found =
                self.search_layer(vectors, q, layer, &entries, self.params.ef_construction);
            let selected = self.select_neighbors(vectors, &found, self.max_degree(layer));

            self.layers[layer][id as usize] = selected.clone();
            for &n in &selected {
                // Links are bidirectional; adding the back-edge may overfill the
                // neighbour, which is then re-pruned by the same heuristic.
                self.layers[layer][n as usize].push(id);
                if self.layers[layer][n as usize].len() > self.max_degree(layer) {
                    let nv = vectors.row(n as usize);
                    let cands: Vec<Candidate> = self.layers[layer][n as usize]
                        .iter()
                        .map(|&o| Candidate { dist: distance(nv, vectors.row(o as usize)), id: o })
                        .collect();
                    self.layers[layer][n as usize] =
                        self.select_neighbors(vectors, &cands, self.max_degree(layer));
                }
            }
            entries = found.iter().map(|c| c.id).collect();
        }

        if level as usize > self.levels[entry as usize] as usize {
            self.entry = Some(id);
        }
        self.len += 1;
    }

    fn greedy_descend(
        &self,
        vectors: &Vectors,
        q: &[f32],
        layer: usize,
        mut cur: u32,
        mut cur_d: f32,
    ) -> (u32, f32) {
        loop {
            let mut improved = false;
            for &n in &self.layers[layer][cur as usize] {
                let d = distance(q, vectors.row(n as usize));
                if d < cur_d {
                    cur_d = d;
                    cur = n;
                    improved = true;
                }
            }
            if !improved {
                return (cur, cur_d);
            }
        }
    }

    /// Best-first search on one layer, returning up to `ef` results sorted by distance.
    fn search_layer(
        &mut self,
        vectors: &Vectors,
        q: &[f32],
        layer: usize,
        entries: &[u32],
        ef: usize,
    ) -> Vec<Candidate> {
        self.visited.clear();
        let mut frontier: BinaryHeap<MinFirst> = BinaryHeap::new();
        let mut result: BinaryHeap<Candidate> = BinaryHeap::new();

        for &e in entries {
            let c = Candidate { dist: distance(q, vectors.row(e as usize)), id: e };
            self.visited.test_and_set(e);
            frontier.push(MinFirst(c));
            result.push(c);
        }

        while let Some(MinFirst(best)) = frontier.pop() {
            // Stop once the nearest unexplored candidate is further than the worst
            // kept result: nothing reachable through it can improve the answer.
            if let Some(worst) = result.peek() {
                if best.dist > worst.dist && result.len() >= ef {
                    break;
                }
            }
            for ni in 0..self.layers[layer][best.id as usize].len() {
                let n = self.layers[layer][best.id as usize][ni];
                if self.visited.test_and_set(n) {
                    continue;
                }
                let c = Candidate { dist: distance(q, vectors.row(n as usize)), id: n };
                let worst = result.peek().map(|w| w.dist).unwrap_or(f32::INFINITY);
                if result.len() < ef || c.dist < worst {
                    frontier.push(MinFirst(c));
                    result.push(c);
                    if result.len() > ef {
                        result.pop();
                    }
                }
            }
        }

        let mut out = result.into_vec();
        out.sort_unstable();
        out
    }

    /// Neighbour selection heuristic.
    ///
    /// Keeping simply the `m` nearest produces clustered neighbourhoods that all
    /// point the same way, leaving regions unreachable. A candidate is kept only if
    /// it is closer to the query than to any already-kept neighbour, which favours
    /// edges pointing in *different directions* and is what keeps the graph navigable.
    fn select_neighbors(
        &self,
        vectors: &Vectors,
        candidates: &[Candidate],
        m: usize,
    ) -> Vec<u32> {
        let mut sorted: Vec<Candidate> = candidates.to_vec();
        sorted.sort_unstable();
        sorted.dedup_by_key(|c| c.id);

        let mut kept: Vec<u32> = Vec::with_capacity(m);
        for c in &sorted {
            if kept.len() >= m {
                break;
            }
            let cv = vectors.row(c.id as usize);
            let dominated = kept
                .iter()
                .any(|&k| distance(cv, vectors.row(k as usize)) < c.dist);
            if !dominated {
                kept.push(c.id);
            }
        }
        // If the heuristic was too strict to fill the quota, top up with the nearest
        // remaining candidates; an under-connected node is worse than a clustered one.
        if kept.len() < m {
            for c in &sorted {
                if kept.len() >= m {
                    break;
                }
                if !kept.contains(&c.id) {
                    kept.push(c.id);
                }
            }
        }
        kept
    }

    /// Search for the `k` nearest neighbours of `query`, exploring `ef` candidates.
    ///
    /// `ef` must be at least `k`; it is the quality/cost dial at query time.
    pub fn search(&mut self, vectors: &Vectors, query: &[f32], k: usize, ef: usize) -> Vec<(u32, f32)> {
        let Some(entry) = self.entry else { return Vec::new() };
        let ef = ef.max(k);
        let mut cur = entry;
        let mut cur_d = distance(query, vectors.row(cur as usize));
        for layer in (1..self.layers.len()).rev() {
            (cur, cur_d) = self.greedy_descend(vectors, query, layer, cur, cur_d);
        }
        let found = self.search_layer(vectors, query, 0, &[cur], ef);
        found
            .into_iter()
            .take(k)
            // Report similarity, matching `exact_top_k`, so callers compare like with like.
            .map(|c| (c.id, 1.0 - c.dist))
            .collect()
    }

    /// Statistics used by the layout and locality analysis.
    pub fn stats(&self) -> HnswStats {
        let mut edges = 0usize;
        let mut nodes_per_layer = Vec::new();
        for layer in &self.layers {
            let n = layer.iter().filter(|v| !v.is_empty()).count();
            edges += layer.iter().map(|v| v.len()).sum::<usize>();
            nodes_per_layer.push(n);
        }
        HnswStats { edges, nodes_per_layer, layers: self.layers.len() }
    }
}

#[derive(Debug)]
pub struct HnswStats {
    pub edges: usize,
    pub layers: usize,
    pub nodes_per_layer: Vec<usize>,
}
