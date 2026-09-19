//! Vamana graphs (the index behind DiskANN).
//!
//! Vamana is a *single* flat graph of fixed out-degree `r`, not a hierarchy. Both
//! properties are the reason it suits an HTTP-backed reader where HNSW does not:
//!
//! * **Flat** means one kind of record. Every node's neighbour list is the same
//!   length, so node `i` sits at byte `i * record_size` and the page holding it is
//!   arithmetic rather than a lookup. HNSW's per-node level makes records variable
//!   and forces an index of indexes.
//! * **Fixed degree** means a node and its entire adjacency fit a known budget, so
//!   a record can be sized to divide evenly into a page.
//!
//! The construction differs from HNSW's neighbour heuristic in one parameter, and it
//! is the important one. [`robust_prune`] keeps a candidate `v` only if no already
//! kept neighbour `p*` satisfies `alpha * d(p*, v) <= d(p, v)`. With `alpha = 1` this
//! is HNSW's rule: drop `v` when some kept neighbour is closer to it than `p` is.
//! With `alpha > 1` the test is deliberately harder to pass, so edges survive that
//! the strict rule would cut — long-range edges that shrink the graph's diameter.
//! Fewer hops is the entire game when a hop is a network round-trip, because hops
//! are *dependent*: the next one cannot be issued until the current returns.
//!
//! Construction runs two passes, `alpha = 1.0` then `alpha > 1`, as in the paper. The
//! first pass builds a reasonable graph from a random start; the second re-prunes it
//! with the relaxed rule, which only makes sense once neighbourhoods are meaningful.

use crate::vectors::{dot, Vectors};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

#[inline]
fn distance(a: &[f32], b: &[f32]) -> f32 {
    1.0 - dot(a, b)
}

#[derive(Clone, Copy, Debug)]
pub struct VamanaParams {
    /// Maximum out-degree. Also fixes the on-disk record size.
    pub r: usize,
    /// Search list size during construction. Larger explores more and builds better.
    pub l_build: usize,
    /// Pruning relaxation. Useful range is 1.0-1.2; see [`robust_prune`].
    pub alpha: f32,
    pub seed: u64,
}

impl Default for VamanaParams {
    fn default() -> Self {
        Self { r: 64, l_build: 100, alpha: 1.1, seed: 0xDA7A }
    }
}

pub struct Vamana {
    pub params: VamanaParams,
    /// Flat adjacency, `r` slots per node, `len[i]` of them valid.
    adj: Vec<u32>,
    deg: Vec<u32>,
    /// Search entry point: the medoid, which is roughly equidistant from everything
    /// and so gives greedy descent a neutral start regardless of query direction.
    pub medoid: u32,
    n: usize,
}

/// Visit stamps reused across searches.
///
/// Construction runs one search per node per pass, so allocating and zeroing an
/// n-element buffer each time would make the build quadratic in the corpus size
/// before a single distance was computed. Bumping a generation counter makes the
/// reset free. It is an explicit argument rather than interior state so that several
/// searches can run on different threads over the same immutable graph.
pub struct Scratch {
    seen: Vec<u32>,
    generation: u32,
}

impl Scratch {
    pub fn new(n: usize) -> Self {
        Self { seen: vec![0; n], generation: 0 }
    }

    fn begin(&mut self) -> u32 {
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 {
            self.seen.iter_mut().for_each(|s| *s = 0);
            self.generation = 1;
        }
        self.generation
    }
}

impl Vamana {
    pub fn build(vectors: &Vectors, params: VamanaParams) -> Self {
        let n = vectors.len();
        assert!(n > 0, "cannot build over an empty set");
        let r = params.r.min(n.saturating_sub(1)).max(1);

        let mut idx = Self {
            params: VamanaParams { r, ..params },
            adj: vec![u32::MAX; n * r],
            deg: vec![0; n],
            medoid: medoid(vectors),
            n,
        };
        idx.random_init(&mut ChaCha8Rng::seed_from_u64(params.seed));

        let mut rng = ChaCha8Rng::seed_from_u64(params.seed ^ 0xABCD);
        let mut order: Vec<u32> = (0..n as u32).collect();
        for i in (1..order.len()).rev() {
            order.swap(i, rng.gen_range(0..=i));
        }

        // Pass 1 at alpha = 1.0 turns the random graph into a navigable one; pass 2
        // at the configured alpha re-prunes it, which is only meaningful once
        // neighbourhoods carry information.
        let mut scratch = Scratch::new(n);
        for &alpha in &[1.0f32, params.alpha.max(1.0)] {
            for &p in &order {
                idx.insert_pass(vectors, p, alpha, &mut scratch);
            }
        }
        idx
    }

    /// Build with the greedy searches run in parallel.
    ///
    /// Insertion is inherently sequential -- each node's neighbours depend on the
    /// graph as it stands -- but the *search* half is read-only, and it is where the
    /// time goes. Nodes are therefore processed in batches: every search in a batch
    /// runs concurrently against the graph as it was at the start of the batch, then
    /// the pruning and edge installation are applied one node at a time.
    ///
    /// The approximation is that a node early in a batch does not see edges added by
    /// a node later in the same batch. Smaller batches approach the sequential build
    /// exactly; `batch` is exposed so that trade is explicit rather than hidden.
    /// Both passes still run, and the second pass re-prunes against the finished
    /// first-pass graph, which absorbs most of the difference.
    pub fn build_parallel(vectors: &Vectors, params: VamanaParams, batch: usize) -> Self {
        use rayon::prelude::*;

        let n = vectors.len();
        assert!(n > 0, "cannot build over an empty set");
        let r = params.r.min(n.saturating_sub(1)).max(1);
        let mut idx = Self {
            params: VamanaParams { r, ..params },
            adj: vec![u32::MAX; n * r],
            deg: vec![0; n],
            medoid: medoid(vectors),
            n,
        };
        idx.random_init(&mut ChaCha8Rng::seed_from_u64(params.seed));

        let mut rng = ChaCha8Rng::seed_from_u64(params.seed ^ 0xABCD);
        let mut order: Vec<u32> = (0..n as u32).collect();
        for i in (1..order.len()).rev() {
            order.swap(i, rng.gen_range(0..=i));
        }

        let batch = batch.max(1);
        // Construction at a million nodes runs for hours. Reporting progress is not
        // decoration: without it there is no way to tell a slow build from a stuck
        // one, and no basis for deciding whether to wait. Cost per insert grows with
        // the graph, so the estimate is deliberately based on the rate observed so
        // far in this pass rather than on a constant extrapolated from the start.
        let started = std::time::Instant::now();
        let total_inserts = (n * 2) as f64;
        let mut done_inserts = 0usize;
        let mut last_report = std::time::Instant::now();

        for (pass, &alpha) in [1.0f32, params.alpha.max(1.0)].iter().enumerate() {
            for chunk in order.chunks(batch) {
                let found: Vec<(u32, Vec<u32>)> = chunk
                    .par_iter()
                    .map_init(
                        || Scratch::new(n),
                        |scratch, &p| {
                            let (_, visited) = idx.greedy_search_with(
                                vectors,
                                vectors.row(p as usize),
                                1,
                                params.l_build,
                                scratch,
                            );
                            (p, visited)
                        },
                    )
                    .collect();
                for (p, visited) in found {
                    idx.apply(vectors, p, visited, alpha);
                }

                done_inserts += chunk.len();
                if last_report.elapsed().as_secs() >= 30 {
                    let frac = done_inserts as f64 / total_inserts;
                    let elapsed = started.elapsed().as_secs_f64();
                    let remaining = if frac > 0.0 { elapsed / frac - elapsed } else { 0.0 };
                    eprintln!(
                        "    pass {}/2  {:>6.2}%  {:.0}s elapsed, ~{:.0}s remaining",
                        pass + 1,
                        frac * 100.0,
                        elapsed,
                        remaining
                    );
                    last_report = std::time::Instant::now();
                }
            }
        }
        eprintln!("    graph complete in {:.0}s", started.elapsed().as_secs_f64());
        idx
    }

    /// Seed with a random r-regular graph so the first greedy searches have
    /// somewhere to go; every edge here is expected to be replaced by pass 1.
    fn random_init(&mut self, rng: &mut ChaCha8Rng) {
        let r = self.params.r;
        for i in 0..self.n {
            let mut count = 0;
            while count < r {
                let cand = rng.gen_range(0..self.n) as u32;
                if cand as usize != i && !self.adj[i * r..i * r + count].contains(&cand) {
                    self.adj[i * r + count] = cand;
                    count += 1;
                }
            }
            self.deg[i] = r as u32;
        }
    }

    fn insert_pass(&mut self, vectors: &Vectors, p: u32, alpha: f32, scratch: &mut Scratch) {
        let (_, visited) =
            self.greedy_search_with(vectors, vectors.row(p as usize), 1, self.params.l_build, scratch);
        self.apply(vectors, p, visited, alpha);
    }

    /// Prune `p`'s candidate set and install the resulting edges, including the
    /// back-edges. Separated from the search so a batch of searches can run in
    /// parallel and their results be applied one at a time.
    fn apply(&mut self, vectors: &Vectors, p: u32, visited: Vec<u32>, alpha: f32) {
        let mut candidates: Vec<u32> = visited.into_iter().filter(|&v| v != p).collect();
        candidates.extend(self.neighbors(p).iter().copied().filter(|&v| v != p));
        candidates.sort_unstable();
        candidates.dedup();

        let pruned = robust_prune(vectors, p, candidates, alpha, self.params.r);
        self.set_neighbors(p, &pruned);

        // Edges are bidirectional. Adding the back-edge may overflow the neighbour's
        // budget, in which case that neighbour is re-pruned by the same rule.
        for &j in &pruned {
            if self.neighbors(j).contains(&p) {
                continue;
            }
            if (self.deg[j as usize] as usize) < self.params.r {
                let d = self.deg[j as usize] as usize;
                self.adj[j as usize * self.params.r + d] = p;
                self.deg[j as usize] += 1;
            } else {
                let mut cand: Vec<u32> = self.neighbors(j).to_vec();
                cand.push(p);
                let kept = robust_prune(vectors, j, cand, alpha, self.params.r);
                self.set_neighbors(j, &kept);
            }
        }
    }

    #[inline]
    pub fn neighbors(&self, id: u32) -> &[u32] {
        let r = self.params.r;
        let base = id as usize * r;
        &self.adj[base..base + self.deg[id as usize] as usize]
    }

    fn set_neighbors(&mut self, id: u32, list: &[u32]) {
        let r = self.params.r;
        let base = id as usize * r;
        let k = list.len().min(r);
        self.adj[base..base + k].copy_from_slice(&list[..k]);
        self.deg[id as usize] = k as u32;
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Greedy best-first search. Returns the `k` best and the full visited set, the
    /// latter being what construction prunes against and what the page-locality
    /// analysis counts.
    pub fn greedy_search(
        &self,
        vectors: &Vectors,
        query: &[f32],
        k: usize,
        l: usize,
    ) -> (Vec<(u32, f32)>, Vec<u32>) {
        let mut scratch = Scratch::new(self.n);
        self.greedy_search_with(vectors, query, k, l, &mut scratch)
    }

    /// As [`Vamana::greedy_search`], reusing a caller-owned scratch buffer.
    pub fn greedy_search_with(
        &self,
        vectors: &Vectors,
        query: &[f32],
        k: usize,
        l: usize,
        scratch: &mut Scratch,
    ) -> (Vec<(u32, f32)>, Vec<u32>) {
        let l = l.max(k).max(1);
        let mut list: Vec<(u32, f32, bool)> =
            vec![(self.medoid, distance(query, vectors.row(self.medoid as usize)), false)];
        let generation = scratch.begin();
        let seen = &mut scratch.seen;
        seen[self.medoid as usize] = generation;
        let mut visited = Vec::new();

        loop {
            // The closest node not yet expanded.
            let Some(pos) = list
                .iter()
                .enumerate()
                .filter(|(_, e)| !e.2)
                .min_by(|a, b| a.1 .1.total_cmp(&b.1 .1))
                .map(|(i, _)| i)
            else {
                break;
            };
            list[pos].2 = true;
            let node = list[pos].0;
            visited.push(node);

            for &nb in self.neighbors(node) {
                if seen[nb as usize] == generation {
                    continue;
                }
                seen[nb as usize] = generation;
                list.push((nb, distance(query, vectors.row(nb as usize)), false));
            }
            if list.len() > l {
                list.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
                list.truncate(l);
            }
        }

        list.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        let best = list.iter().take(k).map(|e| (e.0, 1.0 - e.1)).collect();
        (best, visited)
    }

    /// Search returning similarities, matching `exact_top_k`'s orientation.
    pub fn search(&self, vectors: &Vectors, query: &[f32], k: usize, l: usize) -> Vec<(u32, f32)> {
        self.greedy_search(vectors, query, k, l).0
    }

    pub fn stats(&self) -> VamanaStats {
        let edges: usize = self.deg.iter().map(|&d| d as usize).sum();
        let max = self.deg.iter().copied().max().unwrap_or(0) as usize;
        let orphans = self.deg.iter().filter(|&&d| d == 0).count();
        VamanaStats { edges, mean_degree: edges as f64 / self.n as f64, max_degree: max, orphans }
    }
}

#[derive(Debug)]
pub struct VamanaStats {
    pub edges: usize,
    pub mean_degree: f64,
    pub max_degree: usize,
    pub orphans: usize,
}

/// The point minimising total distance to all others, approximated by the point
/// closest to the centroid. An exact medoid is O(n^2); this is O(n) and the entry
/// point only has to be *central*, not optimal.
fn medoid(vectors: &Vectors) -> u32 {
    let dim = vectors.dim;
    let mut centre = vec![0f64; dim];
    for row in vectors.rows() {
        for (c, &x) in centre.iter_mut().zip(row) {
            *c += x as f64;
        }
    }
    let inv = 1.0 / vectors.len() as f64;
    let centre: Vec<f32> = centre.iter().map(|&c| (c * inv) as f32).collect();
    vectors
        .rows()
        .enumerate()
        .min_by(|a, b| {
            let da: f32 = a.1.iter().zip(&centre).map(|(x, y)| (x - y) * (x - y)).sum();
            let db: f32 = b.1.iter().zip(&centre).map(|(x, y)| (x - y) * (x - y)).sum();
            da.total_cmp(&db)
        })
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

/// Vamana's pruning rule.
///
/// Repeatedly take the nearest surviving candidate `p*`, keep it, then discard every
/// candidate `v` for which `alpha * d(p*, v) <= d(p, v)` — read as "`p*` already
/// covers the direction of `v` well enough".
///
/// The effect of `alpha` is worth stating precisely, because it is easy to get
/// backwards and the opposite reading was measured and rejected (RESEARCH_LOG.md
/// section 11.2). Raising `alpha` makes the discard test *harder* to satisfy, so
/// fewer candidates are occluded and the kept list fills from the nearest end of the
/// pool. Higher `alpha` therefore yields a denser graph with *shorter* edges, not
/// longer ones. Past about 1.4 that degenerates into an approximate k-nearest-
/// neighbour graph, which is exactly the badly-navigable structure diversified
/// pruning exists to avoid: measured recall collapses from 0.93 to 0.06 because
/// greedy search can no longer escape the medoid's neighbourhood.
///
/// The useful range is 1.0-1.2. Scaling the other side of the inequality instead
/// (`d(p*, v) <= alpha * d(p, v)`) prunes ever more aggressively and was measured to
/// strip the graph to mean degree 1.3 and recall 0.005.
pub fn robust_prune(
    vectors: &Vectors,
    p: u32,
    candidates: Vec<u32>,
    alpha: f32,
    r: usize,
) -> Vec<u32> {
    let pv = vectors.row(p as usize);
    let mut pool: Vec<(u32, f32)> = candidates
        .into_iter()
        .filter(|&c| c != p)
        .map(|c| (c, distance(pv, vectors.row(c as usize))))
        .collect();
    pool.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));

    let mut kept: Vec<u32> = Vec::with_capacity(r);
    let mut alive = vec![true; pool.len()];
    for i in 0..pool.len() {
        if !alive[i] {
            continue;
        }
        let (star, _) = pool[i];
        kept.push(star);
        if kept.len() >= r {
            break;
        }
        let sv = vectors.row(star as usize);
        for j in (i + 1)..pool.len() {
            if alive[j] && alpha * distance(sv, vectors.row(pool[j].0 as usize)) <= pool[j].1 {
                alive[j] = false;
            }
        }
    }
    kept
}
