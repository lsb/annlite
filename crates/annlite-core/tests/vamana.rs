//! Vamana graph properties.

use annlite_core::layout::{access_cost, bfs_order, cluster_order, PageLayout, Permutation};
use annlite_core::vamana::{Vamana, VamanaParams};
use annlite_core::vectors::{dot, exact_top_k, Vectors};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

fn clustered(n: usize, dim: usize, clusters: usize, seed: u64) -> Vectors {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let centres: Vec<Vec<f32>> = (0..clusters)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect())
        .collect();
    let mut data = Vec::with_capacity(n * dim);
    for i in 0..n {
        let c = &centres[i % clusters];
        let mut v: Vec<f32> = c.iter().map(|x| x + rng.gen_range(-0.25f32..0.25)).collect();
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter_mut().for_each(|x| *x /= norm);
        data.extend_from_slice(&v);
    }
    Vectors { data, dim }
}

fn recall_at_10(idx: &Vamana, v: &Vectors, queries: &Vectors, l: usize) -> f64 {
    let mut hits = 0usize;
    for qi in 0..queries.len() {
        let q = queries.row(qi);
        let gold: Vec<u32> = exact_top_k(v, q, 10).iter().map(|x| x.0).collect();
        let got: Vec<u32> = idx.search(v, q, 10, l).iter().map(|x| x.0).collect();
        hits += gold.iter().filter(|g| got.contains(g)).count();
    }
    hits as f64 / (queries.len() * 10) as f64
}

#[test]
fn finds_near_neighbours() {
    let v = clustered(2000, 64, 40, 1);
    let q = clustered(50, 64, 40, 77);
    let idx = Vamana::build(&v, VamanaParams { r: 32, l_build: 64, alpha: 1.2, seed: 1 });
    let r = recall_at_10(&idx, &v, &q, 64);
    assert!(r > 0.85, "recall@10 was {r}");
}

#[test]
fn every_node_is_reachable() {
    // An orphan is a document that can never be returned, however good the search.
    let v = clustered(1000, 32, 20, 3);
    let idx = Vamana::build(&v, VamanaParams { r: 24, l_build: 50, alpha: 1.2, seed: 2 });
    assert_eq!(idx.stats().orphans, 0);

    let perm = bfs_order(v.len(), idx.medoid, |n| idx.neighbors(n).to_vec());
    // BFS from the medoid should reach everything; if it did not, the tail of the
    // permutation would be the unreachable remainder appended in original order.
    let mut seen = vec![false; v.len()];
    let mut reached = 0usize;
    let mut queue = std::collections::VecDeque::from([idx.medoid]);
    seen[idx.medoid as usize] = true;
    while let Some(n) = queue.pop_front() {
        reached += 1;
        for &nb in idx.neighbors(n) {
            if !seen[nb as usize] {
                seen[nb as usize] = true;
                queue.push_back(nb);
            }
        }
    }
    assert_eq!(reached, v.len(), "graph is disconnected");
    assert_eq!(perm.len(), v.len());
}

#[test]
fn degree_never_exceeds_r() {
    let v = clustered(800, 32, 16, 4);
    let r = 20;
    let idx = Vamana::build(&v, VamanaParams { r, l_build: 40, alpha: 1.2, seed: 5 });
    assert!(idx.stats().max_degree <= r, "max degree {} exceeds R={r}", idx.stats().max_degree);
    for i in 0..v.len() as u32 {
        let nb = idx.neighbors(i);
        assert!(!nb.contains(&i), "node {i} links to itself");
        let mut u = nb.to_vec();
        u.sort_unstable();
        u.dedup();
        assert_eq!(u.len(), nb.len(), "node {i} has duplicate neighbours");
    }
}

#[test]
fn build_is_deterministic() {
    let v = clustered(500, 32, 10, 6);
    let p = VamanaParams { r: 16, l_build: 32, alpha: 1.2, seed: 9 };
    let a = Vamana::build(&v, p);
    let b = Vamana::build(&v, p);
    assert_eq!(a.medoid, b.medoid);
    for i in 0..v.len() as u32 {
        assert_eq!(a.neighbors(i), b.neighbors(i), "node {i} differs between builds");
    }
}

#[test]
fn alpha_controls_graph_density_and_degenerates_when_too_high() {
    // Measured behaviour, which is the opposite of the intuitive reading: raising
    // alpha makes the occlusion test harder to pass, so fewer candidates are pruned,
    // the graph gets denser and its edges get *shorter*. Past ~1.4 it becomes an
    // approximate kNN graph and greedy search can no longer escape the medoid's
    // neighbourhood. This test pins both halves so a future change to the rule
    // cannot silently flip them.
    let v = clustered(1500, 64, 30, 7);
    let q = clustered(40, 64, 30, 71);
    let stats_for = |alpha: f32| {
        let idx = Vamana::build(&v, VamanaParams { r: 32, l_build: 64, alpha, seed: 11 });
        let mut total = 0f64;
        let mut count = 0usize;
        for i in 0..v.len() as u32 {
            for &nb in idx.neighbors(i) {
                total += (1.0 - dot(v.row(i as usize), v.row(nb as usize))) as f64;
                count += 1;
            }
        }
        (count, total / count as f64, recall_at_10(&idx, &v, &q, 64))
    };

    let (e_low, d_low, r_low) = stats_for(1.0);
    let (e_mid, d_mid, r_mid) = stats_for(1.2);
    let (e_high, _, r_high) = stats_for(2.0);

    assert!(e_mid > e_low, "alpha=1.2 should prune less than 1.0: {e_mid} vs {e_low}");
    assert!(e_high > e_mid, "alpha=2.0 should prune less still: {e_high} vs {e_mid}");
    assert!(d_mid < d_low, "denser graph should have shorter edges: {d_mid} vs {d_low}");
    assert!(r_low > 0.8 && r_mid > 0.8, "usable range lost recall: {r_low}, {r_mid}");
    assert!(r_high < 0.3, "alpha=2.0 should degenerate, but recall held at {r_high}");
}

#[test]
fn reordering_improves_page_locality() {
    // The project's central claim: node ids decide pages, so ordering nodes by
    // similarity should cut the pages a traversal touches, without touching the
    // graph or the search at all.
    let v = clustered(4000, 64, 40, 13);
    let q = clustered(40, 64, 40, 131);
    let idx = Vamana::build(&v, VamanaParams { r: 32, l_build: 64, alpha: 1.2, seed: 17 });
    let layout = PageLayout { page_bytes: 4096, record_bytes: 64 + 2 + 32 * 4 };

    let identity = Permutation::identity(v.len());
    let bfs = bfs_order(v.len(), idx.medoid, |n| idx.neighbors(n).to_vec());
    let cluster = cluster_order(&v, 64, 10, 19);

    let mean_pages = |perm: &Permutation| -> f64 {
        let mut total = 0usize;
        for qi in 0..q.len() {
            let (_, visited) = idx.greedy_search(&v, q.row(qi), 10, 64);
            total += access_cost(&visited, perm, layout).distinct_pages;
        }
        total as f64 / q.len() as f64
    };

    let (pi, pb, pc) = (mean_pages(&identity), mean_pages(&bfs), mean_pages(&cluster));
    println!("mean distinct pages -- identity {pi:.1}, bfs {pb:.1}, cluster {pc:.1}");
    assert!(pb < pi, "BFS ordering ({pb}) did not beat insertion order ({pi})");
    assert!(pc < pi, "cluster ordering ({pc}) did not beat insertion order ({pi})");
}

#[test]
fn access_cost_counts_runs_correctly() {
    let layout = PageLayout { page_bytes: 100, record_bytes: 10 };
    assert_eq!(layout.records_per_page(), 10);
    let perm = Permutation::identity(100);
    // ids 0..9 -> page 0; 10..19 -> page 1; 50 -> page 5
    let c = access_cost(&[0, 5, 9, 11, 50], &perm, layout);
    assert_eq!(c.nodes, 5);
    assert_eq!(c.distinct_pages, 3, "pages 0, 1 and 5");
    assert_eq!(c.contiguous_runs, 2, "pages 0-1 form one run, page 5 another");
    assert_eq!(access_cost(&[], &perm, layout).contiguous_runs, 0);
}
