//! Vamana against HNSW on the metric that matters here: recall per *dependent hop*.
//!
//! Distance computations are nearly free over a network; a hop is not, because the
//! next one cannot be issued until the current returns. A graph that reaches the
//! same recall in fewer hops wins even if it does far more arithmetic.

use annlite_core::hnsw::{Hnsw, HnswParams};
use annlite_core::vamana::{Vamana, VamanaParams};
use annlite_core::vectors::{exact_top_k, Vectors};
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let docs = Vectors::load(Path::new("data/embeddings/docs-10k.f32"), 384)?;
    let queries = Vectors::load(Path::new("data/embeddings/queries-10k.f32"), 384)?;
    let n_q = 200.min(queries.len());
    let gold: Vec<Vec<u32>> = (0..n_q)
        .map(|i| exact_top_k(&docs, queries.row(i), 10).iter().map(|x| x.0).collect())
        .collect();

    println!("=== Vamana ===");
    for &(r, alpha) in &[(32usize, 1.0f32), (32, 1.2), (64, 1.2), (64, 1.5)] {
        let t = std::time::Instant::now();
        let idx = Vamana::build(&docs, VamanaParams { r, l_build: 100, alpha, seed: 0xDA7A });
        let build = t.elapsed().as_secs_f64();
        let st = idx.stats();
        println!(
            "\nR={r} alpha={alpha}: build {build:.1}s, mean degree {:.1}, max {}, orphans {}",
            st.mean_degree, st.max_degree, st.orphans
        );
        println!("  {:>5} {:>10} {:>9} {:>10}", "L", "recall@10", "hops", "ms/query");
        for &l in &[10usize, 20, 50, 100, 200] {
            let t = std::time::Instant::now();
            let (mut hits, mut hops) = (0usize, 0usize);
            for i in 0..n_q {
                let (best, visited) = idx.greedy_search(&docs, queries.row(i), 10, l);
                let ids: Vec<u32> = best.iter().map(|x| x.0).collect();
                hits += gold[i].iter().filter(|g| ids.contains(g)).count();
                hops += visited.len();
            }
            let ms = t.elapsed().as_secs_f64() * 1000.0 / n_q as f64;
            println!("  {l:>5} {:>10.3} {:>9.1} {ms:>10.3}",
                     hits as f64 / (n_q * 10) as f64, hops as f64 / n_q as f64);
        }
    }

    println!("\n=== HNSW, for comparison ===");
    let mut h = Hnsw::build(&docs, HnswParams { m: 32, m0: 64, ef_construction: 100, seed: 7 });
    println!("  {:>5} {:>10} {:>10}", "ef", "recall@10", "ms/query");
    for &ef in &[10usize, 50, 100, 200] {
        let t = std::time::Instant::now();
        let mut hits = 0usize;
        for i in 0..n_q {
            let ids: Vec<u32> =
                h.search(&docs, queries.row(i), 10, ef).iter().map(|x| x.0).collect();
            hits += gold[i].iter().filter(|g| ids.contains(g)).count();
        }
        let ms = t.elapsed().as_secs_f64() * 1000.0 / n_q as f64;
        println!("  {ef:>5} {:>10.3} {ms:>10.3}", hits as f64 / (n_q * 10) as f64);
    }
    Ok(())
}
