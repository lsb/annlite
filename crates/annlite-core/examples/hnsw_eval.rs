//! HNSW recall and cost against exact search, on real embeddings.

use annlite_core::hnsw::{Hnsw, HnswParams};
use annlite_core::vectors::{exact_top_k, Vectors};
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let docs = Vectors::load(Path::new("data/embeddings/docs-10k.f32"), 384)?;
    let queries = Vectors::load(Path::new("data/embeddings/queries-10k.f32"), 384)?;
    let n_q = 200.min(queries.len());

    let gold: Vec<Vec<u32>> = (0..n_q)
        .map(|i| exact_top_k(&docs, queries.row(i), 10).iter().map(|x| x.0).collect())
        .collect();

    let t0 = std::time::Instant::now();
    let exact_ms = {
        let t = std::time::Instant::now();
        for i in 0..n_q {
            std::hint::black_box(exact_top_k(&docs, queries.row(i), 10));
        }
        t.elapsed().as_secs_f64() * 1000.0 / n_q as f64
    };
    let _ = t0;
    println!("exact brute force: {exact_ms:.3} ms/query over {} docs", docs.len());

    for &(m, efc) in &[(8usize, 100usize), (16, 200), (32, 200)] {
        let params = HnswParams { m, m0: 2 * m, ef_construction: efc, seed: 0x5EED };
        let t = std::time::Instant::now();
        let mut idx = Hnsw::build(&docs, params);
        let build_s = t.elapsed().as_secs_f64();
        let st = idx.stats();
        println!(
            "\nM={m} efC={efc}: built in {build_s:.1}s, {} layers, {} edges ({:.1} per node), \
             nodes/layer {:?}",
            st.layers, st.edges, st.edges as f64 / docs.len() as f64, st.nodes_per_layer
        );
        println!("  {:>5} {:>9} {:>11} {:>9}", "ef", "recall@10", "ms/query", "speedup");
        for &ef in &[10usize, 20, 50, 100, 200] {
            let t = std::time::Instant::now();
            let mut hits = 0usize;
            for i in 0..n_q {
                let got = idx.search(&docs, queries.row(i), 10, ef);
                let ids: Vec<u32> = got.iter().map(|x| x.0).collect();
                hits += gold[i].iter().filter(|g| ids.contains(g)).count();
            }
            let ms = t.elapsed().as_secs_f64() * 1000.0 / n_q as f64;
            println!("  {ef:>5} {:>9.3} {ms:>11.3} {:>8.1}x",
                     hits as f64 / (n_q * 10) as f64, exact_ms / ms);
        }
    }
    Ok(())
}
