//! Is low HNSW recall a bug in the implementation, or a property of the data?
//! Runs the same index over synthetic data with known cluster structure and over
//! the real embeddings, and reports the intrinsic difficulty of each.

use annlite_core::hnsw::{Hnsw, HnswParams};
use annlite_core::vectors::{dot, exact_top_k, Vectors};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::path::Path;

fn clustered(n: usize, dim: usize, clusters: usize, spread: f32, seed: u64) -> Vectors {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let centres: Vec<Vec<f32>> = (0..clusters)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect())
        .collect();
    let mut data = Vec::with_capacity(n * dim);
    for i in 0..n {
        let c = &centres[i % clusters];
        let mut v: Vec<f32> = c.iter().map(|x| x + rng.gen_range(-spread..spread)).collect();
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter_mut().for_each(|x| *x /= norm);
        data.extend_from_slice(&v);
    }
    Vectors { data, dim }
}

/// Relative contrast: mean similarity gap between the 10th neighbour and a random
/// point, scaled by the spread of all similarities. Low contrast means "nearest" is
/// barely distinguishable from "typical", which is the regime where every
/// graph-based index degrades regardless of implementation.
fn contrast(v: &Vectors, trials: usize) -> (f64, f64, f64) {
    let mut near = 0f64;
    let mut mean = 0f64;
    let mut sd = 0f64;
    for t in 0..trials {
        let q = v.row((t * 97) % v.len());
        let mut sims: Vec<f32> = (0..v.len()).map(|i| dot(q, v.row(i))).collect();
        sims.sort_unstable_by(|a, b| b.total_cmp(a));
        near += sims[10] as f64;
        let m = sims.iter().map(|&x| x as f64).sum::<f64>() / sims.len() as f64;
        mean += m;
        sd += (sims.iter().map(|&x| (x as f64 - m).powi(2)).sum::<f64>() / sims.len() as f64).sqrt();
    }
    (near / trials as f64, mean / trials as f64, sd / trials as f64)
}

fn eval(name: &str, v: &Vectors, queries: &Vectors, n_q: usize) {
    let (near, mean, sd) = contrast(v, 20);
    println!(
        "\n=== {name} ({} docs, {}d) ===\n  10th-NN sim {near:.3}, mean sim {mean:.3}, sd {sd:.3} \
         -> contrast (near-mean)/sd = {:.2}",
        v.len(), v.dim, (near - mean) / sd
    );
    let gold: Vec<Vec<u32>> = (0..n_q)
        .map(|i| exact_top_k(v, queries.row(i), 10).iter().map(|x| x.0).collect())
        .collect();
    let mut idx = Hnsw::build(v, HnswParams { m: 16, m0: 32, ef_construction: 200, seed: 7 });
    print!("  recall@10:");
    for &ef in &[10usize, 50, 100, 200] {
        let mut hits = 0usize;
        for i in 0..n_q {
            let ids: Vec<u32> = idx.search(v, queries.row(i), 10, ef).iter().map(|x| x.0).collect();
            hits += gold[i].iter().filter(|g| ids.contains(g)).count();
        }
        print!("  ef={ef} {:.3}", hits as f64 / (n_q * 10) as f64);
    }
    println!();
}

fn main() -> anyhow::Result<()> {
    let n = 10000;
    // Tight clusters: the easy case any correct HNSW must ace.
    let tight = clustered(n, 384, 100, 0.15, 1);
    // Held-out queries drawn from the same generator with a different seed. Using
    // corpus members as queries would rig the comparison: greedy descent finds a
    // point identical to the query almost for free, which is not the task the real
    // query set poses.
    let tq = clustered(200, 384, 100, 0.15, 99);
    eval("synthetic, tight clusters", &tight, &tq, 200);

    // Near-uniform on the sphere: the hard case, no structure to exploit.
    let mut rng = ChaCha8Rng::seed_from_u64(2);
    let mut data = Vec::with_capacity(n * 384);
    for _ in 0..n {
        let mut v: Vec<f32> = (0..384).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter_mut().for_each(|x| *x /= norm);
        data.extend_from_slice(&v);
    }
    let uni = Vectors { data, dim: 384 };
    let mut qdata = Vec::with_capacity(200 * 384);
    for _ in 0..200 {
        let mut v: Vec<f32> = (0..384).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter_mut().for_each(|x| *x /= norm);
        qdata.extend_from_slice(&v);
    }
    let uq = Vectors { data: qdata, dim: 384 };
    eval("synthetic, uniform on sphere", &uni, &uq, 200);

    let docs = Vectors::load(Path::new("data/embeddings/docs-10k.f32"), 384)?;
    let queries = Vectors::load(Path::new("data/embeddings/queries-10k.f32"), 384)?;
    eval("real MiniLM embeddings of word bags", &docs, &queries, 200);
    Ok(())
}
