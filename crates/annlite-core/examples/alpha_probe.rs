//! What does Vamana's alpha actually buy? Measure hops-to-recall, not edge length.
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

fn main() {
    let v = clustered(4000, 64, 40, 13);
    let q = clustered(100, 64, 40, 131);
    let gold: Vec<Vec<u32>> = (0..q.len())
        .map(|i| exact_top_k(&v, q.row(i), 10).iter().map(|x| x.0).collect())
        .collect();

    println!("{:>7} {:>8} {:>12} {:>10} {:>10} {:>10}",
             "alpha", "edges", "mean_edge_d", "L=20 rec", "L=64 rec", "L=64 hops");
    for &alpha in &[1.0f32, 1.1, 1.2, 1.4, 1.6, 2.0] {
        let idx = Vamana::build(&v, VamanaParams { r: 32, l_build: 64, alpha, seed: 11 });
        let st = idx.stats();
        let mut total_d = 0f64;
        let mut cnt = 0usize;
        for i in 0..v.len() as u32 {
            for &nb in idx.neighbors(i) {
                total_d += (1.0 - dot(v.row(i as usize), v.row(nb as usize))) as f64;
                cnt += 1;
            }
        }
        let mut rec = [0f64; 2];
        let mut hops = 0usize;
        for (j, &l) in [20usize, 64].iter().enumerate() {
            let mut hits = 0usize;
            for qi in 0..q.len() {
                let (best, visited) = idx.greedy_search(&v, q.row(qi), 10, l);
                let ids: Vec<u32> = best.iter().map(|x| x.0).collect();
                hits += gold[qi].iter().filter(|g| ids.contains(g)).count();
                if j == 1 { hops += visited.len(); }
            }
            rec[j] = hits as f64 / (q.len() * 10) as f64;
        }
        println!("{alpha:>7.1} {:>8} {:>12.4} {:>10.3} {:>10.3} {:>10.1}",
                 st.edges, total_d / cnt as f64, rec[0], rec[1], hops as f64 / q.len() as f64);
    }
}
