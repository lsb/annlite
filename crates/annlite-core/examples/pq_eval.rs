//! Measure PQ quality on real embeddings: how much of the exact top-k survives
//! quantized scoring, at several code sizes and candidate-pool depths.

use annlite_core::pq::ProductQuantizer;
use annlite_core::vectors::{dot, exact_top_k, Vectors};
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let docs = Vectors::load(Path::new("data/embeddings/docs-10k.f32"), 384)?;
    let queries = Vectors::load(Path::new("data/embeddings/queries-10k.f32"), 384)?;
    println!("docs {} queries {}", docs.len(), queries.len());

    let n_q = 200.min(queries.len());
    let gold: Vec<Vec<u32>> = (0..n_q)
        .map(|i| exact_top_k(&docs, queries.row(i), 100).iter().map(|x| x.0).collect())
        .collect();

    println!("\n{:>4} {:>6} {:>8}  {:>9} {:>9} {:>9} {:>9}",
             "m", "bytes", "train_s", "R@10/10", "R@10/50", "R@10/100", "R@100/100");
    println!("{}", "-".repeat(66));

    for &m in &[16usize, 32, 64, 96] {
        let t0 = std::time::Instant::now();
        // Train on the first bulk insert, as the brief specifies.
        let pq = ProductQuantizer::train(&docs, m, 20, 0xA11CE)?;
        let train_s = t0.elapsed().as_secs_f64();
        let codes = pq.encode_all(&docs);

        let (mut r10_10, mut r10_50, mut r10_100, mut r100_100) = (0usize, 0, 0, 0);
        for qi in 0..n_q {
            let table = pq.score_table(queries.row(qi));
            let mut approx: Vec<(u32, f32)> = (0..docs.len())
                .map(|i| (i as u32, table.score(&codes[i * m..(i + 1) * m])))
                .collect();
            approx.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
            let ranked: Vec<u32> = approx.iter().map(|x| x.0).collect();
            let g10 = &gold[qi][..10];
            r10_10 += g10.iter().filter(|g| ranked[..10].contains(g)).count();
            r10_50 += g10.iter().filter(|g| ranked[..50].contains(g)).count();
            r10_100 += g10.iter().filter(|g| ranked[..100].contains(g)).count();
            r100_100 += gold[qi].iter().filter(|g| ranked[..100].contains(g)).count();
        }
        let d10 = (n_q * 10) as f64;
        let d100 = (n_q * 100) as f64;
        println!("{m:>4} {:>6} {train_s:>8.1}  {:>9.3} {:>9.3} {:>9.3} {:>9.3}",
                 m, r10_10 as f64 / d10, r10_50 as f64 / d10,
                 r10_100 as f64 / d10, r100_100 as f64 / d100);
    }

    // How much of the loss is quantization versus genuinely tied scores?
    let pq = ProductQuantizer::train(&docs, 64, 20, 0xA11CE)?;
    let codes = pq.encode_all(&docs);
    let mut abs_err = 0f64;
    let mut count = 0usize;
    for qi in 0..20 {
        let table = pq.score_table(queries.row(qi));
        for i in (0..docs.len()).step_by(37) {
            abs_err += (table.score(&codes[i * 64..(i + 1) * 64])
                - dot(queries.row(qi), docs.row(i))) as f64;
            count += 1;
        }
    }
    println!("\nmean signed ADC score error at m=64: {:.5} over {count} pairs",
             abs_err / count as f64);
    Ok(())
}
