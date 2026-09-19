//! PLAID-style compression and staged retrieval on the real code corpus.
//!
//! Exact MaxSim over uncompressed token vectors is the quality ceiling and the cost
//! floor; this measures how much of the first survives the second.

use annlite_core::late::{maxsim, LateIndex, MultiVector};
use std::path::Path;

fn read_lengths(p: &Path) -> anyhow::Result<Vec<usize>> {
    let b = std::fs::read(p)?;
    Ok(b.chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as usize)
        .collect())
}

fn read_multi(p: &Path, lengths: &[usize], dim: usize) -> anyhow::Result<Vec<MultiVector>> {
    let bytes = std::fs::read(p)?;
    let all: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let mut out = Vec::with_capacity(lengths.len());
    let mut off = 0usize;
    for &n in lengths {
        out.push(MultiVector { data: all[off * dim..(off + n) * dim].to_vec(), dim });
        off += n;
    }
    anyhow::ensure!(off * dim == all.len(), "lengths do not account for every vector");
    Ok(out)
}

fn main() -> anyhow::Result<()> {
    let dim = 48;
    let d = Path::new("data/embeddings");
    let dl = read_lengths(&d.join("code-late-lengths.i32"))?;
    let ql = read_lengths(&d.join("code-late-qlengths.i32"))?;
    let docs = read_multi(&d.join("code-late.f32"), &dl, dim)?;
    let queries = read_multi(&d.join("code-late-q.f32"), &ql, dim)?;
    let total_tokens: usize = dl.iter().sum();
    println!("{} docs ({} tokens), {} queries", docs.len(), total_tokens, queries.len());

    // Ground truth: the docstring's own function, and the exact MaxSim ranking.
    let t0 = std::time::Instant::now();
    let exact_rank: Vec<usize> = (0..queries.len())
        .map(|qi| {
            let mut s: Vec<(usize, f32)> =
                (0..docs.len()).map(|i| (i, maxsim(&queries[qi], &docs[i]))).collect();
            s.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
            s.iter().position(|x| x.0 == qi).unwrap() + 1
        })
        .collect();
    let exact_ms = t0.elapsed().as_secs_f64() * 1000.0 / queries.len() as f64;
    let succ = |r: &[usize], k: usize| r.iter().filter(|&&x| x <= k).count() as f64 / r.len() as f64;
    println!(
        "exact MaxSim: succ@1 {:.3} succ@10 {:.3}  {:.1} ms/query  {} B/doc",
        succ(&exact_rank, 1), succ(&exact_rank, 10), exact_ms,
        total_tokens * dim * 4 / docs.len()
    );

    println!("\n{:>8} {:>8} {:>9} {:>10} {:>9} {:>9} {:>10} {:>9}",
             "k", "probe", "B/doc", "compress", "cand@1", "cand@10", "rerank@1", "ms");
    println!("{}", "-".repeat(80));
    for &k in &[512usize, 1024, 2048] {
        let t0 = std::time::Instant::now();
        // Centroids are fitted on a strided sample; every token is still assigned.
        let idx = LateIndex::build_sampled(&docs, k, 10, 0xC0DE, 60_000)?;
        let build_s = t0.elapsed().as_secs_f64();
        let (compressed, raw) = idx.footprint();
        for &probe in &[4usize, 16, 32] {
            let t0 = std::time::Instant::now();
            let (mut c1, mut c10, mut r1) = (0usize, 0usize, 0usize);
            for qi in 0..queries.len() {
                let cands = idx.candidates(&queries[qi], probe, 200);
                let pos = cands.iter().position(|x| x.0 as usize == qi);
                if let Some(p) = pos {
                    if p == 0 { c1 += 1; }
                    if p < 10 { c10 += 1; }
                }
                // Stage 3: exact MaxSim over the shallow pool.
                let depth = 100.min(cands.len());
                let mut rescored: Vec<(u32, f32)> = cands[..depth]
                    .iter()
                    .map(|&(dd, _)| (dd, maxsim(&queries[qi], &docs[dd as usize])))
                    .collect();
                rescored.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
                if rescored.first().map(|x| x.0 as usize) == Some(qi) { r1 += 1; }
            }
            let ms = t0.elapsed().as_secs_f64() * 1000.0 / queries.len() as f64;
            let n = queries.len() as f64;
            println!("{k:>8} {probe:>8} {:>9} {:>9.1}x {:>9.3} {:>9.3} {:>10.3} {ms:>9.1}",
                     compressed / docs.len(), raw as f64 / compressed as f64,
                     c1 as f64 / n, c10 as f64 / n, r1 as f64 / n);
        }
        println!("  (k={k} built in {build_s:.1}s)");
    }
    Ok(())
}
