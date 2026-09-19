//! Late-interaction index behaviour, against exact MaxSim.

use annlite_core::late::{maxsim, LateIndex, MultiVector};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Documents whose tokens are drawn from a shared pool of "concepts", which is what
/// makes centroid compression possible at all: real token vectors are redundant
/// across a corpus, and a corpus of independent noise would have nothing to cluster.
fn corpus(n_docs: usize, tokens: usize, dim: usize, concepts: usize, seed: u64) -> Vec<MultiVector> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let pool: Vec<Vec<f32>> = (0..concepts)
        .map(|_| {
            let mut v: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
            let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            v.iter_mut().for_each(|x| *x /= n);
            v
        })
        .collect();
    (0..n_docs)
        .map(|_| {
            let mut data = Vec::with_capacity(tokens * dim);
            for _ in 0..tokens {
                let c = &pool[rng.gen_range(0..concepts)];
                let mut v: Vec<f32> = c.iter().map(|x| x + rng.gen_range(-0.05f32..0.05)).collect();
                let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                v.iter_mut().for_each(|x| *x /= n);
                data.extend_from_slice(&v);
            }
            MultiVector { data, dim }
        })
        .collect()
}

#[test]
fn maxsim_rewards_partial_matches_that_a_mean_would_dilute() {
    // The property that motivates late interaction: a document matching one query
    // token perfectly and ignoring the rest should beat one that matches everything
    // weakly. Averaging the document into a single vector loses that distinction.
    let dim = 4;
    let q = MultiVector { data: vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dim };
    let sharp = MultiVector { data: vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dim };
    let diffuse = MultiVector { data: vec![0.71, 0.71, 0.0, 0.0], dim };
    assert!(maxsim(&q, &sharp) > maxsim(&q, &diffuse));
    assert!((maxsim(&q, &sharp) - 2.0).abs() < 1e-5, "two perfect matches should sum to 2");
}

#[test]
fn compression_is_substantial() {
    let docs = corpus(400, 40, 48, 60, 1);
    let idx = LateIndex::build(&docs, 128, 10, 3).unwrap();
    let (compressed, raw) = idx.footprint();
    let ratio = raw as f64 / compressed as f64;
    // Storing a centroid id instead of a 48-d float32 vector is 4 bytes instead of
    // 192, less the shared centroid table.
    assert!(ratio > 10.0, "compression ratio was only {ratio:.1}x");
}

#[test]
fn candidate_generation_recalls_the_true_best() {
    let docs = corpus(500, 40, 48, 60, 5);
    let idx = LateIndex::build(&docs, 128, 15, 7).unwrap();
    let queries = corpus(30, 8, 48, 60, 99);

    let mut hits = 0usize;
    for q in &queries {
        let mut exact: Vec<(u32, f32)> =
            (0..docs.len()).map(|i| (i as u32, maxsim(q, &docs[i]))).collect();
        exact.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        let gold: Vec<u32> = exact[..10].iter().map(|x| x.0).collect();
        let cands: Vec<u32> = idx.candidates(q, 8, 100).iter().map(|x| x.0).collect();
        hits += gold.iter().filter(|g| cands.contains(g)).count();
    }
    let recall = hits as f64 / (queries.len() * 10) as f64;
    assert!(recall > 0.75, "centroid stage recalled only {recall:.3} of the true top 10");
}

#[test]
fn reranking_recovers_exact_ranking() {
    // Stage 3 exists because centroid-only scoring ranks approximately. After
    // rescoring the candidate pool exactly, the top result should be the true one.
    let docs = corpus(300, 40, 48, 50, 11);
    let idx = LateIndex::build(&docs, 96, 15, 13).unwrap();
    let queries = corpus(20, 8, 48, 50, 131);

    let mut top1 = 0usize;
    for q in &queries {
        let mut exact: Vec<(u32, f32)> =
            (0..docs.len()).map(|i| (i as u32, maxsim(q, &docs[i]))).collect();
        exact.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        let got = idx.search(&docs, q, 10, 8, 100, 100);
        if got[0].0 == exact[0].0 {
            top1 += 1;
        }
    }
    let acc = top1 as f64 / queries.len() as f64;
    assert!(acc > 0.8, "exact rerank put the true best first only {acc:.2} of the time");
}

#[test]
fn more_probes_never_shrink_the_candidate_pool() {
    let docs = corpus(200, 30, 32, 40, 17);
    let idx = LateIndex::build(&docs, 64, 10, 19).unwrap();
    let q = &corpus(1, 6, 32, 40, 23)[0];
    let narrow = idx.candidates(q, 1, 1000).len();
    let wide = idx.candidates(q, 16, 1000).len();
    assert!(wide >= narrow, "probing more centroids returned fewer candidates: {wide} < {narrow}");
}

#[test]
fn rejects_mismatched_dimensions() {
    let a = MultiVector { data: vec![1.0, 0.0], dim: 2 };
    let b = MultiVector { data: vec![1.0, 0.0, 0.0], dim: 3 };
    assert!(LateIndex::build(&[a, b], 4, 5, 1).is_err());
}
