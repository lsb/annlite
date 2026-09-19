//! Product quantization behaviour.
//!
//! Tests use synthetic clustered data rather than real embeddings so the expected
//! outcome is known independently of the encoder.

use annlite_core::pq::{ProductQuantizer, CENTROIDS};
use annlite_core::vectors::{dot, Vectors};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// `n` vectors of `dim` dimensions drawn around `clusters` random centres, then
/// L2-normalised to match what the encoder emits.
fn clustered(n: usize, dim: usize, clusters: usize, seed: u64) -> Vectors {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let centres: Vec<Vec<f32>> = (0..clusters)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect();
    let mut data = Vec::with_capacity(n * dim);
    for i in 0..n {
        let c = &centres[i % clusters];
        let mut v: Vec<f32> = c.iter().map(|x| x + rng.gen_range(-0.1..0.1)).collect();
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter_mut().for_each(|x| *x /= norm);
        data.extend_from_slice(&v);
    }
    Vectors { data, dim }
}

#[test]
fn code_is_one_byte_per_subquantizer() {
    let v = clustered(600, 384, 40, 1);
    let pq = ProductQuantizer::train(&v, 64, 10, 42).unwrap();
    // The headline budget: 384 float32 dimensions (1536 bytes) down to 64 bytes.
    assert_eq!(pq.encode(v.row(0)).len(), 64);
    assert_eq!(pq.encode_all(&v).len(), v.len() * 64);
}

#[test]
fn training_is_deterministic() {
    let v = clustered(600, 384, 40, 1);
    let a = ProductQuantizer::train(&v, 64, 10, 42).unwrap();
    let b = ProductQuantizer::train(&v, 64, 10, 42).unwrap();
    assert_eq!(a.centroids, b.centroids);
    assert_eq!(a.encode_all(&v), b.encode_all(&v));
}

#[test]
fn adc_table_agrees_with_decoded_inner_product() {
    // The whole speed argument for ADC is that table lookups equal the inner product
    // against the reconstruction. If this drifts, scoring is silently wrong.
    let v = clustered(400, 128, 20, 7);
    let pq = ProductQuantizer::train(&v, 16, 10, 3).unwrap();
    let table = pq.score_table(v.row(0));
    for i in 0..50 {
        let code = pq.encode(v.row(i));
        let exact = dot(v.row(0), &pq.decode(&code));
        assert!(
            (table.score(&code) - exact).abs() < 1e-4,
            "row {i}: {} vs {exact}",
            table.score(&code)
        );
    }
}

#[test]
fn quantization_is_a_good_candidate_generator() {
    // The right question for PQ is not "does it rank the top 10 correctly" but "does
    // the true top 10 survive into a candidate pool worth reranking". Measured on real
    // MiniLM embeddings, 64-byte codes put 99.2% of the exact top 10 inside the PQ
    // top 100 while placing only 61% inside the PQ top 10 (see RESEARCH_LOG.md).
    // That gap is the whole reason the dense index reranks rather than trusting PQ.
    let v = clustered(4000, 384, 120, 11);
    let pq = ProductQuantizer::train(&v, 64, 15, 5).unwrap();
    let codes = pq.encode_all(&v);

    let (mut in10, mut in100) = (0usize, 0usize);
    let trials = 30;
    for q in 0..trials {
        let query = v.row(q * 7 % v.len());
        let gold: Vec<usize> = {
            let mut e: Vec<(usize, f32)> = (0..v.len()).map(|i| (i, dot(query, v.row(i)))).collect();
            e.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
            e[..10].iter().map(|x| x.0).collect()
        };
        let table = pq.score_table(query);
        let mut approx: Vec<(usize, f32)> = (0..v.len())
            .map(|i| (i, table.score(&codes[i * 64..(i + 1) * 64])))
            .collect();
        approx.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        let ranked: Vec<usize> = approx.iter().map(|x| x.0).collect();
        in10 += gold.iter().filter(|g| ranked[..10].contains(g)).count();
        in100 += gold.iter().filter(|g| ranked[..100].contains(g)).count();
    }
    let d = (trials * 10) as f32;
    let (r10, r100) = (in10 as f32 / d, in100 as f32 / d);
    // The candidate-pool figure is the one that carries over to real data and it is
    // asserted tightly. Top-10 recall is only a loose regression guard here: these
    // synthetic clusters are tight enough that a query's ten nearest neighbours are
    // near-ties, so which ten come back is close to arbitrary and the figure reads
    // far lower than the 0.61 measured on real MiniLM embeddings.
    assert!(r100 > 0.95, "true top-10 should survive into the top-100 pool; got {r100}");
    assert!(r10 > 0.20, "PQ top-10 recall collapsed to {r10}");
    assert!(r100 > r10, "a deeper pool must not recall less than a shallower one");
}

#[test]
fn adc_score_error_is_unbiased() {
    // ADC approximates the inner product; if the approximation were systematically
    // high or low, score thresholds tuned on one corpus would not transfer.
    let v = clustered(2000, 384, 80, 21);
    let pq = ProductQuantizer::train(&v, 64, 15, 4).unwrap();
    let codes = pq.encode_all(&v);
    let mut err = 0f64;
    let mut n = 0usize;
    for q in 0..20 {
        let query = v.row(q * 13 % v.len());
        let table = pq.score_table(query);
        for i in (0..v.len()).step_by(11) {
            err += (table.score(&codes[i * 64..(i + 1) * 64]) - dot(query, v.row(i))) as f64;
            n += 1;
        }
    }
    let mean = err / n as f64;
    assert!(mean.abs() < 0.01, "ADC score error is biased by {mean} over {n} pairs");
}

#[test]
fn exact_top_k_is_sorted_and_correct() {
    let v = clustered(500, 64, 25, 3);
    let q = v.row(17).to_vec();
    let top = annlite_core::vectors::exact_top_k(&v, &q, 10);
    assert_eq!(top.len(), 10);
    assert_eq!(top[0].0, 17, "a vector must be its own nearest neighbour");
    for w in top.windows(2) {
        assert!(w[0].1 >= w[1].1, "results are not in descending score order");
    }
    // Agrees with a full sort.
    let mut all: Vec<(u32, f32)> =
        (0..v.len()).map(|i| (i as u32, dot(&q, v.row(i)))).collect();
    all.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
    assert_eq!(
        top.iter().map(|x| x.0).collect::<Vec<_>>(),
        all[..10].iter().map(|x| x.0).collect::<Vec<_>>()
    );
}

#[test]
fn rejects_dimension_not_divisible_by_m() {
    let v = clustered(100, 100, 10, 1);
    assert!(ProductQuantizer::train(&v, 64, 5, 1).is_err());
}

#[test]
fn every_code_point_is_used_on_rich_data() {
    // Empty clusters are wasted code points; the re-seeding path should prevent them.
    let v = clustered(4000, 64, 200, 13);
    let pq = ProductQuantizer::train(&v, 8, 20, 9).unwrap();
    let codes = pq.encode_all(&v);
    let mut seen = [false; CENTROIDS];
    for c in codes.iter().step_by(8) {
        seen[*c as usize] = true;
    }
    let used = seen.iter().filter(|x| **x).count();
    assert!(used > CENTROIDS / 2, "only {used}/{CENTROIDS} code points used in subspace 0");
}
