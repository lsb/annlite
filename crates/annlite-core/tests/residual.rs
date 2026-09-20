//! Residual quantization behaviour.

use annlite_core::residual::{ResidualCodec, ResidualStore};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

fn residual_sample(n: usize, seed: u64) -> Vec<f32> {
    // Residuals from a nearby centroid are sharply peaked at zero, which is the
    // property equal-mass bucketing exists to exploit.
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let u: f32 = rng.gen_range(-1.0..1.0);
            u * u * u * 0.5
        })
        .collect()
}

#[test]
fn round_trip_error_falls_as_bits_rise() {
    let sample = residual_sample(20_000, 1);
    let mut prev = f64::INFINITY;
    for bits in [1u8, 2, 4] {
        let codec = ResidualCodec::train(&sample, bits).unwrap();
        assert_eq!(codec.buckets(), 1 << bits);
        let dim = 48;
        let mut packed = vec![0u8; codec.bytes_per_token(dim)];
        let mut out = vec![0f32; dim];
        let mut err = 0f64;
        for chunk in sample.chunks_exact(dim).take(200) {
            codec.encode_into(chunk, &mut packed);
            codec.decode_into(&packed, &mut out);
            err += chunk.iter().zip(&out).map(|(a, b)| (a - b).abs() as f64).sum::<f64>();
        }
        assert!(err < prev, "error at {bits} bits ({err}) did not improve on {prev}");
        prev = err;
    }
}

#[test]
fn packing_is_exact_at_every_bit_width() {
    // 3, 5, 6 and 7 bits straddle byte boundaries; a code must survive that.
    let sample = residual_sample(8_000, 2);
    for bits in 1u8..=8 {
        let codec = ResidualCodec::train(&sample, bits).unwrap();
        let dim = 48;
        let mut packed = vec![0u8; codec.bytes_per_token(dim)];
        let mut out = vec![0f32; dim];
        let src: Vec<f32> = sample[..dim].to_vec();
        codec.encode_into(&src, &mut packed);
        codec.decode_into(&packed, &mut out);
        // Every reconstruction must be one of the codec's centers.
        for v in &out {
            assert!(
                codec.centers.iter().any(|c| (c - v).abs() < 1e-9),
                "{bits} bits produced {v}, not a center"
            );
        }
        // And re-encoding a reconstruction must be a fixed point.
        let mut again = vec![0u8; packed.len()];
        let mut out2 = vec![0f32; dim];
        codec.encode_into(&out, &mut again);
        codec.decode_into(&again, &mut out2);
        assert_eq!(out, out2, "{bits} bits is not idempotent");
    }
}

#[test]
fn bytes_per_token_matches_the_bit_budget() {
    let sample = residual_sample(4_000, 3);
    for (bits, expect) in [(1u8, 6usize), (2, 12), (4, 24), (8, 48)] {
        let codec = ResidualCodec::train(&sample, bits).unwrap();
        assert_eq!(codec.bytes_per_token(48), expect, "{bits} bits at 48 dimensions");
    }
}

#[test]
fn reconstruction_beats_the_centroid_alone() {
    // The whole point: centroid plus residual must be closer to the token than the
    // centroid is, or the extra bytes buy nothing.
    let mut rng = ChaCha8Rng::seed_from_u64(9);
    let dim = 48;
    let k = 32;
    let centroids: Vec<f32> = (0..k * dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
    let n = 2_000;
    let mut tokens = Vec::with_capacity(n * dim);
    let mut codes = Vec::with_capacity(n);
    for i in 0..n {
        let c = i % k;
        codes.push(c as u32);
        for d in 0..dim {
            tokens.push(centroids[c * dim + d] + rng.gen_range(-0.3f32..0.3));
        }
    }
    let store = ResidualStore::build(&tokens, &codes, &centroids, dim, 2, 0).unwrap();

    let mut recon = vec![0f32; dim];
    let (mut with_residual, mut centroid_only) = (0f64, 0f64);
    for i in 0..n {
        let c = codes[i] as usize;
        store.reconstruct_into(i, codes[i], &centroids, &mut recon);
        for d in 0..dim {
            let t = tokens[i * dim + d];
            with_residual += (t - recon[d]).abs() as f64;
            centroid_only += (t - centroids[c * dim + d]).abs() as f64;
        }
    }
    assert!(
        with_residual < centroid_only * 0.6,
        "residuals cut error only to {:.3} of centroid-only",
        with_residual / centroid_only
    );
}

#[test]
fn rejects_nonsense_configuration() {
    assert!(ResidualCodec::train(&[], 2).is_err());
    assert!(ResidualCodec::train(&[0.1, 0.2], 0).is_err());
    assert!(ResidualCodec::train(&[0.1, 0.2], 9).is_err());
}
