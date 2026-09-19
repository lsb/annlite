//! Per-request randomness that survives replay.
//!
//! A benchmark that reports "12 round-trips, 340 ms" is worthless if rerunning it
//! draws different delays, so the simulated network must be a pure function of the
//! request sequence. That rules out thread-local or time-seeded generators: the
//! server hands requests to a pool of worker threads, and any generator shared
//! across them would make the delay depend on thread scheduling rather than on the
//! workload.
//!
//! Instead each request gets its own ChaCha8 stream, seeded by
//! `SHA-256("annlite/netsim/v1/" ‖ domain ‖ seed ‖ ordinal)`. Two consequences:
//!
//! 1. Request *n* draws the same numbers no matter when it is served, by which
//!    thread, or whether requests 0..n were served at all — so a client that
//!    replays the same sequence sees the same simulated network.
//! 2. Domain separation keeps the latency draw for a request statistically
//!    independent of its bandwidth draw, rather than one being a shifted view of
//!    the other stream.
//!
//! This mirrors `annlite-corpus`'s RNG discipline; the draws below are written out
//! rather than taken from `rand`'s distribution helpers so the numbers stay pinned
//! if `rand` changes its internals.

use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use sha2::{Digest, Sha256};

/// The stream for one (domain, seed, request ordinal) triple.
pub fn stream(domain: &str, seed: u64, ordinal: u64) -> ChaCha8Rng {
    let mut h = Sha256::new();
    h.update(b"annlite/netsim/v1/");
    h.update(domain.as_bytes());
    h.update(b"/");
    h.update(seed.to_le_bytes());
    h.update(ordinal.to_le_bytes());
    ChaCha8Rng::from_seed(h.finalize().into())
}

/// Uniform in `[0, 1)` with 53 bits of resolution, the most an `f64` can hold
/// without rounding to exactly 1.0.
pub fn uniform01(rng: &mut ChaCha8Rng) -> f64 {
    (rng.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
}

/// Standard normal by Box-Muller. Only one of the two variates it produces is
/// used: keeping the spare would make a draw depend on how many draws preceded it
/// within the request, which is exactly the coupling this module avoids.
pub fn standard_normal(rng: &mut ChaCha8Rng) -> f64 {
    // u1 is shifted off zero because ln(0) is -inf.
    let u1 = uniform01(rng).max(f64::MIN_POSITIVE);
    let u2 = uniform01(rng);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// Lognormal with the given median and log-scale `sigma`.
///
/// Parameterised by the *median* rather than the mean so that the profile tables
/// can quote a familiar "typical RTT" number and have it mean what a reader
/// expects: half the requests land below it. The mean of the draw is
/// `median · exp(σ²/2)`, a little higher — that right skew is the point.
pub fn lognormal(rng: &mut ChaCha8Rng, median: f64, sigma: f64) -> f64 {
    if sigma <= 0.0 {
        return median;
    }
    median * (sigma * standard_normal(rng)).exp()
}

/// Exponential with the given mean, by inverse transform.
pub fn exponential(rng: &mut ChaCha8Rng, mean: f64) -> f64 {
    let u = uniform01(rng).max(f64::MIN_POSITIVE);
    -mean * u.ln()
}
