//! Reproducible randomness.
//!
//! Every random choice in corpus generation flows through a ChaCha8 stream whose
//! 32-byte seed is the SHA-256 of a domain string plus a base seed. Two properties
//! matter and neither is offered by `shuf`:
//!
//! 1. ChaCha8's output is specified by the algorithm, not by a library's internals,
//!    so the same seed yields the same bytes on every platform, architecture and
//!    release. `shuf --random-source` is reproducible only within one coreutils
//!    build, and plain `shuf` additionally depends on locale collation.
//! 2. Domain separation means the permutation for round 7 is statistically
//!    independent of the one for round 8, rather than a shifted view of one stream.

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use sha2::{Digest, Sha256};

/// Derive a ChaCha8 stream from a domain label and a base seed.
pub fn stream(domain: &str, base_seed: u64) -> ChaCha8Rng {
    let mut h = Sha256::new();
    h.update(b"annlite/v1/");
    h.update(domain.as_bytes());
    h.update(b"/");
    h.update(base_seed.to_le_bytes());
    ChaCha8Rng::from_seed(h.finalize().into())
}

/// Fisher-Yates, written out rather than delegated to `SliceRandom::shuffle` so the
/// permutation stays pinned to this exact sequence of draws even if `rand` changes
/// its internal shuffle strategy across versions.
pub fn shuffle<T>(items: &mut [T], rng: &mut ChaCha8Rng) {
    for i in (1..items.len()).rev() {
        let j = below(rng, (i + 1) as u64) as usize;
        items.swap(i, j);
    }
}

/// Uniform integer in `[0, n)` by rejection sampling: unbiased, and independent of
/// `rand`'s `gen_range` implementation details.
pub fn below(rng: &mut ChaCha8Rng, n: u64) -> u64 {
    assert!(n > 0);
    let zone = u64::MAX - (u64::MAX % n);
    loop {
        let v = rand::RngCore::next_u64(rng);
        if v < zone {
            return v % n;
        }
    }
}
