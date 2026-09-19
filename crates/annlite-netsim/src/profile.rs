//! Named link profiles and the delay model they parameterise.
//!
//! The question this crate exists to answer is "how long would my phone take to
//! fetch these SQLite pages", so the numbers below have to be defensible rather
//! than merely plausible. Each profile quotes a *median* round-trip time and a
//! sustained downlink rate, with the reasoning in its doc comment. Every value is
//! overridable on the command line; the profile is a starting point, not a claim
//! that all LTE looks alike.
//!
//! # The delay model
//!
//! A response costs `latency + bytes / rate`:
//!
//! * **Latency** is charged once, before the first body byte, and stands for the
//!   whole request round trip — DNS and TLS are assumed warm, as they are for the
//!   second and subsequent range requests of a page-faulting SQLite reader, which
//!   is where essentially all the round-trips are.
//! * **Rate** shapes the body (see [`crate::server`]), so a 4 KiB SQLite page and a
//!   400 KiB index segment differ in the way they really differ.
//!
//! In `Mode::Random` the latency is a lognormal draw around the profile median.
//! Lognormal because measured cell RTT is right-skewed: it cannot go below the
//! radio's scheduling floor but can stretch arbitrarily far above it, and its log
//! is near-normal. A pure lognormal still understates what a cell link actually
//! does, though — it produces no multi-hundred-millisecond stalls, and those are
//! precisely what ruins an index that needs twelve dependent round-trips. So the
//! model adds an explicit spike component: with probability `spike_prob`, an extra
//! exponential delay of mean `spike_ms` on top of the lognormal body. The two
//! parameters are separate because their causes are separate — the lognormal body
//! is ordinary scheduling jitter, the spike is a retransmission, a handover or a
//! transition out of an idle radio state.

use serde::Serialize;
use std::str::FromStr;
use std::time::Duration;

/// How the per-request draws behave.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    /// Every request gets exactly the configured latency and rate. The control
    /// condition for A/B work: any difference between two runs is the index, not
    /// the network.
    Fixed,
    /// Latency and rate are drawn per request. Still exactly reproducible — see
    /// [`crate::rng`] — but with the variance that makes tail behaviour visible.
    Random,
}

impl FromStr for Mode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "fixed" | "deterministic" => Ok(Mode::Fixed),
            "random" | "probabilistic" => Ok(Mode::Random),
            other => Err(format!("unknown mode {other:?} (expected fixed or random)")),
        }
    }
}

/// One named link, in the units the underlying sources quote.
#[derive(Clone, Copy, Debug)]
pub struct Profile {
    pub name: &'static str,
    /// Median round-trip time in milliseconds.
    pub rtt_ms: f64,
    /// Log-scale sigma of the lognormal RTT body. 0.2 is a wired-ish link whose
    /// 95th percentile sits ~1.4x its median; 0.5 is a radio link where it sits
    /// ~2.3x.
    pub rtt_sigma: f64,
    /// Probability that a request also draws a tail spike.
    pub spike_prob: f64,
    /// Mean of the exponential spike, in milliseconds.
    pub spike_ms: f64,
    /// Sustained downlink, megabits per second. 0 means unlimited.
    pub mbps: f64,
    /// Log-scale sigma of the per-request rate draw. Bandwidth varies more slowly
    /// than latency in reality (it tracks cell load and signal, not individual
    /// packet scheduling), so these are smaller than `rtt_sigma`.
    pub bw_sigma: f64,
}

/// The control condition: no latency, no shaping. Everything is measured against
/// this, so that "this index needs 40 round-trips" and "40 round-trips cost 2.8 s
/// on LTE" stay separable findings.
pub const IDEAL: Profile = Profile {
    name: "ideal",
    rtt_ms: 0.0,
    rtt_sigma: 0.0,
    spike_prob: 0.0,
    spike_ms: 0.0,
    mbps: 0.0,
    bw_sigma: 0.0,
};

/// Home Wi-Fi to a nearby CDN edge: a couple of milliseconds of 802.11 access plus
/// a short terrestrial hop, which is where a warm HTTP request to a well-peered
/// CDN typically lands. Rate is a conservative single-client share of a domestic
/// connection rather than the link's headline figure.
pub const WIFI: Profile = Profile {
    name: "wifi",
    rtt_ms: 15.0,
    rtt_sigma: 0.22,
    spike_prob: 0.005,
    spike_ms: 120.0,
    mbps: 50.0,
    bw_sigma: 0.15,
};

/// 5G NR mid-band (n78-style). Sub-frame scheduling cuts radio latency well below
/// LTE's, and the downlink is fast enough that for SQLite-page-sized bodies the
/// round-trip dominates completely — which is the interesting regime.
pub const FIVE_G: Profile = Profile {
    name: "5g",
    rtt_ms: 35.0,
    rtt_sigma: 0.35,
    spike_prob: 0.015,
    spike_ms: 250.0,
    mbps: 100.0,
    bw_sigma: 0.35,
};

/// LTE / 4G, the default assumption for "on a phone, outdoors, decent signal".
/// LTE's radio access network adds roughly 40-60 ms over the wired path, and
/// national-operator medians for 4G downlink sit in the 10-25 Mbit/s band; 70 ms
/// and 15 Mbit/s are the middle of both.
pub const LTE: Profile = Profile {
    name: "lte",
    rtt_ms: 70.0,
    rtt_sigma: 0.45,
    spike_prob: 0.03,
    spike_ms: 400.0,
    mbps: 15.0,
    bw_sigma: 0.45,
};

/// 3G / HSPA+. The rate matches Chrome DevTools' "Fast 3G" preset (1.6 Mbit/s
/// down), which is the figure most web-performance work is calibrated against.
/// The RTT does not: DevTools charges 562 ms, deliberately pessimistic, while
/// measured HSPA+ round-trips cluster nearer 150-250 ms. 200 ms is the honest
/// number; use `--latency-ms 562` to reproduce DevTools exactly.
pub const THREE_G: Profile = Profile {
    name: "3g",
    rtt_ms: 200.0,
    rtt_sigma: 0.55,
    spike_prob: 0.06,
    spike_ms: 800.0,
    mbps: 1.6,
    bw_sigma: 0.5,
};

/// Chrome DevTools' "Slow 3G" preset: 400 kbit/s at 2000 ms RTT. Not a measurement
/// of any real network — it is the industry's agreed worst case, useful as the
/// upper bound on what a round-trip-hungry index would cost.
pub const SLOW_3G: Profile = Profile {
    name: "slow-3g",
    rtt_ms: 2000.0,
    rtt_sigma: 0.4,
    spike_prob: 0.08,
    spike_ms: 1500.0,
    mbps: 0.4,
    bw_sigma: 0.5,
};

/// Geostationary satellite. The latency here is physics, not engineering: a GEO
/// arc sits 35,786 km up, so one hop up-and-down is ~239 ms at c and a round trip
/// is ~477 ms before any processing. 600 ms accounts for the terrestrial tail and
/// modem framing. Rate is a modern Ka-band consumer service. This is the profile
/// that punishes dependent round-trips hardest, which makes it the sharpest test
/// of whether an index's accesses can be batched.
pub const SATELLITE: Profile = Profile {
    name: "satellite",
    rtt_ms: 600.0,
    rtt_sigma: 0.25,
    spike_prob: 0.04,
    spike_ms: 900.0,
    mbps: 20.0,
    bw_sigma: 0.4,
};

/// Low-earth-orbit satellite (Starlink-class). ~550 km altitude makes propagation
/// almost irrelevant (~4 ms round trip); the 45 ms is the ground network plus the
/// constellation's periodic satellite hand-offs, which also drive the higher spike
/// probability — LEO stalls come from reconfiguration, not distance.
pub const LEO: Profile = Profile {
    name: "leo",
    rtt_ms: 45.0,
    rtt_sigma: 0.35,
    spike_prob: 0.05,
    spike_ms: 350.0,
    mbps: 80.0,
    bw_sigma: 0.45,
};

/// All profiles, in the order `--list-profiles` prints them.
pub const ALL: &[Profile] = &[IDEAL, WIFI, FIVE_G, LTE, LEO, THREE_G, SLOW_3G, SATELLITE];

/// Look up a profile by name. `4g` is accepted as a synonym for `lte` and `wifi`
/// for `wi-fi`, because both spellings turn up in scripts.
pub fn by_name(name: &str) -> Option<Profile> {
    let key = name.trim().to_ascii_lowercase();
    let key = match key.as_str() {
        "4g" => "lte",
        "wi-fi" | "wlan" => "wifi",
        "geo" => "satellite",
        "starlink" => "leo",
        "none" | "localhost" => "ideal",
        other => other,
    };
    ALL.iter().copied().find(|p| p.name == key)
}

/// A fully resolved simulation: a profile with any command-line overrides applied.
#[derive(Clone, Copy, Debug)]
pub struct Sim {
    pub profile_name: &'static str,
    pub mode: Mode,
    /// Median (or, in `Fixed` mode, exact) first-byte delay, milliseconds.
    pub latency_ms: f64,
    pub latency_sigma: f64,
    pub spike_prob: f64,
    pub spike_ms: f64,
    /// Body shaping rate in **bytes** per second; `None` is unlimited.
    pub bytes_per_sec: Option<f64>,
    pub bandwidth_sigma: f64,
    pub seed: u64,
}

impl Sim {
    pub fn from_profile(p: Profile, mode: Mode, seed: u64) -> Sim {
        Sim {
            profile_name: p.name,
            mode,
            latency_ms: p.rtt_ms,
            latency_sigma: p.rtt_sigma,
            spike_prob: p.spike_prob,
            spike_ms: p.spike_ms,
            // Mbit/s to byte/s: 1e6 bits, 8 bits to the byte. Network vendors quote
            // decimal megabits, not mebibits, so this is 1e6 and not 1<<20.
            bytes_per_sec: (p.mbps > 0.0).then(|| p.mbps * 1e6 / 8.0),
            bandwidth_sigma: p.bw_sigma,
            seed,
        }
    }

    /// The first-byte delay for request `ordinal`.
    ///
    /// Depends only on `(seed, ordinal)`, never on wall-clock time or thread
    /// identity, so replaying a request sequence replays its delays exactly.
    pub fn latency(&self, ordinal: u64) -> Duration {
        if self.latency_ms <= 0.0 {
            return Duration::ZERO;
        }
        let ms = match self.mode {
            Mode::Fixed => self.latency_ms,
            Mode::Random => {
                let mut r = crate::rng::stream("latency", self.seed, ordinal);
                let body = crate::rng::lognormal(&mut r, self.latency_ms, self.latency_sigma);
                // The spike coin is always flipped, even when spike_prob is 0, so
                // that changing spike_prob alone does not shift the body draws of
                // every later request.
                let coin = crate::rng::uniform01(&mut r);
                let spike = crate::rng::exponential(&mut r, self.spike_ms.max(0.0));
                if self.spike_prob > 0.0 && coin < self.spike_prob {
                    body + spike
                } else {
                    body
                }
            }
        };
        Duration::from_secs_f64((ms / 1000.0).max(0.0))
    }

    /// The body rate for request `ordinal`, in bytes per second.
    ///
    /// One draw per request rather than per chunk: within the few tens of
    /// milliseconds a page fetch lasts, the bottleneck's capacity is effectively
    /// constant, and per-chunk redraws would average out to the mean and hide
    /// exactly the slow-response tail this is meant to expose.
    pub fn rate(&self, ordinal: u64) -> Option<f64> {
        let base = self.bytes_per_sec?;
        match self.mode {
            Mode::Fixed => Some(base),
            Mode::Random => {
                let mut r = crate::rng::stream("bandwidth", self.seed, ordinal);
                // Floor at a tenth of nominal: a lognormal tail can otherwise draw
                // a rate near zero and hang a benchmark for minutes.
                Some(crate::rng::lognormal(&mut r, base, self.bandwidth_sigma).max(base * 0.1))
            }
        }
    }
}
