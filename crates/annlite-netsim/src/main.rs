//! Command line front end. Every profile parameter is individually overridable,
//! because the profiles are starting points for measurement, not claims about how
//! all LTE behaves — a result that only holds at exactly 70 ms is worth knowing.

use annlite_netsim::profile::{self, Mode, Sim};
use annlite_netsim::server::{Config, Server};
use anyhow::{Context, Result};
use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "annlite-netsim",
    about = "HTTP byte-range server with reproducible latency and throughput simulation"
)]
struct Cli {
    /// Directory to serve. Nothing outside it is reachable.
    #[arg(long, default_value = ".")]
    root: PathBuf,

    #[arg(long, default_value = "127.0.0.1:8080")]
    addr: String,

    /// Named link profile: ideal, wifi, 5g, lte (4g), leo, 3g, slow-3g, satellite.
    #[arg(long, default_value = "lte")]
    profile: String,

    /// `random` draws latency and rate per request; `fixed` applies the profile's
    /// numbers exactly. Both are reproducible; `fixed` removes variance entirely.
    #[arg(long, default_value = "random")]
    mode: Mode,

    /// Seed for the per-request draws. The same seed and the same request sequence
    /// give the same delays, on any machine.
    #[arg(long, default_value_t = 0)]
    seed: u64,

    /// Override the profile's median first-byte delay, milliseconds.
    #[arg(long)]
    latency_ms: Option<f64>,

    /// Override the log-scale sigma of the latency draw. 0 makes latency constant
    /// while leaving bandwidth variable.
    #[arg(long)]
    latency_sigma: Option<f64>,

    /// Override the probability of a tail spike, 0..1.
    #[arg(long)]
    spike_prob: Option<f64>,

    /// Override the mean size of a tail spike, milliseconds.
    #[arg(long)]
    spike_ms: Option<f64>,

    /// Override downlink in megabits per second (decimal, as vendors quote it).
    /// 0 means unlimited.
    #[arg(long, conflicts_with = "bandwidth_bps")]
    bandwidth_mbps: Option<f64>,

    /// Override downlink in bytes per second, for when the number that matters is
    /// "how long does a 4 KiB page take".
    #[arg(long)]
    bandwidth_bps: Option<f64>,

    /// Override the log-scale sigma of the per-request rate draw.
    #[arg(long)]
    bandwidth_sigma: Option<f64>,

    /// Worker threads; also the number of requests that can be in flight.
    #[arg(long, default_value_t = 4)]
    threads: usize,

    /// Bytes written between pacing sleeps. Below 1024 tiny_http's write buffer
    /// coalesces chunks and the shaping stops reaching the socket.
    #[arg(long, default_value_t = 8192)]
    chunk_bytes: usize,

    /// JSONL request log; `-` writes to stdout. Without it requests are counted
    /// but not recorded.
    #[arg(long)]
    log: Option<PathBuf>,

    /// Value for the `Cache-Control` response header. The default keeps a browser
    /// cache from hiding repeat fetches from the measurement.
    #[arg(long, default_value = "no-store")]
    cache_control: String,

    /// Print the profile table and exit.
    #[arg(long)]
    list_profiles: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    if cli.list_profiles {
        println!("{:<10} {:>9} {:>6} {:>7} {:>8} {:>9} {:>6}", "profile", "rtt_ms", "sigma", "spike_p", "spike_ms", "mbps", "bw_sig");
        for p in profile::ALL {
            println!(
                "{:<10} {:>9.0} {:>6.2} {:>7.3} {:>8.0} {:>9.1} {:>6.2}",
                p.name, p.rtt_ms, p.rtt_sigma, p.spike_prob, p.spike_ms, p.mbps, p.bw_sigma
            );
        }
        return Ok(());
    }

    let base = profile::by_name(&cli.profile)
        .with_context(|| format!("unknown profile {:?}; try --list-profiles", cli.profile))?;
    let mut sim = Sim::from_profile(base, cli.mode, cli.seed);
    if let Some(v) = cli.latency_ms {
        sim.latency_ms = v;
    }
    if let Some(v) = cli.latency_sigma {
        sim.latency_sigma = v;
    }
    if let Some(v) = cli.spike_prob {
        sim.spike_prob = v;
    }
    if let Some(v) = cli.spike_ms {
        sim.spike_ms = v;
    }
    if let Some(v) = cli.bandwidth_mbps {
        sim.bytes_per_sec = (v > 0.0).then(|| v * 1e6 / 8.0);
    }
    if let Some(v) = cli.bandwidth_bps {
        sim.bytes_per_sec = (v > 0.0).then_some(v);
    }
    if let Some(v) = cli.bandwidth_sigma {
        sim.bandwidth_sigma = v;
    }

    let mut cfg = Config::new(cli.root, sim);
    cfg.addr = cli.addr;
    cfg.threads = cli.threads;
    cfg.chunk_bytes = cli.chunk_bytes;
    cfg.log = cli.log;
    cfg.cache_control = cli.cache_control;

    let server = Server::bind(cfg.clone())?;
    let rate = match sim.bytes_per_sec {
        Some(b) => format!("{:.2} Mbit/s", b * 8.0 / 1e6),
        None => "unlimited".to_string(),
    };
    eprintln!(
        "annlite-netsim: http://{} root={} profile={} mode={:?} seed={}\n  \
         latency {:.0} ms (sigma {:.2}, spike {:.1}% x {:.0} ms mean), downlink {} (sigma {:.2})",
        server.local_addr(),
        cfg.root.display(),
        sim.profile_name,
        sim.mode,
        sim.seed,
        sim.latency_ms,
        sim.latency_sigma,
        sim.spike_prob * 100.0,
        sim.spike_ms,
        rate,
        sim.bandwidth_sigma,
    );
    server.run()
}
