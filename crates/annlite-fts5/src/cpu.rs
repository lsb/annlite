//! CPU time consumed by this process, in nanoseconds.
//!
//! Wall-clock latency is the wrong instrument on a machine that is running other
//! benchmarks: the figures move by more than a factor of two with background load,
//! which is why RESEARCH_LOG.md section 15.1 omits timings from the head-to-head
//! table altogether. CPU time is not immune — cache and memory-bandwidth contention
//! still inflate it — but it excludes the time the process spent descheduled, which
//! is the dominant term, so it is the more honest of the two numbers to publish
//! beside a "contended" flag.
//!
//! Summed over `/proc/self/task/*/schedstat`, whose first field is nanoseconds that
//! task spent on a CPU. Every thread is counted, not just the calling one, because a
//! parallel phase would otherwise read as nearly free: the brute-force ground truth
//! this is used to time fans out over a Rayon pool, and the main thread's own
//! schedstat misses all of it. `None` when the files are absent (not Linux, or a
//! kernel without `CONFIG_SCHEDSTATS`), in which case the caller reports no CPU
//! figure rather than a fabricated one.

/// Nanoseconds this process's threads have spent running, or `None` where the kernel
/// does not say.
pub fn process_cpu_nanos() -> Option<u64> {
    let mut total = 0u64;
    let mut seen = false;
    for entry in std::fs::read_dir("/proc/self/task").ok()? {
        let Ok(entry) = entry else { continue };
        let Ok(raw) = std::fs::read_to_string(entry.path().join("schedstat")) else { continue };
        let Some(ns) = raw.split_whitespace().next().and_then(|f| f.parse::<u64>().ok()) else {
            continue;
        };
        total += ns;
        seen = true;
    }
    seen.then_some(total)
}

/// CPU milliseconds elapsed since `start`, or `None` if either reading failed.
///
/// A thread that exits inside the window takes its total with it, so a long-lived
/// worker pool is measured exactly while a phase that spawns and joins short-lived
/// threads is undercounted. Every caller here runs against Rayon's persistent pool.
pub fn since_ms(start: Option<u64>) -> Option<f64> {
    let (a, b) = (start?, process_cpu_nanos()?);
    Some(b.saturating_sub(a) as f64 / 1e6)
}

/// Competing load observed while a measurement ran.
///
/// Whether a timing was taken under contention is a fact about the machine, not
/// something a benchmark should assert. Earlier runs hardcoded `contended: true`,
/// which was honest at the time and became wrong the moment the box went quiet —
/// and nothing in the output would have said so. This samples instead.
///
/// The signal is the runnable-task count from `/proc/loadavg`'s fourth field
/// (`running/total`). A single-threaded benchmark with the machine to itself sits at
/// 1; anything above the sampler's own footprint is somebody else's work. The
/// one-minute load average is recorded alongside it because the runnable count is
/// instantaneous and can miss a neighbour that is briefly blocked on I/O.
#[derive(Debug, Clone, Copy, Default)]
pub struct LoadWitness {
    pub peak_runnable: u32,
    pub load_1min: f64,
    samples: u32,
}

impl LoadWitness {
    pub fn new() -> Self {
        Self::default()
    }

    /// Take one observation. Call periodically through a measurement.
    pub fn sample(&mut self) {
        let Ok(raw) = std::fs::read_to_string("/proc/loadavg") else { return };
        let mut fields = raw.split_whitespace();
        if let Some(load) = fields.next().and_then(|f| f.parse::<f64>().ok()) {
            self.load_1min = self.load_1min.max(load);
        }
        // Fourth field is "running/total".
        if let Some(runnable) = fields
            .nth(2)
            .and_then(|f| f.split('/').next().and_then(|r| r.parse::<u32>().ok()))
        {
            self.peak_runnable = self.peak_runnable.max(runnable);
        }
        self.samples += 1;
    }

    /// Whether anything beyond this process was competing for CPU.
    ///
    /// Two runnable tasks is the measuring process plus the sampler's own read, so
    /// the threshold sits above that. Returns `None` when nothing was sampled, so a
    /// caller reports an unknown rather than a confident "clean".
    pub fn contended(&self) -> Option<bool> {
        (self.samples > 0).then(|| self.peak_runnable > 2 || self.load_1min > 1.5)
    }
}
