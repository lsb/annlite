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
