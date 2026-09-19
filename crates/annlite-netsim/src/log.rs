//! One JSONL line per request.
//!
//! Attributing round-trips and bytes to individual queries is the entire point of
//! this crate: the headline result of the project is meant to be "query X costs N
//! round-trips and B bytes", and that can only be measured at the server, because
//! the browser cannot see what its own VFS coalesced or cached.
//!
//! Two things make a log span attributable to one query. A **label**, which the
//! benchmark sets either out of band (`/__netsim/label`) or per request
//! (`X-Netsim-Label`), and a **reset**, which returns the ordinal counter to zero.
//! Resetting is not just bookkeeping: the ordinal is the index into the seeded
//! delay stream, so resetting before each query makes every query see the *same*
//! sequence of simulated delays. Two indexes compared that way differ because of
//! their access patterns, not because one happened to draw a spike.

use anyhow::{Context, Result};
use serde::Serialize;
use std::fs::{File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

/// One served request. Field order here is the field order in the JSONL, which is
/// stable so that `jq -c` output and diffs between runs line up.
#[derive(Debug, Clone, Serialize)]
pub struct Record {
    /// Position in the request sequence since the last reset. This is also the
    /// index into the seeded delay stream, so it identifies the draw exactly.
    pub ordinal: u64,
    /// Milliseconds since the last reset, taken when the request was accepted.
    pub t_ms: f64,
    pub label: String,
    pub method: String,
    pub path: String,
    /// The `Range` header verbatim, or null if the client sent none.
    pub range: Option<String>,
    pub status: u16,
    /// First byte served, and how many. For a `HEAD` this is what a `GET` would
    /// have sent; `transfer_ms` is 0 because no body crossed the wire.
    pub offset: u64,
    pub length: u64,
    /// Length of the whole file, so a reader can compute what fraction was fetched.
    pub total: u64,
    /// Simulated first-byte delay actually slept.
    pub latency_ms: f64,
    /// Simulated body transfer time, `length / rate`.
    pub transfer_ms: f64,
    /// Real wall clock from accepting the request to finishing the response. It
    /// should track `latency_ms + transfer_ms` closely; a persistent gap means the
    /// host cannot keep up with the simulation and the numbers are suspect.
    pub service_ms: f64,
    pub profile: &'static str,
    pub mode: crate::profile::Mode,
}

/// Totals since the last reset.
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct Stats {
    /// Round-trips. The number the project is ultimately trying to reduce.
    pub requests: u64,
    pub bytes: u64,
    pub latency_ms: f64,
    pub transfer_ms: f64,
    pub service_ms: f64,
    /// Wall clock since the reset, which for a sequence of dependent fetches is
    /// the number a user would feel.
    pub elapsed_ms: f64,
}

enum Sink {
    Discard,
    Stdout,
    File(File),
}

struct Inner {
    sink: Sink,
    label: String,
    stats: Stats,
    epoch: Instant,
}

pub struct RequestLog {
    ordinal: AtomicU64,
    inner: Mutex<Inner>,
}

impl RequestLog {
    /// `path` of `-` writes to stdout; `None` discards records but still counts
    /// them, so `/__netsim/stats` works even with logging off.
    pub fn open(path: Option<&Path>) -> Result<Self> {
        let sink = match path {
            None => Sink::Discard,
            Some(p) if p.as_os_str() == "-" => Sink::Stdout,
            Some(p) => Sink::File(
                OpenOptions::new()
                    .create(true)
                    .write(true)
                    .truncate(true)
                    .open(p)
                    .with_context(|| format!("opening request log {}", p.display()))?,
            ),
        };
        Ok(RequestLog {
            ordinal: AtomicU64::new(0),
            inner: Mutex::new(Inner {
                sink,
                label: String::new(),
                stats: Stats::default(),
                epoch: Instant::now(),
            }),
        })
    }

    /// Claim the next ordinal. Called once per request from the accept loop, so
    /// ordinals follow arrival order even though the work happens on a pool.
    pub fn next_ordinal(&self) -> u64 {
        self.ordinal.fetch_add(1, Ordering::SeqCst)
    }

    pub fn elapsed_ms(&self) -> f64 {
        self.inner.lock().unwrap().epoch.elapsed().as_secs_f64() * 1e3
    }

    pub fn label(&self) -> String {
        self.inner.lock().unwrap().label.clone()
    }

    pub fn set_label(&self, label: &str) {
        self.inner.lock().unwrap().label = label.to_string();
    }

    /// Start a new attribution span: zero the ordinal (and so the delay stream),
    /// zero the totals, restart the clock, and optionally relabel. Returns the
    /// totals of the span that just ended, which is what a benchmark harness wants
    /// to record.
    pub fn reset(&self, label: Option<&str>) -> Stats {
        let mut inner = self.inner.lock().unwrap();
        let previous = Stats {
            elapsed_ms: inner.epoch.elapsed().as_secs_f64() * 1e3,
            ..inner.stats
        };
        inner.stats = Stats::default();
        inner.epoch = Instant::now();
        if let Some(l) = label {
            inner.label = l.to_string();
        }
        // After the totals are swapped, so a concurrent request cannot be counted
        // in the old span but numbered in the new one.
        self.ordinal.store(0, Ordering::SeqCst);
        previous
    }

    pub fn stats(&self) -> Stats {
        let inner = self.inner.lock().unwrap();
        Stats {
            elapsed_ms: inner.epoch.elapsed().as_secs_f64() * 1e3,
            ..inner.stats
        }
    }

    /// Append one record. Each line is flushed as it is written: a benchmark that
    /// reads the log between queries, or one that is killed mid-run, must still see
    /// every request that was actually served.
    pub fn write(&self, rec: &Record) {
        let mut inner = self.inner.lock().unwrap();
        inner.stats.requests += 1;
        inner.stats.bytes += rec.length;
        inner.stats.latency_ms += rec.latency_ms;
        inner.stats.transfer_ms += rec.transfer_ms;
        inner.stats.service_ms += rec.service_ms;

        let line = match serde_json::to_string(rec) {
            Ok(s) => s,
            Err(_) => return,
        };
        let _ = match &mut inner.sink {
            Sink::Discard => Ok(()),
            Sink::Stdout => {
                let out = std::io::stdout();
                let mut h = out.lock();
                writeln!(h, "{line}").and_then(|()| h.flush())
            }
            Sink::File(f) => writeln!(f, "{line}").and_then(|()| f.flush()),
        };
    }

    /// Empty the log file, used by `/__netsim/reset?truncate=1` when a harness
    /// wants one file per query rather than one file per run.
    pub fn truncate(&self) {
        let mut inner = self.inner.lock().unwrap();
        if let Sink::File(f) = &mut inner.sink {
            let _ = f.set_len(0);
            let _ = f.seek(SeekFrom::Start(0));
        }
    }
}
