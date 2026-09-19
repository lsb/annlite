//! End-to-end tests against a real socket.
//!
//! These bind an ephemeral port and speak HTTP over TCP rather than calling the
//! handler directly. The things most likely to break the browser client are
//! exactly the things an in-process test cannot see: a missing `Content-Length`
//! because the body crossed a chunking threshold, a header that never made it onto
//! the wire, a 206 whose body is off by one byte. So the client here is a few
//! dozen lines of `std::net` and everything is checked on the bytes as received.

use annlite_netsim::profile::{self, Mode, Sim};
use annlite_netsim::server::{Config, Server, Shutdown};
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

struct Fixture {
    addr: SocketAddr,
    shutdown: Shutdown,
    dir: PathBuf,
    body: Vec<u8>,
}

impl Drop for Fixture {
    fn drop(&mut self) {
        self.shutdown.stop();
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

static COUNTER: AtomicU32 = AtomicU32::new(0);

/// A server over a scratch directory holding `data.txt`, a deterministic
/// pseudo-text file big enough (96 KiB) to cross tiny_http's 32 KiB chunking
/// threshold — the regression that would silently drop `Content-Length`.
fn fixture(configure: impl FnOnce(&mut Config)) -> Fixture {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!("annlite-netsim-{}-{n}", std::process::id()));
    std::fs::create_dir_all(dir.join("sub")).unwrap();
    let body: Vec<u8> = (0..96 * 1024u32).map(|i| (i % 251) as u8).collect();
    std::fs::write(dir.join("data.txt"), &body).unwrap();
    std::fs::write(dir.join("sub/inner.txt"), b"inner").unwrap();
    std::fs::write(dir.join("empty.txt"), b"").unwrap();

    let sim = Sim::from_profile(profile::IDEAL, Mode::Fixed, 0);
    let mut cfg = Config::new(dir.clone(), sim);
    cfg.addr = "127.0.0.1:0".into();
    configure(&mut cfg);
    let server = Server::bind(cfg).unwrap();
    let addr = server.local_addr();
    let shutdown = server.shutdown_handle();
    std::thread::spawn(move || server.run().unwrap());
    Fixture { addr, shutdown, dir, body }
}

struct Resp {
    status: u16,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

impl Resp {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }
    fn expect(&self, name: &str) -> &str {
        self.header(name)
            .unwrap_or_else(|| panic!("response is missing header {name}"))
    }
}

/// Minimal HTTP/1.1 client: one request per connection, `Connection: close`, read
/// to EOF. No keep-alive, so no response framing ambiguity to get wrong.
fn request(addr: SocketAddr, method: &str, path: &str, headers: &[(&str, &str)]) -> Resp {
    let mut sock = TcpStream::connect(addr).unwrap();
    sock.set_read_timeout(Some(Duration::from_secs(30))).unwrap();
    let mut req = format!("{method} {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n");
    for (k, v) in headers {
        req.push_str(&format!("{k}: {v}\r\n"));
    }
    req.push_str("\r\n");
    sock.write_all(req.as_bytes()).unwrap();
    sock.flush().unwrap();

    let mut raw = Vec::new();
    sock.read_to_end(&mut raw).unwrap();
    let split = raw
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .expect("response has no header terminator");
    let head = String::from_utf8_lossy(&raw[..split]).into_owned();
    let body = raw[split + 4..].to_vec();

    let mut lines = head.split("\r\n");
    let status_line = lines.next().unwrap();
    let status: u16 = status_line.split(' ').nth(1).unwrap().parse().unwrap();
    let headers = lines
        .filter_map(|l| l.split_once(": "))
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    Resp { status, headers, body }
}

fn get(addr: SocketAddr, path: &str, range: Option<&str>) -> Resp {
    match range {
        Some(r) => request(addr, "GET", path, &[("Range", r)]),
        None => request(addr, "GET", path, &[]),
    }
}

// ---------------------------------------------------------------------------
// Range semantics
// ---------------------------------------------------------------------------

#[test]
fn no_range_serves_the_whole_file() {
    let f = fixture(|_| {});
    let r = get(f.addr, "/data.txt", None);
    assert_eq!(r.status, 200);
    assert_eq!(r.expect("Accept-Ranges"), "bytes");
    assert_eq!(r.expect("Content-Length"), f.body.len().to_string());
    assert!(r.header("Content-Range").is_none());
    assert_eq!(r.body, f.body, "200 body differs from the file");
    // A 96 KiB body must still be identity-coded, not chunked.
    assert!(r.header("Transfer-Encoding").is_none());
}

#[test]
fn ranges_return_exactly_the_requested_bytes() {
    let f = fixture(|_| {});
    let total = f.body.len() as u64;
    // Page-aligned reads like a SQLite VFS makes, plus the awkward edges.
    for (start, end) in [
        (0u64, 0u64),
        (0, 4095),
        (4096, 8191),
        (1, 2),
        (total - 1, total - 1),
        (total - 4096, total - 1),
        (0, total - 1),
        (30_000, 70_000), // spans the 32 KiB chunking threshold
    ] {
        let r = get(f.addr, "/data.txt", Some(&format!("bytes={start}-{end}")));
        let len = end - start + 1;
        assert_eq!(r.status, 206, "bytes={start}-{end}");
        assert_eq!(
            r.expect("Content-Range"),
            format!("bytes {start}-{end}/{total}"),
            "bytes={start}-{end}"
        );
        assert_eq!(r.expect("Content-Length"), len.to_string(), "bytes={start}-{end}");
        assert_eq!(r.body.len() as u64, len, "bytes={start}-{end}");
        assert_eq!(
            r.body,
            f.body[start as usize..=end as usize],
            "served bytes differ from the file for bytes={start}-{end}"
        );
    }
}

#[test]
fn open_ended_range_runs_to_the_end() {
    let f = fixture(|_| {});
    let total = f.body.len() as u64;
    let r = get(f.addr, "/data.txt", Some("bytes=98000-"));
    assert_eq!(r.status, 206);
    assert_eq!(r.expect("Content-Range"), format!("bytes 98000-{}/{total}", total - 1));
    assert_eq!(r.body, f.body[98000..]);
}

#[test]
fn suffix_range_counts_back_from_the_end() {
    let f = fixture(|_| {});
    let total = f.body.len() as u64;
    let r = get(f.addr, "/data.txt", Some("bytes=-100"));
    assert_eq!(r.status, 206);
    assert_eq!(
        r.expect("Content-Range"),
        format!("bytes {}-{}/{total}", total - 100, total - 1)
    );
    assert_eq!(r.body, f.body[f.body.len() - 100..]);

    // A suffix longer than the file is the whole file, not an error.
    let r = get(f.addr, "/data.txt", Some("bytes=-999999999"));
    assert_eq!(r.status, 206);
    assert_eq!(r.body.len(), f.body.len());
}

#[test]
fn unsatisfiable_range_is_416_with_the_total() {
    let f = fixture(|_| {});
    let total = f.body.len() as u64;
    for spec in ["bytes=999999-1000000", "bytes=-0"] {
        let r = get(f.addr, "/data.txt", Some(spec));
        assert_eq!(r.status, 416, "{spec}");
        assert_eq!(r.expect("Content-Range"), format!("bytes */{total}"), "{spec}");
    }
    // An empty file cannot satisfy any range either.
    let r = get(f.addr, "/empty.txt", Some("bytes=0-0"));
    assert_eq!(r.status, 416);
    assert_eq!(r.expect("Content-Range"), "bytes */0");
}

#[test]
fn invalid_range_is_ignored_rather_than_refused() {
    // RFC 7233: an unparseable Range must be ignored, which means 200 and not 416.
    let f = fixture(|_| {});
    for spec in ["bytes=5-3", "bytes=abc", "items=0-9", "bytes=0-9,20-29"] {
        let r = get(f.addr, "/data.txt", Some(spec));
        assert_eq!(r.status, 200, "{spec}");
        assert_eq!(r.body.len(), f.body.len(), "{spec}");
    }
}

#[test]
fn head_matches_get_headers_and_sends_no_body() {
    let f = fixture(|_| {});
    let h = request(f.addr, "HEAD", "/data.txt", &[("Range", "bytes=0-4095")]);
    let g = get(f.addr, "/data.txt", Some("bytes=0-4095"));
    assert_eq!(h.status, g.status);
    assert_eq!(h.expect("Content-Length"), g.expect("Content-Length"));
    assert_eq!(h.expect("Content-Range"), g.expect("Content-Range"));
    assert_eq!(h.expect("Accept-Ranges"), "bytes");
    assert!(h.body.is_empty(), "HEAD returned {} body bytes", h.body.len());

    // The length-discovery HEAD that sql.js-httpvfs opens with.
    let h = request(f.addr, "HEAD", "/data.txt", &[]);
    assert_eq!(h.status, 200);
    assert_eq!(h.expect("Content-Length"), f.body.len().to_string());
    assert!(h.body.is_empty());
}

// ---------------------------------------------------------------------------
// CORS and path safety
// ---------------------------------------------------------------------------

#[test]
fn cors_exposes_the_headers_the_browser_needs() {
    let f = fixture(|_| {});
    let r = get(f.addr, "/data.txt", Some("bytes=0-9"));
    assert_eq!(r.expect("Access-Control-Allow-Origin"), "*");
    let exposed = r.expect("Access-Control-Expose-Headers").to_ascii_lowercase();
    for needed in ["content-range", "content-length", "accept-ranges"] {
        assert!(exposed.contains(needed), "Expose-Headers lacks {needed}: {exposed}");
    }

    let pre = request(f.addr, "OPTIONS", "/data.txt", &[("Origin", "https://example.com")]);
    assert_eq!(pre.status, 204);
    assert_eq!(pre.expect("Access-Control-Allow-Origin"), "*");
    assert!(pre.expect("Access-Control-Allow-Methods").contains("GET"));
    assert!(pre
        .expect("Access-Control-Allow-Headers")
        .to_ascii_lowercase()
        .contains("range"));
}

#[test]
fn path_traversal_is_refused() {
    let f = fixture(|_| {});
    // Every one of these resolves outside the root, or would if it were honoured.
    for path in [
        "/../data.txt",
        "/sub/../../data.txt",
        "/%2e%2e/data.txt",
        "/..%2fdata.txt",
        "//etc/passwd",
        "/etc/passwd",
        "/sub/%2e%2e/%2e%2e/etc/passwd",
    ] {
        let r = get(f.addr, path, None);
        assert!(
            r.status == 403 || r.status == 404,
            "{path} returned {} instead of being refused",
            r.status
        );
        assert!(
            !r.body.windows(4).any(|w| w == b"root"),
            "{path} leaked file contents"
        );
    }
    // A legitimate subdirectory still works, so the check is not just refusing.
    let r = get(f.addr, "/sub/inner.txt", None);
    assert_eq!(r.status, 200);
    assert_eq!(r.body, b"inner");
}

// ---------------------------------------------------------------------------
// Simulation
// ---------------------------------------------------------------------------

/// Collect the latency each request was charged, as the server reports it.
fn latency_sequence(sim: Sim, n: u64) -> Vec<String> {
    let f = fixture(|cfg| cfg.sim = sim);
    (0..n)
        .map(|i| {
            let r = get(f.addr, "/data.txt", Some(&format!("bytes={}-{}", i * 4096, i * 4096 + 63)));
            assert_eq!(r.status, 206);
            r.expect("X-Netsim-Latency-Ms").to_string()
        })
        .collect()
}

#[test]
fn seeded_probabilistic_latency_replays_identically() {
    // A fast link with heavy variance: the point is the sequence of draws, and the
    // test should not spend seconds sleeping to observe it.
    let mut sim = Sim::from_profile(profile::LTE, Mode::Random, 20260919);
    sim.latency_ms = 6.0;
    sim.spike_ms = 15.0;
    sim.spike_prob = 0.25;
    sim.bytes_per_sec = None;

    let first = latency_sequence(sim, 24);
    let second = latency_sequence(sim, 24);
    assert_eq!(first, second, "the same seed and request sequence must replay exactly");

    // ...and it is a genuine distribution, not a constant dressed up as one.
    let distinct: std::collections::HashSet<&String> = first.iter().collect();
    assert!(distinct.len() > 15, "expected varied draws, saw {distinct:?}");
    let spread = first
        .iter()
        .map(|s| s.parse::<f64>().unwrap())
        .fold((f64::MAX, 0.0f64), |(lo, hi), v| (lo.min(v), hi.max(v)));
    assert!(spread.1 > spread.0 * 1.5, "draws are suspiciously flat: {spread:?}");

    // A different seed must give a different sequence, or the seed is decorative.
    let mut other = sim;
    other.seed = 1;
    assert_ne!(first, latency_sequence(other, 24));
}

#[test]
fn fixed_mode_charges_exactly_the_configured_latency() {
    let mut sim = Sim::from_profile(profile::LTE, Mode::Fixed, 0);
    sim.latency_ms = 7.0;
    sim.bytes_per_sec = None;
    let seq = latency_sequence(sim, 5);
    assert!(seq.iter().all(|v| v == "7.000"), "{seq:?}");
}

#[test]
fn throughput_shaping_slows_the_body_measurably() {
    // 64 KiB at 256 KB/s is a quarter of a second of transfer that must actually
    // elapse, against a baseline that has to be far below it.
    let mut sim = Sim::from_profile(profile::IDEAL, Mode::Fixed, 0);
    sim.bytes_per_sec = Some(256.0 * 1024.0);
    let f = fixture(|cfg| cfg.sim = sim);
    let t0 = Instant::now();
    let r = get(f.addr, "/data.txt", Some("bytes=0-65535"));
    let shaped = t0.elapsed();
    assert_eq!(r.status, 206);
    assert_eq!(r.body, f.body[..65536]);

    let unshaped = {
        let g = fixture(|_| {});
        let t = Instant::now();
        assert_eq!(get(g.addr, "/data.txt", Some("bytes=0-65535")).status, 206);
        t.elapsed()
    };
    assert!(
        shaped >= Duration::from_millis(200),
        "64 KiB at 256 KB/s took only {shaped:?}"
    );
    assert!(
        shaped > unshaped * 4,
        "shaped {shaped:?} is not meaningfully slower than unshaped {unshaped:?}"
    );
}

// ---------------------------------------------------------------------------
// Logging and control plane
// ---------------------------------------------------------------------------

#[test]
fn requests_are_logged_as_jsonl_and_attributable_to_a_label() {
    let dir = std::env::temp_dir().join(format!("annlite-netsim-log-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let log_path = dir.join("requests.jsonl");
    let f = fixture(|cfg| cfg.log = Some(log_path.clone()));

    request(f.addr, "GET", "/__netsim/label?name=warmup", &[]);
    get(f.addr, "/data.txt", Some("bytes=0-4095"));
    let reset = request(f.addr, "GET", "/__netsim/reset?label=query-7", &[]);
    assert_eq!(reset.status, 200);
    let reset: serde_json::Value = serde_json::from_slice(&reset.body).unwrap();
    assert_eq!(reset["previous"]["requests"], 1);
    assert_eq!(reset["previous"]["bytes"], 4096);

    get(f.addr, "/data.txt", Some("bytes=8192-12287"));
    request(f.addr, "GET", "/data.txt", &[("X-Netsim-Label", "override")]);

    let stats: serde_json::Value =
        serde_json::from_slice(&request(f.addr, "GET", "/__netsim/stats", &[]).body).unwrap();
    assert_eq!(stats["stats"]["requests"], 2);

    let lines: Vec<serde_json::Value> = std::fs::read_to_string(&log_path)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(lines.len(), 3, "control endpoints must not be logged");

    assert_eq!(lines[0]["label"], "warmup");
    assert_eq!(lines[0]["ordinal"], 0);
    assert_eq!(lines[0]["status"], 206);
    assert_eq!(lines[0]["offset"], 0);
    assert_eq!(lines[0]["length"], 4096);
    assert_eq!(lines[0]["total"], f.body.len());
    assert_eq!(lines[0]["range"], "bytes=0-4095");
    assert_eq!(lines[0]["method"], "GET");
    assert_eq!(lines[0]["path"], "/data.txt");

    // The reset restarts the ordinal, which is also the index into the delay
    // stream: that is what makes two queries see the same simulated network.
    assert_eq!(lines[1]["ordinal"], 0);
    assert_eq!(lines[1]["label"], "query-7");
    assert_eq!(lines[1]["offset"], 8192);

    assert_eq!(lines[2]["ordinal"], 1);
    assert_eq!(lines[2]["label"], "override", "a per-request label must win");
    assert!(lines[2]["service_ms"].as_f64().unwrap() >= 0.0);
    assert!(lines[2]["t_ms"].as_f64().unwrap() >= 0.0);

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn logged_timings_account_for_the_simulated_link() {
    let dir = std::env::temp_dir().join(format!("annlite-netsim-time-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let log_path = dir.join("requests.jsonl");
    let mut sim = Sim::from_profile(profile::IDEAL, Mode::Fixed, 0);
    sim.latency_ms = 40.0;
    sim.bytes_per_sec = Some(512.0 * 1024.0);
    let f = fixture(|cfg| {
        cfg.sim = sim;
        cfg.log = Some(log_path.clone());
    });

    get(f.addr, "/data.txt", Some("bytes=0-65535"));
    let rec: serde_json::Value =
        serde_json::from_str(std::fs::read_to_string(&log_path).unwrap().lines().next().unwrap())
            .unwrap();
    let latency = rec["latency_ms"].as_f64().unwrap();
    let transfer = rec["transfer_ms"].as_f64().unwrap();
    let service = rec["service_ms"].as_f64().unwrap();
    assert!((latency - 40.0).abs() < 1e-6);
    assert!((transfer - 125.0).abs() < 1e-6, "65536 B at 512 KiB/s is 125 ms, got {transfer}");
    // The real service time must cover the simulation; a large overshoot would
    // mean the host, not the simulated link, is the bottleneck.
    assert!(service >= latency + transfer - 5.0, "service {service} < simulated {latency}+{transfer}");
    assert!(service < latency + transfer + 100.0, "service {service} overshot badly");

    std::fs::remove_dir_all(&dir).ok();
}
