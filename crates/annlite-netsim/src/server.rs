//! The server: RFC 7233 range semantics over a simulated link.
//!
//! Built on `tiny_http` because it is blocking and gives us the response body as a
//! `Read` we control. That is the whole trick behind throughput shaping: the body
//! is a reader that sleeps between chunks, so the bytes leave the socket at the
//! simulated rate instead of all at once. An async server would need a custom
//! timer-driven stream to do the same thing, for no benefit here — the workload is
//! a handful of concurrent connections, and threads that spend their lives asleep
//! are exactly what the simulation is made of.
//!
//! # Concurrency and reproducibility
//!
//! The accept loop runs on one thread and assigns each request its ordinal, then
//! hands it to a worker pool. Ordinals therefore follow arrival order, while the
//! sleeping happens in parallel — a browser that opens six connections is not
//! serialised into six sequential round-trips, but the delay each request draws
//! still depends only on its position in the sequence.
//!
//! Control requests (`/__netsim/*`) are answered on the accept thread without
//! taking an ordinal. They are not part of the workload, so they must not consume
//! draws or count as round-trips, and answering them in arrival order gives
//! `reset` a well-defined position relative to the requests around it.

use crate::log::{Record, RequestLog};
use crate::profile::Sim;
use crate::range::{self, Resolved};
use anyhow::{Context, Result};
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use tiny_http::{Header, Request, Response, StatusCode};

#[derive(Clone, Debug)]
pub struct Config {
    /// Only files under this directory are served, enforced both lexically and by
    /// canonicalising the resolved path.
    pub root: PathBuf,
    pub addr: String,
    pub sim: Sim,
    /// Worker threads. Four matches a browser's per-origin connection limit
    /// closely enough that queueing effects look like the real thing.
    pub threads: usize,
    /// Body chunk size for throughput shaping. Must stay above tiny_http's 1 KiB
    /// write buffer, or chunks are coalesced and the pacing never reaches the
    /// socket; 8 KiB also matches `io::copy`'s buffer, so one chunk is one write.
    pub chunk_bytes: usize,
    pub log: Option<PathBuf>,
    /// Sent verbatim as `Cache-Control`. The default is `no-store` because a
    /// browser cache silently absorbing repeat page fetches would make an index
    /// look better than it is; set it to a real policy only when caching is the
    /// thing under test.
    pub cache_control: String,
}

impl Config {
    pub fn new(root: PathBuf, sim: Sim) -> Config {
        Config {
            root,
            addr: "127.0.0.1:0".into(),
            sim,
            threads: 4,
            chunk_bytes: 8192,
            log: None,
            cache_control: "no-store".into(),
        }
    }
}

pub struct Server {
    http: Arc<tiny_http::Server>,
    cfg: Arc<Config>,
    root: PathBuf,
    log: Arc<RequestLog>,
}

/// Lets another thread stop `Server::run`.
#[derive(Clone)]
pub struct Shutdown(Arc<tiny_http::Server>);

impl Shutdown {
    pub fn stop(&self) {
        self.0.unblock();
    }
}

impl Server {
    pub fn bind(cfg: Config) -> Result<Server> {
        let root = cfg
            .root
            .canonicalize()
            .with_context(|| format!("serving root {} does not exist", cfg.root.display()))?;
        anyhow::ensure!(root.is_dir(), "serving root {} is not a directory", root.display());
        let log = RequestLog::open(cfg.log.as_deref())?;
        let http = tiny_http::Server::http(&cfg.addr)
            .map_err(|e| anyhow::anyhow!("binding {}: {e}", cfg.addr))?;
        Ok(Server {
            http: Arc::new(http),
            cfg: Arc::new(cfg),
            root,
            log: Arc::new(log),
        })
    }

    pub fn local_addr(&self) -> SocketAddr {
        match self.http.server_addr() {
            tiny_http::ListenAddr::IP(a) => a,
            #[allow(unreachable_patterns)]
            _ => unreachable!("only TCP listeners are created"),
        }
    }

    pub fn shutdown_handle(&self) -> Shutdown {
        Shutdown(self.http.clone())
    }

    pub fn log(&self) -> Arc<RequestLog> {
        self.log.clone()
    }

    /// Serve until [`Shutdown::stop`] is called.
    pub fn run(self) -> Result<()> {
        let (tx, rx) = mpsc::channel::<Job>();
        let rx = Arc::new(Mutex::new(rx));
        let mut workers = Vec::new();
        for _ in 0..self.cfg.threads.max(1) {
            let rx = rx.clone();
            let cfg = self.cfg.clone();
            let root = self.root.clone();
            let log = self.log.clone();
            workers.push(thread::spawn(move || loop {
                let job = {
                    let guard = rx.lock().unwrap();
                    guard.recv()
                };
                match job {
                    Ok(job) => serve(&cfg, &root, &log, job),
                    Err(_) => break,
                }
            }));
        }

        for request in self.http.incoming_requests() {
            let accepted = Instant::now();
            if request.url().starts_with(CONTROL_PREFIX) {
                control(&self.cfg, &self.log, request);
                continue;
            }
            let job = Job {
                ordinal: self.log.next_ordinal(),
                t_ms: self.log.elapsed_ms(),
                accepted,
                request,
            };
            if tx.send(job).is_err() {
                break;
            }
        }
        drop(tx);
        for w in workers {
            let _ = w.join();
        }
        Ok(())
    }
}

struct Job {
    ordinal: u64,
    t_ms: f64,
    accepted: Instant,
    request: Request,
}

const CONTROL_PREFIX: &str = "/__netsim/";

// ---------------------------------------------------------------------------
// Request handling
// ---------------------------------------------------------------------------

fn serve(cfg: &Config, root: &Path, log: &RequestLog, job: Job) {
    let Job { ordinal, t_ms, accepted, request } = job;
    let method = request.method().as_str().to_string();
    let (raw_path, _query) = split_url(request.url());
    let range_header = header_value(&request, "Range");
    let label = header_value(&request, "X-Netsim-Label").unwrap_or_else(|| log.label());

    let mut rec = Record {
        ordinal,
        t_ms,
        label,
        method: method.clone(),
        path: raw_path.to_string(),
        range: range_header.clone(),
        status: 0,
        offset: 0,
        length: 0,
        total: 0,
        latency_ms: 0.0,
        transfer_ms: 0.0,
        service_ms: 0.0,
        profile: cfg.sim.profile_name,
        mode: cfg.sim.mode,
    };

    // The latency is charged to every request that reaches the network, including
    // preflights and errors: the browser waits for those round-trips too.
    let latency = cfg.sim.latency(ordinal);
    rec.latency_ms = latency.as_secs_f64() * 1e3;

    let outcome = match method.as_str() {
        "OPTIONS" => Outcome::Preflight,
        "GET" | "HEAD" => match resolve_path(root, raw_path) {
            Err(status) => Outcome::Error(status, "path not served"),
            Ok(path) => match File::open(&path).and_then(|f| Ok((f.metadata()?.len(), f))) {
                Err(_) => Outcome::Error(404, "not found"),
                Ok((total, file)) => Outcome::File { file, total, path },
            },
        },
        _ => Outcome::Error(405, "method not allowed"),
    };

    sleep(latency);

    let result = match outcome {
        Outcome::Preflight => {
            rec.status = 204;
            request.respond(finish(preflight_response(cfg)))
        }
        Outcome::Error(status, msg) => {
            rec.status = status;
            let body = format!("{status} {msg}\n");
            let mut resp = Response::from_string(body)
                .with_status_code(StatusCode(status))
                .boxed();
            add_common_headers(&mut resp, cfg, ordinal, rec.latency_ms);
            resp.add_header(text_header("Content-Type", "text/plain; charset=utf-8"));
            request.respond(finish(resp))
        }
        Outcome::File { file, total, path } => {
            rec.total = total;
            let resolved = range::resolve(range_header.as_deref(), total);
            let (offset, length) = resolved.span(total);
            rec.offset = offset;
            rec.length = length;
            let rate = cfg.sim.rate(ordinal);
            // A HEAD moves no body, so it costs a round-trip and nothing else.
            let head = method == "HEAD";
            rec.transfer_ms = match (head, rate) {
                (true, _) | (_, None) => 0.0,
                (false, Some(r)) => length as f64 / r * 1e3,
            };
            rec.status = match resolved {
                Resolved::Full => 200,
                Resolved::Partial { .. } => 206,
                Resolved::Unsatisfiable => 416,
            };
            match file_response(cfg, file, &path, total, resolved, ordinal, rec.latency_ms, rate) {
                Ok(resp) => request.respond(finish(resp)),
                Err(_) => {
                    rec.status = 500;
                    rec.length = 0;
                    request.respond(Response::from_string("500 read error\n").with_status_code(500))
                }
            }
        }
    };
    // A client hanging up mid-body is ordinary (a browser abandoning a fetch) and
    // must not take the worker down. The record is written either way, so an
    // abandoned transfer still shows up as a round-trip that was paid for.
    let _ = result;

    rec.service_ms = accepted.elapsed().as_secs_f64() * 1e3;
    log.write(&rec);
}

enum Outcome {
    Preflight,
    Error(u16, &'static str),
    File { file: File, total: u64, path: PathBuf },
}

/// Force identity transfer coding.
///
/// tiny_http switches to chunked for any body of 32 KiB or more, which drops
/// `Content-Length`. sql.js-httpvfs reads `Content-Length` (and browsers expose it
/// through the Fetch API's headers), so a chunked 206 would break the client in a
/// way that only shows up once a range crosses 32 KiB — a nasty size-dependent
/// bug. Raising the threshold out of reach pins every response to identity.
fn finish(resp: tiny_http::ResponseBox) -> tiny_http::ResponseBox {
    resp.with_chunked_threshold(usize::MAX)
}

#[allow(clippy::too_many_arguments)]
fn file_response(
    cfg: &Config,
    mut file: File,
    path: &Path,
    total: u64,
    resolved: Resolved,
    ordinal: u64,
    latency_ms: f64,
    rate: Option<f64>,
) -> io::Result<tiny_http::ResponseBox> {
    let (offset, length) = resolved.span(total);

    if resolved == Resolved::Unsatisfiable {
        let mut resp = Response::from_string("416 range not satisfiable\n")
            .with_status_code(StatusCode(416))
            .boxed();
        add_common_headers(&mut resp, cfg, ordinal, latency_ms);
        // RFC 7233 §4.4: the `*/total` form is how the client learns the real
        // length after guessing wrong.
        resp.add_header(text_header("Content-Range", &format!("bytes */{total}")));
        return Ok(resp);
    }

    file.seek(SeekFrom::Start(offset))?;
    let body = Paced::new(file.take(length), rate, cfg.chunk_bytes);
    let status = if matches!(resolved, Resolved::Partial { .. }) { 206 } else { 200 };
    let mut resp = Response::new(
        StatusCode(status),
        Vec::new(),
        Box::new(body) as Box<dyn Read + Send>,
        Some(length as usize),
        None,
    );
    add_common_headers(&mut resp, cfg, ordinal, latency_ms);
    resp.add_header(text_header("Content-Type", content_type(path)));
    if let Some(tag) = etag(path) {
        resp.add_header(text_header("ETag", &tag));
    }
    if let Resolved::Partial { start, end } = resolved {
        resp.add_header(text_header(
            "Content-Range",
            &format!("bytes {start}-{end}/{total}"),
        ));
    }
    Ok(resp)
}

fn preflight_response(cfg: &Config) -> tiny_http::ResponseBox {
    let mut resp = Response::empty(StatusCode(204)).boxed();
    add_common_headers(&mut resp, cfg, 0, 0.0);
    resp.add_header(text_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS"));
    resp.add_header(text_header(
        "Access-Control-Allow-Headers",
        "Range, If-Range, X-Netsim-Label, Cache-Control",
    ));
    resp.add_header(text_header("Access-Control-Max-Age", "86400"));
    resp
}

/// Headers every response carries.
///
/// `Access-Control-Expose-Headers` is the one that is easy to forget and fatal to
/// omit: a cross-origin `fetch` can read only a handful of allow-listed response
/// headers, and `Content-Range` is not among them. Without this line the browser
/// receives a correct 206 and sql.js-httpvfs still cannot determine the file
/// length, which presents as an unreadable database rather than as a CORS error.
fn add_common_headers(resp: &mut tiny_http::ResponseBox, cfg: &Config, ordinal: u64, latency_ms: f64) {
    resp.add_header(text_header("Accept-Ranges", "bytes"));
    resp.add_header(text_header("Access-Control-Allow-Origin", "*"));
    resp.add_header(text_header(
        "Access-Control-Expose-Headers",
        "Content-Range, Content-Length, Accept-Ranges, ETag, X-Netsim-Ordinal, X-Netsim-Latency-Ms",
    ));
    resp.add_header(text_header("Cache-Control", &cfg.cache_control));
    // Echoed so a client-side trace can be joined to the server's JSONL without
    // guessing which request was which.
    resp.add_header(text_header("X-Netsim-Ordinal", &ordinal.to_string()));
    resp.add_header(text_header("X-Netsim-Latency-Ms", &format!("{latency_ms:.3}")));
}

// ---------------------------------------------------------------------------
// Throughput shaping
// ---------------------------------------------------------------------------

/// A reader that hands back at most `chunk` bytes per call and then sleeps until
/// the simulated link would have delivered them.
///
/// Pacing is against a deadline computed from the *cumulative* byte count rather
/// than by sleeping a fixed amount per chunk. Sleep always overshoots a little, and
/// per-chunk sleeps compound that error: over the ~550 chunks of a 4.5 MB file a
/// 100 µs overshoot would add half a second of drift. Against a deadline the
/// overshoot is absorbed by the next chunk instead of accumulating.
struct Paced<R> {
    inner: R,
    /// Bytes per second; `None` serves at host speed.
    rate: Option<f64>,
    chunk: usize,
    sent: u64,
    start: Option<Instant>,
}

impl<R: Read> Paced<R> {
    fn new(inner: R, rate: Option<f64>, chunk: usize) -> Self {
        Paced { inner, rate, chunk: chunk.max(1), sent: 0, start: None }
    }
}

impl<R: Read> Read for Paced<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let cap = buf.len().min(self.chunk);
        let n = self.inner.read(&mut buf[..cap])?;
        if n == 0 {
            return Ok(0);
        }
        if let Some(rate) = self.rate {
            let start = *self.start.get_or_insert_with(Instant::now);
            self.sent += n as u64;
            let due = start + Duration::from_secs_f64(self.sent as f64 / rate);
            // The sleep happens before the bytes are returned to the writer, so
            // byte k leaves the socket at roughly k/rate, which is what a client
            // measuring time-to-last-byte sees.
            if let Some(wait) = due.checked_duration_since(Instant::now()) {
                sleep(wait);
            }
        }
        Ok(n)
    }
}

/// `thread::sleep` on a zero duration still costs a syscall, and the zero case is
/// the `ideal` profile's entire hot path.
fn sleep(d: Duration) {
    if d > Duration::ZERO {
        thread::sleep(d);
    }
}

// ---------------------------------------------------------------------------
// Control plane
// ---------------------------------------------------------------------------

/// `/__netsim/*`: label, reset, stats, profile.
///
/// Answered without latency, without an ordinal and without a log line, so a
/// harness can bracket a query with control calls without perturbing what it is
/// measuring.
fn control(cfg: &Config, log: &RequestLog, request: Request) {
    let (path, query) = split_url(request.url());
    let body = match path.trim_end_matches('/') {
        "/__netsim/label" => {
            if let Some(l) = query_param(query, "name") {
                log.set_label(&l);
            }
            serde_json::json!({ "ok": true, "label": log.label() })
        }
        "/__netsim/reset" => {
            let previous = log.reset(query_param(query, "label").as_deref());
            if query_param(query, "truncate").is_some() {
                log.truncate();
            }
            serde_json::json!({ "ok": true, "label": log.label(), "previous": previous })
        }
        "/__netsim/stats" => {
            serde_json::json!({ "ok": true, "label": log.label(), "stats": log.stats() })
        }
        "/__netsim/profile" => serde_json::json!({
            "ok": true,
            "profile": cfg.sim.profile_name,
            "mode": cfg.sim.mode,
            "latency_ms": cfg.sim.latency_ms,
            "latency_sigma": cfg.sim.latency_sigma,
            "spike_prob": cfg.sim.spike_prob,
            "spike_ms": cfg.sim.spike_ms,
            "bytes_per_sec": cfg.sim.bytes_per_sec,
            "bandwidth_sigma": cfg.sim.bandwidth_sigma,
            "seed": cfg.sim.seed,
            "chunk_bytes": cfg.chunk_bytes,
        }),
        other => serde_json::json!({ "ok": false, "error": format!("no control endpoint {other}") }),
    };
    let ok = body.get("ok").and_then(|v| v.as_bool()).unwrap_or(false);
    let mut resp = Response::from_string(body.to_string() + "\n")
        .with_status_code(StatusCode(if ok { 200 } else { 404 }))
        .boxed();
    add_common_headers(&mut resp, cfg, 0, 0.0);
    resp.add_header(text_header("Content-Type", "application/json"));
    let _ = request.respond(finish(resp));
}

// ---------------------------------------------------------------------------
// Paths and headers
// ---------------------------------------------------------------------------

fn split_url(url: &str) -> (&str, &str) {
    match url.split_once('?') {
        Some((p, q)) => (p, q),
        None => (url, ""),
    }
}

fn query_param(query: &str, key: &str) -> Option<String> {
    query.split('&').find_map(|pair| {
        let (k, v) = pair.split_once('=')?;
        (k == key).then(|| percent_decode(&v.replace('+', " ")))
    })
}

/// Map a request path to a file under `root`, or a status code to refuse with.
///
/// Refused lexically first — a `..` component never reaches the filesystem — and
/// then again after canonicalising, which catches a symlink inside the root that
/// points out of it. Both checks are cheap and they fail differently, so both stay.
fn resolve_path(root: &Path, raw: &str) -> Result<PathBuf, u16> {
    let mut out = root.to_path_buf();
    let mut any = false;
    // Decoding each component separately means an encoded separator (`%2F`) can
    // never manufacture a new path segment after the fact.
    for segment in raw.trim_start_matches('/').split('/') {
        if segment.is_empty() {
            continue;
        }
        let decoded = percent_decode(segment);
        if decoded.is_empty()
            || decoded == "."
            || decoded == ".."
            || decoded.contains('/')
            || decoded.contains('\\')
            || decoded.contains('\0')
        {
            return Err(403);
        }
        out.push(decoded);
        any = true;
    }
    if !any {
        return Err(404);
    }
    let canonical = out.canonicalize().map_err(|_| 404u16)?;
    if !canonical.starts_with(root) {
        return Err(403);
    }
    if !canonical.is_file() {
        return Err(404);
    }
    Ok(canonical)
}

fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let (Some(hi), Some(lo)) = (hex(bytes[i + 1]), hex(bytes[i + 2])) {
                out.push(hi << 4 | lo);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    // Invalid UTF-8 is replaced rather than rejected; such a path will simply fail
    // to open, and no decoded byte can introduce a separator (that is checked by
    // the caller on the decoded string).
    String::from_utf8_lossy(&out).into_owned()
}

fn hex(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

fn header_value(request: &Request, name: &'static str) -> Option<String> {
    request
        .headers()
        .iter()
        .find(|h| h.field.equiv(name))
        .map(|h| h.value.as_str().to_string())
}

fn text_header(name: &str, value: &str) -> Header {
    Header::from_bytes(name.as_bytes(), value.as_bytes())
        .unwrap_or_else(|_| panic!("header {name} is not valid ASCII"))
}

/// Enough of a type table for what this serves: corpora, SQLite files, and the
/// static assets of the browser demo.
fn content_type(path: &Path) -> &'static str {
    match path.extension().and_then(|e| e.to_str()).unwrap_or("") {
        "txt" => "text/plain; charset=utf-8",
        "json" | "jsonl" => "application/json",
        "html" => "text/html; charset=utf-8",
        "js" | "mjs" => "text/javascript",
        "wasm" => "application/wasm",
        "db" | "sqlite" | "sqlite3" => "application/vnd.sqlite3",
        _ => "application/octet-stream",
    }
}

/// Strong validator from length and mtime. Cheap, and stable across restarts so a
/// benchmark that does want caching can exercise revalidation.
fn etag(path: &Path) -> Option<String> {
    let meta = std::fs::metadata(path).ok()?;
    let mtime = meta
        .modified()
        .ok()?
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_nanos();
    Some(format!("\"{:x}-{:x}\"", meta.len(), mtime))
}
