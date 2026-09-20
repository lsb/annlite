//! Late interaction in SQLite, measured in pages — the third leg of the comparison.
//!
//! Quality for this system was already known (RESEARCH_LOG.md section 15) and so was
//! its compressed size (section 15.2), but both were measured in memory. Neither
//! says anything about the axis this project is about, which is how many *dependent
//! round-trips* and how many *pages* a client that holds no copy of the file has to
//! pay. This binary supplies that, on the same corpus and the same 500 queries as
//! the FTS5 and dense measurements, so the three can be read side by side.
//!
//! Quality is computed exactly as `tools/analyze/code_eval.py::metrics` computes it,
//! on purpose: a benchmark whose three systems each define success@k slightly
//! differently is not a comparison. `success@1` has a ceiling of 0.962 on this query
//! set (section 15.1) and is reported against that, not against 1.0.

use annlite_core::late::{LateIndex, MultiVector};
use annlite_sqlite::late_search::{late_search, LateDb};
use annlite_sqlite::late_store::{write_late_index, PAGE_BYTES};
use anyhow::Result;
use rusqlite::Connection;
use std::io::Write;
use std::path::Path;

fn read_lengths(p: &Path) -> Result<Vec<usize>> {
    let b = std::fs::read(p)?;
    Ok(b.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as usize).collect())
}

fn read_multi(p: &Path, lengths: &[usize], dim: usize) -> Result<Vec<MultiVector>> {
    let bytes = std::fs::read(p)?;
    let all: Vec<f32> =
        bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let mut out = Vec::with_capacity(lengths.len());
    let mut off = 0usize;
    for &n in lengths {
        out.push(MultiVector { data: all[off * dim..(off + n) * dim].to_vec(), dim });
        off += n;
    }
    anyhow::ensure!(off * dim == all.len(), "lengths do not account for every vector");
    Ok(out)
}

/// The gold document for each query, read from the corpus rather than assumed.
///
/// The construction makes `source_doc` equal the query index, but the whole
/// benchmark rests on that, so it is checked rather than relied on.
fn gold(path: &Path, n: usize) -> Result<Vec<usize>> {
    let text = std::fs::read_to_string(path)?;
    let mut out = Vec::with_capacity(n);
    for (i, line) in text.lines().take(n).enumerate() {
        let v: serde_json::Value = serde_json::from_str(line)?;
        let src = v["source_doc"].as_u64().unwrap() as usize;
        let qid = v["qid"].as_u64().unwrap() as usize;
        anyhow::ensure!(qid == i, "query file is out of order at line {i}");
        out.push(src);
    }
    Ok(out)
}

/// Identical to `metrics()` in `tools/analyze/code_eval.py`. A rank is the 1-indexed
/// position of the gold document, or `ABSENT` if it was never returned.
const ABSENT: usize = 1_000_000;

fn metrics(ranks: &[usize]) -> serde_json::Value {
    let n = ranks.len() as f64;
    let succ = |k: usize| ranks.iter().filter(|&&r| r <= k).count() as f64 / n;
    let mrr = ranks.iter().map(|&r| if r <= 10 { 1.0 / r as f64 } else { 0.0 }).sum::<f64>() / n;
    serde_json::json!({
        "success@1": succ(1), "success@10": succ(10), "success@100": succ(100), "mrr@10": mrr,
    })
}

/// `numpy.percentile` with linear interpolation, so the three systems' p95 figures
/// are the same statistic.
fn pct(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let pos = q * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo as f64)
}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len().max(1) as f64
}

/// Process CPU time, in seconds, from `/proc/self/stat`.
///
/// Wall clock on this machine moves by more than 2x with background load, which is
/// why every timing here is CPU time and why the emitted records still carry
/// a measured `contended` flag — CPU time is steadier than wall clock, not immune.
fn cpu_seconds() -> f64 {
    let s = std::fs::read_to_string("/proc/self/stat").unwrap_or_default();
    // Field 2 is the command name, parenthesised and free to contain spaces, so the
    // fields after it are found from the last ')' rather than by splitting.
    let Some(i) = s.rfind(')') else { return 0.0 };
    let f: Vec<&str> = s[i + 1..].split_whitespace().collect();
    // What follows is state, ppid, pgrp, session, tty_nr, tpgid, flags and the four
    // fault counters, putting utime and stime at offsets 11 and 12. Units are clock
    // ticks, 100 per second on every Linux this runs on.
    let get = |i: usize| f.get(i).and_then(|x| x.parse::<f64>().ok()).unwrap_or(0.0);
    (get(11) + get(12)) / 100.0
}

/// Which corpus this run measures, and where its pieces live.
///
/// The code corpus is the one late interaction was built for and the only one with
/// a published success@1 ceiling, but the cost axis this project is about -- pages,
/// requests, hops -- is meaningful on the word corpora too, and the FTS5 and dense
/// baselines are already measured there. Naming the target rather than hardcoding it
/// is what lets all three systems be compared at the same scale.
struct Target {
    label: String,
    emb_prefix: String,
    gold_path: std::path::PathBuf,
    out_path: std::path::PathBuf,
    /// Highest success@1 the query set admits, where one has been established.
    ceiling: Option<f64>,
}

impl Target {
    fn new(label: &str) -> Result<Self> {
        Ok(match label {
            // 7.4% of code queries share a docstring with another function and cannot
            // be answered as known-item retrieval (RESEARCH_LOG 15.1).
            "code" => Target {
                label: label.into(),
                emb_prefix: "code-late".into(),
                gold_path: "data/corpus/code-queries.jsonl".into(),
                out_path: "bench/results/tri-late.jsonl".into(),
                ceiling: Some(0.962),
            },
            // Word corpora: `words-10k` reads the embeddings written for docs-10k.txt
            // and the first 500 queries of queries-10k.jsonl, which are all
            // known-item with a gold document, exactly as the dense sweep uses them.
            scale if scale.starts_with("words-") => {
                let s = &scale["words-".len()..];
                Target {
                    label: label.into(),
                    emb_prefix: format!("{label}-late"),
                    gold_path: format!("data/corpus/queries-{s}.jsonl").into(),
                    out_path: format!("bench/results/late-{label}.jsonl").into(),
                    ceiling: None,
                }
            }
            other => anyhow::bail!(
                "unknown corpus {other:?}; expected \"code\" or \"words-<scale>\""
            ),
        })
    }
}

fn main() -> Result<()> {
    let dim = 48;
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |name: &str| -> Option<String> {
        args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
    };
    // A bare positional number stays supported: `late_pages -- 500` is how the
    // committed code-corpus rows were produced.
    let n_q: usize = flag("--queries")
        .or_else(|| args.first().filter(|a| a.parse::<usize>().is_ok()).cloned())
        .and_then(|s| s.parse().ok())
        .unwrap_or(500);
    let target = Target::new(&flag("--corpus").unwrap_or_else(|| "code".into()))?;
    let ks: Vec<usize> = match flag("--k") {
        Some(v) => v.split(',').map(|x| x.trim().parse::<usize>()).collect::<Result<_, _>>()?,
        None => vec![512, 1024, 2048],
    };

    let emb = Path::new("data/embeddings");
    let pfx = &target.emb_prefix;
    let dl = read_lengths(&emb.join(format!("{pfx}-lengths.i32")))?;
    let ql = read_lengths(&emb.join(format!("{pfx}-qlengths.i32")))?;
    let docs = read_multi(&emb.join(format!("{pfx}.f32")), &dl, dim)?;
    let queries = read_multi(&emb.join(format!("{pfx}-q.f32")), &ql, dim)?;
    let n_q = n_q.min(queries.len());
    let gold = gold(&target.gold_path, n_q)?;
    let total_tokens: usize = dl.iter().sum();

    let mut lens: Vec<f64> = dl.iter().map(|&x| x as f64).collect();
    lens.sort_by(f64::total_cmp);
    eprintln!(
        "{} docs, {total_tokens} tokens (mean {:.1}, median {:.0}, max {:.0} per doc), {n_q} queries",
        docs.len(),
        total_tokens as f64 / docs.len() as f64,
        pct(&lens, 0.5),
        lens.last().unwrap()
    );

    std::fs::create_dir_all("data/db")?;
    std::fs::create_dir_all("bench/results")?;
    // Truncated, not appended. The readers in tools/analyze pick the *first* row
    // matching a (system, config), so an appended re-run does not replace the old
    // measurement -- it hides behind it. That is exactly what happened to the
    // section 18.2 re-measurement: every late row was written twice and the stale
    // contended one kept being published. One run now produces one complete file.
    let out_path = target.out_path.clone();
    let mut out = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&out_path)?;

    println!(
        "\n{:>6} {:>6} {:>7} {:>8} {:>8} {:>8} {:>9} {:>9} {:>9} {:>9} {:>8} {:>6} {:>8}",
        "k", "probe", "rerank", "succ@1", "succ@10", "mrr@10", "post pg", "cent pg", "rrnk pg",
        "pages", "requests", "hops", "cpu ms"
    );
    println!("{}", "-".repeat(110));

    for &k in &ks {
        let t0 = std::time::Instant::now();
        let cpu0 = cpu_seconds();
        // Centroids are fitted on a strided sample; every token is still assigned.
        // 60,000 is what section 15.2 used, so these indexes are the same ones whose
        // quality is already published.
        let idx = LateIndex::build_sampled(&docs, k, 10, 0xC0DE, 60_000)?;
        let build_cpu = cpu_seconds() - cpu0;
        let build_wall = t0.elapsed().as_secs_f64();

        let db_path = format!("data/db/late-{}-k{k}.db", target.label);
        let _ = std::fs::remove_file(&db_path);
        let mut conn = Connection::open(&db_path)?;
        let meta = write_late_index(&mut conn, &idx, &docs)?;
        let db = LateDb::open(&conn)?;
        let file_bytes = std::fs::metadata(&db_path)?.len() as usize;
        let contiguous = db.postings_arena.is_contiguous()
            && db.codes_arena.is_contiguous()
            && db.tokens_arena.is_contiguous();
        eprintln!(
            "  k={k}: built in {build_cpu:.1}s cpu ({build_wall:.1}s wall), written in \
             {:.1}s, file {:.1} MB, resident {:.0} KB, arenas contiguous: {contiguous}",
            meta.write_seconds,
            file_bytes as f64 / 1e6,
            db.resident_bytes() as f64 / 1024.0
        );
        anyhow::ensure!(contiguous, "an arena's overflow chain is not consecutive");

        for &probe in &[4usize, 8, 32] {
            // Probe width is swept only at the centroid count section 15.2 settled
            // on; at the others it is held at 8 so the k comparison is clean.
            if ks.len() > 1 && k != 1024 && probe != 8 {
                continue;
            }
            for &rerank in &[0usize, 100] {
                let mut ranks = Vec::with_capacity(n_q);
                let (mut pages, mut runs, mut hops) = (Vec::new(), Vec::new(), Vec::new());
                let mut stage = [[0f64; 2]; 3]; // [stage][pages, payload bytes]
                let mut cands = 0f64;
                // Sampled through the loop so the contention flag is an observation
                // rather than an assertion; see `annlite_fts5::cpu::LoadWitness`.
                let mut witness = annlite_fts5::cpu::LoadWitness::new();
                let cpu0 = cpu_seconds();
                for qi in 0..n_q {
                    if qi % 25 == 0 {
                        witness.sample();
                    }
                    let res = late_search(&conn, &db, &queries[qi], probe, docs.len(), rerank)?;
                    let pos = res.results.iter().position(|x| x.0 as usize == gold[qi]);
                    ranks.push(pos.map(|p| p + 1).unwrap_or(ABSENT));
                    let c = &res.cost;
                    pages.push(c.distinct_pages as f64);
                    runs.push(c.contiguous_runs as f64);
                    hops.push(c.hops as f64);
                    cands += c.candidates as f64;
                    for (i, s) in [&c.postings, &c.centroid, &c.rerank].iter().enumerate() {
                        stage[i][0] += s.distinct_pages as f64;
                        stage[i][1] += s.bytes as f64;
                    }
                }
                witness.sample();
                let cpu_ms = (cpu_seconds() - cpu0) * 1000.0 / n_q as f64;
                let q = metrics(&ranks);
                let mut sp = pages.clone();
                sp.sort_by(f64::total_cmp);
                let mut sr = runs.clone();
                sr.sort_by(f64::total_cmp);
                let f = n_q as f64;
                let index_bytes = db.index_bytes(&conn, rerank > 0)?;

                println!(
                    "{k:>6} {probe:>6} {rerank:>7} {:>8.3} {:>8.3} {:>8.3} {:>9.1} {:>9.1} \
                     {:>9.1} {:>9.1} {:>8.1} {:>6.1} {cpu_ms:>8.1}",
                    q["success@1"].as_f64().unwrap(),
                    q["success@10"].as_f64().unwrap(),
                    q["mrr@10"].as_f64().unwrap(),
                    stage[0][0] / f,
                    stage[1][0] / f,
                    stage[2][0] / f,
                    mean(&pages),
                    mean(&runs),
                    mean(&hops)
                );

                let note = format!(
                    "k-means build {build_wall:.1}s wall / {build_cpu:.1}s cpu (rayon, so cpu \
                     exceeds wall); write {:.1}s. SQLite arenas + resident offsets directory \
                     ({} bytes: {} centroid table, \
                     {} doc offsets, {} posting offsets). Stage bytes are the payload of the byte \
                     ranges requested; a page-granular client transfers pages*{PAGE_BYTES} \
                     instead, {:.0} B/query here. index_bytes counts late_meta+late_postings+\
                     late_codes{}, from dbstat; whole file is {file_bytes} bytes. \
                     Candidate pool {:.0} of {} documents ({:.1}%), so the inverted list prunes \
                     almost nothing at this scale (RESEARCH_LOG 15.3): stage 2 touches {:.0} of \
                     the centroid-code arena's {} pages, i.e. it is a full scan and the arena would be \
                     better made resident. Arena overflow chains are \
                     consecutive, so each stage's runs are near 1 per contiguous span. \
                     cpu_ms_per_query is /proc/self/stat utime+stime, not wall clock.{}",
                    meta.write_seconds,
                    db.resident_bytes(),
                    db.centroids.len() * 4,
                    db.doc_offsets.len() * 4,
                    db.posting_offsets.len() * 4,
                    mean(&pages) * PAGE_BYTES as f64,
                    if rerank > 0 { "+late_tokens" } else { "" },
                    cands / f,
                    docs.len(),
                    cands / f / docs.len() as f64 * 100.0,
                    stage[1][0] / f,
                    db.codes_arena.pages(),
                    match target.ceiling {
                        Some(c) => format!(" success@1 ceiling on this query set is {c}."),
                        // The word corpora have no published ceiling: their gold is one
                        // specific source document by construction, and nothing here
                        // establishes how often another document is equally correct.
                        None => String::new(),
                    },
                );
                writeln!(
                    out,
                    "{}",
                    serde_json::json!({
                        "record": "system", "corpus": target.label, "docs": docs.len(), "queries": n_q,
                        "system": "late",
                        "config": format!("k={k}/probe={probe}/rerank={rerank}"),
                        "build_seconds": build_wall,
                        "index_bytes": index_bytes,
                        "bytes_per_doc": index_bytes as f64 / docs.len() as f64,
                        "quality": q,
                        "pages": {
                            "mean": mean(&pages), "median": pct(&sp, 0.5), "p95": pct(&sp, 0.95),
                        },
                        "requests": { "mean": mean(&runs), "median": pct(&sr, 0.5) },
                        "stages": {
                            "postings": {"pages": stage[0][0] / f, "bytes": stage[0][1] / f},
                            "centroid": {"pages": stage[1][0] / f, "bytes": stage[1][1] / f},
                            "rerank":   {"pages": stage[2][0] / f, "bytes": stage[2][1] / f},
                        },
                        "hops": mean(&hops),
                        "cpu_ms_per_query": cpu_ms,
                        "contended": witness.contended(),
                        "peak_runnable": witness.peak_runnable,
                        "load_1min": witness.load_1min,
                        "notes": note,
                    })
                )?;
                out.flush()?;
            }
        }
    }
    eprintln!("\nresults -> {}", out_path.display());
    Ok(())
}
