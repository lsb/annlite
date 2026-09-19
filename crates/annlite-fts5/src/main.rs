//! Runs the FTS5 baseline at one corpus scale and writes JSONL results.
//!
//! Scale is a CLI argument rather than a loop over all three because the 1M-document
//! run takes minutes and should be startable, interruptible and re-runnable on its own.

use annlite_fts5::{db, index, metrics, pages, query, scale_by_name};
use anyhow::{Context, Result};
use clap::Parser;
use metrics::Dist;
use rusqlite::Connection;
use serde::Serialize;
use serde_json::json;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "annlite-fts5", about = "FTS5 baseline: build, latency, quality, page access")]
struct Cli {
    /// Corpus scale: 100, 10k or 1m.
    #[arg(long)]
    scale: String,
    #[arg(long, default_value = "data")]
    data_dir: PathBuf,
    #[arg(long, default_value = "bench/results")]
    out_dir: PathBuf,
    /// Result depth. 100 is the deepest cutoff quality is reported at, so the same
    /// run serves success@1, @10 and @100.
    #[arg(long, default_value_t = 100)]
    limit: usize,
    /// Use only the first N queries (development aid; the reported run uses all).
    #[arg(long)]
    sample: Option<usize>,
    /// Queries in the fresh-connection cross-check of the page counter.
    #[arg(long, default_value_t = 50)]
    crosscheck: usize,
    /// Skip the `dbstat` scan, which reads the whole database to attribute pages to
    /// shadow tables.
    #[arg(long)]
    no_dbstat: bool,
    /// Reuse an existing database instead of rebuilding (skips the build measurement).
    #[arg(long)]
    reuse_db: bool,
}

/// What one query cost, from the timed pass.
#[derive(Serialize, Clone, Debug)]
struct LatencyOutcome {
    qid: usize,
    kind: String,
    k: usize,
    latency_us: f64,
    n_results: usize,
}

/// What one query retrieved, from the untimed quality pass.
#[derive(Clone, Debug)]
struct QualityOutcome {
    qid: usize,
    kind: String,
    k: usize,
    /// 1-based rank of the gold document, if it appeared within `limit`.
    rank: Option<usize>,
    /// Total documents matching the query, independent of `limit`.
    match_count: i64,
    /// Best (most negative) BM25 score returned, if any row matched.
    best_score: Option<f64>,
    /// How many of the returned rows share that best score, capped at `limit`.
    tied_at_best: usize,
}

fn sql_for(limit_param: bool) -> String {
    let t = index::TABLE;
    if limit_param {
        format!("SELECT rowid, bm25({t}) FROM {t} WHERE {t} MATCH ?1 ORDER BY bm25({t}) LIMIT ?2")
    } else {
        format!("SELECT count(*) FROM {t} WHERE {t} MATCH ?1")
    }
}

fn run_latency(conn: &Connection, queries: &[query::Query], limit: usize) -> Result<Vec<LatencyOutcome>> {
    let mut stmt = conn.prepare(&sql_for(true))?;
    let mut out = Vec::with_capacity(queries.len());
    for q in queries {
        let Some(expr) = query::match_expression(&q.text) else { continue };
        let t0 = Instant::now();
        let mut rows = stmt.query(rusqlite::params![expr, limit as i64])?;
        let mut n = 0usize;
        while let Some(r) = rows.next()? {
            let _: i64 = r.get(0)?;
            let _: f64 = r.get(1)?;
            n += 1;
        }
        let latency_us = t0.elapsed().as_secs_f64() * 1e6;
        out.push(LatencyOutcome { qid: q.qid, kind: q.kind.clone(), k: q.k, latency_us, n_results: n });
    }
    Ok(out)
}

fn run_quality(conn: &Connection, queries: &[query::Query], limit: usize) -> Result<Vec<QualityOutcome>> {
    let mut top = conn.prepare(&sql_for(true))?;
    let mut count = conn.prepare(&sql_for(false))?;
    let mut out = Vec::with_capacity(queries.len());
    for q in queries {
        let Some(expr) = query::match_expression(&q.text) else { continue };
        let mut rank = None;
        let mut best_score = None;
        let mut tied = 0usize;
        let mut rows = top.query(rusqlite::params![expr, limit as i64])?;
        let mut i = 0usize;
        while let Some(r) = rows.next()? {
            let rowid: i64 = r.get(0)?;
            let score: f64 = r.get(1)?;
            i += 1;
            if i == 1 {
                best_score = Some(score);
            }
            // BM25 scores here are exact sums of identical per-term contributions, so
            // equality is the right test; a tolerance would only blur the tie count.
            if Some(score) == best_score {
                tied += 1;
            }
            if rank.is_none() && q.source_doc == Some(rowid as usize) {
                rank = Some(i);
            }
        }
        drop(rows);
        let match_count: i64 = count.query_row(rusqlite::params![expr], |r| r.get(0))?;
        out.push(QualityOutcome {
            qid: q.qid,
            kind: q.kind.clone(),
            k: q.k,
            rank,
            match_count,
            best_score,
            tied_at_best: tied,
        });
    }
    Ok(out)
}

/// Group label used in the summary records: overall, per kind, and per (kind, k).
fn groups(queries: &[query::Query]) -> Vec<(String, Box<dyn Fn(&str, usize) -> bool>)> {
    let mut ks: Vec<usize> = queries.iter().map(|q| q.k).collect();
    ks.sort_unstable();
    ks.dedup();
    let mut g: Vec<(String, Box<dyn Fn(&str, usize) -> bool>)> =
        vec![("all".to_string(), Box::new(|_, _| true))];
    for kind in ["known_item", "random"] {
        g.push((format!("kind={kind}"), Box::new(move |kk: &str, _| kk == kind)));
        for k in ks.clone() {
            g.push((
                format!("kind={kind},k={k}"),
                Box::new(move |kk: &str, qk: usize| kk == kind && qk == k),
            ));
        }
    }
    g
}

struct Out {
    file: std::io::BufWriter<std::fs::File>,
}

impl Out {
    fn write(&mut self, v: serde_json::Value) -> Result<()> {
        writeln!(self.file, "{}", serde_json::to_string(&v)?)?;
        Ok(())
    }
}

fn fmt_dist(label: &str, d: &Dist, unit: &str) -> String {
    format!(
        "{label:<22} n={:<5} mean={:>10.2} med={:>10.2} p95={:>10.2} p99={:>10.2} max={:>10.2} {unit}",
        d.n, d.mean, d.median, d.p95, d.p99, d.max
    )
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let scale = scale_by_name(&cli.scale)
        .with_context(|| format!("unknown scale {:?}; expected one of 100, 10k, 1m", cli.scale))?;

    let corpus = cli.data_dir.join(format!("corpus/docs-{}.txt", scale.name));
    let qpath = cli.data_dir.join(format!("corpus/queries-{}.jsonl", scale.name));
    let manifest = cli.data_dir.join(format!("corpus/docs-{}.manifest.json", scale.name));
    let db_path = cli.data_dir.join(format!("db/fts5-{}.db", scale.name));
    std::fs::create_dir_all(&cli.out_dir)?;
    let out_path = cli.out_dir.join(format!("fts5-{}.jsonl", scale.name));
    let mut out = Out { file: std::io::BufWriter::new(std::fs::File::create(&out_path)?) };

    let mut queries = query::load(&qpath)?;
    if let Some(n) = cli.sample {
        queries.truncate(n);
    }
    let with_dbstat = !cli.no_dbstat;

    // ---- build -----------------------------------------------------------------
    let build = if cli.reuse_db {
        eprintln!("[{}] reusing existing database {}", scale.name, db_path.display());
        None
    } else {
        eprintln!("[{}] building FTS5 index from {}", scale.name, corpus.display());
        Some(index::build(&db_path, &corpus, with_dbstat)?)
    };

    let conn = Connection::open(&db_path)?;
    db::assert_fts5(&conn)?;
    let meta = json!({
        "record": "meta",
        "scale": scale.name,
        "n_docs_expected": scale.n_docs,
        "timestamp_utc": db::utc_now(&conn)?,
        "sqlite_version": db::sqlite_version(&conn)?,
        "page_size": db::PAGE_SIZE,
        "result_limit": cli.limit,
        "n_queries": queries.len(),
        "corpus": corpus.display().to_string(),
        "corpus_manifest": std::fs::read_to_string(&manifest).ok()
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok()),
        "db_path": db_path.display().to_string(),
        "fts5_table": format!("CREATE VIRTUAL TABLE {} USING fts5(body)", index::TABLE),
        "match_expression": "terms OR-ed, each double-quoted; ORDER BY bm25() ASC",
    });
    out.write(meta.clone())?;
    println!("== FTS5 baseline, scale {} ==", scale.name);
    println!("sqlite {}  page_size {}  queries {}  limit {}",
        db::sqlite_version(&conn)?, db::PAGE_SIZE, queries.len(), cli.limit);

    if let Some(b) = &build {
        out.write(json!({ "record": "build", "scale": scale.name, "build": b }))?;
        println!(
            "\nbuild: {} docs in {:.2}s = {:.0} docs/s ({:.1} MiB/s of corpus text)",
            b.n_docs, b.build_secs, b.docs_per_sec, b.mib_per_sec
        );
        println!(
            "       db {} bytes, {} pages of {} B, {:.1} B/doc, {:.2}x corpus size",
            b.stats.file_bytes, b.stats.page_count, b.stats.page_size, b.bytes_per_doc,
            b.stats.file_bytes as f64 / b.corpus_bytes as f64
        );
        if let Some(t) = &b.stats.per_table {
            for t in t {
                println!("       {:<16} {:>10} pages {:>14} payload bytes", t.name, t.pages, t.payload_bytes);
            }
        }
    }

    let segs_before = index::segment_count(&conn)?;
    drop(conn);

    // ---- measurement of one phase ----------------------------------------------
    let run_phase = |phase: &str, out: &mut Out| -> Result<()> {
        let conn = Connection::open(&db_path)?;
        conn.pragma_update(None, "cache_size", -262_144)?;
        // Warm the pager cache first so the timed pass measures steady-state query
        // cost rather than the first-touch cost of the index; the cold-cache cost is
        // what the page-access phase below reports, in pages rather than in seconds.
        let _ = run_latency(&conn, &queries, cli.limit)?;
        let lat = run_latency(&conn, &queries, cli.limit)?;
        let qual = run_quality(&conn, &queries, cli.limit)?;
        drop(conn);

        println!("\n-- {phase}: query latency (warm cache, µs) --");
        for (label, pred) in groups(&queries) {
            let v: Vec<f64> = lat.iter().filter(|o| pred(&o.kind, o.k)).map(|o| o.latency_us).collect();
            if v.is_empty() {
                continue;
            }
            let d = Dist::of(&v);
            let res: Vec<f64> = lat.iter().filter(|o| pred(&o.kind, o.k)).map(|o| o.n_results as f64).collect();
            out.write(json!({
                "record": "latency", "scale": scale.name, "phase": phase,
                "group": label, "latency_us": d, "n_results": Dist::of(&res),
            }))?;
            println!("{}", fmt_dist(&label, &d, "µs"));
        }

        let ki: Vec<&QualityOutcome> = qual.iter().filter(|q| q.kind == "known_item").collect();
        let ranks: Vec<Option<usize>> = ki.iter().map(|q| q.rank).collect();
        let overall = metrics::known_item_quality(&ranks);
        let mut per_k = serde_json::Map::new();
        let mut ks: Vec<usize> = ki.iter().map(|q| q.k).collect();
        ks.sort_unstable();
        ks.dedup();
        for k in &ks {
            let r: Vec<Option<usize>> = ki.iter().filter(|q| q.k == *k).map(|q| q.rank).collect();
            let tied: Vec<f64> = ki.iter().filter(|q| q.k == *k).map(|q| q.tied_at_best as f64).collect();
            let mc: Vec<f64> = ki.iter().filter(|q| q.k == *k).map(|q| q.match_count as f64).collect();
            per_k.insert(
                k.to_string(),
                json!({
                    "quality": metrics::known_item_quality(&r),
                    "tied_at_best_score": Dist::of(&tied),
                    "match_count": Dist::of(&mc),
                }),
            );
        }
        let rnd: Vec<&QualityOutcome> = qual.iter().filter(|q| q.kind == "random").collect();
        let rnd_counts: Vec<f64> = rnd.iter().map(|q| q.match_count as f64).collect();
        let rnd_scores: Vec<f64> = rnd.iter().filter_map(|q| q.best_score).collect();
        let rnd_zero = rnd.iter().filter(|q| q.match_count == 0).count();
        let mut rnd_per_k = serde_json::Map::new();
        for k in &ks {
            let c: Vec<f64> = rnd.iter().filter(|q| q.k == *k).map(|q| q.match_count as f64).collect();
            let s: Vec<f64> = rnd.iter().filter(|q| q.k == *k).filter_map(|q| q.best_score).collect();
            rnd_per_k.insert(
                k.to_string(),
                json!({
                    "match_count": Dist::of(&c),
                    "best_bm25": Dist::of(&s),
                    "zero_result_queries": rnd.iter().filter(|q| q.k == *k && q.match_count == 0).count(),
                }),
            );
        }
        out.write(json!({
            "record": "quality", "scale": scale.name, "phase": phase,
            "known_item": { "overall": overall, "per_k": per_k },
            "random": {
                "n_queries": rnd.len(),
                "zero_result_queries": rnd_zero,
                "match_count": Dist::of(&rnd_counts),
                "best_bm25": Dist::of(&rnd_scores),
                "per_k": rnd_per_k,
            },
        }))?;

        println!("\n-- {phase}: known-item quality --");
        println!("{:<6} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>12} {:>12}",
            "k", "succ@1", "succ@10", "succ@100", "mrr@1", "mrr@10", "mrr@100", "tied@best", "matches");
        for k in &ks {
            let v = &per_k[&k.to_string()];
            let q = &v["quality"];
            println!("{:<6} {:>9.3} {:>9.3} {:>9.3} {:>9.3} {:>9.3} {:>9.3} {:>12.1} {:>12.1}",
                k,
                q["success_at"][0][1].as_f64().unwrap(), q["success_at"][1][1].as_f64().unwrap(),
                q["success_at"][2][1].as_f64().unwrap(),
                q["mrr_at"][0][1].as_f64().unwrap(), q["mrr_at"][1][1].as_f64().unwrap(),
                q["mrr_at"][2][1].as_f64().unwrap(),
                v["tied_at_best_score"]["median"].as_f64().unwrap(),
                v["match_count"]["median"].as_f64().unwrap());
        }
        println!("overall success@1/@10/@100 = {:.3} / {:.3} / {:.3}   MRR@100 = {:.4}   never retrieved: {}",
            overall.success_at[0].1, overall.success_at[1].1, overall.success_at[2].1,
            overall.mrr_at[2].1, overall.n_unretrieved);
        println!("\n-- {phase}: random queries --");
        println!("{}", fmt_dist("match_count", &Dist::of(&rnd_counts), "docs"));
        println!("{}", fmt_dist("best bm25", &Dist::of(&rnd_scores), ""));
        println!("zero-result random queries: {} / {}", rnd_zero, rnd.len());

        // ---- page access --------------------------------------------------------
        let base = pages::open_prepare_baseline(&db_path, &sql_for(true))?;
        let po = pages::measure(&db_path, &queries, &sql_for(true), cli.limit, db::PAGE_SIZE as u64)?;
        let cross = pages::measure_fresh_connection(
            &db_path, &queries, &sql_for(true), cli.limit, db::PAGE_SIZE as u64, cli.crosscheck,
        )?;
        println!("\n-- {phase}: pages touched per query (cold pager cache, 4 KiB pages) --");
        println!("connection open + prepare alone: {} distinct pages, {} xRead calls",
            base.pages(db::PAGE_SIZE as u64).len(), base.read_calls());
        for (label, pred) in groups(&queries) {
            let sel: Vec<&pages::PageOutcome> = po.iter().filter(|o| pred(&o.kind, o.k)).collect();
            if sel.is_empty() {
                continue;
            }
            let dp = Dist::of(&sel.iter().map(|o| o.distinct_pages as f64).collect::<Vec<_>>());
            let runs = Dist::of(&sel.iter().map(|o| o.contiguous_runs as f64).collect::<Vec<_>>());
            let rc = Dist::of(&sel.iter().map(|o| o.read_calls as f64).collect::<Vec<_>>());
            let by = Dist::of(&sel.iter().map(|o| o.bytes_read as f64).collect::<Vec<_>>());
            let cm = Dist::of(&sel.iter().map(|o| o.cache_miss as f64).collect::<Vec<_>>());
            out.write(json!({
                "record": "pages", "scale": scale.name, "phase": phase, "group": label,
                "method": "xRead interception, PRAGMA shrink_memory before each query",
                "distinct_pages": dp, "contiguous_runs": runs, "read_calls": rc,
                "bytes_read": by, "cache_miss_dbstatus": cm,
            }))?;
            println!("{}", fmt_dist(&label, &dp, "pages"));
        }
        let agree = po.iter().filter(|o| o.cache_miss as usize == o.distinct_pages).count();
        let cross_d = Dist::of(&cross.iter().map(|&x| x as f64).collect::<Vec<_>>());
        out.write(json!({
            "record": "pages_crosscheck", "scale": scale.name, "phase": phase,
            "open_prepare_pages": base.pages(db::PAGE_SIZE as u64).len(),
            "open_prepare_read_calls": base.read_calls(),
            "cache_miss_equals_distinct_pages": agree,
            "n_queries": po.len(),
            "fresh_connection_per_query": { "sample": cross.len(), "distinct_pages": cross_d },
            "shrink_memory_same_queries": Dist::of(
                &po.iter().take(cross.len()).map(|o| o.distinct_pages as f64).collect::<Vec<_>>()),
        }))?;
        println!("cross-check: DBSTATUS_CACHE_MISS equals distinct pages on {}/{} queries",
            agree, po.len());
        println!("cross-check: fresh connection per query (first {}) median {:.1} pages vs {:.1} with shrink_memory",
            cross.len(), cross_d.median,
            Dist::of(&po.iter().take(cross.len()).map(|o| o.distinct_pages as f64).collect::<Vec<_>>()).median);

        // Per-query detail, so distributions can be recomputed without a re-run.
        let page_by_qid: std::collections::HashMap<usize, &pages::PageOutcome> =
            po.iter().map(|o| (o.qid, o)).collect();
        for (l, q) in lat.iter().zip(qual.iter()) {
            debug_assert_eq!(l.qid, q.qid, "latency and quality passes visited queries in different orders");
            let p = page_by_qid.get(&l.qid);
            out.write(json!({
                "record": "query", "scale": scale.name, "phase": phase,
                "qid": l.qid, "kind": l.kind, "k": l.k,
                "latency_us": l.latency_us, "n_results": l.n_results,
                "match_count": q.match_count, "rank": q.rank,
                "best_bm25": q.best_score, "tied_at_best": q.tied_at_best,
                "distinct_pages": p.map(|p| p.distinct_pages),
                "contiguous_runs": p.map(|p| p.contiguous_runs),
                "read_calls": p.map(|p| p.read_calls),
                "bytes_read": p.map(|p| p.bytes_read),
                "cache_miss": p.map(|p| p.cache_miss),
            }))?;
        }
        out.file.flush()?;
        Ok(())
    };

    run_phase("pre_optimize", &mut out)?;

    // ---- optimize ----------------------------------------------------------------
    eprintln!("[{}] running FTS5 optimize", scale.name);
    let opt = index::optimize(&db_path, with_dbstat)?;
    let conn = Connection::open(&db_path)?;
    let segs_after = index::segment_count(&conn)?;
    drop(conn);
    out.write(json!({
        "record": "optimize", "scale": scale.name,
        "optimize": opt, "segments_before": segs_before, "segments_after": segs_after,
    }))?;
    println!("\noptimize: {:.2}s, segments {} -> {}, db {} bytes / {} pages",
        opt.optimize_secs, segs_before, segs_after, opt.stats.file_bytes, opt.stats.page_count);
    if let Some(t) = &opt.stats.per_table {
        for t in t {
            println!("       {:<16} {:>10} pages {:>14} payload bytes", t.name, t.pages, t.payload_bytes);
        }
    }

    run_phase("post_optimize", &mut out)?;

    out.file.flush()?;
    println!("\nresults -> {}", out_path.display());
    Ok(())
}
