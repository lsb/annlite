//! Per-stage request counts for an already-built late index.
//!
//! `late_pages` emits the aggregate `requests` figure, because that is the field the
//! three-way comparison joins on. But the interesting question -- which stage a
//! client's round-trips are actually spent in -- needs the coalesced run count broken
//! out the same way pages and bytes are. This reads the databases `late_pages` wrote
//! and reports that, without refitting centroids: k-means is the expensive part and
//! it is deterministic, so re-running it to learn one more column would be waste.

use annlite_core::late::MultiVector;
use annlite_sqlite::late_search::{late_search, LateDb};
use annlite_sqlite::late_store::PAGE_BYTES;
use anyhow::Result;
use rusqlite::Connection;
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

fn main() -> Result<()> {
    let dim = 48;
    let n_q: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(500);
    let emb = Path::new("data/embeddings");
    let ql = read_lengths(&emb.join("code-late-qlengths.i32"))?;
    let queries = read_multi(&emb.join("code-late-q.f32"), &ql, dim)?;
    let n_q = n_q.min(queries.len());

    println!(
        "\n{:>6} {:>7} {:>10} {:>8} {:>8} {:>10} {:>8} {:>10} {:>8} {:>8}",
        "k", "rerank", "stage", "pages", "runs", "payload B", "page B", "arena pg", "hops", "resident"
    );
    println!("{}", "-".repeat(100));

    for &k in &[512usize, 1024, 2048] {
        let path = format!("data/db/late-code-k{k}.db");
        if !Path::new(&path).exists() {
            eprintln!("{path} not built; run `make late-pages` first");
            continue;
        }
        let conn = Connection::open(&path)?;
        let db = LateDb::open(&conn)?;
        let arena_pages =
            [db.postings_arena.pages(), db.codes_arena.pages(), db.tokens_arena.pages()];
        for &rerank in &[0usize, 100] {
            let mut acc = [[0f64; 4]; 3]; // [stage][pages, runs, payload bytes, _]
            let mut hops = 0f64;
            for q in queries.iter().take(n_q) {
                let res = late_search(&conn, &db, q, 8, db.count, rerank)?;
                hops += res.cost.hops as f64;
                for (i, s) in
                    [&res.cost.postings, &res.cost.centroid, &res.cost.rerank].iter().enumerate()
                {
                    acc[i][0] += s.distinct_pages as f64;
                    acc[i][1] += s.contiguous_runs as f64;
                    acc[i][2] += s.bytes as f64;
                }
            }
            let f = n_q as f64;
            for (i, name) in ["postings", "centroid", "rerank"].iter().enumerate() {
                println!(
                    "{k:>6} {rerank:>7} {name:>10} {:>8.1} {:>8.1} {:>10.0} {:>8.0} {:>10} {:>8} {:>8}",
                    acc[i][0] / f,
                    acc[i][1] / f,
                    acc[i][2] / f,
                    acc[i][0] / f * PAGE_BYTES as f64,
                    arena_pages[i],
                    if i == 0 { format!("{:.1}", hops / f) } else { String::new() },
                    if i == 0 { format!("{}", db.resident_bytes()) } else { String::new() },
                );
            }
        }
    }
    Ok(())
}
