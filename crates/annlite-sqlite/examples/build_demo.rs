//! Build a single self-contained demo database: graph, codes, vectors and the
//! document text, ordered breadth-first for page locality.

use annlite_core::layout::{bfs_order, Ordering};
use annlite_core::pq::ProductQuantizer;
use annlite_core::vamana::{Vamana, VamanaParams};
use annlite_core::vectors::Vectors;
use annlite_sqlite::store::write_index;
use rusqlite::Connection;
use std::io::BufRead;
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let n: usize = std::env::args().nth(1).unwrap_or_else(|| "2000".into()).parse()?;
    let out = std::env::args().nth(2).unwrap_or_else(|| "web/demo/annlite-demo.db".into());
    let dim = 384usize;
    let (m, r) = (32usize, 24usize);

    let all = Vectors::load(Path::new("data/embeddings/docs-10k.f32"), dim)?;
    anyhow::ensure!(all.len() >= n, "only {} embeddings available", all.len());
    let docs = Vectors { data: all.data[..n * dim].to_vec(), dim };

    let text: Vec<String> = std::io::BufReader::new(std::fs::File::open("data/corpus/docs-10k.txt")?)
        .lines()
        .take(n)
        .collect::<Result<_, _>>()?;

    let pq = ProductQuantizer::train(&docs, m, 20, 0xA11CE)?;
    let codes = pq.encode_all(&docs);
    let graph = Vamana::build(&docs, VamanaParams { r, l_build: 80, alpha: 1.1, seed: 0xDA7A });
    let perm = bfs_order(n, graph.medoid, |x| graph.neighbors(x).to_vec());

    if let Some(parent) = Path::new(&out).parent() {
        std::fs::create_dir_all(parent)?;
    }
    let _ = std::fs::remove_file(&out);
    let mut conn = Connection::open(&out)?;
    let meta = write_index(&mut conn, &graph, &pq, &codes, &docs, &perm, Ordering::Bfs, Some(&text))?;
    // A demo database is read-only once built, so reclaim the free pages: a client
    // fetching ranges over HTTP would otherwise pay for holes it never reads.
    conn.execute_batch("VACUUM;")?;

    // Also emit the graph and codes as flat sidecar files.
    //
    // The demo showed that sql.js-httpvfs escalates its read-ahead to megabyte
    // requests and pulls the whole database on the first query, which makes any
    // page-locality work in the index invisible. These files are what the fixed-size
    // record format was designed for: record i begins at byte i * record_bytes, so a
    // client can range-fetch exactly the bytes it needs and nothing else. A CDN
    // serves three static files as readily as one.
    let rec_len = m + 2 + r * 4;
    let dir = Path::new(&out).parent().unwrap_or(Path::new("."));
    let mut graph_bin = vec![0u8; n * rec_len];
    let mut codes_bin = vec![0u8; n * m];
    for new_id in 0..n {
        let old = perm.old_id_of[new_id] as usize;
        let neighbors: Vec<u32> = graph
            .neighbors(old as u32)
            .iter()
            .map(|&o| perm.new_id_of[o as usize])
            .collect();
        let rec = annlite_sqlite::format::RecordFormat { m, r }
            .encode(&codes[old * m..(old + 1) * m], &neighbors);
        graph_bin[new_id * rec_len..(new_id + 1) * rec_len].copy_from_slice(&rec);
        codes_bin[new_id * m..(new_id + 1) * m]
            .copy_from_slice(&codes[old * m..(old + 1) * m]);
    }
    std::fs::write(dir.join("annlite-graph.bin"), &graph_bin)?;
    std::fs::write(dir.join("annlite-codes.bin"), &codes_bin)?;
    println!("  sidecars: annlite-graph.bin {} B, annlite-codes.bin {} B",
             graph_bin.len(), codes_bin.len());

    let bytes = std::fs::metadata(&out)?.len();
    println!(
        "{out}: {} docs, {} bytes ({:.1} KB), m={m} r={r}, medoid={}",
        meta.count, bytes, bytes as f64 / 1024.0, meta.medoid_new_id
    );
    println!("  record_bytes={} code_blob={} bytes", m + 2 + r * 4, n * m);
    Ok(())
}
