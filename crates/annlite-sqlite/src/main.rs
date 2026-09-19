//! Build annlite indexes into SQLite and measure what a network client would pay.
//!
//! The experiment this binary runs is the project's central one. A Vamana graph and
//! a PQ codebook are built once, then written out repeatedly under different node
//! orderings. Graph, codes, queries and search parameters are held identical across
//! orderings, so any difference in pages touched is attributable to node numbering
//! alone -- which is the claim being tested.

use annlite_core::layout::{bfs_order, cluster_order, Ordering, Permutation};
use annlite_core::pq::ProductQuantizer;
use annlite_core::vamana::{Vamana, VamanaParams};
use annlite_core::vectors::{exact_top_k, Vectors};
use annlite_sqlite::search::{search, Index};
use annlite_sqlite::store::{node_page_stats, write_index};
use anyhow::Result;
use clap::Parser;
use rusqlite::Connection;
use std::io::Write;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "annlite-sqlite", about = "Build and measure annlite indexes in SQLite")]
struct Cli {
    #[arg(long)]
    docs: PathBuf,
    #[arg(long)]
    queries: PathBuf,
    #[arg(long, default_value = "10k")]
    scale: String,
    #[arg(long, default_value_t = 384)]
    dim: usize,
    /// PQ subquantizers; also bytes per document code.
    #[arg(long, default_value_t = 64)]
    m: usize,
    /// Vamana out-degree.
    #[arg(long, default_value_t = 32)]
    r: usize,
    #[arg(long, default_value_t = 1.1)]
    alpha: f32,
    #[arg(long, default_value_t = 100)]
    l_build: usize,
    /// Cap on documents, for quick runs.
    #[arg(long, default_value_t = 0)]
    limit: usize,
    #[arg(long, default_value_t = 200)]
    n_queries: usize,
    #[arg(long, default_value = "data/db")]
    db_dir: PathBuf,
    #[arg(long, default_value = "bench/results")]
    out_dir: PathBuf,
    /// Train PQ on at most this many vectors; 0 uses all of them.
    #[arg(long, default_value_t = 100_000)]
    pq_train: usize,
    /// Beam widths to sweep. Beam trades bytes for round-trips: a whole frontier is
    /// one batched fetch, so widening it cuts dependent hops while reading more
    /// records. That trade runs the opposite way from an in-memory index.
    #[arg(long, value_delimiter = ',', default_value = "1,4,16")]
    beams: Vec<usize>,
    #[arg(long, value_delimiter = ',', default_value = "32,64,128")]
    search_l: Vec<usize>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    std::fs::create_dir_all(&cli.db_dir)?;
    std::fs::create_dir_all(&cli.out_dir)?;

    let mut docs = Vectors::load(&cli.docs, cli.dim)?;
    if cli.limit > 0 && cli.limit < docs.len() {
        docs.data.truncate(cli.limit * cli.dim);
    }
    let queries = Vectors::load(&cli.queries, cli.dim)?;
    let n_q = cli.n_queries.min(queries.len());
    eprintln!("docs {} queries {n_q} dim {}", docs.len(), cli.dim);

    eprintln!("computing exact ground truth...");
    let t0 = std::time::Instant::now();
    let gold: Vec<Vec<u32>> = (0..n_q)
        .map(|i| exact_top_k(&docs, queries.row(i), 100).iter().map(|x| x.0).collect())
        .collect();
    let exact_ms = t0.elapsed().as_secs_f64() * 1000.0 / n_q as f64;
    eprintln!("  exact brute force: {exact_ms:.2} ms/query");

    // PQ trains on the first bulk insert, per the brief. A sample suffices: 256
    // centroids per 6-dimensional subspace are well determined by 100k points, and
    // training cost is linear in the sample.
    let train = if cli.pq_train > 0 && cli.pq_train < docs.len() {
        Vectors { data: docs.data[..cli.pq_train * cli.dim].to_vec(), dim: cli.dim }
    } else {
        Vectors { data: docs.data.clone(), dim: cli.dim }
    };
    eprintln!("training PQ (m={}) on {} vectors...", cli.m, train.len());
    let t0 = std::time::Instant::now();
    let pq = ProductQuantizer::train(&train, cli.m, 20, 0xA11CE)?;
    let pq_train_s = t0.elapsed().as_secs_f64();
    let codes = pq.encode_all(&docs);
    eprintln!("  trained in {pq_train_s:.1}s, {} bytes/doc", cli.m);

    eprintln!("building Vamana (R={} alpha={} L={})...", cli.r, cli.alpha, cli.l_build);
    let t0 = std::time::Instant::now();
    let graph = Vamana::build(
        &docs,
        VamanaParams { r: cli.r, l_build: cli.l_build, alpha: cli.alpha, seed: 0xDA7A },
    );
    let build_s = t0.elapsed().as_secs_f64();
    let gs = graph.stats();
    eprintln!("  built in {build_s:.1}s, mean degree {:.1}, orphans {}", gs.mean_degree, gs.orphans);

    let out_path = cli.out_dir.join(format!("ann-{}.jsonl", cli.scale));
    let mut out = std::fs::File::create(&out_path)?;
    writeln!(out, "{}", serde_json::json!({
        "record": "meta", "scale": cli.scale, "docs": docs.len(), "queries": n_q,
        "dim": cli.dim, "m": cli.m, "r": cli.r, "alpha": cli.alpha, "l_build": cli.l_build,
        "pq_train_vectors": train.len(), "pq_train_seconds": pq_train_s,
        "vamana_build_seconds": build_s, "mean_degree": gs.mean_degree, "orphans": gs.orphans,
        "exact_ms_per_query": exact_ms,
    }))?;

    println!("\n{:>10} {:>5} {:>5} {:>6} {:>8} {:>9} {:>8} {:>7} {:>7} {:>8}",
             "ordering", "L", "beam", "rrank", "recall10", "nodes", "pages", "runs", "hops", "ms");
    println!("{}", "-".repeat(90));

    for ordering in [Ordering::Identity, Ordering::Bfs, Ordering::Cluster] {
        let perm = match ordering {
            Ordering::Identity => Permutation::identity(docs.len()),
            Ordering::Bfs => bfs_order(docs.len(), graph.medoid, |n| graph.neighbors(n).to_vec()),
            Ordering::Cluster => {
                // Roughly one cluster per page's worth of records, so a cluster is
                // about the granularity a single fetch can deliver.
                let k = (docs.len() / 64).clamp(2, 4096);
                cluster_order(&docs, k, 10, 19)
            }
        };

        let db_path = cli.db_dir.join(format!("ann-{}-{:?}.db", cli.scale, ordering).to_lowercase());
        let _ = std::fs::remove_file(&db_path);
        let mut conn = Connection::open(&db_path)?;
        let t0 = std::time::Instant::now();
        let meta = write_index(&mut conn, &graph, &pq, &codes, &docs, &perm, ordering, None)?;
        let write_s = t0.elapsed().as_secs_f64();
        let (node_pages, per_page) = node_page_stats(&conn)?;
        let db_bytes = std::fs::metadata(&db_path)?.len();

        let idx = Index::open(&conn)?;
        writeln!(out, "{}", serde_json::json!({
            "record": "index", "ordering": format!("{ordering:?}"), "write_seconds": write_s,
            "db_bytes": db_bytes, "node_leaf_pages": node_pages, "records_per_page": per_page,
            "record_bytes": idx.fmt.len(), "count": meta.count,
        }))?;

        for &l in &cli.search_l {
          for &beam in &cli.beams {
            for &rerank in &[0usize, 100] {
                let t0 = std::time::Instant::now();
                let (mut hits, mut nodes, mut pages, mut runs, mut hops, mut vecs, mut vpages) =
                    (0usize, 0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
                for qi in 0..n_q {
                    let res = search(&conn, &idx, queries.row(qi), 10, l, beam, rerank)?;
                    let ids: Vec<u32> = res.results.iter().map(|x| x.0).collect();
                    // Gold ids are in the original numbering; map through the permutation.
                    hits += gold[qi][..10]
                        .iter()
                        .filter(|g| ids.contains(&perm.new_id_of[**g as usize]))
                        .count();
                    nodes += res.cost.nodes_read;
                    pages += res.cost.distinct_pages;
                    hops += res.cost.hops;
                    vecs += res.cost.vectors_read;
                    vpages += res.cost.rerank_pages;
                    runs += res.cost.contiguous_runs_estimate();
                }
                let ms = t0.elapsed().as_secs_f64() * 1000.0 / n_q as f64;
                let f = n_q as f64;
                let recall = hits as f64 / (n_q * 10) as f64;
                println!("{:>10} {l:>5} {beam:>5} {rerank:>6} {recall:>8.3} {:>9.1} {:>8.1} {:>7.1} {:>7.1} {ms:>8.2}",
                         format!("{ordering:?}"), nodes as f64 / f, pages as f64 / f,
                         runs as f64 / f, hops as f64 / f);
                writeln!(out, "{}", serde_json::json!({
                    "record": "query_set", "ordering": format!("{ordering:?}"),
                    "l": l, "beam": beam, "rerank": rerank, "recall_at_10": recall,
                    "mean_nodes_read": nodes as f64 / f, "mean_distinct_pages": pages as f64 / f,
                    "mean_contiguous_runs": runs as f64 / f, "mean_hops": hops as f64 / f,
                    "mean_vectors_read": vecs as f64 / f, "mean_rerank_pages": vpages as f64 / f,
                    "ms_per_query": ms,
                }))?;
            }
          }
        }
    }
    eprintln!("\nresults -> {}", out_path.display());
    Ok(())
}
