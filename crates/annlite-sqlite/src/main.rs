//! Build annlite indexes into SQLite and measure what a network client would pay.
//!
//! The experiment this binary runs is the project's central one. A Vamana graph and
//! a PQ codebook are built once, then written out repeatedly under different node
//! orderings. Graph, codes, queries and search parameters are held identical across
//! orderings, so any difference in pages touched is attributable to node numbering
//! alone -- which is the claim being tested.

use annlite_core::layout::{bfs_order, cluster_order_sampled, Ordering, Permutation};
use annlite_core::pq::ProductQuantizer;
use annlite_core::vamana::{Vamana, VamanaParams};
use annlite_core::vectors::{exact_top_k, Vectors};
use annlite_sqlite::search::{search, search_resident, Index, ResidentCodes};
use annlite_fts5::metrics::Dist;
use annlite_sqlite::store::{node_page_stats, table_page_stats, write_index, PAGE_BYTES};
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
    /// Batch size for the parallel graph build; 0 builds sequentially.
    #[arg(long, default_value_t = 1024)]
    build_batch: usize,
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
    /// Node orderings to compare: identity, bfs, cluster.
    #[arg(long, value_delimiter = ',', default_value = "identity,bfs,cluster")]
    orderings: Vec<String>,
    /// Fit cluster-ordering centroids on at most this many vectors.
    #[arg(long, default_value_t = 100_000)]
    cluster_train: usize,
    /// Query set with a `source_doc` per query, turning the sweep into a known-item
    /// benchmark alongside the recall-against-brute-force one.
    ///
    /// Recall@10 answers "does the graph find what brute force finds", which is a
    /// question about the index. success@k answers "is the one right document near the
    /// top", which is a question about retrieval, and is the only one comparable with
    /// the FTS5 baseline and the late-interaction runs.
    #[arg(long)]
    gold: Option<PathBuf>,
    /// Result depth. Quality is reported at 1, 10 and 100, so the default retrieves
    /// the deepest of those; note that success@100 is also bounded by `--search-l`,
    /// since the candidate list is never longer than L.
    #[arg(long, default_value_t = 100)]
    k: usize,
}

/// 1-based rank of `gold` among `ids`, or `None` if it was never returned.
///
/// A missed document counts in the denominator rather than being dropped: scoring
/// only the queries a system answered would flatter every system by its own failures.
fn rank_of(ids: &[u32], gold: u32) -> Option<usize> {
    ids.iter().position(|&x| x == gold).map(|i| i + 1)
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

    // Gold ids are in the original document numbering, the same numbering the
    // embeddings were written in, so they line up with rows of `docs` before any
    // reordering. Each ordering maps them through its own permutation below.
    let known: Option<Vec<u32>> = match &cli.gold {
        None => None,
        Some(path) => {
            let qs = annlite_fts5::query::load(path)?;
            anyhow::ensure!(
                qs.len() >= n_q,
                "{} holds {} queries, fewer than the {n_q} being measured",
                path.display(), qs.len()
            );
            let g: Vec<u32> = qs[..n_q]
                .iter()
                .map(|q| {
                    q.source_doc
                        .ok_or_else(|| anyhow::anyhow!("query {} has no source_doc", q.qid))
                        .map(|d| d as u32)
                })
                .collect::<Result<_>>()?;
            anyhow::ensure!(
                g.iter().all(|&d| (d as usize) < docs.len()),
                "a gold document id is outside the {} embedded documents", docs.len()
            );
            Some(g)
        }
    };

    eprintln!("computing exact ground truth...");
    let cpu0 = annlite_fts5::cpu::process_cpu_nanos();
    let t0 = std::time::Instant::now();
    let gold: Vec<Vec<u32>> = (0..n_q)
        .map(|i| exact_top_k(&docs, queries.row(i), 100).iter().map(|x| x.0).collect())
        .collect();
    let exact_ms = t0.elapsed().as_secs_f64() * 1000.0 / n_q as f64;
    let exact_cpu_ms = annlite_fts5::cpu::since_ms(cpu0).map(|t| t / n_q as f64);
    eprintln!("  exact brute force: {exact_ms:.2} ms/query");

    // The ceiling every approximate configuration below is measured against: what
    // these same vectors score when nothing is approximated at all. Quoting an index's
    // success@1 without it confuses a loss from quantization with a loss from the
    // embeddings.
    let exact_quality = known.as_ref().map(|g| {
        let ranks: Vec<Option<usize>> =
            (0..n_q).map(|i| rank_of(&gold[i], g[i])).collect();
        annlite_fts5::metrics::known_item_quality(&ranks)
    });
    if let Some(q) = &exact_quality {
        eprintln!(
            "  exact search known-item: success@1 {:.3} @10 {:.3} @100 {:.3}  MRR@10 {:.4}",
            q.success_at[0].1, q.success_at[1].1, q.success_at[2].1, q.mrr_at[1].1
        );
    }

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
    let vparams = VamanaParams { r: cli.r, l_build: cli.l_build, alpha: cli.alpha, seed: 0xDA7A };
    let graph = if cli.build_batch > 0 {
        Vamana::build_parallel(&docs, vparams, cli.build_batch)
    } else {
        Vamana::build(&docs, vparams)
    };
    let build_s = t0.elapsed().as_secs_f64();
    let gs = graph.stats();
    eprintln!("  built in {build_s:.1}s, mean degree {:.1}, orphans {}", gs.mean_degree, gs.orphans);

    let out_path = cli.out_dir.join(format!("ann-{}.jsonl", cli.scale));
    let mut out = std::fs::File::create(&out_path)?;
    writeln!(out, "{}", serde_json::json!({
        "record": "meta", "scale": cli.scale, "docs": docs.len(), "queries": n_q,
        "dim": cli.dim, "m": cli.m, "r": cli.r, "alpha": cli.alpha, "l_build": cli.l_build,
        "pq_train_vectors": train.len(), "pq_train_seconds": pq_train_s,
        "build_batch": cli.build_batch,
        "vamana_build_seconds": build_s, "mean_degree": gs.mean_degree, "orphans": gs.orphans,
        "exact_ms_per_query": exact_ms, "exact_cpu_ms_per_query": exact_cpu_ms,
        "gold": cli.gold.as_ref().map(|p| p.display().to_string()),
        "k": cli.k,
        "exact_known_item": exact_quality,
    }))?;

    println!("\n{:>10} {:>9} {:>5} {:>5} {:>6} {:>8} {:>9} {:>8} {:>7} {:>7} {:>8}",
             "ordering", "mode", "L", "beam", "rrank", "recall10", "nodes", "pages", "runs",
             "hops", "ms");
    println!("{}", "-".repeat(101));

    let chosen: Vec<Ordering> = cli
        .orderings
        .iter()
        .map(|s| match s.to_lowercase().as_str() {
            "identity" => Ok(Ordering::Identity),
            "bfs" => Ok(Ordering::Bfs),
            "cluster" => Ok(Ordering::Cluster),
            other => Err(anyhow::anyhow!("unknown ordering {other}")),
        })
        .collect::<Result<_>>()?;

    for ordering in chosen {
        let perm = match ordering {
            Ordering::Identity => Permutation::identity(docs.len()),
            Ordering::Bfs => bfs_order(docs.len(), graph.medoid, |n| graph.neighbors(n).to_vec()),
            Ordering::Cluster => {
                // Roughly one cluster per page's worth of records, so a cluster is
                // about the granularity a single fetch can deliver.
                let k = (docs.len() / 64).clamp(2, 4096);
                cluster_order_sampled(&docs, k, 10, 19, cli.cluster_train)
            }
        };

        let t_perm = std::time::Instant::now();
        let db_path = cli.db_dir.join(format!("ann-{}-{:?}.db", cli.scale, ordering).to_lowercase());
        eprintln!("  {:?} ordering computed in {:.1}s", ordering, t_perm.elapsed().as_secs_f64());
        let _ = std::fs::remove_file(&db_path);
        let mut conn = Connection::open(&db_path)?;
        let t_write = std::time::Instant::now();
        let meta = write_index(&mut conn, &graph, &pq, &codes, &docs, &perm, ordering, None)?;
        let write_s = t_write.elapsed().as_secs_f64();
        let (node_pages, per_page) = node_page_stats(&conn)?;
        let by_table = table_page_stats(&conn)?;
        let db_bytes = std::fs::metadata(&db_path)?.len();

        // Queries run against a read-only connection through the FTS5 baseline's
        // pass-through VFS, so the pages this sweep reports are the same kind of thing
        // that baseline reports: 4 KiB file pages SQLite actually asked for, b-tree
        // interior pages and all. The layout-derived counts are kept alongside them --
        // they are what RESEARCH_LOG.md sections 11 and 16 report, and they are the
        // only ones that isolate node numbering from SQLite's own bookkeeping -- but
        // only the VFS figures can be set beside FTS5's without an asterisk.
        drop(conn);
        let conn = annlite_fts5::pages::open_counting(&db_path)?;
        let idx = Index::open(&conn)?;
        let resident = ResidentCodes::load(&conn, &idx)?;
        writeln!(out, "{}", serde_json::json!({
            "record": "index", "ordering": format!("{ordering:?}"), "write_seconds": write_s,
            "db_bytes": db_bytes, "node_leaf_pages": node_pages, "records_per_page": per_page,
            "record_bytes": idx.fmt.len(), "count": meta.count,
            "code_blob_bytes": resident.bytes(),
            "bytes_by_table": by_table
                .into_iter()
                .map(|(n, p, b)| serde_json::json!({"name": n, "pages": p, "bytes": b}))
                .collect::<Vec<_>>(),
        }))?;

        for &l in &cli.search_l {
          for &beam in &cli.beams {
            for &rerank in &[0usize, 100] {
              for mode in ["ondisk", "resident"] {
                // CPU rather than wall clock: this machine runs other benchmarks
                // concurrently and the wall-clock figures move by more than 2x with
                // background load. Both are recorded; neither is clean.
                let cpu0 = annlite_fts5::cpu::process_cpu_nanos();
                let t0 = std::time::Instant::now();
                let (mut hits, mut nodes, mut pages, mut runs, mut hops, mut vecs, mut vpages) =
                    (0usize, 0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
                let mut vruns = 0usize;
                let mut ranks: Vec<Option<usize>> = Vec::with_capacity(n_q);
                let (mut vfs_page_per_q, mut vfs_run_per_q) = (Vec::new(), Vec::new());
                // Per-query, not just totalled: the distribution is what a client
                // feels, and a mean hides the query that costs five times it.
                let (mut page_per_q, mut run_per_q) = (Vec::new(), Vec::new());
                for qi in 0..n_q {
                    // Cold pager cache per query, exactly as the FTS5 measurement does
                    // it: without this the second query would read almost nothing.
                    conn.execute_batch("PRAGMA shrink_memory")?;
                    annlite_fts5::vfs::record_start();
                    let res = if mode == "resident" {
                        search_resident(
                            &conn, &idx, &resident, queries.row(qi), cli.k, l, beam, rerank,
                        )?
                    } else {
                        search(&conn, &idx, queries.row(qi), cli.k, l, beam, rerank)?
                    };
                    let trace = annlite_fts5::vfs::record_take();
                    vfs_page_per_q.push(trace.pages(PAGE_BYTES as u64).len() as f64);
                    vfs_run_per_q.push(trace.contiguous_runs(PAGE_BYTES as u64) as f64);
                    let ids: Vec<u32> = res.results.iter().map(|x| x.0).collect();
                    // Gold ids are in the original numbering; map through the permutation.
                    hits += gold[qi][..10]
                        .iter()
                        .filter(|g| ids.contains(&perm.new_id_of[**g as usize]))
                        .count();
                    if let Some(g) = &known {
                        ranks.push(rank_of(&ids, perm.new_id_of[g[qi] as usize]));
                    }
                    nodes += res.cost.nodes_read;
                    pages += res.cost.distinct_pages;
                    hops += res.cost.hops;
                    vecs += res.cost.vectors_read;
                    vpages += res.cost.rerank_pages;
                    vruns += res.cost.rerank_runs;
                    runs += res.cost.contiguous_runs_estimate();
                    // Reranking reads a second table, so its pages are part of what
                    // the query costs even though they are counted apart from the
                    // traversal's. RESEARCH_LOG.md 16.2 charges them the same way.
                    page_per_q.push((res.cost.distinct_pages + res.cost.rerank_pages) as f64);
                    run_per_q.push((res.cost.contiguous_runs_estimate() + res.cost.rerank_runs) as f64);
                }
                let ms = t0.elapsed().as_secs_f64() * 1000.0 / n_q as f64;
                let cpu_ms = annlite_fts5::cpu::since_ms(cpu0).map(|t| t / n_q as f64);
                let f = n_q as f64;
                let recall = hits as f64 / (n_q * 10) as f64;
                let quality = known
                    .as_ref()
                    .map(|_| annlite_fts5::metrics::known_item_quality(&ranks));
                println!("{:>10} {:>9} {l:>5} {beam:>5} {rerank:>6} {recall:>8.3} {:>9.1} {:>8.1} {:>7.1} {:>7.1} {ms:>8.2}",
                         format!("{ordering:?}"), mode, nodes as f64 / f, pages as f64 / f,
                         runs as f64 / f, hops as f64 / f);
                writeln!(out, "{}", serde_json::json!({
                    "record": "query_set", "ordering": format!("{ordering:?}"), "mode": mode,
                    "l": l,
                    // The search widens L to at least k so it can return k results,
                    // so a run asking for more results than L would silently be a
                    // different run from the one its label claims.
                    "l_effective": l.max(cli.k).max(1),
                    "beam": beam, "rerank": rerank, "recall_at_10": recall,
                    "mean_nodes_read": nodes as f64 / f, "mean_distinct_pages": pages as f64 / f,
                    "mean_contiguous_runs": runs as f64 / f, "mean_hops": hops as f64 / f,
                    "mean_vectors_read": vecs as f64 / f, "mean_rerank_pages": vpages as f64 / f,
                    "mean_rerank_runs": vruns as f64 / f,
                    "pages_per_query": Dist::of(&page_per_q),
                    "runs_per_query": Dist::of(&run_per_q),
                    "vfs_pages_per_query": Dist::of(&vfs_page_per_q),
                    "vfs_runs_per_query": Dist::of(&vfs_run_per_q),
                    "ms_per_query": ms, "cpu_ms_per_query": cpu_ms,
                    "k": cli.k, "known_item": quality,
                }))?;
              }
            }
          }
        }
    }
    eprintln!("\nresults -> {}", out_path.display());
    Ok(())
}
