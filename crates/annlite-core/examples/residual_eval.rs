//! Does reranking in the compressed domain keep exact quality?
//!
//! Section 15.2's unresolved gap: PLAID compresses candidate generation but exact
//! reranking reads uncompressed vectors, so the accurate configuration stores 29,196
//! bytes per document against 928 for the cheap one. This measures where residual
//! quantization lands between them.
//!
//! Writes `bench/results/residual-code.jsonl` as well as printing the table, because
//! every other number in RESEARCH_LOG.md traces to a committed measurement file and
//! this one used to trace only to a screenful of stdout. The rows are deterministic
//! — k-means is seeded, and nothing here is timed — so re-running reproduces the
//! file rather than perturbing it.

use annlite_core::late::{maxsim, LateIndex, MultiVector};
use annlite_core::residual::ResidualStore;
use std::io::Write;
use std::path::{Path, PathBuf};

fn read_lengths(p: &Path) -> anyhow::Result<Vec<usize>> {
    Ok(std::fs::read(p)?
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as usize)
        .collect())
}

fn read_multi(p: &Path, lengths: &[usize], dim: usize) -> anyhow::Result<Vec<MultiVector>> {
    let all: Vec<f32> = std::fs::read(p)?
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let mut out = Vec::with_capacity(lengths.len());
    let mut off = 0usize;
    for &n in lengths {
        out.push(MultiVector { data: all[off * dim..(off + n) * dim].to_vec(), dim });
        off += n;
    }
    anyhow::ensure!(off * dim == all.len(), "lengths do not account for every vector");
    Ok(out)
}

/// The rank of the gold document for each query, 1-indexed, `usize::MAX` if absent.
fn succ(r: &[usize], k: usize) -> f64 {
    r.iter().filter(|&&x| x <= k).count() as f64 / r.len() as f64
}

fn mrr(r: &[usize]) -> f64 {
    r.iter().map(|&x| if x <= 10 { 1.0 / x as f64 } else { 0.0 }).sum::<f64>() / r.len() as f64
}

fn quality(r: &[usize]) -> serde_json::Value {
    serde_json::json!({
        "success@1": succ(r, 1),
        "success@10": succ(r, 10),
        "mrr@10": mrr(r),
    })
}

fn main() -> anyhow::Result<()> {
    let dim = 48;
    let d = Path::new("data/embeddings");
    let out_path: PathBuf = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "bench/results/residual-code.jsonl".into())
        .into();

    let dl = read_lengths(&d.join("code-late-lengths.i32"))?;
    let ql = read_lengths(&d.join("code-late-qlengths.i32"))?;
    let docs = read_multi(&d.join("code-late.f32"), &dl, dim)?;
    let queries = read_multi(&d.join("code-late-q.f32"), &ql, dim)?;
    let total: usize = dl.iter().sum();
    println!("{} docs ({total} tokens), {} queries", docs.len(), queries.len());

    let mut rows: Vec<serde_json::Value> = Vec::new();
    let common = |config: &str, per_token: usize, per_doc: usize, ranks: &[usize], notes: &str| {
        serde_json::json!({
            "record": "system",
            "corpus": "code",
            "docs": 3366,
            "queries": 500,
            "system": "late",
            "config": config,
            "bytes_per_token": per_token,
            "bytes_per_doc": per_doc,
            "compression_vs_exact": (48 * 4) as f64 / per_token as f64,
            "quality": quality(ranks),
            "notes": notes,
        })
    };

    // Exact MaxSim over uncompressed vectors: the ceiling and the expensive baseline.
    let rank_exact: Vec<usize> = (0..queries.len())
        .map(|qi| {
            let mut s: Vec<(usize, f32)> =
                (0..docs.len()).map(|i| (i, maxsim(&queries[qi], &docs[i]))).collect();
            s.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
            s.iter().position(|x| x.0 == qi).unwrap() + 1
        })
        .collect();
    println!(
        "\nexact MaxSim (uncompressed): succ@1 {:.3} succ@10 {:.3} MRR@10 {:.3}  {} B/doc",
        succ(&rank_exact, 1),
        succ(&rank_exact, 10),
        mrr(&rank_exact),
        total * dim * 4 / docs.len()
    );
    rows.push(common(
        "exact/full-scan",
        dim * 4,
        total * dim * 4 / docs.len(),
        &rank_exact,
        "MaxSim against every document's uncompressed token vectors -- no candidate \
         generation at all, so this is the ceiling the quantized rows are measured \
         against. It reproduces the success@1 of the k=1024/probe=8/rerank=100 row in \
         RESEARCH_LOG 15.4, which is what establishes that the 100-candidate pool is \
         not the binding constraint: the same 0.454 is reached whether 100 documents \
         are reranked or all 3,366. success@1 ceiling on this query set is 0.962.",
    ));

    let k = 1024usize;
    let idx = LateIndex::build_sampled(&docs, k, 10, 0xC0DE, 60_000)?;

    // Flatten tokens and their centroid assignments in document order.
    let mut flat = Vec::with_capacity(total * dim);
    let mut codes = Vec::with_capacity(total);
    for (i, doc) in docs.iter().enumerate() {
        flat.extend_from_slice(&doc.data);
        codes.extend_from_slice(idx.doc_codes(i as u32));
    }

    println!(
        "\n{:>6} {:>10} {:>10} {:>10} {:>9} {:>9} {:>9}",
        "bits", "B/token", "B/doc", "vs exact", "succ@1", "succ@10", "MRR@10"
    );
    println!("{}", "-".repeat(68));

    // Centroid only, for the bottom of the range.
    {
        let per_doc = (total * 4 + idx.centroids.len() * 4) / docs.len();
        let ranks: Vec<usize> = (0..queries.len())
            .map(|qi| {
                let c = idx.candidates(&queries[qi], 8, docs.len());
                c.iter().position(|x| x.0 as usize == qi).map(|p| p + 1).unwrap_or(usize::MAX)
            })
            .collect();
        println!(
            "{:>6} {:>10} {:>10} {:>10} {:>9.3} {:>9.3} {:>9.3}",
            "none",
            4,
            per_doc,
            format!("{:.0}x", (dim * 4) as f64 / 4.0),
            succ(&ranks, 1),
            succ(&ranks, 10),
            mrr(&ranks)
        );
        rows.push(common(
            "residual/bits=none/rerank=0",
            4,
            per_doc,
            &ranks,
            "Centroid ids only: ranking is the PLAID candidate score, with no \
             reranking stage and no residual stored. This is the cheap end of the \
             range section 15.2 left open -- k=1024, probe=8, k-means seeded 0xC0DE \
             on a 60,000-token sample.",
        ));
    }

    for bits in [1u8, 2, 4] {
        let store = ResidualStore::build(&flat, &codes, &idx.centroids, dim, bits, 200_000)?;
        let per_token = store.codec.bytes_per_token(dim) + 4; // residual + centroid id
        let per_doc = (store.bytes()
            + total * 4
            + idx.centroids.len() * 4
            + store.codec.table_bytes())
            / docs.len();

        // Reconstruct each document once, then MaxSim against the reconstruction --
        // which is exactly what a client with the compressed index would do.
        let mut recon: Vec<MultiVector> = Vec::with_capacity(docs.len());
        let mut cursor = 0usize;
        let mut tok = vec![0f32; dim];
        for doc in docs.iter() {
            let n = doc.tokens();
            let mut data = Vec::with_capacity(n * dim);
            for t in 0..n {
                store.reconstruct_into(cursor + t, codes[cursor + t], &idx.centroids, &mut tok);
                data.extend_from_slice(&tok);
            }
            cursor += n;
            recon.push(MultiVector { data, dim });
        }

        // Stage 1 and 2 generate candidates; stage 3 reranks against reconstructions.
        let ranks: Vec<usize> = (0..queries.len())
            .map(|qi| {
                let cands = idx.candidates(&queries[qi], 8, 100);
                let mut r: Vec<(u32, f32)> = cands
                    .iter()
                    .map(|&(dd, _)| (dd, maxsim(&queries[qi], &recon[dd as usize])))
                    .collect();
                r.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
                r.iter().position(|x| x.0 as usize == qi).map(|p| p + 1).unwrap_or(usize::MAX)
            })
            .collect();

        println!(
            "{:>6} {:>10} {:>10} {:>10} {:>9.3} {:>9.3} {:>9.3}",
            bits,
            per_token,
            per_doc,
            format!("{:.1}x", (dim * 4) as f64 / per_token as f64),
            succ(&ranks, 1),
            succ(&ranks, 10),
            mrr(&ranks)
        );
        rows.push(common(
            &format!("residual/bits={bits}/rerank=100"),
            per_token,
            per_doc,
            &ranks,
            &format!(
                "Top-100 candidates from stages 1-2 reranked against centroid+residual \
                 reconstructions, never against a stored full vector -- so bytes_per_doc \
                 is everything a client holds. {bits}-bit residual, bucket cutoffs are \
                 equal-mass quantiles of the residual distribution pooled across all \
                 dimensions (ColBERTv2's choice: residuals are offsets from a nearby \
                 centroid, so their spread is set by distance from the cluster centre \
                 rather than by axis, and a per-dimension codebook would spend 48x the \
                 table on the same distribution). Trained on a 200,000-token sample.",
            ),
        ));
    }

    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut f = std::fs::File::create(&out_path)?;
    for row in &rows {
        writeln!(f, "{}", serde_json::to_string(row)?)?;
    }
    println!("\n-> {} ({} rows)", out_path.display(), rows.len());
    Ok(())
}
